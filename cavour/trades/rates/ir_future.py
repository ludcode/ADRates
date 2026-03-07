##############################################################################

##############################################################################

"""
Short-Term Interest Rate (STIR) Futures implementation.

Provides the IRFuture class for creating, valuing, and analyzing STIR futures
contracts (3-month interest rate futures based on overnight indices like SOFR,
SONIA, and ESTR).

Key features:
- Support for IMM quarterly futures (Mar/Jun/Sep/Dec)
- Support for serial monthly futures (all months)
- Support for FOMC meeting-dated futures (USD SOFR)
- Support for BOE MPC meeting-dated futures (GBP SONIA)
- Price-to-rate conversion (Price = 100 - implied_rate * 100)
- Integration with curve bootstrapping (OISCurve)
- Optional convexity adjustment (futures vs forwards)

Contract specifications:
- USD SOFR Futures: $1M notional, ACT/360, 3M tenor (CME: SFR)
- GBP SONIA Futures: £1M notional, ACT/365F, 3M tenor (ICE: FS)
- EUR ESTR Futures: €1M notional, ACT/360, 3M tenor (Eurex: FES)

Example:
    >>> # Create SOFR futures for March 2024 expiry at price 97.50 (2.5% implied)
    >>> value_dt = Date(15, 1, 2024)
    >>> future = IRFuture(
    ...     effective_dt=value_dt,
    ...     expiry_date_or_contract="H24",  # March 2024 IMM
    ...     futures_price=97.50,
    ...     contract_type=FutureContractTypes.IMM,
    ...     currency=CurrencyTypes.USD,
    ...     floating_index=CurveTypes.USD_OIS_SOFR
    ... )
    >>>
    >>> # Get implied rate
    >>> rate = future.implied_rate()  # 0.025 (2.5%)
    >>>
    >>> # Value the future
    >>> pv = future.value(value_dt, discount_curve)
"""

import numpy as np
import jax.numpy as jnp

from cavour.utils.error import LibError
from cavour.utils.date import Date
from cavour.utils.day_count import DayCount, DayCountTypes
from cavour.utils.global_types import (
    CurveTypes, FutureContractTypes, InstrumentTypes
)
from cavour.utils.calendar import CalendarTypes, Calendar, BusDayAdjustTypes
from cavour.utils.helpers import check_argument_types, label_to_string
from cavour.utils.currency import CurrencyTypes
from cavour.market.curves.discount_curve import DiscountCurve
from cavour.market.position.position import Position

###############################################################################

# Futures month codes (CME standard)
FUTURES_MONTH_CODES = {
    1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
    7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
}

# Reverse mapping for parsing
MONTH_CODE_TO_NUMBER = {v: k for k, v in FUTURES_MONTH_CODES.items()}

# IMM months only (quarterly)
IMM_MONTHS = [3, 6, 9, 12]  # March, June, September, December

# Standard contract sizes by currency
CONTRACT_SIZES = {
    CurrencyTypes.USD: 1_000_000,  # $1M for SOFR futures
    CurrencyTypes.GBP: 1_000_000,  # £1M for SONIA futures
    CurrencyTypes.EUR: 1_000_000,  # €1M for ESTR futures
}

# Standard day count conventions by currency
DEFAULT_DAY_COUNTS = {
    CurrencyTypes.USD: DayCountTypes.ACT_360,
    CurrencyTypes.GBP: DayCountTypes.ACT_365F,
    CurrencyTypes.EUR: DayCountTypes.ACT_360,
}

# Bloomberg ticker roots by currency
TICKER_ROOTS = {
    CurrencyTypes.USD: "SFR",   # CME SOFR 3M Futures
    CurrencyTypes.GBP: "FS",    # ICE SONIA 3M Futures (Short Sterling replacement)
    CurrencyTypes.EUR: "FES",   # Eurex Three-Month ESTR Futures
}

###############################################################################


class IRFuture:
    """
    Short-Term Interest Rate (STIR) Futures implementation.

    A STIR future is an exchange-traded contract where:
    - Price = 100 - implied_3M_rate * 100
    - Example: Price 97.50 implies 2.50% 3-month rate
    - Settlement at expiry delivers the 3-month interest rate
    - Used for hedging and speculating on short-term rates

    STIR futures are used for:
    - Building the 3M-2Y section of OIS curves (most liquid instruments)
    - Hedging short-term interest rate exposure
    - Expressing views on central bank policy paths

    Market conventions:
    - USD SOFR: $1M contract, ACT/360, quarterly IMM + serial monthly
    - GBP SONIA: £1M contract, ACT/365F, quarterly IMM + serial monthly
    - EUR ESTR: €1M contract, ACT/360, quarterly IMM + serial monthly

    Contract types:
    - IMM: Quarterly futures (Mar, Jun, Sep, Dec) - most liquid
    - SERIAL_MONTHLY: Monthly futures (all months) - less liquid
    - FOMC: FOMC meeting-dated (USD only)
    - BOE: BOE MPC meeting-dated (GBP only)
    """

    def __init__(self,
                 effective_dt: Date,
                 expiry_date_or_contract: (Date, str),
                 futures_price: float,
                 contract_type: FutureContractTypes,
                 currency: CurrencyTypes,
                 floating_index: CurveTypes,
                 dc_type=None,  # Optional: defaults by currency
                 contract_size=None,  # Optional: defaults by currency
                 settlement_lag: int = 0,
                 convexity_adjustment: float = 0.0,
                 cal_type: CalendarTypes = CalendarTypes.WEEKEND,
                 bd_type: BusDayAdjustTypes = BusDayAdjustTypes.FOLLOWING):
        """
        Create a STIR futures contract.

        Args:
            effective_dt: Valuation/trade date
            expiry_date_or_contract: Expiry date OR contract code string
                - Date object: explicit expiry date
                - "H24": March 2024 (month code + 2-digit year)
                - "SFRH24": SOFR March 2024 (ticker + month code + year)
                - "IMM1": 1st IMM date from effective_dt
                - "IMM2": 2nd IMM date from effective_dt
            futures_price: Market price (e.g., 97.50 for 2.50% implied rate)
            contract_type: FutureContractTypes enum (IMM, SERIAL, FOMC, BOE)
            currency: Currency of the futures contract
            floating_index: Index type (USD_OIS_SOFR, GBP_OIS_SONIA, EUR_OIS_ESTR)
            dc_type: Day count convention (defaults by currency: USD=ACT/360, GBP=ACT/365F)
            contract_size: Notional amount (defaults by currency: $1M, £1M, €1M)
            settlement_lag: Business days from expiry to settlement (default=0)
            convexity_adjustment: Adjustment from futures to forward rate (default=0.0)
                - Positive adjustment: forward_rate = futures_rate - convexity_adj
                - Typical adjustment: ~0.5 * vol^2 * T1 * T2 (requires volatility data)
            cal_type: Calendar for business day adjustments
            bd_type: Business day adjustment convention
        """

        check_argument_types(self.__init__, locals())

        # Set instrument type
        self.derivative_type = InstrumentTypes.STIR_FUTURE

        # Store basic parameters
        self._effective_dt = effective_dt
        self._futures_price = futures_price
        self._contract_type = contract_type
        self._currency = currency
        self._floating_index = floating_index
        self._convexity_adjustment = convexity_adjustment
        self._cal_type = cal_type
        self._bd_type = bd_type
        self._settlement_lag = settlement_lag

        # Set defaults for day count and contract size
        self._dc_type = dc_type if dc_type is not None else DEFAULT_DAY_COUNTS.get(currency, DayCountTypes.ACT_360)
        self._notional = contract_size if contract_size is not None else CONTRACT_SIZES.get(currency, 1_000_000)

        # Parse expiry date from contract code or use explicit date
        if isinstance(expiry_date_or_contract, Date):
            self._expiry_dt = expiry_date_or_contract
        else:
            self._expiry_dt = self._parse_contract_code(
                expiry_date_or_contract, effective_dt, contract_type
            )

        # Compute settlement date (expiry + settlement lag business days)
        calendar = Calendar(cal_type)
        if settlement_lag > 0:
            self._settlement_dt = calendar.add_business_days(
                self._expiry_dt, settlement_lag
            )
        else:
            self._settlement_dt = self._expiry_dt

        # Compute 3-month accrual period dates
        # For STIR futures, the accrual period is typically:
        # - Start: expiry date (or settlement date depending on contract)
        # - End: start + 3 months
        self._accrual_start_dt = self._settlement_dt
        self._accrual_end_dt = self._accrual_start_dt.add_months(3)

        # Apply business day adjustment to accrual end
        self._accrual_end_dt = calendar.adjust(self._accrual_end_dt, bd_type)

        # Compute year fraction for 3-month period
        dcc = DayCount(self._dc_type)
        self._year_frac, _, _ = dcc.year_frac(
            self._accrual_start_dt,
            self._accrual_end_dt
        )

        # Compute year fraction from effective_dt to accrual_start_dt
        # This is needed for forward-starting instrument support in curve bootstrap
        self._year_frac_to_start, _, _ = dcc.year_frac(
            self._effective_dt,
            self._accrual_start_dt
        )

        # Compute implied rate from futures price
        # Price = 100 - implied_rate * 100
        # => implied_rate = (100 - price) / 100
        self._implied_rate = (100.0 - futures_price) / 100.0

        # Adjust for convexity (futures rate -> forward rate)
        # Forward rate = Futures rate - convexity adjustment
        self._forward_rate = self._implied_rate - convexity_adjustment

        # CRITICAL: Curve builder compatibility attributes
        # OISCurve expects these attributes for bootstrapping
        self._adjusted_fixed_dts = [self._accrual_end_dt]  # Single payment date
        self._fixed_coupon = self._forward_rate  # Convexity-adjusted rate
        self._fixed_year_fracs = [self._year_frac]  # Single element list
        self._start_dt = self._accrual_start_dt  # Start of accrual period
        self._maturity_dt = self._accrual_end_dt  # End of accrual period
        self._termination_dt = self._accrual_end_dt  # Alias for maturity

        # Forward-starting instrument support
        # This tells the curve builder that the futures accrual period starts
        # at a future date (accrual_start_dt), not at effective_dt
        self._start_year_fracs = [self._year_frac_to_start]

###############################################################################

    def _parse_contract_code(
        self,
        contract_str: str,
        effective_dt: Date,
        contract_type: FutureContractTypes
    ) -> Date:
        """
        Parse futures contract code to determine expiry date.

        Supported formats:
        - "H24": Month code + 2-digit year (e.g., H24 = March 2024)
        - "SFRH24": Ticker + month code + 2-digit year
        - "IMM1", "IMM2", ...: Nth IMM date from effective_dt

        Args:
            contract_str: Contract code string
            effective_dt: Reference date for relative codes (IMM1, IMM2, etc.)
            contract_type: Type of contract (IMM, SERIAL, FOMC, BOE)

        Returns:
            Date object for contract expiry

        Raises:
            LibError: If contract code cannot be parsed
        """
        contract_str = contract_str.upper().strip()

        # Pattern 1: "IMMn" notation (e.g., IMM1, IMM2, ...)
        if contract_str.startswith("IMM"):
            try:
                n = int(contract_str[3:])  # Extract number after "IMM"
                if n < 1:
                    raise LibError("IMM number must be >= 1")
                expiry_dt = effective_dt.get_nth_imm_date(n)
                return expiry_dt
            except (ValueError, IndexError):
                raise LibError(f"Invalid IMM notation: {contract_str}. Use IMM1, IMM2, etc.")

        # Pattern 2: "FOMC1", "FOMC2", ... (FOMC meeting-dated futures)
        if contract_str.startswith("FOMC"):
            if contract_type != FutureContractTypes.FOMC:
                raise LibError("FOMC notation requires contract_type=FutureContractTypes.FOMC")
            try:
                n = int(contract_str[4:])  # Extract number after "FOMC"
                if n < 1 or n > 12:
                    raise LibError("FOMC number must be 1-12")
                # Get FOMC meeting dates for current and next year
                fomc_dates = (
                    Date.get_fomc_meeting_dates(effective_dt._y) +
                    Date.get_fomc_meeting_dates(effective_dt._y + 1)
                )
                # Filter to future dates only
                future_fomc_dates = [d for d in fomc_dates if d > effective_dt]
                if n > len(future_fomc_dates):
                    raise LibError(f"FOMC{n} not found in next 2 years")
                return future_fomc_dates[n - 1]
            except (ValueError, IndexError):
                raise LibError(f"Invalid FOMC notation: {contract_str}. Use FOMC1, FOMC2, etc.")

        # Pattern 3: "BOE1", "BOE2", ... (BOE MPC meeting-dated futures)
        if contract_str.startswith("BOE"):
            if contract_type != FutureContractTypes.BOE:
                raise LibError("BOE notation requires contract_type=FutureContractTypes.BOE")
            try:
                n = int(contract_str[3:])  # Extract number after "BOE"
                if n < 1 or n > 12:
                    raise LibError("BOE number must be 1-12")
                # Get BOE meeting dates for current and next year
                boe_dates = (
                    Date.get_boe_meeting_dates(effective_dt._y) +
                    Date.get_boe_meeting_dates(effective_dt._y + 1)
                )
                # Filter to future dates only
                future_boe_dates = [d for d in boe_dates if d > effective_dt]
                if n > len(future_boe_dates):
                    raise LibError(f"BOE{n} not found in next 2 years")
                return future_boe_dates[n - 1]
            except (ValueError, IndexError):
                raise LibError(f"Invalid BOE notation: {contract_str}. Use BOE1, BOE2, etc.")

        # Pattern 4: "SFRH24", "FSM24", etc. (Ticker + month code + year)
        # Strip ticker prefix if present
        for ticker_root in TICKER_ROOTS.values():
            if contract_str.startswith(ticker_root):
                contract_str = contract_str[len(ticker_root):]
                break

        # Pattern 5: "H24", "M24", etc. (Month code + 2-digit year)
        if len(contract_str) >= 3:
            month_code = contract_str[0]
            year_str = contract_str[1:]

            # Validate month code
            if month_code not in MONTH_CODE_TO_NUMBER:
                raise LibError(
                    f"Invalid month code '{month_code}'. "
                    f"Must be one of: {list(MONTH_CODE_TO_NUMBER.keys())}"
                )

            # Parse year
            try:
                if len(year_str) == 2:
                    # 2-digit year: assume 20xx for 00-49, 19xx for 50-99
                    year_2digit = int(year_str)
                    if year_2digit <= 49:
                        year = 2000 + year_2digit
                    else:
                        year = 1900 + year_2digit
                elif len(year_str) == 4:
                    year = int(year_str)
                else:
                    raise ValueError("Year must be 2 or 4 digits")
            except ValueError:
                raise LibError(f"Invalid year format: {year_str}")

            month = MONTH_CODE_TO_NUMBER[month_code]

            # For IMM contracts, validate month is an IMM month
            if contract_type == FutureContractTypes.IMM:
                if month not in IMM_MONTHS:
                    raise LibError(
                        f"Month {month} is not an IMM month. "
                        f"IMM months are: Mar(H), Jun(M), Sep(U), Dec(Z)"
                    )

            # Compute expiry date (3rd Wednesday of the month)
            expiry_day = Date(1, month, year).third_wednesday_of_month(month, year)
            expiry_dt = Date(expiry_day, month, year)

            return expiry_dt

        raise LibError(
            f"Cannot parse contract code: {contract_str}. "
            f"Use formats: 'H24', 'SFRH24', 'IMM1', 'FOMC1', 'BOE1'"
        )

###############################################################################

    def position(self, model):
        """
        Create a Position object for this future within a model.

        Args:
            model: The model containing curves and market data

        Returns:
            Position object for computing VALUE, DELTA, GAMMA, etc.
        """
        return Position(self, model)

###############################################################################

    def implied_rate(self):
        """
        Get the implied 3-month rate from the futures price.

        Formula: implied_rate = (100 - futures_price) / 100

        Returns:
            Implied 3-month rate (annualized, as decimal)

        Example:
            >>> future = IRFuture(..., futures_price=97.50, ...)
            >>> future.implied_rate()  # 0.025 (2.5%)
        """
        return self._implied_rate

###############################################################################

    def forward_rate_from_curve(self, value_dt, discount_curve):
        """
        Calculate the forward rate implied by the discount curve.

        This is the "fair" futures rate based on the curve.
        Compare to implied_rate() to assess futures pricing.

        Formula:
            Forward_Rate = (DF(start) / DF(end) - 1) / year_frac

        Args:
            value_dt: Valuation date
            discount_curve: Discount curve for rate calculation

        Returns:
            Forward rate (annualized, as decimal) from the curve

        Note:
            This does NOT include convexity adjustment.
            For futures vs forwards: futures_rate ≈ forward_rate + convexity_adj
        """
        if self._accrual_end_dt <= value_dt:
            return 0.0

        # Get discount factors at start and end of accrual period
        df_start = discount_curve.df(self._accrual_start_dt, self._dc_type)
        df_end = discount_curve.df(self._accrual_end_dt, self._dc_type)

        # Compute forward rate
        forward_rate = (df_start / df_end - 1.0) / self._year_frac

        return forward_rate

###############################################################################

    def value(self,
              value_dt: Date,
              discount_curve: DiscountCurve = None,
              ois_curve: DiscountCurve = None,
              xccy_discount_curve: DiscountCurve = None,
              spot_fx: float = None,
              collateral_type=None,
              first_fixing_rate=None):
        """
        Value the STIR futures contract on a valuation date.

        For futures, the value represents the PV of the difference between
        the futures implied rate and the forward rate from the curve.

        Args:
            value_dt: Valuation date
            discount_curve: Discount curve for PV calculation (primary)
            ois_curve: OIS curve (alternative if discount_curve not provided)
            xccy_discount_curve: Cross-currency discount curve (unused for futures)
            spot_fx: FX rate (unused for futures)
            collateral_type: Collateral type (unused for futures)
            first_fixing_rate: First fixing rate (unused for futures)

        Returns:
            Present value of the futures contract

        Note:
            For curve bootstrapping, the future should reprice to ~0 when valued
            on its own curve (with proper convexity adjustment).
        """
        # Select discount curve (prefer explicit discount_curve, fallback to ois_curve)
        if discount_curve is None:
            if ois_curve is None:
                raise ValueError("Either discount_curve or ois_curve must be provided")
            discount_curve = ois_curve

        # If accrual period has ended, futures has zero value
        if self._accrual_end_dt <= value_dt:
            return 0.0

        # Get forward rate from curve
        curve_forward_rate = self.forward_rate_from_curve(value_dt, discount_curve)

        # Futures value = notional * rate differential * year_frac
        # Rate differential: curve_forward_rate - futures_implied_rate
        # (This should be ~0 if futures is priced correctly)
        rate_differential = curve_forward_rate - self._forward_rate

        # Futures are marked-to-market daily with immediate cash settlement
        # No discounting applied - daily variation margin eliminates credit risk
        # See: Market convention for SOFR/EURIBOR futures valuation
        # PV = notional * rate_diff * year_frac (NO discount factor)
        pv = self._notional * rate_differential * self._year_frac

        return pv

###############################################################################

    def print_details(self):
        """Print futures contract details."""
        print("OBJECT TYPE:", type(self).__name__)
        print("Contract Type:", self._contract_type.name)
        print("Currency:", self._currency.name)
        print("Expiry Date:", self._expiry_dt)
        print("Settlement Date:", self._settlement_dt)
        print("Accrual Period:", f"{self._accrual_start_dt} to {self._accrual_end_dt}")
        print("Futures Price:", f"{self._futures_price:.4f}")
        print("Implied Rate:", f"{self._implied_rate*100:.4f}%")
        print("Forward Rate (convexity-adj):", f"{self._forward_rate*100:.4f}%")
        print("Convexity Adjustment:", f"{self._convexity_adjustment*10000:.2f} bps")
        print("Contract Size:", f"{self._notional:,.0f} {self._currency.name}")
        print("Year Fraction:", f"{self._year_frac:.6f}")
        print("Day Count:", self._dc_type.name)

###############################################################################

    def __repr__(self):
        s = label_to_string("OBJECT TYPE", type(self).__name__)
        s += label_to_string("Contract Type", self._contract_type.name)
        s += label_to_string("Currency", self._currency.name)
        s += label_to_string("Expiry Date", self._expiry_dt)
        s += label_to_string("Settlement Date", self._settlement_dt)
        s += label_to_string("Accrual Period", f"{self._accrual_start_dt} to {self._accrual_end_dt}")
        s += label_to_string("Futures Price", f"{self._futures_price:.4f}")
        s += label_to_string("Implied Rate", f"{self._implied_rate*100:.4f}%")
        s += label_to_string("Forward Rate", f"{self._forward_rate*100:.4f}%")
        s += label_to_string("Contract Size", f"{self._notional:,.0f}")
        s += label_to_string("Day Count", self._dc_type.name)
        s += label_to_string("Year Fraction", f"{self._year_frac:.6f}")
        return s

###############################################################################
