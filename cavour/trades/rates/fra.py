##############################################################################

##############################################################################

"""
Forward Rate Agreement (FRA) implementation for interest rate hedging.

Provides the FRA class for creating, valuing, and analyzing forward rate
agreements where a buyer locks in a borrowing rate for a future period.

Key features:
- Standard FRA notation support ("3x6", "6x9", "9x12")
- Forward rate calculation from discount curve
- Settlement date conventions (typically T+2 from fixing)
- Integration with automatic differentiation for Greeks
- Support for various day count conventions

A FRA is a contract where:
- Buyer locks in a borrowing rate (the FRA rate) for a future period
- At fixing date, the realized rate is compared to the FRA rate
- Settlement payment = Notional × (Forward_Rate - FRA_Rate) × Year_Frac
- Payment is discounted to settlement date

Example:
    >>> # Create a 3x6 FRA (3M fixing, 6M maturity) at 5.25%
    >>> value_dt = Date(15, 6, 2023)
    >>> fra = FRA(
    ...     effective_dt=value_dt,
    ...     fra_notation="3x6",
    ...     fra_rate=0.0525,
    ...     dc_type=DayCountTypes.ACT_360,
    ...     floating_index=CurveTypes.USD_OIS_SOFR,
    ...     currency=CurrencyTypes.USD,
    ...     notional=1_000_000
    ... )
    >>>
    >>> # Value the FRA
    >>> pv = fra.value(value_dt, discount_curve)
    >>> forward_rate = fra.forward_rate(value_dt, discount_curve)
"""

import numpy as np
import jax.numpy as jnp

from cavour.utils.error import LibError
from cavour.utils.date import Date
from cavour.utils.day_count import DayCount, DayCountTypes
from cavour.utils.global_types import CurveTypes
from cavour.utils.calendar import CalendarTypes, Calendar, BusDayAdjustTypes
from cavour.utils.helpers import check_argument_types, label_to_string
from cavour.utils.math import ONE_MILLION
from cavour.utils.global_types import SwapTypes, InstrumentTypes
from cavour.utils.currency import CurrencyTypes
from cavour.market.curves.discount_curve import DiscountCurve
from cavour.market.position.position import Position

###############################################################################


class FRA:
    """
    Forward Rate Agreement for locking in future borrowing rates.

    A FRA is an OTC derivative contract where two parties agree to exchange
    interest payments based on a notional principal for a future period.
    The buyer locks in a borrowing rate (the FRA rate).

    Valuation:
        - Forward Rate: (DF(start) / DF(end) - 1) / year_frac
        - Payoff: Notional × (Forward_Rate - FRA_Rate) × Year_Frac
        - PV: Payoff × DF(settlement) / DF(value_date)

    Settlement:
        - Typically T+2 business days from fixing date
        - Payment can be upfront (at start_dt) or in arrears (at end_dt)
        - Standard is upfront settlement

    Market conventions:
    - USD: ACT/360 day count
    - GBP: ACT/365F day count
    - EUR: ACT/360 day count

    FRA Notation:
    - "3x6": 3M fixing date, 6M end date (3M forward starting 3M deposit)
    - "6x9": 6M fixing date, 9M end date (6M forward starting 3M deposit)
    - "9x12": 9M fixing date, 12M end date (9M forward starting 3M deposit)
    """

    def __init__(self,
                 effective_dt: Date,
                 fra_notation: str = None,
                 fixing_dt: Date = None,
                 start_dt: Date = None,
                 end_dt: Date = None,
                 fra_rate: float = 0.0,
                 dc_type: DayCountTypes = DayCountTypes.ACT_360,
                 floating_index: CurveTypes = CurveTypes.USD_OIS_SOFR,
                 currency: CurrencyTypes = CurrencyTypes.USD,
                 notional: float = ONE_MILLION,
                 settlement_lag: int = 2,
                 cal_type: CalendarTypes = CalendarTypes.WEEKEND,
                 bd_type: BusDayAdjustTypes = BusDayAdjustTypes.MODIFIED_FOLLOWING):
        """
        Create a Forward Rate Agreement.

        Can be specified either using standard FRA notation (e.g., "3x6")
        or by providing explicit dates.

        Args:
            effective_dt: Trade date / valuation date reference
            fra_notation: Standard FRA notation (e.g., "3x6", "6x9", "9x12")
            fixing_dt: Explicit fixing date (alternative to fra_notation)
            start_dt: Explicit accrual start date (alternative to fra_notation)
            end_dt: Explicit accrual end date (alternative to fra_notation)
            fra_rate: Agreed FRA rate (annualized, as decimal, e.g., 0.05 for 5%)
            dc_type: Day count convention (ACT/360 for USD, ACT/365F for GBP)
            floating_index: Index type (SOFR, SONIA, ESTR)
            currency: Currency of the FRA
            notional: Notional amount (default 1,000,000)
            settlement_lag: Business days from fixing to settlement (default 2)
            cal_type: Calendar for business day adjustments
            bd_type: Business day adjustment convention

        Examples:
            >>> # Using FRA notation
            >>> fra = FRA(value_dt, fra_notation="3x6", fra_rate=0.0525, ...)
            >>>
            >>> # Using explicit dates
            >>> fixing = value_dt.add_months(3)
            >>> start = fixing.add_days(2)
            >>> end = fixing.add_months(6)
            >>> fra = FRA(value_dt, fixing_dt=fixing, start_dt=start,
            ...          end_dt=end, fra_rate=0.0525, ...)
        """

        # Set instrument type
        self.derivative_type = InstrumentTypes.FRA

        # Parse FRA notation or use explicit dates
        # Do this BEFORE check_argument_types to avoid validation errors on None
        if fra_notation is not None:
            # Parse notation like "3x6" (3M fixing, 6M maturity)
            fixing_dt, start_dt, end_dt = self._parse_fra_notation(
                effective_dt, fra_notation, cal_type, bd_type
            )
        else:
            # Use explicit dates
            if fixing_dt is None or start_dt is None or end_dt is None:
                raise ValueError(
                    "Must provide either fra_notation or "
                    "(fixing_dt, start_dt, end_dt)"
                )

        # Now check argument types with resolved parameters
        check_argument_types(self.__init__, locals())

        # Store dates
        self._effective_dt = effective_dt
        self._fixing_dt = fixing_dt
        self._start_dt = start_dt
        self._end_dt = end_dt

        # Compute settlement date (T+2 from fixing, typically)
        calendar = Calendar(cal_type)
        self._settlement_dt = calendar.add_business_days(fixing_dt, settlement_lag)

        # Validation
        if fixing_dt <= effective_dt:
            raise LibError("Fixing date must be after effective date")
        if start_dt < fixing_dt:
            raise LibError("Start date cannot be before fixing date")
        if end_dt <= start_dt:
            raise LibError("End date must be after start date")

        # Store parameters
        self._fra_rate = fra_rate
        self._dc_type = dc_type
        self._floating_index = floating_index
        self._currency = currency
        self._notional = notional
        self._settlement_lag = settlement_lag
        self._cal_type = cal_type
        self._bd_type = bd_type

        # Compute accrual year fraction (start to end) - used for FRA valuation
        dcc = DayCount(dc_type)
        self._year_frac, _, _ = dcc.year_frac(start_dt, end_dt)

        # Compute fixing-to-start year fraction (for reference)
        self._fixing_year_frac, _, _ = dcc.year_frac(fixing_dt, start_dt)

        # CRITICAL: Store attributes for curve builder compatibility
        # OISCurve expects these attributes to extract instrument data
        # For FRAs, we use the accrual end date for curve calibration
        # This ensures the curve has a DF at end_dt, which is needed for forward rate calculation
        self._adjusted_fixed_dts = [self._end_dt]  # Accrual end date (maturity)
        self._fixed_coupon = fra_rate  # The FRA rate

        # FORWARD-STARTING INSTRUMENT SUPPORT:
        # For forward-starting instruments like FRAs, we need to store TWO time periods:
        # 1. _fixed_year_fracs: The accrual period (start_dt to end_dt) used in bootstrap formula
        # 2. _start_year_fracs: When the forward period begins (effective_dt to start_dt)
        # This allows the curve builder to interpolate DF at start_dt and apply forward formula
        self._fixed_year_fracs = [self._year_frac]  # Accrual period ONLY (start_dt to end_dt)

        # NEW: Start time for forward-starting bootstrap
        start_year_frac, _, _ = dcc.year_frac(effective_dt, start_dt)
        self._start_year_fracs = [start_year_frac]  # When forward period begins

        self._start_dt_attr = effective_dt  # Renamed to avoid conflict

        # Store maturity for compatibility
        self._maturity_dt = end_dt
        self._termination_dt = end_dt

###############################################################################

    def _parse_fra_notation(self, effective_dt, fra_notation, cal_type, bd_type):
        """
        Parse standard FRA notation like "3x6" into dates.

        FRA notation "AxB" means:
        - A months from effective_dt to fixing date
        - B months from effective_dt to end date
        - (B-A) is the accrual period length

        Args:
            effective_dt: Reference date (typically trade date)
            fra_notation: String like "3x6", "6x9", "9x12"
            cal_type: Calendar for business day adjustments
            bd_type: Business day adjustment convention

        Returns:
            Tuple of (fixing_dt, start_dt, end_dt)

        Examples:
            "3x6" → 3M fixing, 3M-6M accrual period (3M forward 3M deposit)
            "6x9" → 6M fixing, 6M-9M accrual period (6M forward 3M deposit)
            "9x15" → 9M fixing, 9M-15M accrual period (9M forward 6M deposit)
        """
        # Parse notation
        parts = fra_notation.lower().replace(" ", "").split("x")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid FRA notation: {fra_notation}. "
                f"Expected format like '3x6', '6x9', etc."
            )

        try:
            fixing_months = int(parts[0])
            end_months = int(parts[1])
        except ValueError:
            raise ValueError(
                f"Invalid FRA notation: {fra_notation}. "
                f"Expected numeric months like '3x6'"
            )

        if end_months <= fixing_months:
            raise ValueError(
                f"Invalid FRA notation: {fra_notation}. "
                f"End month ({end_months}) must be > fixing month ({fixing_months})"
            )

        # Compute dates
        calendar = Calendar(cal_type)
        fixing_dt_unadj = effective_dt.add_months(fixing_months)
        fixing_dt = calendar.adjust(fixing_dt_unadj, bd_type)

        # Start date is typically T+2 from fixing (spot settlement)
        start_dt = calendar.add_business_days(fixing_dt, 2)

        # End date is at the accrual period maturity
        # Calculate from effective_dt + end_months, then adjust
        end_dt_unadj = effective_dt.add_months(end_months)
        end_dt = calendar.adjust(end_dt_unadj, bd_type)

        return fixing_dt, start_dt, end_dt

###############################################################################

    def position(self, model):
        """
        Create a Position object for this FRA within a model.

        Args:
            model: The model containing curves and market data

        Returns:
            Position object for computing VALUE, DELTA, GAMMA, etc.
        """
        return Position(self, model)

###############################################################################

    def forward_rate(self, value_dt, discount_curve):
        """
        Calculate the forward rate implied by the discount curve.

        The forward rate is the market rate for borrowing over [start_dt, end_dt]
        as of the fixing date.

        Formula:
            Forward_Rate = (DF(start) / DF(end) - 1) / year_frac

        Args:
            value_dt: Valuation date
            discount_curve: Discount curve for rate extraction

        Returns:
            Forward rate (annualized, as decimal)
        """
        if self._end_dt <= value_dt:
            return 0.0

        # Get discount factors at start and end of accrual period
        df_start = discount_curve.df(self._start_dt, self._dc_type)
        df_end = discount_curve.df(self._end_dt, self._dc_type)

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
        Value the FRA on a valuation date.

        FRA Valuation:
        1. Compute forward rate from curve: (DF(start)/DF(end) - 1) / year_frac
        2. Compute payoff: Notional × (Forward_Rate - FRA_Rate) × Year_Frac
        3. Discount to settlement date
        4. Discount to valuation date

        Args:
            value_dt: Valuation date
            discount_curve: Discount curve for PV calculation (primary)
            ois_curve: OIS curve (alternative if discount_curve not provided)
            xccy_discount_curve: Cross-currency discount curve (unused for FRAs)
            spot_fx: FX rate (unused for FRAs)
            collateral_type: Collateral type (unused for FRAs)
            first_fixing_rate: First fixing rate (unused for FRAs)

        Returns:
            Present value of the FRA

        Note:
            - Positive value means FRA buyer benefits (rates rose above FRA rate)
            - Negative value means FRA seller benefits (rates fell below FRA rate)
            - At inception with market FRA rate, value should be ~0
        """
        # Select discount curve
        if discount_curve is None:
            if ois_curve is None:
                raise ValueError("Either discount_curve or ois_curve must be provided")
            discount_curve = ois_curve

        # If settlement date is in the past, FRA has zero value
        if self._settlement_dt <= value_dt:
            return 0.0

        # Compute forward rate from curve
        fwd_rate = self.forward_rate(value_dt, discount_curve)

        # FRA payoff: Notional × (Forward_Rate - FRA_Rate) × Year_Frac
        # This is the settlement amount paid at settlement_dt
        rate_diff = fwd_rate - self._fra_rate
        payoff_at_settlement = self._notional * rate_diff * self._year_frac

        # Discount settlement payment to value date
        df_settlement = discount_curve.df(self._settlement_dt, self._dc_type)
        df_value = discount_curve.df(value_dt, self._dc_type)

        pv = payoff_at_settlement * (df_settlement / df_value)

        return pv

###############################################################################

    def implied_fra_rate(self, value_dt, discount_curve):
        """
        Calculate the FRA rate implied by the discount curve.

        This is the market FRA rate that makes the FRA worth zero.
        It equals the forward rate.

        Args:
            value_dt: Valuation date
            discount_curve: Discount curve for rate calculation

        Returns:
            Implied FRA rate (annualized, as decimal)
        """
        return self.forward_rate(value_dt, discount_curve)

###############################################################################

    def print_details(self):
        """Print FRA details."""
        print("OBJECT TYPE:", type(self).__name__)
        print("Effective Date:", self._effective_dt)
        print("Fixing Date:", self._fixing_dt)
        print("Start Date:", self._start_dt)
        print("End Date:", self._end_dt)
        print("Settlement Date:", self._settlement_dt)
        print("FRA Rate:", f"{self._fra_rate*100:.4f}%")
        print("Notional:", f"{self._notional:,.0f} {self._currency.name}")
        print("Accrual Year Fraction:", f"{self._year_frac:.6f}")
        print("Day Count:", self._dc_type.name)
        print("Floating Index:", self._floating_index.name)

###############################################################################

    def __repr__(self):
        s = label_to_string("OBJECT TYPE", type(self).__name__)
        s += label_to_string("Effective Date", self._effective_dt)
        s += label_to_string("Fixing Date", self._fixing_dt)
        s += label_to_string("Start Date", self._start_dt)
        s += label_to_string("End Date", self._end_dt)
        s += label_to_string("Settlement Date", self._settlement_dt)
        s += label_to_string("FRA Rate", f"{self._fra_rate*100:.4f}%")
        s += label_to_string("Notional", f"{self._notional:,.0f}")
        s += label_to_string("Currency", self._currency.name)
        s += label_to_string("Day Count", self._dc_type.name)
        s += label_to_string("Year Fraction", f"{self._year_frac:.6f}")
        s += label_to_string("Floating Index", self._floating_index.name)
        return s

###############################################################################
