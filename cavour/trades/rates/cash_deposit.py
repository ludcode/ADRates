##############################################################################

##############################################################################

"""
Cash deposit (money market deposit) implementation for short-term funding.

Provides the CashDeposit class for creating, valuing, and analyzing short-term
money market deposits where principal is invested at a fixed rate and returned
with interest at maturity.

Key features:
- Simple interest calculation (not compounded for deposits)
- Support for various day count conventions (ACT/360, ACT/365F)
- Business day adjustments
- Integration with automatic differentiation for Greeks calculation
- Single cashflow at maturity (principal + interest)

Typical maturities: Overnight, 1W, 1M, 2M, 3M, 6M, 9M, 12M

Example:
    >>> # Create a 3M USD deposit at 5.25%
    >>> value_dt = Date(15, 6, 2023)
    >>> deposit = CashDeposit(
    ...     effective_dt=value_dt,
    ...     term_dt_or_tenor="3M",
    ...     deposit_rate=0.0525,
    ...     dc_type=DayCountTypes.ACT_360,
    ...     floating_index=CurveTypes.USD_OIS_SOFR,
    ...     currency=CurrencyTypes.USD,
    ...     notional=1_000_000
    ... )
    >>>
    >>> # Value the deposit
    >>> pv = deposit.value(value_dt, discount_curve)
    >>> par_rate = deposit.implied_rate(value_dt, discount_curve)
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


class CashDeposit:
    """
    Money market deposit instrument with simple interest.

    A cash deposit is a short-term investment where:
    - Principal is invested at effective_dt
    - Interest accrues at a fixed rate using simple interest
    - Principal + interest is returned at maturity_dt

    Cash deposits are used for:
    - Building the short end of OIS curves (0-12M)
    - Anchoring discount curves at short maturities
    - Money market trading and funding

    Market conventions:
    - USD: ACT/360 day count
    - GBP: ACT/365F day count
    - EUR: ACT/360 day count
    """

    def __init__(self,
                 effective_dt: Date,
                 term_dt_or_tenor: (Date, str),
                 deposit_rate: float,
                 dc_type: DayCountTypes,
                 floating_index: CurveTypes,
                 currency: CurrencyTypes,
                 notional: float = ONE_MILLION,
                 cal_type: CalendarTypes = CalendarTypes.WEEKEND,
                 bd_type: BusDayAdjustTypes = BusDayAdjustTypes.MODIFIED_FOLLOWING):
        """
        Create a cash deposit instrument.

        Args:
            effective_dt: Start date when principal is invested
            term_dt_or_tenor: Maturity date or tenor string (e.g., "3M", "6M")
            deposit_rate: Fixed deposit rate (annualized, as decimal, e.g., 0.05 for 5%)
            dc_type: Day count convention (ACT/360 for USD, ACT/365F for GBP)
            floating_index: Index type (determines currency and curve association)
            currency: Currency of the deposit
            notional: Notional amount (default 1,000,000)
            cal_type: Calendar for business day adjustments
            bd_type: Business day adjustment convention
        """

        check_argument_types(self.__init__, locals())

        # Set instrument type
        self.derivative_type = InstrumentTypes.CASH_DEPOSIT

        # Compute maturity date
        if isinstance(term_dt_or_tenor, Date):
            self._maturity_dt = term_dt_or_tenor
        else:
            self._maturity_dt = effective_dt.add_tenor(term_dt_or_tenor)

        # Apply business day adjustment
        calendar = Calendar(cal_type)
        self._maturity_dt = calendar.adjust(self._maturity_dt, bd_type)

        if effective_dt > self._maturity_dt:
            raise LibError("Effective date after maturity date")

        # Store parameters
        self._effective_dt = effective_dt
        self._deposit_rate = deposit_rate
        self._dc_type = dc_type
        self._floating_index = floating_index
        self._currency = currency
        self._notional = notional
        self._cal_type = cal_type
        self._bd_type = bd_type

        # Compute year fraction using day count convention
        dcc = DayCount(dc_type)
        self._year_frac, _, _ = dcc.year_frac(effective_dt, self._maturity_dt)

        # Compute payment amount at maturity: notional * (1 + rate * year_frac)
        # This is the simple interest formula
        self._payment_amt = notional * (1.0 + deposit_rate * self._year_frac)

        # CRITICAL: Store attributes for curve builder compatibility
        # OISCurve expects these attributes to extract instrument data
        self._adjusted_fixed_dts = [self._maturity_dt]  # Single payment date
        self._fixed_coupon = deposit_rate  # The deposit rate
        self._fixed_year_fracs = [self._year_frac]  # Single year fraction
        self._start_dt = effective_dt  # Start date

        # FORWARD-STARTING INSTRUMENT SUPPORT:
        # Deposits are spot-starting (not forward-starting), so the forward period
        # begins immediately at t=0. This is indicated by start_year_frac = 0.0
        self._start_year_fracs = [0.0]  # Spot-starting instrument

        # For compatibility with swap-like interface
        self._termination_dt = self._maturity_dt

###############################################################################

    def position(self, model):
        """
        Create a Position object for this deposit within a model.

        Args:
            model: The model containing curves and market data

        Returns:
            Position object for computing VALUE, DELTA, GAMMA, etc.
        """
        return Position(self, model)

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
        Value the cash deposit on a valuation date.

        The deposit value is the PV of the maturity payment minus the invested principal.
        At inception (value_dt = effective_dt), this should be approximately zero
        for a market deposit rate.

        Args:
            value_dt: Valuation date
            discount_curve: Discount curve for PV calculation (primary)
            ois_curve: OIS curve (alternative if discount_curve not provided)
            xccy_discount_curve: Cross-currency discount curve (unused for deposits)
            spot_fx: FX rate (unused for deposits)
            collateral_type: Collateral type (unused for deposits)
            first_fixing_rate: First fixing rate (unused for deposits)

        Returns:
            Present value of the deposit

        Note:
            For curve bootstrapping, the deposit should reprice to ~0 when valued
            on its own curve at the effective date.
        """
        # Select discount curve (prefer explicit discount_curve, fallback to ois_curve)
        if discount_curve is None:
            if ois_curve is None:
                raise ValueError("Either discount_curve or ois_curve must be provided")
            discount_curve = ois_curve

        # If maturity is in the past, deposit has zero value
        if self._maturity_dt <= value_dt:
            return 0.0

        # Get discount factors
        df_maturity = discount_curve.df(self._maturity_dt, self._dc_type)
        df_value = discount_curve.df(value_dt, self._dc_type)

        # Present value of maturity payment
        pv_payment = self._payment_amt * (df_maturity / df_value)

        # Deposit value = PV(payment) - notional invested
        # If value_dt == effective_dt, this should be ≈0 for market rates
        if value_dt <= self._effective_dt:
            # Before or at inception: PV(payment) - notional
            value = pv_payment - self._notional
        else:
            # After inception: PV(payment) only (notional already invested)
            value = pv_payment

        return value

###############################################################################

    def implied_rate(self, value_dt, discount_curve):
        """
        Calculate the deposit rate implied by the discount curve.

        This is the market deposit rate that makes the deposit worth zero.

        Args:
            value_dt: Valuation date
            discount_curve: Discount curve for rate calculation

        Returns:
            Implied deposit rate (annualized, as decimal)
        """
        if self._maturity_dt <= value_dt:
            return 0.0

        # Get discount factors
        df_effective = discount_curve.df(self._effective_dt, self._dc_type)
        df_maturity = discount_curve.df(self._maturity_dt, self._dc_type)

        # Implied rate: (DF_start / DF_end - 1) / year_frac
        implied_rate = (df_effective / df_maturity - 1.0) / self._year_frac

        return implied_rate

###############################################################################

    def print_details(self):
        """Print deposit details."""
        print("OBJECT TYPE:", type(self).__name__)
        print("Effective Date:", self._effective_dt)
        print("Maturity Date:", self._maturity_dt)
        print("Deposit Rate:", f"{self._deposit_rate*100:.4f}%")
        print("Notional:", f"{self._notional:,.0f} {self._currency.name}")
        print("Year Fraction:", f"{self._year_frac:.6f}")
        print("Payment at Maturity:", f"{self._payment_amt:,.2f}")
        print("Day Count:", self._dc_type.name)

###############################################################################

    def __repr__(self):
        s = label_to_string("OBJECT TYPE", type(self).__name__)
        s += label_to_string("Effective Date", self._effective_dt)
        s += label_to_string("Maturity Date", self._maturity_dt)
        s += label_to_string("Deposit Rate", f"{self._deposit_rate*100:.4f}%")
        s += label_to_string("Notional", f"{self._notional:,.0f}")
        s += label_to_string("Currency", self._currency.name)
        s += label_to_string("Day Count", self._dc_type.name)
        s += label_to_string("Year Fraction", f"{self._year_frac:.6f}")
        s += label_to_string("Payment at Maturity", f"{self._payment_amt:,.2f}")
        return s

###############################################################################
