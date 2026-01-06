"""
Zero-Coupon Inflation Swap (ZCIS) implementation.

Provides the ZeroCouponInflationSwap class for valuing inflation swaps where
a fixed return is exchanged for inflation-linked return at maturity.

Key features:
- Single payment at maturity (zero-coupon structure)
- Fixed leg pays: Notional × [(1 + fixed_rate)^T - 1]
- Inflation leg pays: Notional × [I(T-lag) / I(Base-lag) - 1]
- Breakeven inflation rate calculation
- Support for cross-currency collateral discounting
- PV01 sensitivity computation

Example:
    >>> # Create a 5Y GBP RPI inflation swap
    >>> value_dt = Date(15, 6, 2023)
    >>> maturity_dt = Date(15, 6, 2028)
    >>>
    >>> # Create RPI index
    >>> rpi_index = InflationIndex(
    ...     index_type=InflationIndexTypes.UK_RPI,
    ...     base_date=Date(1, 3, 2023),
    ...     base_index=293.0,
    ...     currency=CurrencyTypes.GBP,
    ...     lag_months=3
    ... )
    >>>
    >>> # Create ZCIS
    >>> zcis = ZeroCouponInflationSwap(
    ...     effective_dt=value_dt,
    ...     term_dt_or_tenor="5Y",
    ...     fixed_leg_type=SwapTypes.PAY,
    ...     fixed_rate=0.03,  # 3% annual compounded
    ...     inflation_index=rpi_index,
    ...     notional=10_000_000
    ... )
    >>>
    >>> # Value the swap
    >>> pv = zcis.value(value_dt, discount_curve, inflation_curve)
    >>> breakeven = zcis.breakeven_inflation_rate(value_dt, discount_curve, inflation_curve)
"""

import numpy as np
from cavour.utils.error import LibError
from cavour.utils.date import Date
from cavour.utils.math import ONE_MILLION
from cavour.utils.day_count import DayCount, DayCountTypes
from cavour.utils.calendar import CalendarTypes, BusDayAdjustTypes, Calendar
from cavour.utils.helpers import check_argument_types, label_to_string, format_table
from cavour.utils.global_types import SwapTypes, InstrumentTypes
from cavour.utils.currency import CurrencyTypes
from cavour.market.curves.discount_curve import DiscountCurve
from cavour.market.indices.inflation_index import InflationIndex
from .swap_inflation_leg import SwapInflationLeg

###############################################################################


class ZeroCouponInflationSwap:
    """
    Zero-Coupon Inflation Swap (ZCIS) contract.

    A ZCIS exchanges a fixed compounded return for an inflation-linked return
    at a single maturity date. Unlike standard inflation swaps with periodic
    payments, the ZCIS has only one payment at maturity.

    Structure:
    - Fixed leg: Pays Notional × [(1 + r)^T - 1] at maturity
    - Inflation leg: Pays Notional × [I(T-lag) / I(Base-lag) - 1] at maturity

    The swap is typically entered at zero cost (breakeven rate).

    Market conventions:
    - UK: RPI-linked, ACT/365F day count, 3-month lag
    - US: CPI-U-linked, ACT/365F day count, 3-month lag
    - EUR: HICP-linked, ACT/365F day count, 3-month lag
    """

    def __init__(self,
                 effective_dt: Date,
                 term_dt_or_tenor: (Date, str),
                 fixed_leg_type: SwapTypes,
                 fixed_rate: float,
                 inflation_index: InflationIndex,
                 notional: float = ONE_MILLION,
                 payment_lag: int = 0,
                 dc_type: DayCountTypes = DayCountTypes.ACT_365F,
                 cal_type: CalendarTypes = CalendarTypes.WEEKEND,
                 bd_type: BusDayAdjustTypes = BusDayAdjustTypes.FOLLOWING):
        """
        Create a zero-coupon inflation swap.

        Args:
            effective_dt: Start date (base date for inflation calculation)
            term_dt_or_tenor: Maturity date or tenor (e.g., "5Y")
            fixed_leg_type: PAY or RECEIVE for fixed leg
            fixed_rate: Fixed annual rate (e.g., 0.03 for 3%)
            inflation_index: InflationIndex object for CPI lookups
            notional: Notional amount
            payment_lag: Days after maturity for payment (default 0)
            dc_type: Day count convention for fixed leg accrual
            cal_type: Calendar for business day adjustments
            bd_type: Business day adjustment convention

        Example:
            >>> zcis = ZeroCouponInflationSwap(
            ...     effective_dt=Date(15, 6, 2023),
            ...     term_dt_or_tenor="5Y",
            ...     fixed_leg_type=SwapTypes.PAY,
            ...     fixed_rate=0.03,
            ...     inflation_index=rpi_index,
            ...     notional=10_000_000
            ... )
        """
        check_argument_types(self.__init__, locals())

        self.instrument_type = InstrumentTypes.ZCIS

        # Handle maturity date
        if isinstance(term_dt_or_tenor, Date):
            self._termination_dt = term_dt_or_tenor
        else:
            self._termination_dt = effective_dt.add_tenor(term_dt_or_tenor)

        # Adjust maturity for business days
        calendar = Calendar(cal_type)
        self._maturity_dt = calendar.adjust(self._termination_dt, bd_type)

        if effective_dt > self._maturity_dt:
            raise LibError("Start date after maturity date")

        self._effective_dt = effective_dt
        self._fixed_leg_type = fixed_leg_type
        self._fixed_rate = fixed_rate
        self._inflation_index = inflation_index
        self._notional = notional
        self._payment_lag = payment_lag
        self._dc_type = dc_type
        self._cal_type = cal_type
        self._bd_type = bd_type

        # Calculate payment date
        if payment_lag == 0:
            self._payment_dt = self._maturity_dt
        else:
            self._payment_dt = calendar.add_business_days(self._maturity_dt, payment_lag)

        # Determine inflation leg type (opposite of fixed leg)
        if fixed_leg_type == SwapTypes.PAY:
            inflation_leg_type = SwapTypes.RECEIVE
        else:
            inflation_leg_type = SwapTypes.PAY

        # Create inflation leg
        self._inflation_leg = SwapInflationLeg(
            effective_dt=effective_dt,
            end_dt=self._termination_dt,
            leg_type=inflation_leg_type,
            inflation_index=inflation_index,
            notional=notional,
            payment_lag=payment_lag,
            cal_type=cal_type,
            bd_type=bd_type
        )

        # Valuation results (populated by value())
        self._fixed_return = None
        self._fixed_payment = None
        self._fixed_pv = None
        self._inflation_pv = None
        self._payment_df = None

###############################################################################

    def value(self,
              value_dt: Date,
              discount_curve: DiscountCurve,
              inflation_curve=None) -> float:
        """
        Value the zero-coupon inflation swap.

        Calculates the present value of both legs and returns the net PV.

        Process:
        1. Calculate fixed leg payment: Notional × [(1 + r)^T - 1]
        2. Calculate inflation leg payment via SwapInflationLeg
        3. Discount both to valuation date
        4. Return net PV (fixed + inflation)

        Args:
            value_dt: Valuation date
            discount_curve: Curve for discounting cashflows
            inflation_curve: Curve for projecting future CPI (optional)

        Returns:
            Present value of the swap in natural currency

        Example:
            >>> pv = zcis.value(value_dt, ois_curve, inflation_curve)
        """
        # Calculate year fraction for fixed leg
        day_counter = DayCount(self._dc_type)
        (year_frac, num_days, _) = day_counter.year_frac(
            self._effective_dt,
            self._maturity_dt
        )

        # Calculate fixed return: (1 + r)^T - 1
        self._fixed_return = ((1.0 + self._fixed_rate) ** year_frac) - 1.0

        # Calculate fixed payment
        self._fixed_payment = self._notional * self._fixed_return

        # Discount fixed payment
        if self._payment_dt > value_dt:
            df_value = discount_curve.df(value_dt, DayCountTypes.ACT_365F)
            df_payment = discount_curve.df(self._payment_dt, DayCountTypes.ACT_365F)
            self._payment_df = df_payment / df_value
            self._fixed_pv = self._fixed_payment * self._payment_df
        else:
            # Payment in the past
            self._payment_df = 0.0
            self._fixed_pv = 0.0

        # Apply fixed leg direction
        if self._fixed_leg_type == SwapTypes.PAY:
            self._fixed_pv *= -1.0

        # Value inflation leg
        self._inflation_pv = self._inflation_leg.value(
            value_dt,
            discount_curve,
            inflation_curve
        )

        # Net PV
        return self._fixed_pv + self._inflation_pv

###############################################################################

    def breakeven_inflation_rate(self,
                                 value_dt: Date,
                                 discount_curve: DiscountCurve,
                                 inflation_curve=None) -> float:
        """
        Calculate the breakeven annual inflation rate.

        The breakeven rate is the constant annual inflation rate that would
        make the swap worth zero. It's calculated by comparing the fixed
        compounded return with the projected inflation return.

        Returns:
            Annual breakeven inflation rate (e.g., 0.03 for 3%)

        Example:
            >>> breakeven = zcis.breakeven_inflation_rate(value_dt, ois_curve, infl_curve)
            >>> print(f"Breakeven inflation: {breakeven*100:.2f}%")
        """
        # Value the inflation leg to get projected inflation return
        self._inflation_leg.value(value_dt, discount_curve, inflation_curve)
        inflation_return = self._inflation_leg._inflation_return

        # Calculate year fraction
        day_counter = DayCount(self._dc_type)
        (year_frac, num_days, _) = day_counter.year_frac(
            self._effective_dt,
            self._maturity_dt
        )

        # Solve: (1 + breakeven)^T = 1 + inflation_return
        # breakeven = (1 + inflation_return)^(1/T) - 1
        if year_frac <= 0:
            raise LibError("Year fraction must be positive")

        if inflation_return <= -1.0:
            raise LibError(f"Inflation return too negative: {inflation_return}")

        breakeven_rate = ((1.0 + inflation_return) ** (1.0 / year_frac)) - 1.0
        return breakeven_rate

###############################################################################

    def pv01(self, value_dt: Date, discount_curve: DiscountCurve) -> float:
        """
        Calculate PV01 - the value of 1bp change in fixed rate.

        Returns:
            PV01 value (always positive)

        Example:
            >>> pv01 = zcis.pv01(value_dt, discount_curve)
        """
        # Calculate year fraction
        day_counter = DayCount(self._dc_type)
        (year_frac, num_days, _) = day_counter.year_frac(
            self._effective_dt,
            self._maturity_dt
        )

        # Get discount factor
        if self._payment_dt > value_dt:
            df_value = discount_curve.df(value_dt, DayCountTypes.ACT_365F)
            df_payment = discount_curve.df(self._payment_dt, DayCountTypes.ACT_365F)
            df = df_payment / df_value
        else:
            df = 0.0

        # PV01 = dPV/d(rate) × 0.0001
        # For fixed payment = N × [(1+r)^T - 1]
        # dPV/dr = N × T × (1+r)^(T-1) × df
        # PV01 = N × T × (1+r)^(T-1) × df × 0.0001

        dpv_dr = self._notional * year_frac * ((1.0 + self._fixed_rate) ** (year_frac - 1.0)) * df
        pv01 = abs(dpv_dr) * 0.0001  # 1bp = 0.0001

        return pv01

###############################################################################

    def print_payments(self):
        """
        Print swap payment schedule.

        Shows both fixed and inflation legs with payment dates and amounts.
        """
        print("="*80)
        print("ZERO-COUPON INFLATION SWAP PAYMENTS")
        print("="*80)
        print("START DATE:", self._effective_dt)
        print("MATURITY DATE:", self._maturity_dt)
        print("PAYMENT DATE:", self._payment_dt)
        print("NOTIONAL:", f"{self._notional:,.2f}")
        print("FIXED RATE:", f"{self._fixed_rate*100:.4f}%")
        print()

        # Calculate year fraction
        day_counter = DayCount(self._dc_type)
        (year_frac, num_days, _) = day_counter.year_frac(
            self._effective_dt,
            self._maturity_dt
        )

        print("FIXED LEG:")
        print(f"  Leg Type: {self._fixed_leg_type.name}")
        print(f"  Year Fraction: {year_frac:.6f} ({num_days} days)")
        print(f"  Fixed Return: {((1.0 + self._fixed_rate)**year_frac - 1.0)*100:.6f}%")

        if self._fixed_payment is not None:
            print(f"  Fixed Payment: {self._fixed_payment:,.2f}")

        print()
        print("INFLATION LEG:")
        self._inflation_leg.print_payments()

###############################################################################

    def print_valuation(self):
        """
        Print swap valuation details.

        Shows PV breakdown by leg, returns, and discount factors.
        """
        if self._fixed_pv is None:
            print("\nValuation not yet performed. Call value() first.")
            return

        print("="*80)
        print("ZERO-COUPON INFLATION SWAP VALUATION")
        print("="*80)
        print("START DATE:", self._effective_dt)
        print("MATURITY DATE:", self._maturity_dt)
        print("PAYMENT DATE:", self._payment_dt)
        print("NOTIONAL:", f"{self._notional:,.2f}")
        print()

        # Calculate year fraction
        day_counter = DayCount(self._dc_type)
        (year_frac, num_days, _) = day_counter.year_frac(
            self._effective_dt,
            self._maturity_dt
        )

        header = ["Leg", "Type", "Return", "Payment", "DF", "PV"]
        rows = [
            [
                "Fixed",
                self._fixed_leg_type.name,
                f"{self._fixed_return*100:.6f}%",
                f"{abs(self._fixed_payment):,.2f}",
                f"{self._payment_df:.6f}",
                f"{self._fixed_pv:,.2f}"
            ],
            [
                "Inflation",
                self._inflation_leg._leg_type.name,
                f"{self._inflation_leg._inflation_return*100:.6f}%",
                f"{abs(self._inflation_leg._payment_amount):,.2f}",
                f"{self._inflation_leg._payment_df:.6f}",
                f"{self._inflation_pv:,.2f}"
            ],
            ["", "", "", "", "", ""],
            [
                "NET PV",
                "",
                "",
                "",
                "",
                f"{self._fixed_pv + self._inflation_pv:,.2f}"
            ]
        ]

        table = format_table(header, rows)
        print(table)

###############################################################################

    def __repr__(self):
        s = label_to_string("OBJECT TYPE", type(self).__name__)
        s += label_to_string("START DATE", self._effective_dt)
        s += label_to_string("TERMINATION DATE", self._termination_dt)
        s += label_to_string("MATURITY DATE", self._maturity_dt)
        s += label_to_string("PAYMENT DATE", self._payment_dt)
        s += label_to_string("NOTIONAL", self._notional)
        s += label_to_string("FIXED LEG TYPE", self._fixed_leg_type)
        s += label_to_string("FIXED RATE", f"{self._fixed_rate*100:.4f}%")
        s += label_to_string("INFLATION INDEX", self._inflation_index._index_type)
        s += label_to_string("INDEX LAG (MONTHS)", self._inflation_index._lag_months)
        s += label_to_string("DAY COUNT", self._dc_type)
        return s

###############################################################################

    def _print(self):
        """Print swap details."""
        print(self)

###############################################################################
