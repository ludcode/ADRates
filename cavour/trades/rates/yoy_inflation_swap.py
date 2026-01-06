"""
Year-on-Year Inflation Swap implementation.

Provides the YoYInflationSwap class for valuing periodic inflation swaps where
a fixed rate is exchanged for year-on-year inflation rates.

Key features:
- Periodic payments (Annual, Semi-annual, Quarterly, Monthly)
- Fixed leg pays: Notional × year_frac × fixed_rate
- Inflation leg pays: Notional × year_frac × (YoY_inflation_rate + spread)
- YoY inflation: [I(t) / I(t-1y)] - 1
- Breakeven rate calculation
- Support for inflation spreads
- PV01 sensitivity computation

Example:
    >>> # Create a 5Y GBP RPI YoY inflation swap
    >>> value_dt = Date(15, 6, 2023)
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
    >>> # Create YoY IIS (receive fixed, pay inflation)
    >>> yoy_iis = YoYInflationSwap(
    ...     effective_dt=value_dt,
    ...     term_dt_or_tenor="5Y",
    ...     fixed_leg_type=SwapTypes.RECEIVE,
    ...     fixed_rate=0.03,  # 3% annual
    ...     inflation_index=rpi_index,
    ...     freq_type=FrequencyTypes.ANNUAL,
    ...     notional=10_000_000
    ... )
    >>>
    >>> # Value the swap
    >>> pv = yoy_iis.value(value_dt, discount_curve, inflation_curve)
    >>> breakeven = yoy_iis.breakeven_rate(value_dt, discount_curve, inflation_curve)
"""

import numpy as np
from cavour.utils.error import LibError
from cavour.utils.date import Date
from cavour.utils.math import ONE_MILLION
from cavour.utils.day_count import DayCount, DayCountTypes
from cavour.utils.frequency import FrequencyTypes
from cavour.utils.calendar import CalendarTypes, BusDayAdjustTypes, DateGenRuleTypes, Calendar
from cavour.utils.helpers import check_argument_types, label_to_string, format_table
from cavour.utils.global_types import SwapTypes, InstrumentTypes, CurveTypes
from cavour.utils.currency import CurrencyTypes
from cavour.market.curves.discount_curve import DiscountCurve
from cavour.market.indices.inflation_index import InflationIndex
from .swap_fixed_leg import SwapFixedLeg
from .swap_yoy_inflation_leg import SwapYoYInflationLeg

###############################################################################


class YoYInflationSwap:
    """
    Year-on-Year Inflation Swap (YoY IIS) contract.

    A YoY IIS exchanges periodic fixed-rate payments for periodic payments
    linked to year-on-year inflation rates. Unlike zero-coupon inflation swaps
    which have a single payment at maturity, YoY IIS have regular periodic
    payments throughout the life of the swap.

    Structure:
    - Fixed leg: Pays Notional × year_frac × fixed_rate at each period
    - Inflation leg: Pays Notional × year_frac × (YoY_rate + spread) at each period
    - YoY rate: [I(t) / I(t-1y)] - 1

    The swap is typically entered at zero cost (breakeven rate).

    Market conventions:
    - UK: RPI-linked, annual frequency, ACT/365F day count, 3-month lag
    - US: CPI-U-linked, annual frequency, ACT/365F day count, 3-month lag
    - EUR: HICP-linked, annual frequency, ACT/365F day count, 3-month lag
    """

    def __init__(self,
                 effective_dt: Date,
                 term_dt_or_tenor: (Date, str),
                 fixed_leg_type: SwapTypes,
                 fixed_rate: float,
                 inflation_index: InflationIndex,
                 freq_type: FrequencyTypes,
                 notional: float = ONE_MILLION,
                 inflation_spread: float = 0.0,
                 dc_type: DayCountTypes = DayCountTypes.ACT_365F,
                 payment_lag: int = 0,
                 cal_type: CalendarTypes = CalendarTypes.WEEKEND,
                 bd_type: BusDayAdjustTypes = BusDayAdjustTypes.FOLLOWING,
                 dg_type: DateGenRuleTypes = DateGenRuleTypes.BACKWARD,
                 end_of_month: bool = False):
        """
        Create a year-on-year inflation swap.

        Args:
            effective_dt: Start date (base date for inflation calculation)
            term_dt_or_tenor: Maturity date or tenor (e.g., "5Y")
            fixed_leg_type: PAY or RECEIVE for fixed leg
            fixed_rate: Fixed annual rate (e.g., 0.03 for 3%)
            inflation_index: InflationIndex object for CPI lookups
            freq_type: Payment frequency (ANNUAL, SEMI_ANNUAL, QUARTERLY, MONTHLY)
            notional: Notional amount
            inflation_spread: Spread over YoY inflation rate (e.g., 0.01 for +1%)
            dc_type: Day count convention for accrual
            payment_lag: Days after accrual end for payment (default 0)
            cal_type: Calendar for business day adjustments
            bd_type: Business day adjustment convention
            dg_type: Date generation rule (BACKWARD, FORWARD)
            end_of_month: Whether to use end-of-month adjustment

        Example:
            >>> yoy_iis = YoYInflationSwap(
            ...     effective_dt=Date(15, 6, 2023),
            ...     term_dt_or_tenor="5Y",
            ...     fixed_leg_type=SwapTypes.RECEIVE,
            ...     fixed_rate=0.03,
            ...     inflation_index=rpi_index,
            ...     freq_type=FrequencyTypes.ANNUAL,
            ...     notional=10_000_000
            ... )
        """
        check_argument_types(self.__init__, locals())

        self.instrument_type = InstrumentTypes.YOY_INFLATION_SWAP
        self.derivative_type = InstrumentTypes.YOY_INFLATION_SWAP

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
        self._freq_type = freq_type
        self._notional = notional
        self._inflation_spread = inflation_spread
        self._dc_type = dc_type
        self._payment_lag = payment_lag
        self._cal_type = cal_type
        self._bd_type = bd_type
        self._dg_type = dg_type
        self._end_of_month = end_of_month

        # Determine inflation leg type (opposite of fixed leg)
        if fixed_leg_type == SwapTypes.PAY:
            inflation_leg_type = SwapTypes.RECEIVE
        else:
            inflation_leg_type = SwapTypes.PAY

        # Get currency from inflation index
        currency = inflation_index._currency

        # Determine floating index based on currency (for fixed leg compatibility)
        # Note: This is a convention mapping - the fixed leg doesn't actually use it
        if currency == CurrencyTypes.GBP:
            floating_index = CurveTypes.GBP_OIS_SONIA
        elif currency == CurrencyTypes.USD:
            floating_index = CurveTypes.USD_OIS_SOFR
        elif currency == CurrencyTypes.EUR:
            floating_index = CurveTypes.EUR_OIS_ESTR
        else:
            # Default to USD convention
            floating_index = CurveTypes.USD_OIS_SOFR

        # Create fixed leg
        self._fixed_leg = SwapFixedLeg(
            effective_dt=effective_dt,
            end_dt=self._termination_dt,
            leg_type=fixed_leg_type,
            coupon=fixed_rate,
            freq_type=freq_type,
            dc_type=dc_type,
            floating_index=floating_index,
            currency=currency,
            notional=notional,
            payment_lag=payment_lag,
            cal_type=cal_type,
            bd_type=bd_type,
            dg_type=dg_type,
            end_of_month=end_of_month
        )

        # Create YoY inflation leg
        self._inflation_leg = SwapYoYInflationLeg(
            effective_dt=effective_dt,
            end_dt=self._termination_dt,
            leg_type=inflation_leg_type,
            inflation_index=inflation_index,
            freq_type=freq_type,
            dc_type=dc_type,
            notional=notional,
            spread=inflation_spread,
            payment_lag=payment_lag,
            cal_type=cal_type,
            bd_type=bd_type,
            dg_type=dg_type,
            end_of_month=end_of_month
        )

        # Valuation results (populated by value())
        self._fixed_pv = None
        self._inflation_pv = None

###############################################################################

    def value(self,
              value_dt: Date,
              discount_curve: DiscountCurve,
              inflation_curve=None) -> float:
        """
        Value the year-on-year inflation swap.

        Calculates the present value of both legs and returns the net PV.

        Process:
        1. Value fixed leg via SwapFixedLeg
        2. Value inflation leg via SwapYoYInflationLeg
        3. Return net PV (fixed + inflation)

        Args:
            value_dt: Valuation date
            discount_curve: Curve for discounting cashflows
            inflation_curve: Curve for projecting future CPI (optional)

        Returns:
            Present value of the swap in natural currency

        Example:
            >>> pv = yoy_iis.value(value_dt, ois_curve, inflation_curve)
        """
        # Value fixed leg
        self._fixed_pv = self._fixed_leg.value(value_dt, discount_curve)

        # Value inflation leg
        self._inflation_pv = self._inflation_leg.value(
            value_dt,
            discount_curve,
            inflation_curve
        )

        # Net PV
        return self._fixed_pv + self._inflation_pv

###############################################################################

    def breakeven_rate(self,
                      value_dt: Date,
                      discount_curve: DiscountCurve,
                      inflation_curve=None) -> float:
        """
        Calculate the breakeven fixed rate.

        The breakeven rate is the fixed rate that would make the swap worth zero.
        It's calculated by solving for the fixed rate where:
        PV(fixed leg) + PV(inflation leg) = 0

        Since PV(fixed leg) = fixed_rate × annuity, we can solve:
        breakeven_rate = -PV(inflation leg) / annuity

        Args:
            value_dt: Valuation date
            discount_curve: Curve for discounting
            inflation_curve: Curve for projecting CPI

        Returns:
            Breakeven fixed rate (e.g., 0.03 for 3%)

        Example:
            >>> breakeven = yoy_iis.breakeven_rate(value_dt, ois_curve, infl_curve)
            >>> print(f"Breakeven rate: {breakeven*100:.2f}%")
        """
        # Value the inflation leg to get its PV
        inflation_pv = self._inflation_leg.value(
            value_dt,
            discount_curve,
            inflation_curve
        )

        # Calculate annuity (sum of discounted year fractions)
        # This is the PV01 of the fixed leg (before multiplying by rate)
        annuity = 0.0

        for i in range(len(self._fixed_leg._payment_dts)):
            payment_dt = self._fixed_leg._payment_dts[i]

            if payment_dt <= value_dt:
                continue  # Skip past payments

            # Get discount factor
            df_value = discount_curve.df(value_dt, DayCountTypes.ACT_365F)
            df_payment = discount_curve.df(payment_dt, DayCountTypes.ACT_365F)
            df = df_payment / df_value

            # Add to annuity
            year_frac = self._fixed_leg._year_fracs[i]
            annuity += year_frac * df

        if annuity <= 0:
            raise LibError("Annuity must be positive for breakeven calculation")

        # Calculate breakeven rate
        # For PAY fixed: PV_fixed = -notional × rate × annuity
        # For RECEIVE fixed: PV_fixed = +notional × rate × annuity
        # We want: PV_fixed + PV_inflation = 0
        # So: ±notional × rate × annuity = -PV_inflation

        if self._fixed_leg_type == SwapTypes.PAY:
            # PV_fixed = -notional × rate × annuity
            # -notional × rate × annuity + PV_inflation = 0
            # rate = PV_inflation / (notional × annuity)
            breakeven = inflation_pv / (self._notional * annuity)
        else:
            # PV_fixed = +notional × rate × annuity
            # notional × rate × annuity + PV_inflation = 0
            # rate = -PV_inflation / (notional × annuity)
            breakeven = -inflation_pv / (self._notional * annuity)

        return breakeven

###############################################################################

    def pv01(self, value_dt: Date, discount_curve: DiscountCurve) -> float:
        """
        Calculate PV01 - the value of 1bp change in fixed rate.

        Returns:
            PV01 value (always positive)

        Example:
            >>> pv01 = yoy_iis.pv01(value_dt, discount_curve)
        """
        # Calculate annuity (sum of discounted year fractions)
        annuity = 0.0

        for i in range(len(self._fixed_leg._payment_dts)):
            payment_dt = self._fixed_leg._payment_dts[i]

            if payment_dt <= value_dt:
                continue  # Skip past payments

            # Get discount factor
            df_value = discount_curve.df(value_dt, DayCountTypes.ACT_365F)
            df_payment = discount_curve.df(payment_dt, DayCountTypes.ACT_365F)
            df = df_payment / df_value

            # Add to annuity
            year_frac = self._fixed_leg._year_fracs[i]
            annuity += year_frac * df

        # PV01 = notional × annuity × 0.0001
        pv01 = abs(self._notional * annuity * 0.0001)

        return pv01

###############################################################################

    def print_payments(self):
        """
        Print swap payment schedule.

        Shows both fixed and inflation legs with payment dates and amounts.
        """
        print("="*80)
        print("YEAR-ON-YEAR INFLATION SWAP PAYMENTS")
        print("="*80)
        print("START DATE:", self._effective_dt)
        print("MATURITY DATE:", self._maturity_dt)
        print("NOTIONAL:", f"{self._notional:,.2f}")
        print("FIXED RATE:", f"{self._fixed_rate*100:.4f}%")
        print("FREQUENCY:", self._freq_type.name)
        print("DAY COUNT:", self._dc_type.name)
        print()

        print("FIXED LEG:")
        print(f"  Leg Type: {self._fixed_leg_type.name}")
        print(f"  Number of Payments: {len(self._fixed_leg._payment_dts)}")
        print()

        # Print fixed leg payment schedule
        fixed_header = ["Payment #", "Accrual Start", "Accrual End", "Payment Date", "Year Frac", "Rate", "Payment"]
        fixed_rows = []

        for i in range(len(self._fixed_leg._payment_dts)):
            fixed_rows.append([
                str(i + 1),
                str(self._fixed_leg._start_accrued_dts[i]),
                str(self._fixed_leg._end_accrued_dts[i]),
                str(self._fixed_leg._payment_dts[i]),
                f"{self._fixed_leg._year_fracs[i]:.6f}",
                f"{self._fixed_leg._rates[i]*100:.4f}%",
                f"{self._fixed_leg._payments[i]:,.2f}"
            ])

        print(format_table(fixed_header, fixed_rows))
        print()

        print("INFLATION LEG:")
        print(f"  Leg Type: {self._inflation_leg._leg_type.name}")
        print(f"  Number of Payments: {len(self._inflation_leg._payment_dts)}")
        print(f"  Inflation Spread: {self._inflation_spread*100:.4f}%")
        print()

        # Print inflation leg payment schedule
        self._inflation_leg.print_payments()

###############################################################################

    def print_valuation(self):
        """
        Print swap valuation details.

        Shows PV breakdown by leg and net PV.
        """
        if self._fixed_pv is None or self._inflation_pv is None:
            print("\nValuation not yet performed. Call value() first.")
            return

        print("="*80)
        print("YEAR-ON-YEAR INFLATION SWAP VALUATION")
        print("="*80)
        print("START DATE:", self._effective_dt)
        print("MATURITY DATE:", self._maturity_dt)
        print("NOTIONAL:", f"{self._notional:,.2f}")
        print("FREQUENCY:", self._freq_type.name)
        print()

        header = ["Leg", "Type", "PV"]
        rows = [
            [
                "Fixed",
                self._fixed_leg_type.name,
                f"{self._fixed_pv:,.2f}"
            ],
            [
                "Inflation (YoY)",
                self._inflation_leg._leg_type.name,
                f"{self._inflation_pv:,.2f}"
            ],
            ["", "", ""],
            [
                "NET PV",
                "",
                f"{self._fixed_pv + self._inflation_pv:,.2f}"
            ]
        ]

        table = format_table(header, rows)
        print(table)

        # Print detailed leg valuations
        print()
        print("DETAILED INFLATION LEG VALUATION:")
        print("-"*80)
        self._inflation_leg.print_valuation()

###############################################################################

    def __repr__(self):
        s = label_to_string("OBJECT TYPE", type(self).__name__)
        s += label_to_string("START DATE", self._effective_dt)
        s += label_to_string("TERMINATION DATE", self._termination_dt)
        s += label_to_string("MATURITY DATE", self._maturity_dt)
        s += label_to_string("NOTIONAL", self._notional)
        s += label_to_string("FIXED LEG TYPE", self._fixed_leg_type)
        s += label_to_string("FIXED RATE", f"{self._fixed_rate*100:.4f}%")
        s += label_to_string("INFLATION SPREAD", f"{self._inflation_spread*100:.4f}%")
        s += label_to_string("FREQUENCY", self._freq_type)
        s += label_to_string("DAY COUNT", self._dc_type)
        s += label_to_string("INFLATION INDEX", self._inflation_index._index_type)
        s += label_to_string("INDEX LAG (MONTHS)", self._inflation_index._lag_months)
        return s

###############################################################################

    def _print(self):
        """Print swap details."""
        print(self)

###############################################################################
