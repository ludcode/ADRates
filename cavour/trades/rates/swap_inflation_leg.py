"""
Inflation leg implementation for zero-coupon inflation swaps.

Provides the SwapInflationLeg class for managing inflation-linked payments
in swap contracts. Supports CPI-linked cashflows with publication lag,
index interpolation, and forward projection.

Key features:
- Single payment at maturity (zero-coupon structure)
- Inflation index ratio calculation: I(T-lag) / I(Base-lag)
- Publication lag handling (typically 3 months)
- Integration with InflationIndex for fixings and interpolation
- Forward CPI projection from inflation curves
- Business day adjustments and calendar handling

Used as a component in zero-coupon inflation swaps (ZCIS) where the
inflation leg pays based on cumulative CPI growth.

Example:
    >>> # Create inflation leg for 5Y ZCIS
    >>> effective = Date(15, 6, 2023)
    >>> maturity = Date(15, 6, 2028)
    >>>
    >>> # Create inflation index
    >>> rpi_index = InflationIndex(
    ...     index_type=InflationIndexTypes.UK_RPI,
    ...     base_date=Date(1, 3, 2023),
    ...     base_index=125.4,
    ...     currency=CurrencyTypes.GBP,
    ...     lag_months=3
    ... )
    >>>
    >>> # Create inflation leg
    >>> infl_leg = SwapInflationLeg(
    ...     effective_dt=effective,
    ...     end_dt=maturity,
    ...     leg_type=SwapTypes.RECEIVE,
    ...     inflation_index=rpi_index,
    ...     notional=1_000_000
    ... )
    >>>
    >>> # Value the leg
    >>> pv = infl_leg.value(effective, discount_curve, inflation_curve)
    >>> infl_leg.print_valuation()
"""

from cavour.utils.error import LibError
from cavour.utils.date import Date
from cavour.utils.math import ONE_MILLION
from cavour.utils.day_count import DayCountTypes
from cavour.utils.calendar import CalendarTypes, DateGenRuleTypes
from cavour.utils.calendar import Calendar, BusDayAdjustTypes
from cavour.utils.helpers import format_table, label_to_string, check_argument_types
from cavour.utils.global_types import SwapTypes, InstrumentTypes
from cavour.market.curves.discount_curve import DiscountCurve
from cavour.market.indices.inflation_index import InflationIndex

###############################################################################


class SwapInflationLeg:
    """
    Inflation-linked leg of a zero-coupon inflation swap.

    An inflation leg consists of a single payment at maturity based on the
    cumulative inflation between the base date and maturity date. The payment
    is calculated as:

        Payment = Notional × [I(T-lag) / I(Base-lag) - 1]

    where:
    - I(T-lag) = Inflation index at maturity minus publication lag
    - I(Base-lag) = Inflation index at base date minus publication lag
    - Lag typically 3 months for CPI/RPI/HICP

    The leg handles:
    - Historical fixings via InflationIndex
    - Forward CPI projection via inflation curves
    - Intra-month interpolation (FLAT, LINEAR, COMPOUND)
    - Business day adjustments
    - Discounting to valuation date

    Market conventions:
    - UK RPI: 3-month lag, linear interpolation, ACT/365F
    - US CPI-U: 3-month lag, linear interpolation, ACT/365F
    - EUR HICP: 3-month lag, linear interpolation, ACT/365F
    """

    def __init__(self,
                 effective_dt: Date,
                 end_dt: (Date, str),
                 leg_type: SwapTypes,
                 inflation_index: InflationIndex,
                 notional: float = ONE_MILLION,
                 payment_lag: int = 0,
                 cal_type: CalendarTypes = CalendarTypes.WEEKEND,
                 bd_type: BusDayAdjustTypes = BusDayAdjustTypes.FOLLOWING):
        """
        Create an inflation-linked leg.

        Args:
            effective_dt: Start date (base index reference before lag)
            end_dt: Maturity date or tenor (e.g., "5Y")
            leg_type: PAY or RECEIVE
            inflation_index: InflationIndex object with base CPI and lag
            notional: Notional amount
            payment_lag: Days after maturity for payment (default 0)
            cal_type: Calendar for date adjustments
            bd_type: Business day convention

        Example:
            >>> infl_leg = SwapInflationLeg(
            ...     effective_dt=Date(15, 6, 2023),
            ...     end_dt="5Y",
            ...     leg_type=SwapTypes.RECEIVE,
            ...     inflation_index=rpi_index,
            ...     notional=1_000_000
            ... )
        """
        check_argument_types(self.__init__, locals())

        self.instrument_type = InstrumentTypes.SWAP_INFLATION_LEG

        # Handle end date
        if isinstance(end_dt, Date):
            self._termination_dt = end_dt
        else:
            self._termination_dt = effective_dt.add_tenor(end_dt)

        # Adjust maturity for business days
        calendar = Calendar(cal_type)
        self._maturity_dt = calendar.adjust(self._termination_dt, bd_type)

        if effective_dt > self._maturity_dt:
            raise LibError("Start date after maturity date")

        self._effective_dt = effective_dt
        self._leg_type = leg_type
        self._inflation_index = inflation_index
        self._notional = notional
        self._payment_lag = payment_lag
        self._cal_type = cal_type
        self._bd_type = bd_type

        # Calculate payment date
        if payment_lag == 0:
            self._payment_dt = self._maturity_dt
        else:
            self._payment_dt = calendar.add_business_days(self._maturity_dt, payment_lag)

        # Calculate CPI reference dates (with lag applied by inflation index)
        # The inflation index will apply lag internally
        self._base_cpi_ref_dt = effective_dt
        self._final_cpi_ref_dt = self._maturity_dt

        # Valuation results (populated by value())
        self._base_index = None
        self._final_index = None
        self._inflation_return = None
        self._payment_amount = None
        self._payment_df = None
        self._payment_pv = None

###############################################################################

    def value(self,
              value_dt: Date,
              discount_curve: DiscountCurve,
              inflation_curve=None) -> float:
        """
        Value the inflation leg.

        Calculates the present value of the inflation-linked payment at maturity.

        Process:
        1. Get base CPI: I(effective_dt - lag) from fixings or curve
        2. Get final CPI: I(maturity_dt - lag) from fixings or curve
        3. Calculate inflation return: (I_final / I_base) - 1
        4. Calculate payment: Notional × inflation_return
        5. Discount to valuation date

        Args:
            value_dt: Valuation date
            discount_curve: Curve for discounting cashflows
            inflation_curve: Curve for projecting future CPI (optional)

        Returns:
            Present value of inflation leg in natural currency

        Example:
            >>> pv = infl_leg.value(value_dt, ois_curve, inflation_curve)
        """
        # Set curve if provided
        if inflation_curve is not None:
            self._inflation_index.set_inflation_curve(inflation_curve)

        # Get base and final CPI (lag applied internally by inflation index)
        self._base_index = self._inflation_index.get_index(
            self._base_cpi_ref_dt,
            apply_lag=True
        )

        self._final_index = self._inflation_index.get_index(
            self._final_cpi_ref_dt,
            apply_lag=True
        )

        # Calculate inflation return: (I_final / I_base - 1)
        if self._base_index <= 0.0:
            raise LibError(f"Base index must be positive, got {self._base_index}")

        self._inflation_return = (self._final_index / self._base_index) - 1.0

        # Calculate payment amount
        self._payment_amount = self._notional * self._inflation_return

        # Discount to valuation date
        if self._payment_dt > value_dt:
            df_value = discount_curve.df(value_dt, DayCountTypes.ACT_365F)
            df_payment = discount_curve.df(self._payment_dt, DayCountTypes.ACT_365F)
            self._payment_df = df_payment / df_value
            self._payment_pv = self._payment_amount * self._payment_df
            leg_pv = self._payment_pv
        else:
            # Payment in the past
            self._payment_df = 0.0
            self._payment_pv = 0.0
            leg_pv = 0.0

        # Apply leg direction
        if self._leg_type == SwapTypes.PAY:
            leg_pv *= -1.0

        return leg_pv

###############################################################################

    def print_payments(self):
        """
        Print inflation leg payment schedule.

        Shows the inflation index dates, CPI values, and payment calculation.
        """
        print("START DATE:", self._effective_dt)
        print("MATURITY DATE:", self._maturity_dt)
        print("PAYMENT DATE:", self._payment_dt)
        print("NOTIONAL:", f"{self._notional:,.2f}")
        print("LEG TYPE:", self._leg_type.name)
        print("INDEX TYPE:", self._inflation_index._index_type.name)
        print("INDEX LAG (MONTHS):", self._inflation_index._lag_months)
        print()

        # Calculate lagged dates for display
        base_lagged = self._inflation_index._apply_lag(self._base_cpi_ref_dt)
        final_lagged = self._inflation_index._apply_lag(self._final_cpi_ref_dt)

        print("CPI REFERENCE DATES:")
        print(f"  Base Ref Date: {self._base_cpi_ref_dt} → {base_lagged} (lagged)")
        print(f"  Final Ref Date: {self._final_cpi_ref_dt} → {final_lagged} (lagged)")

        if self._base_index is not None:
            print()
            print("CPI VALUES:")
            print(f"  Base Index: {self._base_index:.4f}")
            print(f"  Final Index: {self._final_index:.4f}")
            print(f"  Inflation Return: {self._inflation_return*100:.6f}%")
            print()
            print("PAYMENT:")
            print(f"  Amount: {self._payment_amount:,.2f}")

###############################################################################

    def print_valuation(self):
        """
        Print inflation leg valuation details.

        Shows CPI values, inflation return, payment amount, discount factor,
        and present value.
        """
        print("START DATE:", self._effective_dt)
        print("MATURITY DATE:", self._maturity_dt)
        print("PAYMENT DATE:", self._payment_dt)
        print("NOTIONAL:", f"{self._notional:,.2f}")
        print("LEG TYPE:", self._leg_type.name)

        if self._base_index is None:
            print("\nValuation not yet performed. Call value() first.")
            return

        # Calculate lagged dates
        base_lagged = self._inflation_index._apply_lag(self._base_cpi_ref_dt)
        final_lagged = self._inflation_index._apply_lag(self._final_cpi_ref_dt)

        print()
        print("="*80)
        print("INFLATION LEG VALUATION")
        print("="*80)

        header = ["Description", "Date", "Value"]
        rows = [
            ["Base CPI Ref (pre-lag)", str(self._base_cpi_ref_dt), ""],
            ["Base CPI Ref (lagged)", str(base_lagged), f"{self._base_index:.4f}"],
            ["Final CPI Ref (pre-lag)", str(self._final_cpi_ref_dt), ""],
            ["Final CPI Ref (lagged)", str(final_lagged), f"{self._final_index:.4f}"],
            ["", "", ""],
            ["Inflation Return", "", f"{self._inflation_return*100:.6f}%"],
            ["Payment Amount", str(self._payment_dt), f"{self._payment_amount:,.2f}"],
            ["Discount Factor", "", f"{self._payment_df:.6f}"],
            ["Present Value", "", f"{self._payment_pv:,.2f}"],
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
        s += label_to_string("LEG TYPE", self._leg_type)
        s += label_to_string("INFLATION INDEX", self._inflation_index._index_type)
        s += label_to_string("INDEX LAG (MONTHS)", self._inflation_index._lag_months)
        s += label_to_string("CALENDAR", self._cal_type)
        s += label_to_string("BUS DAY ADJUST", self._bd_type)
        return s

###############################################################################

    def _print(self):
        """Print inflation leg details."""
        print(self)

###############################################################################
