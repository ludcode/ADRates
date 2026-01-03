"""
Year-on-Year Inflation leg implementation for inflation swaps.

Provides the SwapYoYInflationLeg class for managing year-on-year inflation-linked
periodic payments in swap contracts. Each payment is based on the annual inflation
rate between consecutive periods.

Key features:
- Periodic payments based on YoY inflation: [I(t) / I(t-1y)] - 1
- ISDA-compliant schedule generation
- Publication lag handling (typically 3 months)
- Integration with InflationIndex for fixings and interpolation
- Forward CPI projection from inflation curves
- Business day adjustments and calendar handling
- Optional spread over inflation rate

Used as a component in year-on-year inflation swaps where the inflation leg
pays the annual inflation rate on each payment date.

Example:
    >>> # Create YoY inflation leg for 5Y swap, annual payments
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
    >>> # Create YoY inflation leg
    >>> yoy_leg = SwapYoYInflationLeg(
    ...     effective_dt=effective,
    ...     end_dt=maturity,
    ...     leg_type=SwapTypes.RECEIVE,
    ...     inflation_index=rpi_index,
    ...     freq_type=FrequencyTypes.ANNUAL,
    ...     dc_type=DayCountTypes.ACT_ACT_ISDA,
    ...     notional=1_000_000
    ... )
    >>>
    >>> # Value the leg
    >>> pv = yoy_leg.value(effective, discount_curve, inflation_curve)
    >>> yoy_leg.print_valuation()
"""

from cavour.utils.error import LibError
from cavour.utils.date import Date
from cavour.utils.math import ONE_MILLION
from cavour.utils.day_count import DayCount, DayCountTypes
from cavour.utils.frequency import FrequencyTypes
from cavour.utils.calendar import CalendarTypes, DateGenRuleTypes
from cavour.utils.calendar import Calendar, BusDayAdjustTypes
from cavour.utils.schedule import Schedule
from cavour.utils.helpers import format_table, label_to_string, check_argument_types
from cavour.utils.global_types import SwapTypes, InstrumentTypes
from cavour.market.curves.discount_curve import DiscountCurve
from cavour.market.indices.inflation_index import InflationIndex

###############################################################################


class SwapYoYInflationLeg:
    """
    Year-on-Year inflation-linked leg of an inflation swap.

    A YoY inflation leg consists of periodic payments based on the annual
    inflation rate measured between consecutive periods. Each payment is:

        Payment_i = Notional × year_frac_i × [I(t_i) / I(t_i - 1y) - 1 + spread]

    where:
    - I(t_i) = Inflation index at payment date i (minus publication lag)
    - I(t_i - 1y) = Inflation index one year before payment date i
    - spread = Optional spread over inflation rate
    - year_frac_i = Accrual fraction for period i

    The leg handles:
    - Historical fixings via InflationIndex
    - Forward CPI projection via inflation curves
    - Intra-month interpolation (FLAT, LINEAR, COMPOUND)
    - Business day adjustments
    - Discounting to valuation date
    - Spread over inflation rate

    Market conventions:
    - UK RPI: Annual frequency, ACT/ACT, 3-month lag
    - US CPI-U: Annual frequency, ACT/ACT, 3-month lag
    - EUR HICP: Annual frequency, ACT/ACT, 3-month lag
    """

    def __init__(self,
                 effective_dt: Date,
                 end_dt: (Date, str),
                 leg_type: SwapTypes,
                 inflation_index: InflationIndex,
                 freq_type: FrequencyTypes,
                 dc_type: DayCountTypes,
                 notional: float = ONE_MILLION,
                 spread: float = 0.0,
                 payment_lag: int = 0,
                 cal_type: CalendarTypes = CalendarTypes.WEEKEND,
                 bd_type: BusDayAdjustTypes = BusDayAdjustTypes.FOLLOWING,
                 dg_type: DateGenRuleTypes = DateGenRuleTypes.BACKWARD,
                 end_of_month: bool = False):
        """
        Create a year-on-year inflation-linked leg.

        Args:
            effective_dt: Start date (first accrual period start)
            end_dt: Maturity date or tenor (e.g., "5Y")
            leg_type: PAY or RECEIVE
            inflation_index: InflationIndex object with base CPI and lag
            freq_type: Payment frequency (ANNUAL, SEMI_ANNUAL, QUARTERLY)
            dc_type: Day count convention for year fractions
            notional: Notional amount
            spread: Spread over inflation rate (e.g., 0.001 for 10bp)
            payment_lag: Days after accrual end for payment (default 0)
            cal_type: Calendar for date adjustments
            bd_type: Business day convention
            dg_type: Date generation rule
            end_of_month: End of month flag for schedule generation

        Example:
            >>> yoy_leg = SwapYoYInflationLeg(
            ...     effective_dt=Date(15, 6, 2023),
            ...     end_dt="5Y",
            ...     leg_type=SwapTypes.RECEIVE,
            ...     inflation_index=rpi_index,
            ...     freq_type=FrequencyTypes.ANNUAL,
            ...     dc_type=DayCountTypes.ACT_ACT_ISDA,
            ...     notional=1_000_000
            ... )
        """
        check_argument_types(self.__init__, locals())

        self.instrument_type = InstrumentTypes.SWAP_YOY_INFLATION_LEG

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
        self._end_dt = end_dt
        self._leg_type = leg_type
        self._inflation_index = inflation_index
        self._freq_type = freq_type
        self._dc_type = dc_type
        self._notional = notional
        self._spread = spread
        self._payment_lag = payment_lag
        self._cal_type = cal_type
        self._bd_type = bd_type
        self._dg_type = dg_type
        self._end_of_month = end_of_month

        # Payment schedule arrays (populated by generate_payment_schedule())
        self._start_accrued_dts = []
        self._end_accrued_dts = []
        self._payment_dts = []
        self._year_fracs = []
        self._accrued_days = []

        # YoY-specific: dates for inflation measurement
        self._yoy_start_dts = []  # CPI reference dates for start of YoY period
        self._yoy_end_dts = []    # CPI reference dates for end of YoY period

        # Valuation results (populated by value())
        self._start_cpis = []
        self._end_cpis = []
        self._yoy_rates = []
        self._payments = []
        self._dfs = []
        self._pvs = []

        # Generate payment schedule
        self.generate_payment_schedule()

###############################################################################

    def generate_payment_schedule(self):
        """
        Generate payment schedule for YoY inflation leg.

        Creates:
        - Accrual period dates
        - Payment dates (with lag if applicable)
        - Year fractions for each period
        - CPI reference dates for YoY calculations
        """
        # Generate ISDA schedule
        schedule = Schedule(
            self._effective_dt,
            self._termination_dt,
            self._freq_type,
            self._cal_type,
            self._bd_type,
            self._dg_type,
            end_of_month=self._end_of_month
        )

        schedule_dts = schedule._adjusted_dts

        if len(schedule_dts) < 2:
            raise LibError("Schedule has none or only one date")

        # Clear arrays
        self._start_accrued_dts = []
        self._end_accrued_dts = []
        self._payment_dts = []
        self._year_fracs = []
        self._accrued_days = []
        self._yoy_start_dts = []
        self._yoy_end_dts = []

        calendar = Calendar(self._cal_type)
        day_counter = DayCount(self._dc_type)

        # For each accrual period
        for i in range(1, len(schedule_dts)):
            start_dt = schedule_dts[i - 1]
            end_dt = schedule_dts[i]

            # Calculate year fraction for accrual
            (year_frac, num_days, denom_days) = day_counter.year_frac(
                start_dt,
                end_dt,
                None,  # No next coupon date needed for most conventions
                None   # No frequency needed for most conventions
            )

            # Calculate payment date (end date + payment lag)
            if self._payment_lag == 0:
                payment_dt = end_dt
            else:
                payment_dt = calendar.add_business_days(end_dt, self._payment_lag)

            # For YoY calculation:
            # - End CPI: measured at end of accrual period
            # - Start CPI: measured 1 year before end of accrual period
            yoy_end_dt = end_dt
            yoy_start_dt = end_dt.add_months(-12)  # 1 year before

            # Store schedule data
            self._start_accrued_dts.append(start_dt)
            self._end_accrued_dts.append(end_dt)
            self._payment_dts.append(payment_dt)
            self._year_fracs.append(year_frac)
            self._accrued_days.append(num_days)
            self._yoy_start_dts.append(yoy_start_dt)
            self._yoy_end_dts.append(yoy_end_dt)

###############################################################################

    def value(self,
              value_dt: Date,
              discount_curve: DiscountCurve,
              inflation_curve=None) -> float:
        """
        Value the YoY inflation leg.

        Calculates the present value of all periodic inflation-linked payments.

        Process for each payment:
        1. Get start CPI: I(payment_dt - 1y - lag) from fixings or curve
        2. Get end CPI: I(payment_dt - lag) from fixings or curve
        3. Calculate YoY inflation rate: (I_end / I_start) - 1
        4. Calculate payment: Notional × year_frac × (yoy_rate + spread)
        5. Discount to valuation date
        6. Sum all PVs

        Args:
            value_dt: Valuation date
            discount_curve: Curve for discounting cashflows
            inflation_curve: Curve for projecting future CPI (optional)

        Returns:
            Present value of YoY inflation leg in natural currency

        Example:
            >>> pv = yoy_leg.value(value_dt, ois_curve, inflation_curve)
        """
        # Set curve if provided
        if inflation_curve is not None:
            self._inflation_index.set_inflation_curve(inflation_curve)

        # Clear valuation arrays
        self._start_cpis = []
        self._end_cpis = []
        self._yoy_rates = []
        self._payments = []
        self._dfs = []
        self._pvs = []

        leg_pv = 0.0

        # Value each payment
        for i in range(len(self._payment_dts)):
            payment_dt = self._payment_dts[i]

            # Only value future payments
            if payment_dt <= value_dt:
                # Payment in the past - skip
                self._start_cpis.append(0.0)
                self._end_cpis.append(0.0)
                self._yoy_rates.append(0.0)
                self._payments.append(0.0)
                self._dfs.append(0.0)
                self._pvs.append(0.0)
                continue

            # Get CPI values (lag applied internally by inflation index)
            start_cpi = self._inflation_index.get_index(
                self._yoy_start_dts[i],
                apply_lag=True
            )

            end_cpi = self._inflation_index.get_index(
                self._yoy_end_dts[i],
                apply_lag=True
            )

            # Calculate YoY inflation rate: (I_end / I_start) - 1
            if start_cpi <= 0.0:
                raise LibError(f"Start CPI must be positive, got {start_cpi}")

            yoy_rate = (end_cpi / start_cpi) - 1.0

            # Calculate payment: Notional × year_frac × (yoy_rate + spread)
            payment = self._notional * self._year_fracs[i] * (yoy_rate + self._spread)

            # Discount to valuation date
            df_value = discount_curve.df(value_dt, self._dc_type)
            df_payment = discount_curve.df(payment_dt, self._dc_type)
            df = df_payment / df_value

            pv = payment * df

            # Store results
            self._start_cpis.append(start_cpi)
            self._end_cpis.append(end_cpi)
            self._yoy_rates.append(yoy_rate)
            self._payments.append(payment)
            self._dfs.append(df)
            self._pvs.append(pv)

            leg_pv += pv

        # Apply leg direction
        if self._leg_type == SwapTypes.PAY:
            leg_pv *= -1.0

        return leg_pv

###############################################################################

    def print_payments(self):
        """
        Print YoY inflation leg payment schedule.

        Shows the payment dates, accrual periods, YoY inflation measurement
        dates, and calculated payments.
        """
        print("="*100)
        print("YEAR-ON-YEAR INFLATION LEG PAYMENT SCHEDULE")
        print("="*100)
        print("EFFECTIVE DATE:", self._effective_dt)
        print("MATURITY DATE:", self._maturity_dt)
        print("NOTIONAL:", f"{self._notional:,.2f}")
        print("LEG TYPE:", self._leg_type.name)
        print("FREQUENCY:", self._freq_type.name)
        print("DAY COUNT:", self._dc_type.name)
        print("SPREAD:", f"{self._spread * 10000:.2f} bp")
        print("INDEX TYPE:", self._inflation_index._index_type.name)
        print("INDEX LAG (MONTHS):", self._inflation_index._lag_months)
        print()

        if len(self._payments) == 0:
            print("No valuation performed yet. Call value() first.")
            return

        header = ["Period", "Start", "End", "Payment", "YFrac",
                  "YoY Start", "YoY End", "Start CPI", "End CPI",
                  "YoY Rate", "Payment"]
        rows = []

        for i in range(len(self._payment_dts)):
            rows.append([
                f"{i+1}",
                str(self._start_accrued_dts[i]),
                str(self._end_accrued_dts[i]),
                str(self._payment_dts[i]),
                f"{self._year_fracs[i]:.6f}",
                str(self._yoy_start_dts[i]),
                str(self._yoy_end_dts[i]),
                f"{self._start_cpis[i]:.4f}" if self._start_cpis[i] > 0 else "N/A",
                f"{self._end_cpis[i]:.4f}" if self._end_cpis[i] > 0 else "N/A",
                f"{self._yoy_rates[i]*100:.4f}%" if self._yoy_rates[i] != 0 else "N/A",
                f"{self._payments[i]:,.2f}" if self._payments[i] != 0 else "N/A"
            ])

        table = format_table(header, rows)
        print(table)

###############################################################################

    def print_valuation(self):
        """
        Print YoY inflation leg valuation details.

        Shows payment schedule, CPI values, YoY rates, discount factors,
        and present values for each payment.
        """
        print("="*100)
        print("YEAR-ON-YEAR INFLATION LEG VALUATION")
        print("="*100)

        if len(self._pvs) == 0:
            print("No valuation performed yet. Call value() first.")
            return

        header = ["Period", "Payment Date", "YoY Rate", "Spread", "Total Rate",
                  "Notional", "Year Frac", "Payment", "DF", "PV"]
        rows = []

        total_pv = 0.0
        for i in range(len(self._payment_dts)):
            if self._pvs[i] == 0.0:
                continue  # Skip past payments

            total_rate = self._yoy_rates[i] + self._spread

            rows.append([
                f"{i+1}",
                str(self._payment_dts[i]),
                f"{self._yoy_rates[i]*100:.4f}%",
                f"{self._spread*10000:.2f}bp",
                f"{total_rate*100:.4f}%",
                f"{self._notional:,.0f}",
                f"{self._year_fracs[i]:.6f}",
                f"{self._payments[i]:,.2f}",
                f"{self._dfs[i]:.6f}",
                f"{self._pvs[i]:,.2f}"
            ])

            total_pv += self._pvs[i]

        table = format_table(header, rows)
        print(table)
        print()
        print(f"TOTAL PV: {total_pv:,.2f}")

        # Apply leg direction for display
        if self._leg_type == SwapTypes.PAY:
            total_pv *= -1.0
            print(f"LEG PV (after direction): {total_pv:,.2f}")

###############################################################################

    def __repr__(self):
        s = label_to_string("OBJECT TYPE", type(self).__name__)
        s += label_to_string("START DATE", self._effective_dt)
        s += label_to_string("TERMINATION DATE", self._termination_dt)
        s += label_to_string("MATURITY DATE", self._maturity_dt)
        s += label_to_string("NOTIONAL", self._notional)
        s += label_to_string("LEG TYPE", self._leg_type)
        s += label_to_string("FREQUENCY", self._freq_type)
        s += label_to_string("DAY COUNT", self._dc_type)
        s += label_to_string("SPREAD", f"{self._spread * 10000:.2f} bp")
        s += label_to_string("INFLATION INDEX", self._inflation_index._index_type)
        s += label_to_string("INDEX LAG (MONTHS)", self._inflation_index._lag_months)
        s += label_to_string("NUM PAYMENTS", len(self._payment_dts))
        s += label_to_string("CALENDAR", self._cal_type)
        s += label_to_string("BUS DAY ADJUST", self._bd_type)
        return s

###############################################################################

    def _print(self):
        """Print YoY inflation leg details."""
        print(self)

###############################################################################
