##############################################################################

##############################################################################

from ...utils.error import LibError
from ...utils.date import Date
from ...utils.math import ONE_MILLION
from ...utils.day_count import DayCount, DayCountTypes
from ...utils.frequency import FrequencyTypes
from ...utils.global_types import CurveTypes
from ...utils.calendar import CalendarTypes,  DateGenRuleTypes
from ...utils.calendar import Calendar, BusDayAdjustTypes
from ...utils.schedule import Schedule
from ...utils.helpers import format_table, label_to_string, check_argument_types
from ...utils.global_types import SwapTypes
from ...utils.currency import CurrencyTypes
from ...market.curves.discount_curve import DiscountCurve

##########################################################################


class SwapFloatLeg:
    """ Class for managing the floating leg of a swap. A float leg consists of
    a sequence of flows calculated according to an ISDA schedule and with a
    coupon determined by an index curve which changes over life of the swap."""

    def __init__(self,
                 effective_dt: Date,  # Date interest starts to accrue
                 end_dt: (Date, str),  # Date contract ends
                 leg_type: SwapTypes,
                 spread: (float),
                 freq_type: FrequencyTypes,
                 dc_type: DayCountTypes,
                 notional: float = ONE_MILLION,
                 principal: float = 0.0,
                 payment_lag: int = 0,
                 cal_type: CalendarTypes = CalendarTypes.WEEKEND,
                 bd_type: BusDayAdjustTypes = BusDayAdjustTypes.FOLLOWING,
                 dg_type: DateGenRuleTypes = DateGenRuleTypes.BACKWARD,
                 end_of_month: bool = False,
                 notional_exchange: bool = False,
                 floating_index: CurveTypes = CurveTypes.GBP_OIS_SONIA,
                 currency: CurrencyTypes = CurrencyTypes.GBP):        
        """ Create the fixed leg of a swap contract giving the contract start
        date, its maturity, fixed coupon, fixed leg frequency, fixed leg day
        count convention and notional.  """

        check_argument_types(self.__init__, locals())

        if type(end_dt) == Date:
            self._termination_dt = end_dt
        else:
            self._termination_dt = effective_dt.add_tenor(end_dt)

        calendar = Calendar(cal_type)

        self._maturity_dt = calendar.adjust(self._termination_dt,
                                            bd_type)

        if effective_dt > self._maturity_dt:
            raise LibError("Start date after maturity date")

        self._effective_dt = effective_dt
        self._end_dt = end_dt
        self._leg_type = leg_type
        self._freq_type = freq_type
        self._payment_lag = payment_lag
        self._principal = 0.0
        self._notional = notional
        self._notional_array = []
        self._spread = spread
        self._floating_index = floating_index
        self._currency = currency
        self._notional_exchange = notional_exchange
        self._dc_type = dc_type
        self._cal_type = cal_type
        self._bd_type = bd_type
        self._dg_type = dg_type
        self._end_of_month = end_of_month

        self._start_accrued_dts = []
        self._end_accrued_dts = []
        self._payment_dts = []
        self._payments = []
        self._year_fracs = []
        self._accrued_days = []

        self.generate_payment_dts()

###############################################################################

    def generate_payment_dts(self):
        """ Generate the floating leg payment dates and accrual factors. The
        coupons cannot be generated yet as we do not have the index curve. """

        schedule = Schedule(self._effective_dt,
                            self._termination_dt,
                            self._freq_type,
                            self._cal_type,
                            self._bd_type,
                            self._dg_type,
                            end_of_month=self._end_of_month)

        schedule_dts = schedule._adjusted_dts

        if len(schedule_dts) < 2:
            raise LibError("Schedule has none or only one date")

        self._start_accrued_dts = []
        self._end_accrued_dts = []
        self._payment_dts = []
        self._payment_dts_float = []
        self._payment_dts_ad = []
        self._year_fracs = []
        self._accrued_days = []

        prev_dt = schedule_dts[0]

        day_counter = DayCount(self._dc_type)
        calendar = Calendar(self._cal_type)

        payment_dts_float_sum = 0

        # All of the lists end up with the same length
        for next_dt in schedule_dts[1:]:

            self._start_accrued_dts.append(prev_dt)
            self._end_accrued_dts.append(next_dt)

            if self._payment_lag == 0:
                payment_dt = next_dt
            else:
                payment_dt = calendar.add_business_days(next_dt,
                                                        self._payment_lag)

            (year_frac, _, _) = day_counter.year_frac(self._effective_dt,
                                                next_dt)
            self._payment_dts_ad.append(year_frac)
            self._payment_dts.append(payment_dt)

            (year_frac, num, _) = day_counter.year_frac(prev_dt,
                                                        next_dt)
            payment_dts_float_sum += year_frac
            self._payment_dts_float.append(payment_dts_float_sum)
            self._year_fracs.append(year_frac)
            self._accrued_days.append(num)

            prev_dt = next_dt

###############################################################################

    def value(self,
              value_dt: Date,  # This should be the settlement date
              discount_curve: DiscountCurve,
              index_curve: DiscountCurve,
              first_fixing_rate: float = None):
        """ Value the floating leg with payments from an index curve and
        discounting based on a supplied discount curve as of the valuation date
        supplied. For an existing swap, the user must enter the next fixing
        coupon. """

        if discount_curve is None:
            raise LibError("Discount curve is None")

        if index_curve is None:
            index_curve = discount_curve

        self._rates = []
        self._payments = []
        self._payment_dfs = []
        self._payment_pvs = []
        self._cumulative_pvs = []

        df_value = discount_curve.df(value_dt,self._dc_type)
        leg_pv = 0.0
        num_payments = len(self._payment_dts)
        first_payment = False

        if not len(self._notional_array):
            self._notional_array = [self._notional] * num_payments

        index_basis = index_curve._dc_type
        index_day_counter = DayCount(index_basis)

        for i_pmnt in range(0, num_payments):

            pmnt_dt = self._payment_dts[i_pmnt]

            if pmnt_dt > value_dt:

                start_accrued_dt = self._start_accrued_dts[i_pmnt]
                end_accrued_dt = self._end_accrued_dts[i_pmnt]
                pay_alpha = self._year_fracs[i_pmnt]

                (index_alpha, num, _) = index_day_counter.year_frac(start_accrued_dt,
                                                                    end_accrued_dt)

                if first_payment is False and first_fixing_rate is not None:

                    fwd_rate = first_fixing_rate
                    first_payment = True

                else:

                    df_start = index_curve.df(start_accrued_dt,self._dc_type)
                    df_end = index_curve.df(end_accrued_dt,self._dc_type)
                    fwd_rate = (df_start / df_end - 1.0) / index_alpha

                pmntAmount = (fwd_rate + self._spread) * \
                    pay_alpha * self._notional_array[i_pmnt]

                df_pmnt = discount_curve.df(pmnt_dt,self._dc_type) / df_value
                pmnt_pv = pmntAmount * df_pmnt
                leg_pv += pmnt_pv

                self._rates.append(fwd_rate)
                self._payments.append(pmntAmount)
                self._payment_dfs.append(df_pmnt)
                self._payment_pvs.append(pmnt_pv)
                self._cumulative_pvs.append(leg_pv)

            else:

                self._rates.append(0.0)
                self._payments.append(0.0)
                self._payment_dfs.append(0.0)
                self._payment_pvs.append(0.0)
                self._cumulative_pvs.append(leg_pv)

        if pmnt_dt > value_dt:
            payment_pv = self._principal * df_pmnt * self._notional_array[-1]
            self._payment_pvs[-1] += payment_pv
            leg_pv += payment_pv
            self._cumulative_pvs[-1] = leg_pv

        if self._notional_exchange:
            if self._effective_dt > value_dt:
                df_start = discount_curve.df(self._effective_dt, self._dc_type) / df_value
                leg_pv += self._notional * df_start
            if pmnt_dt > value_dt:
                df_end = discount_curve.df(self._termination_dt, self._dc_type) / df_value
                end_pv = -self._notional * df_end
                leg_pv += end_pv
                self._payment_pvs[-1] += end_pv
                self._cumulative_pvs[-1] = leg_pv

        if self._leg_type == SwapTypes.PAY:
            leg_pv = leg_pv * (-1.0)

        return leg_pv
    
###############################################################################

    def print_payments(self):
        """ Prints the fixed leg dates, accrual factors, discount factors,
        cash amounts, their present value and their cumulative PV using the
        last valuation performed. """

        print("START DATE:", self._effective_dt)
        print("MATURITY DATE:", self._maturity_dt)
        print("SPREAD (bp):", self._spread * 10000)
        print("FREQUENCY:", str(self._freq_type))
        print("DAY COUNT:", str(self._dc_type))

        if len(self._payment_dts) == 0:
            print("Payments Dates not calculated.")
            return


        header = ["PAY_NUM", "PAY_dt", "ACCR_START", "ACCR_END",
                  "DAYS", "YEARFRAC", "RATE", "PMNT"]

        rows = []
        num_flows = len(self._payment_dts)
        for i_flow in range(0, num_flows):
            rows.append([
                i_flow + 1,
                self._payment_dts[i_flow],
                self._start_accrued_dts[i_flow],
                self._end_accrued_dts[i_flow],
                self._accrued_days[i_flow],
                round(self._year_fracs[i_flow], 4),
                round(self._rates[i_flow] * 100.0, 4),
                round(self._payments[i_flow], 2),
            ])

        table = format_table(header, rows)
        print("\nPAYMENTS SCHEDULE:")
        print(table)

###############################################################################

    def print_valuation(self):
        """ Prints the fixed leg dates, accrual factors, discount factors,
        cash amounts, their present value and their cumulative PV using the
        last valuation performed. """

        print("START DATE:", self._effective_dt)
        print("MATURITY DATE:", self._maturity_dt)
        print("SPREAD (BPS):", self._spread * 10000)
        print("FREQUENCY:", str(self._freq_type))
        print("DAY COUNT:", str(self._dc_type))

        if len(self._payments) == 0:
            print("Payments not calculated.")
            return

        header = ["PAY_NUM", "PAY_dt",  "NOTIONAL",
                  "IBOR", "PMNT", "DF", "PV", "CUM_PV"]

        rows = []
        num_flows = len(self._payment_dts)
        for i_flow in range(0, num_flows):
            rows.append([
                i_flow + 1,
                self._payment_dts[i_flow],
                round(self._notional_array[i_flow], 0),
                round(self._rates[i_flow] * 100.0, 4),
                round(self._payments[i_flow], 2),
                round(self._payment_dfs[i_flow], 4),
                round(self._payment_pvs[i_flow], 2),
                round(self._cumulative_pvs[i_flow], 2),
            ])

        table = format_table(header, rows)
        print("\nPAYMENTS VALUATION:")
        print(table)

###############################################################################

    def __repr__(self):
        s = label_to_string("OBJECT TYPE", type(self).__name__)
        s += label_to_string("START DATE", self._effective_dt)
        s += label_to_string("TERMINATION DATE", self._termination_dt)
        s += label_to_string("MATURITY DATE", self._maturity_dt)
        s += label_to_string("NOTIONAL", self._notional)
        s += label_to_string("SWAP TYPE", self._leg_type)
        s += label_to_string("SPREAD (BPS)", self._spread*10000)
        s += label_to_string("FREQUENCY", self._freq_type)
        s += label_to_string("DAY COUNT", self._dc_type)
        s += label_to_string("CALENDAR", self._cal_type)
        s += label_to_string("BUS DAY ADJUST", self._bd_type)
        s += label_to_string("DATE GEN TYPE", self._dg_type)
        return s

###############################################################################

    def _print(self):
        """ Print a list of the unadjusted coupon payment dates used in
        analytic calculations for the bond. """
        print(self)

###############################################################################