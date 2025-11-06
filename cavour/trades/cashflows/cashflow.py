##############################################################################

##############################################################################

"""
Single cashflow valuation for fixed payment instruments.

Provides the SingleFixedCashflow class for pricing standalone fixed cashflows
with business day adjustments, payment lags, and calendar handling.

Used as a building block for more complex instruments like bonds and swaps.
"""

from typing import Union

from cavour.utils.error import LibError
from cavour.utils.date import Date
from cavour.utils.math import ONE_MILLION
from cavour.utils.day_count import DayCount, DayCountTypes
from cavour.utils.frequency import FrequencyTypes
from cavour.utils.global_types import CurveTypes
from cavour.utils.calendar import CalendarTypes,  DateGenRuleTypes
from cavour.utils.calendar import Calendar, BusDayAdjustTypes
from cavour.utils.schedule import Schedule
from cavour.utils.helpers import format_table, label_to_string, check_argument_types
from cavour.utils.global_types import SwapTypes, InstrumentTypes
from cavour.utils.currency import CurrencyTypes
from cavour.market.curves.discount_curve import DiscountCurve

##########################################################################


class SingleFixedCashflow:
    """
    Represents a single fixed cashflow at a specified payment date.

    Used as a building block for pricing bonds, swaps, and other fixed-income
    instruments. Handles business day adjustments, payment lags, and calendar
    conventions automatically.

    Attributes:
        _effective_dt (Date): Accrual start date for the cashflow
        _payment_dt (Date): Adjusted payment date after lag and business day conventions
        _leg_type (SwapTypes): PAY (negative PV) or RECEIVE (positive PV)
        _amount (float): Fixed payment amount in currency units
        _dc_type (DayCountTypes): Day count convention for discounting
        _payment_lag (int): Business day lag applied to payment date
        _currency (CurrencyTypes): Currency denomination

    Example:
        >>> # Create a £1,000,000 payment 6 months from today
        >>> effective = Date(15, 6, 2023)
        >>> cashflow = SingleFixedCashflow(
        ...     effective_dt=effective,
        ...     payment_dt="6M",
        ...     leg_type=SwapTypes.RECEIVE,
        ...     amount=1_000_000.0,
        ...     dc_type=DayCountTypes.ACT_365F
        ... )
        >>>
        >>> # Value the cashflow
        >>> value_dt = Date(15, 6, 2023)
        >>> pv = cashflow.value(value_dt, discount_curve)
        >>> print(f"Present Value: {pv:,.2f}")
    """

    def __init__(self,
                 effective_dt: Date,                # accrual start date
                 payment_dt: Union[Date, str],      # exact date or tenor (e.g. '6M')
                 leg_type: SwapTypes,               # PAY or RECEIVE
                 amount: float,                     # fixed payment amount
                 dc_type: DayCountTypes,            # day-count for discounting
                 payment_lag: int = 0,              # business-day lag
                 cal_type: CalendarTypes = CalendarTypes.WEEKEND,
                 bd_type: BusDayAdjustTypes = BusDayAdjustTypes.FOLLOWING,
                 currency: CurrencyTypes = CurrencyTypes.GBP):
        check_argument_types(self.__init__, locals())

        self._effective_dt = effective_dt
        self._leg_type     = leg_type
        self._amount       = amount
        self._dc_type      = dc_type
        self._payment_lag  = payment_lag
        self._currency     = currency

        cal = Calendar(cal_type)

        # support tenor strings
        if isinstance(payment_dt, str):
            raw_dt = effective_dt.add_tenor(payment_dt)
        else:
            raw_dt = payment_dt

        # apply lag then business-day adjust
        lagged = cal.add_business_days(raw_dt, payment_lag)
        self._payment_dt = cal.adjust(lagged, bd_type)

    def value(self, value_dt: Date, discount_curve: DiscountCurve) -> float:
        """
        Discount the single payment back to value_dt.
        Returns positive for RECEIVE, negative for PAY legs.
        """
        # today‐zero
        df_ref   = discount_curve.df(value_dt, self._dc_type)
        df_pmt   = discount_curve.df(self._payment_dt, self._dc_type)

        if self._payment_dt <= value_dt:
            pv = 0.0
        else:
            pv = self._amount * (df_pmt / df_ref)

        return -pv if self._leg_type == SwapTypes.PAY else pv
    
###########################################################################
    
    def print_valuation(self, value_dt: Date, discount_curve: DiscountCurve):
        """
        Prints the single cashflow’s payment date, amount, discount factor,
        present value and cumulative PV (which equals PV for one flow).
        """
        # discount to today
        df_ref = discount_curve.df(value_dt, self._dc_type)
        df_pmt = discount_curve.df(self._payment_dt, self._dc_type)
        if self._payment_dt <= value_dt:
            pv = 0.0
            df_disp = 0.0
        else:
            df_disp = df_pmt / df_ref
            pv      = self._amount * df_disp

        # flip sign for PAY legs
        if self._leg_type == SwapTypes.PAY:
            pv      = -pv

        # build table
        header = ["PAY_NUM", "PAY_dt", "AMOUNT", "DF", "PV", "CUM_PV"]
        rows   = [[
            1,
            self._payment_dt,
            round(self._amount, 2),
            round(df_disp,      6),
            round(pv,           2),
            round(pv,           2),
        ]]
        table = format_table(header, rows)

        print(f"START DATE:   {self._effective_dt}")
        print(f"PAYMENT DATE: {self._payment_dt}")
        print("\nSINGLE CASHFLOW VALUATION:")
        print(table)

###########################################################################

    def __repr__(self):
        s  = label_to_string("OBJECT TYPE",    type(self).__name__)
        s += label_to_string("START DATE",     self._effective_dt)
        s += label_to_string("PAYMENT DATE",   self._payment_dt)
        s += label_to_string("AMOUNT",         self._amount)
        s += label_to_string("LEG TYPE",       self._leg_type)
        s += label_to_string("DAY COUNT",      self._dc_type)
        s += label_to_string("PAYMENT LAG",    self._payment_lag)
        return s
    
###########################################################################

    def _print(self):
        """Alias for printing the repr."""
        print(self)