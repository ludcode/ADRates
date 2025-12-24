##############################################################################

##############################################################################

"""
Cross-Currency Fixed-Float Swap implementation.

Provides the XccyFixFloat class for creating and valuing cross-currency swaps
where a fixed leg in domestic currency is exchanged against a floating leg in
foreign currency. Includes notional exchanges at start and maturity.

Key features:
- Fixed leg in domestic currency (pays fixed coupon)
- Floating leg in foreign currency (pays floating rate + spread) with notional exchange
- Forward rate projection from separate index curves
- Discounting using cross-currency discount curve
- FX conversion for consistent domestic currency valuation

The XCCY fixed-float swap structure:
- Start date: XCCY spot date (T_spot = value date + FX spot lag)
- Domestic fixed leg (pays/receives fixed coupon)
- Foreign floating leg (pays/receives floating + spread) with independent calendar
- Initial notional exchange at T_spot
- Final notional exchange at maturity

Example:
    >>> # Create a 5Y USD/GBP fixed-float swap
    >>> # Domestic (GBP): pay fixed 4.5%
    >>> # Foreign (USD): receive floating SOFR + 25bp
    >>> value_dt = Date(15, 6, 2023)
    >>> xccy_swap = XccyFixFloat(
    ...     effective_dt=value_dt,
    ...     term_dt_or_tenor="5Y",
    ...     domestic_notional=1_000_000,  # GBP
    ...     foreign_notional=1_270_000,   # USD (at spot FX)
    ...     domestic_leg_type=SwapTypes.PAY,  # Pay fixed GBP
    ...     domestic_coupon=0.045,  # 4.5% fixed
    ...     foreign_spread=0.0025,  # 25bp spread on USD float
    ...     domestic_freq_type=FrequencyTypes.ANNUAL,
    ...     foreign_freq_type=FrequencyTypes.QUARTERLY,
    ...     domestic_dc_type=DayCountTypes.ACT_365F,
    ...     foreign_dc_type=DayCountTypes.ACT_360,
    ...     domestic_currency=CurrencyTypes.GBP,
    ...     foreign_currency=CurrencyTypes.USD
    ... )
    >>>
    >>> # Value the swap
    >>> pv = xccy_swap.value(value_dt, gbp_curve, usd_curve, xccy_curve, spot_fx)
"""

from cavour.utils.error import LibError
from cavour.utils.date import Date
from cavour.utils.day_count import DayCountTypes
from cavour.utils.frequency import FrequencyTypes
from cavour.utils.global_types import CurveTypes
from cavour.utils.calendar import CalendarTypes, DateGenRuleTypes
from cavour.utils.calendar import Calendar, BusDayAdjustTypes
from cavour.utils.helpers import check_argument_types, label_to_string
from cavour.utils.math import ONE_MILLION
from cavour.utils.global_types import SwapTypes, InstrumentTypes
from cavour.utils.currency import CurrencyTypes
from cavour.market.curves.discount_curve import DiscountCurve

from .swap_fixed_leg import SwapFixedLeg
from .swap_float_leg import SwapFloatLeg

###############################################################################


class XccyFixFloat:
    """
    Cross-currency fixed-float swap with notional exchange.

    A fixed-float swap exchanges fixed rate payments in domestic currency
    against floating rate payments in foreign currency. Notional amounts are
    exchanged at both the start and maturity of the swap.

    Valuation:
    - Domestic fixed leg: discounted with domestic curve
    - Foreign floating leg: valued using foreign index curve, discounted with XCCY curve
    - Foreign PV converted to domestic using spot FX rate
    - Total PV = PV_domestic + spot_FX * PV_foreign_in_domestic
    """

    def __init__(self,
                 effective_dt: Date,
                 term_dt_or_tenor: (Date, str),
                 domestic_notional: float,
                 foreign_notional: float,
                 domestic_leg_type: SwapTypes,
                 domestic_coupon: float,
                 foreign_spread: float,
                 domestic_freq_type: FrequencyTypes,
                 foreign_freq_type: FrequencyTypes,
                 domestic_dc_type: DayCountTypes,
                 foreign_dc_type: DayCountTypes,
                 domestic_floating_index: CurveTypes,
                 foreign_floating_index: CurveTypes,
                 domestic_currency: CurrencyTypes,
                 foreign_currency: CurrencyTypes,
                 domestic_payment_lag: int = 0,
                 foreign_payment_lag: int = 0,
                 domestic_cal_type: CalendarTypes = CalendarTypes.WEEKEND,
                 foreign_cal_type: CalendarTypes = CalendarTypes.WEEKEND,
                 domestic_bd_type: BusDayAdjustTypes = BusDayAdjustTypes.FOLLOWING,
                 foreign_bd_type: BusDayAdjustTypes = BusDayAdjustTypes.FOLLOWING,
                 domestic_dg_type: DateGenRuleTypes = DateGenRuleTypes.BACKWARD,
                 foreign_dg_type: DateGenRuleTypes = DateGenRuleTypes.BACKWARD,
                 domestic_end_of_month: bool = False,
                 foreign_end_of_month: bool = False):
        """
        Create a cross-currency fixed-float swap.

        Args:
            effective_dt: Date interest starts to accrue (XCCY spot date)
            term_dt_or_tenor: Maturity date or tenor string (e.g., "5Y")
            domestic_notional: Notional amount in domestic currency
            foreign_notional: Notional amount in foreign currency
            domestic_leg_type: SwapTypes.PAY or SwapTypes.RECEIVE for domestic fixed leg
            domestic_coupon: Fixed coupon rate for domestic leg (decimal, e.g., 0.045 for 4.5%)
            foreign_spread: Spread added to foreign floating leg (decimal)
            domestic_freq_type: Payment frequency for domestic fixed leg
            foreign_freq_type: Payment frequency for foreign floating leg
            domestic_dc_type: Day count convention for domestic fixed leg
            foreign_dc_type: Day count convention for foreign floating leg
            domestic_floating_index: Index curve type for domestic currency (for discounting)
            foreign_floating_index: Index curve type for foreign leg (for projection)
            domestic_currency: Currency of domestic leg
            foreign_currency: Currency of foreign leg
            domestic_payment_lag: Payment lag in days for domestic leg
            foreign_payment_lag: Payment lag in days for foreign leg
            domestic_cal_type: Calendar for domestic leg
            foreign_cal_type: Calendar for foreign leg
            domestic_bd_type: Business day adjustment for domestic leg
            foreign_bd_type: Business day adjustment for foreign leg
            domestic_dg_type: Date generation rule for domestic leg
            foreign_dg_type: Date generation rule for foreign leg
            domestic_end_of_month: End of month adjustment for domestic leg
            foreign_end_of_month: End of month adjustment for foreign leg
        """
        check_argument_types(self.__init__, locals())

        self.derivative_type = InstrumentTypes.XCCY_SWAP

        if isinstance(term_dt_or_tenor, Date):
            self._termination_dt = term_dt_or_tenor
        else:
            self._termination_dt = effective_dt.add_tenor(term_dt_or_tenor)

        # Use domestic calendar for maturity adjustment
        calendar = Calendar(domestic_cal_type)
        self._maturity_dt = calendar.adjust(self._termination_dt, domestic_bd_type)

        if effective_dt > self._maturity_dt:
            raise LibError("Start date after maturity date")

        self._effective_dt = effective_dt
        self._domestic_notional = domestic_notional
        self._foreign_notional = foreign_notional
        self._domestic_currency = domestic_currency
        self._foreign_currency = foreign_currency
        self._domestic_floating_index = domestic_floating_index
        self._foreign_floating_index = foreign_floating_index
        self._domestic_leg_type = domestic_leg_type

        # Domestic leg: fixed coupon (PAY or RECEIVE as specified)
        self._domestic_leg = SwapFixedLeg(
            effective_dt=effective_dt,
            end_dt=self._termination_dt,
            leg_type=domestic_leg_type,
            coupon=domestic_coupon,
            freq_type=domestic_freq_type,
            dc_type=domestic_dc_type,
            floating_index=domestic_floating_index,
            currency=domestic_currency,
            notional=domestic_notional,
            principal=0.0,  # No principal in SwapFixedLeg, handled manually
            payment_lag=domestic_payment_lag,
            cal_type=domestic_cal_type,
            bd_type=domestic_bd_type,
            dg_type=domestic_dg_type,
            end_of_month=domestic_end_of_month
        )

        # Foreign leg: floating + spread (opposite direction of domestic fixed)
        foreign_leg_type = SwapTypes.PAY if domestic_leg_type == SwapTypes.RECEIVE else SwapTypes.RECEIVE

        self._foreign_leg = SwapFloatLeg(
            effective_dt=effective_dt,
            end_dt=self._termination_dt,
            leg_type=foreign_leg_type,
            spread=foreign_spread,
            freq_type=foreign_freq_type,
            dc_type=foreign_dc_type,
            floating_index=foreign_floating_index,
            currency=foreign_currency,
            notional=foreign_notional,
            principal=0.0,
            payment_lag=foreign_payment_lag,
            cal_type=foreign_cal_type,
            bd_type=foreign_bd_type,
            dg_type=foreign_dg_type,
            end_of_month=foreign_end_of_month,
            notional_exchange=True  # Foreign floating leg has notional exchange
        )

    def value(self,
              value_dt: Date,
              domestic_discount_curve: DiscountCurve,
              foreign_discount_curve: DiscountCurve,
              xccy_discount_curve: DiscountCurve,
              spot_fx: float,
              first_fixing_rate_foreign = None):
        """
        Value the cross-currency fixed-float swap.

        Args:
            value_dt: Valuation date
            domestic_discount_curve: Domestic OIS discount curve
            foreign_discount_curve: Foreign OIS discount curve (for projection)
            xccy_discount_curve: Cross-currency discount curve (for foreign leg discounting)
            spot_fx: Spot FX rate (domestic currency per unit of foreign currency)
            first_fixing_rate_foreign: First fixing rate for foreign floating leg (if applicable)

        Returns:
            Present value of the swap in domestic currency
        """
        check_argument_types(self.value, locals())

        # Value domestic fixed leg using domestic curve
        domestic_leg_value = self._domestic_leg.value(
            value_dt=value_dt,
            discount_curve=domestic_discount_curve
        )

        # Add notional exchanges for domestic fixed leg (manual handling)
        # Start exchange: -notional at effective date (outflow)
        # End exchange: +notional at maturity date (inflow)
        if self._effective_dt >= value_dt:
            df_start = domestic_discount_curve.df(self._effective_dt)
            start_exchange_pv = -self._domestic_notional * df_start
            # Apply leg type sign
            if self._domestic_leg_type == SwapTypes.RECEIVE:
                domestic_leg_value += start_exchange_pv
            else:
                domestic_leg_value -= start_exchange_pv

        if self._maturity_dt >= value_dt:
            df_end = domestic_discount_curve.df(self._maturity_dt)
            end_exchange_pv = self._domestic_notional * df_end
            # Apply leg type sign
            if self._domestic_leg_type == SwapTypes.RECEIVE:
                domestic_leg_value += end_exchange_pv
            else:
                domestic_leg_value -= end_exchange_pv

        # Value foreign floating leg
        # Use foreign curve for projection, XCCY curve for discounting
        foreign_leg_value = self._foreign_leg.value(
            value_dt=value_dt,
            discount_curve=xccy_discount_curve,
            index_curve=foreign_discount_curve,
            first_fixing_rate=first_fixing_rate_foreign
        )

        # Convert foreign PV to domestic currency and sum
        value = domestic_leg_value + spot_fx * foreign_leg_value

        return value

    def print_valuation(self):
        """Print valuation details for both legs."""
        print("="*80)
        print("DOMESTIC FIXED LEG VALUATION:")
        print("="*80)
        self._domestic_leg.print_valuation()

        print("\n" + "="*80)
        print("FOREIGN FLOATING LEG VALUATION:")
        print("="*80)
        self._foreign_leg.print_valuation()

    def __repr__(self):
        s = label_to_string("OBJECT TYPE", type(self).__name__)
        s += label_to_string("EFFECTIVE DATE", self._effective_dt)
        s += label_to_string("MATURITY DATE", self._maturity_dt)
        s += label_to_string("DOMESTIC NOTIONAL", self._domestic_notional)
        s += label_to_string("FOREIGN NOTIONAL", self._foreign_notional)
        s += label_to_string("DOMESTIC CURRENCY", self._domestic_currency)
        s += label_to_string("FOREIGN CURRENCY", self._foreign_currency)
        s += label_to_string("DOMESTIC LEG TYPE", self._domestic_leg_type)
        return s
