##############################################################################

##############################################################################

"""
Cross-Currency Basis Swap implementation for float-float XCCY swaps.

Provides the XccyBasisSwap class for creating and valuing cross-currency basis
swaps where two floating legs in different currencies are exchanged, with a
basis spread applied to one leg. Includes notional exchanges at start and
maturity.

Key features:
- Dual floating legs in different currencies with notional exchange
- Basis spread applied to specified leg (domestic or foreign)
- Forward rate projection from separate index curves
- Discounting using cross-currency discount curve
- FX conversion for consistent domestic currency valuation

The XCCY basis swap structure:
- Start date: XCCY spot date (T_spot = value date + FX spot lag)
- Two floating legs with independent calendars and conventions
- Initial notional exchange at T_spot
- Final notional exchange at maturity
- Basis spread added to one leg (typically foreign)

Example:
    >>> # Create a 5Y USD/GBP basis swap with 25bp spread on USD leg
    >>> value_dt = Date(15, 6, 2023)
    >>> xccy_swap = XccyBasisSwap(
    ...     effective_dt=value_dt,
    ...     term_dt_or_tenor="5Y",
    ...     domestic_notional=1_000_000,  # GBP
    ...     foreign_notional=1_270_000,   # USD (at spot FX)
    ...     domestic_spread=0.0,
    ...     foreign_spread=0.0025,  # 25bp
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

from .swap_float_leg import SwapFloatLeg

###############################################################################


class XccyBasisSwap:
    """
    Cross-currency basis swap with two floating legs and notional exchange.

    A basis swap exchanges floating rate payments in two currencies, with a
    basis spread applied to one leg. Notional amounts are exchanged at both
    the start and maturity of the swap.

    Valuation:
    - Domestic leg: valued using domestic index curve, discounted with domestic curve
    - Foreign leg: valued using foreign index curve, discounted with XCCY curve
    - Foreign PV converted to domestic using spot FX rate
    - Total PV = PV_domestic + spot_FX * PV_foreign_in_domestic
    """

    def __init__(self,
                 effective_dt: Date,
                 term_dt_or_tenor: (Date, str),
                 domestic_notional: float,
                 foreign_notional: float,
                 domestic_spread: float,
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
        Create a cross-currency basis swap with two floating legs.

        Args:
            effective_dt: Date interest starts to accrue (XCCY spot date)
            term_dt_or_tenor: Maturity date or tenor string (e.g., "5Y")
            domestic_notional: Notional amount in domestic currency
            foreign_notional: Notional amount in foreign currency
            domestic_spread: Spread added to domestic leg (decimal, e.g., 0.0025 for 25bp)
            foreign_spread: Spread added to foreign leg (decimal)
            domestic_freq_type: Payment frequency for domestic leg
            foreign_freq_type: Payment frequency for foreign leg
            domestic_dc_type: Day count convention for domestic leg
            foreign_dc_type: Day count convention for foreign leg
            domestic_floating_index: Index curve type for domestic leg
            foreign_floating_index: Index curve type for foreign leg
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

        # Domestic leg: receive floating (from domestic perspective)
        # Convention: we receive domestic, pay foreign
        self._domestic_leg = SwapFloatLeg(
            effective_dt=effective_dt,
            end_dt=self._termination_dt,
            leg_type=SwapTypes.RECEIVE,
            spread=domestic_spread,
            freq_type=domestic_freq_type,
            dc_type=domestic_dc_type,
            floating_index=domestic_floating_index,
            currency=domestic_currency,
            notional=domestic_notional,
            principal=0.0,
            payment_lag=domestic_payment_lag,
            cal_type=domestic_cal_type,
            bd_type=domestic_bd_type,
            dg_type=domestic_dg_type,
            end_of_month=domestic_end_of_month,
            notional_exchange=True
        )

        # Foreign leg: pay floating (from domestic perspective)
        self._foreign_leg = SwapFloatLeg(
            effective_dt=effective_dt,
            end_dt=self._termination_dt,
            leg_type=SwapTypes.PAY,
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
            notional_exchange=True
        )

        # Store attributes needed for curve construction
        self._domestic_spread = domestic_spread
        self._foreign_spread = foreign_spread
        self._adjusted_domestic_dts = self._domestic_leg._payment_dts
        self._adjusted_foreign_dts = self._foreign_leg._payment_dts

###############################################################################

    def value(self,
              value_dt: Date,
              domestic_discount_curve: DiscountCurve,
              foreign_discount_curve: DiscountCurve,
              xccy_discount_curve: DiscountCurve,
              spot_fx: float,
              first_fixing_rate_domestic: float = None,
              first_fixing_rate_foreign: float = None):
        """
        Value the cross-currency basis swap.

        Args:
            value_dt: Valuation date
            domestic_discount_curve: Discount curve for domestic currency (domestic collateral)
            foreign_discount_curve: Discount curve for foreign currency (foreign collateral)
            xccy_discount_curve: XCCY curve for foreign cashflows in domestic collateral
            spot_fx: Spot FX rate (domestic per unit of foreign)
            first_fixing_rate_domestic: First fixing rate for domestic leg (if applicable)
            first_fixing_rate_foreign: First fixing rate for foreign leg (if applicable)

        Returns:
            float: Present value in domestic currency

        Notes:
            - Domestic leg is valued with domestic_discount_curve for both projection and discounting
            - Foreign leg is valued with foreign_discount_curve for projection, xccy_discount_curve for discounting
            - Foreign leg PV is converted to domestic using spot_fx
        """
        # Domestic leg: use domestic curves for both projection and discounting
        domestic_leg_value = self._domestic_leg.value(
            value_dt=value_dt,
            discount_curve=domestic_discount_curve,
            index_curve=domestic_discount_curve,
            first_fixing_rate=first_fixing_rate_domestic
        )

        # Foreign leg: use foreign curve for projection, XCCY curve for discounting
        foreign_leg_value = self._foreign_leg.value(
            value_dt=value_dt,
            discount_curve=xccy_discount_curve,
            index_curve=foreign_discount_curve,
            first_fixing_rate=first_fixing_rate_foreign
        )

        # Convert foreign leg to domestic currency
        # foreign_leg is already PAY, so its PV is negative when we owe money
        # Multiply by spot_fx to convert to domestic
        foreign_leg_value_domestic = spot_fx * foreign_leg_value

        # Total value in domestic currency
        value = domestic_leg_value + foreign_leg_value_domestic
        return value

###############################################################################

    def print_payments(self):
        """Print payment schedules for both legs."""
        print("\n" + "="*80)
        print("DOMESTIC LEG:")
        print("="*80)
        self._domestic_leg.print_payments()

        print("\n" + "="*80)
        print("FOREIGN LEG:")
        print("="*80)
        self._foreign_leg.print_payments()

###############################################################################

    def print_valuation(self):
        """Print valuation details for both legs."""
        print("\n" + "="*80)
        print("DOMESTIC LEG VALUATION:")
        print("="*80)
        self._domestic_leg.print_valuation()

        print("\n" + "="*80)
        print("FOREIGN LEG VALUATION:")
        print("="*80)
        self._foreign_leg.print_valuation()

###############################################################################

    def __repr__(self):
        s = label_to_string("OBJECT TYPE", type(self).__name__)
        s += label_to_string("EFFECTIVE DATE", self._effective_dt)
        s += label_to_string("TERMINATION DATE", self._termination_dt)
        s += label_to_string("MATURITY DATE", self._maturity_dt)
        s += label_to_string("DOMESTIC CURRENCY", self._domestic_currency)
        s += label_to_string("FOREIGN CURRENCY", self._foreign_currency)
        s += label_to_string("DOMESTIC NOTIONAL", self._domestic_notional)
        s += label_to_string("FOREIGN NOTIONAL", self._foreign_notional)
        s += label_to_string("DOMESTIC SPREAD (BPS)", self._domestic_spread * 10000)
        s += label_to_string("FOREIGN SPREAD (BPS)", self._foreign_spread * 10000)
        s += "\n"
        s += label_to_string("DOMESTIC LEG", "")
        s += self._domestic_leg.__repr__()
        s += "\n"
        s += label_to_string("FOREIGN LEG", "")
        s += self._foreign_leg.__repr__()
        return s

###############################################################################
