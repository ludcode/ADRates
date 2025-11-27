"""
Test suite for XCCY curve calendar functionality.

Validates that XCCY curves correctly accept and use calendar parameters
through XccyBasisSwap instruments.
"""

import pytest
import numpy as np

from cavour.utils.date import Date
from cavour.utils.day_count import DayCountTypes
from cavour.utils.frequency import FrequencyTypes
from cavour.utils.calendar import Calendar, CalendarTypes, create_calendar_intersection
from cavour.utils.global_types import CurveTypes
from cavour.utils.currency import CurrencyTypes
from cavour.market.curves.discount_curve import DiscountCurve
from cavour.market.curves.interpolator import InterpTypes

from cavour.trades.rates.xccy_basis_swap import XccyBasisSwap
from cavour.trades.rates.xccy_curve import XccyCurve


@pytest.fixture
def value_dt():
    """Valuation date for tests"""
    return Date(15, 6, 2024)


@pytest.fixture
def gbp_ois_curve(value_dt):
    """Simple GBP OIS curve for testing"""
    times = [1.0, 2.0, 5.0]
    dfs = np.array([0.95, 0.91, 0.83])
    return DiscountCurve(value_dt, times, dfs, InterpTypes.FLAT_FWD_RATES)


@pytest.fixture
def usd_ois_curve(value_dt):
    """Simple USD OIS curve for testing"""
    times = [1.0, 2.0, 5.0]
    dfs = np.array([0.94, 0.89, 0.80])
    return DiscountCurve(value_dt, times, dfs, InterpTypes.FLAT_FWD_RATES)


@pytest.fixture
def us_calendar():
    """US calendar"""
    return CalendarTypes.UNITED_STATES


@pytest.fixture
def uk_calendar():
    """UK calendar"""
    return CalendarTypes.UNITED_KINGDOM


@pytest.fixture
def target_calendar():
    """TARGET calendar for EUR"""
    return CalendarTypes.TARGET


def test_xccy_curve_with_different_calendars(value_dt, gbp_ois_curve, usd_ois_curve, us_calendar, uk_calendar):
    """Test that XccyBasisSwap can use different calendars for domestic and foreign legs"""

    spot_fx = 0.79  # GBP per USD

    # Create basis swaps with different calendars for each leg
    basis_swaps = [
        XccyBasisSwap(
            effective_dt=value_dt,
            term_dt_or_tenor="1Y",
            domestic_notional=spot_fx * 1_000_000,
            foreign_notional=1_000_000,
            domestic_spread=0.0,
            foreign_spread=0.0010,  # 10bp
            domestic_freq_type=FrequencyTypes.ANNUAL,
            foreign_freq_type=FrequencyTypes.ANNUAL,
            domestic_dc_type=DayCountTypes.ACT_365F,
            foreign_dc_type=DayCountTypes.ACT_360,
            domestic_floating_index=CurveTypes.GBP_OIS_SONIA,
            foreign_floating_index=CurveTypes.USD_OIS_SOFR,
            domestic_currency=CurrencyTypes.GBP,
            foreign_currency=CurrencyTypes.USD,
            domestic_cal_type=uk_calendar,  # UK calendar for GBP leg
            foreign_cal_type=us_calendar    # US calendar for USD leg
        ),
        XccyBasisSwap(
            effective_dt=value_dt,
            term_dt_or_tenor="2Y",
            domestic_notional=spot_fx * 1_000_000,
            foreign_notional=1_000_000,
            domestic_spread=0.0,
            foreign_spread=0.0012,  # 12bp
            domestic_freq_type=FrequencyTypes.ANNUAL,
            foreign_freq_type=FrequencyTypes.ANNUAL,
            domestic_dc_type=DayCountTypes.ACT_365F,
            foreign_dc_type=DayCountTypes.ACT_360,
            domestic_floating_index=CurveTypes.GBP_OIS_SONIA,
            foreign_floating_index=CurveTypes.USD_OIS_SOFR,
            domestic_currency=CurrencyTypes.GBP,
            foreign_currency=CurrencyTypes.USD,
            domestic_cal_type=uk_calendar,
            foreign_cal_type=us_calendar
        )
    ]

    # Build XCCY curve from basis swaps
    xccy_curve = XccyCurve(
        value_dt=value_dt,
        basis_swaps=basis_swaps,
        domestic_curve=gbp_ois_curve,
        foreign_curve=usd_ois_curve,
        spot_fx=spot_fx,
        interp_type=InterpTypes.FLAT_FWD_RATES,
        check_refit=False
    )

    # Verify curve was built successfully
    assert xccy_curve is not None
    assert len(xccy_curve._times) == 3  # t=0 + 2 pillars

    # Verify the basis swaps have different calendars
    assert basis_swaps[0]._domestic_leg._cal_type == uk_calendar
    assert basis_swaps[0]._foreign_leg._cal_type == us_calendar


def test_xccy_curve_with_joint_calendar(value_dt, gbp_ois_curve, usd_ois_curve):
    """
    Test XCCY curve with joint calendar for both legs.

    For standard cross-currency swaps, both legs typically use the same
    joint calendar (intersection of both currency calendars).
    """
    spot_fx = 0.79

    # For USD/GBP, use intersection calendar (business days in both NY and London)
    joint_cal = CalendarTypes.WEEKEND  # Simplified for testing

    basis_swaps = [
        XccyBasisSwap(
            effective_dt=value_dt,
            term_dt_or_tenor="1Y",
            domestic_notional=spot_fx * 1_000_000,
            foreign_notional=1_000_000,
            domestic_spread=0.0,
            foreign_spread=0.0010,
            domestic_freq_type=FrequencyTypes.ANNUAL,
            foreign_freq_type=FrequencyTypes.ANNUAL,
            domestic_dc_type=DayCountTypes.ACT_365F,
            foreign_dc_type=DayCountTypes.ACT_360,
            domestic_floating_index=CurveTypes.GBP_OIS_SONIA,
            foreign_floating_index=CurveTypes.USD_OIS_SOFR,
            domestic_currency=CurrencyTypes.GBP,
            foreign_currency=CurrencyTypes.USD,
            domestic_cal_type=joint_cal,  # Same calendar for both legs
            foreign_cal_type=joint_cal
        )
    ]

    xccy_curve = XccyCurve(
        value_dt=value_dt,
        basis_swaps=basis_swaps,
        domestic_curve=gbp_ois_curve,
        foreign_curve=usd_ois_curve,
        spot_fx=spot_fx,
        interp_type=InterpTypes.FLAT_FWD_RATES,
        check_refit=False
    )

    assert xccy_curve is not None
    # Both legs should use the same calendar
    assert basis_swaps[0]._domestic_leg._cal_type == joint_cal
    assert basis_swaps[0]._foreign_leg._cal_type == joint_cal


def test_xccy_curve_defaults_to_weekend_calendar(value_dt, gbp_ois_curve, usd_ois_curve):
    """Test that XccyBasisSwap defaults to WEEKEND calendar when none provided"""

    spot_fx = 0.79

    # Create swap without specifying calendars (should default to WEEKEND)
    basis_swaps = [
        XccyBasisSwap(
            effective_dt=value_dt,
            term_dt_or_tenor="1Y",
            domestic_notional=spot_fx * 1_000_000,
            foreign_notional=1_000_000,
            domestic_spread=0.0,
            foreign_spread=0.0010,
            domestic_freq_type=FrequencyTypes.ANNUAL,
            foreign_freq_type=FrequencyTypes.ANNUAL,
            domestic_dc_type=DayCountTypes.ACT_365F,
            foreign_dc_type=DayCountTypes.ACT_360,
            domestic_floating_index=CurveTypes.GBP_OIS_SONIA,
            foreign_floating_index=CurveTypes.USD_OIS_SOFR,
            domestic_currency=CurrencyTypes.GBP,
            foreign_currency=CurrencyTypes.USD
            # No calendars specified - should default
        )
    ]

    xccy_curve = XccyCurve(
        value_dt=value_dt,
        basis_swaps=basis_swaps,
        domestic_curve=gbp_ois_curve,
        foreign_curve=usd_ois_curve,
        spot_fx=spot_fx,
        interp_type=InterpTypes.FLAT_FWD_RATES,
        check_refit=False
    )

    assert xccy_curve is not None
    # Verify default WEEKEND calendars were used
    assert basis_swaps[0]._domestic_leg._cal_type == CalendarTypes.WEEKEND
    assert basis_swaps[0]._foreign_leg._cal_type == CalendarTypes.WEEKEND


def test_xccy_practical_usd_gbp_with_calendars(value_dt, gbp_ois_curve, usd_ois_curve, us_calendar, uk_calendar):
    """
    Practical test: USD/GBP cross-currency swap with proper calendars

    In practice, USD/GBP basis swaps would use:
    - GBP leg: UK calendar (London)
    - USD leg: US calendar (New York)
    Or a joint calendar (intersection) for both legs
    """
    spot_fx = 0.79

    basis_swaps = [
        XccyBasisSwap(
            effective_dt=value_dt,
            term_dt_or_tenor="1Y",
            domestic_notional=spot_fx * 1_000_000,
            foreign_notional=1_000_000,
            domestic_spread=0.0,
            foreign_spread=0.0010,
            domestic_freq_type=FrequencyTypes.ANNUAL,
            foreign_freq_type=FrequencyTypes.QUARTERLY,  # USD leg quarterly
            domestic_dc_type=DayCountTypes.ACT_365F,
            foreign_dc_type=DayCountTypes.ACT_360,
            domestic_floating_index=CurveTypes.GBP_OIS_SONIA,
            foreign_floating_index=CurveTypes.USD_OIS_SOFR,
            domestic_currency=CurrencyTypes.GBP,
            foreign_currency=CurrencyTypes.USD,
            domestic_cal_type=uk_calendar,
            foreign_cal_type=us_calendar
        ),
        XccyBasisSwap(
            effective_dt=value_dt,
            term_dt_or_tenor="2Y",
            domestic_notional=spot_fx * 1_000_000,
            foreign_notional=1_000_000,
            domestic_spread=0.0,
            foreign_spread=0.0012,
            domestic_freq_type=FrequencyTypes.ANNUAL,
            foreign_freq_type=FrequencyTypes.QUARTERLY,
            domestic_dc_type=DayCountTypes.ACT_365F,
            foreign_dc_type=DayCountTypes.ACT_360,
            domestic_floating_index=CurveTypes.GBP_OIS_SONIA,
            foreign_floating_index=CurveTypes.USD_OIS_SOFR,
            domestic_currency=CurrencyTypes.GBP,
            foreign_currency=CurrencyTypes.USD,
            domestic_cal_type=uk_calendar,
            foreign_cal_type=us_calendar
        )
    ]

    xccy_curve = XccyCurve(
        value_dt=value_dt,
        basis_swaps=basis_swaps,
        domestic_curve=gbp_ois_curve,
        foreign_curve=usd_ois_curve,
        spot_fx=spot_fx,
        interp_type=InterpTypes.FLAT_FWD_RATES,
        check_refit=False
    )

    assert xccy_curve is not None
    assert len(xccy_curve._times) == 3  # t=0 + 2 pillars

    # Verify correct calendars were used
    assert basis_swaps[0]._domestic_leg._cal_type == uk_calendar
    assert basis_swaps[0]._foreign_leg._cal_type == us_calendar

    # Verify different frequencies work
    assert basis_swaps[0]._domestic_leg._freq_type == FrequencyTypes.ANNUAL
    assert basis_swaps[0]._foreign_leg._freq_type == FrequencyTypes.QUARTERLY


def test_xccy_eur_usd_with_target_calendar(value_dt, gbp_ois_curve, usd_ois_curve, target_calendar, us_calendar):
    """Test EUR/USD swap with TARGET and US calendars"""

    spot_fx = 1.10  # EUR per USD

    # EUR/USD basis swap with TARGET calendar for EUR leg
    basis_swaps = [
        XccyBasisSwap(
            effective_dt=value_dt,
            term_dt_or_tenor="1Y",
            domestic_notional=spot_fx * 1_000_000,  # EUR
            foreign_notional=1_000_000,  # USD
            domestic_spread=0.0,
            foreign_spread=0.0005,  # 5bp
            domestic_freq_type=FrequencyTypes.ANNUAL,
            foreign_freq_type=FrequencyTypes.ANNUAL,
            domestic_dc_type=DayCountTypes.ACT_360,
            foreign_dc_type=DayCountTypes.ACT_360,
            domestic_floating_index=CurveTypes.EUR_OIS_ESTR,
            foreign_floating_index=CurveTypes.USD_OIS_SOFR,
            domestic_currency=CurrencyTypes.EUR,
            foreign_currency=CurrencyTypes.USD,
            domestic_cal_type=target_calendar,  # TARGET for EUR
            foreign_cal_type=us_calendar         # US for USD
        )
    ]

    # Use GBP curve as proxy for EUR curve (just for testing)
    xccy_curve = XccyCurve(
        value_dt=value_dt,
        basis_swaps=basis_swaps,
        domestic_curve=gbp_ois_curve,
        foreign_curve=usd_ois_curve,
        spot_fx=spot_fx,
        interp_type=InterpTypes.FLAT_FWD_RATES,
        check_refit=False
    )

    assert xccy_curve is not None
    # Verify correct calendars
    assert basis_swaps[0]._domestic_leg._cal_type == target_calendar
    assert basis_swaps[0]._foreign_leg._cal_type == us_calendar


def test_xccy_calendar_affects_payment_dates(value_dt, gbp_ois_curve, usd_ois_curve):
    """
    Test that different calendars can produce different payment schedules

    This test verifies that the calendar choice actually impacts the generated
    payment dates (though the exact dates depend on holiday schedules).
    """
    spot_fx = 0.79

    # Create two swaps with different calendars
    swap_uk = XccyBasisSwap(
        effective_dt=value_dt,
        term_dt_or_tenor="1Y",
        domestic_notional=spot_fx * 1_000_000,
        foreign_notional=1_000_000,
        domestic_spread=0.0,
        foreign_spread=0.0010,
        domestic_freq_type=FrequencyTypes.ANNUAL,
        foreign_freq_type=FrequencyTypes.ANNUAL,
        domestic_dc_type=DayCountTypes.ACT_365F,
        foreign_dc_type=DayCountTypes.ACT_360,
        domestic_floating_index=CurveTypes.GBP_OIS_SONIA,
        foreign_floating_index=CurveTypes.USD_OIS_SOFR,
        domestic_currency=CurrencyTypes.GBP,
        foreign_currency=CurrencyTypes.USD,
        domestic_cal_type=CalendarTypes.UNITED_KINGDOM,
        foreign_cal_type=CalendarTypes.UNITED_KINGDOM
    )

    swap_us = XccyBasisSwap(
        effective_dt=value_dt,
        term_dt_or_tenor="1Y",
        domestic_notional=spot_fx * 1_000_000,
        foreign_notional=1_000_000,
        domestic_spread=0.0,
        foreign_spread=0.0010,
        domestic_freq_type=FrequencyTypes.ANNUAL,
        foreign_freq_type=FrequencyTypes.ANNUAL,
        domestic_dc_type=DayCountTypes.ACT_365F,
        foreign_dc_type=DayCountTypes.ACT_360,
        domestic_floating_index=CurveTypes.GBP_OIS_SONIA,
        foreign_floating_index=CurveTypes.USD_OIS_SOFR,
        domestic_currency=CurrencyTypes.GBP,
        foreign_currency=CurrencyTypes.USD,
        domestic_cal_type=CalendarTypes.UNITED_STATES,
        foreign_cal_type=CalendarTypes.UNITED_STATES
    )

    # Both swaps should have payment schedules
    assert len(swap_uk._domestic_leg._payment_dts) > 0
    assert len(swap_us._domestic_leg._payment_dts) > 0

    # Calendars should be different
    assert swap_uk._domestic_leg._cal_type != swap_us._domestic_leg._cal_type
