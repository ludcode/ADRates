"""
Test suite for XCCY curve calendar functionality.

Validates that XCCY curves correctly accept and use calendar parameters.
"""

import pytest
from cavour.utils.date import Date
from cavour.utils.calendar import (
    Calendar,
    CalendarTypes,
    create_calendar_intersection
)
from cavour.trades.rates.experimental.xccy_curve import XCCYCurve
from cavour.market.curves.discount_curve import DiscountCurve
from cavour.utils.currency import CurrencyTypes
from cavour.utils.global_types import CurveTypes
import numpy as np


@pytest.fixture
def value_dt():
    """Valuation date for tests"""
    return Date(15, 6, 2024)


@pytest.fixture
def gbp_ois_curve(value_dt):
    """Simple GBP OIS curve for testing"""
    times = [0.0, 1.0, 2.0, 5.0]
    dfs = np.array([1.0, 0.95, 0.91, 0.83])
    return DiscountCurve(value_dt, times, dfs)


@pytest.fixture
def usd_ois_curve(value_dt):
    """Simple USD OIS curve for testing"""
    times = [0.0, 1.0, 2.0, 5.0]
    dfs = np.array([1.0, 0.94, 0.89, 0.80])
    return DiscountCurve(value_dt, times, dfs)


@pytest.fixture
def us_calendar():
    """US calendar"""
    return Calendar(CalendarTypes.UNITED_STATES)


@pytest.fixture
def uk_calendar():
    """UK calendar"""
    return Calendar(CalendarTypes.UNITED_KINGDOM)


@pytest.fixture
def joint_calendar(us_calendar, uk_calendar):
    """Joint US-UK calendar for cross-currency swaps"""
    return create_calendar_intersection(us_calendar, uk_calendar)


def test_xccy_curve_accepts_calendar_parameters(value_dt, gbp_ois_curve, usd_ois_curve, us_calendar, uk_calendar):
    """Test that XCCYCurve accepts separate calendars for target and collateral legs"""

    xccy = XCCYCurve(
        value_dt=value_dt,
        target_ois_curve=gbp_ois_curve,
        collateral_ois_curve=usd_ois_curve,
        basis_tenors=["1Y", "2Y"],
        basis_spreads=[10.0, 12.0],
        fx_rate=1.27,
        target_currency=CurrencyTypes.GBP,
        collateral_currency=CurrencyTypes.USD,
        target_calendar=uk_calendar,
        collateral_calendar=us_calendar
    )

    # Verify calendars are stored
    assert xccy._target_calendar is uk_calendar
    assert xccy._collateral_calendar is us_calendar


def test_xccy_curve_accepts_joint_calendar(value_dt, gbp_ois_curve, usd_ois_curve, joint_calendar):
    """Test that XCCYCurve accepts joint calendar for both legs"""

    xccy = XCCYCurve(
        value_dt=value_dt,
        target_ois_curve=gbp_ois_curve,
        collateral_ois_curve=usd_ois_curve,
        basis_tenors=["1Y", "2Y"],
        basis_spreads=[10.0, 12.0],
        fx_rate=1.27,
        target_currency=CurrencyTypes.GBP,
        collateral_currency=CurrencyTypes.USD,
        target_calendar=joint_calendar,
        collateral_calendar=joint_calendar
    )

    # Verify joint calendar is stored
    assert xccy._target_calendar is joint_calendar
    assert xccy._collateral_calendar is joint_calendar
    assert xccy._target_calendar._cal_type == CalendarTypes.INTERSECTION


def test_xccy_curve_defaults_to_weekend_calendar(value_dt, gbp_ois_curve, usd_ois_curve):
    """Test that XCCYCurve defaults to WEEKEND calendar when none provided"""

    xccy = XCCYCurve(
        value_dt=value_dt,
        target_ois_curve=gbp_ois_curve,
        collateral_ois_curve=usd_ois_curve,
        basis_tenors=["1Y", "2Y"],
        basis_spreads=[10.0, 12.0],
        fx_rate=1.27,
        target_currency=CurrencyTypes.GBP,
        collateral_currency=CurrencyTypes.USD
        # No calendars provided
    )

    # Verify default WEEKEND calendars are created
    assert xccy._target_calendar._cal_type == CalendarTypes.WEEKEND
    assert xccy._collateral_calendar._cal_type == CalendarTypes.WEEKEND


def test_xccy_curve_stores_calendar_types(value_dt, gbp_ois_curve, usd_ois_curve, us_calendar, uk_calendar):
    """Test that XCCY stores both Calendar objects and CalendarTypes for leg creation"""

    xccy = XCCYCurve(
        value_dt=value_dt,
        target_ois_curve=gbp_ois_curve,
        collateral_ois_curve=usd_ois_curve,
        basis_tenors=["1Y"],
        basis_spreads=[10.0],
        fx_rate=1.27,
        target_currency=CurrencyTypes.GBP,
        collateral_currency=CurrencyTypes.USD,
        target_calendar=uk_calendar,
        collateral_calendar=us_calendar
    )

    # Verify Calendar objects are stored
    assert xccy._target_calendar is uk_calendar
    assert xccy._collateral_calendar is us_calendar

    # Verify CalendarTypes are extracted for leg creation
    assert xccy._target_cal_type == CalendarTypes.UNITED_KINGDOM
    assert xccy._collateral_cal_type == CalendarTypes.UNITED_STATES

    # Verify legs can be created (they use the CalendarTypes internally)
    maturity_dt = value_dt.add_tenor("1Y")
    target_leg = xccy._create_target_leg(maturity_dt, 0.001)
    collateral_leg = xccy._create_collateral_leg(maturity_dt)

    # Legs should exist and be properly configured
    assert target_leg is not None
    assert collateral_leg is not None


def test_xccy_practical_usd_gbp_joint_calendar(value_dt, gbp_ois_curve, usd_ois_curve):
    """
    Practical test: USD/GBP cross-currency swap with joint calendar

    For a USD/GBP basis swap, payment dates must be business days in
    both New York and London, so we use an intersection calendar.
    """
    us_cal = Calendar(CalendarTypes.UNITED_STATES)
    uk_cal = Calendar(CalendarTypes.UNITED_KINGDOM)
    joint_cal = create_calendar_intersection(us_cal, uk_cal)

    xccy = XCCYCurve(
        value_dt=value_dt,
        target_ois_curve=gbp_ois_curve,
        collateral_ois_curve=usd_ois_curve,
        basis_tenors=["1Y", "2Y", "5Y"],
        basis_spreads=[10.0, 12.0, 15.0],
        fx_rate=1.27,
        target_currency=CurrencyTypes.GBP,
        collateral_currency=CurrencyTypes.USD,
        target_index=CurveTypes.GBP_OIS_SONIA,
        collateral_index=CurveTypes.USD_OIS_SOFR,
        target_calendar=joint_cal,
        collateral_calendar=joint_cal
    )

    # Verify both legs store the joint calendar
    assert xccy._target_calendar._cal_type == CalendarTypes.INTERSECTION
    assert xccy._collateral_calendar._cal_type == CalendarTypes.INTERSECTION

    # Verify CalendarTypes are INTERSECTION for both
    assert xccy._target_cal_type == CalendarTypes.INTERSECTION
    assert xccy._collateral_cal_type == CalendarTypes.INTERSECTION

    # Create a swap leg - it will use INTERSECTION calendar type
    maturity_dt = value_dt.add_tenor("2Y")
    target_leg = xccy._create_target_leg(maturity_dt, 0.0012)

    # Verify the leg was created successfully
    assert target_leg is not None
    payment_dts = target_leg._payment_dts
    assert len(payment_dts) > 0

    # NOTE: SwapFloatLeg creates its own Calendar from CalendarTypes.INTERSECTION
    # The Schedule class will use the intersection logic from calendar.py
    # to ensure payment dates are business days in both US and UK calendars


def test_xccy_eur_usd_with_target_calendar(value_dt, gbp_ois_curve, usd_ois_curve):
    """Test EUR/USD swap with TARGET and US calendars"""
    target_cal = Calendar(CalendarTypes.TARGET)
    us_cal = Calendar(CalendarTypes.UNITED_STATES)

    xccy = XCCYCurve(
        value_dt=value_dt,
        target_ois_curve=gbp_ois_curve,  # Using GBP curve as proxy for EUR
        collateral_ois_curve=usd_ois_curve,
        basis_tenors=["1Y"],
        basis_spreads=[5.0],
        fx_rate=1.10,
        target_currency=CurrencyTypes.EUR,
        collateral_currency=CurrencyTypes.USD,
        target_index=CurveTypes.EUR_OIS_ESTR,
        collateral_index=CurveTypes.USD_OIS_SOFR,
        target_calendar=target_cal,
        collateral_calendar=us_cal
    )

    # Verify correct calendars
    assert xccy._target_calendar._cal_type == CalendarTypes.TARGET
    assert xccy._collateral_calendar._cal_type == CalendarTypes.UNITED_STATES


def test_xccy_mixed_calendars_independent_legs(value_dt, gbp_ois_curve, usd_ois_curve):
    """
    Test that target and collateral legs can have different calendars

    This is useful when the two legs settle independently (non-standard case).
    """
    uk_cal = Calendar(CalendarTypes.UNITED_KINGDOM)
    us_cal = Calendar(CalendarTypes.UNITED_STATES)

    xccy = XCCYCurve(
        value_dt=value_dt,
        target_ois_curve=gbp_ois_curve,
        collateral_ois_curve=usd_ois_curve,
        basis_tenors=["1Y"],
        basis_spreads=[10.0],
        fx_rate=1.27,
        target_calendar=uk_cal,
        collateral_calendar=us_cal
    )

    # Verify different calendars are stored
    assert xccy._target_calendar is uk_cal
    assert xccy._collateral_calendar is us_cal

    # Verify different CalendarTypes are extracted
    assert xccy._target_cal_type == CalendarTypes.UNITED_KINGDOM
    assert xccy._collateral_cal_type == CalendarTypes.UNITED_STATES

    # Create legs - they will use different calendar types
    maturity_dt = value_dt.add_tenor("1Y")
    target_leg = xccy._create_target_leg(maturity_dt, 0.001)
    collateral_leg = xccy._create_collateral_leg(maturity_dt)

    # Verify legs are created successfully
    assert target_leg is not None
    assert collateral_leg is not None

    # Payment dates may differ based on calendar holidays
    # (This is inherent in the SwapFloatLeg schedule generation)
