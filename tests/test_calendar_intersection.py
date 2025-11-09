"""
Test suite for calendar intersection functionality.

Validates that joint calendars correctly handle business days and holidays
across multiple regional calendars.
"""

import pytest
from cavour.utils.date import Date
from cavour.utils.calendar import (
    Calendar,
    CalendarTypes,
    BusDayAdjustTypes,
    create_calendar_intersection
)
from cavour.utils.error import LibError


@pytest.fixture
def us_calendar():
    """US calendar"""
    return Calendar(CalendarTypes.UNITED_STATES)


@pytest.fixture
def uk_calendar():
    """UK calendar"""
    return Calendar(CalendarTypes.UNITED_KINGDOM)


@pytest.fixture
def target_calendar():
    """European TARGET calendar"""
    return Calendar(CalendarTypes.TARGET)


@pytest.fixture
def us_uk_intersection(us_calendar, uk_calendar):
    """Joint US-UK calendar using convenience function"""
    return create_calendar_intersection(us_calendar, uk_calendar)


def test_create_intersection_convenience_function(us_calendar, uk_calendar):
    """Test creating intersection calendar using convenience function"""
    joint_cal = create_calendar_intersection(us_calendar, uk_calendar)

    assert joint_cal._cal_type == CalendarTypes.INTERSECTION
    assert len(joint_cal._constituent_calendars) == 2
    assert us_calendar in joint_cal._constituent_calendars
    assert uk_calendar in joint_cal._constituent_calendars


def test_create_intersection_direct_construction(us_calendar, uk_calendar):
    """Test creating intersection calendar using direct constructor"""
    joint_cal = Calendar(CalendarTypes.INTERSECTION, [us_calendar, uk_calendar])

    assert joint_cal._cal_type == CalendarTypes.INTERSECTION
    assert len(joint_cal._constituent_calendars) == 2


def test_create_intersection_needs_at_least_two_calendars(us_calendar):
    """Test that intersection requires at least 2 calendars"""
    with pytest.raises(LibError, match="at least 2 calendars"):
        create_calendar_intersection(us_calendar)


def test_create_intersection_validates_calendar_types():
    """Test that intersection only accepts Calendar objects"""
    us_cal = Calendar(CalendarTypes.UNITED_STATES)

    with pytest.raises(LibError, match="must be Calendar objects"):
        create_calendar_intersection(us_cal, "not a calendar")


def test_business_day_both_calendars_agree(us_uk_intersection):
    """Test date that is business day in both US and UK"""
    # Wednesday, June 5, 2024 - regular business day in both
    dt = Date(5, 6, 2024)

    assert us_uk_intersection.is_business_day(dt) is True


def test_holiday_in_us_only(us_calendar, uk_calendar, us_uk_intersection):
    """Test US Independence Day (July 4) - holiday in US but not UK"""
    # Thursday, July 4, 2024
    dt = Date(4, 7, 2024)

    # Verify individual calendars
    assert us_calendar.is_holiday(dt) is True
    assert uk_calendar.is_holiday(dt) is False

    # Intersection: holiday in ANY calendar = not a business day
    assert us_uk_intersection.is_holiday(dt) is True
    assert us_uk_intersection.is_business_day(dt) is False


def test_holiday_in_uk_only(us_calendar, uk_calendar, us_uk_intersection):
    """Test UK Summer Bank Holiday - holiday in UK but not US"""
    # Monday, August 26, 2024
    dt = Date(26, 8, 2024)

    # Verify individual calendars
    assert uk_calendar.is_holiday(dt) is True
    assert us_calendar.is_holiday(dt) is False

    # Intersection: holiday in ANY calendar = not a business day
    assert us_uk_intersection.is_holiday(dt) is True
    assert us_uk_intersection.is_business_day(dt) is False


def test_holiday_in_both_calendars(us_calendar, uk_calendar, us_uk_intersection):
    """Test Christmas Day - holiday in both calendars"""
    # Wednesday, December 25, 2024
    dt = Date(25, 12, 2024)

    # Verify individual calendars
    assert us_calendar.is_holiday(dt) is True
    assert uk_calendar.is_holiday(dt) is True

    # Intersection: definitely a holiday
    assert us_uk_intersection.is_holiday(dt) is True
    assert us_uk_intersection.is_business_day(dt) is False


def test_weekend_not_business_day(us_uk_intersection):
    """Test that weekends are not business days"""
    # Saturday, June 1, 2024
    sat = Date(1, 6, 2024)
    assert us_uk_intersection.is_business_day(sat) is False

    # Sunday, June 2, 2024
    sun = Date(2, 6, 2024)
    assert us_uk_intersection.is_business_day(sun) is False


def test_adjustment_following(us_uk_intersection):
    """Test FOLLOWING adjustment skips holidays in either calendar"""
    # July 4, 2024 (Thursday) - US holiday
    us_holiday = Date(4, 7, 2024)

    adjusted = us_uk_intersection.adjust(us_holiday, BusDayAdjustTypes.FOLLOWING)

    # Should move to next joint business day (Friday, July 5)
    assert adjusted == Date(5, 7, 2024)
    assert us_uk_intersection.is_business_day(adjusted) is True


def test_adjustment_preceding(us_uk_intersection):
    """Test PRECEDING adjustment moves backward if needed"""
    # July 4, 2024 (Thursday) - US holiday
    us_holiday = Date(4, 7, 2024)

    adjusted = us_uk_intersection.adjust(us_holiday, BusDayAdjustTypes.PRECEDING)

    # Should move to previous joint business day (Wednesday, July 3)
    assert adjusted == Date(3, 7, 2024)
    assert us_uk_intersection.is_business_day(adjusted) is True


def test_adjustment_modified_following(us_uk_intersection):
    """Test MODIFIED_FOLLOWING stays in same month"""
    # August 26, 2024 (Monday) - UK holiday
    uk_holiday = Date(26, 8, 2024)

    adjusted = us_uk_intersection.adjust(uk_holiday, BusDayAdjustTypes.MODIFIED_FOLLOWING)

    # Should move to next business day (Tuesday, August 27)
    assert adjusted == Date(27, 8, 2024)
    assert adjusted.m() == 8  # Same month
    assert us_uk_intersection.is_business_day(adjusted) is True


def test_adjustment_none_returns_same_date(us_uk_intersection):
    """Test that NONE adjustment returns the date unchanged"""
    dt = Date(4, 7, 2024)  # US holiday

    adjusted = us_uk_intersection.adjust(dt, BusDayAdjustTypes.NONE)

    assert adjusted == dt


def test_three_calendar_intersection(us_calendar, uk_calendar, target_calendar):
    """Test intersection with more than 2 calendars"""
    triple_cal = create_calendar_intersection(us_calendar, uk_calendar, target_calendar)

    assert len(triple_cal._constituent_calendars) == 3

    # May 1 - Labor Day in TARGET, not in US/UK
    may_day = Date(1, 5, 2024)

    # Should be holiday in intersection since TARGET has it
    assert triple_cal.is_holiday(may_day) is True
    assert triple_cal.is_business_day(may_day) is False


def test_add_business_days_with_intersection(us_uk_intersection):
    """Test adding business days with intersection calendar"""
    # Start on Wednesday, July 3, 2024
    start_dt = Date(3, 7, 2024)

    # Add 3 business days
    # July 4 (Thu) = US holiday (skip)
    # July 5 (Fri) = business day 1
    # July 6-7 (Sat-Sun) = weekend (skip)
    # July 8 (Mon) = business day 2
    # July 9 (Tue) = business day 3
    result = us_uk_intersection.add_business_days(start_dt, 3)

    assert result == Date(9, 7, 2024)
    assert us_uk_intersection.is_business_day(result) is True


def test_add_negative_business_days(us_uk_intersection):
    """Test subtracting business days with intersection calendar"""
    # Start on Friday, July 5, 2024
    start_dt = Date(5, 7, 2024)

    # Subtract 1 business day (should skip July 4 US holiday)
    result = us_uk_intersection.add_business_days(start_dt, -1)

    # Should land on Wednesday, July 3
    assert result == Date(3, 7, 2024)
    assert us_uk_intersection.is_business_day(result) is True


def test_intersection_calendar_string_representation(us_uk_intersection):
    """Test string representation of intersection calendar"""
    s = str(us_uk_intersection)
    assert s == "INTERSECTION"


def test_practical_xccy_scenario():
    """
    Practical test: USD/GBP cross-currency swap calendar

    For a USD/GBP cross-currency basis swap, payment dates must be
    business days in BOTH New York and London.
    """
    us_cal = Calendar(CalendarTypes.UNITED_STATES)
    uk_cal = Calendar(CalendarTypes.UNITED_KINGDOM)
    xccy_cal = create_calendar_intersection(us_cal, uk_cal)

    # Test various dates around holidays
    test_cases = [
        # (date, expected_is_business_day, description)
        (Date(5, 6, 2024), True, "Regular Wednesday"),
        (Date(4, 7, 2024), False, "US Independence Day"),
        (Date(26, 8, 2024), False, "UK Summer Bank Holiday"),
        (Date(25, 12, 2024), False, "Christmas in both"),
        (Date(1, 1, 2024), False, "New Year in both"),
        (Date(2, 1, 2024), True, "Regular business day"),
    ]

    for dt, expected, description in test_cases:
        result = xccy_cal.is_business_day(dt)
        assert result == expected, f"Failed for {description}: {dt}"


def test_weekend_calendar():
    """Test WEEKEND calendar as baseline"""
    weekend_cal = Calendar(CalendarTypes.WEEKEND)

    # Saturday
    assert weekend_cal.is_business_day(Date(1, 6, 2024)) is False
    # Sunday
    assert weekend_cal.is_business_day(Date(2, 6, 2024)) is False
    # Monday (no holidays in WEEKEND calendar)
    assert weekend_cal.is_business_day(Date(3, 6, 2024)) is True
    # Christmas is NOT a holiday in WEEKEND calendar
    assert weekend_cal.is_holiday(Date(25, 12, 2024)) is False
