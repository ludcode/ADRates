"""
Comprehensive tests for date arithmetic and operations.

Tests the Date class including:
- add_days, add_months, add_years
- add_tenor parsing (1Y, 3M, 1W, 1D)
- add_weekdays (business day arithmetic)
- Date comparisons (>, <, ==, >=, <=)
- Date subtraction (days between dates)
- is_weekend, is_eom
- Excel date compatibility
- from_string, from_date constructors
- Leap year handling
- Month-end roll conventions
"""

import pytest
import datetime
import numpy as np
from cavour.utils.date import Date, datediff, days_in_month, is_leap_year, date_range


class TestDateAddDays:
    """Test add_days method"""

    def test_add_days_simple(self):
        """Test adding days to a date"""
        dt = Date(15, 6, 2023)
        dt2 = dt.add_days(10)

        assert dt2.d() == 25
        assert dt2.m() == 6
        assert dt2.y() == 2023

    def test_add_days_across_month(self):
        """Test adding days across month boundary"""
        dt = Date(25, 6, 2023)
        dt2 = dt.add_days(10)

        assert dt2.d() == 5
        assert dt2.m() == 7
        assert dt2.y() == 2023

    def test_add_days_across_year(self):
        """Test adding days across year boundary"""
        dt = Date(25, 12, 2023)
        dt2 = dt.add_days(10)

        assert dt2.d() == 4
        assert dt2.m() == 1
        assert dt2.y() == 2024

    def test_add_negative_days(self):
        """Test subtracting days (negative addition)"""
        dt = Date(15, 6, 2023)
        dt2 = dt.add_days(-10)

        assert dt2.d() == 5
        assert dt2.m() == 6
        assert dt2.y() == 2023

    def test_add_zero_days(self):
        """Test adding zero days"""
        dt = Date(15, 6, 2023)
        dt2 = dt.add_days(0)

        assert dt2 == dt


class TestDateAddMonths:
    """Test add_months method"""

    def test_add_months_simple(self):
        """Test adding months"""
        dt = Date(15, 6, 2023)
        dt2 = dt.add_months(3)

        assert dt2.d() == 15
        assert dt2.m() == 9
        assert dt2.y() == 2023

    def test_add_months_across_year(self):
        """Test adding months across year boundary"""
        dt = Date(15, 10, 2023)
        dt2 = dt.add_months(5)

        assert dt2.d() == 15
        assert dt2.m() == 3
        assert dt2.y() == 2024

    def test_add_months_month_end_adjustment(self):
        """Test month-end adjustment when target month is shorter"""
        dt = Date(31, 1, 2023)
        dt2 = dt.add_months(1)  # Feb has only 28 days in 2023

        assert dt2.d() == 28  # Adjusted to last day of Feb
        assert dt2.m() == 2
        assert dt2.y() == 2023

    def test_add_months_leap_year_february(self):
        """Test adding months to land on Feb in leap year"""
        dt = Date(31, 12, 2023)
        dt2 = dt.add_months(2)  # Feb 2024 (leap year)

        assert dt2.d() == 29  # Adjusted to Feb 29
        assert dt2.m() == 2
        assert dt2.y() == 2024

    def test_add_negative_months(self):
        """Test subtracting months"""
        dt = Date(15, 6, 2023)
        dt2 = dt.add_months(-3)

        assert dt2.d() == 15
        assert dt2.m() == 3
        assert dt2.y() == 2023

    def test_add_months_array(self):
        """Test adding multiple months (array input)"""
        dt = Date(15, 6, 2023)
        dts = dt.add_months([0, 3, 6, 12])

        assert len(dts) == 4
        assert dts[0].m() == 6  # +0 months
        assert dts[1].m() == 9  # +3 months
        assert dts[2].m() == 12  # +6 months
        assert dts[3].y() == 2024  # +12 months crosses year


class TestDateAddYears:
    """Test add_years method"""

    def test_add_years_simple(self):
        """Test adding years"""
        dt = Date(15, 6, 2023)
        dt2 = dt.add_years(5)

        assert dt2.d() == 15
        assert dt2.m() == 6
        assert dt2.y() == 2028

    def test_add_years_leap_year_handling(self):
        """Test adding years with Feb 29"""
        dt = Date(29, 2, 2024)  # Leap year
        dt2 = dt.add_years(1)  # 2025 is not leap year

        assert dt2.d() == 28  # Adjusted to Feb 28
        assert dt2.m() == 2
        assert dt2.y() == 2025

    def test_add_fractional_years(self):
        """Test adding fractional years"""
        dt = Date(15, 6, 2023)
        dt2 = dt.add_years(0.5)  # 6 months

        assert dt2.m() == 12  # Approximately 6 months later
        assert dt2.y() == 2023

    def test_add_years_array(self):
        """Test adding multiple years (array input)"""
        dt = Date(15, 6, 2023)
        dts = dt.add_years([1, 5, 10])

        assert len(dts) == 3
        assert dts[0].y() == 2024
        assert dts[1].y() == 2028
        assert dts[2].y() == 2033


class TestDateAddTenor:
    """Test add_tenor method (string parsing)"""

    def test_add_tenor_days(self):
        """Test adding days via tenor string"""
        dt = Date(15, 6, 2023)
        dt2 = dt.add_tenor("10D")

        assert dt2 == dt.add_days(10)

    def test_add_tenor_weeks(self):
        """Test adding weeks via tenor string"""
        dt = Date(15, 6, 2023)
        dt2 = dt.add_tenor("2W")

        assert dt2 == dt.add_days(14)

    def test_add_tenor_months(self):
        """Test adding months via tenor string"""
        dt = Date(15, 6, 2023)
        dt2 = dt.add_tenor("3M")

        assert dt2.m() == 9
        assert dt2.y() == 2023

    def test_add_tenor_years(self):
        """Test adding years via tenor string"""
        dt = Date(15, 6, 2023)
        dt2 = dt.add_tenor("5Y")

        assert dt2.y() == 2028

    def test_add_tenor_overnight(self):
        """Test ON (overnight) tenor"""
        dt = Date(15, 6, 2023)
        dt2 = dt.add_tenor("ON")

        assert dt2 == dt.add_days(1)

    def test_add_tenor_tomorrow_next(self):
        """Test TN (tomorrow-next) tenor"""
        dt = Date(15, 6, 2023)
        dt2 = dt.add_tenor("TN")

        assert dt2 == dt.add_days(1)

    def test_add_tenor_case_insensitive(self):
        """Test that tenor parsing is case-insensitive"""
        dt = Date(15, 6, 2023)
        dt_upper = dt.add_tenor("3M")
        dt_lower = dt.add_tenor("3m")

        assert dt_upper == dt_lower

    def test_add_tenor_array(self):
        """Test adding multiple tenors"""
        dt = Date(15, 6, 2023)
        dts = dt.add_tenor(["1M", "3M", "6M", "1Y"])

        assert len(dts) == 4
        assert dts[0].m() == 7  # +1M
        assert dts[1].m() == 9  # +3M
        assert dts[2].m() == 12  # +6M
        assert dts[3].y() == 2024  # +1Y


class TestDateAddWeekdays:
    """Test add_weekdays method (business days)"""

    def test_add_weekdays_within_week(self):
        """Test adding weekdays within same week"""
        dt = Date(12, 6, 2023)  # Monday
        dt2 = dt.add_weekdays(3)  # Wednesday

        assert not dt2.is_weekend()

    def test_add_weekdays_skip_weekend(self):
        """Test that add_weekdays skips weekends"""
        dt = Date(16, 6, 2023)  # Friday
        dt2 = dt.add_weekdays(1)  # Should be Monday

        assert not dt2.is_weekend()
        assert dt2.weekday() == Date.MON

    def test_add_weekdays_multiple_weeks(self):
        """Test adding weekdays across multiple weeks"""
        dt = Date(12, 6, 2023)  # Monday
        dt2 = dt.add_weekdays(10)  # 2 full weeks

        # 10 business days = 2 weeks
        days_diff = dt2 - dt
        assert days_diff == 14  # 10 weekdays = 14 calendar days

    def test_add_negative_weekdays(self):
        """Test subtracting weekdays"""
        dt = Date(16, 6, 2023)  # Friday
        dt2 = dt.add_weekdays(-5)  # Previous Friday

        assert not dt2.is_weekend()
        days_diff = dt - dt2
        assert days_diff == 7  # 5 weekdays back = 7 calendar days

    def test_add_weekdays_from_weekend(self):
        """Test adding weekdays starting from weekend"""
        dt = Date(17, 6, 2023)  # Saturday
        dt2 = dt.add_weekdays(1)

        assert not dt2.is_weekend()


class TestDateComparisons:
    """Test date comparison operators"""

    def test_date_equality(self):
        """Test date equality"""
        dt1 = Date(15, 6, 2023)
        dt2 = Date(15, 6, 2023)

        assert dt1 == dt2

    def test_date_inequality(self):
        """Test date inequality"""
        dt1 = Date(15, 6, 2023)
        dt2 = Date(16, 6, 2023)

        assert dt1 != dt2

    def test_date_less_than(self):
        """Test less than comparison"""
        dt1 = Date(15, 6, 2023)
        dt2 = Date(16, 6, 2023)

        assert dt1 < dt2
        assert not (dt2 < dt1)

    def test_date_greater_than(self):
        """Test greater than comparison"""
        dt1 = Date(15, 6, 2023)
        dt2 = Date(16, 6, 2023)

        assert dt2 > dt1
        assert not (dt1 > dt2)

    def test_date_less_equal(self):
        """Test less than or equal comparison"""
        dt1 = Date(15, 6, 2023)
        dt2 = Date(15, 6, 2023)
        dt3 = Date(16, 6, 2023)

        assert dt1 <= dt2
        assert dt1 <= dt3

    def test_date_greater_equal(self):
        """Test greater than or equal comparison"""
        dt1 = Date(15, 6, 2023)
        dt2 = Date(15, 6, 2023)
        dt3 = Date(14, 6, 2023)

        assert dt1 >= dt2
        assert dt1 >= dt3


class TestDateSubtraction:
    """Test date subtraction (datediff)"""

    def test_date_subtraction(self):
        """Test subtracting dates"""
        dt1 = Date(15, 6, 2023)
        dt2 = Date(25, 6, 2023)

        diff = dt2 - dt1
        assert diff == 10

    def test_datediff_function(self):
        """Test datediff function"""
        dt1 = Date(1, 1, 2023)
        dt2 = Date(1, 1, 2024)

        diff = datediff(dt1, dt2)
        assert diff == 365  # 2023 is not leap year

    def test_datediff_leap_year(self):
        """Test datediff across leap year"""
        dt1 = Date(1, 1, 2024)
        dt2 = Date(1, 1, 2025)

        diff = datediff(dt1, dt2)
        assert diff == 366  # 2024 is leap year


class TestDateProperties:
    """Test date property methods"""

    def test_is_weekend_saturday(self):
        """Test is_weekend for Saturday"""
        dt = Date(17, 6, 2023)  # Saturday
        assert dt.is_weekend()

    def test_is_weekend_sunday(self):
        """Test is_weekend for Sunday"""
        dt = Date(18, 6, 2023)  # Sunday
        assert dt.is_weekend()

    def test_is_weekend_weekday(self):
        """Test is_weekend for weekday"""
        dt = Date(19, 6, 2023)  # Monday
        assert not dt.is_weekend()

    def test_is_eom_true(self):
        """Test is_eom for end of month"""
        dt = Date(30, 6, 2023)  # June has 30 days
        assert dt.is_eom()

    def test_is_eom_false(self):
        """Test is_eom for non-end of month"""
        dt = Date(29, 6, 2023)
        assert not dt.is_eom()

    def test_is_eom_february_non_leap(self):
        """Test is_eom for Feb 28 in non-leap year"""
        dt = Date(28, 2, 2023)
        assert dt.is_eom()

    def test_is_eom_february_leap(self):
        """Test is_eom for Feb 29 in leap year"""
        dt = Date(29, 2, 2024)
        assert dt.is_eom()

    def test_eom_method(self):
        """Test eom() method returns last day of month"""
        dt = Date(15, 6, 2023)
        eom_dt = dt.eom()

        assert eom_dt.d() == 30
        assert eom_dt.is_eom()

    def test_weekday_property(self):
        """Test weekday() method"""
        dt = Date(12, 6, 2023)  # Monday
        assert dt.weekday() == Date.MON


class TestDateConstructors:
    """Test date construction methods"""

    def test_from_string(self):
        """Test from_string constructor"""
        dt = Date.from_string("15-06-2023", "%d-%m-%Y")

        assert dt.d() == 15
        assert dt.m() == 6
        assert dt.y() == 2023

    def test_from_date_datetime(self):
        """Test from_date constructor with datetime.date"""
        py_date = datetime.date(2023, 6, 15)
        dt = Date.from_date(py_date)

        assert dt.d() == 15
        assert dt.m() == 6
        assert dt.y() == 2023

    def test_datetime_method(self):
        """Test datetime() method returns python datetime"""
        dt = Date(15, 6, 2023)
        py_date = dt.datetime()

        assert isinstance(py_date, datetime.date)
        assert py_date.day == 15
        assert py_date.month == 6
        assert py_date.year == 2023


class TestLeapYearHandling:
    """Test leap year handling"""

    def test_is_leap_year_true(self):
        """Test is_leap_year for leap years"""
        assert is_leap_year(2024)
        assert is_leap_year(2020)
        assert is_leap_year(2000)

    def test_is_leap_year_false(self):
        """Test is_leap_year for non-leap years"""
        assert not is_leap_year(2023)
        assert not is_leap_year(2100)  # Divisible by 100 but not 400
        assert not is_leap_year(1900)

    def test_days_in_month_function(self):
        """Test days_in_month helper function"""
        assert days_in_month(2, 2024) == 29  # Feb in leap year
        assert days_in_month(2, 2023) == 28  # Feb in non-leap year
        assert days_in_month(6, 2023) == 30
        assert days_in_month(12, 2023) == 31


class TestDateRange:
    """Test date_range helper function"""

    def test_date_range_daily(self):
        """Test date_range with daily frequency"""
        start = Date(1, 6, 2023)
        end = Date(5, 6, 2023)

        dates = date_range(start, end, "1D")

        assert len(dates) == 5
        assert dates[0] == start
        assert dates[-1] == end

    def test_date_range_monthly(self):
        """Test date_range with monthly frequency"""
        start = Date(15, 1, 2023)
        end = Date(15, 6, 2023)

        dates = date_range(start, end, "1M")

        assert len(dates) == 6
        assert dates[0].m() == 1
        assert dates[-1].m() == 6

    def test_date_range_empty(self):
        """Test date_range when start > end"""
        start = Date(15, 6, 2023)
        end = Date(14, 6, 2023)

        dates = date_range(start, end)

        assert len(dates) == 0


class TestDateEdgeCases:
    """Test edge cases"""

    def test_date_creation_validation(self):
        """Test that invalid dates raise errors"""
        with pytest.raises(Exception):
            Date(32, 1, 2023)  # Invalid day

        with pytest.raises(Exception):
            Date(29, 2, 2023)  # Feb 29 in non-leap year

    def test_date_year_validation(self):
        """Test that years before 1900 raise errors"""
        with pytest.raises(Exception):
            Date(1, 1, 1899)

    def test_excel_date_compatibility(self):
        """Test Excel date number is generated"""
        dt = Date(1, 1, 2023)
        excel_dt = dt.excel_dt()

        assert excel_dt > 0  # Should be positive integer

    def test_intraday_time_support(self):
        """Test that Date supports hours/minutes/seconds"""
        dt = Date(15, 6, 2023, 14, 30, 45)

        # Date should store time components
        assert dt._hh == 14
        assert dt._mm == 30
        assert dt._ss == 45

    def test_add_hours(self):
        """Test add_hours method"""
        dt = Date(15, 6, 2023, 10, 0, 0)
        dt2 = dt.add_hours(5)

        assert dt2._hh == 15
        assert dt2.d() == 15  # Same day

    def test_add_hours_across_midnight(self):
        """Test add_hours across midnight"""
        dt = Date(15, 6, 2023, 22, 0, 0)
        dt2 = dt.add_hours(5)

        assert dt2.d() == 16  # Next day
        assert dt2._hh == 3  # 3 AM
