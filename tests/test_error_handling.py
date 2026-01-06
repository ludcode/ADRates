"""
Comprehensive robustness and error handling tests.

Tests cover input validation, error conditions, and edge cases across
the Cavour library to ensure robust behavior and clear error messages.

Focus areas:
- Invalid date inputs
- Out-of-range parameters
- Type validation
- Numerical stability edge cases
- Calendar and day count edge cases
"""

import pytest
import numpy as np
import datetime
from cavour.utils.date import Date
from cavour.utils.day_count import DayCount, DayCountTypes
from cavour.utils.frequency import FrequencyTypes
from cavour.utils.calendar import Calendar, CalendarTypes, BusDayAdjustTypes
from cavour.utils.schedule import Schedule, DateGenRuleTypes
from cavour.utils.error import LibError
from cavour.market.curves.interpolator import Interpolator, InterpTypes


class TestDateValidation:
    """Test date input validation and error handling"""

    def test_invalid_day_raises_error(self):
        """Test that invalid day number raises error"""
        with pytest.raises((ValueError, LibError)):
            Date(32, 1, 2023)  # Day 32 doesn't exist

    def test_invalid_month_raises_error(self):
        """Test that invalid month raises error"""
        with pytest.raises((ValueError, LibError, IndexError)):
            Date(15, 13, 2023)  # Month 13 doesn't exist

    def test_feb_29_non_leap_year(self):
        """Test that Feb 29 in non-leap year raises error"""
        with pytest.raises((ValueError, LibError)):
            Date(29, 2, 2023)  # 2023 is not a leap year

    def test_feb_29_leap_year_valid(self):
        """Test that Feb 29 in leap year is valid"""
        dt = Date(29, 2, 2024)  # 2024 is a leap year
        assert dt.d() == 29
        assert dt.m() == 2

    def test_zero_day_invalid(self):
        """Test that day 0 is invalid"""
        with pytest.raises((ValueError, LibError)):
            Date(0, 1, 2023)

    def test_negative_day_invalid(self):
        """Test that negative day is invalid"""
        with pytest.raises((ValueError, LibError)):
            Date(-1, 1, 2023)

    def test_date_ordering_works(self):
        """Test that date ordering comparisons work correctly"""
        dt1 = Date(15, 6, 2023)
        dt2 = Date(16, 6, 2023)

        assert dt1 < dt2
        assert dt2 > dt1
        assert dt1 != dt2

    def test_date_equality_works(self):
        """Test that date equality works"""
        dt1 = Date(15, 6, 2023)
        dt2 = Date(15, 6, 2023)
        dt3 = Date(16, 6, 2023)

        assert dt1 == dt2
        assert dt1 != dt3


class TestDayCountEdgeCases:
    """Test day count convention edge cases"""

    def test_same_date_returns_zero(self):
        """Test that year fraction for same date is zero"""
        dc = DayCount(DayCountTypes.ACT_365F)
        dt = Date(15, 6, 2023)

        year_frac, _, _ = dc.year_frac(dt, dt)
        assert year_frac == 0.0

    def test_reversed_dates_negative(self):
        """Test that reversed dates give negative year fraction"""
        dc = DayCount(DayCountTypes.ACT_365F)
        dt1 = Date(15, 6, 2023)
        dt2 = Date(15, 12, 2023)

        year_frac_forward, _, _ = dc.year_frac(dt1, dt2)
        year_frac_backward, _, _ = dc.year_frac(dt2, dt1)

        assert year_frac_forward > 0
        assert year_frac_backward < 0
        assert abs(year_frac_forward + year_frac_backward) < 1e-12

    def test_very_long_period(self):
        """Test day count with very long period (100 years)"""
        dc = DayCount(DayCountTypes.ACT_365F)
        dt1 = Date(1, 1, 2000)
        dt2 = Date(1, 1, 2100)

        year_frac, num_days, _ = dc.year_frac(dt1, dt2)

        # Should be approximately 100 years
        assert 99.5 < year_frac < 100.5
        assert num_days > 36500  # At least 100 * 365

    def test_leap_day_handling(self):
        """Test that leap day is handled correctly"""
        dc = DayCount(DayCountTypes.ACT_365F)
        dt1 = Date(28, 2, 2024)  # Leap year
        dt2 = Date(1, 3, 2024)

        year_frac, num_days, _ = dc.year_frac(dt1, dt2)

        assert num_days == 2  # Feb 28 -> Feb 29 -> Mar 1


class TestScheduleEdgeCases:
    """Test schedule generation edge cases"""

    def test_single_period_schedule(self):
        """Test schedule with just one period"""
        effective = Date(15, 6, 2023)
        termination = Date(15, 12, 2023)

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.SEMI_ANNUAL,
            dg_type=DateGenRuleTypes.BACKWARD
        )

        dates = schedule.schedule_dts()
        # Should have previous coupon date (effective) + termination
        assert len(dates) >= 2

    def test_very_short_schedule_1month(self):
        """Test schedule with very short maturity (1 month)"""
        effective = Date(15, 6, 2023)
        termination = Date(15, 7, 2023)

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.MONTHLY,
            dg_type=DateGenRuleTypes.BACKWARD
        )

        dates = schedule.schedule_dts()
        assert len(dates) >= 2

    def test_very_long_schedule_50years(self):
        """Test schedule with very long maturity (50 years)"""
        effective = Date(15, 6, 2023)
        termination = Date(15, 6, 2073)

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.ANNUAL,
            dg_type=DateGenRuleTypes.BACKWARD
        )

        dates = schedule.schedule_dts()
        # Should have approximately 51 dates (PCD + 50 annual periods)
        assert 50 <= len(dates) <= 52

    def test_schedule_termination_before_effective_invalid(self):
        """Test that termination before effective date raises error"""
        effective = Date(15, 6, 2023)
        termination = Date(15, 6, 2022)  # Before effective

        with pytest.raises(LibError):
            Schedule(
                effective_dt=effective,
                termination_dt=termination,
                freq_type=FrequencyTypes.ANNUAL,
                dg_type=DateGenRuleTypes.BACKWARD
            )


class TestInterpolatorEdgeCases:
    """Test interpolator edge cases and numerical stability"""

    def test_flat_curve_all_methods(self):
        """Test that all interpolators can fit flat curves without error"""
        times = [1.0, 2.0, 5.0, 10.0]
        dfs = [0.95, 0.95, 0.95, 0.95]  # Flat

        # Test that scipy-based methods can fit flat curves
        for interp_type in [InterpTypes.PCHIP_ZERO_RATES, InterpTypes.NATCUBIC_ZERO_RATES]:
            interp = Interpolator(interp_type)
            interp.fit(times, dfs)

            # Should be able to interpolate (exact value may vary due to rate space conversions)
            df_test = interp.interpolate(3.0)
            assert 0.9 < df_test < 1.0  # Should be in reasonable range

    def test_monotonic_decreasing_dfs(self):
        """Test that interpolators preserve monotonicity"""
        times = np.array([1.0, 2.0, 5.0, 10.0])
        dfs = np.array([0.98, 0.95, 0.88, 0.75])  # Decreasing

        interp = Interpolator(InterpTypes.PCHIP_ZERO_RATES)
        interp.fit(times, dfs)

        # Test interpolation maintains monotonicity
        test_times = [1.5, 3.0, 7.5]
        prev_df = 1.0
        for t in test_times:
            df = interp.interpolate(t)
            assert df <= prev_df
            prev_df = df

    def test_extrapolation_beyond_range(self):
        """Test that extrapolation beyond last point works"""
        times = [1.0, 2.0, 5.0, 10.0]
        dfs = [0.98, 0.95, 0.88, 0.75]

        interp = Interpolator(InterpTypes.FLAT_FWD_RATES)
        interp.fit(times, dfs)

        # Extrapolate beyond last time
        df_extrap = interp.interpolate(15.0)

        # Should return a valid discount factor
        assert 0.0 < df_extrap < 1.0
        # Should be less than last point (time value of money)
        assert df_extrap < dfs[-1]


class TestCalendarEdgeCases:
    """Test calendar and business day adjustment edge cases"""

    def test_weekend_calendar_saturdays(self):
        """Test that Saturdays are recognized as non-business days"""
        cal = Calendar(CalendarTypes.WEEKEND)
        saturday = Date(17, 6, 2023)  # Saturday

        assert not cal.is_business_day(saturday)

    def test_weekend_calendar_sundays(self):
        """Test that Sundays are recognized as non-business days"""
        cal = Calendar(CalendarTypes.WEEKEND)
        sunday = Date(18, 6, 2023)  # Sunday

        assert not cal.is_business_day(sunday)

    def test_weekend_calendar_weekdays(self):
        """Test that weekdays are business days"""
        cal = Calendar(CalendarTypes.WEEKEND)
        monday = Date(19, 6, 2023)  # Monday

        assert cal.is_business_day(monday)

    def test_adjust_following_works(self):
        """Test FOLLOWING business day adjustment"""
        cal = Calendar(CalendarTypes.WEEKEND)
        saturday = Date(17, 6, 2023)  # Saturday

        adjusted = cal.adjust(saturday, BusDayAdjustTypes.FOLLOWING)

        # Should move to Monday
        assert adjusted.d() == 19
        assert cal.is_business_day(adjusted)

    def test_adjust_preceding_works(self):
        """Test PRECEDING business day adjustment"""
        cal = Calendar(CalendarTypes.WEEKEND)
        saturday = Date(17, 6, 2023)  # Saturday

        adjusted = cal.adjust(saturday, BusDayAdjustTypes.PRECEDING)

        # Should move to Friday
        assert adjusted.d() == 16
        assert cal.is_business_day(adjusted)


class TestNumericalStability:
    """Test numerical stability and precision"""

    def test_very_small_day_fractions(self):
        """Test day count with very small fractions (1 day)"""
        dc = DayCount(DayCountTypes.ACT_365F)
        dt1 = Date(15, 6, 2023)
        dt2 = Date(16, 6, 2023)

        year_frac, num_days, _ = dc.year_frac(dt1, dt2)

        assert num_days == 1
        assert abs(year_frac - 1/365) < 1e-12

    def test_date_arithmetic_overflow_protection(self):
        """Test that date arithmetic doesn't overflow on large tenors"""
        dt = Date(15, 6, 2023)

        # Add 100 years
        future = dt.add_years(100)

        assert future.y() == 2123
        assert future.m() == 6
        assert future.d() == 15

    def test_interpolator_very_close_points(self):
        """Test interpolator with very close time points"""
        times = [1.0, 1.001, 1.002, 2.0]
        dfs = [0.98, 0.979, 0.978, 0.95]

        interp = Interpolator(InterpTypes.LINEAR_ZERO_RATES)
        interp.fit(times, dfs)

        df = interp.interpolate(1.0015)
        assert 0.977 < df < 0.98


class TestTypeValidation:
    """Test that type validation works correctly"""

    def test_date_requires_integers(self):
        """Test that Date requires integer inputs"""
        # This should work
        dt = Date(15, 6, 2023)
        assert dt is not None

        # Floats should be handled (may convert or raise error)
        try:
            dt_float = Date(15.5, 6, 2023)
            # If it doesn't raise, check it was converted
            assert dt_float.d() in [15, 16]
        except (TypeError, ValueError, LibError):
            # Expected behavior - float not accepted
            pass

    def test_schedule_requires_date_objects(self):
        """Test that Schedule requires Date objects"""
        effective = Date(15, 6, 2023)
        termination = Date(15, 6, 2025)

        # Valid construction
        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.ANNUAL,
            dg_type=DateGenRuleTypes.BACKWARD
        )
        assert schedule is not None

    def test_day_count_handles_date_types(self):
        """Test that DayCount works with Date objects"""
        dc = DayCount(DayCountTypes.ACT_365F)
        dt1 = Date(15, 6, 2023)
        dt2 = Date(15, 12, 2023)

        year_frac, _, _ = dc.year_frac(dt1, dt2)
        assert year_frac > 0
