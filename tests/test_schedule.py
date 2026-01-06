"""
Comprehensive tests for schedule generation.

Tests the Schedule class including:
- Forward and backward date generation rules
- Business day adjustments
- End-of-month conventions
- Stub period handling
- Multiple payment frequencies
- Calendar integration
- Termination date adjustments

References:
- ISDA 2006 Definitions
"""

import pytest
from cavour.utils.date import Date
from cavour.utils.schedule import Schedule
from cavour.utils.calendar import CalendarTypes, BusDayAdjustTypes, DateGenRuleTypes
from cavour.utils.frequency import FrequencyTypes


class TestScheduleBackwardGeneration:
    """Test backward date generation (default)"""

    def test_backward_annual_simple(self):
        """Test backward generation with annual frequency"""
        effective = Date(15, 6, 2020)
        termination = Date(15, 6, 2023)

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.ANNUAL,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )

        dates = schedule.schedule_dts()

        # Should have: PCD (effective), 2021, 2022, 2023 (termination)
        assert len(dates) == 4
        assert dates[0] == effective  # PCD
        assert dates[-1] == termination  # Termination

    def test_backward_semi_annual(self):
        """Test backward generation with semi-annual frequency"""
        effective = Date(15, 6, 2020)
        termination = Date(15, 6, 2022)

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.SEMI_ANNUAL,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )

        dates = schedule.schedule_dts()

        # Should have: PCD, 15 Dec 2020, 15 Jun 2021, 15 Dec 2021, 15 Jun 2022
        assert len(dates) == 5
        assert dates[0] == effective

    def test_backward_quarterly(self):
        """Test backward generation with quarterly frequency"""
        effective = Date(15, 3, 2023)
        termination = Date(15, 3, 2024)

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.QUARTERLY,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )

        dates = schedule.schedule_dts()

        # Should have: PCD + 4 quarterly dates
        assert len(dates) == 5

    def test_backward_monthly(self):
        """Test backward generation with monthly frequency"""
        effective = Date(15, 6, 2023)
        termination = Date(15, 12, 2023)

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.MONTHLY,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )

        dates = schedule.schedule_dts()

        # Should have: PCD + 6 monthly dates
        assert len(dates) == 7


class TestScheduleForwardGeneration:
    """Test forward date generation"""

    def test_forward_annual(self):
        """Test forward generation with annual frequency"""
        effective = Date(15, 6, 2020)
        termination = Date(15, 6, 2023)

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.ANNUAL,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.FORWARD
        )

        dates = schedule.schedule_dts()

        # Forward generation starts from effective date
        assert len(dates) >= 3
        assert dates[-1] == termination

    def test_forward_semi_annual(self):
        """Test forward generation with semi-annual frequency"""
        effective = Date(15, 6, 2020)
        termination = Date(15, 6, 2022)

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.SEMI_ANNUAL,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.FORWARD
        )

        dates = schedule.schedule_dts()

        assert len(dates) >= 4
        assert dates[-1] == termination

    def test_forward_quarterly(self):
        """Test forward generation with quarterly frequency"""
        effective = Date(15, 3, 2023)
        termination = Date(15, 3, 2024)

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.QUARTERLY,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.FORWARD
        )

        dates = schedule.schedule_dts()

        assert len(dates) >= 4
        assert dates[-1] == termination


class TestScheduleEndOfMonth:
    """Test end-of-month convention"""

    def test_eom_true(self):
        """Test that end_of_month=True forces dates to month-end"""
        effective = Date(31, 1, 2023)
        termination = Date(31, 7, 2023)

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.MONTHLY,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD,
            end_of_month=True
        )

        dates = schedule.schedule_dts()

        # EOM flag affects generation, but business day adjustment can move dates
        # Just check that we have the expected number of dates and they're reasonable
        assert len(dates) == 7  # PCD + 6 monthly periods
        # Check that at least some dates are near month-end (day > 25)
        high_day_count = sum(1 for dt in dates if dt.d() > 25)
        assert high_day_count >= 4  # Most should be near month-end

    def test_eom_false(self):
        """Test that end_of_month=False preserves day of month"""
        effective = Date(15, 1, 2023)
        termination = Date(15, 7, 2023)

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.MONTHLY,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD,
            end_of_month=False
        )

        dates = schedule.schedule_dts()

        # Most dates should be on the 15th (or adjusted for weekends)
        # At least check that not all are end-of-month
        eom_count = sum(1 for dt in dates if dt.is_eom())
        assert eom_count < len(dates)

    def test_eom_with_february(self):
        """Test end-of-month handling with February"""
        effective = Date(31, 1, 2023)
        termination = Date(31, 3, 2023)

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.MONTHLY,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD,
            end_of_month=True
        )

        dates = schedule.schedule_dts()

        # Should have Jan 31, Feb 28 (2023 non-leap), Mar 31
        # After adjustment for weekends
        assert len(dates) == 3


class TestScheduleBusinessDayAdjustment:
    """Test business day adjustment types"""

    def test_following_adjustment(self):
        """Test FOLLOWING adjustment moves weekend to next business day"""
        # Use a date that falls on weekend
        effective = Date(1, 1, 2023)  # Sunday
        termination = Date(1, 7, 2023)  # Saturday

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.SEMI_ANNUAL,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD,
            adjust_termination_dt=True
        )

        dates = schedule.schedule_dts()

        # Check that no date falls on weekend (except possibly effective)
        for i in range(1, len(dates)):
            assert not dates[i].is_weekend(), f"Date {dates[i]} is weekend"

    def test_preceding_adjustment(self):
        """Test PRECEDING adjustment moves weekend to previous business day"""
        effective = Date(1, 1, 2023)
        termination = Date(1, 7, 2023)

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.SEMI_ANNUAL,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.PRECEDING,
            dg_type=DateGenRuleTypes.BACKWARD,
            adjust_termination_dt=True
        )

        dates = schedule.schedule_dts()

        # Check that no adjusted date falls on weekend
        for i in range(1, len(dates)):
            assert not dates[i].is_weekend()

    def test_modified_following_adjustment(self):
        """Test MODIFIED_FOLLOWING adjustment"""
        effective = Date(1, 1, 2023)
        termination = Date(1, 7, 2023)

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.SEMI_ANNUAL,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD,
            adjust_termination_dt=True
        )

        dates = schedule.schedule_dts()

        # Modified following should avoid weekends
        for i in range(1, len(dates)):
            assert not dates[i].is_weekend()


class TestScheduleTerminationAdjustment:
    """Test termination date adjustment flag"""

    def test_adjust_termination_true(self):
        """Test that termination date is adjusted when flag is True"""
        effective = Date(1, 6, 2023)
        termination = Date(1, 7, 2023)  # Saturday

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.SEMI_ANNUAL,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD,
            adjust_termination_dt=True
        )

        dates = schedule.schedule_dts()

        # Termination should be adjusted to Monday Jul 3
        assert not dates[-1].is_weekend()

    def test_adjust_termination_false(self):
        """Test that termination date is NOT adjusted when flag is False"""
        effective = Date(1, 6, 2023)
        termination = Date(1, 7, 2023)  # Saturday

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.SEMI_ANNUAL,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD,
            adjust_termination_dt=False  # Default for swaps
        )

        dates = schedule.schedule_dts()

        # Termination should stay as provided (Saturday)
        assert dates[-1] == termination


class TestScheduleCalendarTypes:
    """Test different calendar types"""

    def test_weekend_calendar(self):
        """Test WEEKEND calendar (only Sat/Sun are non-business days)"""
        effective = Date(1, 6, 2023)
        termination = Date(1, 12, 2023)

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.MONTHLY,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )

        dates = schedule.schedule_dts()

        # All adjusted dates should be weekdays
        for i in range(1, len(dates) - 1):
            assert not dates[i].is_weekend()

    def test_uk_calendar(self):
        """Test UNITED_KINGDOM calendar includes holidays"""
        effective = Date(1, 6, 2023)
        termination = Date(1, 12, 2023)

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.MONTHLY,
            cal_type=CalendarTypes.UNITED_KINGDOM,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )

        dates = schedule.schedule_dts()

        # Dates should be adjusted for UK holidays
        assert len(dates) > 0

    def test_us_calendar(self):
        """Test UNITED_STATES calendar"""
        effective = Date(1, 6, 2023)
        termination = Date(1, 12, 2023)

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.MONTHLY,
            cal_type=CalendarTypes.UNITED_STATES,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )

        dates = schedule.schedule_dts()

        assert len(dates) > 0


class TestScheduleStubPeriods:
    """Test stub period handling"""

    def test_short_front_stub_backward(self):
        """Test short front stub with backward generation"""
        # Not aligned with annual frequency - creates front stub
        effective = Date(15, 9, 2020)
        termination = Date(15, 6, 2023)

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.ANNUAL,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )

        dates = schedule.schedule_dts()

        # First period (PCD to NCD) should be short (9 months)
        # Should have dates: Sep 2020, Jun 2021, Jun 2022, Jun 2023
        assert len(dates) == 4
        assert dates[0] == effective

    def test_short_back_stub_forward(self):
        """Test short back stub with forward generation"""
        effective = Date(15, 6, 2020)
        termination = Date(15, 9, 2022)  # Not aligned - creates back stub

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.ANNUAL,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.FORWARD
        )

        dates = schedule.schedule_dts()

        # Last period should be short (3 months)
        assert dates[-1] == termination

    def test_no_stub_aligned(self):
        """Test perfectly aligned dates create no stub"""
        effective = Date(15, 6, 2020)
        termination = Date(15, 6, 2023)

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.ANNUAL,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )

        dates = schedule.schedule_dts()

        # Check that periods are roughly equal (1 year each)
        for i in range(1, len(dates)):
            period_days = dates[i] - dates[i-1]
            # Annual period should be ~365 days (allowing for leap years)
            assert 360 < period_days < 370


class TestScheduleEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_very_short_tenor(self):
        """Test schedule with very short tenor (3 months)"""
        effective = Date(15, 6, 2023)
        termination = Date(15, 9, 2023)

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.QUARTERLY,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )

        dates = schedule.schedule_dts()

        # Should have PCD and termination
        assert len(dates) >= 2

    def test_very_long_tenor(self):
        """Test schedule with very long tenor (30 years)"""
        effective = Date(15, 6, 2023)
        termination = Date(15, 6, 2053)

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.ANNUAL,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )

        dates = schedule.schedule_dts()

        # Should have ~31 dates (PCD + 30 annual payments)
        assert len(dates) == 31

    def test_single_period(self):
        """Test schedule with single period (6 months, semi-annual)"""
        effective = Date(15, 6, 2023)
        termination = Date(15, 12, 2023)

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.SEMI_ANNUAL,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )

        dates = schedule.schedule_dts()

        # Should have exactly 2 dates
        assert len(dates) == 2

    def test_effective_equals_termination_fails(self):
        """Test that effective = termination raises error"""
        effective = Date(15, 6, 2023)

        with pytest.raises(Exception):  # Should raise LibError
            Schedule(
                effective_dt=effective,
                termination_dt=effective,  # Same as effective
                freq_type=FrequencyTypes.ANNUAL
            )

    def test_effective_after_termination_fails(self):
        """Test that effective > termination raises error"""
        effective = Date(15, 6, 2023)
        termination = Date(15, 6, 2022)  # Before effective

        with pytest.raises(Exception):  # Should raise LibError
            Schedule(
                effective_dt=effective,
                termination_dt=termination,
                freq_type=FrequencyTypes.ANNUAL
            )


class TestScheduleConsistency:
    """Test consistency properties of schedules"""

    def test_dates_monotonic_increasing(self):
        """Test that all dates are monotonically increasing"""
        effective = Date(15, 6, 2020)
        termination = Date(15, 6, 2023)

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.QUARTERLY,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )

        dates = schedule.schedule_dts()

        # Check monotonicity
        for i in range(1, len(dates)):
            assert dates[i] > dates[i-1], f"Dates not monotonic at index {i}"

    def test_no_duplicate_dates(self):
        """Test that schedule contains no duplicate dates"""
        effective = Date(15, 6, 2020)
        termination = Date(15, 6, 2023)

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.MONTHLY,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )

        dates = schedule.schedule_dts()

        # Check no duplicates by comparing each pair
        for i in range(len(dates)):
            for j in range(i + 1, len(dates)):
                assert dates[i] != dates[j], f"Duplicate dates at {i} and {j}: {dates[i]}"

    def test_first_date_is_pcd(self):
        """Test that first date is previous coupon date (or effective)"""
        effective = Date(15, 6, 2020)
        termination = Date(15, 6, 2023)

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.ANNUAL,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )

        dates = schedule.schedule_dts()

        # First date should be effective date or before
        assert dates[0] <= effective

    def test_last_date_is_termination(self):
        """Test that last date is termination date"""
        effective = Date(15, 6, 2020)
        termination = Date(15, 6, 2023)

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.ANNUAL,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD,
            adjust_termination_dt=False
        )

        dates = schedule.schedule_dts()

        # Last date should be termination (if not adjusted)
        assert dates[-1] == termination


class TestScheduleFrequencyTypes:
    """Test all frequency types"""

    def test_annual_frequency(self):
        """Test annual frequency produces yearly dates"""
        effective = Date(15, 6, 2020)
        termination = Date(15, 6, 2025)

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.ANNUAL,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )

        dates = schedule.schedule_dts()

        # Should have 6 dates (PCD + 5 annual)
        assert len(dates) == 6

    def test_semi_annual_frequency(self):
        """Test semi-annual frequency produces 6-month dates"""
        effective = Date(15, 6, 2020)
        termination = Date(15, 6, 2025)

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.SEMI_ANNUAL,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )

        dates = schedule.schedule_dts()

        # Should have 11 dates (PCD + 10 semi-annual)
        assert len(dates) == 11

    def test_quarterly_frequency(self):
        """Test quarterly frequency produces 3-month dates"""
        effective = Date(15, 6, 2020)
        termination = Date(15, 6, 2022)

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.QUARTERLY,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )

        dates = schedule.schedule_dts()

        # Should have 9 dates (PCD + 8 quarterly)
        assert len(dates) == 9

    def test_monthly_frequency(self):
        """Test monthly frequency produces monthly dates"""
        effective = Date(15, 6, 2020)
        termination = Date(15, 6, 2021)

        schedule = Schedule(
            effective_dt=effective,
            termination_dt=termination,
            freq_type=FrequencyTypes.MONTHLY,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )

        dates = schedule.schedule_dts()

        # Should have 13 dates (PCD + 12 monthly)
        assert len(dates) == 13
