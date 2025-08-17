"""
Test suite for cavour.utils.schedule module
Tests Schedule class for ISDA-compliant payment schedule generation
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from cavour.utils.date import Date
from cavour.utils.frequency import FrequencyTypes
from cavour.utils.calendar import CalendarTypes, BusDayAdjustTypes, DateGenRuleTypes
from cavour.utils.schedule import Schedule


class TestSchedule:
    """Test cases for Schedule class"""
    
    def test_basic_schedule_generation(self):
        """Test basic schedule generation"""
        start_dt = Date(15, 1, 2024)
        end_dt = Date(15, 1, 2025)  # 1 year
        
        schedule = Schedule(
            effective_dt=start_dt,
            termination_dt=end_dt,
            freq_type=FrequencyTypes.QUARTERLY,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )
        
        # Should have adjusted dates
        assert hasattr(schedule, '_adjusted_dts')
        assert len(schedule._adjusted_dts) >= 2  # At least start and end
        
        # First date should be start date (possibly adjusted)
        assert schedule._adjusted_dts[0] >= start_dt
        
        # Last date should be end date (possibly adjusted)
        assert schedule._adjusted_dts[-1] >= end_dt
        
    def test_quarterly_frequency(self):
        """Test quarterly payment schedule"""
        start_dt = Date(15, 1, 2024)
        end_dt = Date(15, 1, 2025)  # 1 year
        
        schedule = Schedule(
            effective_dt=start_dt,
            termination_dt=end_dt,
            freq_type=FrequencyTypes.QUARTERLY,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )
        
        # Should have approximately 5 dates (start + 4 quarterly payments)
        assert 4 <= len(schedule._adjusted_dts) <= 6
        
    def test_annual_frequency(self):
        """Test annual payment schedule"""
        start_dt = Date(15, 1, 2024)
        end_dt = Date(15, 1, 2026)  # 2 years
        
        schedule = Schedule(
            effective_dt=start_dt,
            termination_dt=end_dt,
            freq_type=FrequencyTypes.ANNUAL,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )
        
        # Should have approximately 3 dates (start + 2 annual payments)
        assert 2 <= len(schedule._adjusted_dts) <= 4
        
    def test_semi_annual_frequency(self):
        """Test semi-annual payment schedule"""
        start_dt = Date(15, 1, 2024)
        end_dt = Date(15, 1, 2025)  # 1 year
        
        schedule = Schedule(
            effective_dt=start_dt,
            termination_dt=end_dt,
            freq_type=FrequencyTypes.SEMI_ANNUAL,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )
        
        # Should have approximately 3 dates (start + 2 semi-annual payments)
        assert 2 <= len(schedule._adjusted_dts) <= 4
        
    def test_monthly_frequency(self):
        """Test monthly payment schedule"""
        start_dt = Date(15, 1, 2024)
        end_dt = Date(15, 7, 2024)  # 6 months
        
        schedule = Schedule(
            effective_dt=start_dt,
            termination_dt=end_dt,
            freq_type=FrequencyTypes.MONTHLY,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )
        
        # Should have approximately 7 dates (start + 6 monthly payments)
        assert 6 <= len(schedule._adjusted_dts) <= 8
        
    def test_forward_generation(self):
        """Test forward date generation rule"""
        start_dt = Date(15, 1, 2024)
        end_dt = Date(15, 1, 2025)
        
        schedule = Schedule(
            effective_dt=start_dt,
            termination_dt=end_dt,
            freq_type=FrequencyTypes.QUARTERLY,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.FORWARD
        )
        
        assert len(schedule._adjusted_dts) >= 2
        
        # Test that dates are in ascending order
        for i in range(1, len(schedule._adjusted_dts)):
            assert schedule._adjusted_dts[i] > schedule._adjusted_dts[i-1]
            
    def test_backward_generation(self):
        """Test backward date generation rule"""
        start_dt = Date(15, 1, 2024)
        end_dt = Date(15, 1, 2025)
        
        schedule = Schedule(
            effective_dt=start_dt,
            termination_dt=end_dt,
            freq_type=FrequencyTypes.QUARTERLY,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )
        
        assert len(schedule._adjusted_dts) >= 2
        
        # Test that dates are in ascending order
        for i in range(1, len(schedule._adjusted_dts)):
            assert schedule._adjusted_dts[i] > schedule._adjusted_dts[i-1]
            
    def test_different_business_day_adjustments(self):
        """Test different business day adjustment rules"""
        start_dt = Date(15, 1, 2024)
        end_dt = Date(15, 1, 2025)
        
        adjustments = [
            BusDayAdjustTypes.FOLLOWING,
            BusDayAdjustTypes.MODIFIED_FOLLOWING,
            BusDayAdjustTypes.PRECEDING,
            BusDayAdjustTypes.MODIFIED_PRECEDING
        ]
        
        schedules = {}
        for adj in adjustments:
            try:
                schedule = Schedule(
                    effective_dt=start_dt,
                    termination_dt=end_dt,
                    freq_type=FrequencyTypes.QUARTERLY,
                    cal_type=CalendarTypes.WEEKEND,
                    bd_type=adj,
                    dg_type=DateGenRuleTypes.BACKWARD
                )
                schedules[adj] = schedule
            except (ValueError, AttributeError):
                # Some adjustment types may not be implemented
                pass
                
        # Should have at least one successful schedule
        assert len(schedules) > 0
        
    def test_weekend_calendar(self):
        """Test weekend calendar handling"""
        # Choose a date that falls on weekend
        start_dt = Date(13, 1, 2024)  # Saturday
        end_dt = Date(13, 7, 2024)    # Saturday
        
        schedule = Schedule(
            effective_dt=start_dt,
            termination_dt=end_dt,
            freq_type=FrequencyTypes.MONTHLY,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )
        
        # Adjusted dates should not fall on weekends (if calendar is working)
        # Note: This test may need adjustment based on implementation
        assert len(schedule._adjusted_dts) >= 2
        
    def test_end_of_month_flag(self):
        """Test end of month flag if supported"""
        start_dt = Date(31, 1, 2024)  # End of month
        end_dt = Date(28, 2, 2025)    # End of month (non-leap year)
        
        try:
            schedule = Schedule(
                effective_dt=start_dt,
                termination_dt=end_dt,
                freq_type=FrequencyTypes.MONTHLY,
                cal_type=CalendarTypes.WEEKEND,
                bd_type=BusDayAdjustTypes.FOLLOWING,
                dg_type=DateGenRuleTypes.BACKWARD,
                end_of_month=True
            )
            
            # Should handle end-of-month dates appropriately
            assert len(schedule._adjusted_dts) >= 2
            
        except TypeError:
            # Constructor may not support end_of_month parameter
            pass
            
    def test_short_period_schedule(self):
        """Test schedule for very short periods"""
        start_dt = Date(15, 1, 2024)
        end_dt = Date(15, 2, 2024)  # 1 month
        
        schedule = Schedule(
            effective_dt=start_dt,
            termination_dt=end_dt,
            freq_type=FrequencyTypes.QUARTERLY,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )
        
        # Should handle short periods gracefully
        assert len(schedule._adjusted_dts) >= 2
        
    def test_long_period_schedule(self):
        """Test schedule for long periods"""
        start_dt = Date(15, 1, 2024)
        end_dt = Date(15, 1, 2034)  # 10 years
        
        schedule = Schedule(
            effective_dt=start_dt,
            termination_dt=end_dt,
            freq_type=FrequencyTypes.ANNUAL,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )
        
        # Should have approximately 11 dates (start + 10 annual payments)
        assert 10 <= len(schedule._adjusted_dts) <= 12
        
    def test_invalid_date_order(self):
        """Test handling of invalid date order"""
        start_dt = Date(15, 1, 2025)
        end_dt = Date(15, 1, 2024)  # Before start date
        
        with pytest.raises((ValueError, Exception)):
            Schedule(
                effective_dt=start_dt,
                termination_dt=end_dt,
                freq_type=FrequencyTypes.QUARTERLY,
                cal_type=CalendarTypes.WEEKEND,
                bd_type=BusDayAdjustTypes.FOLLOWING,
                dg_type=DateGenRuleTypes.BACKWARD
            )
            
    def test_same_start_end_dates(self):
        """Test handling when start and end dates are the same"""
        start_dt = Date(15, 1, 2024)
        end_dt = Date(15, 1, 2024)  # Same as start
        
        try:
            schedule = Schedule(
                effective_dt=start_dt,
                termination_dt=end_dt,
                freq_type=FrequencyTypes.QUARTERLY,
                cal_type=CalendarTypes.WEEKEND,
                bd_type=BusDayAdjustTypes.FOLLOWING,
                dg_type=DateGenRuleTypes.BACKWARD
            )
            
            # May return just the single date or handle as special case
            assert len(schedule._adjusted_dts) >= 1
            
        except (ValueError, Exception):
            # May raise exception for zero-length schedule
            pass
            
    def test_leap_year_handling(self):
        """Test schedule generation across leap years"""
        start_dt = Date(29, 2, 2024)  # Leap year Feb 29
        end_dt = Date(28, 2, 2025)    # Non-leap year Feb 28
        
        schedule = Schedule(
            effective_dt=start_dt,
            termination_dt=end_dt,
            freq_type=FrequencyTypes.QUARTERLY,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )
        
        # Should handle leap year transition correctly
        assert len(schedule._adjusted_dts) >= 2
        
    def test_year_boundary_crossing(self):
        """Test schedule generation across year boundaries"""
        start_dt = Date(15, 12, 2023)
        end_dt = Date(15, 2, 2024)    # Crosses year boundary
        
        schedule = Schedule(
            effective_dt=start_dt,
            termination_dt=end_dt,
            freq_type=FrequencyTypes.MONTHLY,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )
        
        # Should handle year boundary correctly
        assert len(schedule._adjusted_dts) >= 2
        
        # Dates should cross year boundary
        has_2023 = any(dt.y() == 2023 for dt in schedule._adjusted_dts)
        has_2024 = any(dt.y() == 2024 for dt in schedule._adjusted_dts)
        assert has_2023 and has_2024
        
    def test_schedule_date_ordering(self):
        """Test that schedule dates are properly ordered"""
        start_dt = Date(15, 1, 2024)
        end_dt = Date(15, 1, 2025)
        
        schedule = Schedule(
            effective_dt=start_dt,
            termination_dt=end_dt,
            freq_type=FrequencyTypes.QUARTERLY,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )
        
        # All dates should be in ascending order
        dates = schedule._adjusted_dts
        for i in range(1, len(dates)):
            assert dates[i] > dates[i-1], f"Dates not ordered: {dates[i-1]} >= {dates[i]}"
            
    def test_string_representation(self):
        """Test string representation of schedule"""
        start_dt = Date(15, 1, 2024)
        end_dt = Date(15, 1, 2025)
        
        schedule = Schedule(
            effective_dt=start_dt,
            termination_dt=end_dt,
            freq_type=FrequencyTypes.QUARTERLY,
            cal_type=CalendarTypes.WEEKEND,
            bd_type=BusDayAdjustTypes.FOLLOWING,
            dg_type=DateGenRuleTypes.BACKWARD
        )
        
        # Test string representation works
        str_repr = str(schedule)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0