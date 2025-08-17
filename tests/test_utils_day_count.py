"""
Test suite for cavour.utils.day_count module
Tests day count convention calculations and year fraction computations
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from cavour.utils.date import Date
from cavour.utils.day_count import DayCount, DayCountTypes


class TestDayCount:
    """Test cases for DayCount class and day count conventions"""
    
    def test_day_count_types_enum(self):
        """Test that all day count types are available"""
        # Test common day count conventions exist
        assert hasattr(DayCountTypes, 'ACT_360')
        assert hasattr(DayCountTypes, 'ACT_365F')
        assert hasattr(DayCountTypes, 'THIRTY_360_BOND')
        
    def test_act_360_calculation(self):
        """Test ACT/360 day count convention"""
        dc = DayCount(DayCountTypes.ACT_360)
        
        start_date = Date(15, 1, 2024)
        end_date = Date(15, 7, 2024)  # 6 months later
        
        year_frac, days, _ = dc.year_frac(start_date, end_date)
        
        # ACT/360: actual days / 360
        assert days > 0
        assert abs(year_frac - days / 360.0) < 1e-10
        
        # Test half year is approximately 0.5
        assert abs(year_frac - 0.5) < 0.1
        
    def test_act_365f_calculation(self):
        """Test ACT/365F day count convention"""
        dc = DayCount(DayCountTypes.ACT_365F)
        
        start_date = Date(1, 1, 2024)
        end_date = Date(1, 1, 2025)  # Exactly 1 year
        
        year_frac, days, _ = dc.year_frac(start_date, end_date)
        
        # ACT/365F: actual days / 365 (fixed)
        assert days > 360  # Should be 365 or 366 (leap year)
        assert abs(year_frac - days / 365.0) < 1e-10
        
        # For 1 year, should be close to 1.0
        assert abs(year_frac - 1.0) < 0.01
        
    def test_thirty_360_calculation(self):
        """Test 30/360 day count convention"""
        if hasattr(DayCountTypes, 'THIRTY_360_BOND'):
            dc = DayCount(DayCountTypes.THIRTY_360_BOND)
            
            start_date = Date(15, 1, 2024)
            end_date = Date(15, 7, 2024)  # 6 months
            
            year_frac, days, _ = dc.year_frac(start_date, end_date)
            
            # 30/360: assumes 30 days per month, 360 days per year
            # 6 months = 180 days, year_frac = 180/360 = 0.5
            assert abs(year_frac - 0.5) < 1e-10
            
    def test_same_date_calculation(self):
        """Test day count when start and end dates are the same"""
        dc = DayCount(DayCountTypes.ACT_360)
        
        date = Date(15, 6, 2024)
        year_frac, days, _ = dc.year_frac(date, date)
        
        assert days == 0
        assert year_frac == 0.0
        
    def test_reverse_date_calculation(self):
        """Test day count when end date is before start date"""
        dc = DayCount(DayCountTypes.ACT_360)
        
        start_date = Date(15, 7, 2024)
        end_date = Date(15, 1, 2024)  # Before start date
        
        year_frac, days, _ = dc.year_frac(start_date, end_date)
        
        # Should handle negative periods
        assert days < 0
        assert year_frac < 0
        
    def test_leap_year_handling(self):
        """Test day count calculations across leap years"""
        dc_365f = DayCount(DayCountTypes.ACT_365F)
        dc_360 = DayCount(DayCountTypes.ACT_360)
        
        # Test leap year Feb 29
        leap_start = Date(1, 2, 2024)
        leap_end = Date(1, 3, 2024)  # Crosses Feb 29
        
        year_frac_365f, days_365f, _ = dc_365f.year_frac(leap_start, leap_end)
        year_frac_360, days_360, _ = dc_360.year_frac(leap_start, leap_end)
        
        # Should include Feb 29 in day count
        assert days_365f == days_360  # Same actual days
        assert days_365f == 29  # Feb has 29 days in 2024
        
        # But different year fractions due to different denominators
        assert abs(year_frac_365f - days_365f / 365.0) < 1e-10
        assert abs(year_frac_360 - days_360 / 360.0) < 1e-10
        
    def test_month_end_dates(self):
        """Test day count with month-end dates"""
        dc = DayCount(DayCountTypes.ACT_360)
        
        # Test various month ends
        jan_end = Date(31, 1, 2024)
        feb_end = Date(29, 2, 2024)  # Leap year
        
        year_frac, days, _ = dc.year_frac(jan_end, feb_end)
        
        assert days > 0
        assert year_frac > 0
        
    def test_year_boundary_crossing(self):
        """Test day count calculations across year boundaries"""
        dc = DayCount(DayCountTypes.ACT_365F)
        
        year_end = Date(31, 12, 2023)
        next_year = Date(1, 1, 2024)
        
        year_frac, days, _ = dc.year_frac(year_end, next_year)
        
        assert days == 1
        assert abs(year_frac - 1.0 / 365.0) < 1e-10
        
    def test_long_periods(self):
        """Test day count for long periods (multiple years)"""
        dc = DayCount(DayCountTypes.ACT_360)
        
        start = Date(1, 1, 2020)
        end = Date(1, 1, 2025)  # 5 years
        
        year_frac, days, _ = dc.year_frac(start, end)
        
        # Should be approximately 5 years
        assert abs(year_frac - 5.0) < 0.1
        assert days > 1800  # More than 5 * 360
        
    def test_weekend_and_holiday_handling(self):
        """Test that day count includes all calendar days"""
        dc = DayCount(DayCountTypes.ACT_360)
        
        # Test across a weekend
        friday = Date(7, 6, 2024)   # Friday
        monday = Date(10, 6, 2024)  # Monday
        
        year_frac, days, _ = dc.year_frac(friday, monday)
        
        # Should include weekend days
        assert days == 3  # Sat, Sun, Mon
        
    def test_consistency_across_conventions(self):
        """Test that different conventions give consistent relative results"""
        start_date = Date(1, 1, 2024)
        end_date = Date(1, 7, 2024)  # 6 months
        
        dc_360 = DayCount(DayCountTypes.ACT_360)
        dc_365f = DayCount(DayCountTypes.ACT_365F)
        
        year_frac_360, days_360, _ = dc_360.year_frac(start_date, end_date)
        year_frac_365f, days_365f, _ = dc_365f.year_frac(start_date, end_date)
        
        # Same actual days
        assert days_360 == days_365f
        
        # ACT/360 should give higher year fraction than ACT/365F
        assert year_frac_360 > year_frac_365f
        
        # Ratio should be 365/360
        ratio = year_frac_360 / year_frac_365f
        assert abs(ratio - 365.0 / 360.0) < 1e-6
        
    def test_invalid_day_count_type(self):
        """Test handling of invalid day count types"""
        # This test depends on implementation - may not be applicable
        # if all enum values are valid
        pass
        
    def test_precision_and_rounding(self):
        """Test precision of year fraction calculations"""
        dc = DayCount(DayCountTypes.ACT_360)
        
        start = Date(1, 1, 2024)
        end = Date(2, 1, 2024)  # 1 day
        
        year_frac, days, _ = dc.year_frac(start, end)
        
        assert days == 1
        # Should be precise to many decimal places
        expected = 1.0 / 360.0
        assert abs(year_frac - expected) < 1e-12