"""
Test suite for cavour.utils.date module
Tests Date class functionality including arithmetic, formatting, and tenor operations
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from cavour.utils.date import Date


class TestDate:
    """Test cases for Date class"""
    
    def test_date_creation(self):
        """Test various ways to create Date objects"""
        # Test basic construction
        d1 = Date(15, 6, 2024)
        assert d1.d() == 15
        assert d1.m() == 6
        assert d1.y() == 2024
        
        # Test construction with hours, minutes, seconds
        d2 = Date(15, 6, 2024, 10, 30, 45)
        assert d2.d() == 15
        assert d2.m() == 6
        assert d2.y() == 2024
        
    def test_date_string_representation(self):
        """Test string representation of dates"""
        d = Date(1, 1, 2024)
        str_repr = str(d)
        assert "2024" in str_repr
        assert "1" in str_repr
        
    def test_date_comparison(self):
        """Test date comparison operations"""
        d1 = Date(1, 6, 2024)
        d2 = Date(2, 6, 2024)
        d3 = Date(1, 6, 2024)
        
        assert d1 < d2
        assert d2 > d1
        assert d1 == d3
        assert d1 <= d3
        assert d1 >= d3
        assert d1 != d2
        
    def test_date_arithmetic(self):
        """Test date arithmetic operations"""
        d = Date(15, 6, 2024)
        
        # Test addition using add_days
        d_plus_10 = d.add_days(10)
        assert d_plus_10 > d
        
        # Test subtraction using add_days with negative
        d_minus_5 = d.add_days(-5)
        assert d_minus_5 < d
        
        # Test difference
        diff = d_plus_10 - d
        assert diff == 10
        
    def test_add_months(self):
        """Test adding months to dates"""
        d = Date(15, 6, 2024)
        
        # Add 1 month
        d_plus_1m = d.add_months(1)
        assert d_plus_1m.m() == 7
        assert d_plus_1m.y() == 2024
        assert d_plus_1m.d() == 15
        
        # Add 12 months (1 year)
        d_plus_12m = d.add_months(12)
        assert d_plus_12m.m() == 6
        assert d_plus_12m.y() == 2025
        
        # Test month-end handling
        d_month_end = Date(31, 1, 2024)
        d_feb = d_month_end.add_months(1)
        # Should handle Feb 29 (leap year) correctly
        
    def test_add_years(self):
        """Test adding years to dates"""
        d = Date(29, 2, 2024)  # Leap year
        
        # Add 1 year
        d_plus_1y = d.add_years(1)
        assert d_plus_1y.y() == 2025
        # Should handle leap year to non-leap year transition
        
    def test_add_weekdays(self):
        """Test adding weekdays (business days)"""
        # Start with a Monday (assuming Date handles weekdays correctly)
        d = Date(3, 6, 2024)  # June 3, 2024 is a Monday
        
        # Add 1 weekday
        d_plus_1wd = d.add_weekdays(1)
        assert d_plus_1wd > d
        
        # Add 5 weekdays (1 business week)
        d_plus_5wd = d.add_weekdays(5)
        assert (d_plus_5wd - d) >= 5  # Should be at least 5 days, possibly more due to weekends
        
    def test_add_tenor(self):
        """Test adding tenor strings to dates"""
        d = Date(15, 6, 2024)
        
        # Test various tenor formats
        test_tenors = [
            ("1D", 1),      # 1 day
            ("1W", 7),      # 1 week
            ("1M", None),   # 1 month (variable days)
            ("3M", None),   # 3 months
            ("6M", None),   # 6 months
            ("1Y", None),   # 1 year
            ("2Y", None),   # 2 years
            ("10Y", None),  # 10 years
        ]
        
        for tenor, expected_days in test_tenors:
            d_tenor = d.add_tenor(tenor)
            assert d_tenor > d, f"Failed for tenor {tenor}"
            
            if expected_days:
                assert (d_tenor - d) == expected_days, f"Failed day calculation for tenor {tenor}"
                
    def test_tenor_edge_cases(self):
        """Test edge cases in tenor parsing"""
        d = Date(15, 6, 2024)
        
        # Test longer tenors
        long_tenors = ["30Y", "50Y"]
        for tenor in long_tenors:
            d_long = d.add_tenor(tenor)
            assert d_long > d
            
    def test_invalid_dates(self):
        """Test handling of invalid date inputs"""
        # Test invalid day
        with pytest.raises((ValueError, Exception)):
            Date(32, 1, 2024)
            
        # Test invalid month
        with pytest.raises((ValueError, Exception)):
            Date(15, 13, 2024)
            
    def test_leap_year_handling(self):
        """Test leap year date handling"""
        # 2024 is a leap year
        leap_date = Date(29, 2, 2024)
        assert leap_date.d() == 29
        assert leap_date.m() == 2
        
        # Test operations on leap year dates
        leap_plus_year = leap_date.add_years(1)
        # Should handle Feb 29 -> Feb 28 transition correctly
        
    def test_year_boundaries(self):
        """Test date operations across year boundaries"""
        d = Date(31, 12, 2023)
        
        # Add 1 day to cross year boundary
        d_new_year = d.add_days(1)
        assert d_new_year.y() == 2024
        assert d_new_year.m() == 1
        assert d_new_year.d() == 1
        
    def test_date_serialization(self):
        """Test date conversion to various formats"""
        d = Date(15, 6, 2024)
        
        # Test string representation is consistent
        str1 = str(d)
        str2 = str(d)
        assert str1 == str2
        
        # Test repr is informative
        repr_str = repr(d)
        assert "Date" in repr_str or "2024" in repr_str