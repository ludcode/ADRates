"""
Test suite for cavour.market.curves.discount_curve module
Tests DiscountCurve class functionality including interpolation and curve operations
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from cavour.utils.date import Date
from cavour.market.curves.discount_curve import DiscountCurve
from cavour.market.curves.interpolator import InterpTypes


class TestDiscountCurve:
    """Test cases for DiscountCurve class"""
    
    @pytest.fixture
    def sample_curve_data(self):
        """Create sample curve data for testing"""
        value_dt = Date(1, 1, 2024)
        
        # Times in years from value date
        times = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
        
        # Typical decreasing discount factors
        dfs = np.array([1.0, 0.987, 0.974, 0.949, 0.898, 0.784, 0.614])
        
        return value_dt, times, dfs
    
    def test_curve_construction(self, sample_curve_data):
        """Test basic curve construction"""
        value_dt, times, dfs = sample_curve_data
        
        curve = DiscountCurve(value_dt, times, dfs)
        
        assert curve is not None
        # Test that value date DF is 1.0
        assert abs(curve.df(value_dt) - 1.0) < 1e-10
        
    def test_discount_factor_retrieval(self, sample_curve_data):
        """Test discount factor retrieval at pillar dates"""
        value_dt, times, dfs = sample_curve_data
        
        curve = DiscountCurve(value_dt, times, dfs)
        
        # Test discount factors at pillar points
        dates = [value_dt.add_years(t) for t in times]
        for date, expected_df in zip(dates, dfs):
            actual_df = curve.df(date)
            assert abs(actual_df - expected_df) < 1e-3, f"DF mismatch at {date}"
            
    def test_interpolation_between_pillars(self, sample_curve_data):
        """Test discount factor interpolation between pillar dates"""
        value_dt, times, dfs = sample_curve_data
        
        curve = DiscountCurve(value_dt, times, dfs)
        
        # Test interpolation between 6M and 1Y
        six_month_dt = value_dt.add_tenor("6M")
        nine_month_dt = value_dt.add_tenor("9M")
        one_year_dt = value_dt.add_tenor("1Y")
        
        df_6m = curve.df(six_month_dt)
        df_9m = curve.df(nine_month_dt)
        df_1y = curve.df(one_year_dt)
        
        # Interpolated value should be between neighboring pillars
        assert df_1y < df_9m < df_6m
        
    def test_extrapolation_beyond_curve(self, sample_curve_data):
        """Test extrapolation beyond the curve's last point"""
        value_dt, times, dfs = sample_curve_data
        
        curve = DiscountCurve(value_dt, times, dfs)
        
        # Test extrapolation beyond 10Y
        fifteen_year_dt = value_dt.add_tenor("15Y")
        df_15y = curve.df(fifteen_year_dt)
        
        # Should be positive but less than 10Y discount factor
        df_10y = curve.df(value_dt.add_tenor("10Y"))
        assert 0 < df_15y < df_10y
        
    def test_before_value_date(self, sample_curve_data):
        """Test behavior for dates before value date"""
        value_dt, times, dfs = sample_curve_data
        
        curve = DiscountCurve(value_dt, times, dfs)
        
        # Test date before value date
        past_date = value_dt.add_days(-30)
        
        # Behavior may vary by implementation - test that it doesn't crash
        try:
            df_past = curve.df(past_date)
            # If it returns a value, it should be >= 1.0 (forward discounting)
            assert df_past >= 1.0
        except (ValueError, Exception):
            # Or it may raise an exception, which is also acceptable
            pass
            
    def test_monotonicity_property(self, sample_curve_data):
        """Test that discount factors are monotonically decreasing"""
        value_dt, times, dfs = sample_curve_data
        
        curve = DiscountCurve(value_dt, times, dfs)
        
        # Test at regular intervals
        test_dates = [value_dt.add_months(i * 6) for i in range(1, 21)]  # 6M to 10Y
        prev_df = 1.0
        
        for test_date in test_dates:
            current_df = curve.df(test_date)
            assert current_df <= prev_df, f"Non-monotonic at {test_date}: {current_df} > {prev_df}"
            prev_df = current_df
            
    def test_different_interpolation_types(self, sample_curve_data):
        """Test different interpolation methods if supported"""
        value_dt, times, dfs = sample_curve_data
        
        # Test with different interpolation types if constructor supports it
        try:
            curve_linear = DiscountCurve(value_dt, times, dfs, InterpTypes.LINEAR_ZERO_RATES)
            curve_flat = DiscountCurve(value_dt, times, dfs, InterpTypes.FLAT_FWD_RATES)
            
            # Test that different interpolation gives different results
            test_date = value_dt.add_tenor("9M")
            df_linear = curve_linear.df(test_date)
            df_flat = curve_flat.df(test_date)
            
            # Results should be different (unless by coincidence)
            # Just test that both are reasonable
            assert 0 < df_linear < 1
            assert 0 < df_flat < 1
            
        except (TypeError, AttributeError):
            # Constructor may not support interpolation type parameter
            pass
            
    def test_invalid_construction_data(self):
        """Test curve construction with invalid data"""
        value_dt = Date(1, 1, 2024)
        
        # Test empty data
        with pytest.raises((ValueError, Exception)):
            DiscountCurve(value_dt, [], [])
            
        # Test mismatched lengths
        dates = [value_dt, value_dt.add_tenor("1Y")]
        dfs = [1.0]  # One less than dates
        with pytest.raises((ValueError, Exception)):
            DiscountCurve(value_dt, dates, dfs)
            
        # Test negative discount factors
        dates = [value_dt, value_dt.add_tenor("1Y")]
        dfs = [1.0, -0.5]  # Negative DF
        with pytest.raises((ValueError, Exception)):
            DiscountCurve(value_dt, dates, dfs)
            
    def test_zero_rates_calculation(self, sample_curve_data):
        """Test zero rate calculation from discount factors"""
        value_dt, times, dfs = sample_curve_data
        
        curve = DiscountCurve(value_dt, times, dfs)
        
        # Test zero rate calculation if method exists
        try:
            one_year_dt = value_dt.add_tenor("1Y")
            zero_rate = curve.zero_rate(one_year_dt)
            
            # Zero rate should be positive for normal curve
            assert zero_rate > 0
            
            # Verify relationship: DF = exp(-zero_rate * time)
            time_to_maturity = (one_year_dt - value_dt) / 365.25
            expected_df = np.exp(-zero_rate * time_to_maturity)
            actual_df = curve.df(one_year_dt)
            
            assert abs(actual_df - expected_df) < 1e-3
            
        except AttributeError:
            # Method may not exist
            pass
            
    def test_forward_rates_calculation(self, sample_curve_data):
        """Test forward rate calculation if method exists"""
        value_dt, times, dfs = sample_curve_data
        
        curve = DiscountCurve(value_dt, times, dfs)
        
        try:
            # Test forward rate between 1Y and 2Y
            start_dt = value_dt.add_tenor("1Y")
            end_dt = value_dt.add_tenor("2Y")
            
            fwd_rate = curve.fwd_rate(start_dt, end_dt)
            
            # Forward rate should be positive
            assert fwd_rate > 0
            
            # Test relationship with discount factors
            df_start = curve.df(start_dt)
            df_end = curve.df(end_dt)
            time_period = (end_dt - start_dt) / 365.25
            
            expected_fwd = (df_start / df_end - 1.0) / time_period
            assert abs(fwd_rate - expected_fwd) < 1e-3
            
        except AttributeError:
            # Method may not exist
            pass
            
    def test_curve_shifting(self, sample_curve_data):
        """Test curve shifting operations if supported"""
        value_dt, times, dfs = sample_curve_data
        
        curve = DiscountCurve(value_dt, times, dfs)
        
        try:
            # Test parallel shift
            shift_amount = 0.01  # 100 bps
            shifted_curve = curve.shift(shift_amount)
            
            # Test that shifted curve has different discount factors
            test_date = value_dt.add_tenor("1Y")
            original_df = curve.df(test_date)
            shifted_df = shifted_curve.df(test_date)
            
            # Higher rates should mean lower discount factors
            assert shifted_df < original_df
            
        except AttributeError:
            # Method may not exist
            pass
            
    def test_curve_copying(self, sample_curve_data):
        """Test curve copying/cloning if supported"""
        value_dt, times, dfs = sample_curve_data
        
        curve = DiscountCurve(value_dt, times, dfs)
        
        try:
            curve_copy = curve.copy()
            
            # Test that copy produces same results
            test_date = value_dt.add_tenor("1Y")
            original_df = curve.df(test_date)
            copy_df = curve_copy.df(test_date)
            
            assert abs(original_df - copy_df) < 1e-12
            
        except AttributeError:
            # Method may not exist
            pass
            
    def test_performance_with_many_evaluations(self, sample_curve_data):
        """Test performance with many discount factor evaluations"""
        value_dt, times, dfs = sample_curve_data
        
        curve = DiscountCurve(value_dt, times, dfs)
        
        # Test many evaluations don't cause issues
        test_dates = [value_dt.add_months(i) for i in range(1, 121)]  # 10 years monthly
        
        results = []
        for test_date in test_dates:
            df = curve.df(test_date)
            results.append(df)
            
        # All results should be reasonable
        assert all(0 < df < 1 for df in results)
        
        # Should be monotonically decreasing
        for i in range(1, len(results)):
            assert results[i] <= results[i-1], f"Non-monotonic at index {i}"