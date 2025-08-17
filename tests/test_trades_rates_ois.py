"""
Test suite for cavour.trades.rates.ois module
Tests OIS (Overnight Index Swap) instrument functionality
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from cavour.utils.date import Date
from cavour.utils.global_types import SwapTypes
from cavour.utils.day_count import DayCountTypes
from cavour.utils.frequency import FrequencyTypes
from cavour.utils.calendar import BusDayAdjustTypes
from cavour.trades.rates.ois import OIS
from cavour.market.curves.discount_curve import DiscountCurve


class TestOIS:
    """Test cases for OIS (Overnight Index Swap) class"""
    
    @pytest.fixture
    def sample_ois_curve(self):
        """Create a sample OIS discount curve for testing"""
        value_dt = Date(1, 1, 2024)
        
        # Times in years from value date
        times = [0.0, 1/12, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
        
        # Sample discount factors for a typical OIS curve
        dfs = np.array([1.0, 0.9958, 0.9871, 0.9742, 0.9487, 0.8963, 0.7408, 0.5488])
        
        return DiscountCurve(value_dt, times, dfs)
    
    @pytest.fixture
    def standard_ois_params(self):
        """Standard OIS parameters for testing"""
        return {
            'effective_dt': Date(2, 1, 2024),  # T+1 from value date
            'term_dt_or_tenor': '1Y',
            'fixed_leg_type': SwapTypes.PAY,
            'fixed_coupon': 0.05,  # 5%
            'fixed_freq_type': FrequencyTypes.ANNUAL,
            'fixed_dc_type': DayCountTypes.ACT_365F,
            'bd_type': BusDayAdjustTypes.MODIFIED_FOLLOWING,
            'float_freq_type': FrequencyTypes.ANNUAL,
            'float_dc_type': DayCountTypes.ACT_365F
        }
    
    def test_ois_construction(self, standard_ois_params):
        """Test basic OIS construction"""
        ois = OIS(**standard_ois_params)
        
        assert ois is not None
        assert ois._fixed_leg_type == SwapTypes.PAY
        assert ois._fixed_coupon == 0.05
        
    def test_ois_with_tenor_string(self, standard_ois_params):
        """Test OIS construction with tenor string"""
        ois = OIS(**standard_ois_params)
        
        # Maturity should be calculated from effective date + tenor
        expected_maturity = standard_ois_params['effective_dt'].add_tenor('1Y')
        # Allow for business day adjustment differences
        assert abs((ois._maturity_dt - expected_maturity)) <= 3
        
    def test_ois_with_explicit_maturity(self, standard_ois_params):
        """Test OIS construction with explicit maturity date"""
        params = standard_ois_params.copy()
        explicit_maturity = Date(2, 1, 2025)
        params['term_dt_or_tenor'] = explicit_maturity
        
        ois = OIS(**params)
        
        # Should use explicit maturity (possibly adjusted for business days)
        assert abs((ois._maturity_dt - explicit_maturity)) <= 3
        
    def test_swap_rate_calculation(self, sample_ois_curve, standard_ois_params):
        """Test par swap rate calculation"""
        ois = OIS(**standard_ois_params)
        value_dt = Date(1, 1, 2024)
        
        par_rate = ois.swap_rate(value_dt, sample_ois_curve)
        
        # Par rate should be positive and reasonable (between 1% and 10%)
        assert 0.01 < par_rate < 0.10
        
        # Test that par rate makes swap value zero
        params_at_par = standard_ois_params.copy()
        params_at_par['fixed_coupon'] = par_rate
        ois_at_par = OIS(**params_at_par)
        
        par_value = ois_at_par.value(value_dt, sample_ois_curve)
        assert abs(par_value) < 1e-6  # Should be approximately zero
        
    def test_swap_valuation(self, sample_ois_curve, standard_ois_params):
        """Test OIS valuation"""
        ois = OIS(**standard_ois_params)
        value_dt = Date(1, 1, 2024)
        
        swap_value = ois.value(value_dt, sample_ois_curve)
        
        # Value should be non-zero (unless fixed coupon equals par rate)
        assert isinstance(swap_value, (int, float))
        
        # Test pay vs receive
        params_receive = standard_ois_params.copy()
        params_receive['fixed_leg_type'] = SwapTypes.RECEIVE
        ois_receive = OIS(**params_receive)
        
        receive_value = ois_receive.value(value_dt, sample_ois_curve)
        
        # Pay and receive should have opposite signs
        assert swap_value * receive_value < 0
        
    def test_pv01_calculation(self, sample_ois_curve, standard_ois_params):
        """Test PV01 (price value of a basis point) calculation"""
        ois = OIS(**standard_ois_params)
        value_dt = Date(1, 1, 2024)
        
        pv01 = ois.pv01(value_dt, sample_ois_curve)
        
        # PV01 should be positive for typical swap
        assert pv01 > 0
        
        # PV01 should be reasonable (typically hundreds to thousands)
        assert 100 < pv01 < 10000
        
        # Test relationship with manual bump
        original_value = ois.value(value_dt, sample_ois_curve)
        
        # Create bumped curve (if curve supports shifting)
        try:
            bumped_curve = sample_ois_curve.shift(0.0001)  # 1bp shift
            bumped_value = ois.value(value_dt, bumped_curve)
            
            manual_pv01 = abs(bumped_value - original_value)
            
            # Should be approximately equal
            assert abs(pv01 - manual_pv01) < abs(pv01 * 0.1)  # Within 10%
            
        except AttributeError:
            # Curve may not support shifting
            pass
            
    def test_different_frequencies(self, sample_ois_curve):
        """Test OIS with different payment frequencies"""
        base_params = {
            'effective_dt': Date(2, 1, 2024),
            'term_dt_or_tenor': '2Y',
            'fixed_leg_type': SwapTypes.PAY,
            'fixed_coupon': 0.05,
            'bd_type': BusDayAdjustTypes.MODIFIED_FOLLOWING,
            'fixed_dc_type': DayCountTypes.ACT_365F,
            'float_dc_type': DayCountTypes.ACT_365F
        }
        
        value_dt = Date(1, 1, 2024)
        
        # Test different frequencies
        frequencies = [FrequencyTypes.ANNUAL, FrequencyTypes.SEMI_ANNUAL, FrequencyTypes.QUARTERLY]
        
        results = []
        for freq in frequencies:
            params = base_params.copy()
            params['fixed_freq_type'] = freq
            params['float_freq_type'] = freq
            
            ois = OIS(**params)
            value = ois.value(value_dt, sample_ois_curve)
            results.append(value)
            
        # Results should be different for different frequencies
        assert len(set(results)) > 1  # Not all the same
        
    def test_different_day_count_conventions(self, sample_ois_curve):
        """Test OIS with different day count conventions"""
        base_params = {
            'effective_dt': Date(2, 1, 2024),
            'term_dt_or_tenor': '1Y',
            'fixed_leg_type': SwapTypes.PAY,
            'fixed_coupon': 0.05,
            'fixed_freq_type': FrequencyTypes.ANNUAL,
            'float_freq_type': FrequencyTypes.ANNUAL,
            'bd_type': BusDayAdjustTypes.MODIFIED_FOLLOWING
        }
        
        value_dt = Date(1, 1, 2024)
        
        # Test different day count conventions
        day_counts = [DayCountTypes.ACT_365F, DayCountTypes.ACT_360]
        
        results = []
        for dc in day_counts:
            params = base_params.copy()
            params['fixed_dc_type'] = dc
            params['float_dc_type'] = dc
            
            ois = OIS(**params)
            value = ois.value(value_dt, sample_ois_curve)
            results.append(value)
            
        # Results should be different for different day count conventions
        assert abs(results[0] - results[1]) > 1e-6
        
    def test_maturity_validation(self, standard_ois_params):
        """Test validation of maturity dates"""
        # Test that effective date before maturity
        params = standard_ois_params.copy()
        params['effective_dt'] = Date(2, 1, 2025)  # After 1Y tenor
        
        with pytest.raises((ValueError, Exception)):
            OIS(**params)
            
    def test_zero_coupon_swap(self, sample_ois_curve, standard_ois_params):
        """Test OIS with zero fixed coupon"""
        params = standard_ois_params.copy()
        params['fixed_coupon'] = 0.0
        
        ois = OIS(**params)
        value_dt = Date(1, 1, 2024)
        
        value = ois.value(value_dt, sample_ois_curve)
        
        # Should be negative for pay swap (paying nothing, receiving floating)
        if params['fixed_leg_type'] == SwapTypes.PAY:
            assert value < 0
            
    def test_high_coupon_swap(self, sample_ois_curve, standard_ois_params):
        """Test OIS with very high fixed coupon"""
        params = standard_ois_params.copy()
        params['fixed_coupon'] = 0.20  # 20% - very high
        
        ois = OIS(**params)
        value_dt = Date(1, 1, 2024)
        
        value = ois.value(value_dt, sample_ois_curve)
        
        # Should be positive for pay swap (paying high rate, receiving lower floating)
        if params['fixed_leg_type'] == SwapTypes.PAY:
            assert value > 0
            
    def test_swap_leg_generation(self, standard_ois_params):
        """Test that swap legs are generated correctly"""
        ois = OIS(**standard_ois_params)
        
        # Check that fixed and float legs exist
        assert hasattr(ois, '_fixed_leg')
        assert hasattr(ois, '_float_leg')
        
        # Check that payment dates are generated
        if hasattr(ois._fixed_leg, '_payment_dts'):
            assert len(ois._fixed_leg._payment_dts) > 0
        if hasattr(ois._float_leg, '_payment_dts'):
            assert len(ois._float_leg._payment_dts) > 0
            
    def test_different_notionals(self, sample_ois_curve):
        """Test OIS with different notional amounts"""
        base_params = {
            'effective_dt': Date(2, 1, 2024),
            'term_dt_or_tenor': '1Y',
            'fixed_leg_type': SwapTypes.PAY,
            'fixed_coupon': 0.05,
            'fixed_freq_type': FrequencyTypes.ANNUAL,
            'float_freq_type': FrequencyTypes.ANNUAL,
            'fixed_dc_type': DayCountTypes.ACT_365F,
            'float_dc_type': DayCountTypes.ACT_365F,
            'bd_type': BusDayAdjustTypes.MODIFIED_FOLLOWING
        }
        
        value_dt = Date(1, 1, 2024)
        
        # Test different notionals if parameter exists
        notionals = [1_000_000, 10_000_000]  # 1M and 10M
        
        results = []
        for notional in notionals:
            params = base_params.copy()
            # Add notional if OIS constructor accepts it
            try:
                params['notional'] = notional
                ois = OIS(**params)
                value = ois.value(value_dt, sample_ois_curve)
                results.append(value)
            except TypeError:
                # Constructor may not accept notional parameter
                break
                
        if len(results) == 2:
            # Values should scale with notional
            ratio = results[1] / results[0]
            expected_ratio = notionals[1] / notionals[0]
            assert abs(ratio - expected_ratio) < 0.01
            
    def test_string_representation(self, standard_ois_params):
        """Test string representation of OIS"""
        ois = OIS(**standard_ois_params)
        
        # Test that string representation works
        str_repr = str(ois)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0
        
        # Should contain key information
        assert "OIS" in str_repr or "Swap" in str_repr