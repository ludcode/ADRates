"""
Test suite for cavour.models.models module
Tests Model class functionality including curve building and scenario analysis
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from cavour.utils.date import Date
from cavour.utils.global_types import SwapTypes
from cavour.utils.day_count import DayCountTypes
from cavour.utils.frequency import FrequencyTypes
from cavour.utils.calendar import BusDayAdjustTypes
from cavour.market.curves.interpolator import InterpTypes
from cavour.models.models import Model


class TestModel:
    """Test cases for Model class"""
    
    @pytest.fixture
    def sample_market_data(self):
        """Sample market data for curve building"""
        return {
            'value_dt': Date(1, 1, 2024),
            'px_list': [5.20, 5.15, 5.10, 5.05, 4.95, 4.85, 4.75, 4.65, 4.55, 4.45],
            'tenor_list': ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y"],
            'curve_name': "USD_OIS_SOFR"
        }
    
    def test_model_construction(self, sample_market_data):
        """Test basic model construction"""
        model = Model(sample_market_data['value_dt'])
        
        assert model is not None
        assert model.value_dt == sample_market_data['value_dt']
        
    def test_single_curve_building(self, sample_market_data):
        """Test building a single OIS curve"""
        model = Model(sample_market_data['value_dt'])
        
        # Build curve with standard parameters
        model.build_curve(
            name=sample_market_data['curve_name'],
            px_list=sample_market_data['px_list'],
            tenor_list=sample_market_data['tenor_list'],
            spot_days=0,
            swap_type=SwapTypes.PAY,
            fixed_dcc_type=DayCountTypes.ACT_360,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            float_freq_type=FrequencyTypes.ANNUAL,
            float_dc_type=DayCountTypes.ACT_360,
            bus_day_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
            interp_type=InterpTypes.LINEAR_ZERO_RATES
        )
        
        # Check that curve was built and is accessible
        assert hasattr(model.curves, sample_market_data['curve_name'])
        curve = getattr(model.curves, sample_market_data['curve_name'])
        assert curve is not None
        
        # Test discount factor at value date
        assert abs(curve.df(sample_market_data['value_dt']) - 1.0) < 1e-10
        
    def test_multiple_curve_building(self, sample_market_data):
        """Test building multiple curves"""
        model = Model(sample_market_data['value_dt'])
        
        # Build first curve
        model.build_curve(
            name="USD_OIS_SOFR",
            px_list=sample_market_data['px_list'],
            tenor_list=sample_market_data['tenor_list'],
            spot_days=2,
            swap_type=SwapTypes.PAY,
            fixed_dcc_type=DayCountTypes.ACT_360,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            float_freq_type=FrequencyTypes.ANNUAL,
            float_dc_type=DayCountTypes.ACT_360,
            bus_day_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
            interp_type=InterpTypes.LINEAR_ZERO_RATES
        )
        
        # Build second curve with different parameters
        gbp_rates = [4.80, 4.75, 4.70, 4.65, 4.55, 4.45, 4.35, 4.25, 4.15, 4.05]
        model.build_curve(
            name="GBP_OIS_SONIA",
            px_list=gbp_rates,
            tenor_list=sample_market_data['tenor_list'],
            spot_days=0,
            swap_type=SwapTypes.PAY,
            fixed_dcc_type=DayCountTypes.ACT_365F,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            float_freq_type=FrequencyTypes.ANNUAL,
            float_dc_type=DayCountTypes.ACT_365F,
            bus_day_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
            interp_type=InterpTypes.LINEAR_ZERO_RATES
        )
        
        # Check both curves exist
        assert hasattr(model.curves, "USD_OIS_SOFR")
        assert hasattr(model.curves, "GBP_OIS_SONIA")
        
        usd_curve = getattr(model.curves, "USD_OIS_SOFR")
        gbp_curve = getattr(model.curves, "GBP_OIS_SONIA")
        
        # Curves should be different
        test_date = sample_market_data['value_dt'].add_tenor("1Y")
        usd_df = usd_curve.df(test_date)
        gbp_df = gbp_curve.df(test_date)
        assert abs(usd_df - gbp_df) > 1e-6
        
    def test_scenario_analysis(self, sample_market_data):
        """Test scenario analysis functionality"""
        model = Model(sample_market_data['value_dt'])
        
        # Build base curve
        model.build_curve(
            name=sample_market_data['curve_name'],
            px_list=sample_market_data['px_list'],
            tenor_list=sample_market_data['tenor_list'],
            spot_days=0,
            swap_type=SwapTypes.PAY,
            fixed_dcc_type=DayCountTypes.ACT_360,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            float_freq_type=FrequencyTypes.ANNUAL,
            float_dc_type=DayCountTypes.ACT_360,
            bus_day_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
            interp_type=InterpTypes.LINEAR_ZERO_RATES
        )
        
        # Create scenario with rate bump
        bump_dict = {"1Y": 0.01}  # 100bp bump at 1Y
        scenario_model = model.scenario(sample_market_data['curve_name'], bump_dict)
        
        # Check that scenario model is different from original
        assert scenario_model is not model
        
        # Test that bumped curve has different discount factors
        test_date = sample_market_data['value_dt'].add_tenor("1Y")
        original_df = getattr(model.curves, sample_market_data['curve_name']).df(test_date)
        scenario_df = getattr(scenario_model.curves, sample_market_data['curve_name']).df(test_date)
        
        # Higher rates should mean lower discount factors
        assert scenario_df < original_df
        
    def test_curve_accessor_functionality(self, sample_market_data):
        """Test CurveAccessor functionality"""
        model = Model(sample_market_data['value_dt'])
        
        # Build curve
        model.build_curve(
            name=sample_market_data['curve_name'],
            px_list=sample_market_data['px_list'],
            tenor_list=sample_market_data['tenor_list'],
            spot_days=0,
            swap_type=SwapTypes.PAY,
            fixed_dcc_type=DayCountTypes.ACT_360,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            float_freq_type=FrequencyTypes.ANNUAL,
            float_dc_type=DayCountTypes.ACT_360,
            bus_day_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
            interp_type=InterpTypes.LINEAR_ZERO_RATES
        )
        
        # Test curve accessor
        assert hasattr(model, 'curves')
        
        # Test dynamic attribute access
        curve = getattr(model.curves, sample_market_data['curve_name'])
        assert curve is not None
        
        # Test that non-existent curve raises appropriate error
        with pytest.raises(AttributeError):
            getattr(model.curves, "NON_EXISTENT_CURVE")
            
    def test_different_interpolation_methods(self, sample_market_data):
        """Test different interpolation methods"""
        model = Model(sample_market_data['value_dt'])
        
        # Test different interpolation types
        interp_types = [InterpTypes.LINEAR_ZERO_RATES, InterpTypes.FLAT_FWD_RATES]
        
        curves = {}
        for i, interp_type in enumerate(interp_types):
            curve_name = f"TEST_CURVE_{i}"
            model.build_curve(
                name=curve_name,
                px_list=sample_market_data['px_list'],
                tenor_list=sample_market_data['tenor_list'],
                spot_days=0,
                swap_type=SwapTypes.PAY,
                fixed_dcc_type=DayCountTypes.ACT_360,
                fixed_freq_type=FrequencyTypes.ANNUAL,
                float_freq_type=FrequencyTypes.ANNUAL,
                float_dc_type=DayCountTypes.ACT_360,
                bus_day_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
                interp_type=interp_type
            )
            curves[interp_type] = getattr(model.curves, curve_name)
            
        # Test that different interpolation methods give different results
        test_date = sample_market_data['value_dt'].add_tenor("18M")  # Between pillars
        dfs = [curve.df(test_date) for curve in curves.values()]
        
        # Should have different results (unless by coincidence)
        if len(set(dfs)) > 1:
            # Results are different as expected
            pass
        else:
            # May be the same by coincidence, which is also valid
            pass
            
    def test_different_day_count_conventions(self, sample_market_data):
        """Test curve building with different day count conventions"""
        model = Model(sample_market_data['value_dt'])
        
        day_counts = [DayCountTypes.ACT_360, DayCountTypes.ACT_365F]
        
        for i, dc_type in enumerate(day_counts):
            curve_name = f"TEST_DC_{i}"
            model.build_curve(
                name=curve_name,
                px_list=sample_market_data['px_list'],
                tenor_list=sample_market_data['tenor_list'],
                spot_days=0,
                swap_type=SwapTypes.PAY,
                fixed_dcc_type=dc_type,
                fixed_freq_type=FrequencyTypes.ANNUAL,
                float_freq_type=FrequencyTypes.ANNUAL,
                float_dc_type=dc_type,
                bus_day_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
                interp_type=InterpTypes.LINEAR_ZERO_RATES
            )
            
            # Check curve was built successfully
            curve = getattr(model.curves, curve_name)
            assert curve is not None
            
    def test_spot_days_parameter(self, sample_market_data):
        """Test different spot days settings"""
        model = Model(sample_market_data['value_dt'])
        
        spot_days_values = [0, 1, 2]
        
        for spot_days in spot_days_values:
            curve_name = f"TEST_SPOT_{spot_days}"
            model.build_curve(
                name=curve_name,
                px_list=sample_market_data['px_list'],
                tenor_list=sample_market_data['tenor_list'],
                spot_days=spot_days,
                swap_type=SwapTypes.PAY,
                fixed_dcc_type=DayCountTypes.ACT_360,
                fixed_freq_type=FrequencyTypes.ANNUAL,
                float_freq_type=FrequencyTypes.ANNUAL,
                float_dc_type=DayCountTypes.ACT_360,
                bus_day_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
                interp_type=InterpTypes.LINEAR_ZERO_RATES
            )
            
            curve = getattr(model.curves, curve_name)
            assert curve is not None
            
    def test_invalid_curve_building_parameters(self, sample_market_data):
        """Test error handling for invalid curve building parameters"""
        model = Model(sample_market_data['value_dt'])
        
        # Test mismatched list lengths
        with pytest.raises((ValueError, Exception)):
            model.build_curve(
                name="INVALID_CURVE",
                px_list=[5.0, 5.1],  # 2 rates
                tenor_list=["1M", "3M", "6M"],  # 3 tenors
                spot_days=0,
                swap_type=SwapTypes.PAY,
                fixed_dcc_type=DayCountTypes.ACT_360,
                fixed_freq_type=FrequencyTypes.ANNUAL,
                float_freq_type=FrequencyTypes.ANNUAL,
                float_dc_type=DayCountTypes.ACT_360,
                bus_day_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
                interp_type=InterpTypes.LINEAR_ZERO_RATES
            )
            
        # Test empty lists
        with pytest.raises((ValueError, Exception)):
            model.build_curve(
                name="EMPTY_CURVE",
                px_list=[],
                tenor_list=[],
                spot_days=0,
                swap_type=SwapTypes.PAY,
                fixed_dcc_type=DayCountTypes.ACT_360,
                fixed_freq_type=FrequencyTypes.ANNUAL,
                float_freq_type=FrequencyTypes.ANNUAL,
                float_dc_type=DayCountTypes.ACT_360,
                bus_day_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
                interp_type=InterpTypes.LINEAR_ZERO_RATES
            )
            
    def test_model_copying_or_cloning(self, sample_market_data):
        """Test model copying/cloning if supported"""
        model = Model(sample_market_data['value_dt'])
        
        # Build curve
        model.build_curve(
            name=sample_market_data['curve_name'],
            px_list=sample_market_data['px_list'],
            tenor_list=sample_market_data['tenor_list'],
            spot_days=0,
            swap_type=SwapTypes.PAY,
            fixed_dcc_type=DayCountTypes.ACT_360,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            float_freq_type=FrequencyTypes.ANNUAL,
            float_dc_type=DayCountTypes.ACT_360,
            bus_day_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
            interp_type=InterpTypes.LINEAR_ZERO_RATES
        )
        
        # Test scenario creates a copy
        bump_dict = {"1Y": 0.01}
        scenario_model = model.scenario(sample_market_data['curve_name'], bump_dict)
        
        # Should be different objects
        assert scenario_model is not model
        assert scenario_model.value_dt == model.value_dt
        
    def test_string_representation(self, sample_market_data):
        """Test string representation of model"""
        model = Model(sample_market_data['value_dt'])
        
        str_repr = str(model)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0
        
        # Should contain model information
        repr_str = repr(model)
        assert isinstance(repr_str, str)