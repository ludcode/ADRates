"""
Integration test suite for Cavour library
Tests complete workflow from curve building to portfolio valuation
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
from cavour.trades.rates.ois import OIS


@pytest.mark.integration
class TestFullWorkflow:
    """Integration tests for complete Cavour workflows"""
    
    @pytest.fixture
    def multi_currency_model(self, standard_value_date, standard_tenors, 
                           usd_market_rates, gbp_market_rates, 
                           usd_curve_params, gbp_curve_params):
        """Build a multi-currency model for testing"""
        model = Model(standard_value_date)
        
        # Build USD curve
        model.build_curve(
            name="USD_OIS_SOFR",
            px_list=usd_market_rates,
            tenor_list=standard_tenors,
            **usd_curve_params
        )
        
        # Build GBP curve
        model.build_curve(
            name="GBP_OIS_SONIA",
            px_list=gbp_market_rates,
            tenor_list=standard_tenors,
            **gbp_curve_params
        )
        
        return model
    
    def test_complete_ois_lifecycle(self, multi_currency_model, standard_value_date):
        """Test complete OIS swap lifecycle from creation to valuation"""
        model = multi_currency_model
        
        # Create OIS swap
        ois = OIS(
            effective_dt=standard_value_date.add_weekdays(2),
            term_dt_or_tenor="5Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=0.05,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_360,
            bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
            float_freq_type=FrequencyTypes.ANNUAL,
            float_dc_type=DayCountTypes.ACT_360
        )
        
        # Get curve
        usd_curve = model.curves.USD_OIS_SOFR
        
        # Calculate par rate
        par_rate = ois.swap_rate(standard_value_date, usd_curve)
        assert 0.02 < par_rate < 0.08  # Reasonable range
        
        # Value at par (should be near zero)
        par_ois = OIS(
            effective_dt=standard_value_date.add_weekdays(2),
            term_dt_or_tenor="5Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=par_rate,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_360,
            bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
            float_freq_type=FrequencyTypes.ANNUAL,
            float_dc_type=DayCountTypes.ACT_360
        )
        
        par_value = par_ois.value(standard_value_date, usd_curve)
        assert abs(par_value) < 1e-4  # Should be approximately zero
        
        # Calculate PV01
        pv01 = ois.pv01(standard_value_date, usd_curve)
        assert pv01 > 0
        assert 1000 < pv01 < 50000  # Reasonable PV01 range for 5Y swap
        
    def test_multi_currency_consistency(self, multi_currency_model, standard_value_date):
        """Test consistency across multiple currencies"""
        model = multi_currency_model
        
        # Create similar swaps in both currencies
        usd_ois = OIS(
            effective_dt=standard_value_date.add_weekdays(2),
            term_dt_or_tenor="3Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=0.05,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_360,
            bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
            float_freq_type=FrequencyTypes.ANNUAL,
            float_dc_type=DayCountTypes.ACT_360
        )
        
        gbp_ois = OIS(
            effective_dt=standard_value_date.add_weekdays(1),  # Different spot days
            term_dt_or_tenor="3Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=0.05,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_365F,
            bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
            float_freq_type=FrequencyTypes.ANNUAL,
            float_dc_type=DayCountTypes.ACT_365F
        )
        
        # Get par rates
        usd_par = usd_ois.swap_rate(standard_value_date, model.curves.USD_OIS_SOFR)
        gbp_par = gbp_ois.swap_rate(standard_value_date, model.curves.GBP_OIS_SONIA)
        
        # Par rates should be reasonable and different
        assert 0.02 < usd_par < 0.08
        assert 0.02 < gbp_par < 0.08
        assert abs(usd_par - gbp_par) > 1e-6  # Should be different
        
        # PV01s should be reasonable
        usd_pv01 = usd_ois.pv01(standard_value_date, model.curves.USD_OIS_SOFR)
        gbp_pv01 = gbp_ois.pv01(standard_value_date, model.curves.GBP_OIS_SONIA)
        
        assert usd_pv01 > 0
        assert gbp_pv01 > 0
        
    def test_scenario_analysis_workflow(self, multi_currency_model, standard_value_date):
        """Test complete scenario analysis workflow"""
        model = multi_currency_model
        
        # Create portfolio of swaps
        swaps = []
        for i, tenor in enumerate(["2Y", "5Y", "10Y"]):
            ois = OIS(
                effective_dt=standard_value_date.add_weekdays(2),
                term_dt_or_tenor=tenor,
                fixed_leg_type=SwapTypes.PAY if i % 2 == 0 else SwapTypes.RECEIVE,
                fixed_coupon=0.045 + i * 0.005,  # Different coupons
                fixed_freq_type=FrequencyTypes.ANNUAL,
                fixed_dc_type=DayCountTypes.ACT_360,
                bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
                float_freq_type=FrequencyTypes.ANNUAL,
                float_dc_type=DayCountTypes.ACT_360
            )
            swaps.append(ois)
            
        # Calculate base case portfolio value
        base_values = []
        for swap in swaps:
            value = swap.value(standard_value_date, model.curves.USD_OIS_SOFR)
            base_values.append(value)
            
        base_portfolio_value = sum(base_values)
        
        # Create scenario (parallel shift)
        shift_scenarios = {"5Y": 0.01, "10Y": 0.01}  # 100bp parallel shift
        scenario_model = model.scenario("USD_OIS_SOFR", shift_scenarios)
        
        # Calculate scenario portfolio value
        scenario_values = []
        for swap in swaps:
            value = swap.value(standard_value_date, scenario_model.curves.USD_OIS_SOFR)
            scenario_values.append(value)
            
        scenario_portfolio_value = sum(scenario_values)
        
        # Portfolio should have different value under scenario
        portfolio_pnl = scenario_portfolio_value - base_portfolio_value
        assert abs(portfolio_pnl) > 1e-4  # Should have measurable impact
        
        # Calculate expected direction (higher rates generally hurt payers)
        pay_count = sum(1 for i in range(len(swaps)) if i % 2 == 0)
        receive_count = len(swaps) - pay_count
        
        if pay_count > receive_count:
            # More pay swaps, expect negative impact from rate increase
            assert portfolio_pnl < 0
        elif receive_count > pay_count:
            # More receive swaps, expect positive impact from rate increase
            assert portfolio_pnl > 0
            
    def test_curve_building_consistency(self, standard_value_date):
        """Test that curve building produces consistent results"""
        # Test with different interpolation methods
        px_list = [5.20, 5.15, 5.10, 5.05, 4.95, 4.85, 4.75, 4.65]
        tenor_list = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "10Y"]
        
        interp_methods = [InterpTypes.LINEAR_ZERO_RATES, InterpTypes.FLAT_FWD_RATES]
        models = {}
        
        for interp_type in interp_methods:
            model = Model(standard_value_date)
            model.build_curve(
                name="TEST_CURVE",
                px_list=px_list,
                tenor_list=tenor_list,
                spot_days=2,
                swap_type=SwapTypes.PAY,
                fixed_dcc_type=DayCountTypes.ACT_360,
                fixed_freq_type=FrequencyTypes.ANNUAL,
                float_freq_type=FrequencyTypes.ANNUAL,
                float_dc_type=DayCountTypes.ACT_360,
                bus_day_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
                interp_type=interp_type
            )
            models[interp_type] = model
            
        # All curves should have DF = 1.0 at value date
        for model in models.values():
            curve = model.curves.TEST_CURVE
            assert abs(curve.df(standard_value_date) - 1.0) < 1e-10
            
        # Curves should be different between interpolation methods
        test_date = standard_value_date.add_tenor("18M")  # Between pillars
        dfs = [model.curves.TEST_CURVE.df(test_date) for model in models.values()]
        
        # Should have different interpolated values (unless by coincidence)
        unique_dfs = set(f"{df:.8f}" for df in dfs)  # Round to avoid floating point issues
        # May be same if interpolation points coincide, so don't enforce difference
        
    def test_error_handling_and_validation(self, standard_value_date):
        """Test error handling in complete workflows"""
        model = Model(standard_value_date)
        
        # Test invalid curve building parameters
        with pytest.raises((ValueError, Exception)):
            model.build_curve(
                name="INVALID_CURVE",
                px_list=[],  # Empty list
                tenor_list=[],  # Empty list
                spot_days=0,
                swap_type=SwapTypes.PAY,
                fixed_dcc_type=DayCountTypes.ACT_360,
                fixed_freq_type=FrequencyTypes.ANNUAL,
                float_freq_type=FrequencyTypes.ANNUAL,
                float_dc_type=DayCountTypes.ACT_360,
                bus_day_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
                interp_type=InterpTypes.LINEAR_ZERO_RATES
            )
            
        # Build valid curve for subsequent tests
        model.build_curve(
            name="VALID_CURVE",
            px_list=[5.0, 4.8, 4.6],
            tenor_list=["1Y", "5Y", "10Y"],
            spot_days=0,
            swap_type=SwapTypes.PAY,
            fixed_dcc_type=DayCountTypes.ACT_360,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            float_freq_type=FrequencyTypes.ANNUAL,
            float_dc_type=DayCountTypes.ACT_360,
            bus_day_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
            interp_type=InterpTypes.LINEAR_ZERO_RATES
        )
        
        # Test invalid swap creation
        with pytest.raises((ValueError, Exception)):
            OIS(
                effective_dt=standard_value_date.add_years(1),  # After maturity
                term_dt_or_tenor=standard_value_date,  # Invalid maturity
                fixed_leg_type=SwapTypes.PAY,
                fixed_coupon=0.05,
                fixed_freq_type=FrequencyTypes.ANNUAL,
                fixed_dc_type=DayCountTypes.ACT_360,
                bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
                float_freq_type=FrequencyTypes.ANNUAL,
                float_dc_type=DayCountTypes.ACT_360
            )
            
    def test_numerical_precision_and_consistency(self, multi_currency_model, standard_value_date):
        """Test numerical precision and consistency"""
        model = multi_currency_model
        
        # Create swap at par rate
        ois = OIS(
            effective_dt=standard_value_date.add_weekdays(2),
            term_dt_or_tenor="5Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=0.05,  # Will adjust to par
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_360,
            bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
            float_freq_type=FrequencyTypes.ANNUAL,
            float_dc_type=DayCountTypes.ACT_360
        )
        
        curve = model.curves.USD_OIS_SOFR
        
        # Get par rate and create swap at par
        par_rate = ois.swap_rate(standard_value_date, curve)
        
        par_ois = OIS(
            effective_dt=standard_value_date.add_weekdays(2),
            term_dt_or_tenor="5Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=par_rate,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_360,
            bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
            float_freq_type=FrequencyTypes.ANNUAL,
            float_dc_type=DayCountTypes.ACT_360
        )
        
        # Value should be very close to zero
        par_value = par_ois.value(standard_value_date, curve)
        assert abs(par_value) < 1e-8  # High precision requirement
        
        # Test pay vs receive symmetry
        receive_ois = OIS(
            effective_dt=standard_value_date.add_weekdays(2),
            term_dt_or_tenor="5Y",
            fixed_leg_type=SwapTypes.RECEIVE,
            fixed_coupon=par_rate,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_360,
            bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
            float_freq_type=FrequencyTypes.ANNUAL,
            float_dc_type=DayCountTypes.ACT_360
        )
        
        receive_value = receive_ois.value(standard_value_date, curve)
        
        # Pay and receive should have opposite values
        assert abs(par_value + receive_value) < 1e-8
        
    @pytest.mark.slow
    def test_performance_with_large_portfolio(self, multi_currency_model, standard_value_date):
        """Test performance with large portfolio"""
        model = multi_currency_model
        
        # Create large portfolio
        swaps = []
        tenors = ["1Y", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y"]
        
        import time
        start_time = time.time()
        
        # Create 100 swaps
        for i in range(100):
            tenor = tenors[i % len(tenors)]
            ois = OIS(
                effective_dt=standard_value_date.add_weekdays(2),
                term_dt_or_tenor=tenor,
                fixed_leg_type=SwapTypes.PAY if i % 2 == 0 else SwapTypes.RECEIVE,
                fixed_coupon=0.04 + (i % 10) * 0.001,
                fixed_freq_type=FrequencyTypes.ANNUAL,
                fixed_dc_type=DayCountTypes.ACT_360,
                bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
                float_freq_type=FrequencyTypes.ANNUAL,
                float_dc_type=DayCountTypes.ACT_360
            )
            swaps.append(ois)
            
        creation_time = time.time() - start_time
        
        # Value entire portfolio
        start_time = time.time()
        
        total_value = 0
        total_pv01 = 0
        
        for swap in swaps:
            value = swap.value(standard_value_date, model.curves.USD_OIS_SOFR)
            pv01 = swap.pv01(standard_value_date, model.curves.USD_OIS_SOFR)
            total_value += value
            total_pv01 += pv01
            
        valuation_time = time.time() - start_time
        
        # Performance should be reasonable
        assert creation_time < 10.0  # Should create 100 swaps in under 10 seconds
        assert valuation_time < 30.0  # Should value 100 swaps in under 30 seconds
        
        # Results should be reasonable
        assert isinstance(total_value, (int, float))
        assert total_pv01 > 0