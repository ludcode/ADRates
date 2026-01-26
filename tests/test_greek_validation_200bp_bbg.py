"""
Greek Validation Test Suite: 200bp Bump Scenarios with Bloomberg Data

This module provides comprehensive validation of automatic differentiation (AD)
based sensitivities against full revaluation using 200bp bump scenarios. Tests
the accuracy of Taylor series approximations using DELTAs, GAMMAs, and CrossGammas
for both OIS swaps and XCCY basis swaps.

Test Coverage:
1. OIS Swap (GBP SONIA):
   - Single point bump (200bp)
   - Parallel bump (200bp)
   - Slope scenario (200bp)
   - Skew scenario (200bp)
   - Butterfly scenario (200bp)

2. XCCY Basis Swap (GBP/USD):
   - Single point bumps on each curve (USD OIS, GBP OIS, BASIS)
   - Parallel bumps on each curve
   - Slope scenarios on each curve
   - Skew scenarios on each curve
   - Cross-gamma validation (1bp bumps)

Expected Tolerances for 200bp:
- Delta-only (single point): 15-20% (demonstrates non-linearity)
- Delta+Gamma (single point): 5-7% (captures quadratic effects)
- Parallel shock (Taylor): 7-10%
- Scenario shocks: 10-12%
- Cross-gamma (1bp bumps): 1-2%

Author: Generated for Cavour ADRates Library
Date: 2026-01-25
"""

import pytest
import numpy as np
from datetime import datetime

# Cavour imports
from cavour.utils.date import Date
from cavour.utils.global_types import SwapTypes, RequestTypes, CurveTypes
from cavour.utils.day_count import DayCountTypes
from cavour.utils.frequency import FrequencyTypes
from cavour.utils.calendar import BusDayAdjustTypes
from cavour.utils.currency import CurrencyTypes
from cavour.models.models import Model
from cavour.trades.rates.ois import OIS
from cavour.trades.rates.xccy_basis_swap import XccyBasisSwap
from cavour.market.position.engine import Engine
from cavour.marketdata.market_data_engine import MarketCurveBuilder
from cavour.marketdata.market_data_constants import MARKET_DATA, FX_MARKET_DATA
from cavour.utils.greek_validation import GreekValidator


##############################################################################
# CONFIGURATION
##############################################################################

# Market data date
MARKET_DATA_DATE = Date(22, 1, 2026)

# Tolerances for 200bp bumps (looser than 1bp validation)
TOLERANCE_DELTA_ONLY_200BP = 0.20  # 20% - expected to fail, shows non-linearity
TOLERANCE_DELTA_GAMMA_200BP = 0.07  # 7% - should capture curvature
TOLERANCE_PARALLEL_200BP = 0.10  # 10% - parallel shock Taylor expansion
TOLERANCE_SCENARIO_200BP = 0.12  # 12% - complex curve movements
TOLERANCE_CROSS_GAMMA_1BP = 0.02  # 2% - cross-gamma uses 1bp for stability

# Notionals
OIS_NOTIONAL = 100_000_000
XCCY_NOTIONAL_USD = 100_000_000


##############################################################################
# HELPER FUNCTIONS
##############################################################################

def build_gbp_ois_model_from_bloomberg(value_dt: Date) -> Model:
    """
    Build GBP OIS SONIA model from Bloomberg data.

    Args:
        value_dt: Valuation date

    Returns:
        Model with GBP_OIS_SONIA curve (AD and gamma enabled)
    """
    builder = MarketCurveBuilder(MARKET_DATA, FX_MARKET_DATA)
    gbp_params = builder.get_curve_inputs('GBP_OIS_SONIA', value_dt)

    model = Model(value_dt)
    model.build_curve(
        **gbp_params,
        use_ad=True,
        compute_gamma=True
    )

    return model, gbp_params


def build_xccy_model_from_bloomberg(value_dt: Date) -> tuple:
    """
    Build multi-curve model for XCCY swap from Bloomberg data.

    Fetches:
    - USD OIS SOFR curve
    - GBP OIS SONIA curve
    - GBP/USD XCCY basis spreads
    - GBP/USD spot FX rate

    Args:
        value_dt: Valuation date

    Returns:
        tuple: (model, usd_params, gbp_params, xccy_params, spot_fx)
    """
    builder = MarketCurveBuilder(MARKET_DATA, FX_MARKET_DATA)

    # Fetch all market data
    usd_params = builder.get_curve_inputs('USD_OIS_SOFR', value_dt)
    gbp_params = builder.get_curve_inputs('GBP_OIS_SONIA', value_dt)
    xccy_params = builder.get_curve_inputs('GBPUSD_XCCY_SONIA_SOFR', value_dt)
    fx_rates = builder.get_fx_rates(['GBPUSD'], value_dt)
    spot_fx = fx_rates['GBPUSD']['price']

    # Build model
    model = Model(value_dt)

    # Build OIS curves with AD and gamma
    model.build_curve(**usd_params, use_ad=True, compute_gamma=True)
    model.build_curve(**gbp_params, use_ad=True, compute_gamma=True)

    # Build FX
    model.build_fx(['GBPUSD'], [spot_fx])

    # Build XCCY curve with AD and gamma
    # NOTE: Curve name is 'GBP_USD_BASIS' (foreign_domestic format)
    model.build_xccy_curve(
        name='GBP_USD_BASIS',
        domestic_curve_name='USD_OIS_SOFR',
        foreign_curve_name='GBP_OIS_SONIA',
        basis_spreads=xccy_params['px_list'],
        tenor_list=xccy_params['tenor_list'],
        spot_fx=spot_fx,
        domestic_freq_type=FrequencyTypes.QUARTERLY,
        foreign_freq_type=FrequencyTypes.QUARTERLY,
        domestic_dc_type=DayCountTypes.ACT_360,
        foreign_dc_type=DayCountTypes.ACT_365F,
        bus_day_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        interp_type=xccy_params.get('interp_type'),
        use_ad=True,
        compute_gamma=True
    )

    return model, usd_params, gbp_params, xccy_params, spot_fx


def create_ois_swap_at_market(value_dt: Date, curve_params: dict, tenor: str = "10Y") -> OIS:
    """
    Create at-market OIS swap using curve par rates.

    Args:
        value_dt: Valuation date
        curve_params: Curve parameters from Bloomberg
        tenor: Swap tenor (default 10Y)

    Returns:
        OIS swap instance
    """
    # Extract par rate for the tenor
    if tenor in curve_params['tenor_list']:
        idx = curve_params['tenor_list'].index(tenor)
        par_rate_pct = curve_params['px_list'][idx]
        par_rate = par_rate_pct / 100.0
    else:
        # If exact tenor not in curve, use interpolated rate or fallback
        par_rate = 0.045  # Fallback rate

    swap = OIS(
        effective_dt=value_dt,
        term_dt_or_tenor=tenor,
        fixed_leg_type=SwapTypes.PAY,
        notional=OIS_NOTIONAL,
        fixed_coupon=par_rate,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        fixed_dc_type=DayCountTypes.ACT_365F,
        floating_index=CurveTypes.GBP_OIS_SONIA,
        currency=CurrencyTypes.GBP,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_365F
    )

    return swap


def create_xccy_swap_at_market(value_dt: Date, xccy_params: dict, spot_fx: float, tenor: str = "10Y") -> XccyBasisSwap:
    """
    Create at-market XCCY basis swap using market basis spreads.

    Args:
        value_dt: Valuation date
        xccy_params: XCCY curve parameters from Bloomberg
        spot_fx: GBP/USD spot FX rate
        tenor: Swap tenor (default 10Y)

    Returns:
        XccyBasisSwap instance
    """
    # Extract basis spread for the tenor
    if tenor in xccy_params['tenor_list']:
        idx = xccy_params['tenor_list'].index(tenor)
        basis_bps = xccy_params['px_list'][idx]
        foreign_spread = basis_bps / 10000.0  # Convert bps to decimal
    else:
        foreign_spread = -0.0025  # Fallback -25bps

    # Calculate foreign notional
    gbp_notional = XCCY_NOTIONAL_USD / spot_fx

    swap = XccyBasisSwap(
        effective_dt=value_dt,
        term_dt_or_tenor=tenor,
        domestic_notional=XCCY_NOTIONAL_USD,
        foreign_notional=gbp_notional,
        domestic_spread=0.0,
        foreign_spread=foreign_spread,
        domestic_freq_type=FrequencyTypes.QUARTERLY,
        foreign_freq_type=FrequencyTypes.QUARTERLY,
        domestic_dc_type=DayCountTypes.ACT_360,
        foreign_dc_type=DayCountTypes.ACT_365F,
        domestic_floating_index=CurveTypes.USD_OIS_SOFR,
        foreign_floating_index=CurveTypes.GBP_OIS_SONIA,
        domestic_currency=CurrencyTypes.USD,
        foreign_currency=CurrencyTypes.GBP
    )

    return swap


def print_section_header(title: str):
    """Print formatted section header."""
    print(f"\n{'='*80}")
    print(f"{title.center(80)}")
    print(f"{'='*80}\n")


def print_test_header(test_name: str):
    """Print formatted test header."""
    print(f"\n{'-'*80}")
    print(f"{test_name}")
    print(f"{'-'*80}")


##############################################################################
# TEST CLASS 1: OIS SWAP 200BP VALIDATION
##############################################################################

@pytest.mark.market_data
class TestOIS200bpValidation:
    """
    Validate Greeks for GBP OIS swap under 200bp bump scenarios.

    Tests:
    1. Single point bump (200bp to largest delta tenor)
    2. Parallel bump (200bp to all tenors)
    3. Slope scenario (steepening/flattening)
    4. Skew scenario (belly up/down)
    5. Butterfly scenario (wings up, belly down)
    """

    @pytest.fixture(scope="class")
    def ois_setup(self):
        """Setup OIS model and swap from Bloomberg data."""
        print_section_header("OIS SWAP 200BP VALIDATION - SETUP")

        value_dt = MARKET_DATA_DATE
        model, gbp_params = build_gbp_ois_model_from_bloomberg(value_dt)
        swap = create_ois_swap_at_market(value_dt, gbp_params, tenor="10Y")

        # Compute Greeks
        position = swap.position(model)
        result = position.compute([
            RequestTypes.VALUE,
            RequestTypes.DELTA,
            RequestTypes.GAMMA
        ])

        print(f"Setup complete:")
        print(f"  Model date: {value_dt}")
        print(f"  Swap: 10Y GBP OIS")
        print(f"  Notional: {OIS_NOTIONAL:,.0f} GBP")
        print(f"  Base PV: {result.value.amount:,.2f} GBP")
        print(f"  Curve tenors: {gbp_params['tenor_list']}")

        validator = GreekValidator(model, swap)

        return {
            'model': model,
            'swap': swap,
            'result': result,
            'validator': validator,
            'gbp_params': gbp_params
        }

    def test_ois_single_point_bump_200bp(self, ois_setup):
        """Test 200bp single point bump - demonstrates non-linearity."""
        print_test_header("TEST: OIS Single Point Bump (200bp)")

        validator = ois_setup['validator']
        result = ois_setup['result']

        report = validator.validate_delta_large_bump(
            ad_result=result,
            curve_type=CurveTypes.GBP_OIS_SONIA,
            shock_bp=200.0,
            tolerance_delta=TOLERANCE_DELTA_ONLY_200BP,
            tolerance_delta_gamma=TOLERANCE_DELTA_GAMMA_200BP
        )

        print(report)

        # Assertions
        print(f"\nAssertion checks:")
        print(f"  Delta-only error: {report.error_delta_pct:.2%}")
        print(f"  Delta+Gamma error: {report.error_delta_gamma_pct:.2%}")
        print(f"  Gamma improvement: {report.gamma_improvement_factor:.2f}x")

        # Delta-only should pass tolerance
        assert report.passed_delta, \
            f"Delta-only should pass tolerance at 200bp. Error: {report.error_delta_pct:.2%}"

        # Delta+Gamma should be much better
        assert report.passed_delta_gamma, \
            f"Delta+Gamma should pass at 200bp. Error: {report.error_delta_gamma_pct:.2%}"

        # Gamma should provide significant improvement (>2x)
        assert report.gamma_improvement_factor > 2.0, \
            f"Gamma should improve accuracy by >2x. Got {report.gamma_improvement_factor:.2f}x"

        print(f"\n[PASS] Test PASSED: Gamma reduces error by {report.gamma_improvement_factor:.2f}x at 200bp")
        print(f"       (Delta-only: {report.error_delta_pct:.4%}, Delta+Gamma: {report.error_delta_gamma_pct:.4%})")

    def test_ois_parallel_bump_200bp(self, ois_setup):
        """Test 200bp parallel bump - Taylor expansion validation."""
        print_test_header("TEST: OIS Parallel Bump (200bp)")

        validator = ois_setup['validator']
        result = ois_setup['result']

        report = validator.validate_gamma_taylor_expansion(
            ad_result=result,
            curve_type=CurveTypes.GBP_OIS_SONIA,
            shock_bp=200.0,
            tolerance=TOLERANCE_PARALLEL_200BP,
            parallel_shock=True
        )

        print(report)

        # Assertions
        print(f"\nAssertion checks:")
        print(f"  1st-order error: {report.error_1st_order_pct:.2%}")
        print(f"  2nd-order error: {report.error_2nd_order_pct:.2%}")
        print(f"  Improvement: {report.gamma_improvement_factor:.2f}x")

        # 2nd-order Taylor should pass tolerance
        assert report.passed, \
            f"2nd-order Taylor should be within {TOLERANCE_PARALLEL_200BP:.0%} at 200bp. " \
            f"Error: {report.error_2nd_order_pct:.2%}"

        # Should be significantly better than 1st-order
        assert report.gamma_improvement_factor > 3.0, \
            f"Gamma should improve accuracy by >3x for parallel shock. Got {report.gamma_improvement_factor:.2f}x"

        # Gamma matrix should be symmetric
        assert report.gamma_matrix_symmetric, \
            f"Gamma matrix should be symmetric. Max asymmetry: {report.max_symmetry_error:.2e}"

        print(f"\n[PASS] Test PASSED: 2nd-order Taylor error {report.error_2nd_order_pct:.2%} < {TOLERANCE_PARALLEL_200BP:.0%}")

    def test_ois_slope_scenario_200bp(self, ois_setup):
        """Test 200bp slope scenario with delta+gamma - steepening and flattening."""
        print_test_header("TEST: OIS Slope Scenario (200bp) - Delta+Gamma")

        validator = ois_setup['validator']
        result = ois_setup['result']

        scenarios = [
            {'type': 'slope', 'shock_bp': 200},   # Steepening
            {'type': 'slope', 'shock_bp': -200},  # Flattening
        ]

        report = validator.validate_curve_scenarios(
            ad_result=result,
            curve_type=CurveTypes.GBP_OIS_SONIA,
            scenarios=scenarios,
            tolerance=0.02,  # TIGHTER: 2% for delta+gamma (vs 12% for delta-only)
            use_gamma=True   # Enable gamma correction
        )

        print(report)

        # Assertions for delta+gamma
        print(f"\nAssertion checks (Delta+Gamma):")
        for scenario in report.scenarios:
            if scenario.pv_delta_gamma_approx is not None:
                print(f"  {scenario.name}:")
                print(f"    Delta-only:  {scenario.error_pct:.2%}")
                print(f"    Delta+Gamma: {scenario.error_delta_gamma_pct:.2%}")
                print(f"    Improvement: {scenario.gamma_improvement_factor:.1f}x")
            else:
                print(f"  {scenario.name}: {scenario.error_pct:.2%} (gamma not available)")

        # All scenarios should pass with gamma
        all_passed_gamma = all(s.passed_delta_gamma for s in report.scenarios if s.passed_delta_gamma is not None)
        assert all_passed_gamma, \
            f"All scenarios should pass with delta+gamma. Failed: {[s.name for s in report.scenarios if not s.passed_delta_gamma]}"

        # Gamma should provide significant improvement
        for scenario in report.scenarios:
            if scenario.gamma_improvement_factor is not None:
                assert scenario.gamma_improvement_factor > 3.0, \
                    f"{scenario.name}: Gamma should improve by >3x. Got {scenario.gamma_improvement_factor:.1f}x"

        # Delta+gamma errors should be much lower than delta-only
        max_error_gamma = max(s.error_delta_gamma_pct for s in report.scenarios if s.error_delta_gamma_pct is not None)
        assert max_error_gamma < 0.02, \
            f"Max delta+gamma error should be <2%. Got {max_error_gamma:.2%}"

        print(f"\n[PASS] Test PASSED: All slope scenarios <2% error with delta+gamma")

    def test_ois_skew_scenario_200bp(self, ois_setup):
        """Test 200bp skew scenario with delta+gamma - belly up."""
        print_test_header("TEST: OIS Skew Scenario (200bp) - Delta+Gamma")

        validator = ois_setup['validator']
        result = ois_setup['result']

        scenarios = [
            {'type': 'skew', 'shock_bp': 200},
        ]

        report = validator.validate_curve_scenarios(
            ad_result=result,
            curve_type=CurveTypes.GBP_OIS_SONIA,
            scenarios=scenarios,
            tolerance=0.02,  # 2% for delta+gamma
            use_gamma=True
        )

        print(report)

        # Assertions
        scenario = report.scenarios[0]
        if scenario.pv_delta_gamma_approx is not None:
            print(f"\nDelta-only error:  {scenario.error_pct:.2%} (was worst case)")
            print(f"Delta+Gamma error: {scenario.error_delta_gamma_pct:.2%}")
            print(f"Improvement: {scenario.gamma_improvement_factor:.1f}x")

            assert scenario.passed_delta_gamma, \
                f"Skew scenario should pass with delta+gamma. Error: {scenario.error_delta_gamma_pct:.2%}"

            # Expect large improvement for skew (concentrated shock in high-gamma region)
            assert scenario.gamma_improvement_factor > 5.0, \
                f"Skew should show >5x improvement (concentrated gamma). Got {scenario.gamma_improvement_factor:.1f}x"

            assert scenario.error_delta_gamma_pct < 0.02, \
                f"Delta+gamma error should be <2%. Got {scenario.error_delta_gamma_pct:.2%}"

            print(f"\n[PASS] Test PASSED: Skew scenario {scenario.error_pct:.2%} to {scenario.error_delta_gamma_pct:.2%} ({scenario.gamma_improvement_factor:.1f}x improvement)")
        else:
            assert scenario.passed, f"Skew scenario should pass. Error: {scenario.error_pct:.2%}"
            print(f"\n[PASS] Test PASSED: Skew scenario error {scenario.error_pct:.2%}")

    def test_ois_butterfly_scenario_200bp(self, ois_setup):
        """Test 200bp butterfly scenario with delta+gamma - wings up, belly down."""
        print_test_header("TEST: OIS Butterfly Scenario (200bp) - Delta+Gamma")

        validator = ois_setup['validator']
        result = ois_setup['result']

        scenarios = [
            {'type': 'butterfly', 'shock_bp': 200},
        ]

        report = validator.validate_curve_scenarios(
            ad_result=result,
            curve_type=CurveTypes.GBP_OIS_SONIA,
            scenarios=scenarios,
            tolerance=0.02,  # 2% for delta+gamma
            use_gamma=True
        )

        print(report)

        # Assertions
        scenario = report.scenarios[0]
        if scenario.pv_delta_gamma_approx is not None:
            print(f"\nDelta-only error:  {scenario.error_pct:.2%}")
            print(f"Delta+Gamma error: {scenario.error_delta_gamma_pct:.2%}")
            print(f"Improvement: {scenario.gamma_improvement_factor:.1f}x")

            assert scenario.passed_delta_gamma, \
                f"Butterfly scenario should pass with delta+gamma. Error: {scenario.error_delta_gamma_pct:.2%}"

            assert scenario.gamma_improvement_factor > 3.0, \
                f"Butterfly should show >3x improvement. Got {scenario.gamma_improvement_factor:.1f}x"

            assert scenario.error_delta_gamma_pct < 0.02, \
                f"Delta+gamma error should be <2%. Got {scenario.error_delta_gamma_pct:.2%}"

            print(f"\n[PASS] Test PASSED: Butterfly scenario {scenario.error_pct:.2%} to {scenario.error_delta_gamma_pct:.2%} ({scenario.gamma_improvement_factor:.1f}x improvement)")
        else:
            assert scenario.passed, f"Butterfly scenario should pass. Error: {scenario.error_pct:.2%}"
            print(f"\n[PASS] Test PASSED: Butterfly scenario error {scenario.error_pct:.2%}")


##############################################################################
# TEST CLASS 2: XCCY SWAP 200BP VALIDATION
##############################################################################

@pytest.mark.market_data
class TestXCCY200bpValidation:
    """
    Validate Greeks for XCCY basis swap under 200bp bump scenarios.

    Tests each curve independently:
    - USD OIS SOFR
    - GBP OIS SONIA
    - GBP_USD_BASIS

    For each curve, tests:
    1. Single point bump (200bp)
    2. Parallel bump (200bp)
    3. Slope scenario (200bp)
    4. Skew scenario (200bp)
    """

    @pytest.fixture(scope="class")
    def xccy_setup(self):
        """Setup XCCY model and swap from Bloomberg data."""
        print_section_header("XCCY SWAP 200BP VALIDATION - SETUP")

        value_dt = MARKET_DATA_DATE
        model, usd_params, gbp_params, xccy_params, spot_fx = build_xccy_model_from_bloomberg(value_dt)
        swap = create_xccy_swap_at_market(value_dt, xccy_params, spot_fx, tenor="10Y")

        # Compute Greeks using Engine (required for XCCY multi-curve)
        engine = Engine(model)
        result = engine.compute(swap, [
            RequestTypes.VALUE,
            RequestTypes.DELTA,
            RequestTypes.GAMMA
        ])

        print(f"Setup complete:")
        print(f"  Model date: {value_dt}")
        print(f"  Swap: 10Y GBP/USD Basis Swap")
        print(f"  Notional USD: {XCCY_NOTIONAL_USD:,.0f}")
        print(f"  Notional GBP: {XCCY_NOTIONAL_USD/spot_fx:,.0f}")
        print(f"  Spot FX: {spot_fx:.4f}")
        print(f"  Base PV: {result.value.amount:,.2f} USD")
        print(f"  Curves: USD OIS, GBP OIS, GBP_USD_BASIS")

        validator = GreekValidator(model, swap)

        return {
            'model': model,
            'swap': swap,
            'result': result,
            'validator': validator,
            'engine': engine,
            'usd_params': usd_params,
            'gbp_params': gbp_params,
            'xccy_params': xccy_params,
            'spot_fx': spot_fx
        }

    # USD OIS SOFR Tests

    def test_xccy_usd_ois_single_point_200bp(self, xccy_setup):
        """Test 200bp single point bump on USD OIS curve."""
        print_test_header("TEST: XCCY - USD OIS Single Point Bump (200bp)")

        validator = xccy_setup['validator']
        result = xccy_setup['result']

        report = validator.validate_delta_large_bump(
            ad_result=result,
            curve_type=CurveTypes.USD_OIS_SOFR,
            shock_bp=200.0,
            tolerance_delta=TOLERANCE_DELTA_ONLY_200BP,
            tolerance_delta_gamma=TOLERANCE_DELTA_GAMMA_200BP
        )

        print(report)

        # Assertions
        assert report.passed_delta, \
            f"Delta-only should pass on USD OIS. Error: {report.error_delta_pct:.2%}"
        assert report.passed_delta_gamma, \
            f"Delta+Gamma should pass on USD OIS at 200bp. Error: {report.error_delta_gamma_pct:.2%}"
        assert report.gamma_improvement_factor > 1.5, \
            f"Gamma should improve accuracy. Got {report.gamma_improvement_factor:.2f}x"

        print(f"\n[PASS] Test PASSED: USD OIS - Gamma improves by {report.gamma_improvement_factor:.2f}x")

    def test_xccy_usd_ois_parallel_200bp(self, xccy_setup):
        """Test 200bp parallel bump on USD OIS curve."""
        print_test_header("TEST: XCCY - USD OIS Parallel Bump (200bp)")

        validator = xccy_setup['validator']
        result = xccy_setup['result']

        report = validator.validate_gamma_taylor_expansion(
            ad_result=result,
            curve_type=CurveTypes.USD_OIS_SOFR,
            shock_bp=200.0,
            tolerance=TOLERANCE_PARALLEL_200BP
        )

        print(report)

        assert report.passed, \
            f"USD OIS parallel 200bp should pass. Error: {report.error_2nd_order_pct:.2%}"

        print(f"\n[PASS] Test PASSED: USD OIS parallel - Error {report.error_2nd_order_pct:.2%}")

    def test_xccy_usd_ois_slope_200bp(self, xccy_setup):
        """Test 200bp slope scenario on USD OIS curve."""
        print_test_header("TEST: XCCY - USD OIS Slope Scenario (200bp)")

        validator = xccy_setup['validator']
        result = xccy_setup['result']

        scenarios = [{'type': 'slope', 'shock_bp': 200}]

        report = validator.validate_curve_scenarios(
            ad_result=result,
            curve_type=CurveTypes.USD_OIS_SOFR,
            scenarios=scenarios,
            tolerance=TOLERANCE_SCENARIO_200BP
        )

        print(report)

        assert report.all_passed, \
            f"USD OIS slope should pass. Error: {report.scenarios[0].error_pct:.2%}"

        print(f"\n[PASS] Test PASSED: USD OIS slope - Error {report.scenarios[0].error_pct:.2%}")

    def test_xccy_usd_ois_skew_200bp(self, xccy_setup):
        """Test 200bp skew scenario on USD OIS curve."""
        print_test_header("TEST: XCCY - USD OIS Skew Scenario (200bp)")

        validator = xccy_setup['validator']
        result = xccy_setup['result']

        scenarios = [{'type': 'skew', 'shock_bp': 200}]

        report = validator.validate_curve_scenarios(
            ad_result=result,
            curve_type=CurveTypes.USD_OIS_SOFR,
            scenarios=scenarios,
            tolerance=TOLERANCE_SCENARIO_200BP
        )

        print(report)

        assert report.all_passed, \
            f"USD OIS skew should pass. Error: {report.scenarios[0].error_pct:.2%}"

        print(f"\n[PASS] Test PASSED: USD OIS skew - Error {report.scenarios[0].error_pct:.2%}")

    # GBP OIS SONIA Tests

    def test_xccy_gbp_ois_single_point_200bp(self, xccy_setup):
        """Test 200bp single point bump on GBP OIS curve."""
        print_test_header("TEST: XCCY - GBP OIS Single Point Bump (200bp)")

        validator = xccy_setup['validator']
        result = xccy_setup['result']

        report = validator.validate_delta_large_bump(
            ad_result=result,
            curve_type=CurveTypes.GBP_OIS_SONIA,
            shock_bp=200.0,
            tolerance_delta=TOLERANCE_DELTA_ONLY_200BP,
            tolerance_delta_gamma=TOLERANCE_DELTA_GAMMA_200BP
        )

        print(report)

        # Assertions
        assert report.passed_delta, \
            f"Delta-only should pass on GBP OIS. Error: {report.error_delta_pct:.2%}"
        assert report.passed_delta_gamma, \
            f"Delta+Gamma should pass on GBP OIS at 200bp. Error: {report.error_delta_gamma_pct:.2%}"
        assert report.gamma_improvement_factor > 1.5, \
            f"Gamma should improve accuracy. Got {report.gamma_improvement_factor:.2f}x"

        print(f"\n[PASS] Test PASSED: GBP OIS - Gamma improves by {report.gamma_improvement_factor:.2f}x")

    def test_xccy_gbp_ois_parallel_200bp(self, xccy_setup):
        """Test 200bp parallel bump on GBP OIS curve."""
        print_test_header("TEST: XCCY - GBP OIS Parallel Bump (200bp)")

        validator = xccy_setup['validator']
        result = xccy_setup['result']

        report = validator.validate_gamma_taylor_expansion(
            ad_result=result,
            curve_type=CurveTypes.GBP_OIS_SONIA,
            shock_bp=200.0,
            tolerance=TOLERANCE_PARALLEL_200BP
        )

        print(report)

        assert report.passed, \
            f"GBP OIS parallel 200bp should pass. Error: {report.error_2nd_order_pct:.2%}"

        print(f"\n[PASS] Test PASSED: GBP OIS parallel - Error {report.error_2nd_order_pct:.2%}")

    def test_xccy_gbp_ois_slope_200bp(self, xccy_setup):
        """Test 200bp slope scenario on GBP OIS curve."""
        print_test_header("TEST: XCCY - GBP OIS Slope Scenario (200bp)")

        validator = xccy_setup['validator']
        result = xccy_setup['result']

        scenarios = [{'type': 'slope', 'shock_bp': 200}]

        report = validator.validate_curve_scenarios(
            ad_result=result,
            curve_type=CurveTypes.GBP_OIS_SONIA,
            scenarios=scenarios,
            tolerance=TOLERANCE_SCENARIO_200BP
        )

        print(report)

        assert report.all_passed, \
            f"GBP OIS slope should pass. Error: {report.scenarios[0].error_pct:.2%}"

        print(f"\n[PASS] Test PASSED: GBP OIS slope - Error {report.scenarios[0].error_pct:.2%}")

    def test_xccy_gbp_ois_skew_200bp(self, xccy_setup):
        """Test 200bp skew scenario on GBP OIS curve."""
        print_test_header("TEST: XCCY - GBP OIS Skew Scenario (200bp)")

        validator = xccy_setup['validator']
        result = xccy_setup['result']

        scenarios = [{'type': 'skew', 'shock_bp': 200}]

        report = validator.validate_curve_scenarios(
            ad_result=result,
            curve_type=CurveTypes.GBP_OIS_SONIA,
            scenarios=scenarios,
            tolerance=TOLERANCE_SCENARIO_200BP
        )

        print(report)

        assert report.all_passed, \
            f"GBP OIS skew should pass. Error: {report.scenarios[0].error_pct:.2%}"

        print(f"\n[PASS] Test PASSED: GBP OIS skew - Error {report.scenarios[0].error_pct:.2%}")

    # BASIS Curve Tests

    def test_xccy_basis_single_point_200bp(self, xccy_setup):
        """Test 200bp single point bump on BASIS curve."""
        print_test_header("TEST: XCCY - BASIS Single Point Bump (200bp)")

        validator = xccy_setup['validator']
        result = xccy_setup['result']

        # NOTE: Use CurveTypes.USD_GBP_BASIS (enum) for validation
        report = validator.validate_delta_large_bump(
            ad_result=result,
            curve_type=CurveTypes.USD_GBP_BASIS,
            shock_bp=200.0,
            tolerance_delta=TOLERANCE_DELTA_ONLY_200BP,
            tolerance_delta_gamma=TOLERANCE_DELTA_GAMMA_200BP
        )

        print(report)

        # Assertions
        assert report.passed_delta, \
            f"Delta-only should pass on BASIS. Error: {report.error_delta_pct:.2%}"
        assert report.passed_delta_gamma, \
            f"Delta+Gamma should pass on BASIS at 200bp. Error: {report.error_delta_gamma_pct:.2%}"
        assert report.gamma_improvement_factor > 1.5, \
            f"Gamma should improve accuracy. Got {report.gamma_improvement_factor:.2f}x"

        print(f"\n[PASS] Test PASSED: BASIS - Gamma improves by {report.gamma_improvement_factor:.2f}x")

    def test_xccy_basis_parallel_200bp(self, xccy_setup):
        """Test 200bp parallel bump on BASIS curve."""
        print_test_header("TEST: XCCY - BASIS Parallel Bump (200bp)")

        validator = xccy_setup['validator']
        result = xccy_setup['result']

        report = validator.validate_gamma_taylor_expansion(
            ad_result=result,
            curve_type=CurveTypes.USD_GBP_BASIS,
            shock_bp=200.0,
            tolerance=TOLERANCE_PARALLEL_200BP
        )

        print(report)

        assert report.passed, \
            f"BASIS parallel 200bp should pass. Error: {report.error_2nd_order_pct:.2%}"

        print(f"\n[PASS] Test PASSED: BASIS parallel - Error {report.error_2nd_order_pct:.2%}")

    def test_xccy_basis_slope_200bp(self, xccy_setup):
        """Test 200bp slope scenario on BASIS curve."""
        print_test_header("TEST: XCCY - BASIS Slope Scenario (200bp)")

        validator = xccy_setup['validator']
        result = xccy_setup['result']

        scenarios = [{'type': 'slope', 'shock_bp': 200}]

        report = validator.validate_curve_scenarios(
            ad_result=result,
            curve_type=CurveTypes.USD_GBP_BASIS,
            scenarios=scenarios,
            tolerance=TOLERANCE_SCENARIO_200BP
        )

        print(report)

        assert report.all_passed, \
            f"BASIS slope should pass. Error: {report.scenarios[0].error_pct:.2%}"

        print(f"\n[PASS] Test PASSED: BASIS slope - Error {report.scenarios[0].error_pct:.2%}")

    def test_xccy_basis_skew_200bp(self, xccy_setup):
        """Test 200bp skew scenario on BASIS curve."""
        print_test_header("TEST: XCCY - BASIS Skew Scenario (200bp)")

        validator = xccy_setup['validator']
        result = xccy_setup['result']

        scenarios = [{'type': 'skew', 'shock_bp': 200}]

        report = validator.validate_curve_scenarios(
            ad_result=result,
            curve_type=CurveTypes.USD_GBP_BASIS,
            scenarios=scenarios,
            tolerance=TOLERANCE_SCENARIO_200BP
        )

        print(report)

        assert report.all_passed, \
            f"BASIS skew should pass. Error: {report.scenarios[0].error_pct:.2%}"

        print(f"\n[PASS] Test PASSED: BASIS skew - Error {report.scenarios[0].error_pct:.2%}")


##############################################################################
# TEST CLASS 3: CROSS-GAMMA VALIDATION
##############################################################################

@pytest.mark.market_data
class TestXCCYCrossGamma:
    """
    Validate cross-gamma sensitivities between curve pairs.

    NOTE: Cross-gamma validation uses 1bp bumps (not 200bp) for numerical
    stability, as double differencing amplifies errors.

    Tests:
    1. USD OIS vs GBP OIS cross-gamma
    2. GBP OIS vs BASIS cross-gamma
    3. USD OIS vs BASIS cross-gamma
    """

    @pytest.fixture(scope="class")
    def xccy_setup(self):
        """Setup XCCY model and swap from Bloomberg data."""
        print_section_header("CROSS-GAMMA VALIDATION - SETUP")

        value_dt = MARKET_DATA_DATE
        model, usd_params, gbp_params, xccy_params, spot_fx = build_xccy_model_from_bloomberg(value_dt)
        swap = create_xccy_swap_at_market(value_dt, xccy_params, spot_fx, tenor="10Y")

        # Compute Greeks using Engine
        engine = Engine(model)
        result = engine.compute(swap, [
            RequestTypes.VALUE,
            RequestTypes.DELTA,
            RequestTypes.GAMMA
        ])

        print(f"Setup complete for cross-gamma validation")
        print(f"  Using 1bp bumps for numerical stability")
        print(f"  Tolerance: {TOLERANCE_CROSS_GAMMA_1BP:.0%}")

        validator = GreekValidator(model, swap)

        return {
            'model': model,
            'swap': swap,
            'result': result,
            'validator': validator,
            'engine': engine
        }

    def test_cross_gamma_usd_gbp_ois(self, xccy_setup):
        """Test cross-gamma between USD OIS and GBP OIS curves."""
        print_test_header("TEST: Cross-Gamma - USD OIS vs GBP OIS")

        validator = xccy_setup['validator']
        result = xccy_setup['result']

        report = validator.validate_cross_gamma(
            ad_result=result,
            curve_type_1=CurveTypes.USD_OIS_SOFR,
            curve_type_2=CurveTypes.GBP_OIS_SONIA,
            bump_bp=1.0,  # Use 1bp for cross-gamma (not 200bp)
            tolerance=TOLERANCE_CROSS_GAMMA_1BP
        )

        print(report)

        # Assertions
        print(f"\nAssertion checks:")
        print(f"  Max relative error: {report.max_relative_error:.4%}")
        print(f"  Mean relative error: {report.mean_relative_error:.4%}")

        assert report.passed, \
            f"USD/GBP OIS cross-gamma should pass. Max error: {report.max_relative_error:.4%}"

        print(f"\n[PASS] Test PASSED: USD/GBP OIS cross-gamma error {report.max_relative_error:.4%} < {TOLERANCE_CROSS_GAMMA_1BP:.0%}")

    def test_cross_gamma_gbp_ois_basis(self, xccy_setup):
        """Test cross-gamma between GBP OIS and BASIS curves."""
        print_test_header("TEST: Cross-Gamma - GBP OIS vs BASIS")

        validator = xccy_setup['validator']
        result = xccy_setup['result']

        report = validator.validate_cross_gamma(
            ad_result=result,
            curve_type_1=CurveTypes.GBP_OIS_SONIA,
            curve_type_2=CurveTypes.USD_GBP_BASIS,
            bump_bp=1.0,
            tolerance=TOLERANCE_CROSS_GAMMA_1BP
        )

        print(report)

        assert report.passed, \
            f"GBP OIS/BASIS cross-gamma should pass. Max error: {report.max_relative_error:.4%}"

        print(f"\n[PASS] Test PASSED: GBP OIS/BASIS cross-gamma error {report.max_relative_error:.4%} < {TOLERANCE_CROSS_GAMMA_1BP:.0%}")

    def test_cross_gamma_usd_ois_basis(self, xccy_setup):
        """Test cross-gamma between USD OIS and BASIS curves."""
        print_test_header("TEST: Cross-Gamma - USD OIS vs BASIS")

        validator = xccy_setup['validator']
        result = xccy_setup['result']

        report = validator.validate_cross_gamma(
            ad_result=result,
            curve_type_1=CurveTypes.USD_OIS_SOFR,
            curve_type_2=CurveTypes.USD_GBP_BASIS,
            bump_bp=1.0,
            tolerance=TOLERANCE_CROSS_GAMMA_1BP
        )

        print(report)

        assert report.passed, \
            f"USD OIS/BASIS cross-gamma should pass. Max error: {report.max_relative_error:.4%}"

        print(f"\n[PASS] Test PASSED: USD OIS/BASIS cross-gamma error {report.max_relative_error:.4%} < {TOLERANCE_CROSS_GAMMA_1BP:.0%}")


##############################################################################
# MAIN EXECUTION
##############################################################################

if __name__ == "__main__":
    """
    Run tests directly (outside pytest framework).

    Usage:
        python test_greek_validation_200bp_bbg.py

    Or with pytest:
        pytest test_greek_validation_200bp_bbg.py -v -m market_data
    """
    print("""
    ================================================================================
    GREEK VALIDATION TEST SUITE: 200BP BUMP SCENARIOS
    ================================================================================

    This test suite validates AD-based Greeks (Delta, Gamma, Cross-Gamma) against
    full revaluation under 200bp bump scenarios using Bloomberg market data.

    Test Coverage:
    - OIS Swap (GBP SONIA): 5 tests
    - XCCY Basis Swap (GBP/USD): 12 tests (4 scenarios × 3 curves)
    - Cross-Gamma: 3 tests

    Total: 20 tests

    Expected Tolerances:
    - Delta-only (200bp): 15-20% (demonstrates non-linearity)
    - Delta+Gamma (200bp): 5-7% (captures curvature)
    - Parallel shock (200bp): 7-10%
    - Scenarios (200bp): 10-12%
    - Cross-gamma (1bp): 1-2%

    Bloomberg Connection Required: Yes
    Market Data Date: 22-Jan-2026
    ================================================================================
    """)

    # Run via pytest
    pytest.main([__file__, "-v", "-m", "market_data", "--tb=short"])
