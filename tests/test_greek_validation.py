"""
Comprehensive tests for Greek validation framework.

Tests the validation of AD-based deltas and gammas against:
- Finite difference approximations (for delta)
- Taylor series expansion (for gamma)
"""

import pytest
import numpy as np
from cavour.utils.date import Date
from cavour.utils.global_types import SwapTypes, RequestTypes, CurveTypes
from cavour.utils.day_count import DayCountTypes
from cavour.utils.frequency import FrequencyTypes
from cavour.utils.calendar import BusDayAdjustTypes
from cavour.utils.currency import CurrencyTypes
from cavour.market.curves.interpolator import InterpTypes
from cavour.models.models import Model
from cavour.trades.rates.ois import OIS
from cavour.trades.rates.xccy_fix_float_swap import XccyFixFloat
from cavour.market.position.engine import Engine
from cavour.utils.greek_validation import GreekValidator


class TestDeltaValidation:
    """Test delta validation against finite difference."""

    def test_ois_delta_validation_5y_simple(self):
        """
        Validate delta for simple 5Y OIS swap against FD.

        This test:
        1. Builds a GBP OIS curve
        2. Creates a 5Y OIS swap
        3. Computes AD deltas
        4. Validates deltas vs central finite difference
        """
        # Setup
        value_dt = Date(30, 4, 2024)

        # Build GBP OIS curve (5 tenors)
        model = Model(value_dt=value_dt)
        model.build_curve(
            name="GBP_OIS_SONIA",
            px_list=[5.19, 5.13, 5.04, 4.75, 4.24],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            fixed_dcc_type=DayCountTypes.ACT_365F,
            use_ad=True,  # Enable AD for delta computation
            compute_gamma=False  # Don't need gamma for this test
        )

        # Create 5Y OIS swap
        swap = OIS(
            effective_dt=value_dt,
            term_dt_or_tenor="5Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=0.045,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_365F,
            floating_index=CurveTypes.GBP_OIS_SONIA,
            currency=CurrencyTypes.GBP
        )

        # Compute AD deltas
        position = swap.position(model)
        ad_result = position.compute([RequestTypes.VALUE, RequestTypes.DELTA])

        # Validate deltas
        validator = GreekValidator(model, swap)
        delta_report = validator.validate_delta_vs_fd(
            ad_result=ad_result,
            curve_type=CurveTypes.GBP_OIS_SONIA,
            bump_bp=1.0,
            fd_method='central',
            tolerance=0.01  # 1% relative error tolerance
        )

        # Print report
        print(delta_report)

        # Assertions
        assert delta_report.max_relative_error < 0.01, \
            f"Delta validation failed: max error {delta_report.max_relative_error:.6f} exceeds 1%"
        assert delta_report.passed, "Delta validation did not pass"

        # Check that errors are reasonable
        assert delta_report.mean_relative_error < 0.005, \
            f"Mean delta error {delta_report.mean_relative_error:.6f} exceeds 0.5%"

    def test_ois_delta_validation_forward_vs_central(self):
        """
        Compare forward difference vs central difference for delta validation.

        Central difference should be more accurate (O(h^2) vs O(h) error).
        """
        # Setup
        value_dt = Date(30, 4, 2024)

        # Build curve
        model = Model(value_dt=value_dt)
        model.build_curve(
            name="GBP_OIS_SONIA",
            px_list=[5.19, 5.13, 5.04, 4.75, 4.24],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            fixed_dcc_type=DayCountTypes.ACT_365F,
            use_ad=True
        )

        # Create swap
        swap = OIS(
            effective_dt=value_dt,
            term_dt_or_tenor="5Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=0.045,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_365F,
            floating_index=CurveTypes.GBP_OIS_SONIA,
            currency=CurrencyTypes.GBP
        )

        # Compute AD deltas
        ad_result = swap.position(model).compute([RequestTypes.VALUE, RequestTypes.DELTA])

        # Validate with central difference
        validator = GreekValidator(model, swap)
        report_central = validator.validate_delta_vs_fd(
            ad_result, fd_method='central', tolerance=0.01
        )

        # Validate with forward difference
        report_forward = validator.validate_delta_vs_fd(
            ad_result, fd_method='forward', tolerance=0.01
        )

        print("\n=== Central Difference ===")
        print(f"Max error: {report_central.max_relative_error:.6f}")
        print(f"Mean error: {report_central.mean_relative_error:.6f}")

        print("\n=== Forward Difference ===")
        print(f"Max error: {report_forward.max_relative_error:.6f}")
        print(f"Mean error: {report_forward.mean_relative_error:.6f}")

        # Central should generally be more accurate
        # (though for some cases they might be similar)
        assert report_central.passed
        assert report_forward.passed


class TestGammaValidation:
    """Test gamma validation via Taylor series expansion."""

    def test_ois_gamma_taylor_100bp_shock(self):
        """
        Validate gamma via 100bp parallel shock.

        Second-order Taylor should significantly outperform first-order.
        """
        # Setup
        value_dt = Date(30, 4, 2024)

        # Build curve with AD and gamma
        model = Model(value_dt=value_dt)
        model.build_curve(
            name="GBP_OIS_SONIA",
            px_list=[5.19, 5.13, 5.04, 4.75, 4.24],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            fixed_dcc_type=DayCountTypes.ACT_365F,
            use_ad=True,
            compute_gamma=True  # Enable gamma computation
        )

        # Create swap
        swap = OIS(
            effective_dt=value_dt,
            term_dt_or_tenor="5Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=0.045,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_365F,
            floating_index=CurveTypes.GBP_OIS_SONIA,
            currency=CurrencyTypes.GBP
        )

        # Compute AD deltas and gammas
        ad_result = swap.position(model).compute([
            RequestTypes.VALUE,
            RequestTypes.DELTA,
            RequestTypes.GAMMA
        ])

        # Validate gamma via Taylor expansion
        validator = GreekValidator(model, swap)
        gamma_report = validator.validate_gamma_taylor_expansion(
            ad_result=ad_result,
            curve_type=CurveTypes.GBP_OIS_SONIA,
            shock_bp=100.0,
            tolerance=0.05  # 5% error for 2nd-order Taylor
        )

        # Print report
        print(gamma_report)

        # Assertions
        assert gamma_report.passed, \
            f"Gamma validation failed: 2nd-order error {gamma_report.error_2nd_order_pct:.4%} exceeds 5%"

        # Gamma should improve accuracy significantly
        assert gamma_report.gamma_improvement_factor > 2.0, \
            f"Gamma only improved accuracy by {gamma_report.gamma_improvement_factor:.2f}x (expected >2x)"

        # Gamma matrix should be symmetric
        assert gamma_report.gamma_matrix_symmetric, \
            f"Gamma matrix not symmetric (max error: {gamma_report.max_symmetry_error:.6e})"

    def test_ois_gamma_taylor_10bp_shock(self):
        """
        Validate gamma via smaller 10bp shock.

        For smaller shocks, both 1st and 2nd order should be accurate,
        but 2nd order should still be better.
        """
        # Setup
        value_dt = Date(30, 4, 2024)

        # Build curve
        model = Model(value_dt=value_dt)
        model.build_curve(
            name="GBP_OIS_SONIA",
            px_list=[5.19, 5.13, 5.04, 4.75, 4.24],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            fixed_dcc_type=DayCountTypes.ACT_365F,
            use_ad=True,
            compute_gamma=True
        )

        # Create swap
        swap = OIS(
            effective_dt=value_dt,
            term_dt_or_tenor="5Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=0.045,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_365F,
            floating_index=CurveTypes.GBP_OIS_SONIA,
            currency=CurrencyTypes.GBP
        )

        # Compute Greeks
        ad_result = swap.position(model).compute([
            RequestTypes.VALUE,
            RequestTypes.DELTA,
            RequestTypes.GAMMA
        ])

        # Validate with small shock
        validator = GreekValidator(model, swap)
        gamma_report = validator.validate_gamma_taylor_expansion(
            ad_result=ad_result,
            shock_bp=10.0,
            tolerance=0.005  # 0.5% error for small shock
        )

        print(gamma_report)

        # For small shocks, errors should be very small
        assert gamma_report.error_2nd_order_pct < 0.005, \
            f"2nd-order error {gamma_report.error_2nd_order_pct:.4%} too large for 10bp shock"

        # Still expect gamma to improve accuracy
        assert gamma_report.gamma_improvement_factor > 1.0


class TestDeltaParallelShift:
    """Test parallel shift delta validation (Phase 1)."""

    def test_ois_parallel_shift_100bp(self):
        """
        Validate that sum of tenor deltas matches parallel shift FD.

        Tests that individual tenor deltas aggregate correctly to capture
        uniform curve movements.
        """
        # Setup
        value_dt = Date(30, 4, 2024)

        # Build curve
        model = Model(value_dt=value_dt)
        model.build_curve(
            name="GBP_OIS_SONIA",
            px_list=[5.19, 5.13, 5.04, 4.75, 4.24],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            fixed_dcc_type=DayCountTypes.ACT_365F,
            use_ad=True
        )

        # Create 5Y OIS swap
        swap = OIS(
            effective_dt=value_dt,
            term_dt_or_tenor="5Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=0.045,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_365F,
            floating_index=CurveTypes.GBP_OIS_SONIA,
            currency=CurrencyTypes.GBP
        )

        # Compute AD deltas
        ad_result = swap.position(model).compute([RequestTypes.VALUE, RequestTypes.DELTA])

        # Validate parallel shift
        validator = GreekValidator(model, swap)
        report = validator.validate_delta_parallel_shift(
            ad_result,
            shock_bp=100.0,
            tolerance=0.001  # 0.1% tolerance
        )

        # Print report
        print(report)

        # Assertions
        assert report.passed, f"Parallel shift validation failed: error {report.relative_error:.6f}"
        assert report.relative_error < 0.001, f"Relative error {report.relative_error:.6f} exceeds 0.1%"

        # Verify sum of contributions equals AD sum
        contrib_sum = sum(report.tenor_contributions.values())
        assert abs(contrib_sum - report.delta_ad_sum) < 1e-10

    def test_ois_parallel_shift_10bp(self):
        """Test parallel shift with smaller 10bp shock."""
        value_dt = Date(30, 4, 2024)

        model = Model(value_dt=value_dt)
        model.build_curve(
            name="GBP_OIS_SONIA",
            px_list=[5.19, 5.13, 5.04, 4.75, 4.24],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            fixed_dcc_type=DayCountTypes.ACT_365F,
            use_ad=True
        )

        swap = OIS(
            effective_dt=value_dt,
            term_dt_or_tenor="5Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=0.045,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_365F,
            floating_index=CurveTypes.GBP_OIS_SONIA,
            currency=CurrencyTypes.GBP
        )

        ad_result = swap.position(model).compute([RequestTypes.VALUE, RequestTypes.DELTA])

        validator = GreekValidator(model, swap)
        report = validator.validate_delta_parallel_shift(
            ad_result,
            shock_bp=10.0,  # Smaller shock
            tolerance=0.001
        )

        print(f"\n10bp Parallel Shift Report:")
        print(f"  AD sum: {report.delta_ad_sum:.4f}")
        print(f"  FD parallel: {report.delta_fd_parallel:.4f}")
        print(f"  Relative error: {report.relative_error*100:.6f}%")

        assert report.passed


class TestDeltaLargeBump:
    """Test large bump non-linearity validation (Phase 2)."""

    def test_ois_large_bump_200bp(self):
        """
        Test 200bp bump to demonstrate non-linearity.

        Delta-only approximation should have significant error for such a large move,
        while delta+gamma should be much more accurate.
        """
        # Setup
        value_dt = Date(30, 4, 2024)

        # Build curve with gamma enabled
        model = Model(value_dt=value_dt)
        model.build_curve(
            name="GBP_OIS_SONIA",
            px_list=[5.19, 5.13, 5.04, 4.75, 4.24],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            fixed_dcc_type=DayCountTypes.ACT_365F,
            use_ad=True,
            compute_gamma=True  # Enable gamma
        )

        # Create swap
        swap = OIS(
            effective_dt=value_dt,
            term_dt_or_tenor="5Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=0.045,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_365F,
            floating_index=CurveTypes.GBP_OIS_SONIA,
            currency=CurrencyTypes.GBP
        )

        # Compute deltas and gammas
        ad_result = swap.position(model).compute([
            RequestTypes.VALUE,
            RequestTypes.DELTA,
            RequestTypes.GAMMA
        ])

        # Validate large bump
        validator = GreekValidator(model, swap)
        report = validator.validate_delta_large_bump(
            ad_result,
            shock_bp=200.0,
            tolerance_delta=0.20,        # 20% for delta-only
            tolerance_delta_gamma=0.05   # 5% for delta+gamma
        )

        # Print report
        print(report)

        # Assertions
        # Delta-only should have significant error
        assert report.error_delta_pct > 0.05, \
            f"Expected significant non-linearity, but delta error only {report.error_delta_pct:.4%}"

        # Delta+gamma should be much better
        if report.error_delta_gamma_pct is not None:
            assert report.error_delta_gamma_pct < report.error_delta_pct, \
                "Gamma should improve accuracy"
            assert report.passed_delta_gamma, \
                f"Delta+gamma error {report.error_delta_gamma_pct:.4%} exceeds tolerance"

            # Gamma should provide significant improvement
            assert report.gamma_improvement_factor > 2.0, \
                f"Gamma improvement factor {report.gamma_improvement_factor:.2f} should be >2x"

    def test_ois_large_bump_specific_tenor(self):
        """Test 200bp bump on a specific tenor (1Y)."""
        value_dt = Date(30, 4, 2024)

        model = Model(value_dt=value_dt)
        model.build_curve(
            name="GBP_OIS_SONIA",
            px_list=[5.19, 5.13, 5.04, 4.75, 4.24],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            fixed_dcc_type=DayCountTypes.ACT_365F,
            use_ad=True,
            compute_gamma=True
        )

        swap = OIS(
            effective_dt=value_dt,
            term_dt_or_tenor="5Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=0.045,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_365F,
            floating_index=CurveTypes.GBP_OIS_SONIA,
            currency=CurrencyTypes.GBP
        )

        ad_result = swap.position(model).compute([
            RequestTypes.VALUE,
            RequestTypes.DELTA,
            RequestTypes.GAMMA
        ])

        validator = GreekValidator(model, swap)

        # Test specific tenor (1Y is index 3)
        report = validator.validate_delta_large_bump(
            ad_result,
            tenor_idx=3,  # 1Y tenor
            shock_bp=200.0,
            tolerance_delta=0.20,
            tolerance_delta_gamma=0.05
        )

        print(f"\nLarge Bump Report (1Y tenor):")
        print(f"  Tenor: {report.tenor}")
        print(f"  Delta error: {report.error_delta_pct:.4%}")
        if report.error_delta_gamma_pct:
            print(f"  Delta+Gamma error: {report.error_delta_gamma_pct:.4%}")
            print(f"  Improvement: {report.gamma_improvement_factor:.2f}x")

        assert report.tenor == "1Y"
        assert report.passed_delta  # Should pass 20% tolerance

    def test_ois_large_bump_without_gamma(self):
        """Test large bump when gamma is not available."""
        value_dt = Date(30, 4, 2024)

        # Build without gamma
        model = Model(value_dt=value_dt)
        model.build_curve(
            name="GBP_OIS_SONIA",
            px_list=[5.19, 5.13, 5.04, 4.75, 4.24],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            fixed_dcc_type=DayCountTypes.ACT_365F,
            use_ad=True,
            compute_gamma=False  # No gamma
        )

        swap = OIS(
            effective_dt=value_dt,
            term_dt_or_tenor="5Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=0.045,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_365F,
            floating_index=CurveTypes.GBP_OIS_SONIA,
            currency=CurrencyTypes.GBP
        )

        ad_result = swap.position(model).compute([RequestTypes.VALUE, RequestTypes.DELTA])

        validator = GreekValidator(model, swap)
        report = validator.validate_delta_large_bump(
            ad_result,
            shock_bp=200.0,
            tolerance_delta=0.20
        )

        # Gamma fields should be None
        assert report.pv_delta_gamma_approx is None
        assert report.error_delta_gamma_pct is None
        assert report.gamma_improvement_factor is None


class TestCurveScenarios:
    """Test curve scenario validation (Phase 3)."""

    def test_slope_scenario_validation(self):
        """
        Test delta validation under slope scenario (steepening).

        Slope scenario: short-end +shock, long-end -shock.
        """
        # Setup
        value_dt = Date(30, 4, 2024)

        model = Model(value_dt=value_dt)
        model.build_curve(
            name="GBP_OIS_SONIA",
            px_list=[5.19, 5.13, 5.04, 4.75, 4.24],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            fixed_dcc_type=DayCountTypes.ACT_365F,
            use_ad=True
        )

        swap = OIS(
            effective_dt=value_dt,
            term_dt_or_tenor="5Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=0.045,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_365F,
            floating_index=CurveTypes.GBP_OIS_SONIA,
            currency=CurrencyTypes.GBP
        )

        ad_result = swap.position(model).compute([RequestTypes.VALUE, RequestTypes.DELTA])

        # Validate with slope scenario
        validator = GreekValidator(model, swap)
        scenarios = [{'type': 'slope', 'shock_bp': 100}]
        report = validator.validate_curve_scenarios(
            ad_result,
            scenarios=scenarios,
            tolerance=0.05  # 5% tolerance
        )

        print(report)

        # Assertions
        assert len(report.scenarios) == 1
        assert report.scenarios[0].name == "slope_+100bp"
        assert report.all_passed, f"Slope scenario failed: error {report.scenarios[0].error_pct:.4%}"

    def test_skew_scenario_validation(self):
        """Test delta validation under skew scenario (belly up)."""
        value_dt = Date(30, 4, 2024)

        model = Model(value_dt=value_dt)
        model.build_curve(
            name="GBP_OIS_SONIA",
            px_list=[5.19, 5.13, 5.04, 4.75, 4.24],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            fixed_dcc_type=DayCountTypes.ACT_365F,
            use_ad=True
        )

        swap = OIS(
            effective_dt=value_dt,
            term_dt_or_tenor="5Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=0.045,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_365F,
            floating_index=CurveTypes.GBP_OIS_SONIA,
            currency=CurrencyTypes.GBP
        )

        ad_result = swap.position(model).compute([RequestTypes.VALUE, RequestTypes.DELTA])

        validator = GreekValidator(model, swap)
        scenarios = [{'type': 'skew', 'shock_bp': 100}]
        report = validator.validate_curve_scenarios(
            ad_result,
            scenarios=scenarios,
            tolerance=0.05
        )

        assert report.all_passed

    def test_multiple_scenarios_validation(self):
        """Test validation with multiple scenarios (slope, skew, butterfly)."""
        value_dt = Date(30, 4, 2024)

        model = Model(value_dt=value_dt)
        model.build_curve(
            name="GBP_OIS_SONIA",
            px_list=[5.19, 5.13, 5.04, 4.75, 4.24],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            fixed_dcc_type=DayCountTypes.ACT_365F,
            use_ad=True
        )

        swap = OIS(
            effective_dt=value_dt,
            term_dt_or_tenor="5Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=0.045,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_365F,
            floating_index=CurveTypes.GBP_OIS_SONIA,
            currency=CurrencyTypes.GBP
        )

        ad_result = swap.position(model).compute([RequestTypes.VALUE, RequestTypes.DELTA])

        validator = GreekValidator(model, swap)

        # Test with default scenarios (will include slope +/-, skew, butterfly)
        report = validator.validate_curve_scenarios(
            ad_result,
            scenarios=None,  # Use defaults
            tolerance=0.05
        )

        print(f"\nMultiple Scenarios Report:")
        print(f"  Number of scenarios: {len(report.scenarios)}")
        print(f"  All passed: {report.all_passed}")
        print(report.to_dataframe())

        # Should have 4 default scenarios
        assert len(report.scenarios) == 4
        assert report.all_passed

        # Check scenario names
        scenario_names = [s.name for s in report.scenarios]
        assert any('slope_+100bp' in name for name in scenario_names)
        assert any('slope_-100bp' in name for name in scenario_names)

    def test_scenario_report_dataframe(self):
        """Test scenario report DataFrame export."""
        value_dt = Date(30, 4, 2024)

        model = Model(value_dt=value_dt)
        model.build_curve(
            name="GBP_OIS_SONIA",
            px_list=[5.19, 5.13, 5.04],
            tenor_list=["1M", "3M", "6M"],
            fixed_dcc_type=DayCountTypes.ACT_365F,
            use_ad=True
        )

        swap = OIS(
            effective_dt=value_dt,
            term_dt_or_tenor="6M",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=0.05,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_365F,
            floating_index=CurveTypes.GBP_OIS_SONIA,
            currency=CurrencyTypes.GBP
        )

        ad_result = swap.position(model).compute([RequestTypes.VALUE, RequestTypes.DELTA])

        validator = GreekValidator(model, swap)
        scenarios = [
            {'type': 'slope', 'shock_bp': 50},
            {'type': 'skew', 'shock_bp': 50}
        ]
        report = validator.validate_curve_scenarios(ad_result, scenarios=scenarios)

        # Export to DataFrame
        df = report.to_dataframe()

        # Check structure
        assert 'Scenario' in df.columns
        assert 'PV_Shocked' in df.columns
        assert 'PV_Delta_Approx' in df.columns
        assert 'Error_%' in df.columns
        assert 'Passed' in df.columns
        assert len(df) == 2

        print("\n" + str(df))


class TestValidationReports:
    """Test validation report formatting and output."""

    def test_delta_report_to_dataframe(self):
        """Test delta report DataFrame export."""
        # Setup
        value_dt = Date(30, 4, 2024)
        model = Model(value_dt=value_dt)
        model.build_curve(
            name="GBP_OIS_SONIA",
            px_list=[5.19, 5.13, 5.04],
            tenor_list=["1M", "3M", "6M"],
            fixed_dcc_type=DayCountTypes.ACT_365F,
            use_ad=True
        )

        swap = OIS(
            effective_dt=value_dt,
            term_dt_or_tenor="6M",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=0.05,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_365F,
            floating_index=CurveTypes.GBP_OIS_SONIA,
            currency=CurrencyTypes.GBP
        )

        ad_result = swap.position(model).compute([RequestTypes.VALUE, RequestTypes.DELTA])

        validator = GreekValidator(model, swap)
        delta_report = validator.validate_delta_vs_fd(ad_result)

        # Export to DataFrame
        df = delta_report.to_dataframe()

        # Check structure
        assert 'Tenor' in df.columns
        assert 'Delta_AD' in df.columns
        assert 'Delta_FD' in df.columns
        assert 'Abs_Error' in df.columns
        assert 'Rel_Error_%' in df.columns
        assert len(df) == 3  # 3 tenors

        print("\n" + str(df))

    def test_delta_report_string_format(self):
        """Test delta report string representation."""
        value_dt = Date(30, 4, 2024)
        model = Model(value_dt=value_dt)
        model.build_curve(
            name="GBP_OIS_SONIA",
            px_list=[5.19, 5.13],
            tenor_list=["1M", "3M"],
            fixed_dcc_type=DayCountTypes.ACT_365F,
            use_ad=True
        )

        swap = OIS(
            effective_dt=value_dt,
            term_dt_or_tenor="3M",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=0.05,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_365F,
            floating_index=CurveTypes.GBP_OIS_SONIA,
            currency=CurrencyTypes.GBP
        )

        ad_result = swap.position(model).compute([RequestTypes.VALUE, RequestTypes.DELTA])
        validator = GreekValidator(model, swap)
        delta_report = validator.validate_delta_vs_fd(ad_result)

        # String representation should be readable
        report_str = str(delta_report)
        assert "DELTA VALIDATION REPORT" in report_str
        assert "GBP_OIS_SONIA" in report_str
        assert "PASSED" in report_str or "FAILED" in report_str

        print(report_str)


class TestXccyCurveRebuild:
    """Test XCCY curve rebuild infrastructure (Phase 4A verification)."""

    def test_xccy_depends_on_curve_domestic(self):
        """
        Test that _xccy_depends_on_curve() correctly identifies domestic OIS dependency.
        """
        value_dt = Date(30, 4, 2024)
        model = Model(value_dt=value_dt)

        # Build domestic USD OIS curve
        model.build_curve(
            name="USD_OIS_SOFR",
            px_list=[5.33, 5.25, 5.10, 4.80, 4.30],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            fixed_dcc_type=DayCountTypes.ACT_360,
            use_ad=True
        )

        # Build foreign GBP OIS curve
        model.build_curve(
            name="GBP_OIS_SONIA",
            px_list=[5.19, 5.13, 5.04, 4.75, 4.24],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            fixed_dcc_type=DayCountTypes.ACT_365F,
            use_ad=True
        )

        # Build XCCY curve (USD domestic, GBP foreign)
        model.build_xccy_curve(
            name="GBP_USD_BASIS",
            domestic_curve_name="USD_OIS_SOFR",
            foreign_curve_name="GBP_OIS_SONIA",
            spot_fx=1.2500,
            basis_spreads=[-30, -28, -25, -20, -15],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            use_ad=True
        )

        # Create dummy swap for validator
        swap = OIS(
            effective_dt=value_dt,
            term_dt_or_tenor="5Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=0.045,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_365F,
            floating_index=CurveTypes.GBP_OIS_SONIA,
            currency=CurrencyTypes.GBP
        )

        validator = GreekValidator(model, swap)

        # Test domestic dependency
        assert validator._xccy_depends_on_curve("GBP_USD_BASIS", "USD_OIS_SOFR") is True
        print("XCCY correctly depends on domestic USD_OIS_SOFR")

    def test_xccy_depends_on_curve_foreign(self):
        """
        Test that _xccy_depends_on_curve() correctly identifies foreign OIS dependency.
        """
        value_dt = Date(30, 4, 2024)
        model = Model(value_dt=value_dt)

        # Build curves
        model.build_curve(
            name="USD_OIS_SOFR",
            px_list=[5.33, 5.25, 5.10, 4.80, 4.30],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            fixed_dcc_type=DayCountTypes.ACT_360,
            use_ad=True
        )

        model.build_curve(
            name="GBP_OIS_SONIA",
            px_list=[5.19, 5.13, 5.04, 4.75, 4.24],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            fixed_dcc_type=DayCountTypes.ACT_365F,
            use_ad=True
        )

        model.build_xccy_curve(
            name="GBP_USD_BASIS",
            domestic_curve_name="USD_OIS_SOFR",
            foreign_curve_name="GBP_OIS_SONIA",
            spot_fx=1.2500,
            basis_spreads=[-30, -28, -25, -20, -15],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            use_ad=True
        )

        swap = OIS(
            effective_dt=value_dt,
            term_dt_or_tenor="5Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=0.045,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_365F,
            floating_index=CurveTypes.GBP_OIS_SONIA,
            currency=CurrencyTypes.GBP
        )

        validator = GreekValidator(model, swap)

        # Test foreign dependency
        assert validator._xccy_depends_on_curve("GBP_USD_BASIS", "GBP_OIS_SONIA") is True
        print("XCCY correctly depends on foreign GBP_OIS_SONIA")

    def test_xccy_rebuild_on_domestic_bump(self):
        """
        Test that XCCY curve rebuilds when domestic OIS curve is bumped.

        This verifies the critical fix from Phase 4A.2.
        """
        value_dt = Date(30, 4, 2024)
        model = Model(value_dt=value_dt)

        # Build curves
        model.build_curve(
            name="USD_OIS_SOFR",
            px_list=[5.33, 5.25, 5.10, 4.80, 4.30],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            fixed_dcc_type=DayCountTypes.ACT_360,
            use_ad=True
        )

        model.build_curve(
            name="GBP_OIS_SONIA",
            px_list=[5.19, 5.13, 5.04, 4.75, 4.24],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            fixed_dcc_type=DayCountTypes.ACT_365F,
            use_ad=True
        )

        model.build_xccy_curve(
            name="GBP_USD_BASIS",
            domestic_curve_name="USD_OIS_SOFR",
            foreign_curve_name="GBP_OIS_SONIA",
            spot_fx=1.2500,
            basis_spreads=[-30, -28, -25, -20, -15],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            use_ad=True
        )

        swap = OIS(
            effective_dt=value_dt,
            term_dt_or_tenor="5Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=0.045,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_365F,
            floating_index=CurveTypes.GBP_OIS_SONIA,
            currency=CurrencyTypes.GBP
        )

        validator = GreekValidator(model, swap)

        # Bump domestic curve (USD_OIS_SOFR) at 5Y tenor (index 4)
        model_bumped = validator._rebuild_model_with_bumped_curve(
            curve_name="USD_OIS_SOFR",
            tenor_index=4,
            bump_amount=0.0001  # 1bp
        )

        # Verify bumped model has GBP_USD_BASIS curve rebuilt (not just copied)
        assert "GBP_USD_BASIS" in model_bumped._curves_dict
        print("XCCY curve exists in bumped model")

        # The XCCY curve should be different due to rebuild
        # (it depends on USD_OIS_SOFR which was bumped)
        xccy_original = model._curves_dict["GBP_USD_BASIS"]
        xccy_bumped = model_bumped._curves_dict["GBP_USD_BASIS"]

        # Check that curves are different objects (not copied)
        assert xccy_original is not xccy_bumped
        print("XCCY curve was rebuilt (not copied)")

    def test_xccy_rebuild_on_foreign_bump(self):
        """
        Test that XCCY curve rebuilds when foreign OIS curve is bumped.

        This is the critical test that would fail without Phase 4A fixes.
        """
        value_dt = Date(30, 4, 2024)
        model = Model(value_dt=value_dt)

        # Build curves
        model.build_curve(
            name="USD_OIS_SOFR",
            px_list=[5.33, 5.25, 5.10, 4.80, 4.30],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            fixed_dcc_type=DayCountTypes.ACT_360,
            use_ad=True
        )

        model.build_curve(
            name="GBP_OIS_SONIA",
            px_list=[5.19, 5.13, 5.04, 4.75, 4.24],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            fixed_dcc_type=DayCountTypes.ACT_365F,
            use_ad=True
        )

        model.build_xccy_curve(
            name="GBP_USD_BASIS",
            domestic_curve_name="USD_OIS_SOFR",
            foreign_curve_name="GBP_OIS_SONIA",
            spot_fx=1.2500,
            basis_spreads=[-30, -28, -25, -20, -15],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            use_ad=True
        )

        swap = OIS(
            effective_dt=value_dt,
            term_dt_or_tenor="5Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=0.045,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_365F,
            floating_index=CurveTypes.GBP_OIS_SONIA,
            currency=CurrencyTypes.GBP
        )

        validator = GreekValidator(model, swap)

        # Bump foreign curve (GBP_OIS_SONIA) at 5Y tenor (index 4)
        model_bumped = validator._rebuild_model_with_bumped_curve(
            curve_name="GBP_OIS_SONIA",
            tenor_index=4,
            bump_amount=0.0001  # 1bp
        )

        # Verify XCCY curve was rebuilt
        assert "GBP_USD_BASIS" in model_bumped._curves_dict
        xccy_original = model._curves_dict["GBP_USD_BASIS"]
        xccy_bumped = model_bumped._curves_dict["GBP_USD_BASIS"]

        # Check that curves are different objects (not copied)
        assert xccy_original is not xccy_bumped
        print("XCCY curve correctly rebuilds when foreign OIS curve is bumped")


class TestXccyDeltaValidation:
    """Test XCCY multi-curve delta validation (Phase 4B verification)."""

    def _build_xccy_model(self, value_dt, use_ad=True):
        """
        Helper to build a complete XCCY model with all curves.

        Returns:
            Model with USD_OIS_SOFR, GBP_OIS_SONIA, and GBP_USD_BASIS curves
        """
        model = Model(value_dt=value_dt)

        # Build domestic USD OIS curve
        model.build_curve(
            name="USD_OIS_SOFR",
            px_list=[5.33, 5.25, 5.10, 4.80, 4.30],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            fixed_dcc_type=DayCountTypes.ACT_360,
            use_ad=use_ad
        )

        # Build foreign GBP OIS curve
        model.build_curve(
            name="GBP_OIS_SONIA",
            px_list=[5.19, 5.13, 5.04, 4.75, 4.24],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            fixed_dcc_type=DayCountTypes.ACT_365F,
            use_ad=use_ad
        )

        # Build XCCY curve (USD domestic, GBP foreign)
        model.build_xccy_curve(
            name="GBP_USD_BASIS",
            domestic_curve_name="USD_OIS_SOFR",
            foreign_curve_name="GBP_OIS_SONIA",
            spot_fx=1.2700,
            basis_spreads=[-30, -28, -25, -20, -15],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            use_ad=use_ad
        )

        return model

    def test_xccy_delta_validation_domestic_curve(self):
        """
        Validate delta for domestic OIS curve in XCCY swap.

        Tests that AD deltas match FD deltas for the USD_OIS_SOFR curve
        when used in an XCCY swap context.
        """
        value_dt = Date(30, 4, 2024)
        model = self._build_xccy_model(value_dt)

        # Create 5Y XCCY fixed-float swap
        # Domestic (USD): pay fixed 4.3%
        # Foreign (GBP): receive floating SONIA + 0bp
        swap = XccyFixFloat(
            effective_dt=value_dt,
            term_dt_or_tenor="5Y",
            domestic_notional=1_000_000,  # USD
            foreign_notional=787_402,  # GBP (at spot FX 1.27)
            domestic_leg_type=SwapTypes.PAY,
            domestic_coupon=0.043,
            foreign_spread=0.0,
            domestic_freq_type=FrequencyTypes.ANNUAL,
            foreign_freq_type=FrequencyTypes.ANNUAL,
            domestic_dc_type=DayCountTypes.ACT_360,
            foreign_dc_type=DayCountTypes.ACT_365F,
            domestic_floating_index=CurveTypes.USD_OIS_SOFR,
            foreign_floating_index=CurveTypes.GBP_OIS_SONIA,
            domestic_currency=CurrencyTypes.USD,
            foreign_currency=CurrencyTypes.GBP
        )

        # Compute AD deltas (should return Risk object with deltas for all curves)
        engine = Engine(model)
        ad_result = engine.compute(swap, [RequestTypes.VALUE, RequestTypes.DELTA])

        # Validate domestic curve deltas
        validator = GreekValidator(model, swap)

        # Extract delta for domestic curve only
        from cavour.requests.analytics_results import AnalyticsResult
        from cavour.requests.risk_result import Delta

        # For XCCY swaps, ad_result.risk should be a Risk object with multiple deltas
        # Extract just the USD_OIS_SOFR delta
        if hasattr(ad_result.risk, 'deltas'):
            usd_delta = ad_result.risk.deltas[CurveTypes.USD_OIS_SOFR]
            usd_result = AnalyticsResult(value=ad_result.value, risk=usd_delta)
        else:
            # Fallback if risk is already a single Delta object
            usd_result = ad_result

        delta_report = validator.validate_delta_vs_fd(
            ad_result=usd_result,
            curve_type=CurveTypes.USD_OIS_SOFR,
            bump_bp=1.0,
            tolerance=0.01,  # 1% relative error
            fd_method='central'
        )

        print(f"\nDomestic USD curve delta validation:")
        print(delta_report)

        assert delta_report.passed, \
            f"Domestic curve delta validation failed: max error {delta_report.max_relative_error:.6f}"

    def test_xccy_delta_validation_foreign_curve(self):
        """
        Validate delta for foreign OIS curve in XCCY swap.

        This is the critical test that validates the Phase 4A XCCY cascade fixes.
        Without proper cascade rebuild, this would fail.
        """
        value_dt = Date(30, 4, 2024)
        model = self._build_xccy_model(value_dt)

        swap = XccyFixFloat(
            effective_dt=value_dt,
            term_dt_or_tenor="5Y",
            domestic_notional=1_000_000,
            foreign_notional=787_402,
            domestic_leg_type=SwapTypes.PAY,
            domestic_coupon=0.043,
            foreign_spread=0.0,
            domestic_freq_type=FrequencyTypes.ANNUAL,
            foreign_freq_type=FrequencyTypes.ANNUAL,
            domestic_dc_type=DayCountTypes.ACT_360,
            foreign_dc_type=DayCountTypes.ACT_365F,
            domestic_floating_index=CurveTypes.USD_OIS_SOFR,
            foreign_floating_index=CurveTypes.GBP_OIS_SONIA,
            domestic_currency=CurrencyTypes.USD,
            foreign_currency=CurrencyTypes.GBP
        )

        engine = Engine(model)
        ad_result = engine.compute(swap, [RequestTypes.VALUE, RequestTypes.DELTA])

        validator = GreekValidator(model, swap)

        # Extract delta for foreign curve only
        from cavour.requests.analytics_results import AnalyticsResult

        if hasattr(ad_result.risk, 'deltas'):
            gbp_delta = ad_result.risk.deltas[CurveTypes.GBP_OIS_SONIA]
            gbp_result = AnalyticsResult(value=ad_result.value, risk=gbp_delta)
        else:
            gbp_result = ad_result

        delta_report = validator.validate_delta_vs_fd(
            ad_result=gbp_result,
            curve_type=CurveTypes.GBP_OIS_SONIA,
            bump_bp=1.0,
            tolerance=0.01,
            fd_method='central'
        )

        print(f"\nForeign GBP curve delta validation:")
        print(delta_report)

        assert delta_report.passed, \
            f"Foreign curve delta validation failed: max error {delta_report.max_relative_error:.6f}"
        print("Foreign curve delta validation PASSED - Phase 4A fixes working correctly!")

    def test_xccy_delta_validation_xccy_curve(self):
        """
        Validate delta for XCCY basis curve.

        Tests deltas for the GBP_USD_BASIS basis spread curve.
        """
        value_dt = Date(30, 4, 2024)
        model = self._build_xccy_model(value_dt)

        swap = XccyFixFloat(
            effective_dt=value_dt,
            term_dt_or_tenor="5Y",
            domestic_notional=1_000_000,
            foreign_notional=787_402,
            domestic_leg_type=SwapTypes.PAY,
            domestic_coupon=0.043,
            foreign_spread=0.0,
            domestic_freq_type=FrequencyTypes.ANNUAL,
            foreign_freq_type=FrequencyTypes.ANNUAL,
            domestic_dc_type=DayCountTypes.ACT_360,
            foreign_dc_type=DayCountTypes.ACT_365F,
            domestic_floating_index=CurveTypes.USD_OIS_SOFR,
            foreign_floating_index=CurveTypes.GBP_OIS_SONIA,
            domestic_currency=CurrencyTypes.USD,
            foreign_currency=CurrencyTypes.GBP
        )

        engine = Engine(model)
        ad_result = engine.compute(swap, [RequestTypes.VALUE, RequestTypes.DELTA])

        validator = GreekValidator(model, swap)

        # Extract delta for XCCY curve only
        from cavour.requests.analytics_results import AnalyticsResult

        if hasattr(ad_result.risk, 'deltas'):
            xccy_delta = ad_result.risk.deltas[CurveTypes.GBP_USD_BASIS]
            xccy_result = AnalyticsResult(value=ad_result.value, risk=xccy_delta)
        else:
            xccy_result = ad_result

        delta_report = validator.validate_delta_vs_fd(
            ad_result=xccy_result,
            curve_type=CurveTypes.GBP_USD_BASIS,
            bump_bp=1.0,
            tolerance=0.01,
            fd_method='central'
        )

        print(f"\nXCCY basis curve delta validation:")
        print(delta_report)

        assert delta_report.passed, \
            f"XCCY curve delta validation failed: max error {delta_report.max_relative_error:.6f}"

    def test_xccy_multi_curve_validation(self):
        """
        Validate all deltas simultaneously using validate_xccy_multi_curve().

        This is the comprehensive multi-curve validation test that validates
        deltas across all three curves (USD OIS, GBP OIS, USD_GBP XCCY) in a
        single call.
        """
        value_dt = Date(30, 4, 2024)
        model = self._build_xccy_model(value_dt)

        swap = XccyFixFloat(
            effective_dt=value_dt,
            term_dt_or_tenor="5Y",
            domestic_notional=1_000_000,
            foreign_notional=787_402,
            domestic_leg_type=SwapTypes.PAY,
            domestic_coupon=0.043,
            foreign_spread=0.0,
            domestic_freq_type=FrequencyTypes.ANNUAL,
            foreign_freq_type=FrequencyTypes.ANNUAL,
            domestic_dc_type=DayCountTypes.ACT_360,
            foreign_dc_type=DayCountTypes.ACT_365F,
            domestic_floating_index=CurveTypes.USD_OIS_SOFR,
            foreign_floating_index=CurveTypes.GBP_OIS_SONIA,
            domestic_currency=CurrencyTypes.USD,
            foreign_currency=CurrencyTypes.GBP
        )

        engine = Engine(model)
        ad_result = engine.compute(swap, [RequestTypes.VALUE, RequestTypes.DELTA])

        validator = GreekValidator(model, swap)

        # Use multi-curve validation method
        multi_report = validator.validate_xccy_multi_curve(
            ad_result=ad_result,
            bump_bp=1.0,
            tolerance=0.001,  # 0.1% for multi-curve
            fd_method='central'
        )

        print(f"\n{'='*70}")
        print("MULTI-CURVE VALIDATION REPORT")
        print(f"{'='*70}")
        print(multi_report)
        print(f"\nDataFrame summary:")
        print(multi_report.to_dataframe())

        # Check that all curves passed
        assert multi_report.all_passed, \
            f"Multi-curve validation failed. See report above for details."

        # Check that we validated all expected curves
        expected_curves = {CurveTypes.USD_OIS_SOFR, CurveTypes.GBP_OIS_SONIA, CurveTypes.GBP_USD_BASIS}
        assert set(multi_report.curves) == expected_curves, \
            f"Expected curves {expected_curves}, got {set(multi_report.curves)}"

        print("\nAll XCCY curves validated successfully!")


class TestXccyParallelShift:
    """Test XCCY parallel shift validation (Phase 4B.3 verification)."""

    def _build_xccy_model(self, value_dt, use_ad=True):
        """Helper to build a complete XCCY model with all curves."""
        model = Model(value_dt=value_dt)

        model.build_curve(
            name="USD_OIS_SOFR",
            px_list=[5.33, 5.25, 5.10, 4.80, 4.30],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            fixed_dcc_type=DayCountTypes.ACT_360,
            use_ad=use_ad
        )

        model.build_curve(
            name="GBP_OIS_SONIA",
            px_list=[5.19, 5.13, 5.04, 4.75, 4.24],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            fixed_dcc_type=DayCountTypes.ACT_365F,
            use_ad=use_ad
        )

        model.build_xccy_curve(
            name="GBP_USD_BASIS",
            domestic_curve_name="USD_OIS_SOFR",
            foreign_curve_name="GBP_OIS_SONIA",
            spot_fx=1.2700,
            basis_spreads=[-30, -28, -25, -20, -15],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            use_ad=use_ad
        )

        return model

    def test_xccy_parallel_shift_100bp(self):
        """
        Test parallel shift validation across all XCCY curves with 100bp shock.
        """
        value_dt = Date(30, 4, 2024)
        model = self._build_xccy_model(value_dt)

        swap = XccyFixFloat(
            effective_dt=value_dt,
            term_dt_or_tenor="5Y",
            domestic_notional=1_000_000,
            foreign_notional=787_402,
            domestic_leg_type=SwapTypes.PAY,
            domestic_coupon=0.043,
            foreign_spread=0.0,
            domestic_freq_type=FrequencyTypes.ANNUAL,
            foreign_freq_type=FrequencyTypes.ANNUAL,
            domestic_dc_type=DayCountTypes.ACT_360,
            foreign_dc_type=DayCountTypes.ACT_365F,
            domestic_floating_index=CurveTypes.USD_OIS_SOFR,
            foreign_floating_index=CurveTypes.GBP_OIS_SONIA,
            domestic_currency=CurrencyTypes.USD,
            foreign_currency=CurrencyTypes.GBP
        )

        engine = Engine(model)
        ad_result = engine.compute(swap, [RequestTypes.VALUE, RequestTypes.DELTA])

        validator = GreekValidator(model, swap)

        # Validate parallel shift across all curves
        parallel_report = validator.validate_xccy_parallel_shift(
            ad_result=ad_result,
            shock_bp=100.0,
            tolerance=0.001,  # 0.1% tolerance
            fd_method='central'
        )

        print(f"\n{'='*70}")
        print("XCCY PARALLEL SHIFT VALIDATION (100bp)")
        print(f"{'='*70}")
        print(parallel_report)
        print(f"\nDataFrame summary:")
        print(parallel_report.to_dataframe())

        assert parallel_report.all_passed, \
            f"Parallel shift validation failed. See report above."

        print("\nAll curves passed 100bp parallel shift validation!")

    def test_xccy_parallel_shift_50bp(self):
        """
        Test parallel shift validation with smaller 50bp shock.
        """
        value_dt = Date(30, 4, 2024)
        model = self._build_xccy_model(value_dt)

        swap = XccyFixFloat(
            effective_dt=value_dt,
            term_dt_or_tenor="5Y",
            domestic_notional=1_000_000,
            foreign_notional=787_402,
            domestic_leg_type=SwapTypes.PAY,
            domestic_coupon=0.043,
            foreign_spread=0.0,
            domestic_freq_type=FrequencyTypes.ANNUAL,
            foreign_freq_type=FrequencyTypes.ANNUAL,
            domestic_dc_type=DayCountTypes.ACT_360,
            foreign_dc_type=DayCountTypes.ACT_365F,
            domestic_floating_index=CurveTypes.USD_OIS_SOFR,
            foreign_floating_index=CurveTypes.GBP_OIS_SONIA,
            domestic_currency=CurrencyTypes.USD,
            foreign_currency=CurrencyTypes.GBP
        )

        engine = Engine(model)
        ad_result = engine.compute(swap, [RequestTypes.VALUE, RequestTypes.DELTA])

        validator = GreekValidator(model, swap)

        parallel_report = validator.validate_xccy_parallel_shift(
            ad_result=ad_result,
            shock_bp=50.0,
            tolerance=0.001,
            fd_method='central'
        )

        print(f"\n50bp parallel shift validation:")
        print(parallel_report.to_dataframe())

        assert parallel_report.all_passed, \
            f"50bp parallel shift validation failed"

        print("\nAll curves passed 50bp parallel shift validation!")


class TestXccyScenarios:
    """Test XCCY scenario validation (Phase 4B.4 verification)."""

    def _build_xccy_model(self, value_dt, use_ad=True):
        """Helper to build a complete XCCY model with all curves."""
        model = Model(value_dt=value_dt)

        model.build_curve(
            name="USD_OIS_SOFR",
            px_list=[5.33, 5.25, 5.10, 4.80, 4.30],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            fixed_dcc_type=DayCountTypes.ACT_360,
            use_ad=use_ad
        )

        model.build_curve(
            name="GBP_OIS_SONIA",
            px_list=[5.19, 5.13, 5.04, 4.75, 4.24],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            fixed_dcc_type=DayCountTypes.ACT_365F,
            use_ad=use_ad
        )

        model.build_xccy_curve(
            name="GBP_USD_BASIS",
            domestic_curve_name="USD_OIS_SOFR",
            foreign_curve_name="GBP_OIS_SONIA",
            spot_fx=1.2700,
            basis_spreads=[-30, -28, -25, -20, -15],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            use_ad=use_ad
        )

        return model

    def test_xccy_slope_scenario(self):
        """
        Test slope scenario (steepening) across all XCCY curves.
        """
        value_dt = Date(30, 4, 2024)
        model = self._build_xccy_model(value_dt)

        swap = XccyFixFloat(
            effective_dt=value_dt,
            term_dt_or_tenor="5Y",
            domestic_notional=1_000_000,
            foreign_notional=787_402,
            domestic_leg_type=SwapTypes.PAY,
            domestic_coupon=0.043,
            foreign_spread=0.0,
            domestic_freq_type=FrequencyTypes.ANNUAL,
            foreign_freq_type=FrequencyTypes.ANNUAL,
            domestic_dc_type=DayCountTypes.ACT_360,
            foreign_dc_type=DayCountTypes.ACT_365F,
            domestic_floating_index=CurveTypes.USD_OIS_SOFR,
            foreign_floating_index=CurveTypes.GBP_OIS_SONIA,
            domestic_currency=CurrencyTypes.USD,
            foreign_currency=CurrencyTypes.GBP
        )

        engine = Engine(model)
        ad_result = engine.compute(swap, [RequestTypes.VALUE, RequestTypes.DELTA])

        validator = GreekValidator(model, swap)

        # Run slope scenario for all curves
        scenarios = [{'type': 'slope', 'shock_bp': 100}]
        scenario_reports = validator.validate_xccy_scenarios(
            ad_result=ad_result,
            scenarios=scenarios,
            tolerance=0.05  # 5% tolerance for scenarios
        )

        print(f"\n{'='*70}")
        print("XCCY SLOPE SCENARIO VALIDATION")
        print(f"{'='*70}")

        for curve_type, report in scenario_reports.items():
            print(f"\n{curve_type.name}:")
            print(report)

        # Check all curves passed
        for curve_type, report in scenario_reports.items():
            assert report.all_passed, \
                f"Slope scenario failed for {curve_type.name}"

        print("\nAll curves passed slope scenario validation!")

    def test_xccy_skew_scenario(self):
        """
        Test skew scenario (belly-up) across all XCCY curves.
        """
        value_dt = Date(30, 4, 2024)
        model = self._build_xccy_model(value_dt)

        swap = XccyFixFloat(
            effective_dt=value_dt,
            term_dt_or_tenor="5Y",
            domestic_notional=1_000_000,
            foreign_notional=787_402,
            domestic_leg_type=SwapTypes.PAY,
            domestic_coupon=0.043,
            foreign_spread=0.0,
            domestic_freq_type=FrequencyTypes.ANNUAL,
            foreign_freq_type=FrequencyTypes.ANNUAL,
            domestic_dc_type=DayCountTypes.ACT_360,
            foreign_dc_type=DayCountTypes.ACT_365F,
            domestic_floating_index=CurveTypes.USD_OIS_SOFR,
            foreign_floating_index=CurveTypes.GBP_OIS_SONIA,
            domestic_currency=CurrencyTypes.USD,
            foreign_currency=CurrencyTypes.GBP
        )

        engine = Engine(model)
        ad_result = engine.compute(swap, [RequestTypes.VALUE, RequestTypes.DELTA])

        validator = GreekValidator(model, swap)

        scenarios = [{'type': 'skew', 'shock_bp': 100}]
        scenario_reports = validator.validate_xccy_scenarios(
            ad_result=ad_result,
            scenarios=scenarios,
            tolerance=0.05
        )

        print(f"\nXCCY skew scenario validation:")
        for curve_type, report in scenario_reports.items():
            assert report.all_passed, \
                f"Skew scenario failed for {curve_type.name}"

        print("All curves passed skew scenario validation!")

    def test_xccy_butterfly_scenario(self):
        """
        Test butterfly scenario (wings vs belly) across all XCCY curves.
        """
        value_dt = Date(30, 4, 2024)
        model = self._build_xccy_model(value_dt)

        swap = XccyFixFloat(
            effective_dt=value_dt,
            term_dt_or_tenor="5Y",
            domestic_notional=1_000_000,
            foreign_notional=787_402,
            domestic_leg_type=SwapTypes.PAY,
            domestic_coupon=0.043,
            foreign_spread=0.0,
            domestic_freq_type=FrequencyTypes.ANNUAL,
            foreign_freq_type=FrequencyTypes.ANNUAL,
            domestic_dc_type=DayCountTypes.ACT_360,
            foreign_dc_type=DayCountTypes.ACT_365F,
            domestic_floating_index=CurveTypes.USD_OIS_SOFR,
            foreign_floating_index=CurveTypes.GBP_OIS_SONIA,
            domestic_currency=CurrencyTypes.USD,
            foreign_currency=CurrencyTypes.GBP
        )

        engine = Engine(model)
        ad_result = engine.compute(swap, [RequestTypes.VALUE, RequestTypes.DELTA])

        validator = GreekValidator(model, swap)

        scenarios = [{'type': 'butterfly', 'shock_bp': 100}]
        scenario_reports = validator.validate_xccy_scenarios(
            ad_result=ad_result,
            scenarios=scenarios,
            tolerance=0.05
        )

        print(f"\nXCCY butterfly scenario validation:")
        for curve_type, report in scenario_reports.items():
            assert report.all_passed, \
                f"Butterfly scenario failed for {curve_type.name}"

        print("All curves passed butterfly scenario validation!")

    def test_xccy_all_scenarios(self):
        """
        Test all scenarios (slope, skew, butterfly) across all XCCY curves.
        """
        value_dt = Date(30, 4, 2024)
        model = self._build_xccy_model(value_dt)

        swap = XccyFixFloat(
            effective_dt=value_dt,
            term_dt_or_tenor="5Y",
            domestic_notional=1_000_000,
            foreign_notional=787_402,
            domestic_leg_type=SwapTypes.PAY,
            domestic_coupon=0.043,
            foreign_spread=0.0,
            domestic_freq_type=FrequencyTypes.ANNUAL,
            foreign_freq_type=FrequencyTypes.ANNUAL,
            domestic_dc_type=DayCountTypes.ACT_360,
            foreign_dc_type=DayCountTypes.ACT_365F,
            domestic_floating_index=CurveTypes.USD_OIS_SOFR,
            foreign_floating_index=CurveTypes.GBP_OIS_SONIA,
            domestic_currency=CurrencyTypes.USD,
            foreign_currency=CurrencyTypes.GBP
        )

        engine = Engine(model)
        ad_result = engine.compute(swap, [RequestTypes.VALUE, RequestTypes.DELTA])

        validator = GreekValidator(model, swap)

        # Test all scenario types
        scenarios = [
            {'type': 'slope', 'shock_bp': 100},
            {'type': 'skew', 'shock_bp': 100},
            {'type': 'butterfly', 'shock_bp': 100}
        ]

        scenario_reports = validator.validate_xccy_scenarios(
            ad_result=ad_result,
            scenarios=scenarios,
            tolerance=0.05
        )

        print(f"\n{'='*70}")
        print("XCCY ALL SCENARIOS VALIDATION")
        print(f"{'='*70}")

        # Aggregate results across all curves and scenarios
        total_scenarios = 0
        passed_scenarios = 0

        for curve_type, report in scenario_reports.items():
            print(f"\n{curve_type.name}: {len(report.scenarios)} scenarios")
            for scenario in report.scenarios:
                total_scenarios += 1
                if scenario.passed:
                    passed_scenarios += 1
                print(f"  {scenario.name}: {'PASSED' if scenario.passed else 'FAILED'}")

        print(f"\nOverall: {passed_scenarios}/{total_scenarios} scenarios passed")

        # Check all scenarios passed for all curves
        for curve_type, report in scenario_reports.items():
            assert report.all_passed, \
                f"Not all scenarios passed for {curve_type.name}"

        print("\nAll scenarios passed for all XCCY curves!")


class TestPhase5CrossGamma:
    """
    Phase 5: Multi-curve cross-gamma validation.

    Tests cross-gamma (d^2PV / d(curve1) * d(curve2)) validation via double
    finite difference for cross-currency swaps.
    """

    def _build_xccy_model(self, value_dt, use_ad=True):
        """
        Helper to build a complete XCCY model with all curves.

        Returns:
            Model with USD_OIS_SOFR, GBP_OIS_SONIA, and GBP_USD_BASIS curves
        """
        model = Model(value_dt=value_dt)

        # Build domestic USD OIS curve
        model.build_curve(
            name="USD_OIS_SOFR",
            px_list=[5.33, 5.25, 5.10, 4.80, 4.30],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            fixed_dcc_type=DayCountTypes.ACT_360,
            use_ad=use_ad
        )

        # Build foreign GBP OIS curve
        model.build_curve(
            name="GBP_OIS_SONIA",
            px_list=[5.19, 5.13, 5.04, 4.75, 4.24],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            fixed_dcc_type=DayCountTypes.ACT_365F,
            use_ad=use_ad
        )

        # Build XCCY curve (USD domestic, GBP foreign)
        model.build_xccy_curve(
            name="GBP_USD_BASIS",
            domestic_curve_name="USD_OIS_SOFR",
            foreign_curve_name="GBP_OIS_SONIA",
            spot_fx=1.2700,
            basis_spreads=[-30, -28, -25, -20, -15],
            tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            use_ad=use_ad
        )

        return model

    def test_cross_gamma_validation_foreign_vs_basis(self):
        """
        Validate cross-gamma between Foreign OIS and XCCY Basis curves.

        Tests that AD cross-gamma matches FD cross-gamma computed via double
        finite difference: CrossGamma[i,j] = (PV_11 - PV_10 - PV_01 + PV_00) / bump^2

        This is the primary cross-gamma currently computed by the Engine.
        """
        value_dt = Date(30, 4, 2024)
        model = self._build_xccy_model(value_dt)

        # Create 5Y XCCY fixed-float swap
        swap = XccyFixFloat(
            effective_dt=value_dt,
            term_dt_or_tenor="5Y",
            domestic_notional=1_000_000,  # USD
            foreign_notional=787_402,  # GBP (at spot FX 1.27)
            domestic_leg_type=SwapTypes.PAY,
            domestic_coupon=0.043,
            foreign_spread=0.0,
            domestic_freq_type=FrequencyTypes.ANNUAL,
            foreign_freq_type=FrequencyTypes.ANNUAL,
            domestic_dc_type=DayCountTypes.ACT_360,
            foreign_dc_type=DayCountTypes.ACT_365F,
            domestic_floating_index=CurveTypes.USD_OIS_SOFR,
            foreign_floating_index=CurveTypes.GBP_OIS_SONIA,
            domestic_currency=CurrencyTypes.USD,
            foreign_currency=CurrencyTypes.GBP
        )

        # Compute AD gamma and cross-gamma
        engine = Engine(model)
        ad_result = engine.compute(swap, [RequestTypes.VALUE, RequestTypes.GAMMA])

        # Check that cross-gamma exists
        cross_gamma_obj = ad_result.risk.cross_gamma(
            CurveTypes.GBP_OIS_SONIA,
            CurveTypes.GBP_USD_BASIS
        )
        assert cross_gamma_obj is not None, \
            "Cross-gamma between GBP_OIS_SONIA and GBP_USD_BASIS should exist"

        print(f"\nCross-gamma shape: {cross_gamma_obj.risk_matrix.shape}")
        print(f"Tenors curve 1 (GBP_OIS_SONIA): {cross_gamma_obj.tenors_curve1}")
        print(f"Tenors curve 2 (GBP_USD_BASIS): {cross_gamma_obj.tenors_curve2}")
        print(f"Total cross-gamma value: {cross_gamma_obj.value}")

        # Validate cross-gamma via double FD
        validator = GreekValidator(model, swap)
        report = validator.validate_cross_gamma(
            ad_result=ad_result,
            curve_type_1=CurveTypes.GBP_OIS_SONIA,
            curve_type_2=CurveTypes.GBP_USD_BASIS,
            bump_bp=1.0,  # 1bp for numerical stability in double differencing
            tolerance=0.02  # 2% tolerance (higher than delta due to double differencing)
        )

        # Print detailed report
        print(report)

        # Check validation passed
        assert report.passed, \
            f"Cross-gamma validation failed with max error {report.max_relative_error*100:.4f}%"

        # Check reasonable error levels
        assert report.max_relative_error < 0.05, \
            f"Cross-gamma max relative error too high: {report.max_relative_error*100:.2f}%"
        assert report.mean_relative_error < 0.01, \
            f"Cross-gamma mean relative error too high: {report.mean_relative_error*100:.2f}%"

        print("\nCross-gamma validation PASSED!")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
