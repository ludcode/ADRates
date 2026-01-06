"""
Comprehensive tests for interpolation methods.

Tests all interpolation types with focus on:
- PCHIP and cubic splines (scipy-based, most reliable)
- Flat forward and linear methods (JAX-based)
- Exact reproduction at knot points
- Monotonicity
- Realistic curve shapes
"""

import pytest
import numpy as np
import jax.numpy as jnp
from cavour.market.curves.interpolator import Interpolator, InterpTypes, interpolate


class TestInterpolatorPCHIP:
    """Test PCHIP interpolation methods (scipy-based, reliable)"""

    def test_pchip_zero_at_knot_points(self):
        """Test PCHIP_ZERO_RATES exactly reproduces knot points"""
        times = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        dfs = np.array([0.98, 0.95, 0.90, 0.80, 0.70])

        interp = Interpolator(InterpTypes.PCHIP_ZERO_RATES)
        interp.fit(times, dfs)

        for i, t in enumerate(times):
            df = interp.interpolate(t)
            assert abs(df - dfs[i]) < 1e-10, f"Failed at knot point {i}: {df} vs {dfs[i]}"

    def test_pchip_log_discount_at_knot_points(self):
        """Test PCHIP_LOG_DISCOUNT exactly reproduces knot points"""
        times = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        dfs = np.array([0.98, 0.95, 0.90, 0.80, 0.70])

        interp = Interpolator(InterpTypes.PCHIP_LOG_DISCOUNT)
        interp.fit(times, dfs)

        for i, t in enumerate(times):
            df = interp.interpolate(t)
            assert abs(df - dfs[i]) < 1e-10, f"Failed at knot point {i}"

    def test_pchip_monotonic_decreasing(self):
        """Test that PCHIP preserves monotonicity"""
        times = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        dfs = np.array([0.98, 0.95, 0.90, 0.80, 0.70])

        interp = Interpolator(InterpTypes.PCHIP_ZERO_RATES)
        interp.fit(times, dfs)

        test_times = np.linspace(0.5, 10.0, 100)
        prev_df = 1.0
        for t in test_times:
            df = interp.interpolate(t)
            assert df <= prev_df + 1e-10, f"Not monotonic at t={t}: {df} > {prev_df}"
            prev_df = df

    def test_pchip_between_knots(self):
        """Test PCHIP interpolation between knot points"""
        times = np.array([1.0, 2.0, 5.0])
        dfs = np.array([0.95, 0.90, 0.80])

        interp = Interpolator(InterpTypes.PCHIP_ZERO_RATES)
        interp.fit(times, dfs)

        df_mid = interp.interpolate(1.5)
        assert 0.80 < df_mid < 0.95, f"Interpolated value {df_mid} out of range"

    def test_pchip_smooth(self):
        """Test that PCHIP produces smooth curves (no large jumps)"""
        times = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        dfs = np.array([0.98, 0.95, 0.90, 0.80, 0.70])

        interp = Interpolator(InterpTypes.PCHIP_ZERO_RATES)
        interp.fit(times, dfs)

        test_times = np.linspace(0.5, 10.0, 200)
        test_dfs = [interp.interpolate(t) for t in test_times]

        for i in range(1, len(test_dfs)):
            change = abs(test_dfs[i] - test_dfs[i-1])
            assert change < 0.01, f"Large jump at index {i}"

    def test_pchip_array_input(self):
        """Test PCHIP with array of times"""
        times = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        dfs = np.array([0.98, 0.95, 0.90, 0.80, 0.70])

        interp = Interpolator(InterpTypes.PCHIP_ZERO_RATES)
        interp.fit(times, dfs)

        test_times = np.array([1.0, 2.0, 5.0])
        result = interp.interpolate(test_times)

        # Should match knot points
        assert abs(result[0] - 0.95) < 1e-10
        assert abs(result[1] - 0.90) < 1e-10
        assert abs(result[2] - 0.80) < 1e-10


class TestInterpolatorCubicSplines:
    """Test cubic spline interpolation methods (scipy-based)"""

    def test_natcubic_zero_at_knot_points(self):
        """Test NATCUBIC_ZERO_RATES at knot points"""
        times = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        dfs = np.array([0.98, 0.95, 0.90, 0.80, 0.70])

        interp = Interpolator(InterpTypes.NATCUBIC_ZERO_RATES)
        interp.fit(times, dfs)

        for i, t in enumerate(times):
            df = interp.interpolate(t)
            assert abs(df - dfs[i]) < 1e-10, f"Failed at knot point {i}"

    def test_natcubic_log_discount_at_knot_points(self):
        """Test NATCUBIC_LOG_DISCOUNT at knot points"""
        times = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        dfs = np.array([0.98, 0.95, 0.90, 0.80, 0.70])

        interp = Interpolator(InterpTypes.NATCUBIC_LOG_DISCOUNT)
        interp.fit(times, dfs)

        for i, t in enumerate(times):
            df = interp.interpolate(t)
            assert abs(df - dfs[i]) < 1e-10

    def test_fincubic_zero_at_knot_points(self):
        """Test FINCUBIC_ZERO_RATES at knot points"""
        times = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        dfs = np.array([0.98, 0.95, 0.90, 0.80, 0.70])

        interp = Interpolator(InterpTypes.FINCUBIC_ZERO_RATES)
        interp.fit(times, dfs)

        for i, t in enumerate(times):
            df = interp.interpolate(t)
            assert abs(df - dfs[i]) < 1e-10

    def test_cubic_smooth(self):
        """Test that cubic splines produce smooth curves"""
        times = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        dfs = np.array([0.98, 0.95, 0.90, 0.80, 0.70])

        interp = Interpolator(InterpTypes.NATCUBIC_ZERO_RATES)
        interp.fit(times, dfs)

        test_times = np.linspace(0.5, 10.0, 200)
        test_dfs = [interp.interpolate(t) for t in test_times]

        for i in range(1, len(test_dfs)):
            change = abs(test_dfs[i] - test_dfs[i-1])
            assert change < 0.01

    def test_cubic_between_knots(self):
        """Test cubic interpolation between knots"""
        times = np.array([1.0, 2.0, 5.0])
        dfs = np.array([0.95, 0.90, 0.80])

        interp = Interpolator(InterpTypes.NATCUBIC_ZERO_RATES)
        interp.fit(times, dfs)

        df_mid = interp.interpolate(1.5)
        assert 0.80 < df_mid < 0.95


class TestInterpolatorJAXMethods:
    """Test JAX-based interpolation methods"""

    def test_flat_fwd_with_jax_arrays(self):
        """Test FLAT_FWD_RATES with JAX arrays"""
        times = jnp.array([0.0, 1.0, 2.0, 5.0, 10.0])
        dfs = jnp.array([1.0, 0.95, 0.90, 0.80, 0.70])

        interp = Interpolator(InterpTypes.FLAT_FWD_RATES)
        interp.fit(times, dfs)

        # Test at knot points
        for i, t in enumerate(times):
            df = interp.interpolate(float(t))
            df_val = float(df) if hasattr(df, '__float__') else df
            assert abs(df_val - float(dfs[i])) < 1e-10

    def test_linear_zero_with_jax_arrays(self):
        """Test LINEAR_ZERO_RATES with JAX arrays"""
        times = jnp.array([0.5, 1.0, 2.0, 5.0])
        dfs = jnp.array([0.98, 0.95, 0.90, 0.80])

        interp = Interpolator(InterpTypes.LINEAR_ZERO_RATES)
        interp.fit(times, dfs)

        # Test at knot points
        for i, t in enumerate(times):
            df = interp.interpolate(float(t))
            df_val = float(df) if hasattr(df, '__float__') else df
            assert abs(df_val - float(dfs[i])) < 1e-9

    def test_linear_fwd_with_jax_arrays(self):
        """Test LINEAR_FWD_RATES with JAX arrays"""
        times = jnp.array([0.0, 1.0, 2.0, 5.0])
        dfs = jnp.array([1.0, 0.95, 0.90, 0.80])

        interp = Interpolator(InterpTypes.LINEAR_FWD_RATES)
        interp.fit(times, dfs)

        # Test at knot points
        for i, t in enumerate(times):
            df = interp.interpolate(float(t))
            df_val = float(df) if hasattr(df, '__float__') else df
            assert abs(df_val - float(dfs[i])) < 1e-9


class TestInterpolatorEdgeCases:
    """Test edge cases"""

    def test_single_point_pchip(self):
        """Test that single point causes fit to skip interpolator creation"""
        times = np.array([1.0])
        dfs = np.array([0.95])

        interp = Interpolator(InterpTypes.PCHIP_ZERO_RATES)
        interp.fit(times, dfs)

        # Single point: interpolator not created, but fit() should succeed
        # (This is a limitation of scipy interpolators which need >= 2 points)
        assert interp._times is not None
        assert interp._dfs is not None

    def test_two_points_pchip(self):
        """Test PCHIP with two data points"""
        times = np.array([1.0, 2.0])
        dfs = np.array([0.95, 0.90])

        interp = Interpolator(InterpTypes.PCHIP_ZERO_RATES)
        interp.fit(times, dfs)

        df_mid = interp.interpolate(1.5)
        assert 0.90 < df_mid < 0.95

    def test_zero_time_returns_one(self):
        """Test that t=0 returns df=1.0"""
        times = np.array([0.0, 1.0, 2.0])
        dfs = np.array([1.0, 0.95, 0.90])

        interp = Interpolator(InterpTypes.PCHIP_LOG_DISCOUNT)
        interp.fit(times, dfs)

        df_zero = interp.interpolate(0.0)
        assert abs(df_zero - 1.0) < 1e-12

    def test_very_small_time(self):
        """Test interpolation at very small time"""
        times = np.array([0.01, 1.0, 2.0])
        dfs = np.array([0.9999, 0.95, 0.90])

        interp = Interpolator(InterpTypes.PCHIP_ZERO_RATES)
        interp.fit(times, dfs)

        df = interp.interpolate(0.005)
        assert 0.90 < df <= 1.0

    def test_large_time_range(self):
        """Test interpolation over large time range (50 years)"""
        times = np.array([0.5, 1.0, 5.0, 10.0, 30.0, 50.0])
        dfs = np.array([0.98, 0.95, 0.85, 0.75, 0.50, 0.30])

        interp = Interpolator(InterpTypes.PCHIP_ZERO_RATES)
        interp.fit(times, dfs)

        for t in [2.5, 7.0, 20.0, 40.0]:
            df = interp.interpolate(t)
            assert 0.0 < df < 1.0


class TestInterpolatorRealisticCurves:
    """Test with realistic yield curve shapes"""

    def test_realistic_ois_curve(self):
        """Test with realistic OIS curve"""
        times = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
        rates = np.array([0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.032, 0.033])
        dfs = np.exp(-rates * times)

        interp = Interpolator(InterpTypes.PCHIP_ZERO_RATES)
        interp.fit(times, dfs)

        # Test at knot points
        for i, t in enumerate(times):
            df = interp.interpolate(t)
            assert abs(df - dfs[i]) < 1e-10

        # Test between points
        df_mid = interp.interpolate(3.0)
        assert 0.0 < df_mid < 1.0

    def test_realistic_inverted_curve(self):
        """Test with inverted yield curve"""
        times = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
        rates = np.array([0.050, 0.045, 0.040, 0.035, 0.030, 0.028])  # Inverted
        dfs = np.exp(-rates * times)

        interp = Interpolator(InterpTypes.PCHIP_LOG_DISCOUNT)
        interp.fit(times, dfs)

        # Verify monotonicity of DFs
        test_times = np.linspace(0.25, 10.0, 50)
        prev_df = 1.0
        for t in test_times:
            df = interp.interpolate(t)
            assert df <= prev_df + 1e-10, f"Not monotonic at t={t}"
            prev_df = df

    def test_realistic_steep_curve(self):
        """Test with steep yield curve"""
        times = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
        rates = np.array([0.001, 0.005, 0.015, 0.030, 0.040, 0.045, 0.048])  # Steep
        dfs = np.exp(-rates * times)

        interp = Interpolator(InterpTypes.PCHIP_ZERO_RATES)
        interp.fit(times, dfs)

        # All DFs should be positive and <= 1
        test_times = np.linspace(0.25, 30.0, 100)
        for t in test_times:
            df = interp.interpolate(t)
            assert 0.0 < df <= 1.0


class TestLegacyInterpolateFunction:
    """Test the standalone interpolate() function"""

    def test_legacy_flat_fwd(self):
        """Test legacy interpolate function with FLAT_FWD_RATES"""
        times = np.array([0.0, 1.0, 2.0, 5.0])
        dfs = np.array([1.0, 0.95, 0.90, 0.80])

        df = interpolate(1.5, times, dfs, InterpTypes.FLAT_FWD_RATES.value)
        assert 0.80 < df < 0.95

    def test_legacy_linear_zero(self):
        """Test legacy interpolate function with LINEAR_ZERO_RATES"""
        times = np.array([1.0, 2.0, 5.0])
        dfs = np.array([0.95, 0.90, 0.80])

        df = interpolate(1.5, times, dfs, InterpTypes.LINEAR_ZERO_RATES.value)
        assert 0.80 < df < 0.95

    def test_legacy_array_input(self):
        """Test legacy interpolate function with array input"""
        times = np.array([0.0, 1.0, 2.0, 5.0])
        dfs = np.array([1.0, 0.95, 0.90, 0.80])

        test_times = np.array([0.5, 1.5, 3.0])
        results = interpolate(test_times, times, dfs, InterpTypes.FLAT_FWD_RATES.value)

        assert len(results) == len(test_times)
        for df in results:
            assert 0.0 < df <= 1.0


class TestInterpolatorConsistency:
    """Test consistency across methods"""

    def test_all_scipy_methods_at_knot_points(self):
        """Test that all scipy methods reproduce knot points exactly"""
        times = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        dfs = np.array([0.98, 0.95, 0.90, 0.80, 0.70])

        methods = [
            InterpTypes.PCHIP_ZERO_RATES,
            InterpTypes.PCHIP_LOG_DISCOUNT,
            InterpTypes.NATCUBIC_ZERO_RATES,
            InterpTypes.NATCUBIC_LOG_DISCOUNT,
            InterpTypes.FINCUBIC_ZERO_RATES,
        ]

        for method in methods:
            interp = Interpolator(method)
            interp.fit(times, dfs)

            for i, t in enumerate(times):
                df = interp.interpolate(t)
                assert abs(df - dfs[i]) < 1e-9, f"Failed for {method} at knot {i}"

    def test_all_scipy_methods_monotonic(self):
        """Test that scipy methods preserve monotonicity"""
        times = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        dfs = np.array([0.98, 0.95, 0.90, 0.80, 0.70])

        methods = [
            InterpTypes.PCHIP_ZERO_RATES,
            InterpTypes.PCHIP_LOG_DISCOUNT,
        ]

        for method in methods:
            interp = Interpolator(method)
            interp.fit(times, dfs)

            test_times = np.linspace(0.5, 10.0, 50)
            prev_df = 1.0
            for t in test_times:
                df = interp.interpolate(t)
                assert df <= prev_df + 1e-10, f"Not monotonic for {method} at t={t}"
                prev_df = df
