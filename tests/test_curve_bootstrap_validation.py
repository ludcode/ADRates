"""
Comprehensive tests for curve bootstrap validation.

Tests that bootstrapped curves satisfy fundamental financial constraints:
- Discount factors are monotonically decreasing
- Discount factors are in valid range (0, 1]
- Forward rates are within reasonable bounds
- Zero rates are monotonically increasing (for normal curves)
- Curve smoothness (no sudden jumps)
- Extrapolation behavior

Tests focus on GBP SONIA curve with realistic market data.

NOTE: Some tenor combinations trigger an IndexError in the OIS curve
bootstrap logic (ois_curve.py:187). This is a known library issue.
"""

import pytest
import numpy as np
from cavour.utils.date import Date
from cavour.utils.global_types import SwapTypes, CurveTypes
from cavour.utils.day_count import DayCountTypes
from cavour.utils.frequency import FrequencyTypes
from cavour.utils.calendar import BusDayAdjustTypes
from cavour.market.curves.interpolator import InterpTypes
from cavour.models.models import Model


@pytest.fixture
def gbp_sonia_curve():
    """Build a GBP SONIA curve for validation"""
    value_date = Date(30, 4, 2024)

    px_list = [5.1998, 5.2014, 5.2003, 5.2027, 5.2023, 5.19281,
               5.1656, 5.1482, 5.1342, 5.1173, 5.1013, 5.0862,
               5.0701, 5.054, 5.0394, 4.8707, 4.75483, 4.532,
               4.3628, 4.2428, 4.16225, 4.1132, 4.08505, 4.0762,
               4.078, 4.0961, 4.12195, 4.1315, 4.113, 4.07724, 3.984, 3.88]

    tenor_list = ["1D", "1W", "2W", "1M", "2M", "3M", "4M", "5M", "6M",
                  "7M", "8M", "9M", "10M", "11M", "1Y", "18M", "2Y",
                  "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y",
                  "12Y", "15Y", "20Y", "25Y", "30Y", "40Y", "50Y"]

    model = Model(value_date)
    model.build_curve(
        name="GBP_OIS_SONIA",
        px_list=px_list,
        tenor_list=tenor_list,
        spot_days=0,
        swap_type=SwapTypes.PAY,
        fixed_dcc_type=DayCountTypes.ACT_365F,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_365F,
        bus_day_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        interp_type=InterpTypes.LINEAR_ZERO_RATES
    )

    return model.curves.GBP_OIS_SONIA


class TestDiscountFactorMonotonicity:
    """Test that discount factors are monotonically decreasing"""

    def test_gbp_dfs_monotonic_decreasing(self, gbp_sonia_curve):
        """Test GBP SONIA discount factors are monotonically decreasing"""
        times = gbp_sonia_curve._times
        dfs = gbp_sonia_curve._dfs

        # Check that each DF is <= previous DF
        for i in range(1, len(dfs)):
            assert dfs[i] <= dfs[i-1], \
                f"DF not monotonic at index {i}: {dfs[i]} > {dfs[i-1]}"

    def test_gbp_dfs_strictly_decreasing(self, gbp_sonia_curve):
        """Test GBP SONIA discount factors are strictly decreasing (no duplicates)"""
        dfs = gbp_sonia_curve._dfs

        # Check that each DF is actually < previous DF (not just <=)
        for i in range(1, len(dfs)):
            assert dfs[i] < dfs[i-1], \
                f"DF not strictly decreasing at index {i}: {dfs[i]} >= {dfs[i-1]}"

    def test_interpolated_dfs_monotonic(self, gbp_sonia_curve):
        """Test that interpolated DFs between pillars are also monotonic"""
        # Create dense grid of times starting from first pillar
        min_time = float(gbp_sonia_curve._times[0])
        max_time = float(gbp_sonia_curve._times[-1])
        test_times = np.linspace(min_time, max_time, 500)

        prev_df = float(gbp_sonia_curve.df_ad(min_time))
        for t in test_times[1:]:
            df = float(gbp_sonia_curve.df_ad(t))
            assert df <= prev_df + 1e-10, \
                f"Interpolated DF not monotonic at t={t}: {df} > {prev_df}"
            prev_df = df


class TestDiscountFactorBounds:
    """Test that discount factors are in valid range (0, 1]"""

    def test_gbp_dfs_in_valid_range(self, gbp_sonia_curve):
        """Test GBP SONIA DFs are in (0, 1]"""
        dfs = gbp_sonia_curve._dfs

        for i, df in enumerate(dfs):
            assert 0.0 < df <= 1.0, \
                f"DF out of range at index {i}: {df}"

    def test_first_pillar_df_near_one(self, gbp_sonia_curve):
        """Test that first pillar DF is close to 1.0"""
        first_df = gbp_sonia_curve._dfs[0]
        assert 0.99 < first_df <= 1.0, f"First pillar DF {first_df} seems unreasonable"


class TestForwardRateBounds:
    """Test that forward rates are within reasonable economic bounds"""

    def test_gbp_forward_rates_reasonable(self, gbp_sonia_curve):
        """Test GBP SONIA forward rates are in reasonable range"""
        times = gbp_sonia_curve._times
        dfs = gbp_sonia_curve._dfs

        # Calculate instantaneous forward rates
        for i in range(1, len(times)):
            dt = times[i] - times[i-1]
            fwd = -(np.log(dfs[i]) - np.log(dfs[i-1])) / dt

            # Forward rates should be between -5% and +20% (very generous bounds)
            assert -0.05 < fwd < 0.20, \
                f"Forward rate at time {times[i]} is {fwd*100:.2f}% - unreasonable"

    def test_forward_rates_positive_normal_curve(self, gbp_sonia_curve):
        """Test that forward rates are positive for normal upward-sloping curve"""
        times = gbp_sonia_curve._times
        dfs = gbp_sonia_curve._dfs

        # Most forward rates should be positive
        positive_count = 0
        total_count = 0

        for i in range(1, len(times)):
            dt = times[i] - times[i-1]
            fwd = -(np.log(dfs[i]) - np.log(dfs[i-1])) / dt
            total_count += 1
            if fwd > 0:
                positive_count += 1

        # At least 80% should be positive
        assert positive_count / total_count > 0.8, \
            f"Only {positive_count}/{total_count} forward rates are positive"


class TestZeroRateBehavior:
    """Test zero rate properties"""

    def test_zero_rates_calculable(self, gbp_sonia_curve):
        """Test that zero rates can be calculated from DFs"""
        times = gbp_sonia_curve._times
        dfs = gbp_sonia_curve._dfs

        for i in range(len(times)):
            if times[i] > 0:
                zero_rate = -np.log(dfs[i]) / times[i]
                # Zero rate should be reasonable
                assert -0.05 < zero_rate < 0.20, \
                    f"Zero rate at time {times[i]} is {zero_rate*100:.2f}%"

    def test_zero_rates_no_extreme_inversions(self, gbp_sonia_curve):
        """Test that zero rates don't have extreme inversions"""
        times = gbp_sonia_curve._times
        dfs = gbp_sonia_curve._dfs

        zero_rates = []
        for i in range(len(times)):
            if times[i] > 0:
                zero_rates.append(-np.log(dfs[i]) / times[i])

        # Check that zero rate changes are not too extreme
        for i in range(1, len(zero_rates)):
            abs_change = abs(zero_rates[i] - zero_rates[i-1])
            # Zero rate should not jump more than 2% (200 bps)
            assert abs_change < 0.02, \
                f"Extreme zero rate change at index {i}: {abs_change*10000:.0f} bps"


class TestCurveSmoothness:
    """Test that curves are smooth (no sudden jumps)"""

    def test_gbp_df_smoothness(self, gbp_sonia_curve):
        """Test that GBP DFs don't have sudden jumps"""
        times = gbp_sonia_curve._times
        dfs = gbp_sonia_curve._dfs

        for i in range(1, len(dfs)):
            pct_change = abs((dfs[i] - dfs[i-1]) / dfs[i-1])
            # No single pillar should have more than 5% change
            assert pct_change < 0.05, \
                f"Large DF jump at index {i}: {pct_change*100:.2f}%"

    def test_interpolated_smoothness(self, gbp_sonia_curve):
        """Test that interpolated values are smooth"""
        min_time = float(gbp_sonia_curve._times[0])
        max_time = float(gbp_sonia_curve._times[-1])
        test_times = np.linspace(min_time, max_time, 1000)

        dfs = [float(gbp_sonia_curve.df_ad(t)) for t in test_times]

        # Check that adjacent interpolated values don't jump
        for i in range(1, len(dfs)):
            pct_change = abs((dfs[i] - dfs[i-1]) / dfs[i-1])
            # Allow up to 0.5% change between adjacent points (generous for 1000 points over 50Y)
            assert pct_change < 0.005, \
                f"Large jump in interpolated DF at index {i}: {pct_change*100:.2f}%"

    def test_forward_rate_smoothness(self, gbp_sonia_curve):
        """Test that forward rates don't have extreme spikes"""
        times = gbp_sonia_curve._times
        dfs = gbp_sonia_curve._dfs

        fwd_rates = []
        for i in range(1, len(times)):
            dt = times[i] - times[i-1]
            fwd = -(np.log(dfs[i]) - np.log(dfs[i-1])) / dt
            fwd_rates.append(fwd)

        # Check for extreme changes in forward rates
        for i in range(1, len(fwd_rates)):
            abs_change = abs(fwd_rates[i] - fwd_rates[i-1])
            # Forward rate shouldn't jump more than 2% (200 bps)
            assert abs_change < 0.02, \
                f"Large forward rate jump at index {i}: {abs_change*10000:.0f} bps"


class TestCurveExtrapolation:
    """Test curve behavior beyond last pillar"""

    def test_extrapolation_beyond_last_pillar(self, gbp_sonia_curve):
        """Test that curve can extrapolate beyond last pillar"""
        max_time = float(gbp_sonia_curve._times[-1])

        # Test at 1.5x last pillar
        t_extrap = max_time * 1.5
        df = float(gbp_sonia_curve.df_ad(t_extrap))

        # Should still return valid DF
        assert 0.0 < df < 1.0, f"Extrapolated DF {df} out of range"

        # Should be less than last pillar DF
        df_last = gbp_sonia_curve._dfs[-1]
        assert df < df_last, "Extrapolated DF not monotonic"

    def test_very_long_maturity_extrapolation(self, gbp_sonia_curve):
        """Test extrapolation to very long maturities (100 years)"""
        df_100y = float(gbp_sonia_curve.df_ad(100.0))

        # Should still be positive
        assert df_100y > 0, f"100Y DF is {df_100y}, should be positive"

        # Should be significantly discounted
        assert df_100y < 0.5, f"100Y DF is {df_100y}, seems too high"
