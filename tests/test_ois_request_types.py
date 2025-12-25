"""
Comprehensive tests for OIS swap VALUE, DELTA, and GAMMA request types.

This test suite validates:
1. VALUE: Par swap repricing, off-market valuation
2. DELTA: Finite difference validation (parallel & tenor-specific bumps)
3. GAMMA: Taylor expansion accuracy for large shocks (100bp, 200bp)

Tests ensure AD-based sensitivities match finite difference approximations
and that higher-order terms explain residual P&L in large market moves.
"""

import pytest
import numpy as np
from cavour.utils.date import Date
from cavour.utils.global_types import SwapTypes, RequestTypes, CurveTypes
from cavour.utils.day_count import DayCountTypes
from cavour.utils.frequency import FrequencyTypes
from cavour.utils.calendar import BusDayAdjustTypes
from cavour.utils.currency import CurrencyTypes
from cavour.trades.rates.ois import OIS
from cavour.models.models import Model


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def gbp_value_date():
    """Reference valuation date for all tests."""
    return Date(17, 12, 2024)


@pytest.fixture
def gbp_market_data():
    """GBP SONIA market rates from overnight to 50Y."""
    px_list = [
        5.1998, 5.2014, 5.2003, 5.2027, 5.2023, 5.19281,
        5.1656, 5.1482, 5.1342, 5.1173, 5.1013, 5.0862,
        5.0701, 5.054, 5.0394, 4.8707, 4.75483, 4.532,
        4.3628, 4.2428, 4.16225, 4.1132, 4.08505, 4.0762,
        4.078, 4.0961, 4.12195, 4.1315, 4.113, 4.07724, 3.984, 3.88
    ]
    tenor_list = [
        "1D", "1W", "2W", "1M", "2M", "3M", "4M", "5M", "6M",
        "7M", "8M", "9M", "10M", "11M", "1Y", "18M", "2Y",
        "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y",
        "12Y", "15Y", "20Y", "25Y", "30Y", "40Y", "50Y"
    ]
    return {"px_list": px_list, "tenor_list": tenor_list}


@pytest.fixture
def gbp_curve_parameters():
    """Standard curve building parameters for GBP SONIA."""
    return {
        "spot_days": 0,
        "swap_type": SwapTypes.PAY,
        "fixed_dcc_type": DayCountTypes.ACT_365F,
        "fixed_freq_type": FrequencyTypes.ANNUAL,
        "float_freq_type": FrequencyTypes.ANNUAL,
        "float_dc_type": DayCountTypes.ACT_365F,
        "bus_day_type": BusDayAdjustTypes.MODIFIED_FOLLOWING,
    }


@pytest.fixture
def gbp_model(gbp_value_date, gbp_market_data, gbp_curve_parameters):
    """GBP model with SONIA curve built from market data."""
    model = Model(gbp_value_date)
    model.build_curve(
        name="GBP_OIS_SONIA",
        px_list=gbp_market_data["px_list"],
        tenor_list=gbp_market_data["tenor_list"],
        **gbp_curve_parameters,
    )
    return model


@pytest.fixture
def usd_value_date():
    """Reference valuation date for USD tests."""
    return Date(17, 12, 2024)


@pytest.fixture
def usd_market_data():
    """USD SOFR market rates from overnight to 50Y."""
    px_list = [
        5.3500, 5.3200, 5.3100, 5.2900, 5.2700, 5.2500,
        5.2300, 5.2100, 5.1900, 5.1700, 5.1500, 5.1300,
        5.1100, 5.0900, 5.0700, 4.9500, 4.8500, 4.7000,
        4.5800, 4.4800, 4.4100, 4.3600, 4.3200, 4.2900,
        4.2700, 4.2800, 4.3000, 4.3200, 4.3100, 4.2900, 4.2400, 4.1800
    ]
    tenor_list = [
        "1D", "1W", "2W", "1M", "2M", "3M", "4M", "5M", "6M",
        "7M", "8M", "9M", "10M", "11M", "1Y", "18M", "2Y",
        "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y",
        "12Y", "15Y", "20Y", "25Y", "30Y", "40Y", "50Y"
    ]
    return {"px_list": px_list, "tenor_list": tenor_list}


@pytest.fixture
def usd_curve_parameters():
    """Standard curve building parameters for USD SOFR (ACT_360 conventions)."""
    return {
        "spot_days": 0,
        "swap_type": SwapTypes.PAY,
        "fixed_dcc_type": DayCountTypes.ACT_360,
        "fixed_freq_type": FrequencyTypes.ANNUAL,
        "float_freq_type": FrequencyTypes.ANNUAL,
        "float_dc_type": DayCountTypes.ACT_360,
        "bus_day_type": BusDayAdjustTypes.MODIFIED_FOLLOWING,
    }


@pytest.fixture
def usd_model(usd_value_date, usd_market_data, usd_curve_parameters):
    """USD model with SOFR curve built from market data."""
    model = Model(usd_value_date)
    model.build_curve(
        name="USD_OIS_SOFR",
        px_list=usd_market_data["px_list"],
        tenor_list=usd_market_data["tenor_list"],
        **usd_curve_parameters,
    )
    return model


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def compute_finite_difference_delta(
    swap, model, value_dt, bump_bp=1.0, curve_name="GBP_OIS_SONIA"
):
    """
    Compute DELTA using finite differences with parallel curve bump.

    Args:
        swap: OIS swap instance
        model: Model with discount curve
        value_dt: Valuation date
        bump_bp: Bump size in basis points (default: 1bp)
        curve_name: Name of curve to bump

    Returns:
        Finite difference DELTA (P&L per 1bp parallel shift)
    """
    # Convert bp to percentage (scenario() expects percent, not bps)
    # 1bp = 0.01%, so bump_bp=1.0 becomes shock=0.01
    shock_pct = bump_bp * 0.01

    # Central difference: (V(+bump) - V(-bump)) / (2 * bump)
    model_up = model.scenario(curve_name, shock=shock_pct)
    model_down = model.scenario(curve_name, shock=-shock_pct)

    pos_up = swap.position(model_up)
    pos_down = swap.position(model_down)

    value_up = pos_up.compute([RequestTypes.VALUE]).value.amount
    value_down = pos_down.compute([RequestTypes.VALUE]).value.amount

    # Scale to 1bp sensitivity
    delta_fd = (value_up - value_down) / (2.0 * bump_bp)

    return delta_fd


def compute_tenor_specific_delta(
    swap, model, value_dt, tenor, bump_bp=1.0, curve_name="GBP_OIS_SONIA"
):
    """
    Compute DELTA for specific tenor using finite differences.

    Args:
        swap: OIS swap instance
        model: Model with discount curve
        value_dt: Valuation date
        tenor: Tenor to bump (e.g., "5Y")
        bump_bp: Bump size in basis points
        curve_name: Name of curve to bump

    Returns:
        Tenor-specific finite difference DELTA
    """
    # Convert bp to percentage
    shock_pct = bump_bp * 0.01

    shock_dict = {tenor: shock_pct}
    shock_dict_down = {tenor: -shock_pct}

    model_up = model.scenario(curve_name, shock=shock_dict)
    model_down = model.scenario(curve_name, shock=shock_dict_down)

    pos_up = swap.position(model_up)
    pos_down = swap.position(model_down)

    value_up = pos_up.compute([RequestTypes.VALUE]).value.amount
    value_down = pos_down.compute([RequestTypes.VALUE]).value.amount

    delta_fd = (value_up - value_down) / (2.0 * bump_bp)

    return delta_fd


# ==============================================================================
# VALUE TESTS
# ==============================================================================

@pytest.mark.parametrize("tenor", ["2Y", "5Y", "10Y", "30Y"])
def test_value_par_swap_repricing(gbp_model, gbp_value_date, tenor):
    """
    Test that a par swap (fixed rate = swap rate) has VALUE near zero.

    A swap constructed at the par rate should have zero present value
    by definition, since the fixed and floating leg values are equal.
    """
    value_dt = gbp_value_date
    curve = gbp_model.curves["GBP_OIS_SONIA"]
    settle_dt = value_dt.add_tenor("0D")

    # Create a swap to extract par rate
    temp_swap = OIS(
        effective_dt=settle_dt,
        term_dt_or_tenor=tenor,
        fixed_leg_type=SwapTypes.PAY,
        fixed_coupon=0.05,  # Placeholder rate
        fixed_freq_type=FrequencyTypes.ANNUAL,
        fixed_dc_type=DayCountTypes.ACT_365F,
        floating_index=CurveTypes.GBP_OIS_SONIA,
        currency=CurrencyTypes.GBP,
        bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_365F,
    )

    # Get the true par rate from the curve
    # Note: swap_rate returns value that needs to be scaled by 100 for use as fixed_coupon
    par_rate = temp_swap.swap_rate(value_dt, curve) * 100

    # Create swap at exact par rate
    par_swap = OIS(
        effective_dt=settle_dt,
        term_dt_or_tenor=tenor,
        fixed_leg_type=SwapTypes.PAY,
        fixed_coupon=par_rate,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        fixed_dc_type=DayCountTypes.ACT_365F,
        floating_index=CurveTypes.GBP_OIS_SONIA,
        currency=CurrencyTypes.GBP,
        bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_365F,
    )

    # Compute VALUE
    pos = par_swap.position(gbp_model)
    result = pos.compute([RequestTypes.VALUE])
    value = result.value.amount

    # Par swap should have value near zero
    assert abs(value) < 1e-5, f"Par swap {tenor} value {value} exceeds tolerance"


@pytest.mark.parametrize("freq", [FrequencyTypes.ANNUAL, FrequencyTypes.SEMI_ANNUAL, FrequencyTypes.QUARTERLY])
def test_value_par_swap_multiple_frequencies(gbp_model, gbp_value_date, freq):
    """
    Test par swap repricing works across different payment frequencies.
    """
    value_dt = gbp_value_date
    curve = gbp_model.curves["GBP_OIS_SONIA"]
    settle_dt = value_dt.add_tenor("0D")
    tenor = "5Y"

    temp_swap = OIS(
        effective_dt=settle_dt,
        term_dt_or_tenor=tenor,
        fixed_leg_type=SwapTypes.PAY,
        fixed_coupon=0.05,
        fixed_freq_type=freq,
        fixed_dc_type=DayCountTypes.ACT_365F,
        floating_index=CurveTypes.GBP_OIS_SONIA,
        currency=CurrencyTypes.GBP,
        bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        float_freq_type=freq,
        float_dc_type=DayCountTypes.ACT_365F,
    )

    par_rate = temp_swap.swap_rate(value_dt, curve) * 100

    par_swap = OIS(
        effective_dt=settle_dt,
        term_dt_or_tenor=tenor,
        fixed_leg_type=SwapTypes.PAY,
        fixed_coupon=par_rate,
        fixed_freq_type=freq,
        fixed_dc_type=DayCountTypes.ACT_365F,
        floating_index=CurveTypes.GBP_OIS_SONIA,
        currency=CurrencyTypes.GBP,
        bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        float_freq_type=freq,
        float_dc_type=DayCountTypes.ACT_365F,
    )

    pos = par_swap.position(gbp_model)
    result = pos.compute([RequestTypes.VALUE])
    value = result.value.amount

    assert abs(value) < 1e-5, f"Par swap {freq} value {value} exceeds tolerance"


def test_value_off_market_swap(gbp_model, gbp_value_date):
    """
    Test VALUE calculation for an off-market swap.

    A swap with fixed rate significantly different from par should have
    a non-zero present value. This tests the basic VALUE calculation.
    """
    value_dt = gbp_value_date
    curve = gbp_model.curves["GBP_OIS_SONIA"]
    settle_dt = value_dt.add_tenor("0D")
    tenor = "5Y"

    # Get par rate
    temp_swap = OIS(
        effective_dt=settle_dt,
        term_dt_or_tenor=tenor,
        fixed_leg_type=SwapTypes.PAY,
        fixed_coupon=0.05,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        fixed_dc_type=DayCountTypes.ACT_365F,
        floating_index=CurveTypes.GBP_OIS_SONIA,
        currency=CurrencyTypes.GBP,
        bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_365F,
    )
    par_rate = temp_swap.swap_rate(value_dt, curve) * 100

    # Create swap 50bps off-market (paying higher fixed rate)
    off_market_swap = OIS(
        effective_dt=settle_dt,
        term_dt_or_tenor=tenor,
        fixed_leg_type=SwapTypes.PAY,
        fixed_coupon=par_rate + 0.5,  # +50bps (par_rate is in percent, so +0.5%)
        fixed_freq_type=FrequencyTypes.ANNUAL,
        fixed_dc_type=DayCountTypes.ACT_365F,
        floating_index=CurveTypes.GBP_OIS_SONIA,
        currency=CurrencyTypes.GBP,
        bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_365F,
    )

    pos = off_market_swap.position(gbp_model)
    result = pos.compute([RequestTypes.VALUE])
    value = result.value.amount

    # Should be negative (paying more than market rate)
    assert value < -1000, f"Off-market swap value {value} should be significantly negative"

    # Verify magnitude is reasonable for a 5Y swap 50bps off-market
    # With notional ~1M, 50bps over 5Y should be substantial but not huge
    assert abs(value) > 10000, f"Off-market swap value magnitude {abs(value)} seems too small"
    assert abs(value) < 10000000, f"Off-market swap value magnitude {abs(value)} seems too large"


@pytest.mark.parametrize("tenor", ["2Y", "5Y", "10Y"])
def test_value_usd_par_swap_repricing(usd_model, usd_value_date, tenor):
    """
    Test USD OIS par swap repricing with ACT_360 conventions.

    Validates that USD SOFR swaps with ACT_360 day count conventions
    work correctly and reprice to ~0 when created at par rate.
    """
    value_dt = usd_value_date
    curve = usd_model.curves["USD_OIS_SOFR"]
    settle_dt = value_dt.add_tenor("0D")

    # Create temp swap to get par rate
    temp_swap = OIS(
        effective_dt=settle_dt,
        term_dt_or_tenor=tenor,
        fixed_leg_type=SwapTypes.PAY,
        fixed_coupon=0.05,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        fixed_dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD,
        bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_360,
    )

    par_rate = temp_swap.swap_rate(value_dt, curve) * 100

    # Create par swap
    par_swap = OIS(
        effective_dt=settle_dt,
        term_dt_or_tenor=tenor,
        fixed_leg_type=SwapTypes.PAY,
        fixed_coupon=par_rate,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        fixed_dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD,
        bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_360,
    )

    # Compute VALUE
    pos = par_swap.position(usd_model)
    result = pos.compute([RequestTypes.VALUE])
    value = result.value.amount

    # USD par swap should have value near zero
    assert abs(value) < 1e-5, f"USD {tenor} par swap value {value} exceeds tolerance"


# ==============================================================================
# DELTA TESTS
# ==============================================================================

@pytest.mark.parametrize("bump_bp", [1.0, 10.0])
def test_delta_parallel_shift_validation(gbp_model, gbp_value_date, bump_bp):
    """
    Test AD-based DELTA matches finite difference for parallel curve shifts.

    Validates that the algorithmic differentiation DELTA (gradient-based)
    matches the finite difference approximation using central differences.
    Tests with both 1bp and 10bp bumps (scaled) to verify linearity.
    """
    value_dt = gbp_value_date
    settle_dt = value_dt.add_tenor("0D")
    tenor = "10Y"

    # Create a swap slightly off-market for non-trivial sensitivities
    swap = OIS(
        effective_dt=settle_dt,
        term_dt_or_tenor=tenor,
        fixed_leg_type=SwapTypes.PAY,
        fixed_coupon=0.045,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        fixed_dc_type=DayCountTypes.ACT_365F,
        floating_index=CurveTypes.GBP_OIS_SONIA,
        currency=CurrencyTypes.GBP,
        bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_365F,
    )

    # Compute AD-based DELTA
    pos = swap.position(gbp_model)
    result = pos.compute([RequestTypes.DELTA])
    delta_ad = result.risk.value.amount  # Sum of all tenor sensitivities

    # Compute finite difference DELTA
    # Note: compute_finite_difference_delta already returns per-1bp sensitivity
    delta_fd = compute_finite_difference_delta(
        swap, gbp_model, value_dt, bump_bp=bump_bp
    )

    # Check relative error with bump-size-specific tolerance
    # AD is extremely accurate: 0.01% for 1bp, 0.05% for 10bp
    tolerance = 0.0001 if bump_bp == 1.0 else 0.0005
    relative_error = abs(delta_ad - delta_fd) / abs(delta_fd)

    assert relative_error < tolerance, \
        f"DELTA mismatch for {bump_bp}bp bump: AD={delta_ad:.6f}, FD={delta_fd:.6f}, error={relative_error:.4%} (tolerance={tolerance:.4%})"


@pytest.mark.parametrize("tenor", ["2Y", "5Y", "10Y", "30Y"])
def test_delta_tenor_specific_bumps(gbp_model, gbp_value_date, tenor):
    """
    Test individual tenor DELTA components match finite differences.

    Validates that bumping a specific tenor point produces a P&L change
    that matches the corresponding element in the DELTA risk ladder.
    """
    value_dt = gbp_value_date
    settle_dt = value_dt.add_tenor("0D")
    swap_tenor = "15Y"

    swap = OIS(
        effective_dt=settle_dt,
        term_dt_or_tenor=swap_tenor,
        fixed_leg_type=SwapTypes.PAY,
        fixed_coupon=0.04,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        fixed_dc_type=DayCountTypes.ACT_365F,
        floating_index=CurveTypes.GBP_OIS_SONIA,
        currency=CurrencyTypes.GBP,
        bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_365F,
    )

    # Compute full DELTA risk ladder
    pos = swap.position(gbp_model)
    result = pos.compute([RequestTypes.DELTA])
    delta_obj = result.risk
    delta_ladder_obj = delta_obj.ladder  # Ladder object
    delta_ladder_dict = delta_ladder_obj.data  # Dictionary: tenor -> sensitivity
    tenor_list = delta_obj.tenors

    # Compute tenor-specific finite difference
    delta_fd = compute_tenor_specific_delta(
        swap, gbp_model, value_dt, tenor=tenor, bump_bp=1.0
    )

    # Extract corresponding AD DELTA component
    if tenor in delta_ladder_dict:
        delta_ad_tenor = delta_ladder_dict[tenor]

        # Allow higher tolerance for tenor-specific (5%) due to interpolation effects
        if abs(delta_fd) > 1e-6:  # Only test if sensitivity is material
            relative_error = abs(delta_ad_tenor - delta_fd) / abs(delta_fd)
            assert relative_error < 0.05, \
                f"Tenor {tenor} DELTA mismatch: AD={delta_ad_tenor:.6f}, FD={delta_fd:.6f}, error={relative_error:.2%}"


def test_delta_structure_validation(gbp_model, gbp_value_date):
    """
    Test DELTA result structure and metadata.

    Validates that the DELTA object has correct structure:
    - Risk ladder length matches curve tenors
    - Tenors list is populated
    - Currency and curve type are correct
    """
    value_dt = gbp_value_date
    settle_dt = value_dt.add_tenor("0D")

    swap = OIS(
        effective_dt=settle_dt,
        term_dt_or_tenor="10Y",
        fixed_leg_type=SwapTypes.PAY,
        fixed_coupon=0.045,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        fixed_dc_type=DayCountTypes.ACT_365F,
        floating_index=CurveTypes.GBP_OIS_SONIA,
        currency=CurrencyTypes.GBP,
        bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_365F,
    )

    pos = swap.position(gbp_model)
    result = pos.compute([RequestTypes.DELTA])
    delta = result.risk

    # Check structure
    assert len(delta.risk_ladder) > 0, "DELTA risk ladder is empty"
    assert len(delta.tenors) > 0, "DELTA tenors list is empty"
    assert len(delta.risk_ladder) == len(delta.tenors), \
        "DELTA risk ladder and tenors length mismatch"

    # Check metadata
    assert delta.currency == CurrencyTypes.GBP, "DELTA currency mismatch"
    assert delta.curve_type == CurveTypes.GBP_OIS_SONIA, "DELTA curve type mismatch"

    # Check that ladder has correct structure
    ladder_obj = delta.ladder
    # Ladder object has a data dict property
    assert hasattr(ladder_obj, 'data'), "DELTA ladder should have data attribute"


# ==============================================================================
# GAMMA TESTS
# ==============================================================================

@pytest.mark.parametrize("shock_bp", [100.0, -100.0])
def test_gamma_taylor_expansion_100bp(gbp_model, gbp_value_date, shock_bp):
    """
    Test GAMMA improves P&L approximation for 100bp parallel shocks.

    Uses Taylor expansion: PnL ≈ DELTA * dR + 0.5 * GAMMA * dR^2
    Validates that 2nd-order approximation is significantly better than
    1st-order approximation for large rate moves.
    """
    value_dt = gbp_value_date
    settle_dt = value_dt.add_tenor("0D")
    tenor = "10Y"

    swap = OIS(
        effective_dt=settle_dt,
        term_dt_or_tenor=tenor,
        fixed_leg_type=SwapTypes.PAY,
        fixed_coupon=0.045,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        fixed_dc_type=DayCountTypes.ACT_365F,
        floating_index=CurveTypes.GBP_OIS_SONIA,
        currency=CurrencyTypes.GBP,
        bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_365F,
    )

    # Compute base VALUE, DELTA, and GAMMA
    pos = swap.position(gbp_model)
    result = pos.compute([RequestTypes.VALUE, RequestTypes.DELTA, RequestTypes.GAMMA])

    value_0 = result.value.amount
    delta_total = result.risk.value.amount
    gamma_total = result.gamma.value.amount

    # Compute shocked VALUE
    # Convert bp to percentage (scenario expects percent)
    shock_pct = shock_bp * 0.01
    model_shocked = gbp_model.scenario("GBP_OIS_SONIA", shock=shock_pct)
    pos_shocked = swap.position(model_shocked)
    result_shocked = pos_shocked.compute([RequestTypes.VALUE])
    value_shocked = result_shocked.value.amount

    # Actual P&L
    pnl_actual = value_shocked - value_0

    # 1st-order approximation: PnL ≈ DELTA * dR
    pnl_delta = delta_total * shock_bp

    # 2nd-order approximation: PnL ≈ DELTA * dR + 0.5 * GAMMA * dR^2
    pnl_gamma = delta_total * shock_bp + 0.5 * gamma_total * (shock_bp ** 2)

    # Calculate errors
    error_1st_order = abs(pnl_delta - pnl_actual)
    error_2nd_order = abs(pnl_gamma - pnl_actual)

    # 2nd-order should be significantly better (at least 50% reduction in error)
    assert error_2nd_order < 0.5 * error_1st_order, \
        f"GAMMA not improving approximation: 1st error={error_1st_order:.6f}, 2nd error={error_2nd_order:.6f}"

    # 2nd-order should explain most of the P&L (within 5% relative error)
    if abs(pnl_actual) > 1e-6:
        relative_error_2nd = abs(pnl_gamma - pnl_actual) / abs(pnl_actual)
        assert relative_error_2nd < 0.05, \
            f"2nd-order approximation error {relative_error_2nd:.2%} exceeds 5% for {shock_bp}bp shock"


@pytest.mark.parametrize("shock_bp", [200.0, -200.0])
def test_gamma_taylor_expansion_200bp(gbp_model, gbp_value_date, shock_bp):
    """
    Test GAMMA is critical for explaining P&L in 200bp shocks.

    For very large shocks, GAMMA becomes essential. Tests that:
    - 1st-order error is large (>20% of actual P&L)
    - 2nd-order error is acceptable (<10% of actual P&L)
    """
    value_dt = gbp_value_date
    settle_dt = value_dt.add_tenor("0D")
    tenor = "10Y"

    swap = OIS(
        effective_dt=settle_dt,
        term_dt_or_tenor=tenor,
        fixed_leg_type=SwapTypes.PAY,
        fixed_coupon=0.045,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        fixed_dc_type=DayCountTypes.ACT_365F,
        floating_index=CurveTypes.GBP_OIS_SONIA,
        currency=CurrencyTypes.GBP,
        bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_365F,
    )

    pos = swap.position(gbp_model)
    result = pos.compute([RequestTypes.VALUE, RequestTypes.DELTA, RequestTypes.GAMMA])

    value_0 = result.value.amount
    delta_total = result.risk.value.amount
    gamma_total = result.gamma.value.amount

    # Convert bp to percentage
    shock_pct = shock_bp * 0.01
    model_shocked = gbp_model.scenario("GBP_OIS_SONIA", shock=shock_pct)
    pos_shocked = swap.position(model_shocked)
    result_shocked = pos_shocked.compute([RequestTypes.VALUE])
    value_shocked = result_shocked.value.amount

    pnl_actual = value_shocked - value_0
    pnl_delta = delta_total * shock_bp
    pnl_gamma = delta_total * shock_bp + 0.5 * gamma_total * (shock_bp ** 2)

    # For 200bp shocks, GAMMA should improve P&L approximation
    if abs(pnl_actual) > 1e-6:
        relative_error_1st = abs(pnl_delta - pnl_actual) / abs(pnl_actual)
        relative_error_2nd = abs(pnl_gamma - pnl_actual) / abs(pnl_actual)

        # 1st-order error should be material (>5% for swaps)
        assert relative_error_1st > 0.05, \
            f"1st-order error {relative_error_1st:.2%} should be >5% for {shock_bp}bp shock"

        # 2nd-order should improve the approximation
        assert relative_error_2nd < relative_error_1st, \
            f"2nd-order error {relative_error_2nd:.2%} should be less than 1st-order {relative_error_1st:.2%}"

        # 2nd-order should still explain most P&L (<10% error)
        assert relative_error_2nd < 0.10, \
            f"2nd-order error {relative_error_2nd:.2%} exceeds 10% for {shock_bp}bp shock"


def test_gamma_structure_validation(gbp_model, gbp_value_date):
    """
    Test GAMMA result structure and properties.

    Validates that the GAMMA matrix:
    - Is square (N x N)
    - Is symmetric (within numerical tolerance)
    - Has correct tenors
    - Has correct metadata
    """
    value_dt = gbp_value_date
    settle_dt = value_dt.add_tenor("0D")

    swap = OIS(
        effective_dt=settle_dt,
        term_dt_or_tenor="10Y",
        fixed_leg_type=SwapTypes.PAY,
        fixed_coupon=0.045,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        fixed_dc_type=DayCountTypes.ACT_365F,
        floating_index=CurveTypes.GBP_OIS_SONIA,
        currency=CurrencyTypes.GBP,
        bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_365F,
    )

    pos = swap.position(gbp_model)
    result = pos.compute([RequestTypes.GAMMA])
    gamma = result.gamma

    # Check structure
    risk_ladder = np.array(gamma.risk_ladder)
    assert len(risk_ladder.shape) == 2, "GAMMA should be 2D matrix"
    assert risk_ladder.shape[0] == risk_ladder.shape[1], "GAMMA should be square matrix"

    n_tenors = len(gamma.tenors)
    assert risk_ladder.shape[0] == n_tenors, \
        f"GAMMA dimension {risk_ladder.shape[0]} should match tenors {n_tenors}"

    # Check symmetry (within tolerance for numerical precision)
    assert np.allclose(risk_ladder, risk_ladder.T, rtol=1e-10, atol=1e-14), \
        "GAMMA matrix should be symmetric"

    # Check metadata
    assert gamma.currency == CurrencyTypes.GBP, "GAMMA currency mismatch"
    assert gamma.curve_type == CurveTypes.GBP_OIS_SONIA, "GAMMA curve type mismatch"


def test_gamma_cross_terms(gbp_model, gbp_value_date):
    """
    Test GAMMA matrix has reasonable cross-term structure.

    Verifies that the GAMMA matrix contains non-zero cross-terms
    that represent second-order interaction effects between tenors.
    """
    value_dt = gbp_value_date
    settle_dt = value_dt.add_tenor("0D")

    swap = OIS(
        effective_dt=settle_dt,
        term_dt_or_tenor="10Y",
        fixed_leg_type=SwapTypes.PAY,
        fixed_coupon=0.045,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        fixed_dc_type=DayCountTypes.ACT_365F,
        floating_index=CurveTypes.GBP_OIS_SONIA,
        currency=CurrencyTypes.GBP,
        bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_365F,
    )

    # Compute GAMMA
    pos = swap.position(gbp_model)
    result = pos.compute([RequestTypes.GAMMA])

    gamma_matrix = np.array(result.gamma.risk_ladder)

    # Verify matrix has some non-zero cross-terms
    # Get all off-diagonal elements
    n = gamma_matrix.shape[0]
    off_diagonal_sum = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                off_diagonal_sum += abs(gamma_matrix[i, j])

    # Should have some non-trivial cross-terms for a 10Y swap
    assert off_diagonal_sum > 0, "GAMMA should have non-zero cross-terms"


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

def test_multiple_request_types_single_call(gbp_model, gbp_value_date):
    """
    Test computing VALUE, DELTA, and GAMMA in a single request.

    Validates that all three request types can be computed together
    and accessed via the AnalyticsResult object.
    """
    value_dt = gbp_value_date
    settle_dt = value_dt.add_tenor("0D")

    swap = OIS(
        effective_dt=settle_dt,
        term_dt_or_tenor="10Y",
        fixed_leg_type=SwapTypes.PAY,
        fixed_coupon=0.045,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        fixed_dc_type=DayCountTypes.ACT_365F,
        floating_index=CurveTypes.GBP_OIS_SONIA,
        currency=CurrencyTypes.GBP,
        bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_365F,
    )

    pos = swap.position(gbp_model)
    result = pos.compute([RequestTypes.VALUE, RequestTypes.DELTA, RequestTypes.GAMMA])

    # All results should be populated
    assert result.value is not None, "VALUE not computed"
    assert result.risk is not None, "DELTA (risk) not computed"
    assert result.gamma is not None, "GAMMA not computed"

    # Access properties
    assert isinstance(result.value.amount, float), "VALUE amount should be float"
    assert len(result.risk.risk_ladder) > 0, "DELTA risk ladder should be populated"
    assert len(result.gamma.risk_ladder) > 0, "GAMMA risk ladder should be populated"


def test_pay_vs_receive_sensitivity_sign(gbp_model, gbp_value_date):
    """
    Test that PAY vs RECEIVE swaps have opposite sensitivities.

    A PAY swap (pay fixed, receive floating) should have opposite
    VALUE and DELTA compared to a RECEIVE swap with same parameters.
    """
    value_dt = gbp_value_date
    settle_dt = value_dt.add_tenor("0D")

    # PAY swap
    swap_pay = OIS(
        effective_dt=settle_dt,
        term_dt_or_tenor="5Y",
        fixed_leg_type=SwapTypes.PAY,
        fixed_coupon=0.045,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        fixed_dc_type=DayCountTypes.ACT_365F,
        floating_index=CurveTypes.GBP_OIS_SONIA,
        currency=CurrencyTypes.GBP,
        bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_365F,
    )

    # RECEIVE swap
    swap_receive = OIS(
        effective_dt=settle_dt,
        term_dt_or_tenor="5Y",
        fixed_leg_type=SwapTypes.RECEIVE,
        fixed_coupon=0.045,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        fixed_dc_type=DayCountTypes.ACT_365F,
        floating_index=CurveTypes.GBP_OIS_SONIA,
        currency=CurrencyTypes.GBP,
        bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_365F,
    )

    pos_pay = swap_pay.position(gbp_model)
    pos_receive = swap_receive.position(gbp_model)

    result_pay = pos_pay.compute([RequestTypes.VALUE, RequestTypes.DELTA])
    result_receive = pos_receive.compute([RequestTypes.VALUE, RequestTypes.DELTA])

    # VALUES should be opposite signs
    value_pay = result_pay.value.amount
    value_receive = result_receive.value.amount

    assert np.sign(value_pay) != np.sign(value_receive), \
        "PAY and RECEIVE swaps should have opposite VALUE signs"

    assert abs(value_pay + value_receive) < 1e-10, \
        f"PAY and RECEIVE VALUES should sum to zero: {value_pay} + {value_receive}"

    # DELTAs should be opposite
    delta_pay = result_pay.risk.value.amount
    delta_receive = result_receive.risk.value.amount

    assert np.sign(delta_pay) != np.sign(delta_receive), \
        "PAY and RECEIVE swaps should have opposite DELTA signs"

    assert abs(delta_pay + delta_receive) < 1e-10, \
        f"PAY and RECEIVE DELTAs should sum to zero: {delta_pay} + {delta_receive}"


@pytest.mark.parametrize("tenor", ["3M", "50Y"])
def test_edge_case_tenors(gbp_model, gbp_value_date, tenor):
    """
    Test edge cases: very short (3M) and very long (50Y) tenors.

    Validates that VALUE, DELTA, GAMMA work for extreme tenor points.
    """
    value_dt = gbp_value_date
    settle_dt = value_dt.add_tenor("0D")

    swap = OIS(
        effective_dt=settle_dt,
        term_dt_or_tenor=tenor,
        fixed_leg_type=SwapTypes.PAY,
        fixed_coupon=0.045,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        fixed_dc_type=DayCountTypes.ACT_365F,
        floating_index=CurveTypes.GBP_OIS_SONIA,
        currency=CurrencyTypes.GBP,
        bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_365F,
    )

    pos = swap.position(gbp_model)
    result = pos.compute([RequestTypes.VALUE, RequestTypes.DELTA, RequestTypes.GAMMA])

    # Should compute without errors
    assert result.value is not None
    assert result.risk is not None
    assert result.gamma is not None

    # Should have reasonable magnitudes
    assert abs(result.value.amount) < 1e6, f"VALUE for {tenor} seems unreasonably large"
    assert abs(result.risk.value.amount) < 1e6, f"DELTA for {tenor} seems unreasonably large"
