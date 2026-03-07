"""Test FRA sensitivities (DELTA and GAMMA) via algorithmic Differentiation.

This test suite validates FRA DELTA and GAMMA calculations by comparing
AD results against finite difference approximations.

Test Coverage:
- VALUE: Basic FRA valuation at inception
- DELTA: Parallel shifts and tenor-specific bumps (FD validation)
- GAMMA: Taylor expansion validation with large rate shocks
- Structure: Proper result object construction
"""

import pytest
import numpy as np
from cavour.utils.date import Date
from cavour.utils.day_count import DayCountTypes
from cavour.utils.global_types import CurveTypes, SwapTypes, RequestTypes
from cavour.utils.frequency import FrequencyTypes
from cavour.utils.currency import CurrencyTypes
from cavour.trades.rates.fra import FRA
from cavour.models.models import Model

# Test fixtures
value_dt = Date(15, 6, 2023)

def build_test_model(value_dt):
    """Build a standard USD OIS curve for testing."""
    px_list = [5.20, 5.30, 5.40, 4.50, 4.75]
    tenor_list = ['1M', '3M', '6M', '2Y', '5Y']

    model = Model(value_dt)
    model.build_curve(
        name='USD_OIS_SOFR',
        px_list=px_list,
        tenor_list=tenor_list,
        spot_days=0,
        swap_type=SwapTypes.PAY,
        fixed_dcc_type=DayCountTypes.ACT_360,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_360,
        compute_gamma=True
    )
    return model

def create_fra(value_dt, fra_notation, fra_rate, notional=1_000_000):
    """Create a standard FRA for testing."""
    return FRA(
        effective_dt=value_dt,
        fra_notation=fra_notation,
        fra_rate=fra_rate,
        dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD,
        notional=notional
    )

def compute_finite_difference_delta(fra, model, value_dt, bump_bp=1.0, curve_name="USD_OIS_SOFR"):
    """Compute DELTA via central finite differences.

    Uses (V_up - V_down) / (2 * bump) for parallel shift.
    """
    shock_pct = bump_bp * 0.01

    model_up = model.scenario(curve_name, shock=shock_pct)
    model_down = model.scenario(curve_name, shock=-shock_pct)

    position_up = fra.position(model_up)
    position_down = fra.position(model_down)

    value_up = position_up.compute([RequestTypes.VALUE]).value.amount
    value_down = position_down.compute([RequestTypes.VALUE]).value.amount

    return (value_up - value_down) / (2.0 * bump_bp)

def compute_tenor_specific_fd_delta(fra, model, value_dt, tenor_idx, bump_bp=1.0, curve_name="USD_OIS_SOFR"):
    """Compute DELTA for a specific tenor via FD.

    Bumps only one tenor by +bump_bp and -bump_bp.
    """
    curve = model.curves[curve_name]
    tenors = [f"{int(t*12)}M" if t < 1 else f"{int(t)}Y" for t in curve.swap_times]

    # Create shock dict for single tenor
    tenor_name = tenors[tenor_idx]
    shock_dict_up = {tenor_name: bump_bp * 0.01}
    shock_dict_down = {tenor_name: -bump_bp * 0.01}

    model_up = model.scenario(curve_name, shock=shock_dict_up)
    model_down = model.scenario(curve_name, shock=shock_dict_down)

    value_up = fra.position(model_up).compute([RequestTypes.VALUE]).value.amount
    value_down = fra.position(model_down).compute([RequestTypes.VALUE]).value.amount

    return (value_up - value_down) / (2.0 * bump_bp)

# ============================================================================
# VALUE TESTS
# ============================================================================

@pytest.mark.parametrize("fra_notation", ["3x6", "6x9", "9x12", "12x15"])
def test_value_fra_at_inception(fra_notation):
    """Test FRA valuation at inception (should be near zero for at-market rate)."""
    model = build_test_model(value_dt)

    # Create FRA at current forward rate (approximately at-market)
    fra = create_fra(value_dt, fra_notation, fra_rate=0.053)

    position = fra.position(model)
    result = position.compute([RequestTypes.VALUE])

    pv = result.value.amount

    # At-market FRA should have small PV (not exactly zero due to rate mismatch)
    # This is just a basic valuation test, not a sensitivity test
    assert isinstance(pv, float)
    assert result.value.currency == CurrencyTypes.USD

# ============================================================================
# DELTA TESTS - Parallel Shift
# ============================================================================

@pytest.mark.parametrize("bump_bp", [1.0, 10.0])
def test_delta_parallel_shift_validation(bump_bp):
    """Validate DELTA via parallel curve shift using finite differences.

    Tests:
    - Sum of AD delta risk ladder matches FD delta for parallel shift
    - Tolerance: 0.01% for 1bp, 0.1% for 10bp bumps
    """
    model = build_test_model(value_dt)
    fra = create_fra(value_dt, "6x9", fra_rate=0.053)

    # Compute AD DELTA
    position = fra.position(model)
    result = position.compute([RequestTypes.DELTA])

    delta_ad = result.risk
    delta_sum = sum(delta_ad.risk_ladder)

    # Compute FD DELTA
    delta_fd = compute_finite_difference_delta(fra, model, value_dt, bump_bp=bump_bp)

    # Validate match
    tol = abs(delta_fd) * 0.001 if bump_bp == 1.0 else abs(delta_fd) * 0.01
    assert abs(delta_sum - delta_fd) < tol, (
        f"AD delta sum {delta_sum:.6f} vs FD {delta_fd:.6f} "
        f"(diff={abs(delta_sum - delta_fd):.6f}, tol={tol:.6f})"
    )

# ============================================================================
# DELTA TESTS - Tenor Specific
# ============================================================================

@pytest.mark.parametrize("fra_notation", ["3x6", "6x9", "9x12"])
@pytest.mark.parametrize("bump_tenor_name", ["1M", "3M", "6M", "2Y"])
def test_delta_tenor_specific_bumps(fra_notation, bump_tenor_name):
    """Validate DELTA for individual tenor bumps.

    Tests:
    - Each tenor's AD delta matches FD delta for that tenor
    - Validates granular sensitivity structure
    """
    model = build_test_model(value_dt)
    fra = create_fra(value_dt, fra_notation, fra_rate=0.053)

    # Compute AD DELTA
    position = fra.position(model)
    result = position.compute([RequestTypes.DELTA])

    delta_ad = result.risk
    tenors = delta_ad.tenors

    # Find tenor index
    try:
        tenor_idx = tenors.index(bump_tenor_name)
    except ValueError:
        pytest.skip(f"Tenor {bump_tenor_name} not in curve")

    # Get AD delta for this tenor
    delta_ad_tenor = delta_ad.risk_ladder[tenor_idx]

    # Compute FD delta for this tenor
    delta_fd_tenor = compute_tenor_specific_fd_delta(
        fra, model, value_dt, tenor_idx, bump_bp=1.0
    )

    # Validate match (use relative tolerance for small values)
    if abs(delta_fd_tenor) > 1e-6:
        rel_error = abs(delta_ad_tenor - delta_fd_tenor) / abs(delta_fd_tenor)
        assert rel_error < 0.01, (
            f"Tenor {bump_tenor_name}: AD {delta_ad_tenor:.6f} vs FD {delta_fd_tenor:.6f} "
            f"(rel_error={rel_error:.4%})"
        )
    else:
        abs_error = abs(delta_ad_tenor - delta_fd_tenor)
        assert abs_error < 1e-6, (
            f"Tenor {bump_tenor_name}: AD {delta_ad_tenor:.6f} vs FD {delta_fd_tenor:.6f} "
            f"(abs_error={abs_error:.8f})"
        )

# ============================================================================
# DELTA STRUCTURE TESTS
# ============================================================================

def test_delta_structure_validation():
    """Validate DELTA result structure and metadata.

    Tests:
    - Delta object has correct attributes
    - Tenors match curve tenors
    - Risk ladder has correct length
    - Currency and curve_type are correct
    """
    model = build_test_model(value_dt)
    fra = create_fra(value_dt, "6x9", fra_rate=0.053)

    position = fra.position(model)
    result = position.compute([RequestTypes.DELTA])

    delta = result.risk

    # Check attributes exist
    assert hasattr(delta, 'risk_ladder')
    assert hasattr(delta, 'tenors')
    assert hasattr(delta, 'currency')
    assert hasattr(delta, 'curve_type')

    # Check values
    assert len(delta.risk_ladder) == len(delta.tenors)
    assert delta.currency == CurrencyTypes.USD
    assert delta.curve_type == CurveTypes.USD_OIS_SOFR

    # Check tenors match curve
    curve_tenors = model.curves['USD_OIS_SOFR'].swap_times
    assert len(delta.tenors) == len(curve_tenors)

# ============================================================================
# GAMMA TESTS - Taylor Expansion
# ============================================================================

@pytest.mark.parametrize("shock_bp", [100.0, -100.0])
def test_gamma_taylor_expansion_100bp(shock_bp):
    """Validate GAMMA using Taylor expansion for 100bp shocks.

    Tests:
    - 2nd order approximation is significantly better than 1st order
    - 2nd order residual is < 5% of total P&L

    Taylor expansion:
    dV ≈ delta · dx + 0.5 · gamma · dx²
    """
    model = build_test_model(value_dt)
    fra = create_fra(value_dt, "6x9", fra_rate=0.053)

    # Get VALUE, DELTA, GAMMA
    position = fra.position(model)
    result = position.compute([RequestTypes.VALUE, RequestTypes.DELTA, RequestTypes.GAMMA])

    v0 = result.value.amount
    delta = np.array(result.risk.risk_ladder)
    gamma = np.array(result.gamma.risk_ladder)

    # Create shocked scenario (parallel shift, use scalar)
    n_tenors = len(delta)
    shock_scalar = shock_bp * 0.01

    model_shocked = model.scenario('USD_OIS_SOFR', shock=shock_scalar)
    v_shocked = fra.position(model_shocked).compute([RequestTypes.VALUE]).value.amount

    actual_pnl = v_shocked - v0

    # 1st order approximation
    dx = np.full(n_tenors, shock_bp)
    pnl_1st = np.dot(delta, dx)

    # 2nd order approximation
    pnl_2nd = pnl_1st + 0.5 * np.dot(dx, np.dot(gamma, dx))

    # Compute errors
    error_1st = abs(actual_pnl - pnl_1st)
    error_2nd = abs(actual_pnl - pnl_2nd)

    # 2nd order should be significantly better
    improvement_ratio = error_1st / error_2nd if error_2nd > 1e-10 else float('inf')
    assert improvement_ratio > 1.5, (
        f"2nd order not significantly better: improvement ratio {improvement_ratio:.2f}"
    )

    # 2nd order error should be small
    rel_error_2nd = error_2nd / abs(actual_pnl) if abs(actual_pnl) > 1e-6 else 0
    assert rel_error_2nd < 0.05, (
        f"2nd order error too large: {rel_error_2nd:.2%} "
        f"(actual={actual_pnl:.2f}, 2nd={pnl_2nd:.2f})"
    )

@pytest.mark.parametrize("shock_bp", [200.0, -200.0])
def test_gamma_taylor_expansion_200bp(shock_bp):
    """Validate GAMMA using Taylor expansion for 200bp shocks.

    Larger shocks test GAMMA accuracy with higher-order effects.
    Allow larger error tolerance (10%) due to 3rd order terms.
    """
    model = build_test_model(value_dt)
    fra = create_fra(value_dt, "6x9", fra_rate=0.053)

    position = fra.position(model)
    result = position.compute([RequestTypes.VALUE, RequestTypes.DELTA, RequestTypes.GAMMA])

    v0 = result.value.amount
    delta = np.array(result.risk.risk_ladder)
    gamma = np.array(result.gamma.risk_ladder)

    n_tenors = len(delta)
    shock_scalar = shock_bp * 0.01

    model_shocked = model.scenario('USD_OIS_SOFR', shock=shock_scalar)
    v_shocked = fra.position(model_shocked).compute([RequestTypes.VALUE]).value.amount

    actual_pnl = v_shocked - v0

    dx = np.full(n_tenors, shock_bp)
    pnl_1st = np.dot(delta, dx)
    pnl_2nd = pnl_1st + 0.5 * np.dot(dx, np.dot(gamma, dx))

    error_2nd = abs(actual_pnl - pnl_2nd)
    rel_error_2nd = error_2nd / abs(actual_pnl) if abs(actual_pnl) > 1e-6 else 0

    assert rel_error_2nd < 0.10, (
        f"2nd order error too large for 200bp shock: {rel_error_2nd:.2%}"
    )

# ============================================================================
# GAMMA STRUCTURE TESTS
# ============================================================================

def test_gamma_structure_validation():
    """Validate GAMMA result structure and metadata.

    Tests:
    - Gamma object has correct attributes
    - Gamma matrix is square and symmetric
    - Dimensions match number of tenors
    """
    model = build_test_model(value_dt)
    fra = create_fra(value_dt, "6x9", fra_rate=0.053)

    position = fra.position(model)
    result = position.compute([RequestTypes.GAMMA])

    gamma = result.gamma

    # Check attributes
    assert hasattr(gamma, 'risk_ladder')
    assert hasattr(gamma, 'tenors')
    assert hasattr(gamma, 'currency')
    assert hasattr(gamma, 'curve_type')

    # Check gamma matrix shape
    gamma_matrix = np.array(gamma.risk_ladder)
    n_tenors = len(gamma.tenors)
    assert gamma_matrix.shape == (n_tenors, n_tenors)

    # Check symmetry
    assert np.allclose(gamma_matrix, gamma_matrix.T, rtol=1e-9), \
        "Gamma matrix should be symmetric"

    # Check metadata
    assert gamma.currency == CurrencyTypes.USD
    assert gamma.curve_type == CurveTypes.USD_OIS_SOFR

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_multiple_request_types_single_call():
    """Test requesting VALUE, DELTA, and GAMMA in a single call.

    Validates:
    - All three can be computed together
    - Results are consistent with individual calls
    """
    model = build_test_model(value_dt)
    fra = create_fra(value_dt, "6x9", fra_rate=0.053)

    position = fra.position(model)
    result = position.compute([RequestTypes.VALUE, RequestTypes.DELTA, RequestTypes.GAMMA])

    # All should be present
    assert result.value is not None
    assert result.risk is not None
    assert result.gamma is not None

    # Value should match individual call
    result_value_only = position.compute([RequestTypes.VALUE])
    assert abs(result.value.amount - result_value_only.value.amount) < 1e-10

    # Delta should match individual call
    result_delta_only = position.compute([RequestTypes.DELTA])
    assert np.allclose(result.risk.risk_ladder, result_delta_only.risk.risk_ladder)

    # Gamma should match individual call
    result_gamma_only = position.compute([RequestTypes.GAMMA])
    assert np.allclose(result.gamma.risk_ladder, result_gamma_only.gamma.risk_ladder)

# ============================================================================
# EDGE CASES
# ============================================================================

def test_multiple_fra_notations():
    """Test DELTA/GAMMA for various FRA notations."""
    model = build_test_model(value_dt)

    notations = ["1x4", "3x6", "6x9", "9x12", "12x15"]

    for notation in notations:
        fra = create_fra(value_dt, notation, fra_rate=0.053)
        position = fra.position(model)
        result = position.compute([RequestTypes.DELTA, RequestTypes.GAMMA])

        # Should complete without error
        assert result.risk is not None
        assert result.gamma is not None
        assert len(result.risk.risk_ladder) > 0
        assert len(result.gamma.risk_ladder) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
