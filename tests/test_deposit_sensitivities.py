"""
Comprehensive tests for CashDeposit VALUE, DELTA, and GAMMA request types.

This test suite validates:
1. VALUE: Deposit repricing at inception
2. DELTA: Finite difference validation (parallel & tenor-specific bumps)
3. GAMMA: Taylor expansion accuracy for large shocks (100bp, 200bp)

Tests ensure AD-based sensitivities match finite difference approximations
for cash deposits (single-cashflow instruments).
"""

import pytest
import numpy as np
from cavour.utils.date import Date
from cavour.utils.global_types import RequestTypes, CurveTypes
from cavour.utils.day_count import DayCountTypes
from cavour.utils.currency import CurrencyTypes
from cavour.trades.rates.cash_deposit import CashDeposit
from cavour.models.models import Model
from cavour.utils.global_types import SwapTypes
from cavour.utils.frequency import FrequencyTypes


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def value_date():
    """Reference valuation date for all tests."""
    return Date(15, 6, 2023)


@pytest.fixture
def usd_market_data():
    """USD SOFR market rates from ON to 10Y."""
    px_list = [5.20, 5.30, 5.40, 5.50, 4.50, 4.75]
    tenor_list = ['1M', '3M', '6M', '1Y', '2Y', '5Y']
    return {"px_list": px_list, "tenor_list": tenor_list}


@pytest.fixture
def usd_model(value_date, usd_market_data):
    """USD model with SOFR curve built from deposit market data."""
    model = Model(value_date)
    model.build_curve(
        name="USD_OIS_SOFR",
        px_list=usd_market_data["px_list"],
        tenor_list=usd_market_data["tenor_list"],
        instrument_type='DEPOSIT',  # Build curve from deposits, not OIS swaps
        spot_days=0,
        swap_type=SwapTypes.PAY,
        fixed_dcc_type=DayCountTypes.ACT_360,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_360,
        compute_gamma=True  # Enable GAMMA computation
    )
    return model


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def compute_finite_difference_delta(deposit, model, value_dt, bump_bp=1.0, curve_name="USD_OIS_SOFR"):
    """
    Compute DELTA via finite differences (central difference).

    Args:
        deposit: CashDeposit instance
        model: Model with discount curve
        value_dt: Valuation date
        bump_bp: Bump size in basis points
        curve_name: Name of curve to bump

    Returns:
        Total finite difference DELTA (per 1bp move)
    """
    # Convert bp to percentage for parallel bump
    shock_pct = bump_bp * 0.01

    model_up = model.scenario(curve_name, shock=shock_pct)
    model_down = model.scenario(curve_name, shock=-shock_pct)

    pos_up = deposit.position(model_up)
    pos_down = deposit.position(model_down)

    value_up = pos_up.compute([RequestTypes.VALUE]).value.amount
    value_down = pos_down.compute([RequestTypes.VALUE]).value.amount

    # Return per-1bp sensitivity
    delta_fd = (value_up - value_down) / (2.0 * bump_bp)

    return delta_fd


def compute_tenor_specific_delta(deposit, model, value_dt, tenor, bump_bp=1.0, curve_name="USD_OIS_SOFR"):
    """
    Compute tenor-specific DELTA via finite differences.

    Args:
        deposit: CashDeposit instance
        model: Model with discount curve
        value_dt: Valuation date
        tenor: Tenor to bump (e.g., "3M")
        bump_bp: Bump size in basis points
        curve_name: Name of curve to bump

    Returns:
        Tenor-specific finite difference DELTA
    """
    shock_pct = bump_bp * 0.01

    shock_dict = {tenor: shock_pct}
    shock_dict_down = {tenor: -shock_pct}

    model_up = model.scenario(curve_name, shock=shock_dict)
    model_down = model.scenario(curve_name, shock=shock_dict_down)

    pos_up = deposit.position(model_up)
    pos_down = deposit.position(model_down)

    value_up = pos_up.compute([RequestTypes.VALUE]).value.amount
    value_down = pos_down.compute([RequestTypes.VALUE]).value.amount

    delta_fd = (value_up - value_down) / (2.0 * bump_bp)

    return delta_fd


# ==============================================================================
# VALUE TESTS
# ==============================================================================

@pytest.mark.parametrize("tenor", ["1M", "3M", "6M", "1Y"])
def test_value_deposit_at_inception(usd_model, value_date, usd_market_data, tenor):
    """
    Test that a cash deposit has VALUE near zero at inception.

    At inception, a deposit should have PV ≈ 0 since the implied
    forward rate equals the market rate used to build the curve.
    """
    # Map tenor to market rate used in curve construction
    tenor_to_rate = {
        t: r * 0.01 for t, r in zip(usd_market_data["tenor_list"], usd_market_data["px_list"])
    }

    # Use the exact market rate for this tenor
    deposit_rate = tenor_to_rate[tenor]

    deposit = CashDeposit(
        effective_dt=value_date,
        term_dt_or_tenor=tenor,
        deposit_rate=deposit_rate,
        dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD,
        notional=1_000_000
    )

    # Compute VALUE
    pos = deposit.position(usd_model)
    result = pos.compute([RequestTypes.VALUE])
    value = result.value.amount

    # Deposit at market rate should reprice to near zero
    # Small tolerance for numerical precision
    assert abs(value) < 1.0, f"{tenor} deposit value {value} exceeds tolerance (expected ~0)"


# ==============================================================================
# DELTA TESTS
# ==============================================================================

@pytest.mark.parametrize("bump_bp", [1.0, 10.0])
def test_delta_parallel_shift_validation(usd_model, value_date, bump_bp):
    """
    Test AD-based DELTA matches finite difference for parallel curve shifts.

    Validates that the automatic differentiation DELTA (gradient-based)
    matches the finite difference approximation using central differences.
    Tests with both 1bp and 10bp bumps to verify linearity.
    """
    # Create a 6M deposit
    deposit = CashDeposit(
        effective_dt=value_date,
        term_dt_or_tenor="6M",
        deposit_rate=0.0540,  # 5.40%
        dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD,
        notional=1_000_000
    )

    # Compute AD-based DELTA
    pos = deposit.position(usd_model)
    result = pos.compute([RequestTypes.DELTA])
    delta_ad = result.risk.value.amount  # Sum of all tenor sensitivities

    # Compute finite difference DELTA
    delta_fd = compute_finite_difference_delta(
        deposit, usd_model, value_date, bump_bp=bump_bp
    )

    # Check relative error with bump-size-specific tolerance
    # Deposits should be highly accurate: 0.01% for 1bp, 0.1% for 10bp
    tolerance = 0.0001 if bump_bp == 1.0 else 0.001
    relative_error = abs(delta_ad - delta_fd) / abs(delta_fd) if abs(delta_fd) > 1e-10 else 0.0

    assert relative_error < tolerance, \
        f"DELTA mismatch for {bump_bp}bp bump: AD={delta_ad:.6f}, FD={delta_fd:.6f}, error={relative_error:.4%} (tolerance={tolerance:.4%})"


@pytest.mark.parametrize("deposit_tenor", ["1M", "3M", "6M"])
@pytest.mark.parametrize("bump_tenor", ["1M", "3M", "6M", "1Y"])
def test_delta_tenor_specific_bumps(usd_model, value_date, deposit_tenor, bump_tenor):
    """
    Test individual tenor DELTA components match finite differences.

    Validates that bumping a specific tenor point produces a P&L change
    that matches the corresponding element in the DELTA risk ladder.
    """
    deposit = CashDeposit(
        effective_dt=value_date,
        term_dt_or_tenor=deposit_tenor,
        deposit_rate=0.0520,  # 5.20%
        dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD,
        notional=1_000_000
    )

    # Compute full DELTA risk ladder
    pos = deposit.position(usd_model)
    result = pos.compute([RequestTypes.DELTA])
    delta_obj = result.risk
    delta_ladder_obj = delta_obj.ladder  # Ladder object
    delta_ladder_dict = delta_ladder_obj.data  # Dictionary: tenor -> sensitivity

    # Compute tenor-specific finite difference
    delta_fd = compute_tenor_specific_delta(
        deposit, usd_model, value_date, tenor=bump_tenor, bump_bp=1.0
    )

    # Extract corresponding AD DELTA component
    if bump_tenor in delta_ladder_dict:
        delta_ad_tenor = delta_ladder_dict[bump_tenor]

        # Allow tolerance for tenor-specific (5%) due to interpolation effects
        if abs(delta_fd) > 1e-6:  # Only test if sensitivity is material
            relative_error = abs(delta_ad_tenor - delta_fd) / abs(delta_fd)
            assert relative_error < 0.05, \
                f"Deposit {deposit_tenor}, bump {bump_tenor}: AD={delta_ad_tenor:.6f}, FD={delta_fd:.6f}, error={relative_error:.2%}"


def test_delta_structure_validation(usd_model, value_date):
    """
    Validate structure and metadata of DELTA result.

    Checks that:
    - risk attribute is a Delta object
    - Delta has ladder, tenors, currency, curve_type attributes
    - Sum of tenor sensitivities matches total DELTA
    """
    deposit = CashDeposit(
        effective_dt=value_date,
        term_dt_or_tenor="3M",
        deposit_rate=0.0530,  # 5.30%
        dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD,
        notional=1_000_000
    )

    pos = deposit.position(usd_model)
    result = pos.compute([RequestTypes.DELTA])

    # Validate structure
    assert hasattr(result, 'risk'), "Result missing 'risk' attribute"
    delta_obj = result.risk

    assert hasattr(delta_obj, 'ladder'), "Delta missing 'ladder' attribute"
    assert hasattr(delta_obj, 'tenors'), "Delta missing 'tenors' attribute"
    assert hasattr(delta_obj, 'currency'), "Delta missing 'currency' attribute"
    assert hasattr(delta_obj, 'curve_type'), "Delta missing 'curve_type' attribute"

    # Validate metadata
    assert delta_obj.currency == CurrencyTypes.USD
    assert delta_obj.curve_type == CurveTypes.USD_OIS_SOFR
    assert len(delta_obj.tenors) > 0, "Delta should have tenor list"

    # Validate total matches sum
    delta_total = delta_obj.value.amount
    delta_ladder_dict = delta_obj.ladder.data
    delta_sum = sum(delta_ladder_dict.values())

    assert abs(delta_total - delta_sum) < 1e-6, \
        f"Total DELTA {delta_total} doesn't match sum of ladder {delta_sum}"


# ==============================================================================
# GAMMA TESTS
# ==============================================================================

@pytest.mark.parametrize("shock_bp", [100.0, -100.0])
def test_gamma_taylor_expansion_100bp(usd_model, value_date, shock_bp):
    """
    Test GAMMA using Taylor expansion for 100bp shocks.

    Validates that:
    - 1st-order: PnL ≈ DELTA × dR
    - 2nd-order: PnL ≈ DELTA × dR + 0.5 × GAMMA × dR²

    The 2nd-order approximation should be significantly better.
    """
    deposit = CashDeposit(
        effective_dt=value_date,
        term_dt_or_tenor="6M",
        deposit_rate=0.0540,  # 5.40%
        dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD,
        notional=1_000_000
    )

    # Compute base VALUE, DELTA, and GAMMA
    pos = deposit.position(usd_model)
    result = pos.compute([RequestTypes.VALUE, RequestTypes.DELTA, RequestTypes.GAMMA])
    value_0 = result.value.amount
    delta_total = result.risk.value.amount
    gamma_total = result.gamma.value.amount

    # Compute shocked VALUE
    model_shocked = usd_model.scenario("USD_OIS_SOFR", shock=shock_bp * 0.01)
    value_shocked = deposit.position(model_shocked).compute([RequestTypes.VALUE]).value.amount
    pnl_actual = value_shocked - value_0

    # 1st-order approximation: PnL ≈ DELTA × dR
    pnl_delta = delta_total * shock_bp

    # 2nd-order approximation: PnL ≈ DELTA × dR + 0.5 × GAMMA × dR²
    pnl_gamma = delta_total * shock_bp + 0.5 * gamma_total * (shock_bp ** 2)

    # Calculate errors
    error_1st = abs(pnl_delta - pnl_actual)
    error_2nd = abs(pnl_gamma - pnl_actual)

    # 2nd-order should be significantly better
    assert error_2nd < 0.5 * error_1st, \
        f"2nd-order error {error_2nd:.2f} not better than 1st-order {error_1st:.2f}"

    # 2nd-order should explain most P&L (within 5% for 100bp shock)
    relative_error_2nd = abs(pnl_gamma - pnl_actual) / abs(pnl_actual) if abs(pnl_actual) > 1e-6 else 0.0
    assert relative_error_2nd < 0.05, \
        f"2nd-order relative error {relative_error_2nd:.2%} exceeds 5% tolerance"


@pytest.mark.parametrize("shock_bp", [200.0, -200.0])
def test_gamma_taylor_expansion_200bp(usd_model, value_date, shock_bp):
    """
    Test GAMMA using Taylor expansion for 200bp shocks.

    For larger shocks, 2nd-order approximation is even more critical.
    """
    deposit = CashDeposit(
        effective_dt=value_date,
        term_dt_or_tenor="1Y",
        deposit_rate=0.0550,  # 5.50%
        dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD,
        notional=1_000_000
    )

    # Compute base VALUE, DELTA, and GAMMA
    pos = deposit.position(usd_model)
    result = pos.compute([RequestTypes.VALUE, RequestTypes.DELTA, RequestTypes.GAMMA])
    value_0 = result.value.amount
    delta_total = result.risk.value.amount
    gamma_total = result.gamma.value.amount

    # Compute shocked VALUE
    model_shocked = usd_model.scenario("USD_OIS_SOFR", shock=shock_bp * 0.01)
    value_shocked = deposit.position(model_shocked).compute([RequestTypes.VALUE]).value.amount
    pnl_actual = value_shocked - value_0

    # 1st-order and 2nd-order approximations
    pnl_delta = delta_total * shock_bp
    pnl_gamma = delta_total * shock_bp + 0.5 * gamma_total * (shock_bp ** 2)

    error_1st = abs(pnl_delta - pnl_actual)
    error_2nd = abs(pnl_gamma - pnl_actual)

    # 2nd-order must be better
    assert error_2nd < 0.5 * error_1st, \
        f"2nd-order error {error_2nd:.2f} not better than 1st-order {error_1st:.2f}"

    # Allow 10% tolerance for 200bp shock (higher-order terms matter)
    relative_error_2nd = abs(pnl_gamma - pnl_actual) / abs(pnl_actual) if abs(pnl_actual) > 1e-6 else 0.0
    assert relative_error_2nd < 0.10, \
        f"2nd-order relative error {relative_error_2nd:.2%} exceeds 10% tolerance"


def test_gamma_structure_validation(usd_model, value_date):
    """
    Validate structure and metadata of GAMMA result.

    Checks that:
    - gamma attribute is a Gamma object
    - Gamma has risk_ladder (matrix), tenors, currency, curve_type
    - Matrix is symmetric
    - Dimensions match number of tenors
    """
    deposit = CashDeposit(
        effective_dt=value_date,
        term_dt_or_tenor="6M",
        deposit_rate=0.0540,  # 5.40%
        dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD,
        notional=1_000_000
    )

    pos = deposit.position(usd_model)
    result = pos.compute([RequestTypes.GAMMA])

    # Validate structure
    assert hasattr(result, 'gamma'), "Result missing 'gamma' attribute"
    gamma_obj = result.gamma

    assert hasattr(gamma_obj, 'risk_ladder'), "Gamma missing 'risk_ladder' attribute"
    assert hasattr(gamma_obj, 'tenors'), "Gamma missing 'tenors' attribute"
    assert hasattr(gamma_obj, 'currency'), "Gamma missing 'currency' attribute"
    assert hasattr(gamma_obj, 'curve_type'), "Gamma missing 'curve_type' attribute"

    # Validate metadata
    assert gamma_obj.currency == CurrencyTypes.USD
    assert gamma_obj.curve_type == CurveTypes.USD_OIS_SOFR

    # Validate matrix structure
    gamma_matrix = np.array(gamma_obj.risk_ladder)
    n_tenors = len(gamma_obj.tenors)

    assert gamma_matrix.shape == (n_tenors, n_tenors), \
        f"Gamma matrix shape {gamma_matrix.shape} doesn't match tenor count {n_tenors}"

    # Check symmetry (Hessian should be symmetric)
    symmetry_error = np.max(np.abs(gamma_matrix - gamma_matrix.T))
    assert symmetry_error < 1e-6, f"Gamma matrix not symmetric: max error {symmetry_error}"


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

def test_multiple_request_types_single_call(usd_model, value_date):
    """
    Test that VALUE, DELTA, and GAMMA can be computed in a single call.

    Validates that requesting multiple analytics together works correctly
    and returns consistent results.
    """
    deposit = CashDeposit(
        effective_dt=value_date,
        term_dt_or_tenor="3M",
        deposit_rate=0.0530,  # 5.30%
        dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD,
        notional=1_000_000
    )

    # Compute all three together
    pos = deposit.position(usd_model)
    result = pos.compute([RequestTypes.VALUE, RequestTypes.DELTA, RequestTypes.GAMMA])

    # All should be present
    assert result.value is not None, "VALUE missing in combined request"
    assert result.risk is not None, "DELTA missing in combined request"
    assert result.gamma is not None, "GAMMA missing in combined request"

    # Verify they're of correct types
    assert hasattr(result.value, 'amount'), "VALUE missing 'amount' attribute"
    assert hasattr(result.risk, 'value'), "DELTA missing 'value' attribute"
    assert hasattr(result.gamma, 'value'), "GAMMA missing 'value' attribute"
