"""
Comprehensive Automatic Differentiation (AD) Test for DELTA and GAMMA

Tests JAX-based AD computation of first-order (DELTA) and second-order (GAMMA)
sensitivities for:
- Cash Deposits (overnight, 1M, 3M)
- IR Futures (IMM quarterly)
- OIS Swaps (2Y, 5Y, 10Y)

Validates AD results against finite differences and Taylor expansion.

Target tolerance: SWAP_TOL = 1e-10 from ois_curve.py
"""

import sys
sys.path.insert(0, 'C:\\Projects\\Cavour')

import pytest
import numpy as np
from cavour.utils.date import Date
from cavour.utils.day_count import DayCountTypes
from cavour.utils.global_types import CurveTypes, SwapTypes, FutureContractTypes, RequestTypes
from cavour.utils.currency import CurrencyTypes
from cavour.utils.frequency import FrequencyTypes
from cavour.trades.rates.cash_deposit import CashDeposit
from cavour.trades.rates.ir_future import IRFuture
from cavour.trades.rates.ois import OIS
from cavour.models.models import Model


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def value_date():
    """Test value date."""
    return Date(26, 2, 2026)


@pytest.fixture
def usd_sofr_market_data():
    """Realistic USD SOFR market data for curve construction."""
    return {
        'deposits': {
            'tenors': ['1M', '2M', '3M'],
            'rates': [5.30, 5.32, 5.35]
        },
        'futures': {
            'contracts': ['H26', 'M26', 'U26', 'Z26', 'H27', 'M27', 'U27', 'Z27'],
            'prices': [94.62, 94.60, 94.58, 94.57, 94.56, 94.55, 94.54, 94.53]
        },
        'ois': {
            'tenors': ['2Y', '3Y', '5Y', '7Y', '10Y', '15Y', '20Y', '30Y'],
            'rates': [5.45, 5.43, 5.38, 5.34, 5.28, 5.22, 5.18, 5.15]
        }
    }


@pytest.fixture
def usd_model(value_date, usd_sofr_market_data):
    """USD SOFR model with AD-enabled curve (DELTA + GAMMA)."""
    from cavour.trades.rates.ois_curve import OISCurve
    from cavour.market.curves.interpolator import InterpTypes

    model = Model(value_date)

    # Build deposits
    deposits = []
    for tenor, rate in zip(usd_sofr_market_data['deposits']['tenors'],
                          usd_sofr_market_data['deposits']['rates']):
        dep = CashDeposit(
            effective_dt=value_date,
            term_dt_or_tenor=tenor,
            deposit_rate=rate / 100.0,
            dc_type=DayCountTypes.ACT_360,
            floating_index=CurveTypes.USD_OIS_SOFR,
            currency=CurrencyTypes.USD,
            notional=1_000_000
        )
        deposits.append(dep)

    # Build futures
    futures = []
    for contract, price in zip(usd_sofr_market_data['futures']['contracts'],
                              usd_sofr_market_data['futures']['prices']):
        fut = IRFuture(
            effective_dt=value_date,
            expiry_date_or_contract=contract,
            futures_price=price,
            contract_type=FutureContractTypes.IMM,
            currency=CurrencyTypes.USD,
            floating_index=CurveTypes.USD_OIS_SOFR,
            dc_type=DayCountTypes.ACT_360,
            contract_size=1_000_000
        )
        futures.append(fut)

    # Build OIS swaps
    ois_swaps = []
    for tenor, rate in zip(usd_sofr_market_data['ois']['tenors'],
                          usd_sofr_market_data['ois']['rates']):
        swap = OIS(
            effective_dt=value_date,
            term_dt_or_tenor=tenor,
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=rate / 100.0,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_360,
            floating_index=CurveTypes.USD_OIS_SOFR,
            currency=CurrencyTypes.USD,
            float_freq_type=FrequencyTypes.ANNUAL,
            float_dc_type=DayCountTypes.ACT_360,
            notional=1_000_000
        )
        ois_swaps.append(swap)

    # Build curve with instruments directly (not via Model.build_curve)
    all_instruments = deposits + futures + ois_swaps
    curve = OISCurve(
        value_dt=value_date,
        instruments=all_instruments,
        interp_type=InterpTypes.LINEAR_ZERO_RATES,
        check_refit=False,
        use_ad=True,  # Enable AD for DELTA
        compute_gamma=True  # CRITICAL: Enable Hessian computation for GAMMA
    )

    # Add curve to model
    model._curves_dict['USD_OIS_SOFR'] = curve

    return model


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_fd_delta_parallel(instrument, model, value_date, bump_bp=1.0):
    """
    Compute DELTA via parallel curve shift using central finite differences.

    Args:
        instrument: Cash deposit, IR future, or OIS swap
        model: Model with USD_OIS_SOFR curve
        value_date: Valuation date
        bump_bp: Bump size in basis points (1bp = 0.01%)

    Returns:
        Total DELTA (sum across all tenors) in 1bp units
    """
    curve_name = 'USD_OIS_SOFR'
    shock_pct = bump_bp * 0.01  # Convert bp to percentage

    # Bump curve up
    model_up = model.scenario(curve_name, shock=shock_pct)
    value_up = instrument.position(model_up).compute([RequestTypes.VALUE]).value.amount

    # Bump curve down
    model_down = model.scenario(curve_name, shock=-shock_pct)
    value_down = instrument.position(model_down).compute([RequestTypes.VALUE]).value.amount

    # Central difference: (V+ - V-) / (2 * bump)
    delta_fd = (value_up - value_down) / (2.0 * bump_bp)

    return delta_fd


def compute_fd_delta_tenor(instrument, model, value_date, tenor_name, bump_bp=1.0):
    """
    Compute DELTA for a specific tenor via tenor-specific bump.

    Args:
        instrument: Cash deposit, IR future, or OIS swap
        model: Model with USD_OIS_SOFR curve
        value_date: Valuation date
        tenor_name: Tenor to bump (e.g., '3M', '1Y', '5Y')
        bump_bp: Bump size in basis points

    Returns:
        DELTA for that specific tenor in 1bp units
    """
    curve_name = 'USD_OIS_SOFR'
    shock_pct = bump_bp * 0.01

    # Bump specific tenor up
    shock_dict_up = {tenor_name: shock_pct}
    model_up = model.scenario(curve_name, shock=shock_dict_up)
    value_up = instrument.position(model_up).compute([RequestTypes.VALUE]).value.amount

    # Bump specific tenor down
    shock_dict_down = {tenor_name: -shock_pct}
    model_down = model.scenario(curve_name, shock=shock_dict_down)
    value_down = instrument.position(model_down).compute([RequestTypes.VALUE]).value.amount

    # Central difference
    delta_fd_tenor = (value_up - value_down) / (2.0 * bump_bp)

    return delta_fd_tenor


# ============================================================================
# VALUE TESTS
# ============================================================================

def test_value_cash_deposit(value_date, usd_model):
    """Test VALUE computation for cash deposit at inception."""
    # Create 3M deposit at market rate
    dep = CashDeposit(
        effective_dt=value_date,
        term_dt_or_tenor='3M',
        deposit_rate=5.35 / 100.0,  # Market rate
        dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD,
        notional=1_000_000
    )

    position = dep.position(usd_model)
    result = position.compute([RequestTypes.VALUE])
    value = result.value.amount

    print(f"\nCash Deposit (3M) VALUE: ${value:,.2f}")

    # At inception with market rate, value should be near zero
    assert abs(value) < 10.0, f"Expected near-zero value at inception, got ${value}"


def test_value_ir_future(value_date, usd_model):
    """Test VALUE computation for IR future at inception."""
    # Create H26 future at market price
    fut = IRFuture(
        effective_dt=value_date,
        expiry_date_or_contract='H26',
        futures_price=94.62,  # Market price
        contract_type=FutureContractTypes.IMM,
        currency=CurrencyTypes.USD,
        floating_index=CurveTypes.USD_OIS_SOFR,
        dc_type=DayCountTypes.ACT_360,
        contract_size=1_000_000
    )

    position = fut.position(usd_model)
    result = position.compute([RequestTypes.VALUE])
    value = result.value.amount

    print(f"\nIR Future (H26) VALUE: ${value:,.2f}")

    # At inception with market price, value should be near zero
    assert abs(value) < 10.0, f"Expected near-zero value at inception, got ${value}"


def test_value_ois_swap(value_date, usd_model):
    """Test VALUE computation for OIS swap at inception."""
    # Create 5Y OIS at market rate
    swap = OIS(
        effective_dt=value_date,
        term_dt_or_tenor='5Y',
        fixed_leg_type=SwapTypes.PAY,
        fixed_coupon=5.38 / 100.0,  # Market rate
        fixed_freq_type=FrequencyTypes.ANNUAL,
        fixed_dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_360,
        notional=1_000_000
    )

    position = swap.position(usd_model)
    result = position.compute([RequestTypes.VALUE])
    value = result.value.amount

    print(f"\nOIS Swap (5Y) VALUE: ${value:,.2f}")

    # At inception with market rate, value should be near zero
    assert abs(value) < 100.0, f"Expected near-zero value at inception, got ${value}"


# ============================================================================
# DELTA TESTS - PARALLEL SHIFT VALIDATION
# ============================================================================

@pytest.mark.parametrize("bump_bp", [1.0, 10.0])
def test_delta_parallel_cash_deposit(value_date, usd_model, bump_bp):
    """Validate Cash Deposit DELTA via parallel curve shift (AD vs FD)."""
    # Create overnight deposit (not at market rate to have non-zero DELTA)
    dep = CashDeposit(
        effective_dt=value_date,
        term_dt_or_tenor='1D',
        deposit_rate=5.00 / 100.0,  # Off-market rate
        dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD,
        notional=1_000_000
    )

    # Compute AD DELTA
    position = dep.position(usd_model)
    result = position.compute([RequestTypes.DELTA])
    delta_ad = sum(result.risk.risk_ladder)

    # Compute FD DELTA
    delta_fd = compute_fd_delta_parallel(dep, usd_model, value_date, bump_bp=bump_bp)

    # Validate match
    error = abs(delta_ad - delta_fd)
    tolerance = abs(delta_fd) * (0.001 if bump_bp == 1.0 else 0.01)

    print(f"\nCash Deposit DELTA (parallel {bump_bp}bp):")
    print(f"  AD:    {delta_ad:>12.6f}")
    print(f"  FD:    {delta_fd:>12.6f}")
    print(f"  Error: {error:>12.6f} (tol: {tolerance:.6f})")

    assert error < tolerance, f"AD vs FD mismatch: {error:.6f} > {tolerance:.6f}"


@pytest.mark.parametrize("bump_bp", [1.0, 10.0])
def test_delta_parallel_ir_future(value_date, usd_model, bump_bp):
    """Validate IR Future DELTA via parallel curve shift (AD vs FD)."""
    # Create future at off-market price
    fut = IRFuture(
        effective_dt=value_date,
        expiry_date_or_contract='M26',
        futures_price=95.00,  # Off-market price
        contract_type=FutureContractTypes.IMM,
        currency=CurrencyTypes.USD,
        floating_index=CurveTypes.USD_OIS_SOFR,
        dc_type=DayCountTypes.ACT_360,
        contract_size=1_000_000
    )

    # Compute AD DELTA
    position = fut.position(usd_model)
    result = position.compute([RequestTypes.DELTA])
    delta_ad = sum(result.risk.risk_ladder)

    # Compute FD DELTA
    delta_fd = compute_fd_delta_parallel(fut, usd_model, value_date, bump_bp=bump_bp)

    # Validate match
    error = abs(delta_ad - delta_fd)
    tolerance = abs(delta_fd) * (0.001 if bump_bp == 1.0 else 0.01)

    print(f"\nIR Future DELTA (parallel {bump_bp}bp):")
    print(f"  AD:    {delta_ad:>12.6f}")
    print(f"  FD:    {delta_fd:>12.6f}")
    print(f"  Error: {error:>12.6f} (tol: {tolerance:.6f})")

    assert error < tolerance, f"AD vs FD mismatch: {error:.6f} > {tolerance:.6f}"


@pytest.mark.parametrize("bump_bp", [1.0, 10.0])
def test_delta_parallel_ois_swap(value_date, usd_model, bump_bp):
    """Validate OIS Swap DELTA via parallel curve shift (AD vs FD)."""
    # Create 5Y swap at off-market rate
    swap = OIS(
        effective_dt=value_date,
        term_dt_or_tenor='5Y',
        fixed_leg_type=SwapTypes.PAY,
        fixed_coupon=5.00 / 100.0,  # Off-market rate
        fixed_freq_type=FrequencyTypes.ANNUAL,
        fixed_dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_360,
        notional=1_000_000
    )

    # Compute AD DELTA
    position = swap.position(usd_model)
    result = position.compute([RequestTypes.DELTA])
    delta_ad = sum(result.risk.risk_ladder)

    # Compute FD DELTA
    delta_fd = compute_fd_delta_parallel(swap, usd_model, value_date, bump_bp=bump_bp)

    # Validate match
    error = abs(delta_ad - delta_fd)
    tolerance = abs(delta_fd) * (0.001 if bump_bp == 1.0 else 0.01)

    print(f"\nOIS Swap DELTA (parallel {bump_bp}bp):")
    print(f"  AD:    {delta_ad:>12.6f}")
    print(f"  FD:    {delta_fd:>12.6f}")
    print(f"  Error: {error:>12.6f} (tol: {tolerance:.6f})")

    assert error < tolerance, f"AD vs FD mismatch: {error:.6f} > {tolerance:.6f}"


# ============================================================================
# DELTA TESTS - TENOR-SPECIFIC VALIDATION
# ============================================================================

@pytest.mark.parametrize("tenor", ['3M', '1Y', '5Y', '10Y'])
def test_delta_tenor_specific_ois(value_date, usd_model, tenor):
    """Validate OIS DELTA for specific tenors (AD vs FD)."""
    # Create 10Y swap at off-market rate (has exposure to all tenors)
    swap = OIS(
        effective_dt=value_date,
        term_dt_or_tenor='10Y',
        fixed_leg_type=SwapTypes.PAY,
        fixed_coupon=5.00 / 100.0,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        fixed_dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_360,
        notional=1_000_000
    )

    # Compute AD DELTA
    position = swap.position(usd_model)
    result = position.compute([RequestTypes.DELTA])
    delta_ad_tenors = result.risk.risk_ladder
    delta_tenors_list = result.risk.tenors

    # Find the tenor index
    try:
        tenor_idx = delta_tenors_list.index(tenor)
        delta_ad_tenor = delta_ad_tenors[tenor_idx]
    except ValueError:
        pytest.skip(f"Tenor {tenor} not in risk ladder: {delta_tenors_list}")

    # Compute FD DELTA for that tenor
    delta_fd_tenor = compute_fd_delta_tenor(swap, usd_model, value_date, tenor, bump_bp=1.0)

    # Validate match
    error = abs(delta_ad_tenor - delta_fd_tenor)

    # Use relative error if delta is significant
    if abs(delta_fd_tenor) > 1e-6:
        rel_error = error / abs(delta_fd_tenor)
        print(f"\nOIS DELTA (tenor {tenor}):")
        print(f"  AD:        {delta_ad_tenor:>12.6f}")
        print(f"  FD:        {delta_fd_tenor:>12.6f}")
        print(f"  Rel Error: {rel_error:>12.6%}")
        assert rel_error < 0.05, f"Relative error {rel_error:.2%} > 5%"
    else:
        print(f"\nOIS DELTA (tenor {tenor}):")
        print(f"  AD:        {delta_ad_tenor:>12.6f}")
        print(f"  FD:        {delta_fd_tenor:>12.6f}")
        print(f"  Abs Error: {error:>12.6f}")
        assert error < 1e-3, f"Absolute error {error:.6f} > 0.001"


# ============================================================================
# GAMMA TESTS - TAYLOR EXPANSION VALIDATION
# ============================================================================

@pytest.mark.parametrize("shock_bp", [100.0, 200.0])
def test_gamma_taylor_expansion_ois(value_date, usd_model, shock_bp):
    """
    Validate OIS GAMMA using Taylor expansion for large shocks.

    Taylor expansion:
        dV ≈ delta · dx + 0.5 · gamma · dx²

    For large shocks (100bp, 200bp), 2nd-order approximation should be
    significantly better than 1st-order (DELTA-only) approximation.
    """
    # Create 10Y swap at off-market rate
    swap = OIS(
        effective_dt=value_date,
        term_dt_or_tenor='10Y',
        fixed_leg_type=SwapTypes.PAY,
        fixed_coupon=5.00 / 100.0,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        fixed_dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_360,
        notional=1_000_000
    )

    # Get VALUE, DELTA, GAMMA
    position = swap.position(usd_model)
    result = position.compute([RequestTypes.VALUE, RequestTypes.DELTA, RequestTypes.GAMMA])

    v0 = result.value.amount
    delta = np.array(result.risk.risk_ladder)
    gamma = np.array(result.gamma.risk_ladder)

    # Create shocked scenario (parallel shift)
    shock_scalar = shock_bp * 0.01  # Convert bp to percentage
    model_shocked = usd_model.scenario('USD_OIS_SOFR', shock=shock_scalar)
    v_shocked = swap.position(model_shocked).compute([RequestTypes.VALUE]).value.amount

    actual_pnl = v_shocked - v0

    # Compute dx vector (all tenors shifted by same amount)
    n_tenors = len(delta)
    dx = np.full(n_tenors, shock_bp)  # All tenors shifted by shock_bp basis points

    # 1st order approximation: DELTA · dx
    pnl_1st = np.dot(delta, dx)

    # 2nd order approximation: DELTA · dx + 0.5 · GAMMA · dx²
    pnl_2nd = pnl_1st + 0.5 * np.dot(dx, np.dot(gamma, dx))

    # Compute errors
    error_1st = abs(actual_pnl - pnl_1st)
    error_2nd = abs(actual_pnl - pnl_2nd)

    # 2nd order should improve over 1st order
    improvement_ratio = error_1st / error_2nd if error_2nd > 1e-10 else float('inf')

    print(f"\nGAMMA Taylor Expansion ({shock_bp:.0f}bp shock):")
    print(f"  Actual P&L:        {actual_pnl:>15,.2f}")
    print(f"  1st order (DELTA): {pnl_1st:>15,.2f}  (error: {error_1st:>12,.2f})")
    print(f"  2nd order (+GAMMA):{pnl_2nd:>15,.2f}  (error: {error_2nd:>12,.2f})")
    print(f"  Improvement:       {improvement_ratio:>15.2f}x")

    # 2nd order should be significantly better than 1st order
    assert improvement_ratio > 1.5, f"GAMMA did not improve approximation: {improvement_ratio:.2f}x < 1.5x"

    # 2nd order residual error should be small
    rel_error_2nd = error_2nd / abs(actual_pnl) if abs(actual_pnl) > 1e-6 else 0.0
    max_rel_error = 0.05 if shock_bp == 100.0 else 0.10  # 5% for 100bp, 10% for 200bp

    print(f"  2nd order rel err: {rel_error_2nd:>15.2%} (max: {max_rel_error:.2%})")

    assert rel_error_2nd < max_rel_error, f"2nd order error {rel_error_2nd:.2%} > {max_rel_error:.2%}"


# ============================================================================
# GAMMA TESTS - SYMMETRY VALIDATION
# ============================================================================

def test_gamma_symmetry_ois(value_date, usd_model):
    """Validate that GAMMA matrix is symmetric."""
    # Create 10Y swap
    swap = OIS(
        effective_dt=value_date,
        term_dt_or_tenor='10Y',
        fixed_leg_type=SwapTypes.PAY,
        fixed_coupon=5.00 / 100.0,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        fixed_dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_360,
        notional=1_000_000
    )

    # Get GAMMA
    position = swap.position(usd_model)
    result = position.compute([RequestTypes.GAMMA])
    gamma = np.array(result.gamma.risk_ladder)

    # Check symmetry: gamma[i,j] = gamma[j,i]
    max_asymmetry = 0.0
    for i in range(gamma.shape[0]):
        for j in range(i+1, gamma.shape[1]):
            asymmetry = abs(gamma[i, j] - gamma[j, i])
            max_asymmetry = max(max_asymmetry, asymmetry)

    print(f"\nGAMMA Symmetry Check:")
    print(f"  Matrix shape:      {gamma.shape}")
    print(f"  Max asymmetry:     {max_asymmetry:.2e}")

    # Hessian should be symmetric (Schwarz's theorem)
    assert max_asymmetry < 1e-10, f"GAMMA not symmetric: max asymmetry {max_asymmetry:.2e}"


# ============================================================================
# SUMMARY TEST
# ============================================================================

def test_summary_all_instruments(value_date, usd_model):
    """
    Summary test showing DELTA and GAMMA for all three instrument types.
    Demonstrates the complete AD infrastructure in action.
    """
    print("\n" + "="*100)
    print("COMPREHENSIVE AD DELTA/GAMMA SUMMARY")
    print("="*100)

    instruments = []

    # Cash Deposit
    dep = CashDeposit(
        effective_dt=value_date,
        term_dt_or_tenor='3M',
        deposit_rate=5.00 / 100.0,
        dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD,
        notional=1_000_000
    )
    instruments.append(('Cash Deposit (3M)', dep))

    # IR Future
    fut = IRFuture(
        effective_dt=value_date,
        expiry_date_or_contract='M26',
        futures_price=95.00,
        contract_type=FutureContractTypes.IMM,
        currency=CurrencyTypes.USD,
        floating_index=CurveTypes.USD_OIS_SOFR,
        dc_type=DayCountTypes.ACT_360,
        contract_size=1_000_000
    )
    instruments.append(('IR Future (M26)', fut))

    # OIS Swap
    swap = OIS(
        effective_dt=value_date,
        term_dt_or_tenor='5Y',
        fixed_leg_type=SwapTypes.PAY,
        fixed_coupon=5.00 / 100.0,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        fixed_dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_360,
        notional=1_000_000
    )
    instruments.append(('OIS Swap (5Y)', swap))

    # Compute sensitivities for all
    print("\nInstrument              VALUE            DELTA (total)   GAMMA (trace)")
    print("-" * 100)

    for name, instrument in instruments:
        position = instrument.position(usd_model)
        result = position.compute([RequestTypes.VALUE, RequestTypes.DELTA, RequestTypes.GAMMA])

        value = result.value.amount
        delta_total = sum(result.risk.risk_ladder)
        gamma_matrix = np.array(result.gamma.risk_ladder)
        gamma_trace = np.trace(gamma_matrix)  # Sum of diagonal elements

        print(f"{name:<22s}  {value:>15,.2f}  {delta_total:>15,.2f}  {gamma_trace:>15,.2f}")

    print("="*100)
    print("\nAD DELTA/GAMMA computation successful for all instrument types!")
    print("- VALUE: Present value using AD-compatible curve interpolation")
    print("- DELTA: First derivative dV/dr via jax.grad() + chain rule")
    print("- GAMMA: Second derivative d²V/dr² via jax.hessian() + chain rule")
    print("="*100)


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "-s"])
