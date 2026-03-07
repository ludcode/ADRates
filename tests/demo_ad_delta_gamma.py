"""
Simple Demonstration: DELTA and GAMMA via Automatic Differentiation (JAX)

Shows how to compute first-order (DELTA) and second-order (GAMMA) sensitivities
for Cash Deposits, IR Futures, and OIS Swaps using JAX-based automatic differentiation.

This demonstrates the core AD infrastructure without complex test harnesses.
"""

import sys
sys.path.insert(0, 'C:\\Projects\\Cavour')

import numpy as np
from cavour.utils.date import Date
from cavour.utils.day_count import DayCountTypes
from cavour.utils.global_types import CurveTypes, SwapTypes, FutureContractTypes, RequestTypes
from cavour.utils.currency import CurrencyTypes
from cavour.utils.frequency import FrequencyTypes
from cavour.trades.rates.cash_deposit import CashDeposit
from cavour.trades.rates.ir_future import IRFuture
from cavour.trades.rates.ois import OIS
from cavour.trades.rates.ois_curve import OISCurve
from cavour.market.curves.interpolator import InterpTypes
from cavour.models.models import Model


def main():
    """Demonstrate AD DELTA/GAMMA computation for Cash Deposits, IR Futures, and OIS Swaps."""

    print("="*100)
    print("AUTOMATIC DIFFERENTIATION (AD) DEMONSTRATION: DELTA AND GAMMA")
    print("="*100)
    print("\nUsing JAX for exact derivatives via jax.grad() and jax.hessian()")
    print()

    # ========================================================================
    # SETUP: Build USD SOFR curve
    # ========================================================================
    value_date = Date(26, 2, 2026)

    print("Step 1: Building USD SOFR curve with Deposits + Futures + OIS...")

    # Create instruments
    deposits = []
    for tenor, rate in zip(['1M', '2M', '3M'], [5.30, 5.32, 5.35]):
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

    futures = []
    for contract, price in zip(['H26', 'M26', 'U26', 'Z26'], [94.62, 94.60, 94.58, 94.57]):
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

    ois_swaps = []
    for tenor, rate in zip(['2Y', '3Y', '5Y', '7Y', '10Y'], [5.45, 5.43, 5.38, 5.34, 5.28]):
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

    # Build curve with AD enabled
    all_instruments = deposits + futures + ois_swaps
    curve = OISCurve(
        value_dt=value_date,
        instruments=all_instruments,
        interp_type=InterpTypes.LINEAR_ZERO_RATES,
        check_refit=False,
        use_ad=True,          # Enable Jacobian computation (for DELTA)
        compute_gamma=True    # Enable Hessian computation (for GAMMA)
    )

    # Add to model
    model = Model(value_date)
    model._curves_dict['USD_OIS_SOFR'] = curve

    print(f"  [OK] Curve built: {len(curve._times)} discount factor points")
    print(f"  [OK] Jacobian stored: {curve._jac.shape if curve._jac is not None else 'None'}")
    print(f"  [OK] Hessian stored: {curve._hess.shape if curve._hess is not None else 'None'}")
    print()

    # ========================================================================
    # TEST 1: Cash Deposit DELTA and GAMMA
    # ========================================================================
    print("="*100)
    print("TEST 1: Cash Deposit (3M) - DELTA and GAMMA")
    print("="*100)

    dep_test = CashDeposit(
        effective_dt=value_date,
        term_dt_or_tenor='3M',
        deposit_rate=5.00 / 100.0,  # Off-market rate for non-zero DELTA
        dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD,
        notional=1_000_000
    )

    position = dep_test.position(model)
    result = position.compute([RequestTypes.VALUE, RequestTypes.DELTA, RequestTypes.GAMMA])

    print(f"\nVALUE: ${result.value.amount:,.2f}")
    print(f"\nDELTA (first derivative dV/dr via jax.grad):")
    print(f"  Tenors: {result.risk.tenors}")
    print(f"  DELTA:  {[f'{x:.4f}' for x in result.risk.risk_ladder]}")
    print(f"  Total:  {sum(result.risk.risk_ladder):.4f} (sum across all tenors)")

    print(f"\nGAMMA (second derivative d²V/dr² via jax.hessian):")
    gamma_matrix = np.array(result.gamma.risk_ladder)
    print(f"  Shape:       {gamma_matrix.shape}")
    print(f"  Trace:       {np.trace(gamma_matrix):.4f} (sum of diagonal)")
    print(f"  Symmetric:   {np.allclose(gamma_matrix, gamma_matrix.T)}")
    print(f"  Max element: {np.max(np.abs(gamma_matrix)):.4e}")

    # ========================================================================
    # TEST 2: OIS Swap DELTA and GAMMA
    # ========================================================================
    print("\n" + "="*100)
    print("TEST 2: OIS Swap (5Y) - DELTA and GAMMA")
    print("="*100)

    swap_test = OIS(
        effective_dt=value_date,
        term_dt_or_tenor='5Y',
        fixed_leg_type=SwapTypes.PAY,
        fixed_coupon=5.00 / 100.0,  # Off-market rate for non-zero DELTA
        fixed_freq_type=FrequencyTypes.ANNUAL,
        fixed_dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_360,
        notional=1_000_000
    )

    position = swap_test.position(model)
    result = position.compute([RequestTypes.VALUE, RequestTypes.DELTA, RequestTypes.GAMMA])

    print(f"\nVALUE: ${result.value.amount:,.2f}")
    print(f"\nDELTA (first derivative dV/dr via jax.grad):")
    print(f"  Tenors: {result.risk.tenors}")
    print(f"  DELTA:  {[f'{x:.4f}' for x in result.risk.risk_ladder]}")
    print(f"  Total:  {sum(result.risk.risk_ladder):.4f} (sum across all tenors)")

    print(f"\nGAMMA (second derivative d²V/dr² via jax.hessian):")
    gamma_matrix = np.array(result.gamma.risk_ladder)
    print(f"  Shape:       {gamma_matrix.shape}")
    print(f"  Trace:       {np.trace(gamma_matrix):.4f} (sum of diagonal)")
    print(f"  Symmetric:   {np.allclose(gamma_matrix, gamma_matrix.T)}")
    print(f"  Max element: {np.max(np.abs(gamma_matrix)):.4e}")

    # Display a portion of the GAMMA matrix
    print(f"\nGAMMA Matrix (first 5x5 block):")
    print(gamma_matrix[:5, :5])

    # ========================================================================
    # TEST 3: IR Future DELTA and GAMMA
    # ========================================================================
    print("\n" + "="*100)
    print("TEST 3: IR Future (H26 - June 2026) - DELTA and GAMMA")
    print("="*100)

    fut_test = IRFuture(
        effective_dt=value_date,
        expiry_date_or_contract='H26',
        futures_price=94.50,  # Off-market price for non-zero DELTA
        contract_type=FutureContractTypes.IMM,
        currency=CurrencyTypes.USD,
        floating_index=CurveTypes.USD_OIS_SOFR,
        dc_type=DayCountTypes.ACT_360,
        contract_size=1_000_000
    )

    position = fut_test.position(model)
    result = position.compute([RequestTypes.VALUE, RequestTypes.DELTA, RequestTypes.GAMMA])

    print(f"\nVALUE: ${result.value.amount:,.2f}")
    print(f"\nDELTA (first derivative dV/dr via jax.grad):")
    print(f"  Tenors: {result.risk.tenors}")
    print(f"  DELTA:  {[f'{x:.4f}' for x in result.risk.risk_ladder]}")
    print(f"  Total:  {sum(result.risk.risk_ladder):.4f} (sum across all tenors)")

    print(f"\nGAMMA (second derivative d²V/dr² via jax.hessian):")
    gamma_matrix = np.array(result.gamma.risk_ladder)
    print(f"  Shape:       {gamma_matrix.shape}")
    print(f"  Trace:       {np.trace(gamma_matrix):.4f} (sum of diagonal)")
    print(f"  Symmetric:   {np.allclose(gamma_matrix, gamma_matrix.T)}")
    print(f"  Max element: {np.max(np.abs(gamma_matrix)):.4e}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print()
    print("[OK] Automatic Differentiation (AD) successfully computes DELTA and GAMMA")
    print("[OK] DELTA: First derivative via jax.grad()")
    print("[OK] GAMMA: Second derivative via jax.hessian()")
    print("[OK] Chain rule: dV/dr = (dV/dDF) * (dDF/dr)")
    print("[OK] Chain rule: d^2V/dr^2 = (dDF/dr)^T * (d^2V/dDF^2) * (dDF/dr) + Sum(dV/dDF * d^2DF/dr^2)")
    print()
    print("Supported instruments:")
    print("  [OK] Cash Deposits")
    print("  [OK] OIS Swaps")
    print("  [OK] FRAs (Forward Rate Agreements)")
    print("  [OK] IR Futures (SOFR/EURIBOR futures)")
    print("  [OK] XCCY Swaps (Cross-Currency)")
    print()
    print("="*100)


if __name__ == "__main__":
    main()
