"""
Test USD SOFR curve construction using Deposits + FRAs + OIS (no overlap).

Validates that all instruments used to build the curve reprice within tolerance
as defined in ois_curve.py (SWAP_TOL = 1e-10).

Curve Structure (Non-Overlapping):
- Cash Deposits: 0-3M (1M, 2M, 3M)
- FRAs: 3M-18M (3x6, 6x9, 9x12, 12x15, 15x18)
- OIS Swaps: 2Y+ (2Y, 3Y, 5Y, 7Y, 10Y, 15Y, 20Y, 30Y)
"""

import sys
sys.path.insert(0, 'C:\\Projects\\Cavour')

from tests.sample_market_data_usd_sofr import (
    VALUE_DATE,
    get_deposit_instruments,
    get_fra_instruments,
    get_ois_instruments,
)
from cavour.trades.rates.ois_curve import OISCurve, SWAP_TOL
from cavour.market.curves.interpolator import InterpTypes


def test_curve_1_deposits_fras_ois():
    """Build Curve 1: Deposits + FRAs + OIS and validate instrument refit."""

    print("=" * 100)
    print("CURVE 1: Deposits + FRAs + OIS (Non-Overlapping)")
    print("=" * 100)
    print(f"Value Date: {VALUE_DATE}")
    print(f"Refit Tolerance: {SWAP_TOL} (absolute value / notional)")
    print()

    # ========================================================================
    # Step 1: Create instruments
    # ========================================================================
    deposits = get_deposit_instruments()
    fras = get_fra_instruments()
    ois_swaps = get_ois_instruments()

    print(f"Instruments:")
    print(f"  - Deposits: {len(deposits)} (1M, 2M, 3M)")
    print(f"  - FRAs:     {len(fras)} (3x6, 6x9, 9x12, 12x15, 15x18)")
    print(f"  - OIS:      {len(ois_swaps)} (2Y, 3Y, 5Y, 7Y, 10Y, 15Y, 20Y, 30Y)")
    print(f"  - Total:    {len(deposits) + len(fras) + len(ois_swaps)} instruments")
    print()

    # ========================================================================
    # Step 2: Build curve
    # ========================================================================
    all_instruments = deposits + fras + ois_swaps

    print("Building curve with LINEAR_ZERO_RATES interpolation...")
    curve = OISCurve(
        value_dt=VALUE_DATE,
        instruments=all_instruments,
        interp_type=InterpTypes.LINEAR_ZERO_RATES,
        check_refit=False,  # We'll do manual validation
        use_ad=True,
        compute_gamma=False
    )
    print(f"Curve built: {len(curve._times)} discount factor points")
    print()

    # ========================================================================
    # Step 3: Validate refit for each instrument type
    # ========================================================================

    results = {
        "deposits": [],
        "fras": [],
        "ois": []
    }

    # --- DEPOSITS ---
    print("=" * 100)
    print("DEPOSIT REFIT VALIDATION")
    print("=" * 100)
    print(f"{'Maturity':<15s} {'PV ($)':<20s} {'Abs Error':<20s} {'Status':<10s}")
    print("-" * 100)

    for dep in deposits:
        pv = dep.value(VALUE_DATE, ois_curve=curve)
        abs_error = abs(pv) / dep._notional
        status = "PASS" if abs_error <= SWAP_TOL else "FAIL"

        results["deposits"].append({
            "instrument": dep,
            "maturity": dep._maturity_dt,
            "pv": pv,
            "abs_error": abs_error,
            "status": status
        })

        print(f"{str(dep._maturity_dt):<15s} {pv:>18.10f}  {abs_error:>18.10e}  {status:<10s}")

    # --- FRAs ---
    print()
    print("=" * 100)
    print("FRA REFIT VALIDATION")
    print("=" * 100)
    print(f"{'Period':<15s} {'PV ($)':<20s} {'Abs Error':<20s} {'Status':<10s}")
    print("-" * 100)

    for fra in fras:
        pv = fra.value(VALUE_DATE, ois_curve=curve)
        abs_error = abs(pv) / fra._notional
        status = "PASS" if abs_error <= SWAP_TOL else "FAIL"

        period_str = f"{fra._start_dt} to {fra._end_dt}"

        results["fras"].append({
            "instrument": fra,
            "period": period_str,
            "pv": pv,
            "abs_error": abs_error,
            "status": status
        })

        # Truncate period string if too long
        period_display = period_str if len(period_str) <= 30 else period_str[:27] + "..."
        print(f"{period_display:<15s} {pv:>18.10f}  {abs_error:>18.10e}  {status:<10s}")

    # --- OIS SWAPS ---
    print()
    print("=" * 100)
    print("OIS SWAP REFIT VALIDATION")
    print("=" * 100)
    print(f"{'Maturity':<15s} {'PV ($)':<20s} {'Abs Error':<20s} {'Status':<10s}")
    print("-" * 100)

    for swap in ois_swaps:
        pv = swap.value(VALUE_DATE, ois_curve=curve)
        abs_error = abs(pv) / swap._notional
        status = "PASS" if abs_error <= SWAP_TOL else "FAIL"

        results["ois"].append({
            "instrument": swap,
            "maturity": swap._maturity_dt,
            "pv": pv,
            "abs_error": abs_error,
            "status": status
        })

        print(f"{str(swap._maturity_dt):<15s} {pv:>18.10f}  {abs_error:>18.10e}  {status:<10s}")

    # ========================================================================
    # Step 4: Summary Report
    # ========================================================================
    print()
    print("=" * 100)
    print("SUMMARY REPORT")
    print("=" * 100)

    total_instruments = 0
    total_pass = 0
    total_fail = 0
    max_error = 0.0
    max_error_instrument = None

    for inst_type, inst_results in results.items():
        type_pass = sum(1 for r in inst_results if r["status"] == "PASS")
        type_fail = sum(1 for r in inst_results if r["status"] == "FAIL")
        type_total = len(inst_results)

        total_instruments += type_total
        total_pass += type_pass
        total_fail += type_fail

        type_max_error = max([r["abs_error"] for r in inst_results]) if inst_results else 0.0

        if type_max_error > max_error:
            max_error = type_max_error
            max_error_instrument = inst_type

        print(f"{inst_type.upper():<10s}: {type_pass:>3d}/{type_total:<3d} passed, "
              f"max error = {type_max_error:.10e}")

    print("-" * 100)
    print(f"{'TOTAL':<10s}: {total_pass:>3d}/{total_instruments:<3d} passed ({total_pass/total_instruments*100:.1f}%)")
    print(f"{'FAILED':<10s}: {total_fail:>3d}/{total_instruments:<3d}")
    print(f"\nMax absolute error: {max_error:.10e} ({max_error_instrument})")
    print(f"Target tolerance:   {SWAP_TOL:.10e}")

    if total_fail > 0:
        print(f"\nWARNING: {total_fail} instrument(s) failed to meet tolerance {SWAP_TOL}")
    else:
        print(f"\nSUCCESS: All instruments reprice within tolerance {SWAP_TOL}")

    print("=" * 100)

    return results, curve


if __name__ == "__main__":
    results, curve = test_curve_1_deposits_fras_ois()
