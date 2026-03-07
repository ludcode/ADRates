"""
Robustness Testing: Multiple Combinations of Deposits + Futures + OIS

Tests various realistic curve structures to validate implementation robustness:
- Different deposit coverage (short vs long)
- 1-month vs 3-month futures
- Mixed future types
- Different OIS starting points

Target: All combinations should achieve SWAP_TOL = 1e-10
"""

import sys
sys.path.insert(0, 'C:\\Projects\\Cavour')

from cavour.utils.date import Date
from cavour.utils.day_count import DayCountTypes
from cavour.utils.global_types import CurveTypes, SwapTypes, FutureContractTypes
from cavour.utils.currency import CurrencyTypes
from cavour.utils.frequency import FrequencyTypes
from cavour.trades.rates.cash_deposit import CashDeposit
from cavour.trades.rates.ir_future import IRFuture
from cavour.trades.rates.ois import OIS
from cavour.trades.rates.ois_curve import OISCurve, SWAP_TOL
from cavour.market.curves.interpolator import InterpTypes


VALUE_DATE = Date(26, 2, 2026)


def create_deposits(tenors, rates):
    """Create cash deposit instruments."""
    deposits = []
    for tenor, rate in zip(tenors, rates):
        dep = CashDeposit(
            effective_dt=VALUE_DATE,
            term_dt_or_tenor=tenor,
            deposit_rate=rate / 100.0,
            dc_type=DayCountTypes.ACT_360,
            floating_index=CurveTypes.USD_OIS_SOFR,
            currency=CurrencyTypes.USD,
            notional=1_000_000
        )
        deposits.append(dep)
    return deposits


def create_futures_3m(contracts, prices):
    """Create 3-month SOFR futures (IMM)."""
    futures = []
    for contract, price in zip(contracts, prices):
        fut = IRFuture(
            effective_dt=VALUE_DATE,
            expiry_date_or_contract=contract,
            futures_price=price,
            contract_type=FutureContractTypes.IMM,
            currency=CurrencyTypes.USD,
            floating_index=CurveTypes.USD_OIS_SOFR,
            dc_type=DayCountTypes.ACT_360,
            contract_size=1_000_000
        )
        futures.append(fut)
    return futures


def create_futures_1m(contracts, prices):
    """Create 1-month SOFR futures (Serial Monthly)."""
    futures = []
    for contract, price in zip(contracts, prices):
        fut = IRFuture(
            effective_dt=VALUE_DATE,
            expiry_date_or_contract=contract,
            futures_price=price,
            contract_type=FutureContractTypes.SERIAL_MONTHLY,
            currency=CurrencyTypes.USD,
            floating_index=CurveTypes.USD_OIS_SOFR,
            dc_type=DayCountTypes.ACT_360,
            contract_size=1_000_000
        )
        futures.append(fut)
    return futures


def create_ois(tenors, rates):
    """Create OIS instruments."""
    swaps = []
    for tenor, rate in zip(tenors, rates):
        swap = OIS(
            effective_dt=VALUE_DATE,
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
        swaps.append(swap)
    return swaps


def test_curve(name, deposits, futures, ois_swaps):
    """Test a single curve configuration."""
    print(f"\n{'='*100}")
    print(f"TEST: {name}")
    print(f"{'='*100}")

    all_instruments = deposits + futures + ois_swaps

    print(f"Structure:")
    print(f"  - Deposits: {len(deposits)}")
    print(f"  - Futures:  {len(futures)}")
    print(f"  - OIS:      {len(ois_swaps)}")
    print(f"  - Total:    {len(all_instruments)} instruments")

    # Build curve
    try:
        curve = OISCurve(
            value_dt=VALUE_DATE,
            instruments=all_instruments,
            interp_type=InterpTypes.LINEAR_ZERO_RATES,
            check_refit=False,
            use_ad=True,
            compute_gamma=False
        )
        print(f"  - Curve:    {len(curve._times)} discount factor points")
    except Exception as e:
        print(f"\n[FAIL] Curve construction failed: {e}")
        return {"name": name, "status": "CONSTRUCTION_FAILED", "pass_rate": 0.0}

    # Validate refit
    results = {"deposits": [], "futures": [], "ois": []}

    for dep in deposits:
        pv = dep.value(VALUE_DATE, ois_curve=curve)
        abs_error = abs(pv) / dep._notional
        status = "PASS" if abs_error <= SWAP_TOL else "FAIL"
        results["deposits"].append({"pv": pv, "error": abs_error, "status": status})

    for fut in futures:
        pv = fut.value(VALUE_DATE, ois_curve=curve)
        abs_error = abs(pv) / fut._notional
        status = "PASS" if abs_error <= SWAP_TOL else "FAIL"
        results["futures"].append({"pv": pv, "error": abs_error, "status": status})

    for swap in ois_swaps:
        pv = swap.value(VALUE_DATE, ois_curve=curve)
        abs_error = abs(pv) / swap._notional
        status = "PASS" if abs_error <= SWAP_TOL else "FAIL"
        results["ois"].append({"pv": pv, "error": abs_error, "status": status})

    # Summary
    dep_pass = sum(1 for r in results["deposits"] if r["status"] == "PASS")
    fut_pass = sum(1 for r in results["futures"] if r["status"] == "PASS")
    ois_pass = sum(1 for r in results["ois"] if r["status"] == "PASS")

    total_pass = dep_pass + fut_pass + ois_pass
    total_inst = len(all_instruments)
    pass_rate = total_pass / total_inst * 100

    dep_max = max([r["error"] for r in results["deposits"]]) if results["deposits"] else 0.0
    fut_max = max([r["error"] for r in results["futures"]]) if results["futures"] else 0.0
    ois_max = max([r["error"] for r in results["ois"]]) if results["ois"] else 0.0
    max_error = max(dep_max, fut_max, ois_max)

    print(f"\nResults:")
    print(f"  - Deposits: {dep_pass}/{len(deposits)} PASS, max_err={dep_max:.2e}")
    print(f"  - Futures:  {fut_pass}/{len(futures)} PASS, max_err={fut_max:.2e}")
    print(f"  - OIS:      {ois_pass}/{len(ois_swaps)} PASS, max_err={ois_max:.2e}")
    print(f"  - TOTAL:    {total_pass}/{total_inst} PASS ({pass_rate:.1f}%)")

    status = "PASS" if total_pass == total_inst else "FAIL"
    print(f"\n[{status}] Max error: {max_error:.2e} (target: {SWAP_TOL:.2e})")

    return {
        "name": name,
        "status": status,
        "pass_rate": pass_rate,
        "max_error": max_error,
        "dep_pass": dep_pass,
        "dep_total": len(deposits),
        "fut_pass": fut_pass,
        "fut_total": len(futures),
        "ois_pass": ois_pass,
        "ois_total": len(ois_swaps)
    }


def run_all_tests():
    """Run all robustness tests."""

    print("="*100)
    print("ROBUSTNESS TESTING: Deposits + Futures + OIS Combinations")
    print("="*100)
    print(f"Value Date: {VALUE_DATE}")
    print(f"Target Tolerance: {SWAP_TOL:.2e}")

    all_results = []

    # ========================================================================
    # TEST 1: Minimal deposits, 3M futures, standard OIS
    # ========================================================================
    deposits = create_deposits(
        tenors=["1M", "3M"],
        rates=[5.30, 5.35]
    )
    futures = create_futures_3m(
        contracts=["H26", "M26", "U26", "Z26", "H27", "M27", "U27", "Z27"],
        prices=[94.62, 94.60, 94.58, 94.57, 94.56, 94.55, 94.54, 94.53]
    )
    ois = create_ois(
        tenors=["2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"],
        rates=[5.45, 5.43, 5.38, 5.34, 5.28, 5.22, 5.18, 5.15]
    )
    all_results.append(test_curve("Minimal Deposits (1M, 3M) + 3M Futures + OIS", deposits, futures, ois))

    # ========================================================================
    # TEST 2: Extended deposits, 3M futures, standard OIS
    # ========================================================================
    deposits = create_deposits(
        tenors=["1M", "2M", "3M", "6M", "9M", "12M"],
        rates=[5.30, 5.32, 5.35, 5.40, 5.45, 5.48]
    )
    futures = create_futures_3m(
        contracts=["H27", "M27", "U27", "Z27"],
        prices=[5.56, 5.55, 5.54, 5.53]
    )
    ois = create_ois(
        tenors=["2Y", "5Y", "10Y", "30Y"],
        rates=[5.45, 5.38, 5.28, 5.15]
    )
    all_results.append(test_curve("Extended Deposits (1M-12M) + 3M Futures + OIS", deposits, futures, ois))

    # ========================================================================
    # TEST 3: Short deposits, mixed 1M+3M futures, OIS
    # ========================================================================
    deposits = create_deposits(
        tenors=["1M", "2M"],
        rates=[5.30, 5.32]
    )
    # Note: 1M futures use SERIAL_MONTHLY contract type
    futures_1m = create_futures_1m(
        contracts=["J26", "K26", "M26"],  # Apr, May, Jun 2026
        prices=[94.65, 94.63, 94.62]
    )
    futures_3m = create_futures_3m(
        contracts=["U26", "Z26", "H27"],
        prices=[94.58, 94.57, 94.56]
    )
    ois = create_ois(
        tenors=["2Y", "5Y", "10Y"],
        rates=[5.45, 5.38, 5.28]
    )
    all_results.append(test_curve("Short Deposits + Mixed 1M/3M Futures + OIS",
                                   deposits, futures_1m + futures_3m, ois))

    # ========================================================================
    # TEST 4: Overnight + short deposits, 3M futures, OIS
    # ========================================================================
    deposits = create_deposits(
        tenors=["1D", "1W", "2W", "1M", "2M", "3M"],
        rates=[5.28, 5.29, 5.295, 5.30, 5.32, 5.35]
    )
    futures = create_futures_3m(
        contracts=["H26", "M26", "U26", "Z26"],
        prices=[94.62, 94.60, 94.58, 94.57]
    )
    ois = create_ois(
        tenors=["2Y", "3Y", "5Y", "7Y", "10Y"],
        rates=[5.45, 5.43, 5.38, 5.34, 5.28]
    )
    all_results.append(test_curve("Very Short Deposits (1D-3M) + 3M Futures + OIS", deposits, futures, ois))

    # ========================================================================
    # TEST 5: Standard deposits, many 3M futures, sparse OIS
    # ========================================================================
    deposits = create_deposits(
        tenors=["1M", "3M", "6M"],
        rates=[5.30, 5.35, 5.40]
    )
    futures = create_futures_3m(
        contracts=["H26", "M26", "U26", "Z26", "H27", "M27", "U27", "Z27", "H28", "M28"],
        prices=[94.62, 94.60, 94.58, 94.57, 94.56, 94.55, 94.54, 94.53, 94.52, 94.51]
    )
    ois = create_ois(
        tenors=["5Y", "10Y", "30Y"],
        rates=[5.38, 5.28, 5.15]
    )
    all_results.append(test_curve("Standard Deposits + Many Futures (10) + Sparse OIS", deposits, futures, ois))

    # ========================================================================
    # TEST 6: Deposits up to 6M, early OIS start (18M)
    # ========================================================================
    deposits = create_deposits(
        tenors=["1M", "2M", "3M", "6M"],
        rates=[5.30, 5.32, 5.35, 5.40]
    )
    futures = create_futures_3m(
        contracts=["H26", "M26", "U26"],
        prices=[94.62, 94.60, 94.58]
    )
    ois = create_ois(
        tenors=["18M", "2Y", "3Y", "5Y", "10Y"],
        rates=[5.42, 5.45, 5.43, 5.38, 5.28]
    )
    all_results.append(test_curve("Deposits + Few Futures + Early OIS (18M start)", deposits, futures, ois))

    # ========================================================================
    # TEST 7: Dense short end, 3M futures, long OIS
    # ========================================================================
    deposits = create_deposits(
        tenors=["1W", "2W", "1M", "2M", "3M", "4M", "5M", "6M"],
        rates=[5.29, 5.295, 5.30, 5.32, 5.35, 5.37, 5.39, 5.40]
    )
    futures = create_futures_3m(
        contracts=["H26", "M26", "U26", "Z26"],
        prices=[94.62, 94.60, 94.58, 94.57]
    )
    ois = create_ois(
        tenors=["2Y", "5Y", "10Y", "15Y", "20Y", "25Y", "30Y", "40Y", "50Y"],
        rates=[5.45, 5.38, 5.28, 5.22, 5.18, 5.16, 5.15, 5.14, 5.13]
    )
    all_results.append(test_curve("Dense Short End + Futures + Long OIS (up to 50Y)", deposits, futures, ois))

    # ========================================================================
    # SUMMARY REPORT
    # ========================================================================
    print("\n\n")
    print("="*100)
    print("SUMMARY: ALL ROBUSTNESS TESTS")
    print("="*100)
    print()
    print(f"{'Test Name':<60s} {'Status':<10s} {'Pass Rate':<12s} {'Max Error':<15s}")
    print("-"*100)

    for r in all_results:
        status_display = f"[{r['status']}]"
        pass_rate_display = f"{r['pass_rate']:.1f}%"
        max_err_display = f"{r['max_error']:.2e}" if r['status'] != "CONSTRUCTION_FAILED" else "N/A"
        print(f"{r['name']:<60s} {status_display:<10s} {pass_rate_display:<12s} {max_err_display:<15s}")

    print("-"*100)

    total_tests = len(all_results)
    passed_tests = sum(1 for r in all_results if r['status'] == "PASS")
    failed_tests = total_tests - passed_tests

    print(f"\nOverall: {passed_tests}/{total_tests} configurations PASSED ({passed_tests/total_tests*100:.1f}%)")

    if failed_tests > 0:
        print(f"\n[WARNING] {failed_tests} configuration(s) failed to meet tolerance {SWAP_TOL:.2e}")
        print("\nFailed configurations:")
        for r in all_results:
            if r['status'] != "PASS":
                print(f"  - {r['name']}")
                if r['status'] != "CONSTRUCTION_FAILED":
                    print(f"    Deposits: {r['dep_pass']}/{r['dep_total']}, Futures: {r['fut_pass']}/{r['fut_total']}, OIS: {r['ois_pass']}/{r['ois_total']}")
    else:
        print(f"\n[SUCCESS] All {total_tests} configurations achieve tolerance {SWAP_TOL:.2e}")

    print("="*100)


if __name__ == "__main__":
    run_all_tests()
