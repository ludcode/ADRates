"""
Regression Test Suite for Refactoring Validation

This test captures the current behavior of OIS and XCCY swap valuations
(VALUE, DELTA, GAMMA) before refactoring engine.py. After each refactoring
step, we run this test to ensure identical behavior within tolerance.

Test Coverage:
1. OIS Swap (USD SOFR) - 10Y tenor with full curve (deposits, futures, OIS)
2. XCCY Swap (USD/EUR) - 10Y tenor with 3 curves (USD OIS, EUR OIS, XCCY basis)

Golden values are stored in JSON format and compared after refactoring.

Tolerances:
- VALUE: 1e-10 (absolute)
- DELTA: 1e-6 (per basis point)
- GAMMA: 1e-6 (per basis point squared)

Author: Generated for Cavour refactoring (March 2026)
"""

import sys
sys.path.insert(0, 'C:\\Projects\\Cavour')

import pytest
import json
import numpy as np
from pathlib import Path

from cavour.utils.date import Date
from cavour.utils.day_count import DayCountTypes
from cavour.utils.global_types import (
    CurveTypes, SwapTypes, FutureContractTypes, RequestTypes
)
from cavour.utils.currency import CurrencyTypes
from cavour.utils.frequency import FrequencyTypes
from cavour.utils.calendar import BusDayAdjustTypes
from cavour.models.models import Model
from cavour.trades.rates.cash_deposit import CashDeposit
from cavour.trades.rates.ir_future import IRFuture
from cavour.trades.rates.ois import OIS
from cavour.trades.rates.xccy_basis_swap import XccyBasisSwap
from cavour.market.position.engine import Engine


##############################################################################
# CONFIGURATION
##############################################################################

VALUE_DATE = Date(15, 3, 2026)
GOLDEN_VALUES_DIR = Path(__file__).parent / "golden_values"
OIS_GOLDEN_FILE = GOLDEN_VALUES_DIR / "ois_swap_golden.json"
XCCY_GOLDEN_FILE = GOLDEN_VALUES_DIR / "xccy_swap_golden.json"

# Tolerances
TOL_VALUE = 1e-10  # Absolute tolerance for VALUE
TOL_DELTA = 1e-6   # Per basis point tolerance for DELTA
TOL_GAMMA = 1e-6   # Per basis point squared tolerance for GAMMA


##############################################################################
# HARDCODED MARKET DATA
##############################################################################

# USD SOFR OIS Market Data (as of 15-MAR-2026)
USD_OIS_DATA = {
    'deposits': {
        'tenors': ['1M', '2M', '3M'],
        'rates': [4.75, 4.78, 4.82]  # Percentage rates
    },
    'futures': {
        'contracts': ['M26', 'U26', 'Z26', 'H27', 'M27', 'U27', 'Z27', 'H28'],
        'prices': [95.15, 95.12, 95.10, 95.08, 95.06, 95.05, 95.04, 95.03]
    },
    'ois': {
        'tenors': ['2Y', '3Y', '5Y', '7Y', '10Y', '15Y', '20Y', '30Y'],
        'rates': [4.85, 4.82, 4.75, 4.68, 4.60, 4.52, 4.48, 4.45]
    }
}

# EUR OIS (ESTR) Market Data (as of 15-MAR-2026)
EUR_OIS_DATA = {
    'deposits': {
        'tenors': ['1M', '2M', '3M'],
        'rates': [3.25, 3.28, 3.32]
    },
    'futures': {
        'contracts': ['M26', 'U26', 'Z26', 'H27', 'M27', 'U27', 'Z27', 'H28'],
        'prices': [96.65, 96.63, 96.62, 96.60, 96.58, 96.57, 96.56, 96.55]
    },
    'ois': {
        'tenors': ['2Y', '3Y', '5Y', '7Y', '10Y', '15Y', '20Y', '30Y'],
        'rates': [3.35, 3.32, 3.28, 3.22, 3.15, 3.08, 3.05, 3.02]
    }
}

# USD/EUR XCCY Basis Spreads (EUR leg spread over ESTR, as of 15-MAR-2026)
XCCY_BASIS_DATA = {
    'tenors': ['2Y', '3Y', '5Y', '7Y', '10Y', '15Y', '20Y', '30Y'],
    'spreads_bps': [-15.5, -14.2, -12.8, -11.5, -10.2, -9.0, -8.5, -8.0]
}

# FX Spot Rate (EUR per USD)
FX_SPOT_EURUSD = 0.92  # 1 USD = 0.92 EUR


##############################################################################
# HELPER FUNCTIONS - OIS SWAP
##############################################################################

def build_usd_ois_model(value_dt: Date) -> Model:
    """
    Build USD SOFR OIS model from hardcoded market data.

    Returns:
        Model with USD_OIS_SOFR curve (AD-enabled with GAMMA)
    """
    from cavour.trades.rates.ois_curve import OISCurve
    from cavour.market.curves.interpolator import InterpTypes

    model = Model(value_dt)

    # Build deposits
    deposits = []
    for tenor, rate in zip(USD_OIS_DATA['deposits']['tenors'],
                          USD_OIS_DATA['deposits']['rates']):
        dep = CashDeposit(
            effective_dt=value_dt,
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
    for contract, price in zip(USD_OIS_DATA['futures']['contracts'],
                              USD_OIS_DATA['futures']['prices']):
        fut = IRFuture(
            effective_dt=value_dt,
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
    for tenor, rate in zip(USD_OIS_DATA['ois']['tenors'],
                          USD_OIS_DATA['ois']['rates']):
        swap = OIS(
            effective_dt=value_dt,
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

    # Build curve with all instruments
    all_instruments = deposits + futures + ois_swaps
    curve = OISCurve(
        value_dt=value_dt,
        instruments=all_instruments,
        interp_type=InterpTypes.LINEAR_ZERO_RATES,
        check_refit=True,  # Verify that deposits, futures, and OIS swaps refit precisely
        use_ad=True,
        compute_gamma=True
    )

    # Add curve to model
    model._curves_dict['USD_OIS_SOFR'] = curve

    return model


def create_10y_ois_swap(value_dt: Date) -> OIS:
    """
    Create 10Y USD SOFR OIS swap (payer) at market rate.

    Returns:
        OIS swap instance
    """
    # Use 10Y market rate from USD_OIS_DATA
    idx = USD_OIS_DATA['ois']['tenors'].index('10Y')
    par_rate = USD_OIS_DATA['ois']['rates'][idx] / 100.0

    swap = OIS(
        effective_dt=value_dt,
        term_dt_or_tenor='10Y',
        fixed_leg_type=SwapTypes.PAY,
        fixed_coupon=par_rate,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        fixed_dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_360,
        notional=100_000_000  # $100MM notional
    )

    return swap


def compute_ois_analytics(swap: OIS, model: Model, value_dt: Date) -> dict:
    """
    Compute VALUE, DELTA, GAMMA for OIS swap using Engine.

    Args:
        swap: OIS swap instance
        model: Model with USD_OIS_SOFR curve
        value_dt: Valuation date

    Returns:
        dict with keys: 'value', 'delta', 'gamma'
    """
    engine = Engine(model)

    # Compute VALUE, DELTA, GAMMA
    reqs = {RequestTypes.VALUE, RequestTypes.DELTA, RequestTypes.GAMMA}
    result = engine.compute(swap, reqs)

    # Extract results
    value = float(result.value.amount)

    # DELTA: Extract risk ladder from USD_OIS_SOFR curve
    delta_ladder = [float(x) for x in result.risk.risk_ladder]
    delta_tenors = result.risk.tenors

    # GAMMA: Extract full matrix
    gamma_matrix = [[float(x) for x in row] for row in result.gamma.risk_ladder]

    return {
        'value': value,
        'delta': {
            'tenors': delta_tenors,
            'risk_ladder': delta_ladder
        },
        'gamma': {
            'risk_ladder': gamma_matrix
        }
    }


##############################################################################
# HELPER FUNCTIONS - XCCY SWAP
##############################################################################

def build_eur_ois_model(value_dt: Date, model: Model) -> Model:
    """
    Build EUR OIS (ESTR) curve and add to existing model.

    Args:
        value_dt: Valuation date
        model: Existing model (will be modified)

    Returns:
        Model with EUR_OIS_ESTR curve added
    """
    from cavour.trades.rates.ois_curve import OISCurve
    from cavour.market.curves.interpolator import InterpTypes

    # Build deposits
    deposits = []
    for tenor, rate in zip(EUR_OIS_DATA['deposits']['tenors'],
                          EUR_OIS_DATA['deposits']['rates']):
        dep = CashDeposit(
            effective_dt=value_dt,
            term_dt_or_tenor=tenor,
            deposit_rate=rate / 100.0,
            dc_type=DayCountTypes.ACT_360,
            floating_index=CurveTypes.EUR_OIS_ESTR,
            currency=CurrencyTypes.EUR,
            notional=1_000_000
        )
        deposits.append(dep)

    # Build futures
    futures = []
    for contract, price in zip(EUR_OIS_DATA['futures']['contracts'],
                              EUR_OIS_DATA['futures']['prices']):
        fut = IRFuture(
            effective_dt=value_dt,
            expiry_date_or_contract=contract,
            futures_price=price,
            contract_type=FutureContractTypes.IMM,
            currency=CurrencyTypes.EUR,
            floating_index=CurveTypes.EUR_OIS_ESTR,
            dc_type=DayCountTypes.ACT_360,
            contract_size=1_000_000
        )
        futures.append(fut)

    # Build OIS swaps
    ois_swaps = []
    for tenor, rate in zip(EUR_OIS_DATA['ois']['tenors'],
                          EUR_OIS_DATA['ois']['rates']):
        swap = OIS(
            effective_dt=value_dt,
            term_dt_or_tenor=tenor,
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=rate / 100.0,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_360,
            floating_index=CurveTypes.EUR_OIS_ESTR,
            currency=CurrencyTypes.EUR,
            float_freq_type=FrequencyTypes.ANNUAL,
            float_dc_type=DayCountTypes.ACT_360,
            notional=1_000_000
        )
        ois_swaps.append(swap)

    # Build curve
    all_instruments = deposits + futures + ois_swaps
    curve = OISCurve(
        value_dt=value_dt,
        instruments=all_instruments,
        interp_type=InterpTypes.LINEAR_ZERO_RATES,
        check_refit=True,  # Verify that deposits, futures, and OIS swaps refit precisely
        use_ad=True,
        compute_gamma=True
    )

    # Add curve to model
    model._curves_dict['EUR_OIS_ESTR'] = curve

    return model


def build_xccy_curve(value_dt: Date, model: Model, spot_fx: float) -> Model:
    """
    Build XCCY basis curve (USD/EUR) and add to model.

    Args:
        value_dt: Valuation date
        model: Existing model with USD_OIS_SOFR and EUR_OIS_ESTR curves
        spot_fx: EUR per USD spot rate

    Returns:
        Model with USD_EUR_BASIS curve added
    """
    # Build XCCY curve using model method
    model.build_xccy_curve(
        name='USD_EUR_BASIS',
        domestic_curve_name='USD_OIS_SOFR',
        foreign_curve_name='EUR_OIS_ESTR',
        basis_spreads=[s / 10000.0 for s in XCCY_BASIS_DATA['spreads_bps']],
        tenor_list=XCCY_BASIS_DATA['tenors'],
        spot_fx=spot_fx,
        domestic_freq_type=FrequencyTypes.QUARTERLY,
        foreign_freq_type=FrequencyTypes.QUARTERLY,
        domestic_dc_type=DayCountTypes.ACT_360,
        foreign_dc_type=DayCountTypes.ACT_360,
        bus_day_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        use_ad=True,
        compute_gamma=True
    )

    return model


def build_multi_curve_model(value_dt: Date) -> tuple:
    """
    Build multi-curve model for XCCY swap.

    Returns:
        tuple: (model, spot_fx)
    """
    # Build USD OIS curve first
    model = build_usd_ois_model(value_dt)

    # Add EUR OIS curve
    model = build_eur_ois_model(value_dt, model)

    # Add FX spot
    model.build_fx(['EURUSD'], [FX_SPOT_EURUSD])

    # Build XCCY basis curve
    model = build_xccy_curve(value_dt, model, FX_SPOT_EURUSD)

    return model, FX_SPOT_EURUSD


def create_10y_xccy_swap(value_dt: Date, spot_fx: float) -> XccyBasisSwap:
    """
    Create 10Y USD/EUR XCCY basis swap at market spread.

    Args:
        value_dt: Valuation date
        spot_fx: EUR per USD spot rate

    Returns:
        XccyBasisSwap instance
    """
    # Use 10Y market basis spread
    idx = XCCY_BASIS_DATA['tenors'].index('10Y')
    basis_bps = XCCY_BASIS_DATA['spreads_bps'][idx]
    foreign_spread = basis_bps / 10000.0

    # USD notional
    usd_notional = 100_000_000  # $100MM
    eur_notional = usd_notional * spot_fx  # Convert to EUR

    swap = XccyBasisSwap(
        effective_dt=value_dt,
        term_dt_or_tenor='10Y',
        domestic_notional=usd_notional,
        foreign_notional=eur_notional,
        domestic_spread=0.0,  # No spread on USD leg
        foreign_spread=foreign_spread,  # Basis spread on EUR leg
        domestic_freq_type=FrequencyTypes.QUARTERLY,
        foreign_freq_type=FrequencyTypes.QUARTERLY,
        domestic_dc_type=DayCountTypes.ACT_360,
        foreign_dc_type=DayCountTypes.ACT_360,
        domestic_floating_index=CurveTypes.USD_OIS_SOFR,
        foreign_floating_index=CurveTypes.EUR_OIS_ESTR,
        domestic_currency=CurrencyTypes.USD,
        foreign_currency=CurrencyTypes.EUR,
        domestic_bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        foreign_bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING
    )

    return swap


def compute_xccy_analytics(swap: XccyBasisSwap, model: Model, value_dt: Date) -> dict:
    """
    Compute VALUE, DELTA, GAMMA for XCCY swap using Engine.

    Args:
        swap: XCCY basis swap instance
        model: Model with USD_OIS_SOFR, EUR_OIS_ESTR, USD_EUR_BASIS curves
        value_dt: Valuation date

    Returns:
        dict with keys: 'value', 'delta_domestic', 'delta_foreign', 'delta_basis', 'gamma'
    """
    engine = Engine(model)

    # Compute VALUE, DELTA, GAMMA
    reqs = {RequestTypes.VALUE, RequestTypes.DELTA, RequestTypes.GAMMA}
    result = engine.compute(swap, reqs)

    # Extract results
    value = float(result.value.amount)

    # DELTA: Extract separate risk ladders for each curve
    # Result.risk contains attributes named after curves: USD_OIS_SOFR, EUR_OIS_ESTR, USD_EUR_BASIS
    delta_domestic = {
        'tenors': result.risk.USD_OIS_SOFR.tenors,
        'risk_ladder': [float(x) for x in result.risk.USD_OIS_SOFR.risk_ladder]
    }

    delta_foreign = {
        'tenors': result.risk.EUR_OIS_ESTR.tenors,
        'risk_ladder': [float(x) for x in result.risk.EUR_OIS_ESTR.risk_ladder]
    }

    # Find the basis curve attribute - it might not be named exactly USD_EUR_BASIS
    basis_attr = None
    for attr in dir(result.risk):
        if 'BASIS' in attr and not attr.startswith('_'):
            basis_attr = attr
            break

    if basis_attr is None:
        raise ValueError(f"Could not find BASIS curve in risk object. Available: {dir(result.risk)}")

    delta_basis = {
        'tenors': getattr(result.risk, basis_attr).tenors,
        'risk_ladder': [float(x) for x in getattr(result.risk, basis_attr).risk_ladder]
    }

    # GAMMA: Extract full matrices for all curves
    # For XCCY, gamma is a Risk object with curve attributes (same structure as delta)
    gamma_matrices = {}
    gamma_matrices['domestic'] = [[float(x) for x in row] for row in result.gamma.USD_OIS_SOFR.risk_ladder]
    gamma_matrices['foreign'] = [[float(x) for x in row] for row in result.gamma.EUR_OIS_ESTR.risk_ladder]
    gamma_matrices['basis'] = [[float(x) for x in row] for row in getattr(result.gamma, basis_attr).risk_ladder]

    return {
        'value': value,
        'delta_domestic': delta_domestic,
        'delta_foreign': delta_foreign,
        'delta_basis': delta_basis,
        'gamma': gamma_matrices  # Dict with keys: 'domestic', 'foreign', 'basis'
    }


##############################################################################
# GOLDEN VALUE MANAGEMENT
##############################################################################

def save_golden_values(data: dict, filepath: Path):
    """Save golden values to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n[OK] Golden values saved to: {filepath}")


def load_golden_values(filepath: Path) -> dict:
    """Load golden values from JSON file."""
    if not filepath.exists():
        return None
    with open(filepath, 'r') as f:
        return json.load(f)


def compare_values(current: dict, golden: dict, name: str) -> bool:
    """
    Compare current results against golden values within tolerance.

    Args:
        current: Current computation results
        golden: Golden reference values
        name: Test name for reporting

    Returns:
        bool: True if all values match within tolerance
    """
    print(f"\n{'='*80}")
    print(f"COMPARING: {name}")
    print(f"{'='*80}")

    all_passed = True

    # Compare VALUE
    val_current = current['value']
    val_golden = golden['value']
    val_diff = abs(val_current - val_golden)
    val_passed = val_diff < TOL_VALUE

    print(f"\nVALUE:")
    print(f"  Current:  {val_current:.12e}")
    print(f"  Golden:   {val_golden:.12e}")
    print(f"  Diff:     {val_diff:.12e}")
    print(f"  Status:   {'[PASS]' if val_passed else '[FAIL]'} (tol={TOL_VALUE:.1e})")

    if not val_passed:
        all_passed = False

    # Compare DELTA (handle both single and multi-curve cases)
    if 'delta' in current:
        # Single curve (OIS)
        delta_ladders = [('DELTA', current['delta'], golden['delta'])]
    else:
        # Multi-curve (XCCY)
        delta_ladders = [
            ('DELTA_DOMESTIC', current['delta_domestic'], golden['delta_domestic']),
            ('DELTA_FOREIGN', current['delta_foreign'], golden['delta_foreign']),
            ('DELTA_BASIS', current['delta_basis'], golden['delta_basis'])
        ]

    for label, delta_curr, delta_gold in delta_ladders:
        ladder_curr = np.array(delta_curr['risk_ladder'])
        ladder_gold = np.array(delta_gold['risk_ladder'])
        diff = np.abs(ladder_curr - ladder_gold)
        max_diff = np.max(diff)
        delta_passed = max_diff < TOL_DELTA

        print(f"\n{label}:")
        print(f"  Max diff: {max_diff:.12e}")
        print(f"  Status:   {'[PASS]' if delta_passed else '[FAIL]'} (tol={TOL_DELTA:.1e})")

        if not delta_passed:
            all_passed = False
            # Print failing tenors
            failing_indices = np.where(diff >= TOL_DELTA)[0]
            if len(failing_indices) > 0:
                print(f"  Failing tenors: {[delta_curr['tenors'][i] for i in failing_indices[:5]]}")

    # Compare GAMMA (handle both single matrix and multi-curve dict)
    if isinstance(current['gamma'], dict) and 'domestic' in current['gamma']:
        # Multi-curve XCCY gamma
        print(f"\nGAMMA:")
        for curve_name in ['domestic', 'foreign', 'basis']:
            gamma_curr = np.array(current['gamma'][curve_name])
            gamma_gold = np.array(golden['gamma'][curve_name])
            diff = np.abs(gamma_curr - gamma_gold)
            max_diff = np.max(diff)
            gamma_passed = max_diff < TOL_GAMMA

            print(f"  {curve_name.upper()}:")
            print(f"    Matrix shape: {gamma_curr.shape}")
            print(f"    Max diff:     {max_diff:.12e}")
            print(f"    Status:       {'[PASS]' if gamma_passed else '[FAIL]'} (tol={TOL_GAMMA:.1e})")

            if not gamma_passed:
                all_passed = False
    else:
        # Single curve OIS gamma
        if 'risk_ladder' in current['gamma']:
            gamma_curr = np.array(current['gamma']['risk_ladder'])
            gamma_gold = np.array(golden['gamma']['risk_ladder'])
        else:
            gamma_curr = np.array(current['gamma'])
            gamma_gold = np.array(golden['gamma'])

        diff = np.abs(gamma_curr - gamma_gold)
        max_diff = np.max(diff)
        gamma_passed = max_diff < TOL_GAMMA

        print(f"\nGAMMA:")
        print(f"  Matrix shape: {gamma_curr.shape}")
        print(f"  Max diff:     {max_diff:.12e}")
        print(f"  Status:       {'[PASS]' if gamma_passed else '[FAIL]'} (tol={TOL_GAMMA:.1e})")

        if not gamma_passed:
            all_passed = False

    # Final verdict
    print(f"\n{'='*80}")
    if all_passed:
        print(f"[PASS] ALL CHECKS PASSED: {name}")
    else:
        print(f"[FAIL] SOME CHECKS FAILED: {name}")
    print(f"{'='*80}\n")

    return all_passed


##############################################################################
# PYTEST TEST FUNCTIONS
##############################################################################

def test_ois_swap_regression():
    """
    Test OIS swap VALUE, DELTA, GAMMA computation.

    First run: Captures golden values
    Subsequent runs: Validates against golden values
    """
    print("\n" + "="*80)
    print("TEST: OIS SWAP REGRESSION (10Y USD SOFR)")
    print("="*80)

    # Build model and swap
    model = build_usd_ois_model(VALUE_DATE)
    swap = create_10y_ois_swap(VALUE_DATE)

    # Compute analytics
    result = compute_ois_analytics(swap, model, VALUE_DATE)

    # Load golden values if they exist
    golden = load_golden_values(OIS_GOLDEN_FILE)

    if golden is None:
        # First run: Save golden values
        save_golden_values(result, OIS_GOLDEN_FILE)
        print("\n[OK] First run: Golden values captured")
        print(f"  VALUE: {result['value']:.12e}")
        print(f"  DELTA tenors: {len(result['delta']['tenors'])}")
        print(f"  GAMMA shape: {np.array(result['gamma']['risk_ladder']).shape}")
        pytest.skip("First run - golden values captured, skipping comparison")
    else:
        # Subsequent runs: Compare against golden
        passed = compare_values(result, golden, "OIS Swap (10Y USD SOFR)")
        assert passed, "OIS swap results do not match golden values!"


def test_xccy_swap_regression():
    """
    Test XCCY swap VALUE, DELTA, GAMMA computation.

    First run: Captures golden values
    Subsequent runs: Validates against golden values
    """
    print("\n" + "="*80)
    print("TEST: XCCY SWAP REGRESSION (10Y USD/EUR)")
    print("="*80)

    # Build model and swap
    model, spot_fx = build_multi_curve_model(VALUE_DATE)
    swap = create_10y_xccy_swap(VALUE_DATE, spot_fx)

    # Compute analytics
    result = compute_xccy_analytics(swap, model, VALUE_DATE)

    # Load golden values if they exist
    golden = load_golden_values(XCCY_GOLDEN_FILE)

    if golden is None:
        # First run: Save golden values
        save_golden_values(result, XCCY_GOLDEN_FILE)
        print("\n[OK] First run: Golden values captured")
        print(f"  VALUE: {result['value']:.12e}")
        print(f"  DELTA_DOMESTIC tenors: {len(result['delta_domestic']['tenors'])}")
        print(f"  DELTA_FOREIGN tenors: {len(result['delta_foreign']['tenors'])}")
        print(f"  DELTA_BASIS tenors: {len(result['delta_basis']['tenors'])}")
        print(f"  GAMMA_DOMESTIC shape: {np.array(result['gamma']['domestic']).shape}")
        print(f"  GAMMA_FOREIGN shape: {np.array(result['gamma']['foreign']).shape}")
        print(f"  GAMMA_BASIS shape: {np.array(result['gamma']['basis']).shape}")
        pytest.skip("First run - golden values captured, skipping comparison")
    else:
        # Subsequent runs: Compare against golden
        passed = compare_values(result, golden, "XCCY Swap (10Y USD/EUR)")
        assert passed, "XCCY swap results do not match golden values!"


##############################################################################
# MAIN EXECUTION (for standalone testing)
##############################################################################

if __name__ == '__main__':
    print("\n" + "="*80)
    print("REFACTORING REGRESSION TEST SUITE")
    print("="*80)

    # Run OIS test
    try:
        test_ois_swap_regression()
    except Exception as e:
        print(f"\n[ERROR] OIS test failed: {e}")

    # Run XCCY test
    try:
        test_xccy_swap_regression()
    except Exception as e:
        print(f"\n[ERROR] XCCY test failed: {e}")

    print("\n" + "="*80)
    print("REGRESSION TESTS COMPLETE")
    print("="*80)
