"""
Bloomberg Benchmark Test: 20Y USD/GBP XCCY Swap (22-JAN-2026)

This test provides a comprehensive benchmark for pricing and risk calculations
of a 20Y USD/GBP cross-currency basis swap using Bloomberg market data.

Two operational modes:
1. Live Bloomberg mode (pytest -m market_data): Fetches real market data
2. Hardcoded benchmark mode (default): Uses saved market data for validation

Test Coverage:
- Multi-curve model construction (USD OIS, GBP OIS, XCCY BASIS)
- 20Y par XCCY basis swap creation using market spreads
- VALUE computation (present value in USD)
- DELTA ladder extraction for all three curves

Market Data Date: 22-JAN-2026
Swap Structure: 20Y USD/GBP XCCY basis swap with quarterly payments

Author: Generated for Cavour ADRates Library
"""

import pytest
import json
from pathlib import Path

from cavour.utils.date import Date
from cavour.utils.global_types import SwapTypes, RequestTypes, CurveTypes
from cavour.utils.day_count import DayCountTypes
from cavour.utils.frequency import FrequencyTypes
from cavour.utils.calendar import BusDayAdjustTypes
from cavour.utils.currency import CurrencyTypes
from cavour.models.models import Model
from cavour.trades.rates.xccy_basis_swap import XccyBasisSwap
from cavour.market.position.engine import Engine
from cavour.marketdata.market_data_engine import MarketCurveBuilder
from cavour.marketdata.market_data_constants import MARKET_DATA, FX_MARKET_DATA


##############################################################################
# CONFIGURATION
##############################################################################

MARKET_DATA_DATE = Date(22, 1, 2026)
TENOR = "20Y"
NOTIONAL_USD = 100_000_000  # $100MM USD notional

# Tolerance for hardcoded benchmark validation
TOLERANCE_VALUE_PCT = 0.0001  # 0.01% relative tolerance for PV
TOLERANCE_DELTA_ABS = 1.0     # 1 USD absolute tolerance per delta pillar

# Output file for saving market data and results
OUTPUT_FILE = Path(__file__).parent / "test_xccy_20y_benchmark_data.json"


##############################################################################
# HARDCODED MARKET DATA (populated after first live run)
##############################################################################

HARDCODED_DATA = {
    "market_date": "22-JAN-2026",
    "usd_ois_sofr": {
        "tenor_list": [],  # To be populated
        "px_list": []      # To be populated
    },
    "gbp_ois_sonia": {
        "tenor_list": [],  # To be populated
        "px_list": []      # To be populated
    },
    "gbpusd_xccy_basis": {
        "tenor_list": [],  # To be populated
        "px_list": []      # To be populated (basis spreads in bps)
    },
    "fx_spot": {
        "GBPUSD": 0.0      # To be populated (GBP per USD)
    },
    "expected_results": {
        "value_usd": 0.0,  # To be populated
        "delta_usd_ois": [],     # To be populated
        "delta_gbp_ois": [],     # To be populated
        "delta_basis": []        # To be populated
    }
}


##############################################################################
# HELPER FUNCTIONS
##############################################################################

def build_xccy_model_from_bloomberg(value_dt: Date) -> tuple:
    """
    Build multi-curve model for XCCY swap from Bloomberg data.

    Fetches:
    - USD OIS SOFR curve
    - GBP OIS SONIA curve
    - GBP/USD XCCY basis spreads
    - GBP/USD spot FX rate

    Args:
        value_dt: Valuation date

    Returns:
        tuple: (model, usd_params, gbp_params, xccy_params, spot_fx)
    """
    builder = MarketCurveBuilder(MARKET_DATA, FX_MARKET_DATA)

    # Fetch all market data
    usd_params = builder.get_curve_inputs('USD_OIS_SOFR', value_dt)
    gbp_params = builder.get_curve_inputs('GBP_OIS_SONIA', value_dt)
    xccy_params = builder.get_curve_inputs('GBPUSD_XCCY_SONIA_SOFR', value_dt)
    fx_rates = builder.get_fx_rates(['GBPUSD'], value_dt)
    spot_fx = fx_rates['GBPUSD']['price']

    # Build model
    model = Model(value_dt)

    # Build OIS curves with AD and gamma
    model.build_curve(**usd_params, use_ad=True, compute_gamma=True)
    model.build_curve(**gbp_params, use_ad=True, compute_gamma=True)

    # Build FX
    model.build_fx(['GBPUSD'], [spot_fx])

    # Build XCCY curve with AD and gamma
    # NOTE: Curve name is 'USD_GBP_BASIS' (domestic_foreign format - matches engine expectation)
    model.build_xccy_curve(
        name='USD_GBP_BASIS',
        domestic_curve_name='USD_OIS_SOFR',
        foreign_curve_name='GBP_OIS_SONIA',
        basis_spreads=xccy_params['px_list'],
        tenor_list=xccy_params['tenor_list'],
        spot_fx=spot_fx,
        domestic_freq_type=FrequencyTypes.QUARTERLY,
        foreign_freq_type=FrequencyTypes.QUARTERLY,
        domestic_dc_type=DayCountTypes.ACT_360,
        foreign_dc_type=DayCountTypes.ACT_365F,
        bus_day_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        interp_type=xccy_params.get('interp_type'),
        use_ad=True,
        compute_gamma=True
    )

    return model, usd_params, gbp_params, xccy_params, spot_fx


def build_xccy_model_from_hardcoded(value_dt: Date, data: dict) -> tuple:
    """
    Build multi-curve model from hardcoded market data.

    Args:
        value_dt: Valuation date
        data: Hardcoded market data dictionary

    Returns:
        tuple: (model, spot_fx)
    """
    model = Model(value_dt)

    # Build USD OIS curve
    model.build_curve(
        name='USD_OIS_SOFR',
        px_list=data['usd_ois_sofr']['px_list'],
        tenor_list=data['usd_ois_sofr']['tenor_list'],
        spot_days=0,
        swap_type=SwapTypes.PAY,
        fixed_dcc_type=DayCountTypes.ACT_360,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_360,
        bus_day_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        use_ad=True,
        compute_gamma=True
    )

    # Build GBP OIS curve
    model.build_curve(
        name='GBP_OIS_SONIA',
        px_list=data['gbp_ois_sonia']['px_list'],
        tenor_list=data['gbp_ois_sonia']['tenor_list'],
        spot_days=0,
        swap_type=SwapTypes.PAY,
        fixed_dcc_type=DayCountTypes.ACT_365F,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_365F,
        bus_day_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        use_ad=True,
        compute_gamma=True
    )

    # Get spot FX
    spot_fx = data['fx_spot']['GBPUSD']

    # Build FX
    model.build_fx(['GBPUSD'], [spot_fx])

    # Build XCCY curve
    model.build_xccy_curve(
        name='USD_GBP_BASIS',
        domestic_curve_name='USD_OIS_SOFR',
        foreign_curve_name='GBP_OIS_SONIA',
        basis_spreads=data['gbpusd_xccy_basis']['px_list'],
        tenor_list=data['gbpusd_xccy_basis']['tenor_list'],
        spot_fx=spot_fx,
        domestic_freq_type=FrequencyTypes.QUARTERLY,
        foreign_freq_type=FrequencyTypes.QUARTERLY,
        domestic_dc_type=DayCountTypes.ACT_360,
        foreign_dc_type=DayCountTypes.ACT_365F,
        bus_day_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        use_ad=True,
        compute_gamma=True
    )

    return model, spot_fx


def create_20y_par_xccy_swap(value_dt: Date, xccy_params: dict, spot_fx: float) -> XccyBasisSwap:
    """
    Create 20Y par XCCY basis swap using market basis spread.

    Args:
        value_dt: Valuation date
        xccy_params: XCCY curve parameters (must include 20Y tenor)
        spot_fx: GBP/USD spot FX rate (GBP per USD)

    Returns:
        XccyBasisSwap instance
    """
    # Extract 20Y basis spread
    if TENOR not in xccy_params['tenor_list']:
        raise ValueError(f"20Y tenor not found in XCCY params. Available: {xccy_params['tenor_list']}")

    idx = xccy_params['tenor_list'].index(TENOR)
    basis_bps = xccy_params['px_list'][idx]
    foreign_spread = basis_bps / 10000.0  # Convert bps to decimal

    # Calculate GBP notional using spot FX
    gbp_notional = NOTIONAL_USD * spot_fx

    swap = XccyBasisSwap(
        effective_dt=value_dt,
        term_dt_or_tenor=TENOR,
        domestic_notional=NOTIONAL_USD,      # USD (domestic)
        foreign_notional=gbp_notional,        # GBP (foreign)
        domestic_spread=0.0,                  # No spread on USD leg
        foreign_spread=foreign_spread,        # Basis spread on GBP leg
        domestic_freq_type=FrequencyTypes.QUARTERLY,
        foreign_freq_type=FrequencyTypes.QUARTERLY,
        domestic_dc_type=DayCountTypes.ACT_360,
        foreign_dc_type=DayCountTypes.ACT_365F,
        domestic_floating_index=CurveTypes.USD_OIS_SOFR,
        foreign_floating_index=CurveTypes.GBP_OIS_SONIA,
        domestic_currency=CurrencyTypes.USD,
        foreign_currency=CurrencyTypes.GBP
    )

    return swap


def print_section_header(title: str):
    """Print formatted section header."""
    print(f"\n{'='*80}")
    print(f"{title.center(80)}")
    print(f"{'='*80}\n")


def save_market_data_and_results(usd_params: dict, gbp_params: dict, xccy_params: dict,
                                  spot_fx: float, result: any, output_file: Path):
    """
    Save market data and results to JSON file for hardcoding.

    Args:
        usd_params: USD OIS curve parameters
        gbp_params: GBP OIS curve parameters
        xccy_params: XCCY curve parameters
        spot_fx: Spot FX rate
        result: Engine computation result
        output_file: Path to output JSON file
    """
    data = {
        "market_date": str(MARKET_DATA_DATE),
        "tenor": TENOR,
        "notional_usd": NOTIONAL_USD,
        "usd_ois_sofr": {
            "tenor_list": usd_params['tenor_list'],
            "px_list": [float(px) for px in usd_params['px_list']]
        },
        "gbp_ois_sonia": {
            "tenor_list": gbp_params['tenor_list'],
            "px_list": [float(px) for px in gbp_params['px_list']]
        },
        "gbpusd_xccy_basis": {
            "tenor_list": xccy_params['tenor_list'],
            "px_list": [float(px) for px in xccy_params['px_list']]
        },
        "fx_spot": {
            "GBPUSD": float(spot_fx)
        },
        "results": {
            "value_usd": float(result.value.amount),
            "delta_usd_ois": [float(x) for x in result.risk.USD_OIS_SOFR.risk_ladder],
            "delta_gbp_ois": [float(x) for x in result.risk.GBP_OIS_SONIA.risk_ladder],
            "delta_basis": [float(x) for x in result.risk.USD_GBP_BASIS.risk_ladder]
        }
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nMarket data and results saved to: {output_file}")


##############################################################################
# TEST CLASS
##############################################################################

@pytest.mark.market_data
class TestXCCY20YBloombergLive:
    """
    Live Bloomberg test - fetches real market data and computes results.

    Requires Bloomberg connection. Run with: pytest -m market_data

    This test:
    1. Fetches market data from Bloomberg for 22-JAN-2026
    2. Builds multi-curve model (USD OIS, GBP OIS, XCCY BASIS)
    3. Creates 20Y par XCCY swap
    4. Computes VALUE and DELTA ladders
    5. Saves results to JSON file for hardcoding
    """

    def test_live_bloomberg_20y_xccy(self):
        """Fetch Bloomberg data, price 20Y XCCY swap, extract VALUE and DELTA."""
        print_section_header("LIVE BLOOMBERG TEST: 20Y USD/GBP XCCY SWAP")

        value_dt = MARKET_DATA_DATE

        # Build model from Bloomberg
        print("Fetching market data from Bloomberg...")
        model, usd_params, gbp_params, xccy_params, spot_fx = build_xccy_model_from_bloomberg(value_dt)

        print("\nMarket Data Summary:")
        print(f"  Date: {value_dt}")
        print(f"  USD OIS tenors: {usd_params['tenor_list']}")
        print(f"  GBP OIS tenors: {gbp_params['tenor_list']}")
        print(f"  XCCY BASIS tenors: {xccy_params['tenor_list']}")
        print(f"  GBPUSD spot FX: {spot_fx:.6f}")

        # Extract 20Y rates
        usd_20y_idx = usd_params['tenor_list'].index('20Y')
        gbp_20y_idx = gbp_params['tenor_list'].index('20Y')
        xccy_20y_idx = xccy_params['tenor_list'].index('20Y')

        print(f"\n20Y Market Rates:")
        print(f"  USD OIS SOFR:  {usd_params['px_list'][usd_20y_idx]:.4f}%")
        print(f"  GBP OIS SONIA: {gbp_params['px_list'][gbp_20y_idx]:.4f}%")
        print(f"  XCCY Basis:    {xccy_params['px_list'][xccy_20y_idx]:.2f} bps")

        # Create 20Y par swap
        print(f"\nCreating 20Y par XCCY swap...")
        swap = create_20y_par_xccy_swap(value_dt, xccy_params, spot_fx)

        gbp_notional = NOTIONAL_USD * spot_fx
        print(f"  Notional USD: ${NOTIONAL_USD:,.0f}")
        print(f"  Notional GBP: £{gbp_notional:,.0f}")
        print(f"  Tenor: {TENOR}")
        print(f"  Foreign spread: {swap._foreign_spread*10000:.2f} bps")

        # Compute VALUE and DELTA using Engine
        print("\nComputing VALUE and DELTA...")
        engine = Engine(model)
        result = engine.compute(swap, [RequestTypes.VALUE, RequestTypes.DELTA])

        # Extract results
        pv_usd = result.value.amount
        delta_usd_ois = result.risk.USD_OIS_SOFR.risk_ladder
        delta_gbp_ois = result.risk.GBP_OIS_SONIA.risk_ladder
        delta_basis = result.risk.USD_GBP_BASIS.risk_ladder

        # Print results
        print_section_header("RESULTS")

        print(f"VALUE:")
        print(f"  PV (USD): ${pv_usd:,.2f}")
        print(f"  PV relative to notional: {pv_usd/NOTIONAL_USD*100:.6f}%")

        print(f"\nDELTA LADDERS:")
        print(f"\nUSD OIS SOFR:")
        for i, tenor in enumerate(usd_params['tenor_list']):
            print(f"  {tenor:>5}: {delta_usd_ois[i]:>12,.2f}")

        print(f"\nGBP OIS SONIA:")
        for i, tenor in enumerate(gbp_params['tenor_list']):
            print(f"  {tenor:>5}: {delta_gbp_ois[i]:>12,.2f}")

        print(f"\nUSD_GBP_BASIS:")
        for i, tenor in enumerate(xccy_params['tenor_list']):
            print(f"  {tenor:>5}: {delta_basis[i]:>12,.2f}")

        # Save to file
        save_market_data_and_results(usd_params, gbp_params, xccy_params,
                                      spot_fx, result, OUTPUT_FILE)

        print_section_header("TEST PASSED")
        print("To use this data for hardcoded benchmark:")
        print(f"1. Copy the data from: {OUTPUT_FILE}")
        print("2. Paste into HARDCODED_DATA dictionary in this file")
        print("3. Run: pytest test_xccy_20y_benchmark_bbg.py::TestXCCY20YBenchmark")

        # Basic sanity checks
        assert abs(pv_usd) < NOTIONAL_USD * 0.01, \
            f"PV should be small for par swap. Got: ${pv_usd:,.2f}"

        print("\n[PASS] Live Bloomberg test completed successfully!")


class TestXCCY20YBenchmark:
    """
    Hardcoded benchmark test - validates against saved market data.

    Does NOT require Bloomberg connection. Run with: pytest (no marker)

    This test:
    1. Uses hardcoded market data from 22-JAN-2026
    2. Builds multi-curve model
    3. Creates 20Y par XCCY swap
    4. Computes VALUE and DELTA ladders
    5. Validates against expected hardcoded results
    """

    def test_hardcoded_benchmark_20y_xccy(self):
        """Validate 20Y XCCY swap pricing using hardcoded market data."""
        print_section_header("HARDCODED BENCHMARK TEST: 20Y USD/GBP XCCY SWAP")

        # Check if hardcoded data is populated
        if not HARDCODED_DATA['usd_ois_sofr']['tenor_list']:
            pytest.skip("Hardcoded data not yet populated. Run live Bloomberg test first.")

        value_dt = MARKET_DATA_DATE

        # Build model from hardcoded data
        print("Building model from hardcoded market data...")
        model, spot_fx = build_xccy_model_from_hardcoded(value_dt, HARDCODED_DATA)

        print(f"\nHardcoded Market Data:")
        print(f"  Date: {HARDCODED_DATA['market_date']}")
        print(f"  USD OIS tenors: {len(HARDCODED_DATA['usd_ois_sofr']['tenor_list'])}")
        print(f"  GBP OIS tenors: {len(HARDCODED_DATA['gbp_ois_sonia']['tenor_list'])}")
        print(f"  XCCY BASIS tenors: {len(HARDCODED_DATA['gbpusd_xccy_basis']['tenor_list'])}")
        print(f"  GBPUSD spot FX: {spot_fx:.6f}")

        # Create xccy_params for swap creation
        xccy_params = {
            'tenor_list': HARDCODED_DATA['gbpusd_xccy_basis']['tenor_list'],
            'px_list': HARDCODED_DATA['gbpusd_xccy_basis']['px_list']
        }

        # Create 20Y par swap
        print(f"\nCreating 20Y par XCCY swap...")
        swap = create_20y_par_xccy_swap(value_dt, xccy_params, spot_fx)

        # Compute VALUE and DELTA
        print("\nComputing VALUE and DELTA...")
        engine = Engine(model)
        result = engine.compute(swap, [RequestTypes.VALUE, RequestTypes.DELTA])

        # Extract results
        pv_usd = result.value.amount
        delta_usd_ois = result.risk.USD_OIS_SOFR.risk_ladder
        delta_gbp_ois = result.risk.GBP_OIS_SONIA.risk_ladder
        delta_basis = result.risk.USD_GBP_BASIS.risk_ladder

        # Expected results
        expected_pv = HARDCODED_DATA['expected_results']['value_usd']
        expected_delta_usd = HARDCODED_DATA['expected_results']['delta_usd_ois']
        expected_delta_gbp = HARDCODED_DATA['expected_results']['delta_gbp_ois']
        expected_delta_basis = HARDCODED_DATA['expected_results']['delta_basis']

        # Print results
        print_section_header("RESULTS VALIDATION")

        print(f"VALUE:")
        print(f"  Computed PV (USD): ${pv_usd:,.2f}")
        print(f"  Expected PV (USD): ${expected_pv:,.2f}")
        print(f"  Difference:        ${pv_usd - expected_pv:,.2f}")
        print(f"  Relative error:    {abs(pv_usd - expected_pv)/abs(NOTIONAL_USD)*100:.6f}%")

        # Validate VALUE
        rel_error_pv = abs(pv_usd - expected_pv) / abs(NOTIONAL_USD)
        assert rel_error_pv < TOLERANCE_VALUE_PCT, \
            f"PV error {rel_error_pv*100:.4f}% exceeds tolerance {TOLERANCE_VALUE_PCT*100:.4f}%"

        print(f"\n[PASS] VALUE matches within {TOLERANCE_VALUE_PCT*100:.4f}% tolerance")

        # Validate DELTA ladders
        print(f"\nDELTA VALIDATION:")

        # USD OIS
        print(f"\nUSD OIS SOFR DELTA:")
        max_delta_usd_error = 0
        for i in range(len(delta_usd_ois)):
            diff = abs(delta_usd_ois[i] - expected_delta_usd[i])
            max_delta_usd_error = max(max_delta_usd_error, diff)
            if diff > TOLERANCE_DELTA_ABS:
                print(f"  {HARDCODED_DATA['usd_ois_sofr']['tenor_list'][i]:>5}: "
                      f"Computed={delta_usd_ois[i]:>12,.2f}, "
                      f"Expected={expected_delta_usd[i]:>12,.2f}, "
                      f"Diff={diff:>8,.2f} [FAIL]")
        assert max_delta_usd_error < TOLERANCE_DELTA_ABS, \
            f"USD OIS DELTA error {max_delta_usd_error:.2f} exceeds tolerance {TOLERANCE_DELTA_ABS:.2f}"
        print(f"  Max error: {max_delta_usd_error:.2f} < {TOLERANCE_DELTA_ABS:.2f} [PASS]")

        # GBP OIS
        print(f"\nGBP OIS SONIA DELTA:")
        max_delta_gbp_error = 0
        for i in range(len(delta_gbp_ois)):
            diff = abs(delta_gbp_ois[i] - expected_delta_gbp[i])
            max_delta_gbp_error = max(max_delta_gbp_error, diff)
            if diff > TOLERANCE_DELTA_ABS:
                print(f"  {HARDCODED_DATA['gbp_ois_sonia']['tenor_list'][i]:>5}: "
                      f"Computed={delta_gbp_ois[i]:>12,.2f}, "
                      f"Expected={expected_delta_gbp[i]:>12,.2f}, "
                      f"Diff={diff:>8,.2f} [FAIL]")
        assert max_delta_gbp_error < TOLERANCE_DELTA_ABS, \
            f"GBP OIS DELTA error {max_delta_gbp_error:.2f} exceeds tolerance {TOLERANCE_DELTA_ABS:.2f}"
        print(f"  Max error: {max_delta_gbp_error:.2f} < {TOLERANCE_DELTA_ABS:.2f} [PASS]")

        # BASIS
        print(f"\nGBP_USD_BASIS DELTA:")
        max_delta_basis_error = 0
        for i in range(len(delta_basis)):
            diff = abs(delta_basis[i] - expected_delta_basis[i])
            max_delta_basis_error = max(max_delta_basis_error, diff)
            if diff > TOLERANCE_DELTA_ABS:
                print(f"  {HARDCODED_DATA['gbpusd_xccy_basis']['tenor_list'][i]:>5}: "
                      f"Computed={delta_basis[i]:>12,.2f}, "
                      f"Expected={expected_delta_basis[i]:>12,.2f}, "
                      f"Diff={diff:>8,.2f} [FAIL]")
        assert max_delta_basis_error < TOLERANCE_DELTA_ABS, \
            f"BASIS DELTA error {max_delta_basis_error:.2f} exceeds tolerance {TOLERANCE_DELTA_ABS:.2f}"
        print(f"  Max error: {max_delta_basis_error:.2f} < {TOLERANCE_DELTA_ABS:.2f} [PASS]")

        print_section_header("BENCHMARK TEST PASSED")
        print(f"All results match hardcoded benchmark within tolerance!")
        print(f"  VALUE tolerance: {TOLERANCE_VALUE_PCT*100:.4f}%")
        print(f"  DELTA tolerance: {TOLERANCE_DELTA_ABS:.2f} USD")


##############################################################################
# MAIN EXECUTION
##############################################################################

if __name__ == "__main__":
    """
    Run tests directly (outside pytest framework).

    Usage:
        # Run live Bloomberg test (requires connection):
        pytest test_xccy_20y_benchmark_bbg.py -m market_data -v

        # Run hardcoded benchmark test (no Bloomberg required):
        pytest test_xccy_20y_benchmark_bbg.py::TestXCCY20YBenchmark -v
    """
    print("""
    ================================================================================
    20Y USD/GBP XCCY SWAP BENCHMARK TEST
    ================================================================================

    Two test modes:

    1. Live Bloomberg Mode (pytest -m market_data):
       - Fetches real market data from Bloomberg
       - Prices 20Y XCCY swap
       - Extracts VALUE and DELTA ladders
       - Saves results to JSON for hardcoding

    2. Hardcoded Benchmark Mode (default):
       - Uses saved market data (no Bloomberg required)
       - Validates VALUE and DELTA against expected results
       - Ensures reproducibility and regression testing

    Market Data Date: 22-JAN-2026
    Swap: 20Y USD/GBP cross-currency basis swap (quarterly payments)
    Notional: $100MM USD

    ================================================================================
    """)

    pytest.main([__file__, "-v", "--tb=short"])
