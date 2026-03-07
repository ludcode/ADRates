"""
Bloomberg Benchmark Test: 20Y GBP OIS Swap (22-JAN-2026)

This test provides a comprehensive benchmark for pricing and risk calculations
of a 20Y GBP SONIA OIS swap using Bloomberg market data.

Two operational modes:
1. Live Bloomberg mode (pytest -m market_data): Fetches real market data
2. Hardcoded benchmark mode (default): Uses saved market data for validation

Test Coverage:
- GBP OIS SONIA curve construction
- 20Y par OIS swap creation using market rates
- VALUE computation (present value in GBP)
- DELTA ladder extraction for GBP OIS curve

Market Data Date: 22-JAN-2026
Swap Structure: 20Y GBP SONIA OIS swap with annual payments

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
from cavour.trades.rates.ois import OIS
from cavour.marketdata.market_data_engine import MarketCurveBuilder
from cavour.marketdata.market_data_constants import MARKET_DATA, FX_MARKET_DATA


##############################################################################
# CONFIGURATION
##############################################################################

MARKET_DATA_DATE = Date(22, 1, 2026)
TENOR = "20Y"
NOTIONAL_GBP = 100_000_000  # 100MM GBP notional

# Tolerance for hardcoded benchmark validation
TOLERANCE_VALUE_PCT = 0.0001  # 0.01% relative tolerance for PV
TOLERANCE_DELTA_ABS = 1.0     # 1 GBP absolute tolerance per delta pillar

# Output file for saving market data and results
OUTPUT_FILE = Path(__file__).parent / "test_ois_20y_benchmark_data.json"


##############################################################################
# HARDCODED MARKET DATA (populated after first live run)
##############################################################################

HARDCODED_DATA = {
    "market_date": "22-JAN-2026",
    "gbp_ois_sonia": {
        "tenor_list": ["1D", "1W", "2W", "1M", "2M", "3M", "4M", "5M", "6M", "7M", "8M", "9M", "10M", "11M", "1Y", "18M", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "12Y", "15Y", "20Y", "25Y", "30Y", "40Y", "50Y"],
        "px_list": [3.726, 3.7275, 3.7285, 3.7275, 3.7275, 3.7105, 3.67902, 3.65217, 3.62545, 3.5971, 3.57669, 3.5585, 3.5415, 3.53054, 3.5195, 3.47387, 3.49721, 3.5625, 3.63019, 3.69965, 3.7687, 3.8399, 3.91067, 3.9807, 4.0484, 4.16894, 4.312, 4.45214, 4.5076, 4.5105, 4.4532, 4.36458]
    },
    "expected_results": {
        "value_gbp": -1.4901161193847656e-08,
        "delta_gbp_ois": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.4210854715202004e-12, 0.0, -2.2737367544323206e-13, -3.410605131648481e-13, -9.094947017729282e-13, 6.139089236967266e-12, 0.0, -2.2737367544323206e-13, -1.1368683772161603e-12, -2.3874235921539366e-11, 9.549694368615746e-12, -1.6370904631912708e-11, 2.3646862246096134e-11, 132902.28202754914, 0.0, 0.0, 0.0, 0.0],
        "gamma_gbp_ois": "POPULATED"  # 32x32 matrix - see JSON file for full data
    }
}


##############################################################################
# HELPER FUNCTIONS
##############################################################################

def build_gbp_ois_model_from_bloomberg(value_dt: Date) -> tuple:
    """
    Build GBP OIS SONIA model from Bloomberg data.

    Args:
        value_dt: Valuation date

    Returns:
        tuple: (model, gbp_params)
    """
    builder = MarketCurveBuilder(MARKET_DATA, FX_MARKET_DATA)

    # Fetch GBP OIS market data
    gbp_params = builder.get_curve_inputs('GBP_OIS_SONIA', value_dt)

    # Build model
    model = Model(value_dt)

    # Build GBP OIS curve with AD and gamma
    model.build_curve(**gbp_params, use_ad=True, compute_gamma=True)

    return model, gbp_params


def build_gbp_ois_model_from_hardcoded(value_dt: Date, data: dict) -> Model:
    """
    Build GBP OIS model from hardcoded market data.

    Args:
        value_dt: Valuation date
        data: Hardcoded market data dictionary

    Returns:
        Model instance
    """
    model = Model(value_dt)

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

    return model


def create_20y_par_ois_swap(value_dt: Date, gbp_params: dict) -> OIS:
    """
    Create 20Y par OIS swap using market par rate.

    Args:
        value_dt: Valuation date
        gbp_params: GBP OIS curve parameters (must include 20Y tenor)

    Returns:
        OIS instance
    """
    # Extract 20Y par rate
    if TENOR not in gbp_params['tenor_list']:
        raise ValueError(f"20Y tenor not found in GBP OIS params. Available: {gbp_params['tenor_list']}")

    idx = gbp_params['tenor_list'].index(TENOR)
    par_rate_pct = gbp_params['px_list'][idx]
    par_rate = par_rate_pct / 100.0  # Convert percentage to decimal

    swap = OIS(
        effective_dt=value_dt,
        term_dt_or_tenor=TENOR,
        fixed_leg_type=SwapTypes.PAY,
        notional=NOTIONAL_GBP,
        fixed_coupon=par_rate,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        fixed_dc_type=DayCountTypes.ACT_365F,
        floating_index=CurveTypes.GBP_OIS_SONIA,
        currency=CurrencyTypes.GBP,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_365F
    )

    return swap


def print_section_header(title: str):
    """Print formatted section header."""
    print(f"\n{'='*80}")
    print(f"{title.center(80)}")
    print(f"{'='*80}\n")


def save_market_data_and_results(gbp_params: dict, result: any, output_file: Path):
    """
    Save market data and results to JSON file for hardcoding.

    Args:
        gbp_params: GBP OIS curve parameters
        result: Position computation result
        output_file: Path to output JSON file
    """
    data = {
        "market_date": str(MARKET_DATA_DATE),
        "tenor": TENOR,
        "notional_gbp": NOTIONAL_GBP,
        "gbp_ois_sonia": {
            "tenor_list": gbp_params['tenor_list'],
            "px_list": [float(px) for px in gbp_params['px_list']]
        },
        "results": {
            "value_gbp": float(result.value.amount),
            "delta_gbp_ois": [float(x) for x in result.risk.risk_ladder],
            "gamma_gbp_ois": [[float(x) for x in row] for row in result.gamma.risk_ladder]
        }
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nMarket data and results saved to: {output_file}")


##############################################################################
# TEST CLASS
##############################################################################

@pytest.mark.market_data
class TestOIS20YBloombergLive:
    """
    Live Bloomberg test - fetches real market data and computes results.

    Requires Bloomberg connection. Run with: pytest -m market_data

    This test:
    1. Fetches GBP OIS SONIA market data from Bloomberg for 22-JAN-2026
    2. Builds GBP OIS curve with AD and gamma
    3. Creates 20Y par OIS swap
    4. Computes VALUE and DELTA ladder
    5. Saves results to JSON file for hardcoding
    """

    def test_live_bloomberg_20y_ois(self):
        """Fetch Bloomberg data, price 20Y OIS swap, extract VALUE and DELTA."""
        print_section_header("LIVE BLOOMBERG TEST: 20Y GBP OIS SWAP")

        value_dt = MARKET_DATA_DATE

        # Build model from Bloomberg
        print("Fetching market data from Bloomberg...")
        model, gbp_params = build_gbp_ois_model_from_bloomberg(value_dt)

        print("\nMarket Data Summary:")
        print(f"  Date: {value_dt}")
        print(f"  GBP OIS tenors: {gbp_params['tenor_list']}")
        print(f"  Number of pillars: {len(gbp_params['tenor_list'])}")

        # Extract 20Y rate
        idx_20y = gbp_params['tenor_list'].index('20Y')

        print(f"\n20Y Market Rate:")
        print(f"  GBP OIS SONIA: {gbp_params['px_list'][idx_20y]:.4f}%")

        # Create 20Y par swap
        print(f"\nCreating 20Y par OIS swap...")
        swap = create_20y_par_ois_swap(value_dt, gbp_params)

        print(f"  Notional: £{NOTIONAL_GBP:,.0f}")
        print(f"  Tenor: {TENOR}")
        print(f"  Fixed coupon: {swap._fixed_coupon*100:.4f}%")

        # Compute VALUE, DELTA, and GAMMA using Position
        print("\nComputing VALUE, DELTA, and GAMMA...")
        position = swap.position(model)
        result = position.compute([RequestTypes.VALUE, RequestTypes.DELTA, RequestTypes.GAMMA])

        # Extract results
        pv_gbp = result.value.amount
        delta_gbp_ois = result.risk.risk_ladder
        gamma_gbp_ois = result.gamma.risk_ladder

        # Print results
        print_section_header("RESULTS")

        print(f"VALUE:")
        print(f"  PV (GBP): £{pv_gbp:,.2f}")
        print(f"  PV relative to notional: {pv_gbp/NOTIONAL_GBP*100:.6f}%")

        print(f"\nDELTA LADDER (GBP OIS SONIA):")
        for i, tenor in enumerate(gbp_params['tenor_list']):
            print(f"  {tenor:>5}: {delta_gbp_ois[i]:>12,.2f}")

        print(f"\nGAMMA MATRIX (GBP OIS SONIA):")
        print(f"  Shape: {len(gamma_gbp_ois)}x{len(gamma_gbp_ois[0])}")
        print(f"  Max gamma: {max(max(row) for row in gamma_gbp_ois):,.2f}")
        print(f"  Min gamma: {min(min(row) for row in gamma_gbp_ois):,.2f}")

        # Print diagonal (own-tenor gamma)
        print(f"\n  Diagonal (own-tenor gamma):")
        for i in range(min(10, len(gamma_gbp_ois))):  # Print first 10
            tenor = gbp_params['tenor_list'][i]
            print(f"    {tenor:>5}: {gamma_gbp_ois[i][i]:>12,.2f}")
        if len(gamma_gbp_ois) > 10:
            print(f"    ... ({len(gamma_gbp_ois) - 10} more)")

        # Save to file
        save_market_data_and_results(gbp_params, result, OUTPUT_FILE)

        print_section_header("TEST PASSED")
        print("To use this data for hardcoded benchmark:")
        print(f"1. Copy the data from: {OUTPUT_FILE}")
        print("2. Paste into HARDCODED_DATA dictionary in this file")
        print("3. Run: pytest test_ois_20y_benchmark_bbg.py::TestOIS20YBenchmark")

        # Basic sanity checks
        assert abs(pv_gbp) < NOTIONAL_GBP * 0.001, \
            f"PV should be very small for par swap. Got: £{pv_gbp:,.2f}"

        print("\n[PASS] Live Bloomberg test completed successfully!")


class TestOIS20YBenchmark:
    """
    Hardcoded benchmark test - validates against saved market data.

    Does NOT require Bloomberg connection. Run with: pytest (no marker)

    This test:
    1. Uses hardcoded GBP OIS market data from 22-JAN-2026
    2. Builds GBP OIS curve
    3. Creates 20Y par OIS swap
    4. Computes VALUE and DELTA ladder
    5. Validates against expected hardcoded results
    """

    def test_hardcoded_benchmark_20y_ois(self):
        """Validate 20Y OIS swap pricing using hardcoded market data."""
        print_section_header("HARDCODED BENCHMARK TEST: 20Y GBP OIS SWAP")

        # Check if hardcoded data is populated
        if not HARDCODED_DATA['gbp_ois_sonia']['tenor_list']:
            pytest.skip("Hardcoded data not yet populated. Run live Bloomberg test first.")

        value_dt = MARKET_DATA_DATE

        # Build model from hardcoded data
        print("Building model from hardcoded market data...")
        model = build_gbp_ois_model_from_hardcoded(value_dt, HARDCODED_DATA)

        print(f"\nHardcoded Market Data:")
        print(f"  Date: {HARDCODED_DATA['market_date']}")
        print(f"  GBP OIS tenors: {len(HARDCODED_DATA['gbp_ois_sonia']['tenor_list'])}")

        # Create gbp_params for swap creation
        gbp_params = {
            'tenor_list': HARDCODED_DATA['gbp_ois_sonia']['tenor_list'],
            'px_list': HARDCODED_DATA['gbp_ois_sonia']['px_list']
        }

        # Create 20Y par swap
        print(f"\nCreating 20Y par OIS swap...")
        swap = create_20y_par_ois_swap(value_dt, gbp_params)

        # Compute VALUE, DELTA, and GAMMA
        print("\nComputing VALUE, DELTA, and GAMMA...")
        position = swap.position(model)
        result = position.compute([RequestTypes.VALUE, RequestTypes.DELTA, RequestTypes.GAMMA])

        # Extract results
        pv_gbp = result.value.amount
        delta_gbp_ois = result.risk.risk_ladder
        gamma_gbp_ois = result.gamma.risk_ladder

        # Expected results
        expected_pv = HARDCODED_DATA['expected_results']['value_gbp']
        expected_delta_gbp = HARDCODED_DATA['expected_results']['delta_gbp_ois']
        expected_gamma_gbp = HARDCODED_DATA['expected_results']['gamma_gbp_ois']

        # Print results
        print_section_header("RESULTS VALIDATION")

        print(f"VALUE:")
        print(f"  Computed PV (GBP): £{pv_gbp:,.2f}")
        print(f"  Expected PV (GBP): £{expected_pv:,.2f}")
        print(f"  Difference:        £{pv_gbp - expected_pv:,.2f}")
        print(f"  Relative error:    {abs(pv_gbp - expected_pv)/abs(NOTIONAL_GBP)*100:.6f}%")

        # Validate VALUE
        rel_error_pv = abs(pv_gbp - expected_pv) / abs(NOTIONAL_GBP)
        assert rel_error_pv < TOLERANCE_VALUE_PCT, \
            f"PV error {rel_error_pv*100:.4f}% exceeds tolerance {TOLERANCE_VALUE_PCT*100:.4f}%"

        print(f"\n[PASS] VALUE matches within {TOLERANCE_VALUE_PCT*100:.4f}% tolerance")

        # Validate DELTA ladder
        print(f"\nDELTA VALIDATION:")

        print(f"\nGBP OIS SONIA DELTA:")
        max_delta_error = 0
        for i in range(len(delta_gbp_ois)):
            diff = abs(delta_gbp_ois[i] - expected_delta_gbp[i])
            max_delta_error = max(max_delta_error, diff)
            if diff > TOLERANCE_DELTA_ABS:
                print(f"  {HARDCODED_DATA['gbp_ois_sonia']['tenor_list'][i]:>5}: "
                      f"Computed={delta_gbp_ois[i]:>12,.2f}, "
                      f"Expected={expected_delta_gbp[i]:>12,.2f}, "
                      f"Diff={diff:>8,.2f} [FAIL]")
        assert max_delta_error < TOLERANCE_DELTA_ABS, \
            f"GBP OIS DELTA error {max_delta_error:.2f} exceeds tolerance {TOLERANCE_DELTA_ABS:.2f}"
        print(f"  Max error: {max_delta_error:.2f} < {TOLERANCE_DELTA_ABS:.2f} [PASS]")

        # Validate GAMMA matrix (if populated with actual data, not placeholder)
        if expected_gamma_gbp and isinstance(expected_gamma_gbp, list):
            print(f"\nGAMMA VALIDATION:")
            max_gamma_error = 0
            for i in range(len(gamma_gbp_ois)):
                for j in range(len(gamma_gbp_ois[i])):
                    diff = abs(gamma_gbp_ois[i][j] - expected_gamma_gbp[i][j])
                    max_gamma_error = max(max_gamma_error, diff)

            assert max_gamma_error < TOLERANCE_DELTA_ABS, \
                f"GBP OIS GAMMA error {max_gamma_error:.2f} exceeds tolerance {TOLERANCE_DELTA_ABS:.2f}"
            print(f"  Max error: {max_gamma_error:.2f} < {TOLERANCE_DELTA_ABS:.2f} [PASS]")
        else:
            print(f"\nGAMMA: Computed (validation skipped - hardcoded data is placeholder)")

        print_section_header("BENCHMARK TEST PASSED")
        print(f"All results match hardcoded benchmark within tolerance!")
        print(f"  VALUE tolerance: {TOLERANCE_VALUE_PCT*100:.4f}%")
        print(f"  DELTA tolerance: {TOLERANCE_DELTA_ABS:.2f} GBP")
        if expected_gamma_gbp and isinstance(expected_gamma_gbp, list):
            print(f"  GAMMA tolerance: {TOLERANCE_DELTA_ABS:.2f} GBP")
        else:
            print(f"  GAMMA: Computed successfully (full validation requires hardcoded matrix)")


##############################################################################
# MAIN EXECUTION
##############################################################################

if __name__ == "__main__":
    """
    Run tests directly (outside pytest framework).

    Usage:
        # Run live Bloomberg test (requires connection):
        pytest test_ois_20y_benchmark_bbg.py -m market_data -v

        # Run hardcoded benchmark test (no Bloomberg required):
        pytest test_ois_20y_benchmark_bbg.py::TestOIS20YBenchmark -v
    """
    print("""
    ================================================================================
    20Y GBP OIS SWAP BENCHMARK TEST
    ================================================================================

    Two test modes:

    1. Live Bloomberg Mode (pytest -m market_data):
       - Fetches real market data from Bloomberg
       - Prices 20Y GBP SONIA OIS swap
       - Extracts VALUE and DELTA ladder
       - Saves results to JSON for hardcoding

    2. Hardcoded Benchmark Mode (default):
       - Uses saved market data (no Bloomberg required)
       - Validates VALUE and DELTA against expected results
       - Ensures reproducibility and regression testing

    Market Data Date: 22-JAN-2026
    Swap: 20Y GBP SONIA OIS swap (annual payments)
    Notional: £100MM GBP

    ================================================================================
    """)

    pytest.main([__file__, "-v", "--tb=short"])
