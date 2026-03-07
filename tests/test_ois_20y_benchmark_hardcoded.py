"""
GBP OIS 20Y Par Swap Benchmark Test (Hardcoded Data Only)

This test validates VALUE, DELTA, and GAMMA calculations against hardcoded
benchmark data. No Bloomberg connection required.

Purpose:
- Regression testing for OIS curve construction and Greek calculations
- Validates algorithmic differentiation accuracy
- Performance benchmarking

Market Data Date: 22-JAN-2026
Swap Structure: 20Y GBP OIS par swap

Author: Generated for Cavour ADRates Library
"""

import pytest

from cavour.utils.date import Date
from cavour.utils.global_types import SwapTypes, RequestTypes, CurveTypes
from cavour.utils.day_count import DayCountTypes
from cavour.utils.frequency import FrequencyTypes
from cavour.utils.calendar import BusDayAdjustTypes, CalendarTypes
from cavour.utils.currency import CurrencyTypes
from cavour.models.models import Model
from cavour.trades.rates.ois import OIS


##############################################################################
# TEST CONFIGURATION
##############################################################################

MARKET_DATA_DATE = Date(22, 1, 2026)
TENOR = "20Y"
NOTIONAL_GBP = 100_000_000

# Tolerances for validation
TOLERANCE_VALUE_PCT = 0.0001  # 0.01% relative error for PV
TOLERANCE_DELTA_ABS = 1.0     # 1 GBP absolute error for delta
TOLERANCE_GAMMA_ABS = 0.01    # 0.01 GBP absolute error for gamma


##############################################################################
# HARDCODED MARKET DATA (from Bloomberg 22-JAN-2026)
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
        "gamma_gbp_ois": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.7755575615628914e-16, 0.0, 2.393918396847994e-16, 2.42861286636753e-17, 1.734723475976807e-17, -1.734723475976807e-16, -4.85722573273506e-17, 2.7755575615628914e-17, 0.0, 7.28583859910259e-17, -1.0061396160665481e-16, 1.457167719820518e-16, 3.3306690738754696e-16, -0.43650313760097126, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.6020852139652106e-16, 0.0, 5.412337245047638e-16, 4.85722573273506e-17, 2.7755575615628914e-17, -3.5388358909926865e-16, -1.1796119636642288e-16, 4.85722573273506e-17, 6.938893903907228e-18, 1.5265566588595902e-16, -1.8041124150158794e-16, 2.7755575615628914e-16, 7.077671781985373e-16, -0.8957533825172465, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.469446951953614e-18, 0.0, 7.494005416219807e-16, -1.0269562977782698e-15, 1.5681900222830336e-15, -5.551115123125783e-16, -1.249000902703301e-16, 5.551115123125783e-17, 2.7755575615628914e-17, 2.498001805406602e-16, -3.2612801348363973e-16, 4.718447854656915e-16, 9.992007221626409e-16, -1.3531654358816556, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.85722573273506e-17, 0.0, 1.0408340855860843e-16, 2.0122792321330962e-16, -3.58046925441613e-15, -3.1363800445660672e-15, -1.249000902703301e-16, 5.551115123125783e-17, -1.3877787807814457e-17, 3.608224830031759e-16, -3.885780586188048e-16, 6.38378239159465e-16, 1.4710455076283324e-15, -1.8397247332532458, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0469737016526324e-16, 0.0, -3.885780586188048e-16, -5.967448757360216e-16, -7.91033905045424e-16, 6.966649479522857e-15, 3.219646771412954e-15, 1.6653345369377348e-16, -1.3877787807814457e-17, 5.134781488891349e-16, -4.996003610813204e-16, 7.494005416219807e-16, 1.887379141862766e-15, -2.339577383954746, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.5102810375396984e-17, 0.0, -1.1796119636642288e-16, -1.3877787807814457e-16, -1.6653345369377348e-16, 3.2057689836051395e-15, -2.220446049250313e-16, 4.8433479449272454e-15, -6.938893903907228e-17, 5.689893001203927e-16, -6.800116025829084e-16, 9.159339953157541e-16, 2.3314683517128287e-15, -2.857298130331144, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.7755575615628914e-17, 0.0, -6.245004513516506e-17, -9.020562075079397e-17, -1.5265566588595902e-16, -1.1102230246251565e-16, 4.496403249731884e-15, 5.551115123125783e-16, -5.551115123125783e-17, 6.661338147750939e-16, -7.216449660063518e-16, 1.1102230246251565e-15, 2.7755575615628914e-15, -3.425132643596397, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.204170427930421e-17, 0.0, 1.0408340855860843e-16, 1.6653345369377348e-16, 2.914335439641036e-16, 3.0531133177191805e-16, 2.7755575615628914e-16, -1.149080830487037e-14, -1.3711254354120683e-14, 1.5376588891058418e-14, -8.881784197001252e-16, 1.3322676295501878e-15, 3.219646771412954e-15, -3.9429167302708468, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.326672684688674e-17, 0.0, 1.5265566588595902e-16, 2.498001805406602e-16, 3.469446951953614e-16, 4.718447854656915e-16, 5.412337245047638e-16, 6.661338147750939e-16, 1.5376588891058418e-14, 1.9817480989559044e-14, -1.0269562977782698e-15, 1.5543122344752192e-15, 3.4416913763379853e-15, -4.517504677430466, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9.71445146547012e-17, 0.0, -1.8735013540549517e-16, -3.0531133177191805e-16, -3.885780586188048e-16, -5.273559366969494e-16, -6.661338147750939e-16, -7.771561172376096e-16, -9.43689570931383e-16, -9.43689570931383e-16, -2.1649348980190553e-15, 1.942890293094024e-15, 4.107825191113079e-15, -5.129490899607331, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5265566588595902e-16, 0.0, 3.191891195797325e-16, 5.134781488891349e-16, 6.938893903907228e-16, 7.771561172376096e-16, 1.0547118733938987e-15, 1.1102230246251565e-15, 1.3322676295501878e-15, 1.4988010832439613e-15, 1.7763568394002505e-15, -1.9539925233402755e-14, 9.992007221626409e-15, -12.17191894076823, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.5388358909926865e-16, 0.0, 7.632783294297951e-16, 1.0824674490095276e-15, 1.5265566588595902e-15, 1.887379141862766e-15, 2.4980018054066022e-15, 2.7755575615628914e-15, 3.219646771412954e-15, 3.6637359812630166e-15, 4.218847493575595e-15, 1.0436096431476471e-14, -2.0605739337042905e-13, -23.405975612189344, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4365031376009711, 0.0, -0.8957533825172465, -1.353165435881655, -1.839724733253246, -2.339577383954746, -2.857298130331144, -3.425132643596397, -3.9429167302708463, -4.517504677430466, -5.129490899607333, -12.171918940768231, -23.405975612189348, -109.71964976600808, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    }
}


##############################################################################
# HELPER FUNCTIONS
##############################################################################

def print_section_header(title):
    """Print a formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"{title:^80}")
    print(f"{'=' * 80}\n")


def create_20y_par_ois_swap(value_dt, gbp_params):
    """
    Create a 20Y GBP OIS par swap using the 20Y market rate.

    Args:
        value_dt: Valuation date
        gbp_params: Dict with 'tenor_list' and 'px_list' for GBP OIS SONIA

    Returns:
        OIS swap object
    """
    # Get 20Y par rate from market data
    tenor_idx = gbp_params['tenor_list'].index(TENOR)
    par_rate = gbp_params['px_list'][tenor_idx] / 100.0  # Convert to decimal

    # Create OIS swap
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


##############################################################################
# TEST CLASS
##############################################################################

class TestOIS20YBenchmarkHardcoded:
    """Test GBP OIS 20Y par swap using hardcoded benchmark data."""

    def setup_method(self):
        """Setup called before each test."""
        print_section_header("Starting Cavour Test Session")

    def teardown_method(self):
        """Teardown called after each test."""
        print_section_header("Ending Cavour Test Session")

    def test_hardcoded_benchmark_20y_ois(self):
        """Validate VALUE, DELTA, and GAMMA against hardcoded benchmark."""
        print_section_header("HARDCODED BENCHMARK TEST: 20Y GBP OIS SWAP")

        value_dt = MARKET_DATA_DATE

        # Build model from hardcoded data
        print("Building model from hardcoded data...")
        model = Model(value_dt)

        # Build GBP OIS curve with AD and gamma
        model.build_curve(
            name='GBP_OIS_SONIA',
            px_list=HARDCODED_DATA['gbp_ois_sonia']['px_list'],
            tenor_list=HARDCODED_DATA['gbp_ois_sonia']['tenor_list'],
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

        # Validate VALUE
        relative_error = abs(pv_gbp - expected_pv) / max(abs(expected_pv), 1e-6)
        assert relative_error < TOLERANCE_VALUE_PCT, \
            f"GBP PV relative error {relative_error*100:.4f}% exceeds tolerance {TOLERANCE_VALUE_PCT*100:.4f}%"
        print(f"  Relative error: {relative_error*100:.6f}% < {TOLERANCE_VALUE_PCT*100:.4f}% [PASS]")

        # Validate DELTA
        print(f"\nDELTA VALIDATION:")
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

        # Validate GAMMA matrix
        print(f"\nGAMMA VALIDATION:")
        max_gamma_error = 0
        for i in range(len(gamma_gbp_ois)):
            for j in range(len(gamma_gbp_ois[i])):
                diff = abs(gamma_gbp_ois[i][j] - expected_gamma_gbp[i][j])
                max_gamma_error = max(max_gamma_error, diff)

        assert max_gamma_error < TOLERANCE_GAMMA_ABS, \
            f"GBP OIS GAMMA error {max_gamma_error:.6f} exceeds tolerance {TOLERANCE_GAMMA_ABS:.6f}"
        print(f"  Max error: {max_gamma_error:.6f} < {TOLERANCE_GAMMA_ABS:.6f} [PASS]")

        # Print summary
        print_section_header("BENCHMARK TEST PASSED")
        print(f"All results match hardcoded benchmark within tolerance!")
        print(f"  VALUE tolerance: {TOLERANCE_VALUE_PCT*100:.4f}%")
        print(f"  DELTA tolerance: {TOLERANCE_DELTA_ABS:.2f} GBP")
        print(f"  GAMMA tolerance: {TOLERANCE_GAMMA_ABS:.6f} GBP")
        print(f"\nMarket Date: {HARDCODED_DATA['market_date']}")
        print(f"Notional: £{NOTIONAL_GBP:,.0f}")
        print(f"Tenor: {TENOR}")
        print(f"Curve Pillars: {len(HARDCODED_DATA['gbp_ois_sonia']['tenor_list'])}")


##############################################################################
# MAIN EXECUTION
##############################################################################

if __name__ == "__main__":
    """
    Run the hardcoded benchmark test.

    Usage with pytest (recommended):
        pytest tests/test_ois_20y_benchmark_hardcoded.py -v
        pytest tests/test_ois_20y_benchmark_hardcoded.py -v -s  # with full output

    Or directly:
        python -m pytest tests/test_ois_20y_benchmark_hardcoded.py -v
    """
    test = TestOIS20YBenchmarkHardcoded()
    test.setup_method()
    test.test_hardcoded_benchmark_20y_ois()
    test.teardown_method()
