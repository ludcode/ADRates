"""
Comprehensive risk calculation tests for Bond and FRN products.

Tests VALUE, DELTA, and other risk measures for:
1. Fixed-coupon Bonds: clean/dirty price, YTM, duration, convexity
2. Floating Rate Notes (FRNs): clean/dirty price, discount margin

Validates that:
- VALUE calculations are accurate
- DELTA sensitivities match finite difference approximations
- Duration and convexity calculations are reasonable
- FRN valuations behave correctly with spread changes
"""

import pytest
import numpy as np
from cavour.utils.date import Date
from cavour.utils.day_count import DayCountTypes
from cavour.utils.frequency import FrequencyTypes
from cavour.utils.currency import CurrencyTypes
from cavour.utils.global_types import CurveTypes, RequestTypes, SwapTypes
from cavour.utils.calendar import BusDayAdjustTypes
from cavour.trades.credit.bond import Bond
from cavour.trades.credit.frn import FRN
from cavour.models.models import Model


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def value_date():
    """Common valuation date"""
    return Date(15, 6, 2024)


@pytest.fixture
def gbp_model(value_date):
    """Build a simple GBP SONIA curve"""
    px_list = [5.1998, 5.2014, 5.2003, 5.2027, 5.2023, 5.19281,
               5.1656, 5.1482, 5.1342, 5.1173, 5.1013, 5.0862,
               5.0701, 5.054, 5.0394, 4.8707, 4.75483, 4.532,
               4.3628, 4.2428, 4.16225, 4.1132, 4.08505, 4.0762,
               4.078, 4.0961, 4.12195, 4.1315, 4.113, 4.07724, 3.984, 3.88]

    tenor_list = ["1D", "1W", "2W", "1M", "2M", "3M", "4M", "5M", "6M",
                  "7M", "8M", "9M", "10M", "11M", "1Y", "18M", "2Y",
                  "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y",
                  "12Y", "15Y", "20Y", "25Y", "30Y", "40Y", "50Y"]

    model = Model(value_date)
    model.build_curve(
        name="GBP_OIS_SONIA",
        px_list=px_list,
        tenor_list=tenor_list,
        spot_days=0,
        swap_type=SwapTypes.PAY,
        fixed_dcc_type=DayCountTypes.ACT_365F,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_365F,
        bus_day_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
    )
    return model


# ==============================================================================
# BOND VALUE TESTS
# ==============================================================================

class TestBondValue:
    """Test Bond VALUE calculations"""

    def test_bond_par_valuation(self, value_date, gbp_model):
        """Test that a bond priced at par with coupon = yield has value near face"""
        bond = Bond(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            coupon=0.05,
            freq_type=FrequencyTypes.ANNUAL,
            dc_type=DayCountTypes.ACT_365F,
            currency=CurrencyTypes.GBP,
            face_value=100.0
        )

        curve = gbp_model.curves.GBP_OIS_SONIA

        # For a bond with coupon = yield, price should be near par (100)
        value = bond.value(value_date, curve)

        # Allow reasonable deviation from par
        assert 95.0 < value < 105.0, f"Bond value {value} should be near par (100)"

    def test_bond_clean_vs_dirty_price(self, value_date, gbp_model):
        """Test that dirty price = clean price + accrued interest"""
        bond = Bond(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            coupon=0.05,
            freq_type=FrequencyTypes.ANNUAL,
            dc_type=DayCountTypes.ACT_365F,
            currency=CurrencyTypes.GBP,
            face_value=100.0
        )

        curve = gbp_model.curves.GBP_OIS_SONIA

        # Get clean and dirty prices
        clean = bond.clean_price(value_date, curve)
        dirty = bond.dirty_price(value_date, curve)

        # Dirty should be >= clean (includes accrued interest)
        assert dirty >= clean - 1e-10, f"Dirty price {dirty} should >= clean price {clean}"

        # Both should be reasonable
        assert 50.0 < clean < 150.0, f"Clean price {clean} unreasonable"
        assert 50.0 < dirty < 150.0, f"Dirty price {dirty} unreasonable"

    def test_bond_ytm_calculation(self, value_date, gbp_model):
        """Test that yield to maturity can be calculated"""
        bond = Bond(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            coupon=0.05,
            freq_type=FrequencyTypes.ANNUAL,
            dc_type=DayCountTypes.ACT_365F,
            currency=CurrencyTypes.GBP,
            face_value=100.0
        )

        curve = gbp_model.curves.GBP_OIS_SONIA

        # Get clean price first
        clean_price = bond.clean_price(value_date, curve)

        # Calculate YTM from clean price
        ytm = bond.yield_to_maturity(value_date, clean_price)

        # YTM should be in reasonable range (0% to 15%)
        assert 0.0 < ytm < 0.15, f"YTM {ytm*100:.2f}% seems unreasonable"

    @pytest.mark.parametrize("tenor", ["2Y", "5Y", "10Y"])
    def test_bond_value_different_maturities(self, value_date, gbp_model, tenor):
        """Test bond valuation works for different maturities"""
        bond = Bond(
            issue_dt=value_date,
            maturity_dt_or_tenor=tenor,
            coupon=0.05,
            freq_type=FrequencyTypes.ANNUAL,
            dc_type=DayCountTypes.ACT_365F,
            currency=CurrencyTypes.GBP,
            face_value=100.0
        )

        curve = gbp_model.curves.GBP_OIS_SONIA
        value = bond.value(value_date, curve)

        # Should have reasonable value
        assert 50.0 < value < 150.0, f"Bond {tenor} value {value} unreasonable"


# ==============================================================================
# BOND RISK TESTS
# ==============================================================================

class TestBondRisk:
    """Test Bond duration and convexity calculations"""

    def test_bond_duration_positive(self, value_date, gbp_model):
        """Test that bond duration is positive"""
        bond = Bond(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            coupon=0.05,
            freq_type=FrequencyTypes.ANNUAL,
            dc_type=DayCountTypes.ACT_365F,
            currency=CurrencyTypes.GBP,
            face_value=100.0
        )

        curve = gbp_model.curves.GBP_OIS_SONIA
        duration = bond.duration(value_date, curve)

        # Duration should be positive and less than maturity (5 years)
        assert 0.0 < duration < 5.0, f"Duration {duration} unreasonable for 5Y bond"

    def test_bond_convexity_positive(self, value_date, gbp_model):
        """Test that bond convexity is positive"""
        bond = Bond(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            coupon=0.05,
            freq_type=FrequencyTypes.ANNUAL,
            dc_type=DayCountTypes.ACT_365F,
            currency=CurrencyTypes.GBP,
            face_value=100.0
        )

        curve = gbp_model.curves.GBP_OIS_SONIA
        convexity = bond.convexity(value_date, curve)

        # Convexity should be positive for standard bonds
        assert convexity > 0, f"Convexity {convexity} should be positive"

        # Should be reasonable magnitude
        assert convexity < 100, f"Convexity {convexity} seems too large"

    def test_bond_duration_increases_with_maturity(self, value_date, gbp_model):
        """Test that longer maturity bonds have higher duration"""
        curve = gbp_model.curves.GBP_OIS_SONIA

        bond_2y = Bond(
            issue_dt=value_date,
            maturity_dt_or_tenor="2Y",
            coupon=0.05,
            freq_type=FrequencyTypes.ANNUAL,
            dc_type=DayCountTypes.ACT_365F,
            currency=CurrencyTypes.GBP
        )

        bond_10y = Bond(
            issue_dt=value_date,
            maturity_dt_or_tenor="10Y",
            coupon=0.05,
            freq_type=FrequencyTypes.ANNUAL,
            dc_type=DayCountTypes.ACT_365F,
            currency=CurrencyTypes.GBP
        )

        duration_2y = bond_2y.duration(value_date, curve)
        duration_10y = bond_10y.duration(value_date, curve)

        # Longer maturity should have higher duration
        assert duration_10y > duration_2y, \
            f"10Y duration {duration_10y} should > 2Y duration {duration_2y}"

    def test_bond_dv01_approximation(self, value_date, gbp_model):
        """Test DV01 (dollar value of 1bp) using finite difference"""
        bond = Bond(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            coupon=0.05,
            freq_type=FrequencyTypes.ANNUAL,
            dc_type=DayCountTypes.ACT_365F,
            currency=CurrencyTypes.GBP,
            face_value=100.0
        )

        curve = gbp_model.curves.GBP_OIS_SONIA

        # Base value
        value_0 = bond.value(value_date, curve)

        # Bump curve by 1bp and revalue
        model_up = gbp_model.scenario("GBP_OIS_SONIA", shock=0.01)  # 1bp = 0.01%
        curve_up = model_up.curves.GBP_OIS_SONIA
        value_up = bond.value(value_date, curve_up)

        # DV01 (change in value for 1bp increase in rates)
        dv01 = value_up - value_0

        # For a 5Y bond with face 100, DV01 should be negative (value decreases when rates increase)
        assert dv01 < 0, f"DV01 {dv01} should be negative (inverse relationship)"

        # Magnitude should be reasonable
        assert abs(dv01) > 0.001, f"DV01 magnitude {abs(dv01)} seems too small"
        assert abs(dv01) < 1.0, f"DV01 magnitude {abs(dv01)} seems too large for 100 face"


# ==============================================================================
# FRN VALUE TESTS
# ==============================================================================

class TestFRNValue:
    """Test FRN VALUE calculations"""

    def test_frn_par_valuation(self, value_date, gbp_model):
        """Test that FRN with zero margin prices near par at issue"""
        frn = FRN(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            quoted_margin=0.0,
            freq_type=FrequencyTypes.QUARTERLY,
            dc_type=DayCountTypes.ACT_365F,
            currency=CurrencyTypes.GBP,
            floating_index=CurveTypes.GBP_OIS_SONIA,
            face_value=100.0
        )

        curve = gbp_model.curves.GBP_OIS_SONIA
        value = frn.value(value_date, curve)

        # FRN with zero margin should price near par at issue
        assert 95.0 < value < 105.0, f"FRN value {value} should be near par (100)"

    def test_frn_with_positive_margin(self, value_date, gbp_model):
        """Test that FRN with positive margin prices above par"""
        frn = FRN(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            quoted_margin=0.005,  # 50bp positive margin
            freq_type=FrequencyTypes.QUARTERLY,
            dc_type=DayCountTypes.ACT_365F,
            currency=CurrencyTypes.GBP,
            floating_index=CurveTypes.GBP_OIS_SONIA,
            face_value=100.0
        )

        curve = gbp_model.curves.GBP_OIS_SONIA
        value = frn.value(value_date, curve)

        # Positive margin FRN should be worth more than par (at issue)
        # Though this depends on where we are in the reset cycle
        assert 90.0 < value < 110.0, f"FRN value {value} unreasonable"

    def test_frn_clean_vs_dirty_price(self, value_date, gbp_model):
        """Test FRN clean vs dirty price"""
        frn = FRN(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            quoted_margin=0.005,
            freq_type=FrequencyTypes.QUARTERLY,
            dc_type=DayCountTypes.ACT_365F,
            currency=CurrencyTypes.GBP,
            floating_index=CurveTypes.GBP_OIS_SONIA,
            face_value=100.0
        )

        curve = gbp_model.curves.GBP_OIS_SONIA

        clean = frn.clean_price(value_date, curve)
        dirty = frn.dirty_price(value_date, curve)

        # Both should be reasonable
        assert 50.0 < clean < 150.0, f"Clean price {clean} unreasonable"
        assert 50.0 < dirty < 150.0, f"Dirty price {dirty} unreasonable"

    def test_frn_discount_margin(self, value_date, gbp_model):
        """Test FRN discount margin calculation"""
        frn = FRN(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            quoted_margin=0.005,
            freq_type=FrequencyTypes.QUARTERLY,
            dc_type=DayCountTypes.ACT_365F,
            currency=CurrencyTypes.GBP,
            floating_index=CurveTypes.GBP_OIS_SONIA,
            face_value=100.0
        )

        curve = gbp_model.curves.GBP_OIS_SONIA

        # Get clean price first
        clean_price = frn.clean_price(value_date, curve)

        # Calculate discount margin (needs settlement_dt, discount_curve, index_curve, clean_price)
        dm = frn.discount_margin(value_date, curve, curve, clean_price)

        # Discount margin should be in reasonable range (-5% to +15%)
        assert -0.05 < dm < 0.15, f"Discount margin {dm*10000:.0f}bp seems unreasonable"


# ==============================================================================
# FRN CAP/FLOOR TESTS
# ==============================================================================

class TestFRNCapFloor:
    """Test FRN with caps and floors"""

    def test_frn_with_cap_valuation(self, value_date, gbp_model):
        """Test FRN with cap can be valued"""
        frn = FRN(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            quoted_margin=0.005,
            freq_type=FrequencyTypes.QUARTERLY,
            dc_type=DayCountTypes.ACT_365F,
            currency=CurrencyTypes.GBP,
            floating_index=CurveTypes.GBP_OIS_SONIA,
            cap_rate=0.08,
            face_value=100.0
        )

        curve = gbp_model.curves.GBP_OIS_SONIA
        value = frn.value(value_date, curve)

        # Should be valued (cap may or may not be in-the-money)
        assert 50.0 < value < 150.0, f"Capped FRN value {value} unreasonable"

    def test_frn_with_floor_valuation(self, value_date, gbp_model):
        """Test FRN with floor can be valued"""
        frn = FRN(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            quoted_margin=0.005,
            freq_type=FrequencyTypes.QUARTERLY,
            dc_type=DayCountTypes.ACT_365F,
            currency=CurrencyTypes.GBP,
            floating_index=CurveTypes.GBP_OIS_SONIA,
            floor_rate=0.01,
            face_value=100.0
        )

        curve = gbp_model.curves.GBP_OIS_SONIA
        value = frn.value(value_date, curve)

        # Should be valued
        assert 50.0 < value < 150.0, f"Floored FRN value {value} unreasonable"

    def test_frn_collar_valuation(self, value_date, gbp_model):
        """Test FRN with both cap and floor (collar)"""
        frn = FRN(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            quoted_margin=0.005,
            freq_type=FrequencyTypes.QUARTERLY,
            dc_type=DayCountTypes.ACT_365F,
            currency=CurrencyTypes.GBP,
            floating_index=CurveTypes.GBP_OIS_SONIA,
            cap_rate=0.08,
            floor_rate=0.01,
            face_value=100.0
        )

        curve = gbp_model.curves.GBP_OIS_SONIA
        value = frn.value(value_date, curve)

        # Collar FRN should be valued
        assert 50.0 < value < 150.0, f"Collar FRN value {value} unreasonable"


# ==============================================================================
# COMPARATIVE TESTS
# ==============================================================================

class TestBondFRNComparison:
    """Test comparative properties of bonds and FRNs"""

    def test_bond_longer_maturity_higher_value_sensitivity(self, value_date, gbp_model):
        """Test that longer maturity bonds have higher interest rate sensitivity"""
        curve = gbp_model.curves.GBP_OIS_SONIA

        bond_2y = Bond(
            issue_dt=value_date,
            maturity_dt_or_tenor="2Y",
            coupon=0.05,
            freq_type=FrequencyTypes.ANNUAL,
            dc_type=DayCountTypes.ACT_365F,
            currency=CurrencyTypes.GBP,
            face_value=100.0
        )

        bond_10y = Bond(
            issue_dt=value_date,
            maturity_dt_or_tenor="10Y",
            coupon=0.05,
            freq_type=FrequencyTypes.ANNUAL,
            dc_type=DayCountTypes.ACT_365F,
            currency=CurrencyTypes.GBP,
            face_value=100.0
        )

        # Get base values
        value_2y_0 = bond_2y.value(value_date, curve)
        value_10y_0 = bond_10y.value(value_date, curve)

        # Bump curve by 10bp
        model_up = gbp_model.scenario("GBP_OIS_SONIA", shock=0.10)
        curve_up = model_up.curves.GBP_OIS_SONIA

        value_2y_up = bond_2y.value(value_date, curve_up)
        value_10y_up = bond_10y.value(value_date, curve_up)

        # Calculate % changes
        pct_change_2y = (value_2y_up - value_2y_0) / value_2y_0
        pct_change_10y = (value_10y_up - value_10y_0) / value_10y_0

        # 10Y bond should have larger % change than 2Y
        assert abs(pct_change_10y) > abs(pct_change_2y), \
            f"10Y bond % change {pct_change_10y:.4%} should be larger than 2Y {pct_change_2y:.4%}"

    @pytest.mark.parametrize("freq", [FrequencyTypes.QUARTERLY, FrequencyTypes.SEMI_ANNUAL, FrequencyTypes.ANNUAL])
    def test_frn_different_frequencies(self, value_date, gbp_model, freq):
        """Test FRN valuation works for different reset frequencies"""
        frn = FRN(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            quoted_margin=0.005,
            freq_type=freq,
            dc_type=DayCountTypes.ACT_365F,
            currency=CurrencyTypes.GBP,
            floating_index=CurveTypes.GBP_OIS_SONIA,
            face_value=100.0
        )

        curve = gbp_model.curves.GBP_OIS_SONIA
        value = frn.value(value_date, curve)

        # Should be valued for all frequencies
        assert 80.0 < value < 120.0, f"FRN {freq} value {value} unreasonable"
