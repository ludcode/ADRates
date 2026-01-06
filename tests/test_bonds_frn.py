"""
Comprehensive tests for Bond and FRN (Floating Rate Note) products.

Tests cover both fixed-coupon bonds and floating-rate notes with:
- Construction with various conventions (frequencies, day counts)
- Edge cases (zero coupon, high coupons, short/long maturities)
- Multiple currencies and calendars

Reference:
- cavour/trades/credit/bond.py
- cavour/trades/credit/frn.py
"""

import pytest
from cavour.utils.date import Date
from cavour.utils.day_count import DayCountTypes
from cavour.utils.frequency import FrequencyTypes
from cavour.utils.calendar import CalendarTypes, BusDayAdjustTypes, DateGenRuleTypes
from cavour.utils.currency import CurrencyTypes
from cavour.utils.global_types import CurveTypes
from cavour.trades.credit.bond import Bond
from cavour.trades.credit.frn import FRN


@pytest.fixture
def value_date():
    """Common valuation date"""
    return Date(15, 6, 2024)


class TestBondConstruction:
    """Test fixed-coupon bond construction"""

    def test_create_5y_annual_bond(self, value_date):
        """Test creating a 5Y annual coupon bond"""
        bond = Bond(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            coupon=0.05,
            freq_type=FrequencyTypes.ANNUAL,
            dc_type=DayCountTypes.ACT_365F,
            currency=CurrencyTypes.GBP
        )

        assert bond is not None
        assert bond._coupon == 0.05
        assert bond._freq_type == FrequencyTypes.ANNUAL

    def test_create_10y_semiannual_bond(self, value_date):
        """Test creating a 10Y semiannual coupon bond"""
        bond = Bond(
            issue_dt=value_date,
            maturity_dt_or_tenor="10Y",
            coupon=0.04,
            freq_type=FrequencyTypes.SEMI_ANNUAL,
            dc_type=DayCountTypes.ACT_360,
            currency=CurrencyTypes.USD
        )

        assert bond._coupon == 0.04
        assert bond._freq_type == FrequencyTypes.SEMI_ANNUAL
        assert bond._currency == CurrencyTypes.USD

    def test_zero_coupon_bond(self, value_date):
        """Test creating a zero-coupon bond"""
        bond = Bond(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            coupon=0.0,
            freq_type=FrequencyTypes.ZERO,
            dc_type=DayCountTypes.ACT_365F,
            currency=CurrencyTypes.GBP
        )

        assert bond._coupon == 0.0
        assert bond._freq_type == FrequencyTypes.ZERO

    def test_bond_with_explicit_maturity_date(self, value_date):
        """Test creating bond with explicit maturity date"""
        maturity_date = Date(15, 6, 2034)
        bond = Bond(
            issue_dt=value_date,
            maturity_dt_or_tenor=maturity_date,
            coupon=0.05,
            freq_type=FrequencyTypes.ANNUAL,
            dc_type=DayCountTypes.ACT_365F,
            currency=CurrencyTypes.GBP
        )

        assert bond._maturity_dt == maturity_date

    def test_bond_different_face_values(self, value_date):
        """Test bonds with different face values"""
        face_values = [100.0, 1000.0, 10000.0]

        for fv in face_values:
            bond = Bond(
                issue_dt=value_date,
                maturity_dt_or_tenor="5Y",
                coupon=0.05,
                freq_type=FrequencyTypes.ANNUAL,
                dc_type=DayCountTypes.ACT_365F,
                currency=CurrencyTypes.GBP,
                face_value=fv
            )
            assert bond._face_value == fv


class TestBondFrequencies:
    """Test bonds with different payment frequencies"""

    @pytest.mark.parametrize("freq_type", [
        FrequencyTypes.ANNUAL,
        FrequencyTypes.SEMI_ANNUAL,
        FrequencyTypes.QUARTERLY,
        FrequencyTypes.MONTHLY
    ])
    def test_bond_all_frequencies(self, value_date, freq_type):
        """Test bond construction with all payment frequencies"""
        bond = Bond(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            coupon=0.05,
            freq_type=freq_type,
            dc_type=DayCountTypes.ACT_365F,
            currency=CurrencyTypes.GBP
        )

        assert bond._freq_type == freq_type


class TestBondDayCountConventions:
    """Test bonds with different day count conventions"""

    @pytest.mark.parametrize("dc_type", [
        DayCountTypes.ACT_365F,
        DayCountTypes.ACT_360,
        DayCountTypes.ACT_ACT_ISDA,
        DayCountTypes.THIRTY_360_BOND
    ])
    def test_bond_all_day_counts(self, value_date, dc_type):
        """Test bond construction with all day count conventions"""
        bond = Bond(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            coupon=0.05,
            freq_type=FrequencyTypes.ANNUAL,
            dc_type=dc_type,
            currency=CurrencyTypes.GBP
        )

        assert bond._dc_type == dc_type


class TestBondEdgeCases:
    """Test bond edge cases and boundary conditions"""

    def test_short_maturity_bond_1month(self, value_date):
        """Test bond with very short maturity (1 month)"""
        bond = Bond(
            issue_dt=value_date,
            maturity_dt_or_tenor="1M",
            coupon=0.05,
            freq_type=FrequencyTypes.MONTHLY,
            dc_type=DayCountTypes.ACT_365F,
            currency=CurrencyTypes.GBP
        )

        assert bond is not None

    def test_long_maturity_bond_30y(self, value_date):
        """Test bond with very long maturity (30 years)"""
        bond = Bond(
            issue_dt=value_date,
            maturity_dt_or_tenor="30Y",
            coupon=0.04,
            freq_type=FrequencyTypes.SEMI_ANNUAL,
            dc_type=DayCountTypes.ACT_365F,
            currency=CurrencyTypes.GBP
        )

        assert bond is not None

    def test_high_coupon_bond(self, value_date):
        """Test bond with high coupon rate (15%)"""
        bond = Bond(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            coupon=0.15,
            freq_type=FrequencyTypes.ANNUAL,
            dc_type=DayCountTypes.ACT_365F,
            currency=CurrencyTypes.GBP
        )

        assert bond._coupon == 0.15

    def test_low_coupon_bond(self, value_date):
        """Test bond with very low coupon rate (0.5%)"""
        bond = Bond(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            coupon=0.005,
            freq_type=FrequencyTypes.ANNUAL,
            dc_type=DayCountTypes.ACT_365F,
            currency=CurrencyTypes.GBP
        )

        assert bond._coupon == 0.005


class TestBondMultipleCurrencies:
    """Test bonds in different currencies"""

    @pytest.mark.parametrize("currency", [
        CurrencyTypes.GBP,
        CurrencyTypes.USD,
        CurrencyTypes.EUR
    ])
    def test_bond_all_currencies(self, value_date, currency):
        """Test bond construction in all major currencies"""
        bond = Bond(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            coupon=0.05,
            freq_type=FrequencyTypes.ANNUAL,
            dc_type=DayCountTypes.ACT_365F,
            currency=currency
        )

        assert bond._currency == currency


# ================================================================================
# FRN (Floating Rate Note) Tests
# ================================================================================


class TestFRNConstruction:
    """Test FRN construction with various conventions"""

    def test_create_5y_quarterly_frn(self, value_date):
        """Test creating a 5Y FRN with quarterly resets"""
        frn = FRN(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            quoted_margin=0.005,  # 50bp spread
            freq_type=FrequencyTypes.QUARTERLY,
            dc_type=DayCountTypes.ACT_360,
            currency=CurrencyTypes.USD,
            floating_index=CurveTypes.USD_OIS_SOFR
        )

        assert frn is not None
        assert frn._quoted_margin == 0.005
        assert frn._freq_type == FrequencyTypes.QUARTERLY

    def test_create_frn_semiannual_sonia(self, value_date):
        """Test creating FRN with semiannual resets on SONIA"""
        frn = FRN(
            issue_dt=value_date,
            maturity_dt_or_tenor="10Y",
            quoted_margin=0.0025,  # 25bp spread
            freq_type=FrequencyTypes.SEMI_ANNUAL,
            dc_type=DayCountTypes.ACT_365F,
            currency=CurrencyTypes.GBP,
            floating_index=CurveTypes.GBP_OIS_SONIA
        )

        assert frn._quoted_margin == 0.0025
        assert frn._floating_index == CurveTypes.GBP_OIS_SONIA

    def test_zero_margin_frn(self, value_date):
        """Test FRN with zero quoted margin"""
        frn = FRN(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            quoted_margin=0.0,
            freq_type=FrequencyTypes.QUARTERLY,
            dc_type=DayCountTypes.ACT_360,
            currency=CurrencyTypes.USD,
            floating_index=CurveTypes.USD_OIS_SOFR
        )

        assert frn._quoted_margin == 0.0

    def test_frn_with_cap(self, value_date):
        """Test FRN with coupon cap"""
        frn = FRN(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            quoted_margin=0.005,
            freq_type=FrequencyTypes.QUARTERLY,
            dc_type=DayCountTypes.ACT_360,
            currency=CurrencyTypes.USD,
            floating_index=CurveTypes.USD_OIS_SOFR,
            cap_rate=0.08  # 8% cap
        )

        assert frn._cap_rate == 0.08

    def test_frn_with_floor(self, value_date):
        """Test FRN with coupon floor"""
        frn = FRN(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            quoted_margin=0.005,
            freq_type=FrequencyTypes.QUARTERLY,
            dc_type=DayCountTypes.ACT_360,
            currency=CurrencyTypes.USD,
            floating_index=CurveTypes.USD_OIS_SOFR,
            floor_rate=0.01  # 1% floor
        )

        assert frn._floor_rate == 0.01

    def test_frn_with_cap_and_floor(self, value_date):
        """Test FRN with both cap and floor (collar)"""
        frn = FRN(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            quoted_margin=0.005,
            freq_type=FrequencyTypes.QUARTERLY,
            dc_type=DayCountTypes.ACT_360,
            currency=CurrencyTypes.USD,
            floating_index=CurveTypes.USD_OIS_SOFR,
            cap_rate=0.08,
            floor_rate=0.01
        )

        assert frn._cap_rate == 0.08
        assert frn._floor_rate == 0.01


class TestFRNFrequencies:
    """Test FRNs with different reset frequencies"""

    @pytest.mark.parametrize("freq_type", [
        FrequencyTypes.MONTHLY,
        FrequencyTypes.QUARTERLY,
        FrequencyTypes.SEMI_ANNUAL,
        FrequencyTypes.ANNUAL
    ])
    def test_frn_all_frequencies(self, value_date, freq_type):
        """Test FRN construction with all reset frequencies"""
        frn = FRN(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            quoted_margin=0.005,
            freq_type=freq_type,
            dc_type=DayCountTypes.ACT_360,
            currency=CurrencyTypes.USD,
            floating_index=CurveTypes.USD_OIS_SOFR
        )

        assert frn._freq_type == freq_type


class TestFRNEdgeCases:
    """Test FRN edge cases and boundary conditions"""

    def test_short_maturity_frn(self, value_date):
        """Test FRN with short maturity (1 year)"""
        frn = FRN(
            issue_dt=value_date,
            maturity_dt_or_tenor="1Y",
            quoted_margin=0.005,
            freq_type=FrequencyTypes.QUARTERLY,
            dc_type=DayCountTypes.ACT_360,
            currency=CurrencyTypes.USD,
            floating_index=CurveTypes.USD_OIS_SOFR
        )

        assert frn is not None

    def test_long_maturity_frn(self, value_date):
        """Test FRN with long maturity (30 years)"""
        frn = FRN(
            issue_dt=value_date,
            maturity_dt_or_tenor="30Y",
            quoted_margin=0.005,
            freq_type=FrequencyTypes.QUARTERLY,
            dc_type=DayCountTypes.ACT_360,
            currency=CurrencyTypes.USD,
            floating_index=CurveTypes.USD_OIS_SOFR
        )

        assert frn is not None

    def test_high_margin_frn(self, value_date):
        """Test FRN with high quoted margin (500bp)"""
        frn = FRN(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            quoted_margin=0.05,
            freq_type=FrequencyTypes.QUARTERLY,
            dc_type=DayCountTypes.ACT_360,
            currency=CurrencyTypes.USD,
            floating_index=CurveTypes.USD_OIS_SOFR
        )

        assert frn._quoted_margin == 0.05

    def test_negative_margin_frn(self, value_date):
        """Test FRN with negative quoted margin"""
        frn = FRN(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            quoted_margin=-0.001,  # -10bp spread
            freq_type=FrequencyTypes.QUARTERLY,
            dc_type=DayCountTypes.ACT_360,
            currency=CurrencyTypes.USD,
            floating_index=CurveTypes.USD_OIS_SOFR
        )

        assert frn._quoted_margin == -0.001


class TestFRNMultipleIndices:
    """Test FRNs linked to different floating indices"""

    def test_frn_sofr_linked(self, value_date):
        """Test FRN linked to USD SOFR"""
        frn = FRN(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            quoted_margin=0.005,
            freq_type=FrequencyTypes.QUARTERLY,
            dc_type=DayCountTypes.ACT_360,
            currency=CurrencyTypes.USD,
            floating_index=CurveTypes.USD_OIS_SOFR
        )

        assert frn._floating_index == CurveTypes.USD_OIS_SOFR

    def test_frn_sonia_linked(self, value_date):
        """Test FRN linked to GBP SONIA"""
        frn = FRN(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            quoted_margin=0.0025,
            freq_type=FrequencyTypes.QUARTERLY,
            dc_type=DayCountTypes.ACT_365F,
            currency=CurrencyTypes.GBP,
            floating_index=CurveTypes.GBP_OIS_SONIA
        )

        assert frn._floating_index == CurveTypes.GBP_OIS_SONIA

    def test_frn_estr_linked(self, value_date):
        """Test FRN linked to EUR ESTR"""
        frn = FRN(
            issue_dt=value_date,
            maturity_dt_or_tenor="5Y",
            quoted_margin=0.003,
            freq_type=FrequencyTypes.QUARTERLY,
            dc_type=DayCountTypes.ACT_360,
            currency=CurrencyTypes.EUR,
            floating_index=CurveTypes.EUR_OIS_ESTR
        )

        assert frn._floating_index == CurveTypes.EUR_OIS_ESTR
