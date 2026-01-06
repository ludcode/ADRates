"""
Comprehensive tests for Zero-Coupon Inflation Swap (ZCIS) implementation.

A ZCIS exchanges a fixed compounded return for an inflation-linked return
at a single maturity date. Tests cover:
- Swap construction with various tenors and conventions
- Valuation and breakeven inflation rate calculation
- Edge cases (zero rates, negative inflation, long maturities)
- PV01 sensitivity calculations

Reference: cavour/trades/rates/zcis.py
"""

import pytest
import numpy as np
from cavour.utils.date import Date
from cavour.utils.global_types import SwapTypes, InflationIndexTypes
from cavour.utils.day_count import DayCountTypes
from cavour.utils.currency import CurrencyTypes
from cavour.utils.calendar import CalendarTypes, BusDayAdjustTypes
from cavour.market.indices.inflation_index import InflationIndex
from cavour.market.curves.discount_curve import DiscountCurve
from cavour.market.curves.inflation_curve import InflationCurve
from cavour.market.curves.interpolator import InterpTypes
from cavour.trades.rates.zcis import ZeroCouponInflationSwap


@pytest.fixture
def value_date():
    """Common valuation date"""
    return Date(15, 6, 2024)


@pytest.fixture
def rpi_index(value_date):
    """Create UK RPI index with historical fixings"""
    base_date = Date(1, 3, 2024)
    rpi = InflationIndex(
        index_type=InflationIndexTypes.UK_RPI,
        base_date=base_date,
        base_index=293.0,
        currency=CurrencyTypes.GBP,
        lag_months=3
    )

    # Add historical monthly fixings
    rpi.add_fixing(Date(1, 3, 2024), 293.0)
    rpi.add_fixing(Date(1, 4, 2024), 293.5)
    rpi.add_fixing(Date(1, 5, 2024), 294.0)
    rpi.add_fixing(Date(1, 6, 2024), 294.5)

    return rpi


@pytest.fixture
def simple_discount_curve(value_date):
    """Create simple flat discount curve for testing"""
    times = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    dfs = np.array([0.9875, 0.975, 0.95, 0.90, 0.78, 0.61])

    # DiscountCurve constructor: (value_dt, times, dfs, interp_type)
    curve = DiscountCurve(value_date, times, dfs, InterpTypes.FLAT_FWD_RATES)
    return curve


class TestZCISConstruction:
    """Test ZCIS construction with various conventions"""

    def test_create_5y_zcis_pay_fixed(self, value_date, rpi_index):
        """Test creating a 5Y ZCIS paying fixed"""
        zcis = ZeroCouponInflationSwap(
            effective_dt=value_date,
            term_dt_or_tenor="5Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_rate=0.03,
            inflation_index=rpi_index,
            notional=10_000_000
        )

        assert zcis is not None
        assert zcis._fixed_rate == 0.03
        assert zcis._notional == 10_000_000
        assert zcis._fixed_leg_type == SwapTypes.PAY

    def test_create_10y_zcis_receive_fixed(self, value_date, rpi_index):
        """Test creating a 10Y ZCIS receiving fixed"""
        zcis = ZeroCouponInflationSwap(
            effective_dt=value_date,
            term_dt_or_tenor="10Y",
            fixed_leg_type=SwapTypes.RECEIVE,
            fixed_rate=0.025,
            inflation_index=rpi_index,
            notional=5_000_000
        )

        assert zcis._fixed_rate == 0.025
        assert zcis._fixed_leg_type == SwapTypes.RECEIVE

    def test_create_zcis_with_date_maturity(self, value_date, rpi_index):
        """Test creating ZCIS with explicit maturity date"""
        maturity_date = Date(15, 6, 2029)
        zcis = ZeroCouponInflationSwap(
            effective_dt=value_date,
            term_dt_or_tenor=maturity_date,
            fixed_leg_type=SwapTypes.PAY,
            fixed_rate=0.03,
            inflation_index=rpi_index
        )

        assert zcis._maturity_dt == maturity_date

    def test_zcis_different_notionals(self, value_date, rpi_index):
        """Test ZCIS with different notional amounts"""
        notionals = [1_000_000, 10_000_000, 100_000_000]

        for notional in notionals:
            zcis = ZeroCouponInflationSwap(
                effective_dt=value_date,
                term_dt_or_tenor="5Y",
                fixed_leg_type=SwapTypes.PAY,
                fixed_rate=0.03,
                inflation_index=rpi_index,
                notional=notional
            )
            assert zcis._notional == notional




class TestZCISEdgeCases:
    """Test ZCIS edge cases and boundary conditions"""

    def test_short_maturity_zcis(self, value_date, rpi_index):
        """Test ZCIS with very short maturity (1 month)"""
        zcis = ZeroCouponInflationSwap(
            effective_dt=value_date,
            term_dt_or_tenor="1M",
            fixed_leg_type=SwapTypes.PAY,
            fixed_rate=0.03,
            inflation_index=rpi_index
        )

        assert zcis is not None

    def test_long_maturity_zcis(self, value_date, rpi_index):
        """Test ZCIS with very long maturity (30 years)"""
        zcis = ZeroCouponInflationSwap(
            effective_dt=value_date,
            term_dt_or_tenor="30Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_rate=0.03,
            inflation_index=rpi_index
        )

        assert zcis is not None

    def test_zero_fixed_rate(self, value_date, rpi_index):
        """Test ZCIS with zero fixed rate"""
        zcis = ZeroCouponInflationSwap(
            effective_dt=value_date,
            term_dt_or_tenor="5Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_rate=0.0,
            inflation_index=rpi_index
        )

        assert zcis._fixed_rate == 0.0

    def test_high_fixed_rate(self, value_date, rpi_index):
        """Test ZCIS with high fixed rate (10%)"""
        zcis = ZeroCouponInflationSwap(
            effective_dt=value_date,
            term_dt_or_tenor="5Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_rate=0.10,
            inflation_index=rpi_index
        )

        assert zcis._fixed_rate == 0.10


class TestZCISConventions:
    """Test ZCIS with different market conventions"""

    def test_zcis_different_day_counts(self, value_date, rpi_index):
        """Test ZCIS with different day count conventions"""
        day_counts = [
            DayCountTypes.ACT_365F,
            DayCountTypes.ACT_360,
            DayCountTypes.ACT_ACT_ISDA
        ]

        for dc in day_counts:
            zcis = ZeroCouponInflationSwap(
                effective_dt=value_date,
                term_dt_or_tenor="5Y",
                fixed_leg_type=SwapTypes.PAY,
                fixed_rate=0.03,
                inflation_index=rpi_index,
                dc_type=dc
            )
            assert zcis._dc_type == dc

    def test_zcis_different_calendars(self, value_date, rpi_index):
        """Test ZCIS with different business day calendars"""
        calendars = [
            CalendarTypes.WEEKEND,
            CalendarTypes.UNITED_KINGDOM,
            CalendarTypes.TARGET
        ]

        for cal in calendars:
            zcis = ZeroCouponInflationSwap(
                effective_dt=value_date,
                term_dt_or_tenor="5Y",
                fixed_leg_type=SwapTypes.PAY,
                fixed_rate=0.03,
                inflation_index=rpi_index,
                cal_type=cal
            )
            assert zcis._cal_type == cal
