"""
Integration tests for money market curves using mixed instrument types.

Tests the full curve construction workflow:
- Deposit-only curves
- FRA-only curves
- Mixed deposit + FRA + OIS curves
- Curve continuity and smoothness
- Refit accuracy across instrument types
- Model.build_curve() integration
"""

import pytest
import numpy as np
from cavour.utils.date import Date
from cavour.utils.day_count import DayCountTypes
from cavour.utils.global_types import CurveTypes, RequestTypes
from cavour.utils.currency import CurrencyTypes
from cavour.utils.calendar import CalendarTypes, BusDayAdjustTypes
from cavour.trades.rates.cash_deposit import CashDeposit
from cavour.trades.rates.fra import FRA
from cavour.trades.rates.ois import OIS
from cavour.trades.rates.ois_curve import OISCurve
from cavour.market.curves.interpolator import InterpTypes
from cavour.models.models import Model
from cavour.utils.frequency import FrequencyTypes

###############################################################################
# Test Fixtures
###############################################################################


@pytest.fixture
def value_date():
    """Standard valuation date for tests."""
    return Date(15, 6, 2023)


@pytest.fixture
def usd_model(value_date):
    """Create a USD model for testing."""
    return Model(value_date)


###############################################################################
# Deposit-Only Curve Tests
###############################################################################


def test_deposit_only_curve_construction(value_date):
    """Test curve construction from deposits only (0-12M)."""
    deposits = [
        CashDeposit(value_date, "1M", 0.0500, DayCountTypes.ACT_360,
                   CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
        CashDeposit(value_date, "3M", 0.0520, DayCountTypes.ACT_360,
                   CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
        CashDeposit(value_date, "6M", 0.0540, DayCountTypes.ACT_360,
                   CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
        CashDeposit(value_date, "12M", 0.0560, DayCountTypes.ACT_360,
                   CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
    ]

    curve = OISCurve(
        value_dt=value_date,
        instruments=deposits,
        interp_type=InterpTypes.LINEAR_ZERO_RATES,
        check_refit=False  # Will check manually
    )

    # Curve should be constructed successfully
    assert len(curve._times) >= len(deposits)
    assert len(curve._dfs) >= len(deposits)
    # First DF should be 1.0
    assert abs(curve._dfs[0] - 1.0) < 1e-10


def test_deposit_curve_refit_accuracy(value_date):
    """Test that deposits reprice accurately after bootstrapping."""
    deposits = [
        CashDeposit(value_date, "1M", 0.0500, DayCountTypes.ACT_360,
                   CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
        CashDeposit(value_date, "3M", 0.0520, DayCountTypes.ACT_360,
                   CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
        CashDeposit(value_date, "6M", 0.0540, DayCountTypes.ACT_360,
                   CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
    ]

    curve = OISCurve(
        value_dt=value_date,
        instruments=deposits,
        interp_type=InterpTypes.LINEAR_ZERO_RATES,
        check_refit=True  # Will raise if refit fails
    )

    # If we reach here, refit check passed
    assert True


def test_deposit_curve_via_model(usd_model):
    """Test deposit curve construction via Model.build_curve()."""
    usd_model.build_curve(
        name="USD_OIS_SOFR",
        px_list=[5.00, 5.20, 5.40, 5.60],
        tenor_list=["1M", "3M", "6M", "12M"],
        instrument_type="DEPOSIT",
        fixed_dcc_type=DayCountTypes.ACT_360
    )

    # Curve should be accessible
    curve = usd_model.curves.USD_OIS_SOFR
    assert curve is not None
    assert len(curve.swap_rates) == 4


###############################################################################
# FRA-Only Curve Tests
###############################################################################


def test_fra_only_curve_construction(value_date):
    """Test curve construction from FRAs only."""
    fras = [
        FRA(value_date, "3x6", 0.0525, DayCountTypes.ACT_360,
           CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
        FRA(value_date, "6x9", 0.0530, DayCountTypes.ACT_360,
           CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
        FRA(value_date, "9x12", 0.0535, DayCountTypes.ACT_360,
           CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
    ]

    curve = OISCurve(
        value_dt=value_date,
        instruments=fras,
        interp_type=InterpTypes.LINEAR_ZERO_RATES,
        check_refit=False
    )

    # Curve should be constructed successfully
    assert len(curve._times) >= len(fras)
    assert len(curve._dfs) >= len(fras)


def test_fra_curve_refit_accuracy(value_date):
    """Test that FRAs reprice accurately after bootstrapping."""
    fras = [
        FRA(value_date, "3x6", 0.0525, DayCountTypes.ACT_360,
           CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
        FRA(value_date, "6x9", 0.0530, DayCountTypes.ACT_360,
           CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
    ]

    curve = OISCurve(
        value_dt=value_date,
        instruments=fras,
        interp_type=InterpTypes.LINEAR_ZERO_RATES,
        check_refit=True  # Will raise if refit fails
    )

    # If we reach here, refit check passed
    assert True


def test_fra_curve_via_model(usd_model):
    """Test FRA curve construction via Model.build_curve()."""
    usd_model.build_curve(
        name="USD_OIS_SOFR",
        px_list=[5.25, 5.30, 5.35],
        tenor_list=["3x6", "6x9", "9x12"],
        instrument_type="FRA",
        fixed_dcc_type=DayCountTypes.ACT_360
    )

    # Curve should be accessible
    curve = usd_model.curves.USD_OIS_SOFR
    assert curve is not None
    assert len(curve.swap_rates) == 3


###############################################################################
# Mixed Instrument Curve Tests
###############################################################################


def test_mixed_deposit_fra_curve(value_date):
    """Test curve from mixed deposits and FRAs."""
    instruments = [
        # Deposits for short end (0-3M)
        CashDeposit(value_date, "1M", 0.0500, DayCountTypes.ACT_360,
                   CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
        CashDeposit(value_date, "3M", 0.0520, DayCountTypes.ACT_360,
                   CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
        # FRAs for medium term (3M-12M)
        FRA(value_date, "3x6", 0.0525, DayCountTypes.ACT_360,
           CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
        FRA(value_date, "6x9", 0.0530, DayCountTypes.ACT_360,
           CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
        FRA(value_date, "9x12", 0.0535, DayCountTypes.ACT_360,
           CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
    ]

    curve = OISCurve(
        value_dt=value_date,
        instruments=instruments,
        interp_type=InterpTypes.LINEAR_ZERO_RATES,
        check_refit=True
    )

    # Curve should be sorted and bootstrapped correctly
    assert len(curve._times) >= len(instruments)


def test_mixed_deposit_fra_ois_curve(value_date):
    """Test full term structure: deposits + FRAs + OIS."""
    instruments = [
        # Deposits for 0-3M
        CashDeposit(value_date, "1M", 0.0500, DayCountTypes.ACT_360,
                   CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
        CashDeposit(value_date, "3M", 0.0520, DayCountTypes.ACT_360,
                   CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
        # FRAs for 3M-12M
        FRA(value_date, "3x6", 0.0525, DayCountTypes.ACT_360,
           CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
        FRA(value_date, "6x9", 0.0530, DayCountTypes.ACT_360,
           CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
        # OIS for 1Y+
        OIS(
            effective_dt=value_date,
            term_dt_or_tenor="2Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=0.0540,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_360,
            floating_index=CurveTypes.USD_OIS_SOFR,
            currency=CurrencyTypes.USD
        ),
    ]

    from cavour.utils.global_types import SwapTypes

    curve = OISCurve(
        value_dt=value_date,
        instruments=instruments,
        interp_type=InterpTypes.LINEAR_ZERO_RATES,
        check_refit=True
    )

    # Full term structure built successfully
    assert len(curve._times) >= 5


###############################################################################
# Curve Continuity Tests
###############################################################################


def test_curve_forward_rates_monotonic(value_date):
    """Test that forward rates are reasonable (no arbitrage)."""
    instruments = [
        CashDeposit(value_date, "1M", 0.0500, DayCountTypes.ACT_360,
                   CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
        CashDeposit(value_date, "3M", 0.0520, DayCountTypes.ACT_360,
                   CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
        CashDeposit(value_date, "6M", 0.0540, DayCountTypes.ACT_360,
                   CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
    ]

    curve = OISCurve(
        value_dt=value_date,
        instruments=instruments,
        interp_type=InterpTypes.LINEAR_ZERO_RATES,
        check_refit=False
    )

    # Extract discount factors
    dfs = curve._dfs

    # DFs should be monotonically decreasing (increasing rates)
    for i in range(len(dfs) - 1):
        assert dfs[i] >= dfs[i+1], f"DF[{i}]={dfs[i]} should be >= DF[{i+1}]={dfs[i+1]}"


def test_curve_no_negative_forward_rates(value_date):
    """Test that curve produces no negative forward rates."""
    instruments = [
        CashDeposit(value_date, "1M", 0.0500, DayCountTypes.ACT_360,
                   CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
        CashDeposit(value_date, "6M", 0.0520, DayCountTypes.ACT_360,
                   CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
        CashDeposit(value_date, "12M", 0.0540, DayCountTypes.ACT_360,
                   CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
    ]

    curve = OISCurve(
        value_dt=value_date,
        instruments=instruments,
        interp_type=InterpTypes.LINEAR_ZERO_RATES,
        check_refit=False
    )

    # Check forward rates between consecutive points
    times = curve._times
    dfs = curve._dfs

    for i in range(len(times) - 1):
        if times[i+1] > times[i]:
            dt = times[i+1] - times[i]
            fwd_df = dfs[i+1] / dfs[i]
            fwd_rate = (1.0 / fwd_df - 1.0) / dt

            # Forward rate should be non-negative (allowing small numerical errors)
            assert fwd_rate > -1e-6, f"Negative forward rate at t={times[i]}: {fwd_rate}"


###############################################################################
# Instrument Sorting Tests
###############################################################################


def test_instrument_sorting_by_maturity(value_date):
    """Test that instruments are sorted by maturity before bootstrapping."""
    # Create instruments in random order
    instruments = [
        CashDeposit(value_date, "6M", 0.0540, DayCountTypes.ACT_360,
                   CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
        CashDeposit(value_date, "1M", 0.0500, DayCountTypes.ACT_360,
                   CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
        CashDeposit(value_date, "3M", 0.0520, DayCountTypes.ACT_360,
                   CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
    ]

    curve = OISCurve(
        value_dt=value_date,
        instruments=instruments,
        interp_type=InterpTypes.LINEAR_ZERO_RATES,
        check_refit=False
    )

    # Check that swap_times are in increasing order
    times = curve.swap_times
    for i in range(len(times) - 1):
        assert times[i] <= times[i+1], f"Times not sorted: {times[i]} > {times[i+1]}"


###############################################################################
# Multi-Currency Tests
###############################################################################


@pytest.mark.parametrize("currency,curve_type,dc_type", [
    (CurrencyTypes.USD, CurveTypes.USD_OIS_SOFR, DayCountTypes.ACT_360),
    (CurrencyTypes.GBP, CurveTypes.GBP_OIS_SONIA, DayCountTypes.ACT_365F),
    (CurrencyTypes.EUR, CurveTypes.EUR_OIS_ESTR, DayCountTypes.ACT_360),
])
def test_mixed_curve_multi_currency(value_date, currency, curve_type, dc_type):
    """Test mixed curves for different currencies."""
    instruments = [
        CashDeposit(value_date, "1M", 0.05, dc_type, curve_type, currency),
        CashDeposit(value_date, "3M", 0.052, dc_type, curve_type, currency),
        FRA(value_date, "3x6", 0.053, dc_type, curve_type, currency),
    ]

    curve = OISCurve(
        value_dt=value_date,
        instruments=instruments,
        interp_type=InterpTypes.LINEAR_ZERO_RATES,
        check_refit=True
    )

    # Curve built successfully for this currency
    assert len(curve.swap_rates) == 3


###############################################################################
# Model Integration Tests
###############################################################################


def test_model_mixed_curve_construction(usd_model, value_date):
    """Test Model with mixed instrument curves."""
    # Build deposit curve
    usd_model.build_curve(
        name="USD_OIS_SOFR",
        px_list=[5.00, 5.20],
        tenor_list=["1M", "3M"],
        instrument_type="DEPOSIT",
        fixed_dcc_type=DayCountTypes.ACT_360
    )

    curve = usd_model.curves.USD_OIS_SOFR
    assert curve is not None

    # Create a deposit and value it
    deposit = CashDeposit(
        value_date, "1M", 0.05, DayCountTypes.ACT_360,
        CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD
    )

    pos = deposit.position(usd_model)
    result = pos.compute([RequestTypes.VALUE])

    # Should have a value
    assert result.value is not None


###############################################################################
# Error Handling Tests
###############################################################################


def test_mixed_instruments_with_inconsistent_dc_raises_warning(value_date):
    """Test that mixed day count conventions work (use first instrument's)."""
    instruments = [
        CashDeposit(value_date, "1M", 0.05, DayCountTypes.ACT_360,
                   CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
        CashDeposit(value_date, "3M", 0.052, DayCountTypes.ACT_365F,  # Different DC
                   CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
    ]

    # Should not raise, but uses first instrument's day count
    curve = OISCurve(
        value_dt=value_date,
        instruments=instruments,
        interp_type=InterpTypes.LINEAR_ZERO_RATES,
        check_refit=False
    )

    # Curve uses first instrument's day count
    assert curve._dc_type == DayCountTypes.ACT_360


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
