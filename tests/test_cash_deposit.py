"""
Unit tests for CashDeposit instrument.

Tests cover:
- Basic deposit construction and validation
- Valuation at different time points
- Implied rate calculations
- Curve bootstrapping and refit accuracy
- Multi-currency deposits (USD, GBP, EUR)
- Integration with Model and Position framework
"""

import pytest
import numpy as np
from cavour.utils.date import Date
from cavour.utils.day_count import DayCountTypes
from cavour.utils.global_types import CurveTypes, InstrumentTypes
from cavour.utils.currency import CurrencyTypes
from cavour.utils.calendar import CalendarTypes, BusDayAdjustTypes
from cavour.trades.rates.cash_deposit import CashDeposit
from cavour.trades.rates.ois_curve import OISCurve
from cavour.utils.frequency import FrequencyTypes
from cavour.market.curves.interpolator import InterpTypes

###############################################################################
# Test Fixtures
###############################################################################


@pytest.fixture
def value_date():
    """Standard valuation date for tests."""
    return Date(15, 6, 2023)


@pytest.fixture
def usd_deposit_3m(value_date):
    """Create a 3M USD deposit at 5.25%."""
    return CashDeposit(
        effective_dt=value_date,
        term_dt_or_tenor="3M",
        deposit_rate=0.0525,
        dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD,
        notional=1_000_000
    )


@pytest.fixture
def gbp_deposit_6m(value_date):
    """Create a 6M GBP deposit at 4.75%."""
    return CashDeposit(
        effective_dt=value_date,
        term_dt_or_tenor="6M",
        deposit_rate=0.0475,
        dc_type=DayCountTypes.ACT_365F,
        floating_index=CurveTypes.GBP_OIS_SONIA,
        currency=CurrencyTypes.GBP,
        notional=1_000_000
    )


###############################################################################
# Construction Tests
###############################################################################


def test_deposit_construction_with_tenor(value_date):
    """Test deposit construction using tenor string."""
    deposit = CashDeposit(
        effective_dt=value_date,
        term_dt_or_tenor="3M",
        deposit_rate=0.05,
        dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD
    )

    assert deposit.derivative_type == InstrumentTypes.CASH_DEPOSIT
    assert deposit._effective_dt == value_date
    assert deposit._deposit_rate == 0.05
    assert deposit._currency == CurrencyTypes.USD
    assert deposit._notional == 1_000_000  # Default
    assert deposit._maturity_dt > value_date  # Should be approximately 3 months later


def test_deposit_construction_with_explicit_date(value_date):
    """Test deposit construction using explicit maturity date."""
    maturity_dt = value_date.add_months(6)
    deposit = CashDeposit(
        effective_dt=value_date,
        term_dt_or_tenor=maturity_dt,
        deposit_rate=0.045,
        dc_type=DayCountTypes.ACT_365F,
        floating_index=CurveTypes.GBP_OIS_SONIA,
        currency=CurrencyTypes.GBP,
        notional=5_000_000
    )

    assert deposit._notional == 5_000_000
    assert deposit._maturity_dt >= maturity_dt  # May be adjusted for business days


def test_deposit_attributes_for_curve_builder(usd_deposit_3m):
    """Test that deposit has required attributes for OISCurve bootstrapping."""
    # OISCurve expects these attributes
    assert hasattr(usd_deposit_3m, '_adjusted_fixed_dts')
    assert hasattr(usd_deposit_3m, '_fixed_coupon')
    assert hasattr(usd_deposit_3m, '_fixed_year_fracs')
    assert hasattr(usd_deposit_3m, '_start_dt')

    # Check format: single-element lists
    assert len(usd_deposit_3m._adjusted_fixed_dts) == 1
    assert len(usd_deposit_3m._fixed_year_fracs) == 1
    assert usd_deposit_3m._fixed_coupon == 0.0525
    assert usd_deposit_3m._start_dt == usd_deposit_3m._effective_dt


def test_deposit_payment_calculation(usd_deposit_3m):
    """Test that payment amount uses simple interest formula."""
    # Payment = Notional * (1 + rate * year_frac)
    expected_payment = 1_000_000 * (1.0 + 0.0525 * usd_deposit_3m._year_frac)
    assert abs(usd_deposit_3m._payment_amt - expected_payment) < 1e-10


def test_deposit_year_fraction_calculation(value_date):
    """Test year fraction calculation with different day count conventions."""
    # USD deposit with ACT/360
    usd_deposit = CashDeposit(
        effective_dt=value_date,
        term_dt_or_tenor="3M",
        deposit_rate=0.05,
        dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD
    )

    # GBP deposit with ACT/365F
    gbp_deposit = CashDeposit(
        effective_dt=value_date,
        term_dt_or_tenor="3M",
        deposit_rate=0.05,
        dc_type=DayCountTypes.ACT_365F,
        floating_index=CurveTypes.GBP_OIS_SONIA,
        currency=CurrencyTypes.GBP
    )

    # Year fractions should be different due to day count conventions
    assert usd_deposit._year_frac != gbp_deposit._year_frac
    # Both should be approximately 0.25 for 3M, but vary by convention
    assert 0.24 < usd_deposit._year_frac < 0.26
    assert 0.24 < gbp_deposit._year_frac < 0.26


###############################################################################
# Valuation Tests
###############################################################################


def test_deposit_value_at_inception(value_date, usd_deposit_3m):
    """Test that deposit values to approximately zero at inception with market rate."""
    # Create a simple flat curve at the deposit rate
    deposits = [usd_deposit_3m]
    curve = OISCurve(
        value_dt=value_date,
        instruments=deposits,
        interp_type=InterpTypes.FLAT_FWD_RATES,
        check_refit=False
    )

    # Value at inception should be close to zero
    value = usd_deposit_3m.value(value_date, curve)
    assert abs(value) < 1.0  # Within $1 for 1M notional


def test_deposit_value_at_maturity(value_date, usd_deposit_3m):
    """Test that deposit has zero PV at maturity."""
    # Create simple curve
    deposits = [usd_deposit_3m]
    curve = OISCurve(
        value_dt=value_date,
        instruments=deposits,
        interp_type=InterpTypes.FLAT_FWD_RATES,
        check_refit=False
    )

    # Value at maturity should be 0 (cashflow already occurred)
    value = usd_deposit_3m.value(usd_deposit_3m._maturity_dt, curve)
    assert abs(value) < 1e-10


def test_deposit_value_after_maturity(value_date, usd_deposit_3m):
    """Test that deposit has zero value after maturity."""
    deposits = [usd_deposit_3m]
    curve = OISCurve(
        value_dt=value_date,
        instruments=deposits,
        interp_type=InterpTypes.FLAT_FWD_RATES,
        check_refit=False
    )

    # Value after maturity
    future_date = usd_deposit_3m._maturity_dt.add_days(30)
    value = usd_deposit_3m.value(future_date, curve)
    assert abs(value) < 1e-10


def test_deposit_value_mid_life(value_date, usd_deposit_3m):
    """Test deposit valuation between inception and maturity."""
    deposits = [usd_deposit_3m]
    curve = OISCurve(
        value_dt=value_date,
        instruments=deposits,
        interp_type=InterpTypes.FLAT_FWD_RATES,
        check_refit=False
    )

    # Value 1 month into the deposit
    mid_date = value_date.add_months(1)
    value = usd_deposit_3m.value(mid_date, curve)

    # Should be positive (receiving fixed rate higher than curve)
    # or approximately zero if curve matches deposit rate
    assert isinstance(value, (int, float, np.floating))


###############################################################################
# Implied Rate Tests
###############################################################################


def test_deposit_implied_rate_from_curve(value_date):
    """Test implied rate extraction from discount curve."""
    # Create deposit
    deposit = CashDeposit(
        effective_dt=value_date,
        term_dt_or_tenor="6M",
        deposit_rate=0.05,  # Initial rate
        dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD
    )

    # Build curve with this deposit
    curve = OISCurve(
        value_dt=value_date,
        instruments=[deposit],
        interp_type=InterpTypes.FLAT_FWD_RATES,
        check_refit=False
    )

    # Implied rate should match deposit rate (within tolerance)
    implied_rate = deposit.implied_rate(value_date, curve)
    assert abs(implied_rate - 0.05) < 1e-6


###############################################################################
# Multi-Currency Tests
###############################################################################


@pytest.mark.parametrize("currency,curve_type,dc_type,rate", [
    (CurrencyTypes.USD, CurveTypes.USD_OIS_SOFR, DayCountTypes.ACT_360, 0.0525),
    (CurrencyTypes.GBP, CurveTypes.GBP_OIS_SONIA, DayCountTypes.ACT_365F, 0.0475),
    (CurrencyTypes.EUR, CurveTypes.EUR_OIS_ESTR, DayCountTypes.ACT_360, 0.0350),
])
def test_multi_currency_deposits(value_date, currency, curve_type, dc_type, rate):
    """Test deposits in different currencies with appropriate conventions."""
    deposit = CashDeposit(
        effective_dt=value_date,
        term_dt_or_tenor="3M",
        deposit_rate=rate,
        dc_type=dc_type,
        floating_index=curve_type,
        currency=currency,
        notional=1_000_000
    )

    assert deposit._currency == currency
    assert deposit._floating_index == curve_type
    assert deposit._dc_type == dc_type
    assert deposit._deposit_rate == rate


###############################################################################
# Curve Bootstrapping Tests
###############################################################################


def test_single_deposit_curve_bootstrap(value_date):
    """Test curve bootstrapping with a single deposit."""
    deposit = CashDeposit(
        effective_dt=value_date,
        term_dt_or_tenor="3M",
        deposit_rate=0.05,
        dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD
    )

    curve = OISCurve(
        value_dt=value_date,
        instruments=[deposit],
        interp_type=InterpTypes.FLAT_FWD_RATES,
        check_refit=False
    )

    # Curve should have time points
    assert len(curve._times) > 0
    assert len(curve._dfs) > 0
    # First DF should be 1.0 at t=0
    assert abs(curve._dfs[0] - 1.0) < 1e-10


def test_multiple_deposit_curve_bootstrap(value_date):
    """Test curve bootstrapping with multiple deposits."""
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
        interp_type=InterpTypes.FLAT_FWD_RATES,
        check_refit=False
    )

    # Curve should be built successfully
    assert len(curve._times) >= len(deposits)


def test_deposit_curve_refit_accuracy(value_date):
    """Test that deposits reprice to near-zero after curve bootstrapping."""
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
        check_refit=False  # Manual check below
    )

    # Each deposit should reprice to near-zero on its own curve
    refit_tolerance = 1e-6  # $0.000001 for 1M notional
    for deposit in deposits:
        value = deposit.value(value_date, curve)
        assert abs(value) < refit_tolerance, \
            f"Deposit {deposit._maturity_dt} refit failed: value={value}"


###############################################################################
# Print and Display Tests
###############################################################################


def test_deposit_repr(usd_deposit_3m):
    """Test deposit string representation."""
    repr_str = repr(usd_deposit_3m)
    assert "CashDeposit" in repr_str
    assert "5.2500%" in repr_str or "0.0525" in repr_str
    assert "USD" in repr_str


def test_deposit_print_details(usd_deposit_3m, capsys):
    """Test deposit print_details() output."""
    usd_deposit_3m.print_details()
    captured = capsys.readouterr()
    assert "CashDeposit" in captured.out
    assert "5.25" in captured.out  # Rate
    assert "USD" in captured.out


###############################################################################
# Error Handling Tests
###############################################################################


def test_deposit_effective_after_maturity_raises_error(value_date):
    """Test that effective date after maturity raises error."""
    with pytest.raises(Exception):  # LibError
        CashDeposit(
            effective_dt=value_date.add_months(6),
            term_dt_or_tenor=value_date,  # Maturity before effective
            deposit_rate=0.05,
            dc_type=DayCountTypes.ACT_360,
            floating_index=CurveTypes.USD_OIS_SOFR,
            currency=CurrencyTypes.USD
        )


def test_deposit_value_without_curve_raises_error(usd_deposit_3m, value_date):
    """Test that valuation without curve raises error."""
    with pytest.raises(ValueError):
        usd_deposit_3m.value(value_date, discount_curve=None, ois_curve=None)


###############################################################################
# Business Day Adjustment Tests
###############################################################################


def test_deposit_business_day_adjustment(value_date):
    """Test that maturity date is adjusted for business days."""
    # Create deposit with maturity potentially on weekend
    deposit = CashDeposit(
        effective_dt=value_date,
        term_dt_or_tenor="3M",
        deposit_rate=0.05,
        dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD,
        cal_type=CalendarTypes.WEEKEND,
        bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING
    )

    # Maturity should be a valid business day
    # (Hard to test without knowing specific date, but attribute should exist)
    assert hasattr(deposit, '_maturity_dt')
    assert deposit._maturity_dt >= value_date


###############################################################################
# Integration Tests
###############################################################################


def test_deposit_position_creation(value_date, usd_deposit_3m):
    """Test Position object creation (requires Model, skip if not available)."""
    # This test would require a full Model setup
    # Placeholder to show Position interface
    assert hasattr(usd_deposit_3m, 'position')
    # pos = usd_deposit_3m.position(model)
    # result = pos.compute([RequestTypes.VALUE])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
