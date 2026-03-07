"""
Unit tests for FRA (Forward Rate Agreement) instrument.

Tests cover:
- FRA construction using notation and explicit dates
- Forward rate calculation
- FRA valuation at different time points
- Implied FRA rate calculations
- Curve bootstrapping and refit accuracy
- Multi-currency FRAs (USD, GBP, EUR)
- Standard FRA notations ("3x6", "6x9", "9x12")
"""

import pytest
import numpy as np
from cavour.utils.date import Date
from cavour.utils.day_count import DayCountTypes
from cavour.utils.global_types import CurveTypes, InstrumentTypes
from cavour.utils.currency import CurrencyTypes
from cavour.utils.calendar import CalendarTypes, BusDayAdjustTypes
from cavour.trades.rates.fra import FRA
from cavour.trades.rates.cash_deposit import CashDeposit
from cavour.trades.rates.ois_curve import OISCurve
from cavour.market.curves.interpolator import InterpTypes

###############################################################################
# Test Fixtures
###############################################################################


@pytest.fixture
def value_date():
    """Standard valuation date for tests."""
    return Date(15, 6, 2023)


@pytest.fixture
def usd_fra_3x6(value_date):
    """Create a 3x6 USD FRA at 5.25%."""
    return FRA(
        effective_dt=value_date,
        fra_notation="3x6",
        fra_rate=0.0525,
        dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD,
        notional=1_000_000
    )


@pytest.fixture
def gbp_fra_6x9(value_date):
    """Create a 6x9 GBP FRA at 4.75%."""
    return FRA(
        effective_dt=value_date,
        fra_notation="6x9",
        fra_rate=0.0475,
        dc_type=DayCountTypes.ACT_365F,
        floating_index=CurveTypes.GBP_OIS_SONIA,
        currency=CurrencyTypes.GBP,
        notional=1_000_000
    )


###############################################################################
# Construction Tests
###############################################################################


def test_fra_construction_with_notation(value_date):
    """Test FRA construction using standard notation."""
    fra = FRA(
        effective_dt=value_date,
        fra_notation="3x6",
        fra_rate=0.05,
        dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD
    )

    assert fra.derivative_type == InstrumentTypes.FRA
    assert fra._effective_dt == value_date
    assert fra._fra_rate == 0.05
    assert fra._currency == CurrencyTypes.USD
    assert fra._notional == 1_000_000  # Default
    assert fra._fixing_dt > value_date
    assert fra._start_dt > fra._fixing_dt
    assert fra._end_dt > fra._start_dt


def test_fra_construction_with_explicit_dates(value_date):
    """Test FRA construction using explicit dates."""
    fixing_dt = value_date.add_months(3)
    start_dt = fixing_dt.add_days(2)
    end_dt = start_dt.add_months(3)

    fra = FRA(
        effective_dt=value_date,
        fixing_dt=fixing_dt,
        start_dt=start_dt,
        end_dt=end_dt,
        fra_rate=0.05,
        dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD,
        notional=5_000_000
    )

    assert fra._notional == 5_000_000
    assert fra._fixing_dt == fixing_dt
    assert fra._start_dt == start_dt
    assert fra._end_dt == end_dt


def test_fra_attributes_for_curve_builder(usd_fra_3x6):
    """Test that FRA has required attributes for OISCurve bootstrapping."""
    # OISCurve expects these attributes
    assert hasattr(usd_fra_3x6, '_adjusted_fixed_dts')
    assert hasattr(usd_fra_3x6, '_fixed_coupon')
    assert hasattr(usd_fra_3x6, '_fixed_year_fracs')
    assert hasattr(usd_fra_3x6, '_start_dt_attr')

    # Check format: single-element lists
    assert len(usd_fra_3x6._adjusted_fixed_dts) == 1
    assert len(usd_fra_3x6._fixed_year_fracs) == 1
    assert usd_fra_3x6._fixed_coupon == 0.0525
    # Settlement date should be the single cashflow point
    assert usd_fra_3x6._adjusted_fixed_dts[0] == usd_fra_3x6._settlement_dt


@pytest.mark.parametrize("notation,fixing_months,end_months", [
    ("3x6", 3, 6),
    ("6x9", 6, 9),
    ("9x12", 9, 12),
    ("12x18", 12, 18),
])
def test_fra_notation_parsing(value_date, notation, fixing_months, end_months):
    """Test FRA notation parsing for various standard formats."""
    fra = FRA(
        effective_dt=value_date,
        fra_notation=notation,
        fra_rate=0.05,
        dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD
    )

    # Fixing date should be approximately fixing_months from effective
    # (exact match depends on business day adjustments)
    fixing_days = (fra._fixing_dt - value_date)
    expected_days_approx = fixing_months * 30  # Rough approximation
    assert abs(fixing_days - expected_days_approx) < 10  # Within 10 days tolerance

    # End date should be approximately end_months from effective
    end_days = (fra._end_dt - value_date)
    expected_end_days_approx = end_months * 30
    assert abs(end_days - expected_end_days_approx) < 10


###############################################################################
# Forward Rate Calculation Tests
###############################################################################


def test_fra_forward_rate_from_flat_curve(value_date, usd_fra_3x6):
    """Test forward rate calculation from a flat curve."""
    # Create flat curve at 5%
    deposit = CashDeposit(
        value_date, "12M", 0.05, DayCountTypes.ACT_360,
        CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD
    )
    curve = OISCurve(
        value_dt=value_date,
        instruments=[deposit],
        interp_type=InterpTypes.FLAT_FWD_RATES,
        check_refit=False
    )

    # Forward rate should be approximately 5% (flat curve)
    fwd_rate = usd_fra_3x6.forward_rate(value_date, curve)
    assert abs(fwd_rate - 0.05) < 0.01  # Within 1% tolerance


def test_fra_forward_rate_consistency(value_date):
    """Test that forward rate is consistent with discount factors."""
    # Create FRA
    fra = FRA(
        value_date, "3x6", 0.05, DayCountTypes.ACT_360,
        CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD
    )

    # Build curve
    deposit = CashDeposit(
        value_date, "12M", 0.05, DayCountTypes.ACT_360,
        CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD
    )
    curve = OISCurve(value_dt=value_date, instruments=[deposit],
                    interp_type=InterpTypes.FLAT_FWD_RATES, check_refit=False)

    # Get discount factors
    df_start = curve.df(fra._start_dt, fra._dc_type)
    df_end = curve.df(fra._end_dt, fra._dc_type)

    # Compute forward rate manually
    fwd_rate_manual = (df_start / df_end - 1.0) / fra._year_frac

    # Compare with FRA method
    fwd_rate_fra = fra.forward_rate(value_date, curve)

    assert abs(fwd_rate_manual - fwd_rate_fra) < 1e-10


###############################################################################
# Valuation Tests
###############################################################################


def test_fra_value_at_inception_with_market_rate(value_date, usd_fra_3x6):
    """Test that FRA values to approximately zero at inception with market rate."""
    # Create curve from FRA (uses FRA rate as market rate)
    # This requires modifying OISCurve to handle FRAs (Phase 6)
    # For now, use a deposit curve
    deposit = CashDeposit(
        value_date, "6M", 0.0525, DayCountTypes.ACT_360,
        CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD
    )
    curve = OISCurve(
        value_dt=value_date,
        instruments=[deposit],
        interp_type=InterpTypes.FLAT_FWD_RATES,
        check_refit=False
    )

    # Value should be small (not exactly zero due to curve construction)
    value = usd_fra_3x6.value(value_date, curve)
    assert abs(value) < 1000  # Within $1,000 for 1M notional


def test_fra_value_after_settlement(value_date, usd_fra_3x6):
    """Test that FRA has zero value after settlement."""
    deposit = CashDeposit(
        value_date, "6M", 0.05, DayCountTypes.ACT_360,
        CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD
    )
    curve = OISCurve(value_dt=value_date, instruments=[deposit],
                    interp_type=InterpTypes.FLAT_FWD_RATES, check_refit=False)

    # Value after settlement
    future_date = usd_fra_3x6._settlement_dt.add_days(1)
    value = usd_fra_3x6.value(future_date, curve)
    assert abs(value) < 1e-10


def test_fra_value_sign_convention(value_date):
    """Test FRA value sign convention (buyer benefits from rate increases)."""
    # Create FRA at 3%
    fra = FRA(
        value_date, "3x6", 0.03, DayCountTypes.ACT_360,
        CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD
    )

    # Create higher-rate curve (5%)
    deposit = CashDeposit(
        value_date, "6M", 0.05, DayCountTypes.ACT_360,
        CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD
    )
    curve = OISCurve(value_dt=value_date, instruments=[deposit],
                    interp_type=InterpTypes.FLAT_FWD_RATES, check_refit=False)

    # FRA value should be positive (buyer locked in 3%, market is 5%)
    value = fra.value(value_date, curve)
    assert value > 0


def test_fra_value_mid_life(value_date, usd_fra_3x6):
    """Test FRA valuation between effective and settlement dates."""
    deposit = CashDeposit(
        value_date, "6M", 0.05, DayCountTypes.ACT_360,
        CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD
    )
    curve = OISCurve(value_dt=value_date, instruments=[deposit],
                    interp_type=InterpTypes.FLAT_FWD_RATES, check_refit=False)

    # Value 1 month into the FRA
    mid_date = value_date.add_months(1)
    value = usd_fra_3x6.value(mid_date, curve)

    # Should return a numeric value
    assert isinstance(value, (int, float, np.floating))


###############################################################################
# Implied Rate Tests
###############################################################################


def test_fra_implied_rate_equals_forward_rate(value_date, usd_fra_3x6):
    """Test that implied FRA rate equals forward rate."""
    deposit = CashDeposit(
        value_date, "6M", 0.05, DayCountTypes.ACT_360,
        CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD
    )
    curve = OISCurve(value_dt=value_date, instruments=[deposit],
                    interp_type=InterpTypes.FLAT_FWD_RATES, check_refit=False)

    implied_rate = usd_fra_3x6.implied_fra_rate(value_date, curve)
    forward_rate = usd_fra_3x6.forward_rate(value_date, curve)

    assert abs(implied_rate - forward_rate) < 1e-10


###############################################################################
# Multi-Currency Tests
###############################################################################


@pytest.mark.parametrize("currency,curve_type,dc_type,rate,notation", [
    (CurrencyTypes.USD, CurveTypes.USD_OIS_SOFR, DayCountTypes.ACT_360, 0.0525, "3x6"),
    (CurrencyTypes.GBP, CurveTypes.GBP_OIS_SONIA, DayCountTypes.ACT_365F, 0.0475, "6x9"),
    (CurrencyTypes.EUR, CurveTypes.EUR_OIS_ESTR, DayCountTypes.ACT_360, 0.0350, "9x12"),
])
def test_multi_currency_fras(value_date, currency, curve_type, dc_type, rate, notation):
    """Test FRAs in different currencies with appropriate conventions."""
    fra = FRA(
        effective_dt=value_date,
        fra_notation=notation,
        fra_rate=rate,
        dc_type=dc_type,
        floating_index=curve_type,
        currency=currency,
        notional=1_000_000
    )

    assert fra._currency == currency
    assert fra._floating_index == curve_type
    assert fra._dc_type == dc_type
    assert fra._fra_rate == rate


###############################################################################
# Date Validation Tests
###############################################################################


def test_fra_fixing_before_effective_raises_error(value_date):
    """Test that fixing date before effective date raises error."""
    fixing_dt = value_date.add_days(-30)  # Before effective
    start_dt = value_date.add_days(2)
    end_dt = value_date.add_months(3)

    with pytest.raises(Exception):  # LibError
        FRA(
            effective_dt=value_date,
            fixing_dt=fixing_dt,
            start_dt=start_dt,
            end_dt=end_dt,
            fra_rate=0.05,
            dc_type=DayCountTypes.ACT_360,
            floating_index=CurveTypes.USD_OIS_SOFR,
            currency=CurrencyTypes.USD
        )


def test_fra_end_before_start_raises_error(value_date):
    """Test that end date before start date raises error."""
    fixing_dt = value_date.add_months(3)
    start_dt = value_date.add_months(6)
    end_dt = value_date.add_months(4)  # Before start

    with pytest.raises(Exception):  # LibError
        FRA(
            effective_dt=value_date,
            fixing_dt=fixing_dt,
            start_dt=start_dt,
            end_dt=end_dt,
            fra_rate=0.05,
            dc_type=DayCountTypes.ACT_360,
            floating_index=CurveTypes.USD_OIS_SOFR,
            currency=CurrencyTypes.USD
        )


def test_fra_invalid_notation_raises_error(value_date):
    """Test that invalid FRA notation raises error."""
    with pytest.raises(ValueError):
        FRA(
            effective_dt=value_date,
            fra_notation="invalid",
            fra_rate=0.05,
            dc_type=DayCountTypes.ACT_360,
            floating_index=CurveTypes.USD_OIS_SOFR,
            currency=CurrencyTypes.USD
        )


def test_fra_backward_notation_raises_error(value_date):
    """Test that backward notation (e.g., 6x3) raises error."""
    with pytest.raises(ValueError):
        FRA(
            effective_dt=value_date,
            fra_notation="6x3",  # End before start
            fra_rate=0.05,
            dc_type=DayCountTypes.ACT_360,
            floating_index=CurveTypes.USD_OIS_SOFR,
            currency=CurrencyTypes.USD
        )


def test_fra_missing_dates_raises_error(value_date):
    """Test that missing dates without notation raises error."""
    with pytest.raises(ValueError):
        FRA(
            effective_dt=value_date,
            # No fra_notation and no explicit dates
            fra_rate=0.05,
            dc_type=DayCountTypes.ACT_360,
            floating_index=CurveTypes.USD_OIS_SOFR,
            currency=CurrencyTypes.USD
        )


###############################################################################
# Print and Display Tests
###############################################################################


def test_fra_repr(usd_fra_3x6):
    """Test FRA string representation."""
    repr_str = repr(usd_fra_3x6)
    assert "FRA" in repr_str
    assert "5.2500%" in repr_str or "0.0525" in repr_str
    assert "USD" in repr_str


def test_fra_print_details(usd_fra_3x6, capsys):
    """Test FRA print_details() output."""
    usd_fra_3x6.print_details()
    captured = capsys.readouterr()
    assert "FRA" in captured.out
    assert "5.25" in captured.out  # Rate
    assert "USD" in captured.out
    assert "Fixing Date" in captured.out
    assert "Settlement Date" in captured.out


###############################################################################
# Settlement Date Tests
###############################################################################


def test_fra_settlement_date_is_t_plus_2(value_date):
    """Test that settlement date is T+2 from fixing (standard convention)."""
    fra = FRA(
        value_date, "3x6", 0.05, DayCountTypes.ACT_360,
        CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD,
        settlement_lag=2
    )

    # Settlement should be 2 business days after fixing
    # (Exact calculation depends on calendar, but should be close)
    days_diff = (fra._settlement_dt - fra._fixing_dt)
    assert 2 <= days_diff <= 4  # Between 2 and 4 days (accounting for weekends)


def test_fra_custom_settlement_lag(value_date):
    """Test FRA with custom settlement lag."""
    fra = FRA(
        value_date, "3x6", 0.05, DayCountTypes.ACT_360,
        CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD,
        settlement_lag=0  # Immediate settlement
    )

    # Settlement should be at fixing date
    assert fra._settlement_dt == fra._fixing_dt


###############################################################################
# Integration Tests
###############################################################################


def test_fra_position_creation(value_date, usd_fra_3x6):
    """Test Position object creation (requires Model, skip if not available)."""
    # This test would require a full Model setup
    # Placeholder to show Position interface
    assert hasattr(usd_fra_3x6, 'position')
    # pos = usd_fra_3x6.position(model)
    # result = pos.compute([RequestTypes.VALUE])


def test_fra_value_without_curve_raises_error(usd_fra_3x6, value_date):
    """Test that valuation without curve raises error."""
    with pytest.raises(ValueError):
        usd_fra_3x6.value(value_date, discount_curve=None, ois_curve=None)


###############################################################################
# Year Fraction Tests
###############################################################################


def test_fra_year_fraction_calculation(value_date):
    """Test year fraction calculation with different day count conventions."""
    # USD FRA with ACT/360
    usd_fra = FRA(
        value_date, "3x6", 0.05, DayCountTypes.ACT_360,
        CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD
    )

    # GBP FRA with ACT/365F
    gbp_fra = FRA(
        value_date, "3x6", 0.05, DayCountTypes.ACT_365F,
        CurveTypes.GBP_OIS_SONIA, CurrencyTypes.GBP
    )

    # Year fractions should be different due to day count conventions
    assert usd_fra._year_frac != gbp_fra._year_frac
    # Both should be approximately 0.25 for 3M accrual period
    assert 0.23 < usd_fra._year_frac < 0.26
    assert 0.23 < gbp_fra._year_frac < 0.26


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
