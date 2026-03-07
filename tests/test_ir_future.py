"""
Unit tests for IRFuture (STIR futures) implementation.

Tests cover:
- Construction with various contract code formats
- Contract type support (IMM, SERIAL, FOMC, BOE)
- Implied rate calculation from futures price
- Forward rate calculation from discount curves
- Valuation logic
- Curve builder compatibility attributes
- Edge cases and error handling
"""

import pytest
from cavour.utils.date import Date
from cavour.utils.day_count import DayCountTypes
from cavour.utils.global_types import (
    CurveTypes, FutureContractTypes, InstrumentTypes
)
from cavour.utils.currency import CurrencyTypes
from cavour.utils.calendar import CalendarTypes, BusDayAdjustTypes
from cavour.utils.error import LibError
from cavour.trades.rates.ir_future import IRFuture
from cavour.trades.rates.ois_curve import OISCurve
from cavour.trades.rates.cash_deposit import CashDeposit
from cavour.market.curves.interpolator import InterpTypes

###############################################################################
# Fixtures
###############################################################################


@pytest.fixture
def value_date():
    """Standard valuation date for tests."""
    return Date(15, 1, 2024)


@pytest.fixture
def usd_sofr_imm_h24(value_date):
    """USD SOFR IMM futures for March 2024 at 97.50 (2.5% implied)."""
    return IRFuture(
        effective_dt=value_date,
        expiry_date_or_contract="H24",  # March 2024
        futures_price=97.50,
        contract_type=FutureContractTypes.IMM,
        currency=CurrencyTypes.USD,
        floating_index=CurveTypes.USD_OIS_SOFR
    )


@pytest.fixture
def gbp_sonia_imm_m24(value_date):
    """GBP SONIA IMM futures for June 2024 at 95.25 (4.75% implied)."""
    return IRFuture(
        effective_dt=value_date,
        expiry_date_or_contract="M24",  # June 2024
        futures_price=95.25,
        contract_type=FutureContractTypes.IMM,
        currency=CurrencyTypes.GBP,
        floating_index=CurveTypes.GBP_OIS_SONIA
    )


###############################################################################
# Construction Tests
###############################################################################


def test_ir_future_construction_basic(usd_sofr_imm_h24):
    """Test basic IRFuture construction."""
    assert usd_sofr_imm_h24.derivative_type == InstrumentTypes.STIR_FUTURE
    assert usd_sofr_imm_h24._currency == CurrencyTypes.USD
    assert usd_sofr_imm_h24._contract_type == FutureContractTypes.IMM
    assert usd_sofr_imm_h24._futures_price == 97.50
    assert usd_sofr_imm_h24._notional == 1_000_000  # $1M standard


def test_ir_future_expiry_date_from_month_code(value_date):
    """Test expiry date calculation from month code."""
    future = IRFuture(
        effective_dt=value_date,
        expiry_date_or_contract="H24",  # March 2024
        futures_price=97.50,
        contract_type=FutureContractTypes.IMM,
        currency=CurrencyTypes.USD,
        floating_index=CurveTypes.USD_OIS_SOFR
    )

    # March 2024 IMM date is 3rd Wednesday = March 20, 2024
    assert future._expiry_dt._m == 3
    assert future._expiry_dt._y == 2024
    assert future._expiry_dt.is_imm_date()


def test_ir_future_expiry_date_from_ticker_code(value_date):
    """Test expiry date parsing from full ticker code."""
    future = IRFuture(
        effective_dt=value_date,
        expiry_date_or_contract="SFRH24",  # SOFR March 2024
        futures_price=97.50,
        contract_type=FutureContractTypes.IMM,
        currency=CurrencyTypes.USD,
        floating_index=CurveTypes.USD_OIS_SOFR
    )

    # Should parse same as "H24"
    assert future._expiry_dt._m == 3
    assert future._expiry_dt._y == 2024


def test_ir_future_expiry_date_from_imm_notation(value_date):
    """Test expiry date from IMM1, IMM2 notation."""
    future1 = IRFuture(
        effective_dt=value_date,  # Jan 15, 2024
        expiry_date_or_contract="IMM1",  # Next IMM = March 20, 2024
        futures_price=97.50,
        contract_type=FutureContractTypes.IMM,
        currency=CurrencyTypes.USD,
        floating_index=CurveTypes.USD_OIS_SOFR
    )

    future2 = IRFuture(
        effective_dt=value_date,
        expiry_date_or_contract="IMM2",  # Second IMM = June 19, 2024
        futures_price=97.25,
        contract_type=FutureContractTypes.IMM,
        currency=CurrencyTypes.USD,
        floating_index=CurveTypes.USD_OIS_SOFR
    )

    # IMM1 should be March 2024
    assert future1._expiry_dt._m == 3
    assert future1._expiry_dt._y == 2024

    # IMM2 should be June 2024
    assert future2._expiry_dt._m == 6
    assert future2._expiry_dt._y == 2024


def test_ir_future_expiry_date_explicit(value_date):
    """Test explicit expiry date."""
    expiry_dt = Date(20, 3, 2024)  # March 20, 2024
    future = IRFuture(
        effective_dt=value_date,
        expiry_date_or_contract=expiry_dt,
        futures_price=97.50,
        contract_type=FutureContractTypes.IMM,
        currency=CurrencyTypes.USD,
        floating_index=CurveTypes.USD_OIS_SOFR
    )

    assert future._expiry_dt == expiry_dt


def test_ir_future_serial_monthly(value_date):
    """Test serial monthly futures (non-IMM months)."""
    # February is not an IMM month (only Mar, Jun, Sep, Dec)
    future = IRFuture(
        effective_dt=value_date,
        expiry_date_or_contract="G24",  # February 2024
        futures_price=97.50,
        contract_type=FutureContractTypes.SERIAL_MONTHLY,
        currency=CurrencyTypes.USD,
        floating_index=CurveTypes.USD_OIS_SOFR
    )

    assert future._expiry_dt._m == 2
    assert future._expiry_dt._y == 2024
    assert future._contract_type == FutureContractTypes.SERIAL_MONTHLY


def test_ir_future_invalid_imm_month_raises_error(value_date):
    """Test that using non-IMM month with IMM contract type raises error."""
    with pytest.raises(LibError, match="not an IMM month"):
        IRFuture(
            effective_dt=value_date,
            expiry_date_or_contract="G24",  # February (not IMM)
            futures_price=97.50,
            contract_type=FutureContractTypes.IMM,  # IMM contract type
            currency=CurrencyTypes.USD,
            floating_index=CurveTypes.USD_OIS_SOFR
        )


def test_ir_future_invalid_contract_code_raises_error(value_date):
    """Test invalid contract code formats."""
    with pytest.raises(LibError):
        IRFuture(
            effective_dt=value_date,
            expiry_date_or_contract="INVALID",
            futures_price=97.50,
            contract_type=FutureContractTypes.IMM,
            currency=CurrencyTypes.USD,
            floating_index=CurveTypes.USD_OIS_SOFR
        )


def test_ir_future_invalid_month_code_raises_error(value_date):
    """Test invalid month code."""
    with pytest.raises(LibError, match="Invalid month code"):
        IRFuture(
            effective_dt=value_date,
            expiry_date_or_contract="A24",  # 'A' is not a valid month code
            futures_price=97.50,
            contract_type=FutureContractTypes.IMM,
            currency=CurrencyTypes.USD,
            floating_index=CurveTypes.USD_OIS_SOFR
        )


###############################################################################
# Implied Rate Tests
###############################################################################


def test_ir_future_implied_rate(usd_sofr_imm_h24):
    """Test implied rate calculation from futures price."""
    # Price = 100 - rate * 100
    # 97.50 => rate = (100 - 97.50) / 100 = 2.5%
    implied_rate = usd_sofr_imm_h24.implied_rate()
    assert abs(implied_rate - 0.025) < 1e-10


def test_ir_future_implied_rate_high_price(value_date):
    """Test implied rate for high price (low rate)."""
    future = IRFuture(
        effective_dt=value_date,
        expiry_date_or_contract="H24",
        futures_price=99.50,  # Very high price => 0.5% rate
        contract_type=FutureContractTypes.IMM,
        currency=CurrencyTypes.USD,
        floating_index=CurveTypes.USD_OIS_SOFR
    )

    implied_rate = future.implied_rate()
    assert abs(implied_rate - 0.005) < 1e-10


def test_ir_future_implied_rate_low_price(value_date):
    """Test implied rate for low price (high rate)."""
    future = IRFuture(
        effective_dt=value_date,
        expiry_date_or_contract="H24",
        futures_price=90.00,  # Low price => 10% rate
        contract_type=FutureContractTypes.IMM,
        currency=CurrencyTypes.USD,
        floating_index=CurveTypes.USD_OIS_SOFR
    )

    implied_rate = future.implied_rate()
    assert abs(implied_rate - 0.10) < 1e-10


###############################################################################
# Accrual Period Tests
###############################################################################


def test_ir_future_accrual_period_3_months(usd_sofr_imm_h24):
    """Test that accrual period is 3 months."""
    # Expiry = March 20, 2024
    # Accrual start = March 20, 2024 (settlement date)
    # Accrual end = June 20, 2024 (3 months later, adjusted)

    assert usd_sofr_imm_h24._accrual_start_dt == usd_sofr_imm_h24._expiry_dt
    assert usd_sofr_imm_h24._accrual_end_dt._m == 6  # June

    # Year fraction should be approximately 0.25 (3M / 12M)
    # Exact value depends on day count (ACT/360 for USD)
    assert 0.24 < usd_sofr_imm_h24._year_frac < 0.26


def test_ir_future_day_count_usd(usd_sofr_imm_h24):
    """Test USD SOFR uses ACT/360 day count."""
    assert usd_sofr_imm_h24._dc_type == DayCountTypes.ACT_360


def test_ir_future_day_count_gbp(gbp_sonia_imm_m24):
    """Test GBP SONIA uses ACT/365F day count."""
    assert gbp_sonia_imm_m24._dc_type == DayCountTypes.ACT_365F


def test_ir_future_contract_size_usd(usd_sofr_imm_h24):
    """Test USD SOFR has $1M contract size."""
    assert usd_sofr_imm_h24._notional == 1_000_000


def test_ir_future_contract_size_gbp(gbp_sonia_imm_m24):
    """Test GBP SONIA has £1M contract size."""
    assert gbp_sonia_imm_m24._notional == 1_000_000


###############################################################################
# Curve Builder Compatibility Tests
###############################################################################


def test_ir_future_curve_builder_attributes(usd_sofr_imm_h24):
    """Test that IRFuture has required attributes for OISCurve bootstrapping."""
    # CRITICAL: These attributes are required by OISCurve
    assert hasattr(usd_sofr_imm_h24, '_fixed_coupon')
    assert hasattr(usd_sofr_imm_h24, '_adjusted_fixed_dts')
    assert hasattr(usd_sofr_imm_h24, '_fixed_year_fracs')
    assert hasattr(usd_sofr_imm_h24, '_start_dt')
    assert hasattr(usd_sofr_imm_h24, '_maturity_dt')

    # Check types
    assert isinstance(usd_sofr_imm_h24._adjusted_fixed_dts, list)
    assert isinstance(usd_sofr_imm_h24._fixed_year_fracs, list)
    assert isinstance(usd_sofr_imm_h24._fixed_coupon, float)

    # Single cashflow => single-element lists
    assert len(usd_sofr_imm_h24._adjusted_fixed_dts) == 1
    assert len(usd_sofr_imm_h24._fixed_year_fracs) == 1


def test_ir_future_fixed_coupon_equals_forward_rate(usd_sofr_imm_h24):
    """Test that _fixed_coupon is the convexity-adjusted rate."""
    # With zero convexity adjustment, _fixed_coupon should equal implied_rate
    assert abs(usd_sofr_imm_h24._fixed_coupon - usd_sofr_imm_h24._implied_rate) < 1e-10


def test_ir_future_convexity_adjustment(value_date):
    """Test convexity adjustment reduces forward rate."""
    convexity_adj = 0.0005  # 5 bps adjustment

    future = IRFuture(
        effective_dt=value_date,
        expiry_date_or_contract="H24",
        futures_price=97.50,  # 2.5% implied
        contract_type=FutureContractTypes.IMM,
        currency=CurrencyTypes.USD,
        floating_index=CurveTypes.USD_OIS_SOFR,
        convexity_adjustment=convexity_adj
    )

    # forward_rate = implied_rate - convexity_adjustment
    # 2.5% - 5bps = 2.45%
    assert abs(future._forward_rate - 0.0245) < 1e-10
    assert abs(future._fixed_coupon - 0.0245) < 1e-10  # Used for curve building


###############################################################################
# Forward Rate from Curve Tests
###############################################################################


def test_ir_future_forward_rate_from_flat_curve(value_date, usd_sofr_imm_h24):
    """Test forward rate calculation from a flat discount curve."""
    # Create flat curve at 2.5%
    deposit = CashDeposit(
        value_date, "12M", 0.025, DayCountTypes.ACT_360,
        CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD
    )
    curve = OISCurve(
        value_dt=value_date,
        instruments=[deposit],
        interp_type=InterpTypes.FLAT_FWD_RATES,
        check_refit=False
    )

    # Forward rate should be approximately 2.5% (flat curve)
    fwd_rate = usd_sofr_imm_h24.forward_rate_from_curve(value_date, curve)
    assert abs(fwd_rate - 0.025) < 0.01  # Within 1% tolerance


def test_ir_future_forward_rate_consistency_with_dfs(value_date):
    """Test forward rate formula consistency."""
    # Create a simple futures
    future = IRFuture(
        effective_dt=value_date,
        expiry_date_or_contract="H24",
        futures_price=97.50,
        contract_type=FutureContractTypes.IMM,
        currency=CurrencyTypes.USD,
        floating_index=CurveTypes.USD_OIS_SOFR
    )

    # Create curve
    deposit = CashDeposit(
        value_date, "12M", 0.025, DayCountTypes.ACT_360,
        CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD
    )
    curve = OISCurve(
        value_dt=value_date,
        instruments=[deposit],
        interp_type=InterpTypes.FLAT_FWD_RATES,
        check_refit=False
    )

    # Manual forward rate calculation
    df_start = curve.df(future._accrual_start_dt, future._dc_type)
    df_end = curve.df(future._accrual_end_dt, future._dc_type)
    fwd_rate_manual = (df_start / df_end - 1.0) / future._year_frac

    # Method calculation
    fwd_rate_method = future.forward_rate_from_curve(value_date, curve)

    # Should match within tolerance
    assert abs(fwd_rate_manual - fwd_rate_method) < 1e-10


###############################################################################
# Valuation Tests
###############################################################################


def test_ir_future_value_zero_when_curve_matches_futures_rate(value_date):
    """Test that futures values to ~0 when curve matches futures implied rate."""
    # Create futures at 2.5%
    future = IRFuture(
        effective_dt=value_date,
        expiry_date_or_contract="H24",
        futures_price=97.50,  # 2.5% implied
        contract_type=FutureContractTypes.IMM,
        currency=CurrencyTypes.USD,
        floating_index=CurveTypes.USD_OIS_SOFR
    )

    # Create curve at 2.5% (matching futures rate)
    deposit = CashDeposit(
        value_date, "12M", 0.025, DayCountTypes.ACT_360,
        CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD
    )
    curve = OISCurve(
        value_dt=value_date,
        instruments=[deposit],
        interp_type=InterpTypes.FLAT_FWD_RATES,
        check_refit=False
    )

    # Value should be close to zero
    pv = future.value(value_date, curve)
    assert abs(pv) < 1000  # Within $1,000 for $1M notional


def test_ir_future_value_expired_is_zero(value_date):
    """Test that expired futures have zero value."""
    # Create futures expiring before valuation date
    past_date = Date(20, 1, 2023)  # January 2023 (past)
    future = IRFuture(
        effective_dt=past_date,
        expiry_date_or_contract=Date(20, 3, 2023),  # March 2023 expiry
        futures_price=97.50,
        contract_type=FutureContractTypes.IMM,
        currency=CurrencyTypes.USD,
        floating_index=CurveTypes.USD_OIS_SOFR
    )

    # Create dummy curve
    deposit = CashDeposit(
        value_date, "1Y", 0.03, DayCountTypes.ACT_360,
        CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD
    )
    curve = OISCurve(
        value_dt=value_date,
        instruments=[deposit],
        interp_type=InterpTypes.FLAT_FWD_RATES,
        check_refit=False
    )

    # Value of expired future should be zero
    pv = future.value(value_date, curve)
    assert pv == 0.0


def test_ir_future_value_no_curve_raises_error(value_date, usd_sofr_imm_h24):
    """Test that valuation without curve raises error."""
    with pytest.raises(ValueError, match="discount_curve or ois_curve must be provided"):
        usd_sofr_imm_h24.value(value_date)


###############################################################################
# Position Integration Tests
###############################################################################


def test_ir_future_position_creation(usd_sofr_imm_h24):
    """Test Position wrapper creation."""
    # Position requires a model, so we'll just test the method exists
    assert hasattr(usd_sofr_imm_h24, 'position')
    assert callable(usd_sofr_imm_h24.position)


###############################################################################
# Representation Tests
###############################################################################


def test_ir_future_repr(usd_sofr_imm_h24):
    """Test string representation."""
    repr_str = repr(usd_sofr_imm_h24)
    assert "IRFuture" in repr_str
    assert "IMM" in repr_str
    assert "USD" in repr_str


def test_ir_future_print_details(usd_sofr_imm_h24, capsys):
    """Test print_details method."""
    usd_sofr_imm_h24.print_details()
    captured = capsys.readouterr()
    assert "IRFuture" in captured.out
    assert "97.5000" in captured.out  # Futures price
    assert "2.5000%" in captured.out  # Implied rate


###############################################################################
# Edge Cases
###############################################################################


def test_ir_future_settlement_lag(value_date):
    """Test settlement lag (T+2 settlement)."""
    future = IRFuture(
        effective_dt=value_date,
        expiry_date_or_contract="H24",
        futures_price=97.50,
        contract_type=FutureContractTypes.IMM,
        currency=CurrencyTypes.USD,
        floating_index=CurveTypes.USD_OIS_SOFR,
        settlement_lag=2  # T+2 settlement
    )

    # Settlement date should be 2 business days after expiry
    # (Exact date depends on calendar, but should be different from expiry)
    assert future._settlement_dt >= future._expiry_dt


def test_ir_future_custom_contract_size(value_date):
    """Test custom contract size."""
    custom_size = 5_000_000  # $5M instead of standard $1M
    future = IRFuture(
        effective_dt=value_date,
        expiry_date_or_contract="H24",
        futures_price=97.50,
        contract_type=FutureContractTypes.IMM,
        currency=CurrencyTypes.USD,
        floating_index=CurveTypes.USD_OIS_SOFR,
        contract_size=custom_size
    )

    assert future._notional == custom_size


def test_ir_future_custom_day_count(value_date):
    """Test custom day count convention."""
    future = IRFuture(
        effective_dt=value_date,
        expiry_date_or_contract="H24",
        futures_price=97.50,
        contract_type=FutureContractTypes.IMM,
        currency=CurrencyTypes.USD,
        floating_index=CurveTypes.USD_OIS_SOFR,
        dc_type=DayCountTypes.ACT_365F  # Override default ACT/360
    )

    assert future._dc_type == DayCountTypes.ACT_365F


###############################################################################
# Multi-Currency Tests
###############################################################################


def test_ir_future_eur_estr(value_date):
    """Test EUR ESTR futures."""
    future = IRFuture(
        effective_dt=value_date,
        expiry_date_or_contract="H24",
        futures_price=96.75,  # 3.25% implied
        contract_type=FutureContractTypes.IMM,
        currency=CurrencyTypes.EUR,
        floating_index=CurveTypes.EUR_OIS_ESTR
    )

    assert future._currency == CurrencyTypes.EUR
    assert future._dc_type == DayCountTypes.ACT_360  # EUR default
    assert future._notional == 1_000_000  # €1M
    assert abs(future.implied_rate() - 0.0325) < 1e-10


###############################################################################
# Year Parsing Tests
###############################################################################


def test_ir_future_2_digit_year_2000s(value_date):
    """Test 2-digit year in 2000s range."""
    future = IRFuture(
        effective_dt=value_date,
        expiry_date_or_contract="H25",  # 2025
        futures_price=97.50,
        contract_type=FutureContractTypes.IMM,
        currency=CurrencyTypes.USD,
        floating_index=CurveTypes.USD_OIS_SOFR
    )

    assert future._expiry_dt._y == 2025


def test_ir_future_4_digit_year(value_date):
    """Test 4-digit year format."""
    future = IRFuture(
        effective_dt=value_date,
        expiry_date_or_contract="H2025",  # 4-digit year
        futures_price=97.50,
        contract_type=FutureContractTypes.IMM,
        currency=CurrencyTypes.USD,
        floating_index=CurveTypes.USD_OIS_SOFR
    )

    assert future._expiry_dt._y == 2025
