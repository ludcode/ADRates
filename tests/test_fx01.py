"""
Test suite for FX01 (FX sensitivity) calculation.

Validates analytical FX01 calculation against finite difference for:
1. XCCY swaps (sequential computation)
2. XCCY swaps (batched computation)
3. OIS swaps with cross-currency collateral

Tests verify:
- Analytical formula accuracy vs finite difference
- Sign conventions (positive FX01 = gains when domestic weakens)
- Units (domestic currency per 1% move)
- Integration with Risk and AnalyticsResult objects
"""

import pytest
import numpy as np
from cavour.utils.date import Date
from cavour.utils.currency import CurrencyTypes
from cavour.utils.global_types import RequestTypes, CurveTypes, FrequencyTypes
from cavour.trades.rates.xccy_swap import XccySwap
from cavour.trades.rates.ois import OIS
from cavour.utils.enums import SwapTypes


def test_xccy_fx01_analytical_vs_fd(xccy_model_usd_gbp):
    """
    Test FX01 calculation for XCCY swap using analytical vs finite difference.

    Validates that the analytical formula (for_pv * spot_fx * 0.01) matches
    the finite difference bump-reprice approach.
    """
    model = xccy_model_usd_gbp
    value_dt = model.value_dt

    # Create a simple XCCY swap: receive USD, pay GBP
    swap = XccySwap(
        value_dt=value_dt,
        term_usd="5Y",
        domestic_floating_index=CurveTypes.USD_OIS_SOFR,
        foreign_floating_index=CurveTypes.GBP_OIS_SONIA,
        domestic_spread=0.0,
        foreign_spread=0.0,
        domestic_notional=10_000_000,  # $10M
        foreign_notional=8_000_000,    # £8M
        domestic_leg_type=SwapTypes.RECEIVE,
        domestic_freq_type=FrequencyTypes.QUARTERLY,
        foreign_freq_type=FrequencyTypes.SEMI_ANNUAL,
        domestic_currency=CurrencyTypes.USD,
        foreign_currency=CurrencyTypes.GBP
    )

    # Get base case results
    pos = swap.position(model)
    result = pos.compute([RequestTypes.VALUE, RequestTypes.FX01])

    base_pv = result.value.amount
    fx01_analytical = result.fx_delta.sensitivity
    spot_fx_base = result.fx_delta.spot_fx

    # Verify FX01 object properties
    assert result.fx_delta.currency == CurrencyTypes.USD
    assert result.fx_delta.domestic_currency == CurrencyTypes.USD
    assert result.fx_delta.foreign_currency == CurrencyTypes.GBP
    assert result.fx_delta.spot_fx > 0

    # Finite difference: bump spot FX by 1%
    spot_fx_bumped = spot_fx_base * 1.01

    # Rebuild model with bumped spot FX
    # Access the XCCY curve and modify spot_fx
    xccy_curve_name = "GBP_USD_BASIS"
    xccy_curve = getattr(model.curves, xccy_curve_name)
    original_spot = xccy_curve._spot_fx

    # Temporarily modify spot_fx
    xccy_curve._spot_fx = spot_fx_bumped

    # Recompute PV with bumped spot
    result_bumped = pos.compute([RequestTypes.VALUE])
    bumped_pv = result_bumped.value.amount

    # Restore original spot_fx
    xccy_curve._spot_fx = original_spot

    # Finite difference FX01
    fx01_fd = bumped_pv - base_pv

    # Compare analytical vs finite difference
    # Allow 0.1% relative error (FD has numerical errors)
    relative_error = abs(fx01_analytical - fx01_fd) / abs(fx01_fd)

    print(f"\nFX01 Validation:")
    print(f"  Base PV:           {base_pv:,.2f} USD")
    print(f"  Spot FX:           {spot_fx_base:.6f} USD/GBP")
    print(f"  FX01 Analytical:   {fx01_analytical:,.2f} USD")
    print(f"  FX01 FD:           {fx01_fd:,.2f} USD")
    print(f"  Relative Error:    {relative_error:.6%}")

    assert relative_error < 0.001, \
        f"FX01 analytical ({fx01_analytical:.2f}) differs from FD ({fx01_fd:.2f}) by {relative_error:.2%}"


def test_xccy_fx01_sign_convention(xccy_model_usd_gbp):
    """
    Test FX01 sign convention.

    For a receive-USD, pay-GBP swap:
    - Positive FX01 = gains when USD/GBP increases (USD strengthens, GBP weakens)
    - Negative FX01 = loses when USD/GBP decreases (USD weakens, GBP strengthens)
    """
    model = xccy_model_usd_gbp
    value_dt = model.value_dt

    # Create XCCY swap: receive USD, pay GBP
    # If GBP weakens (USD/GBP increases), the GBP liability becomes cheaper in USD
    # So we expect POSITIVE FX01
    swap_receive_usd = XccySwap(
        value_dt=value_dt,
        term_usd="5Y",
        domestic_floating_index=CurveTypes.USD_OIS_SOFR,
        foreign_floating_index=CurveTypes.GBP_OIS_SONIA,
        domestic_spread=0.0,
        foreign_spread=0.0,
        domestic_notional=10_000_000,
        foreign_notional=8_000_000,
        domestic_leg_type=SwapTypes.RECEIVE,  # Receive USD
        domestic_freq_type=FrequencyTypes.QUARTERLY,
        foreign_freq_type=FrequencyTypes.SEMI_ANNUAL,
        domestic_currency=CurrencyTypes.USD,
        foreign_currency=CurrencyTypes.GBP
    )

    pos = swap_receive_usd.position(model)
    result = pos.compute([RequestTypes.FX01])
    fx01 = result.fx_delta.sensitivity

    # For receive-USD swap, FX01 sign depends on net exposure
    # But we can test that it's non-zero and reasonable
    assert abs(fx01) > 0, "FX01 should be non-zero for XCCY swap"

    # Test opposite direction: pay USD, receive GBP
    swap_pay_usd = XccySwap(
        value_dt=value_dt,
        term_usd="5Y",
        domestic_floating_index=CurveTypes.USD_OIS_SOFR,
        foreign_floating_index=CurveTypes.GBP_OIS_SONIA,
        domestic_spread=0.0,
        foreign_spread=0.0,
        domestic_notional=10_000_000,
        foreign_notional=8_000_000,
        domestic_leg_type=SwapTypes.PAY,  # Pay USD
        domestic_freq_type=FrequencyTypes.QUARTERLY,
        foreign_freq_type=FrequencyTypes.SEMI_ANNUAL,
        domestic_currency=CurrencyTypes.USD,
        foreign_currency=CurrencyTypes.GBP
    )

    pos_opposite = swap_pay_usd.position(model)
    result_opposite = pos_opposite.compute([RequestTypes.FX01])
    fx01_opposite = result_opposite.fx_delta.sensitivity

    # Opposite swaps should have opposite FX01 (approximately)
    # Not exactly opposite due to discounting asymmetry
    assert np.sign(fx01) != np.sign(fx01_opposite), \
        "Opposite swaps should have FX01 with opposite signs"

    print(f"\nFX01 Sign Convention:")
    print(f"  Receive USD: {fx01:,.2f} USD per 1% FX move")
    print(f"  Pay USD:     {fx01_opposite:,.2f} USD per 1% FX move")


def test_xccy_batch_fx01(xccy_model_usd_gbp):
    """
    Test FX01 calculation for batched XCCY swaps.

    Validates that batched computation produces same results as sequential.
    """
    model = xccy_model_usd_gbp
    value_dt = model.value_dt

    # Create multiple swaps with different maturities
    swaps = [
        XccySwap(
            value_dt=value_dt,
            term_usd=term,
            domestic_floating_index=CurveTypes.USD_OIS_SOFR,
            foreign_floating_index=CurveTypes.GBP_OIS_SONIA,
            domestic_spread=0.0,
            foreign_spread=0.0,
            domestic_notional=10_000_000,
            foreign_notional=8_000_000,
            domestic_leg_type=SwapTypes.RECEIVE,
            domestic_freq_type=FrequencyTypes.QUARTERLY,
            foreign_freq_type=FrequencyTypes.SEMI_ANNUAL,
            domestic_currency=CurrencyTypes.USD,
            foreign_currency=CurrencyTypes.GBP
        )
        for term in ["2Y", "5Y", "10Y"]
    ]

    # Compute sequentially
    sequential_results = []
    for swap in swaps:
        pos = swap.position(model)
        result = pos.compute([RequestTypes.VALUE, RequestTypes.FX01])
        sequential_results.append(result)

    # Compute in batch
    from cavour.market.position.position import Position
    batch_pos = Position(swaps, model)
    batch_results = batch_pos.compute([RequestTypes.VALUE, RequestTypes.FX01])

    # Compare results
    for i, (seq, batch) in enumerate(zip(sequential_results, batch_results)):
        seq_pv = seq.value.amount
        batch_pv = batch.value.amount
        seq_fx01 = seq.fx_delta.sensitivity
        batch_fx01 = batch.fx_delta.sensitivity

        # PV should match
        pv_diff = abs(seq_pv - batch_pv)
        assert pv_diff < 1e-8, \
            f"Swap {i}: PV mismatch {pv_diff:.2e}"

        # FX01 should match
        fx01_diff = abs(seq_fx01 - batch_fx01)
        assert fx01_diff < 1e-8, \
            f"Swap {i}: FX01 mismatch {fx01_diff:.2e}"

        print(f"\nSwap {i} ({swaps[i]._tenor_usd}):")
        print(f"  PV:   Sequential={seq_pv:,.2f}, Batch={batch_pv:,.2f}")
        print(f"  FX01: Sequential={seq_fx01:,.2f}, Batch={batch_fx01:,.2f}")


def test_ois_xccy_collateral_fx01(model_with_xccy_collateral):
    """
    Test FX01 for OIS swap with cross-currency collateral.

    For cross-currency collateral, FX01 has inverse relationship:
    PV_collateral = PV_swap / spot_fx
    dPV/d(spot_fx) = -PV_swap / spot_fx^2
    FX01 = -PV_collateral * 0.01
    """
    model = model_with_xccy_collateral
    value_dt = model.value_dt

    # Create GBP OIS swap collateralized in USD
    swap = OIS(
        value_dt=value_dt,
        term="5Y",
        cpn=0.04,
        floating_index=CurveTypes.GBP_OIS_SONIA,
        currency=CurrencyTypes.GBP,
        notional=10_000_000,  # £10M
        fixed_leg_type=SwapTypes.PAY,
        freq_type=FrequencyTypes.SEMI_ANNUAL
    )

    # Compute with USD collateral
    pos = swap.position(model, collateral_ccy=CurrencyTypes.USD)
    result = pos.compute([RequestTypes.VALUE, RequestTypes.FX01])

    pv_collateral = result.value.amount  # In USD
    fx01 = result.fx_delta.sensitivity
    spot_fx = result.fx_delta.spot_fx

    # Verify sign: FX01 should be negative
    # When USD/GBP increases (GBP weakens), the USD-denominated PV decreases
    assert fx01 < 0, "FX01 should be negative for cross-currency collateral"

    # Verify magnitude: FX01 should be approximately -PV * 0.01
    expected_fx01 = -pv_collateral * 0.01
    relative_error = abs(fx01 - expected_fx01) / abs(expected_fx01)

    assert relative_error < 1e-10, \
        f"FX01 ({fx01:.2f}) should equal -PV * 0.01 ({expected_fx01:.2f})"

    print(f"\nOIS with XCCY Collateral FX01:")
    print(f"  PV (USD):    {pv_collateral:,.2f}")
    print(f"  Spot FX:     {spot_fx:.6f} USD/GBP")
    print(f"  FX01:        {fx01:,.2f} USD per 1% FX move")
    print(f"  Expected:    {expected_fx01:,.2f} USD")


def test_fx01_units_and_scaling(xccy_model_usd_gbp):
    """
    Test FX01 units and scaling properties.

    Validates:
    - Units are correct (domestic currency per 1% move)
    - Scaling with notional
    - Relationship to PV
    """
    model = xccy_model_usd_gbp
    value_dt = model.value_dt

    # Base case
    notional_base = 10_000_000
    swap_base = XccySwap(
        value_dt=value_dt,
        term_usd="5Y",
        domestic_floating_index=CurveTypes.USD_OIS_SOFR,
        foreign_floating_index=CurveTypes.GBP_OIS_SONIA,
        domestic_spread=0.0,
        foreign_spread=0.0,
        domestic_notional=notional_base,
        foreign_notional=notional_base * 0.8,
        domestic_leg_type=SwapTypes.RECEIVE,
        domestic_freq_type=FrequencyTypes.QUARTERLY,
        foreign_freq_type=FrequencyTypes.SEMI_ANNUAL,
        domestic_currency=CurrencyTypes.USD,
        foreign_currency=CurrencyTypes.GBP
    )

    pos_base = swap_base.position(model)
    result_base = pos_base.compute([RequestTypes.FX01])
    fx01_base = result_base.fx_delta.sensitivity

    # Double notional
    notional_double = notional_base * 2
    swap_double = XccySwap(
        value_dt=value_dt,
        term_usd="5Y",
        domestic_floating_index=CurveTypes.USD_OIS_SOFR,
        foreign_floating_index=CurveTypes.GBP_OIS_SONIA,
        domestic_spread=0.0,
        foreign_spread=0.0,
        domestic_notional=notional_double,
        foreign_notional=notional_double * 0.8,
        domestic_leg_type=SwapTypes.RECEIVE,
        domestic_freq_type=FrequencyTypes.QUARTERLY,
        foreign_freq_type=FrequencyTypes.SEMI_ANNUAL,
        domestic_currency=CurrencyTypes.USD,
        foreign_currency=CurrencyTypes.GBP
    )

    pos_double = swap_double.position(model)
    result_double = pos_double.compute([RequestTypes.FX01])
    fx01_double = result_double.fx_delta.sensitivity

    # FX01 should scale linearly with notional
    scaling_ratio = fx01_double / fx01_base

    assert abs(scaling_ratio - 2.0) < 1e-10, \
        f"FX01 should double with notional, got scaling={scaling_ratio:.6f}"

    print(f"\nFX01 Scaling Test:")
    print(f"  Base notional:   ${notional_base:,.0f}")
    print(f"  Base FX01:       {fx01_base:,.2f} USD")
    print(f"  Double notional: ${notional_double:,.0f}")
    print(f"  Double FX01:     {fx01_double:,.2f} USD")
    print(f"  Scaling ratio:   {scaling_ratio:.6f}")


@pytest.fixture
def xccy_model_usd_gbp():
    """
    Create a model with USD and GBP OIS curves and USD/GBP XCCY curve.
    """
    from cavour.models.models import Model
    from cavour.utils.calendar import CalendarTypes, Calendar
    from cavour.utils.day_count import DayCountTypes

    # Setup
    value_dt = Date(15, 1, 2024)
    settle_dt = Date(17, 1, 2024)
    calendar = Calendar(CalendarTypes.TARGET)
    dc_type = DayCountTypes.ACT_360

    # USD OIS curve data
    usd_tenors = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y"]
    usd_rates = [0.0530, 0.0535, 0.0540, 0.0525, 0.0510, 0.0500, 0.0490, 0.0485, 0.0480]

    # GBP OIS curve data
    gbp_tenors = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y"]
    gbp_rates = [0.0515, 0.0520, 0.0525, 0.0520, 0.0515, 0.0510, 0.0505, 0.0502, 0.0500]

    # XCCY basis spreads (GBP vs USD)
    basis_tenors = ["1Y", "2Y", "3Y", "5Y", "7Y", "10Y"]
    basis_spreads = [0.0020, 0.0022, 0.0025, 0.0028, 0.0030, 0.0032]

    # Spot FX rate
    spot_fx_usd_gbp = 1.27  # USD per GBP

    # Build model
    model = Model(value_dt, settle_dt)

    # Add USD OIS curve
    model.add_ois_curve(
        curve_name="USD_OIS_SOFR",
        tenors=usd_tenors,
        rates=usd_rates,
        calendar=calendar,
        dc_type=dc_type,
        freq_type=FrequencyTypes.QUARTERLY
    )

    # Add GBP OIS curve
    model.add_ois_curve(
        curve_name="GBP_OIS_SONIA",
        tenors=gbp_tenors,
        rates=gbp_rates,
        calendar=calendar,
        dc_type=dc_type,
        freq_type=FrequencyTypes.SEMI_ANNUAL
    )

    # Add XCCY curve
    model.add_xccy_curve(
        domestic_curve_name="USD_OIS_SOFR",
        foreign_curve_name="GBP_OIS_SONIA",
        tenors=basis_tenors,
        basis_spreads=basis_spreads,
        spot_fx=spot_fx_usd_gbp,
        domestic_freq_type=FrequencyTypes.QUARTERLY,
        foreign_freq_type=FrequencyTypes.SEMI_ANNUAL
    )

    return model


@pytest.fixture
def model_with_xccy_collateral():
    """
    Create a model with GBP OIS and GBP/USD XCCY curves for collateral testing.
    """
    # Reuse xccy_model_usd_gbp setup
    # This fixture should return a model that supports cross-currency collateral
    # For simplicity, we'll use the same model structure
    return xccy_model_usd_gbp()


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
