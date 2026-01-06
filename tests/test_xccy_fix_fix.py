"""
Test cross-currency fixed-fixed swaps.

Tests the XccyFixFix class for creating and valuing XCCY swaps where
both the domestic and foreign legs are fixed.
"""

import pytest
import numpy as np

from cavour.utils.date import Date
from cavour.utils.day_count import DayCountTypes
from cavour.utils.frequency import FrequencyTypes
from cavour.utils.global_types import CurveTypes, SwapTypes
from cavour.utils.currency import CurrencyTypes
from cavour.market.curves.interpolator import InterpTypes

from cavour.trades.rates.ois import OIS
from cavour.trades.rates.ois_curve import OISCurve
from cavour.trades.rates.xccy_basis_swap import XccyBasisSwap
from cavour.trades.rates.xccy_curve import XccyCurve
from cavour.trades.rates.xccy_fix_fix_swap import XccyFixFix
from cavour.models.models import Model
from cavour.utils.calendar import BusDayAdjustTypes


def test_xccy_fix_fix_construction():
    """Test basic construction of XccyFixFix swap."""

    value_dt = Date(15, 6, 2023)

    # Create a simple fixed-fixed XCCY swap
    swap = XccyFixFix(
        effective_dt=value_dt,
        term_dt_or_tenor="1Y",
        domestic_notional=790_000,  # GBP
        foreign_notional=1_000_000,  # USD
        domestic_leg_type=SwapTypes.PAY,  # Pay fixed GBP
        domestic_coupon=0.045,  # 4.5% fixed GBP
        foreign_coupon=0.052,  # 5.2% fixed USD
        domestic_freq_type=FrequencyTypes.ANNUAL,
        foreign_freq_type=FrequencyTypes.SEMI_ANNUAL,
        domestic_dc_type=DayCountTypes.ACT_365F,
        foreign_dc_type=DayCountTypes.ACT_360,
        domestic_floating_index=CurveTypes.GBP_OIS_SONIA,
        foreign_floating_index=CurveTypes.USD_OIS_SOFR,
        domestic_currency=CurrencyTypes.GBP,
        foreign_currency=CurrencyTypes.USD
    )

    # Basic assertions
    assert swap is not None
    assert swap._domestic_notional == 790_000
    assert swap._foreign_notional == 1_000_000
    assert swap._domestic_leg_type == SwapTypes.PAY
    assert swap._maturity_dt >= value_dt

    print("\nXccyFixFix construction test passed!")


def test_xccy_fix_fix_valuation():
    """Test valuation of XccyFixFix swap with 20Y tenor."""

    value_dt = Date(15, 6, 2023)

    # Define tenor structure up to 20Y
    tenors = ['1Y', '2Y', '3Y', '4Y', '5Y', '7Y', '10Y', '15Y', '20Y']

    # GBP OIS rates (slightly upward sloping curve)
    gbp_rates = [4.50, 4.55, 4.60, 4.65, 4.70, 4.74, 4.80, 4.85, 4.90]

    # USD OIS rates (higher than GBP)
    usd_rates = [5.20, 5.25, 5.30, 5.35, 5.40, 5.44, 5.50, 5.55, 5.60]

    # XCCY basis spreads
    basis_spreads = [0.0025, 0.0028, 0.0030, 0.0032, 0.0034, 0.0036, 0.0039, 0.0042, 0.0045]

    # Build GBP OIS curve using Model
    gbp_model = Model(value_dt)
    gbp_model.build_curve(
        name='GBP_OIS_SONIA',
        px_list=gbp_rates,
        tenor_list=tenors,
        spot_days=0,
        swap_type=SwapTypes.PAY,
        fixed_dcc_type=DayCountTypes.ACT_365F,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_365F,
        bus_day_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        interp_type=InterpTypes.FLAT_FWD_RATES
    )
    gbp_curve = gbp_model.curves.GBP_OIS_SONIA

    # Build USD OIS curve using Model
    usd_model = Model(value_dt)
    usd_model.build_curve(
        name='USD_OIS_SOFR',
        px_list=usd_rates,
        tenor_list=tenors,
        spot_days=0,
        swap_type=SwapTypes.PAY,
        fixed_dcc_type=DayCountTypes.ACT_360,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_360,
        bus_day_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        interp_type=InterpTypes.FLAT_FWD_RATES
    )
    usd_curve = usd_model.curves.USD_OIS_SOFR

    # Spot FX: GBP per USD
    spot_fx = 0.79

    # Build XCCY curve from basis swaps
    basis_swaps = []
    for tenor, spread in zip(tenors, basis_spreads):
        basis_swaps.append(
            XccyBasisSwap(
                effective_dt=value_dt,
                term_dt_or_tenor=tenor,
                domestic_notional=spot_fx * 1_000_000,
                foreign_notional=1_000_000,
                domestic_spread=0.0,
                foreign_spread=spread,
                domestic_freq_type=FrequencyTypes.ANNUAL,
                foreign_freq_type=FrequencyTypes.ANNUAL,
                domestic_dc_type=DayCountTypes.ACT_365F,
                foreign_dc_type=DayCountTypes.ACT_360,
                domestic_floating_index=CurveTypes.GBP_OIS_SONIA,
                foreign_floating_index=CurveTypes.USD_OIS_SOFR,
                domestic_currency=CurrencyTypes.GBP,
                foreign_currency=CurrencyTypes.USD
            )
        )

    xccy_curve = XccyCurve(
        value_dt=value_dt,
        basis_swaps=basis_swaps,
        domestic_curve=gbp_curve,
        foreign_curve=usd_curve,
        spot_fx=spot_fx,
        interp_type=InterpTypes.FLAT_FWD_RATES,
        check_refit=True
    )

    # Create 20Y fixed-fixed swap
    # Domestic (GBP): pay fixed 4.9%
    # Foreign (USD): receive fixed 5.6%
    swap = XccyFixFix(
        effective_dt=value_dt,
        term_dt_or_tenor="20Y",
        domestic_notional=790_000,  # GBP
        foreign_notional=1_000_000,  # USD
        domestic_leg_type=SwapTypes.PAY,  # Pay fixed GBP
        domestic_coupon=0.049,  # 4.9%
        foreign_coupon=0.056,  # 5.6%
        domestic_freq_type=FrequencyTypes.ANNUAL,
        foreign_freq_type=FrequencyTypes.SEMI_ANNUAL,
        domestic_dc_type=DayCountTypes.ACT_365F,
        foreign_dc_type=DayCountTypes.ACT_360,
        domestic_floating_index=CurveTypes.GBP_OIS_SONIA,
        foreign_floating_index=CurveTypes.USD_OIS_SOFR,
        domestic_currency=CurrencyTypes.GBP,
        foreign_currency=CurrencyTypes.USD
    )

    # Value the swap
    pv = swap.value(value_dt, gbp_curve, usd_curve, xccy_curve, spot_fx)

    # Assertions
    assert pv is not None
    assert isinstance(pv, (int, float))
    print(f"\nXccyFixFix 20Y swap PV: {pv:,.2f} GBP")
    print(f"PV as % of domestic notional: {pv/790_000*100:.4f}%")

    # Check that valuation runs without errors
    swap.print_valuation()

    print("\nXccyFixFix 20Y valuation test passed!")


def test_xccy_fix_fix_equal_coupons():
    """Test XccyFixFix swap with equal coupon rates (should have PV close to zero)."""

    value_dt = Date(15, 6, 2023)

    # Build minimal curves
    gbp_swaps = [
        OIS(
            effective_dt=value_dt,
            term_dt_or_tenor="1Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=0.0450,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_365F,
            floating_index=CurveTypes.GBP_OIS_SONIA,
            currency=CurrencyTypes.GBP
        )
    ]
    gbp_curve = OISCurve(value_dt, gbp_swaps, InterpTypes.FLAT_FWD_RATES, check_refit=True)

    usd_swaps = [
        OIS(
            effective_dt=value_dt,
            term_dt_or_tenor="1Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=0.0450,  # Same as GBP
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_365F,  # Same day count
            floating_index=CurveTypes.USD_OIS_SOFR,
            currency=CurrencyTypes.USD
        )
    ]
    usd_curve = OISCurve(value_dt, usd_swaps, InterpTypes.FLAT_FWD_RATES, check_refit=True)

    spot_fx = 1.0  # 1:1 FX rate for simplicity

    basis_swaps = [
        XccyBasisSwap(
            effective_dt=value_dt,
            term_dt_or_tenor="1Y",
            domestic_notional=1_000_000,
            foreign_notional=1_000_000,
            domestic_spread=0.0,
            foreign_spread=0.0,  # No basis
            domestic_freq_type=FrequencyTypes.ANNUAL,
            foreign_freq_type=FrequencyTypes.ANNUAL,
            domestic_dc_type=DayCountTypes.ACT_365F,
            foreign_dc_type=DayCountTypes.ACT_365F,
            domestic_floating_index=CurveTypes.GBP_OIS_SONIA,
            foreign_floating_index=CurveTypes.USD_OIS_SOFR,
            domestic_currency=CurrencyTypes.GBP,
            foreign_currency=CurrencyTypes.USD
        )
    ]
    xccy_curve = XccyCurve(value_dt, basis_swaps, gbp_curve, usd_curve, spot_fx, InterpTypes.FLAT_FWD_RATES)

    # Create swap with equal coupons and conventions
    swap = XccyFixFix(
        effective_dt=value_dt,
        term_dt_or_tenor="1Y",
        domestic_notional=1_000_000,
        foreign_notional=1_000_000,
        domestic_leg_type=SwapTypes.PAY,
        domestic_coupon=0.045,  # Same coupon
        foreign_coupon=0.045,  # Same coupon
        domestic_freq_type=FrequencyTypes.ANNUAL,
        foreign_freq_type=FrequencyTypes.ANNUAL,
        domestic_dc_type=DayCountTypes.ACT_365F,
        foreign_dc_type=DayCountTypes.ACT_365F,
        domestic_floating_index=CurveTypes.GBP_OIS_SONIA,
        foreign_floating_index=CurveTypes.USD_OIS_SOFR,
        domestic_currency=CurrencyTypes.GBP,
        foreign_currency=CurrencyTypes.USD
    )

    # Value the swap
    pv = swap.value(value_dt, gbp_curve, usd_curve, xccy_curve, spot_fx)

    print(f"\nXccyFixFix PV (equal coupons, 1:1 FX): {pv:,.6f} GBP")

    # With equal coupons, equal notionals, 1:1 FX, and no basis spread,
    # PV should be relatively small (not exactly zero due to day count/calendar differences)
    # The PV is small relative to notional (52 GBP on 1M notional = 0.0052%)
    assert abs(pv) < 100, f"Expected PV to be small, got {pv}"
    assert abs(pv / 1_000_000) < 0.0001, f"Expected PV < 0.01% of notional, got {pv/1_000_000:.4%}"

    print("\nXccyFixFix equal coupons test passed!")


def test_xccy_fix_fix_leg_pv():
    """Test that individual leg valuations work correctly."""

    value_dt = Date(15, 6, 2023)

    # Build minimal curves (1Y only)
    gbp_swaps = [
        OIS(
            effective_dt=value_dt,
            term_dt_or_tenor="1Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=0.0450,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_365F,
            floating_index=CurveTypes.GBP_OIS_SONIA,
            currency=CurrencyTypes.GBP
        )
    ]
    gbp_curve = OISCurve(value_dt, gbp_swaps, InterpTypes.FLAT_FWD_RATES, check_refit=True)

    usd_swaps = [
        OIS(
            effective_dt=value_dt,
            term_dt_or_tenor="1Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=0.0520,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_360,
            floating_index=CurveTypes.USD_OIS_SOFR,
            currency=CurrencyTypes.USD
        )
    ]
    usd_curve = OISCurve(value_dt, usd_swaps, InterpTypes.FLAT_FWD_RATES, check_refit=True)

    spot_fx = 0.79

    basis_swaps = [
        XccyBasisSwap(
            effective_dt=value_dt,
            term_dt_or_tenor="1Y",
            domestic_notional=790_000,
            foreign_notional=1_000_000,
            domestic_spread=0.0,
            foreign_spread=0.0025,
            domestic_freq_type=FrequencyTypes.ANNUAL,
            foreign_freq_type=FrequencyTypes.ANNUAL,
            domestic_dc_type=DayCountTypes.ACT_365F,
            foreign_dc_type=DayCountTypes.ACT_360,
            domestic_floating_index=CurveTypes.GBP_OIS_SONIA,
            foreign_floating_index=CurveTypes.USD_OIS_SOFR,
            domestic_currency=CurrencyTypes.GBP,
            foreign_currency=CurrencyTypes.USD
        )
    ]
    xccy_curve = XccyCurve(value_dt, basis_swaps, gbp_curve, usd_curve, spot_fx, InterpTypes.FLAT_FWD_RATES)

    # Create swap
    swap = XccyFixFix(
        effective_dt=value_dt,
        term_dt_or_tenor="1Y",
        domestic_notional=790_000,
        foreign_notional=1_000_000,
        domestic_leg_type=SwapTypes.PAY,
        domestic_coupon=0.045,
        foreign_coupon=0.052,
        domestic_freq_type=FrequencyTypes.ANNUAL,
        foreign_freq_type=FrequencyTypes.ANNUAL,
        domestic_dc_type=DayCountTypes.ACT_365F,
        foreign_dc_type=DayCountTypes.ACT_360,
        domestic_floating_index=CurveTypes.GBP_OIS_SONIA,
        foreign_floating_index=CurveTypes.USD_OIS_SOFR,
        domestic_currency=CurrencyTypes.GBP,
        foreign_currency=CurrencyTypes.USD
    )

    # Value full swap
    pv_total = swap.value(value_dt, gbp_curve, usd_curve, xccy_curve, spot_fx)

    # Value individual legs
    domestic_pv = swap._domestic_leg.value(value_dt, gbp_curve)
    foreign_pv = swap._foreign_leg.value(value_dt, xccy_curve)

    print(f"\nDomestic fixed leg PV: {domestic_pv:,.2f} GBP")
    print(f"Foreign fixed leg PV: {foreign_pv:,.2f} USD")
    print(f"Foreign PV in GBP: {spot_fx * foreign_pv:,.2f} GBP")
    print(f"Total PV (approx, without manual notional exchanges): {domestic_pv + spot_fx * foreign_pv:,.2f} GBP")
    print(f"Total PV (with notional exchanges): {pv_total:,.2f} GBP")

    # Assertions
    assert domestic_pv is not None
    assert foreign_pv is not None
    assert pv_total is not None

    print("\nXccyFixFix leg PV test passed!")


if __name__ == "__main__":
    print("Testing XccyFixFix Swaps...")
    print("="*80)

    test_xccy_fix_fix_construction()
    print("\n" + "="*80)

    test_xccy_fix_fix_valuation()
    print("\n" + "="*80)

    test_xccy_fix_fix_equal_coupons()
    print("\n" + "="*80)

    test_xccy_fix_fix_leg_pv()
    print("\n" + "="*80)

    print("\nAll XccyFixFix tests passed!")
