"""
Test cross-currency fixed-float swaps.

Tests the XccyFixFloat class for creating and valuing XCCY swaps where
the domestic leg is fixed and the foreign leg is floating.
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
from cavour.trades.rates.xccy_fix_float_swap import XccyFixFloat
from cavour.trades.rates.xccy_fix_fix_swap import XccyFixFix
from cavour.models.models import Model
from cavour.utils.calendar import BusDayAdjustTypes


def test_xccy_fix_float_construction():
    """Test basic construction of XccyFixFloat swap."""

    value_dt = Date(15, 6, 2023)

    # Create a simple fixed-float XCCY swap
    swap = XccyFixFloat(
        effective_dt=value_dt,
        term_dt_or_tenor="1Y",
        domestic_notional=790_000,  # GBP
        foreign_notional=1_000_000,  # USD
        domestic_leg_type=SwapTypes.PAY,  # Pay fixed GBP
        domestic_coupon=0.045,  # 4.5% fixed
        foreign_spread=0.0025,  # 25bp spread on USD float
        domestic_freq_type=FrequencyTypes.ANNUAL,
        foreign_freq_type=FrequencyTypes.QUARTERLY,
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

    print("\nXccyFixFloat construction test passed!")


def test_xccy_fix_float_valuation():
    """Test valuation of XccyFixFloat swap with 20Y tenor."""

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

    # Create 20Y fixed-float swap
    # Domestic (GBP): pay fixed 4.9%
    # Foreign (USD): receive floating SOFR + 45bp
    swap = XccyFixFloat(
        effective_dt=value_dt,
        term_dt_or_tenor="20Y",
        domestic_notional=790_000,  # GBP
        foreign_notional=1_000_000,  # USD
        domestic_leg_type=SwapTypes.PAY,  # Pay fixed GBP
        domestic_coupon=0.049,  # 4.9%
        foreign_spread=0.0045,  # 45bp
        domestic_freq_type=FrequencyTypes.ANNUAL,
        foreign_freq_type=FrequencyTypes.QUARTERLY,
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
    print(f"\nXccyFixFloat 20Y swap PV: {pv:,.2f} GBP")
    print(f"PV as % of domestic notional: {pv/790_000*100:.4f}%")

    # Check that valuation runs without errors
    swap.print_valuation()

    print("\nXccyFixFloat 20Y valuation test passed!")


def test_xccy_fix_float_leg_pv():
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
    swap = XccyFixFloat(
        effective_dt=value_dt,
        term_dt_or_tenor="1Y",
        domestic_notional=790_000,
        foreign_notional=1_000_000,
        domestic_leg_type=SwapTypes.PAY,
        domestic_coupon=0.045,
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

    # Value full swap
    pv_total = swap.value(value_dt, gbp_curve, usd_curve, xccy_curve, spot_fx)

    # Value individual legs
    domestic_pv = swap._domestic_leg.value(value_dt, gbp_curve)
    foreign_pv = swap._foreign_leg.value(value_dt, xccy_curve, usd_curve)

    print(f"\nDomestic fixed leg PV: {domestic_pv:,.2f} GBP")
    print(f"Foreign floating leg PV: {foreign_pv:,.2f} USD")
    print(f"Foreign PV in GBP: {spot_fx * foreign_pv:,.2f} GBP")
    print(f"Total PV (approx, without manual notional exchanges): {domestic_pv + spot_fx * foreign_pv:,.2f} GBP")
    print(f"Total PV (with notional exchanges): {pv_total:,.2f} GBP")

    # Assertions
    assert domestic_pv is not None
    assert foreign_pv is not None
    assert pv_total is not None

    print("\nXccyFixFloat leg PV test passed!")


def test_xccy_fix_float_decomposition():
    """
    Test #5: Cross-validation using decomposition principle.

    Theory: A cross-currency fixed-float swap can be decomposed as:
        XCCY Fixed-Float = XCCY Fixed-Fixed + Vanilla Foreign Swap

    Where:
    - If domestic pays fixed, foreign receives floating:
      PV(fix-float) = PV(fix-fix with same domestic, foreign fixed at par)
                      - spot_fx * PV(vanilla foreign swap: pay floating, receive fixed at par)

    This test validates the fixed-float pricing by constructing it synthetically
    from a fixed-fixed swap and a vanilla foreign OIS swap.
    """

    value_dt = Date(15, 6, 2023)

    # Define tenor structure
    tenors = ['1Y', '2Y', '3Y', '4Y', '5Y', '7Y', '10Y']

    # GBP OIS rates
    gbp_rates = [4.50, 4.55, 4.60, 4.65, 4.70, 4.74, 4.80]

    # USD OIS rates
    usd_rates = [5.20, 5.25, 5.30, 5.35, 5.40, 5.44, 5.50]

    # XCCY basis spreads
    basis_spreads = [0.0025, 0.0028, 0.0030, 0.0032, 0.0034, 0.0036, 0.0039]

    # Build GBP OIS curve
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

    # Build USD OIS curve
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

    # Spot FX
    spot_fx = 0.79

    # Build XCCY curve
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

    # Test parameters
    tenor = "5Y"
    domestic_notional = 790_000
    foreign_notional = 1_000_000
    domestic_coupon = 0.047  # 4.7% fixed GBP
    foreign_spread = 0.0034  # 34bp spread on USD floating

    # Create the fixed-float swap (what we want to validate)
    fix_float_swap = XccyFixFloat(
        effective_dt=value_dt,
        term_dt_or_tenor=tenor,
        domestic_notional=domestic_notional,
        foreign_notional=foreign_notional,
        domestic_leg_type=SwapTypes.PAY,  # Pay fixed GBP
        domestic_coupon=domestic_coupon,
        foreign_spread=foreign_spread,
        domestic_freq_type=FrequencyTypes.ANNUAL,
        foreign_freq_type=FrequencyTypes.QUARTERLY,
        domestic_dc_type=DayCountTypes.ACT_365F,
        foreign_dc_type=DayCountTypes.ACT_360,
        domestic_floating_index=CurveTypes.GBP_OIS_SONIA,
        foreign_floating_index=CurveTypes.USD_OIS_SOFR,
        domestic_currency=CurrencyTypes.GBP,
        foreign_currency=CurrencyTypes.USD
    )

    pv_fix_float = fix_float_swap.value(value_dt, gbp_curve, usd_curve, xccy_curve, spot_fx)

    # Find the par fixed rate for USD OIS at 5Y with quarterly frequency
    # Binary search for par rate (quarterly compounding)
    low, high = 0.03, 0.08
    par_foreign_rate = None
    for _ in range(100):
        mid = (low + high) / 2

        # Reconstruct OIS each iteration with new coupon
        test_ois = OIS(
            effective_dt=value_dt,
            term_dt_or_tenor=tenor,
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=mid,
            fixed_freq_type=FrequencyTypes.QUARTERLY,
            fixed_dc_type=DayCountTypes.ACT_360,
            floating_index=CurveTypes.USD_OIS_SOFR,
            currency=CurrencyTypes.USD,
            notional=foreign_notional
        )

        pv = test_ois.value(value_dt, usd_curve)

        if abs(pv) < 1e-4:
            par_foreign_rate = mid
            break
        if pv > 0:
            low = mid
        else:
            high = mid

    if par_foreign_rate is None:
        par_foreign_rate = mid

    print(f"Par foreign fixed rate (quarterly): {par_foreign_rate*100:.6f}%")

    # Create fixed-fixed swap with foreign leg at par + spread
    foreign_fixed_rate = par_foreign_rate + foreign_spread

    fix_fix_swap = XccyFixFix(
        effective_dt=value_dt,
        term_dt_or_tenor=tenor,
        domestic_notional=domestic_notional,
        foreign_notional=foreign_notional,
        domestic_leg_type=SwapTypes.PAY,  # Pay fixed GBP
        domestic_coupon=domestic_coupon,
        foreign_coupon=foreign_fixed_rate,  # Par + spread
        domestic_freq_type=FrequencyTypes.ANNUAL,
        foreign_freq_type=FrequencyTypes.QUARTERLY,
        domestic_dc_type=DayCountTypes.ACT_365F,
        foreign_dc_type=DayCountTypes.ACT_360,
        domestic_floating_index=CurveTypes.GBP_OIS_SONIA,
        foreign_floating_index=CurveTypes.USD_OIS_SOFR,
        domestic_currency=CurrencyTypes.GBP,
        foreign_currency=CurrencyTypes.USD
    )

    pv_fix_fix = fix_fix_swap.value(value_dt, gbp_curve, usd_curve, xccy_curve, spot_fx)

    # Create floating leg component valued with XCCY curve (not vanilla OIS!)
    # The key insight: we need to value the floating leg with the XCCY curve, not USD curve
    from cavour.trades.rates.swap_float_leg import SwapFloatLeg

    # Create foreign floating leg identical to the one in fix-float swap
    foreign_float_leg = SwapFloatLeg(
        effective_dt=value_dt,
        end_dt=value_dt.add_tenor(tenor),
        leg_type=SwapTypes.RECEIVE,  # Receive floating (from domestic perspective)
        spread=foreign_spread,
        freq_type=FrequencyTypes.QUARTERLY,
        dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD,
        notional=foreign_notional,
        principal=0.0,
        payment_lag=0,
        notional_exchange=True  # Include notional exchanges
    )

    # Value with XCCY curve (projection with USD curve, discounting with XCCY curve)
    pv_foreign_float = foreign_float_leg.value(value_dt, xccy_curve, usd_curve)

    # Create foreign fixed leg identical to the one in fix-fix swap
    from cavour.trades.rates.swap_fixed_leg import SwapFixedLeg

    foreign_fixed_leg = SwapFixedLeg(
        effective_dt=value_dt,
        end_dt=value_dt.add_tenor(tenor),
        leg_type=SwapTypes.RECEIVE,  # Receive fixed (from domestic perspective)
        coupon=foreign_fixed_rate,
        freq_type=FrequencyTypes.QUARTERLY,
        dc_type=DayCountTypes.ACT_360,
        floating_index=CurveTypes.USD_OIS_SOFR,
        currency=CurrencyTypes.USD,
        notional=foreign_notional,
        principal=0.0,
        payment_lag=0
    )

    # Value with XCCY curve
    pv_foreign_fixed = foreign_fixed_leg.value(value_dt, xccy_curve)

    # Add manual notional exchanges for fixed leg
    df_start = xccy_curve.df(value_dt)
    df_end = xccy_curve.df(value_dt.add_tenor(tenor))
    notional_pv = -foreign_notional * df_start + foreign_notional * df_end
    pv_foreign_fixed += notional_pv

    # Synthetic: Replace fixed leg with floating leg
    # Fix-Float = domestic_fixed + spot_fx * foreign_float
    # Fix-Fix = domestic_fixed + spot_fx * foreign_fixed
    # Therefore: Fix-Float = Fix-Fix - spot_fx * (foreign_fixed - foreign_float)
    pv_synthetic = pv_fix_fix - spot_fx * (pv_foreign_fixed - pv_foreign_float)

    # Compare
    print("\n" + "="*80)
    print("DECOMPOSITION TEST RESULTS:")
    print("="*80)
    print(f"Direct Fixed-Float PV:      {pv_fix_float:>15,.2f} GBP")
    print(f"Fixed-Fixed PV:             {pv_fix_fix:>15,.2f} GBP")
    print(f"Foreign Fixed Leg PV (USD): {pv_foreign_fixed:>15,.2f} USD")
    print(f"Foreign Float Leg PV (USD): {pv_foreign_float:>15,.2f} USD")
    print(f"Leg Difference (USD):       {pv_foreign_fixed - pv_foreign_float:>15,.2f} USD")
    print(f"Leg Difference (GBP):       {spot_fx * (pv_foreign_fixed - pv_foreign_float):>15,.2f} GBP")
    print(f"Synthetic Fixed-Float PV:   {pv_synthetic:>15,.2f} GBP")
    print("="*80)
    print(f"Difference (Direct - Synthetic): {pv_fix_float - pv_synthetic:>10,.2f} GBP")
    print(f"Relative Difference:             {abs(pv_fix_float - pv_synthetic) / abs(domestic_notional) * 100:>10,.6f}%")

    # Assertion
    relative_error = abs(pv_fix_float - pv_synthetic) / abs(domestic_notional)

    # Allow for small numerical differences due to different schedule generation
    # and discounting approaches between fix-fix and vanilla swap
    assert relative_error < 0.001, f"Decomposition mismatch: {relative_error*100:.6f}% of notional"

    print("\nDecomposition test passed - Fixed-Float pricing validated!")


if __name__ == "__main__":
    print("Testing XccyFixFloat Swaps...")
    print("="*80)

    test_xccy_fix_float_construction()
    print("\n" + "="*80)

    test_xccy_fix_float_valuation()
    print("\n" + "="*80)

    test_xccy_fix_float_leg_pv()
    print("\n" + "="*80)

    test_xccy_fix_float_decomposition()
    print("\n" + "="*80)

    print("\nAll XccyFixFloat tests passed!")
