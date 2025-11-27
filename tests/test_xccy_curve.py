"""
Test suite for cross-currency curve construction.

Tests the XccyCurve class for building foreign-in-domestic discount curves
from cross-currency basis swap market quotes.
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


def test_xccy_curve_basic_construction():
    """Test basic XCCY curve construction with simple market data (using 1Y swaps only to avoid OISCurve bug)."""

    # Valuation date
    value_dt = Date(15, 6, 2023)

    # Build domestic (GBP SONIA) OIS curve - 1Y only to avoid OISCurve bug
    gbp_swaps = [
        OIS(
            effective_dt=value_dt,
            term_dt_or_tenor="1Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=0.0450,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_365F,
            floating_index=CurveTypes.GBP_OIS_SONIA,
            currency=CurrencyTypes.GBP,
            notional=1_000_000
        )
    ]

    gbp_curve = OISCurve(
        value_dt=value_dt,
        ois_swaps=gbp_swaps,
        interp_type=InterpTypes.FLAT_FWD_RATES,
        check_refit=True
    )

    # Build foreign (USD SOFR) OIS curve - 1Y only to avoid OISCurve bug
    usd_swaps = [
        OIS(
            effective_dt=value_dt,
            term_dt_or_tenor="1Y",
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=0.0520,
            fixed_freq_type=FrequencyTypes.ANNUAL,
            fixed_dc_type=DayCountTypes.ACT_360,
            floating_index=CurveTypes.USD_OIS_SOFR,
            currency=CurrencyTypes.USD,
            notional=1_000_000
        )
    ]

    usd_curve = OISCurve(
        value_dt=value_dt,
        ois_swaps=usd_swaps,
        interp_type=InterpTypes.FLAT_FWD_RATES,
        check_refit=True
    )

    # Spot FX rate: GBP per USD (e.g., 1 USD = 0.79 GBP)
    spot_fx = 0.79

    # Build XCCY basis swaps (USD/GBP) - 1Y only
    basis_swaps = [
        XccyBasisSwap(
            effective_dt=value_dt,
            term_dt_or_tenor="1Y",
            domestic_notional=spot_fx * 1_000_000,  # GBP
            foreign_notional=1_000_000,  # USD
            domestic_spread=0.0,
            foreign_spread=0.0025,  # 25bp
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

    # Bootstrap XCCY curve
    xccy_curve = XccyCurve(
        value_dt=value_dt,
        basis_swaps=basis_swaps,
        domestic_curve=gbp_curve,
        foreign_curve=usd_curve,
        spot_fx=spot_fx,
        interp_type=InterpTypes.FLAT_FWD_RATES,
        check_refit=True
    )

    # Basic assertions
    assert xccy_curve is not None
    assert len(xccy_curve._times) == 2  # t=0 plus 1 swap maturity
    assert len(xccy_curve._dfs) == 2

    # Check discount factors are positive and decreasing
    for i in range(len(xccy_curve._dfs) - 1):
        assert xccy_curve._dfs[i] > 0
        assert xccy_curve._dfs[i] >= xccy_curve._dfs[i+1]

    # Check we can query discount factors
    df_1y = xccy_curve.df(value_dt.add_years(1))
    assert df_1y > 0
    assert df_1y <= 1.0

    print("\nXCCY Curve constructed successfully!")
    print(xccy_curve)


def test_xccy_swap_valuation():
    """Test that XCCY swaps value correctly with the XCCY curve (using 1Y swaps only to avoid OISCurve bug)."""

    value_dt = Date(15, 6, 2023)

    # Simple flat curves for testing - 1Y only to avoid OISCurve bug
    gbp_swaps = [
        OIS(value_dt, "1Y", SwapTypes.PAY, 0.045, FrequencyTypes.ANNUAL,
            DayCountTypes.ACT_365F, CurveTypes.GBP_OIS_SONIA, CurrencyTypes.GBP)
    ]

    gbp_curve = OISCurve(value_dt, gbp_swaps, InterpTypes.FLAT_FWD_RATES, check_refit=True)

    usd_swaps = [
        OIS(value_dt, "1Y", SwapTypes.PAY, 0.050, FrequencyTypes.ANNUAL,
            DayCountTypes.ACT_360, CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD)
    ]

    usd_curve = OISCurve(value_dt, usd_swaps, InterpTypes.FLAT_FWD_RATES, check_refit=True)

    spot_fx = 0.79

    basis_swaps = [
        XccyBasisSwap(
            value_dt, "1Y", spot_fx * 1_000_000, 1_000_000, 0.0, 0.0020,
            FrequencyTypes.ANNUAL, FrequencyTypes.ANNUAL,
            DayCountTypes.ACT_365F, DayCountTypes.ACT_360,
            CurveTypes.GBP_OIS_SONIA, CurveTypes.USD_OIS_SOFR,
            CurrencyTypes.GBP, CurrencyTypes.USD
        )
    ]

    xccy_curve = XccyCurve(value_dt, basis_swaps, gbp_curve, usd_curve,
                           spot_fx, InterpTypes.FLAT_FWD_RATES, check_refit=True)

    # Test that we can value a basis swap
    test_swap = basis_swaps[0]
    pv = test_swap.value(value_dt, gbp_curve, usd_curve, xccy_curve, spot_fx)

    # Should be very close to zero (calibration instrument)
    assert abs(pv / test_swap._domestic_notional) < 1e-8

    print(f"\nTest swap PV: {pv}")
    print(f"Normalized PV: {pv / test_swap._domestic_notional}")


if __name__ == "__main__":
    print("Testing XCCY Curve Construction...")
    test_xccy_curve_basic_construction()
    print("\nTesting XCCY Swap Valuation...")
    test_xccy_swap_valuation()
    print("\nAll tests passed!")
