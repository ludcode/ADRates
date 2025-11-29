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
from cavour.utils.calendar import BusDayAdjustTypes
from cavour.utils.global_types import CurveTypes, SwapTypes
from cavour.utils.currency import CurrencyTypes
from cavour.market.curves.interpolator import InterpTypes
from cavour.models.models import Model

from cavour.trades.rates.xccy_basis_swap import XccyBasisSwap
from cavour.trades.rates.xccy_curve import XccyCurve


def test_xccy_curve_basic_construction():
    """Test XCCY curve construction with full basis curve (1Y to 20Y)."""

    # Valuation date
    value_dt = Date(15, 6, 2023)

    # Full tenor structure: 1Y-10Y annual, then 12Y, 15Y, 20Y
    tenors = ['1Y', '2Y', '3Y', '4Y', '5Y', '6Y', '7Y', '8Y', '9Y', '10Y', '12Y', '15Y', '20Y']

    # GBP OIS rates (slightly upward sloping curve)
    gbp_rates = [4.50, 4.55, 4.60, 4.65, 4.70, 4.72, 4.74, 4.76, 4.78, 4.80, 4.82, 4.85, 4.90]

    # USD OIS rates (higher than GBP, also upward sloping)
    usd_rates = [5.20, 5.25, 5.30, 5.35, 5.40, 5.42, 5.44, 5.46, 5.48, 5.50, 5.52, 5.55, 5.60]

    # XCCY basis spreads (widening with tenor)
    basis_spreads = [0.0025, 0.0028, 0.0030, 0.0032, 0.0034, 0.0035, 0.0036, 0.0037, 0.0038, 0.0039, 0.0040, 0.0042, 0.0045]

    # Build domestic (GBP SONIA) OIS curve
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

    # Build foreign (USD SOFR) OIS curve
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

    # Spot FX rate: GBP per USD
    spot_fx = 0.79

    # Build XCCY basis swaps for all tenors
    basis_swaps = []
    for tenor, spread in zip(tenors, basis_spreads):
        basis_swaps.append(
            XccyBasisSwap(
                effective_dt=value_dt,
                term_dt_or_tenor=tenor,
                domestic_notional=spot_fx * 1_000_000,  # GBP
                foreign_notional=1_000_000,  # USD
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

    # Bootstrap XCCY curve
    print(f"\n{'='*80}")
    print(f"Building XCCY curve with {len(tenors)} basis swaps...")
    print(f"Tenors: {tenors}")
    print(f"{'='*80}\n")

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
    # Note: curve may have intermediate nodes for payment dates between pillars
    assert len(xccy_curve._times) >= len(tenors) + 1  # At least t=0 plus all swap maturities
    assert len(xccy_curve._dfs) == len(xccy_curve._times)

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
    """Test that XCCY swaps value correctly with full curve (repricing check)."""

    value_dt = Date(15, 6, 2023)

    # Full tenor structure
    tenors = ['1Y', '2Y', '3Y', '4Y', '5Y', '6Y', '7Y', '8Y', '9Y', '10Y', '12Y', '15Y', '20Y']

    # Flat curves for simplicity
    gbp_rates = [4.50] * len(tenors)
    usd_rates = [5.00] * len(tenors)
    basis_spreads = [0.0020, 0.0022, 0.0024, 0.0026, 0.0028, 0.0029, 0.0030, 0.0031, 0.0032, 0.0033, 0.0034, 0.0036, 0.0040]

    # Build curves
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

    spot_fx = 0.79

    # Build basis swaps
    basis_swaps = []
    for tenor, spread in zip(tenors, basis_spreads):
        basis_swaps.append(
            XccyBasisSwap(
                value_dt, tenor, spot_fx * 1_000_000, 1_000_000, 0.0, spread,
                FrequencyTypes.ANNUAL, FrequencyTypes.ANNUAL,
                DayCountTypes.ACT_365F, DayCountTypes.ACT_360,
                CurveTypes.GBP_OIS_SONIA, CurveTypes.USD_OIS_SOFR,
                CurrencyTypes.GBP, CurrencyTypes.USD
            )
        )

    print(f"\n{'='*80}")
    print(f"Testing repricing with {len(tenors)} basis swaps...")
    print(f"{'='*80}\n")

    xccy_curve = XccyCurve(value_dt, basis_swaps, gbp_curve, usd_curve,
                           spot_fx, InterpTypes.FLAT_FWD_RATES, check_refit=True)

    # Test that calibration swaps reprice to zero
    print(f"\nRepricing check for calibration instruments:")
    for i, swap in enumerate(basis_swaps):
        pv = swap.value(value_dt, gbp_curve, usd_curve, xccy_curve, spot_fx)
        normalized_pv = pv / swap._domestic_notional
        print(f"  {tenors[i]:>4s}: PV = {pv:12.6e}, Normalized = {normalized_pv:12.6e}")

        # Check that it's close to zero
        assert abs(normalized_pv) < 1e-8, f"{tenors[i]} swap did not reprice: {normalized_pv}"

    print(f"\nAll swaps repriced successfully!")


if __name__ == "__main__":
    print("Testing XCCY Curve Construction...")
    test_xccy_curve_basic_construction()
    print("\nTesting XCCY Swap Valuation...")
    test_xccy_swap_valuation()
    print("\nAll tests passed!")
