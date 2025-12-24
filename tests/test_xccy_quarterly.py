"""
Test XCCY curve with quarterly payment frequency.

This test verifies that the XccyCurve bootstrapping works correctly
when basis swaps use quarterly payment frequencies instead of annual.
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


def test_xccy_curve_quarterly_frequency():
    """Test XCCY curve construction with quarterly payment frequency."""

    # Valuation date
    value_dt = Date(15, 6, 2023)

    # Use fewer tenors for this test to keep it simple
    tenors = ['1Y', '2Y', '3Y', '5Y', '7Y', '10Y']

    # GBP OIS rates (slightly upward sloping curve)
    gbp_rates = [4.50, 4.55, 4.60, 4.70, 4.74, 4.80]

    # USD OIS rates (higher than GBP)
    usd_rates = [5.20, 5.25, 5.30, 5.40, 5.44, 5.50]

    # XCCY basis spreads
    basis_spreads = [0.0025, 0.0028, 0.0030, 0.0034, 0.0036, 0.0039]

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

    # Build XCCY basis swaps with QUARTERLY frequency
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
                domestic_freq_type=FrequencyTypes.QUARTERLY,  # Quarterly
                foreign_freq_type=FrequencyTypes.QUARTERLY,   # Quarterly
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
    print(f"Building XCCY curve with QUARTERLY frequency and {len(tenors)} basis swaps...")
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
    # With quarterly frequency, we expect many more nodes (payment dates)
    print(f"Number of curve nodes: {len(xccy_curve._times)}")
    print(f"Number of swaps: {len(tenors)}")
    assert len(xccy_curve._times) >= len(tenors) + 1
    assert len(xccy_curve._dfs) == len(xccy_curve._times)

    # Check discount factors are positive and decreasing
    for i in range(len(xccy_curve._dfs) - 1):
        assert xccy_curve._dfs[i] > 0
        assert xccy_curve._dfs[i] >= xccy_curve._dfs[i+1]

    # Check we can query discount factors
    df_1y = xccy_curve.df(value_dt.add_years(1))
    assert df_1y > 0
    assert df_1y <= 1.0

    print("\nXCCY Curve with quarterly frequency constructed successfully!")
    print(xccy_curve)

    # Test repricing - swaps should value to zero
    print(f"\n{'='*80}")
    print(f"Repricing check for quarterly calibration instruments:")
    print(f"{'='*80}\n")

    max_error = 0.0
    for i, swap in enumerate(basis_swaps):
        pv = swap.value(value_dt, gbp_curve, usd_curve, xccy_curve, spot_fx)
        normalized_pv = pv / swap._domestic_notional
        max_error = max(max_error, abs(normalized_pv))
        print(f"  {tenors[i]:>4s}: PV = {pv:12.6e}, Normalized = {normalized_pv:12.6e}")

        # Check that it's close to zero
        assert abs(normalized_pv) < 1e-8, f"{tenors[i]} swap did not reprice: {normalized_pv}"

    print(f"\nAll swaps repriced successfully!")
    print(f"Maximum repricing error: {max_error:.2e}")


def test_xccy_curve_mixed_frequency():
    """Test XCCY curve with mixed frequencies (domestic quarterly, foreign semi-annual)."""

    value_dt = Date(15, 6, 2023)
    tenors = ['1Y', '2Y', '5Y']

    gbp_rates = [4.50, 4.55, 4.70]
    usd_rates = [5.20, 5.25, 5.40]
    basis_spreads = [0.0025, 0.0028, 0.0034]

    # Build curves (same as before)
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

    # Build XCCY basis swaps with MIXED frequencies
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
                domestic_freq_type=FrequencyTypes.QUARTERLY,     # Quarterly
                foreign_freq_type=FrequencyTypes.SEMI_ANNUAL,   # Semi-annual
                domestic_dc_type=DayCountTypes.ACT_365F,
                foreign_dc_type=DayCountTypes.ACT_360,
                domestic_floating_index=CurveTypes.GBP_OIS_SONIA,
                foreign_floating_index=CurveTypes.USD_OIS_SOFR,
                domestic_currency=CurrencyTypes.GBP,
                foreign_currency=CurrencyTypes.USD
            )
        )

    print(f"\n{'='*80}")
    print(f"Building XCCY curve with MIXED frequencies (domestic quarterly, foreign semi-annual)...")
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

    assert xccy_curve is not None
    print(f"Number of curve nodes: {len(xccy_curve._times)}")

    # Test repricing
    print(f"\nRepricing check:")
    for i, swap in enumerate(basis_swaps):
        pv = swap.value(value_dt, gbp_curve, usd_curve, xccy_curve, spot_fx)
        normalized_pv = pv / swap._domestic_notional
        print(f"  {tenors[i]:>4s}: Normalized PV = {normalized_pv:12.6e}")
        assert abs(normalized_pv) < 1e-8

    print(f"\nMixed frequency test passed!")


if __name__ == "__main__":
    print("Testing XCCY Curve with Quarterly Frequency...")
    test_xccy_curve_quarterly_frequency()
    print("\n" + "="*80)
    print("\nTesting XCCY Curve with Mixed Frequencies...")
    test_xccy_curve_mixed_frequency()
    print("\n" + "="*80)
    print("\nAll quarterly frequency tests passed!")
