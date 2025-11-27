"""
Simple test for XCCY curve - testing just construction without OIS curve issues.
"""

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


def test_xccy_with_ois_curves():
    """Test XCCY curve construction with OIS curves (using 1Y swaps only to avoid OIS bug)."""

    value_dt = Date(15, 6, 2023)

    # Build domestic (GBP SONIA) OIS curve - 1Y only
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

    # Build foreign (USD SOFR) OIS curve - 1Y only
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

    # Spot FX: GBP per USD
    spot_fx = 0.79

    # Create basis swaps - 1Y only
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

    print("\nTesting XCCY Curve Construction...")
    print("="*80)

    # Build XCCY curve
    xccy_curve = XccyCurve(
        value_dt=value_dt,
        basis_swaps=basis_swaps,
        domestic_curve=gbp_curve,
        foreign_curve=usd_curve,
        spot_fx=spot_fx,
        interp_type=InterpTypes.FLAT_FWD_RATES,
        check_refit=True  # Check that calibration swaps reprice to zero
    )

    print("\nXCCY Curve built successfully!")
    print(f"Number of nodes: {len(xccy_curve._times)}")
    print(f"Times: {xccy_curve._times}")
    print(f"Discount factors: {xccy_curve._dfs}")

    # Check basic properties
    assert xccy_curve is not None
    assert len(xccy_curve._times) == 2  # t=0 plus 1 swap maturity
    assert all(df > 0 for df in xccy_curve._dfs)

    # Check discount factors are decreasing
    for i in range(len(xccy_curve._dfs) - 1):
        assert xccy_curve._dfs[i] >= xccy_curve._dfs[i+1], \
            f"Discount factors not decreasing: {xccy_curve._dfs[i]} < {xccy_curve._dfs[i+1]}"

    # Query discount factor
    df_1y = xccy_curve.df(value_dt.add_years(1))
    print(f"\nDiscount factor at 1Y: {df_1y}")
    assert 0 < df_1y <= 1.0

    # Test that calibration swap reprices to zero
    test_swap = basis_swaps[0]
    pv = test_swap.value(value_dt, gbp_curve, usd_curve, xccy_curve, spot_fx)
    pv_normalized = pv / test_swap._domestic_notional

    print(f"\nCalibration swap PV: {pv}")
    print(f"Normalized PV: {pv_normalized}")
    assert abs(pv_normalized) < 1e-8, f"Calibration swap does not reprice to zero: {pv_normalized}"

    print("\n" + "="*80)
    print(xccy_curve)
    print("="*80)

    print("\nAll tests PASSED!")


if __name__ == "__main__":
    test_xccy_with_ois_curves()
