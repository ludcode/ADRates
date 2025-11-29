"""Debug script to analyze XCCY curve repricing errors."""

from cavour.utils.date import Date
from cavour.utils.day_count import DayCountTypes
from cavour.utils.frequency import FrequencyTypes
from cavour.utils.calendar import BusDayAdjustTypes
from cavour.utils.global_types import SwapTypes, CurveTypes
from cavour.utils.currency import CurrencyTypes
from cavour.market.curves.interpolator import InterpTypes
from cavour.models.models import Model
from cavour.trades.rates.xccy_basis_swap import XccyBasisSwap
from cavour.trades.rates.xccy_curve import XccyCurve

def test_xccy_repricing():
    """Test XCCY curve repricing with full diagnostic output."""

    # Valuation date
    value_dt = Date(15, 6, 2023)

    # Full tenor structure
    tenors = ['1Y', '2Y', '3Y', '4Y', '5Y', '6Y', '7Y', '8Y', '9Y', '10Y', '12Y', '15Y', '20Y']

    # Market rates
    gbp_rates = [4.50, 4.55, 4.60, 4.65, 4.70, 4.72, 4.74, 4.76, 4.78, 4.80, 4.82, 4.85, 4.90]
    usd_rates = [5.20, 5.25, 5.30, 5.35, 5.40, 5.42, 5.44, 5.46, 5.48, 5.50, 5.52, 5.55, 5.60]
    basis_spreads = [0.0025, 0.0028, 0.0030, 0.0032, 0.0034, 0.0035, 0.0036, 0.0037, 0.0038, 0.0039, 0.0040, 0.0042, 0.0045]

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

    # Build basis swaps
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

    # Build XCCY curve WITHOUT refit check
    print(f"\n{'='*80}")
    print(f"Building XCCY curve with {len(tenors)} basis swaps...")
    print(f"{'='*80}\n")

    xccy_curve = XccyCurve(
        value_dt=value_dt,
        basis_swaps=basis_swaps,
        domestic_curve=gbp_curve,
        foreign_curve=usd_curve,
        spot_fx=spot_fx,
        interp_type=InterpTypes.FLAT_FWD_RATES,
        check_refit=False  # Don't check during construction
    )

    print("XCCY curve built successfully!\n")

    # Now manually check repricing for ALL swaps
    print(f"{'='*80}")
    print("REPRICING ANALYSIS:")
    print(f"{'='*80}\n")
    print(f"{'Tenor':<8} {'Maturity':<15} {'PV':<20} {'Normalized PV':<20}")
    print(f"{'-'*80}")

    for i, swap in enumerate(basis_swaps):
        tenor = tenors[i]
        maturity = swap._maturity_dt

        # Reprice swap
        pv = swap.value(
            value_dt=value_dt,
            domestic_discount_curve=gbp_curve,
            foreign_discount_curve=usd_curve,
            xccy_discount_curve=xccy_curve,
            spot_fx=spot_fx
        )

        pv_normalized = pv / swap._domestic_notional

        print(f"{tenor:<8} {str(maturity):<15} {pv:<20.10f} {pv_normalized:<20.10e}")

    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    test_xccy_repricing()
