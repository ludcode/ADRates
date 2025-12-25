"""
Diagnostic script to analyze DELTA relative errors across all tenors.

This will help determine if the 1% tolerance in test_delta_parallel_shift_validation
is appropriate by showing actual errors for each tenor.
"""

import numpy as np
from cavour.utils.date import Date
from cavour.utils.global_types import SwapTypes, RequestTypes, CurveTypes
from cavour.utils.day_count import DayCountTypes
from cavour.utils.frequency import FrequencyTypes
from cavour.utils.calendar import BusDayAdjustTypes
from cavour.utils.currency import CurrencyTypes
from cavour.trades.rates.ois import OIS
from cavour.models.models import Model


def compute_finite_difference_delta(swap, model, value_dt, bump_bp=1.0, curve_name="GBP_OIS_SONIA"):
    """Compute finite difference DELTA."""
    shock_pct = bump_bp * 0.01

    model_up = model.scenario(curve_name, shock=shock_pct)
    model_down = model.scenario(curve_name, shock=-shock_pct)

    pos_up = swap.position(model_up)
    pos_down = swap.position(model_down)

    value_up = pos_up.compute([RequestTypes.VALUE]).value.amount
    value_down = pos_down.compute([RequestTypes.VALUE]).value.amount

    delta_fd = (value_up - value_down) / (2.0 * bump_bp)
    return delta_fd


def main():
    # Setup market data
    value_dt = Date(17, 12, 2024)

    px_list = [
        5.1998, 5.2014, 5.2003, 5.2027, 5.2023, 5.19281,
        5.1656, 5.1482, 5.1342, 5.1173, 5.1013, 5.0862,
        5.0701, 5.054, 5.0394, 4.8707, 4.75483, 4.532,
        4.3628, 4.2428, 4.16225, 4.1132, 4.08505, 4.0762,
        4.078, 4.0961, 4.12195, 4.1315, 4.113, 4.07724, 3.984, 3.88
    ]
    tenor_list = [
        "1D", "1W", "2W", "1M", "2M", "3M", "4M", "5M", "6M",
        "7M", "8M", "9M", "10M", "11M", "1Y", "18M", "2Y",
        "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y",
        "12Y", "15Y", "20Y", "25Y", "30Y", "40Y", "50Y"
    ]

    # Build model
    gbp_model = Model(value_dt)
    gbp_model.build_curve(
        name="GBP_OIS_SONIA",
        px_list=px_list,
        tenor_list=tenor_list,
        spot_days=0,
        swap_type=SwapTypes.PAY,
        fixed_dcc_type=DayCountTypes.ACT_365F,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_365F,
        bus_day_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
    )

    settle_dt = value_dt.add_tenor("0D")

    print("=" * 100)
    print("DELTA RELATIVE ERROR ANALYSIS")
    print("=" * 100)
    print()

    # Test different bump sizes
    bump_sizes = [1.0, 10.0]

    for bump_bp in bump_sizes:
        print(f"\nBUMP SIZE: {bump_bp} bp")
        print("-" * 100)
        print(f"{'Tenor':<8} {'AD Delta':>15} {'FD Delta':>15} {'Abs Error':>15} {'Rel Error':>12}")
        print("-" * 100)

        for tenor in tenor_list:
            try:
                # Create swap
                swap = OIS(
                    effective_dt=settle_dt,
                    term_dt_or_tenor=tenor,
                    fixed_leg_type=SwapTypes.PAY,
                    fixed_coupon=0.045,
                    fixed_freq_type=FrequencyTypes.ANNUAL,
                    fixed_dc_type=DayCountTypes.ACT_365F,
                    floating_index=CurveTypes.GBP_OIS_SONIA,
                    currency=CurrencyTypes.GBP,
                    bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
                    float_freq_type=FrequencyTypes.ANNUAL,
                    float_dc_type=DayCountTypes.ACT_365F,
                )

                # Compute AD DELTA
                pos = swap.position(gbp_model)
                result = pos.compute([RequestTypes.DELTA])
                delta_ad = result.risk.value.amount

                # Compute FD DELTA
                delta_fd = compute_finite_difference_delta(
                    swap, gbp_model, value_dt, bump_bp=bump_bp
                )

                # Calculate errors
                abs_error = abs(delta_ad - delta_fd)
                if abs(delta_fd) > 1e-10:
                    rel_error = abs_error / abs(delta_fd)
                    rel_error_pct = rel_error * 100
                else:
                    rel_error_pct = 0.0

                print(f"{tenor:<8} {delta_ad:>15.6f} {delta_fd:>15.6f} {abs_error:>15.6f} {rel_error_pct:>11.4f}%")

            except Exception as e:
                print(f"{tenor:<8} ERROR: {str(e)}")

        print("-" * 100)

    print("\n" + "=" * 100)
    print("ANALYSIS SUMMARY")
    print("=" * 100)
    print("\nKey questions:")
    print("1. Are relative errors consistently below 1%?")
    print("2. Do errors vary significantly by tenor or bump size?")
    print("3. Are there any outliers that suggest numerical issues?")
    print()


if __name__ == "__main__":
    main()
