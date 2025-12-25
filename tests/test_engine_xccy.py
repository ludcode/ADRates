"""
Test Engine.compute() for cross-currency swaps - VALUE only.

Validates that engine.py produces the same results as direct valuation.
"""

from cavour.utils.date import Date
from cavour.utils.day_count import DayCountTypes
from cavour.utils.frequency import FrequencyTypes
from cavour.utils.global_types import CurveTypes, SwapTypes, RequestTypes
from cavour.utils.currency import CurrencyTypes
from cavour.market.curves.interpolator import InterpTypes
from cavour.utils.calendar import BusDayAdjustTypes

from cavour.trades.rates.xccy_fix_float_swap import XccyFixFloat
from cavour.models.models import Model
from cavour.market.position.engine import Engine


def test_engine_xccy_value_simple():
    """Test engine VALUE for a simple XCCY fixed-float swap with notional exchanges."""

    value_dt = Date(15, 6, 2023)

    # Simple curve structure for testing
    tenors = ['1Y', '2Y', '3Y', '4Y', '5Y']
    gbp_rates = [4.50, 4.55, 4.60, 4.65, 4.70]
    usd_rates = [5.20, 5.25, 5.30, 5.35, 5.40]

    # Build model with both curves
    model = Model(value_dt)

    # Build GBP curve
    model.build_curve(
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

    # Build USD curve
    model.build_curve(
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

    # Add FX rate (USDGBP = GBP per USD)
    spot_fx = 0.79  # GBP per USD
    model.build_fx(["USDGBP"], [spot_fx])

    # Create a simple 3Y XCCY fixed-float swap
    tenor = "3Y"
    domestic_notional = 790_000  # GBP
    foreign_notional = 1_000_000  # USD
    domestic_coupon = 0.046  # 4.6% fixed GBP
    foreign_spread = 0.0030  # 30bp spread on USD floating

    xccy_swap = XccyFixFloat(
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

    # Value using direct method (Note: For now, we don't have XCCY curve, so pass USD curve)
    gbp_curve = model.curves.GBP_OIS_SONIA
    usd_curve = model.curves.USD_OIS_SOFR

    # Value legs separately for comparison
    domestic_leg_pv_direct = xccy_swap._domestic_leg.value(value_dt, gbp_curve)
    foreign_leg_pv_direct = xccy_swap._foreign_leg.value(value_dt, usd_curve, usd_curve)

    # Direct valuation (using USD curve for XCCY discounting - simplified for now)
    pv_direct = xccy_swap.value(
        value_dt=value_dt,
        domestic_discount_curve=gbp_curve,
        foreign_discount_curve=usd_curve,
        xccy_discount_curve=usd_curve,  # Simplified: use USD curve instead of XCCY
        spot_fx=spot_fx
    )

    print(f"\nDirect valuation leg PVs:")
    print(f"  Domestic (fixed): {domestic_leg_pv_direct:,.2f} GBP")
    print(f"  Foreign (floating): {foreign_leg_pv_direct:,.2f} USD")

    # Value using Engine
    engine = Engine(model)
    result = engine.compute(xccy_swap, [RequestTypes.VALUE])

    pv_engine = result.value.amount

    # Compare
    print("\n" + "="*80)
    print("ENGINE XCCY VALUE TEST:")
    print("="*80)
    print(f"Direct valuation:  {pv_direct:>15,.2f} GBP")
    print(f"Engine valuation:  {pv_engine:>15,.2f} GBP")
    print(f"Difference:        {pv_direct - pv_engine:>15,.2f} GBP")
    print(f"Relative error:    {abs(pv_direct - pv_engine) / abs(domestic_notional) * 100:>15,.6f}%")
    print("="*80)

    # Assert they match within numerical precision
    # The engine and direct methods compute identical valuations, so they should match exactly
    # Allow only for floating-point epsilon differences
    rel_error = abs(pv_direct - pv_engine) / abs(domestic_notional)
    epsilon = 0.0001  # 0.0001% = 1e-6 relative error
    assert rel_error < epsilon / 100, f"Engine mismatch > {epsilon}%: {pv_direct} vs {pv_engine} (rel error: {rel_error*100:.6f}%)"

    print("\nEngine XCCY VALUE test passed!")


def test_engine_xccy_delta_finite_diff():
    """Test engine DELTA for XCCY swap using finite difference validation.

    Compares JAX automatic differentiation against manual bumping of curves +/- 1bp.
    """
    value_dt = Date(15, 6, 2023)

    # Curve structure
    tenors = ['1Y', '2Y', '3Y', '4Y', '5Y']
    gbp_rates_base = [4.50, 4.55, 4.60, 4.65, 4.70]
    usd_rates_base = [5.20, 5.25, 5.30, 5.35, 5.40]
    spot_fx = 0.79  # GBP per USD

    # Helper function to build model AND create fresh swap
    def build_model_and_swap(gbp_rates, usd_rates):
        model = Model(value_dt)

        model.build_curve(
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

        model.build_curve(
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

        model.build_fx(["USDGBP"], [spot_fx])

        # Create fresh swap for this model
        swap = XccyFixFloat(
            effective_dt=value_dt,
            term_dt_or_tenor="3Y",
            domestic_notional=790_000,
            foreign_notional=1_000_000,
            domestic_leg_type=SwapTypes.PAY,
            domestic_coupon=0.046,
            foreign_spread=0.0030,
            domestic_freq_type=FrequencyTypes.ANNUAL,
            foreign_freq_type=FrequencyTypes.QUARTERLY,
            domestic_dc_type=DayCountTypes.ACT_365F,
            foreign_dc_type=DayCountTypes.ACT_360,
            domestic_floating_index=CurveTypes.GBP_OIS_SONIA,
            foreign_floating_index=CurveTypes.USD_OIS_SOFR,
            domestic_currency=CurrencyTypes.GBP,
            foreign_currency=CurrencyTypes.USD
        )

        # Initialize leg state
        gbp_curve = model.curves.GBP_OIS_SONIA
        usd_curve = model.curves.USD_OIS_SOFR
        _ = swap._domestic_leg.value(value_dt, gbp_curve)
        _ = swap._foreign_leg.value(value_dt, usd_curve, usd_curve)

        return model, swap

    # Build base model and swap
    model_base, xccy_swap_base = build_model_and_swap(gbp_rates_base, usd_rates_base)

    # Compute DELTA using engine
    engine_base = Engine(model_base)
    result_base = engine_base.compute(xccy_swap_base, [RequestTypes.VALUE, RequestTypes.DELTA])

    pv_base = result_base.value.amount
    delta_gbp = result_base.risk.GBP_OIS_SONIA
    delta_usd = result_base.risk.USD_OIS_SOFR

    print("\n" + "="*80)
    print("ENGINE XCCY DELTA TEST (Finite Difference Validation)")
    print("="*80)
    print(f"\nBase PV: {pv_base:,.2f} GBP")
    print(f"\nEngine DELTAs (JAX auto-diff):")
    print(f"  GBP curve: {delta_gbp.risk_ladder}")
    print(f"  USD curve: {delta_usd.risk_ladder}")

    # Validate GBP DELTA using finite difference
    print(f"\nValidating GBP DELTA (bumping GBP curve +/- 1bp):")
    delta_gbp_fd = []
    bump_size = 0.01  # 1bp = 0.01% = 0.01 in percentage point units

    for i, tenor in enumerate(tenors):
        # Bump UP
        gbp_rates_up = gbp_rates_base.copy()
        gbp_rates_up[i] += bump_size
        model_up, swap_up = build_model_and_swap(gbp_rates_up, usd_rates_base)
        engine_up = Engine(model_up)
        result_up = engine_up.compute(swap_up, [RequestTypes.VALUE])
        pv_up = result_up.value.amount

        # Bump DOWN
        gbp_rates_down = gbp_rates_base.copy()
        gbp_rates_down[i] -= bump_size
        model_down, swap_down = build_model_and_swap(gbp_rates_down, usd_rates_base)
        engine_down = Engine(model_down)
        result_down = engine_down.compute(swap_down, [RequestTypes.VALUE])
        pv_down = result_down.value.amount

        # Finite difference: (PV_up - PV_down) / 2
        # Note: Engine DELTA is already scaled for 1bp shift (multiplied by 1e-4)
        # So we just need the change in PV divided by 2
        fd = (pv_up - pv_down) / 2
        delta_gbp_fd.append(fd)

        # Compare to engine DELTA
        engine_delta = delta_gbp.risk_ladder[i]
        diff = abs(engine_delta - fd)
        rel_error = diff / max(abs(fd), 1e-6) * 100 if fd != 0 else 0

        print(f"  {tenor}: Engine={engine_delta:>10.4f}, FD={fd:>10.4f}, Diff={diff:>8.4f}, RelErr={rel_error:>6.2f}%")

    # Validate USD DELTA using finite difference
    print(f"\nValidating USD DELTA (bumping USD curve +/- 1bp):")
    delta_usd_fd = []

    for i, tenor in enumerate(tenors):
        # Bump UP
        usd_rates_up = usd_rates_base.copy()
        usd_rates_up[i] += bump_size
        model_up, swap_up = build_model_and_swap(gbp_rates_base, usd_rates_up)
        engine_up = Engine(model_up)
        result_up = engine_up.compute(swap_up, [RequestTypes.VALUE])
        pv_up = result_up.value.amount

        # Bump DOWN
        usd_rates_down = usd_rates_base.copy()
        usd_rates_down[i] -= bump_size
        model_down, swap_down = build_model_and_swap(gbp_rates_base, usd_rates_down)
        engine_down = Engine(model_down)
        result_down = engine_down.compute(swap_down, [RequestTypes.VALUE])
        pv_down = result_down.value.amount

        # Finite difference: (PV_up - PV_down) / 2
        # Note: Engine DELTA is already scaled for 1bp shift (multiplied by 1e-4)
        fd = (pv_up - pv_down) / 2
        delta_usd_fd.append(fd)

        # Compare to engine DELTA
        engine_delta = delta_usd.risk_ladder[i]
        diff = abs(engine_delta - fd)
        rel_error = diff / max(abs(fd), 1e-6) * 100 if fd != 0 else 0

        print(f"  {tenor}: Engine={engine_delta:>10.4f}, FD={fd:>10.4f}, Diff={diff:>8.4f}, RelErr={rel_error:>6.2f}%")

    # Assert all DELTAs match within tolerance
    print(f"\nAsserting DELTAs match within tolerance...")
    tolerance = 1.0  # 1.0 GBP absolute tolerance per pillar

    for i, tenor in enumerate(tenors):
        # GBP curve
        engine_delta = delta_gbp.risk_ladder[i]
        fd_delta = delta_gbp_fd[i]
        diff = abs(engine_delta - fd_delta)
        assert diff < tolerance, f"GBP {tenor} DELTA mismatch: {engine_delta} vs {fd_delta} (diff={diff})"

        # USD curve
        engine_delta = delta_usd.risk_ladder[i]
        fd_delta = delta_usd_fd[i]
        diff = abs(engine_delta - fd_delta)
        assert diff < tolerance, f"USD {tenor} DELTA mismatch: {engine_delta} vs {fd_delta} (diff={diff})"

    print("="*80)
    print("All DELTAs match finite difference within tolerance!")
    print("Engine XCCY DELTA test passed!")
    print("="*80)


if __name__ == "__main__":
    test_engine_xccy_value_simple()
    test_engine_xccy_delta_finite_diff()
