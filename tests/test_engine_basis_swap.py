"""
Test Engine for XccyBasisSwap (Float-Float Cross-Currency Basis Swaps)

Tests VALUE and DELTA computations using JAX automatic differentiation
and validates against direct valuation and finite difference methods.
"""

import pytest
from cavour.utils.date import Date
from cavour.utils.frequency import FrequencyTypes
from cavour.utils.day_count import DayCountTypes
from cavour.utils.calendar import CalendarTypes, BusDayAdjustTypes
from cavour.utils.global_types import CurveTypes, SwapTypes, RequestTypes
from cavour.utils.currency import CurrencyTypes
from cavour.market.curves.interpolator import InterpTypes
from cavour.models.models import Model
from cavour.market.position.engine import Engine
from cavour.trades.rates.xccy_basis_swap import XccyBasisSwap


print("\n=== Starting Cavour Test Session ===\n")


def test_engine_basis_swap_value():
    """Test engine VALUE for XccyBasisSwap against direct valuation.

    Creates a GBP/USD basis swap with:
    - Domestic (GBP) floating leg: receive SONIA + 0bp
    - Foreign (USD) floating leg: pay SOFR + 25bp
    - Notional exchanges at start and maturity

    Validates engine valuation matches direct swap valuation to machine precision.
    """
    value_dt = Date(15, 6, 2023)

    # Build GBP and USD curves
    tenors = ['1Y', '2Y', '3Y', '4Y', '5Y']
    gbp_rates = [4.50, 4.55, 4.60, 4.65, 4.70]
    usd_rates = [5.20, 5.25, 5.30, 5.35, 5.40]
    spot_fx = 0.79  # GBP per USD

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

    # Add FX rate
    model.build_fx(["USDGBP"], [spot_fx])

    # Get curves for direct valuation
    gbp_curve = model.curves.GBP_OIS_SONIA
    usd_curve = model.curves.USD_OIS_SOFR

    # Create basis swap: receive GBP SONIA, pay USD SOFR + 25bp
    basis_swap = XccyBasisSwap(
        effective_dt=value_dt,
        term_dt_or_tenor="3Y",
        domestic_notional=790_000,  # GBP notional
        foreign_notional=1_000_000,  # USD notional
        domestic_spread=0.0,  # No spread on GBP leg
        foreign_spread=0.0025,  # 25bp spread on USD leg
        domestic_freq_type=FrequencyTypes.ANNUAL,
        foreign_freq_type=FrequencyTypes.QUARTERLY,
        domestic_dc_type=DayCountTypes.ACT_365F,
        foreign_dc_type=DayCountTypes.ACT_360,
        domestic_floating_index=CurveTypes.GBP_OIS_SONIA,
        foreign_floating_index=CurveTypes.USD_OIS_SOFR,
        domestic_currency=CurrencyTypes.GBP,
        foreign_currency=CurrencyTypes.USD,
    )

    # Direct valuation using swap's value() method
    # Note: For now, using USD curve for XCCY discounting (no separate XCCY curve yet)
    direct_value = basis_swap.value(
        value_dt=value_dt,
        domestic_discount_curve=gbp_curve,
        foreign_discount_curve=usd_curve,
        xccy_discount_curve=usd_curve,  # Using USD curve as XCCY curve for now
        spot_fx=spot_fx
    )

    # Get leg PVs for diagnostics
    domestic_leg_value = basis_swap._domestic_leg.value(
        value_dt=value_dt,
        discount_curve=gbp_curve,
        index_curve=gbp_curve
    )
    foreign_leg_value = basis_swap._foreign_leg.value(
        value_dt=value_dt,
        discount_curve=usd_curve,  # Using USD curve as XCCY curve
        index_curve=usd_curve
    )

    print(f"Direct valuation leg PVs:")
    print(f"  Domestic (floating): {domestic_leg_value:,.2f} GBP")
    print(f"  Foreign (floating): {foreign_leg_value:,.2f} USD")
    print()

    # Engine valuation
    engine = Engine(model)
    result = engine.compute(basis_swap, [RequestTypes.VALUE])
    engine_value = result.value.amount

    # Compare values
    diff = abs(engine_value - direct_value)
    rel_error = abs(diff / direct_value) * 100 if direct_value != 0 else 0

    print("=" * 80)
    print("ENGINE BASIS SWAP VALUE TEST:")
    print("=" * 80)
    print(f"Direct valuation:         {direct_value:,.2f} GBP")
    print(f"Engine valuation:         {engine_value:,.2f} GBP")
    print(f"Difference:               {diff:>10.2f} GBP")
    print(f"Relative error:           {rel_error:.6f}%")
    print("=" * 80)
    print()

    # Assert match to machine precision
    tolerance = 0.01  # 1 penny tolerance
    assert diff < tolerance, f"VALUE mismatch: {diff:.6f} GBP (expected < {tolerance})"

    print("Engine BASIS SWAP VALUE test passed!")


def test_engine_basis_swap_delta_finite_diff():
    """Test engine DELTA for XccyBasisSwap using finite difference validation.

    Compares JAX automatic differentiation DELTA against manual bumping of curves +/- 1bp.
    Validates sensitivities to both GBP and USD curves.
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

        # Add FX rate
        model.build_fx(["USDGBP"], [spot_fx])

        # Create fresh swap for this model
        swap = XccyBasisSwap(
            effective_dt=value_dt,
            term_dt_or_tenor="3Y",
            domestic_notional=790_000,
            foreign_notional=1_000_000,
            domestic_spread=0.0,
            foreign_spread=0.0025,  # 25bp
            domestic_freq_type=FrequencyTypes.ANNUAL,
            foreign_freq_type=FrequencyTypes.QUARTERLY,
            domestic_dc_type=DayCountTypes.ACT_365F,
            foreign_dc_type=DayCountTypes.ACT_360,
            domestic_floating_index=CurveTypes.GBP_OIS_SONIA,
            foreign_floating_index=CurveTypes.USD_OIS_SOFR,
            domestic_currency=CurrencyTypes.GBP,
            foreign_currency=CurrencyTypes.USD,
        )

        # Initialize leg state
        gbp_curve = model.curves.GBP_OIS_SONIA
        usd_curve = model.curves.USD_OIS_SOFR
        _ = swap._domestic_leg.value(value_dt, gbp_curve, gbp_curve)
        _ = swap._foreign_leg.value(value_dt, usd_curve, usd_curve)

        return model, swap

    # Build base model and swap
    model_base, basis_swap_base = build_model_and_swap(gbp_rates_base, usd_rates_base)

    # Compute DELTA using engine
    engine_base = Engine(model_base)
    result_base = engine_base.compute(basis_swap_base, [RequestTypes.VALUE, RequestTypes.DELTA])

    pv_base = result_base.value.amount
    delta_gbp = result_base.risk.GBP_OIS_SONIA
    delta_usd = result_base.risk.USD_OIS_SOFR

    print("=" * 80)
    print("ENGINE BASIS SWAP DELTA TEST (Finite Difference Validation)")
    print("=" * 80)
    print()
    print(f"Base PV: {pv_base:,.2f} GBP")
    print()
    print("Engine DELTAs (JAX auto-diff):")
    print(f"  GBP curve: {delta_gbp.risk_ladder}")
    print(f"  USD curve: {delta_usd.risk_ladder}")
    print()

    # Validate GBP DELTA using finite difference
    delta_gbp_fd = []
    bump_size = 0.01  # 1bp = 0.01% = 0.01 in percentage point units

    print("Validating GBP DELTA (bumping GBP curve +/- 1bp):")
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
        fd = (pv_up - pv_down) / 2
        delta_gbp_fd.append(fd)

        engine_delta = delta_gbp.risk_ladder[i]
        diff = abs(engine_delta - fd)
        rel_err = abs(diff / fd) * 100 if fd != 0 else 0

        print(f"  {tenor}: Engine={engine_delta:>10.4f}, FD={fd:>10.4f}, Diff={diff:>8.4f}, RelErr={rel_err:>6.2f}%")

    print()

    # Validate USD DELTA using finite difference
    delta_usd_fd = []

    print("Validating USD DELTA (bumping USD curve +/- 1bp):")
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
        fd = (pv_up - pv_down) / 2
        delta_usd_fd.append(fd)

        engine_delta = delta_usd.risk_ladder[i]
        diff = abs(engine_delta - fd)
        rel_err = abs(diff / fd) * 100 if fd != 0 else 0

        print(f"  {tenor}: Engine={engine_delta:>10.4f}, FD={fd:>10.4f}, Diff={diff:>8.4f}, RelErr={rel_err:>6.2f}%")

    print()
    print("Asserting DELTAs match within tolerance...")

    # Assert all DELTAs match within tolerance
    tolerance = 1.0  # 1.0 GBP absolute tolerance per pillar

    for i, tenor in enumerate(tenors):
        engine_delta = delta_gbp.risk_ladder[i]
        fd_delta = delta_gbp_fd[i]
        diff = abs(engine_delta - fd_delta)
        assert diff < tolerance, f"GBP {tenor} DELTA mismatch: {engine_delta} vs {fd_delta} (diff={diff})"

    for i, tenor in enumerate(tenors):
        engine_delta = delta_usd.risk_ladder[i]
        fd_delta = delta_usd_fd[i]
        diff = abs(engine_delta - fd_delta)
        assert diff < tolerance, f"USD {tenor} DELTA mismatch: {engine_delta} vs {fd_delta} (diff={diff})"

    print("=" * 80)
    print("All DELTAs match finite difference within tolerance!")
    print("Engine BASIS SWAP DELTA test passed!")
    print("=" * 80)


print("=== Ending Cavour Test Session ===\n")
