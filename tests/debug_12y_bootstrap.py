"""Debug script to trace the 12Y pillar bootstrap in detail."""

import numpy as np
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

# Valuation date
value_dt = Date(15, 6, 2023)

# Tenors through 12Y
tenors = ['1Y', '2Y', '3Y', '4Y', '5Y', '6Y', '7Y', '8Y', '9Y', '10Y', '12Y']
gbp_rates = [4.50, 4.55, 4.60, 4.65, 4.70, 4.72, 4.74, 4.76, 4.78, 4.80, 4.82]
usd_rates = [5.20, 5.25, 5.30, 5.35, 5.40, 5.42, 5.44, 5.46, 5.48, 5.50, 5.52]
basis_spreads = [0.0025, 0.0028, 0.0030, 0.0032, 0.0034, 0.0035, 0.0036, 0.0037, 0.0038, 0.0039, 0.0040]

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

# Build XCCY curve
print(f"\n{'='*80}")
print(f"Building XCCY curve...")
print(f"{'='*80}\n")

xccy_curve = XccyCurve(
    value_dt=value_dt,
    basis_swaps=basis_swaps,
    domestic_curve=gbp_curve,
    foreign_curve=usd_curve,
    spot_fx=spot_fx,
    interp_type=InterpTypes.FLAT_FWD_RATES,
    check_refit=False
)

print("\nXCCY Curve Times and DFs:")
print(f"{'Time':<15} {'DF':<20}")
print(f"{'-'*35}")
for t, df in zip(xccy_curve._times, xccy_curve._dfs):
    print(f"{t:<15.10f} {df:<20.15f}")

# Now reprice the 12Y swap
swap_12y = basis_swaps[10]  # 12Y swap

print(f"\n{'='*80}")
print(f"Repricing 12Y Swap:")
print(f"{'='*80}\n")

pv = swap_12y.value(
    value_dt=value_dt,
    domestic_discount_curve=gbp_curve,
    foreign_discount_curve=usd_curve,
    xccy_discount_curve=xccy_curve,
    spot_fx=spot_fx
)

print(f"PV: {pv}")
print(f"Normalized PV: {pv / swap_12y._domestic_notional}")

# Check DFs at each payment date
print(f"\n12Y Swap Foreign Leg Payment DFs:")
print(f"{'Payment Date':<20} {'Time (ACT/365F)':<20} {'DF from XCCY':<20}")
print(f"{'-'*60}")
for pmnt_dt in swap_12y._foreign_leg._payment_dts:
    t = (pmnt_dt - value_dt) / 365.0
    df = xccy_curve.df(pmnt_dt)
    print(f"{str(pmnt_dt):<20} {t:<20.10f} {df:<20.15f}")

print(f"\n{'='*80}\n")
