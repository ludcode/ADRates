"""
Debug script to understand XCCY swap leg valuations
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

value_dt = Date(15, 6, 2023)

# Build GBP curve
gbp_swap = OIS(value_dt, "1Y", SwapTypes.PAY, 0.0450, FrequencyTypes.ANNUAL,
               DayCountTypes.ACT_365F, CurveTypes.GBP_OIS_SONIA, CurrencyTypes.GBP)
gbp_curve = OISCurve(value_dt, [gbp_swap], InterpTypes.FLAT_FWD_RATES, check_refit=True)

# Build USD curve
usd_swap = OIS(value_dt, "1Y", SwapTypes.PAY, 0.0520, FrequencyTypes.ANNUAL,
               DayCountTypes.ACT_360, CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD)
usd_curve = OISCurve(value_dt, [usd_swap], InterpTypes.FLAT_FWD_RATES, check_refit=True)

spot_fx = 0.79

# Create XCCY swap
xccy_swap = XccyBasisSwap(
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

print("\n" + "="*80)
print("DOMESTIC LEG (GBP):")
print("="*80)
print(f"Leg type: {xccy_swap._domestic_leg._leg_type}")
print(f"Notional: {xccy_swap._domestic_leg._notional:,.2f}")
print(f"Spread: {xccy_swap._domestic_leg._spread * 10000:.2f} bp")

# Value domestic leg
pv_domestic = xccy_swap._domestic_leg.value(value_dt, gbp_curve, gbp_curve)
print(f"\nDomestic leg PV: {pv_domestic:,.2f}")
print(f"Payment PVs: {xccy_swap._domestic_leg._payment_pvs}")
print(f"Cumulative PVs: {xccy_swap._domestic_leg._cumulative_pvs}")

print("\n" + "="*80)
print("FOREIGN LEG (USD):")
print("="*80)
print(f"Leg type: {xccy_swap._foreign_leg._leg_type}")
print(f"Notional: {xccy_swap._foreign_leg._notional:,.2f}")
print(f"Spread: {xccy_swap._foreign_leg._spread * 10000:.2f} bp")

# Value foreign leg with USD curve for both projection and discounting
pv_foreign = xccy_swap._foreign_leg.value(value_dt, usd_curve, usd_curve)
print(f"\nForeign leg PV (in USD, discounted with USD curve): {pv_foreign:,.2f}")
print(f"Payment PVs: {xccy_swap._foreign_leg._payment_pvs}")
print(f"Cumulative PVs: {xccy_swap._foreign_leg._cumulative_pvs}")

print("\n" + "="*80)
print("TOTAL SWAP PV (using USD curve for foreign discounting):")
print("="*80)
pv_total_wrong = pv_domestic + spot_fx * pv_foreign
print(f"PV_domestic + FX * PV_foreign = {pv_domestic:,.2f} + {spot_fx} * {pv_foreign:,.2f}")
print(f"Total PV = {pv_total_wrong:,.2f}")
print("\nThis should NOT be zero because foreign leg should use XCCY curve for discounting")
