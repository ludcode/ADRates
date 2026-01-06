"""Debug script to check if payment dates match pillar dates."""

from cavour.utils.date import Date
from cavour.utils.day_count import DayCountTypes
from cavour.utils.frequency import FrequencyTypes
from cavour.utils.calendar import BusDayAdjustTypes
from cavour.utils.global_types import SwapTypes, CurveTypes
from cavour.utils.currency import CurrencyTypes
from cavour.market.curves.interpolator import InterpTypes
from cavour.models.models import Model
from cavour.trades.rates.xccy_basis_swap import XccyBasisSwap

# Valuation date
value_dt = Date(15, 6, 2023)

# Create a 12Y basis swap
swap_12y = XccyBasisSwap(
    effective_dt=value_dt,
    term_dt_or_tenor='12Y',
    domestic_notional=0.79 * 1_000_000,
    foreign_notional=1_000_000,
    domestic_spread=0.0,
    foreign_spread=0.0040,
    domestic_freq_type=FrequencyTypes.ANNUAL,
    foreign_freq_type=FrequencyTypes.ANNUAL,
    domestic_dc_type=DayCountTypes.ACT_365F,
    foreign_dc_type=DayCountTypes.ACT_360,
    domestic_floating_index=CurveTypes.GBP_OIS_SONIA,
    foreign_floating_index=CurveTypes.USD_OIS_SOFR,
    domestic_currency=CurrencyTypes.GBP,
    foreign_currency=CurrencyTypes.USD
)

print(f"\n{'='*80}")
print(f"12Y SWAP PAYMENT DATES ANALYSIS:")
print(f"{'='*80}\n")

print(f"Effective date: {swap_12y._effective_dt}")
print(f"Maturity date: {swap_12y._maturity_dt}")
print(f"\nForeign leg payment dates:")

for i, dt in enumerate(swap_12y._foreign_leg._payment_dts):
    # Calculate time from value_dt using ACT/365F
    t_365 = (dt - value_dt) / 365.0
    # Calculate time from value_dt using ACT/360
    t_360 = (dt - value_dt) / 360.0
    # Calculate days
    days = dt - value_dt

    print(f"  Payment {i+1}: {dt} (days={days}, t_365={t_365:.10f}, t_360={t_360:.10f})")

print(f"\n{'='*80}\n")
