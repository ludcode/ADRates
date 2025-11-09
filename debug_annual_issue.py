"""
Debug script to see what error occurs with annual frequency + DELTA/GAMMA
"""

from cavour.utils.date import Date
from cavour.utils.global_types import SwapTypes, RequestTypes
from cavour.utils.day_count import DayCountTypes
from cavour.utils.frequency import FrequencyTypes
from cavour.utils.calendar import BusDayAdjustTypes
from cavour.market.curves.interpolator import InterpTypes
from cavour.models.models import Model
from cavour.trades.rates.ois import OIS

# Test parameters
value_dt = Date(30, 4, 2024)

# Just test first few swaps
px_list = [5.1998, 5.2014, 5.2003, 5.2027]
tenor_list = ["1D", "1W", "2W", "1M"]

# Build model with ANNUAL frequency
model = Model(value_dt)
model.build_curve(
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
    interp_type=InterpTypes.LINEAR_ZERO_RATES,
)

# Try to value 1D swap with VALUE only
print("Testing 1D swap with VALUE only:")
swap = OIS(
    effective_dt=value_dt,
    term_dt_or_tenor="1D",
    fixed_leg_type=SwapTypes.PAY,
    fixed_coupon=5.1998 / 100,
    fixed_freq_type=FrequencyTypes.ANNUAL,
    fixed_dc_type=DayCountTypes.ACT_365F,
    bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
    float_freq_type=FrequencyTypes.ANNUAL,
    float_dc_type=DayCountTypes.ACT_365F
)

try:
    pos = swap.position(model)
    res = pos.compute([RequestTypes.VALUE])
    print(f"  VALUE: {res.value.amount}")
    print("  SUCCESS with VALUE only")
except Exception as e:
    print(f"  FAILED with VALUE: {type(e).__name__}: {e}")

# Try with DELTA and GAMMA
print("\nTesting 1D swap with VALUE, DELTA, GAMMA:")
try:
    pos = swap.position(model)
    res = pos.compute([RequestTypes.VALUE, RequestTypes.DELTA, RequestTypes.GAMMA])
    print(f"  VALUE: {res.value.amount}")
    print("  SUCCESS with VALUE, DELTA, GAMMA")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Try with 1W swap
print("\nTesting 1W swap with VALUE, DELTA, GAMMA:")
swap1w = OIS(
    effective_dt=value_dt,
    term_dt_or_tenor="1W",
    fixed_leg_type=SwapTypes.PAY,
    fixed_coupon=5.2014 / 100,
    fixed_freq_type=FrequencyTypes.ANNUAL,
    fixed_dc_type=DayCountTypes.ACT_365F,
    bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
    float_freq_type=FrequencyTypes.ANNUAL,
    float_dc_type=DayCountTypes.ACT_365F
)

try:
    pos = swap1w.position(model)
    res = pos.compute([RequestTypes.VALUE, RequestTypes.DELTA, RequestTypes.GAMMA])
    print(f"  VALUE: {res.value.amount}")
    print("  SUCCESS with VALUE, DELTA, GAMMA")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
