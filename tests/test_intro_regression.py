import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

#np = pytest.importorskip("numpy")
#jax = pytest.importorskip("jax")

from cavour.utils import date
from cavour.utils.global_types import SwapTypes
from cavour.utils.day_count import DayCountTypes
from cavour.utils.calendar import BusDayAdjustTypes
from cavour.utils.frequency import FrequencyTypes
from cavour.market.curves.interpolator import InterpTypes
from cavour.models.models import Model
from cavour.trades.rates.ois import OIS


def build_model():
    value_dt = date.Date(30, 4, 2024)
    px_list = [5.1998, 5.2014, 5.2003, 5.2027, 5.2023, 5.19281,
               5.1656, 5.1482, 5.1342, 5.1173, 5.1013, 5.0862,
               5.0701, 5.054, 5.0394, 4.8707, 4.75483, 4.532,
               4.3628, 4.2428, 4.16225, 4.1132, 4.08505, 4.0762,
               4.078, 4.0961, 4.12195, 4.1315, 4.113, 4.07724,
               3.984, 3.88]
    tenor_list = ["1D","1W","2W","1M","2M","3M","4M","5M","6M",
                  "7M","8M","9M","10M","11M","1Y","18M","2Y",
                  "3Y","4Y","5Y","6Y","7Y","8Y","9Y","10Y",
                  "12Y","15Y","20Y","25Y","30Y","40Y","50Y"]

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
    return model


def test_swap_value_and_pv01():
    model = build_model()
    value_dt = model.value_dt
    settle_dt = value_dt.add_weekdays(0)

    px = 0.04078
    tenor = "10Y"
    swap = OIS(
        effective_dt=settle_dt,
        term_dt_or_tenor=tenor,
        fixed_leg_type=SwapTypes.PAY,
        fixed_coupon=px,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        fixed_dc_type=DayCountTypes.ACT_365F,
        bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_365F,
    )

    par_rate = swap.swap_rate(value_dt, model.curves.GBP_OIS_SONIA) * 10000
    assert pytest.approx(4.078, rel=1e-3) == par_rate

    builtin_value = swap.value(value_dt, model.curves.GBP_OIS_SONIA)
    assert abs(builtin_value) < 1e-6

    pv01_builtin = swap.pv01(value_dt, model.curves.GBP_OIS_SONIA)
    assert pytest.approx(803.6445, rel=1e-3) == pv01_builtin

    bump_model = model.scenario("GBP_OIS_SONIA", {"10Y": 0.01})
    bumped_value = swap.value(value_dt, bump_model.curves.GBP_OIS_SONIA)
    assert pytest.approx(803.5675, rel=1e-3) == bumped_value