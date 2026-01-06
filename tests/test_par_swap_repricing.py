"""
Par swap repricing validation tests.

A fundamental requirement of curve bootstrapping is that the input par swaps
(the swaps used to build the curve) must reprice at exactly zero PV when
valued using the bootstrapped curve.

This is a critical validation that ensures:
- The curve bootstrap algorithm is working correctly
- The curve accurately reflects market prices
- There are no numerical precision issues
- The pricing and curve building logic are consistent

These tests complement test_refit_curves.py by testing internal curve validation.

NOTE: Some tenor combinations trigger an IndexError in the OIS curve
bootstrap logic (ois_curve.py:187). This is a known library issue.
Tests use tenor combinations that avoid this bug.
"""

import pytest
import numpy as np
from cavour.utils.date import Date
from cavour.utils.global_types import SwapTypes
from cavour.utils.day_count import DayCountTypes
from cavour.utils.frequency import FrequencyTypes
from cavour.utils.calendar import BusDayAdjustTypes
from cavour.market.curves.interpolator import InterpTypes
from cavour.models.models import Model


class TestGBPSONIAParRepricing:
    """
    Test that GBP SONIA par swaps reprice at zero using internal curve validation.

    Note: These tests validate the fundamental principle that bootstrapped curves
    must accurately reprice the input par swaps. This is achieved by using the
    curve's internal _check_refits() method.

    Comprehensive repricing tests using the position engine are in test_refit_curves.py.
    """

    def test_gbp_sonia_full_curve_reprices(self):
        """Test that full GBP SONIA curve with realistic market data reprices correctly"""
        value_date = Date(30, 4, 2024)

        # Realistic GBP SONIA market rates (32 tenors)
        # This is real-world data that should reprice within strict tolerances
        px_list = [5.1998, 5.2014, 5.2003, 5.2027, 5.2023, 5.19281,
                   5.1656, 5.1482, 5.1342, 5.1173, 5.1013, 5.0862,
                   5.0701, 5.054, 5.0394, 4.8707, 4.75483, 4.532,
                   4.3628, 4.2428, 4.16225, 4.1132, 4.08505, 4.0762,
                   4.078, 4.0961, 4.12195, 4.1315, 4.113, 4.07724, 3.984, 3.88]

        tenor_list = ["1D", "1W", "2W", "1M", "2M", "3M", "4M", "5M", "6M",
                      "7M", "8M", "9M", "10M", "11M", "1Y", "18M", "2Y",
                      "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y",
                      "12Y", "15Y", "20Y", "25Y", "30Y", "40Y", "50Y"]

        model = Model(value_date)
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
            interp_type=InterpTypes.LINEAR_ZERO_RATES
        )

        curve = model.curves.GBP_OIS_SONIA

        # Use curve's internal validation method
        # Standard tolerance: all swaps should reprice within 1e-5 (0.001 bps)
        swap_tol = 1e-5
        curve._check_refits(swap_tol=swap_tol)


