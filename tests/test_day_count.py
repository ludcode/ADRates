"""
Comprehensive tests for day count conventions.

Tests all day count types including:
- ACT_365F, ACT_360, ACT_ACT_ISDA, ACT_ACT_ICMA, ACT_365L
- THIRTY_360_BOND, THIRTY_E_360, THIRTY_E_360_ISDA, THIRTY_E_PLUS_360
- Edge cases: leap years, month-end dates, February, year boundaries

References:
- ISDA 2006 Definitions
- OpenGamma day count documentation
"""

import pytest
from cavour.utils.date import Date
from cavour.utils.day_count import DayCount, DayCountTypes, is_last_day_of_feb
from cavour.utils.frequency import FrequencyTypes


class TestDayCountACT365F:
    """Test ACT/365F convention - always uses 365 as denominator"""

    def test_act_365f_simple_period(self):
        """Test ACT/365F for simple 6-month period"""
        dc = DayCount(DayCountTypes.ACT_365F)
        dt1 = Date(15, 6, 2023)
        dt2 = Date(15, 12, 2023)

        year_frac, num_days, denom = dc.year_frac(dt1, dt2)

        assert num_days == 183  # Actual days
        assert denom == 365
        assert abs(year_frac - 183/365) < 1e-12

    def test_act_365f_leap_year(self):
        """Test ACT/365F across leap year - denom stays 365"""
        dc = DayCount(DayCountTypes.ACT_365F)
        dt1 = Date(1, 1, 2024)  # Leap year
        dt2 = Date(1, 1, 2025)

        year_frac, num_days, denom = dc.year_frac(dt1, dt2)

        assert num_days == 366  # 2024 has 366 days
        assert denom == 365  # But denom is still 365
        assert abs(year_frac - 366/365) < 1e-12

    def test_act_365f_february_leap(self):
        """Test ACT/365F across Feb 29 in leap year"""
        dc = DayCount(DayCountTypes.ACT_365F)
        dt1 = Date(28, 2, 2024)
        dt2 = Date(1, 3, 2024)

        year_frac, num_days, denom = dc.year_frac(dt1, dt2)

        assert num_days == 2  # 29th and 1st
        assert denom == 365

    def test_act_365f_same_date(self):
        """Test ACT/365F with same date"""
        dc = DayCount(DayCountTypes.ACT_365F)
        dt1 = Date(15, 6, 2023)

        year_frac, num_days, denom = dc.year_frac(dt1, dt1)

        assert num_days == 0
        assert denom == 365
        assert year_frac == 0.0


class TestDayCountACT360:
    """Test ACT/360 convention - money market basis"""

    def test_act_360_simple(self):
        """Test ACT/360 for simple period"""
        dc = DayCount(DayCountTypes.ACT_360)
        dt1 = Date(15, 6, 2023)
        dt2 = Date(15, 9, 2023)

        year_frac, num_days, denom = dc.year_frac(dt1, dt2)

        assert num_days == 92
        assert denom == 360
        assert abs(year_frac - 92/360) < 1e-12

    def test_act_360_leap_year(self):
        """Test ACT/360 in leap year"""
        dc = DayCount(DayCountTypes.ACT_360)
        dt1 = Date(1, 1, 2024)
        dt2 = Date(1, 1, 2025)

        year_frac, num_days, denom = dc.year_frac(dt1, dt2)

        assert num_days == 366
        assert denom == 360
        assert abs(year_frac - 366/360) < 1e-12


class TestDayCountACTACTISDA:
    """Test ACT/ACT ISDA - splits across years"""

    def test_act_act_isda_same_year(self):
        """Test ACT/ACT ISDA within same year"""
        dc = DayCount(DayCountTypes.ACT_ACT_ISDA)
        dt1 = Date(15, 6, 2023)
        dt2 = Date(15, 12, 2023)

        year_frac, num_days, denom = dc.year_frac(dt1, dt2)

        assert num_days == 183
        assert denom == 365  # 2023 is not leap year
        assert abs(year_frac - 183/365) < 1e-12

    def test_act_act_isda_across_years(self):
        """Test ACT/ACT ISDA across year boundary"""
        dc = DayCount(DayCountTypes.ACT_ACT_ISDA)
        dt1 = Date(1, 7, 2023)
        dt2 = Date(1, 7, 2024)

        year_frac, _, _ = dc.year_frac(dt1, dt2)

        # Implementation: calculates days from dt1 to end of year 1, and from start of year 2 to dt2
        # Jul 1 2023 to Jan 1 2024: 184 days
        # Jan 1 2024 to Jul 1 2024: 182 days
        # year_diff = 2024 - 2023 - 1 = 0
        # So: 184/365 + 182/366 + 0
        expected = 184/365 + 182/366
        assert abs(year_frac - expected) < 1e-12

    def test_act_act_isda_leap_to_non_leap(self):
        """Test ACT/ACT ISDA from leap to non-leap year"""
        dc = DayCount(DayCountTypes.ACT_ACT_ISDA)
        dt1 = Date(1, 1, 2024)  # Leap year
        dt2 = Date(1, 1, 2025)  # Non-leap year

        year_frac, _, _ = dc.year_frac(dt1, dt2)

        # All 366 days in 2024
        expected = 366/366
        assert abs(year_frac - expected) < 1e-12
        assert abs(year_frac - 1.0) < 1e-12

    def test_act_act_isda_multi_year(self):
        """Test ACT/ACT ISDA spanning multiple years"""
        dc = DayCount(DayCountTypes.ACT_ACT_ISDA)
        dt1 = Date(1, 6, 2023)
        dt2 = Date(1, 6, 2025)

        year_frac, _, _ = dc.year_frac(dt1, dt2)

        # Jun 1 2023 to Jan 1 2024: 214 days / 365
        # Jan 1 2025 to Jun 1 2025: 151 days / 365
        # year_diff = 2025 - 2023 - 1 = 1
        expected = 214/365 + 151/365 + 1.0
        assert abs(year_frac - expected) < 1e-10


class TestDayCountACTACTICMA:
    """Test ACT/ACT ICMA - used for bonds"""

    def test_act_act_icma_semi_annual(self):
        """Test ACT/ACT ICMA with semi-annual frequency"""
        dc = DayCount(DayCountTypes.ACT_ACT_ICMA)
        dt1 = Date(15, 6, 2023)   # Previous coupon date
        dt2 = Date(15, 9, 2023)   # Settlement date
        dt3 = Date(15, 12, 2023)  # Next coupon date

        year_frac, num_days, denom = dc.year_frac(
            dt1, dt2, dt3, FrequencyTypes.SEMI_ANNUAL
        )

        assert num_days == 92
        # denom = freq * (dt3 - dt1) = 2 * 183 = 366
        expected_denom = 2 * 183
        assert denom == expected_denom
        assert abs(year_frac - 92/366) < 1e-12

    def test_act_act_icma_quarterly(self):
        """Test ACT/ACT ICMA with quarterly frequency"""
        dc = DayCount(DayCountTypes.ACT_ACT_ICMA)
        dt1 = Date(15, 6, 2023)
        dt2 = Date(15, 7, 2023)
        dt3 = Date(15, 9, 2023)

        year_frac, num_days, denom = dc.year_frac(
            dt1, dt2, dt3, FrequencyTypes.QUARTERLY
        )

        assert num_days == 30
        # denom = freq * (dt3 - dt1) = 4 * 92 = 368
        expected_denom = 4 * 92
        assert denom == expected_denom

    def test_act_act_icma_requires_dt3(self):
        """Test that ACT/ACT ICMA requires dt3"""
        dc = DayCount(DayCountTypes.ACT_ACT_ICMA)
        dt1 = Date(15, 6, 2023)
        dt2 = Date(15, 12, 2023)

        with pytest.raises(Exception):  # Should raise LibError
            dc.year_frac(dt1, dt2, None, FrequencyTypes.ANNUAL)


class TestDayCountACT365L:
    """Test ACT/365L - accounts for leap day"""

    def test_act_365l_non_leap_year(self):
        """Test ACT/365L in non-leap year"""
        dc = DayCount(DayCountTypes.ACT_365L)
        dt1 = Date(1, 1, 2023)
        dt2 = Date(1, 7, 2023)
        dt3 = Date(1, 1, 2024)

        year_frac, num_days, denom = dc.year_frac(
            dt1, dt2, dt3, FrequencyTypes.ANNUAL
        )

        assert num_days == 181
        assert denom == 365  # No leap day in period

    def test_act_365l_with_leap_day(self):
        """Test ACT/365L spanning leap day"""
        dc = DayCount(DayCountTypes.ACT_365L)
        dt1 = Date(1, 1, 2024)
        dt2 = Date(1, 7, 2024)
        dt3 = Date(1, 1, 2025)

        year_frac, num_days, denom = dc.year_frac(
            dt1, dt2, dt3, FrequencyTypes.ANNUAL
        )

        assert num_days == 182  # Jan 1 to Jul 1 is 182 days
        assert denom == 366  # Leap day in period


class TestDayCountThirty360Bond:
    """Test THIRTY_360_BOND convention"""

    def test_thirty_360_bond_simple(self):
        """Test 30/360 Bond for simple period"""
        dc = DayCount(DayCountTypes.THIRTY_360_BOND)
        dt1 = Date(15, 6, 2023)
        dt2 = Date(15, 12, 2023)

        year_frac, num_days, denom = dc.year_frac(dt1, dt2)

        # 360 * (2023 - 2023) + 30 * (12 - 6) + (15 - 15) = 180
        assert num_days == 180
        assert denom == 360
        assert abs(year_frac - 0.5) < 1e-12

    def test_thirty_360_bond_day_31(self):
        """Test 30/360 Bond with day 31"""
        dc = DayCount(DayCountTypes.THIRTY_360_BOND)
        dt1 = Date(31, 1, 2023)
        dt2 = Date(28, 2, 2023)

        year_frac, num_days, denom = dc.year_frac(dt1, dt2)

        # d1=31 -> d1=30, d2=28
        # 360*0 + 30*(2-1) + (28-30) = 30 - 2 = 28
        assert num_days == 28
        assert denom == 360

    def test_thirty_360_bond_both_31(self):
        """Test 30/360 Bond with both dates on 31st"""
        dc = DayCount(DayCountTypes.THIRTY_360_BOND)
        dt1 = Date(31, 1, 2023)
        dt2 = Date(31, 3, 2023)

        year_frac, num_days, denom = dc.year_frac(dt1, dt2)

        # d1=31 -> d1=30, d2=31 and d1==30 -> d2=30
        # 360*0 + 30*(3-1) + (30-30) = 60
        assert num_days == 60
        assert denom == 360

    def test_thirty_360_bond_february_28(self):
        """Test 30/360 Bond with February 28 (non-leap)"""
        dc = DayCount(DayCountTypes.THIRTY_360_BOND)
        dt1 = Date(28, 2, 2023)  # Not leap year
        dt2 = Date(31, 3, 2023)

        year_frac, num_days, denom = dc.year_frac(dt1, dt2)

        # d1=28, d2=31 (d2 stays 31 because d1 != 30)
        # 360*0 + 30*(3-2) + (31-28) = 30 + 3 = 33
        assert num_days == 33
        assert denom == 360


class TestDayCountThirtyE360:
    """Test THIRTY_E_360 convention (Eurobond)"""

    def test_thirty_e_360_simple(self):
        """Test 30E/360 for simple period"""
        dc = DayCount(DayCountTypes.THIRTY_E_360)
        dt1 = Date(15, 6, 2023)
        dt2 = Date(15, 12, 2023)

        year_frac, num_days, denom = dc.year_frac(dt1, dt2)

        assert num_days == 180
        assert denom == 360
        assert abs(year_frac - 0.5) < 1e-12

    def test_thirty_e_360_both_31(self):
        """Test 30E/360 with both dates on 31st"""
        dc = DayCount(DayCountTypes.THIRTY_E_360)
        dt1 = Date(31, 1, 2023)
        dt2 = Date(31, 3, 2023)

        year_frac, num_days, denom = dc.year_frac(dt1, dt2)

        # Both 31s become 30, independently
        # 360*0 + 30*(3-1) + (30-30) = 60
        assert num_days == 60
        assert denom == 360

    def test_thirty_e_360_end_31_start_15(self):
        """Test 30E/360 where end is 31st but start is not"""
        dc = DayCount(DayCountTypes.THIRTY_E_360)
        dt1 = Date(15, 1, 2023)
        dt2 = Date(31, 3, 2023)

        year_frac, num_days, denom = dc.year_frac(dt1, dt2)

        # d1=15, d2=31->30
        # 360*0 + 30*(3-1) + (30-15) = 60 + 15 = 75
        assert num_days == 75
        assert denom == 360


class TestDayCountThirtyE360ISDA:
    """Test THIRTY_E_360_ISDA convention"""

    def test_thirty_e_360_isda_feb_last_day(self):
        """Test 30E/360 ISDA with last day of February"""
        dc = DayCount(DayCountTypes.THIRTY_E_360_ISDA)
        dt1 = Date(28, 2, 2023)  # Last day of Feb (non-leap)
        dt2 = Date(31, 3, 2023)

        year_frac, num_days, denom = dc.year_frac(
            dt1, dt2, isTerminationDate=False
        )

        # d1=28 (last day of Feb) -> 30
        # d2=31 -> 30 (not termination date)
        # 360*0 + 30*(3-2) + (30-30) = 30
        assert num_days == 30
        assert denom == 360

    def test_thirty_e_360_isda_feb_termination(self):
        """Test 30E/360 ISDA with Feb as termination date"""
        dc = DayCount(DayCountTypes.THIRTY_E_360_ISDA)
        dt1 = Date(31, 1, 2023)
        dt2 = Date(28, 2, 2023)  # Last day of Feb, termination

        year_frac, num_days, denom = dc.year_frac(
            dt1, dt2, isTerminationDate=True
        )

        # d1=31 -> 30
        # d2=28 (last day of Feb, termination) -> stays 28
        # 360*0 + 30*(2-1) + (28-30) = 30 - 2 = 28
        assert num_days == 28
        assert denom == 360

    def test_thirty_e_360_isda_leap_year_feb(self):
        """Test 30E/360 ISDA with Feb 29 in leap year"""
        dc = DayCount(DayCountTypes.THIRTY_E_360_ISDA)
        dt1 = Date(29, 2, 2024)  # Last day of Feb (leap)
        dt2 = Date(31, 3, 2024)

        year_frac, num_days, denom = dc.year_frac(
            dt1, dt2, isTerminationDate=False
        )

        # d1=29 (last day of Feb) -> 30
        # d2=31 -> 30
        # 360*0 + 30*(3-2) + (30-30) = 30
        assert num_days == 30
        assert denom == 360


class TestDayCountThirtyEPlus360:
    """Test THIRTY_E_PLUS_360 convention"""

    def test_thirty_e_plus_360_day_31_rolls(self):
        """Test 30E+/360 where day 31 rolls to next month"""
        dc = DayCount(DayCountTypes.THIRTY_E_PLUS_360)
        dt1 = Date(15, 1, 2023)
        dt2 = Date(31, 3, 2023)

        year_frac, num_days, denom = dc.year_frac(dt1, dt2)

        # d1=15, d2=31 -> m2=4, d2=1 (rolls to April 1st)
        # 360*0 + 30*(4-1) + (1-15) = 90 - 14 = 76
        assert num_days == 76
        assert denom == 360

    def test_thirty_e_plus_360_start_31(self):
        """Test 30E+/360 where start date is 31st"""
        dc = DayCount(DayCountTypes.THIRTY_E_PLUS_360)
        dt1 = Date(31, 1, 2023)
        dt2 = Date(15, 3, 2023)

        year_frac, num_days, denom = dc.year_frac(dt1, dt2)

        # d1=31 -> 30, d2=15
        # 360*0 + 30*(3-1) + (15-30) = 60 - 15 = 45
        assert num_days == 45
        assert denom == 360


class TestHelperFunctions:
    """Test helper functions"""

    def test_is_last_day_of_feb_non_leap(self):
        """Test is_last_day_of_feb for non-leap year"""
        dt = Date(28, 2, 2023)
        assert is_last_day_of_feb(dt) is True

        dt2 = Date(27, 2, 2023)
        # Function returns None when not last day of Feb, not False
        assert is_last_day_of_feb(dt2) is not True

    def test_is_last_day_of_feb_leap(self):
        """Test is_last_day_of_feb for leap year"""
        dt = Date(29, 2, 2024)
        assert is_last_day_of_feb(dt) is True

        dt2 = Date(28, 2, 2024)
        # Function returns None when not last day of Feb, not False
        assert is_last_day_of_feb(dt2) is not True

    def test_is_last_day_of_feb_other_month(self):
        """Test is_last_day_of_feb for non-February date"""
        dt = Date(31, 3, 2023)
        assert is_last_day_of_feb(dt) is False


class TestDayCountEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_year_boundary(self):
        """Test day count across year boundary"""
        dc = DayCount(DayCountTypes.ACT_365F)
        dt1 = Date(31, 12, 2023)
        dt2 = Date(1, 1, 2024)

        year_frac, num_days, denom = dc.year_frac(dt1, dt2)

        assert num_days == 1
        assert denom == 365

    def test_leap_year_feb_29_to_mar_1(self):
        """Test Feb 29 to Mar 1 in leap year"""
        dc = DayCount(DayCountTypes.ACT_365F)
        dt1 = Date(29, 2, 2024)
        dt2 = Date(1, 3, 2024)

        year_frac, num_days, denom = dc.year_frac(dt1, dt2)

        assert num_days == 1
        assert denom == 365

    def test_month_end_30_vs_31(self):
        """Test month-end dates with 30 vs 31 days"""
        dc = DayCount(DayCountTypes.ACT_365F)
        dt1 = Date(30, 4, 2023)  # April has 30 days
        dt2 = Date(31, 5, 2023)  # May has 31 days

        year_frac, num_days, denom = dc.year_frac(dt1, dt2)

        assert num_days == 31
        assert denom == 365

    def test_all_conventions_same_dates(self):
        """Test that all conventions give 0 for same dates"""
        dt = Date(15, 6, 2023)

        for dc_type in DayCountTypes:
            if dc_type == DayCountTypes.ZERO:
                continue  # Skip ZERO type
            if dc_type == DayCountTypes.ACT_ACT_ICMA:
                continue  # Requires dt3
            if dc_type == DayCountTypes.ACT_365L:
                continue  # Requires dt3

            dc = DayCount(dc_type)
            year_frac, _, _ = dc.year_frac(dt, dt)
            assert year_frac == 0.0, f"Failed for {dc_type}"


class TestDayCountConsistency:
    """Test consistency across conventions"""

    def test_act_365f_vs_act_360_numerator(self):
        """Test that ACT/365F and ACT/360 have same numerator"""
        dt1 = Date(15, 6, 2023)
        dt2 = Date(15, 12, 2023)

        dc_365f = DayCount(DayCountTypes.ACT_365F)
        dc_360 = DayCount(DayCountTypes.ACT_360)

        _, num_365f, _ = dc_365f.year_frac(dt1, dt2)
        _, num_360, _ = dc_360.year_frac(dt1, dt2)

        assert num_365f == num_360  # Same actual days

    def test_thirty_conventions_six_months(self):
        """Test that 30/360 variants all give 180 for exact 6 months"""
        dt1 = Date(15, 6, 2023)
        dt2 = Date(15, 12, 2023)

        for dc_type in [DayCountTypes.THIRTY_360_BOND,
                        DayCountTypes.THIRTY_E_360]:
            dc = DayCount(dc_type)
            year_frac, num_days, _ = dc.year_frac(dt1, dt2)
            assert num_days == 180, f"Failed for {dc_type}"
            assert abs(year_frac - 0.5) < 1e-12, f"Failed for {dc_type}"


class TestDaysInYear:
    """Test days_in_year method"""

    def test_days_in_year_360_conventions(self):
        """Test days_in_year for 30/360 conventions"""
        for dc_type in [DayCountTypes.THIRTY_360_BOND,
                        DayCountTypes.THIRTY_E_360,
                        DayCountTypes.THIRTY_E_360_ISDA,
                        DayCountTypes.THIRTY_E_PLUS_360,
                        DayCountTypes.ACT_360]:
            dc = DayCount(dc_type)
            assert dc.days_in_year() == 360

    def test_days_in_year_365f(self):
        """Test days_in_year for ACT/365F"""
        dc = DayCount(DayCountTypes.ACT_365F)
        assert dc.days_in_year() == 365

    def test_days_in_year_raises_for_variable_conventions(self):
        """Test that variable conventions raise error"""
        for dc_type in [DayCountTypes.ACT_ACT_ISDA,
                        DayCountTypes.ACT_365L,
                        DayCountTypes.ACT_ACT_ICMA]:
            dc = DayCount(dc_type)
            with pytest.raises(Exception):
                dc.days_in_year()
