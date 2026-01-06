"""
Inflation Index implementation for managing CPI/RPI/HICP fixings and projections.

Provides the InflationIndex class for:
- Storing historical inflation index fixings
- Interpolating monthly CPI values for intra-month dates
- Applying publication lag (typically 3 months)
- Projecting forward CPI values from inflation curves
- Calculating inflation ratios between dates

Key features:
- Support for multiple index types (UK RPI/CPI, US CPI-U, EUR HICP)
- Flexible interpolation methods (FLAT, LINEAR, COMPOUND)
- Integration with inflation curves for forward projection
- Lag adjustment for publication delays

Example:
    >>> # Create UK RPI index
    >>> base_dt = Date(1, 1, 2023)
    >>> rpi_index = InflationIndex(
    ...     index_type=InflationIndexTypes.UK_RPI,
    ...     base_date=base_dt,
    ...     base_index=125.4,
    ...     currency=CurrencyTypes.GBP,
    ...     lag_months=3
    ... )
    >>>
    >>> # Add historical fixings
    >>> rpi_index.add_fixing(Date(1, 1, 2023), 125.4)
    >>> rpi_index.add_fixing(Date(1, 2, 2023), 126.1)
    >>>
    >>> # Get interpolated value
    >>> cpi_value = rpi_index.get_index(Date(15, 1, 2023))
"""

import numpy as np
from typing import Dict, Optional

from cavour.utils.error import LibError
from cavour.utils.date import Date
from cavour.utils.day_count import DayCount, DayCountTypes
from cavour.utils.global_types import InflationIndexTypes, InflationInterpTypes
from cavour.utils.currency import CurrencyTypes
from cavour.utils.helpers import check_argument_types, label_to_string

###############################################################################


class InflationIndex:
    """
    Inflation index for managing CPI/RPI/HICP fixings and forward projections.

    An inflation index represents a price index (e.g., UK RPI, US CPI-U) with:
    - Historical monthly fixings
    - Base reference date and value
    - Publication lag (typically 3 months)
    - Interpolation method for intra-month dates
    - Optional inflation curve for forward projections

    The index handles the complexity of:
    1. Monthly publication with lag (e.g., Jan CPI published in April)
    2. Daily interpolation for swap cashflow calculations
    3. Forward projection when fixings are not yet available

    Market conventions:
    - UK RPI: 3-month lag, linear interpolation
    - US CPI-U: 3-month lag, linear interpolation
    - EUR HICP: 3-month lag, linear interpolation
    """

    def __init__(self,
                 index_type: InflationIndexTypes,
                 base_date: Date,
                 base_index: float,
                 currency: CurrencyTypes,
                 lag_months: int = 3,
                 interp_type: InflationInterpTypes = InflationInterpTypes.LINEAR,
                 seasonality_factors: Optional[Dict[int, float]] = None):
        """
        Create an inflation index.

        Args:
            index_type: Type of inflation index (RPI, CPI, HICP, etc.)
            base_date: Base reference date for the index
            base_index: Base CPI value at base_date (e.g., 125.4)
            currency: Currency of the index
            lag_months: Publication lag in months (typically 3)
            interp_type: Interpolation method for daily values
            seasonality_factors: Optional dict of monthly seasonality adjustments.
                                Maps month (1-12) to multiplicative factor.
                                Factors should average to ~1.0 over the year.
                                Example: {1: 1.015, 2: 1.008, ..., 12: 0.992}

        Example:
            >>> rpi = InflationIndex(
            ...     index_type=InflationIndexTypes.UK_RPI,
            ...     base_date=Date(1, 1, 2020),
            ...     base_index=100.0,
            ...     currency=CurrencyTypes.GBP,
            ...     lag_months=3
            ... )
            >>>
            >>> # With seasonality
            >>> uk_seasonality = {1: 1.015, 2: 1.008, ..., 12: 0.992}
            >>> rpi_seasonal = InflationIndex(
            ...     index_type=InflationIndexTypes.UK_RPI,
            ...     base_date=Date(1, 1, 2020),
            ...     base_index=100.0,
            ...     currency=CurrencyTypes.GBP,
            ...     lag_months=3,
            ...     seasonality_factors=uk_seasonality
            ... )
        """
        check_argument_types(self.__init__, locals())

        if base_index <= 0.0:
            raise LibError("Base index must be positive")

        if lag_months < 0:
            raise LibError("Lag months must be non-negative")

        # Validate seasonality factors if provided
        if seasonality_factors is not None:
            self._validate_seasonality_factors(seasonality_factors)

        self._index_type = index_type
        self._base_date = base_date
        self._base_index = base_index
        self._currency = currency
        self._lag_months = lag_months
        self._interp_type = interp_type
        self._seasonality_factors = seasonality_factors or {}
        self._use_seasonality = len(self._seasonality_factors) > 0

        # Historical fixings: {excel_dt: (Date, float)}
        # Store using excel serial date as key (since Date is not hashable)
        self._fixings: Dict[int, tuple] = {}

        # Add base fixing
        self._fixings[base_date._excel_dt] = (base_date, base_index)

        # Inflation curve for forward projections (set later)
        self._inflation_curve = None

###############################################################################

    def _validate_seasonality_factors(self, factors: Dict[int, float]):
        """
        Validate seasonality factors.

        Checks:
        1. All months (1-12) are represented
        2. All factors are positive
        3. Factors average to approximately 1.0 (within 1% tolerance)

        Args:
            factors: Dict mapping month (1-12) to seasonality factor

        Raises:
            LibError if validation fails
        """
        # Check all months present
        if set(factors.keys()) != set(range(1, 13)):
            raise LibError(
                f"Seasonality factors must include all months 1-12. "
                f"Got: {sorted(factors.keys())}"
            )

        # Check all positive
        for month, factor in factors.items():
            if factor <= 0:
                raise LibError(
                    f"Seasonality factors must be positive. "
                    f"Month {month} has factor {factor}"
                )

        # Check average is approximately 1.0 (within 1% tolerance)
        avg_factor = sum(factors.values()) / 12.0
        if abs(avg_factor - 1.0) > 0.01:
            raise LibError(
                f"Seasonality factors should average to 1.0 (within 1% tolerance). "
                f"Got average: {avg_factor:.6f}"
            )

###############################################################################

    def _apply_seasonality(self, date: Date, cpi_value: float) -> float:
        """
        Apply seasonality adjustment to CPI value.

        Args:
            date: Date to get seasonality factor for
            cpi_value: Raw CPI value

        Returns:
            Seasonally adjusted CPI value
        """
        if not self._use_seasonality:
            return cpi_value

        month = date._m  # Month number (1-12)
        factor = self._seasonality_factors.get(month, 1.0)
        return cpi_value * factor

###############################################################################

    def add_fixing(self, fixing_date: Date, index_value: float):
        """
        Add a historical CPI fixing.

        Args:
            fixing_date: Date of the fixing (typically first of month)
            index_value: CPI value at that date

        Example:
            >>> rpi.add_fixing(Date(1, 2, 2020), 101.5)
            >>> rpi.add_fixing(Date(1, 3, 2020), 102.3)
        """
        if index_value <= 0.0:
            raise LibError(f"Index value must be positive, got {index_value}")

        self._fixings[fixing_date._excel_dt] = (fixing_date, index_value)

###############################################################################

    def set_inflation_curve(self, inflation_curve):
        """
        Attach inflation curve for forward CPI projections.

        Args:
            inflation_curve: InflationCurve object for projecting future CPI

        Example:
            >>> rpi.set_inflation_curve(infl_curve)
        """
        self._inflation_curve = inflation_curve

###############################################################################

    def get_index(self, ref_date: Date, apply_lag: bool = True) -> float:
        """
        Get CPI value at reference date (with optional lag and seasonality applied).

        This is the core method for retrieving CPI values:
        1. Apply lag if requested (shift back by lag_months)
        2. Check if we have a fixing for that month
        3. If yes: return fixing (with interpolation for intra-month dates)
        4. If no: project from inflation curve
        5. Apply seasonality adjustment if enabled

        Args:
            ref_date: Reference date for CPI lookup
            apply_lag: Whether to apply publication lag (default True)

        Returns:
            CPI value at the (lagged) reference date, with seasonality if enabled

        Example:
            >>> # Get CPI for 15-Jun-2023 with 3-month lag
            >>> # Actually looks up CPI for 15-Mar-2023
            >>> cpi = rpi.get_index(Date(15, 6, 2023))
        """
        # Apply lag if requested
        if apply_lag:
            lookup_date = self._apply_lag(ref_date)
        else:
            lookup_date = ref_date

        # Try to get from historical fixings
        index_value = self._get_historical_index(lookup_date)

        if index_value is not None:
            # Apply seasonality adjustment before returning
            return self._apply_seasonality(lookup_date, index_value)

        # If not in fixings, try to project from inflation curve
        if self._inflation_curve is not None:
            curve_value = self._inflation_curve.forward_index(lookup_date)
            # Apply seasonality adjustment before returning
            return self._apply_seasonality(lookup_date, curve_value)

        # No fixings and no curve
        raise LibError(
            f"No fixing available for {lookup_date} and no inflation curve set. "
            f"Add fixings via add_fixing() or set curve via set_inflation_curve()."
        )

###############################################################################

    def inflation_ratio(self,
                       start_dt: Date,
                       end_dt: Date,
                       apply_lag: bool = True) -> float:
        """
        Calculate inflation index ratio: I(end) / I(start).

        This is the fundamental calculation for zero-coupon inflation swaps.

        Args:
            start_dt: Start date (base date)
            end_dt: End date (final date)
            apply_lag: Whether to apply publication lag

        Returns:
            Index ratio (e.g., 1.15 means 15% cumulative inflation)

        Example:
            >>> # Calculate 5-year inflation
            >>> ratio = rpi.inflation_ratio(
            ...     Date(1, 1, 2020),
            ...     Date(1, 1, 2025)
            ... )
            >>> inflation_pct = (ratio - 1.0) * 100
        """
        index_start = self.get_index(start_dt, apply_lag=apply_lag)
        index_end = self.get_index(end_dt, apply_lag=apply_lag)

        if index_start <= 0.0:
            raise LibError(f"Start index must be positive, got {index_start}")

        return index_end / index_start

###############################################################################

    def _apply_lag(self, ref_date: Date) -> Date:
        """
        Apply publication lag by shifting date back by lag_months.

        Args:
            ref_date: Original reference date

        Returns:
            Lagged date (shifted back by lag_months)

        Example:
            >>> # With 3-month lag: 15-Jun-2023 → 15-Mar-2023
            >>> lagged = rpi._apply_lag(Date(15, 6, 2023))
        """
        return ref_date.add_months(-self._lag_months)

###############################################################################

    def _get_historical_index(self, lookup_date: Date) -> Optional[float]:
        """
        Get CPI from historical fixings with interpolation.

        Handles three cases:
        1. Exact fixing exists: return it
        2. Between two fixings: interpolate
        3. Before/after all fixings: return None

        Args:
            lookup_date: Date to lookup (already lagged if applicable)

        Returns:
            Interpolated CPI value, or None if outside fixing range
        """
        if not self._fixings:
            return None

        # Get sorted fixing excel dates and convert to Date objects
        sorted_excel_dts = sorted(self._fixings.keys())
        fixing_dates = [self._fixings[excel_dt][0] for excel_dt in sorted_excel_dts]

        # Check if before earliest fixing
        if lookup_date < fixing_dates[0]:
            return None

        # Check if after latest fixing
        if lookup_date > fixing_dates[-1]:
            return None

        # Check for exact match
        if lookup_date._excel_dt in self._fixings:
            return self._fixings[lookup_date._excel_dt][1]

        # Find bracketing fixings
        lower_date = None
        upper_date = None
        lower_value = None
        upper_value = None

        for i in range(len(fixing_dates) - 1):
            if fixing_dates[i] <= lookup_date <= fixing_dates[i + 1]:
                lower_date = fixing_dates[i]
                upper_date = fixing_dates[i + 1]
                lower_value = self._fixings[sorted_excel_dts[i]][1]
                upper_value = self._fixings[sorted_excel_dts[i + 1]][1]
                break

        if lower_date is None or upper_date is None:
            return None

        # Interpolate between lower and upper
        return self._interpolate(
            lookup_date,
            lower_date,
            upper_date,
            lower_value,
            upper_value
        )

###############################################################################

    def _interpolate(self,
                     target_date: Date,
                     lower_date: Date,
                     upper_date: Date,
                     lower_value: float,
                     upper_value: float) -> float:
        """
        Interpolate CPI between two monthly fixings.

        Supports three methods:
        - FLAT: Use lower (prior month) value
        - LINEAR: Linear interpolation between months
        - COMPOUND: Compound rate interpolation

        Args:
            target_date: Date to interpolate for
            lower_date: Earlier fixing date
            upper_date: Later fixing date
            lower_value: CPI at lower_date
            upper_value: CPI at upper_date

        Returns:
            Interpolated CPI value
        """
        if self._interp_type == InflationInterpTypes.FLAT:
            # Use prior month's value
            return lower_value

        elif self._interp_type == InflationInterpTypes.LINEAR:
            # Linear interpolation by days
            day_counter = DayCount(DayCountTypes.ACT_365F)

            (total_days, _, _) = day_counter.year_frac(lower_date, upper_date)
            (elapsed_days, _, _) = day_counter.year_frac(lower_date, target_date)

            if total_days == 0:
                return lower_value

            # Linear blend
            weight = elapsed_days / total_days
            return lower_value + weight * (upper_value - lower_value)

        elif self._interp_type == InflationInterpTypes.COMPOUND:
            # Compound rate interpolation
            day_counter = DayCount(DayCountTypes.ACT_365F)

            (total_years, _, _) = day_counter.year_frac(lower_date, upper_date)
            (elapsed_years, _, _) = day_counter.year_frac(lower_date, target_date)

            if total_years == 0:
                return lower_value

            # Compound: I(t) = I0 × (If/I0)^(t/T)
            ratio = upper_value / lower_value
            weight = elapsed_years / total_years
            return lower_value * (ratio ** weight)

        else:
            raise LibError(f"Unknown interpolation type: {self._interp_type}")

###############################################################################

    def get_all_fixings(self) -> list:
        """
        Get all historical fixings.

        Returns:
            List of tuples [(Date, CPI value), ...]

        Note:
            Returns a list instead of dict because Date objects are not hashable.
            To check if a specific date has a fixing, use get_index() instead.
        """
        # Convert internal format to user-friendly format
        # Return as list of tuples since Date is not hashable
        return [(date, value) for date, value in self._fixings.values()]

###############################################################################

    def __repr__(self):
        s = label_to_string("OBJECT TYPE", type(self).__name__)
        s += label_to_string("INDEX TYPE", self._index_type)
        s += label_to_string("BASE DATE", self._base_date)
        s += label_to_string("BASE INDEX", self._base_index)
        s += label_to_string("CURRENCY", self._currency)
        s += label_to_string("LAG (MONTHS)", self._lag_months)
        s += label_to_string("INTERPOLATION", self._interp_type)
        s += label_to_string("NUM FIXINGS", len(self._fixings))
        s += label_to_string("HAS CURVE", self._inflation_curve is not None)
        s += label_to_string("SEASONALITY", "Enabled" if self._use_seasonality else "Disabled")
        return s

###############################################################################
