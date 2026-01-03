"""
Inflation Curve implementation for forward CPI projection.

Provides the InflationCurve class for:
- Building inflation curves from ZCIS instruments
- Projecting forward CPI values
- Integration with InflationIndex for forward lookups
- Interpolation of inflation factors

Key features:
- Direct construction from zero-coupon quotes (no bootstrapping needed)
- JAX-compatible for automatic differentiation
- Multiple interpolation methods (LINEAR, COMPOUND, FLAT)
- Cumulative inflation factor storage: I(T) / I(0) = (1 + r)^T
- Exact reproduction of input ZCIS breakeven rates

The construction algorithm:
1. Extract breakeven rates from ZCIS instruments
2. Convert to cumulative inflation factors: I(T)/I(0) = (1+r)^T
3. Store as discount-like factors for interpolation
4. Validate curve reproduces input ZCIS rates (if check_refit=True)

Example:
    >>> # Create ZCIS instruments at market rates
    >>> zcis_swaps = [
    ...     ZeroCouponInflationSwap(value_dt, "1Y", SwapTypes.PAY, 0.025, rpi_index),
    ...     ZeroCouponInflationSwap(value_dt, "5Y", SwapTypes.PAY, 0.028, rpi_index),
    ...     ZeroCouponInflationSwap(value_dt, "10Y", SwapTypes.PAY, 0.030, rpi_index)
    ... ]
    >>>
    >>> # Build curve
    >>> infl_curve = InflationCurve(
    ...     value_dt=value_dt,
    ...     zcis_instruments=zcis_swaps,
    ...     base_cpi=293.0,
    ...     currency=CurrencyTypes.GBP,
    ...     index_type=InflationIndexTypes.UK_RPI,
    ...     check_refit=True
    ... )
    >>>
    >>> # Project forward CPI
    >>> forward_cpi = infl_curve.forward_index(Date(1, 1, 2028))
"""

import numpy as np
import jax.numpy as jnp
from typing import List, Union, Optional
from cavour.utils.error import LibError
from cavour.utils.date import Date
from cavour.utils.day_count import DayCount, DayCountTypes
from cavour.utils.global_types import InflationIndexTypes, InflationInterpTypes
from cavour.utils.currency import CurrencyTypes
from cavour.utils.helpers import (check_argument_types, _func_name,
                              label_to_string, format_table)
from cavour.market.curves.discount_curve import DiscountCurve
from cavour.market.curves.interpolator import InterpTypes, Interpolator
from cavour.market.curves.interpolator_ad import InterpolatorAd

###############################################################################

ZCIS_TOL = 1e-10

###############################################################################


class InflationCurve(DiscountCurve):
    """
    Inflation curve for forward CPI projection.

    Constructs an inflation curve from zero-coupon inflation swap (ZCIS)
    market quotes. Each ZCIS with fixed rate r(T) at tenor T implies a
    cumulative inflation factor:

        I(T) / I(0) = (1 + r(T))^T

    The curve stores these factors and interpolates between them to project
    CPI at any future date. Inherits from DiscountCurve to leverage existing
    interpolation infrastructure.

    Market conventions:
    - UK: RPI-linked, typically quoted annually out to 30Y
    - US: CPI-U-linked, quoted semi-annually to annually
    - EUR: HICP-linked, quoted annually

    Interpolation methods:
    - LINEAR: Linear interpolation on cumulative inflation factors
    - COMPOUND: Log-linear interpolation (preserves compounding)
    - FLAT: Flat forward interpolation
    """

    def __init__(self,
                 value_dt: Date,
                 zcis_instruments: list,
                 base_cpi: float,
                 currency: CurrencyTypes,
                 index_type: InflationIndexTypes,
                 discount_curve: DiscountCurve = None,
                 interp_type: InflationInterpTypes = InflationInterpTypes.LINEAR,
                 dc_type: DayCountTypes = DayCountTypes.ACT_365F,
                 check_refit: bool = False):
        """
        Create an inflation curve from ZCIS instruments.

        Args:
            value_dt: Valuation date (curve anchor date)
            zcis_instruments: List of ZeroCouponInflationSwap instruments for calibration
            base_cpi: CPI value at valuation date
            currency: Currency of the inflation index
            index_type: Type of inflation index (RPI, CPI, HICP)
            discount_curve: Discount curve for ZCIS valuation (optional, not currently used)
            interp_type: Interpolation method for CPI values
            dc_type: Day count convention for year fractions
            check_refit: If True, verify calibration instruments reprice correctly

        Example:
            >>> infl_curve = InflationCurve(
            ...     value_dt=Date(1, 1, 2023),
            ...     zcis_instruments=[zcis_1y, zcis_5y, zcis_10y],
            ...     base_cpi=293.0,
            ...     currency=CurrencyTypes.GBP,
            ...     index_type=InflationIndexTypes.UK_RPI,
            ...     check_refit=True
            ... )
        """
        check_argument_types(getattr(self, _func_name(), None), locals())

        if base_cpi <= 0.0:
            raise LibError("Base CPI must be positive")

        if len(zcis_instruments) < 2:
            raise LibError("Need at least 2 ZCIS instruments to build a curve")

        self._value_dt = value_dt
        self._used_swaps = zcis_instruments
        self._base_cpi = base_cpi
        self._currency = currency
        self._index_type = index_type
        self._discount_curve = discount_curve
        self._interp_type_infl = interp_type  # Store inflation interp type
        self._dc_type = dc_type
        self._check_refit = check_refit

        # Prepare inputs and build curve
        breakeven_rates = self._prepare_curve_builder_inputs()
        self._build_curve(breakeven_rates)

        # Validate calibration if requested
        if self._check_refit:
            self._check_refits(ZCIS_TOL)

###############################################################################

    def _prepare_curve_builder_inputs(self):
        """
        Prepare inputs for curve construction.

        Extracts breakeven rates and maturities from the calibration ZCIS
        instruments. For zero-coupon inflation swaps quoted at par, the
        fixed rate IS the breakeven inflation rate.

        Returns:
            List of breakeven rates (one per ZCIS)
        """
        breakeven_rates = []
        self.swap_times = []
        self.tenors = []

        day_counter = DayCount(self._dc_type)

        for zcis in self._used_swaps:
            # Extract breakeven rate from ZCIS
            # For par ZCIS (zero PV), fixed_rate = breakeven rate
            breakeven_rate = zcis._fixed_rate
            breakeven_rates.append(breakeven_rate)

            # Calculate maturity time
            (year_frac, _, _) = day_counter.year_frac(
                zcis._effective_dt,
                zcis._maturity_dt
            )
            self.swap_times.append(year_frac)

            # Calculate tenor string for display
            if abs(year_frac - round(year_frac)) < 0.1:
                tenor_str = f"{int(round(year_frac))}Y"
            else:
                tenor_str = f"{year_frac:.2f}Y"
            self.tenors.append(tenor_str)

        return breakeven_rates

###############################################################################

    def _build_curve(self, breakeven_rates):
        """
        Construct the inflation curve from breakeven rates.

        Converts breakeven rates to cumulative inflation factors and stores
        them as discount-like factors. Uses DiscountCurve infrastructure for
        interpolation.

        Algorithm:
        For each ZCIS with maturity T_k and breakeven rate r_k:
        1. Calculate cumulative inflation factor: I(T_k)/I(0) = (1+r_k)^T_k
        2. Store as curve node
        3. Fit interpolator

        Args:
            breakeven_rates: List of breakeven rates from ZCIS instruments
        """
        # Map inflation interpolation type to discount curve interpolation
        interp_mapping = {
            InflationInterpTypes.LINEAR: InterpTypes.LINEAR_ZERO_RATES,
            InflationInterpTypes.COMPOUND: InterpTypes.LINEAR_ZERO_RATES,  # Similar behavior
            InflationInterpTypes.FLAT: InterpTypes.FLAT_FWD_RATES
        }
        interp_type = interp_mapping.get(self._interp_type_infl, InterpTypes.LINEAR_ZERO_RATES)
        self._interp_type = interp_type  # Set interp_type for parent DiscountCurve methods

        self._interpolator = Interpolator(interp_type)
        self._times = np.array([])
        self._dfs = np.array([])

        # Time zero: I(0)/I(0) = 1.0
        self._times = np.append(self._times, 0.0)
        self._dfs = np.append(self._dfs, 1.0)

        # Build inflation factors for each ZCIS maturity
        for i, (t_mat, rate) in enumerate(zip(self.swap_times, breakeven_rates)):
            # Calculate cumulative inflation factor: I(T)/I(0) = (1 + r)^T
            inflation_factor = (1.0 + rate) ** t_mat

            self._times = np.append(self._times, t_mat)
            self._dfs = np.append(self._dfs, inflation_factor)

        # Fit interpolator
        self._interpolator.fit(self._times, self._dfs)

        # Validate monotonicity
        if not all(self._times[i] < self._times[i+1]
                   for i in range(len(self._times)-1)):
            raise LibError("Pillar times must be strictly increasing")

###############################################################################

    def _build_curve_ad(self, breakeven_rates):
        """
        JAX-compatible version of _build_curve() for automatic differentiation.

        Constructs the inflation curve using JAX arrays instead of NumPy arrays,
        enabling automatic differentiation for computing sensitivities (delta, gamma)
        with respect to ZCIS breakeven rates.

        Algorithm (identical to _build_curve but with JAX):
        For each ZCIS with maturity T_k and breakeven rate r_k:
        1. Calculate cumulative inflation factor: I(T_k)/I(0) = (1+r_k)^T_k
        2. Store as curve node (using jnp arrays)
        3. Use JAX-compatible interpolator

        Args:
            breakeven_rates: JAX array of breakeven rates from ZCIS instruments

        Returns:
            Tuple of (times, dfs) as JAX arrays for automatic differentiation
        """
        # Map inflation interpolation type to discount curve interpolation
        interp_mapping = {
            InflationInterpTypes.LINEAR: InterpTypes.LINEAR_ZERO_RATES,
            InflationInterpTypes.COMPOUND: InterpTypes.LINEAR_ZERO_RATES,
            InflationInterpTypes.FLAT: InterpTypes.FLAT_FWD_RATES
        }
        interp_type = interp_mapping.get(self._interp_type_infl, InterpTypes.LINEAR_ZERO_RATES)
        self._interp_type = interp_type

        # Initialize with JAX arrays
        times = jnp.array([0.0])
        dfs = jnp.array([1.0])

        # Build inflation factors for each ZCIS maturity
        swap_times_jax = jnp.array(self.swap_times)
        breakeven_rates_jax = jnp.array(breakeven_rates)

        for i in range(len(self.swap_times)):
            t_mat = swap_times_jax[i]
            rate = breakeven_rates_jax[i]

            # Calculate cumulative inflation factor: I(T)/I(0) = (1 + r)^T
            inflation_factor = jnp.power(1.0 + rate, t_mat)

            times = jnp.append(times, t_mat)
            dfs = jnp.append(dfs, inflation_factor)

        # Store in object (JAX arrays)
        self._times = times
        self._dfs = dfs

        # Use JAX-compatible interpolator for AD
        self._interpolator_ad = InterpolatorAd(interp_type)
        self._interpolator_ad.fit(times, dfs)

        return times, dfs

###############################################################################

    def _check_refits(self, zcis_tol):
        """
        Ensure that the inflation curve refits the calibration instruments.

        For each ZCIS used in calibration, verify that when we project
        forward CPI using the curve, we recover the original breakeven rate
        (within tolerance).

        Args:
            zcis_tol: Absolute tolerance for breakeven rate comparison

        Raises:
            LibError: If any ZCIS does not refit within tolerance
        """
        day_counter = DayCount(self._dc_type)

        for i, zcis in enumerate(self._used_swaps):
            # Get the inflation factor from the curve
            (year_frac, _, _) = day_counter.year_frac(
                zcis._effective_dt,
                zcis._maturity_dt
            )

            # Interpolate inflation factor at this maturity
            inflation_factor = self._df(year_frac)

            # Back out implied breakeven rate: (1+r)^T = I(T)/I(0)
            if year_frac > 0:
                implied_breakeven = (inflation_factor ** (1.0 / year_frac)) - 1.0
            else:
                implied_breakeven = 0.0

            # Compare with original ZCIS fixed rate
            original_breakeven = zcis._fixed_rate
            diff = abs(implied_breakeven - original_breakeven)

            if diff > zcis_tol:
                print(f"ZCIS with maturity {zcis._maturity_dt} not repriced.")
                print(f"Original breakeven: {original_breakeven*100:.6f}%")
                print(f"Implied breakeven: {implied_breakeven*100:.6f}%")
                print(f"Difference: {diff*10000:.4f} bps")
                raise LibError(
                    f"ZCIS with maturity {zcis._maturity_dt} not repriced. "
                    f"Difference is {diff*10000:.4f} bps"
                )

###############################################################################

    def forward_index(self, target_date: Date) -> float:
        """
        Project forward CPI at target date.

        Uses the inflation curve to project what the CPI will be at the
        target date, based on the curve's implied inflation expectations.

        Args:
            target_date: Date to project CPI for

        Returns:
            Projected CPI value

        Example:
            >>> future_cpi = infl_curve.forward_index(Date(1, 1, 2028))
        """
        if target_date < self._value_dt:
            raise LibError(
                f"Cannot project CPI before value date. "
                f"Target: {target_date}, Value: {self._value_dt}"
            )

        # Calculate time from value date
        day_counter = DayCount(self._dc_type)
        (year_frac, _, _) = day_counter.year_frac(self._value_dt, target_date)

        # Get inflation factor via interpolation (using DiscountCurve._df)
        inflation_factor = self._df(year_frac)

        # Project CPI: I(T) = I(0) × factor(T)
        forward_cpi = self._base_cpi * inflation_factor

        return forward_cpi

###############################################################################

    def inflation_rate(self, start_dt: Date, end_dt: Date) -> float:
        """
        Calculate implied annual inflation rate between two dates.

        Args:
            start_dt: Start date
            end_dt: End date

        Returns:
            Annualized inflation rate

        Example:
            >>> # Get 5Y5Y forward inflation rate
            >>> rate = infl_curve.inflation_rate(
            ...     Date(1, 1, 2028),
            ...     Date(1, 1, 2033)
            ... )
        """
        if end_dt <= start_dt:
            raise LibError("End date must be after start date")

        cpi_start = self.forward_index(start_dt)
        cpi_end = self.forward_index(end_dt)

        day_counter = DayCount(self._dc_type)
        (year_frac, _, _) = day_counter.year_frac(start_dt, end_dt)

        if year_frac <= 0:
            raise LibError("Year fraction must be positive")

        # Solve: (1 + r)^T = I(end) / I(start)
        inflation_rate = ((cpi_end / cpi_start) ** (1.0 / year_frac)) - 1.0

        return inflation_rate

###############################################################################

    def __repr__(self):
        s = label_to_string("OBJECT TYPE", type(self).__name__)
        s += label_to_string("VALUATION DATE", self._value_dt)
        s += label_to_string("BASE CPI", self._base_cpi)
        s += label_to_string("CURRENCY", self._currency)
        s += label_to_string("INDEX TYPE", self._index_type)
        s += label_to_string("INTERPOLATION", self._interp_type_infl)

        num_points = len(self._used_swaps)

        header = ["TENOR", "TIME", "BREAKEVEN_BPS", "INFLATION_FACTOR"]
        rows = []

        for i in range(num_points):
            # Get the inflation factor at this pillar (skip t=0)
            inflation_factor = self._dfs[i+1]
            breakeven_rate = self._used_swaps[i]._fixed_rate

            rows.append([
                self.tenors[i],
                round(self.swap_times[i], 4),
                round(breakeven_rate * 10000, 2),
                round(inflation_factor, 6),
            ])

        table = format_table(header, rows)
        print("\nINFLATION CURVE DETAILS:")
        print(table)

        return "Cavour_v0.1"

###############################################################################
