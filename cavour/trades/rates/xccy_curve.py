##############################################################################

##############################################################################

"""
Cross-currency discount curve construction via basis swap bootstrapping.

Provides the XccyCurve class for building foreign-in-domestic discount curves
from cross-currency basis swap market quotes. Uses a cashflow-based bootstrapping
approach that solves directly for discount factors without requiring iterative
solvers.

Key features:
- JAX-compatible automatic differentiation for computing sensitivities
- Closed-form bootstrap solution (no iterative solver needed)
- Multiple interpolation schemes (flat forward, linear zero rates, cubic)
- Exact reproduction of input basis swap par conditions (within tolerance)
- Support for various day count conventions and calendars

The bootstrapping algorithm:
1. Builds discount factors sequentially for each basis swap maturity
2. Uses par condition: PV_domestic + spot_FX * PV_foreign_in_domestic = 0
3. Solves directly: P_f_d(T_k) = -(PV_d + S0 * PV_f_known) / (S0 * CF_last_f)
4. Stores intermediate discount factors for dense interpolation grid

Theory:
- Constructs P_f_d(T): discount factors for foreign cashflows under domestic collateral
- Given:
  - P_d(T): domestic OIS discount curve (domestic in domestic collateral)
  - P_f_f(T): foreign OIS discount curve (foreign in foreign collateral)
  - Basis spreads: cross-currency basis by tenor
  - Spot FX: S0 (domestic per unit of foreign)
- For each basis tenor, builds an XCCY basis swap with:
  - Domestic floating leg (no spread or with spread)
  - Foreign floating leg (with basis spread)
  - Notional exchanges at start and maturity
- Par condition: Total PV in domestic = 0

Example:
    >>> # Create XCCY basis swaps at market basis spreads
    >>> basis_swaps = [
    ...     XccyBasisSwap(value_dt, "1Y", 1.0, 1.27, 0.0, 0.0025, ...),
    ...     XccyBasisSwap(value_dt, "2Y", 1.0, 1.27, 0.0, 0.0030, ...),
    ...     XccyBasisSwap(value_dt, "5Y", 1.0, 1.27, 0.0, 0.0035, ...)
    ... ]
    >>>
    >>> # Bootstrap XCCY curve
    >>> xccy_curve = XccyCurve(
    ...     value_dt=value_dt,
    ...     basis_swaps=basis_swaps,
    ...     domestic_curve=gbp_ois_curve,
    ...     foreign_curve=usd_ois_curve,
    ...     spot_fx=1.27,  # GBP per USD
    ...     interp_type=InterpTypes.FLAT_FWD_RATES,
    ...     check_refit=True
    ... )
    >>>
    >>> # Query discount factors for foreign cashflows in domestic collateral
    >>> df_1y = xccy_curve.df(value_dt.add_years(1))
"""

import numpy as np
import jax.numpy as jnp
import jax

from cavour.utils.error import LibError
from cavour.utils.date import Date
from cavour.utils.day_count import DayCount, DayCountTypes
from cavour.utils.helpers import (check_argument_types,
                              _func_name,
                              label_to_string,
                              format_table)
from cavour.utils.global_types import SwapTypes
from cavour.market.curves.interpolator import InterpTypes
from cavour.market.curves.discount_curve import DiscountCurve

SWAP_TOL = 1e-10

jax.config.update("jax_enable_x64", True)


class XccyCurve(DiscountCurve):
    """
    Constructs a cross-currency discount curve from basis swap prices.

    Builds the foreign-in-domestic discount curve P_f_d(T), which represents
    discount factors for foreign currency cashflows when collateralized in
    domestic currency.

    The curve is calibrated to cross-currency basis swaps, ensuring that each
    calibration instrument has PV = 0 in domestic currency (within tolerance).

    Bootstrap approach:
    - For each basis swap pillar with maturity T_k and spread B_k:
      1. Value domestic leg using domestic OIS curve
      2. Split foreign leg cashflows into:
         - Known part: dates < T_k (already bootstrapped)
         - Unknown part: cashflows at T_k
      3. Solve closed-form for P_f_d(T_k) from par condition
      4. Store node and continue to next pillar

    The construction depends on:
    - Domestic OIS curve (for domestic leg valuation)
    - Foreign OIS curve (for foreign leg forward rate projection)
    - Spot FX rate (for currency conversion)
    - Basis swap conventions (must match market exactly)
    """

    def __init__(self,
                 value_dt: Date,
                 basis_swaps: list,
                 domestic_curve: DiscountCurve,
                 foreign_curve: DiscountCurve,
                 spot_fx: float,
                 interp_type: InterpTypes = InterpTypes.FLAT_FWD_RATES,
                 check_refit: bool = False):
        """
        Create a cross-currency discount curve from basis swaps.

        Args:
            value_dt: Valuation date (anchor date for the curve)
            basis_swaps: List of XccyBasisSwap instruments for calibration
            domestic_curve: Domestic OIS discount curve (domestic in domestic collateral)
            foreign_curve: Foreign OIS discount curve (foreign in foreign collateral)
            spot_fx: Spot FX rate (domestic currency per unit of foreign)
            interp_type: Interpolation method for discount factors
            check_refit: If True, verify calibration swaps reprice to zero

        Notes:
            - All basis swaps must have same effective date (XCCY spot date)
            - Swaps will be sorted by maturity during construction
            - The curve will assign a discount factor of 1.0 to the valuation date
        """
        check_argument_types(getattr(self, _func_name(), None), locals())

        self._value_dt = value_dt
        self._used_swaps = basis_swaps
        self._domestic_curve = domestic_curve
        self._foreign_curve = foreign_curve
        self._spot_fx = spot_fx
        self._interp_type = interp_type
        self._check_refit = check_refit
        self._interpolator = None

        # Sort swaps by maturity
        self._used_swaps = sorted(self._used_swaps,
                                   key=lambda x: x._maturity_dt)

        # Prepare inputs and build curve
        self._prepare_curve_builder_inputs()
        self._build_curve()

###############################################################################

    def _prepare_curve_builder_inputs(self):
        """
        Prepare inputs for curve construction.

        Extracts basis spreads, maturities, and day count conventions from
        the calibration basis swaps. Initializes curve data structures.
        """
        # Use ACT/365F for curve time calculations (consistent with DiscountCurve/gDaysInYear)
        # This ensures that df() queries return correct values
        from cavour.utils.global_vars import gDaysInYear
        self._dc_type = DayCountTypes.ACT_365F

        self._times = jnp.array([])
        self._dfs = jnp.array([])
        self._repr_dfs = jnp.array([])

        # Time zero is now
        df_mat = 1.0
        self._times = jnp.append(self._times, 0.0)
        self._dfs = jnp.append(self._dfs, df_mat)
        self._repr_dfs = jnp.append(self._repr_dfs, df_mat)

        self.basis_spreads = []
        self.swap_times = []

        for swap in self._used_swaps:
            # Basis spread is on foreign leg
            basis_spread = swap._foreign_spread
            maturity_dt = swap._maturity_dt
            # Use gDaysInYear (365) for consistency with DiscountCurve
            tswap = (maturity_dt - self._value_dt) / gDaysInYear

            self.swap_times.append(tswap)
            self.basis_spreads.append(basis_spread)

###############################################################################

    def _build_curve(self):
        """
        Bootstrap the XCCY discount curve using closed-form solution.

        Algorithm:
        For each basis swap pillar k with maturity T_k:
        1. Build domestic and foreign leg schedules
        2. Compute domestic leg PV using domestic curve
        3. Compute foreign leg PV split into:
           - PV_f_known: cashflows at dates < T_k (use existing XCCY curve nodes)
           - CF_last_f: aggregate cashflow at T_k
        4. Solve for P_f_d(T_k) from par condition:
           P_f_d(T_k) = -(PV_d + S0 * PV_f_known) / (S0 * CF_last_f)
        5. Store P_f_d(T_k) as new curve node

        No iterative solver needed - each node is determined by a closed-form
        algebraic solution.
        """
        # Build a temporary XCCY curve for intermediate calculations
        # Store times and DFs directly to avoid DiscountCurve round-trip conversion errors
        from cavour.market.curves.interpolator import Interpolator

        xccy_nodes_times = []
        xccy_nodes_dfs = []

        for i, swap in enumerate(self._used_swaps):
            t_mat = self.swap_times[i]
            maturity_dt = swap._maturity_dt

            # Value domestic leg using domestic curve
            pv_domestic = swap._domestic_leg.value(
                value_dt=self._value_dt,
                discount_curve=self._domestic_curve,
                index_curve=self._domestic_curve,
                first_fixing_rate=None
            )

            # Before building temp curve, identify any intermediate payment dates
            # that fall between the last node and current maturity
            intermediate_dates = []
            if i > 0:
                last_node_time = xccy_nodes_times[-1]
                for pmnt_dt in swap._foreign_leg._payment_dts:
                    pmnt_time = (pmnt_dt - self._value_dt) / 365.0  # ACT_365F
                    if last_node_time < pmnt_time < t_mat:
                        intermediate_dates.append((pmnt_dt, pmnt_time))

            # Build temporary XCCY curve from current nodes
            # For ALL pillars (including first), create consistent temp curve
            if i == 0:
                # For first pillar, create a single-node curve at t=0, DF=1.0
                # We'll use foreign_curve for all other discount queries, but this
                # ensures consistent ACT_365F day count handling
                temp_times = np.array([0.0])
                temp_dfs = np.array([1.0])

                # Manually construct DiscountCurve
                temp_xccy_curve = DiscountCurve.__new__(DiscountCurve)
                temp_xccy_curve._value_dt = self._value_dt
                temp_xccy_curve._times = temp_times
                temp_xccy_curve._dfs = temp_dfs
                temp_xccy_curve._interp_type = self._interp_type
                temp_xccy_curve._dc_type = DayCountTypes.ACT_365F

                # Manually create and fit interpolator
                temp_interpolator = Interpolator(self._interp_type)
                temp_interpolator.fit(temp_times, temp_dfs)
                temp_xccy_curve._interpolator = temp_interpolator

                # For first pillar, also add foreign curve's DFs to approximate
                # We'll query foreign_curve and copy its structure with ACT_365F override
                # Actually, for the first pillar we need to query foreign curve for DFs at payment dates
                # Let's use a simpler approach: just use flat DF=1.0 since we're solving for the first DF anyway

                # Override df() to query foreign_curve with dates (not times)
                # This allows foreign_curve to use its own day count for time calculations
                def temp_df_override_first(self, dt, day_count=None):
                    # Query foreign curve with dates, letting it use its own day count
                    return xccy_curve_self._foreign_curve.df(dt, xccy_curve_self._foreign_curve._dc_type)

                # Store reference to self for closure
                xccy_curve_self = self

                # Bind the override method to the instance
                import types
                temp_xccy_curve.df = types.MethodType(temp_df_override_first, temp_xccy_curve)
            else:
                # For subsequent pillars, manually construct temp curve to avoid
                # DiscountCurve round-trip conversion error (year_frac -> date -> year_frac)
                # which corrupts the times and introduces accumulating errors

                # Create times and DFs arrays, including intermediate nodes
                temp_times_list = [0.0] + xccy_nodes_times.copy()
                temp_dfs_list = [1.0] + xccy_nodes_dfs.copy()

                # Add intermediate nodes using extrapolation if any exist
                if intermediate_dates:
                    # Use the interpolate function to extrapolate DFs
                    from cavour.market.curves.interpolator import interpolate

                    existing_times = np.array([0.0] + xccy_nodes_times)
                    existing_dfs = np.array([1.0] + xccy_nodes_dfs)

                    # Extrapolate DFs for intermediate dates
                    for interm_dt, interm_time in intermediate_dates:
                        interm_df = interpolate(
                            interm_time,
                            existing_times,
                            existing_dfs,
                            self._interp_type.value
                        )
                        temp_times_list.append(interm_time)
                        temp_dfs_list.append(float(interm_df))

                    # Sort by time to maintain monotonicity
                    sorted_indices = np.argsort(temp_times_list)
                    temp_times_list = [temp_times_list[i] for i in sorted_indices]
                    temp_dfs_list = [temp_dfs_list[i] for i in sorted_indices]

                # Convert to numpy arrays
                temp_times = np.array(temp_times_list)
                temp_dfs = np.array(temp_dfs_list)

                # Manually construct DiscountCurve to preserve exact times
                temp_xccy_curve = DiscountCurve.__new__(DiscountCurve)
                temp_xccy_curve._value_dt = self._value_dt
                temp_xccy_curve._times = temp_times  # Keep as numpy array for interpolator
                temp_xccy_curve._dfs = temp_dfs      # Keep as numpy array for interpolator
                temp_xccy_curve._interp_type = self._interp_type
                temp_xccy_curve._dc_type = DayCountTypes.ACT_365F

                # Manually create and fit interpolator
                temp_interpolator = Interpolator(self._interp_type)
                temp_interpolator.fit(temp_times, temp_dfs)
                temp_xccy_curve._interpolator = temp_interpolator

                # Override df() method to always use ACT_365F (same as XccyCurve.df())
                # This prevents day count mismatch when foreign leg queries with ACT_360
                def temp_df_override(self, dt, day_count=None):
                    from cavour.utils.helpers import times_from_dates
                    times = times_from_dates(dt, self._value_dt, DayCountTypes.ACT_365F)
                    dfs = self._df(times)
                    if isinstance(dfs, float):
                        return dfs
                    else:
                        return np.array(dfs)

                # Bind the override method to the instance
                import types
                temp_xccy_curve.df = types.MethodType(temp_df_override, temp_xccy_curve)

            # For foreign leg, we need to split PV into known and unknown parts
            # The foreign leg is PAY type, so value() will negate the final PV
            foreign_leg = swap._foreign_leg
            maturity_dt = swap._maturity_dt

            # Value foreign leg with temp XCCY curve to get cashflows and DFs
            _ = foreign_leg.value(
                value_dt=self._value_dt,
                discount_curve=temp_xccy_curve,
                index_curve=self._foreign_curve,
                first_fixing_rate=None
            )

            # Now we need to split this into:
            # 1. PV_known: PV of cashflows before maturity (already discounted and signed)
            # 2. CF_last: undiscounted cashflows at maturity (before PAY/RECEIVE sign)
            #
            # The tricky part: the leg.value() method applies the PAY/RECEIVE sign at the END
            # So we need to work with the internal _payment_pvs before the final sign flip

            pv_foreign_known = 0.0
            cf_last_foreign = 0.0

            # Loop through the stored PV values
            for j, pmnt_dt in enumerate(foreign_leg._payment_dts):
                if pmnt_dt > self._value_dt:  # Exclude payments AT value_dt (already have DF=1.0)
                    pmnt_amount = foreign_leg._payments[j]

                    if pmnt_dt == maturity_dt:
                        # This is at maturity - we'll solve for its DF
                        # Use the undiscounted cashflow amount
                        cf_last_foreign += pmnt_amount
                    else:
                        # This is before maturity - use the already-computed PV
                        # The _payment_pvs already have the cashflow sign (positive or negative)
                        # but NOT the final PAY/RECEIVE leg type sign flip
                        pv_foreign_known += foreign_leg._payment_pvs[j]
                elif pmnt_dt == self._value_dt:
                    # Initial notional exchange at value_dt - include in known PV with DF=1.0
                    pv_foreign_known += foreign_leg._payments[j]

            # The PAY/RECEIVE sign flip is applied at the END by the leg.value() method
            # So we need to apply it here too
            if foreign_leg._leg_type == SwapTypes.PAY:
                pv_foreign_known = pv_foreign_known * (-1.0)
                cf_last_foreign = cf_last_foreign * (-1.0)

            # Apply par condition to solve for P_f_d(T_k)
            # Par condition: PV_domestic + spot_fx * PV_foreign_total = 0
            # where PV_foreign_total = PV_foreign_known + cf_last_foreign * P_f_d(T_k)
            #
            # Solving: PV_domestic + spot_fx * (PV_foreign_known + cf_last_foreign * P_f_d(T_k)) = 0
            # P_f_d(T_k) = -(PV_domestic + spot_fx * PV_foreign_known) / (spot_fx * cf_last_foreign)

            if abs(cf_last_foreign) < 1e-12:
                raise LibError(f"No cashflow at maturity for swap {i}. Cannot bootstrap.")

            df_mat = -(pv_domestic + self._spot_fx * pv_foreign_known) / (self._spot_fx * cf_last_foreign)

            # Store intermediate nodes first (if any) to maintain chronological order
            for interm_dt, interm_time in intermediate_dates:
                # Extrapolate DF for intermediate node using existing nodes
                from cavour.market.curves.interpolator import interpolate
                interm_df = interpolate(
                    interm_time,
                    np.array([0.0] + xccy_nodes_times),
                    np.array([1.0] + xccy_nodes_dfs),
                    self._interp_type.value
                )

                # Store intermediate node
                xccy_nodes_times.append(interm_time)
                xccy_nodes_dfs.append(float(interm_df))

                # Update curve arrays
                self._times = jnp.append(self._times, interm_time)
                self._dfs = jnp.append(self._dfs, float(interm_df))
                self._repr_dfs = jnp.append(self._repr_dfs, float(interm_df))

            # Now store the new pillar node using exact times
            xccy_nodes_times.append(t_mat)
            xccy_nodes_dfs.append(df_mat)

            # Update curve arrays for representation
            self._times = jnp.append(self._times, t_mat)
            self._dfs = jnp.append(self._dfs, df_mat)
            self._repr_dfs = jnp.append(self._repr_dfs, df_mat)

        # Initialize the interpolator with the final times and DFs
        from cavour.market.curves.interpolator import Interpolator
        self._interpolator = Interpolator(self._interp_type)
        self._interpolator.fit(self._times, self._dfs)

        # Validate calibration if requested
        if self._check_refit:
            self._check_refits(SWAP_TOL)

###############################################################################

    def df(self, dt, day_count=None):
        """
        Override parent df() method to ensure consistent day count usage.

        XccyCurve uses ACT/365F (gDaysInYear) for all time calculations,
        so we must use the same when querying discount factors.

        Args:
            dt: Date or list of dates to get discount factors for
            day_count: Day count convention (ignored - always uses ACT/365F)

        Returns:
            Discount factor(s) at the given date(s)
        """
        from cavour.utils.helpers import times_from_dates

        # Always use ACT/365F to match how we calculate _times internally
        # Ignore the day_count parameter to ensure consistency
        times = times_from_dates(dt, self._value_dt, DayCountTypes.ACT_365F)
        dfs = self._df(times)

        if isinstance(dfs, float):
            return dfs
        else:
            return np.array(dfs)

###############################################################################

    def _check_refits(self, swap_tol):
        """
        Ensure that the XCCY curve refits the calibration instruments.

        For each basis swap used in calibration, reprice it using the
        constructed XCCY curve and verify that PV is approximately zero
        in domestic currency.

        Args:
            swap_tol: Absolute tolerance for PV (should be close to zero)

        Raises:
            LibError: If any swap does not reprice within tolerance
        """
        for swap in self._used_swaps:
            # Value using the constructed curve (self acts as XCCY curve)
            v = swap.value(
                value_dt=self._value_dt,
                domestic_discount_curve=self._domestic_curve,
                foreign_discount_curve=self._foreign_curve,
                xccy_discount_curve=self,
                spot_fx=self._spot_fx
            )

            # Normalize by notional for comparison
            v_normalized = v / swap._domestic_notional

            if abs(v_normalized) > swap_tol:
                print(f"XCCY Swap with maturity {swap._maturity_dt} not repriced.")
                print(f"PV = {v}, Normalized PV = {v_normalized}")
                swap.print_valuation()
                raise LibError(
                    f"XCCY swap with maturity {swap._maturity_dt} not repriced. "
                    f"Difference is {abs(v_normalized)}"
                )

###############################################################################

    def __repr__(self):
        s = label_to_string("OBJECT TYPE", type(self).__name__)
        s += label_to_string("VALUATION DATE", self._value_dt)
        s += label_to_string("SPOT FX", self._spot_fx)
        s += label_to_string("INTERPOLATION", self._interp_type)

        num_points = len(self.basis_spreads)

        header = ["TENORS", "TIME", "BASIS_SPREAD_BPS", "DFs"]
        rows = []

        for i in range(0, num_points):
            rows.append([
                self._used_swaps[i]._termination_dt,
                round(self.swap_times[i], 4),
                round(self.basis_spreads[i] * 10000, 2),
                round(self._repr_dfs[i+1], 6),
            ])

        table = format_table(header, rows)
        print("\nXCCY CURVE DETAILS:")
        print(table)

        return "Cavour_v0.1"

###############################################################################
