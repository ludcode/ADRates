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
                 check_refit: bool = False,
                 use_ad: bool = False):
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
            use_ad: If True, use JAX-compatible _build_curve_ad() for automatic differentiation

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
        self._use_ad = use_ad
        self._interpolator = None

        # Sort swaps by maturity
        self._used_swaps = sorted(self._used_swaps,
                                   key=lambda x: x._maturity_dt)

        # Prepare inputs and build curve
        self._prepare_curve_builder_inputs()

        # Use AD or regular build method
        if use_ad:
            self._build_curve_ad()
        else:
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
            # Store as (date, time) tuples - DFs will be computed with flat forward basis
            intermediate_dates = []
            intermediate_dates_with_dfs = []  # Will store (date, time, df) after computation

            if i > 0:
                last_node_time = xccy_nodes_times[-1]
                for pmnt_dt in swap._foreign_leg._payment_dts:
                    pmnt_time = (pmnt_dt - self._value_dt) / 365.0  # ACT_365F
                    if last_node_time < pmnt_time < t_mat:
                        intermediate_dates.append((pmnt_dt, pmnt_time))

            # Build temporary XCCY curve from current nodes
            # Apply FLAT FORWARD BASIS ASSUMPTION for intermediate dates
            basis_spread = self.basis_spreads[i]  # Current pillar's basis spread

            if i == 0:
                # For first pillar, create a single-node curve at t=0, DF=1.0
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

                # FLAT FORWARD BASIS ASSUMPTION:
                # Apply the current pillar's basis spread to all intermediate dates
                # Formula: DF_xccy(t) = DF_ois(t) * exp(-basis * t)
                # This assumes constant forward basis spread from 0 to maturity
                def temp_df_override_first(self, dt, day_count=None):
                    from cavour.utils.helpers import times_from_dates

                    # Get time from value date using ACT_365F (consistent with curve)
                    t = times_from_dates(dt, xccy_curve_self._value_dt, DayCountTypes.ACT_365F)

                    # Get foreign OIS DF (uses its own day count internally)
                    df_ois = xccy_curve_self._foreign_curve.df(dt, xccy_curve_self._foreign_curve._dc_type)

                    # Apply flat forward basis adjustment
                    # Scalar or array handling
                    if isinstance(t, (list, np.ndarray)):
                        t_arr = np.array(t)
                        df_ois_arr = np.array(df_ois) if not isinstance(df_ois, np.ndarray) else df_ois
                        df_xccy = df_ois_arr * np.exp(-basis_spread_closure * t_arr)
                    else:
                        df_xccy = df_ois * np.exp(-basis_spread_closure * float(t))

                    return df_xccy

                # Store references for closure
                xccy_curve_self = self
                basis_spread_closure = basis_spread

                # Bind the override method to the instance
                import types
                temp_xccy_curve.df = types.MethodType(temp_df_override_first, temp_xccy_curve)
            else:
                # For subsequent pillars, apply flat forward basis for intermediate dates
                # Create times and DFs arrays with existing nodes
                temp_times_list = [0.0] + xccy_nodes_times.copy()
                temp_dfs_list = [1.0] + xccy_nodes_dfs.copy()

                # FLAT FORWARD BASIS for intermediate nodes
                # For dates between last pillar and current maturity:
                # DF_xccy(t) = DF_xccy(t_prev) * [DF_ois(t) / DF_ois(t_prev)] * exp(-basis * (t - t_prev))
                if intermediate_dates:
                    last_node_time = xccy_nodes_times[-1]
                    last_node_df = xccy_nodes_dfs[-1]

                    # Get the actual last pillar date from the previous swap's maturity
                    last_pillar_dt = self._used_swaps[i-1]._maturity_dt

                    # Compute DF_ois at last pillar
                    df_ois_prev = self._foreign_curve.df(last_pillar_dt, self._foreign_curve._dc_type)

                    for interm_dt, interm_time in intermediate_dates:
                        # Get DF_ois at intermediate date
                        df_ois_t = self._foreign_curve.df(interm_dt, self._foreign_curve._dc_type)

                        # Apply flat forward basis formula
                        # DF_xccy(t) = DF_xccy(t_prev) * [DF_ois(t) / DF_ois(t_prev)] * exp(-basis * (t - t_prev))
                        dt_delta = interm_time - last_node_time
                        interm_df = last_node_df * (df_ois_t / df_ois_prev) * np.exp(-basis_spread * dt_delta)

                        temp_times_list.append(interm_time)
                        temp_dfs_list.append(float(interm_df))

                        # Store the DF for later (to be added as a node in the final curve)
                        intermediate_dates_with_dfs.append((interm_dt, interm_time, float(interm_df)))

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
            # IMPORTANT: Recompute cashflows dynamically (like _build_curve_ad) instead of
            # using pre-computed values from foreign_leg._payments which include spread incorrectly
            foreign_leg = swap._foreign_leg
            maturity_dt = swap._maturity_dt

            # First, call value() to populate the payment schedule (including notional exchanges)
            # We'll discard the PVs but use the payment dates
            _ = foreign_leg.value(
                value_dt=self._value_dt,
                discount_curve=temp_xccy_curve,
                index_curve=self._foreign_curve,
                first_fixing_rate=None
            )

            # Extract payment schedule data from foreign leg (now populated)
            # This allows us to recompute cashflows dynamically with correct spread handling
            payment_dts = foreign_leg._payment_dts
            start_accrual_dts = foreign_leg._start_accrued_dts
            end_accrual_dts = foreign_leg._end_accrued_dts
            year_fracs = foreign_leg._year_fracs
            notionals = foreign_leg._notional_array

            # Get the constant notional for XCCY swaps (all payments use same notional)
            leg_notional = swap._foreign_notional

            # Determine which payments are notional exchanges vs interest payments
            # Notional exchanges have year_frac ~ 0
            from cavour.utils.helpers import times_from_dates

            pv_foreign_known = 0.0
            cf_last_foreign = 0.0

            # Loop through payments and recompute cashflows dynamically
            for j, pmnt_dt in enumerate(payment_dts):
                year_frac = year_fracs[j]
                # Use indexed notional if available, otherwise use leg notional
                notional = notionals[j] if j < len(notionals) else leg_notional
                is_notional_exchange = (year_frac < 1e-10)
                is_last_payment = (j == len(payment_dts) - 1)

                # Compute cashflow dynamically
                if is_notional_exchange:
                    # Notional exchange
                    if is_last_payment:
                        cashflow = notional  # Return notional at maturity
                    else:
                        cashflow = -notional  # Receive notional at start
                else:
                    # Interest payment: recompute from forward rates
                    start_dt = start_accrual_dts[j]
                    end_dt = end_accrual_dts[j]

                    # Get foreign OIS DFs at accrual dates
                    df_start = self._foreign_curve.df(start_dt, self._foreign_curve._dc_type)
                    df_end = self._foreign_curve.df(end_dt, self._foreign_curve._dc_type)

                    # Compute forward rate (no spread)
                    fwd_rate = (df_start / df_end - 1.0) / year_frac if year_frac > 1e-10 else 0.0

                    # Base interest (no spread)
                    base_interest = fwd_rate * year_frac * notional

                    # Add basis spread separately (this is the key fix!)
                    spread_component = basis_spread * year_frac * notional
                    cashflow = base_interest + spread_component

                    # If last payment, add notional return
                    if is_last_payment:
                        cashflow += notional

                # Now discount cashflow or accumulate for maturity
                if pmnt_dt > self._value_dt:
                    if pmnt_dt == maturity_dt:
                        # This is at maturity - we'll solve for its DF
                        cf_last_foreign += cashflow
                    else:
                        # This is before maturity - discount with temp XCCY curve
                        df_xccy = temp_xccy_curve.df(pmnt_dt)
                        pv_foreign_known += cashflow * df_xccy
                elif pmnt_dt == self._value_dt:
                    # Initial exchange at value_dt - DF = 1.0
                    pv_foreign_known += cashflow * 1.0

            # The PAY/RECEIVE sign flip
            # For PAY leg, cashflows are negated
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
            # For i=0, compute intermediate DFs using flat forward basis
            # For i>0, use the DFs already computed and stored in intermediate_dfs_dict
            if i == 0:
                # For first pillar, identify and store DFs for all payment dates before maturity
                for j, pmnt_dt in enumerate(foreign_leg._payment_dts):
                    if self._value_dt < pmnt_dt < maturity_dt:
                        # Compute DF using flat forward basis
                        from cavour.utils.helpers import times_from_dates
                        pmnt_time = float(times_from_dates(pmnt_dt, self._value_dt, DayCountTypes.ACT_365F))
                        df_ois = self._foreign_curve.df(pmnt_dt, self._foreign_curve._dc_type)
                        interm_df = df_ois * np.exp(-basis_spread * pmnt_time)

                        # Store intermediate node
                        xccy_nodes_times.append(pmnt_time)
                        xccy_nodes_dfs.append(float(interm_df))

                        # Update curve arrays
                        self._times = jnp.append(self._times, pmnt_time)
                        self._dfs = jnp.append(self._dfs, float(interm_df))
                        self._repr_dfs = jnp.append(self._repr_dfs, float(interm_df))
            else:
                # For subsequent pillars, use the pre-computed DFs
                for interm_dt, interm_time, interm_df in intermediate_dates_with_dfs:
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

    def _build_curve_ad(self):
        """
        Bootstrap the XCCY discount curve using JAX-compatible operations.

        This method provides the same functionality as _build_curve() but uses
        pure JAX operations for automatic differentiation. The algorithm is
        identical, but implemented using jax.lax.scan instead of Python loops.

        Returns exact same times and DFs as _build_curve() to machine precision.

        Also computes and stores Jacobians and Hessians for sensitivity analysis:
        - _jac_basis: d(xccy_dfs) / d(basis_spreads)
        - _jac_foreign_curve_dfs: d(xccy_dfs) / d(foreign_curve_dfs)
        - _hess_basis: d²(xccy_dfs) / d(basis_spreads)²
        - _mixed_hess_foreign_basis: d²(xccy_dfs) / d(foreign_curve_dfs) d(basis_spreads)
        """
        from cavour.utils.helpers import times_from_dates
        from jax import lax, jacrev

        # Step 1: Pre-extract ALL payment data from ALL swaps
        # This is done outside JAX since it involves object manipulation
        payment_data = self._prepare_ad_inputs()

        # Cache payment_data for later use (e.g., aggregating Jacobian sensitivities)
        self._payment_data_cache = payment_data

        # Step 2: Run JAX bootstrap using lax.scan
        times, dfs = self._run_jax_bootstrap(payment_data)

        # Step 3: Update curve arrays
        self._times = times
        self._dfs = dfs
        self._repr_dfs = dfs

        # Step 4: Compute Jacobians for automatic differentiation
        # Extract current parameter values
        df_foreign_ois = payment_data['df_foreign_ois']

        # 4a: Jacobian w.r.t. basis spreads: d(xccy_dfs) / d(basis_spreads)
        # We want sensitivity to PILLAR-level spreads, not payment-level
        # Extract pillar-level spreads (one per swap)
        swap_idx_array = payment_data['swap_idx']
        n_swaps = payment_data['n_swaps']

        # Create pillar-level basis spread array (one per swap)
        basis_spreads_pillar = jnp.array([self._used_swaps[i]._foreign_spread for i in range(n_swaps)])

        def dfs_from_basis_pillar(pillar_spreads):
            """Rebuild curve with different pillar-level basis spreads.

            Args:
                pillar_spreads: Array of shape (n_swaps,) with one spread per swap/pillar

            Returns:
                DFs after bootstrapping with the new spreads
            """
            # Expand pillar-level spreads to payment-level
            # payment_spreads[i] = pillar_spreads[swap_idx[i]]
            payment_spreads = pillar_spreads[swap_idx_array]

            modified_data = dict(payment_data)
            modified_data['basis_spreads'] = payment_spreads
            _, dfs_out = self._run_jax_bootstrap(modified_data)
            return dfs_out

        self._jac_basis = jacrev(dfs_from_basis_pillar)(basis_spreads_pillar)

        # 4a-ii: Hessian w.r.t. basis spreads: d²(xccy_dfs) / d(basis_spreads)²
        # This is needed for computing gamma (second-order sensitivities)
        #
        # Phase 2: Using JAX automatic differentiation (fast!)
        # jacfwd(jacrev(...)) computes Hessian for vector-valued functions
        # This is equivalent to forward-over-reverse mode AD
        from jax import jacfwd
        hess_basis_func = jacfwd(jacrev(dfs_from_basis_pillar))
        hess_basis = hess_basis_func(basis_spreads_pillar)
        # Result shape: [n_xccy_dfs, n_spreads, n_spreads]
        self._hess_basis = hess_basis

        # 4b: Mixed Hessian w.r.t. foreign OIS CURVE DFs and basis spreads
        # d2(xccy_dfs) / d(foreign_curve_dfs) d(basis_spreads)
        # This is needed for computing cross-gamma between foreign OIS and XCCY basis
        #
        # BREAKTHROUGH: The key to making JAX mixed Hessians work:
        # - Use jacrev(jacfwd(...)) NOT jacfwd(jacrev(...))
        # - Differentiate w.r.t. CURVE DFs (not payment DFs) to match engine.py expectations
        # - Use lax.stop_gradient() on all static arrays
        # - Result shape: [n_xccy, n_foreign_curve, n_basis] → transpose to [n_xccy, n_basis, n_foreign_curve]

        from jax import jacrev, jacfwd
        from jax import lax

        # Get foreign curve times and DFs
        foreign_curve_times = jnp.array(self._foreign_curve._times)
        foreign_curve_dfs = jnp.array(self._foreign_curve._dfs)

        # Static JAX arrays (explicitly marked as non-differentiable)
        foreign_curve_times_static = lax.stop_gradient(foreign_curve_times)
        payment_times_jax = lax.stop_gradient(jnp.array(payment_data['times']))
        swap_idx_jax = lax.stop_gradient(jnp.array(payment_data['swap_idx'], dtype=jnp.int32))

        # Extract static scalars from payment_data (not arrays)
        payment_data_scalars = {
            k: v for k, v in payment_data.items()
            if k not in ['df_foreign_ois', 'basis_spreads', 'times', 'swap_idx']
        }

        # Create function with ONLY the two differentiable inputs as arguments
        def compute_xccy_from_foreign_curve(basis_spreads_arg, foreign_curve_dfs_arg):
            """
            Compute XCCY DFs from basis spreads and foreign curve DFs.

            Args:
                basis_spreads_arg: [n_basis] - pillar basis spreads (DIFFERENTIABLE)
                foreign_curve_dfs_arg: [n_foreign_curve] - foreign curve DFs (DIFFERENTIABLE)

            Returns:
                xccy_dfs: [n_xccy] - bootstrapped XCCY discount factors
            """
            # Interpolate foreign curve DFs to payment times
            log_curve_dfs = jnp.log(foreign_curve_dfs_arg)
            log_payment_dfs = jnp.interp(payment_times_jax, foreign_curve_times_static, log_curve_dfs)
            payment_dfs_foreign = jnp.exp(log_payment_dfs)

            # Expand basis spreads from pillar to payment level
            payment_spreads = basis_spreads_arg[swap_idx_jax]

            # Rebuild payment_data with modified values
            modified_data = dict(payment_data_scalars)
            modified_data['df_foreign_ois'] = payment_dfs_foreign
            modified_data['basis_spreads'] = payment_spreads
            modified_data['times'] = payment_times_jax
            modified_data['swap_idx'] = swap_idx_jax

            # Bootstrap XCCY curve
            _, xccy_dfs = self._run_jax_bootstrap_impl(modified_data)
            return xccy_dfs

        # 4c: Jacobian w.r.t. foreign curve DFs: d(xccy_dfs) / d(foreign_curve_dfs)
        # Use the same function as mixed Hessian but differentiate only w.r.t. foreign DFs
        # Chained with d(foreign_dfs)/d(foreign_rates) in engine.py for foreign OIS sensitivities
        self._jac_foreign_curve_dfs = jacrev(compute_xccy_from_foreign_curve, argnums=1)(basis_spreads_pillar, foreign_curve_dfs)
        # Shape: [n_xccy_dfs, n_foreign_dfs]

        # Compute mixed Hessian: d²(xccy_dfs) / d(basis) d(foreign_curve)
        # KEY: Use jacrev(jacfwd(...)) not jacfwd(jacrev(...))!
        # jacrev(jacfwd(f, argnums=1), argnums=0) gives [n_xccy, n_foreign_curve, n_basis]
        mixed_hess_func = jacrev(jacfwd(compute_xccy_from_foreign_curve, argnums=1), argnums=0)
        mixed_hess_raw = mixed_hess_func(basis_spreads_pillar, foreign_curve_dfs)

        # Verify dimensions
        # Expected shape: [n_xccy, n_foreign_curve, n_basis]
        n_xccy_result = mixed_hess_raw.shape[0]
        n_foreign_result = mixed_hess_raw.shape[1]
        n_basis_result = mixed_hess_raw.shape[2]

        if n_foreign_result == len(foreign_curve_dfs) and n_basis_result == len(basis_spreads_pillar):
            # Already in the correct format [n_xccy, n_foreign_curve, n_basis]
            # But we need [n_xccy, n_basis, n_foreign_curve] for engine.py
            self._mixed_hess_foreign_basis = jnp.transpose(mixed_hess_raw, (0, 2, 1))
        else:
            print(f"WARNING: Mixed Hessian dimensions incorrect!")
            print(f"  Got: [{n_xccy_result}, {n_foreign_result}, {n_basis_result}]")
            print(f"  Expected: [?, {len(foreign_curve_dfs)}, {len(basis_spreads_pillar)}]")
            self._mixed_hess_foreign_basis = None


        # Step 5: Initialize the interpolator with the final times and DFs
        from cavour.market.curves.interpolator import Interpolator
        self._interpolator = Interpolator(self._interp_type)
        self._interpolator.fit(np.array(self._times), np.array(self._dfs))

        # Step 6: Validate calibration if requested
        if self._check_refit:
            self._check_refits(SWAP_TOL)

###############################################################################

    def _prepare_ad_inputs(self):
        """
        Pre-extract all payment data and build static structures for JAX bootstrap.

        Returns a dictionary containing all data needed for the JAX scan operation:
        - All payment times from all swaps (both domestic and foreign legs)
        - Pre-computed OIS discount factors at all payment dates
        - Cashflow amounts (forward rates pre-computed)
        - Dependency indices (which prior XCCY DF to use)
        - Masks for conditional logic
        """
        all_points = []

        # For each basis swap, extract ALL payment data from BOTH legs
        for swap_idx, swap in enumerate(self._used_swaps):
            basis_spread = swap._foreign_spread
            maturity_dt = swap._maturity_dt
            maturity_time = (maturity_dt - self._value_dt) / 365.0  # ACT_365F

            # First, trigger leg valuation to populate internal cashflow data
            # We need to value domestic leg normally
            pv_domestic = swap._domestic_leg.value(
                value_dt=self._value_dt,
                discount_curve=self._domestic_curve,
                index_curve=self._domestic_curve,
                first_fixing_rate=None
            )

            # Value foreign leg to populate cashflows
            # Use foreign OIS curve for projection, XCCY curve will be bootstrapped
            _ = swap._foreign_leg.value(
                value_dt=self._value_dt,
                discount_curve=self._foreign_curve,
                index_curve=self._foreign_curve,
                first_fixing_rate=None
            )

            # Extract ONLY foreign leg payments (domestic PV is already pre-computed)
            for pmt_idx, pmnt_dt in enumerate(swap._foreign_leg._payment_dts):
                if pmnt_dt >= self._value_dt:  # Include payments at or after value date
                    pmnt_time = (pmnt_dt - self._value_dt) / 365.0  # ACT_365F
                    df_foreign_ois = self._foreign_curve.df(pmnt_dt, self._foreign_curve._dc_type)

                    # Extract data needed to recompute cashflows from DFs
                    # Cashflows will be computed inside bootstrap from forward rates
                    if pmt_idx < len(swap._foreign_leg._year_fracs):
                        year_frac = swap._foreign_leg._year_fracs[pmt_idx]
                        notional = swap._foreign_leg._notional_array[pmt_idx] if len(swap._foreign_leg._notional_array) > 0 else swap._foreign_notional

                        # Get accrual dates and convert to times using foreign curve's day count
                        start_accrual_dt = swap._foreign_leg._start_accrued_dts[pmt_idx]
                        end_accrual_dt = swap._foreign_leg._end_accrued_dts[pmt_idx]
                        # Use foreign curve's day count to match foreign_ois_times grid
                        from cavour.utils.helpers import times_from_dates
                        start_accrual_time = times_from_dates(start_accrual_dt, self._value_dt, self._foreign_curve._dc_type)
                        end_accrual_time = times_from_dates(end_accrual_dt, self._value_dt, self._foreign_curve._dc_type)

                        # Check if this is a notional exchange (year_frac=0)
                        is_notional_exchange = (abs(year_frac) < 1e-10)

                        # Check if final payment with notional return
                        is_last_payment = (pmnt_dt == maturity_dt) and swap._foreign_leg._notional_exchange

                        # For spread sensitivity: always year_frac * notional (zero for notional exchanges)
                        spread_sensitivity = year_frac * notional if not is_notional_exchange else 0.0
                    else:
                        # Fallback
                        year_frac = 0.0
                        notional = 0.0
                        start_accrual_time = pmnt_time
                        end_accrual_time = pmnt_time
                        is_notional_exchange = True
                        is_last_payment = False
                        spread_sensitivity = 0.0

                    all_points.append({
                        'time': pmnt_time,
                        'time_key': round(pmnt_time, 4),  # For deduplication
                        'swap_idx': swap_idx,
                        'is_domestic': False,
                        'is_foreign': True,
                        'is_maturity': (pmnt_dt == maturity_dt),
                        'is_at_value_dt': (pmnt_dt == self._value_dt),
                        'basis_spread': basis_spread,
                        # Store data to recompute cashflows from DFs
                        'year_frac': year_frac,
                        'notional': notional,
                        'start_accrual_time': start_accrual_time,
                        'end_accrual_time': end_accrual_time,
                        'is_notional_exchange': is_notional_exchange,
                        'is_last_payment': is_last_payment,
                        'spread_sensitivity': spread_sensitivity,  # For basis spread AD
                        'df_domestic_ois': 1.0,  # Not used for foreign payments
                        'df_foreign_ois': df_foreign_ois,
                        'pv_domestic': pv_domestic,  # Store total domestic PV for this swap
                        'maturity_time': maturity_time
                    })

        # Sort all points by time, then by swap_idx for deterministic ordering
        all_points_sorted = sorted(all_points, key=lambda x: (x['time'], x['swap_idx']))

        # NO deduplication at payment level - each swap needs its payments tracked separately
        unique_points = all_points_sorted

        # Build XCCY node mask (foreign payments after value_dt)
        xccy_node_mask = []
        for point in unique_points:
            is_xccy_node = not point['is_at_value_dt']  # All non-value_dt points are nodes
            xccy_node_mask.append(is_xccy_node)

        # Pre-compute which XCCY nodes are unique by time (for final curve output)
        # This avoids using jnp.unique in the differentiable computation
        # Track first occurrence of each unique time among XCCY nodes
        seen_times = {}
        unique_node_indices = []  # Indices in the FILTERED (xccy_node_mask=True) array
        filtered_idx = 0
        for idx, point in enumerate(unique_points):
            if xccy_node_mask[idx]:  # Only consider XCCY nodes
                time_key = point['time_key']
                if time_key not in seen_times:
                    seen_times[time_key] = filtered_idx
                    unique_node_indices.append(filtered_idx)
                filtered_idx += 1

        # Build dependency graph (prev_idx)
        # For each point, find the previous XCCY node (any swap, in time order)
        n_points = len(unique_points)
        prev_idx_array = np.full(n_points, -1, dtype=np.int32)

        # Track indices of XCCY nodes (excluding value_dt)
        xccy_node_indices = []
        for idx, point in enumerate(unique_points):
            if not point['is_at_value_dt']:
                xccy_node_indices.append(idx)

        # For each XCCY node, previous node is the one before it in the list
        for i, idx in enumerate(xccy_node_indices):
            if i > 0:
                prev_idx_array[idx] = xccy_node_indices[i-1]

        # Convert to JAX arrays
        n_swaps = len(self._used_swaps)
        times_array = jnp.array([p['time'] for p in unique_points])
        basis_spreads_array = jnp.array([p['basis_spread'] for p in unique_points])
        swap_idx_array = jnp.array([p['swap_idx'] for p in unique_points], dtype=jnp.int32)
        is_domestic_array = jnp.array([p['is_domestic'] for p in unique_points])
        is_foreign_array = jnp.array([p['is_foreign'] for p in unique_points])
        is_maturity_array = jnp.array([p['is_maturity'] for p in unique_points])
        is_at_value_dt_array = jnp.array([p['is_at_value_dt'] for p in unique_points])
        # New arrays for recomputing cashflows from DFs
        year_frac_array = jnp.array([p['year_frac'] for p in unique_points])
        notional_array = jnp.array([p['notional'] for p in unique_points])
        start_accrual_time_array = jnp.array([p['start_accrual_time'] for p in unique_points])
        end_accrual_time_array = jnp.array([p['end_accrual_time'] for p in unique_points])
        is_notional_exchange_array = jnp.array([p['is_notional_exchange'] for p in unique_points])
        is_last_payment_array = jnp.array([p['is_last_payment'] for p in unique_points])
        spread_sensitivity_array = jnp.array([p['spread_sensitivity'] for p in unique_points])
        df_domestic_ois_array = jnp.array([p['df_domestic_ois'] for p in unique_points])
        df_foreign_ois_array = jnp.array([p['df_foreign_ois'] for p in unique_points])
        prev_idx_array = jnp.array(prev_idx_array)
        xccy_node_mask_array = jnp.array(xccy_node_mask)

        # Pre-computed domestic PV for each swap (constant throughout)
        # Map each point's swap_idx to the correct domestic PV
        pv_domestic_by_swap_dict = {}
        for i in range(n_swaps):
            pv_dom = self._used_swaps[i]._domestic_leg.value(
                value_dt=self._value_dt,
                discount_curve=self._domestic_curve,
                index_curve=self._domestic_curve,
                first_fixing_rate=None
            )
            pv_domestic_by_swap_dict[i] = pv_dom

        # Create array indexed by swap number
        pv_domestic_by_swap = jnp.array([pv_domestic_by_swap_dict[i] for i in range(n_swaps)])

        # Pre-compute mask matrix for sequential accumulation
        # This avoids swap-indexed state (which causes circular gradient dependencies in JAX)
        # same_swap_mask[i, j] = 1 if point j belongs to same swap as point i AND j < i
        # This enables sequential writes: state[i] = value, then sum using jnp.dot(mask, state)
        # JAX can differentiate through dot products, but not through dynamic indexing
        same_swap_mask = np.zeros((n_points, n_points))
        for i in range(n_points):
            swap_i = unique_points[i]['swap_idx']
            for j in range(i):  # Only previous points
                swap_j = unique_points[j]['swap_idx']
                if swap_i == swap_j:
                    same_swap_mask[i, j] = 1.0
        same_swap_mask_array = jnp.array(same_swap_mask)

        # Get foreign OIS curve grid for interpolation (used to compute forward rates)
        # _foreign_curve._times already starts with 0, so prepend only if needed
        if self._foreign_curve._times[0] > 1e-10:
            foreign_ois_times = jnp.concatenate([jnp.array([0.0]), jnp.array(self._foreign_curve._times)])
            foreign_ois_dfs = jnp.concatenate([jnp.array([1.0]), jnp.array(self._foreign_curve._dfs)])
        else:
            foreign_ois_times = jnp.array(self._foreign_curve._times)
            foreign_ois_dfs = jnp.array(self._foreign_curve._dfs)

        return {
            'n_points': n_points,
            'n_swaps': n_swaps,
            'times': times_array,
            'basis_spreads': basis_spreads_array,
            'swap_idx': swap_idx_array,
            'is_domestic': is_domestic_array,
            'is_foreign': is_foreign_array,
            'is_maturity': is_maturity_array,
            'is_at_value_dt': is_at_value_dt_array,
            # New: data to compute cashflows from DFs (instead of pre-computed cashflows)
            'year_fracs': year_frac_array,
            'notionals': notional_array,
            'start_accrual_times': start_accrual_time_array,
            'end_accrual_times': end_accrual_time_array,
            'is_notional_exchange': is_notional_exchange_array,
            'is_last_payment': is_last_payment_array,
            'spread_sensitivities': spread_sensitivity_array,
            'df_domestic_ois': df_domestic_ois_array,
            'df_foreign_ois': df_foreign_ois_array,  # DFs at payment times (legacy, may be unused)
            'foreign_ois_times': foreign_ois_times,  # Full curve grid for interpolation
            'foreign_ois_dfs': foreign_ois_dfs,  # Full curve DFs for interpolation
            'prev_idx': prev_idx_array,
            'xccy_node_mask': xccy_node_mask_array,
            'unique_node_indices': jnp.array(unique_node_indices, dtype=jnp.int32),
            'pv_domestic_by_swap': pv_domestic_by_swap,
            'same_swap_mask': same_swap_mask_array,  # For sequential accumulation
            'spot_fx': self._spot_fx
        }

###############################################################################

    def _run_jax_bootstrap(self, payment_data):
        """
        Run JAX-based bootstrap using pure JAX automatic differentiation.

        JAX's built-in AD handles lax.scan differentiation correctly,
        including second derivatives (Hessians). Removing the custom_vjp
        wrapper enables forward-mode AD (jvp) required for hessian().

        This allows replacing finite difference Hessian computations with
        fast JAX AD in subsequent optimizations.
        """
        return self._run_jax_bootstrap_impl(payment_data)

###############################################################################

    def _run_jax_bootstrap_impl(self, payment_data):
        """
        Implementation of JAX-based bootstrap (forward pass).

        This contains the actual bootstrap logic using lax.scan.
        Separated from _run_jax_bootstrap to enable custom VJP.
        """
        from jax import lax

        n_points = payment_data['n_points']
        n_swaps = payment_data['n_swaps']
        times = payment_data['times']
        basis_spreads = payment_data['basis_spreads']
        swap_idx = payment_data['swap_idx']
        is_maturity = payment_data['is_maturity']
        is_at_value_dt = payment_data['is_at_value_dt']
        spread_sensitivities = payment_data['spread_sensitivities']
        df_foreign_ois = payment_data['df_foreign_ois']
        prev_idx = payment_data['prev_idx']
        xccy_node_mask = payment_data['xccy_node_mask']
        unique_node_indices = payment_data['unique_node_indices']
        pv_domestic_by_swap = payment_data['pv_domestic_by_swap']
        same_swap_mask = payment_data['same_swap_mask']  # For sequential accumulation
        spot_fx = payment_data['spot_fx']

        # New fields for computing cashflows from DFs
        year_fracs = payment_data['year_fracs']
        notionals = payment_data['notionals']
        start_accrual_times = payment_data['start_accrual_times']
        end_accrual_times = payment_data['end_accrual_times']
        is_notional_exchange = payment_data['is_notional_exchange']
        is_last_payment = payment_data['is_last_payment']
        foreign_ois_times = payment_data['foreign_ois_times']
        foreign_ois_dfs_grid = payment_data['foreign_ois_dfs']

        # PRE-COMPUTE interpolated DFs for all accrual times
        # This avoids JAX scan closure issues with gradient backpropagation
        # Interpolating inside the scan would capture foreign_ois_dfs_grid as a closure variable,
        # which JAX cannot differentiate through. Pre-computing allows gradients to flow correctly.
        #
        # IMPORTANT: Use log-linear interpolation (flat forward rates) to match non-AD version
        # Linear interpolation on DFs would violate the flat forward rate assumption
        log_foreign_ois_dfs = jnp.log(foreign_ois_dfs_grid)
        log_df_start = jnp.interp(start_accrual_times, foreign_ois_times, log_foreign_ois_dfs)
        log_df_end = jnp.interp(end_accrual_times, foreign_ois_times, log_foreign_ois_dfs)
        df_start_accrual_array = jnp.exp(log_df_start)
        df_end_accrual_array = jnp.exp(log_df_end)

        def step(state, inputs):
            """
            Bootstrap one point in the XCCY curve using SEQUENTIAL INDEXING ONLY.

            Key design choice: Sequential writes (state[idx] = value) instead of
            swap-indexed arrays to avoid circular dependencies in JAX gradients.
            Each point writes to its own index, then sums contributions using
            pre-computed masks.

            State structure (all point-indexed, written sequentially):
            - xccy_dfs: [n_points] - XCCY discount factors (sequentially populated)
            - pv_contributions: [n_points] - PV contribution at each point
            - cf_contributions: [n_points] - Cashflow contribution at each point

            At each point idx:
            1. Compute intermediate XCCY DF using flat forward basis
            2. Store PV contribution at this point: state[idx] = value (sequential write)
            3. If maturity: sum all previous contributions using pre-computed mask
            4. Solve par condition if at maturity point

            Why this pattern? JAX can differentiate through:
            - Sequential array writes: state.at[idx].set(value)
            - Dot products for aggregation: jnp.dot(mask, state)

            But NOT through:
            - Dynamic indexing by swap: state[swap_idx[i]]
            - Dictionary/object-based state updates
            """
            (idx, time, basis, prev_idx_i, is_mat, is_val_dt, spread_sens, df_ois, swap_i, mask_row,
             year_frac, notional, is_notl_exch, is_last_pmt, df_start_accrual, df_end_accrual) = inputs

            # Compute cashflow dynamically from foreign OIS DFs
            # This enables gradients to flow through foreign_ois_dfs_grid -> forward rates -> cashflows

            # For notional exchanges: cashflow = notional (no dependency on rates)
            # For interest payments: compute forward rate from DFs
            def compute_interest_cashflow():
                # Use pre-computed interpolated DFs (passed as scan inputs for gradient flow)
                df_start = df_start_accrual
                df_end = df_end_accrual

                # Compute forward rate: fwd = (DF_start / DF_end - 1) / year_frac
                # Use jnp.maximum to avoid division by zero (jnp.where evaluates both branches)
                # For year_frac=0 (notional exchanges), this branch won't be selected anyway
                year_frac_safe = jnp.maximum(year_frac, 1e-10)
                fwd_rate = (df_start / df_end - 1.0) / year_frac_safe

                # Zero out forward rate for notional exchanges (year_frac ~ 0)
                fwd_rate = jnp.where(year_frac > 1e-10, fwd_rate, 0.0)

                # Base interest payment (forward_rate only, NO spread yet)
                # The spread will be added separately via spread_sens
                base_interest = fwd_rate * year_frac * notional

                # If last payment, add notional return
                return jnp.where(is_last_pmt, base_interest + notional, base_interest)

            # Select between notional exchange and interest payment
            # For notional exchanges: at effective_dt (t=0), we RECEIVE notional (negative for PAY leg)
            # at maturity, we RETURN notional (handled by is_last_payment flag above)
            # Since effective_dt notional exchange has year_frac=0 and is_last_payment=False,
            # we need to use -notional for initial exchange
            notional_exchange_cf = jnp.where(is_last_pmt, notional, -notional)
            base_cashflow = jnp.where(is_notl_exch, notional_exchange_cf, compute_interest_cashflow())

            # Add basis spread component (for basis spread AD)
            # spread_sens = year_frac * notional for interest payments, 0 for notional exchanges
            cashflow = base_cashflow + basis * spread_sens

            # Get previous XCCY DF and time for flat forward basis formula
            prev_df = jnp.where(prev_idx_i < 0, 1.0, state['xccy_dfs'][prev_idx_i])
            prev_time = jnp.where(prev_idx_i < 0, 0.0, times[prev_idx_i])
            prev_df_ois = jnp.where(prev_idx_i < 0, 1.0, df_foreign_ois[prev_idx_i])

            # Compute intermediate XCCY DF using flat forward basis
            is_first = (prev_idx_i < 0)

            # First pillar: DF_xccy(t) = DF_ois(t) * exp(-basis * t)
            df_first = df_ois * jnp.exp(-basis * time)

            # Subsequent: DF_xccy(t) = DF_xccy(t_prev) * [DF_ois(t)/DF_ois(t_prev)] * exp(-basis * dt)
            dt_delta = time - prev_time
            df_subsequent = prev_df * (df_ois / prev_df_ois) * jnp.exp(-basis * dt_delta)

            df_intermediate = jnp.where(is_first, df_first, df_subsequent)

            # Compute PV contribution at THIS point (sequential write to idx)
            # PV contribution for non-maturity, non-value_dt points
            is_known = (~is_mat) & (~is_val_dt)
            pv_contrib = jnp.where(is_known, cashflow * df_intermediate, 0.0)

            # PV contribution at value_dt (DF = 1.0)
            pv_at_val_dt_contrib = jnp.where(is_val_dt, cashflow * 1.0, 0.0)

            # Total PV contribution at this point
            total_pv_contribution = pv_contrib + pv_at_val_dt_contrib

            # Store PV contribution at THIS index (sequential write)
            new_pv_contributions = state['pv_contributions'].at[idx].set(total_pv_contribution)

            # Compute cashflow contribution at THIS point
            cf_contribution = jnp.where(is_mat, cashflow, 0.0)

            # Store cashflow contribution at THIS index (sequential write)
            new_cf_contributions = state['cf_contributions'].at[idx].set(cf_contribution)

            # If maturity point: sum all previous contributions for this swap
            # mask_row is pre-computed and passed as input (avoids dynamic indexing)
            # Shape: [n_points], with 1.0 for points in same swap (j < idx), 0.0 otherwise

            # Sum contributions for this swap (mask zeros out other swaps)
            pv_sum_prev = jnp.dot(mask_row, state['pv_contributions'])
            cf_sum_prev = jnp.dot(mask_row, state['cf_contributions'])

            # Total includes current contribution
            pv_foreign_known = pv_sum_prev + total_pv_contribution
            cf_at_maturity = cf_sum_prev + cf_contribution

            # Apply PAY/RECEIVE sign for foreign leg (all foreign legs are PAY type)
            foreign_sign = -1.0
            pv_foreign_known_signed = pv_foreign_known * foreign_sign
            cf_at_maturity_signed = cf_at_maturity * foreign_sign

            # Solve par condition if this is a maturity point
            # Par: PV_domestic + spot_fx * (PV_foreign_known + cf_last * DF_xccy) = 0
            # Solve: DF_xccy = -(PV_domestic + spot_fx * PV_foreign_known) / (spot_fx * cf_last)
            pv_dom = pv_domestic_by_swap[swap_i]
            numerator = -(pv_dom + spot_fx * pv_foreign_known_signed)
            denominator = spot_fx * cf_at_maturity_signed

            # Compute DF from par condition (with safety check)
            # Use jnp.maximum to avoid division by zero (jnp.where evaluates both branches)
            denominator_safe = jnp.where(
                jnp.abs(denominator) > 1e-12,
                denominator,
                jnp.where(denominator >= 0, 1e-12, -1e-12)  # Keep sign
            )
            df_from_par = numerator / denominator_safe

            # Only use par solution if denominator is large enough
            df_from_par = jnp.where(
                jnp.abs(denominator) > 1e-12,
                df_from_par,
                df_intermediate
            )

            # Use par solution if at maturity, otherwise use intermediate DF
            df_final = jnp.where(is_mat, df_from_par, df_intermediate)

            # Update state (all sequential writes to index idx)
            new_xccy_dfs = state['xccy_dfs'].at[idx].set(df_final)
            new_state = {
                'xccy_dfs': new_xccy_dfs,
                'pv_contributions': new_pv_contributions,
                'cf_contributions': new_cf_contributions
            }

            return new_state, df_final

        # Initialize state (all point-indexed for sequential writes)
        init_state = {
            'xccy_dfs': jnp.zeros(n_points),
            'pv_contributions': jnp.zeros(n_points),  # PV contribution at each point
            'cf_contributions': jnp.zeros(n_points)   # CF contribution at each point
        }

        # Prepare scan inputs
        idxs = jnp.arange(n_points)
        scan_inputs = (
            idxs,
            times,
            basis_spreads,
            prev_idx,
            is_maturity,
            is_at_value_dt,
            spread_sensitivities,
            df_foreign_ois,
            swap_idx,
            same_swap_mask,  # Pass mask as input to avoid dynamic indexing
            year_fracs,
            notionals,
            is_notional_exchange,
            is_last_payment,
            df_start_accrual_array,  # Pre-computed interpolated DFs for gradient flow
            df_end_accrual_array     # Pre-computed interpolated DFs for gradient flow
        )

        # Run scan - this is the pure JAX operation
        final_state, all_dfs = lax.scan(step, init_state, scan_inputs)

        # Filter to only XCCY curve nodes (exclude value_dt point)
        filtered_times = times[xccy_node_mask]
        filtered_dfs = all_dfs[xccy_node_mask]

        # Deduplicate by time (keep first occurrence of each unique time)
        # Use pre-computed indices from _prepare_ad_inputs to avoid jnp.unique
        # which is non-differentiable and causes gradient NaN
        unique_times = filtered_times[unique_node_indices]
        unique_dfs = filtered_dfs[unique_node_indices]

        # Prepend t=0, DF=1.0
        final_times = jnp.concatenate([jnp.array([0.0]), unique_times])
        final_dfs = jnp.concatenate([jnp.array([1.0]), unique_dfs])

        return final_times, final_dfs

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
