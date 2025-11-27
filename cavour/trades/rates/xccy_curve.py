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
from cavour.utils.day_count import DayCount
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
        # Use foreign leg day count for curve time calculations
        # (convention: foreign leg typically carries the basis)
        self._dc_type = self._used_swaps[0]._foreign_leg._dc_type

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

        dcc = DayCount(self._dc_type)
        days_in_year = dcc.days_in_year()

        for swap in self._used_swaps:
            # Basis spread is on foreign leg
            basis_spread = swap._foreign_spread
            maturity_dt = swap._maturity_dt
            tswap = (maturity_dt - self._value_dt) / days_in_year

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
        # Initialize with just the anchor point
        xccy_nodes_times = [0.0]
        xccy_nodes_dfs = [1.0]

        for i, swap in enumerate(self._used_swaps):
            t_mat = self.swap_times[i]

            # Value domestic leg using domestic curve
            pv_domestic = swap._domestic_leg.value(
                value_dt=self._value_dt,
                discount_curve=self._domestic_curve,
                index_curve=self._domestic_curve,
                first_fixing_rate=None
            )

            # Build temporary XCCY curve from current nodes
            # For the first pillar, use the foreign OIS curve for initial approximation
            if i == 0:
                # For first pillar, use foreign curve as initial XCCY approximation
                temp_xccy_curve = self._foreign_curve
            else:
                # For subsequent pillars, use bootstrapped XCCY nodes
                temp_xccy_curve = DiscountCurve(
                    value_dt=self._value_dt,
                    df_dts=xccy_nodes_times,
                    df_values=np.array(xccy_nodes_dfs),
                    interp_type=self._interp_type
                )

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
                if pmnt_dt >= self._value_dt:
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

            # Store the new node
            xccy_nodes_times.append(t_mat)
            xccy_nodes_dfs.append(df_mat)

            # Update curve arrays for representation
            self._times = jnp.append(self._times, t_mat)
            self._dfs = jnp.append(self._dfs, df_mat)
            self._repr_dfs = jnp.append(self._repr_dfs, df_mat)

        # Validate calibration if requested
        if self._check_refit:
            self._check_refits(SWAP_TOL)

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
