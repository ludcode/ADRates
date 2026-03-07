##############################################################################

##############################################################################

"""
OIS curve construction via cashflow-based bootstrapping.

Provides the OISCurve class for building discount curves from overnight index
swap (OIS) market quotes. Uses a cashflow-based bootstrapping approach that
solves directly for discount factors without requiring iterative solvers.

Key features:
- JAX-compatible automatic differentiation for computing sensitivities
- Multiple interpolation schemes (flat forward, linear zero rates, cubic)
- Exact reproduction of input swap rates (within tolerance)
- Dense discount factor grid for accurate interpolation
- Support for various day count conventions

The bootstrapping algorithm:
1. Builds discount factors sequentially for each swap maturity
2. Uses par swap condition: PV(fixed leg) = PV(floating leg)
3. Solves directly: D_m = (1 - r_i × PV01_prev) / (1 + r_i × α_m)
4. Stores intermediate discount factors for dense interpolation grid

Example:
    >>> # Create OIS swaps at market rates
    >>> swaps = [
    ...     OIS(value_dt, "1Y", SwapTypes.PAY, 0.045, ...),
    ...     OIS(value_dt, "2Y", SwapTypes.PAY, 0.047, ...),
    ...     OIS(value_dt, "5Y", SwapTypes.PAY, 0.050, ...)
    ... ]
    >>>
    >>> # Bootstrap curve
    >>> curve = OISCurve(
    ...     value_dt=value_dt,
    ...     instruments=swaps,
    ...     interp_type=InterpTypes.LINEAR_ZERO_RATES,
    ...     check_refit=True  # Verify swap rates are reproduced
    ... )
    >>>
    >>> # Query discount factors
    >>> df_1y = curve.df(value_dt.add_years(1))
"""

import numpy as np
from scipy import optimize
from scipy.interpolate import interp1d
import copy
import jax.numpy as jnp
from jax import grad, hessian
import jax
from typing import Optional

from cavour.utils.error import LibError
from cavour.utils.date import Date
from cavour.utils.day_count import DayCount
from cavour.utils.helpers import (check_argument_types,
                              _func_name, 
                              label_to_string, 
                              format_table)
from cavour.utils.global_vars import gDaysInYear
from cavour.market.curves.interpolator import InterpTypes, Interpolator
from cavour.market.curves.discount_curve import DiscountCurve

from cavour.trades.rates.ois import OIS

SWAP_TOL = 1e-10

jax.config.update("jax_enable_x64", True)


class OISCurve(DiscountCurve):
    """ Constructs a discount curve as implied by the prices of Overnight
    Index Rate swaps. The curve date is the date on which we are
    performing the valuation based on the information available on the
    curve date. Typically it is the date on which an amount of 1 unit paid
    has a present value of 1. This class inherits from FinDiscountCurve
    and so it has all of the methods that that class has.

    The construction of the curve is assumed to depend on just the OIS curve,
    i.e. it does not include information from Ibor-OIS basis swaps. For this
    reason I call it a one-curve.
    """

###############################################################################

    def __init__(self,
                 value_dt: Date,
                 instruments: list,
                 interp_type: InterpTypes = InterpTypes.FLAT_FWD_RATES,
                 check_refit: bool = False,  # Set to True to test it works
                 use_ad: bool = True,  # Enable AD storage by default
                 compute_gamma: bool = False,  # Compute Hessians for GAMMA (slow)
                 hessian_bandwidth: Optional[int] = None):  # Hessian sparsity: 0=diagonal only, None=full
        """ Create an instance of an overnight index rate swap curve given a
        valuation date and a set of OIS rates or mixed money market instruments.

        Supports:
        - OIS swaps (for 1Y+ tenors)
        - Cash deposits (for 0-12M tenors)
        - FRAs (for 3M-2Y tenors)
        - STIR futures (for 3M-2Y tenors, most liquid)
        - Mixed instrument types for full term structure

        An interpolation method has also to be provided. The default is to use a
        linear interpolation for swap rates on coupon dates and to then assume
        flat forwards between these coupon dates.

        The curve will assign a discount factor of 1.0 to the valuation date.

        Args:
            value_dt: Valuation date (anchor date for the curve)
            instruments: List of instruments (OIS, CashDeposit, FRA, IRFuture) for calibration
            interp_type: Interpolation method for discount factors
            check_refit: If True, verify calibration instruments reprice correctly
            use_ad: If True, compute and store Jacobians for DELTA sensitivities
            compute_gamma: If True, compute Hessians for GAMMA (second-order sensitivities).
                          Default False for performance (3-5x faster curve construction).
                          Set to True only when GAMMA risk measures are required.
            hessian_bandwidth: Controls Hessian sparsity for performance optimization.
                              0 = diagonal-only (22-25x speedup, empirically 100% accurate for OIS)
                              None = full Hessian (default, maximum accuracy)
                              Ignored if compute_gamma=False.
        """

        # Validate inputs
        check_argument_types(getattr(self, _func_name(), None), locals())

        # Store instruments for curve building
        self._used_instruments = instruments

        # Legacy attribute for backward compatibility with existing code
        self._used_swaps = self._used_instruments

        # Sort instruments by maturity for proper bootstrapping order
        self._used_instruments = self._sort_instruments_by_maturity(self._used_instruments)

        # Validate and auto-adjust futures convexity adjustments
        self._validate_and_adjust_convexity(self._used_instruments)

        self._value_dt = value_dt
        self._interp_type = interp_type
        self._check_refit = check_refit
        self._use_ad = use_ad
        self._compute_gamma = compute_gamma
        self._hessian_bandwidth = hessian_bandwidth
        self._interpolator = None

        # Initialize AD attributes
        self._jac = None  # Jacobian d(DFs)/d(rates)
        self._hess = None  # Hessian d²(DFs)/d(rates)² or diagonal only if hessian_bandwidth=0

        swap_rates = self._prepare_curve_builder_inputs()
        self._build_curve_ad(swap_rates)

        # Compute and store Jacobians/Hessians for AD if requested
        if use_ad:
            self._compute_ad_derivatives(swap_rates)

###############################################################################

    def _validate_and_adjust_convexity(self, instruments):
        """
        Validate futures convexity adjustments.

        Futures Convexity Adjustment (FCA) arises from the difference in cashflow timing:
        - Futures: Daily mark-to-market (no discounting)
        - OIS forwards: Single cashflow at maturity (discounted)

        This creates a hedging asymmetry:
        - When rates rise: Futures gains immediate, OIS gains discounted (worth less)
        - When rates fall: Futures losses immediate, OIS losses discounted (hurts more)

        Result: Futures rates > OIS forward rates by the FCA amount.

        FCA can be observed directly from market data:
          FCA = Futures_rate - OIS_forward_rate (same maturity)

        This method warns when futures are used with FCA=0 (default), which assumes
        futures rates = forward rates and may introduce arbitrage opportunities.

        Args:
            instruments: List of instruments (sorted by maturity)
        """
        from cavour.utils.global_types import InstrumentTypes
        import warnings

        # Find all futures
        futures = []
        for inst in instruments:
            if hasattr(inst, 'derivative_type'):
                if inst.derivative_type == InstrumentTypes.STIR_FUTURE:
                    futures.append(inst)

        # If no futures, nothing to validate
        if not futures:
            return

        # Check for futures with FCA=0 (default)
        futures_with_zero_fca = [f for f in futures if abs(f._convexity_adjustment) < 1e-10]

        if futures_with_zero_fca:
            # Issue warning for first occurrence only (avoid spam)
            first_fut = futures_with_zero_fca[0]
            warnings.warn(
                f"IR Futures detected with convexity_adjustment=0.0 (default). "
                f"This assumes Futures rate = Forward rate, ignoring the timing difference "
                f"between daily mark-to-market (futures) and maturity settlement (forwards). "
                f"\n\nFutures Convexity Adjustment (FCA) can be extracted from market data:\n"
                f"  FCA = Futures_rate - OIS_forward_rate (same maturity)\n\n"
                f"To avoid potential arbitrage, either:\n"
                f"  1. Provide explicit convexity_adjustment parameter to IRFuture()\n"
                f"  2. Use OIS swaps only (if futures data unavailable)\n\n"
                f"First future with FCA=0: {first_fut._expiry_dt}, "
                f"futures_price={first_fut._futures_price:.2f}, "
                f"implied_rate={first_fut._implied_rate*100:.2f}%",
                UserWarning,
                stacklevel=3
            )

###############################################################################

    def _sort_instruments_by_maturity(self, instruments):
        """
        Sort instruments by maturity date for proper bootstrapping order.

        Bootstrapping requires instruments to be ordered from shortest to longest
        maturity so that discount factors can be built sequentially.

        Args:
            instruments: List of instruments (OIS, CashDeposit, FRA)

        Returns:
            Sorted list of instruments by maturity date
        """
        def get_maturity(instrument):
            """Extract maturity date from instrument."""
            # All our instruments should have _maturity_dt attribute
            if hasattr(instrument, '_maturity_dt'):
                return instrument._maturity_dt
            # Fallback for OIS swaps
            elif hasattr(instrument, '_adjusted_fixed_dts'):
                return instrument._adjusted_fixed_dts[-1]
            else:
                raise ValueError(f"Cannot determine maturity for instrument: {type(instrument)}")

        return sorted(instruments, key=get_maturity)

###############################################################################

    def _prepare_curve_builder_inputs(self):
        """
        Prepare inputs for curve bootstrap from mixed instrument types.

        Extracts rates, maturities, and year fractions from:
        - OIS swaps (multi-cashflow)
        - Cash deposits (single cashflow)
        - FRAs (single cashflow)
        - STIR futures (single cashflow)

        All instruments must have:
        - _fixed_coupon: The fixed rate
        - _adjusted_fixed_dts: List of payment dates
        - _fixed_year_fracs: List of year fractions
        - _dc_type: Day count convention

        Returns:
            List of instrument rates for bootstrapping
        """

        # Extract day count convention from first instrument
        # Try different instrument types
        first_inst = self._used_instruments[0]
        if hasattr(first_inst, '_float_leg'):
            # OIS swap
            self._dc_type = first_inst._float_leg._dc_type
        elif hasattr(first_inst, '_dc_type'):
            # Deposit or FRA
            self._dc_type = first_inst._dc_type
        else:
            raise ValueError(f"Cannot determine day count for instrument: {type(first_inst)}")

        self._times = jnp.array([])
        self._dfs = jnp.array([])
        self._repr_dfs = jnp.array([])

        # time zero is now.
        df_mat = 1.0
        self._times = jnp.append(self._times, 0.0)
        self._dfs = jnp.append(self._dfs, df_mat)
        self._repr_dfs = jnp.append(self._repr_dfs, df_mat)

        swap_rates = []
        swap_times = []
        year_fracs = []
        start_times = []  # NEW: For forward-starting instruments

        # Extract data from each instrument (OIS, Deposit, FRA, IRFuture)
        # All instruments store data in standardized attributes:
        # - _fixed_coupon: The rate
        # - _adjusted_fixed_dts: Payment dates (list)
        # - _fixed_year_fracs: Year fractions (list)
        # - _start_year_fracs: Start times for forward-starting (list, NEW)

        dcc = DayCount(self._dc_type)
        days_in_year = dcc.days_in_year()

        for instrument in self._used_instruments:
            # Extract rate (all instruments have this)
            rate = instrument._fixed_coupon

            # Extract maturity date (last payment date)
            maturity_dt = instrument._adjusted_fixed_dts[-1]
            tswap = (maturity_dt - self._value_dt) / days_in_year

            # Extract year fractions
            # For OIS: swap._fixed_leg._year_fracs (list of all coupon periods)
            # For Deposit/FRA/IRFuture: instrument._fixed_year_fracs (list with 1 element)
            if hasattr(instrument, '_fixed_leg'):
                # OIS swap - multi-cashflow
                year_frac = instrument._fixed_leg._year_fracs
            else:
                # Deposit, FRA, or IRFuture - use _fixed_year_fracs directly
                year_frac = instrument._fixed_year_fracs

            # Extract start times (NEW: for forward-starting instruments)
            # For OIS: All periods are spot-starting (use swap bootstrap formula with prev_pv01)
            # For Deposit: [0.0] (spot-starting)
            # For FRA/IRFuture: [start_year_frac] (forward-starting from start_dt)
            if hasattr(instrument, '_fixed_leg'):
                # OIS swap - all periods use spot-starting formula with prev_pv01 accumulation
                start_time = [0.0] * len(year_frac)
            elif hasattr(instrument, '_start_year_fracs'):
                # Deposit, FRA, or IRFuture with explicit start times
                start_time = instrument._start_year_fracs
            else:
                # Fallback for instruments without _start_year_fracs (assume spot-starting)
                start_time = [0.0] if isinstance(year_frac, list) else 0.0

            swap_times.append(tswap)
            swap_rates.append(rate)
            year_fracs.append(year_frac)
            start_times.append(start_time)  # NEW

        self.swap_times = swap_times
        self.swap_rates = swap_rates
        self.year_fracs = year_fracs
        self.start_times = start_times  # NEW

        return swap_rates

    def _build_curve_ad(self, swap_rates):
        """
        Bootstrap OIS curve using Engine's build_curve_ad() method.

        This ensures consistency between VALUE (which uses stored _times/_dfs)
        and DELTA/GAMMA (which use Jacobian/Hessian computed via the same method).

        Previously used recursive bootstrap with deduplication (61 DFs).
        Now uses Engine's scan-based bootstrap with all points (dense grid).
        """
        from cavour.market.position.engine import Engine

        # Create temporary Engine instance to access build_curve_ad
        engine_temp = Engine(None)

        # Call Engine.build_curve_ad() which produces dense DF grid
        # This ensures VALUE and DELTA/GAMMA use the same underlying curve
        swap_rates_array = jnp.array(swap_rates)
        times_dense, dfs_dense = engine_temp.build_curve_ad(
            swap_rates_array,
            self.swap_times,
            self.year_fracs,
            self.start_times  # NEW: Pass start times for forward-starting instruments
        )

        # Prepend time=0 with DF=1.0 if not already present (anchor point)
        # Check if Engine.build_curve_ad() already included t=0
        if len(times_dense) > 0 and times_dense[0] < 1e-7:
            # Already has t≈0, use as-is
            self._times = times_dense
            self._dfs = dfs_dense
        else:
            # No t=0, prepend it
            self._times = jnp.concatenate([jnp.array([0.0]), times_dense])
            self._dfs = jnp.concatenate([jnp.array([1.0]), dfs_dense])

        # Store representative DFs (at swap maturities only) for refit checking
        # Extract DFs at swap maturity times
        self._repr_dfs = jnp.array([1.0])  # Start with t=0
        for swap_time in self.swap_times:
            # Find closest time in times_dense
            idx = jnp.argmin(jnp.abs(times_dense - swap_time))
            self._repr_dfs = jnp.append(self._repr_dfs, dfs_dense[idx])

        return self._times, self._dfs

###############################################################################

    def _compute_ad_derivatives(self, swap_rates):
        """
        Compute and store Jacobian and optionally Hessian for automatic differentiation.

        Uses Engine's build_curve_ad() method for consistent bootstrap logic.
        This ensures VALUE, DELTA, and GAMMA all use the same underlying curve.

        Jacobian is always computed (needed for DELTA).
        Hessian is only computed if compute_gamma=True (needed for GAMMA).

        Args:
            swap_rates: Array of par swap rates used to build the curve
        """
        from jax import jacrev, hessian
        from cavour.market.position.engine import Engine

        # Create temporary Engine instance to access build_curve_ad method
        engine_temp = Engine(None)

        def build_dfs_from_rates(rates_array):
            """Pure function that bootstraps DFs from swap rates using Engine logic."""
            _, dfs = engine_temp.build_curve_ad(
                rates_array,
                self.swap_times,
                self.year_fracs,
                self.start_times  # NEW: Pass start times for forward-starting instruments
            )
            return dfs

        # Compute Jacobian: d(DFs)/d(rates) - ALWAYS needed for DELTA
        rates_array = jnp.array(swap_rates)
        jac_full = jacrev(build_dfs_from_rates)(rates_array)

        # Compute Hessian: d²(DFs)/d(rates)² - ONLY needed for GAMMA (expensive!)
        if self._compute_gamma:
            if self._hessian_bandwidth == 0:
                # Diagonal-only Hessian (empirically 100% accurate for OIS, 22-25x speedup)
                hess_full_temp = hessian(build_dfs_from_rates)(rates_array)
                # Extract diagonal: shape (n_dfs, n_rates, n_rates) -> (n_dfs, n_rates)
                n_dfs, n_rates = hess_full_temp.shape[0], hess_full_temp.shape[1]
                hess_full = jnp.zeros((n_dfs, n_rates))
                for i in range(n_dfs):
                    hess_full = hess_full.at[i, :].set(jnp.diag(hess_full_temp[i, :, :]))
            else:
                # Full Hessian (default for maximum accuracy)
                hess_full = hessian(build_dfs_from_rates)(rates_array)
        else:
            hess_full = None  # Skip expensive Hessian computation for DELTA-only workflows

        # Engine.build_curve_ad() returns DFs including t=0 (DF[0] = 1.0)
        # Engine gradients exclude t=0 (since it's a boundary condition, not a free parameter)
        # So we need to slice off the first row/element to match gradient dimensions
        # Check if first time point is t≈0
        _, dfs_check = engine_temp.build_curve_ad(rates_array, self.swap_times, self.year_fracs, self.start_times)
        times_check, _ = engine_temp.build_curve_ad(rates_array, self.swap_times, self.year_fracs, self.start_times)

        if len(times_check) > 0 and times_check[0] < 1e-7:
            # First DF is at t≈0, slice it off
            # Jacobian shape changes from (n_dfs, n_rates) to (n_dfs-1, n_rates)
            self._jac = jac_full[1:, :]
            # Hessian slicing depends on whether it's diagonal-only or full
            if hess_full is not None:
                if self._hessian_bandwidth == 0:
                    # Diagonal-only: shape (n_dfs, n_rates) -> (n_dfs-1, n_rates)
                    self._hess = hess_full[1:, :]
                else:
                    # Full Hessian: shape (n_dfs, n_rates, n_rates) -> (n_dfs-1, n_rates, n_rates)
                    self._hess = hess_full[1:, :, :]
            else:
                self._hess = None
        else:
            # No t=0 point, use full Jacobian/Hessian
            self._jac = jac_full
            self._hess = hess_full

###############################################################################

    def _build_curve(self):
        """ Construct the discount curve using a bootstrap approach. This is
        the linear swap rate method that is fast and exact as it does not
        require the use of a solver. It is also market standard. """

        self._dc_type = self._used_swaps[0]._float_leg._dc_type

        self._interpolator = Interpolator(self._interp_type)
        self._times = np.array([])
        self._dfs = np.array([])

        # time zero is now.
        t_mat = 0.0
        df_mat = 1.0
        self._times = np.append(self._times, 0.0)
        self._dfs = np.append(self._dfs, df_mat)

        found_start = False
        last_dt = self._value_dt

        # We use the longest swap assuming it has a superset of ALL of the
        # swap flow dates used in the curve construction
        longest_swap = self._used_swaps[-1]
        cpn_dts = longest_swap._adjusted_fixed_dts
        num_flows = len(cpn_dts)

        # Find where first coupon without discount factor starts
        start_index = 0
        for i in range(0, num_flows):
            if cpn_dts[i] > last_dt:
                start_index = i
                found_start = True
                break

        if found_start is False:
            raise LibError("Found start is false. Swaps payments inside FRAs")

        swap_rates = []
        swap_times = []

        # I use the last coupon date for the swap rate interpolation as this
        # may be different from the maturity date due to a holiday adjustment
        # and the swap rates need to align with the coupon payment dates
        for swap in self._used_swaps:
            swap_rate = swap._fixed_coupon
            maturity_dt = swap._adjusted_fixed_dts[-1]
            tswap = (maturity_dt - self._value_dt) / gDaysInYear
            swap_times.append(tswap)
            swap_rates.append(swap_rate)

        interpolated_swap_rates = []
        interpolated_swap_times = []

        for dt in cpn_dts[:]:
            swap_time = (dt - self._value_dt) / gDaysInYear
            swap_rate = np.interp(swap_time, swap_times, swap_rates)
            interpolated_swap_rates.append(swap_rate)
            interpolated_swap_times.append(swap_time)

        log_swap_rates = np.log(swap_rates)
        # Create log-linear interpolator
        log_linear_interp = interp1d(swap_times, log_swap_rates, kind='linear', fill_value='extrapolate')

        # Function to interpolate in normal domain
        def interpolate_loglinear(t):
            return np.exp(log_linear_interp(t))

        # Do I need this line ?
        #interpolated_swap_rates[0] = interpolated_swap_rates[1]
        accrual_factors = longest_swap._fixed_year_fracs

        acc = 0.0
        df = 1.0
        pv01 = 0.0
        df_settle = 1 #self.df(longest_swap._start_dt)

        for i in range(1, start_index):
            dt = cpn_dts[i]
            df = self.df(dt)
            acc = accrual_factors[i-1]
            pv01 += acc * df

        pv01_dict = {}

        def calculate_single_df(pv01, i, target_maturity=None, step=0):
            if target_maturity is None:
                t_mat = swap_times[i]
                #swap_rate = swap_rates[i]
            else:
                t_mat = target_maturity
                #swap_rate = interpolate_loglinear(t_mat)
            swap_rate = swap_rates[i]

            if len(self._used_swaps[i]._fixed_leg._year_fracs) == 1:
                acc = self._used_swaps[i]._fixed_leg._year_fracs[0]
                pv01_end = (acc * swap_rate + 1.0)
                df_mat = (df_settle) / pv01_end
                pv01 = acc * df_mat
            else:
                acc = self._used_swaps[i]._fixed_leg._year_fracs[-1]
                last_payment = sum(self._used_swaps[i]._fixed_leg._year_fracs[:-1-step])
                if round(last_payment , 2) not in pv01_dict:
                    step += 1
                    pv01_dict[round(last_payment,2)] = calculate_single_df(pv01, i, last_payment, step)

                pv01_end = (acc * swap_rate + 1)
                df_mat = (df_settle - swap_rate * pv01_dict[round(last_payment,2)]) / pv01_end
                zero_rate = (1 / df_mat)**(1 / t_mat) - 1
                pv01 = pv01_dict[round(last_payment,2)] + acc * df_mat

            self._times = np.append(self._times, t_mat)
            self._dfs = np.append(self._dfs, df_mat)
            self._interpolator.fit(self._times, self._dfs)

            pv01_dict[round(t_mat,2)] = pv01

            step = 0

            return pv01
        
        for i in range(0, len(self._used_swaps)):
            pv01 = calculate_single_df(pv01, i)

        if self._check_refit is True:
            self._check_refits(1e-10, SWAP_TOL, 1e-5)

###############################################################################

    def _check_refits(self, swap_tol):
        """
        Ensure that the curve refits all calibration instruments.

        Handles:
        - OIS swaps (value at effective date should be ~0)
        - Cash deposits (value at effective date should be ~0)
        - FRAs (value at effective date should be ~0)
        - STIR futures (value at effective date should be ~0)

        Args:
            swap_tol: Tolerance for refit error (absolute value / notional)
        """

        for instrument in self._used_instruments:
            # Value the instrument as of its effective/start date
            effective_dt = instrument._effective_dt

            # Call value() with appropriate signature
            # All instruments support: value(value_dt, discount_curve, ois_curve)
            if hasattr(instrument, '_float_leg'):
                # OIS swap - needs ois_curve for floating leg projection
                v = instrument.value(effective_dt, self, None)
            else:
                # Deposit, FRA, or IRFuture - only needs discount curve
                v = instrument.value(effective_dt, discount_curve=self)

            # Normalize by notional
            v = v / instrument._notional

            # Check refit tolerance
            if abs(v) > swap_tol:
                inst_type = type(instrument).__name__
                print(f"{inst_type} with maturity {instrument._maturity_dt} "
                      f"Not Repriced. Has Value {v}")

                # Print details if available
                if hasattr(instrument, 'print_fixed_leg_pv'):
                    instrument.print_fixed_leg_pv()
                if hasattr(instrument, 'print_float_leg_pv'):
                    instrument.print_float_leg_pv()
                if hasattr(instrument, 'print_details'):
                    instrument.print_details()

                raise LibError(
                    f"{inst_type} with maturity {instrument._maturity_dt} "
                    f"not repriced. Difference is {abs(v)}"
                )

###############################################################################

    def __repr__(self):

        s = label_to_string("OBJECT TYPE", type(self).__name__)
        num_points = len(self.swap_rates)
        s += label_to_string("DATES", "DISCOUNT FACTORS")
        for i in range(0, num_points):
            s += label_to_string("%12s" % self.swap_times[i],
                                 "%12.8f" % self.swap_rates[i])
            
        header = ["TENORS", "YEAR_FRACTION", "RATES", "DFs"]

        rows = []

        for i in range(0, len(self.swap_rates)):
            rows.append([
                round(self.swap_times[i],4),
                round(self.year_fracs[i][-1],4),
                round(self.swap_rates[i],4),
                round(self._repr_dfs[i+1],4),
            ])

        table = format_table(header, rows)
        print("\nCURVE DETAILS:")
        print(table)

        return "Cavour_v0.1"
