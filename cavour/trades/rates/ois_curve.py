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
    ...     ois_swaps=swaps,
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
                 ois_swaps: list,
                 interp_type: InterpTypes = InterpTypes.FLAT_FWD_RATES,
                 check_refit: bool = False):  # Set to True to test it works
        """ Create an instance of an overnight index rate swap curve given a
        valuation date and a set of OIS rates. Some of these may
        be left None and the algorithm will just use what is provided. An
        interpolation method has also to be provided. The default is to use a
        linear interpolation for swap rates on coupon dates and to then assume
        flat forwards between these coupon dates.

        The curve will assign a discount factor of 1.0 to the valuation date.
        """

        check_argument_types(getattr(self, _func_name(), None), locals())

        self._value_dt = value_dt
        self._used_swaps = ois_swaps
        self._interp_type = interp_type
        self._check_refit = check_refit
        self._interpolator = None
        swap_rates = self._prepare_curve_builder_inputs()
        self._build_curve_ad(swap_rates)

###############################################################################

    def _prepare_curve_builder_inputs(self):
        """ Construct the discount curve using a bootstrap approach. This is
        the linear swap rate method that is fast and exact as it does not
        require the use of a solver. It is also market standard.
        Adjusted for algorithmic differentiation """

        self._dc_type = self._used_swaps[0]._float_leg._dc_type
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

        # I use the last coupon date for the swap rate interpolation as this
        # may be different from the maturity date due to a holiday adjustment
        # and the swap rates need to align with the coupon payment dates

        dcc = DayCount(self._dc_type)
        days_in_year = dcc.days_in_year()

        for swap in self._used_swaps:
            swap_rate = swap._fixed_coupon
            maturity_dt = swap._adjusted_fixed_dts[-1]
            tswap = (maturity_dt - self._value_dt) / days_in_year
            year_frac = swap._fixed_leg._year_fracs
            swap_times.append(tswap)
            swap_rates.append(swap_rate)
            year_fracs.append(year_frac)

        self.swap_times = swap_times
        self.swap_rates = swap_rates
        self.year_fracs = year_fracs

        return swap_rates

    def _build_curve_ad(self, swap_rates):

        pv01 = 0.0
        df_settle = 1 

        pv01_dict = {}
        pv01 = 0

        def calculate_single_df(pv01, i, target_maturity=None, step=0):
            if target_maturity is None:
                t_mat = self.swap_times[i]
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
                acc = self._used_swaps[i]._fixed_leg._year_fracs[-1-step]
                last_payment = sum(self._used_swaps[i]._fixed_leg._year_fracs[:-1-step])
                if round(last_payment , 2) not in pv01_dict:
                    step += 1
                    pv01_dict[round(last_payment,2)] = calculate_single_df(pv01, i, last_payment, step)

                pv01_end = (acc * swap_rate + 1)
                df_mat = (df_settle - swap_rate * pv01_dict[round(last_payment,2)]) / pv01_end
                pv01 = pv01_dict[round(last_payment,2)] + acc * df_mat

            self._times = jnp.append(self._times, t_mat)
            self._dfs = jnp.append(self._dfs, df_mat)

            if target_maturity is None:
               self._repr_dfs = jnp.append(self._repr_dfs, df_mat)

            pv01_dict[round(t_mat,2)] = pv01

            step = 0

            return pv01
        
        for i in range(0, len(self._used_swaps)):
            pv01 = calculate_single_df(pv01, i)

        return self._times, self._dfs

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
        """ Ensure that the OIS curve refits the calibration instruments. """


        for swap in self._used_swaps:
            # We value it as of the start date of the swap
            v = swap.value(swap._effective_dt, self,
                           None)
            v = v / swap._notional
            if abs(v) > swap_tol:
                print("Swap with maturity " + str(swap._maturity_dt)
                      + " Not Repriced. Has Value", v)
                swap.print_fixed_leg_pv()
                swap.print_float_leg_pv()
                raise LibError(f"Swap with maturity {swap._maturity_dt} not repriced. Difference is {abs(v)}")

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
