##############################################################################

##############################################################################

from enum import Enum
from numba import njit, float64, int64
import numpy as np
from jax import lax
import jax.numpy as jnp
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import CubicSpline
from ...utils.error import LibError
from ...utils.global_vars import g_small

###############################################################################


class InterpTypes(Enum):
    FLAT_FWD_RATES = 1
    LINEAR_FWD_RATES = 2
    LINEAR_ZERO_RATES = 4
    FINCUBIC_ZERO_RATES = 7
    NATCUBIC_LOG_DISCOUNT = 8
    NATCUBIC_ZERO_RATES = 9
    PCHIP_ZERO_RATES = 10
    PCHIP_LOG_DISCOUNT = 11


# LINEAR_SWAP_RATES = 3

###############################################################################
# TODO: GET RID OF THIS FUNCTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
###############################################################################

def interpolate(t: (float, np.ndarray),  # time or array of times
                times: np.ndarray,  # Vector of times on grid
                dfs: np.ndarray,  # Vector of discount factors
                method: int):  # Interpolation method which is value of enum
    """ Fast interpolation of discount factors at time x given discount factors
    at times provided using one of the methods in the enum InterpTypes. The
    value of x can be an array so that the function is vectorised. """

    if isinstance(t, (float, np.float64)):

        if t < 0.0:
            print(t)
            raise LibError("Interpolate times must all be >= 0")

        u = _uinterpolate(t, times, dfs, method)
        return u
    elif isinstance(t, np.ndarray):

        if np.any(t < 0.0):
            print(t)
            raise LibError("Interpolate times must all be >= 0")

        v = _vinterpolate(t, times, dfs, method)

        return v
    else:
        raise LibError("Unknown input type" + type(t))


###############################################################################


# @njit(float64(float64, float64[:], float64[:], int64),
#       fastmath=True, cache=True, nogil=True)
def _uinterpolate(t, times, dfs, method):
    """ Return the interpolated value of y given x and a vector of x and y.
    The values of x must be monotonic and increasing. The different schemes for
    interpolation are linear in y (as a function of x), linear in log(y) and
    piecewise flat in the continuously compounded forward y rate. """

    small = 1e-10
    num_points = times.size

    if t == times[0]:
        return dfs[0]

    i = 0
    while times[i] < t and i < num_points - 1:
        i = i + 1

    if t > times[i]:
        i = num_points

    yvalue = 0.0

    ###########################################################################
    # linear interpolation of y(x)
    ###########################################################################

    if method == InterpTypes.LINEAR_ZERO_RATES.value:

        if i == 1:
            r1 = -np.log(dfs[i]) / times[i]
            r2 = -np.log(dfs[i]) / times[i]
            dt = times[i] - times[i - 1]
            rvalue = ((times[i] - t) * r1 + (t - times[i - 1]) * r2) / dt
            yvalue = np.exp(-rvalue * t)
        elif i < num_points:
            r1 = -np.log(dfs[i - 1]) / times[i - 1]
            r2 = -np.log(dfs[i]) / times[i]
            dt = times[i] - times[i - 1]
            rvalue = ((times[i] - t) * r1 + (t - times[i - 1]) * r2) / dt
            yvalue = np.exp(-rvalue * t)
        else:
            r1 = -np.log(dfs[i - 1]) / times[i - 1]
            r2 = -np.log(dfs[i - 1]) / times[i - 1]
            dt = times[i - 1] - times[i - 2]
            rvalue = ((times[i - 1] - t) * r1 + (t - times[i - 2]) * r2) / dt
            yvalue = np.exp(-rvalue * t)

        return yvalue

    ###########################################################################
    # linear interpolation of log(y(x)) which means the linear interpolation of
    # continuously compounded zero rates in the case of discount discount
    # This is also FLAT FORWARDS
    ###########################################################################

    elif method == InterpTypes.FLAT_FWD_RATES.value:

        if i == 1:
            rt1 = -np.log(dfs[i - 1])
            rt2 = -np.log(dfs[i])
            dt = times[i] - times[i - 1]
            rtvalue = ((times[i] - t) * rt1 + (t - times[i - 1]) * rt2) / dt
            yvalue = np.exp(-rtvalue)
        elif i < num_points:
            rt1 = -np.log(dfs[i - 1])
            rt2 = -np.log(dfs[i])
            dt = times[i] - times[i - 1]
            rtvalue = ((times[i] - t) * rt1 + (t - times[i - 1]) * rt2) / dt
            yvalue = np.exp(-rtvalue)
        else:
            rt1 = -np.log(dfs[i - 2])
            rt2 = -np.log(dfs[i - 1])
            dt = times[i - 1] - times[i - 2]
            rtvalue = ((times[i - 1] - t) * rt1 +
                       (t - times[i - 2]) * rt2) / dt
            yvalue = np.exp(-rtvalue)

        return yvalue

    elif method == InterpTypes.LINEAR_FWD_RATES.value:

        if i == 1:
            y2 = -np.log(dfs[i] + small)
            yvalue = t * y2 / (times[i] + small)
            yvalue = np.exp(-yvalue)
        elif i < num_points:
            # If you get a math domain error it is because you need negativ
            fwd1 = -np.log(dfs[i - 1] / dfs[i - 2]) / \
                (times[i - 1] - times[i - 2])
            fwd2 = -np.log(dfs[i] / dfs[i - 1]) / (times[i] - times[i - 1])
            dt = times[i] - times[i - 1]
            fwd = ((times[i] - t) * fwd1 + (t - times[i - 1]) * fwd2) / dt
            yvalue = dfs[i - 1] * np.exp(-fwd * (t - times[i - 1]))
        else:
            fwd = -np.log(dfs[i - 1] / dfs[i - 2]) / \
                (times[i - 1] - times[i - 2])
            yvalue = dfs[i - 1] * np.exp(-fwd * (t - times[i - 1]))

        return yvalue

    else:
        print(method)
        raise LibError("Invalid interpolation scheme.")


###############################################################################

# @njit(float64[:](float64[:], float64[:], float64[:], int64),
#       fastmath=True, cache=True, nogil=True)
def _vinterpolate(xValues,
                  xvector,
                  dfs,
                  method):
    """ Return the interpolated values of y given x and a vector of x and y.
    The values of x must be monotonic and increasing. The different schemes for
    interpolation are linear in y (as a function of x), linear in log(y) and
    piecewise flat in the continuously compounded forward y rate. """

    n = xValues.size
    yvalues = np.empty(n)
    for i in range(0, n):
        yvalues[i] = _uinterpolate(xValues[i], xvector, dfs, method)

    return yvalues


###############################################################################


class Interpolator():

    def __init__(self,
                 interpolator_type: InterpTypes):

        self._interp_type = interpolator_type
        self._interp_fn = None
        self._times = None
        self._dfs = None
        self._refit_curve = False

    ###########################################################################

    def fit(self,
            times: np.ndarray,
            dfs: np.ndarray):

        self._times = times
        self._dfs = dfs

        if len(times) == 1:
            return

        if self._interp_type == InterpTypes.PCHIP_LOG_DISCOUNT:

            log_dfs = np.log(self._dfs)
            self._interp_fn = PchipInterpolator(self._times, log_dfs)

        elif self._interp_type == InterpTypes.PCHIP_ZERO_RATES:

            g_small_vector = np.ones(len(self._times)) * g_small
            zero_rates = -np.log(self._dfs) / (self._times + g_small_vector)

            if self._times[0] == 0.0:
                zero_rates[0] = zero_rates[1]

            self._interp_fn = PchipInterpolator(self._times, zero_rates)

        # if self._interp_type == InterpTypes.FINCUBIC_LOG_DISCOUNT:

        #     """ Second derivatives at left is zero and first derivative at
        #     right is clamped to zero. """
        #     log_dfs = np.log(self._dfs)
        #     self._interp_fn = CubicSpline(self._times, log_dfs,
        #                                  bc_type=((2, 0.0), (1, 0.0)))

        elif self._interp_type == InterpTypes.FINCUBIC_ZERO_RATES:

            """ Second derivatives at left is zero and first derivative at
            right is clamped to zero. """
            g_small_vector = np.ones(len(self._times)) * g_small
            zero_rates = -np.log(self._dfs) / (self._times + g_small_vector)

            if self._times[0] == 0.0:
                zero_rates[0] = zero_rates[1]

            self._interp_fn = CubicSpline(self._times, zero_rates,
                                          bc_type=((2, 0.0), (1, 0.0)))

        elif self._interp_type == InterpTypes.NATCUBIC_LOG_DISCOUNT:

            """ Second derivatives are clamped to zero at end points """
            log_dfs = np.log(self._dfs)
            self._interp_fn = CubicSpline(self._times, log_dfs,
                                          bc_type='natural')

        elif self._interp_type == InterpTypes.NATCUBIC_ZERO_RATES:

            """ Second derivatives are clamped to zero at end points """
            g_small_vector = np.ones(len(self._times)) * g_small
            zero_rates = -np.log(self._dfs) / (self._times + g_small_vector)

            if self._times[0] == 0.0:
                zero_rates[0] = zero_rates[1]

            self._interp_fn = CubicSpline(self._times, zero_rates,
                                          bc_type='natural')

    #        elif self._interp_type  == InterpTypes.LINEAR_LOG_DISCOUNT:
    #
    #            log_dfs = np.log(self._dfs)
    #            self._interp_fn = interp1d(self._times, log_dfs,
    #                                      fill_value="extrapolate")

    ###########################################################################

    # @njit(float64(float64, float64[:], float64[:], int64),
    #       fastmath=True, cache=True, nogil=True)
    def _uinterpolate(self, t, times, dfs, method):
        """ Return the interpolated value of y given x and a vector of x and y.
        The values of x must be monotonic and increasing. The different schemes for
        interpolation are linear in y (as a function of x), linear in log(y) and
        piecewise flat in the continuously compounded forward y rate. """

        small = 1e-10
        num_points = times.size

        if t == times[0]:
            return dfs[0]

        i = 0
        while times[i] < t and i < num_points - 1:
            i = i + 1

        if t > times[i]:
            i = num_points

        yvalue = 0.0

        ###########################################################################
        # linear interpolation of y(x)
        ###########################################################################

        if method == InterpTypes.LINEAR_ZERO_RATES.value:

            if i == 1:
                r1 = -jnp.log(dfs[i]) / times[i]
                r2 = -jnp.log(dfs[i]) / times[i]
                dt = times[i] - times[i - 1]
                rvalue = ((times[i] - t) * r1 + (t - times[i - 1]) * r2) / dt
                yvalue = jnp.exp(-rvalue * t)
            elif i < num_points:
                r1 = -jnp.log(dfs[i - 1]) / times[i - 1]
                r2 = -jnp.log(dfs[i]) / times[i]
                dt = times[i] - times[i - 1]
                rvalue = ((times[i] - t) * r1 + (t - times[i - 1]) * r2) / dt
                yvalue = jnp.exp(-rvalue * t)
            else:
                r1 = -jnp.log(dfs[i - 1]) / times[i - 1]
                r2 = -jnp.log(dfs[i - 1]) / times[i - 1]
                dt = times[i - 1] - times[i - 2]
                rvalue = ((times[i - 1] - t) * r1 + (t - times[i - 2]) * r2) / dt
                yvalue = jnp.exp(-rvalue * t)

            return yvalue

        ###########################################################################
        # linear interpolation of log(y(x)) which means the linear interpolation of
        # continuously compounded zero rates in the case of discount discount
        # This is also FLAT FORWARDS
        ###########################################################################

        elif method == InterpTypes.FLAT_FWD_RATES.value:

            if i == 1:
                rt1 = -jnp.log(dfs[i - 1])
                rt2 = -jnp.log(dfs[i])
                dt = times[i] - times[i - 1]
                rtvalue = ((times[i] - t) * rt1 + (t - times[i - 1]) * rt2) / dt
                yvalue = jnp.exp(-rtvalue)
            elif i < num_points:
                rt1 = -jnp.log(dfs[i - 1])
                rt2 = -jnp.log(dfs[i])
                dt = times[i] - times[i - 1]
                rtvalue = ((times[i] - t) * rt1 + (t - times[i - 1]) * rt2) / dt
                yvalue = jnp.exp(-rtvalue)
            else:
                rt1 = -jnp.log(dfs[i - 2])
                rt2 = -jnp.log(dfs[i - 1])
                dt = times[i - 1] - times[i - 2]
                rtvalue = ((times[i - 1] - t) * rt1 +
                        (t - times[i - 2]) * rt2) / dt
                yvalue = jnp.exp(-rtvalue)

            return yvalue

        elif method == InterpTypes.LINEAR_FWD_RATES.value:

            if i == 1:
                y2 = -jnp.log(dfs[i] + small)
                yvalue = t * y2 / (times[i] + small)
                yvalue = jnp.exp(-yvalue)
            elif i < num_points:
                # If you get a math domain error it is because you need negativ
                fwd1 = -jnp.log(dfs[i - 1] / dfs[i - 2]) / \
                    (times[i - 1] - times[i - 2])
                fwd2 = -jnp.log(dfs[i] / dfs[i - 1]) / (times[i] - times[i - 1])
                dt = times[i] - times[i - 1]
                fwd = ((times[i] - t) * fwd1 + (t - times[i - 1]) * fwd2) / dt
                yvalue = dfs[i - 1] * jnp.exp(-fwd * (t - times[i - 1]))
            else:
                fwd = -jnp.log(dfs[i - 1] / dfs[i - 2]) / \
                    (times[i - 1] - times[i - 2])
                yvalue = dfs[i - 1] * jnp.exp(-fwd * (t - times[i - 1]))

            return yvalue

        else:
            print(method)
            raise LibError("Invalid interpolation scheme.")
        

    ###############################################################################

    # @njit(float64[:](float64[:], float64[:], float64[:], int64),
    #       fastmath=True, cache=True, nogil=True)
    def _vinterpolate(self,
                    xValues,
                    xvector,
                    dfs,
                    method):
        """ Return the interpolated values of y given x and a vector of x and y.
        The values of x must be monotonic and increasing. The different schemes for
        interpolation are linear in y (as a function of x), linear in log(y) and
        piecewise flat in the continuously compounded forward y rate. """

        n = xValues.size
        yvalues = jnp.empty(n)
        for i in range(0, n):
            val = self._uinterpolate(xValues[i], xvector, dfs, method)
            yvalues = yvalues.at[i].set(val)

        # def body_fn(i, yvalues):
        #     x_i = lax.dynamic_index_in_dim(xValues, i, keepdims=False)
        #     val = self._uinterpolate(x_i, xvector, dfs, method)
        #     return yvalues.at[i].set(val)
        
        # yvalues = lax.fori_loop(0, n, body_fn, yvalues)

        if yvalues.size == 1:
            yvalues = yvalues.item()

        return yvalues
    
    
    ###########################################################################

    def simple_interpolate(self,
                    t: (float, np.ndarray),  # time or array of times
                    times: np.ndarray,  # Vector of times on grid
                    dfs: np.ndarray,  # Vector of discount factors
                    method: int):  # Interpolation method which is value of enum
        """ Fast interpolation of discount factors at time x given discount factors
        at times provided using one of the methods in the enum InterpTypes. The
        value of x can be an array so that the function is vectorised. """

        if isinstance(t, (float, np.float64)):

            if t < 0.0:
                print(t)
                raise LibError("Interpolate times must all be >= 0")

            u = self._uinterpolate(t, times, dfs, method)
            return u
        elif isinstance(t, np.ndarray):

            if np.any(t < 0.0):
                print(t)
                raise LibError("Interpolate times must all be >= 0")

            v = self._vinterpolate(t, times, dfs, method)

            return v
        else:
            raise LibError("Unknown input type" + type(t))


    ###########################################################################

    def interpolate(self,
                    t: float):
        """ Interpolation of discount factors at time x given discount factors
        at times provided using one of the methods in the enum InterpTypes.
        The value of x can be an array so that the function is vectorised. """

        if self._dfs is None:
            raise LibError("Dfs have not been set.")

        if isinstance(t, (float, np.float64)):

            if t < 0.0:
                print(t)
                raise LibError("Interpolate times must all be >= 0")

            if np.abs(t) < g_small:
                return 1.0

            tvec = np.array([t])

        elif isinstance(t, np.ndarray):

            if np.any(t < 0.0):
                print(t)
                raise LibError("Interpolate times must all be >= 0")

            tvec = t

        else:
            raise LibError("t is not a recognized type")

        if self._interp_type == InterpTypes.PCHIP_LOG_DISCOUNT:

            out = np.exp(self._interp_fn(tvec))

        elif self._interp_type == InterpTypes.PCHIP_ZERO_RATES:

            out = np.exp(-tvec * self._interp_fn(tvec))

        # if self._interp_type == InterpTypes.FINCUBIC_LOG_DISCOUNT:

        #     out = np.exp(self._interp_fn(tvec))

        elif self._interp_type == InterpTypes.FINCUBIC_ZERO_RATES:

            out = np.exp(-tvec * self._interp_fn(tvec))

        elif self._interp_type == InterpTypes.NATCUBIC_LOG_DISCOUNT:

            out = np.exp(self._interp_fn(tvec))

        elif self._interp_type == InterpTypes.NATCUBIC_ZERO_RATES:

            out = np.exp(-tvec * self._interp_fn(tvec))

        #        elif self._interp_type == InterpTypes.LINEAR_LOG_DISCOUNT:
        #
        #            out = np.exp(self._interp_fn(tvec))

        else:
            if isinstance(self._times, jnp.ndarray):
                v_times = self._times
            elif isinstance(self._times, list):
                v_times = jnp.array(self._times)
            else:
                raise LibError(f"{self._times} not a list or np.array")
            
            if isinstance(self._dfs, jnp.ndarray):
                v_dfs = self._dfs
            elif isinstance(self._dfs, list):
                v_dfs = jnp.array(self._dfs)
            else:
                raise LibError(f"{self._dfs} not a list or np.array")


            out = self._vinterpolate(tvec, v_times, v_dfs,
                                self._interp_type.value)

        # if isinstance(t, (float, np.float64)):
        #     return out[0]
        # else:
        #     return out

        return out

###############################################################################