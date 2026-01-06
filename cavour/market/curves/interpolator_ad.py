"""
JAX-compatible interpolation schemes for automatic differentiation.

Provides InterpolatorAd class with various interpolation methods:
- FLAT_FWD_RATES: Piecewise flat forward rates (market standard)
- LINEAR_ZERO_RATES: Linear interpolation on zero rates
- LINEAR_FWD_RATES: Linear interpolation on discount factors
- PCHIP_LOG_DISCOUNT: Monotonic Hermite splines on log(DF)
- PCHIP_ZERO_RATES: Monotonic Hermite splines on zero rates
- Cubic spline variants (natural, financial)

All methods support reverse-mode AD via JAX for computing sensitivities.
"""

import jax
import jax.numpy as jnp
from functools import partial
from enum import Enum
from scipy.interpolate import PchipInterpolator, CubicSpline
from cavour.utils.error import LibError
from cavour.utils.global_vars import g_small
from cavour.utils.global_types import InterpTypes

def _compute_pchip_slopes(x, y):
    """
    Compute monotonic cubic Hermite (PCHIP) slopes for shape-preserving interpolation.

    Args:
        x (jnp.ndarray): Time points (must be strictly increasing)
        y (jnp.ndarray): Values at time points

    Returns:
        jnp.ndarray: Derivative values at each point ensuring monotonicity

    Note:
        Uses weighted harmonic mean to preserve monotonicity between points.
    """
    h = x[1:] - x[:-1]
    m = (y[1:] - y[:-1]) / h
    d = jnp.empty_like(y)
    d = d.at[0].set(m[0])
    d = d.at[-1].set(m[-1])
    def _compute_di(i, val):
        cond = (m[i-1] * m[i]) > 0
        w1 = 2 * h[i] + h[i-1]
        w2 = h[i] + 2 * h[i-1]
        di = jnp.where(cond,
                       (w1 + w2) / ((w1 / m[i-1]) + (w2 / m[i])),
                       0.0)
        return val.at[i].set(di)
    d = jax.lax.fori_loop(1, x.size-1, _compute_di, d)
    return d

@jax.jit
def _pchip_eval(t, x, y, d):
    """
    Evaluate PCHIP interpolation at target time using Hermite basis functions.

    Args:
        t (float): Target time for evaluation
        x (jnp.ndarray): Knot points
        y (jnp.ndarray): Values at knots
        d (jnp.ndarray): Derivative values at knots

    Returns:
        float: Interpolated value at time t
    """
    idx = jnp.clip(jnp.searchsorted(x, t) - 1, 0, x.size-2)
    x0 = x[idx]; x1 = x[idx+1]
    y0 = y[idx]; y1 = y[idx+1]
    d0 = d[idx]; d1 = d[idx+1]
    h = x1 - x0
    s = (t - x0) / h
    h00 = 2*s**3 - 3*s**2 + 1
    h10 = s**3 - 2*s**2 + s
    h01 = -2*s**3 + 3*s**2
    h11 = s**3 - s**2
    return h00*y0 + h10*h*d0 + h01*y1 + h11*h*d1

@jax.jit
def _cubic_eval(t, x, c_coef):
    """
    Evaluate cubic spline at target time using precomputed coefficients.

    Args:
        t (float): Target time
        x (jnp.ndarray): Knot points
        c_coef (jnp.ndarray): Cubic coefficients (4 x n-1 array)

    Returns:
        float: Interpolated value
    """
    idx = jnp.clip(jnp.searchsorted(x, t) - 1, 0, x.size-2)
    u = t - x[idx]
    c0 = c_coef[0, idx]; c1 = c_coef[1, idx]
    c2 = c_coef[2, idx]; c3 = c_coef[3, idx]
    return ((c0*u + c1)*u + c2)*u + c3

@jax.jit
def _linear_interp(t, x, y):
    """
    Linear interpolation between curve points.

    Args:
        t (float): Target time
        x (jnp.ndarray): Time points
        y (jnp.ndarray): Values at time points

    Returns:
        float: Linearly interpolated value at t
    """
    idx = jnp.clip(jnp.searchsorted(x, t) - 1, 0, x.size-2)
    x0 = x[idx]; x1 = x[idx+1]
    y0 = y[idx]; y1 = y[idx+1]
    w = (t - x0) / (x1 - x0)
    return (1 - w)*y0 + w*y1

class InterpolatorAd:
    """
    JAX-compatible interpolator for automatic differentiation of curves.

    Supports multiple interpolation schemes via InterpTypes enum. The fit()
    method precomputes coefficients for complex methods (PCHIP, cubic splines),
    while simple methods compute on-the-fly in simple_interpolate().

    All methods are JIT-compiled for performance and support reverse-mode AD.

    Attributes:
        _interp_type (InterpTypes): Interpolation scheme
        _times (jnp.ndarray): Fitted time points
        _dfs (jnp.ndarray): Fitted discount factors
        _pchip_y (jnp.ndarray): Transformed values for PCHIP
        _pchip_d (jnp.ndarray): PCHIP slopes
        _cubic_coef (jnp.ndarray): Cubic spline coefficients
    """
    def __init__(self, interpolator_type: InterpTypes):
        self._interp_type = interpolator_type
        self._times = None
        self._dfs = None
        self._pchip_y = None
        self._pchip_d = None
        self._cubic_coef = None

    def fit(self, times, dfs):
        """
        Precompute interpolation coefficients for the given curve points.

        Args:
            times (array-like): Time points in years
            dfs (array-like): Discount factors at each time point

        Note:
            For PCHIP and cubic methods, computes transformed values (log, zero rates)
            and spline coefficients. Simple methods (flat forward, linear) do not
            require preprocessing.
        """
        x = jnp.array(times)
        d = jnp.array(dfs)
        self._times = x
        self._dfs = d
        if x.size == 1:
            return
        if self._interp_type == InterpTypes.PCHIP_LOG_DISCOUNT:
            y = jnp.log(d)
            self._pchip_y = y
            self._pchip_d = _compute_pchip_slopes(x, y)
        elif self._interp_type == InterpTypes.PCHIP_ZERO_RATES:
            zero = -jnp.log(d) / (x + g_small)
            zero = zero.at[0].set(jnp.where(x[0]==0, zero[1], zero[0]))
            self._pchip_y = zero
            self._pchip_d = _compute_pchip_slopes(x, zero)
        elif self._interp_type in (InterpTypes.FINCUBIC_ZERO_RATES,
                                    InterpTypes.NATCUBIC_ZERO_RATES,
                                    InterpTypes.NATCUBIC_LOG_DISCOUNT):
            if self._interp_type == InterpTypes.NATCUBIC_LOG_DISCOUNT:
                y = jnp.log(d)
                bc = 'natural'
            else:
                zero = -jnp.log(d) / (x + g_small)
                zero = zero.at[0].set(jnp.where(x[0]==0, zero[1], zero[0]))
                y = zero
                bc = ((2, 0.0), (1, 0.0)) if self._interp_type == InterpTypes.FINCUBIC_ZERO_RATES else 'natural'
            cs = CubicSpline(times, jnp.array(y), bc_type=bc)
            self._cubic_coef = jnp.array(cs.c)

    @partial(jax.jit, static_argnums=(0,4))
    def simple_interpolate(self, t, times, dfs, method):
        """
        Fast interpolation for simple methods (flat forward, linear zero/forward rates).

        Args:
            t (float | jnp.ndarray): Target time(s) for interpolation
            times (jnp.ndarray): Curve time points
            dfs (jnp.ndarray): Discount factors at curve points
            method (int): InterpTypes enum value

        Returns:
            float | jnp.ndarray: Interpolated discount factor(s)

        Raises:
            LibError: If method is not one of LINEAR_ZERO_RATES, FLAT_FWD_RATES,
                      or LINEAR_FWD_RATES

        Note:
            This method is JIT-compiled with static method argument. For complex
            methods (PCHIP, cubic), use interpolate() instead.
        """
        x = jnp.array(times)
        d = jnp.array(dfs)
        def _eval_scalar(tt):
            # if tt < 0:
            #     raise LibError("Interpolate times must all be >= 0")

            # Check if we're at an exact grid point to avoid NaN gradients
            # For exact grid points, return exact value; otherwise interpolate
            from jax import lax

            distances = jnp.abs(tt - x)
            min_dist = jnp.min(distances)
            grid_idx = jnp.argmin(distances)
            at_grid = min_dist < 1e-10

            # Compute interpolated value
            eps = 1e-12
            tt_adjusted = tt + eps

            if method == InterpTypes.LINEAR_ZERO_RATES.value:
                # Avoid division by zero for x[0]=0
                r = -jnp.log(d) / jnp.maximum(x, 1e-15)
                interp_result = jnp.exp(-jnp.interp(tt_adjusted, x, r) * tt)
            elif method == InterpTypes.FLAT_FWD_RATES.value:
                rt = -jnp.log(d)
                interp_result = jnp.exp(-jnp.interp(tt_adjusted, x, rt))
            elif method == InterpTypes.LINEAR_FWD_RATES.value:
                interp_result = jnp.interp(tt_adjusted, x, d)
            else:
                raise LibError("Invalid interpolation scheme.")

            # Use lax.select to avoid computing gradients through inactive branch
            # This prevents NaN when differentiating at exact grid points
            result = lax.select(at_grid, d[grid_idx], interp_result)

            return result
        #return jnp.where(isinstance(t, jnp.ndarray), jax.vmap(_eval_scalar)(t) , _eval_scalar(t))
        tt = jnp.atleast_1d(t)
        out = jax.vmap(_eval_scalar)(tt)
        if tt.shape == (1,):
            return out[0]
        return out

    @partial(jax.jit, static_argnums=(0,))
    def interpolate(self, t: float):
        """
        General interpolation method supporting all InterpTypes.

        Requires fit() to be called first for complex methods. Dispatches to
        appropriate evaluation function based on self._interp_type.

        Args:
            t (float | jnp.ndarray): Target time(s) in years

        Returns:
            float | jnp.ndarray: Interpolated discount factor(s)

        Raises:
            LibError: If fit() has not been called (self._dfs is None)
        """
        if self._dfs is None:
            raise LibError("Dfs have not been set.")
        tt = jnp.atleast_1d(t)
        # assume t >= 0; negative checks should be done before calling this JIT-compiled method
        if self._interp_type == InterpTypes.PCHIP_LOG_DISCOUNT:
            out = jax.vmap(lambda tv: jnp.exp(_pchip_eval(tv, self._times, self._pchip_y, self._pchip_d)))(tt)
        elif self._interp_type == InterpTypes.PCHIP_ZERO_RATES:
            out = jax.vmap(lambda tv: jnp.exp(-tv * _pchip_eval(tv, self._times, self._pchip_y, self._pchip_d)))(tt)
        elif self._interp_type in (InterpTypes.FINCUBIC_ZERO_RATES,
                                   InterpTypes.NATCUBIC_ZERO_RATES,
                                   InterpTypes.NATCUBIC_LOG_DISCOUNT):
            if self._interp_type == InterpTypes.NATCUBIC_LOG_DISCOUNT:
                func = lambda tv: jnp.exp(_cubic_eval(tv, self._times, self._cubic_coef))
            else:
                func = lambda tv: jnp.exp(-tv * _cubic_eval(tv, self._times, self._cubic_coef))
            out = jax.vmap(func)(tt)
        else:
            out = self.simple_interpolate(tt, self._times, self._dfs, self._interp_type.value)
        return out[0] if out.size==1 else out
