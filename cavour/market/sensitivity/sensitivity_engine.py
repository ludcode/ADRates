"""
SensitivityEngine: Centralized computation of DELTA and GAMMA via automatic differentiation.

Eliminates ~1,000 lines of duplicated chain rule code across Engine._compute_*() methods.

The chain rule for risk sensitivities:
- DELTA: dV/d(rates) = (dV/d(DFs)) × (d(DFs)/d(rates))
- GAMMA: d²V/d(rates)² = (d²V/d(DFs)²) × (d(DFs)/d(rates))² + (dV/d(DFs)) × (d²(DFs)/d(rates)²)

This module provides a single implementation that all instrument pricing methods can reuse.
"""

import numpy as np
import jax.numpy as jnp
from jax import grad, hessian
from typing import Callable, Optional

from cavour.requests.results import Delta, Gamma
from cavour.utils.helpers import to_tenor


class SensitivityEngine:
    """
    Centralized computation of DELTA and GAMMA using automatic differentiation.

    Provides static methods for computing first-order (DELTA) and second-order (GAMMA)
    sensitivities using the chain rule with JAX gradients.
    """

    @staticmethod
    def compute_delta(
        pv_fn: Callable[[jnp.ndarray], jnp.ndarray],
        dfs: jnp.ndarray,
        jac: jnp.ndarray,
        swap_times: list,
        currency,
        curve_type
    ) -> Delta:
        """
        Compute DELTA using chain rule: dV/d(rates) = (dV/d(DFs)) × (d(DFs)/d(rates))

        Args:
            pv_fn: JAX function computing PV from discount factors: pv_fn(dfs) -> scalar
            dfs: Current discount factors (array of shape [n_dfs])
            jac: Jacobian d(DFs)/d(rates) from curve (shape [n_dfs, n_rates])
            swap_times: List of swap tenor times for risk ladder labels
            currency: Currency for results (CurrencyTypes enum)
            curve_type: Curve type for results (CurveTypes enum)

        Returns:
            Delta object with risk_ladder in basis points (1bp = 0.01%)

        Notes:
            - pv_fn must return a scalar (or array that can be squeezed to scalar)
            - Risk ladder is multiplied by 1e-4 to convert to 1bp sensitivity units
            - This is the "sensies" convention used throughout the codebase
        """
        # Compute gradient of PV with respect to discount factors
        grad_dfs = grad(lambda d: jnp.squeeze(pv_fn(d)))(dfs)

        # Apply chain rule: sensitivity to rates = grad_dfs @ jacobian
        sensitivities = jnp.dot(grad_dfs, jac)

        # Convert to basis point units (1bp = 0.01% = 0.0001)
        sensies = [float(x) * 1e-4 for x in sensitivities]

        return Delta(
            risk_ladder=sensies,
            tenors=to_tenor(swap_times),
            currency=currency,
            curve_type=curve_type
        )

    @staticmethod
    def compute_gamma(
        pv_fn: Callable[[jnp.ndarray], jnp.ndarray],
        dfs: jnp.ndarray,
        jac: jnp.ndarray,
        hess_curve: jnp.ndarray,
        grad_dfs: Optional[jnp.ndarray],
        swap_times: list,
        currency,
        curve_type
    ) -> Gamma:
        """
        Compute GAMMA using chain rule with two terms.

        The second-order chain rule:
            d²V/d(rates)² = term1 + term2

        where:
            term1 = jac.T @ (d²V/d(DFs)²) @ jac      # Hessian of PV transformed
            term2 = sum_i (dV/dDF_i) * (d²DF_i/d(rates)²)  # First derivative times curve Hessian

        Args:
            pv_fn: JAX function computing PV from discount factors
            dfs: Current discount factors
            jac: Jacobian d(DFs)/d(rates) from curve
            hess_curve: Hessian d²(DFs)/d(rates)² from curve
                       Can be full (n_dfs, n_rates, n_rates) or diagonal (n_dfs, n_rates)
            grad_dfs: Optional pre-computed gradient dV/d(DFs). If None, will be computed.
                     Passing pre-computed gradient avoids redundant computation if also computing DELTA.
            swap_times: List of swap tenor times for risk ladder labels
            currency: Currency for results
            curve_type: Curve type for results

        Returns:
            Gamma object with risk_ladder matrix in basis points squared (bp²)

        Notes:
            - Result is multiplied by 1e-8 to convert to bp² units (1bp² = (0.01%)² = 1e-8)
            - Hessian can be diagonal-only for performance (22-25x speedup, empirically accurate)
            - If hess_curve is None, assumes curve has no Hessian (term2 = 0)
        """
        # Compute gradient if not provided (reuse from DELTA if available)
        if grad_dfs is None:
            grad_dfs = grad(lambda d: jnp.squeeze(pv_fn(d)))(dfs)

        # Compute Hessian of PV with respect to discount factors
        hess_dfs = hessian(lambda d: jnp.squeeze(pv_fn(d)))(dfs)

        # Term 1: Transform PV Hessian via Jacobian
        # jac.T @ hess_dfs @ jac gives sensitivity of sensitivities
        term1 = jac.T @ hess_dfs @ jac

        # Term 2: Contribution from curve Hessian
        # Accounts for fact that DFs themselves are nonlinear in rates
        if hess_curve is not None:
            if hess_curve.ndim == 2:
                # Diagonal Hessian: shape (n_dfs, n_rates)
                # Each DF has diagonal second derivatives w.r.t. rates
                term2_diag = jnp.dot(grad_dfs, hess_curve)
                term2 = jnp.diag(term2_diag)
            elif hess_curve.ndim == 3:
                # Full Hessian: shape (n_dfs, n_rates, n_rates)
                # Sum over DFs: grad_dfs[i] * hess_curve[i, :, :]
                term2 = jnp.sum(grad_dfs[:, None, None] * hess_curve, axis=0)
            else:
                raise ValueError(f"hess_curve must be 2D (diagonal) or 3D (full), got shape {hess_curve.shape}")
        else:
            # No curve Hessian available (term2 = 0)
            term2 = jnp.zeros_like(term1)

        # Combine terms and convert to basis point squared units
        gammas = term1 + term2
        gammas = np.array(gammas, dtype=np.float64) * 1e-8

        return Gamma(
            risk_ladder=gammas,
            tenors=to_tenor(swap_times),
            currency=currency,
            curve_type=curve_type
        )

    @staticmethod
    def compute_delta_gamma(
        pv_fn: Callable[[jnp.ndarray], jnp.ndarray],
        dfs: jnp.ndarray,
        jac: jnp.ndarray,
        hess_curve: Optional[jnp.ndarray],
        swap_times: list,
        currency,
        curve_type
    ) -> tuple[Delta, Gamma]:
        """
        Compute both DELTA and GAMMA efficiently (shares gradient computation).

        When both DELTA and GAMMA are needed, this method avoids redundant computation
        of the gradient dV/d(DFs) by computing it once and reusing it.

        Args:
            pv_fn: JAX function computing PV from discount factors
            dfs: Current discount factors
            jac: Jacobian d(DFs)/d(rates) from curve
            hess_curve: Hessian d²(DFs)/d(rates)² from curve (or None)
            swap_times: List of swap tenor times
            currency: Currency for results
            curve_type: Curve type for results

        Returns:
            Tuple of (delta, gamma) objects

        Notes:
            - More efficient than calling compute_delta() and compute_gamma() separately
            - Gradient is computed once and shared between DELTA and GAMMA calculations
        """
        # Compute gradient once
        grad_dfs = grad(lambda d: jnp.squeeze(pv_fn(d)))(dfs)

        # DELTA using pre-computed gradient
        sensitivities = jnp.dot(grad_dfs, jac)
        sensies = [float(x) * 1e-4 for x in sensitivities]
        delta = Delta(
            risk_ladder=sensies,
            tenors=to_tenor(swap_times),
            currency=currency,
            curve_type=curve_type
        )

        # GAMMA using pre-computed gradient
        gamma = SensitivityEngine.compute_gamma(
            pv_fn=pv_fn,
            dfs=dfs,
            jac=jac,
            hess_curve=hess_curve,
            grad_dfs=grad_dfs,  # Reuse gradient
            swap_times=swap_times,
            currency=currency,
            curve_type=curve_type
        )

        return delta, gamma
