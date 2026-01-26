##############################################################################

##############################################################################

"""
Greek Validation Framework for Cavour

This module provides comprehensive validation of automatic differentiation (AD)
based sensitivities against full revaluation using curve bumps. It enables
rigorous testing of delta, gamma, and cross-gamma calculations by comparing
AD results to finite difference approximations and Taylor series expansions.

Key Features:
- Delta validation: Compare AD deltas to finite difference (FD) deltas
- Gamma validation: Validate via Taylor series expansion accuracy
- Cross-gamma validation: Validate multi-curve second-order sensitivities
- Comprehensive error reporting with tenor-level breakdowns
- Support for OIS and cross-currency swaps
- Automatic handling of curve rebuild dependencies (XCCY cascade)

Validation Approaches:

1. DELTA VALIDATION (Finite Difference Comparison):
   For each curve tenor:
   - Bump rate up/down by small amount (e.g., 1bp)
   - Rebuild curve and revalue derivative
   - Compute FD delta: (PV_up - PV_down) / (2 * bump)
   - Compare to AD delta from Jacobian
   - Report relative errors by tenor

2. GAMMA VALIDATION (Taylor Series Expansion):
   For parallel shock to all tenors:
   - Shock all rates by large amount (e.g., 100bp)
   - Compute PV via full revaluation: PV_shocked
   - Compute PV via 1st-order Taylor: PV_0 + delta * dR
   - Compute PV via 2nd-order Taylor: PV_0 + delta * dR + 0.5 * gamma * dR^2
   - Compare errors: 2nd-order should be significantly more accurate

3. CROSS-GAMMA VALIDATION (Multi-Curve Sensitivities):
   For pairs of curves (e.g., GBP OIS vs USD OIS):
   - Bump both curves simultaneously
   - Compute cross-derivative via finite difference
   - Compare to AD cross-gamma from Hessian
   - Validate cross-curve risk interactions

Example Usage:
    >>> # Build model and derivative
    >>> model = build_model(...)
    >>> swap = create_ois_swap(...)
    >>>
    >>> # Compute Greeks via AD
    >>> result = swap.position(model).compute([
    ...     RequestTypes.VALUE,
    ...     RequestTypes.DELTA,
    ...     RequestTypes.GAMMA
    ... ])
    >>>
    >>> # Validate Greeks
    >>> from cavour.utils.greek_validation import GreekValidator
    >>> validator = GreekValidator(model, swap)
    >>>
    >>> # Validate deltas vs finite difference
    >>> delta_report = validator.validate_delta_vs_fd(
    ...     result,
    ...     bump_bp=1.0,
    ...     tolerance=0.0001  # 0.01% relative error
    ... )
    >>> print(f"Delta validation passed: {delta_report.passed}")
    >>> print(f"Max error: {delta_report.max_relative_error:.6f}")
    >>>
    >>> # Validate gamma via Taylor expansion
    >>> gamma_report = validator.validate_gamma_taylor_expansion(
    ...     result,
    ...     shock_bp=100.0,
    ...     tolerance=0.05  # 5% error for 2nd-order Taylor
    ... )
    >>> print(f"2nd-order Taylor error: {gamma_report.error_2nd_order_pct:.2%}")
    >>>
    >>> # Plot error breakdown
    >>> delta_report.plot()
    >>> gamma_report.plot_taylor_comparison()

Tolerance Guidelines:
- Delta validation: 0.01% - 0.1% relative error (1e-4 to 1e-3)
- Gamma validation (100bp shock): 5% error for 2nd-order Taylor
- Gamma validation (10bp shock): 0.5% error for 2nd-order Taylor
- Cross-gamma: 1% relative error (higher due to mixed partials)

Mathematical Background:

Taylor Series Expansion for P&L:
    PV(r + dR) = PV(r) + delta * dR + 0.5 * gamma * dR^2 + O(dR^3)

Gamma via Chain Rule (two terms):
    d^2(PV)/dr^2 = (dPV/dDF) * (d^2DF/dr^2) + (d^2PV/dDF^2) * (dDF/dr)^2

Cross-Gamma via Mixed Partials:
    d^2(PV)/(dr1 * dr2) = central difference over both curves
"""

import numpy as np
import jax.numpy as jnp
from jax import grad, jacrev
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import copy

from cavour.utils.date import Date
from cavour.utils.global_types import CurveTypes, RequestTypes
from cavour.market.position.engine import Engine
from cavour.requests.results import AnalyticsResult, Delta, Gamma, Risk
from cavour.trades.rates.ois import OIS
from cavour.trades.rates.xccy_basis_swap import XccyBasisSwap

##############################################################################
# CURVE SCENARIO BUILDERS
##############################################################################

def _build_slope_scenario(shock_bp: float, tenors: List[str]) -> Dict[str, float]:
    """
    Build linear slope scenario: short-end +shock, long-end -shock.

    Creates a steepening or flattening scenario depending on shock sign.
    Shock magnitude decreases linearly from front to back of the curve.

    Args:
        shock_bp: Maximum shock size in basis points (applied to first tenor)
        tenors: List of tenor labels (e.g., ["1M", "3M", "6M", "1Y", "5Y"])

    Returns:
        Dictionary mapping tenor -> shock in basis points

    Example:
        >>> _build_slope_scenario(100.0, ["1M", "3M", "1Y", "5Y", "10Y"])
        {"1M": 100.0, "3M": 50.0, "1Y": 0.0, "5Y": -50.0, "10Y": -100.0}

        For positive shock_bp: steepening (shorts up, longs down)
        For negative shock_bp: flattening (shorts down, longs up)
    """
    n = len(tenors)
    if n < 2:
        raise ValueError("Need at least 2 tenors for slope scenario")

    shocks = {}
    for i, tenor in enumerate(tenors):
        # Linear interpolation from +shock to -shock
        weight = 1.0 - (2.0 * i / (n - 1))  # Ranges from +1.0 to -1.0
        shocks[tenor] = weight * shock_bp

    return shocks


def _build_skew_scenario(shock_bp: float, tenors: List[str]) -> Dict[str, float]:
    """
    Build skew/twist scenario: belly up, wings down (or vice versa).

    Creates a butterfly-like shock where intermediate tenors move most,
    and short/long ends move less. Useful for testing convexity in the
    middle of the curve.

    Args:
        shock_bp: Maximum shock size in basis points (applied to middle tenor)
        tenors: List of tenor labels

    Returns:
        Dictionary mapping tenor -> shock in basis points

    Example:
        >>> _build_skew_scenario(100.0, ["1M", "3M", "1Y", "5Y", "10Y"])
        {"1M": 0.0, "3M": 75.0, "1Y": 100.0, "5Y": 75.0, "10Y": 0.0}

        Shock peaks at the middle tenor and decreases towards the wings.
    """
    n = len(tenors)
    if n < 3:
        raise ValueError("Need at least 3 tenors for skew scenario")

    shocks = {}
    mid_idx = n // 2

    for i, tenor in enumerate(tenors):
        # Quadratic: peak at middle, zero at edges
        # Distance from middle normalized to [0, 1]
        dist_from_mid = abs(i - mid_idx) / max(mid_idx, n - mid_idx - 1)
        # Shock decreases quadratically from middle
        shocks[tenor] = shock_bp * (1 - dist_from_mid**2)

    return shocks


def _build_butterfly_scenario(shock_bp: float, tenors: List[str]) -> Dict[str, float]:
    """
    Build butterfly scenario: wings up, belly down (or vice versa).

    Opposite of skew - wings move in one direction, belly moves opposite.
    Classic butterfly trade structure.

    Args:
        shock_bp: Maximum shock size in basis points (applied to wings)
        tenors: List of tenor labels

    Returns:
        Dictionary mapping tenor -> shock in basis points

    Example:
        >>> _build_butterfly_scenario(100.0, ["1M", "3M", "1Y", "5Y", "10Y"])
        {"1M": 100.0, "3M": 25.0, "1Y": -100.0, "5Y": 25.0, "10Y": 100.0}

        Wings (short and long ends) shocked up, belly shocked down.
    """
    n = len(tenors)
    if n < 3:
        raise ValueError("Need at least 3 tenors for butterfly scenario")

    shocks = {}
    mid_idx = n // 2

    for i, tenor in enumerate(tenors):
        # Distance from middle normalized to [0, 1]
        dist_from_mid = abs(i - mid_idx) / max(mid_idx, n - mid_idx - 1)
        # Shock: positive at wings, negative at belly (quadratic)
        # Inverted parabola: -1 at middle, +1 at edges
        shocks[tenor] = shock_bp * (2 * dist_from_mid**2 - 1)

    return shocks


##############################################################################
# VALIDATION REPORT DATACLASSES
##############################################################################

@dataclass
class DeltaValidationReport:
    """
    Results of delta validation comparing AD deltas to finite difference deltas.

    Attributes:
        curve_type: Type of curve validated (e.g., GBP_OIS_SONIA)
        tenors: List of tenor labels (e.g., ["1Y", "2Y", "5Y"])
        delta_ad: AD-computed deltas by tenor (from Jacobian)
        delta_fd: Finite difference deltas by tenor (from bump-and-reprice)
        absolute_errors: Absolute errors by tenor
        relative_errors: Relative errors by tenor (percentage)
        max_absolute_error: Maximum absolute error across all tenors
        max_relative_error: Maximum relative error across all tenors
        mean_absolute_error: Mean absolute error
        mean_relative_error: Mean relative error
        bump_bp: Bump size used for FD (in basis points)
        fd_method: Finite difference method ('central' or 'forward')
        tolerance: Tolerance threshold for pass/fail (relative error)
        passed: Whether validation passed (max_relative_error < tolerance)
    """
    curve_type: CurveTypes
    tenors: List[str]
    delta_ad: Dict[str, float]
    delta_fd: Dict[str, float]
    absolute_errors: Dict[str, float]
    relative_errors: Dict[str, float]
    max_absolute_error: float
    max_relative_error: float
    mean_absolute_error: float
    mean_relative_error: float
    bump_bp: float
    fd_method: str
    tolerance: float
    passed: bool

    def to_dataframe(self) -> pd.DataFrame:
        """Export validation results to pandas DataFrame."""
        df = pd.DataFrame({
            'Tenor': self.tenors,
            'Delta_AD': [self.delta_ad[t] for t in self.tenors],
            'Delta_FD': [self.delta_fd[t] for t in self.tenors],
            'Abs_Error': [self.absolute_errors[t] for t in self.tenors],
            'Rel_Error_%': [self.relative_errors[t] * 100 for t in self.tenors],
        })
        return df

    def __str__(self) -> str:
        """Pretty-print validation report."""
        header = f"\n{'='*70}\n"
        header += f"DELTA VALIDATION REPORT: {self.curve_type.name}\n"
        header += f"{'='*70}\n"

        summary = f"Bump size: {self.bump_bp} bp ({self.fd_method} difference)\n"
        summary += f"Tolerance: {self.tolerance*100:.4f}%\n"
        summary += f"Status: {'PASSED' if self.passed else 'FAILED'}\n\n"

        stats = f"Max absolute error: {self.max_absolute_error:.6f}\n"
        stats += f"Max relative error: {self.max_relative_error*100:.6f}%\n"
        stats += f"Mean absolute error: {self.mean_absolute_error:.6f}\n"
        stats += f"Mean relative error: {self.mean_relative_error*100:.6f}%\n\n"

        table = self.to_dataframe().to_string(index=False)

        return header + summary + stats + table


@dataclass
class GammaValidationReport:
    """
    Results of gamma validation via Taylor series expansion.

    Validates that second-order Taylor expansion using AD gammas provides
    accurate approximation of P&L changes under large curve shocks.

    Attributes:
        shock_bp: Size of parallel shock applied (in basis points)
        pv_base: Base case PV (unshocked)
        pv_shocked: PV after applying shock (full revaluation)
        pv_taylor_1st: PV estimated via 1st-order Taylor (using deltas only)
        pv_taylor_2nd: PV estimated via 2nd-order Taylor (using deltas + gammas)
        error_1st_order: Absolute error for 1st-order Taylor
        error_2nd_order: Absolute error for 2nd-order Taylor
        error_1st_order_pct: Relative error for 1st-order Taylor (%)
        error_2nd_order_pct: Relative error for 2nd-order Taylor (%)
        gamma_improvement_factor: How much gamma improves accuracy (error_1st / error_2nd)
        tolerance: Tolerance threshold for 2nd-order error (relative)
        passed: Whether 2nd-order error is within tolerance

        # Gamma matrix validation
        gamma_matrix_symmetric: Whether gamma matrix is symmetric
        max_symmetry_error: Maximum asymmetry in gamma matrix
        diagonal_gammas: Dictionary of diagonal gamma elements by tenor
        max_offdiag_gamma: Maximum off-diagonal gamma element
    """
    shock_bp: float
    pv_base: float
    pv_shocked: float
    pv_taylor_1st: float
    pv_taylor_2nd: float
    error_1st_order: float
    error_2nd_order: float
    error_1st_order_pct: float
    error_2nd_order_pct: float
    gamma_improvement_factor: float
    tolerance: float
    passed: bool

    # Gamma matrix properties
    gamma_matrix_symmetric: bool
    max_symmetry_error: float
    diagonal_gammas: Dict[str, float]
    max_offdiag_gamma: float

    def plot_taylor_comparison(self):
        """
        Create visualization comparing Taylor approximations to full revaluation.

        Returns bar chart showing:
        - Base PV
        - 1st-order Taylor PV
        - 2nd-order Taylor PV
        - Full revaluation PV
        """
        try:
            import plotly.graph_objects as go

            fig = go.Figure(data=[
                go.Bar(name='Base PV', x=['PV'], y=[self.pv_base]),
                go.Bar(name='1st Order Taylor', x=['PV'], y=[self.pv_taylor_1st]),
                go.Bar(name='2nd Order Taylor', x=['PV'], y=[self.pv_taylor_2nd]),
                go.Bar(name='Full Revaluation', x=['PV'], y=[self.pv_shocked]),
            ])

            fig.update_layout(
                title=f'Taylor Expansion Validation (Shock: {self.shock_bp}bp)',
                yaxis_title='Present Value',
                barmode='group'
            )

            fig.show()
        except ImportError:
            print("Plotly not available. Install with: pip install plotly")

    def __str__(self) -> str:
        """Pretty-print validation report."""
        header = f"\n{'='*70}\n"
        header += f"GAMMA VALIDATION REPORT (Taylor Series Expansion)\n"
        header += f"{'='*70}\n"

        summary = f"Parallel shock: {self.shock_bp} bp\n"
        summary += f"Tolerance: {self.tolerance*100:.4f}%\n"
        summary += f"Status: {'PASSED' if self.passed else 'FAILED'}\n\n"

        pv_section = f"Present Values:\n"
        pv_section += f"  Base (unshocked):        {self.pv_base:>15,.2f}\n"
        pv_section += f"  Full revaluation:        {self.pv_shocked:>15,.2f}\n"
        pv_section += f"  1st-order Taylor:        {self.pv_taylor_1st:>15,.2f}\n"
        pv_section += f"  2nd-order Taylor:        {self.pv_taylor_2nd:>15,.2f}\n\n"

        error_section = f"Taylor Approximation Errors:\n"
        error_section += f"  1st-order error:         {self.error_1st_order:>15,.2f}  ({self.error_1st_order_pct:>6.2%})\n"
        error_section += f"  2nd-order error:         {self.error_2nd_order:>15,.2f}  ({self.error_2nd_order_pct:>6.2%})\n"
        error_section += f"  Gamma improvement:       {self.gamma_improvement_factor:>15.2f}x\n\n"

        gamma_section = f"Gamma Matrix Properties:\n"
        gamma_section += f"  Symmetric:               {'Yes' if self.gamma_matrix_symmetric else 'No'}\n"
        gamma_section += f"  Max symmetry error:      {self.max_symmetry_error:.6e}\n"
        gamma_section += f"  Max off-diagonal gamma:  {self.max_offdiag_gamma:.6f}\n"

        return header + summary + pv_section + error_section + gamma_section


@dataclass
class CrossGammaValidationReport:
    """
    Results of cross-gamma validation between two curves.

    Cross-gamma measures how the delta to one curve changes when a different
    curve moves. For XCCY swaps, this captures interaction between domestic,
    foreign, and basis curves.

    Attributes:
        curve_type_1: First curve type
        curve_type_2: Second curve type
        tenors_1: Tenors for first curve
        tenors_2: Tenors for second curve
        cross_gamma_ad: AD-computed cross-gamma matrix [N1, N2]
        cross_gamma_fd: FD-computed cross-gamma matrix [N1, N2]
        absolute_errors: Absolute errors for each [i, j] element
        relative_errors: Relative errors for each [i, j] element
        max_absolute_error: Maximum absolute error
        max_relative_error: Maximum relative error
        mean_absolute_error: Mean absolute error
        mean_relative_error: Mean relative error
        bump_bp: Bump size used for FD
        tolerance: Tolerance threshold
        passed: Whether validation passed
    """
    curve_type_1: CurveTypes
    curve_type_2: CurveTypes
    tenors_1: List[str]
    tenors_2: List[str]
    cross_gamma_ad: np.ndarray  # [N1, N2]
    cross_gamma_fd: np.ndarray  # [N1, N2]
    absolute_errors: np.ndarray  # [N1, N2]
    relative_errors: np.ndarray  # [N1, N2]
    max_absolute_error: float
    max_relative_error: float
    mean_absolute_error: float
    mean_relative_error: float
    bump_bp: float
    tolerance: float
    passed: bool

    def to_dataframe_ad(self) -> pd.DataFrame:
        """Export AD cross-gamma matrix to DataFrame."""
        df = pd.DataFrame(
            self.cross_gamma_ad,
            index=self.tenors_1,
            columns=self.tenors_2
        )
        df.index.name = f'{self.curve_type_1.name}'
        df.columns.name = f'{self.curve_type_2.name}'
        return df

    def to_dataframe_errors(self) -> pd.DataFrame:
        """Export relative error matrix to DataFrame."""
        df = pd.DataFrame(
            self.relative_errors * 100,  # Convert to percentage
            index=self.tenors_1,
            columns=self.tenors_2
        )
        df.index.name = f'{self.curve_type_1.name}'
        df.columns.name = f'{self.curve_type_2.name}'
        return df

    def __str__(self) -> str:
        """Pretty-print validation report."""
        header = f"\n{'='*70}\n"
        header += f"CROSS-GAMMA VALIDATION REPORT\n"
        header += f"{self.curve_type_1.name} vs {self.curve_type_2.name}\n"
        header += f"{'='*70}\n"

        summary = f"Bump size: {self.bump_bp} bp\n"
        summary += f"Tolerance: {self.tolerance*100:.4f}%\n"
        summary += f"Status: {'PASSED' if self.passed else 'FAILED'}\n\n"

        stats = f"Max absolute error: {self.max_absolute_error:.6f}\n"
        stats += f"Max relative error: {self.max_relative_error*100:.6f}%\n"
        stats += f"Mean absolute error: {self.mean_absolute_error:.6f}\n"
        stats += f"Mean relative error: {self.mean_relative_error*100:.6f}%\n\n"

        ad_table = "AD Cross-Gamma Matrix:\n" + self.to_dataframe_ad().to_string() + "\n\n"
        error_table = "Relative Errors (%):\n" + self.to_dataframe_errors().to_string()

        return header + summary + stats + ad_table + error_table


@dataclass
class DeltaParallelValidationReport:
    """
    Results of parallel shift delta validation.

    Validates that the sum of all tenor deltas matches the finite difference
    delta computed via a parallel curve shift. This ensures that individual
    tenor sensitivities correctly aggregate to capture uniform rate movements.

    Attributes:
        curve_type: Type of curve validated
        shock_bp: Parallel shock size (in basis points)
        delta_ad_sum: Sum of all tenor deltas from AD
        delta_fd_parallel: Finite difference delta via parallel shift
        absolute_error: Absolute difference between AD and FD
        relative_error: Relative error (percentage)
        tenor_contributions: Breakdown of delta by tenor
        tolerance: Tolerance threshold for pass/fail
        passed: Whether validation passed
    """
    curve_type: CurveTypes
    shock_bp: float
    delta_ad_sum: float
    delta_fd_parallel: float
    absolute_error: float
    relative_error: float
    tenor_contributions: Dict[str, float]
    tolerance: float
    passed: bool

    def to_dataframe(self) -> pd.DataFrame:
        """Export tenor contributions to pandas DataFrame."""
        df = pd.DataFrame({
            'Tenor': list(self.tenor_contributions.keys()),
            'Delta_Contribution': list(self.tenor_contributions.values())
        })
        return df

    def __str__(self) -> str:
        """Pretty-print validation report."""
        header = f"\n{'='*70}\n"
        header += f"DELTA PARALLEL SHIFT VALIDATION: {self.curve_type.name}\n"
        header += f"{'='*70}\n"

        summary = f"Parallel shock: {self.shock_bp} bp\n"
        summary += f"Tolerance: {self.tolerance*100:.4f}%\n"
        summary += f"Status: {'PASSED' if self.passed else 'FAILED'}\n\n"

        results = f"Aggregate Deltas:\n"
        results += f"  AD sum(deltas):      {self.delta_ad_sum:>15,.4f}\n"
        results += f"  FD parallel shift:   {self.delta_fd_parallel:>15,.4f}\n"
        results += f"  Absolute error:      {self.absolute_error:>15,.4f}\n"
        results += f"  Relative error:      {self.relative_error*100:>14,.4f}%\n\n"

        contrib_table = "Tenor Contributions:\n" + self.to_dataframe().to_string(index=False)

        return header + summary + results + contrib_table


@dataclass
class DeltaNonlinearityReport:
    """
    Results of large bump non-linearity validation.

    Tests how well deltas (and optionally gammas) approximate P&L changes
    under large single-tenor shocks. Large shocks expose non-linearity,
    demonstrating the limitations of first-order Taylor approximation.

    Attributes:
        curve_type: Type of curve validated
        tenor: Tenor label that was shocked
        tenor_idx: Index of tenor in curve
        shock_bp: Shock size (in basis points, e.g., 200.0)
        pv_base: Base case PV (unshocked)
        pv_shocked: PV after applying large shock (full revaluation)
        pv_delta_approx: PV estimated via 1st-order Taylor (delta only)
        pv_delta_gamma_approx: PV estimated via 2nd-order Taylor (delta + gamma)
        error_delta_pct: Relative error for delta-only approximation
        error_delta_gamma_pct: Relative error for delta+gamma approximation
        gamma_improvement_factor: How much gamma improves accuracy (error_delta / error_delta_gamma)
        non_linearity_factor: Ratio of 200bp error to 1bp error (theoretical: ~200 for linear)
        tolerance_delta: Tolerance for delta-only error
        tolerance_delta_gamma: Tolerance for delta+gamma error
        passed_delta: Whether delta-only passed tolerance
        passed_delta_gamma: Whether delta+gamma passed tolerance
    """
    curve_type: CurveTypes
    tenor: str
    tenor_idx: int
    shock_bp: float
    pv_base: float
    pv_shocked: float
    pv_delta_approx: float
    pv_delta_gamma_approx: Optional[float]
    error_delta_pct: float
    error_delta_gamma_pct: Optional[float]
    gamma_improvement_factor: Optional[float]
    non_linearity_factor: float
    tolerance_delta: float
    tolerance_delta_gamma: float
    passed_delta: bool
    passed_delta_gamma: bool

    def __str__(self) -> str:
        """Pretty-print validation report."""
        header = f"\n{'='*70}\n"
        header += f"DELTA NON-LINEARITY VALIDATION: {self.curve_type.name}\n"
        header += f"Tenor: {self.tenor}, Shock: {self.shock_bp}bp\n"
        header += f"{'='*70}\n"

        summary = f"Delta tolerance: {self.tolerance_delta*100:.2f}%\n"
        if self.pv_delta_gamma_approx is not None:
            summary += f"Delta+Gamma tolerance: {self.tolerance_delta_gamma*100:.2f}%\n"
        summary += f"Status: {'PASSED' if (self.passed_delta and self.passed_delta_gamma) else 'FAILED'}\n\n"

        pv_section = f"Present Values:\n"
        pv_section += f"  Base (unshocked):        {self.pv_base:>15,.2f}\n"
        pv_section += f"  Full revaluation:        {self.pv_shocked:>15,.2f}\n"
        pv_section += f"  Delta approx (1st):      {self.pv_delta_approx:>15,.2f}\n"
        if self.pv_delta_gamma_approx is not None:
            pv_section += f"  Delta+Gamma (2nd):       {self.pv_delta_gamma_approx:>15,.2f}\n"
        pv_section += "\n"

        error_section = f"Approximation Errors:\n"
        error_section += f"  Delta-only error:        {self.error_delta_pct:>14.4f}%  [{'PASS' if self.passed_delta else 'FAIL'}]\n"
        if self.error_delta_gamma_pct is not None:
            error_section += f"  Delta+Gamma error:       {self.error_delta_gamma_pct:>14.4f}%  [{'PASS' if self.passed_delta_gamma else 'FAIL'}]\n"
        if self.gamma_improvement_factor is not None:
            error_section += f"  Gamma improvement:       {self.gamma_improvement_factor:>14.2f}x\n"
        error_section += f"  Non-linearity factor:    {self.non_linearity_factor:>14.2f}x\n"

        return header + summary + pv_section + error_section


@dataclass
class ScenarioResult:
    """
    Result for a single curve scenario validation.

    Represents the outcome of validating deltas (and optionally gammas) under
    one specific shock scenario (e.g., slope, skew, butterfly).

    Attributes:
        name: Scenario name (e.g., "slope_100bp", "skew_neg50bp")
        shock_dict: Dictionary mapping tenor -> shock in basis points
        pv_base: Base case PV (unshocked)
        pv_shocked: PV after applying scenario shocks (full revaluation)
        pv_delta_approx: PV estimated via delta-only approximation
        error_abs: Absolute error for delta-only |PV_shocked - PV_delta_approx|
        error_pct: Relative error for delta-only (percentage)
        passed: Whether delta-only error is within tolerance
        pv_delta_gamma_approx: PV estimated via delta+gamma approximation (optional)
        error_delta_gamma_abs: Absolute error for delta+gamma (optional)
        error_delta_gamma_pct: Relative error for delta+gamma (optional)
        gamma_improvement_factor: error_delta / error_delta_gamma (optional)
        passed_delta_gamma: Whether delta+gamma error is within tolerance (optional)
    """
    name: str
    shock_dict: Dict[str, float]
    pv_base: float
    pv_shocked: float
    pv_delta_approx: float
    error_abs: float
    error_pct: float
    passed: bool
    # Delta+Gamma fields (optional - only populated if gamma available)
    pv_delta_gamma_approx: Optional[float] = None
    error_delta_gamma_abs: Optional[float] = None
    error_delta_gamma_pct: Optional[float] = None
    gamma_improvement_factor: Optional[float] = None
    passed_delta_gamma: Optional[bool] = None


@dataclass
class ScenarioValidationReport:
    """
    Results of curve scenario validation (slope, skew, butterfly).

    Validates that deltas correctly capture P&L changes under non-uniform
    curve shocks. Each scenario applies a different shock pattern to test
    how well individual tenor deltas aggregate.

    Attributes:
        curve_type: Type of curve validated
        scenarios: List of individual scenario results
        tolerance: Tolerance threshold applied to all scenarios
        all_passed: Whether all scenarios passed validation
    """
    curve_type: CurveTypes
    scenarios: List[ScenarioResult]
    tolerance: float
    all_passed: bool

    def to_dataframe(self) -> pd.DataFrame:
        """Export all scenario results to pandas DataFrame (delta-only)."""
        df = pd.DataFrame([
            {
                'Scenario': s.name,
                'PV_Base': s.pv_base,
                'PV_Shocked': s.pv_shocked,
                'PV_Delta_Approx': s.pv_delta_approx,
                'Error_Abs': s.error_abs,
                'Error_%': s.error_pct * 100,
                'Passed': s.passed
            }
            for s in self.scenarios
        ])
        return df

    def to_dataframe_comparison(self) -> pd.DataFrame:
        """
        Export comparison of delta-only vs delta+gamma to pandas DataFrame.

        Returns:
            DataFrame with both approximations and improvement metrics
        """
        rows = []
        for s in self.scenarios:
            row = {
                'Scenario': s.name,
                'PV_Shocked': f"{s.pv_shocked:,.0f}",
                'Delta_Error_%': f"{s.error_pct * 100:.2f}",
                'Delta_Pass': s.passed
            }
            if s.pv_delta_gamma_approx is not None:
                row['DeltaGamma_Error_%'] = f"{s.error_delta_gamma_pct * 100:.4f}"
                row['Improvement'] = f"{s.gamma_improvement_factor:.1f}x"
                row['DG_Pass'] = s.passed_delta_gamma
            rows.append(row)

        return pd.DataFrame(rows)

    def __str__(self) -> str:
        """Pretty-print validation report."""
        # Check if gamma was used
        has_gamma = any(s.pv_delta_gamma_approx is not None for s in self.scenarios)

        header = f"\n{'='*90}\n" if has_gamma else f"\n{'='*70}\n"
        header += f"CURVE SCENARIO VALIDATION: {self.curve_type.name}\n"
        header += f"{'='*90}\n" if has_gamma else f"{'='*70}\n"

        summary = f"Number of scenarios: {len(self.scenarios)}\n"
        summary += f"Approximation: {'Delta + Gamma' if has_gamma else 'Delta Only'}\n"
        summary += f"Tolerance: {self.tolerance*100:.2f}%\n"

        if has_gamma:
            all_passed_gamma = all(s.passed_delta_gamma for s in self.scenarios if s.passed_delta_gamma is not None)
            summary += f"Status (Delta-only): {'ALL PASSED' if self.all_passed else 'SOME FAILED'}\n"
            summary += f"Status (Delta+Gamma): {'ALL PASSED' if all_passed_gamma else 'SOME FAILED'}\n\n"
        else:
            summary += f"Status: {'ALL PASSED' if self.all_passed else 'SOME FAILED'}\n\n"

        # Table
        if has_gamma:
            table = self.to_dataframe_comparison().to_string(index=False)
        else:
            table = self.to_dataframe().to_string(index=False)

        # Summary stats
        stats = f"\n\nSummary Statistics:\n"
        if has_gamma:
            errors_delta = [s.error_pct for s in self.scenarios]
            errors_gamma = [s.error_delta_gamma_pct for s in self.scenarios if s.error_delta_gamma_pct is not None]
            improvements = [s.gamma_improvement_factor for s in self.scenarios if s.gamma_improvement_factor is not None]

            stats += f"  Delta-only errors:  Max={max(errors_delta)*100:.2f}%  Mean={np.mean(errors_delta)*100:.2f}%\n"
            stats += f"  Delta+Gamma errors: Max={max(errors_gamma)*100:.4f}%  Mean={np.mean(errors_gamma)*100:.4f}%\n"
            stats += f"  Gamma improvement:  Max={max(improvements):.1f}x  Mean={np.mean(improvements):.1f}x\n"
        else:
            errors = [s.error_pct for s in self.scenarios]
            stats += f"  Max error:  {max(errors)*100:.4f}%\n"
            stats += f"  Mean error: {np.mean(errors)*100:.4f}%\n"
            stats += f"  Min error:  {min(errors)*100:.4f}%\n"

        return header + summary + table + stats


@dataclass
class MultiCurveValidationReport:
    """
    Results of multi-curve validation for XCCY swaps.

    Validates deltas across all curves involved in a cross-currency swap:
    domestic OIS, foreign OIS, and XCCY basis spreads. Aggregates individual
    curve validation reports and provides overall pass/fail status.

    Attributes:
        derivative_type: Type of derivative validated (e.g., "XccyBasisSwap")
        curves: List of curve types validated
        delta_reports: Dictionary mapping curve_type -> DeltaValidationReport
        all_passed: Whether all curves passed their validation
        tolerance: Tolerance threshold applied
    """
    derivative_type: str
    curves: List[CurveTypes]
    delta_reports: Dict[CurveTypes, DeltaValidationReport]
    all_passed: bool
    tolerance: float

    def to_dataframe(self) -> pd.DataFrame:
        """Export summary of all curve validations to pandas DataFrame."""
        rows = []
        for curve_type in self.curves:
            report = self.delta_reports[curve_type]
            rows.append({
                'Curve': curve_type.name,
                'Max_Abs_Error': report.max_absolute_error,
                'Max_Rel_Error_%': report.max_relative_error * 100,
                'Mean_Rel_Error_%': report.mean_relative_error * 100,
                'Passed': report.passed
            })
        return pd.DataFrame(rows)

    def __str__(self) -> str:
        """Pretty-print multi-curve validation report."""
        header = f"\n{'='*70}\n"
        header += f"MULTI-CURVE VALIDATION: {self.derivative_type}\n"
        header += f"{'='*70}\n"

        summary = f"Number of curves: {len(self.curves)}\n"
        summary += f"Tolerance: {self.tolerance*100:.4f}%\n"
        summary += f"Status: {'ALL PASSED' if self.all_passed else 'SOME FAILED'}\n\n"

        table = self.to_dataframe().to_string(index=False)

        details = "\n\nDetailed Reports:\n"
        for curve_type in self.curves:
            details += f"\n{'-'*70}\n"
            details += f"Curve: {curve_type.name}\n"
            details += f"{'-'*70}\n"
            report = self.delta_reports[curve_type]
            details += f"  Max relative error: {report.max_relative_error*100:.4f}%\n"
            details += f"  Mean relative error: {report.mean_relative_error*100:.4f}%\n"
            details += f"  Status: {'PASSED' if report.passed else 'FAILED'}\n"

        return header + summary + table + details


##############################################################################
# MAIN VALIDATOR CLASS
##############################################################################

class GreekValidator:
    """
    Validates automatic differentiation (AD) based Greeks against full revaluation.

    Provides comprehensive validation suite for delta, gamma, and cross-gamma
    calculations by comparing AD results to finite difference approximations
    and Taylor series expansions.

    Attributes:
        model: Cavour Model instance containing calibrated curves
        derivative: Derivative instrument to validate (OIS, XCCY swap, etc.)
        engine: Engine instance for computing analytics

    Example:
        >>> model = build_model(...)
        >>> swap = create_ois_swap(...)
        >>> validator = GreekValidator(model, swap)
        >>>
        >>> # Compute Greeks
        >>> result = swap.position(model).compute([VALUE, DELTA, GAMMA])
        >>>
        >>> # Validate
        >>> delta_report = validator.validate_delta_vs_fd(result)
        >>> print(delta_report)
    """

    def __init__(self, model, derivative, engine=None):
        """
        Initialize Greek validator.

        Args:
            model: Cavour Model with calibrated curves
            derivative: Derivative instrument (OIS, XCCY swap, etc.)
            engine: Optional Engine instance (will create if not provided)
        """
        self.model = model
        self.derivative = derivative
        self.engine = engine if engine is not None else Engine(model)

        # Cache base valuation for efficiency
        self._base_result = None

    def validate_delta_vs_fd(self,
                            ad_result: AnalyticsResult,
                            curve_type: Optional[CurveTypes] = None,
                            bump_bp: float = 1.0,
                            fd_method: str = 'central',
                            tolerance: float = 0.0001,
                            absolute_tolerance: float = 1e-2) -> DeltaValidationReport:
        """
        Validate AD deltas against finite difference deltas.

        For each tenor in the specified curve:
        1. Bump rate up (and down for central difference)
        2. Rebuild curve (and dependent curves for XCCY)
        3. Revalue derivative to get PV_bumped
        4. Compute FD delta: (PV_up - PV_down) / (2 * bump)
        5. Compare to AD delta from Jacobian

        Args:
            ad_result: AnalyticsResult containing AD deltas
            curve_type: Which curve to validate (if None, validates first curve in result)
            bump_bp: Bump size in basis points (default 1bp)
            fd_method: 'central' (more accurate) or 'forward' (faster)
            tolerance: Relative error tolerance (default 0.01%)
            absolute_tolerance: Absolute error tolerance for small deltas (default 0.01)

        Returns:
            DeltaValidationReport with detailed comparison

        Raises:
            ValueError: If curve_type not found in ad_result
        """
        # Get base PV
        pv_base = ad_result.value.amount

        # Determine which curve to validate
        # Handle both Delta (single curve) and Risk (multiple curves) containers
        if isinstance(ad_result.risk, Delta):
            # Single curve case - risk is a Delta object directly
            delta_obj = ad_result.risk
            if curve_type is None:
                curve_type = delta_obj.curve_type
        elif isinstance(ad_result.risk, Risk):
            # Multiple curves case - risk is a Risk container
            if curve_type is None:
                # Get first curve from risk result
                if hasattr(ad_result.risk, '_risk_dict'):
                    curve_type = list(ad_result.risk._risk_dict.keys())[0]
                else:
                    raise ValueError("Cannot determine curve_type automatically. Please specify explicitly.")
            # Get AD deltas for this curve
            delta_obj = ad_result.risk(curve_type)
        else:
            raise ValueError(f"Unexpected risk type: {type(ad_result.risk)}")
        tenors = delta_obj.tenors
        delta_ad_dict = self._extract_delta_ladder_as_dict(delta_obj)

        # Get curve name
        curve_name = self._get_curve_name_from_type(curve_type)

        # Compute FD deltas for each tenor
        delta_fd_dict = {}
        absolute_errors = {}
        relative_errors = {}

        # Convert bump from basis points to percentage points
        # 1bp = 0.01 percentage points (e.g., 5.19% + 1bp = 5.20%)
        bump_pct_points = bump_bp * 0.01

        for tenor_idx, tenor in enumerate(tenors):
            # Bump UP
            model_up = self._rebuild_model_with_bumped_curve(
                curve_name, tenor_idx, bump_pct_points
            )
            pv_up = self._compute_pv(model_up)

            if fd_method == 'central':
                # Bump DOWN
                model_down = self._rebuild_model_with_bumped_curve(
                    curve_name, tenor_idx, -bump_pct_points
                )
                pv_down = self._compute_pv(model_down)

                # Central difference: delta per 1bp
                # Divide by 2 * bump_pct_points to get delta per percentage point,
                # then the result is already per bp since we bumped by bp
                delta_fd = (pv_up - pv_down) / 2
            else:
                # Forward difference
                delta_fd = (pv_up - pv_base)

            delta_fd_dict[tenor] = delta_fd

            # Compute errors
            delta_ad = delta_ad_dict[tenor]
            abs_error = abs(delta_fd - delta_ad)
            absolute_errors[tenor] = abs_error

            # Relative error with protection for small deltas
            if abs(delta_ad) > absolute_tolerance:
                rel_error = abs_error / abs(delta_ad)
            else:
                # For very small deltas, use absolute error instead
                rel_error = abs_error
            relative_errors[tenor] = rel_error

        # Compute summary statistics
        max_abs_error = max(absolute_errors.values())
        max_rel_error = max(relative_errors.values())
        mean_abs_error = np.mean(list(absolute_errors.values()))
        mean_rel_error = np.mean(list(relative_errors.values()))

        # Determine pass/fail
        passed = max_rel_error < tolerance

        # Create report
        report = DeltaValidationReport(
            curve_type=curve_type,
            tenors=tenors,
            delta_ad=delta_ad_dict,
            delta_fd=delta_fd_dict,
            absolute_errors=absolute_errors,
            relative_errors=relative_errors,
            max_absolute_error=max_abs_error,
            max_relative_error=max_rel_error,
            mean_absolute_error=mean_abs_error,
            mean_relative_error=mean_rel_error,
            bump_bp=bump_bp,
            fd_method=fd_method,
            tolerance=tolerance,
            passed=passed
        )

        return report

    def validate_gamma_taylor_expansion(self,
                                       ad_result: AnalyticsResult,
                                       curve_type: Optional[CurveTypes] = None,
                                       shock_bp: float = 100.0,
                                       tolerance: float = 0.05,
                                       parallel_shock: bool = True) -> GammaValidationReport:
        """
        Validate gamma via Taylor series expansion accuracy.

        Applies large parallel shock to all curve tenors and compares:
        - PV from full revaluation (actual)
        - PV from 1st-order Taylor: PV_0 + delta * dR
        - PV from 2nd-order Taylor: PV_0 + delta * dR + 0.5 * gamma * dR^2

        Second-order Taylor should be significantly more accurate than first-order,
        demonstrating that gamma captures curvature correctly.

        Args:
            ad_result: AnalyticsResult containing AD deltas and gammas
            curve_type: Which curve to shock (if None, uses first curve)
            shock_bp: Shock size in basis points (default 100bp)
            tolerance: Tolerance for 2nd-order Taylor error (default 5%)
            parallel_shock: If True, shock all tenors equally; if False, shock each tenor individually

        Returns:
            GammaValidationReport with Taylor expansion comparison
        """
        # Get base PV
        pv_base = ad_result.value.amount

        # Determine which curve to validate
        # Handle both Delta/Gamma (single curve) and Risk (multiple curves) containers
        if isinstance(ad_result.risk, Delta):
            # Single curve case
            delta_obj = ad_result.risk
            gamma_obj = ad_result.gamma
            if curve_type is None:
                curve_type = delta_obj.curve_type
        elif isinstance(ad_result.risk, Risk):
            # Multiple curves case
            if curve_type is None:
                if hasattr(ad_result.risk, '_risk_dict'):
                    curve_type = list(ad_result.risk._risk_dict.keys())[0]
                else:
                    raise ValueError("Cannot determine curve_type automatically. Please specify explicitly.")
            delta_obj = ad_result.risk(curve_type)
            gamma_obj = ad_result.gamma(curve_type)
        else:
            raise ValueError(f"Unexpected risk type: {type(ad_result.risk)}")

        # Get curve name
        curve_name = self._get_curve_name_from_type(curve_type)

        # Extract deltas as array
        delta_dict = self._extract_delta_ladder_as_dict(delta_obj)
        delta_array = np.array([delta_dict[t] for t in delta_obj.tenors])

        # Extract gamma matrix
        if hasattr(gamma_obj, 'risk_ladder'):
            gamma_matrix = np.array(gamma_obj.risk_ladder)
        else:
            raise ValueError("Cannot extract gamma matrix from Gamma object")

        # Check gamma matrix symmetry
        symmetry_error = np.max(np.abs(gamma_matrix - gamma_matrix.T))
        is_symmetric = symmetry_error < 1e-10

        # Extract diagonal gammas
        n_tenors = len(delta_obj.tenors)
        diagonal_gammas = {delta_obj.tenors[i]: gamma_matrix[i, i] for i in range(n_tenors)}

        # Max off-diagonal gamma
        off_diag_mask = ~np.eye(n_tenors, dtype=bool)
        max_offdiag_gamma = np.max(np.abs(gamma_matrix[off_diag_mask])) if n_tenors > 1 else 0.0

        # Apply parallel shock and revalue
        # Convert shock from basis points to percentage points
        # e.g., 100bp = 1.0 percentage points
        shock_pct_points = shock_bp * 0.01
        model_shocked = self._rebuild_model_parallel_shock(curve_name, shock_pct_points)
        pv_shocked = self._compute_pv(model_shocked)

        # Create shock vector (same shock for all tenors in parallel shock)
        # shock_vector is in basis points for the Taylor expansion
        shock_vector = np.ones(n_tenors) * shock_bp

        # Compute Taylor expansions
        # 1st order: PV_0 + delta^T * dR
        pv_taylor_1st = pv_base + np.dot(delta_array, shock_vector)

        # 2nd order: PV_0 + delta^T * dR + 0.5 * dR^T * gamma * dR
        quadratic_term = 0.5 * np.dot(shock_vector, np.dot(gamma_matrix, shock_vector))
        pv_taylor_2nd = pv_base + np.dot(delta_array, shock_vector) + quadratic_term

        # Compute errors
        error_1st = abs(pv_shocked - pv_taylor_1st)
        error_2nd = abs(pv_shocked - pv_taylor_2nd)

        # Relative errors
        error_1st_pct = error_1st / abs(pv_shocked) if abs(pv_shocked) > 1e-10 else error_1st
        error_2nd_pct = error_2nd / abs(pv_shocked) if abs(pv_shocked) > 1e-10 else error_2nd

        # Improvement factor
        improvement_factor = error_1st / error_2nd if error_2nd > 1e-10 else np.inf

        # Pass/fail
        passed = error_2nd_pct < tolerance

        # Create report
        report = GammaValidationReport(
            shock_bp=shock_bp,
            pv_base=pv_base,
            pv_shocked=pv_shocked,
            pv_taylor_1st=pv_taylor_1st,
            pv_taylor_2nd=pv_taylor_2nd,
            error_1st_order=error_1st,
            error_2nd_order=error_2nd,
            error_1st_order_pct=error_1st_pct,
            error_2nd_order_pct=error_2nd_pct,
            gamma_improvement_factor=improvement_factor,
            tolerance=tolerance,
            passed=passed,
            gamma_matrix_symmetric=is_symmetric,
            max_symmetry_error=symmetry_error,
            diagonal_gammas=diagonal_gammas,
            max_offdiag_gamma=max_offdiag_gamma
        )

        return report

    def validate_cross_gamma(self,
                            ad_result: AnalyticsResult,
                            curve_type_1: CurveTypes,
                            curve_type_2: CurveTypes,
                            bump_bp: float = 1.0,
                            tolerance: float = 0.01) -> CrossGammaValidationReport:
        """
        Validate cross-gamma between two curves via double finite difference.

        Cross-gamma measures d^2(PV) / (d(curve1) * d(curve2)).

        For each pair of tenors (tenor1 from curve1, tenor2 from curve2):
        1. Compute PV with no bumps: PV_00
        2. Compute PV with curve1[tenor1] bumped: PV_10
        3. Compute PV with curve2[tenor2] bumped: PV_01
        4. Compute PV with both bumped: PV_11
        5. Cross-gamma FD = (PV_11 - PV_10 - PV_01 + PV_00) / (bump1 * bump2)
        6. Compare to AD cross-gamma from Hessian

        Args:
            ad_result: AnalyticsResult containing AD cross-gammas
            curve_type_1: First curve type
            curve_type_2: Second curve type
            bump_bp: Bump size in basis points
            tolerance: Relative error tolerance (default 1%)

        Returns:
            CrossGammaValidationReport with detailed comparison
        """
        # Get AD cross-gamma (cross-gammas are stored in gamma field, not risk)
        ad_cross_gamma = ad_result.gamma.cross_gamma(curve_type_1, curve_type_2)
        if ad_cross_gamma is None:
            raise ValueError(
                f"No cross-gamma found for {curve_type_1.name} vs {curve_type_2.name}. "
                f"Ensure model was computed with RequestTypes.GAMMA."
            )

        # Extract AD cross-gamma matrix and tenors
        ad_matrix = np.array(ad_cross_gamma.risk_matrix)  # [N1, N2]
        tenors_1 = ad_cross_gamma.tenors_curve1
        tenors_2 = ad_cross_gamma.tenors_curve2
        n1, n2 = len(tenors_1), len(tenors_2)

        # Convert bump from bp to decimal
        bump_decimal = bump_bp / 100.0

        # Step 1: Compute base PV (PV_00)
        pv_00 = self.engine.compute(self.derivative, [RequestTypes.VALUE]).value.amount

        # Step 2: Compute single-bumped PVs for curve1 (PV_10 for each tenor)
        pv_10_list = []
        for i in range(n1):
            model_10 = self._rebuild_model_with_bumped_curve(
                curve_name=curve_type_1.name,
                tenor_index=i,
                bump_amount=bump_decimal
            )
            pv_10 = Engine(model_10).compute(self.derivative, [RequestTypes.VALUE]).value.amount
            pv_10_list.append(pv_10)

        # Step 3: Compute single-bumped PVs for curve2 (PV_01 for each tenor)
        pv_01_list = []
        for j in range(n2):
            model_01 = self._rebuild_model_with_bumped_curve(
                curve_name=curve_type_2.name,
                tenor_index=j,
                bump_amount=bump_decimal
            )
            pv_01 = Engine(model_01).compute(self.derivative, [RequestTypes.VALUE]).value.amount
            pv_01_list.append(pv_01)

        # Step 4: Compute double-bumped PVs (PV_11 for each pair [i, j])
        # Need to bump both curves simultaneously
        fd_matrix = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                # Bump both curves - need a more sophisticated rebuild method
                # For now, use sequential bumping approach
                model_11 = self._rebuild_model_with_two_bumps(
                    curve_name_1=curve_type_1.name,
                    tenor_index_1=i,
                    bump_amount_1=bump_decimal,
                    curve_name_2=curve_type_2.name,
                    tenor_index_2=j,
                    bump_amount_2=bump_decimal
                )
                pv_11 = Engine(model_11).compute(self.derivative, [RequestTypes.VALUE]).value.amount

                # Compute cross-gamma via double finite difference
                pv_10 = pv_10_list[i]
                pv_01 = pv_01_list[j]

                # CrossGamma[i,j] = (PV_11 - PV_10 - PV_01 + PV_00) / (bump^2)
                fd_matrix[i, j] = (pv_11 - pv_10 - pv_01 + pv_00) / (bump_decimal ** 2)

        # Step 5: Compare AD vs FD element-wise
        absolute_errors = np.abs(ad_matrix - fd_matrix)

        # Relative error with safe division (use absolute error for near-zero values)
        relative_errors = np.zeros_like(absolute_errors)
        for i in range(n1):
            for j in range(n2):
                ad_val = ad_matrix[i, j]
                abs_err = absolute_errors[i, j]

                # Use relative error if AD value is significant (>0.01)
                # Otherwise use absolute error
                if abs(ad_val) > 0.01:
                    relative_errors[i, j] = abs_err / abs(ad_val)
                else:
                    relative_errors[i, j] = abs_err

        # Compute statistics
        max_abs_error = np.max(absolute_errors)
        max_rel_error = np.max(relative_errors)
        mean_abs_error = np.mean(absolute_errors)
        mean_rel_error = np.mean(relative_errors)

        # Pass if all relative errors are within tolerance
        passed = max_rel_error <= tolerance

        return CrossGammaValidationReport(
            curve_type_1=curve_type_1,
            curve_type_2=curve_type_2,
            tenors_1=tenors_1,
            tenors_2=tenors_2,
            cross_gamma_ad=ad_matrix,
            cross_gamma_fd=fd_matrix,
            absolute_errors=absolute_errors,
            relative_errors=relative_errors,
            max_absolute_error=max_abs_error,
            max_relative_error=max_rel_error,
            mean_absolute_error=mean_abs_error,
            mean_relative_error=mean_rel_error,
            bump_bp=bump_bp,
            tolerance=tolerance,
            passed=passed
        )

    def validate_delta_parallel_shift(
        self,
        ad_result: AnalyticsResult,
        curve_type: Optional[CurveTypes] = None,
        shock_bp: float = 100.0,
        tolerance: float = 0.001
    ) -> DeltaParallelValidationReport:
        """
        Validate that sum of tenor deltas matches parallel shift sensitivity.

        This validation ensures that individual tenor deltas correctly aggregate
        to capture uniform curve movements. Compares the sum of AD deltas across
        all tenors to a finite difference delta computed via parallel curve shift.

        Args:
            ad_result: AnalyticsResult containing AD deltas
            curve_type: Which curve to validate (if None, uses first curve)
            shock_bp: Parallel shock size in basis points (default 100bp)
            tolerance: Relative error tolerance (default 0.1%)

        Returns:
            DeltaParallelValidationReport with aggregate comparison

        Example:
            >>> validator = GreekValidator(model, swap)
            >>> report = validator.validate_delta_parallel_shift(
            ...     result, shock_bp=100.0, tolerance=0.001
            ... )
            >>> assert report.passed
        """
        # Get base PV
        pv_base = ad_result.value.amount

        # Determine which curve to validate
        if isinstance(ad_result.risk, Delta):
            delta_obj = ad_result.risk
            if curve_type is None:
                curve_type = delta_obj.curve_type
        elif isinstance(ad_result.risk, Risk):
            if curve_type is None:
                if hasattr(ad_result.risk, '_risk_dict'):
                    curve_type = list(ad_result.risk._risk_dict.keys())[0]
                else:
                    raise ValueError("Cannot determine curve_type automatically. Please specify explicitly.")
            delta_obj = ad_result.risk(curve_type)
        else:
            raise ValueError(f"Unexpected risk type: {type(ad_result.risk)}")

        # Get curve name
        curve_name = self._get_curve_name_from_type(curve_type)

        # Extract individual tenor deltas
        delta_dict = self._extract_delta_ladder_as_dict(delta_obj)
        tenor_contributions = delta_dict.copy()

        # Compute AD aggregate delta (sum of all tenors)
        delta_ad_sum = sum(delta_dict.values())

        # Compute FD delta via parallel shift
        # Convert shock from bp to percentage points
        shock_pct_points = shock_bp * 0.01

        # Bump UP (parallel)
        model_up = self._rebuild_model_parallel_shock(curve_name, shock_pct_points)
        pv_up = self._compute_pv(model_up)

        # Bump DOWN (parallel)
        model_down = self._rebuild_model_parallel_shock(curve_name, -shock_pct_points)
        pv_down = self._compute_pv(model_down)

        # Central difference: compute parallel delta per 1bp
        # We bumped by ±shock_pct_points (e.g., ±1.0 for 100bp)
        # (pv_up - pv_down) / 2 gives PV change for shock_bp basis points
        # AD deltas are per 1bp, so we need to normalize FD to per 1bp
        # Since we shocked by shock_bp, divide by shock_bp to get per 1bp FD
        # Then sum(AD deltas) should match this FD delta
        pv_change_for_shock = (pv_up - pv_down) / 2
        delta_fd_per_bp = pv_change_for_shock / shock_bp

        # For comparison: multiply by shock_bp to get back to shock_bp scale
        # Actually no - AD sum is per 1bp, so just compare directly
        delta_fd_parallel = delta_fd_per_bp

        # Compute errors
        absolute_error = abs(delta_ad_sum - delta_fd_parallel)

        # Relative error with protection
        if abs(delta_fd_parallel) > 1e-10:
            relative_error = absolute_error / abs(delta_fd_parallel)
        else:
            relative_error = absolute_error

        # Determine pass/fail
        passed = relative_error < tolerance

        # Create report
        report = DeltaParallelValidationReport(
            curve_type=curve_type,
            shock_bp=shock_bp,
            delta_ad_sum=delta_ad_sum,
            delta_fd_parallel=delta_fd_parallel,
            absolute_error=absolute_error,
            relative_error=relative_error,
            tenor_contributions=tenor_contributions,
            tolerance=tolerance,
            passed=passed
        )

        return report

    def validate_delta_large_bump(
        self,
        ad_result: AnalyticsResult,
        curve_type: Optional[CurveTypes] = None,
        tenor_idx: Optional[int] = None,
        shock_bp: float = 200.0,
        tolerance_delta: float = 0.20,
        tolerance_delta_gamma: float = 0.05
    ) -> DeltaNonlinearityReport:
        """
        Test non-linearity by shocking single tenor with large bump.

        Large shocks (e.g., 200bp) expose non-linear behavior in interest rate
        derivatives. This validation compares:
        1. Full revaluation (actual P&L)
        2. Delta approximation (1st-order Taylor): PV_0 + delta * dR
        3. Delta+Gamma approximation (2nd-order Taylor): PV_0 + delta*dR + 0.5*gamma*dR^2

        The delta-only approximation should have significant error for large moves,
        while delta+gamma should capture most of the P&L change.

        Args:
            ad_result: AnalyticsResult containing AD deltas (and optionally gammas)
            curve_type: Which curve to validate (if None, uses first curve)
            tenor_idx: Which tenor to shock (if None, uses largest delta tenor)
            shock_bp: Shock size in basis points (default 200bp)
            tolerance_delta: Tolerance for delta-only error (default 20%)
            tolerance_delta_gamma: Tolerance for delta+gamma error (default 5%)

        Returns:
            DeltaNonlinearityReport with non-linearity analysis

        Example:
            >>> report = validator.validate_delta_large_bump(
            ...     result, shock_bp=200.0, tolerance_delta=0.20
            ... )
            >>> assert report.error_delta_pct > 0.10  # Non-linearity visible
            >>> assert report.passed_delta_gamma  # Gamma helps significantly
        """
        # Get base PV
        pv_base = ad_result.value.amount

        # Determine which curve to validate
        if isinstance(ad_result.risk, Delta):
            delta_obj = ad_result.risk
            if curve_type is None:
                curve_type = delta_obj.curve_type
        elif isinstance(ad_result.risk, Risk):
            if curve_type is None:
                if hasattr(ad_result.risk, '_risk_dict'):
                    curve_type = list(ad_result.risk._risk_dict.keys())[0]
                else:
                    raise ValueError("Cannot determine curve_type automatically. Please specify explicitly.")
            delta_obj = ad_result.risk(curve_type)
        else:
            raise ValueError(f"Unexpected risk type: {type(ad_result.risk)}")

        # Get curve name and tenors
        curve_name = self._get_curve_name_from_type(curve_type)
        tenors = delta_obj.tenors

        # Extract deltas
        delta_dict = self._extract_delta_ladder_as_dict(delta_obj)
        delta_array = np.array([delta_dict[t] for t in tenors])

        # Determine which tenor to shock
        if tenor_idx is None:
            # Use tenor with largest absolute delta (most sensitive)
            tenor_idx = int(np.argmax(np.abs(delta_array)))

        tenor = tenors[tenor_idx]
        delta_tenor = delta_array[tenor_idx]

        # Apply large shock to single tenor
        shock_pct_points = shock_bp * 0.01
        model_shocked = self._rebuild_model_with_bumped_curve(
            curve_name, tenor_idx, shock_pct_points
        )
        pv_shocked = self._compute_pv(model_shocked)

        # Compute 1st-order Taylor approximation (delta only)
        pv_delta_approx = pv_base + delta_tenor * shock_bp

        # Compute 2nd-order Taylor approximation (delta + gamma) if available
        pv_delta_gamma_approx = None
        error_delta_gamma_pct = None
        gamma_improvement_factor = None
        passed_delta_gamma = True  # Default to True if gamma not available

        if ad_result.gamma is not None:
            # Extract gamma for this tenor
            if isinstance(ad_result.gamma, Gamma):
                gamma_obj = ad_result.gamma
            elif isinstance(ad_result.risk, Risk):
                gamma_obj = ad_result.gamma(curve_type)
            else:
                gamma_obj = None

            if gamma_obj is not None and hasattr(gamma_obj, 'risk_ladder'):
                gamma_matrix = np.array(gamma_obj.risk_ladder)
                gamma_ii = gamma_matrix[tenor_idx, tenor_idx]

                # 2nd-order Taylor: PV_0 + delta*dR + 0.5*gamma*dR^2
                pv_delta_gamma_approx = pv_base + delta_tenor * shock_bp + 0.5 * gamma_ii * shock_bp**2

                # Compute error
                error_delta_gamma = abs(pv_shocked - pv_delta_gamma_approx)
                error_delta_gamma_pct = error_delta_gamma / abs(pv_shocked) if abs(pv_shocked) > 1e-10 else error_delta_gamma
                passed_delta_gamma = error_delta_gamma_pct < tolerance_delta_gamma

        # Compute delta-only error
        error_delta = abs(pv_shocked - pv_delta_approx)
        error_delta_pct = error_delta / abs(pv_shocked) if abs(pv_shocked) > 1e-10 else error_delta
        passed_delta = error_delta_pct < tolerance_delta

        # Compute gamma improvement factor
        if error_delta_gamma_pct is not None and error_delta_gamma_pct > 1e-10:
            gamma_improvement_factor = error_delta_pct / error_delta_gamma_pct

        # Compute non-linearity factor (compare to 1bp baseline)
        # For small bumps, we'd expect linear behavior
        # For large bumps, error should scale with shock size squared (for quadratic instruments)
        # Theoretical: error_200bp / error_1bp ≈ 200 for linear delta breakdown
        # (This is approximate - actual factor depends on gamma magnitude)
        non_linearity_factor = shock_bp  # Simplified: just use shock ratio

        # Create report
        report = DeltaNonlinearityReport(
            curve_type=curve_type,
            tenor=tenor,
            tenor_idx=tenor_idx,
            shock_bp=shock_bp,
            pv_base=pv_base,
            pv_shocked=pv_shocked,
            pv_delta_approx=pv_delta_approx,
            pv_delta_gamma_approx=pv_delta_gamma_approx,
            error_delta_pct=error_delta_pct,
            error_delta_gamma_pct=error_delta_gamma_pct,
            gamma_improvement_factor=gamma_improvement_factor,
            non_linearity_factor=non_linearity_factor,
            tolerance_delta=tolerance_delta,
            tolerance_delta_gamma=tolerance_delta_gamma,
            passed_delta=passed_delta,
            passed_delta_gamma=passed_delta_gamma
        )

        return report

    def validate_curve_scenarios(
        self,
        ad_result: AnalyticsResult,
        curve_type: Optional[CurveTypes] = None,
        scenarios: Optional[List[Dict[str, Any]]] = None,
        tolerance: float = 0.05,
        use_gamma: bool = True
    ) -> ScenarioValidationReport:
        """
        Validate deltas (and optionally gammas) under slope, skew, and butterfly scenarios.

        Tests how well individual tenor deltas (and gammas) aggregate to capture P&L changes
        under non-uniform curve shocks. Each scenario applies a different shock
        pattern (steepening, belly-up, butterfly, etc.) to validate that Greeks
        correctly handle complex curve movements.

        Args:
            ad_result: AnalyticsResult containing AD deltas (and optionally gammas)
            curve_type: Which curve to validate (if None, uses first curve)
            scenarios: List of scenario definitions. Each dict should have:
                - 'type': 'slope', 'skew', or 'butterfly'
                - 'shock_bp': shock magnitude in basis points
                If None, uses default scenarios
            tolerance: Relative error tolerance (default 5%)
            use_gamma: If True and gamma available, compute delta+gamma approximation (default True)

        Returns:
            ScenarioValidationReport with results for all scenarios

        Example:
            >>> scenarios = [
            ...     {'type': 'slope', 'shock_bp': 100},
            ...     {'type': 'skew', 'shock_bp': 100},
            ...     {'type': 'butterfly', 'shock_bp': 50}
            ... ]
            >>> report = validator.validate_curve_scenarios(result, scenarios=scenarios)
            >>> assert report.all_passed
        """
        # Get base PV
        pv_base = ad_result.value.amount

        # Determine which curve to validate
        if isinstance(ad_result.risk, Delta):
            delta_obj = ad_result.risk
            if curve_type is None:
                curve_type = delta_obj.curve_type
        elif isinstance(ad_result.risk, Risk):
            if curve_type is None:
                if hasattr(ad_result.risk, '_risk_dict'):
                    curve_type = list(ad_result.risk._risk_dict.keys())[0]
                else:
                    raise ValueError("Cannot determine curve_type automatically. Please specify explicitly.")
            delta_obj = ad_result.risk(curve_type)
        else:
            raise ValueError(f"Unexpected risk type: {type(ad_result.risk)}")

        # Get curve name and tenors
        curve_name = self._get_curve_name_from_type(curve_type)
        # Use curve's original tenors from model params (not delta tenors which may differ)
        if curve_name not in self.model._curve_params_dict:
            raise ValueError(f"No stored parameters found for curve '{curve_name}'")
        curve_tenors = self.model._curve_params_dict[curve_name]["tenor_list"]

        # Get delta tenors (may differ in naming, e.g., '5Y' vs '5Y1M')
        delta_tenors = delta_obj.tenors

        # Extract deltas
        delta_dict = self._extract_delta_ladder_as_dict(delta_obj)
        delta_array = np.array([delta_dict[t] for t in delta_tenors])

        # Verify same length
        if len(curve_tenors) != len(delta_tenors):
            raise ValueError(f"Curve has {len(curve_tenors)} tenors but delta has {len(delta_tenors)} tenors")

        # Extract gamma matrix if requested and available
        gamma_matrix = None
        if use_gamma and ad_result.gamma is not None:
            try:
                if isinstance(ad_result.gamma, Gamma):
                    gamma_obj = ad_result.gamma
                elif isinstance(ad_result.risk, Risk):
                    gamma_obj = ad_result.gamma(curve_type)
                else:
                    gamma_obj = None

                if gamma_obj is not None and hasattr(gamma_obj, 'risk_ladder'):
                    gamma_matrix = np.array(gamma_obj.risk_ladder)
                    print(f"  Using delta+gamma approximation (gamma matrix: {gamma_matrix.shape})")
                else:
                    print(f"  Gamma not available, using delta-only approximation")
            except Exception as e:
                print(f"  Warning: Could not extract gamma: {e}. Using delta-only approximation")
                gamma_matrix = None

        # Define default scenarios if none provided
        if scenarios is None:
            scenarios = [
                {'type': 'slope', 'shock_bp': 100},
                {'type': 'slope', 'shock_bp': -100},  # Flattening
                {'type': 'skew', 'shock_bp': 100},
                {'type': 'butterfly', 'shock_bp': 100}
            ]

        # Validate each scenario
        scenario_results = []

        for scenario_spec in scenarios:
            scenario_type = scenario_spec['type']
            shock_bp = scenario_spec['shock_bp']

            # Build shock dictionary based on scenario type using curve tenors
            if scenario_type == 'slope':
                shock_dict = _build_slope_scenario(shock_bp, curve_tenors)
                name = f"slope_{'+' if shock_bp > 0 else ''}{shock_bp:.0f}bp"
            elif scenario_type == 'skew':
                shock_dict = _build_skew_scenario(shock_bp, curve_tenors)
                name = f"skew_{shock_bp:.0f}bp"
            elif scenario_type == 'butterfly':
                shock_dict = _build_butterfly_scenario(shock_bp, curve_tenors)
                name = f"butterfly_{shock_bp:.0f}bp"
            else:
                raise ValueError(f"Unknown scenario type: {scenario_type}")

            # Apply shock and revalue
            model_shocked = self._rebuild_model_with_shock_dict(curve_name, shock_dict)
            pv_shocked = self._compute_pv(model_shocked)

            # Compute delta approximation: sum(delta_i * shock_i)
            # Match curve tenors with delta tenors by index position
            pv_delta_approx = pv_base
            for i, curve_tenor in enumerate(curve_tenors):
                delta_tenor = delta_tenors[i]
                pv_delta_approx += delta_dict[delta_tenor] * shock_dict[curve_tenor]

            # Compute errors for delta-only
            error_abs = abs(pv_shocked - pv_delta_approx)
            error_pct = error_abs / abs(pv_shocked) if abs(pv_shocked) > 1e-10 else error_abs
            passed = error_pct < tolerance

            # Compute delta+gamma approximation if gamma available
            pv_delta_gamma_approx = None
            error_delta_gamma_abs = None
            error_delta_gamma_pct = None
            gamma_improvement_factor = None
            passed_delta_gamma = None

            if gamma_matrix is not None:
                # Build shock vector (aligned with delta tenors)
                shock_vector = np.array([shock_dict[curve_tenors[i]] for i in range(len(curve_tenors))])

                # Compute delta term: Σᵢ(Δᵢ × drᵢ)
                delta_term = np.dot(delta_array, shock_vector)

                # Compute gamma term: 0.5 × dR^T × Γ × dR
                gamma_term = 0.5 * np.dot(shock_vector, np.dot(gamma_matrix, shock_vector))

                # Total delta+gamma approximation
                pv_delta_gamma_approx = pv_base + delta_term + gamma_term

                # Compute errors for delta+gamma
                error_delta_gamma_abs = abs(pv_shocked - pv_delta_gamma_approx)
                error_delta_gamma_pct = error_delta_gamma_abs / abs(pv_shocked) if abs(pv_shocked) > 1e-10 else error_delta_gamma_abs

                # Improvement factor
                gamma_improvement_factor = error_abs / error_delta_gamma_abs if error_delta_gamma_abs > 1e-10 else np.inf

                # Pass/fail for delta+gamma
                passed_delta_gamma = error_delta_gamma_pct < tolerance

            # Create scenario result
            scenario_result = ScenarioResult(
                name=name,
                shock_dict=shock_dict,
                pv_base=pv_base,
                pv_shocked=pv_shocked,
                pv_delta_approx=pv_delta_approx,
                error_abs=error_abs,
                error_pct=error_pct,
                passed=passed,
                pv_delta_gamma_approx=pv_delta_gamma_approx,
                error_delta_gamma_abs=error_delta_gamma_abs,
                error_delta_gamma_pct=error_delta_gamma_pct,
                gamma_improvement_factor=gamma_improvement_factor,
                passed_delta_gamma=passed_delta_gamma
            )
            scenario_results.append(scenario_result)

        # Determine if all scenarios passed
        all_passed = all(s.passed for s in scenario_results)

        # Create report
        report = ScenarioValidationReport(
            curve_type=curve_type,
            scenarios=scenario_results,
            tolerance=tolerance,
            all_passed=all_passed
        )

        return report

    def validate_xccy_multi_curve(
        self,
        ad_result: AnalyticsResult,
        bump_bp: float = 1.0,
        tolerance: float = 0.001,  # 0.1% for multi-curve
        fd_method: str = 'central'
    ) -> MultiCurveValidationReport:
        """
        Validate all deltas for an XCCY swap simultaneously.

        Cross-currency swaps depend on multiple curves:
        - Domestic OIS curve
        - Foreign OIS curve
        - XCCY basis spread curve

        This method validates AD deltas for all curves involved, accounting for
        the cascade effect where bumping an OIS curve triggers XCCY curve rebuild.

        Args:
            ad_result: AnalyticsResult containing AD deltas for multiple curves
            bump_bp: Bump size in basis points (default 1bp)
            tolerance: Relative error tolerance (default 0.1% for multi-curve)
            fd_method: Finite difference method ('central' or 'forward')

        Returns:
            MultiCurveValidationReport with validation results for each curve

        Example:
            >>> # XCCY swap with USD domestic, GBP foreign
            >>> result = xccy_swap.position(model).compute([RequestTypes.VALUE, RequestTypes.DELTA])
            >>> report = validator.validate_xccy_multi_curve(result, bump_bp=1.0)
            >>> assert report.all_passed
            >>> print(report.to_dataframe())
        """
        # Extract deltas for all curves from Risk object
        if not isinstance(ad_result.risk, Risk):
            raise ValueError("AD result must contain a Risk object with deltas for multiple curves")

        # Get all curves that have deltas
        curves = list(ad_result.risk.deltas.keys())

        if len(curves) == 0:
            raise ValueError("No deltas found in AD result Risk object")

        # Validate each curve individually
        delta_reports = {}

        for curve_type in curves:
            # Extract delta for this specific curve
            delta_obj = ad_result.risk.deltas[curve_type]

            # Create single-curve result for validation
            # (validate_delta_vs_fd expects AnalyticsResult with single Delta)
            single_curve_result = AnalyticsResult(
                value=ad_result.value,
                risk=delta_obj  # Pass Delta object directly
            )

            # Run standard delta validation for this curve
            # This automatically handles XCCY cascade via _rebuild_model_with_bumped_curve()
            curve_report = self.validate_delta_vs_fd(
                ad_result=single_curve_result,
                curve_type=curve_type,
                bump_bp=bump_bp,
                tolerance=tolerance,
                fd_method=fd_method
            )

            delta_reports[curve_type] = curve_report

        # Determine if all curves passed
        all_passed = all(report.passed for report in delta_reports.values())

        # Create multi-curve report
        report = MultiCurveValidationReport(
            derivative_type=self.derivative_type,
            curves=curves,
            delta_reports=delta_reports,
            all_passed=all_passed,
            tolerance=tolerance
        )

        return report

    def validate_xccy_parallel_shift(
        self,
        ad_result: AnalyticsResult,
        shock_bp: float = 100.0,
        tolerance: float = 0.001,
        fd_method: str = 'central'
    ) -> MultiCurveValidationReport:
        """
        Validate parallel shift deltas across all XCCY curves.

        Applies a uniform parallel shock to each curve involved in the XCCY swap
        and validates that the delta-based P&L approximation matches the actual
        revalued P&L across all curves.

        This is the multi-curve version of validate_delta_parallel_shift().

        Args:
            ad_result: AnalyticsResult containing AD deltas for multiple curves
            shock_bp: Parallel shock size in basis points (default 100bp)
            tolerance: Relative error tolerance (default 0.1% for multi-curve)
            fd_method: Finite difference method ('central' or 'forward')

        Returns:
            MultiCurveValidationReport with parallel shift validation for each curve

        Example:
            >>> # XCCY swap with USD domestic, GBP foreign
            >>> result = xccy_swap.position(model).compute([RequestTypes.VALUE, RequestTypes.DELTA])
            >>> report = validator.validate_xccy_parallel_shift(result, shock_bp=100)
            >>> assert report.all_passed
            >>> print(report.to_dataframe())
        """
        # Extract deltas for all curves from Risk object
        if not isinstance(ad_result.risk, Risk):
            raise ValueError("AD result must contain a Risk object with deltas for multiple curves")

        # Get all curves that have deltas
        curves = list(ad_result.risk.deltas.keys())

        if len(curves) == 0:
            raise ValueError("No deltas found in AD result Risk object")

        # Validate parallel shift for each curve individually
        delta_reports = {}

        for curve_type in curves:
            # Extract delta for this specific curve
            delta_obj = ad_result.risk.deltas[curve_type]

            # Create single-curve result for validation
            single_curve_result = AnalyticsResult(
                value=ad_result.value,
                risk=delta_obj  # Pass Delta object directly
            )

            # Run parallel shift validation for this curve
            # This automatically handles XCCY cascade via _rebuild_model_parallel_shock()
            parallel_report = self.validate_delta_parallel_shift(
                ad_result=single_curve_result,
                curve_type=curve_type,
                shock_bp=shock_bp,
                tolerance=tolerance
            )

            # Convert DeltaParallelValidationReport to DeltaValidationReport format
            # for consistency with MultiCurveValidationReport
            delta_validation_report = DeltaValidationReport(
                derivative_type=self.derivative_type,
                curve_type=curve_type,
                tenor_deltas_ad=parallel_report.tenor_deltas_ad,
                tenor_deltas_fd=parallel_report.tenor_deltas_fd,
                errors_absolute=parallel_report.errors_absolute,
                errors_relative=parallel_report.errors_relative,
                max_absolute_error=parallel_report.max_absolute_error,
                max_relative_error=parallel_report.max_relative_error,
                mean_relative_error=parallel_report.mean_relative_error,
                tolerance=parallel_report.tolerance,
                passed=parallel_report.passed,
                bump_bp=shock_bp / 100.0,  # Convert to bp for reporting
                fd_method=fd_method
            )

            delta_reports[curve_type] = delta_validation_report

        # Determine if all curves passed
        all_passed = all(report.passed for report in delta_reports.values())

        # Create multi-curve report
        report = MultiCurveValidationReport(
            derivative_type=self.derivative_type,
            curves=curves,
            delta_reports=delta_reports,
            all_passed=all_passed,
            tolerance=tolerance
        )

        return report

    def validate_xccy_scenarios(
        self,
        ad_result: AnalyticsResult,
        scenarios: Optional[List[Dict[str, Any]]] = None,
        tolerance: float = 0.001
    ) -> Dict[CurveTypes, ScenarioValidationReport]:
        """
        Validate scenario deltas across all XCCY curves.

        Applies slope, skew, and butterfly scenarios to each curve involved in
        the XCCY swap and validates that deltas correctly capture P&L changes
        under non-uniform curve shocks.

        This is the multi-curve version of validate_curve_scenarios().

        Args:
            ad_result: AnalyticsResult containing AD deltas for multiple curves
            scenarios: List of scenario definitions. Each dict should have:
                - 'type': 'slope', 'skew', or 'butterfly'
                - 'shock_bp': shock magnitude in basis points
                If None, uses default scenarios (100bp slope, 100bp skew, 100bp butterfly)
            tolerance: Relative error tolerance (default 0.1% for multi-curve)

        Returns:
            Dictionary mapping each CurveTypes to its ScenarioValidationReport

        Example:
            >>> # XCCY swap with USD domestic, GBP foreign
            >>> result = xccy_swap.position(model).compute([RequestTypes.VALUE, RequestTypes.DELTA])
            >>> scenarios = [
            ...     {'type': 'slope', 'shock_bp': 100},
            ...     {'type': 'butterfly', 'shock_bp': 50}
            ... ]
            >>> reports = validator.validate_xccy_scenarios(result, scenarios=scenarios)
            >>> assert all(r.all_passed for r in reports.values())
            >>> for curve, report in reports.items():
            ...     print(f"{curve.name}: {report}")
        """
        # Extract deltas for all curves from Risk object
        if not isinstance(ad_result.risk, Risk):
            raise ValueError("AD result must contain a Risk object with deltas for multiple curves")

        # Get all curves that have deltas
        curves = list(ad_result.risk.deltas.keys())

        if len(curves) == 0:
            raise ValueError("No deltas found in AD result Risk object")

        # Validate scenarios for each curve individually
        scenario_reports = {}

        for curve_type in curves:
            # Extract delta for this specific curve
            delta_obj = ad_result.risk.deltas[curve_type]

            # Create single-curve result for validation
            single_curve_result = AnalyticsResult(
                value=ad_result.value,
                risk=delta_obj  # Pass Delta object directly
            )

            # Run scenario validation for this curve
            # This automatically handles XCCY cascade via _rebuild_model_with_shock_dict()
            curve_scenario_report = self.validate_curve_scenarios(
                ad_result=single_curve_result,
                curve_type=curve_type,
                scenarios=scenarios,
                tolerance=tolerance
            )

            scenario_reports[curve_type] = curve_scenario_report

        return scenario_reports

    ##########################################################################
    # HELPER METHODS
    ##########################################################################

    def _rebuild_model_with_bumped_curve(self,
                                         curve_name: str,
                                         tenor_index: int,
                                         bump_amount: float):
        """
        Rebuild model with one curve tenor bumped, handling dependencies.

        For XCCY swaps, bumping the foreign OIS curve requires rebuilding
        the XCCY curve as well (cascade effect).

        Args:
            curve_name: Name of curve to bump (e.g., "GBP_OIS_SONIA")
            tenor_index: Index of tenor to bump
            bump_amount: Bump size in decimal (e.g., 0.0001 for 1bp)

        Returns:
            New Model instance with bumped curve(s)
        """
        from cavour.models.models import Model
        from cavour.trades.rates.ois_curve import OISCurve
        from cavour.trades.rates.ois import OIS

        # Get curve parameters
        if curve_name not in self.model._curve_params_dict:
            raise ValueError(f"No stored parameters found for curve '{curve_name}'")

        params = self.model._curve_params_dict[curve_name].copy()

        # Detect curve type and bump appropriately
        is_xccy_curve = "basis_spreads" in params

        if is_xccy_curve:
            # XCCY curve: bump basis_spreads
            base_spreads = params["basis_spreads"].copy()
            tenors = params["tenor_list"]
            bumped_spreads = base_spreads.copy()
            bumped_spreads[tenor_index] += bump_amount
            # We'll use these later in Phase 2
            bumped_values = bumped_spreads
        else:
            # OIS curve: bump px_list
            base_px = params["px_list"].copy()
            tenors = params["tenor_list"]
            # Bump the specific tenor (px_list is in percentage points)
            # bump_amount is already in percentage points (e.g., 0.01 for 1bp)
            bumped_px = base_px.copy()
            bumped_px[tenor_index] += bump_amount
            bumped_values = bumped_px

        # Create new model (start fresh to avoid mutation)
        new_model = Model(value_dt=self.model.value_dt)

        # Copy FX data
        new_model._fx_params_dict = copy.deepcopy(self.model._fx_params_dict)

        # Two-phase rebuild to handle XCCY dependencies:
        # Phase 1: Rebuild all OIS curves first (with bumps applied)
        # Phase 2: Rebuild XCCY curves (which depend on OIS curves)

        # Phase 1: Rebuild OIS curves (non-XCCY curves)
        for curve_name_iter in self.model._curves_dict.keys():
            if curve_name_iter not in self.model._curve_params_dict:
                continue  # Skip curves without stored params

            params_iter = self.model._curve_params_dict[curve_name_iter].copy()

            # Check if this is an OIS curve (not XCCY)
            # XCCY curves have 'domestic_curve_name' key, OIS curves do not
            if "domestic_curve_name" in params_iter:
                continue  # Skip XCCY curves in Phase 1

            # This is an OIS curve - rebuild it
            if curve_name_iter == curve_name and not is_xccy_curve:
                # This is the OIS curve being bumped
                params_bumped = params.copy()
                params_bumped["px_list"] = bumped_values
                new_model.build_curve(name=curve_name_iter, **params_bumped)
            else:
                # Other OIS curve - copy unchanged
                new_model.build_curve(name=curve_name_iter, **params_iter)

        # Phase 2: Rebuild XCCY curves (now that OIS curves are ready)
        for curve_name_iter in self.model._curves_dict.keys():
            if curve_name_iter not in self.model._curve_params_dict:
                continue  # Skip curves without stored params

            params_iter = self.model._curve_params_dict[curve_name_iter].copy()

            # Check if this is an XCCY curve
            if "domestic_curve_name" in params_iter:
                # This is an XCCY curve - rebuild with build_xccy_curve()
                if curve_name_iter == curve_name and is_xccy_curve:
                    # This is the XCCY curve being bumped
                    params_bumped = params.copy()
                    params_bumped["basis_spreads"] = bumped_values
                    new_model.build_xccy_curve(name=curve_name_iter, **params_bumped)
                else:
                    # Other XCCY curve - copy unchanged (but picks up any bumped OIS curves)
                    new_model.build_xccy_curve(name=curve_name_iter, **params_iter)

        return new_model

    def _rebuild_model_with_two_bumps(self,
                                      curve_name_1: str,
                                      tenor_index_1: int,
                                      bump_amount_1: float,
                                      curve_name_2: str,
                                      tenor_index_2: int,
                                      bump_amount_2: float):
        """
        Rebuild model with two curve tenors bumped simultaneously.

        Used for cross-gamma validation where we need to bump tenors from
        two different curves and compute the resulting PV.

        Args:
            curve_name_1: Name of first curve to bump (e.g., "GBP_OIS_SONIA")
            tenor_index_1: Index of tenor to bump in curve 1
            bump_amount_1: Bump size in decimal for curve 1
            curve_name_2: Name of second curve to bump (e.g., "USD_GBP_BASIS")
            tenor_index_2: Index of tenor to bump in curve 2
            bump_amount_2: Bump size in decimal for curve 2

        Returns:
            New Model instance with both curves bumped
        """
        from cavour.models.models import Model

        # Get curve parameters for both curves
        if curve_name_1 not in self.model._curve_params_dict:
            raise ValueError(f"No stored parameters found for curve '{curve_name_1}'")
        if curve_name_2 not in self.model._curve_params_dict:
            raise ValueError(f"No stored parameters found for curve '{curve_name_2}'")

        params_1 = self.model._curve_params_dict[curve_name_1].copy()
        params_2 = self.model._curve_params_dict[curve_name_2].copy()

        # Detect curve types and bump appropriately
        is_xccy_1 = "basis_spreads" in params_1
        is_xccy_2 = "basis_spreads" in params_2

        # Bump curve 1
        if is_xccy_1:
            bumped_values_1 = params_1["basis_spreads"].copy()
            bumped_values_1[tenor_index_1] += bump_amount_1
        else:
            bumped_values_1 = params_1["px_list"].copy()
            bumped_values_1[tenor_index_1] += bump_amount_1

        # Bump curve 2
        if is_xccy_2:
            bumped_values_2 = params_2["basis_spreads"].copy()
            bumped_values_2[tenor_index_2] += bump_amount_2
        else:
            bumped_values_2 = params_2["px_list"].copy()
            bumped_values_2[tenor_index_2] += bump_amount_2

        # Create new model
        new_model = Model(value_dt=self.model.value_dt)
        new_model._fx_params_dict = copy.deepcopy(self.model._fx_params_dict)

        # Two-phase rebuild to handle XCCY dependencies:
        # Phase 1: Rebuild all OIS curves first (with bumps applied)
        # Phase 2: Rebuild XCCY curves (which depend on OIS curves)

        # Phase 1: Rebuild OIS curves
        for curve_name_iter in self.model._curves_dict.keys():
            if curve_name_iter not in self.model._curve_params_dict:
                continue

            params_iter = self.model._curve_params_dict[curve_name_iter].copy()

            # Check if this is an OIS curve (not XCCY)
            if "domestic_curve_name" in params_iter:
                continue  # Skip XCCY curves in Phase 1

            # This is an OIS curve - rebuild it
            if curve_name_iter == curve_name_1 and not is_xccy_1:
                # This is curve 1 being bumped (OIS curve)
                params_bumped = params_1.copy()
                params_bumped["px_list"] = bumped_values_1
                new_model.build_curve(name=curve_name_iter, **params_bumped)
            elif curve_name_iter == curve_name_2 and not is_xccy_2:
                # This is curve 2 being bumped (OIS curve)
                params_bumped = params_2.copy()
                params_bumped["px_list"] = bumped_values_2
                new_model.build_curve(name=curve_name_iter, **params_bumped)
            else:
                # Other OIS curve - copy unchanged
                new_model.build_curve(name=curve_name_iter, **params_iter)

        # Phase 2: Rebuild XCCY curves (now that OIS curves are ready)
        for curve_name_iter in self.model._curves_dict.keys():
            if curve_name_iter not in self.model._curve_params_dict:
                continue

            params_iter = self.model._curve_params_dict[curve_name_iter].copy()

            # Check if this is an XCCY curve
            if "domestic_curve_name" in params_iter:
                # This is an XCCY curve - rebuild with build_xccy_curve()
                if curve_name_iter == curve_name_1 and is_xccy_1:
                    # This is curve 1 being bumped (XCCY curve)
                    params_bumped = params_1.copy()
                    params_bumped["basis_spreads"] = bumped_values_1
                    new_model.build_xccy_curve(name=curve_name_iter, **params_bumped)
                elif curve_name_iter == curve_name_2 and is_xccy_2:
                    # This is curve 2 being bumped (XCCY curve)
                    params_bumped = params_2.copy()
                    params_bumped["basis_spreads"] = bumped_values_2
                    new_model.build_xccy_curve(name=curve_name_iter, **params_bumped)
                else:
                    # Other XCCY curve - rebuild with updated OIS curves
                    new_model.build_xccy_curve(name=curve_name_iter, **params_iter)

        return new_model

    def _rebuild_model_parallel_shock(self,
                                      curve_name: str,
                                      shock_amount: float):
        """
        Rebuild model with parallel shock to all tenors of a curve.

        Args:
            curve_name: Name of curve to shock (e.g., "GBP_OIS_SONIA")
            shock_amount: Shock size in decimal (e.g., 0.01 for 100bp)

        Returns:
            New Model instance with shocked curve
        """
        from cavour.models.models import Model

        # Get curve parameters
        if curve_name not in self.model._curve_params_dict:
            raise ValueError(f"No stored parameters found for curve '{curve_name}'")

        params = self.model._curve_params_dict[curve_name].copy()
        base_px = params["px_list"].copy()

        # Apply parallel shock (px_list is in percentage points)
        # shock_amount is already in percentage points (e.g., 1.0 for 100bp)
        shocked_px = [px + shock_amount for px in base_px]

        # Create new model
        new_model = Model(value_dt=self.model.value_dt)
        new_model._fx_params_dict = copy.deepcopy(self.model._fx_params_dict)

        # Two-phase rebuild to handle XCCY dependencies
        # Phase 1: Rebuild all OIS curves first
        for curve_name_iter in self.model._curves_dict.keys():
            if curve_name_iter not in self.model._curve_params_dict:
                continue

            params_iter = self.model._curve_params_dict[curve_name_iter].copy()

            # Skip XCCY curves in Phase 1
            if "domestic_curve_name" in params_iter:
                continue

            # This is an OIS curve - rebuild it
            if curve_name_iter == curve_name:
                # This is the curve being shocked
                params_shocked = params.copy()
                params_shocked["px_list"] = shocked_px
                new_model.build_curve(name=curve_name_iter, **params_shocked)
            else:
                # Other OIS curve - copy unchanged
                new_model.build_curve(name=curve_name_iter, **params_iter)

        # Phase 2: Rebuild XCCY curves (cascade effect)
        for curve_name_iter in self.model._curves_dict.keys():
            if curve_name_iter not in self.model._curve_params_dict:
                continue

            params_iter = self.model._curve_params_dict[curve_name_iter].copy()

            # Check if this is an XCCY curve
            if "domestic_curve_name" in params_iter:
                # Rebuild XCCY curve with updated OIS curves
                new_model.build_xccy_curve(name=curve_name_iter, **params_iter)

        return new_model

    def _rebuild_model_with_shock_dict(
        self,
        curve_name: str,
        shock_dict: Dict[str, float]
    ):
        """
        Rebuild model with tenor-specific shocks.

        Applies different shock magnitudes to different tenors according to
        shock_dict. Used for scenario validation (slope, skew, butterfly).

        Args:
            curve_name: Name of curve to shock (e.g., "GBP_OIS_SONIA")
            shock_dict: Dictionary mapping tenor -> shock in basis points
                e.g., {"1M": 100, "3M": 50, "6M": 0, "1Y": -50, "5Y": -100}

        Returns:
            New Model instance with shocked curve
        """
        from cavour.models.models import Model

        # Get curve parameters
        if curve_name not in self.model._curve_params_dict:
            raise ValueError(f"No stored parameters found for curve '{curve_name}'")

        params = self.model._curve_params_dict[curve_name].copy()
        base_px = params["px_list"].copy()
        tenors = params["tenor_list"]

        # Apply tenor-specific shocks (shock_dict is in basis points, px_list is in percentage points)
        shocked_px = base_px.copy()
        for i, tenor in enumerate(tenors):
            if tenor in shock_dict:
                # Convert shock from bp to percentage points (e.g., 100bp = 1.0)
                shock_pct_points = shock_dict[tenor] * 0.01
                shocked_px[i] += shock_pct_points

        # Create new model
        new_model = Model(value_dt=self.model.value_dt)
        new_model._fx_params_dict = copy.deepcopy(self.model._fx_params_dict)

        # Two-phase rebuild to handle XCCY dependencies
        # Phase 1: Rebuild all OIS curves first
        for curve_name_iter in self.model._curves_dict.keys():
            if curve_name_iter not in self.model._curve_params_dict:
                continue

            params_iter = self.model._curve_params_dict[curve_name_iter].copy()

            # Skip XCCY curves in Phase 1
            if "domestic_curve_name" in params_iter:
                continue

            # This is an OIS curve - rebuild it
            if curve_name_iter == curve_name:
                # This is the curve being shocked
                params_shocked = params.copy()
                params_shocked["px_list"] = shocked_px
                new_model.build_curve(name=curve_name_iter, **params_shocked)
            else:
                # Other OIS curve - copy unchanged
                new_model.build_curve(name=curve_name_iter, **params_iter)

        # Phase 2: Rebuild XCCY curves (cascade effect)
        for curve_name_iter in self.model._curves_dict.keys():
            if curve_name_iter not in self.model._curve_params_dict:
                continue

            params_iter = self.model._curve_params_dict[curve_name_iter].copy()

            # Check if this is an XCCY curve
            if "domestic_curve_name" in params_iter:
                # Rebuild XCCY curve with updated OIS curves
                new_model.build_xccy_curve(name=curve_name_iter, **params_iter)

        return new_model

    def _compute_pv(self, model) -> float:
        """
        Compute present value of derivative using given model.

        Args:
            model: Model instance to use for valuation

        Returns:
            Present value as float
        """
        position = self.derivative.position(model)
        result = position.compute([RequestTypes.VALUE])
        return result.value.amount

    def _get_curve_name_from_type(self, curve_type: CurveTypes) -> str:
        """
        Convert CurveTypes enum to curve name string.

        Args:
            curve_type: CurveTypes enum value

        Returns:
            Curve name as string (e.g., "GBP_OIS_SONIA")
        """
        return curve_type.name

    def _extract_delta_ladder_as_dict(self, delta_obj: Delta) -> Dict[str, float]:
        """
        Extract delta ladder as dictionary with tenor keys.

        Args:
            delta_obj: Delta object from AnalyticsResult

        Returns:
            Dictionary mapping tenor -> delta value
        """
        if hasattr(delta_obj, 'ladder') and hasattr(delta_obj.ladder, 'to_dict'):
            return delta_obj.ladder.to_dict()
        elif hasattr(delta_obj, 'risk_ladder') and hasattr(delta_obj, 'tenors'):
            # Manual extraction
            return {tenor: float(delta_obj.risk_ladder[i])
                    for i, tenor in enumerate(delta_obj.tenors)}
        else:
            raise ValueError("Cannot extract delta ladder from Delta object")

    def _xccy_depends_on_curve(self, xccy_curve_name: str, ois_curve_name: str) -> bool:
        """
        Check if an XCCY curve depends on a given OIS curve.

        XCCY curves depend on both domestic and foreign OIS curves. When
        bumping an OIS curve, any dependent XCCY curves must be rebuilt
        to reflect the change (cascade effect).

        Args:
            xccy_curve_name: Name of potential XCCY curve (e.g., "GBP_USD_BASIS")
            ois_curve_name: Name of OIS curve (e.g., "GBP_OIS_SONIA")

        Returns:
            True if xccy_curve_name depends on ois_curve_name, False otherwise

        Example:
            >>> validator._xccy_depends_on_curve("GBP_USD_BASIS", "GBP_OIS_SONIA")
            True  # GBP_USD_BASIS depends on GBP OIS

            >>> validator._xccy_depends_on_curve("GBP_OIS_SONIA", "USD_OIS_SOFR")
            False  # OIS curve doesn't depend on another OIS
        """
        # Check if curve exists in params dict
        if xccy_curve_name not in self.model._curve_params_dict:
            return False

        params = self.model._curve_params_dict[xccy_curve_name]

        # XCCY curves have 'domestic_curve_name' and 'foreign_curve_name' keys
        # Regular OIS curves do not have these keys
        if "domestic_curve_name" not in params:
            return False

        # Check if this XCCY curve depends on the given OIS curve
        return (params["domestic_curve_name"] == ois_curve_name or
                params["foreign_curve_name"] == ois_curve_name)
