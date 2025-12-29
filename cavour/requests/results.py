"""
Result classes for storing valuation and risk analytics.

Provides dataclasses for:
- Valuation: Monetary amounts with currency
- Delta: First-order interest rate sensitivities
- Gamma: Second-order interest rate sensitivities
- Risk: Container for multiple risk ladders
- AnalyticsResult: Complete result set (PV + Greeks)

All classes support arithmetic operations where appropriate and provide
formatted output for analysis and visualization.
"""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tabulate import tabulate
from typing import List, Tuple, Iterator, Dict, Iterable, Optional, Any


from dataclasses import dataclass, field
from typing import Any, Iterable, Dict, Union
from cavour.utils.currency import CurrencyTypes
from cavour.utils.global_types import RequestTypes, CurveTypes



@dataclass(frozen=True)
class Valuation:
    """
    A monetary amount together with its currency.

    Supports arithmetic operations (+, -, *, /) when currencies match.
    Immutable dataclass suitable for use in aggregations.

    Attributes:
        amount (float): Monetary value
        currency (CurrencyTypes): Currency denomination

    Example:
        >>> v1 = Valuation(1000.0, CurrencyTypes.GBP)
        >>> v2 = Valuation(500.0, CurrencyTypes.GBP)
        >>> total = v1 + v2  # Valuation(1500.0, GBP)
        >>> scaled = v1 * 1.1  # Valuation(1100.0, GBP)
    """
    amount: float
    currency: CurrencyTypes = CurrencyTypes.NONE

    def __post_init__(self):
        if not isinstance(self.currency, CurrencyTypes):
            raise TypeError(
                f"currency must be a CurrencyTypes enum, got {type(self.currency)}"
            )

    def __repr__(self) -> str:
        return f"{self.amount:.2f} {self.currency.name}"

    def __add__(self, other: Any) -> "Valuation":
        # Allow adding two Valuations of same currency
        if not isinstance(other, Valuation):
            return NotImplemented
        if self.currency is not other.currency:
            raise ValueError(
                f"Cannot add {self.currency.name} to {other.currency.name}"
            )
        return Valuation(
            amount=self.amount + other.amount,
            currency=self.currency
        )

    def __sub__(self, other: Any) -> "Valuation":
        if not isinstance(other, Valuation):
            return NotImplemented
        if self.currency is not other.currency:
            raise ValueError(
                f"Cannot subtract {other.currency.name} from {self.currency.name}"
            )
        return Valuation(
            amount=self.amount - other.amount,
            currency=self.currency
        )

    def __mul__(self, factor: float) -> "Valuation":
        return Valuation(
            amount=self.amount * factor,
            currency=self.currency
        )

    def __rmul__(self, factor: float) -> "Valuation":
        return self.__mul__(factor)

    def __truediv__(self, divisor: float) -> "Valuation":
        return Valuation(
            amount=self.amount / divisor,
            currency=self.currency
        )

    def __radd__(self, other: Any) -> "Valuation":
        # support sum() with initial zero
        if other == 0:
            return self
        return self.__add__(other)
    


@dataclass(frozen=True)
class Value:
    """
    A simple monetary amount with currency (lightweight version of Valuation).

    Used for displaying aggregated risk values without arithmetic operations.

    Attributes:
        amount (float): Monetary value
        currency (CurrencyTypes): Currency denomination
    """
    amount: float
    currency: CurrencyTypes = CurrencyTypes.NONE


    
class Ladder:
    """
    Encapsulates a tenor->sensitivity mapping and provides a DataFrame view.

    Wrapper for risk ladder data that can be exported as pandas DataFrame
    for analysis and visualization.

    Attributes:
        data (Dict[str, float]): Mapping of tenor -> sensitivity value
        _curve_name (str): Curve identifier for labeling

    Example:
        >>> data = {"1Y": 10.5, "5Y": -8.2, "10Y": 15.3}
        >>> ladder = Ladder(data, "GBP_OIS_SONIA")
        >>> df = ladder.df  # Returns pandas DataFrame
    """
    def __init__(self, data: Dict[str, float], curve_name: str):
        self.data = data
        self._curve_name = curve_name

    @property
    def df(self) -> pd.DataFrame:
        """
        Return the risk ladder as a pandas DataFrame:
          - index: tenor strings
          - single column: "<CURVE>_Risk"
        """
        df = pd.DataFrame.from_dict(
            self.data,
            orient='index',
            columns=[f"{self._curve_name}_Risk"]
        )
        df.index.name = 'Tenor'
        return df

    def to_dict(self) -> Dict[str, float]:
        """Return the raw tenor->value mapping."""
        return dict(self.data)

    def __repr__(self):
        count = len(self.data)
        return f"Ladder(curve={self._curve_name}, points={count}, curve_data={self.data})"


@dataclass(frozen=True)
class Delta:
    """
    First-order interest rate sensitivity (delta) for a given curve.

    Represents how the position value changes with respect to 1bp parallel
    shifts in each tenor of the curve. Computed via automatic differentiation.

    Attributes:
        risk_ladder (jnp.ndarray): Sensitivities for each tenor (shape: [N])
        tenors (List[str]): Tenor labels (e.g., ["1M", "3M", "1Y"])
        currency (CurrencyTypes): Currency of the sensitivities
        curve_type (CurveTypes): Curve identifier (e.g., GBP_OIS_SONIA)

    Properties:
        value: Sum of ladder as Value object
        ladder: Ladder object for DataFrame export

    Example:
        >>> delta = Delta(
        ...     risk_ladder=jnp.array([10.2, -5.3, 15.8]),
        ...     tenors=["1Y", "5Y", "10Y"],
        ...     currency=CurrencyTypes.GBP,
        ...     curve_type=CurveTypes.GBP_OIS_SONIA
        ... )
        >>> total = delta.value  # Sum of sensitivities
        >>> df = delta.ladder.df  # Export to DataFrame
    """
    risk_ladder: jnp.ndarray       # shape [..., N]
    tenors:       List[str]        # length N
    currency:     CurrencyTypes
    curve_type:   CurveTypes

    def __post_init__(self):
        # convert list to JAX array if needed
        arr = self.risk_ladder
        if isinstance(arr, list):
            arr = jnp.array(arr)
            object.__setattr__(self, 'risk_ladder', arr)
        n = len(self.risk_ladder) #.shape[-1]
        if n != len(self.tenors):
            raise ValueError(
                f"Expected {n} tenors, got {len(self.tenors)}"
            )
        if not isinstance(self.currency, CurrencyTypes):
            raise TypeError(
                f"currency must be CurrencyTypes, got {type(self.currency)}"
            )
        if not isinstance(self.curve_type, CurveTypes):
            raise TypeError(
                f"curve_type must be CurveTypes, got {type(self.curve_type)}"
            )

    @property
    def value(self) -> Value:
        """Sum of the ladder as a Value object."""
        total = float(jnp.sum(self.risk_ladder))
        return Value(amount=total, currency=self.currency)

    @property
    def ladder(self) -> Ladder:
        """Return the tenor->sensitivity mapping as a Ladder object."""
        data = dict(zip(self.tenors, self.risk_ladder.tolist()))
        return Ladder(data, self.curve_type.name)

    def __repr__(self):
        total = self.value.amount
        cur = self.currency.name
        n = len(self.tenors)
        return (
            f"{self.__class__.__name__}("
            f"{self.curve_type.name}: {total:.6g} {cur}, "
            f"points={n})"
        )
    
    def __add__(self, other: Any) -> 'Delta':
        """
        Sum two Delta objects with the same curve_type, currency, and tenors.
        Returns a new Delta with elementwise sum of risk_ladder.
        """
        if not isinstance(other, Delta):
            return NotImplemented
        if (self.curve_type != other.curve_type or
            self.currency  != other.currency or
            self.tenors    != other.tenors):
            raise ValueError(
                "Cannot add Delta with mismatched curve_type, currency, or tenors"
            )
        summed = self.risk_ladder + other.risk_ladder
        return Delta(
            risk_ladder=summed,
            tenors=self.tenors,
            currency=self.currency,
            curve_type=self.curve_type
        )

    __radd__ = __add__  # support sum() accumulation


@dataclass(frozen=True)
class Gamma:
    """
    Second-order interest rate sensitivity (gamma) for a given curve.

    Represents how delta changes with respect to rate movements (curvature).
    Stored as a matrix showing cross-tenor sensitivities. Computed via
    automatic differentiation (Hessian).

    Attributes:
        risk_ladder (jnp.ndarray): Gamma matrix (shape: [N, N]) or vector [N]
        tenors (List[str]): Tenor labels
        currency (CurrencyTypes): Currency of sensitivities
        curve_type (CurveTypes): Curve identifier

    Properties:
        value: Sum of gamma matrix as Value object
        matrix: Display gamma as formatted table (prints to console)
        to_dict: Export gamma as nested dictionary

    Methods:
        plot(): Display interactive Plotly heatmap

    Note:
        Units are value per bp^2, typically scaled by 1e8.

    Example:
        >>> gamma = Gamma(
        ...     risk_ladder=jnp.array([[0.5, 0.1], [0.1, 0.8]]),
        ...     tenors=["1Y", "5Y"],
        ...     currency=CurrencyTypes.GBP,
        ...     curve_type=CurveTypes.GBP_OIS_SONIA
        ... )
        >>> gamma.plot()  # Interactive heatmap
        >>> gamma.matrix  # Formatted table output
    """
    risk_ladder: jnp.ndarray       # shape [N] or [N, N]
    tenors:       List[str]        # length N
    currency:     CurrencyTypes
    curve_type:   CurveTypes

    def __post_init__(self):
        arr = self.risk_ladder
        if isinstance(arr, list):
            arr = jnp.array(arr)
            object.__setattr__(self, 'risk_ladder', arr)
        n = arr.shape[-1]
        if n != len(self.tenors):
            raise ValueError(f"Expected {n} tenors, got {len(self.tenors)}")
        if not isinstance(self.currency, CurrencyTypes):
            raise TypeError(f"currency must be CurrencyTypes, got {type(self.currency)}")
        if not isinstance(self.curve_type, CurveTypes):
            raise TypeError(f"curve_type must be CurveTypes, got {type(self.curve_type)}")

    @property
    def value(self) -> Value:
        """Sum of the gamma ladder as a Value object."""
        total = float(jnp.sum(self.risk_ladder))
        return Value(amount=total, currency=self.currency)

    @property
    def matrix(self) -> dict:
        """
        Return the gamma matrix as a nested dict: {row_tenor: {col_tenor: value}}.
        Assumes risk_ladder is 2D.
        """
        matrix = self.to_dict

        df = pd.DataFrame(matrix)
        # Drop all-zero rows and columns
        df = df.loc[~(df == 0).all(axis=1)]  # drop zero rows
        df = df.loc[:, ~(df == 0).all(axis=0)]  # drop zero cols

        # Round headers/index for display
        df.index = [f"{i:.2f}" for i in df.index]
        df.columns = [f"{c:.2f}" for c in df.columns]

        df.index.name = "Tenors"
        # Pretty print
        print(tabulate(df, headers='keys', tablefmt='grid', floatfmt=".2f"))
    
    @property
    def to_dict(self) -> dict:
        """
        Return the gamma matrix as a nested dict: {row_tenor: {col_tenor: value}}.
        Assumes risk_ladder is 2D.
        """
        gamma_np = np.array(self.risk_ladder)
        if gamma_np.ndim != 2:
            raise ValueError("Gamma risk_ladder must be 2D to access matrix")

        matrix = {
            row_tenor: {
                col_tenor: float(gamma_np[i, j])
                for j, col_tenor in enumerate(self.tenors)
            }
            for i, row_tenor in enumerate(self.tenors)
        }
        return matrix

    def plot(self):
        """
        Plot the gamma heatmap using Plotly.
        Trims outer rows/columns if they are all zero.
        """
        gamma_np = np.array(self.risk_ladder, dtype=np.float64)
        if gamma_np.ndim == 1:
            gamma_np = np.diag(gamma_np)

        # Identify non-zero rows and columns
        nonzero_rows = ~np.all(gamma_np == 0, axis=1)
        nonzero_cols = ~np.all(gamma_np == 0, axis=0)
        keep_mask = nonzero_rows & nonzero_cols

        trimmed_matrix = gamma_np[np.ix_(keep_mask, keep_mask)]
        trimmed_tenors = [t for t, keep in zip(self.tenors, keep_mask) if keep]

        fig = go.Figure(data=go.Heatmap(
            z=trimmed_matrix,
            x=trimmed_tenors,
            y=trimmed_tenors,
            colorscale="RdYlGn_r",
            colorbar=dict(title="Gamma"),
            zmin=np.min(trimmed_matrix),
            zmax=np.max(trimmed_matrix),
        ))

        fig.update_layout(
            title=f"Gamma Heatmap: {self.curve_type.name}",
            xaxis_title="Tenor",
            yaxis_title="Tenor",
            width=800,
            height=700,
        )

        fig.show()

    def __repr__(self):
        total = self.value.amount
        cur = self.currency.name
        n = len(self.tenors)
        return (
            f"{self.__class__.__name__}("
            f"{self.curve_type.name}: {total:.6g} {cur}, "
            f"points={n})"
        )

    def __add__(self, other: Any) -> 'Gamma':
        if not isinstance(other, Gamma):
            return NotImplemented
        if (self.curve_type != other.curve_type or
            self.currency  != other.currency or
            self.tenors    != other.tenors):
            raise ValueError(
                "Cannot add Gamma with mismatched curve_type, currency, or tenors"
            )
        summed = self.risk_ladder + other.risk_ladder
        return Gamma(
            risk_ladder=summed,
            tenors=self.tenors,
            currency=self.currency,
            curve_type=self.curve_type
        )

    __radd__ = __add__

    
@dataclass(frozen=True)
class CrossGamma:
    """
    Cross-curve second-order sensitivity (cross-gamma).

    Represents how delta to one curve changes when a DIFFERENT curve moves.
    This captures second-order dependencies between curves (e.g., how XCCY 
    basis delta changes when foreign OIS rates move).

    Attributes:
        risk_matrix (jnp.ndarray): Cross-gamma matrix (shape: [N1, N2])
        tenors_curve1 (List[str]): Tenor labels for curve 1 (rows)
        tenors_curve2 (List[str]): Tenor labels for curve 2 (columns)
        curve_type_1 (CurveTypes): First curve identifier
        curve_type_2 (CurveTypes): Second curve identifier
        currency (CurrencyTypes): Currency of sensitivities

    Properties:
        value: Sum of cross-gamma matrix as Value object
        matrix: Display cross-gamma as formatted table (prints to console)
        to_dict: Export cross-gamma as nested dictionary

    Methods:
        plot(): Display interactive Plotly heatmap

    Note:
        Units are value per bp^2, typically scaled by 1e8.
        Matrix element [i,j] represents: d²PV / d(curve1_rate_i) d(curve2_rate_j)

    Example:
        >>> cross_gamma = CrossGamma(
        ...     risk_matrix=jnp.array([[0.5, 0.1], [0.1, 0.8]]),
        ...     tenors_curve1=[1Y, 5Y],
        ...     tenors_curve2=[1Y, 3Y],
        ...     curve_type_1=CurveTypes.USD_OIS_SOFR,
        ...     curve_type_2=CurveTypes.USD_GBP_BASIS,
        ...     currency=CurrencyTypes.GBP
        ... )
        >>> cross_gamma.plot()  # Interactive heatmap
    """
    risk_matrix:     jnp.ndarray       # shape [N1, N2]
    tenors_curve1:   List[str]         # length N1 (rows)
    tenors_curve2:   List[str]         # length N2 (columns)
    curve_type_1:    CurveTypes
    curve_type_2:    CurveTypes
    currency:        CurrencyTypes

    def __post_init__(self):
        arr = self.risk_matrix
        if isinstance(arr, list):
            arr = jnp.array(arr)
            object.__setattr__(self, 'risk_matrix', arr)
        
        if arr.ndim != 2:
            raise ValueError(f"CrossGamma risk_matrix must be 2D, got {arr.ndim}D")
        
        n1, n2 = arr.shape
        if n1 != len(self.tenors_curve1):
            raise ValueError(f"Expected {n1} tenors for curve 1, got {len(self.tenors_curve1)}")
        if n2 != len(self.tenors_curve2):
            raise ValueError(f"Expected {n2} tenors for curve 2, got {len(self.tenors_curve2)}")
        
        if not isinstance(self.currency, CurrencyTypes):
            raise TypeError(f"currency must be CurrencyTypes, got {type(self.currency)}")
        if not isinstance(self.curve_type_1, CurveTypes):
            raise TypeError(f"curve_type_1 must be CurveTypes, got {type(self.curve_type_1)}")
        if not isinstance(self.curve_type_2, CurveTypes):
            raise TypeError(f"curve_type_2 must be CurveTypes, got {type(self.curve_type_2)}")

    @property
    def value(self) -> Value:
        """Sum of the cross-gamma matrix as a Value object."""
        total = float(jnp.sum(self.risk_matrix))
        return Value(amount=total, currency=self.currency)

    @property
    def matrix(self) -> dict:
        """
        Return the cross-gamma matrix as a nested dict: {curve1_tenor: {curve2_tenor: value}}.
        """
        matrix_dict = self.to_dict

        df = pd.DataFrame(matrix_dict)
        # Don't drop zero rows/columns for cross-gamma (they may all be informative)

        df.index.name = f"{self.curve_type_1.name} Tenors"
        df.columns.name = f"{self.curve_type_2.name} Tenors"
        
        # Pretty print
        print(tabulate(df, headers='keys', tablefmt='grid', floatfmt=".4f"))
    
    @property
    def to_dict(self) -> dict:
        """
        Return the cross-gamma matrix as a nested dict: {curve1_tenor: {curve2_tenor: value}}.
        """
        gamma_np = np.array(self.risk_matrix)
        
        matrix = {
            row_tenor: {
                col_tenor: float(gamma_np[i, j])
                for j, col_tenor in enumerate(self.tenors_curve2)
            }
            for i, row_tenor in enumerate(self.tenors_curve1)
        }
        return matrix

    def plot(self):
        """
        Plot the cross-gamma heatmap using Plotly.
        Shows curve1 tenors on Y-axis and curve2 tenors on X-axis.
        """
        gamma_np = np.array(self.risk_matrix, dtype=np.float64)

        fig = go.Figure(data=go.Heatmap(
            z=gamma_np,
            x=self.tenors_curve2,
            y=self.tenors_curve1,
            colorscale="RdYlGn_r",
            colorbar=dict(title="Cross-Gamma"),
            zmin=np.min(gamma_np) if gamma_np.size > 0 else 0,
            zmax=np.max(gamma_np) if gamma_np.size > 0 else 1,
        ))

        fig.update_layout(
            title=f"Cross-Gamma: {self.curve_type_1.name} vs {self.curve_type_2.name}",
            xaxis_title=f"{self.curve_type_2.name} Tenors",
            yaxis_title=f"{self.curve_type_1.name} Tenors",
            width=900,
            height=700,
        )

        fig.show()

    def __repr__(self):
        total = self.value.amount
        cur = self.currency.name
        n1, n2 = len(self.tenors_curve1), len(self.tenors_curve2)
        return (
            f"{self.__class__.__name__}("
            f"{self.curve_type_1.name} x {self.curve_type_2.name}: "
            f"{total:.6g} {cur}, shape=[{n1}, {n2}])"
        )

    def __add__(self, other: Any) -> 'CrossGamma':
        if not isinstance(other, CrossGamma):
            return NotImplemented
        if (self.curve_type_1 != other.curve_type_1 or
            self.curve_type_2 != other.curve_type_2 or
            self.currency != other.currency or
            self.tenors_curve1 != other.tenors_curve1 or
            self.tenors_curve2 != other.tenors_curve2):
            raise ValueError(
                "Cannot add CrossGamma with mismatched curves, currency, or tenors"
            )
        summed = self.risk_matrix + other.risk_matrix
        return CrossGamma(
            risk_matrix=summed,
            tenors_curve1=self.tenors_curve1,
            tenors_curve2=self.tenors_curve2,
            curve_type_1=self.curve_type_1,
            curve_type_2=self.curve_type_2,
            currency=self.currency
        )

    __radd__ = __add__


class Risk:
    """
    Container for multiple per-curve Delta and Gamma ladders.

    Provides convenient attribute and callable access to risk by curve.
    Automatically creates attributes from curve names for easy access.

    Args:
        ladders (Iterable[Union[Delta, Gamma]]): List of Delta or Gamma objects
        cross_gammas (Optional[Iterable[CrossGamma]]): List of CrossGamma objects

    Access patterns:
        1. Attribute: risk.GBP_OIS_SONIA.value
        2. Attribute: risk.GBP_OIS_SONIA.ladder (for Delta)
        3. Attribute: risk.GBP_OIS_SONIA.matrix (for Gamma)
        4. Callable: risk(CurveTypes.GBP_OIS_SONIA)
        5. Cross-gamma: risk.cross_gamma(CurveTypes.USD_OIS_SOFR, CurveTypes.USD_GBP_BASIS)

    Example:
        >>> delta1 = Delta([10, -5], ["1Y", "5Y"], CurrencyTypes.GBP, CurveTypes.GBP_OIS_SONIA)
        >>> delta2 = Delta([8, -3], ["1Y", "5Y"], CurrencyTypes.USD, CurveTypes.USD_OIS_SOFR)
        >>> risk = Risk([delta1, delta2])
        >>> gbp_sens = risk.GBP_OIS_SONIA  # Attribute access
        >>> usd_sens = risk(CurveTypes.USD_OIS_SOFR)  # Callable access

    Raises:
        ValueError: If duplicate curve names provided
    """
    def __init__(
        self,
        ladders: Iterable[Union[Delta, Gamma]],
        cross_gammas: Optional[Iterable[CrossGamma]] = None
    ):
        self._by_curve = {}  # type: Dict[str, Union[Delta, Gamma]]
        self._cross_gammas = {}  # type: Dict[Tuple[str, str], CrossGamma]

        for ladder in ladders:
            name = ladder.curve_type.name
            if name in self._by_curve:
                raise ValueError(f"Duplicate curve {name}")
            self._by_curve[name] = ladder
            setattr(self, name, ladder)

        # Store cross-gammas keyed by (curve1_name, curve2_name)
        if cross_gammas is not None:
            for cg in cross_gammas:
                key = (cg.curve_type_1.name, cg.curve_type_2.name)
                if key in self._cross_gammas:
                    raise ValueError(f"Duplicate cross-gamma for {key}")
                self._cross_gammas[key] = cg

    def __call__(self, curve_type: CurveTypes) -> Union[Delta, Gamma]:
        """
        Allow callable access: risk(CurveTypes.GBP_OIS_SONIA) => Delta or Gamma
        """
        try:
            return self._by_curve[curve_type.name]
        except KeyError:
            raise ValueError(f"No risk data for curve: {curve_type.name}")

    def cross_gamma(
        self,
        curve_type_1: CurveTypes,
        curve_type_2: CurveTypes
    ) -> Optional[CrossGamma]:
        """
        Access cross-gamma between two curves.

        Args:
            curve_type_1: First curve (e.g., USD_OIS_SOFR)
            curve_type_2: Second curve (e.g., USD_GBP_BASIS)

        Returns:
            CrossGamma object if exists, None otherwise

        Example:
            >>> cg = risk.cross_gamma(CurveTypes.USD_OIS_SOFR, CurveTypes.USD_GBP_BASIS)
            >>> if cg:
            >>>     print(cg.value)  # Total cross-gamma
            >>>     cg.plot()  # Heatmap
        """
        key = (curve_type_1.name, curve_type_2.name)
        return self._cross_gammas.get(key, None)

    def has_cross_gamma(
        self,
        curve_type_1: CurveTypes,
        curve_type_2: CurveTypes
    ) -> bool:
        """Check if cross-gamma exists for the given curve pair."""
        key = (curve_type_1.name, curve_type_2.name)
        return key in self._cross_gammas

    @property
    def all_cross_gammas(self) -> Dict[Tuple[str, str], CrossGamma]:
        """Return all cross-gammas as a dictionary."""
        return self._cross_gammas.copy()

    def __repr__(self):
        parts = []
        for name, obj in self._by_curve.items():
            mv = obj.value
            parts.append(f"{name}={mv.amount:.6g} {mv.currency.name}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


class AnalyticsResult:
    """
    Complete analytics result set containing valuation and Greeks.

    Central result container returned by position.compute() calls. Holds
    present value, delta, and gamma sensitivities computed via automatic
    differentiation.

    Args:
        value (Optional[Valuation]): Present value with currency
        risk (Optional[Risk]): Delta risk ladders
        gamma (Optional[Gamma]): Gamma (second-order) sensitivities

    Properties:
        value: Valuation object (present value)
        risk: Risk container with delta ladders
        gamma: Gamma matrix

    Example:
        >>> swap = OIS(value_dt, "10Y", 0.04)
        >>> pos = swap.position(model)
        >>> result = pos.compute([RequestTypes.VALUE, RequestTypes.DELTA, RequestTypes.GAMMA])
        >>> print(result.value)  # Valuation(amount=1234.56, currency=GBP)
        >>> print(result.risk.GBP_OIS_SONIA)  # Delta ladder
        >>> result.gamma.plot()  # Interactive gamma heatmap
    """
    def __init__(
        self,
        value: Optional[Valuation] = None,
        risk: Optional[Risk] = None,
        gamma: Optional[Gamma] = None,
    ):
        # store inputs directly
        self._value = value
        self._risk  = risk
        self._gamma  = gamma

    @property
    def value(self) -> Value:
        """Return the Value stored in the Valuation object."""
        # assume Valuation has an attribute `value` of type Value
        return self._value

    @property
    def risk(self) -> Optional[Risk]:
        """Return the Risk object (delta ladders)."""
        return self._risk
    
    @property
    def gamma(self) -> Optional[Risk]:
        """Return the Gamma object (Gamma matrix)."""
        return self._gamma

    def __repr__(self):
        cls = self.__class__.__name__
        parts = []
        # value
        if self._value is not None:
            parts.append(f"value={self._value!r}")
        # risk
        if self._risk is not None:
            parts.append(f"risk={self._risk!r}")
        # gamma
        if self.gamma is not None:
            parts.append(f"gamma={self.gamma!r}")
        return f"{cls}({', '.join(parts)})"
