"Class to store results"

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
    Supports arithmetic operations when currencies match.
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
    A monetary amount together with its currency.
    """
    amount: float
    currency: CurrencyTypes = CurrencyTypes.NONE


    
class Ladder:
    """
    Encapsulates a tenor->sensitivity mapping and provides a DataFrame view.
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
    A delta risk ladder for a given curve.
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
    A gamma risk ladder (second-order sensitivity) for a given curve.
    Units: value per bp^2, typically scaled by 1e8.
    """
    risk_ladder: jnp.ndarray       # shape [N] or [N, N]
    tenors:       List[str]        # length N
    currency:     CurrencyTypes = CurrencyTypes.GBP
    curve_type:   CurveTypes = CurveTypes.SONIA

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
        #return matrix
    
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

    
class Risk:
    """
    Container for multiple per-curve Delta and Gamma ladders.
    Allows access like:
        risk.SONIA.value
        risk.SONIA.ladder  (for Delta)
        risk.SONIA.matrix  (for Gamma)
    Or:
        risk(CurveTypes.SONIA) => Delta or Gamma
    """
    def __init__(self, ladders: Iterable[Union[Delta, Gamma]]):
        self._by_curve = {}  # type: Dict[str, Union[Delta, Gamma]]

        for ladder in ladders:
            name = ladder.curve_type.name
            if name in self._by_curve:
                raise ValueError(f"Duplicate curve {name}")
            self._by_curve[name] = ladder
            setattr(self, name, ladder)

    def __call__(self, curve_type: CurveTypes) -> Union[Delta, Gamma]:
        """
        Allow callable access: risk(CurveTypes.SONIA) => Delta or Gamma
        """
        try:
            return self._by_curve[curve_type.name]
        except KeyError:
            raise ValueError(f"No risk data for curve: {curve_type.name}")

    def __repr__(self):
        parts = []
        for name, obj in self._by_curve.items():
            mv = obj.value
            parts.append(f"{name}={mv.amount:.6g} {mv.currency.name}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


class AnalyticsResult:
    """
    Holds a Valuation-typed PV and its associated greeks (risk ladders).
    Access via:
       res.value  -> Value (wrapped in Valuation)
       res.risk   -> Risk
       res.gamma  -> jnp.ndarray or None
    """
    def __init__(
        self,
        value: Optional[Valuation] = None,
        risk: Optional[Risk] = None,
        gamma: Optional[Gamma] = None, #still need to develop it
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
