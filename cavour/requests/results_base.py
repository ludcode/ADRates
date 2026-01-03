"""
Base classes and mixins for result containers.

This module provides the foundation for all result container classes (Valuation, Delta, Gamma, etc.)
through abstract base classes and reusable mixins. The design follows these principles:

1. Mixins contain only methods, no instance attributes (compatible with frozen dataclasses)
2. Common functionality is centralized to reduce code duplication
3. All mixins are designed to work with JAX arrays and maintain AD compatibility
4. Export methods handle conversion from JAX to standard Python/NumPy types
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime
import json
import numpy as np
import jax.numpy as jnp
import pandas as pd


class BaseResult(ABC):
    """
    Abstract base class for all result containers.

    Defines the common interface that all result types must implement.
    Subclasses must provide validation and dictionary serialization.
    """

    @abstractmethod
    def validate(self) -> bool:
        """
        Validate the result container's internal consistency.

        Returns:
            bool: True if valid, raises exception otherwise
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the result to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary with all relevant data
        """
        pass

    @property
    @abstractmethod
    def df(self) -> pd.DataFrame:
        """
        Convert the result to a pandas DataFrame.

        Returns:
            pd.DataFrame: Tabular representation of the result
        """
        pass


class ArithmeticMixin:
    """
    Mixin providing arithmetic operations for result containers.

    Supports addition for aggregating multiple results (e.g., portfolio aggregation).
    Assumes the class has appropriate attributes (amount, risk_ladder, etc.) that can be added.

    IMPORTANT: This mixin uses duck typing - it calls methods/attributes that the
    concrete class must provide. Not all arithmetic makes sense for all types.
    """

    def __add__(self, other):
        """
        Add two result objects together.

        Subclasses should override this to provide type-specific addition logic.
        This default implementation returns NotImplemented to signal that the
        subclass should handle addition.

        Args:
            other: Another result object of the same type

        Returns:
            Combined result object or NotImplemented
        """
        # Default: delegate to subclass implementation
        return NotImplemented

    def __radd__(self, other):
        """
        Right-hand addition (supports sum() built-in).

        Allows result objects to be summed using sum([result1, result2, ...]).
        When sum() starts with 0, we return self; otherwise delegate to __add__.

        Args:
            other: Left operand (often 0 from sum())

        Returns:
            self if other is 0, otherwise delegates to __add__
        """
        if other == 0:
            return self
        return self.__add__(other)

    def __mul__(self, scalar):
        """
        Multiply result by a scalar.

        Subclasses should override to implement scaling logic.

        Args:
            scalar: Numeric multiplier

        Returns:
            Scaled result object or NotImplemented
        """
        return NotImplemented

    def __rmul__(self, scalar):
        """Right-hand multiplication (scalar * result)."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        """
        Divide result by a scalar.

        Subclasses should override to implement division logic.

        Args:
            scalar: Numeric divisor

        Returns:
            Scaled result object or NotImplemented
        """
        return NotImplemented


class ExportMixin:
    """
    Mixin providing export functionality to various formats.

    Requires the class to implement .to_dict() method.
    Handles conversion from JAX arrays to standard Python types for serialization.
    """

    def to_json(self, indent: Optional[int] = 2) -> str:
        """
        Export result to JSON string.

        Args:
            indent: Number of spaces for indentation (None for compact)

        Returns:
            str: JSON representation
        """
        data = self.to_dict()
        # Convert JAX/NumPy arrays to lists for JSON serialization
        return json.dumps(self._prepare_for_json(data), indent=indent)

    def to_csv(self, filepath: Optional[str] = None) -> Optional[str]:
        """
        Export result to CSV format.

        Args:
            filepath: Optional path to save CSV file. If None, returns CSV string.

        Returns:
            str or None: CSV string if filepath is None, otherwise None
        """
        df = self.df
        if filepath:
            df.to_csv(filepath)
            return None
        else:
            return df.to_csv()

    def to_excel(self, filepath: str, sheet_name: str = 'Sheet1'):
        """
        Export result to Excel file.

        Args:
            filepath: Path to save Excel file
            sheet_name: Name of the Excel sheet
        """
        df = self.df
        df.to_excel(filepath, sheet_name=sheet_name)

    @staticmethod
    def _prepare_for_json(obj):
        """
        Recursively convert JAX/NumPy arrays and custom types to JSON-serializable types.

        Args:
            obj: Object to prepare for JSON serialization

        Returns:
            JSON-serializable version of obj
        """
        if isinstance(obj, (jnp.ndarray, np.ndarray)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: ExportMixin._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [ExportMixin._prepare_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            # Handle enum types and other custom objects
            if hasattr(obj, 'value'):
                return obj.value
            elif hasattr(obj, 'name'):
                return obj.name
            else:
                return str(obj)
        else:
            return obj


class VisualizationMixin:
    """
    Mixin providing visualization functionality.

    Currently provides stub methods for future implementation.
    Subclasses can override to provide meaningful visualizations.
    """

    def plot(self, **kwargs):
        """
        Create a visualization of the result.

        Subclasses should override this method to provide specific visualization.
        For example, Gamma might show a heatmap, Delta might show a bar chart.

        Args:
            **kwargs: Plotting options (backend-specific)

        Returns:
            Plot object (implementation-specific)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.plot() is not yet implemented. "
            "This is a placeholder for future visualization functionality."
        )

    def summary(self) -> str:
        """
        Return a text summary of the result.

        Returns:
            str: Human-readable summary
        """
        return str(self)


class AggregationMixin:
    """
    Mixin providing aggregation functionality for collections of results.

    Useful for Risk and Cashflows containers that hold multiple sub-results.
    """

    def sum(self):
        """
        Sum all elements in the container.

        Subclasses should override to provide meaningful aggregation.

        Returns:
            Aggregated result
        """
        return NotImplemented

    def aggregate(self, func):
        """
        Apply a custom aggregation function.

        Args:
            func: Aggregation function to apply

        Returns:
            Aggregated result
        """
        return NotImplemented


class ValidationMixin:
    """
    Mixin providing common validation utilities.

    Helps ensure consistency across result containers.
    """

    @staticmethod
    def validate_no_nan(arr: jnp.ndarray, name: str = "array") -> bool:
        """
        Validate that array contains no NaN values.

        Args:
            arr: Array to check
            name: Name for error message

        Returns:
            bool: True if valid

        Raises:
            ValueError: If NaN values found
        """
        if jnp.any(jnp.isnan(arr)):
            raise ValueError(f"{name} contains NaN values")
        return True

    @staticmethod
    def validate_no_inf(arr: jnp.ndarray, name: str = "array") -> bool:
        """
        Validate that array contains no infinite values.

        Args:
            arr: Array to check
            name: Name for error message

        Returns:
            bool: True if valid

        Raises:
            ValueError: If infinite values found
        """
        if jnp.any(jnp.isinf(arr)):
            raise ValueError(f"{name} contains infinite values")
        return True

    @staticmethod
    def validate_shape_match(arr: jnp.ndarray, tenors: list, name: str = "array") -> bool:
        """
        Validate that array shape matches tenor count.

        Args:
            arr: Array to check
            tenors: List of tenor labels
            name: Name for error message

        Returns:
            bool: True if valid

        Raises:
            ValueError: If shapes don't match
        """
        if arr.shape[0] != len(tenors):
            raise ValueError(
                f"{name} shape {arr.shape} doesn't match tenor count {len(tenors)}"
            )
        return True

    @staticmethod
    def validate_currency_match(currency1, currency2, operation: str = "operation") -> bool:
        """
        Validate that two currencies match.

        Args:
            currency1: First currency
            currency2: Second currency
            operation: Operation name for error message

        Returns:
            bool: True if valid

        Raises:
            ValueError: If currencies don't match
        """
        if currency1 != currency2:
            raise ValueError(
                f"Cannot perform {operation} with different currencies: "
                f"{currency1} vs {currency2}"
            )
        return True
