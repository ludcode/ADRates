"""
Sensitivity computation module for automatic differentiation.

Provides centralized DELTA and GAMMA calculation using JAX.
"""

from .sensitivity_engine import SensitivityEngine

__all__ = ['SensitivityEngine']
