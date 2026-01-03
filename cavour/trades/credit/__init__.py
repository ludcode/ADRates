"""
Credit instruments module for Cavour.

This module contains implementations of credit-sensitive fixed income instruments
including bonds, credit default swaps, and other credit derivatives.
"""

from .bond import Bond
from .frn import FRN

__all__ = ['Bond', 'FRN']
