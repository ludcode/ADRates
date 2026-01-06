"""
Market indices package for inflation and commodity indices.

Provides index management for:
- Inflation indices (CPI, RPI, HICP) with historical fixings
- Index interpolation for intra-month dates
- Forward index projection from inflation curves
"""

from .inflation_index import InflationIndex

__all__ = ['InflationIndex']
