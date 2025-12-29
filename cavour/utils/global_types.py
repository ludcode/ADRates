"""
Global type enumerations for instruments, requests, and curves.

Provides enumeration types used throughout the Cavour library for:
- Swap leg directions (PAY/RECEIVE)
- Instrument types (OIS, swap legs)
- Request types for calculations (VALUE, DELTA, GAMMA)
- Interpolation methods for curves
- Curve identifiers for major OIS curves

These types ensure type safety and provide clear semantics for
function arguments and return values across the library.

Enumerations:
- SwapTypes: PAY or RECEIVE for swap leg direction
- InstrumentTypes: Classification of financial instruments
- RequestTypes: Types of calculations to perform
- InterpTypes: Curve interpolation methods
- CurveTypes: Named curve identifiers

Interpolation types include:
- FLAT_FWD_RATES: Piecewise constant forward rates
- LINEAR_ZERO_RATES: Linear interpolation of zero rates
- NATCUBIC_ZERO_RATES: Natural cubic spline on zero rates
- PCHIP_ZERO_RATES: PCHIP (monotonic cubic) on zero rates
- And others for log-discount and forward rate interpolation

Example:
    >>> # Create a paying swap
    >>> swap = OIS(
    ...     effective_dt=value_dt,
    ...     term_dt_or_tenor="10Y",
    ...     fixed_leg_type=SwapTypes.PAY,
    ...     fixed_coupon=0.045
    ... )
    >>>
    >>> # Request delta calculation
    >>> pos = swap.position(model)
    >>> results = pos.compute([RequestTypes.DELTA])
    >>>
    >>> # Build curve with interpolation
    >>> curve = OISCurve(
    ...     value_dt=value_dt,
    ...     ois_swaps=swaps,
    ...     interp_type=InterpTypes.LINEAR_ZERO_RATES
    ... )
"""

from enum import Enum


class SwapTypes(Enum):
    PAY = 1
    RECEIVE = 2

class InstrumentTypes(Enum):
    SWAP_FIXED_LEG = 1
    SWAP_FLOAT_LEG = 2
    OIS_SWAP = 3
    XCCY_SWAP = 4

class RequestTypes(Enum):
    VALUE = 1
    DELTA = 2
    GAMMA = 3
    SPEED = 4
    CASHFLOWS = 5

class InterpTypes(Enum):
    FLAT_FWD_RATES = 1
    LINEAR_FWD_RATES = 2
    LINEAR_ZERO_RATES = 4
    FINCUBIC_ZERO_RATES = 7
    NATCUBIC_LOG_DISCOUNT = 8
    NATCUBIC_ZERO_RATES = 9
    PCHIP_ZERO_RATES = 10
    PCHIP_LOG_DISCOUNT = 11

class CurveTypes(Enum):
    GBP_OIS_SONIA = 1
    USD_OIS_SOFR = 2
    EUR_OIS_ESTR = 3
    USD_GBP_BASIS = 4
