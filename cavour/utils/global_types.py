from enum import Enum


class SwapTypes(Enum):
    PAY = 1
    RECEIVE = 2

class InstrumentTypes(Enum):
    SWAP_FIXED_LEG = 1
    SWAP_FLOAT_LEG = 2
    OIS_SWAP = 3

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
    SONIA = 1
    SOFR = 2
    ESTR = 3
