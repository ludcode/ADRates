"""
Currency type enumeration for multi-currency support.

Provides currency codes for use in foreign exchange calculations,
cross-currency swaps, and multi-currency portfolio management.

Supported currencies:
- USD: US Dollar
- EUR: Euro
- GBP: British Pound Sterling
- CHF: Swiss Franc
- CAD: Canadian Dollar
- AUD: Australian Dollar
- NZD: New Zealand Dollar
- DKK: Danish Krone
- SEK: Swedish Krona
- HKD: Hong Kong Dollar
- JPY: Japanese Yen
- NOK: Norwegian Krone
- PLN: Polish Zloty
- RON: Romanian Leu
- NONE: No currency specified

Used in conjunction with:
- FX rate lookups (FXRoutingEngine)
- Cross-currency swap construction (XCCYCurve)
- Position currency conversion
- Model FX rate management

Example:
    >>> # Define currency for a position
    >>> currency = CurrencyTypes.GBP
    >>>
    >>> # Use in swap construction
    >>> swap = OIS(
    ...     effective_dt=value_dt,
    ...     term_dt_or_tenor="10Y",
    ...     currency=CurrencyTypes.USD,
    ...     floating_index=CurveTypes.USD_OIS_SOFR
    ... )
"""

from enum import Enum

###############################################################################

class CurrencyTypes(Enum):
    USD = 1
    EUR = 2
    GBP = 3
    CHF = 4
    CAD = 5
    AUD = 6
    NZD = 7
    DKK = 8
    SEK = 9
    HKD = 10
    JPY = 11
    NOK = 12
    PLN = 13
    RON = 14
    NONE = 15

###############################################################################