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
from cavour.utils.currency import CurrencyTypes


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

class CollateralType(Enum):
    """Collateral types for CSA agreements and discounting.

    Currency-based collateral uses OIS curves for discounting.
    Government bond collateral uses bond-specific curves (future).
    """
    # Currency-based collateral (OIS discounting)
    USD = 1
    GBP = 2
    EUR = 3
    JPY = 4
    CHF = 5
    AUD = 6
    CAD = 7

    # Government bond collateral (future extensibility)
    USD_TIPS = 10  # US Treasury Inflation-Protected Securities
    EUR_OATS = 11  # French government bonds (Obligations Assimilables du Trésor)
    EUR_BUNDS = 12  # German government bonds (Bundesanleihen)
    GBP_GILTS = 13  # UK government bonds
    JGB = 14       # Japanese Government Bonds

    # Special cases
    UNCOLLATERALIZED = 99


###############################################################################
# Collateral Helper Functions
###############################################################################


def collateral_to_currency(collateral_type: CollateralType) -> CurrencyTypes:
    """Map CollateralType to its underlying CurrencyTypes.

    Args:
        collateral_type: The collateral type

    Returns:
        Corresponding CurrencyTypes

    Raises:
        ValueError: If collateral type is UNCOLLATERALIZED or unsupported
    """
    mapping = {
        CollateralType.USD: CurrencyTypes.USD,
        CollateralType.GBP: CurrencyTypes.GBP,
        CollateralType.EUR: CurrencyTypes.EUR,
        CollateralType.JPY: CurrencyTypes.JPY,
        CollateralType.CHF: CurrencyTypes.CHF,
        CollateralType.AUD: CurrencyTypes.AUD,
        CollateralType.CAD: CurrencyTypes.CAD,
        CollateralType.USD_TIPS: CurrencyTypes.USD,  # TIPS are USD-denominated
        CollateralType.EUR_OATS: CurrencyTypes.EUR,  # OATs are EUR-denominated
        CollateralType.EUR_BUNDS: CurrencyTypes.EUR, # Bunds are EUR-denominated
        CollateralType.GBP_GILTS: CurrencyTypes.GBP, # Gilts are GBP-denominated
        CollateralType.JGB: CurrencyTypes.JPY,       # JGBs are JPY-denominated
    }

    if collateral_type not in mapping:
        raise ValueError(
            f"Cannot convert {collateral_type} to currency. "
            f"Use is_currency_collateral() to check first."
        )

    return mapping[collateral_type]


def is_currency_collateral(collateral_type: CollateralType) -> bool:
    """Check if collateral type is currency-based (OIS discounting).

    Args:
        collateral_type: The collateral type to check

    Returns:
        True if currency-based, False otherwise
    """
    currency_types = {
        CollateralType.USD, CollateralType.GBP, CollateralType.EUR,
        CollateralType.JPY, CollateralType.CHF, CollateralType.AUD,
        CollateralType.CAD
    }
    return collateral_type in currency_types


def is_bond_collateral(collateral_type: CollateralType) -> bool:
    """Check if collateral type is government bond-based.

    Args:
        collateral_type: The collateral type to check

    Returns:
        True if bond-based, False otherwise
    """
    bond_types = {
        CollateralType.USD_TIPS, CollateralType.EUR_OATS,
        CollateralType.EUR_BUNDS, CollateralType.GBP_GILTS,
        CollateralType.JGB
    }
    return collateral_type in bond_types


def get_discount_curve_name(
    cashflow_currency: CurrencyTypes,
    collateral_type: CollateralType
) -> str:
    """Generate discount curve name based on cashflow currency and collateral type.

    Args:
        cashflow_currency: Currency of the instrument cashflows
        collateral_type: Type of collateral for discounting

    Returns:
        Curve name string (e.g., "GBP_OIS_SONIA", "GBP_USD_XCCY", "GBP_TIPS_XCCY")

    Raises:
        ValueError: If collateral type is unsupported or UNCOLLATERALIZED

    Examples:
        >>> get_discount_curve_name(CurrencyTypes.GBP, CollateralType.GBP)
        'GBP_OIS_SONIA'  # Natural currency

        >>> get_discount_curve_name(CurrencyTypes.GBP, CollateralType.USD)
        'GBP_USD_XCCY'  # GBP cashflows, USD collateral

        >>> get_discount_curve_name(CurrencyTypes.EUR, CollateralType.EUR_BUNDS)
        'EUR_EUR_BUNDS_XCCY'  # EUR cashflows, Bunds collateral (future)
    """
    # Currency-based collateral (OIS or XCCY curves)
    if is_currency_collateral(collateral_type):
        collateral_ccy = collateral_to_currency(collateral_type)

        if cashflow_currency == collateral_ccy:
            # Natural currency discounting → use OIS curve
            # Map currency to OIS curve name
            ois_curves = {
                CurrencyTypes.USD: "USD_OIS_SOFR",
                CurrencyTypes.GBP: "GBP_OIS_SONIA",
                CurrencyTypes.EUR: "EUR_OIS_ESTR",
                CurrencyTypes.JPY: "JPY_OIS_TONAR",
                CurrencyTypes.CHF: "CHF_OIS_SARON",
                CurrencyTypes.AUD: "AUD_OIS_AONIA",
                CurrencyTypes.CAD: "CAD_OIS_CORRA",
            }

            if cashflow_currency not in ois_curves:
                raise ValueError(f"No OIS curve defined for {cashflow_currency}")

            return ois_curves[cashflow_currency]
        else:
            # Cross-currency discounting → use XCCY curve
            # Format: {CASHFLOW_CCY}_{COLLATERAL_CCY}_XCCY
            return f"{cashflow_currency.name}_{collateral_ccy.name}_XCCY"

    # Government bond collateral (bond-specific curves)
    elif is_bond_collateral(collateral_type):
        # Format: {CASHFLOW_CCY}_{BOND_TYPE}_XCCY
        # e.g., "GBP_USD_TIPS_XCCY" for GBP cashflows with TIPS collateral
        collateral_name = collateral_type.name  # "USD_TIPS", "EUR_BUNDS", etc.
        return f"{cashflow_currency.name}_{collateral_name}_XCCY"

    # Uncollateralized
    elif collateral_type == CollateralType.UNCOLLATERALIZED:
        raise ValueError(
            "Cannot generate curve name for UNCOLLATERALIZED. "
            "Uncollateralized discounting requires separate handling."
        )

    else:
        raise ValueError(f"Unsupported collateral type: {collateral_type}")
