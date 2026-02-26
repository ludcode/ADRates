"""
Market data constants for OIS curves and FX rates.

This module defines Bloomberg ticker mappings and market conventions for:
- OIS curves (SONIA, SOFR)
- Cross-currency basis swaps (XCCY)
- FX spot rates

Each curve definition includes:
- tickers: Mapping of tenors to Bloomberg tickers
- conventions: Day count, frequency, business day adjustments
- currency: Base currency for the curve
- type: Instrument type (OIS, XCCY)
- index: Reference rate (SONIA, SOFR, etc.)
"""

from cavour.utils import *


# OIS curve and XCCY basis swap market data configurations
# Structure: {curve_name: {tickers, conventions, currency, type, index}}
MARKET_DATA = {
    "GBP_OIS_SONIA" : {
        "tickers": {'1D': 'SONIO/N Index',
                    '1W': 'BPSWS1Z BGN Curncy',
                    '2W': 'BPSWS2Z BGN Curncy',
                    '1M': 'BPSWSA BGN Curncy',
                    '2M': 'BPSWSB BGN Curncy',
                    '3M': 'BPSWSC BGN Curncy',
                    '4M': 'BPSWSD BGN Curncy',
                    '5M': 'BPSWSE BGN Curncy',
                    '6M': 'BPSWSF BGN Curncy',
                    '7M': 'BPSWSG BGN Curncy',
                    '8M': 'BPSWSH BGN Curncy',
                    '9M': 'BPSWSI BGN Curncy',
                    '10M': 'BPSWSJ BGN Curncy',
                    '11M': 'BPSWSK BGN Curncy',
                    '1Y': 'BPSWS1 BGN Curncy',
                    '18M': 'BPSWS1F BGN Curncy',
                    '2Y': 'BPSWS2 BGN Curncy',
                    '3Y': 'BPSWS3 BGN Curncy',
                    '4Y': 'BPSWS4 BGN Curncy',
                    '5Y': 'BPSWS5 BGN Curncy',
                    '6Y': 'BPSWS6 BGN Curncy',
                    '7Y': 'BPSWS7 BGN Curncy',
                    '8Y': 'BPSWS8 BGN Curncy',
                    '9Y': 'BPSWS9 BGN Curncy',
                    '10Y': 'BPSWS10 BGN Curncy',
                    '12Y': 'BPSWS12 BGN Curncy',
                    '15Y': 'BPSWS15 BGN Curncy',
                    '20Y': 'BPSWS20 BGN Curncy',
                    '25Y': 'BPSWS25 BGN Curncy',
                    '30Y': 'BPSWS30 BGN Curncy',
                    '40Y': 'BPSWS40 BGN Curncy',
                    '50Y': 'BPSWS50 BGN Curncy'},
        "conventions": {
            "fixed_day_count": DayCountTypes.ACT_365F,
            "fixed_frequency": FrequencyTypes.ANNUAL,
            "business_day_adjustment": BusDayAdjustTypes.MODIFIED_FOLLOWING,
            "float_frequency": FrequencyTypes.ANNUAL,
            "float_day_count": DayCountTypes.ACT_365F,
            "interp_type": InterpTypes.LINEAR_ZERO_RATES,
            "payment_lag" : 0
        },
        "currency": "GBP",
        "type": "OIS",
        "index": "SONIA"
    },

    "USD_OIS_SOFR" : {
        "tickers": {
                    '1D': 'SOFRRATE Index',
                    "1M": "USOSFRA BGNL Curncy",
                    "2M": "USOSFRB BGNL Curncy",
                    "3M": "USOSFRC BGNL Curncy",
                    "4M": "USOSFRD BGNL Curncy",
                    "5M": "USOSFRE BGNL Curncy",
                    "6M": "USOSFRF BGNL Curncy",
                    "9M": "USOSFRI BGNL Curncy",
                    "1Y": "USOSFR1 BGNL Curncy",
                    "18M": "USOSFR1F BGNL Curncy",
                    "2Y": "USOSFR2 BGNL Curncy",
                    "3Y": "USOSFR3 BGNL Curncy",
                    "4Y": "USOSFR4 BGNL Curncy",
                    "5Y": "USOSFR5 BGNL Curncy",
                    "6Y": "USOSFR6 BGNL Curncy",
                    "7Y": "USOSFR7 BGNL Curncy",
                    "8Y": "USOSFR8 BGNL Curncy",
                    "9Y": "USOSFR9 BGNL Curncy",
                    "10Y": "USOSFR10 BGNL Curncy",
                    "12Y": "USOSFR12 BGNL Curncy",
                    "15Y": "USOSFR15 BGNL Curncy",
                    "20Y": "USOSFR20 BGNL Curncy",
                    "25Y": "USOSFR25 BGNL Curncy",
                    "30Y": "USOSFR30 BGNL Curncy",
                    "40Y": "USOSFR40 BGNL Curncy",
                    "50Y": "USOSFR50 BGNL Curncy"
                },
        "conventions": {
            "fixed_day_count": DayCountTypes.ACT_360,
            "fixed_frequency": FrequencyTypes.ANNUAL,
            "business_day_adjustment": BusDayAdjustTypes.MODIFIED_FOLLOWING,
            "float_frequency": FrequencyTypes.ANNUAL,
            "float_day_count": DayCountTypes.ACT_360,
            "interp_type": InterpTypes.LINEAR_ZERO_RATES,
            "payment_lag" : 2
        },
        "currency": "USD",
        "type": "OIS",
        "index": "SOFR"
    },

    "GBPUSD_XCCY_SONIA_SOFR" : {
        "tickers": {
            "3M": "BPXOQQC BGN Curncy",
            "6M": "BPXOQQF BGN Curncy",
            "9M": "BPXOQQI BGN Curncy",
            "1Y": "BPXOQQ1 BGN Curncy",
            "18M": "BPXOQQ1F BGN Curncy",
            "2Y": "BPXOQQ2 BGN Curncy",
            "3Y": "BPXOQQ3 BGN Curncy",
            "4Y": "BPXOQQ4 BGN Curncy",
            "5Y": "BPXOQQ5 BGN Curncy",
            "6Y": "BPXOQQ6 BGN Curncy",
            "7Y": "BPXOQQ7 BGN Curncy",
            "8Y": "BPXOQQ8 BGN Curncy",
            "9Y": "BPXOQQ9 BGN Curncy",
            "10Y": "BPXOQQ10 BGN Curncy",
            "12Y": "BPXOQQ12 BGN Curncy",
            "15Y": "BPXOQQ15 BGN Curncy",
            "20Y": "BPXOQQ20 BGN Curncy",
            "25Y": "BPXOQQ25 BGN Curncy",
            "30Y": "BPXOQQ30 BGN Curncy",
            "40Y": "BPXOQQ40 BGN Curncy",
            "50Y": "BPXOQQ50 BGN Curncy"
        },
        "conventions": {
            "fixed_day_count": DayCountTypes.ACT_360,
            "fixed_frequency": FrequencyTypes.ANNUAL,
            "business_day_adjustment": BusDayAdjustTypes.MODIFIED_FOLLOWING,
            "float_frequency": FrequencyTypes.ANNUAL,
            "float_day_count": DayCountTypes.ACT_360,
            "interp_type": InterpTypes.LINEAR_ZERO_RATES,
            "payment_lag" : 2
        },
        "currency": "GBPUSD",
        "type": "XCCY",
        "index": "SONIA-SOFR"
}
}

# FX spot rate configurations
# Structure: {pair_name: {base, quote, ticker}}
# Includes major USD pairs and European cross rates
FX_MARKET_DATA = {
    # USD Majors
    "EURUSD": {
        "base": CurrencyTypes.EUR,
        "quote": CurrencyTypes.USD,
        "ticker": "EURUSD Curncy"
    },
    "GBPUSD": {
        "base": CurrencyTypes.GBP,
        "quote": CurrencyTypes.USD,
        "ticker": "GBPUSD Curncy"
    },
    "USDCHF": {
        "base": CurrencyTypes.USD,
        "quote": CurrencyTypes.CHF,
        "ticker": "USDCHF Curncy"
    },
    "USDCAD": {
        "base": CurrencyTypes.USD,
        "quote": CurrencyTypes.CAD,
        "ticker": "USDCAD Curncy"
    },
    "AUDUSD": {
        "base": CurrencyTypes.AUD,
        "quote": CurrencyTypes.USD,
        "ticker": "AUDUSD Curncy"
    },
    "NZDUSD": {
        "base": CurrencyTypes.NZD,
        "quote": CurrencyTypes.USD,
        "ticker": "NZDUSD Curncy"
    },
    "USDJPY": {
        "base": CurrencyTypes.USD,
        "quote": CurrencyTypes.JPY,
        "ticker": "USDJPY Curncy"
    },
    "USDSEK": {
        "base": CurrencyTypes.USD,
        "quote": CurrencyTypes.SEK,
        "ticker": "USDSEK Curncy"
    },
    "USDNOK": {
        "base": CurrencyTypes.USD,
        "quote": CurrencyTypes.NOK,
        "ticker": "USDNOK Curncy"
    },
    "USDDKK": {
        "base": CurrencyTypes.USD,
        "quote": CurrencyTypes.DKK,
        "ticker": "USDDKK Curncy"
    },
    "USDHKD": {
        "base": CurrencyTypes.USD,
        "quote": CurrencyTypes.HKD,
        "ticker": "USDHKD Curncy"
    },

    # European currencies vs EUR
    "EURPLN": {
        "base": CurrencyTypes.EUR,
        "quote": CurrencyTypes.PLN,
        "ticker": "EURPLN Curncy"
    },
    "EURRON": {
        "base": CurrencyTypes.EUR,
        "quote": CurrencyTypes.RON,
        "ticker": "EURRON Curncy"
    },

    # Direct USD pairs (often traded too)
    "USDPLN": {
        "base": CurrencyTypes.USD,
        "quote": CurrencyTypes.PLN,
        "ticker": "USDPLN Curncy"
    },
    "USDRON": {
        "base": CurrencyTypes.USD,
        "quote": CurrencyTypes.RON,
        "ticker": "USDRON Curncy"
    }
}

###############################################################################
# STIR Futures Market Data
###############################################################################

# Futures month codes (CME standard)
# Used for constructing futures tickers (e.g., H=March, M=June, U=September, Z=December)
FUTURES_MONTH_CODES = {
    1: 'F',   # January
    2: 'G',   # February
    3: 'H',   # March (IMM)
    4: 'J',   # April
    5: 'K',   # May
    6: 'M',   # June (IMM)
    7: 'N',   # July
    8: 'Q',   # August
    9: 'U',   # September (IMM)
    10: 'V',  # October
    11: 'X',  # November
    12: 'Z'   # December (IMM)
}

# IMM months only (quarterly contracts, most liquid)
IMM_MONTH_CODES = {3: 'H', 6: 'M', 9: 'U', 12: 'Z'}

# Futures ticker roots by currency (Bloomberg convention)
FUTURES_TICKER_ROOTS = {
    "USD": "SFR",   # CME SOFR 3-Month Futures
    "GBP": "FS",    # ICE SONIA 3-Month Futures (Short Sterling replacement)
    "EUR": "FES"    # Eurex Three-Month ESTR Futures
}

# Reverse mapping for parsing tickers
TICKER_ROOT_TO_CURRENCY = {v: k for k, v in FUTURES_TICKER_ROOTS.items()}


def construct_futures_ticker(currency: str, month: int, year: int) -> str:
    """
    Construct Bloomberg STIR futures ticker.

    Args:
        currency: Currency code string ("USD", "GBP", "EUR")
        month: Month number (1-12)
        year: 4-digit year (e.g., 2024)

    Returns:
        Bloomberg ticker string (e.g., "SFRH24 Comdty")

    Examples:
        >>> construct_futures_ticker("USD", 3, 2024)
        'SFRH24 Comdty'  # SOFR March 2024

        >>> construct_futures_ticker("GBP", 6, 2024)
        'FSM24 Comdty'  # SONIA June 2024

        >>> construct_futures_ticker("EUR", 9, 2024)
        'FESU24 Comdty'  # ESTR September 2024

    Raises:
        ValueError: If currency not supported or month invalid
    """
    if currency not in FUTURES_TICKER_ROOTS:
        raise ValueError(
            f"Currency {currency} not supported for futures. "
            f"Supported: {list(FUTURES_TICKER_ROOTS.keys())}"
        )

    if month not in FUTURES_MONTH_CODES:
        raise ValueError(f"Month must be 1-12, got {month}")

    # Get ticker root for currency
    ticker_root = FUTURES_TICKER_ROOTS[currency]

    # Get month code
    month_code = FUTURES_MONTH_CODES[month]

    # Format year as 2 digits
    year_2digit = year % 100

    # Construct ticker: ROOT + MONTH_CODE + YEAR + " Comdty"
    ticker = f"{ticker_root}{month_code}{year_2digit:02d} Comdty"

    return ticker


def construct_futures_ticker_from_date(currency: str, date) -> str:
    """
    Construct futures ticker from a Date object.

    Args:
        currency: Currency code string ("USD", "GBP", "EUR")
        date: Date object for the futures expiry

    Returns:
        Bloomberg ticker string (e.g., "SFRH24 Comdty")

    Example:
        >>> from cavour.utils.date import Date
        >>> expiry = Date(20, 3, 2024)  # March 20, 2024 (IMM date)
        >>> construct_futures_ticker_from_date("USD", expiry)
        'SFRH24 Comdty'
    """
    return construct_futures_ticker(currency, date._m, date._y)


# Example quarterly IMM futures strips (for reference)
# These would typically be fetched dynamically from Bloomberg
FUTURES_EXAMPLES = {
    "USD_SOFR_IMM": {
        "description": "CME SOFR 3-Month Futures (IMM quarterly)",
        "exchange": "CME",
        "contract_size": 1_000_000,  # $1M
        "day_count": "ACT/360",
        "tick_size": 0.0025,  # 0.25 bps = $6.25 per contract
        "example_tickers": [
            "SFRH24 Comdty",  # March 2024
            "SFRM24 Comdty",  # June 2024
            "SFRU24 Comdty",  # September 2024
            "SFRZ24 Comdty",  # December 2024
        ]
    },
    "GBP_SONIA_IMM": {
        "description": "ICE SONIA 3-Month Futures (IMM quarterly)",
        "exchange": "ICE",
        "contract_size": 1_000_000,  # £1M
        "day_count": "ACT/365",
        "tick_size": 0.0025,  # 0.25 bps
        "example_tickers": [
            "FSH24 Comdty",  # March 2024
            "FSM24 Comdty",  # June 2024
            "FSU24 Comdty",  # September 2024
            "FSZ24 Comdty",  # December 2024
        ]
    },
    "EUR_ESTR_IMM": {
        "description": "Eurex Three-Month ESTR Futures (IMM quarterly)",
        "exchange": "Eurex",
        "contract_size": 1_000_000,  # €1M
        "day_count": "ACT/360",
        "tick_size": 0.0025,  # 0.25 bps
        "example_tickers": [
            "FESH24 Comdty",  # March 2024
            "FESM24 Comdty",  # June 2024
            "FESU24 Comdty",  # September 2024
            "FESZ24 Comdty",  # December 2024
        ]
    }
}