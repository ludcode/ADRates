from cavour.utils import *


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
            "interp_type": InterpTypes.LINEAR_ZERO_RATES
        },
        "currency": "GBP",
        "type": "OIS",
        "index": "SONIA"
    }
}

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