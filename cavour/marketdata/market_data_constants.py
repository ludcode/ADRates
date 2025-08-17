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