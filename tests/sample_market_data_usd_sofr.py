"""
Sample USD SOFR market data for testing multi-instrument curve construction.

This file contains realistic market rates as of Feb 26, 2026 (simulated).
Used for reproducible testing of curve bootstrapping with:
- Cash deposits (0-3M)
- FRAs (3M-18M)
- IR Futures (IMM quarterly, 3M-2Y)
- OIS swaps (2Y+)

Market Assumptions:
- Fed funds rate: ~5.25%
- SOFR overnight: ~5.30%
- Curve shape: Slightly inverted near-term, flat long-term
- Based on typical 2024-2026 rate environment
"""

from cavour.utils.date import Date
from cavour.utils.day_count import DayCountTypes
from cavour.utils.global_types import CurveTypes, SwapTypes, FutureContractTypes
from cavour.utils.currency import CurrencyTypes
from cavour.utils.frequency import FrequencyTypes

# Valuation date
VALUE_DATE = Date(26, 2, 2026)

# ============================================================================
# CASH DEPOSITS (0-3M) - ACT/360
# ============================================================================
# Rates reflect short-term SOFR compounding
DEPOSITS = {
    "tickers": ["USOSFRA", "USOSFRB", "USOSFRC"],
    "tenors": ["1M", "2M", "3M"],
    "rates": [5.3000, 5.3200, 5.3500],  # % per annum
}

# ============================================================================
# FRAs (3M-18M) - ACT/360
# ============================================================================
# Forward Rate Agreements for 3-month SOFR periods
FRAS = {
    "tickers": ["3x6", "6x9", "9x12", "12x15", "15x18"],
    "notations": ["3x6", "6x9", "9x12", "12x15", "15x18"],
    "rates": [5.3800, 5.4000, 5.4200, 5.4300, 5.4400],  # % per annum
}

# ============================================================================
# IR FUTURES (IMM Quarterly, ~3M-2Y) - Priced as 100 - implied rate
# ============================================================================
# SOFR 3-Month futures (CME)
# Contract codes: H=Mar, M=Jun, U=Sep, Z=Dec
# As of Feb 26, 2026:
#   - IMM1 (H26 = Mar 2026): expires ~3M from now
#   - IMM2 (M26 = Jun 2026): expires ~6M from now
#   - etc.
FUTURES = {
    "tickers": [
        "SFRH26 Comdty",  # Mar 2026
        "SFRM26 Comdty",  # Jun 2026
        "SFRU26 Comdty",  # Sep 2026
        "SFRZ26 Comdty",  # Dec 2026
        "SFRH27 Comdty",  # Mar 2027
        "SFRM27 Comdty",  # Jun 2027
        "SFRU27 Comdty",  # Sep 2027
        "SFRZ27 Comdty",  # Dec 2027
    ],
    "contract_codes": ["H26", "M26", "U26", "Z26", "H27", "M27", "U27", "Z27"],
    "prices": [94.6200, 94.6000, 94.5800, 94.5700, 94.5600, 94.5500, 94.5400, 94.5300],
    # Implied rates: 5.38%, 5.40%, 5.42%, 5.43%, 5.44%, 5.45%, 5.46%, 5.47%
}

# ============================================================================
# OIS SWAPS (2Y+) - ACT/360, Annual frequency
# ============================================================================
# USD SOFR OIS swap rates (fixed vs compounded SOFR)
OIS_SWAPS = {
    "tickers": [
        "USOSFR2 BGNL Curncy",
        "USOSFR3 BGNL Curncy",
        "USOSFR5 BGNL Curncy",
        "USOSFR7 BGNL Curncy",
        "USOSFR10 BGNL Curncy",
        "USOSFR15 BGNL Curncy",
        "USOSFR20 BGNL Curncy",
        "USOSFR30 BGNL Curncy",
    ],
    "tenors": ["2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"],
    "rates": [5.4500, 5.4300, 5.3800, 5.3400, 5.2800, 5.2200, 5.1800, 5.1500],  # % per annum
    # Slightly inverted 2Y-3Y, then declining long-term (typical term premium compression)
}

# ============================================================================
# MARKET CONVENTIONS
# ============================================================================
USD_SOFR_CONVENTIONS = {
    "day_count": DayCountTypes.ACT_360,
    "fixed_frequency": FrequencyTypes.ANNUAL,
    "float_frequency": FrequencyTypes.ANNUAL,
    "payment_lag": 2,  # T+2 settlement
    "currency": CurrencyTypes.USD,
    "curve_type": CurveTypes.USD_OIS_SOFR,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_deposit_instruments():
    """Create CashDeposit instruments from sample data."""
    from cavour.trades.rates.cash_deposit import CashDeposit

    deposits = []
    for tenor, rate in zip(DEPOSITS["tenors"], DEPOSITS["rates"]):
        dep = CashDeposit(
            effective_dt=VALUE_DATE,
            term_dt_or_tenor=tenor,
            deposit_rate=rate / 100.0,  # Convert % to decimal
            dc_type=USD_SOFR_CONVENTIONS["day_count"],
            floating_index=USD_SOFR_CONVENTIONS["curve_type"],
            currency=USD_SOFR_CONVENTIONS["currency"],
            notional=1_000_000
        )
        deposits.append(dep)
    return deposits


def get_fra_instruments():
    """Create FRA instruments from sample data."""
    from cavour.trades.rates.fra import FRA

    fras = []
    for notation, rate in zip(FRAS["notations"], FRAS["rates"]):
        fra = FRA(
            effective_dt=VALUE_DATE,
            fra_notation=notation,
            fra_rate=rate / 100.0,  # Convert % to decimal
            dc_type=USD_SOFR_CONVENTIONS["day_count"],
            floating_index=USD_SOFR_CONVENTIONS["curve_type"],
            currency=USD_SOFR_CONVENTIONS["currency"],
            notional=1_000_000
        )
        fras.append(fra)
    return fras


def get_future_instruments():
    """Create IRFuture instruments from sample data."""
    from cavour.trades.rates.ir_future import IRFuture

    futures = []
    for contract_code, price in zip(FUTURES["contract_codes"], FUTURES["prices"]):
        fut = IRFuture(
            effective_dt=VALUE_DATE,
            expiry_date_or_contract=contract_code,
            futures_price=price,
            contract_type=FutureContractTypes.IMM,
            currency=USD_SOFR_CONVENTIONS["currency"],
            floating_index=USD_SOFR_CONVENTIONS["curve_type"],
            dc_type=USD_SOFR_CONVENTIONS["day_count"],
            contract_size=1_000_000
        )
        futures.append(fut)
    return futures


def get_ois_instruments():
    """Create OIS instruments from sample data."""
    from cavour.trades.rates.ois import OIS

    swaps = []
    for tenor, rate in zip(OIS_SWAPS["tenors"], OIS_SWAPS["rates"]):
        swap = OIS(
            effective_dt=VALUE_DATE,
            term_dt_or_tenor=tenor,
            fixed_leg_type=SwapTypes.PAY,
            fixed_coupon=rate / 100.0,  # Convert % to decimal
            fixed_freq_type=USD_SOFR_CONVENTIONS["fixed_frequency"],
            fixed_dc_type=USD_SOFR_CONVENTIONS["day_count"],
            floating_index=USD_SOFR_CONVENTIONS["curve_type"],
            currency=USD_SOFR_CONVENTIONS["currency"],
            float_freq_type=USD_SOFR_CONVENTIONS["float_frequency"],
            float_dc_type=USD_SOFR_CONVENTIONS["day_count"],
            notional=1_000_000
        )
        swaps.append(swap)
    return swaps


# ============================================================================
# MAIN DATA EXPORT
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("USD SOFR Sample Market Data")
    print(f"Value Date: {VALUE_DATE}")
    print("=" * 80)

    print("\nCASH DEPOSITS (0-3M):")
    for tenor, rate in zip(DEPOSITS["tenors"], DEPOSITS["rates"]):
        print(f"  {tenor:>4s}: {rate:>6.4f}%")

    print("\nFRAs (3M-18M):")
    for notation, rate in zip(FRAS["notations"], FRAS["rates"]):
        print(f"  {notation:>5s}: {rate:>6.4f}%")

    print("\nIR FUTURES (IMM Quarterly):")
    for code, price in zip(FUTURES["contract_codes"], FUTURES["prices"]):
        implied_rate = 100.0 - price
        print(f"  {code:>5s}: {price:>7.4f} (implied {implied_rate:.4f}%)")

    print("\nOIS SWAPS (2Y+):")
    for tenor, rate in zip(OIS_SWAPS["tenors"], OIS_SWAPS["rates"]):
        print(f"  {tenor:>4s}: {rate:>6.4f}%")
