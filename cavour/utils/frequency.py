"""
Payment frequency types for fixed-income instruments.

Provides frequency enumeration and conversion utilities for coupon
payment frequencies used in bonds, swaps, and other interest rate products.

Frequency types:
- ZERO: Zero coupon (no periodic payments)
- SIMPLE: Simple interest (no compounding)
- ANNUAL: Once per year (frequency = 1)
- SEMI_ANNUAL: Twice per year (frequency = 2)
- TRI_ANNUAL: Three times per year (frequency = 3)
- QUARTERLY: Four times per year (frequency = 4)
- MONTHLY: Twelve times per year (frequency = 12)
- CONTINUOUS: Continuous compounding (frequency = -1)

The annual_frequency() function converts frequency types to their
numeric annual frequency for use in calculations.

Example:
    >>> # Get numeric frequency
    >>> freq = annual_frequency(FrequencyTypes.QUARTERLY)
    >>> print(freq)  # 4.0
    >>>
    >>> # Use in swap construction
    >>> swap = OIS(
    ...     effective_dt=value_dt,
    ...     term_dt_or_tenor="10Y",
    ...     fixed_freq_type=FrequencyTypes.ANNUAL,
    ...     fixed_coupon=0.045
    ... )
"""

from cavour.utils.error import LibError

from enum import Enum


class FrequencyTypes(Enum):
    ZERO = -1
    SIMPLE = 0
    ANNUAL = 1
    SEMI_ANNUAL = 2
    TRI_ANNUAL = 3
    QUARTERLY = 4
    MONTHLY = 12
    CONTINUOUS = 99


def annual_frequency(freq_type: FrequencyTypes):
    """ This is a function that takes in a Frequency Type and returns a
    float value for the number of times a year a payment occurs."""
    if isinstance(freq_type, FrequencyTypes) is False:
        print("FinFrequency:", freq_type)
        raise LibError("Unknown frequency type")

    if freq_type == FrequencyTypes.CONTINUOUS:
        return -1
    elif freq_type == FrequencyTypes.ZERO:
        # This means that there is no coupon and I use 1 to avoid div by zero
        return 1.0
    elif freq_type == FrequencyTypes.ANNUAL:
        return 1.0
    elif freq_type == FrequencyTypes.SEMI_ANNUAL:
        return 2.0
    elif freq_type == FrequencyTypes.TRI_ANNUAL:
        return 3.0
    elif freq_type == FrequencyTypes.QUARTERLY:
        return 4.0
    elif freq_type == FrequencyTypes.MONTHLY:
        return 12.0
