# Money Market Instruments in Cavour

## Overview

Cavour now supports building interest rate curves from three types of money market instruments:

1. **Cash Deposits** - Short-term funding instruments (0-12M)
2. **Forward Rate Agreements (FRAs)** - Interest rate forwards (3M-2Y)
3. **OIS Swaps** - Overnight index swaps (1Y-50Y)

This allows construction of complete term structures using the most liquid instruments for each maturity segment.

## Motivation

Prior to this enhancement, Cavour only supported OIS swaps for curve construction. However, in practice, the short end of interest rate curves (0-2Y) is typically bootstrapped from:
- **Deposits** for the 0-12M segment (most liquid for very short maturities)
- **FRAs or Futures** for the 3M-2Y segment (liquid forward markets)
- **OIS swaps** for the 2Y+ segment (standard for long-dated discounting)

This mixed-instrument approach provides:
- **Better liquidity** - Uses the most actively traded instruments per tenor
- **Tighter bid-ask spreads** - Money market instruments trade with lower spreads than short-dated swaps
- **Market reality** - Reflects actual curve construction practices used by dealers

---

## Cash Deposits

### Description

A cash deposit is a short-term investment where:
- Principal is invested at the effective date
- Interest accrues at a fixed rate using simple (not compounded) interest
- Principal + interest is returned at maturity

### Market Conventions

| Currency | Day Count | Typical Maturities |
|----------|-----------|-------------------|
| USD      | ACT/360   | O/N, 1W, 1M, 2M, 3M, 6M, 9M, 12M |
| GBP      | ACT/365F  | O/N, 1W, 1M, 2M, 3M, 6M, 9M, 12M |
| EUR      | ACT/360   | O/N, 1W, 1M, 2M, 3M, 6M, 9M, 12M |

### Usage Example

```python
from cavour.trades.rates.cash_deposit import CashDeposit
from cavour.utils.date import Date
from cavour.utils.day_count import DayCountTypes
from cavour.utils.global_types import CurveTypes
from cavour.utils.currency import CurrencyTypes

# Create a 3M USD deposit at 5.25%
value_dt = Date(15, 6, 2023)
deposit = CashDeposit(
    effective_dt=value_dt,
    term_dt_or_tenor="3M",
    deposit_rate=0.0525,  # 5.25% as decimal
    dc_type=DayCountTypes.ACT_360,
    floating_index=CurveTypes.USD_OIS_SOFR,
    currency=CurrencyTypes.USD,
    notional=1_000_000
)

# Value the deposit (requires a discount curve)
pv = deposit.value(value_dt, discount_curve=curve)

# Extract implied rate from curve
implied_rate = deposit.implied_rate(value_dt, curve)
```

### Key Methods

- `value(value_dt, discount_curve)` - Compute present value
- `implied_rate(value_dt, discount_curve)` - Extract implied deposit rate from curve
- `position(model)` - Create Position for risk calculations (DELTA, GAMMA)

---

## Forward Rate Agreements (FRAs)

### Description

A FRA is an OTC contract where two parties agree to exchange interest payments based on a notional principal for a future period. The buyer locks in a borrowing rate.

**Valuation Formula:**
```
Forward_Rate = (DF(start) / DF(end) - 1) / year_frac
Payoff = Notional × (Forward_Rate - FRA_Rate) × Year_Frac
PV = Payoff × DF(settlement)
```

### FRA Notation

Standard FRA notation "AxB":
- **A** = Months from effective date to fixing date
- **B** = Months from effective date to end date
- **(B-A)** = Accrual period length

| Notation | Fixing | Accrual Period | Description |
|----------|--------|----------------|-------------|
| 3x6      | 3M     | 3M-6M (3M)     | 3M forward starting 3M deposit |
| 6x9      | 6M     | 6M-9M (3M)     | 6M forward starting 3M deposit |
| 9x12     | 9M     | 9M-12M (3M)    | 9M forward starting 3M deposit |
| 6x12     | 6M     | 6M-12M (6M)    | 6M forward starting 6M deposit |

### Market Conventions

| Currency | Day Count | Settlement | Typical Maturities |
|----------|-----------|------------|-------------------|
| USD      | ACT/360   | T+2        | 3x6, 6x9, 9x12, 12x18, 18x24 |
| GBP      | ACT/365F  | T+2        | 3x6, 6x9, 9x12, 12x15, 15x18 |
| EUR      | ACT/360   | T+2        | 3x6, 6x9, 9x12, 12x18, 18x24 |

### Usage Example

```python
from cavour.trades.rates.fra import FRA

# Create a 3x6 FRA using notation
value_dt = Date(15, 6, 2023)
fra = FRA(
    effective_dt=value_dt,
    fra_notation="3x6",
    fra_rate=0.0525,  # 5.25% as decimal
    dc_type=DayCountTypes.ACT_360,
    floating_index=CurveTypes.USD_OIS_SOFR,
    currency=CurrencyTypes.USD,
    notional=1_000_000
)

# Alternative: Create FRA with explicit dates
fixing_dt = value_dt.add_months(3)
start_dt = fixing_dt.add_days(2)  # T+2 settlement
end_dt = start_dt.add_months(3)

fra_explicit = FRA(
    effective_dt=value_dt,
    fixing_dt=fixing_dt,
    start_dt=start_dt,
    end_dt=end_dt,
    fra_rate=0.0525,
    dc_type=DayCountTypes.ACT_360,
    floating_index=CurveTypes.USD_OIS_SOFR,
    currency=CurrencyTypes.USD
)

# Value the FRA
pv = fra.value(value_dt, discount_curve=curve)

# Extract forward rate
fwd_rate = fra.forward_rate(value_dt, curve)

# Extract implied FRA rate (equals forward rate)
implied_fra_rate = fra.implied_fra_rate(value_dt, curve)
```

### Key Methods

- `value(value_dt, discount_curve)` - Compute present value
- `forward_rate(value_dt, discount_curve)` - Extract forward rate from curve
- `implied_fra_rate(value_dt, discount_curve)` - Extract implied FRA rate (equals forward rate)
- `position(model)` - Create Position for risk calculations

---

## Building Curves with Mixed Instruments

### Using Model.build_curve()

The `Model.build_curve()` method now accepts an `instrument_type` parameter:

```python
from cavour.models.models import Model
from cavour.utils.date import Date

value_dt = Date(15, 6, 2023)
model = Model(value_dt)

# Build deposit curve (0-12M)
model.build_curve(
    name="USD_OIS_SOFR",
    px_list=[5.00, 5.10, 5.20, 5.30],  # Rates in percentage
    tenor_list=["1M", "2M", "3M", "6M"],
    instrument_type="DEPOSIT",  # NEW parameter
    fixed_dcc_type=DayCountTypes.ACT_360
)

# Build FRA curve (3M-2Y)
model.build_curve(
    name="USD_OIS_SOFR",
    px_list=[5.25, 5.30, 5.35, 5.40],
    tenor_list=["3x6", "6x9", "9x12", "12x18"],
    instrument_type="FRA",  # FRA notation
    fixed_dcc_type=DayCountTypes.ACT_360
)

# Build OIS curve (2Y+)
model.build_curve(
    name="USD_OIS_SOFR",
    px_list=[5.40, 5.50, 5.60, 5.50, 5.30],
    tenor_list=["2Y", "3Y", "5Y", "10Y", "30Y"],
    instrument_type="OIS",  # Default
    fixed_dcc_type=DayCountTypes.ACT_360
)
```

### Direct OISCurve Construction

For more control, create instruments directly and pass to `OISCurve`:

```python
from cavour.trades.rates.ois_curve import OISCurve
from cavour.trades.rates.cash_deposit import CashDeposit
from cavour.trades.rates.fra import FRA
from cavour.trades.rates.ois import OIS
from cavour.market.curves.interpolator import InterpTypes

value_dt = Date(15, 6, 2023)

# Build complete term structure
instruments = [
    # Short end: Deposits (0-6M)
    CashDeposit(value_dt, "1M", 0.0500, DayCountTypes.ACT_360,
               CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
    CashDeposit(value_dt, "3M", 0.0510, DayCountTypes.ACT_360,
               CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
    CashDeposit(value_dt, "6M", 0.0520, DayCountTypes.ACT_360,
               CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),

    # Medium term: FRAs (6M-18M)
    FRA(value_dt, "3x6", 0.0525, DayCountTypes.ACT_360,
       CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
    FRA(value_dt, "6x9", 0.0530, DayCountTypes.ACT_360,
       CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
    FRA(value_dt, "9x12", 0.0535, DayCountTypes.ACT_360,
       CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),

    # Long end: OIS swaps (2Y+)
    OIS(value_dt, "2Y", SwapTypes.PAY, 0.0540, FrequencyTypes.ANNUAL,
       DayCountTypes.ACT_360, CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
    OIS(value_dt, "5Y", SwapTypes.PAY, 0.0550, FrequencyTypes.ANNUAL,
       DayCountTypes.ACT_360, CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
    OIS(value_dt, "10Y", SwapTypes.PAY, 0.0560, FrequencyTypes.ANNUAL,
       DayCountTypes.ACT_360, CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
]

# Build curve (instruments are automatically sorted by maturity)
curve = OISCurve(
    value_dt=value_dt,
    instruments=instruments,  # NEW parameter (replaces ois_swaps)
    interp_type=InterpTypes.LINEAR_ZERO_RATES,
    check_refit=True,  # Verify all instruments reprice correctly
    use_ad=True,  # Enable automatic differentiation for DELTA
    compute_gamma=False  # GAMMA computation (slower, optional)
)
```

### Backward Compatibility

The old `ois_swaps` parameter is still supported but deprecated:

```python
# Old way (still works with deprecation warning)
curve = OISCurve(value_dt=value_dt, ois_swaps=swaps)

# New way (preferred)
curve = OISCurve(value_dt=value_dt, instruments=instruments)
```

---

## Risk Calculations

All instruments support computing VALUE, DELTA, and GAMMA via the Position framework:

```python
from cavour.utils.global_types import RequestTypes

# Create deposit
deposit = CashDeposit(value_dt, "3M", 0.05, DayCountTypes.ACT_360,
                     CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD)

# Create position
pos = deposit.position(model)

# Compute analytics
result = pos.compute([RequestTypes.VALUE, RequestTypes.DELTA])

print(f"Value: {result.value}")
print(f"Delta (1bp sensitivities): {result.risk}")

# GAMMA computation (if curve built with compute_gamma=True)
result_gamma = pos.compute([RequestTypes.GAMMA])
print(f"Gamma (second-order sensitivities): {result.gamma}")
```

### DELTA Computation

DELTA sensitivities are computed via automatic differentiation:
- **Deposits**: `dPV/d(rates) = payment_amt × d(DF)/d(rates)`
- **FRAs**: Chain rule applied to forward rate and settlement DF
- **OIS**: Sensitivities to all cashflow dates

### GAMMA Computation

GAMMA (second-order) sensitivities require:
```python
curve = OISCurve(
    value_dt=value_dt,
    instruments=instruments,
    use_ad=True,
    compute_gamma=True,  # Enable Hessian computation
    hessian_bandwidth=0  # Optional: diagonal-only for 20x speedup
)
```

**Note:** FRA GAMMA is not yet implemented. Use finite differences if needed.

---

## Curve Refit Accuracy

All instruments should reprice to near-zero when valued on their own bootstrapped curve. Use `check_refit=True` to verify:

```python
curve = OISCurve(
    value_dt=value_dt,
    instruments=instruments,
    check_refit=True  # Raises LibError if any instrument fails to reprice
)
```

**Tolerance:** Instruments must reprice within `1e-10` relative to notional (default).

---

## Best Practices

### 1. Instrument Selection by Maturity

| Maturity | Recommended Instrument | Rationale |
|----------|------------------------|-----------|
| 0-3M     | Cash Deposits          | Most liquid, tightest spreads |
| 3M-2Y    | FRAs or Futures        | Active forward markets |
| 2Y+      | OIS Swaps              | Standard for discounting |

### 2. Day Count Consistency

Ensure all instruments for a given curve use consistent day count conventions:
- **USD**: ACT/360
- **GBP**: ACT/365F
- **EUR**: ACT/360

### 3. Automatic Sorting

`OISCurve` automatically sorts instruments by maturity before bootstrapping. No need to pre-sort.

### 4. Interpolation Methods

For mixed-instrument curves, recommend:
- `InterpTypes.LINEAR_ZERO_RATES` - Smooth zero rate interpolation
- `InterpTypes.FLAT_FWD_RATES` - Piecewise constant forward rates (conservative)

Avoid cubic splines for curves with few data points.

### 5. AD Performance

For large portfolios:
- Set `use_ad=True` for DELTA (fast)
- Set `compute_gamma=False` unless GAMMA is required (3-5x speedup)
- Use `hessian_bandwidth=0` for diagonal-only GAMMA (20x speedup, empirically accurate for OIS)

---

## Examples

### Example 1: USD Deposit Curve

```python
from cavour.models.models import Model

model = Model(Date(15, 6, 2023))

model.build_curve(
    name="USD_OIS_SOFR",
    px_list=[5.00, 5.10, 5.20, 5.30, 5.40],
    tenor_list=["1M", "2M", "3M", "6M", "12M"],
    instrument_type="DEPOSIT",
    fixed_dcc_type=DayCountTypes.ACT_360,
    use_ad=True
)

curve = model.curves.USD_OIS_SOFR
print(f"Curve built with {len(curve.swap_rates)} deposits")
```

### Example 2: GBP FRA Curve

```python
model.build_curve(
    name="GBP_OIS_SONIA",
    px_list=[4.75, 4.80, 4.85, 4.90],
    tenor_list=["3x6", "6x9", "9x12", "12x15"],
    instrument_type="FRA",
    fixed_dcc_type=DayCountTypes.ACT_365F,
    use_ad=True
)
```

### Example 3: Mixed USD Term Structure

```python
# Build complete term structure in stages
value_dt = Date(15, 6, 2023)

# Stage 1: Short end (deposits)
deposits = [
    CashDeposit(value_dt, "1M", 0.0500, DayCountTypes.ACT_360,
               CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
    CashDeposit(value_dt, "3M", 0.0510, DayCountTypes.ACT_360,
               CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
]

# Stage 2: Medium term (FRAs)
fras = [
    FRA(value_dt, "3x6", 0.0520, DayCountTypes.ACT_360,
       CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
    FRA(value_dt, "6x9", 0.0525, DayCountTypes.ACT_360,
       CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
]

# Stage 3: Long end (OIS)
swaps = [
    OIS(value_dt, "2Y", SwapTypes.PAY, 0.0530, FrequencyTypes.ANNUAL,
       DayCountTypes.ACT_360, CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
    OIS(value_dt, "5Y", SwapTypes.PAY, 0.0540, FrequencyTypes.ANNUAL,
       DayCountTypes.ACT_360, CurveTypes.USD_OIS_SOFR, CurrencyTypes.USD),
]

# Combine and build
all_instruments = deposits + fras + swaps
curve = OISCurve(value_dt=value_dt, instruments=all_instruments,
                interp_type=InterpTypes.LINEAR_ZERO_RATES, check_refit=True)

print(f"Built {len(curve.swap_times)}-point curve")
print(f"Maturities: {curve.swap_times}")
```

---

## Testing

Comprehensive test suites are provided:
- `tests/test_cash_deposit.py` - Deposit unit tests
- `tests/test_fra.py` - FRA unit tests
- `tests/test_money_market_curve.py` - Integration tests for mixed curves

Run tests:
```bash
pytest tests/test_cash_deposit.py -v
pytest tests/test_fra.py -v
pytest tests/test_money_market_curve.py -v
```

---

## Future Enhancements

Planned additions:
1. **Interest Rate Futures** - CME 3M SOFR futures, Eurodollar futures
2. **Convexity Adjustments** - Futures-to-forward rate conversion
3. **Tenor Basis Swaps** - 1M vs 3M SOFR basis
4. **FRA GAMMA** - Second-order sensitivities for FRAs

---

## References

- ISDA OIS Conventions: https://www.isda.org/
- FRA Market Conventions: Various dealer documentation
- SOFR Transition: https://www.newyorkfed.org/markets/reference-rates/sofr

---

## Contact

For questions or issues, please file a GitHub issue or contact the Cavour development team.
