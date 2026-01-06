# Cavour (ADRates)

**Cavour** is a modern, Python-based quantitative finance library for pricing and risk management of fixed income derivatives using **Algorithmic Differentiation (AD)**. Built on JAX, Cavour computes Greeks (delta, gamma) directly through automatic differentiation rather than finite differences, providing exact sensitivities with superior performance.

## Key Features

- **Comprehensive Product Coverage**
  - OIS (Overnight Index Swaps): SONIA, SOFR, ESTR
  - Cross-Currency Swaps: basis swaps, fix-float, fix-fix
  - Inflation Products: Zero-Coupon Inflation Swaps (ZCIS), Year-on-Year Swaps
  - Credit Products: Fixed-coupon Bonds, Floating Rate Notes (FRNs)

- **Algorithmic Differentiation Greeks**
  - First-order sensitivities (Delta) computed via reverse-mode AD
  - Second-order sensitivities (Gamma) computed exactly, not via finite differences
  - Tenor-specific risk ladders for granular risk management

- **Robust Curve Bootstrapping**
  - Cashflow-based bootstrapping ensuring exact repricing of input instruments
  - Multiple interpolation schemes: flat forward, linear zero rates, PCHIP, cubic splines
  - Multi-currency support with cross-currency basis spreads
  - JAX-accelerated curve construction with caching

- **Production-Ready Infrastructure**
  - 340+ comprehensive tests covering all products and edge cases
  - Strict numerical precision (1e-10 to 1e-12 tolerances)
  - ISDA 2006 day count conventions
  - Business day calendars (incl. TARGET)

> **AI/ML Usage Notice**
> This repository's source code is made available under the MIT License **for human review and development only.**
> **No part of this codebase may be used to train, fine-tune, evaluate, or benchmark any machine-learning or AI model** (including large language models) without the express prior written permission of the author.

---

## Installation

```bash
git clone https://github.com/ludcode/cavour.git
cd cavour
pip install -r requirements.txt
```

### Activate Virtual Environment (Windows)

```bash
cd /c/Projects/Cavour && source cavourvenv/Scripts/activate
```

---

## Quick Start

### 1. Building an OIS Curve

```python
from cavour.utils.date import Date
from cavour.utils.global_types import SwapTypes, CurveTypes
from cavour.utils.day_count import DayCountTypes
from cavour.utils.frequency import FrequencyTypes
from cavour.utils.calendar import BusDayAdjustTypes
from cavour.market.curves.interpolator import InterpTypes
from cavour.models.models import Model

# Valuation date
value_date = Date(30, 4, 2024)

# GBP SONIA market rates (realistic data from 1D to 50Y)
px_list = [5.1998, 5.2014, 5.2003, 5.2027, 5.2023, 5.19281,
           5.1656, 5.1482, 5.1342, 5.1173, 5.1013, 5.0862,
           5.0701, 5.054, 5.0394, 4.8707, 4.75483, 4.532,
           4.3628, 4.2428, 4.16225, 4.1132, 4.08505, 4.0762,
           4.078, 4.0961, 4.12195, 4.1315, 4.113, 4.07724, 3.984, 3.88]

tenor_list = ["1D", "1W", "2W", "1M", "2M", "3M", "4M", "5M", "6M",
              "7M", "8M", "9M", "10M", "11M", "1Y", "18M", "2Y",
              "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y",
              "12Y", "15Y", "20Y", "25Y", "30Y", "40Y", "50Y"]

# Build curve
model = Model(value_date)
model.build_curve(
    name="GBP_OIS_SONIA",
    px_list=px_list,
    tenor_list=tenor_list,
    spot_days=0,
    swap_type=SwapTypes.PAY,
    fixed_dcc_type=DayCountTypes.ACT_365F,
    fixed_freq_type=FrequencyTypes.ANNUAL,
    float_freq_type=FrequencyTypes.ANNUAL,
    float_dc_type=DayCountTypes.ACT_365F,
    bus_day_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
    interp_type=InterpTypes.LINEAR_ZERO_RATES
)

# Access the curve
curve = model.curves.GBP_OIS_SONIA

# Get discount factor at 5 years
df_5y = curve.df_ad(5.0)
print(f"5Y Discount Factor: {df_5y:.6f}")
```

### 2. Computing VALUE, DELTA, and GAMMA for an OIS

```python
from cavour.trades.rates.ois import OIS
from cavour.utils.global_types import RequestTypes
from cavour.utils.currency import CurrencyTypes

# Create a 10Y GBP SONIA swap
settle_date = value_date.add_tenor("0D")

swap = OIS(
    effective_dt=settle_date,
    term_dt_or_tenor="10Y",
    fixed_leg_type=SwapTypes.PAY,
    fixed_coupon=0.045,  # 4.5% fixed rate
    fixed_freq_type=FrequencyTypes.ANNUAL,
    fixed_dc_type=DayCountTypes.ACT_365F,
    floating_index=CurveTypes.GBP_OIS_SONIA,
    currency=CurrencyTypes.GBP,
    bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
    float_freq_type=FrequencyTypes.ANNUAL,
    float_dc_type=DayCountTypes.ACT_365F,
    notional=10_000_000
)

# Create position and compute all Greeks
position = swap.position(model)
result = position.compute([RequestTypes.VALUE, RequestTypes.DELTA, RequestTypes.GAMMA])

# Extract results
pv = result.value.amount
delta_total = result.risk.value.amount  # Total delta (sum of tenor deltas)
gamma_total = result.gamma.value.amount  # Total gamma

print(f"Present Value: {pv:,.2f}")
print(f"Delta (1bp parallel): {delta_total:,.2f}")
print(f"Gamma (1bp^2): {gamma_total:,.2f}")

# Access tenor-specific delta ladder
delta_ladder = result.risk.ladder.data  # Dict: tenor -> sensitivity
tenors = result.risk.tenors

print("\nDelta Ladder:")
for tenor in tenors[:10]:  # Show first 10 tenors
    if tenor in delta_ladder:
        print(f"  {tenor}: {delta_ladder[tenor]:,.2f}")

# Access gamma matrix (cross-sensitivities)
gamma_matrix = result.gamma.risk_ladder  # NxN matrix
print(f"\nGamma Matrix Shape: {gamma_matrix.shape}")
```

### 3. Scenario Analysis and P&L Attribution

```python
# Create a 100bp parallel shock scenario
model_shocked = model.scenario("GBP_OIS_SONIA", shock=1.0)  # 100bp = 1.0%

# Revalue under shocked scenario
position_shocked = swap.position(model_shocked)
result_shocked = position_shocked.compute([RequestTypes.VALUE])
pv_shocked = result_shocked.value.amount

# Actual P&L
pnl_actual = pv_shocked - pv

# 1st-order approximation: PnL ≈ Delta * dR
pnl_delta = delta_total * 100  # 100bp shock

# 2nd-order approximation: PnL ≈ Delta * dR + 0.5 * Gamma * dR^2
pnl_gamma = delta_total * 100 + 0.5 * gamma_total * (100 ** 2)

print(f"\n100bp Shock P&L Attribution:")
print(f"  Actual P&L:          {pnl_actual:,.2f}")
print(f"  1st-order approx:    {pnl_delta:,.2f} (error: {abs(pnl_actual - pnl_delta):,.2f})")
print(f"  2nd-order approx:    {pnl_gamma:,.2f} (error: {abs(pnl_actual - pnl_gamma):,.2f})")
```

### 4. Pricing a Fixed-Coupon Bond

```python
from cavour.trades.credit.bond import Bond

# Create a 5Y bond with 5% annual coupon
bond = Bond(
    issue_dt=value_date,
    maturity_dt_or_tenor="5Y",
    coupon=0.05,
    freq_type=FrequencyTypes.ANNUAL,
    dc_type=DayCountTypes.ACT_365F,
    currency=CurrencyTypes.GBP,
    face_value=100.0
)

# Value the bond
curve = model.curves.GBP_OIS_SONIA
clean_price = bond.clean_price(value_date, curve)
dirty_price = bond.dirty_price(value_date, curve)
ytm = bond.yield_to_maturity(value_date, clean_price)
duration = bond.duration(value_date, curve)
convexity = bond.convexity(value_date, curve)

print(f"Bond Valuation:")
print(f"  Clean Price: {clean_price:.4f}")
print(f"  Dirty Price: {dirty_price:.4f}")
print(f"  YTM:         {ytm*100:.2f}%")
print(f"  Duration:    {duration:.2f} years")
print(f"  Convexity:   {convexity:.2f}")
```

### 5. Pricing a Floating Rate Note (FRN)

```python
from cavour.trades.credit.frn import FRN

# Create a 5Y FRN with quarterly resets, 50bp margin, and 8% cap
frn = FRN(
    issue_dt=value_date,
    maturity_dt_or_tenor="5Y",
    quoted_margin=0.005,  # 50bp spread over SONIA
    freq_type=FrequencyTypes.QUARTERLY,
    dc_type=DayCountTypes.ACT_365F,
    currency=CurrencyTypes.GBP,
    floating_index=CurveTypes.GBP_OIS_SONIA,
    cap_rate=0.08,  # 8% coupon cap
    face_value=100.0
)

# Value the FRN
clean_price_frn = frn.clean_price(value_date, curve)
dirty_price_frn = frn.dirty_price(value_date, curve)
discount_margin = frn.discount_margin(value_date, curve, curve, clean_price_frn)

print(f"FRN Valuation:")
print(f"  Clean Price:      {clean_price_frn:.4f}")
print(f"  Dirty Price:      {dirty_price_frn:.4f}")
print(f"  Discount Margin:  {discount_margin*10000:.0f} bps")
```

### 6. Zero-Coupon Inflation Swap (ZCIS)

```python
from cavour.trades.rates.zcis import ZeroCouponInflationSwap
from cavour.market.indices.inflation_index import InflationIndex
from cavour.utils.global_types import InflationIndexTypes

# Create RPI index with historical fixings
base_date = Date(1, 3, 2024)
rpi = InflationIndex(
    index_type=InflationIndexTypes.UK_RPI,
    base_date=base_date,
    base_index=293.0,
    currency=CurrencyTypes.GBP,
    lag_months=3
)

# Add historical fixings
rpi.add_fixing(Date(1, 3, 2024), 293.0)
rpi.add_fixing(Date(1, 4, 2024), 293.5)
rpi.add_fixing(Date(1, 5, 2024), 294.0)
rpi.add_fixing(Date(1, 6, 2024), 294.5)

# Create a 10Y ZCIS (pay 3% fixed, receive inflation)
zcis = ZeroCouponInflationSwap(
    effective_dt=value_date,
    term_dt_or_tenor="10Y",
    fixed_leg_type=SwapTypes.PAY,
    fixed_rate=0.03,  # 3% annual inflation expectation
    inflation_index=rpi,
    notional=10_000_000
)

print(f"ZCIS constructed: 10Y maturity, 3% fixed rate")
```

### 7. Multi-Currency Curves

```python
# Build USD SOFR curve
usd_px_list = [5.3500, 5.3200, 5.3100, 5.2900, 5.2700, 5.2500,
               5.2300, 5.2100, 5.1900, 5.1700, 5.1500, 5.1300,
               5.1100, 5.0900, 5.0700, 4.9500, 4.8500, 4.7000,
               4.5800, 4.4800, 4.4100, 4.3600, 4.3200, 4.2900,
               4.2700, 4.2800, 4.3000, 4.3200, 4.3100, 4.2900, 4.2400, 4.1800]

model.build_curve(
    name="USD_OIS_SOFR",
    px_list=usd_px_list,
    tenor_list=tenor_list,
    spot_days=0,
    swap_type=SwapTypes.PAY,
    fixed_dcc_type=DayCountTypes.ACT_360,  # USD convention
    fixed_freq_type=FrequencyTypes.ANNUAL,
    float_freq_type=FrequencyTypes.ANNUAL,
    float_dc_type=DayCountTypes.ACT_360,
    bus_day_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
    interp_type=InterpTypes.LINEAR_ZERO_RATES
)

# Access both curves
gbp_curve = model.curves.GBP_OIS_SONIA
usd_curve = model.curves.USD_OIS_SOFR

print(f"GBP 10Y DF: {gbp_curve.df_ad(10.0):.6f}")
print(f"USD 10Y DF: {usd_curve.df_ad(10.0):.6f}")
```

---

## Mathematical Foundations

### Curve Bootstrapping

Given *N* par swap rates {*r*₁, *r*₂, ..., *r*ₙ} at maturities {*T*₁, *T*₂, ..., *Tₙ}, we bootstrap discount factors {*D*₁, *D*₂, ..., *Dₙ} by solving the par swap condition:

```
1 = rᵢ × ∑(αⱼ × Dⱼ) + Dₘ
```

where *α*ⱼ are year fractions. Rearranging:

```
Dₘ = (1 - rᵢ × PV01ₚᵣₑᵥ) / (1 + rᵢ × αₘ)
```

This creates a recursive dependency ideal for JAX's `lax.scan`.

### Algorithmic Differentiation for Greeks

#### Delta (First-Order)

```
Δᵢ = ∂PV/∂rᵢ = ∑ⱼ (∂PV/∂Dⱼ) × (∂Dⱼ/∂rᵢ)
```

Implemented via:
```python
grad_dfs = grad(lambda d: pv_fn(d))(dfs)      # ∂PV/∂Dⱼ
sensitivities = jnp.dot(grad_dfs, jac)        # Chain rule
```

#### Gamma (Second-Order)

```
Γᵢⱼ = ∂²PV/∂rᵢ∂rⱼ = J^T × H_PV × J + ∑ₖ (∂PV/∂Dₖ) × H_Dₖ
```

Implemented via:
```python
hess_dfs = hessian(lambda d: pv_fn(d))(dfs)
term1 = jac.T @ hess_dfs @ jac
term2 = jnp.sum(grad_dfs[:, None, None] * hess_curve, axis=0)
gamma = term1 + term2
```

### Interpolation Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `FLAT_FWD_RATES` | Piecewise constant forwards | Market standard |
| `LINEAR_ZERO_RATES` | Linear interpolation on zero rates | Simple, stable |
| `PCHIP_ZERO_RATES` | Monotonic Hermite spline | Shape-preserving |
| `NATCUBIC_LOG_DISCOUNT` | Natural cubic on log(DF) | Smooth forwards |

All methods are JAX-compatible for automatic differentiation.

---

## Architecture

### Core Package Structure

```
cavour/
├── utils/              # Date arithmetic, day counts, schedules, calendars
├── market/
│   ├── curves/         # Discount curves, interpolators, bootstrapping
│   ├── indices/        # Inflation indices, rate fixings
│   └── position/       # Position engine, risk calculations
├── trades/
│   ├── rates/          # OIS, inflation swaps, XCCY swaps
│   └── credit/         # Bonds, FRNs
├── models/             # Model class for multi-curve management
└── requests/           # Request types (VALUE, DELTA, GAMMA)
```

### Key Design Patterns

1. **JAX Integration**: All pricing and curve functions are JAX-compatible for AD
2. **Cashflow-Based Bootstrapping**: Ensures exact repricing of input instruments
3. **Caching**: DFs, Jacobians, and Hessians cached per curve
4. **Position Engine**: Unified interface for computing all risk measures

---

## Product Coverage

| Product | Features | Valuation | Greeks |
|---------|----------|-----------|--------|
| **OIS** | SONIA, SOFR, ESTR | VALUE | DELTA, GAMMA |
| **XCCY Swaps** | Basis, fix-float, fix-fix | VALUE | DELTA, GAMMA |
| **Bonds** | Fixed coupon, various frequencies | Clean/Dirty Price, YTM | Duration, Convexity, DV01 |
| **FRNs** | Floating | Clean/Dirty Price | Discount Margin |
| **ZCIS** | Zero-coupon inflation | VALUE | Breakeven Inflation |
| **YoY Swaps** | Year-on-year inflation | VALUE | Risk measures |

---

## Testing

The library includes **340+ comprehensive tests** covering:

- **Core Infrastructure (161 tests)**: Day counts, schedules, interpolators, date arithmetic
- **Financial Validation (15 tests)**: Curve properties, par swap repricing
- **Product Coverage (47 tests)**: ZCIS, bonds, FRNs construction
- **Risk Calculations (21 tests)**: Bond/FRN duration, convexity, sensitivities
- **Robustness (30 tests)**: Error handling, edge cases, numerical stability
- **Existing Tests (66 tests)**: OIS, XCCY, refit validation

Run tests:
```bash
pytest tests/ -v
pytest tests/test_ois_request_types.py -v  # OIS VALUE/DELTA/GAMMA tests
pytest tests/test_credit_products_risk.py -v  # Bond/FRN tests
```

All tests use strict tolerances (1e-10 to 1e-12) and realistic market data.

---

## Advanced Features

### Cross-Currency Curve Bootstrapping

See `documentation/xccy_bootstrap.md` for detailed mathematical formulation of the cashflow-based cross-currency curve bootstrapping algorithm.

### Bloomberg Integration

Optional Bloomberg data integration via `xbbg` and `blpapi`. Market data tests require Bloomberg connection:

```bash
pytest -m market_data
```

### JAX Acceleration

All curve construction and risk calculations are JIT-compiled:

```python
from jax import jit

# Curve building is automatically JIT-compiled
# Pricing functions are JAX-compatible
df = curve.df_ad(5.0)  # Uses JAX arrays internally
```

---

## Performance Characteristics

- **Curve Bootstrap**: ~50ms for 32-tenor OIS curve (with JIT compilation)
- **Delta Calculation**: ~10ms for 10Y swap (includes Jacobian computation)
- **Gamma Calculation**: ~30ms for 10Y swap (includes Hessian computation)
- **Par Swap Repricing**: <1e-5 absolute error (0.001 bps)

All timings on standard laptop CPU. GPU acceleration available via JAX.


---

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass: `pytest tests/ -v`
2. New features include comprehensive tests
3. Code follows existing patterns (JAX-compatible, type hints)
4. Financial precision maintained (1e-10+ tolerances)

---

## License

MIT License - See LICENSE file for details.

**AI/ML Training Restriction**: No part of this codebase may be used for training, fine-tuning, evaluating, or benchmarking any AI/ML model without express written permission.

---

## Contact

For questions or issues, please open an issue on GitHub.

---

## Acknowledgments

Built with:
- **JAX**: Google's automatic differentiation library
- **NumPy/SciPy**: Numerical computing
- **pytest**: Testing framework

Mathematical foundations based on ISDA 2006 definitions and industry-standard curve construction methodologies.
