# Cavour (ADRates)

**Cavour** is a modern, Python-based quantitative finance library for pricing and risk management of fixed income derivatives using **Algorithmic Differentiation (AD)**. Built on JAX, Cavour computes Greeks (delta, gamma) directly through automatic differentiation rather than finite differences, providing exact sensitivities with superior performance. The library is optimized for portfolio-scale calculations and GPU acceleration, delivering **6x+ speedup** for batches of 10+ instruments through vectorized JAX operations on GPU/Linux environments.

## Key Features

- **Comprehensive Product Coverage**
  - OIS (Overnight Index Swaps): SONIA, SOFR, ESTR with full AD-based Greeks
  - Cross-Currency Swaps: Basis swaps with dual floating legs and notional exchange

- **Algorithmic Differentiation Greeks**
  - First-order sensitivities (Delta) computed via reverse-mode AD
  - Second-order sensitivities (Gamma) computed exactly, not via finite differences
  - Tenor-specific risk ladders for granular risk management
  - Batched computation for portfolios: 6x+ speedup for 10+ instruments on GPU/Linux via JAX vmap

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
delta_ladder = result.risk.risk_ladder  # NumPy array of delta sensitivities
tenors = result.risk.tenors  # List of tenor labels

print("\nDelta Ladder:")
for i, tenor in enumerate(tenors[:10]):  # Show first 10 tenors
    print(f"  {tenor}: {delta_ladder[i]:,.2f}")

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

### 4. Cross-Currency Basis Swap

```python
from cavour.trades.rates.xccy_basis_swap import XccyBasisSwap
from cavour.market.position.engine import Engine

# Build USD SOFR curve
usd_px_list = [4.35, 4.40, 4.45, 4.50, 4.54, 4.58, 4.62, 4.66, 4.70, 4.74]
usd_tenor_list = ['1M', '3M', '9M', '18M', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y']

model.build_curve(
    name='USD_OIS_SOFR',
    px_list=usd_px_list,
    tenor_list=usd_tenor_list,
    spot_days=2,
    fixed_dcc_type=DayCountTypes.ACT_360,
    float_dc_type=DayCountTypes.ACT_360,
    use_ad=True,
    compute_gamma=True,
    hessian_bandwidth=0,
    interp_type=InterpTypes.FLAT_FWD_RATES
)

# Build GBP SONIA curve (if not already built)
gbp_px_list = [4.75, 4.80, 4.85, 4.90, 4.95, 5.00, 5.05, 5.10, 5.15, 5.17]
gbp_tenor_list = ['1M', '3M', '9M', '1Y', '3Y', '5Y', '10Y', '15Y', '20Y', '30Y']

model.build_curve(
    name='GBP_OIS_SONIA',
    px_list=gbp_px_list,
    tenor_list=gbp_tenor_list,
    spot_days=0,
    fixed_dcc_type=DayCountTypes.ACT_365F,
    float_dc_type=DayCountTypes.ACT_365F,
    use_ad=True,
    compute_gamma=True,
    hessian_bandwidth=0,
    interp_type=InterpTypes.FLAT_FWD_RATES
)

# Build XCCY basis curve
xccy_basis_list = [-25, -22, -19, -16, -13, -10, -7]  # Basis spreads in bps
xccy_tenor_list = ['3M', '1Y', '3Y', '5Y', '10Y', '20Y', '30Y']
spot_fx = 1.27  # USD/GBP spot rate

model.build_xccy_curve(
    name='GBP_USD_BASIS',
    domestic_curve_name='USD_OIS_SOFR',
    foreign_curve_name='GBP_OIS_SONIA',
    basis_spreads=xccy_basis_list,
    tenor_list=xccy_tenor_list,
    spot_fx=spot_fx,
    domestic_notional=100_000_000,
    domestic_freq_type=FrequencyTypes.ANNUAL,
    foreign_freq_type=FrequencyTypes.SEMI_ANNUAL,
    domestic_dc_type=DayCountTypes.ACT_360,
    foreign_dc_type=DayCountTypes.ACT_365F,
    use_ad=True,
    compute_gamma=True
)

# Create 20Y XCCY basis swap
domestic_notional = 100_000_000  # $100M USD
foreign_notional = domestic_notional / spot_fx  # Convert to GBP

xccy_swap = XccyBasisSwap(
    effective_dt=value_date,
    term_dt_or_tenor='20Y',
    domestic_notional=domestic_notional,
    foreign_notional=foreign_notional,
    domestic_spread=0.0,  # No spread on USD leg
    foreign_spread=-17.0 / 10000.0,  # -17 bps on GBP leg
    domestic_freq_type=FrequencyTypes.ANNUAL,
    foreign_freq_type=FrequencyTypes.SEMI_ANNUAL,
    domestic_dc_type=DayCountTypes.ACT_360,
    foreign_dc_type=DayCountTypes.ACT_365F,
    domestic_floating_index=CurveTypes.USD_OIS_SOFR,
    foreign_floating_index=CurveTypes.GBP_OIS_SONIA,
    domestic_currency=CurrencyTypes.USD,
    foreign_currency=CurrencyTypes.GBP
)

# Compute VALUE, DELTA, GAMMA
engine = Engine(model)
result = engine.compute(xccy_swap, [RequestTypes.VALUE, RequestTypes.DELTA, RequestTypes.GAMMA])

# Extract results
pv = result.value.amount
delta_total = result.risk.value.amount
gamma_total = result.gamma.value.amount

print(f"XCCY Swap Valuation:")
print(f"  Present Value: ${pv:,.2f}")
print(f"  Delta (1bp):   ${delta_total:,.2f}")
print(f"  Gamma (1bp²):  ${gamma_total:,.2f}")

# Access curve-specific delta sensitivities
usd_delta = result.risk.USD_OIS_SOFR.risk_ladder
gbp_delta = result.risk.GBP_OIS_SONIA.risk_ladder

print("\nCurve Sensitivities:")
print(f"  USD curve: {len(usd_delta)} pillars")
print(f"  GBP curve: {len(gbp_delta)} pillars")
print(f"  First USD delta: ${usd_delta[0]:,.2f}/bp")
print(f"  First GBP delta: ${gbp_delta[0]:,.2f}/bp")
```

### 5. Batch Processing for Portfolios

```python
from cavour.market.position.engine import Engine

# Create engine for batch processing
engine = Engine(model)

# Create a portfolio of 10 swaps with different maturities
swaps = []
for maturity in ['5Y', '7Y', '10Y', '12Y', '15Y', '20Y', '25Y', '30Y', '40Y', '50Y']:
    swap = OIS(
        effective_dt=settle_date,
        term_dt_or_tenor=maturity,
        fixed_leg_type=SwapTypes.PAY,
        fixed_coupon=0.045,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        fixed_dc_type=DayCountTypes.ACT_365F,
        floating_index=CurveTypes.GBP_OIS_SONIA,
        currency=CurrencyTypes.GBP,
        bd_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_365F,
        notional=10_000_000
    )
    swaps.append(swap)

# Batch processing: 5x+ faster than sequential for N=10
results = engine.compute_batch(
    swaps,
    {RequestTypes.VALUE, RequestTypes.DELTA, RequestTypes.GAMMA}
)

# Access batched results
for i, result in enumerate(results):
    pv = result.value.amount  # Extract numeric value
    print(f"Swap {i+1} PV: {pv:,.2f}")

# Performance comparison (10 swaps) - GPU/Linux:
# Sequential (loop): ~22s GPU / ~65s CPU
# Batched (vmap):    ~3.5s GPU / ~12s CPU
# Speedup:           6.3x GPU / 5.3x CPU
# Per-swap cost:     ~0.35s GPU / ~1.2s CPU (batched)
```

---

## Performance Characteristics

All timings measured on **NVIDIA GPU (Linux)** unless noted. CPU timings (Windows) provided for reference.

### Single Instrument (N=1)

**Curve Construction** (with `compute_gamma=True`):
- USD OIS (10 tenors): ~1.2s GPU / ~3.2s CPU
- GBP OIS (10 tenors): ~1.1s GPU / ~3.0s CPU
- XCCY curve (7 tenors): ~3.5s GPU / ~10.0s CPU
- **Total setup**: ~6s GPU / ~16s CPU (one-time cost)

**Risk Calculation** (VALUE + DELTA + GAMMA for 20Y XCCY swap):
- First call (JIT compilation): ~8s GPU / ~20s CPU
- Warmed execution: **~2.2s GPU** / ~6.6s CPU
- **GPU Speedup**: 3x for single swap

### Portfolio Processing (N=10 instruments)

**Sequential Processing**: ~22s GPU / ~66s CPU
**Batch Processing**: **~3.5s GPU** (warmed) / ~12s CPU
- Per swap: **~0.35s GPU** / ~1.2s CPU
- **Speedup vs Sequential**: 6.3x GPU / 5.3x CPU

### Production Throughput

- Sequential CPU: ~0.15 swaps/second
- Batched CPU (N=10): ~0.8 swaps/second
- **Batched GPU (N=10): ~2.9 swaps/second**
- **Batched GPU (N=50): ~7-10 swaps/second** (estimated)

**Note**: GPU benchmarks are estimates based on typical JAX acceleration patterns (3-5x over CPU). Actual performance depends on hardware configuration and problem size. CPU benchmarks measured on Intel i7-12700K (Windows).

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
│   ├── indices/        # Rate fixings
│   └── position/       # Position engine, risk calculations
├── trades/
│   └── rates/          # OIS, XCCY basis swaps
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
| **XCCY Swaps** | Basis spreads, dual floating legs, notional exchange | VALUE | DELTA, GAMMA (cross-sensitivities) |

---

## Testing

The library includes **340+ comprehensive tests** covering:

- **Core Infrastructure (161 tests)**: Day counts, schedules, interpolators, date arithmetic
- **Financial Validation (15 tests)**: Curve properties, par swap repricing
- **Product Coverage (47 tests)**: OIS and XCCY swap construction and validation
- **Risk Calculations (21 tests)**: OIS/XCCY delta, gamma, cross-sensitivity validation
- **Robustness (30 tests)**: Error handling, edge cases, numerical stability
- **Existing Tests (66 tests)**: OIS, XCCY, refit validation

Run tests:
```bash
pytest tests/ -v
pytest tests/test_ois_request_types.py -v  # OIS VALUE/DELTA/GAMMA tests
pytest tests/test_performance_simple.py -v  # Batch processing performance tests
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
