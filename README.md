# ADRates

**ADRates** is a modern, Python-based quantitative finance library for building Overnight Index Swap (OIS) discount curves and computing sensitivities (delta and gamma) of interest rate swaps via Algorithmic Differentiation (AD). This is a fundamental difference with the typical Quant approach of bumping curves.

## 🚀 Features

- **Robust OIS Curve Construction**
  Flexible bootstrapping routines for building arbitrage-free discount and forward curves from market quotes using cashflow-based approaches.

- **Algorithmic Differentiation of Greeks**
  Compute first-order (delta) and second-order (gamma) sensitivities of swap PVs directly via AD, without finite-difference approximations.

- **Modern API & Performance**
  Designed for JAX integration, enabling vectorized, just-in-time compilation and GPU acceleration.

- **Multi-Currency Support**
  Build curves for GBP (SONIA), USD (SOFR), EUR (ESTR) with cross-currency basis spread bootstrapping (in the experimental section).

- **Round-Trip Curve Consistency**
  Build discount and forward curves that reproduce input swap rates exactly, ensuring theoretical consistency.

> **AI/ML Usage Notice**
> This repository's source code is made available under the MIT License **for human review and development only.**
> **No part of this codebase may be used to train, fine-tune, evaluate, or benchmark any machine-learning or AI model** (including large language models) without the express prior written permission of the author.

---

## 📐 Mathematical Foundations

### Curve Bootstrapping

Given a set of *N* par swap rates {*r*₁, *r*₂, ..., *r*ₙ} at maturities {*T*₁, *T*₂, ..., *Tₙ}, we bootstrap discount factors {*D*₁, *D*₂, ..., *Dₙ} by solving the par swap condition iteratively.

For a swap with maturity *Tᵢ* and fixed coupon periods with year fractions {*α*₁, *α*₂, ..., *α*ₘ}, the par condition is:

```
1 = rᵢ × ∑(αⱼ × Dⱼ) + Dₘ
```

Rearranging for the final discount factor:

```
Dₘ = (1 - rᵢ × PV01ₚᵣₑᵥ) / (1 + rᵢ × αₘ)
```

where `PV01ₚᵣₑᵥ = ∑(αⱼ × Dⱼ)` for *j* = 1, ..., *m*-1.

This creates a **recursive dependency** where each discount factor depends on previously computed DFs, making it ideal for sequential bootstrapping.

### Algorithmic Differentiation for Risk Sensitivities

#### Delta (First-Order Sensitivities)

Delta measures how the present value changes with respect to input swap rates:

```
Δᵢ = ∂PV/∂rᵢ
```

Using the **chain rule**, we decompose this as:

```
∂PV/∂rᵢ = ∑ⱼ (∂PV/∂Dⱼ) × (∂Dⱼ/∂rᵢ)
```

In our implementation (`engine.py:412-413`):

```python
grad_dfs = grad(lambda d: pv_fn(d))(dfs)      # ∂PV/∂Dⱼ
sensitivities = jnp.dot(grad_dfs, jac)        # ∑ⱼ (∂PV/∂Dⱼ) × (∂Dⱼ/∂rᵢ)
```

Where:
- `grad_dfs`: gradient of PV w.r.t. discount factors (shape: *M*)
- `jac`: Jacobian matrix ∂**D**/∂**r** (shape: *M* × *N*)
- Result: sensitivities (shape: *N*)

The Jacobian is computed via JAX's `jacrev` (line 262):

```python
jac = jacrev(build_dfs)(rates)  # Reverse-mode AD for efficiency
```

#### Gamma (Second-Order Sensitivities)

Gamma measures the curvature of PV with respect to rate changes:

```
Γᵢⱼ = ∂²PV/∂rᵢ∂rⱼ
```

By differentiating the delta expression, we get:

```
∂²PV/∂rᵢ∂rⱼ = ∑ₖ∑ₗ (∂²PV/∂Dₖ∂Dₗ) × (∂Dₖ/∂rᵢ) × (∂Dₗ/∂rⱼ)
              + ∑ₖ (∂PV/∂Dₖ) × (∂²Dₖ/∂rᵢ∂rⱼ)
```

In matrix form:

```
Γ = J^T × H_PV × J + ∑ₖ (∂PV/∂Dₖ) × H_Dₖ
```

Where:
- **J**: Jacobian ∂**D**/∂**r**
- **H_PV**: Hessian of PV w.r.t. DFs (shape: *M* × *M*)
- **H_Dₖ**: Hessian of *k*-th DF w.r.t. rates (shape: *N* × *N*)

Implementation (`engine.py:422-426`):

```python
hess_dfs = hessian(lambda d: pv_fn(d))(dfs)           # H_PV
term1 = jac.T @ hess_dfs @ jac                        # J^T × H_PV × J
term2 = jnp.sum(grad_dfs[:, None, None] * hess_curve, axis=0)  # Σₖ (∂PV/∂Dₖ) × H_Dₖ
gammas = term1 + term2
```

### Interpolation Methods

The library supports multiple interpolation schemes for constructing continuous discount curves from discrete points:

| Method | Formula | Use Case |
|--------|---------|----------|
| `FLAT_FWD_RATES` | *D*(*t*) = exp(-interp(*t*, **T**, -*r*×**T**)) | Market standard |
| `LINEAR_ZERO_RATES` | *r*(*t*) = interp(*t*, **T**, **r**); *D*(*t*) = exp(-*r*(*t*)×*t*) | Zero rate interpolation |
| `LINEAR_FWD_RATES` | *D*(*t*) = interp(*t*, **T**, **D**) | Direct DF interpolation |
| `PCHIP_ZERO_RATES` | Monotonic Hermite on zero rates | Smooth, shape-preserving |
| `NATCUBIC_LOG_DISCOUNT` | Natural cubic spline on log(*D*) | Smooth forwards |

All interpolation methods in `InterpolatorAd` are JAX-compatible for automatic differentiation.

### Key Implementation Details

1. **Deduplication of Intermediate Points** (`engine.py:205-212`)
   When multiple swaps share intermediate cashflow dates, we deduplicate using rounded maturity keys (1 decimal place) to handle floating-point precision.

2. **Sequential Bootstrapping** (`engine.py:232-250`)
   We use `lax.scan` for efficient sequential computation, where each discount factor depends on previously computed PV01s.

3. **Caching for Performance** (`engine.py:257-279`)
   Discount factors, Jacobians, and Hessians are cached per curve to avoid recomputation during risk calculations.

---

## 📦 Installation

Clone the repository and install dependencies:

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

## 🎯 Quick Start

### Building an OIS Curve

```python
from cavour.utils.date import Date
from cavour.trades.rates.ois import OIS
from cavour.trades.rates.ois_curve import OISCurve
from cavour.utils.global_types import InterpTypes

# Define valuation date
value_dt = Date(30, 4, 2024)

# Hardcoded data, but if you have BBG it connects automatically
tenor_list = ["1D","1W","2W","1M","2M","3M","4M","5M","6M",
              "7M","8M","9M","10M","11M","1Y","18M","2Y",
              "3Y","4Y","5Y","6Y","7Y","8Y","9Y","10Y",
              "12Y","15Y","20Y","25Y","30Y","40Y","50Y"]

px_list = [5.1998, 5.2014, 5.2003, 5.2027, 5.2023, 5.19281, 
           5.1656, 5.1482, 5.1342, 5.1173, 5.1013, 5.0862, 
           5.0701, 5.054, 5.0394, 4.8707, 4.75483, 4.532, 
           4.3628, 4.2428, 4.16225, 4.1132, 4.08505, 4.0762, 
           4.078, 4.0961, 4.12195, 4.1315, 4.113, 4.07724, 3.984, 3.88] 

spot_days = 0
settle_dt = value_dt.add_weekdays(spot_days)

swaps = []
swap_type = SwapTypes.PAY
fixed_dcc_type = DayCountTypes.ACT_365F
fixed_freq_type = FrequencyTypes.ANNUAL
bus_day_type = BusDayAdjustTypes.MODIFIED_FOLLOWING
float_freq_type = FrequencyTypes.ANNUAL
float_dc_type = DayCountTypes.ACT_365F

# Bootstrap curve
model = Model(value_dt)

model.build_curve(
        name = "GBP_OIS_SONIA",
        px_list = px_list,
        tenor_list =  tenor_list,
        spot_days = 0,
        swap_type = SwapTypes.PAY,
        fixed_dcc_type = DayCountTypes.ACT_365F,
        fixed_freq_type = FrequencyTypes.ANNUAL,
        float_freq_type = FrequencyTypes.ANNUAL,
        float_dc_type = DayCountTypes.ACT_365F,
        bus_day_type = BusDayAdjustTypes.MODIFIED_FOLLOWING,
        interp_type = InterpTypes.LINEAR_ZERO_RATES,
)
```

### Computing Sensitivities

```python
from cavour.models.models import Model
from cavour.market.position.engine import Engine
from cavour.utils.global_types import RequestTypes

# Create a swap to value
px = 0.04078
tenor = "10Y"

swap_type = SwapTypes.PAY
fixed_dcc_type = DayCountTypes.ACT_365F
fixed_freq_type = FrequencyTypes.ANNUAL
bus_day_type = BusDayAdjustTypes.MODIFIED_FOLLOWING
float_freq_type = FrequencyTypes.ANNUAL
float_dc_type = DayCountTypes.ACT_365F

swap = OIS(effective_dt= settle_dt,
            term_dt_or_tenor= tenor,
            fixed_leg_type= swap_type,
            fixed_coupon= px,
            fixed_freq_type= fixed_freq_type,
            fixed_dc_type=fixed_dcc_type,
            bd_type= bus_day_type,
            float_freq_type=float_freq_type,
            float_dc_type=float_dc_type
            )

# Select the greeks

requests = [RequestTypes.VALUE,
            RequestTypes.DELTA,
            RequestTypes.GAMMA]

pos = swap.position(model)

# Compute PV, delta, and gamma using Algorithmic Differentiation
results = pos.compute(requests)

print(f"PV: {results.value}")
print(f"Delta: {results.risk}")
print(f"Gamma: {results.gamma}")
```

---

## 🏗️ Architecture

### Core Package Structure

- **`cavour/utils/`** - Date arithmetic, day count conventions, schedules
- **`cavour/market/`** - Curve construction, interpolators, portfolio management
- **`cavour/trades/`** - OIS instruments, swap legs, cashflow projection
- **`cavour/models/`** - Multi-curve model building and scenario analysis
- **`cavour/requests/`** - Risk measure calculations and result aggregation

### Key Design Patterns

1. **Algorithmic Differentiation Integration**
   JAX-compatible interpolators and pricing functions for computing Greeks directly.

2. **Cashflow-Based Bootstrapping**
   Ensures exact reproduction of input swap rates and theoretical consistency.

3. **Curve Caching**
   Discount factors, Jacobians, and Hessians cached for performance.

---

## 📚 Documentation

- **README_TESTING.md** - Comprehensive testing documentation
- **cavour/documentation/** - Technical notes on cross-currency curves and advanced features

---

## 🤝 Contributing

Contributions are welcome! Please ensure all tests pass before submitting a pull request.

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 📧 Contact

For questions or issues, please open an issue on GitHub.
