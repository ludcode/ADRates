# ADRates

**ADRates** is a modern, Python‐based library for building Overnight Index Swap (OIS) discount curves and computing sensitivities (delta and gamma) of interest rate swaps via Automatic Differentiation (AD).

## 🚀 Features

- **Robust OIS Curve Construction**  
  Flexible bootstrapping routines for building arbitrage‐free discount and forward curves from market quotes.

- **Automatic Differentiation of Greeks**  
  Compute first‐order (delta) and second‐order (gamma) sensitivities of swap PVs directly via AD, without finite‐difference approximations.

- **Modern API & Performance**  
  Designed for JAX integration, enabling vectorized, just-in-time compilation and GPU acceleration.

- **Round-Trip Curve Consistency**  
  Build discount and forward curves that reproduce input swap rates exactly, ensuring theoretical consistency.

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/ludocode/ADRates.git
cd ADRates
pip install -r requirements.txt

