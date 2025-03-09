# ADRates

**ADRates** is a limited and modernized fork of an early version of [FinancePy](https://github.com/domokane/FinancePy), created by one of its students. The project aims to enhance the library with improved **Overnight Index Swap (OIS) curve building** and **Automatic Differentiation (AD)** for computing **delta** and **gamma** of interest rate swaps.

## 🚀 Features

- **Enhanced OIS Curve Construction**: More robust and flexible methodologies for discount curve bootstrapping.
- **Automatic Differentiation for Greeks**: Uses AD to compute **delta** and **gamma**, making risk calculations more efficient.
- **Modernization & Bug Fixes**: Improves outdated functionality while preserving core FinancePy concepts. Possible now to build Roundtrip proof OIS discount and forward curves

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/ludocode/ADRates.git
cd ADRates
pip install -r requirements.txt
