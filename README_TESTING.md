# Cavour Library - Testing Guide

This document provides comprehensive information about testing the Cavour financial library.

## Overview

The test suite provides comprehensive coverage of the Cavour library's functionality, from basic utility functions to complete portfolio valuation workflows. Tests are organized by module and include both unit tests and integration tests.

## Test Structure

```
tests/
├── conftest.py                          # Pytest configuration and fixtures
├── pytest.ini                          # Pytest settings
├── test_dummy.py                        # Basic smoke test
├── test_global_types.py                 # Enum validation tests
├── test_intro_regression.py             # Original regression test
├── test_utils_date.py                   # Date utility tests
├── test_utils_day_count.py              # Day count convention tests  
├── test_utils_schedule.py               # Schedule generation tests
├── test_market_curves_discount.py       # Discount curve tests
├── test_trades_rates_ois.py             # OIS swap tests
├── test_models_models.py                # Model building tests
└── test_integration_full_workflow.py    # End-to-end integration tests
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_utils_date.py

# Run specific test function
pytest tests/test_utils_date.py::TestDate::test_date_creation

# Run tests matching pattern
pytest -k "date"
```

### Test Categories

Tests are organized with markers for selective execution:

```bash
# Run only unit tests (default)
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run numerical precision tests
pytest -m numerical

# Run market data tests (requires Bloomberg connection)
pytest -m market_data
```

### Parallel Execution

For faster execution with multiple CPU cores:

```bash
# Install pytest-xdist first
pip install pytest-xdist

# Run tests in parallel
pytest -n auto
```

## Test Coverage

### Core Utilities (`cavour/utils/`)

- **Date Operations**: Date arithmetic, tenor parsing, business day adjustments
- **Day Count Conventions**: ACT/360, ACT/365F, 30/360 calculations
- **Schedule Generation**: ISDA-compliant payment schedules
- **Global Types**: Enum validation and type safety

### Market Data & Curves (`cavour/market/`)

- **Discount Curves**: Construction, interpolation, extrapolation
- **Curve Operations**: Zero rates, forward rates, curve shifting
- **Interpolation Methods**: Linear, flat forward, various schemes

### Financial Instruments (`cavour/trades/`)

- **OIS Swaps**: Construction, valuation, par rate calculation
- **Risk Measures**: PV01 calculation and validation
- **Leg Generation**: Fixed and floating leg cashflow projection

### Model Building (`cavour/models/`)

- **Curve Bootstrapping**: Multi-currency curve construction
- **Scenario Analysis**: Rate bumping and shock scenarios
- **Model Consistency**: Cross-currency validation

### Integration Workflows

- **Complete Valuation Pipeline**: Curve building to portfolio valuation
- **Multi-Currency Operations**: USD/GBP/EUR curve handling
- **Performance Testing**: Large portfolio valuation
- **Error Handling**: Validation and edge case handling

## Test Data and Fixtures

### Shared Fixtures (conftest.py)

- `standard_value_date`: Common valuation date (2024-01-01)
- `standard_discount_curve`: Pre-built curve for testing
- `usd_market_rates`/`gbp_market_rates`: Realistic market rate data
- `standard_ois_params`: Common OIS swap parameters

### Synthetic Data Generation

```python
# Generate test curve data
@pytest.fixture
def curve_test_data_generator():
    def _generate_curve_data(num_points=10, rate_level=0.05, curve_shape="normal"):
        # Returns synthetic rate data for testing
        pass
    return _generate_curve_data
```

## Testing Best Practices

### Numerical Precision

Tests use appropriate tolerances for floating-point comparisons:

```python
# Standard tolerance
assert abs(actual - expected) < 1e-6

# High precision tests
assert abs(par_value) < 1e-8  # Par swaps should be near zero

# Use pytest.approx for convenience
assert actual == pytest.approx(expected, rel=1e-4)
```

### Market Reality Checks

Tests validate that results are financially reasonable:

```python
# Par rates should be in reasonable range
assert 0.01 < par_rate < 0.10  # 1% to 10%

# PV01 should be positive and reasonable
assert 100 < pv01 < 50000  # Typical range for institutional swaps

# Discount factors should be monotonic
assert all(df_t > df_t_plus_1 for df_t, df_t_plus_1 in zip(dfs[:-1], dfs[1:]))
```

### Error Handling

Tests verify proper error handling:

```python
# Invalid construction parameters
with pytest.raises((ValueError, Exception)):
    DiscountCurve(value_dt, [], [])  # Empty data

# Invalid date order
with pytest.raises((ValueError, Exception)):
    OIS(effective_dt=future_date, term_dt_or_tenor=past_date)
```

## Performance Testing

### Benchmarking

```bash
# Run with duration reporting
pytest --durations=10

# Profile slow tests
pytest -m slow --durations=0
```

### Performance Fixtures

```python
@pytest.mark.slow
def test_large_portfolio_performance():
    # Create 1000+ instruments
    # Measure valuation time
    # Assert reasonable performance bounds
```

## Dependencies and Requirements

### Core Testing Dependencies

- `pytest >= 6.0`: Test framework
- `numpy`: Numerical computations
- `pandas`: Data handling (if used in library)

### Optional Testing Extensions

- `pytest-cov`: Coverage reporting
- `pytest-xdist`: Parallel test execution
- `pytest-timeout`: Timeout protection
- `pytest-benchmark`: Performance benchmarking

### Market Data Dependencies

- `xbbg`: Bloomberg data (for market data tests)
- `blpapi`: Bloomberg API (optional)

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', 3.11]
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run tests
      run: pytest -v --cov=cavour
```

## Common Test Patterns

### Curve Building Test

```python
def test_curve_building():
    model = Model(value_date)
    model.build_curve(
        name="TEST_CURVE",
        px_list=[5.0, 4.8, 4.6],
        tenor_list=["1Y", "5Y", "10Y"],
        # ... other parameters
    )
    
    curve = model.curves.TEST_CURVE
    assert curve.df(value_date) == 1.0  # Value date DF
    assert curve.df(value_date.add_tenor("1Y")) < 1.0  # Forward DF
```

### OIS Valuation Test

```python
def test_ois_at_par():
    ois = OIS(effective_dt, "5Y", SwapTypes.PAY, 0.05, ...)
    
    # Calculate par rate
    par_rate = ois.swap_rate(value_dt, curve)
    
    # Create at-par swap
    par_ois = OIS(effective_dt, "5Y", SwapTypes.PAY, par_rate, ...)
    
    # Should value to zero
    assert abs(par_ois.value(value_dt, curve)) < 1e-6
```

### Scenario Analysis Test

```python
def test_scenario_analysis():
    base_model = build_model()
    
    # Create rate shock
    bumps = {"1Y": 0.01, "5Y": 0.01}  # 100bp parallel shift
    shock_model = base_model.scenario("USD_OIS", bumps)
    
    # Compare valuations
    base_value = swap.value(value_dt, base_model.curves.USD_OIS)
    shock_value = swap.value(value_dt, shock_model.curves.USD_OIS)
    
    # Should have measurable impact
    assert abs(shock_value - base_value) > 1e-4
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `PYTHONPATH` includes the cavour package
2. **Numerical Precision**: Use appropriate tolerances for floating-point comparisons
3. **Date Handling**: Be aware of business day adjustments and calendar differences
4. **Market Data**: Mock external data dependencies for reproducible tests

### Debug Mode

```bash
# Run with detailed output
pytest -v -s --tb=long

# Drop into debugger on failure
pytest --pdb

# Run single test with maximum verbosity
pytest -vvv -s tests/test_specific.py::test_function
```

## Contributing Tests

When adding new functionality to Cavour:

1. **Write Tests First**: Follow TDD principles
2. **Cover Edge Cases**: Test boundary conditions and error cases
3. **Validate Financial Logic**: Ensure results are economically sensible
4. **Add Performance Tests**: For computationally intensive features
5. **Update Documentation**: Keep this README current

### Test Naming Conventions

- `test_basic_functionality`: Core feature tests
- `test_edge_cases`: Boundary condition tests  
- `test_error_handling`: Exception and validation tests
- `test_performance_*`: Performance and benchmark tests
- `test_integration_*`: End-to-end workflow tests

## Support

For questions about testing:

1. Check existing test examples in the codebase
2. Review test output and error messages carefully
3. Consider financial/mathematical correctness of test assertions
4. Ensure proper handling of floating-point precision

The test suite is designed to be comprehensive, maintainable, and reflective of real-world usage patterns in quantitative finance.