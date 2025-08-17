"""
Pytest configuration file for Cavour library tests
Provides common fixtures and test configuration
"""
import os
import sys
import pytest

# Add the cavour package to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import key modules for fixtures
from cavour.utils.date import Date
from cavour.utils.global_types import SwapTypes
from cavour.utils.day_count import DayCountTypes
from cavour.utils.frequency import FrequencyTypes
from cavour.utils.calendar import BusDayAdjustTypes
from cavour.market.curves.interpolator import InterpTypes
from cavour.market.curves.discount_curve import DiscountCurve


@pytest.fixture(scope="session")
def standard_value_date():
    """Standard valuation date for tests"""
    return Date(1, 1, 2024)


@pytest.fixture(scope="session")
def sample_curve_dates(standard_value_date):
    """Standard set of curve pillar dates"""
    return [
        standard_value_date,
        standard_value_date.add_tenor("1W"),
        standard_value_date.add_tenor("1M"),
        standard_value_date.add_tenor("3M"),
        standard_value_date.add_tenor("6M"),
        standard_value_date.add_tenor("1Y"),
        standard_value_date.add_tenor("2Y"),
        standard_value_date.add_tenor("3Y"),
        standard_value_date.add_tenor("5Y"),
        standard_value_date.add_tenor("7Y"),
        standard_value_date.add_tenor("10Y"),
        standard_value_date.add_tenor("15Y"),
        standard_value_date.add_tenor("20Y"),
        standard_value_date.add_tenor("30Y")
    ]


@pytest.fixture(scope="session")
def sample_discount_factors():
    """Standard set of discount factors for testing"""
    return [
        1.0000,    # Value date
        0.9996,    # 1W
        0.9958,    # 1M
        0.9871,    # 3M
        0.9742,    # 6M
        0.9487,    # 1Y
        0.8963,    # 2Y
        0.8421,    # 3Y
        0.7408,    # 5Y
        0.6496,    # 7Y
        0.5488,    # 10Y
        0.4165,    # 15Y
        0.3234,    # 20Y
        0.2145     # 30Y
    ]


@pytest.fixture
def standard_discount_curve(standard_value_date, sample_curve_dates, sample_discount_factors):
    """Standard discount curve for testing"""
    return DiscountCurve(standard_value_date, sample_curve_dates, sample_discount_factors)


@pytest.fixture(scope="session")
def usd_market_rates():
    """Sample USD OIS market rates (in percent)"""
    return [5.35, 5.32, 5.28, 5.25, 5.20, 5.15, 5.05, 4.95, 4.80, 4.70, 4.60, 4.45, 4.35, 4.25]


@pytest.fixture(scope="session")
def gbp_market_rates():
    """Sample GBP OIS market rates (in percent)"""
    return [5.20, 5.18, 5.15, 5.12, 5.08, 5.02, 4.92, 4.82, 4.68, 4.58, 4.48, 4.33, 4.23, 4.13]


@pytest.fixture(scope="session")
def standard_tenors():
    """Standard tenor list for curve building"""
    return ["1W", "1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"]


@pytest.fixture
def usd_curve_params():
    """Standard USD curve building parameters"""
    return {
        'swap_type': SwapTypes.PAY,
        'fixed_dcc_type': DayCountTypes.ACT_360,
        'fixed_freq_type': FrequencyTypes.ANNUAL,
        'float_freq_type': FrequencyTypes.ANNUAL,
        'float_dc_type': DayCountTypes.ACT_360,
        'bus_day_type': BusDayAdjustTypes.MODIFIED_FOLLOWING,
        'interp_type': InterpTypes.LINEAR_ZERO_RATES,
        'spot_days': 2
    }


@pytest.fixture
def gbp_curve_params():
    """Standard GBP curve building parameters"""
    return {
        'swap_type': SwapTypes.PAY,
        'fixed_dcc_type': DayCountTypes.ACT_365F,
        'fixed_freq_type': FrequencyTypes.ANNUAL,
        'float_freq_type': FrequencyTypes.ANNUAL,
        'float_dc_type': DayCountTypes.ACT_365F,
        'bus_day_type': BusDayAdjustTypes.MODIFIED_FOLLOWING,
        'interp_type': InterpTypes.LINEAR_ZERO_RATES,
        'spot_days': 0
    }


@pytest.fixture
def standard_ois_params(standard_value_date):
    """Standard OIS swap parameters"""
    return {
        'effective_dt': standard_value_date.add_weekdays(1),
        'term_dt_or_tenor': '5Y',
        'fixed_leg_type': SwapTypes.PAY,
        'fixed_coupon': 0.05,
        'fixed_freq_type': FrequencyTypes.ANNUAL,
        'fixed_dc_type': DayCountTypes.ACT_365F,
        'bd_type': BusDayAdjustTypes.MODIFIED_FOLLOWING,
        'float_freq_type': FrequencyTypes.ANNUAL,
        'float_dc_type': DayCountTypes.ACT_365F
    }


# Test markers for categorizing tests
def pytest_configure(config):
    """Configure custom test markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow (may take longer to run)")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "market_data: marks tests that require market data")
    config.addinivalue_line("markers", "numerical: marks tests with numerical precision requirements")


# Pytest collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add default markers"""
    for item in items:
        # Add 'unit' marker to all tests by default
        if not any(item.iter_markers()):
            item.add_marker(pytest.mark.unit)
            
        # Mark slow tests based on naming convention
        if "performance" in item.name or "long" in item.name:
            item.add_marker(pytest.mark.slow)
            
        # Mark integration tests
        if "integration" in item.name or item.fspath.basename.startswith("test_integration"):
            item.add_marker(pytest.mark.integration)


# Utility functions for tests
@pytest.fixture
def tolerance():
    """Standard numerical tolerance for floating point comparisons"""
    return 1e-6


@pytest.fixture
def strict_tolerance():
    """Strict numerical tolerance for high precision tests"""
    return 1e-10


@pytest.fixture
def approx_equal():
    """Helper function for approximate equality testing"""
    def _approx_equal(a, b, tol=1e-6):
        return abs(a - b) < tol
    return _approx_equal


# Skip conditions for optional dependencies
def pytest_runtest_setup(item):
    """Setup function to skip tests based on conditions"""
    # Skip market data tests if no Bloomberg connection
    if "market_data" in [mark.name for mark in item.iter_markers()]:
        try:
            import xbbg
            # Could add actual Bloomberg connection test here
        except ImportError:
            pytest.skip("Bloomberg market data not available")
            
    # Skip JAX-dependent tests if JAX not available
    if "jax" in [mark.name for mark in item.iter_markers()]:
        try:
            import jax
        except ImportError:
            pytest.skip("JAX not available")


# Session-scoped setup and teardown
@pytest.fixture(scope="session", autouse=True)
def test_session_setup():
    """Setup that runs once per test session"""
    # Could initialize logging, check dependencies, etc.
    print("\n=== Starting Cavour Test Session ===")
    yield
    print("\n=== Ending Cavour Test Session ===")


# Function to generate test data dynamically
@pytest.fixture
def curve_test_data_generator():
    """Generator for creating test curve data"""
    def _generate_curve_data(num_points=10, rate_level=0.05, curve_shape="normal"):
        """Generate synthetic curve data for testing"""
        import numpy as np
        
        if curve_shape == "normal":
            # Normal upward sloping curve
            rates = np.linspace(rate_level - 0.01, rate_level + 0.01, num_points)
        elif curve_shape == "inverted":
            # Inverted yield curve
            rates = np.linspace(rate_level + 0.01, rate_level - 0.01, num_points)
        elif curve_shape == "flat":
            # Flat curve
            rates = np.full(num_points, rate_level)
        else:
            raise ValueError(f"Unknown curve shape: {curve_shape}")
            
        return rates.tolist()
    
    return _generate_curve_data