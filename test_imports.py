#!/usr/bin/env python3
"""
Test script to verify that all imports work correctly after converting relative imports to absolute imports
"""

def test_imports():
    """Test key imports to ensure they work correctly"""
    try:
        # Test utils imports
        from cavour.utils.date import Date
        from cavour.utils.day_count import DayCount, DayCountTypes
        from cavour.utils.frequency import FrequencyTypes
        from cavour.utils.global_types import SwapTypes, CurveTypes
        from cavour.utils.currency import CurrencyTypes
        print("SUCCESS: Utils imports successful")
        
        # Test market curves imports
        from cavour.market.curves.discount_curve import DiscountCurve
        from cavour.market.curves.interpolator import InterpTypes
        print("SUCCESS: Market curves imports successful")
        
        # Test trades imports
        from cavour.trades.rates.ois import OIS
        from cavour.trades.rates.swap_fixed_leg import SwapFixedLeg
        from cavour.trades.rates.swap_float_leg import SwapFloatLeg
        from cavour.trades.rates.ois_curve import OISCurve
        print("SUCCESS: Trades imports successful")
        
        # Test experimental XCCY curve
        from cavour.trades.rates.experimental.xccy_curve import XCCYCurve, bootstrap_xccy_curve
        print("SUCCESS: Experimental XCCY imports successful")
        
        # Test basic functionality
        value_dt = Date(1, 1, 2024)
        print(f"SUCCESS: Date creation successful: {value_dt}")
        
        # Test enum usage
        swap_type = SwapTypes.PAY
        print(f"SUCCESS: Enum usage successful: {swap_type}")
        
        print("\nSUCCESS: All imports working correctly!")
        return True
        
    except ImportError as e:
        print(f"ERROR: Import error: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Other error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1)