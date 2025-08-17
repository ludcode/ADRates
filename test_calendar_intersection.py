#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from cavour.utils.calendar import Calendar, CalendarTypes, create_calendar_intersection
from cavour.utils.date import Date

def test_calendar_intersection_basic():
    """Test basic intersection functionality"""
    print("Testing basic calendar intersection...")
    
    # Create individual calendars
    us_cal = Calendar(CalendarTypes.UNITED_STATES)
    uk_cal = Calendar(CalendarTypes.UNITED_KINGDOM)
    
    # Create intersection calendar
    intersection_cal = create_calendar_intersection(us_cal, uk_cal)
    
    # Verify it's an intersection type
    assert intersection_cal._cal_type == CalendarTypes.INTERSECTION
    assert len(intersection_cal._constituent_calendars) == 2
    
    print("[OK] Basic intersection creation works")

def test_business_day_logic():
    """Test that intersection requires ALL calendars to agree it's a business day"""
    print("Testing business day intersection logic...")
    
    us_cal = Calendar(CalendarTypes.UNITED_STATES)
    uk_cal = Calendar(CalendarTypes.UNITED_KINGDOM)
    intersection_cal = create_calendar_intersection(us_cal, uk_cal)
    
    # Test a regular business day (Wednesday, not a holiday)
    regular_day = Date(15, 3, 2023)  # March 15, 2023 (Wednesday)
    
    us_business = us_cal.is_business_day(regular_day)
    uk_business = uk_cal.is_business_day(regular_day)
    intersection_business = intersection_cal.is_business_day(regular_day)
    
    print(f"  March 15, 2023 - US: {us_business}, UK: {uk_business}, Intersection: {intersection_business}")
    
    # Should be business day in all calendars
    assert us_business == True
    assert uk_business == True
    assert intersection_business == True
    
    print("[OK] Regular business day works correctly")

def test_holiday_logic():
    """Test holidays in intersection calendar"""
    print("Testing holiday intersection logic...")
    
    us_cal = Calendar(CalendarTypes.UNITED_STATES)
    uk_cal = Calendar(CalendarTypes.UNITED_KINGDOM)
    intersection_cal = create_calendar_intersection(us_cal, uk_cal)
    
    # Test Christmas Day (holiday in both calendars)
    christmas = Date(25, 12, 2023)  # December 25, 2023
    
    us_holiday = us_cal.is_holiday(christmas)
    uk_holiday = uk_cal.is_holiday(christmas)
    intersection_holiday = intersection_cal.is_holiday(christmas)
    
    print(f"  Christmas 2023 - US holiday: {us_holiday}, UK holiday: {uk_holiday}, Intersection holiday: {intersection_holiday}")
    
    # Christmas should be a holiday in both
    assert us_holiday == True
    assert uk_holiday == True
    assert intersection_holiday == True
    
    # Business day should be False for intersection
    intersection_business = intersection_cal.is_business_day(christmas)
    assert intersection_business == False
    
    print("[OK] Christmas holiday works correctly")

def test_us_specific_holiday():
    """Test US-specific holiday that UK doesn't observe"""
    print("Testing US-specific holiday...")
    
    us_cal = Calendar(CalendarTypes.UNITED_STATES)
    uk_cal = Calendar(CalendarTypes.UNITED_KINGDOM)
    intersection_cal = create_calendar_intersection(us_cal, uk_cal)
    
    # Test July 4th (Independence Day - US holiday only)
    july_4th = Date(4, 7, 2023)  # July 4, 2023 (Tuesday)
    
    us_holiday = us_cal.is_holiday(july_4th)
    uk_holiday = uk_cal.is_holiday(july_4th)
    intersection_holiday = intersection_cal.is_holiday(july_4th)
    
    us_business = us_cal.is_business_day(july_4th)
    uk_business = uk_cal.is_business_day(july_4th)
    intersection_business = intersection_cal.is_business_day(july_4th)
    
    print(f"  July 4th 2023 - US holiday: {us_holiday}, UK holiday: {uk_holiday}")
    print(f"  July 4th 2023 - US business: {us_business}, UK business: {uk_business}, Intersection business: {intersection_business}")
    
    # July 4th should be holiday in US but not UK
    assert us_holiday == True
    assert uk_holiday == False
    
    # Intersection should treat it as holiday (ANY calendar considers it holiday)
    assert intersection_holiday == True
    
    # Business day logic: US=False, UK=True, Intersection=False (since US considers it non-business)
    assert us_business == False
    assert uk_business == True
    assert intersection_business == False
    
    print("[OK] US-specific holiday works correctly")

def test_weekend():
    """Test weekend handling"""
    print("Testing weekend handling...")
    
    us_cal = Calendar(CalendarTypes.UNITED_STATES)
    uk_cal = Calendar(CalendarTypes.UNITED_KINGDOM)
    intersection_cal = create_calendar_intersection(us_cal, uk_cal)
    
    # Test Saturday
    saturday = Date(18, 3, 2023)  # March 18, 2023 (Saturday)
    
    us_business = us_cal.is_business_day(saturday)
    uk_business = uk_cal.is_business_day(saturday)
    intersection_business = intersection_cal.is_business_day(saturday)
    
    print(f"  Saturday - US: {us_business}, UK: {uk_business}, Intersection: {intersection_business}")
    
    # Weekends should not be business days
    assert us_business == False
    assert uk_business == False
    assert intersection_business == False
    
    print("[OK] Weekend handling works correctly")

def test_multiple_calendars():
    """Test intersection with more than 2 calendars"""
    print("Testing intersection with 3 calendars...")
    
    us_cal = Calendar(CalendarTypes.UNITED_STATES)
    uk_cal = Calendar(CalendarTypes.UNITED_KINGDOM)
    de_cal = Calendar(CalendarTypes.GERMANY)
    
    intersection_cal = create_calendar_intersection(us_cal, uk_cal, de_cal)
    
    assert intersection_cal._cal_type == CalendarTypes.INTERSECTION
    assert len(intersection_cal._constituent_calendars) == 3
    
    # Test a regular business day
    regular_day = Date(15, 3, 2023)  # March 15, 2023
    
    us_business = us_cal.is_business_day(regular_day)
    uk_business = uk_cal.is_business_day(regular_day)
    de_business = de_cal.is_business_day(regular_day)
    intersection_business = intersection_cal.is_business_day(regular_day)
    
    print(f"  March 15, 2023 - US: {us_business}, UK: {uk_business}, DE: {de_business}, Intersection: {intersection_business}")
    
    # Should be business day in all
    assert intersection_business == (us_business and uk_business and de_business)
    
    print("[OK] Multiple calendar intersection works correctly")

def test_error_cases():
    """Test error handling"""
    print("Testing error cases...")
    
    us_cal = Calendar(CalendarTypes.UNITED_STATES)
    
    # Test with only one calendar
    try:
        create_calendar_intersection(us_cal)
        assert False, "Should have raised error"
    except Exception as e:
        print(f"  [OK] Single calendar error: {e}")
    
    # Test with invalid input
    try:
        create_calendar_intersection(us_cal, "not_a_calendar")
        assert False, "Should have raised error"
    except Exception as e:
        print(f"  [OK] Invalid input error: {e}")
    
    print("[OK] Error handling works correctly")

def run_all_tests():
    """Run all tests"""
    print("=== Testing Calendar Intersection Functionality ===\n")
    
    try:
        test_calendar_intersection_basic()
        print()
        
        test_business_day_logic()
        print()
        
        test_holiday_logic()
        print()
        
        test_us_specific_holiday()
        print()
        
        test_weekend()
        print()
        
        test_multiple_calendars()
        print()
        
        test_error_cases()
        print()
        
        print("=== ALL TESTS PASSED! ===")
        
    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)