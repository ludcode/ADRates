# Cross-Currency (XCCY) Curve Bootstrapping - Cashflow-Based Approach

## Overview

This document describes how to bootstrap a cross-currency (XCCY) curve using a cashflow-based approach with the Cavour library. This method eliminates the need for iterative solvers by directly calculating the unknown discount factor through simple division, leveraging the `SwapFloatLeg` class to project all cashflows.

## Key Principle

The fundamental insight is that for each basis swap, we can use `SwapFloatLeg` to project all the cashflows except the final payment. Since all earlier discount factors are already known from previous bootstrap steps, the only unknown is the discount factor for the final payment date. This can be solved directly using:

```
DF_final = (Sum_of_known_PVs_difference) / Final_cashflow_difference
```

## Prerequisites

1. **Pre-calibrated OIS curves** for both currencies
2. **Basis spreads** for sequential maturities
3. **SwapFloatLeg** class to project cashflows

## Bootstrap Algorithm

### Step 1: Cashflow Projection

For each basis spread maturity, use `SwapFloatLeg` to project all cashflows:

```python
# Target currency leg (with basis spread)
target_leg = SwapFloatLeg(
    effective_dt=value_dt,
    end_dt=maturity,
    leg_type=SwapTypes.PAY,
    spread=basis_spread,
    freq_type=FrequencyTypes.QUARTERLY,
    dc_type=DayCountTypes.ACT_360,
    floating_index=CurveTypes.GBP_OIS_SONIA,
    currency=CurrencyTypes.GBP
)

# Collateral currency leg (no spread)
collateral_leg = SwapFloatLeg(
    effective_dt=value_dt,
    end_dt=maturity, 
    leg_type=SwapTypes.RECEIVE,
    spread=0.0,
    freq_type=FrequencyTypes.QUARTERLY,
    dc_type=DayCountTypes.ACT_360,
    floating_index=CurveTypes.USD_OIS_SOFR,
    currency=CurrencyTypes.USD
)

# Generate payment schedules and cashflows
target_leg.generate_payment_dts()
collateral_leg.generate_payment_dts()
```

### Step 2: Calculate Known PVs

Value all cashflows except the final payment using existing discount factors:

```python
def calculate_known_pvs(leg, discount_curve, index_curve, exclude_final=True):
    """
    Calculate PV of all cashflows except the final one
    """
    leg_pv = 0.0
    final_payment_cf = 0.0
    final_payment_df = 0.0
    
    num_payments = len(leg._payment_dts)
    end_index = num_payments - 1 if exclude_final else num_payments
    
    for i in range(end_index):
        # Get cashflow details
        start_dt = leg._start_accrued_dts[i]
        end_dt = leg._end_accrued_dts[i] 
        payment_dt = leg._payment_dts[i]
        year_frac = leg._year_fracs[i]
        
        # Calculate forward rate
        df_start = index_curve.df(start_dt)
        df_end = index_curve.df(end_dt)
        index_alpha = leg._year_fracs[i]  # Using payment leg day count
        fwd_rate = (df_start / df_end - 1.0) / index_alpha
        
        # Calculate cashflow
        cashflow = (fwd_rate + leg._spread) * year_frac * leg._notional
        
        # Discount using known discount factors
        df_payment = discount_curve.df(payment_dt)
        leg_pv += cashflow * df_payment
    
    # Calculate final cashflow amount (but not its PV yet)
    if num_payments > 0:
        i = num_payments - 1
        start_dt = leg._start_accrued_dts[i]
        end_dt = leg._end_accrued_dts[i]
        year_frac = leg._year_fracs[i]
        
        # Final period forward rate
        df_start = index_curve.df(start_dt)
        df_end = index_curve.df(end_dt)
        index_alpha = year_frac
        fwd_rate = (df_start / df_end - 1.0) / index_alpha
        
        final_payment_cf = (fwd_rate + leg._spread) * year_frac * leg._notional
    
    return leg_pv, final_payment_cf
```

### Step 3: Direct Discount Factor Calculation

Solve for the unknown discount factor using the cross-currency basis swap equation:

```python
def bootstrap_single_maturity(basis_spread_data, xccy_curve_partial, 
                            target_ois_curve, collateral_ois_curve, value_dt):
    """
    Bootstrap a single maturity using direct cashflow approach
    """
    maturity = basis_spread_data["maturity"]
    basis_spread = basis_spread_data["spread"]
    
    # Create floating legs
    target_leg = create_target_leg(value_dt, maturity, basis_spread)
    collateral_leg = create_collateral_leg(value_dt, maturity)
    
    # Calculate known PVs (all payments except final)
    target_known_pv, target_final_cf = calculate_known_pvs(
        target_leg, xccy_curve_partial, target_ois_curve, exclude_final=True)
    
    collateral_known_pv, collateral_final_cf = calculate_known_pvs(
        collateral_leg, collateral_ois_curve, collateral_ois_curve, exclude_final=True)
    
    # For cross-currency basis swap: PV_target = PV_collateral
    # Known_PV_target + Final_CF_target * DF_final = Known_PV_collateral + Final_CF_collateral * DF_collateral_final
    
    # Get collateral currency final discount factor (known from collateral OIS curve)
    final_payment_dt = target_leg._payment_dts[-1]  # Should match collateral leg
    df_collateral_final = collateral_ois_curve.df(final_payment_dt)
    
    # Solve for target currency final discount factor
    # target_known_pv + target_final_cf * df_target_final = collateral_known_pv + collateral_final_cf * df_collateral_final
    
    rhs = collateral_known_pv + collateral_final_cf * df_collateral_final
    lhs_known = target_known_pv
    
    df_target_final = (rhs - lhs_known) / target_final_cf
    
    return final_payment_dt, df_target_final
```

### Step 4: Sequential Bootstrap Process

```python
def bootstrap_xccy_curve(value_dt, target_ois_curve, collateral_ois_curve, basis_spreads):
    """
    Bootstrap the complete XCCY curve using sequential cashflow approach
    """
    # Initialize with value date
    xccy_dates = [value_dt]
    xccy_dfs = [1.0]
    
    # Build partial curve for interpolation
    xccy_curve_partial = DiscountCurve(value_dt, xccy_dates, xccy_dfs)
    
    for spread_data in basis_spreads:
        # Bootstrap single maturity
        maturity_dt, df_final = bootstrap_single_maturity(
            spread_data, xccy_curve_partial, target_ois_curve, 
            collateral_ois_curve, value_dt)
        
        # Add to curve
        xccy_dates.append(maturity_dt)
        xccy_dfs.append(df_final)
        
        # Update partial curve for next iteration
        xccy_curve_partial = DiscountCurve(value_dt, xccy_dates.copy(), xccy_dfs.copy())
    
    # Return final bootstrapped curve
    return DiscountCurve(value_dt, xccy_dates, xccy_dfs)
```

## Mathematical Framework

For each cross-currency basis swap at maturity T:

```
Σ(i=1 to n-1) CF_target_i * DF_target_i + CF_target_n * DF_target_n = 
Σ(i=1 to n-1) CF_collateral_i * DF_collateral_i + CF_collateral_n * DF_collateral_n
```

Where:
- `CF_target_i`: Target currency cashflow at payment i (includes basis spread)
- `CF_collateral_i`: Collateral currency cashflow at payment i  
- `DF_target_i`: Target currency discount factor (from XCCY curve)
- `DF_collateral_i`: Collateral currency discount factor (from collateral OIS curve)
- `n`: Final payment index

Rearranging to solve for the unknown `DF_target_n`:

```
DF_target_n = (RHS - LHS_known) / CF_target_n

Where:
RHS = Σ(i=1 to n) CF_collateral_i * DF_collateral_i
LHS_known = Σ(i=1 to n-1) CF_target_i * DF_target_i
```

## Key Advantages

1. **No iterative solver required**: Direct calculation through simple division
2. **Exact solution**: No convergence tolerance or iteration limits
3. **Efficient**: O(n) complexity for n maturities  
4. **Stable**: No numerical instability from iterative processes
5. **Transparent**: Clear mathematical relationship between inputs and outputs
6. **SwapFloatLeg integration**: Leverages existing cashflow projection functionality

## Implementation Details

### Cashflow Projection
- Use `SwapFloatLeg.generate_payment_dts()` to create payment schedules
- Calculate forward rates from the respective OIS curves
- Apply basis spread only to target currency leg
- Handle day count conventions consistently

### Discount Factor Bootstrapping  
- Process maturities sequentially from shortest to longest
- Each new discount factor only affects its own maturity date
- Previously bootstrapped discount factors remain unchanged
- Build partial curves incrementally for intermediate calculations

### Error Handling
- Validate that payment schedules match between legs
- Check for positive discount factors
- Ensure cashflow amounts are reasonable
- Verify curve monotonicity if required

## Example Usage

```python
# Input data
basis_spreads = [
    {"maturity": "3M", "spread": 0.0015},
    {"maturity": "6M", "spread": 0.0018}, 
    {"maturity": "1Y", "spread": 0.0020},
    {"maturity": "2Y", "spread": 0.0025}
]

# Bootstrap curve
xccy_curve = bootstrap_xccy_curve(
    value_dt=Date(2024, 1, 15),
    target_ois_curve=gbp_sonia_curve,
    collateral_ois_curve=usd_sofr_curve,
    basis_spreads=basis_spreads
)

# Use for pricing
cross_currency_swap_pv = price_xccy_swap(xccy_curve, ...)
```

This cashflow-based approach provides a robust, efficient, and mathematically sound method for bootstrapping cross-currency curves without the complexity of iterative solvers.