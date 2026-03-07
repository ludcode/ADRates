# USD SOFR Multi-Instrument Curve Refit Validation Summary

## Executive Summary

Validated two non-overlapping USD SOFR curve structures against target tolerance `SWAP_TOL = 1e-10` (from [ois_curve.py:67](../cavour/trades/rates/ois_curve.py#L67)):

| Curve | Structure | Pass Rate | Max Error | Status |
|-------|-----------|-----------|-----------|--------|
| **Curve 1** | Deposits + FRAs + OIS | **50.0%** (8/16) | 7.32e-05 | PARTIAL |
| **Curve 2** | Deposits + Futures + OIS | **100%** (19/19) | 2.33e-16 | ✓ COMPLETE |

---

## Curve 1: Deposits + FRAs + OIS

### Structure
- **Deposits** (0-3M): 1M, 2M, 3M
- **FRAs** (3M-18M): 3x6, 6x9, 9x12, 12x15, 15x18
- **OIS** (2Y+): 2Y, 3Y, 5Y, 7Y, 10Y, 15Y, 20Y, 30Y

### Results

| Instrument Type | Pass/Total | Max Error | Status |
|----------------|------------|-----------|---------|
| **Deposits** | 3/3 (100%) | 0.00e+00 | ✓ PERFECT |
| **FRAs** | 5/5 (100%) | 6.42e-17 | ✓ PERFECT |
| **OIS Swaps** | 0/8 (0%) | 7.32e-05 | ✗ FAIL |
| **TOTAL** | 8/16 (50.0%) | 7.32e-05 | ⚠ PARTIAL |

### Detailed FRA Results (AFTER FIX)
```
28-MAY-2026 to 26-AUG-2026: PV = -$0.0000, abs_err = 3.77e-17  PASS
28-AUG-2026 to 26-NOV-2026: PV = +$0.0000, abs_err = 6.42e-17  PASS
30-NOV-2026 to 26-FEB-2027: PV = -$0.0000, abs_err = 2.12e-17  PASS
02-MAR-2027 to 26-MAY-2027: PV = -$0.0000, abs_err = 4.81e-17  PASS
28-MAY-2027 to 26-AUG-2027: PV = +$0.0000, abs_err = 5.19e-17  PASS
```

### Detailed OIS Results
```
28-FEB-2028: PV = -$73.24, abs_err = 7.32e-05  FAIL
26-FEB-2029: PV = -$72.97, abs_err = 7.30e-05  FAIL
26-FEB-2031: PV = -$72.30, abs_err = 7.23e-05  FAIL
28-FEB-2033: PV = -$71.76, abs_err = 7.18e-05  FAIL
26-FEB-2036: PV = -$70.95, abs_err = 7.10e-05  FAIL
26-FEB-2041: PV = -$70.15, abs_err = 7.01e-05  FAIL
26-FEB-2046: PV = -$69.61, abs_err = 6.96e-05  FAIL
28-FEB-2056: PV = -$69.21, abs_err = 6.92e-05  FAIL
```

### Key Finding
- **FRAs** achieved machine precision after implementing iterative fixed-point solver
- **OIS swaps** fail with ~7e-5 errors (730x worse than target) even though FRAs are perfect
- **Root cause**: Issue with OIS bootstrap when FRAs are present (NOT from FRA calibration errors propagating)

---

## Curve 2: Deposits + Futures + OIS

### Structure
- **Deposits** (0-3M): 1M, 2M, 3M
- **Futures** (IMM Quarterly, 3M-2Y): H26, M26, U26, Z26, H27, M27, U27, Z27
- **OIS** (2Y+): 2Y, 3Y, 5Y, 7Y, 10Y, 15Y, 20Y, 30Y

### Results

| Instrument Type | Pass/Total | Max Error | Status |
|----------------|------------|-----------|---------|
| **Deposits** | 3/3 (100%) | 0.00e+00 | ✓ PERFECT |
| **Futures** | 8/8 (100%) | 9.18e-17 | ✓ PERFECT |
| **OIS Swaps** | 8/8 (100%) | 2.33e-16 | ✓ PERFECT |
| **TOTAL** | 19/19 (100%) | 2.33e-16 | ✓ COMPLETE |

### Detailed Futures Results (AFTER FIX)
```
18-JUN-2026 (H26): PV = -$0.0000, abs_err = 6.46e-17  PASS
17-SEP-2026 (M26): PV = +$0.0000, abs_err = 2.07e-17  PASS
16-DEC-2026 (U26): PV = +$0.0000, abs_err = 4.37e-17  PASS
16-MAR-2027 (Z26): PV = +$0.0000, abs_err = 9.18e-17  PASS
17-JUN-2027 (H27): PV = +$0.0000, abs_err = 4.13e-17  PASS
16-SEP-2027 (M27): PV = -$0.0000, abs_err = 2.12e-17  PASS
15-DEC-2027 (U27): PV = +$0.0000, abs_err = 5.57e-17  PASS
15-MAR-2028 (Z27): PV = +$0.0000, abs_err = 9.10e-17  PASS
```

### Detailed OIS Results
```
28-FEB-2028: PV = +$0.0000, abs_err = 1.02e-16  PASS
26-FEB-2029: PV = +$0.0000, abs_err = 1.46e-16  PASS
26-FEB-2031: PV = +$0.0000, abs_err = 5.82e-17  PASS
28-FEB-2033: PV = -$0.0000, abs_err = 1.75e-16  PASS
26-FEB-2036: PV = -$0.0000, abs_err = 1.75e-16  PASS
26-FEB-2041: PV = -$0.0000, abs_err = 2.33e-16  PASS
26-FEB-2046: PV = +$0.0000, abs_err = 0.00e+00  PASS
28-FEB-2056: PV = +$0.0000, abs_err = 2.33e-16  PASS
```

### Key Finding
- **All instruments** achieve machine precision after interpolator fix
- **OIS swaps** achieve perfect refit (~1e-16) when using Futures
- **H26 future** fixed from 8.37e-04 to 6.46e-17 via interpolator.py correction
- **100% pass rate** - all 19 instruments meet tolerance 1e-10

---

## Implementation Changes

### FRA Bootstrap Fix ([engine.py:3560-3591](../cavour/market/position/engine.py#L3560-L3591))

**Problem**: Bootstrap-validation inconsistency
- During bootstrap: DF(start) interpolated from points [0, i-1] (excludes FRA end point)
- During validation: DF(start) interpolated from points [0, i] (includes FRA end point)
- Result: ~1e-6 calibration errors

**Solution**: Iterative fixed-point solver
```python
# Fixed-point iteration: df_end^{n+1} = df_start(df_end^n) / (1 + rate * acc)
def fixed_point_step(df_end_guess):
    # Add df_end to curve temporarily
    temp_dfs = dfs_arr.at[i].set(df_end_guess)

    # Interpolate DF(start) from COMPLETE curve (includes df_end)
    df_start_temp = interpolate_zero_rates(start_mat, temp_mats, temp_dfs)

    # Apply FRA formula
    return df_start_temp / (1.0 + rate * acc)

# Run 10 iterations (converges from ~1e-6 to <1e-10)
```

**Result**:
- FRA errors: ~1e-6 → **~6e-17** (machine precision)
- All 5 FRAs now PASS tolerance 1e-10

---

## Outstanding Issues

### Issue 1: OIS Calibration with FRAs Present

**Symptoms**:
- OIS swaps achieve perfect refit (~1e-16) when built with Deposits + OIS only
- OIS swaps achieve perfect refit (~1e-16) when built with Deposits + Futures + OIS
- OIS swaps FAIL (~7e-05) when built with Deposits + FRAs + OIS
- FRAs themselves are perfect (~6e-17) after fix

**Evidence**:
```bash
# Deposits + OIS only: PERFECT
OIS errors: ~1e-16

# Deposits + FRAs only: PERFECT
FRA errors: ~6e-17

# Deposits + FRAs + OIS: FRAs PERFECT, OIS FAIL
FRA errors: ~6e-17
OIS errors: ~7e-05
```

**Hypothesis**:
- Possible deduplication issue between FRA endpoints and OIS cashflows
- FRA-specific PV01 accumulation may affect downstream OIS bootstrap
- Needs further investigation in [engine.py:3580-3587](../cavour/market/position/engine.py#L3580-L3587) (PV01 logic)

**Impact**: 8/16 instruments fail in Curve 1 (50% pass rate)

### Issue 2: First IR Future Calibration (RESOLVED)

**Symptoms** (BEFORE FIX):
- One future (H26, June 2026) failed with ~8e-04 error
- All other futures (7/8) achieved machine precision (~1e-17)

**Evidence** (BEFORE):
```
18-JUN-2026 (H26): PV = -$837.43, abs_err = 8.37e-04  FAIL
17-SEP-2026 (M26): PV = +$0.0000, abs_err = 2.07e-17  PASS
```

**Root Cause**: Bug in `interpolator.py` lines 96-98
- When interpolating in first interval [0, first_curve_point], code used same point twice
- H26's accrual start (t=0.0556) fell in first interval [0, 0.0778]
- Line 97: `r1 = -np.log(dfs[i]) / times[i]` should be `r1 = 0.0 if times[i-1] < 1e-10 else -np.log(dfs[i-1]) / times[i-1]`

**Fix Applied** ([interpolator.py:96-101](../cavour/market/curves/interpolator.py#L96-L101)):
```python
if i == 1:
    # Special case: first interval from t=0 (DF=1.0, zero_rate=0.0) to first curve point
    r1 = 0.0 if times[i - 1] < 1e-10 else -np.log(dfs[i - 1]) / times[i - 1]
    r2 = -np.log(dfs[i]) / times[i]
    dt = times[i] - times[i - 1]
    rvalue = ((times[i] - t) * r1 + (t - times[i - 1]) * r2) / dt
    yvalue = np.exp(-rvalue * t)
```

**Result** (AFTER FIX):
```
18-JUN-2026 (H26): PV = -$0.0000, abs_err = 6.46e-17  PASS
```

**Impact**: Issue RESOLVED - Curve 2 now 100% pass rate (19/19)

---

## Validation Against User's Claim

**User's Original Claim**:
> "All instruments now reprice within ~2.2bp, which meets the requirement for 'extremely small difference'!"

**Reality (AFTER ALL FIXES)**:
- **Target tolerance**: 1e-10 (SWAP_TOL from [ois_curve.py:67](../cavour/trades/rates/ois_curve.py#L67))
- **Curve 2 result**: 100% pass rate (19/19 instruments, max error 2.33e-16)
- **Curve 1 result**: 50% pass rate (8/16 instruments, OIS errors ~7e-05)

**Conclusion**:
- ✓ **Curve 2 (Deposits + Futures + OIS)**: ALL instruments meet 1e-10 tolerance at machine precision
- ✗ **Curve 1 (Deposits + FRAs + OIS)**: OIS instruments still fail with ~0.73bp errors (730x worse than target)
- ✓ Individual instrument types (deposits, FRAs, futures, OIS without FRAs) all achieve machine precision when isolated

---

## Recommendations

### 1. **FRA Implementation** (✓ RESOLVED)
   - Fixed via iterative fixed-point solver in [engine.py:3560-3591](../cavour/market/position/engine.py#L3560-L3591)
   - All FRAs achieve machine precision (~6e-17)
   - Ready for production use

### 2. **IR Future Interpolation** (✓ RESOLVED)
   - Fixed via first-interval correction in [interpolator.py:96-101](../cavour/market/curves/interpolator.py#L96-L101)
   - H26 future now achieves machine precision (~6e-17)
   - All 8 futures pass tolerance 1e-10

### 3. **OIS + FRA Interaction** (⚠ REQUIRES INVESTIGATION)
   - OIS swaps fail (~7e-05) ONLY when FRAs are present in the same curve
   - Works perfectly with Deposits + OIS (machine precision)
   - Works perfectly with Deposits + Futures + OIS (machine precision)
   - Likely issue: deduplication or PV01 accumulation when FRA endpoints coincide with OIS cashflows
   - Needs investigation in [engine.py:3580-3587](../cavour/market/position/engine.py#L3580-L3587) (PV01 logic)

### 4. **Production Recommendation**

**For Immediate Production Use**:
- ✓ **Curve 2 (Deposits + Futures + OIS)**: **100% pass rate** (19/19 instruments)
- All instruments achieve machine precision (max error 2.33e-16)
- Recommended for production deployment

**For Development/Testing Only**:
- ⚠ **Curve 1 (Deposits + FRAs + OIS)**: **50% pass rate** (8/16 instruments)
- FRAs work perfectly in isolation, but OIS fails when combined
- Requires fix before production use
- Use Curve 2 (Futures) instead until OIS+FRA interaction is resolved

---

## Test Files

- [tests/sample_market_data_usd_sofr.py](sample_market_data_usd_sofr.py) - Realistic USD SOFR market data
- [tests/test_usd_curve_deposits_fras_ois.py](test_usd_curve_deposits_fras_ois.py) - Curve 1 validation
- [tests/test_usd_curve_deposits_futures_ois.py](test_usd_curve_deposits_futures_ois.py) - Curve 2 validation
- [tests/test_usd_curve_refit_comparison.py](test_usd_curve_refit_comparison.py) - Comprehensive comparison

**Run tests**:
```bash
cd /c/Projects/Cavour
cavourvenv/Scripts/python.exe tests/test_usd_curve_deposits_fras_ois.py
cavourvenv/Scripts/python.exe tests/test_usd_curve_deposits_futures_ois.py
cavourvenv/Scripts/python.exe tests/test_usd_curve_refit_comparison.py
```
