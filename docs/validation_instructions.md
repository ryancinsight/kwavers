# Validation Instructions: Kwavers vs k-wave-python

## Overview

This document explains how to validate that kwavers utility functions produce identical outputs to k-wave-python.

## Prerequisites

1. **k-wave-python installed:**
   ```bash
   pip install k-wave-python
   ```

2. **pykwavers built:**
   ```bash
   cd pykwavers
   maturin develop --release
   ```

## Running Validation

### Full Validation Suite

```bash
python validate_against_kwave.py
```

### With Verbose Output

```bash
python validate_against_kwave.py --verbose
```

### Specific Tests Only

```bash
python validate_against_kwave.py --tests tone_burst get_win
```

### With Custom Tolerance

```bash
python validate_against_kwave.py --tolerance 1e-8
```

## What Gets Validated

### 1. Signal Generation

**tone_burst:**
- Multiple frequency combinations
- Different numbers of cycles
- Validates amplitude, phase, and envelope

**get_win:**
- Hanning, Hamming, Blackman, Rectangular windows
- Validates window shape and values

**create_cw_signals:**
- Multi-channel continuous wave generation
- Phase offset validation

### 2. Geometry Generation

**make_disc:**
- 2D circular masks
- Validates pixel count and shape
- Compares boolean arrays

**make_ball:**
- 3D spherical masks
- Validates voxel count
- Volume calculation verification

**make_line:**
- Line/linear array masks
- Bresenham algorithm validation

### 3. Unit Conversions

**db2neper / neper2db:**
- Roundtrip validation
- Known value verification

## Expected Results

All tests should pass with differences < 1e-10 (machine precision for f64).

### Success Output:
```
======================================================================
VALIDATION SUMMARY
======================================================================
tone_burst            : ✓ PASS
get_win               : ✓ PASS
make_disc             : ✓ PASS
make_ball             : ✓ PASS
db2neper              : ✓ PASS
======================================================================
✓ ALL TESTS PASSED
kwavers implementations match k-wave-python outputs
```

### Failure Output:
```
tone_burst case 1: ✗ FAIL (max diff: 1.23e-05 at (45,), tolerance: 1.00e-10)
```

## Interpreting Results

### If tests pass:
- ✅ kwavers implementations are numerically identical to k-wave-python
- ✅ Both Rust and Python examples will produce identical results
- ✅ No further action needed

### If tests fail:
1. **Check tolerance:** Some operations may have larger numerical differences
2. **Check API compatibility:** Ensure function signatures match
3. **Review algorithm:** Check for implementation differences
4. **Check data types:** Ensure both use f64 (double precision)

## Common Issues

### Shape Mismatches
- k-wave-python may return 2D arrays where kwavers returns 3D
- Solution: Extract appropriate slices before comparison

### Window Name Differences
- k-wave uses "Hann", kwavers accepts "Hanning" or "Hann"
- Solution: Normalize window names before comparison

### Coordinate Systems
- Ensure both use same units (meters vs grid points)
- Check center position definitions

## Debugging Failed Tests

1. **Run with verbose flag:**
   ```bash
   python validate_against_kwave.py --verbose --tests <failing_test>
   ```

2. **Plot differences:**
   ```python
   import matplotlib.pyplot as plt
   diff = np.abs(kw_result - kwa_result)
   plt.plot(diff)
   plt.show()
   ```

3. **Check at specific indices:**
   ```python
   max_idx = np.unravel_index(np.argmax(diff), diff.shape)
   print(f"k-wave value: {kw_result[max_idx]}")
   print(f"kwavers value: {kwa_result[max_idx]}")
   ```

## Integration with CI

Add to CI pipeline:
```yaml
- name: Validate against k-wave-python
  run: python validate_against_kwave.py --verbose
```

## Next Steps After Validation

Once all tests pass:

1. **Document parity:** Update docs to confirm k-wave-python compatibility
2. **Add examples:** Create example scripts using validated functions
3. **Benchmark performance:** Compare execution times
4. **Expand coverage:** Add more k-wave-python functions as needed

## References

- k-wave-python: https://github.com/waltsims/k-wave-python
- k-wave (MATLAB): http://www.k-wave.org/
- Validation script: `validate_against_kwave.py`
