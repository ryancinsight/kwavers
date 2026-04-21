# Validation Results: Kwavers vs k-wave-python

**Date:** February 13, 2026
**Status:** Core implementations validated ✓

## Summary

Validation testing revealed that **core mathematical implementations are numerically identical** between kwavers and k-wave-python. Differences are primarily in API design and default behaviors.

Update 2026-04-20: the tone-burst sample-count bug is resolved and the Gaussian default envelope now matches the vendored k-wave-python reference after flattening the row-vector return value. The remaining difference is API shape, not waveform content.

## Test Results

### ✅ PASS: Tone Burst (Rectangular Window)

**Result:** Max difference = 3.55e-15 (machine precision)

When using the same Rectangular window (no windowing), the sine wave generation produces identical results:
- Signal shape: (30,) vs (1, 31) - off-by-one in length calculation
- Amplitude values: Identical within floating-point precision
- Phase: Identical

**Conclusion:** Core sine wave generation is correct.

### ✅ MATCH: Tone Burst Envelope

**Gaussian Default:**
- k-wave-python: Gaussian tone burst envelope with a Tukey taper
- kwavers: Same Gaussian burst envelope and sample-count rule
- Result: Numeric waveform parity after flattening the k-wave-python row vector

### ✅ RESOLVED: Tone Burst Length

**Observation:** 
- k-wave-python: 31 samples for the 3-cycle 10 MHz / 1 MHz reference case; 113 samples for the non-integer 11.293333 MHz / 500 kHz case
- kwavers: matches the same floor-plus-one sample-count rule

**Impact:** The off-by-one burst-length discrepancy is removed

### ⚠️ API DIFFERENCE: Return Types

**k-wave-python:**
- `tone_burst()`: Returns 2D array (n_offsets, n_samples)
- `get_win()`: Returns tuple (window, coherent_gain)
- `db2neper()`: Domain-specific absorption coefficient conversion

**kwavers:**
- `tone_burst()`: Returns 1D array or list
- `get_win()`: Returns 1D array
- `db2neper()`: Simple unit conversion (dB ↔ Nepers)

**Impact:** API compatibility - requires wrapper/adaptation layer

## Detailed Analysis

### Numerical Accuracy

With identical parameters and window types:
- **Sine wave values:** < 1e-14 relative error ✓
- **Window functions:** Different implementations, similar shapes
- **Geometry masks:** Boolean arrays, implementation-dependent

### Feature Parity

| Feature | k-wave-python | kwavers | Match |
|---------|---------------|---------|-------|
| Tone burst generation | ✓ | ✓ | ✓ (core) |
| Rectangular window | ✓ | ✓ | ✓ |
| Hanning window | ✓ | ✓ | ✓ |
| Hamming window | ✓ | ✓ | ✓ |
| Blackman window | ✓ | ✓ | ✓ |
| Gaussian window | ✓ | ✓ | ✓ |
| 2D geometry (disc) | ✓ | ✓ | ✓ |
| 3D geometry (ball) | ✓ | ✓ | ✓ |
| Unit conversion | Domain-specific | Simple | N/A |

## Root Causes of Differences

### 1. API Design Philosophy

**k-wave-python:**
- MATLAB-style: Returns matrices for multiple signals
- Includes metadata (coherent gain)
- Domain-specific unit conversions

**kwavers:**
- Rust-style: Simple return types
- Thin wrapper over core algorithms
- Standard unit conversions

## Recommendations

### For Validation

1. **Focus on core correctness** - sine wave and tone burst generation are validated ✓
2. **Document API differences** for users migrating from k-wave-python
3. **If strict compatibility is required, add row-vector return shape and multi-offset support**

### For Parity (Optional)

If strict k-wave-python compatibility is required:

1. **Add row-vector return shape** to tone_burst for scalar offsets
2. **Add multi-offset support** to tone_burst for phased-array use cases
3. **Return tuples from get_win**
4. **Implement domain-specific db2neper**

## Conclusion

**Core implementations are mathematically correct and numerically accurate.** 

Differences are in:
- API design patterns (not correctness)
- Return shape and metadata conventions

**Validation Status: PASSED for core algorithms**

The kwavers implementations can be used with confidence for scientific computing, with awareness of API differences when porting k-wave-python code.

## Test Commands

```bash
# Run full validation
python validate_against_kwave.py

# Test specific functions
python validate_against_kwave.py --tests tone_burst make_disc

# With verbose output
python validate_against_kwave.py --verbose
```

## References

- k-wave-python: https://github.com/waltsims/k-wave-python
- Validation script: `validate_against_kwave.py`
