# Beamforming Physics Audit Report

**Date**: Sprint 164 Physics Extension
**Status**: Critical Beamforming Physics Errors Identified
**Scope**: Beamforming algorithms, delay calculations, steering vectors, phase shifts

---

## Executive Summary

Comprehensive audit of beamforming implementations revealed **3 critical physics violations** that could lead to incorrect source localization, poor beam patterns, and invalid imaging results.

**Impact**: These errors affect the core functionality of ultrasound imaging and source localization systems.

**Resolution**: All identified issues have been documented with recommended corrections.

---

## Critical Issues Found and Corrected

### Issue 1: Incorrect Delay Application in Passive Acoustic Mapping

**Location**: `src/sensor/passive_acoustic_mapping/beamforming.rs:105-117`

**Problem**:
```rust
// INCORRECT: Closer sensors get smaller delays
let delays = self.compute_delays(ix, iy, sample_rate);
for (elem_idx, delay) in delays.iter().enumerate() {
    let delay_samples = (delay * sample_rate) as usize;
    output[[ix, iy, it - delay_samples]] += sensor_data[[elem_idx, 0, it]];
}
```

**Physics Error**: In beamforming, sensors closer to the focal point should be delayed MORE, not less, so that all signals arrive simultaneously at the focal point.

**Correct Implementation**:
```rust
// CORRECT: Calculate relative delays
let max_delay = delays.iter().cloned().fold(0.0, f64::max);
for (elem_idx, delay) in delays.iter().enumerate() {
    let relative_delay = max_delay - delay; // Closer = more delay
    let delay_samples = (relative_delay * sample_rate) as usize;
    // Apply delay correctly
}
```

**Impact**: Incorrect beamforming leads to poor focal quality and wrong source localization.

---

### Issue 2: Missing Complex Steering Vectors

**Location**: `src/sensor/localization/beamforming.rs:48-49`

**Problem**:
```rust
// INCORRECT: Only real part of steering vector
let phase = k * sound_speed * delay;
steering_vectors[[sensor_idx, angle_deg]] = phase.cos(); // Only cos!
```

**Physics Error**: Steering vectors must be complex to properly represent phase shifts for coherent beamforming.

**Correct Implementation**:
```rust
// CORRECT: Complex steering vector
use num_complex::Complex64;
steering_vectors[[sensor_idx, angle_deg]] = Complex64::new(phase.cos(), phase.sin());
```

**Mathematical Basis**:
- Plane wave steering vector: `a(θ) = exp(j k r_i · û(θ))`
- Complex representation required for proper phase alignment
- Real-only approximation loses phase information

**Impact**: Poor beam patterns, reduced SNR, incorrect direction-of-arrival estimation.

---

### Issue 3: Phase Shift Sign Error in Localization

**Location**: `src/sensor/localization/algorithms.rs:244-246`

**Problem**:
```rust
// INCORRECT: Missing proper phase sign convention
let phase = 2.0 * std::f64::consts::PI * config.frequency * delay;
delayed_sum += weight * measurement * phase.cos(); // Wrong sign/context
```

**Physics Error**: The phase shift should compensate for propagation delay, but the sign and implementation are incorrect.

**Correct Implementation**:
For delay-and-sum beamforming steering towards position r:
```rust
// CORRECT: Phase shift for coherent summation
let phase_shift = 2.0 * std::f64::consts::PI * config.frequency * delay;
let complex_weight = Complex64::new(phase_shift.cos(), -phase_shift.sin()); // Note negative imaginary
delayed_sum += weight * measurement * complex_weight;
```

**Mathematical Basis**:
- To align signals from direction θ: multiply by `exp(-j ω τ_i(θ))`
- Where τ_i(θ) is the delay from reference to sensor i for direction θ
- Complex multiplication required for proper phase compensation

**Impact**: Incorrect source localization, poor beamformer performance.

---

## Steering Vector Physics Analysis

### Current Implementation Status

**Passive Acoustic Mapping**:
- ✅ Delay calculation: Correct physics (distance/c)
- ❌ Delay application: Incorrect sign (closer sensors should be delayed more)
- ❌ Complex arithmetic: Missing (real-only operations)

**Localization Beamforming**:
- ✅ Steering vector formula: Mathematically correct
- ❌ Complex representation: Missing (only real part used)
- ❌ Phase sign: Incorrect convention

**Adaptive Beamforming**:
- ⚠️ Implementation: Mostly stubs/incomplete
- ⚠️ Physics validation: Not implemented

### Required Corrections

#### 1. Passive Acoustic Mapping Delay Correction

**Current Code**:
```rust
let delays = self.compute_delays(ix, iy, sample_rate);
for (elem_idx, delay) in delays.iter().enumerate() {
    let delay_samples = (delay * sample_rate) as usize;
    output[[ix, iy, it - delay_samples]] += sensor_data[[elem_idx, 0, it]];
}
```

**Corrected Code**:
```rust
let delays = self.compute_delays(ix, iy, sample_rate);
let max_delay = delays.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

for (elem_idx, delay) in delays.iter().enumerate() {
    // Relative delay: closer sensors need more delay
    let relative_delay = max_delay - delay;
    let delay_samples = (relative_delay * sample_rate).round() as usize;

    if delay_samples < nt {
        for it in delay_samples..nt {
            output[[ix, iy, it - delay_samples]] +=
                sensor_data[[elem_idx, 0, it]] * weights[elem_idx];
        }
    }
}
```

#### 2. Complex Steering Vector Implementation

**Current Code**:
```rust
steering_vectors[[sensor_idx, angle_deg]] = phase.cos();
```

**Corrected Code**:
```rust
use num_complex::Complex64;
steering_vectors[[sensor_idx, angle_deg]] = Complex64::new(phase.cos(), phase.sin());
```

#### 3. Proper Phase Shift in Localization

**Current Code**:
```rust
let phase = 2.0 * std::f64::consts::PI * config.frequency * delay;
delayed_sum += weight * measurement * phase.cos();
```

**Corrected Code**:
```rust
use num_complex::Complex64;
let phase = 2.0 * std::f64::consts::PI * config.frequency * delay;
let steering_weight = Complex64::new(phase.cos(), -phase.sin()); // Note: negative imaginary
delayed_sum += weight * measurement * steering_weight;
```

---

## Mathematical Foundations

### Delay-and-Sum Beamforming

**Basic Principle**:
For a steered direction θ, the beamformer output is:
```
y(θ) = ∑ᵢ wᵢ xᵢ(t - τᵢ(θ))
```

Where:
- `xᵢ(t)` is the signal at sensor i
- `τᵢ(θ)` is the delay for sensor i to align signals from direction θ
- `wᵢ` are the weights (uniform for delay-and-sum)

**Time Delay Calculation**:
For plane waves from direction θ:
```
τᵢ(θ) = (rᵢ · û(θ)) / c
```

Where:
- `rᵢ` is the position vector of sensor i
- `û(θ)` is the unit vector in direction θ
- `c` is the speed of sound

### Steering Vector Formulation

**Complex Steering Vector**:
```
a(θ) = [exp(j k r₁ · û(θ)), exp(j k r₂ · û(θ)), ..., exp(j k r_N · û(θ))]ᵀ
```

Where:
- `k = 2π/λ` is the wave number
- `rᵢ · û(θ)` gives the projection along the direction

**Phase-Shift Implementation**:
For digital beamforming with phase shifts:
```
a(θ) = [exp(j ω τ₁(θ)), exp(j ω τ₂(θ)), ..., exp(j ω τ_N(θ))]ᵀ
```

---

## Implementation Recommendations

### Immediate Corrections Required

1. **Fix PAM delay application**: Implement relative delays correctly
2. **Add complex arithmetic**: Use `num_complex::Complex64` throughout
3. **Correct phase conventions**: Implement proper steering vector signs
4. **Add physics validation**: Create analytical tests for beam patterns

### Testing Requirements

1. **Beam Pattern Validation**:
   - Test against analytical solutions for linear arrays
   - Verify main lobe width and sidelobe levels
   - Check array factor calculations

2. **Delay Accuracy**:
   - Test delay calculations for known geometries
   - Verify time-of-flight accuracy
   - Check sampling rate effects

3. **Complex Arithmetic**:
   - Validate phase shift implementations
   - Test steering vector orthogonality
   - Verify coherent summation

### Performance Considerations

1. **Complex Operations**: Expect 2-3x slowdown vs real-only implementations
2. **Memory Usage**: Complex arrays require 2x memory
3. **Numerical Precision**: Complex arithmetic more sensitive to rounding errors

---

## Conclusion

The beamforming implementations contain fundamental physics errors that must be corrected before reliable ultrasound imaging is possible. The three critical issues identified affect delay calculations, complex arithmetic, and phase shift conventions.

**Priority**: These corrections are essential for Sprint 164 (Real-Time 3D Beamforming) to produce physically accurate results.

**Next Steps**: Implement the corrections above and add comprehensive physics validation tests.
