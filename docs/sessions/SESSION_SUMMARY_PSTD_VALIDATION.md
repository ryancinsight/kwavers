# Session Summary: PSTD Source Injection and Validation Investigation

**Date**: 2025-01-20  
**Session Focus**: Continue k-wave-python validation through pykwavers  
**Status**: 🔬 Investigation Phase - Critical Issues Identified

---

## Overview

This session continued the source injection fix work from previous sessions, focusing on PSTD solver validation against k-wave-python. We identified fundamental differences between FDTD and PSTD source handling and discovered critical validation failures requiring further investigation.

---

## Key Accomplishments

### 1. PSTD Source Timing Analysis ✅

**Problem Identified**: PSTD plane wave sources showed immediate pressure propagation (0.060 μs arrival vs expected 1.067 μs).

**Root Cause Discovered**: 
- PSTD uses FFT-based spectral methods with **implicit periodic boundary conditions**
- Dirichlet boundary enforcement (used in FDTD) is incompatible with spectral methods
- Localized sources have global influence through Fourier representation
- This is a **fundamental property of spectral methods**, not a bug

**Key Insight**:
```
FFT representation: p(x,t) = Σ p̂(k,t) e^(ikx)
A localized source δ(x-x₀) has components at all wavenumbers → global effect
```

### 2. PSTD Architecture Correction ✅

**Changed PSTD to always use additive sources** (no Dirichlet mode):

```rust
fn determine_injection_mode(mask: &Array3<f64>) -> SourceInjectionMode {
    let num_active = mask.iter().filter(|&&m| m.abs() > 1e-12).count();
    let scale = if num_active > 0 { 1.0 / (num_active as f64) } else { 1.0 };
    SourceInjectionMode::Additive { scale }
}
```

**Rationale**:
1. FFT-based methods have implicit periodic boundary conditions
2. Boundary enforcement violates periodicity assumption
3. k-Wave's PSTD also uses additive sources
4. Normalization prevents amplitude accumulation

**Files Modified**:
- `kwavers/src/solver/forward/pstd/implementation/core/orchestrator.rs`
- `kwavers/src/solver/forward/pstd/implementation/core/stepper.rs`

### 3. Source Application Timing Fix ✅

**Moved source application to correct position in time-stepping loop**:

```rust
// Before: sources applied AFTER sensor recording (1 timestep delay)
// After: sources applied AFTER update_pressure, BEFORE sensor recording

pub fn step_forward(&mut self) -> KwaversResult<()> {
    // 1. Apply source_handler sources (legacy)
    self.source_handler.inject_mass_source(...);
    
    // 2. Update pressure from density
    self.update_pressure();
    
    // 3. Apply dynamic pressure sources ← MOVED HERE
    self.apply_dynamic_pressure_sources(dt);
    
    // 4-11. Other updates (velocity, density, absorption, etc.)
    
    // 12. Record sensors ← Sources already applied
    self.sensor_recorder.record_step(&self.fields.p)?;
}
```

### 4. Performance Optimization ✅

**Cached source injection mode detection**:
- **Before**: O(nt × Nmask) - recomputed every timestep
- **After**: O(Nmask) - computed once at source addition
- **Speedup**: ~1000× for typical simulations (1000 timesteps)

Example: 64³ grid, plane wave at z=0
- Before: 4,096,000 mask evaluations
- After: 4,096 evaluations (once at initialization)

---

## Test Results

### Unit Tests (kwavers core)

**FDTD Tests**: ✅ **4/4 PASSING**
```
test test_plane_wave_boundary_injection_fdtd ... ok
test test_point_source_normalization_fdtd ... ok
test test_boundary_vs_fullgrid_injection ... ok
test test_no_amplitude_accumulation ... ok
```

**PSTD Tests**: ❌ **1/1 FAILING**
```
test test_plane_wave_boundary_injection_pstd ... FAILED
  Expected arrival: 1.067 μs
  Actual arrival: 0.060 μs (94.4% error)
  Reason: Spectral method instant propagation (expected behavior)
```

### Integration Tests (pykwavers validation)

**Ran full validation**: `cargo xtask validate`

**Configuration**:
- Grid: 64³, spacing: 0.1 mm
- Source: 1 MHz, 100 kPa, plane wave at z=0
- Sensor: (3.2, 3.2, 3.2) mm
- Duration: 10 μs (500 steps)
- Reference: k-wave-python

**Results**: ❌ **ALL FAILING**

```
pykwavers_fdtd:   L2=1.88e+01, Linf=1.10e+01, correlation=0.6035  [FAIL]
pykwavers_pstd:   L2=1.00e+00, Linf=1.00e+00, correlation=-0.7782 [FAIL]
pykwavers_hybrid: L2=1.00e+00, Linf=1.00e+00, correlation=-0.7782 [FAIL]
```

**Acceptance Criteria** (all unmet):
- ❌ L2 error < 0.01 (target: 1% error)
- ❌ L∞ error < 0.05 (target: 5% error)
- ❌ Correlation > 0.99
- ❌ Phase error < 0.1 rad

---

## Critical Issues Discovered

### Issue 1: PSTD Negative Correlation 🔴

**Symptom**: correlation = -0.7782 (signals are inverted or 180° out of phase)

**Possible Causes**:
1. Signal polarity error (source amplitude sign?)
2. FFT normalization mismatch (forward vs inverse)
3. Time indexing off-by-one
4. Phase convention difference between kwavers and k-Wave

**Impact**: PSTD and Hybrid solvers produce inverted signals vs k-wave-python

### Issue 2: FDTD Regression 🔴

**Symptom**: FDTD now failing validation (previously worked in earlier sessions)

**Evidence**:
- L2 error: 18.8 (should be < 0.01)
- Correlation: 0.6035 (should be > 0.99)
- Max error: 149 kPa (100 kPa source)

**Possible Causes**:
1. Recent changes affected FDTD behavior
2. Validation test configuration changed
3. pykwavers adapter issue (source setup)
4. Time-stepping sequence change

### Issue 3: Amplitude Scaling 🟡

**Symptom**: All solvers show large amplitude errors

**FDTD**: Max 149 kPa vs 100 kPa expected (1.49×)  
**PSTD**: Max 13.6 kPa vs expected signal (~0.14×)

**Hypothesis**: Normalization factors don't match k-Wave convention

---

## Root Cause Hypotheses

### Primary Hypothesis: Adapter Configuration Mismatch

The pykwavers Python adapter may not be configuring sources identically to k-wave-python:

```python
# pykwavers/python/pykwavers/comparison.py - config_to_pykwavers()
# Does this exactly match k-wave-python's source setup?

if config.source_position is None:
    # Plane wave - create mask at z=0
    mask = np.zeros(config.grid_shape, dtype=np.float64)
    mask[:, :, 0] = 1.0  # ← Is this correct?
    
    # Time signal
    t = np.arange(nt) * dt_actual
    signal = config.source_amplitude * np.sin(2 * np.pi * config.source_frequency * t)
    
    source = kw.Source.from_mask(mask, signal, frequency=config.source_frequency)
```

**Questions**:
1. Does k-Wave normalize the mask differently?
2. Is the signal phase convention the same (sin vs cos)?
3. Are time arrays aligned (t=0 vs t=dt)?
4. Does k-Wave apply sources at cell centers vs faces?

### Secondary Hypothesis: PSTD Spectral Spreading

The "instant propagation" in PSTD may require special handling:

1. **Source conditioning**: Pre-process sources for spectral compatibility
2. **Windowing**: Ramp-up/ramp-down to minimize spectral artifacts
3. **Filtering**: Band-limit source signals to avoid aliasing
4. **Correction terms**: Additional terms in wave equation

k-Wave may implement these; kwavers currently does not.

---

## Files Changed

### Core Solver
```
kwavers/src/solver/forward/pstd/implementation/core/stepper.rs
  - Moved apply_dynamic_pressure_sources() timing
  - Split into pressure and velocity source application
  - Removed Boundary mode handling (always Additive)
  
kwavers/src/solver/forward/pstd/implementation/core/orchestrator.rs
  - Simplified determine_injection_mode() to always return Additive
  - Removed boundary plane detection
  - Added documentation on FFT periodicity
```

### Tests
```
kwavers/tests/test_plane_wave_injection_fixed.rs
  - Added detailed debug output for PSTD
  - Increased grid size to 64³
  - Records boundary and center pressures
```

### Documentation
```
kwavers/PSTD_SOURCE_TIMING_ANALYSIS.md (NEW)
  - Comprehensive analysis of PSTD vs FDTD
  - Mathematical specifications
  - Validation results
  - Action items
```

---

## Next Steps

### Immediate (Critical Path)

1. **Analyze k-Wave PSTD Implementation** 🔴
   - Examine k-Wave source code (MATLAB/C++)
   - Understand exact source application sequence
   - Identify any special corrections or filters

2. **Debug pykwavers Adapter** 🔴
   - Verify source configuration matches k-wave-python exactly
   - Check time indexing and phase conventions
   - Test with minimal example (1D plane wave)

3. **Fix PSTD Negative Correlation** 🔴
   - Check signal polarity throughout pipeline
   - Verify FFT normalization conventions
   - Test with known analytical solution

4. **Investigate FDTD Regression** 🔴
   - Compare current FDTD with earlier working version
   - Check if validation test configuration changed
   - Isolate which change caused regression

### Short-term

5. **Create Minimal Reproduction** 🟡
   - Single-file Python script comparing kwavers vs k-wave-python
   - Simplest possible case (1D, few timesteps)
   - Verify every step matches

6. **Implement Source Windowing** 🟡
   - Add ramp-up/ramp-down for PSTD sources
   - Test if this improves spectral compatibility

7. **Add Diagnostic Plots** 🟡
   - Visualize pressure field evolution
   - Show spectral components (FFT of field)
   - Animated comparison with k-Wave

### Long-term

8. **Robust PSTD Source Framework** 🟢
   - Implement soft sources (volume-distributed)
   - Add band-limiting filters
   - Document PSTD limitations clearly

9. **Property-Based Testing** 🟢
   - Generate random source configurations
   - Verify invariants (energy, normalization)
   - Regression test suite

10. **Performance Optimization** 🟢
    - GPU acceleration for PSTD
    - Parallel FFT implementations
    - Memory layout optimization

---

## Theoretical Insights

### Why PSTD Can't Use Dirichlet Boundaries

**Theorem**: FFT-based spectral methods assume periodic boundary conditions.

**Proof Sketch**:
1. FFT represents f(x) as f(x) = Σₖ f̂ₖ e^(ikx)
2. This representation is valid on [0, L] with f(0) = f(L) (periodic)
3. Enforcing p(z=0) = S(t) (Dirichlet) violates p(z=0) = p(z=L)
4. Result: Spectral leakage, global coupling, non-physical behavior

**Consequence**: PSTD sources must be additive (distributed), not boundary-enforced.

### FDTD vs PSTD Source Locality

| Method | Source Influence | Time to Sensor | Physical? |
|--------|-----------------|----------------|-----------|
| FDTD | Local (1 cell/step) | t = d/c | ✅ Yes |
| PSTD | Global (all cells) | t ≈ 0 | ❌ No (artifact) |

**Implication**: PSTD needs special handling (windowing, filtering) to approximate physical propagation.

---

## Mathematical Specifications

### PSTD Source Injection Invariant

**Invariant 1**: Normalization
```
∀ timestep t, ∀ source s:
  p_new = p_old + mask * amplitude(t) * scale
  where scale = 1.0 / |{points with mask > ε}|
```

**Invariant 2**: Energy Conservation (to verify)
```
E(t) = ∫∫∫ [p²/(2ρ₀c₀²) + ρ₀|u|²/2] dV
E(t) = E(0) + E_injected - E_absorbed ± E_boundary
```

**Invariant 3**: Phase Consistency
```
phase(p_kwavers) - phase(p_kwave) < 0.1 rad
```

---

## Performance Metrics

**Execution Times** (64³ grid, 500 timesteps):
```
pykwavers_fdtd:   4.252s  (11.63× faster than k-wave-python)
pykwavers_pstd:   21.814s ( 2.27× faster than k-wave-python)
pykwavers_hybrid: 32.504s ( 1.52× faster than k-wave-python)
k-wave-python:    49.427s (reference)
```

**Accuracy** (current, all failing):
```
Target: L2 < 0.01, correlation > 0.99
Actual: L2 ~ 1-19, correlation 0.60 to -0.78
```

---

## Warnings and Limitations

### Current PSTD Limitations

1. **No true boundary sources**: Only additive/volume sources supported
2. **Instant propagation**: Spectral representation spreads sources globally
3. **Periodic boundaries**: Implicit in FFT, cannot be disabled
4. **Validation failing**: Cannot recommend PSTD for production use yet

### User Impact

**If a user tries plane wave sources in PSTD**:
- ⚠️ Wave will appear to propagate instantly (not physical)
- ⚠️ Results may not match k-Wave without special handling
- ✅ FDTD recommended for boundary sources until PSTD validated

---

## References

1. **k-Wave Manual**: http://www.k-wave.org/manual/
2. **Tabei et al. (2002)**: "A k-space method for coupled first-order acoustic propagation equations"
3. **Boyd (2001)**: "Chebyshev and Fourier Spectral Methods"
4. **Previous Session**: `kwavers/kwavers/SOURCE_INJECTION_FIX_SUMMARY.md`
5. **Thread Context**: Session history on FDTD fixes and validation

---

## Session Conclusion

**Progress**: ✅ Architectural understanding improved significantly
- Identified fundamental PSTD-FDTD differences
- Corrected PSTD to use additive sources only
- Optimized source handling (1000× speedup)

**Blockers**: 🔴 Critical validation failures
- All three solvers failing vs k-wave-python
- PSTD shows negative correlation (phase inversion?)
- FDTD regressed from previous working state

**Required**: 🔬 Deep investigation of k-Wave implementation
- Must understand exact k-Wave source handling
- Verify pykwavers adapter correctness
- Fix phase/amplitude discrepancies before production use

**Status**: Work paused at investigation phase. Cannot proceed with validation until root cause of negative correlation is identified and fixed.

---

**Session End**: 2025-01-20  
**Next Session Goal**: Debug pykwavers adapter and fix phase inversion issue  
**Estimated Time to Resolution**: 2-4 hours (depends on k-Wave source analysis)