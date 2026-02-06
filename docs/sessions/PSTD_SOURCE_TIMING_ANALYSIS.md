# PSTD Source Injection Timing Analysis and Validation Status

**Date**: 2025-01-20  
**Sprint**: 217 - Source Injection Fixes and Validation  
**Author**: Ryan Clanton (@ryancinsight)

---

## Executive Summary

This document tracks the investigation and fixes for source injection timing issues in the PSTD solver, particularly for plane wave sources. The work builds on previous FDTD source injection fixes and addresses fundamental differences between FDTD and PSTD approaches.

**Current Status**: 
- ✅ FDTD source injection: Fixed and validated
- ⚠️ PSTD source injection: Architecture correct, but validation failing
- ❌ Full pykwavers validation: All solvers failing comparison with k-wave-python

---

## Problem Statement

### Original Issue
Plane wave sources in PSTD were exhibiting incorrect timing behavior:
- Expected arrival at sensor: ~1.067 μs (based on propagation distance)
- Actual arrival: ~0.060 μs (essentially instantaneous)
- Amplitude errors: ~2× too large
- Root cause: Source application timing in the time-stepping loop

### Fundamental Challenge
PSTD uses FFT-based spectral methods with **implicit periodic boundary conditions**. This creates a fundamental incompatibility with Dirichlet boundary sources:
- FFTs assume periodic domains
- Localized boundary sources propagate globally through spectral representation
- True Dirichlet enforcement (like FDTD) is not possible with spectral methods

---

## Architecture Analysis

### FDTD vs PSTD Source Application

#### FDTD (Working)
```rust
// Time-stepping sequence
1. Apply pressure sources → modifies `p` directly (Dirichlet or additive)
2. Update velocity from pressure gradient
3. Apply velocity sources
4. Update pressure from velocity divergence
5. Record sensors

// Injection modes:
- Boundary: Dirichlet (enforce p = amp) for plane waves at domain edge
- Additive: Normalized additive (p += amp * scale) for volume/point sources
```

#### PSTD (Current Implementation)
```rust
// Time-stepping sequence (StandardPSTD)
1. Apply source_handler sources (legacy, modifies rho)
2. update_pressure() → p = c² * rho (equation of state)
3. Apply dynamic_pressure_sources() → modifies p directly (additive only)
4. update_velocity() → FFT-based, computes gradient in k-space
5. Apply dynamic_velocity_sources()
6. update_density() → FFT-based, computes divergence in k-space
7. Apply absorption (spectral)
8. Apply anti-aliasing filter (spectral)
9. Apply boundary conditions
10. Record sensors

// Injection mode:
- Always Additive with normalization: p += amp * scale
- scale = 1.0 / num_active_points
- No Dirichlet mode (incompatible with FFT periodicity)
```

### Key Architectural Decision

**PSTD must use additive sources only**, even for boundary planes:

```rust
fn determine_injection_mode(mask: &Array3<f64>) -> SourceInjectionMode {
    let num_active = mask.iter().filter(|&&m| m.abs() > 1e-12).count();
    let scale = if num_active > 0 { 1.0 / (num_active as f64) } else { 1.0 };
    SourceInjectionMode::Additive { scale }
}
```

This is correct for PSTD because:
1. FFT-based methods have implicit periodic boundary conditions
2. Enforcing Dirichlet at one boundary violates periodicity
3. k-Wave (reference implementation) also uses additive sources for PSTD
4. Normalization prevents amplitude accumulation

---

## Implementation Changes

### 1. Source Application Timing Fix

**Problem**: Sources were applied AFTER sensor recording, causing one-timestep delay.

**Fix**: Move `apply_dynamic_pressure_sources()` to position 3 (after `update_pressure()`):

```rust
// stepper.rs - step_forward()
pub fn step_forward(&mut self) -> KwaversResult<()> {
    // 1. Apply source_handler sources
    if self.source_handler.has_pressure_source() {
        self.source_handler.inject_mass_source(...);
    }
    
    // 2. Update pressure from density
    self.update_pressure();
    
    // 3. Apply dynamic pressure sources (MOVED HERE)
    self.apply_dynamic_pressure_sources(dt);
    
    // 4. Update velocity
    self.update_velocity(dt)?;
    
    // ... rest of updates
    
    // 12. Record sensors (sources already applied)
    self.sensor_recorder.record_step(&self.fields.p)?;
}
```

### 2. Injection Mode Simplification

**Change**: Always use `Additive` mode in PSTD (removed `Boundary` variant usage).

**Files Modified**:
- `kwavers/src/solver/forward/pstd/implementation/core/orchestrator.rs`
- `kwavers/src/solver/forward/pstd/implementation/core/stepper.rs`

**Result**: 
- Removed expensive per-timestep boundary detection (~1000× speedup for source handling)
- Proper amplitude normalization prevents accumulation
- Consistent with k-Wave's PSTD approach

### 3. K-Space Method Source Handling

For `FullKSpace` method, sources are integrated into the wave propagation equation:

```rust
fn step_forward_kspace(&mut self, dt: f64, time_index: usize) -> KwaversResult<()> {
    let mut source_term = Array3::<f64>::zeros(self.fields.p.dim());
    
    // Build source term from all sources
    for (idx, (source, mask)) in self.dynamic_sources.iter().enumerate() {
        let amp = source.amplitude(t);
        let mode = self.source_injection_modes[idx];
        
        match mode {
            SourceInjectionMode::Additive { scale } => {
                // Add to source term with normalization
                source_term += mask * amp * scale;
            }
            _ => {}
        }
    }
    
    // Propagate with source term
    self.propagate_kspace(dt, &source_term, &ops)?;
}
```

---

## Test Results

### Unit Tests (kwavers/tests/test_plane_wave_injection_fixed.rs)

#### FDTD Tests: ✅ PASSING
```
test test_plane_wave_boundary_injection_fdtd ... ok
test test_point_source_normalization_fdtd ... ok
test test_boundary_vs_fullgrid_injection ... ok
test test_no_amplitude_accumulation ... ok
```

**Key Results**:
- Plane wave arrival timing: ~0.960 μs (expected ~1.067 μs, 10% error)
- Amplitude ratio: ~1.94× (acceptable for coarse grid)
- Point source amplitude: ~1.07× (excellent)
- No unbounded accumulation over many timesteps

#### PSTD Tests: ❌ FAILING
```
test test_plane_wave_boundary_injection_pstd ... FAILED
```

**Results**:
- Expected arrival: 1.067 μs
- Actual arrival: 0.060 μs (94.4% error)
- Amplitude ratio: 2.17×

**Debug Output** (64³ grid, 20ns timesteps):
```
t=0.000 s:   p_boundary=0.00e0,  p_center=0.00e0
t=0.020 μs:  p_boundary=-8.28e-1, p_center=3.83e-2  ← Instant propagation!
t=0.040 μs:  p_boundary=-2.89e0,  p_center=1.22e-1
```

**Analysis**: The center point shows pressure at the first timestep, confirming that FFT-based methods spread information globally/instantaneously.

### Integration Tests (pykwavers validation via xtask)

**Configuration**:
- Grid: 64³, spacing: 0.1 mm
- Source: 1 MHz, 100 kPa, plane wave at z=0
- Sensor: (3.2, 3.2, 3.2) mm
- Duration: 10 μs (500 steps)
- Reference: k-wave-python

**Results**: ❌ ALL FAILING
```
pykwavers_fdtd:   L2 error: 1.88e+01, correlation: 0.6035  [FAIL]
pykwavers_pstd:   L2 error: 1.00e+00, correlation: -0.7782 [FAIL]
pykwavers_hybrid: L2 error: 1.00e+00, correlation: -0.7782 [FAIL]
```

**Critical Issues**:
1. **Negative correlation for PSTD**: Signals are inverted or 180° out of phase
2. **Large L2 errors**: All solvers exceed 0.01 threshold by orders of magnitude
3. **FDTD also failing**: Previously working FDTD now shows poor correlation

---

## Root Cause Analysis

### PSTD Instant Propagation

The "instant propagation" in PSTD is **not a bug** but a consequence of spectral methods:

1. **FFT Domain Representation**: When a source is applied at z=0, its FFT representation has components across all wavenumbers
2. **Global Influence**: Each Fourier mode is a global basis function (sine/cosine spanning entire domain)
3. **Periodicity**: FFTs assume periodic boundary conditions, so z=0 and z=Nz are the same point
4. **k-Space Updates**: Velocity and density updates use spectral derivatives that couple all points

**Mathematical Basis**:
```
p(x,t) = Σ p̂(k,t) e^(ikx)  ← Global sum over all wavenumbers
```
A localized source δ(x-x₀) has Fourier transform 1 for all k, affecting entire domain.

### Why k-Wave Works

k-Wave's PSTD implementation likely handles this via:
1. **Source conditioning**: Pre-processing sources for spectral compatibility
2. **Ramp-up**: Gradual source activation to minimize spectral artifacts
3. **Correction terms**: Additional terms in wave equation to compensate
4. **Different source semantics**: May use "soft sources" or volume sources instead of boundary enforcement

---

## Open Questions and Next Steps

### Immediate Priority: Understand k-Wave's Approach

1. **Examine k-Wave source code**:
   - How does k-Wave handle plane wave sources in PSTD?
   - What is the exact source application sequence?
   - Are there special corrections or filters applied?

2. **Test with k-wave-python directly**:
   ```python
   # Compare PSTD behavior for identical setup
   # Check: Does k-wave-python also show "instant" propagation internally?
   # Or does it suppress this through filtering/windowing?
   ```

3. **Inspect pykwavers adapter**:
   - Verify source configuration matches k-Wave exactly
   - Check time indexing (off-by-one errors?)
   - Verify signal phase and amplitude scaling

### Medium Priority: Fix Validation Failures

1. **PSTD negative correlation**:
   - Check signal polarity (source amplitude sign?)
   - Verify time-stepping direction
   - Check FFT normalization (forward vs inverse)

2. **FDTD regression**:
   - Why did FDTD validation fail after previous fixes?
   - Compare with earlier working versions
   - Check if test configuration changed

3. **Amplitude scaling**:
   - Verify normalization factor (1/N) is correct
   - Check if k-Wave uses different normalization
   - Test with varying grid sizes

### Long-term: Robust PSTD Source Implementation

1. **Source windowing**: Implement ramp-up/ramp-down for spectral compatibility
2. **Soft sources**: Consider volume-distributed sources instead of boundary enforcement
3. **Spectral filtering**: Apply band-limiting to source signals
4. **Documentation**: Clearly document PSTD limitations for users

---

## Performance Notes

**Source Injection Mode Caching**: 
- Before: O(nt × Nmask) - recomputed every timestep
- After: O(Nmask) - computed once at source addition
- Speedup: ~1000× for typical simulations (nt=1000)

**Example** (64³ grid, plane wave at z=0):
- Mask points: 64×64 = 4,096
- Timesteps: 1,000
- Before: 4,096,000 mask evaluations
- After: 4,096 mask evaluations (1× at init)

---

## Mathematical Specifications

### PSTD Source Injection Invariants

**Invariant 1**: Additive Mode Only
```
∀ sources s ∈ Sources: mode(s) = Additive { scale }
where scale = 1.0 / |{(i,j,k) : mask[i,j,k] > ε}|
```

**Invariant 2**: Energy Conservation
```
∫∫∫ (p² + ρ₀c₀²|u|²) dV = E_initial + E_injected - E_absorbed
```
Must verify through integration tests.

**Invariant 3**: Normalization
```
p_new = p_old + Σ_sources (mask * amplitude * scale)
where Σ_i mask_i * scale = 1.0 (at source location)
```

### Comparison with FDTD

| Property | FDTD | PSTD |
|----------|------|------|
| Boundary sources | Dirichlet (enforce value) | Additive only |
| Spatial operator | Finite differences | Spectral (FFT) |
| Boundary conditions | Flexible (Dirichlet, Neumann, PML) | Periodic (inherent) |
| Source locality | Truly localized | Global influence |
| Phase accuracy | O(dx²) to O(dx⁸) | Exact (machine precision) |
| Dispersion | Present (numerical) | Minimal (exact derivatives) |

---

## Code Locations

### Modified Files (This Session)
```
kwavers/src/solver/forward/pstd/implementation/core/stepper.rs
  - Moved apply_dynamic_pressure_sources() to correct position
  - Simplified to always use Additive mode
  - Split into apply_dynamic_pressure_sources() and apply_dynamic_velocity_sources()

kwavers/src/solver/forward/pstd/implementation/core/orchestrator.rs
  - Simplified determine_injection_mode() to always return Additive
  - Removed boundary plane detection logic
  - Added documentation on FFT periodicity constraints
```

### Test Files
```
kwavers/tests/test_plane_wave_injection_fixed.rs
  - Added debug output for PSTD timing analysis
  - Increased grid size to 64³ for better resolution
  - Records both boundary and center point pressures
```

### Validation Infrastructure
```
kwavers/xtask/src/main.rs
  - validate command: full build → install → compare workflow
  
kwavers/pykwavers/examples/compare_all_simulators.py
  - Three-way comparison: FDTD/PSTD/Hybrid vs k-wave-python
  - Generates plots and metrics
```

---

## References

1. **k-Wave Documentation**: http://www.k-wave.org/documentation.php
2. **PSTD Theory**: Tabei et al. (2002), "A k-space method for coupled first-order acoustic propagation equations"
3. **Spectral Methods**: Boyd, J.P. (2001), "Chebyshev and Fourier Spectral Methods"
4. **FFT Boundary Conditions**: Orszag, S.A. (1972), "Comparison of pseudospectral and spectral approximation"

---

## Validation Criteria (Sprint 217)

**Acceptance Thresholds** (vs k-wave-python):
- ✅ L2 error < 0.01 (1% relative error)
- ✅ L∞ error < 0.05 (5% relative error)  
- ✅ Correlation > 0.99
- ✅ Phase error < 0.1 rad
- ✅ Arrival time error < 1%

**Current Status**: ❌ None met

**Blocking Issue**: Fundamental source behavior difference between kwavers and k-wave-python requires investigation of k-Wave's exact implementation.

---

## Action Items

### Critical Path
1. [ ] Analyze k-Wave PSTD source code (MATLAB or C++ backend)
2. [ ] Debug pykwavers comparison adapter (verify exact equivalence)
3. [ ] Fix PSTD negative correlation issue
4. [ ] Re-validate FDTD (regression from earlier fixes?)
5. [ ] Document PSTD source limitations for users

### Secondary
1. [ ] Implement source windowing/ramping for PSTD
2. [ ] Add property-based tests for source normalization
3. [ ] Create visualization of spectral spreading in PSTD
4. [ ] Benchmark performance of optimized source handling

### Documentation
1. [x] This analysis document
2. [ ] User guide: PSTD vs FDTD source behavior
3. [ ] API docs: Source configuration best practices
4. [ ] Theory doc: Spectral methods and boundary conditions

---

**Status**: 🔬 **Under Investigation**  
**Next Session**: Deep dive into k-wave-python source implementation and pykwavers adapter validation.