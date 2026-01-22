# Benchmark Stub Remediation Plan - Sprint 209 Phase 2
**Date**: 2025-01-14  
**Sprint**: 209 Phase 2  
**Status**: EXECUTION READY  
**Priority**: P0 (Correctness > Functionality)

---

## Executive Summary

Phase 6 TODO audit identified 35+ benchmark stub implementations measuring placeholder operations instead of real physics. Per Dev rules: "Absolute Prohibition: TODOs, stubs, dummy data, zero-filled placeholders" and "Cleanliness: Never create deprecated code; immediately remove obsolete code."

**Decision**: Remove benchmark stubs immediately (Option A) - prevents misleading performance data and optimization waste on incorrect baselines.

**Rationale**:
- **Correctness > Functionality**: Placeholder benchmarks produce invalid performance data
- **No Potemkin Villages**: Benchmarks that don't measure real physics violate architectural purity
- **Zero Tolerance for Error Masking**: Stub benchmarks mask the fact that implementations don't exist
- **Cleanliness**: Remove obsolete/misleading code immediately

---

## Remediation Strategy

### Phase 2A: Remove Performance Benchmark Stubs (1.5 hours)

**File**: `benches/performance_benchmark.rs`

**Action**: Remove or disable 18 stub helper methods and their calling benchmarks

**Stubs to Remove**:
1. `update_velocity_fdtd()` (L913-925) - Empty stub, no staggered grid update
2. `update_pressure_fdtd()` (L934-946) - Empty stub, no wave equation
3. `update_westervelt()` (L955-969) - Empty stub, no nonlinear term
4. `simulate_fft_operations()` (L971-976) - No FFT calls
5. `simulate_angular_spectrum_propagation()` (L977-982) - No angular spectrum
6. `simulate_elastic_wave_step()` (L983-993) - No elastic equation
7. `simulate_displacement_tracking()` (L994-999) - No tracking logic
8. `simulate_stiffness_estimation()` (L1000-1006) - Returns clone (invalid)
9. `simulate_microbubble_scattering()` (L1007-1012) - No Rayleigh-Plesset
10. `simulate_tissue_perfusion()` (L1013-1018) - No perfusion model
11. `simulate_perfusion_analysis()` (L1019-1025) - Returns clone (invalid)
12. `simulate_transducer_element()` (L1026-1037) - No Rayleigh integral
13. `simulate_skull_transmission()` (L1038-1043) - No aberration physics
14. `simulate_thermal_monitoring()` (L1044-1049) - No CEM43 calculation
15. `compute_uncertainty_statistics()` (L1050-1057) - Returns zeros (invalid)
16. `compute_ensemble_mean()` (L1058-1064) - Returns zeros (invalid)
17. `compute_ensemble_variance()` (L1065-1075) - Returns zeros (invalid)
18. `compute_conformity_score()` (L1076-1082) - Returns 0.0 (invalid)
19. `compute_prediction_interval()` (L1083-1089) - Returns clones (invalid)

**Dependent Benchmarks to Disable**:
- `benchmark_fdtd_wave()` - Uses stubs #1, #2
- `benchmark_pstd_wave()` - Uses stub #4
- `benchmark_has_wave()` - Uses stub #5
- `benchmark_westervelt_wave()` - Uses stubs #1, #2, #3
- `benchmark_swe()` - Uses stubs #6, #7, #8
- `benchmark_ceus()` - Uses stubs #9, #10, #11
- `benchmark_transcranial_fus()` - Uses stubs #12, #13, #14
- `benchmark_uncertainty_quantification()` - Uses stubs #15-19

**Replacement Strategy**:
```rust
/// BENCHMARK DISABLED - Awaiting Real Implementation
/// 
/// This benchmark was removed because it measured placeholder operations
/// instead of real physics. See backlog.md Sprint 211-212 for implementation plan.
/// 
/// Required implementations:
/// - FDTD solver with staggered grid velocity update (12-16h)
/// - Pressure update with wave equation (8-12h)
/// - CFL stability enforcement (2-3h)
/// 
/// Related: TODO_AUDIT_PHASE6_SUMMARY.md Section 1.1
/// Tracking: backlog.md "Benchmark Implementations (P2)"
#[allow(dead_code)]
fn benchmark_fdtd_wave_DISABLED() {
    // Benchmark removed - awaiting real FDTD implementation
    // See: src/solver/fdtd/ for production FDTD solver
    // This benchmark will be re-enabled when stubs are replaced with actual physics
}
```

### Phase 2B: Remove Comparative Solver Benchmark Stubs (0.5 hours)

**File**: `benches/comparative_solver_benchmark.rs`

**Stubs to Remove**:
1. `calculate_acoustic_energy()` (L165-166) - "SIMPLIFIED BENCHMARK METRIC"
2. Other simplified metrics identified in audit

**Action**: Add TODO references, disable benchmarks calling simplified metrics

### Phase 2C: Remove PINN Benchmark Stubs (0.5 hours)

**File**: `benches/pinn_performance_benchmarks.rs`

**Stubs to Remove**:
1. `run_pinn_training_benchmark()` - Placeholder training loop
2. `benchmark_memory_usage()` - Stub memory tracking
3. `run_adaptive_sampling_benchmark()` - Uses placeholder adaptive sampling
4. `benchmark_pde_kernel()` - Stub PDE residual

**Action**: Disable until GPU PINN infrastructure ready (Sprint 212+)

### Phase 2D: Update Criterion Benchmark Registrations (0.5 hours)

**Files**: All `benches/*.rs` files with criterion_group! macros

**Action**: Comment out benchmark functions that use stubs

**Example**:
```rust
criterion_group!(
    benches,
    // DISABLED - Awaiting real implementation (Sprint 211-212)
    // benchmark_fdtd_wave,
    // benchmark_pstd_wave,
    // benchmark_westervelt_wave,
    
    // ENABLED - Production benchmarks with real implementations
    benchmark_grid_operations,
    benchmark_memory_allocation,
);
```

---

## Implementation Checklist

### Phase 2A: performance_benchmark.rs
- [ ] Add comprehensive module-level documentation explaining removal
- [ ] Rename stub methods to `*_DISABLED()` with documentation
- [ ] Comment out benchmark functions using stubs
- [ ] Update criterion_group! to exclude disabled benchmarks
- [ ] Add TODO references linking to backlog Sprint 211-212
- [ ] Verify `cargo bench --benches` runs without stub benchmarks

### Phase 2B: comparative_solver_benchmark.rs
- [ ] Review simplified metrics
- [ ] Add TODO documentation
- [ ] Disable benchmarks with invalid metrics
- [ ] Verify compilation

### Phase 2C: pinn_performance_benchmarks.rs
- [ ] Disable PINN training benchmarks
- [ ] Add Sprint 212+ TODO references
- [ ] Verify compilation

### Phase 2D: Criterion Registration
- [ ] Update all criterion_group! macros
- [ ] Verify `cargo bench --list` shows only valid benchmarks
- [ ] Test sample benchmark run

### Phase 2E: Documentation Updates
- [ ] Update `docs/benchmarks.md` (if exists) with removal rationale
- [ ] Create `BENCHMARK_IMPLEMENTATION_ROADMAP.md` with physics requirements
- [ ] Update `backlog.md` with detailed implementation tasks
- [ ] Update `checklist.md` with Phase 2 completion

---

## Validation Plan

### Compilation Tests
```bash
# Verify all benchmarks compile
cargo check --benches

# List available benchmarks (should exclude disabled ones)
cargo bench --benches --list

# Run remaining production benchmarks
cargo bench --benches -- --test
```

### Expected Results
- **Before**: 35+ benchmarks, many measuring placeholder operations
- **After**: ~5-10 benchmarks measuring real implementations (grid ops, memory, etc.)
- **Build**: No errors, only warnings (if any)
- **Performance Data**: Only valid baselines from real implementations

---

## Backlog Updates

### Sprint 211-212: Implement Benchmark Physics (High Priority - P1/P2)

**Total Effort**: 189-263 hours (staged implementation)

#### Stage 1: Core Wave Propagation (Sprint 211 - 45-60h)
1. **FDTD Benchmarks** (20-28h)
   - Implement `update_velocity_fdtd()` with staggered grid
   - Implement `update_pressure_fdtd()` with wave equation
   - Add CFL stability enforcement
   - Validation: Compare against analytical solutions

2. **PSTD Benchmarks** (15-20h)
   - Implement `simulate_fft_operations()` using rustfft
   - Add k-space operator application
   - Validation: Spectral accuracy test

3. **Nonlinear Benchmarks** (10-12h)
   - Implement `update_westervelt()` with β/ρc⁴ nonlinear term
   - Add shock-capturing scheme
   - Validation: Fubini solution comparison

#### Stage 2: Advanced Physics (Sprint 212 - 60-80h)
4. **Elastography Benchmarks** (24-32h)
   - Implement `simulate_elastic_wave_step()` (12-16h)
   - Implement `simulate_displacement_tracking()` (6-8h)
   - Implement `simulate_stiffness_estimation()` inverse solver (6-8h)
   - Validation: Phantom data comparison

5. **CEUS Benchmarks** (16-22h)
   - Implement `simulate_microbubble_scattering()` with R-P dynamics (8-12h)
   - Implement `simulate_tissue_perfusion()` (4-6h)
   - Implement `simulate_perfusion_analysis()` (4-4h)
   - Validation: Clinical reference data

6. **Therapy Benchmarks** (20-26h)
   - Implement `simulate_transducer_element()` Rayleigh integral (8-10h)
   - Implement `simulate_skull_transmission()` aberration (8-12h)
   - Implement `simulate_thermal_monitoring()` CEM43 (4-4h)
   - Validation: FDA reference phantoms

#### Stage 3: Uncertainty Quantification (Sprint 213 - 64-103h)
7. **UQ Benchmarks** (44-63h)
   - Implement `compute_uncertainty_statistics()` (8-12h)
   - Implement `compute_ensemble_mean()` (4-6h)
   - Implement `compute_ensemble_variance()` (6-8h)
   - Implement `compute_conformity_score()` (10-14h)
   - Implement `compute_prediction_interval()` (16-23h)
   - Validation: Monte Carlo ground truth

8. **PINN Benchmarks** (20-40h)
   - Implement `run_pinn_training_benchmark()` with GPU (8-16h)
   - Implement `benchmark_memory_usage()` tracking (4-6h)
   - Implement `run_adaptive_sampling_benchmark()` (6-12h)
   - Implement `benchmark_pde_kernel()` residual (2-6h)
   - Validation: Training convergence plots

---

## Success Criteria

### Immediate (Sprint 209 Phase 2)
- ✅ All benchmark stubs removed or disabled
- ✅ No misleading performance data generated
- ✅ Comprehensive TODO/backlog references added
- ✅ `cargo bench` runs only valid benchmarks
- ✅ Documentation updated (backlog.md, checklist.md)

### Medium-term (Sprint 211-212)
- ⏳ Stage 1 benchmarks implemented (FDTD, PSTD, Westervelt)
- ⏳ Physics validated against analytical solutions
- ⏳ Performance baselines established

### Long-term (Sprint 213+)
- ⏳ All advanced physics benchmarks implemented
- ⏳ Full benchmark suite with real physics
- ⏳ Continuous performance monitoring

---

## Risk Assessment

### Risks Mitigated by Removal
- ❌ **Misleading Performance Data**: Eliminated invalid baselines
- ❌ **Optimization Waste**: Prevents optimizing placeholder code
- ❌ **Architectural Debt**: Removes Potemkin village benchmarks
- ❌ **Credibility Risk**: No false performance claims

### Risks Introduced by Removal
- ⚠️ **Reduced Benchmark Coverage**: Temporary gap until implementations ready
- ⚠️ **Performance Regression Detection**: Cannot detect regressions in unimplemented areas
- ✅ **Mitigation**: Document expected performance characteristics in backlog

### Risk Acceptance
- **Accept**: Temporary benchmark gap (correctness > coverage)
- **Reject**: Keeping invalid benchmarks (violates Dev rules)

---

## References

- **Audit Report**: `TODO_AUDIT_PHASE6_SUMMARY.md` Section 1.1-1.4
- **Backlog**: `backlog.md` Sprint 211-213 (Benchmark Implementation)
- **Dev Rules**: `prompt.yaml` (Correctness > Functionality, No Placeholders)
- **Sprint Context**: `SPRINT_209_PHASE1_COMPLETE.md` (Previous phase results)

---

## Timeline

| Phase | Task | Duration | Completion |
|-------|------|----------|------------|
| 2A | Remove performance_benchmark.rs stubs | 1.5h | Pending |
| 2B | Remove comparative_solver_benchmark.rs stubs | 0.5h | Pending |
| 2C | Remove pinn_performance_benchmarks.rs stubs | 0.5h | Pending |
| 2D | Update criterion registrations | 0.5h | Pending |
| 2E | Documentation updates | 1.0h | Pending |
| **Total** | **Sprint 209 Phase 2** | **4.0h** | **Pending** |

**Estimated Completion**: 2025-01-14 (same day as Phase 1)

---

## Appendix A: Dev Rules Compliance

### Principles Applied
1. ✅ **Correctness > Functionality**: Remove invalid benchmarks rather than keep misleading ones
2. ✅ **Absolute Prohibition**: Eliminate stubs, dummy data, and zero-filled placeholders
3. ✅ **Cleanliness**: Immediately remove obsolete code (benchmark stubs)
4. ✅ **Transparency**: Document removal rationale and future implementation plan
5. ✅ **No Error Masking**: Expose missing implementations instead of hiding behind stubs

### Architectural Purity
- **No Potemkin Villages**: Removed facade benchmarks with no real physics
- **Explicit Invariants**: Document physics requirements for future implementations
- **Single Source of Truth**: Benchmark results reflect only real solver performance

---

## Appendix B: Future Implementation Requirements

### Mathematical Specifications (Examples)

#### FDTD Velocity Update
```
Specification:
  v^(n+1/2) = v^(n-1/2) - (dt/ρ) * ∇p^n

Discretization:
  vx[i,j,k]^(n+1/2) = vx[i,j,k]^(n-1/2) - (dt/ρ) * (p[i+1,j,k] - p[i,j,k])/dx
  vy[i,j,k]^(n+1/2) = vy[i,j,k]^(n-1/2) - (dt/ρ) * (p[i,j+1,k] - p[i,j,k])/dy
  vz[i,j,k]^(n+1/2) = vz[i,j,k]^(n-1/2) - (dt/ρ) * (p[i,j,k+1] - p[i,j,k])/dz

Validation:
  - Plane wave test: p(x,t) = A sin(kx - ωt), v(x,t) = (A/ρc) sin(kx - ωt)
  - Energy conservation: E_total = ∫(p²/2ρc² + ρ|v|²/2)dV = const (lossless)
  - CFL stability: dt ≤ dx/(c√3) for 3D
```

#### Westervelt Nonlinear Term
```
Specification:
  ∂²p/∂t² = c²∇²p + (β/ρc⁴)∂²(p²)/∂t² + (δ/ρc²)∂³p/∂t³

Nonlinear Term:
  N = (β/ρc⁴)∂²(p²)/∂t²
  
Discretization:
  ∂²(p²)/∂t² ≈ (p²[n+1] - 2p²[n] + p²[n-1])/dt²

Validation:
  - Fubini solution: p(x,σ) = p₀ sin(σ)/(1 + Γσ cos(σ))
    where σ = ωt - kx, Γ = (βkp₀)/(2ρc³) (shock formation distance)
```

---

**End of Remediation Plan**