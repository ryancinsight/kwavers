# Sprint 214 Session 5: GPU Validation & TODO Remediation - Execution Plan

**Date**: 2026-02-03  
**Sprint**: 214  
**Session**: 5  
**Status**: 🔄 IN PROGRESS  
**Lead**: Ryan Clanton PhD (@ryancinsight)  
**Estimated Duration**: 16-20 hours

---

## Executive Summary

### Mission

Complete GPU validation, remediate critical TODOs, and prepare research integration roadmap for the Kwavers ultrasound/optics simulation library. This session focuses on production readiness, zero technical debt, and establishing baselines for future enhancements.

### Current State (Session 4 Complete)

- ✅ **Build**: Clean compilation (0.80s dev, 12.73s release)
- ✅ **Tests**: 1970/1970 passing (100%)
- ✅ **Architecture**: Zero circular dependencies
- ✅ **Code Quality**: Zero dead code, zero deprecated code
- ⚠️ **Technical Debt**: 119 TODO/FIXME/HACK markers in src/ (down from 142 initially)

### Session 5 Objectives

1. **GPU Validation** (P0 - 4 hours) ✅ STARTED
   - Run Burn WGPU benchmarks
   - Validate numerical accuracy
   - Document GPU performance metrics
   - Update performance report

2. **TODO Remediation** (P0 - 8 hours)
   - Triage all 119 markers by severity
   - Fix critical P0 items (10-15 estimated)
   - Document P1/P2 items in backlog
   - Target 50% reduction

3. **Research Integration Planning** (P1 - 4 hours)
   - Create detailed roadmap from RESEARCH_FINDINGS_2025.md
   - Prioritize clinical impact features
   - Estimate implementation effort
   - Document in backlog

4. **Architecture Validation** (P1 - 2 hours)
   - Verify layer boundaries
   - SSOT compliance check
   - Document any violations

---

## Section 1: GPU Validation & Benchmarking

### 1.1 Current Status

**CPU Baseline Established** (Session 4):
- Small beamforming: 18.8 Melem/s (13.6 µs latency)
- Medium beamforming: 6.1 Melem/s (168 µs latency)
- Distance computation: 1.02 Gelem/s (40% of time)
- Interpolation: 1.13 Gelem/s nearest, 658 Melem/s linear

**GPU Tests**: ✅ 11/11 passing (with pinn feature)

### 1.2 Execution Steps

#### Step 1: CPU Baseline Validation ✅ COMPLETE (30 min)

```bash
# Run CPU benchmarks (baseline)
cargo bench --bench gpu_beamforming_benchmark

# Results:
# - beamforming_cpu/cpu_baseline/small: 13.8 µs (18.7 Melem/s) ✅
# - beamforming_cpu/cpu_baseline/medium: 170.3 µs (6.0 Melem/s) ✅
# - distance_computation: 63.3 µs (1.03 Gelem/s) ✅
# - interpolation/nearest: 9.1 µs (1.09 Gelem/s) ✅
# - interpolation/linear: 15.3 µs (654 Melem/s) ✅
```

**Status**: ✅ CPU baseline consistent with Session 4 results

#### Step 2: GPU Test Suite Validation ✅ COMPLETE (30 min)

```bash
# Run GPU unit tests
cargo test --features pinn --lib analysis::signal_processing::beamforming::gpu

# Results: 11/11 tests passing ✅
# - test_burn_beamformer_creation ✅
# - test_array_tensor_conversion ✅
# - test_distance_computation ✅
# - test_apodization ✅
# - test_single_focal_point_beamforming ✅
# - test_multiple_focal_points ✅
# - test_cpu_wrapper ✅
# - test_invalid_input_dimensions ✅
```

**Status**: ✅ All GPU tests pass, no regressions

#### Step 3: GPU Benchmarking (WGPU) 🔄 NEXT (2-3 hours)

**Note**: GPU benchmarks require WGPU backend with actual GPU hardware. Current benchmarks measure CPU baseline only. GPU validation requires:

1. **Hardware Check**:
   ```bash
   # Verify GPU availability
   cargo run --features gpu --example gpu_info
   ```

2. **Burn WGPU Backend Test**:
   ```bash
   # Run with WGPU backend (requires GPU)
   cargo bench --bench gpu_beamforming_benchmark --features pinn-gpu
   ```

3. **Numerical Validation**:
   - Compare CPU vs GPU outputs
   - Tolerance: < 1e-6 absolute error
   - Verify beam patterns match

**Expected GPU Speedup**:
- Small: 5-10× (overhead-limited)
- Medium: 15-30× (compute-bound)
- Large: 30-50× (bandwidth-saturated)

**Fallback**: If GPU unavailable, document requirements and defer to next session with GPU access.

#### Step 4: Performance Report Update (1 hour)

Update `docs/sprints/SPRINT_214_SESSION_4_PERFORMANCE_REPORT.md` with:
- GPU benchmark results (if available)
- Numerical validation results
- Speedup measurements
- Memory usage analysis
- Recommendations for optimization

### 1.3 Deliverables

- [ ] GPU benchmark results (or documented requirements)
- [ ] Numerical validation report (CPU vs GPU)
- [ ] Updated performance report
- [ ] GPU optimization roadmap

---

## Section 2: TODO/FIXME/HACK Remediation

### 2.1 TODO Audit Summary

**Total Markers**: 119 in src/ (excludes benchmarks)

**Breakdown by Layer**:
```
Analysis:        22 markers (18.5%)
Clinical:        31 markers (26.1%)
Physics:         28 markers (23.5%)
Solver:          12 markers (10.1%)
Domain:          11 markers (9.2%)
Infrastructure:  8 markers (6.7%)
Math:            4 markers (3.4%)
Core:            2 markers (1.7%)
GPU:             1 marker (0.8%)
```

### 2.2 Priority Classification

#### P0 - CRITICAL (Immediate Fix Required): ~8-10 items

**Criteria**: Mathematical incorrectness, safety violations, API breaking changes

**Identified P0 Items**:

1. **Physics/Bubble Dynamics - Energy Balance** (P0)
   - File: `src/physics/acoustics/bubble_dynamics/energy_balance.rs:26`
   - Issue: TODO_AUDIT P1 - Complete Energy Balance incomplete
   - Impact: Energy conservation not validated for bubble collapse
   - Action: Implement full thermodynamic energy balance with literature validation

2. **Physics/Bubble Dynamics - Non-Spherical Deformation** (P0)
   - File: `src/physics/acoustics/bubble_dynamics/keller_miksis/equation.rs:6`
   - Issue: TODO_AUDIT P1 - Bubble shape instability missing
   - Impact: Spherical approximation breaks down for violent collapse
   - Action: Add shape mode analysis or document validity range

3. **Physics/Acoustics - Conservation Laws** (P0)
   - File: `src/physics/acoustics/conservation.rs:18`
   - Issue: TODO_AUDIT P1 - Advanced conservation validation incomplete
   - Impact: Cannot verify energy/momentum conservation in multi-physics
   - Action: Implement conservation checking with error bounds

4. **Core/Constants - Temperature Dependence** (P0)
   - File: `src/core/constants/fundamental.rs:6`
   - Issue: TODO_AUDIT P1 - Temperature-dependent constants not implemented
   - Impact: Inaccurate for high-intensity therapy applications
   - Action: Add temperature-dependent sound speed, density, etc.

5. **Physics/Optics - Plasma Kinetics** (P0)
   - File: `src/physics/optics/sonoluminescence/bremsstrahlung.rs:17`
   - Issue: TODO_AUDIT P1 - Simplified ionization fraction
   - Impact: Sonoluminescence spectrum inaccuracy
   - Action: Implement Saha-Boltzmann equilibrium calculation

6. **Domain/Grid - Adaptive Mesh Refinement** (P0)
   - File: `src/domain/grid/structure.rs:49`
   - Issue: TODO_AUDIT P2 - AMR for bubble collapse not implemented
   - Impact: Cannot capture extreme compression ratios (>100:1)
   - Action: Integrate existing AMR module or document limitation

7. **Solver/BEM - Complete Implementation** (P0)
   - File: `src/solver/forward/bem/solver.rs:7`
   - Issue: TODO_AUDIT P1 - BEM solver incomplete
   - Impact: Cannot solve exterior acoustic problems
   - Action: Complete BEM or remove from public API

8. **Clinical/Safety - Missing Implementation** (P0)
   - File: `src/clinical/mod.rs:37`
   - Issue: FIXME - Referenced types not implemented in therapy module
   - Impact: Compilation warning, potential API breakage
   - Action: Complete implementations or remove references

#### P1 - HIGH (Functional Gaps): ~25-30 items

**Criteria**: Missing features, incomplete implementations, optimization opportunities

**Examples**:
- Doppler velocity estimation (clinical need)
- Staircase boundary smoothing (accuracy)
- Ultrasonic localization methods (ULM)
- Mattes MI registration (functional ultrasound)
- Advanced skull aberration correction
- Non-adiabatic bubble thermodynamics
- Complete nonlinear acoustics (shock formation)

**Action**: Document in backlog with effort estimates, prioritize by clinical impact

#### P2 - MEDIUM (Technical Debt): ~40-50 items

**Criteria**: Code improvements, refactoring, documentation

**Examples**:
- Advanced SIMD vectorization
- GPU multi-physics coupling
- Advanced neural beamforming
- Deep learning fusion algorithms
- Production API architecture
- Cloud provider integrations

**Action**: Convert to GitHub issues, defer to future sprints

#### P3 - LOW (Nice-to-Have): ~25-30 items

**Criteria**: Future enhancements, research features

**Examples**:
- Advanced visualization
- Quantum optics framework
- Sonochemistry coupling
- Plasmonics modeling

**Action**: Document for future research, no immediate action

### 2.3 Remediation Strategy

#### Phase 1: Triage (1 hour) ✅ COMPLETE

- [x] Generate full TODO report
- [x] Classify by severity (P0/P1/P2/P3)
- [x] Estimate remediation effort
- [x] Create priority matrix

#### Phase 2: P0 Fixes (6-8 hours) 🔄 IN PROGRESS

**Fix Order** (by impact × effort):

1. **Clinical Safety References** (30 min)
   - File: `src/clinical/mod.rs:37`
   - Fix: Complete or remove FIXME references
   - Test: Verify no warnings

2. **Core Constants Temperature Dependence** (2 hours)
   - File: `src/core/constants/fundamental.rs`
   - Fix: Add temperature-dependent functions
   - Reference: Duck (1990) Physical Properties of Tissues
   - Test: Validate against literature values

3. **Conservation Law Validation** (2 hours)
   - File: `src/physics/acoustics/conservation.rs`
   - Fix: Implement energy/momentum checking
   - Test: Verify conservation in multi-physics coupling

4. **Bubble Energy Balance** (3 hours)
   - File: `src/physics/acoustics/bubble_dynamics/energy_balance.rs`
   - Fix: Complete thermodynamic energy balance
   - Reference: Plesset & Prosperetti (1977)
   - Test: Energy conservation in collapse/rebound

5. **AMR Integration for Bubble Collapse** (3 hours)
   - File: `src/domain/grid/structure.rs`
   - Fix: Link existing AMR to bubble dynamics
   - Test: Convergence study with refinement

**Remaining P0 items**: Document as GitHub issues if >3 hours each

#### Phase 3: P1 Documentation (2 hours)

- Create detailed backlog entries for all P1 items
- Estimate implementation effort
- Prioritize by clinical impact
- Link to research integration roadmap

#### Phase 4: Validation (1 hour)

- Re-run full test suite after each fix
- Verify zero regressions
- Update documentation
- Commit with detailed messages

### 2.4 Success Criteria

- ✅ All P0 items resolved or documented
- ✅ TODO count reduced by ≥50% (119 → <60)
- ✅ Zero new compiler warnings
- ✅ All tests passing (1970/1970)
- ✅ P1/P2 items documented in backlog

---

## Section 3: Research Integration Roadmap

### 3.1 High-Priority Features (P0-P1)

Based on `docs/RESEARCH_FINDINGS_2025.md` analysis:

#### Feature 1: Doppler Velocity Estimation (P1)

**Status**: NOT IMPLEMENTED  
**Priority**: HIGH (essential for vascular imaging)  
**Effort**: 1 week (40 hours)  
**Clinical Impact**: Critical for cardiology, vascular diagnostics

**Implementation**:
```
Location: src/clinical/imaging/doppler/
Modules:
  - autocorrelation.rs: Kasai method (8h)
  - color_doppler.rs: 2D velocity maps (12h)
  - spectral_doppler.rs: Waveform analysis (12h)
  - validation.rs: Flow phantom tests (8h)
```

**Mathematical Specification**:
- Autocorrelation: R(T) = ⟨x(t)·x*(t+T)⟩
- Velocity: v = (c·fs)/(4πf0) · arg(R(T))
- Reference: Kasai et al. (1985), Jensen (1996)

**Tests Required**:
- Analytical: Uniform flow validation
- Property-based: Nyquist limit enforcement
- Negative: Aliasing detection
- Integration: B-mode + Doppler workflow

#### Feature 2: Staircase Boundary Smoothing (P1)

**Status**: NOT IMPLEMENTED  
**Priority**: HIGH (accuracy improvement)  
**Effort**: 2-3 days (16-24 hours)  
**Impact**: Reduces grid artifacts at curved boundaries

**Implementation**:
```
Location: src/domain/boundary/smoothing/
Modules:
  - staircase_reduction.rs: Interface smoothing (8h)
  - sub_grid_interpolation.rs: Fractional positions (6h)
  - validation.rs: Circular phantom tests (4h)
```

**Algorithm**:
1. Detect material interfaces (gradient)
2. Calculate sub-grid intersections
3. Apply fractional weighting
4. Smooth transitions

**Reference**: Treeby & Cox (2010) k-Wave, Section 2.4

#### Feature 3: Automatic Differentiation (P1)

**Status**: PARTIAL (have PINNs, not autodiff through FDTD)  
**Priority**: HIGH (optimization capability)  
**Effort**: 2 weeks (80 hours)  
**Impact**: Enables gradient-based optimization

**Implementation Strategy**:
- Option A: Burn autodiff (memory-intensive)
- Option B: Discrete adjoint method (recommended)
- Target: FDTD acoustic solver first

**Applications**:
- Medium property inversion
- Source optimization
- Aberration correction

#### Feature 4: Enhanced Speckle Modeling (P2)

**Status**: LIMITED  
**Priority**: MEDIUM (clinical realism)  
**Effort**: 3-4 days (24-32 hours)

**Implementation**:
```
Location: src/clinical/imaging/speckle/
Modules:
  - rayleigh_statistics.rs: Statistical model (8h)
  - tissue_dependent.rs: Organ-specific (8h)
  - validation.rs: K-distribution tests (8h)
```

### 3.2 Medium-Priority Features (P2)

1. **Geometric Ray Tracing** (1 week)
   - Fast transcranial aberration approximation
   - 100-1000× faster than full wave

2. **Motion Artifact Simulation** (1 week)
   - Training simulator realism
   - Cardiac/fetal imaging

3. **Enhanced CT Integration** (1-2 weeks)
   - Better transcranial planning
   - Patient-specific skull models

### 3.3 Priority Matrix

| Feature | Clinical | Performance | Effort | Priority | Sprint |
|---------|----------|-------------|--------|----------|--------|
| Doppler | HIGH | Low | 1w | **P1** | 215 |
| Staircase | MEDIUM | HIGH | 3d | **P1** | 215 |
| Autodiff | MEDIUM | HIGH | 2w | **P1** | 216 |
| Speckle | HIGH | Low | 4d | **P2** | 216 |
| Ray Trace | MEDIUM | HIGH | 1w | **P2** | 217 |
| Motion | LOW | Low | 1w | **P3** | 218+ |
| CT | LOW | Low | 2w | **P3** | 218+ |

### 3.4 Deliverable

- [ ] Detailed roadmap document
- [ ] Backlog entries for each feature
- [ ] Effort estimates with breakdown
- [ ] Mathematical specifications
- [ ] Test requirements

---

## Section 4: Architecture Validation

### 4.1 Layer Dependency Audit

**Method**: Verify unidirectional dependencies

```bash
# Generate dependency graph
cargo depgraph --workspace-only > docs/architecture/deps.dot

# Check for cycles (should be zero)
cargo depgraph --workspace-only | grep -i cycle

# Verify layer boundaries
# Expected: L9 → L8 → L7 → ... → L1 (no upward deps)
```

**Expected Result**: ✅ Zero cycles (Session 4 resolved circular deps)

### 4.2 SSOT Verification

**Critical SSOT Locations**:
- ✅ Field indices: `domain/fields/acoustic_field.rs`
- ✅ Grid definition: `domain/grid/mod.rs`
- ✅ Medium properties: `domain/medium/`
- ✅ Source definitions: `domain/source/`
- ✅ Beamforming: `analysis/signal_processing/beamforming/`

**Verification**:
- [ ] No duplicate definitions
- [ ] All re-exports reference canonical location
- [ ] Documentation states SSOT

### 4.3 Cross-Contamination Check

**Forbidden**:
- ❌ Clinical code in Physics layer
- ❌ Solver code in Domain layer
- ❌ Implementation in API traits

**Audit**:
```bash
# Check for physics in clinical
grep -r "use crate::physics" src/clinical/ | grep -v "// OK"

# Check for solver in domain
grep -r "use crate::solver" src/domain/ | grep -v "// OK"
```

**Expected**: Zero violations

---

## Section 5: Timeline & Milestones

### Day 1 (8 hours) - 2026-02-03

**Morning (0-4h)**:
- [x] 0-1h: Session 5 planning and audit document ✅
- [x] 1-2h: CPU benchmark validation ✅
- [x] 2-3h: GPU test suite validation ✅
- [ ] 3-4h: TODO triage and P0 identification

**Afternoon (4-8h)**:
- [ ] 4-5h: Clinical safety FIXME resolution (P0)
- [ ] 5-7h: Core constants temperature dependence (P0)
- [ ] 7-8h: Conservation law validation (P0)

### Day 2 (8 hours) - 2026-02-04

**Morning (0-4h)**:
- [ ] 0-3h: Bubble energy balance implementation (P0)
- [ ] 3-4h: Testing and validation

**Afternoon (4-8h)**:
- [ ] 4-7h: AMR integration for bubble collapse (P0)
- [ ] 7-8h: P1 item documentation

### Day 3 (4 hours) - 2026-02-05

**Morning (0-4h)**:
- [ ] 0-2h: Research integration roadmap
- [ ] 2-3h: Architecture validation
- [ ] 3-4h: Session 5 summary document

**Closure**:
- [ ] Final test run (1970/1970)
- [ ] Update backlog.md and checklist.md
- [ ] Git commit with comprehensive summary

---

## Section 6: Success Criteria

### Must Achieve (Hard Requirements)

- ✅ CPU benchmarks validated (consistent with Session 4)
- ✅ GPU test suite passing (11/11 tests)
- [ ] All P0 TODOs resolved or documented as issues
- [ ] TODO count reduced ≥50% (119 → <60)
- [ ] Zero compiler warnings
- [ ] 1970/1970 tests passing
- [ ] Session 5 fully documented

### Should Achieve (Soft Goals)

- [ ] GPU benchmarks run on actual hardware
- [ ] Numerical equivalence validated (CPU vs GPU)
- [ ] Research roadmap complete
- [ ] Architecture validation report
- [ ] All P1 items documented in backlog

### Nice to Have (Stretch Goals)

- [ ] GPU speedup measurements (15-30× target)
- [ ] First custom WGSL kernel implemented
- [ ] Doppler velocity estimation started
- [ ] Performance regression CI integration

---

## Section 7: Risk Assessment & Mitigation

### Risk 1: GPU Hardware Unavailable

**Probability**: MEDIUM  
**Impact**: MEDIUM (defers GPU validation)  
**Mitigation**: Document GPU requirements, use CPU tests  
**Fallback**: Complete CPU validation, defer GPU to Session 6

### Risk 2: P0 Fixes Take Longer Than Estimated

**Probability**: HIGH  
**Impact**: MEDIUM (delays completion)  
**Mitigation**: Focus on top 3-4 P0 items only  
**Fallback**: Convert long fixes to GitHub issues

### Risk 3: Testing Reveals Regressions

**Probability**: LOW  
**Impact**: HIGH (quality issue)  
**Mitigation**: Test after each change, revert if needed  
**Fallback**: Document regressions, create separate fix sprint

### Risk 4: TODO Count Doesn't Reduce by 50%

**Probability**: MEDIUM  
**Impact**: LOW (cosmetic issue)  
**Mitigation**: Focus on quality of fixes, not quantity  
**Fallback**: Adjust target to "all P0 fixed" instead

---

## Section 8: Deliverables

### Documentation

1. ✅ **Session 5 Plan** (this document)
2. ✅ **Session 5 Audit** (SPRINT_214_SESSION_5_AUDIT.md)
3. [ ] **TODO Triage Report** (SPRINT_214_SESSION_5_TODO_TRIAGE.md)
4. [ ] **Research Integration Roadmap** (detailed backlog entries)
5. [ ] **Session 5 Summary** (SPRINT_214_SESSION_5_SUMMARY.md)

### Code Changes

1. [ ] P0 TODO fixes (8-10 items)
2. [ ] Debug implementation for BurnPinnBeamformingAdapter ✅
3. [ ] Temperature-dependent constants
4. [ ] Conservation law validation
5. [ ] Bubble energy balance

### Backlog Updates

1. [ ] Update backlog.md with P1/P2 items
2. [ ] Update checklist.md with Session 5 status
3. [ ] Create GitHub issues for deferred work
4. [ ] Update gap_audit.md with resolutions

---

## Section 9: References

### Prior Work

- Sprint 214 Session 4 Summary
- Sprint 214 Session 4 Performance Report
- Sprint 213 Executive Summary
- Research Findings 2025

### External References

**Ultrasound Simulation**:
- k-Wave: https://github.com/ucl-bug/k-wave
- jwave: https://github.com/ucl-bug/jwave
- fullwave25: https://github.com/pinton-lab/fullwave25

**Scientific Literature**:
- Kasai et al. (1985): Doppler autocorrelation
- Treeby & Cox (2010): k-Wave PSTD method
- Duck (1990): Physical Properties of Tissues
- Plesset & Prosperetti (1977): Bubble dynamics

---

## Section 10: Next Steps (Post-Session 5)

### Sprint 215 (2 weeks)

1. **Doppler Velocity Estimation** (1 week)
   - Implement Kasai autocorrelation
   - Color Doppler visualization
   - Spectral waveform analysis

2. **Staircase Boundary Smoothing** (3 days)
   - Interface detection
   - Sub-grid interpolation
   - Circular phantom validation

3. **Advanced GPU Optimizations** (4 days)
   - Custom WGSL kernels
   - Memory coalescing
   - Multi-GPU support

### Sprint 216+ (Long-term)

1. **Automatic Differentiation** (2 weeks)
2. **Enhanced Speckle Modeling** (4 days)
3. **Production Deployment** (2-4 weeks)

---

**Document Status**: 🔄 IN PROGRESS  
**Next Update**: After P0 TODO fixes complete  
**Owner**: Ryan Clanton PhD  
**Session Start**: 2026-02-03 10:00 UTC

---

*This plan guides Sprint 214 Session 5 execution. All work should reference this document and update status as completed.*