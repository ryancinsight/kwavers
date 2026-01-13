# Sprint 208: Deprecated Code Elimination & Large File Refactoring

**Sprint Duration**: 2025-01-13 to 2025-01-27 (2 weeks)  
**Sprint Goal**: Eliminate all deprecated code, resolve critical TODOs, and refactor large files to achieve zero technical debt  
**Status**: 🔄 IN PROGRESS - Phase 1  

---

## Executive Summary

Sprint 208 continues the comprehensive cleanup initiative started in Sprint 207, focusing on:

1. **Deprecated Code Elimination** (P0): Remove all 20+ deprecated items and update consumers
2. **Critical TODO Resolution** (P0): Implement or remove all P0 TODO items
3. **Large File Refactoring** (P1): Refactor remaining 6 files >900 lines
4. **Test File Refactoring** (P1): Modularize 3 large test files >1200 lines

**Core Principle**: Zero tolerance for technical debt. No deprecated code, no placeholder TODOs, no incomplete implementations.

---

## Phase 1: Deprecated Code Elimination (Week 1) 🔄 IN PROGRESS

### 1.1 CPMLBoundary Method Deprecations 🔴 CRITICAL

**Location**: `domain/boundary/cpml/mod.rs`

**Items to Remove**:
1. `update_acoustic_memory()` - Lines 92-96
   - Deprecated since 3.0.0
   - Replacement: `update_and_apply_gradient_correction`
   - Consumers: Search and update all call sites

2. `apply_gradient_correction()` - Lines 103-107
   - Deprecated since 3.0.0
   - Replacement: `update_and_apply_gradient_correction`
   - Consumers: Search and update all call sites

3. `recreate()` - Lines 160-170
   - Deprecated since 3.1.0
   - Replacement: `.clone()`
   - Consumers: Search and update all call sites

**Action Plan**:
- [ ] Search for all call sites of deprecated methods
- [ ] Update consumers to use replacement APIs
- [ ] Remove deprecated method implementations
- [ ] Run full test suite to verify changes
- [ ] Update any documentation referencing old methods

---

### 1.2 Legacy BoundaryCondition Trait 🔴 CRITICAL

**Location**: `domain/boundary/traits.rs` - Lines 484-487

**Deprecated**: Since 2.15.0  
**Replacement**: New BoundaryCondition trait system

**Action Plan**:
- [ ] Identify all implementations of legacy trait
- [ ] Verify new trait system coverage
- [ ] Remove legacy trait definition
- [ ] Update any ADRs or documentation
- [ ] Verify compilation with `cargo check`

---

### 1.3 Legacy Beamforming Locations 🔴 CRITICAL

**Multiple deprecated re-exports in `domain::sensor::beamforming`**:

1. **MUSIC Algorithm** - `domain/sensor/beamforming/adaptive/algorithms/music.rs:38-42`
   - Deprecated since 0.2.0
   - New location: `analysis::signal_processing::beamforming::adaptive::music`

2. **MVDR Algorithm** - `domain/sensor/beamforming/adaptive/algorithms/mvdr.rs:38-42`
   - Deprecated since 0.2.0
   - New location: `analysis::signal_processing::beamforming::adaptive::mvdr`

3. **Adaptive Beamforming** - `domain/sensor/beamforming/adaptive/mod.rs:92-95, 102-105`
   - MUSIC deprecated since 2.14.0
   - EigenspaceMV deprecated since 2.14.0

4. **Time Domain DAS** - `domain/sensor/beamforming/time_domain/das/mod.rs:45-50, 53-56`
   - Deprecated since 0.2.0
   - New location: `analysis::signal_processing::beamforming::time_domain`

5. **Delay Reference** - `domain/sensor/beamforming/time_domain/delay_reference.rs:42-46`
   - Deprecated since 0.2.0
   - New location: `analysis::signal_processing::beamforming::time_domain::delay_reference`

6. **Time Domain Module** - `domain/sensor/beamforming/time_domain/mod.rs:64-67, 70-73`
   - Deprecated since 0.2.0
   - New location: `analysis::signal_processing::beamforming::time_domain`

7. **BeamformingProcessor** - `domain/sensor/beamforming/processor.rs:230-234`
   - Capon method deprecated since 2.14.0
   - Renamed to `mvdr_unsteered_weights_time_series`

**Action Plan**:
- [ ] Audit all imports across codebase
- [ ] Update imports to use new locations
- [ ] Remove deprecated re-export modules
- [ ] Update migration guide with location mapping
- [ ] Run full test suite
- [ ] Update examples to use new paths

---

### 1.4 Sensor Localization Deprecation 🔴 CRITICAL

**Location**: `domain/sensor/localization/mod.rs:22-25`

**Deprecated Item**: Legacy DOA Beamformer  
**Replacement**: `kwavers::sensor::localization::BeamformSearch` + `LocalizationBeamformSearchConfig`

**Action Plan**:
- [ ] Identify all uses of legacy DOA Beamformer
- [ ] Migrate to new BeamformSearch API
- [ ] Remove legacy implementation
- [ ] Update tests and examples
- [ ] Verify functionality preservation

---

### 1.5 ARFI Radiation Force Deprecations 🔴 CRITICAL

**Location**: `physics/acoustics/imaging/modalities/elastography/radiation_force.rs`

**Items to Remove**:
1. Deprecated displacement method - Lines 250-254
   - Should use `push_pulse_body_force` instead
   - ARFI should be modeled as body-force source term

2. Deprecated multi-directional displacement - Lines 312-316
   - Should use `multi_directional_body_forces` instead
   - Same body-force modeling principle

**Action Plan**:
- [ ] Search for all ARFI method call sites
- [ ] Update to use body-force modeling
- [ ] Remove deprecated displacement methods
- [ ] Verify elastography tests pass
- [ ] Update clinical workflow examples

---

### 1.6 Axisymmetric Medium Deprecations 🔴 CRITICAL

**Location**: `solver/forward/axisymmetric/config.rs`

**Deprecated Items**:
1. `AxisymmetricMedium` struct - Lines 145-149
   - Deprecated since 2.16.0
   - Replacement: `domain::medium::Medium` types with `CylindricalMediumProjection`

2. `homogeneous()` constructor - Lines 164-174
   - Deprecated since 2.16.0
   - Replacement: `HomogeneousMedium::new` with `CylindricalMediumProjection`

3. Other medium constructors - Lines 187-191
   - Deprecated since 2.16.0
   - Replacement: Domain-level medium types

**Action Plan**:
- [ ] Identify all AxisymmetricMedium usage
- [ ] Migrate to domain-level Medium with projections
- [ ] Remove deprecated struct and constructors
- [ ] Update axisymmetric solver tests
- [ ] Verify convergence test compatibility
- [ ] Update ADR-015 (Axisymmetric coordinate system)

---

## Phase 2: Critical TODO Resolution (Week 1) 🔄 PLANNED

### 2.1 Extract Focal Properties Implementation 🔴 P0

**Location**: `analysis/ml/pinn/adapters/source.rs:151-155`

**Current State**: Returns `None` with TODO comment

**Required Implementation**:
```rust
fn extract_focal_properties(_source: &dyn Source) -> Option<FocalProperties> {
    // TODO: Once domain sources expose focal properties, extract them here
    None
}
```

**Action Plan**:
- [ ] Design focal properties trait or interface
- [ ] Extend `domain::source::Source` trait with focal methods
- [ ] Implement focal property extraction for:
  - Focused transducers
  - Piston sources with focus
  - Phased arrays
- [ ] Update PINN adapter to use real extraction
- [ ] Add tests for focal property extraction
- [ ] Remove TODO comment

**Mathematical Specification**:
- Focal depth: Distance from transducer to focal point
- Focal spot size: -6dB beamwidth at focus
- Focal gain: Pressure amplitude ratio (focus/source)
- Focal steering: Angle from transducer normal

---

### 2.2 SIMD Quantization Bug Fix 🔴 P0

**Location**: `analysis/ml/pinn/burn_wave_equation_2d/inference/backend/simd.rs:134-138`

**Bug Description**:
```rust
// If I invoke `matmul_simd_quantized` for hidden, and loop 0..3, 
// it only processes first 3 neurons of previous layer.
// This looks like a BUG in the original code or limitation.
// I will preserve the logic as is for now, but add a comment/TODO.
```

**Action Plan**:
- [ ] Analyze SIMD matmul implementation
- [ ] Determine root cause of neuron limitation
- [ ] Options:
  1. Fix SIMD implementation to handle all neurons
  2. Document limitation and add runtime checks
  3. Remove SIMD path if unfixable
- [ ] Add comprehensive tests for SIMD quantization
- [ ] Benchmark fixed vs original implementation
- [ ] Remove TODO/bug comment once resolved

---

### 2.3 Microbubble Dynamics Implementation 🔴 P0

**Location**: `clinical/therapy/therapy_integration/orchestrator/microbubble.rs:61-71`

**Current State**: Stub implementation with TODO

**Required Implementation**:
Full Rayleigh-Plesset equation solver for microbubble dynamics:
- Bubble radius evolution
- Radiation forces and migration
- Acoustic streaming effects
- Drug release from bubble shell

**Action Plan**:
- [ ] Implement Rayleigh-Plesset ODE solver
- [ ] Add bubble shell mechanics (Marmottant model)
- [ ] Implement radiation force calculations
- [ ] Add acoustic streaming field computation
- [ ] Implement drug release kinetics
- [ ] Add concentration field method to ContrastEnhancedUltrasound
- [ ] Comprehensive validation against literature
- [ ] Remove stub and TODO comments

**Mathematical Specifications**:
- Rayleigh-Plesset: `R*R̈ + (3/2)*Ṙ² = (1/ρ)[P_g(t) - P_∞ - 2σ/R - 4μṘ/R]`
- Shell mechanics: Marmottant model with elastic modulus
- Radiation force: King's formula or Gorkov potential
- References: Marmottant et al. (2005), Sijl et al. (2010)

---

### 2.4 Complex Sparse Matrix Support 🟡 P1

**Location**: `analysis/signal_processing/beamforming/utils/sparse.rs:352-357`

**Current Limitation**: COO format only supports f64, stores magnitude for complex values

**Action Plan**:
- [ ] Extend COO sparse matrix to support Complex64
- [ ] Store real and imaginary parts separately
- [ ] Implement complex sparse matrix operations
- [ ] Update beamforming steering matrix builder
- [ ] Add tests for complex sparse operations
- [ ] Remove TODO comment

---

### 2.5 SensorBeamformer Method Implementations 🟡 P1

**Location**: `domain/sensor/beamforming/sensor_beamformer.rs`

**Methods to Implement**:

1. `calculate_delays()` - Lines 79-86
   - Current: Returns empty delays
   - Required: Proper delay calculation based on sensor geometry

2. `apply_windowing()` - Lines 96-100
   - Current: Returns unmodified delays
   - Required: Proper windowing functions (Hamming, Hanning, etc.)

3. `calculate_steering()` - Lines 110-114
   - Current: Returns identity matrix
   - Required: Proper steering vector calculations

**Action Plan**:
- [ ] Implement geometric delay calculation
- [ ] Add windowing function library (Hamming, Hanning, Blackman, Kaiser)
- [ ] Implement steering vector computation for plane waves and focused beams
- [ ] Add comprehensive tests with known geometries
- [ ] Remove all TODO comments

---

### 2.6 Source Factory Missing Implementations 🟡 P1

**Location**: `domain/source/factory.rs:130-134`

**Missing Source Types**:
- LinearArray
- MatrixArray
- Focused
- Custom

**Action Plan**:
- [ ] Implement LinearArray source creation
- [ ] Implement MatrixArray source creation
- [ ] Implement Focused source creation
- [ ] Implement Custom source creation
- [ ] Add tests for each source type
- [ ] Update factory documentation
- [ ] Remove TODO comment

---

### 2.7 Advanced Fusion Methods 🟢 P2

**Location**: `physics/acoustics/imaging/fusion/algorithms.rs`

**Stub Implementations**:
1. `fuse_feature_based()` - Lines 422-431
2. `fuse_deep_learning()` - Lines 438-447
3. `fuse_maximum_likelihood()` - Lines 453-462

**Action Plan** (Future Sprint):
- Implement proper feature-based fusion
- Implement deep learning fusion (U-Net style)
- Implement MLE fusion with EM algorithm
- Or: Document these as future enhancements and remove TODOs if not critical

---

## Phase 3: Large File Refactoring (Week 2) 📋 PLANNED

### 3.1 Priority 1: SWE 3D Workflows Refactor

**File**: `clinical/therapy/swe_3d_workflows.rs` (975 lines)

**Refactoring Strategy** (Proven Sprint 203-206 Pattern):

```
clinical/therapy/swe_3d_workflows/
├── mod.rs              # Public API, re-exports
├── types.rs            # Workflow types, configurations
├── planning.rs         # Treatment planning
├── execution.rs        # Workflow execution
├── monitoring.rs       # Real-time monitoring
├── validation.rs       # Safety validation
└── optimization.rs     # Parameter optimization
```

**Target Metrics**:
- 6-8 modules, each <500 lines
- 100% API compatibility (no breaking changes)
- 100% test pass rate
- Deep vertical hierarchy with clear SRP

**Action Plan**:
- [ ] Analyze module dependencies and data flow
- [ ] Design module boundaries and interfaces
- [ ] Extract types and core data structures
- [ ] Refactor functions into logical modules
- [ ] Maintain all public APIs unchanged
- [ ] Run full test suite after each module
- [ ] Update internal documentation
- [ ] Verify no performance regression

---

### 3.2 Remaining Large Files (Priority 2-6)

**Files to Refactor** (Weeks 2-4):

1. **infra/api/clinical_handlers.rs** (920 lines)
   - Split by endpoint groups (imaging, therapy, monitoring)
   - Extract request/response types
   - Separate validation logic

2. **physics/optics/sonoluminescence/emission.rs** (956 lines)
   - Split by emission mechanisms
   - Extract spectral models
   - Separate cavitation coupling

3. **analysis/ml/pinn/universal_solver.rs** (912 lines)
   - Split by PDE type
   - Extract training logic
   - Separate validation routines

4. **analysis/ml/pinn/electromagnetic_gpu.rs** (909 lines)
   - Split by GPU kernels
   - Extract device management
   - Separate training loops

5. **analysis/signal_processing/beamforming/adaptive/subspace.rs** (877 lines)
   - Split by algorithm (MUSIC, ESPRIT, Root-MUSIC)
   - Extract eigenspace computation
   - Separate DOA estimation

6. **solver/forward/elastic/swe/gpu.rs** (869 lines)
   - Split by computation stage
   - Extract GPU kernels
   - Separate host-device interface

---

### 3.3 Test File Refactoring

**Large Test Files** (>1200 lines):

1. **tests/pinn_elastic_validation.rs** (1286 lines)
   - Split by validation category
   - Extract test utilities
   - Group by physical phenomenon

2. **tests/ultrasound_physics_validation.rs** (1230 lines)
   - Split by physics model
   - Extract validation helpers
   - Group by equation type

3. **tests/nl_swe_convergence_tests.rs** (1172 lines)
   - Split by convergence metric
   - Extract convergence analysis utilities
   - Group by solver variant

---

## Success Metrics

### Code Quality Gates (Must Pass)
- [ ] Zero deprecated code remaining
- [ ] Zero TODO/FIXME/HACK comments (except documented future enhancements)
- [ ] Zero compilation errors
- [ ] <10 compiler warnings (all justified)
- [ ] All files <500 lines (except tests <800 lines)
- [ ] 100% test pass rate
- [ ] No performance regressions (benchmarks within 5%)

### Architectural Quality Gates
- [ ] All layers respect unidirectional dependencies
- [ ] No circular imports
- [ ] Single source of truth for all concepts
- [ ] Deep vertical hierarchy maintained
- [ ] Clear separation of concerns (SRP)

### Documentation Quality Gates
- [ ] All breaking changes documented in migration guide
- [ ] All refactored modules have module-level docs
- [ ] All public APIs have rustdoc with examples
- [ ] ADRs updated to reflect architectural changes
- [ ] README updated with Sprint 208 achievements

---

## Risk Assessment

### High Risk
1. **Deprecated code removal may break external consumers**
   - Mitigation: Search extensively for usages, provide migration guide
   - Fallback: Keep deprecated with stronger warnings for one sprint

2. **Large refactors may introduce subtle bugs**
   - Mitigation: 100% test coverage before refactoring
   - Fallback: Comprehensive property-based testing

### Medium Risk
1. **TODO implementations may require significant research**
   - Mitigation: Start with mathematical specifications
   - Fallback: Document as future enhancement if too complex

2. **Test file refactoring may lose coverage**
   - Mitigation: Run coverage analysis before/after
   - Fallback: Keep original test structure if needed

### Low Risk
1. **Performance regression from refactoring**
   - Mitigation: Benchmark critical paths
   - Fallback: Optimize hot paths if needed

---

## Dependencies & Prerequisites

### Required Before Sprint
- [x] Sprint 207 Phase 1 complete (cleanup done)
- [x] All compilation errors resolved
- [x] Test suite passing at 100%
- [x] Backlog and gap_audit synchronized

### Parallel Development Opportunities
- Documentation improvements (can proceed alongside refactoring)
- Benchmark suite expansion (can proceed alongside code changes)
- CI/CD enhancements (can proceed in parallel)

---

## Sprint Timeline

### Week 1: Deprecated Code + Critical TODOs
- **Days 1-2**: CPMLBoundary, BoundaryCondition trait removal
- **Days 3-4**: Beamforming location migrations, ARFI updates
- **Day 5**: Axisymmetric medium migration, validation

### Week 2: Large File Refactoring
- **Days 1-3**: SWE 3D workflows refactor (Priority 1)
- **Days 4-5**: Start Priority 2 refactor (clinical_handlers or emission)

### Week 3-4 (If Extended): Remaining Refactors
- Complete Priority 2-6 file refactors
- Test file refactoring
- Final documentation sync

---

## Progress Tracking

### Deprecated Code Elimination Status
- [ ] CPMLBoundary methods (0/3 complete)
- [ ] Legacy BoundaryCondition trait (0/1 complete)
- [ ] Beamforming location migrations (0/7 complete)
- [ ] Sensor localization (0/1 complete)
- [ ] ARFI radiation force methods (0/2 complete)
- [ ] Axisymmetric medium types (0/3 complete)

**Total Progress**: 0/17 deprecated items removed (0%)

### Critical TODO Resolution Status
- [ ] Focal properties extraction (P0)
- [ ] SIMD quantization bug fix (P0)
- [ ] Microbubble dynamics implementation (P0)
- [ ] Complex sparse matrix support (P1)
- [ ] SensorBeamformer methods (P1)
- [ ] Source factory implementations (P1)

**Total Progress**: 0/6 critical TODOs resolved (0%)

### Large File Refactoring Status
- [ ] SWE 3D workflows (Priority 1)
- [ ] Clinical handlers (Priority 2)
- [ ] Sonoluminescence emission (Priority 3)
- [ ] PINN universal solver (Priority 4)
- [ ] PINN electromagnetic GPU (Priority 5)
- [ ] Adaptive subspace beamforming (Priority 6)
- [ ] Elastic SWE GPU (Priority 7)

**Total Progress**: 0/7 large files refactored (0%)

### Test File Refactoring Status
- [ ] PINN elastic validation
- [ ] Ultrasound physics validation
- [ ] Nonlinear SWE convergence

**Total Progress**: 0/3 test files refactored (0%)

---

## Next Steps After Sprint 208

### Sprint 209: Research Integration Phase 1
- k-Wave axisymmetric coordinate system
- kWaveArray-style source/sensor representation
- Differentiable simulation patterns from jwave

### Sprint 210: Performance Optimization
- GPU kernel optimization
- Multi-GPU support
- Distributed computing patterns

### Sprint 211: Advanced Testing Framework
- Property-based testing expansion
- Fuzzing for numerical stability
- Performance regression detection

---

## References

### Internal Documents
- `backlog.md` - Sprint 208 priorities
- `gap_audit.md` - Detailed gap analysis
- `checklist.md` - Historical sprint achievements
- `docs/sprints/SPRINT_207_PHASE_1_COMPLETE.md` - Previous sprint results

### Architectural Documents
- `docs/ADR/` - Architectural decision records
- `docs/planning/` - Strategic planning documents

### Proven Patterns
- Sprint 203: Differential operators refactor
- Sprint 204: Fusion module refactor
- Sprint 205: Photoacoustic module refactor
- Sprint 206: Burn Wave Equation 3D refactor

**Pattern**: Deep vertical hierarchy + 100% API compatibility + 100% test pass rate

---

## Conclusion

Sprint 208 represents the final push toward zero technical debt in the kwavers codebase. By eliminating all deprecated code, resolving critical TODOs, and completing large file refactoring, we establish a clean foundation for research integration in Sprint 209+.

**Success Criteria**: Clean codebase, zero deprecated code, deep vertical hierarchy, 100% test pass rate, ready for advanced feature development.

**Next Review**: End of Week 1 (Deprecated code elimination checkpoint)

---

**Sprint 208 Status**: 🔄 IN PROGRESS  
**Phase**: Phase 1 - Deprecated Code Elimination  
**Target Completion**: 2025-01-27  
**Quality Gate**: Zero tolerance for technical debt