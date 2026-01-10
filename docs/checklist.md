 # Sprint Checklist - Kwavers Development

## Current Sprint: Phase 1 Sprint 4: Beamforming Consolidation

**Status**: ACTIVE - Phase 6 COMPLETE, Phase 7 NEXT
**Previous Sprints**: Sprint 1-3 (Grid, Boundary, Medium) - COMPLETED ✅
**Start Date**: Sprint 4 Phase 3 Execution
**Current Phase**: Phase 6 Deprecation & Documentation - ✅ COMPLETE
**Next Phase**: Phase 7 Final Validation & Testing - 🔴 NOT STARTED
**Success Criteria**: Consolidate beamforming algorithms from `domain::sensor::beamforming` to `analysis::signal_processing::beamforming` with full SSOT enforcement, comprehensive testing, and migration guide

### Sprint Objectives

### Primary Goal
Complete Phase 1 Sprint 4: Consolidate all beamforming algorithms into the analysis layer to eliminate the final cross-contamination pattern and achieve 100% Phase 1 completion.

### Secondary Goals
- Establish canonical beamforming infrastructure (traits, covariance, utils)
- Migrate algorithms from domain layer to analysis layer
- Create comprehensive migration guide and backward compatibility
- Add deprecation warnings and re-exports
- Validate architecture with full test suite and benchmarks
- Complete Phase 1 (100%) by finishing Sprint 4

---

## Task Breakdown

### Phase 1: Foundation & Math Layer (COMPLETED ✅)
- [x] Create math/numerics SSOT with differential, spectral, and interpolation operators
- [x] Add 20 comprehensive unit tests (all passing)
- [x] Document architectural principles and layer boundaries
- [x] Establish verification strategy for algorithm migrations

**Evidence**: Math numerics SSOT created, tested, and documented. Foundation layer complete.

### Phase 2: Domain Layer Purification (IN PROGRESS 🟡)

#### Phase 2A: Structure Creation & ADR (COMPLETED ✅)
- [x] Create `analysis::signal_processing` module structure
- [x] Create `analysis::signal_processing::beamforming` submodule
- [x] Create `analysis::signal_processing::beamforming::time_domain` submodule
- [x] Write ADR 003: Signal Processing Migration to Analysis Layer
- [x] Document migration strategy and backward compatibility plan

**Evidence**: ADR 003 created with complete rationale, migration plan, and verification strategy.

#### Phase 2B: Time-Domain DAS Migration (COMPLETED ✅)
- [x] Migrate `delay_reference.rs` to analysis layer with enhanced documentation
- [x] Migrate `das.rs` (Delay-and-Sum) to analysis layer with enhanced documentation
- [x] Create `time_domain/mod.rs` coordinator with proper exports
- [x] Add 23 comprehensive unit tests (all passing)
- [x] Verify mathematical correctness against analytical models

**Evidence**: Time-domain DAS implementation complete with 23 passing tests. Zero regressions.

#### Phase 2C: Backward Compatibility & Deprecation (COMPLETED ✅)
- [x] Add deprecation warnings to `domain::sensor::beamforming::time_domain::delay_reference`
- [x] Add deprecation warnings to `domain::sensor::beamforming::time_domain::das`
- [x] Create backward-compatible shims (re-exports from new location)
- [x] Add deprecation test to verify old paths still work
- [x] Update module documentation with migration instructions

**Evidence**: Deprecation warnings in place. Backward compatibility verified. Old tests pass with warnings.

#### Phase 2D: Integration & Verification (COMPLETED ✅)
- [x] Update `analysis::signal_processing::beamforming::mod.rs` with exports
- [x] Update `analysis::signal_processing::mod.rs` with re-exports
- [x] Run full test suite: 29 tests passing in new location
- [x] Run deprecated module tests: 2 tests passing (backward compatibility verified)
- [x] Verify zero build errors or test failures

**Evidence**: All 31 tests passing. Zero regressions. Backward compatibility maintained.

### Phase 2E: Documentation & Next Steps (COMPLETED ✅)
- [x] Update checklist.md with Phase 2 completion status
- [x] Update backlog.md with Phase 3 tasks (adaptive beamforming migration)
- [x] Create migration guide for users updating their code
- [x] Update technical documentation with new module paths
- [x] Commit Phase 2 changes to repository

**Evidence**: Phase 2 committed (c78bd052). Documentation updated. All artifacts synchronized.

---

### Sprint 4: Beamforming Consolidation (IN PROGRESS 🟡)

**Overall Progress**: 43% (Phase 3/7 complete)
**Estimated Total Effort**: 38-54 hours
**Completed Effort**: ~6 hours (Phases 2-3)

---

#### Phase 1: Planning & Audit (COMPLETED ✅)
- [x] Conduct architectural audit of beamforming duplication sites
- [x] Identify all beamforming locations (~60 files, ~10.5k LOC)
- [x] Create detailed migration strategy (7 phases)
- [x] Produce effort estimate (38-54h with 20% buffer)
- [x] Document in `PHASE1_SPRINT4_AUDIT.md` and `PHASE1_SPRINT4_EFFORT_ESTIMATE.md`

**Evidence**: Audit complete. Primary duplication in `domain::sensor::beamforming` (~49 files, ~8k LOC).

---

#### Phase 2: Infrastructure Setup (COMPLETED ✅ - 5 hours)
- [x] Create comprehensive trait hierarchy (`traits.rs` - 851 LOC)
  - [x] `Beamformer` root trait
  - [x] `TimeDomainBeamformer` for RF data
  - [x] `FrequencyDomainBeamformer` for FFT data
  - [x] `AdaptiveBeamformer` for covariance-based methods
  - [x] `BeamformerConfig` for initialization
- [x] Create covariance estimation module (`covariance/mod.rs` - 669 LOC)
  - [x] `estimate_sample_covariance()` - Standard estimator
  - [x] `estimate_forward_backward_covariance()` - FB averaging
  - [x] `validate_covariance_matrix()` - Defensive validation
  - [x] `is_hermitian()`, `trace()` - Matrix utilities
- [x] Create utilities module (`utils/mod.rs` - 771 LOC)
  - [x] `plane_wave_steering_vector()` - Far-field model
  - [x] `focused_steering_vector()` - Near-field model
  - [x] `hamming_window()`, `hanning_window()`, `blackman_window()` - Apodization
  - [x] `linear_interpolate()` - Fractional delay
- [x] Create module placeholders
  - [x] `narrowband/mod.rs` - Frequency-domain algorithms (placeholder)
  - [x] `experimental/mod.rs` - Neural/ML algorithms (placeholder)
- [x] Update module exports in `beamforming/mod.rs`
- [x] Create comprehensive migration guide (`BEAMFORMING_MIGRATION_GUIDE.md` - 897 LOC)
- [x] Add 26 infrastructure tests (all passing)
- [x] Document Phase 2 completion in `PHASE1_SPRINT4_PHASE2_SUMMARY.md` (515 LOC)

**Evidence**: 
- All infrastructure tests passing: 85/85 beamforming module tests ✅
- Covariance module: 9/9 tests passing ✅
- Utils module: 11/11 tests passing ✅
- Traits module: 6/6 tests passing ✅
- Total Phase 2 deliverable: 3,665 LOC + 897 LOC migration guide
- Zero compilation errors or test failures

**Status**: ✅ **PHASE 2 COMPLETE** - Ready for Phase 3

---

#### Phase 3: Dead Code Removal (COMPLETED ✅ - 1h actual)

**Substeps:**

**Strategic Decision**: Instead of full algorithm migration (12-16h), performed targeted dead code removal (1h)

##### Removed Files (Dead Code) ✅
- [x] Delete `adaptive/algorithms_old.rs` (~300 LOC) - Explicitly deprecated, unused
- [x] Delete `adaptive/past.rs` (~250 LOC) - Unused subspace tracking, feature-gated
- [x] Delete `adaptive/opast.rs` (~250 LOC) - Unused orthonormal subspace tracking, feature-gated
- [x] Clean up module exports in `adaptive/mod.rs`

##### Verification ✅
- [x] Run full test suite (841/841 tests passing)
- [x] Verify zero usage of removed modules
- [x] Confirm no regressions
- [x] Document dead code removal in `PHASE1_SPRINT4_PHASE3_SUMMARY.md`

##### Deferred to Sprint 5 (Active Code Migration)
- [ ] Configuration types migration (used by localization, PAM)
- [ ] Narrowband processing migration (tightly coupled to localization)
- [ ] 3D beamforming migration (used by clinical workflows)
- [ ] Experimental/AI migration (feature-gated, low priority)

**Evidence**: 
- Files removed: 3 (~800 LOC dead code eliminated)
- Test status: 841/841 passing ✅
- Zero breaking changes ✅
- Clean build with no unused code warnings ✅

**Rationale**: Pragmatic approach - remove dead code first (immediate value, zero risk), defer complex migrations requiring cross-module coordination

**Status**: ✅ **PHASE 3 COMPLETE** - Dead code removed, tests passing

---

#### Phase 4: Transmit Beamforming Refactor (✅ COMPLETE - 2.5h)
- [x] Extract shared delay utilities from `domain::source::transducers::phased_array::beamforming.rs`
- [x] Move shared logic to `analysis::signal_processing::beamforming::utils::delays`
- [x] Keep transmit-specific wrapper in domain (hardware control)
- [x] Update tests and documentation

**Deliverables**:
- ✅ Created `analysis::signal_processing::beamforming::utils::delays` module (727 LOC)
  - `focus_phase_delays()` - Focus delay calculation (SSOT)
  - `plane_wave_phase_delays()` - Plane wave steering delays (SSOT)
  - `spherical_steering_phase_delays()` - Spherical coordinate steering
  - `calculate_beam_width()` - Rayleigh criterion beam width
  - `calculate_focal_zone()` - Depth of field estimation
  - 12 comprehensive tests (property-based, edge cases, validation)
- ✅ Refactored `domain::source::transducers::phased_array::beamforming` to delegate to canonical utilities
  - Removed duplicate geometric calculations (~50 LOC eliminated)
  - Maintained backward-compatible API (zero breaking changes)
  - Added 5 regression tests to ensure behavior preservation
- ✅ Full test suite: **858/858 passing** (10 ignored, zero regressions)
- ✅ Architecture validated: Clean layer separation (Domain → Analysis → Math)

**Status**: ✅ **PHASE 4 COMPLETE** - Transmit/receive beamforming now share SSOT delay utilities

---

#### Phase 5: Sparse Matrix Utilities (✅ COMPLETE - 1.5h)
- [x] Move `core::utils::sparse_matrix::beamforming.rs` to `analysis::signal_processing::beamforming::utils::sparse`
- [x] Refactor and enhance sparse beamforming utilities
- [x] Remove old location (architectural violation)
- [x] Update module exports

**Deliverables**:
- ✅ Created `analysis::signal_processing::beamforming::utils::sparse` module (623 LOC)
  - `SparseSteeringMatrixBuilder` - Sparse steering matrix construction with thresholding
  - `sparse_sample_covariance()` - Sparse covariance estimation with diagonal loading
  - 9 comprehensive tests (validation, edge cases, error handling)
  - Complete documentation with mathematical foundations and literature references
- ✅ Removed `core::utils::sparse_matrix::beamforming.rs` (architectural layer violation)
- ✅ Updated module exports in `core::utils::sparse_matrix::mod.rs`
- ✅ Full test suite: **867/867 passing** (10 ignored, zero regressions)
- ✅ Architecture validated: Beamforming logic removed from core utilities layer

**Status**: ✅ **PHASE 5 COMPLETE** - Sparse matrix utilities migrated to analysis layer

---

#### Phase 6: Deprecation & Documentation (✅ COMPLETE - 2h)
- [x] Audit deprecated code and remove truly dead code
- [x] Update README with Sprint 4 status and architecture improvements
- [x] Add ADR-023 for beamforming consolidation architectural decision
- [x] Update documentation with new architecture and version information
- [x] Verify deprecation notices are comprehensive and correct

**Deliverables**:
- ✅ Updated `README.md` with v2.15.0, Sprint 4 status, and architecture diagram
- ✅ Added ADR-023: Beamforming Consolidation to Analysis Layer (comprehensive decision record)
- ✅ Verified deprecated code: Domain sensor beamforming properly marked, active consumers maintained
- ✅ Maintained backward compatibility: No breaking changes, deprecation warnings in place
- ✅ Full test suite: **867/867 passing** (10 ignored, zero regressions)
- ✅ Documentation quality: Complete migration guides, phase summaries, and ADR

**Status**: ✅ **PHASE 6 COMPLETE** - Documentation updated, deprecation strategy validated

---

#### Phase 7: Testing & Validation (NEXT - 🔴 NOT STARTED - 4-6h)
- [ ] Run full test suite (unit + integration + property)
- [ ] Run benchmarks (compare old vs. new implementations)
- [ ] Run architecture checker (verify no layer violations)
- [ ] Generate coverage report
- [ ] Manual validation on sample projects
- [ ] Finalize Sprint 4 completion report

**Sprint 4 Completion Criteria**: All 7 phases complete, 100% test pass rate, architecture validated, Phase 1 complete (100%)

---

## LEGACY TASKS (Sprint 179 - COMPLETED)

### Phase 1A: Microbubble Contrast Agents (4 hours) ✅
- [ ] Complete microbubble dynamics with encapsulated bubble models
- [ ] Implement nonlinear scattering cross-section calculations
- [ ] Develop contrast-to-tissue ratio computation and imaging
- [ ] Create CEUS perfusion analysis and quantification
- [ ] Integrate microbubble physics with acoustic wave propagation

**Evidence**: Church (1995), Tang & Eckersley (2006) microbubble dynamics

### Phase 1B: Transcranial Ultrasound (3 hours)
- [ ] Complete skull aberration correction algorithms
- [ ] Implement phase aberration calculation and time-reversal correction
- [ ] Develop BBB opening treatment planning and safety monitoring
- [ ] Create transcranial focused ultrasound therapy framework
- [ ] Integrate skull acoustics with wave propagation models

**Evidence**: Aubry (2003), Clement & Hynynen (2002) transcranial ultrasound

### Phase 1C: Sonodynamic Therapy (3 hours)
- [ ] Implement reactive oxygen species (ROS) generation modeling
- [ ] Develop sonosensitizer activation and drug delivery kinetics
- [ ] Create ROS diffusion and cellular damage modeling
- [ ] Integrate sonochemistry with acoustic cavitation physics
- [ ] Establish treatment planning and dosimetry frameworks

**Evidence**: ROS plasma physics and sonochemistry literature

### Phase 2A: Histotripsy & Oncotripsy (4 hours)
- [ ] Implement histotripsy cavitation control and bubble cloud dynamics
- [ ] Develop oncotripsy treatment planning with tumor targeting
- [ ] Create mechanical ablation modeling and tissue fractionation
- [ ] Integrate cavitation detection and feedback control systems
- [ ] Establish safety monitoring and treatment endpoint detection

**Evidence**: Xu et al. (2004), Maxwell et al. (2011) histotripsy literature

### Phase 2B: Clinical Integration Framework (3 hours)
- [ ] Create unified clinical workflow orchestrator for all therapy modalities
- [ ] Implement regulatory compliance frameworks (FDA, IEC standards)
- [ ] Develop safety monitoring and emergency stop systems
- [ ] Establish treatment planning and patient-specific optimization
- [ ] Create clinical decision support and outcome prediction

**Evidence**: IEC 60601-2-37 ultrasound safety standards

### Phase 3A: End-to-End Clinical Workflows (3 hours)
- [ ] Create complete clinical examples for each therapy modality
- [ ] Implement patient-specific treatment planning workflows
- [ ] Develop real-time monitoring and adjustment systems
- [ ] Create clinical outcome prediction and optimization
- [ ] Establish comprehensive safety and efficacy validation

**Evidence**: Clinical trial protocols and GCP standards

### Phase 3B: Documentation & Regulatory Compliance (2 hours)
- [ ] Update gap_audit.md with clinical applications completion status
- [ ] Create comprehensive clinical documentation package
- [ ] Document regulatory compliance frameworks and safety standards
- [ ] Update API documentation with clinical therapy features
- [ ] Create clinical workflow examples and tutorials

**Evidence**: FDA 510(k) and IEC 60601 regulatory documentation standards

---

## Progress Tracking

### Current Status - PHASE 3 EXECUTION (Adaptive Beamforming Migration)

**Sprint 180 - Phase 2: Domain Layer Purification** ✅ COMPLETED
- [x] **Phase 2A**: Structure Creation & ADR (5/5 complete) - **COMPLETED: ADR 003 written**
- [x] **Phase 2B**: Time-Domain DAS Migration (5/5 complete) - **COMPLETED: 23 tests passing**
- [x] **Phase 2C**: Backward Compatibility & Deprecation (5/5 complete) - **COMPLETED: Shims in place**
- [x] **Phase 2D**: Integration & Verification (4/4 complete) - **COMPLETED: 31 tests passing**
- [x] **Phase 2E**: Documentation & Next Steps (5/5 complete) - **COMPLETED: Phase 2 committed**

**Sprint 180 - Phase 3: Adaptive Beamforming Migration** 🟡 IN PROGRESS
- [x] **Phase 3A**: MVDR/Capon Migration (7/7 complete) - **COMPLETED: 14 tests passing**
- [ ] **Phase 3B**: MUSIC & ESMV Migration (0/5 started) - **NEXT: Subspace methods**

**Test Status**: ✅ All 44 tests passing (31 time-domain + 14 adaptive - 1 deprecated)
**Build Status**: ✅ Clean build with deprecation warnings (as intended)
**Regression Status**: ✅ Zero regressions detected

### Previous Status - CLINICAL APPLICATIONS COMPLETED ✅ (Sprint 179)
- [x] **Phase 1A**: Microbubble Contrast Agents (5/5 complete) - **COMPLETED: CEUS workflow with encapsulated bubbles**
- [x] **Phase 1B**: Transcranial Ultrasound (5/5 complete) - **COMPLETED: Aberration correction and BBB opening**
- [x] **Phase 1C**: Sonodynamic Therapy (5/5 complete) - **COMPLETED: ROS generation and drug activation**
- [x] **Phase 2A**: Histotripsy & Oncotripsy (5/5 complete) - **COMPLETED: Cavitation control and tumor targeting**
- [x] **Phase 2B**: Clinical Integration Framework (5/5 complete) - **COMPLETED: Unified therapy orchestrator**
- [x] **Phase 3A**: End-to-End Clinical Workflows (5/5 complete) - **COMPLETED: Multi-modal therapy examples**
- [x] **Phase 3B**: Documentation & Regulatory Compliance (5/5 complete) - **COMPLETED: Clinical documentation package**

**Completion**: **100%** - Complete clinical applications framework implemented with regulatory compliance

### Time Tracking
- **Planned**: 15 hours total
- **Elapsed**: 3 hours
- **Remaining**: 12 hours

### Quality Gates - CONVERGENCE TESTING COMPLETED ✅
- [x] **Gate 1**: Analytical test cases implemented - **PASSED: Nonlinear wave propagation test cases complete**
- [x] **Gate 2**: Hyperelastic model validation complete - **PASSED: Neo-Hookean, Mooney-Rivlin, Ogden validated**
- [x] **Gate 3**: Harmonic generation validated - **PASSED: Chen (2013) theory validation complete**
- [x] **Gate 4**: Convergence framework established - **PASSED: Mesh refinement and error analysis implemented**
- [x] **Gate 5**: Edge cases tested - **PASSED: Extreme strain and boundary conditions validated**
- [ ] **Gate 6**: Integration testing complete - **NEXT: End-to-end workflow validation**

---

## Risk Mitigation

### High Risk Items
- **Analytical Solution Complexity**: Developing accurate analytical test cases for nonlinear hyperelastic waves
  - **Mitigation**: Start with simplified cases and build up complexity gradually
  - **Fallback**: Use numerical reference solutions for validation

### Medium Risk Items
- **Convergence Testing Time**: Comprehensive mesh refinement studies may be computationally intensive
  - **Mitigation**: Implement efficient testing framework with automated convergence analysis
  - **Fallback**: Focus on key test cases with representative parameter ranges

### Low Risk Items
- **Test Framework Integration**: New convergence tests must integrate with existing test infrastructure
  - **Mitigation**: Extend existing test framework with convergence testing utilities
  - **Fallback**: Create standalone convergence testing module

---

## Success Metrics

### Quantitative
- **Analytical Test Cases**: >5 validated test cases covering nonlinear wave propagation
- **Convergence Rate**: <2nd order convergence demonstrated for mesh refinement studies
- **Error Bounds**: <1% error vs analytical solutions for validated test cases
- **Edge Case Coverage**: >90% coverage of material parameter boundaries and singularities
- **Harmonic Accuracy**: <5% error in harmonic amplitude predictions vs theoretical values

### Qualitative
- **Mathematical Rigor**: Analytical validation framework established with literature-backed test cases
- **Numerical Stability**: Comprehensive convergence studies demonstrating algorithm robustness
- **Edge Case Handling**: Proper behavior validation at material model boundaries and extreme conditions
- **Integration Quality**: Seamless integration of all NL-SWE components with validated interfaces
- **Documentation**: Complete convergence testing methodology with reproducible results

---

## Dependencies & Prerequisites

### Required
- [x] Sprint 177 NL-SWE mathematical corrections (complete hyperelastic models available)
- [x] Working nonlinear elastic wave solver with corrected algorithms
- [x] Literature-backed harmonic generation implementation (Chen 2013)
- [x] Complete theorem documentation for hyperelastic models

### Optional
- [ ] Analytical solution libraries for nonlinear wave equations (facilitates testing)
- [ ] Advanced numerical analysis tools for convergence studies (enhances validation)
- [ ] Reference implementations from literature for comparison

---

## Sprint 178 Deliverables

### Core Implementation
- `tests/nl_swe_convergence_tests.rs` - Comprehensive analytical convergence testing suite (300+ lines)
- Analytical test case implementations for nonlinear wave propagation
- Mesh refinement convergence studies and error analysis framework
- Hyperelastic model validation against analytical solutions

### Examples & Validation
- `examples/nl_swe_convergence_validation.rs` - Convergence testing demonstration
- `tests/nl_swe_analytical_validation.rs` - Analytical solution comparison tests
- `tests/nl_swe_edge_cases.rs` - Edge case and robustness testing suite
- Harmonic generation validation examples

### Documentation
- Updated `gap_audit.md` with convergence testing completion status
- `docs/sprint_178_convergence_testing.md` - Complete convergence testing methodology
- API documentation for convergence testing utilities

---

## Sprint 179: Neural Beamforming Remediation

### Objectives
- Replace placeholder `NeuralLayer` with mathematically correct implementation.
- Replace placeholder feature extraction with rigorous signal processing.
- Verify implementations against signal processing theory.

### Task Breakdown

#### Phase 1: Core Neural Architecture (1 hour)
- [ ] Implement `NeuralLayer` with proper matrix multiplication and activation functions.
- [ ] Implement `NeuralBeamformingNetwork` forward pass with rigorous tensor operations.
- [ ] Add unit tests for network forward pass.

#### Phase 2: Feature Extraction (1 hour)
- [ ] Implement FFT-based frequency content analysis (replace gradient proxy).
- [ ] Implement spectral centroid calculation (replace constant).
- [ ] Implement rigorous coherence calculation (replace simplified averaging).

#### Phase 3: Signal Quality & Validation (1 hour)
- [ ] Implement robust SNR estimation.
- [ ] Add integration tests for the full beamforming pipeline.
- [ ] Verify outputs against theoretical expectations.


---

## Emergency Procedures

### If 3D Memory Limits Exceeded
1. **Analysis**: Check volumetric data size and available memory
2. **Mitigation**: Implement chunked processing or reduce resolution
3. **Fallback**: Process smaller sub-volumes sequentially

### If 3D Performance Issues Arise
1. **Analysis**: Profile 3D inversion algorithm bottlenecks
2. **Optimization**: GPU acceleration for volumetric operations
3. **Fallback**: Use 2D SWE for critical regions only

---

## Completion Checklist

### Pre-Commit Validation
- [ ] `cargo check --workspace` passes
- [ ] `cargo clippy --workspace -- -D warnings` passes (0 warnings)
- [ ] `cargo test --workspace --lib` passes (495+ tests)
- [ ] 3D SWE examples execute successfully
- [ ] Performance benchmarks meet targets (<3x slowdown vs 2D)

### Documentation Updates
- [ ] docs/checklist.md updated with completion status
- [ ] docs/backlog.md updated for Sprint 179 planning
- [ ] docs/gap_audit.md reflects 3D SWE capabilities
- [ ] CHANGELOG.md updated with 3D SWE features

---

## Sensor Consolidation Micro-Sprint — Array Processing Unification

Goal: Consolidate beamforming across `sensor` to enforce SSOT and modular boundaries per ADR "Sensor Module Architecture Consolidation".

- [x] Plan consolidation architecture and publish ADR (docs/ADR/sensor_architecture_consolidation.md)
- [ ] Create `BeamformingCoreConfig` and `From` shims from legacy configs
- [x] Move `adaptive_beamforming/*` → `beamforming/adaptive/*` preserving tests and docs
- [x] Replace PAM internal algorithms with `BeamformingProcessor` usage; add `PamBeamformingConfig`
- [x] Refactor localization to use `BeamformingProcessor` for grid search; add `BeamformSearch`
- [x] Feature-gate `beamforming/experimental/neural.rs` with `experimental_neural` feature; update docs
- [x] Update `sensor/mod.rs` re-exports and type aliases for compatibility
- [x] Migrate and consolidate unit/property/integration tests; keep suite green under `cargo nextest`
- [ ] Bench unified Processor hot paths with criterion; capture baselines

Acceptance Criteria:
- Single source of truth for DAS/MVDR/MUSIC/ESMV under `sensor/beamforming`
- PAM and localization consume shared Processor; no duplicate algorithm code remains
- Docs updated; examples compile; tests pass (>90% coverage on beamforming algorithms)
### Final Sign-Off
- [ ] 3D volumetric wave propagation validated against literature
- [ ] Multi-directional shear wave generation working correctly
- [ ] 3D clinical SWE workflow operational
- [ ] Ready for Sprint 179: Supersonic Shear Imaging implementation
