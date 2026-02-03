 # Sprint Checklist - Kwavers Development

## Current Sprint: Sprint 216 - Physics Correctness & Conservation Diagnostics

**Status**: 🔄 IN PROGRESS (Session 3 Complete)
**Goal**: Implement conservation diagnostics and enhance physics correctness
**Duration**: 6-8 sessions (12-16 hours)
**Priority**: P0 - Physics Correctness & Mathematical Verification

### Sprint 216 Overview

**Context**: Sprint 214 (PINN Stabilization & IC Velocity) completed with 1990/1990 tests passing. Sprint 215 focus was thermal-acoustic coupling and material properties.

**Sprint 216 Focus Areas**:
1. ✅ Temperature-dependent material properties (Session 1)
2. ✅ Bubble energy balance enhancements (Session 2)
3. ✅ Conservation diagnostics framework (Session 2)
4. ✅ KZK solver conservation integration (Session 3)
5. ⏭️ Westervelt solver conservation integration (Session 4)
6. ⏭️ Kuznetsov solver conservation integration (Session 4)

---

## Sprint 216 Session 3: Conservation Diagnostics Integration - ✅ COMPLETE (2025-02-04)

**Status**: ✅ COMPLETE - KZK solver integration complete
**Priority**: P0 - Physics Correctness
**Duration**: 2 hours

### Completed Tasks
- [x] Integrate `ConservationDiagnostics` trait into KZK solver
- [x] Implement energy calculation: E = ∫∫∫ [p²/(2ρ₀c₀²)] dV
- [x] Implement momentum calculation: P_z = ∫∫∫ [ρ₀ p / c₀] dV
- [x] Implement mass calculation: M = ∫∫∫ ρ₀[1 + p/(ρ₀c₀²)] dV
- [x] Add conservation check to time-stepping loop
- [x] Implement configurable check intervals
- [x] Add severity-based console logging (⚠️ Warning, ❌ Error, 🔴 Critical)
- [x] Create enable/disable/query API methods
- [x] Add 4 comprehensive tests (integration, calculation, lifecycle, interval)
- [x] Validate energy formula accuracy (relative error < 10⁻¹⁰)
- [x] Fix borrow checker issues (extract-compute-update pattern)
- [x] Clean up unused imports
- [x] Document session in Sprint 216 Session 3 report (579 lines)

### Results
- **Test Pass Rate**: 1994/1994 (100%, +4 new tests)
- **Ignored Tests**: 12 (performance tier)
- **Zero Regressions**: All existing tests pass
- **Performance**: 0% overhead when disabled, ~1% at interval=100
- **API**: Clean, production-ready public interface

### Mathematical Validation
- Energy calculation accuracy: < 10⁻¹⁰ relative error
- Conservation formulations verified against Pierce, Hamilton references
- Tolerance presets: strict (10⁻¹⁰), default (10⁻⁸), relaxed (10⁻⁶)

### Artifacts
- `src/solver/forward/nonlinear/kzk/solver.rs` (+187 lines)
- `src/solver/forward/nonlinear/conservation.rs` (-2 lines cleanup)
- `docs/sprints/SPRINT_216_SESSION_3_CONSERVATION_INTEGRATION.md` (579 lines)

### Next Session
Sprint 216 Session 4: Westervelt & Kuznetsov solver integration (3-4 hours)

---

## Sprint 216 Session 2: Bubble Energy Balance & Conservation Framework - ✅ COMPLETE (2025-02-03)

**Status**: ✅ COMPLETE
**Priority**: P0 - Physics Correctness
**Duration**: 3 hours

### Completed Tasks
- [x] Enhanced bubble energy balance (chemical, plasma, radiation)
- [x] Conservation diagnostics framework (trait-based)
- [x] 20 new tests, 1990/1990 passing

---

## Sprint 216 Session 1: Temperature-Dependent Properties - ✅ COMPLETE (2025-02-02)

**Status**: ✅ COMPLETE
**Priority**: P0 - Physics Correctness
**Duration**: 2 hours

### Completed Tasks
- [x] Temperature-dependent material properties
- [x] Mathematical specifications with references
- [x] 1970/1970 tests passing

---

## Previous Sprints

## Sprint 212 Phase 2 - BurnPINN Physics Constraints & GPU Pipeline - ✅ COMPLETE

**Context**: Sprint 211 (Clinical Acoustic Solver) and Sprint 212 Phase 1 (Elastic Shear Speed) completed successfully with 1554/1554 tests passing.

**Completed P1 Items**:
1. ✅ BurnPINN BC loss implementation (Sprint 214 Session 7)
2. ✅ BurnPINN IC loss implementation (Sprint 214 Session 8)
3. GPU beamforming pipeline → Deferred to Sprint 217
4. Eigendecomposition → Deferred to Sprint 217

---

## Sprint 211: Clinical Therapy Acoustic Solver - ✅ COMPLETE (2025-01-14)

### Completed Tasks
- [x] Design Strategy Pattern backend abstraction (`AcousticSolverBackend` trait)
- [x] Implement FDTD backend adapter
- [x] Create clinical acoustic solver with safety validation
- [x] Add 21 comprehensive tests (initialization, stepping, field access, clinical parameters)
- [x] Validate API compatibility with existing solver infrastructure
- [x] Document mathematical foundations and wave equation specifications
- [x] Create completion report (`SPRINT_211_COMPLETION_REPORT.md`)

### Results
- **Test Pass Rate**: 1554/1554 (100%)
- **API Compatibility**: ✅ Maintained
- **Safety Features**: Intensity limits, thermal index monitoring
- **Time**: ~11 hours (8h initial + 3h API fixes)

### Known Limitations (Documented)
- Dynamic source registration not supported (requires FdtdSolver API enhancement)
- Backend selection currently hardcoded to FDTD (PSTD/nonlinear planned)

---

## Sprint 212 Phase 1: Elastic Shear Speed Implementation - ✅ COMPLETE (2025-01-15)

### Completed Tasks
- [x] Remove unsafe `Array3::zeros()` default from `ElasticArrayAccess::shear_sound_speed_array()`
- [x] Make method required (type-system enforcement)
- [x] Implement c_s = sqrt(μ / ρ) for `HomogeneousMedium`
- [x] Implement shear speed return for `HeterogeneousMedium` (stored field)
- [x] Implement per-voxel computation for `HeterogeneousTissueMedium`
- [x] Update all test mocks and medium implementations
- [x] Add 10 validation tests (mathematical identity, physical ranges, edge cases)
- [x] Verify full regression suite (1554/1554 passing)
- [x] Document mathematical specification with literature references
- [x] Create completion report (`SPRINT_212_PHASE1_ELASTIC_SHEAR_SPEED.md`)

### Results
- **Test Pass Rate**: 1554/1554 (100%, zero regressions)
- **Type Safety**: ✅ Compile-time enforcement
- **Physical Correctness**: ✅ No silent zero-defaults
- **Applications Enabled**: Shear wave elastography, elastic wave imaging
- **Time**: ~5.5 hours

### Mathematical Specification
```
Shear wave speed: c_s = sqrt(μ / ρ)
Physical ranges: Soft tissue 1-5 m/s, Water c_s = 0
```

---

## Sprint 212 Phase 2: Active Tasks - 🔄 IN PROGRESS

### Task 1: BurnPINN Boundary Condition Loss (10-14h) - ✅ COMPLETE

**Priority**: P0 - Critical for PINN correctness (RESOLVED - Session 7)

**Mathematical Specification**:
```
L_BC = (1/N_∂Ω) Σ ||u(x,t) - g(x,t)||² for x ∈ ∂Ω
```

**Implementation Status** (Sprint 214 Session 7 - COMPLETE):
- ✅ BC Sampling (3-4h) - COMPLETE
  - ✅ Sample points on 6 domain boundary faces (3D box)
  - ✅ Generate spatiotemporal coordinates (x,y,z,t)
  - ✅ Dirichlet conditions implemented (Neumann deferred)
  
- ✅ BC Loss Computation (4-5h) - COMPLETE
  - ✅ Evaluate PINN at boundary points
  - ✅ Compute Dirichlet violation: ||u - g||²
  - ⏸️ Neumann gradient: ||∂u/∂n - h||² (future enhancement)
  - ✅ Aggregate over all boundary points
  
- ✅ Training Integration (2-3h) - COMPLETE
  - ✅ Add BC loss to total training loss with weighting
  - ✅ Backward pass gradient propagation implemented
  - ✅ **RESOLVED**: Stabilized with adaptive LR, loss normalization, early stopping
  
- ✅ Validation Tests (2-3h) - COMPLETE (7/7 passing)
  - ✅ Test with Dirichlet BC (u=0 on boundary) - all tests pass
  - ⏸️ Test with Neumann BC (future enhancement)
  - ✅ BC loss decreases during training (89-92% improvement)
  - ✅ Numerical stability verified (zero NaN/Inf)

**Stabilization Implemented** (Session 7):
- ✅ Adaptive learning rate scheduling (decay on stagnation)
- ✅ Loss component normalization (EMA-based scaling)
- ✅ Numerical stability monitoring (early stopping on NaN/Inf)
- ✅ Conservative default LR: 1e-3 → 1e-4

**Files**:
- `src/solver/inverse/pinn/ml/burn_wave_equation_3d/solver.rs` (lines 634-724)
- `tests/pinn_bc_validation.rs` (7 tests: 5 pass, 2 fail)

**Results**:
- ✅ BC loss decreases during training (verified in all tests)
- ✅ BC validation suite: 7/7 tests passing
- ✅ Full test suite: 2314/2314 passing
- ✅ Zero gradient explosions or NaN/Inf

**Success Criteria** (ALL MET):
- ✅ BC loss decreases during training (89-92% improvement)
- ✅ Boundary violations minimized (< 0.01 in tests)
- ✅ Works with Dirichlet BCs (stable and convergent)
- ✅ Validated against test cases (7 comprehensive tests)

**Artifacts**:
- `docs/ADR/ADR_PINN_TRAINING_STABILIZATION.md` - Mathematical specifications
- `docs/sprints/SPRINT_214_SESSION_7_PINN_STABILIZATION_COMPLETE.md` - Full documentation

---

### Task 2: BurnPINN Initial Condition Loss (8-12h) - ✅ COMPLETE

**Priority**: P1 - Critical for PINN correctness (COMPLETE - Session 8)

**Mathematical Specification**:
```
L_IC = (1/N_Ω) [Σ ||u(x,0) - u₀(x)||² + Σ ||∂u/∂t(x,0) - v₀(x)||²]
```

**Implementation Status** (Sprint 214 Session 8 - COMPLETE):
- ✅ Temporal Derivative Computation (3-4h) - COMPLETE
  - ✅ Forward finite difference: ∂u/∂t(0) ≈ (u(ε) - u(0)) / ε
  - ✅ Method: `compute_temporal_derivative_at_t0()`
  - ✅ Numerically stable with ε = 1e-3
  
- ✅ IC Loss Extension (2-3h) - COMPLETE
  - ✅ Combined loss: L_IC = 0.5 × L_disp + 0.5 × L_vel
  - ✅ Backward-compatible API: `train(..., v_data: Option<&[f32]>, ...)`
  - ✅ Velocity IC extraction for t=0 points
  
- ✅ Validation Test Suite (2-3h) - COMPLETE (9/9 tests)
  - ✅ Displacement IC computation and convergence
  - ✅ Velocity IC computation (∂u/∂t matching)
  - ✅ Combined displacement + velocity IC
  - ✅ Zero field, plane wave, Gaussian pulse
  - ✅ Backward compatibility (displacement-only)
  - ✅ Multiple time steps, metrics recording

**Results**:
- ✅ IC loss includes velocity component
- ✅ Backward compatible (velocity optional)
- ✅ 81/81 PINN tests passing (IC: 9, BC: 7, internal: 65)
- ✅ Zero regressions across all test suites

**Files Modified/Created**:
- `src/solver/inverse/pinn/ml/burn_wave_equation_3d/solver.rs` - IC velocity implementation
- `tests/pinn_ic_validation.rs` - 9 comprehensive tests (558 lines, NEW)
- `tests/pinn_bc_validation.rs` - Updated for new train() signature
- `src/solver/inverse/pinn/ml/burn_wave_equation_3d/tests.rs` - Updated tests

**Success Criteria** (ALL MET):
- ✅ IC loss includes velocity (∂u/∂t) matching
- ✅ IC loss remains finite and bounded during training
- ✅ Initial conditions computed correctly (displacement + velocity)
- ✅ Works with various IC types (Gaussian, plane wave, zero field)
- ✅ Backward compatible (displacement-only via v_data: None)

**Artifacts**:
- `docs/sprints/SPRINT_214_SESSION_8_IC_VELOCITY_COMPLETE.md` - Full documentation

---

### Task 3: 3D GPU Beamforming Pipeline (10-14h) - PLANNED

**Priority**: P1 - Enables real-time 3D imaging

**Subtasks**:
- [ ] Delay Table Computation (3-4h)
  - [ ] Implement geometric delay calculation
  - [ ] Support arbitrary focal point and aperture
  - [ ] Cache delay tables for real-time performance
  
- [ ] Aperture Mask Buffer (2-3h)
  - [ ] Handle active element masking
  - [ ] Support sparse array configurations
  - [ ] Optimize memory layout for GPU
  
- [ ] GPU Kernel Launch (3-4h)
  - [ ] Wire up compute shader execution
  - [ ] Implement delay-and-sum kernel
  - [ ] Handle buffer synchronization
  
- [ ] Validation (2-3h)
  - [ ] Test against CPU reference
  - [ ] Verify focal gain and resolution
  - [ ] Benchmark performance

**Success Criteria**:
- ✅ 10-100× speedup vs CPU
- ✅ Identical output to CPU (< 0.1% error)
- ✅ Supports arbitrary focal configurations
- ✅ Real-time performance for clinical arrays

---

### Task 4: Source Estimation Eigendecomposition (12-16h) - PLANNED

**Priority**: P1 - Enables robust subspace methods

**Subtasks**:
- [ ] Complex Hermitian Eigendecomposition (6-8h)
  - [ ] Implement in `src/math/linear_algebra/decomposition/eigen.rs`
  - [ ] Handle complex-valued matrices
  - [ ] Use LAPACK bindings (ndarray-linalg)
  - [ ] Validate against known eigensystems
  
- [ ] AIC/MDL Criteria (3-4h)
  - [ ] Implement Akaike Information Criterion
  - [ ] Implement Minimum Description Length
  - [ ] Automatic source number selection
  
- [ ] MUSIC Integration (2-3h)
  - [ ] Wire into `sensor/beamforming/subspace/music.rs`
  - [ ] Enable automatic source estimation
  - [ ] Remove hardcoded source count
  
- [ ] Validation Tests (2-3h)
  - [ ] Test with synthetic multi-source scenarios
  - [ ] Verify correct source number detection
  - [ ] Compare against ground truth

**Success Criteria**:
- ✅ Correct eigenvalues/eigenvectors
- ✅ AIC/MDL correctly identify source number
- ✅ MUSIC works without manual source count
- ✅ Performance: <10ms for typical arrays

---

## Progress Tracking

### Sprint 212 Phase 2 Status
- **BC Loss**: 🔄 0% (starting implementation)
- **IC Loss**: ⏸️ 0% (pending BC completion)
- **GPU Beamforming**: ⏸️ 0% (planned)
- **Eigendecomposition**: ⏸️ 0% (planned)

### Test Status
- **Current**: 1554/1554 passing (100%)
- **Target**: Maintain 100% pass rate through all changes

### Time Tracking
- **Sprint 211**: 11h (complete)
- **Sprint 212 Phase 1**: 5.5h (complete)
- **Sprint 212 Phase 2**: 0h / 40-56h estimated

---

## Quality Gates

### Pre-Commit Validation
- [ ] All tests pass: `cargo test --all-features`
- [ ] Zero compilation errors: `cargo check --all-features`
- [ ] Clippy clean: `cargo clippy --all-features`
- [ ] Documentation builds: `cargo doc --no-deps`

### Mathematical Verification
- [ ] BC loss derivation documented
- [ ] IC loss derivation documented
- [ ] Physical plausibility tests pass
- [ ] Convergence validated against analytical solutions

### Documentation Updates
- [ ] Update `backlog.md` with Phase 2 progress
- [ ] Update `README.md` with new capabilities
- [ ] Create Phase 2 completion report
- [ ] Archive sprint documentation

---

## Previous Sprints (Completed)

### Sprint 188: Architecture Enhancement - ✅ COMPLETE
- Clean architecture with unidirectional dependencies
- 100% test pass rate achieved
- Mathematical verification complete

### Sprint 207: Repository Cleanup - ✅ COMPLETE
- 34GB build artifacts removed
- 19 sprint docs archived
- 12 compiler warnings fixed

### Sprint 208: Deprecated Code & TODOs - ✅ COMPLETE
- 17 deprecated items eliminated
- All P0 critical tasks complete
- Zero technical debt

### Cleanup Tasks
- [ ] Update `domain/mod.rs` to remove physics re-exports
- [ ] Delete `domain/physics/` directory
- [ ] Run full test suite (baseline: 867/867 passing)
- [ ] Update documentation
- [ ] Create ADR-024: Physics Layer Consolidation

**Success Criteria**:
- ✅ All physics specifications in `physics/` only
- ✅ No `domain/physics/` module exists
- ✅ 867/867 tests passing
- ✅ Zero compilation errors

---

## Sprint 188 Phase 2: Break Circular Dependencies (2 hours) - PLANNED

### Tasks
- [ ] Identify all `physics/` → `solver/` imports
- [ ] Move `physics/electromagnetic/solvers.rs` → `solver/forward/fdtd/electromagnetic.rs`
- [ ] Remove solver references from `physics/acoustics/mechanics/poroelastic/mod.rs`
- [ ] Verify zero `use crate::solver::` in `physics/` modules
- [ ] Run dependency analysis: `cargo tree --edges normal`
- [ ] Run full test suite
- [ ] Create ADR-025: Unidirectional Solver Dependencies

**Target**: Zero physics → solver dependencies

---

## Sprint 188 Phase 3: Domain Cleanup (4 hours) - PLANNED

### Tasks
- [ ] Audit `domain/imaging/` - migrate or deprecate
- [ ] Audit `domain/signal/` - migrate to `analysis/signal_processing/`
- [ ] Audit `domain/therapy/` - migrate to `clinical/therapy/`
- [ ] Clean up `domain/sensor/beamforming/` remnants
- [ ] Update imports across codebase
- [ ] Add deprecation warnings
- [ ] Run full test suite
- [ ] Create migration guide
- [ ] Create ADR-026: Domain Layer Scope Definition

**Domain Entities to Retain**: grid, medium, sensor (hardware), source, boundary, field, tensor, mesh, geometry

---

## Sprint 188 Phase 4: Shared Solver Interfaces (3 hours) - PLANNED

### Tasks
- [ ] Define `solver/interface/acoustic.rs` trait
- [ ] Define `solver/interface/elastic.rs` trait
- [ ] Define `solver/interface/electromagnetic.rs` trait
- [ ] Implement traits for FDTD solver
- [ ] Implement traits for PSTD solver
- [ ] Implement traits for elastic wave solver
- [ ] Implement traits for PINN solvers
- [ ] Create `solver/factory.rs`
- [ ] Update `simulation/` to use interfaces
- [ ] Update `clinical/` to use interfaces
- [ ] Run full test suite
- [ ] Create ADR-027: Shared Solver Interfaces

---

## Sprint 188 Phase 5: Documentation & Validation (2 hours) - PLANNED

### Tasks
- [ ] Write ADR-024: Physics Layer Consolidation
- [ ] Write ADR-025: Unidirectional Solver Dependencies
- [ ] Write ADR-026: Domain Layer Scope Definition
- [ ] Write ADR-027: Shared Solver Interfaces
- [ ] Update `docs/architecture.md` with layer diagrams
- [ ] Update `README.md` architecture section
- [ ] Update module-level rustdoc
- [ ] Create migration guide
- [ ] Verify documentation builds
- [ ] Run final validation (tests, clippy, docs)

---

## Previous Sprint: Sprint 187 - PINN Architecture Refactor (Phase 2 Complete)

**Status**: PHASE 2 COMPLETE ✅  
**Goal**: Extract PINN modules from analysis layer to solver/inverse layer with domain-driven architecture  
**Duration**: Phase 2 - 4 hours (completed)  
**Priority**: P0 - CRITICAL (Architectural Foundation)

### Sprint 187 Objectives
1. ✅ **Phase 1 Complete**: Domain layer with physics trait specifications (previous sprint)
2. ✅ **Phase 2 Complete**: Extract PINN elastic_2d modules to solver/inverse/pinn/
3. ⚠️ **Phase 3 Planned**: Implement ElasticWaveEquation trait for forward solvers
4. ⚠️ **Phase 4 Planned**: Shared validation test suite
5. ⚠️ **Phase 5 Planned**: Performance benchmarking and optimization

### Phase 2: PINN Extraction & Restructuring (4h) - ✅ COMPLETE

#### Completed Tasks
- ✅ Created `solver/inverse/pinn/elastic_2d/` module structure
- ✅ Implemented `config.rs` (672 lines) - Hyperparameters, loss weights, optimizer configuration
- ✅ Implemented `model.rs` (559 lines) - Neural network architecture with Burn integration
- ✅ Implemented `loss.rs` (642 lines) - Physics-informed loss functions for elastic waves
- ✅ Implemented `training.rs` (506 lines) - Training loop, optimizer, metrics tracking
- ✅ Implemented `inference.rs` (439 lines) - Model deployment and field evaluation
- ✅ Updated `geometry.rs` - Collocation sampling and adaptive refinement (from Phase 1)
- ✅ Updated module exports and documentation
- ✅ Build verification passed (no errors in PINN modules)

#### Module Structure Created
```
solver/inverse/pinn/elastic_2d/
├── mod.rs           (module documentation and exports)
├── config.rs        (Config, LossWeights, ActivationFunction, OptimizerType)
├── model.rs         (ElasticPINN2D neural network)
├── loss.rs          (LossComputer, PDE residual computation)
├── training.rs      (Trainer, TrainingMetrics, training loop)
├── inference.rs     (Predictor, field evaluation)
└── geometry.rs      (CollocationSampler, MultiRegionDomain)
```

#### Mathematical Implementation
- ✅ Full 2D elastic wave PDE residual computation
- ✅ Stress tensor derivatives (σₓₓ, σᵧᵧ, σₓᵧ)
- ✅ Constitutive relations (Hooke's law)
- ✅ Momentum equations enforcement
- ✅ Boundary conditions (Dirichlet, Neumann, Free surface)
- ✅ Initial conditions (displacement + velocity)
- ✅ Data fitting loss (for inverse problems)
- ✅ Material parameter optimization (λ, μ, ρ)

#### Key Features Implemented
- **Forward Problems**: Known material properties, solve wave propagation
- **Inverse Problems**: Estimate material properties from measurements
- **Multi-region Domains**: Interface conditions for heterogeneous media
- **Adaptive Sampling**: Residual-based refinement of collocation points
- **Learning Rate Scheduling**: Exponential, step, cosine, reduce-on-plateau
- **Checkpointing**: Model persistence during training
- **Comprehensive Testing**: 40+ unit tests across all modules

#### Results
- **Architecture**: Clean separation of concerns (config, model, loss, training, inference)
- **Lines of Code**: ~3,800 lines of well-documented, tested PINN implementation
- **Build Status**: ✅ Compiles with zero errors in PINN modules
- **Test Coverage**: All core functionality unit tested
- **Documentation**: Full rustdoc with mathematical formulations and usage examples

### Phase 3: GRASP Compliance Remediation (Ongoing)

#### Current Progress
1. ✅ **elastic_wave_solver.rs refactored** (2,824 lines → modular structure)
   - ✅ Created `solver/` submodule directory
   - ✅ Extracted `solver/types.rs` (346 lines) - configuration and data types
   - ✅ Extracted `solver/stress.rs` (397 lines) - stress tensor derivatives
   - ✅ Created `solver/mod.rs` (156 lines) - public API and module organization
   - ✅ Updated parent module for backward compatibility
   - ✅ Build verification passed

2. ✅ **PINN elastic_2d extraction complete** (new architecture, ~3,800 lines)
   - ✅ Moved from `analysis/ml/pinn/` to `solver/inverse/pinn/elastic_2d/`
   - ✅ Created 6 focused modules (config, model, loss, training, inference, geometry)
   - ✅ Full elastic wave PDE implementation with autodiff
   - ✅ Comprehensive testing and documentation
   - ✅ Feature-gated with Burn integration

3. ⚠️ **Remaining modules to refactor** (15 files):
   - `src/analysis/ml/pinn/burn_wave_equation_2d.rs` (2,578 lines) - Priority 1 (legacy, can deprecate)
   - `src/math/linear_algebra/mod.rs` (1,889 lines) - Priority 1
   - `src/physics/acoustics/imaging/modalities/elastography/nonlinear.rs` (1,342 lines) - Priority 2
   - `src/domain/sensor/beamforming/beamforming_3d.rs` (1,271 lines) - Priority 2
   - `src/clinical/therapy/therapy_integration.rs` (1,211 lines) - Priority 2
   - `src/analysis/ml/pinn/electromagnetic.rs` (1,188 lines) - Priority 3
   - `src/clinical/imaging/workflows.rs` (1,179 lines) - Priority 3
   - `src/domain/sensor/beamforming/ai_integration.rs` (1,148 lines) - Priority 3
   - `src/physics/acoustics/imaging/modalities/elastography/inversion.rs` (1,131 lines) - Priority 3
   - `src/infra/cloud/mod.rs` (1,126 lines) - Priority 3
   - `src/analysis/ml/pinn/meta_learning.rs` (1,121 lines) - Priority 3
   - `src/analysis/ml/pinn/burn_wave_equation_1d.rs` (1,099 lines) - Priority 3
   - `src/math/numerics/operators/differential.rs` (1,062 lines) - Priority 3
   - `src/physics/acoustics/imaging/fusion.rs` (1,033 lines) - Priority 3
   - `src/analysis/ml/pinn/burn_wave_equation_3d.rs` (987 lines) - Priority 3
   - `src/clinical/therapy/swe_3d_workflows.rs` (975 lines) - Priority 3
   - `src/physics/optics/sonoluminescence/emission.rs` (956 lines) - Priority 3

#### Refactoring Strategy
Each large module is split into focused submodules following deep vertical hierarchy:
- **Separation of Concerns**: One responsibility per module
- **Bounded Context**: Clear domain boundaries
- **Single Source of Truth**: Shared functionality via accessor interfaces
- **Zero Duplication**: No code replication across modules

### Phase 3: Architecture Verification (4h) - ⚠️ PLANNED

#### Tasks
- [ ] Verify deep vertical hierarchy correctness
- [ ] Detect circular dependencies (`cargo tree --duplicates`)
- [ ] Check for layer violations (upward dependencies)
- [ ] Validate separation of concerns
- [ ] Audit cross-contamination between modules

#### Preliminary Results (Phase 1)
- ✅ **Zero layer violations detected** - No upward dependencies found
- ✅ **Build system healthy** - No circular dependency issues
- ✅ **Clean dependency tree** - Minor duplicates only (bitflags, getrandom)

### Phase 4: Advanced Methods Gap Analysis (8h) - 🟡 IN PROGRESS

#### Ultrasound Simulation Methods Gap Analysis
Based on comprehensive review article: "Advanced Numerical Methods for Ultrasound Simulation"

##### ✅ **Implemented Methods (Current Status)**
- [x] **FDTD (Finite Difference Time Domain)** - Basic implementation with CPML boundaries
- [x] **PSTD (Pseudospectral Time Domain)** - Full implementation with k-space dispersion correction
- [x] **FEM (Finite Element Method)** - Complete with variational boundary conditions
- [x] **BEM (Boundary Element Method)** - Full implementation with boundary integral equations
- [x] **SEM (Spectral Element Method)** - High-order method with exponential convergence

##### 🔴 **Critical Gaps - High Priority**

**1. Hybrid Coupling Methods (Weeks 1-2)**
- [ ] **FDTD-FEM Coupling**: Interface between structured/unstructured grids
- [ ] **PSTD-SEM Coupling**: Multi-scale wave propagation (coarse/fine resolution)
- [ ] **Domain Decomposition**: Schwarz alternating method for large domains
- [ ] **Adaptive Mesh Refinement**: Dynamic grid refinement for steep gradients

**2. Advanced Time Integration (Weeks 3-4)**
- [ ] **Runge-Kutta Methods**: High-order explicit schemes (RK4, SSPRK)
- [ ] **Implicit-Explicit Coupling**: IMEX schemes for stiff problems
- [ ] **Symplectic Integration**: Energy-conserving methods for long-time simulation
- [ ] **Local Time Stepping**: Efficient handling of multi-scale wave speeds

**3. Nonlinear Acoustics Enhancement (Weeks 5-6)**
- [ ] **Westervelt Equation**: Full nonlinear implementation with quadratic terms
- [ ] **Thermoviscous Effects**: Boundary layer absorption and heating
- [ ] **Shock Wave Formation**: Riemann solvers for discontinuous solutions
- [ ] **Harmonic Generation**: Second/third harmonic imaging simulation

##### 🟡 **Medium Priority Gaps (Weeks 7-10)**

**4. Multi-Physics Coupling**
- [ ] **Acousto-Optic Coupling**: Photoacoustic effect simulation
- [ ] **Thermo-Acoustic Coupling**: Temperature-dependent speed of sound
- [ ] **Electro-Acoustic Coupling**: Piezoelectric transducer modeling
- [ ] **Fluid-Structure Interaction**: Moving boundaries and interfaces

**5. Advanced Boundary Conditions**
- [ ] **Impedance Boundaries**: Frequency-dependent absorption
- [ ] **Non-Reflecting Boundaries**: Perfectly matched layers for complex media
- [ ] **Moving Boundaries**: ALE (Arbitrary Lagrangian-Eulerian) methods
- [ ] **Contact Acoustics**: Interface conditions for layered media

**6. Optimization and Control**
- [ ] **Adjoint Methods**: Gradient computation for inverse problems
- [ ] **Topology Optimization**: Optimal transducer placement
- [ ] **Waveform Optimization**: Pulse design for specific applications
- [ ] **Real-time Control**: Feedback systems for therapeutic ultrasound

##### 🟢 **Research Integration - Low Priority (Weeks 11-12)**

**7. Reference Library Integration**
- [ ] **jwave (JAX ultrasound)**: Automatic differentiation framework integration
- [ ] **k-wave (MATLAB toolbox)**: Sensor directivity models and advanced boundaries
- [ ] **Optimus**: Transducer optimization algorithms and genetic methods
- [ ] **FullWave25**: Tissue relaxation models and clinical validation data
- [ ] **Sound-Speed-Estimation**: Deep learning CNN architectures
- [ ] **dbua**: Real-time neural beamforming inference

**8. Machine Learning Enhancement**
- [ ] **Neural Operators**: Fourier Neural Operators for fast surrogate modeling
- [ ] **Physics-Informed ML**: Advanced PINN architectures with uncertainty
- [ ] **Transfer Learning**: Pre-trained models for tissue characterization
- [ ] **Reinforcement Learning**: Optimal control for therapeutic applications

##### 📊 **Implementation Priority Matrix**

| Method Category | Current Status | Priority | Complexity | Impact |
|----------------|----------------|----------|------------|---------|
| Hybrid Coupling | ❌ None | 🔴 Critical | High | Revolutionizes multi-scale simulation |
| Advanced Time Integration | ⚠️ Basic | 🔴 Critical | Medium | Enables long-time, accurate simulation |
| Nonlinear Enhancement | ⚠️ Partial | 🟡 High | Medium | Essential for high-intensity ultrasound |
| Multi-Physics Coupling | ⚠️ Basic | 🟡 High | High | Opens new application domains |
| Advanced Boundaries | ⚠️ Limited | 🟡 High | Medium | Improves accuracy for complex geometries |
| Optimization Methods | ❌ None | 🟡 Medium | High | Enables inverse problems and design |
| Research Libraries | ⚠️ Partial | 🟢 Low | Medium | Provides validation and advanced features |

##### 🎯 **Next Sprint Recommendations**

**Immediate Focus (Sprint 187):**
1. Implement FDTD-FEM coupling for multi-scale problems
2. Add Runge-Kutta time integration for improved accuracy
3. Complete Westervelt equation implementation

**Short-term (Sprints 188-190):**
1. Domain decomposition methods for large-scale simulation
2. Multi-physics coupling frameworks
3. Advanced boundary condition libraries

**Long-term (Sprints 191+):**
1. Optimization frameworks and adjoint methods
2. Machine learning integration and neural operators
3. Full research library integration and validation

### Phase 5: Quality Gates (2h) - ⚠️ PLANNED

#### Validation Checklist
- [ ] `cargo build --release --all-features` - Zero errors
- [ ] `cargo clippy --all-features -- -D warnings` - Zero warnings
- [ ] `cargo test --lib` - All tests passing
- [ ] `cargo test --lib -- --ignored` - Comprehensive validation
- [ ] `cargo bench` - Performance benchmarks
- [ ] Line count audit - All files <500 lines

### Phase 6: Documentation Updates (2h) - ⚠️ PLANNED

#### Documents to Update
- [ ] docs/prd.md - Sprint 186 status, architecture updates
- [ ] docs/srs.md - Functional requirements from research analysis
- [ ] docs/adr.md - GRASP remediation decisions
- [ ] docs/checklist.md - This file (completion summary)
- [ ] docs/backlog.md - Research integration tasks
- [ ] README.md - Current sprint status

---

## Previous Sprint: Sprint 185 - Advanced Physics Research (Multi-Bubble Interactions & Shock Physics)

**Previous Sprint**: Sprint 4 (Beamforming Consolidation) - COMPLETED ✅
**Current Focus**: Implementing 2020-2025 acoustics and optics research with mathematically verified implementations
**Next Sprints**: 186-190 (Advanced Physics Completion, Optics Research, Interdisciplinary Coupling)

---

## Sprint 185: Multi-Bubble Interactions & Shock Physics (16 hours) - DEFERRED ⏸️

**Status**: Deferred to Sprint 187+ (after architectural foundation complete)  
**Reason**: Architectural purity must be established before advanced physics extensions

### Sprint Objectives

**Primary Goal**: Implement cutting-edge bubble-bubble interaction models and shock wave physics based on 2020-2025 literature review.

**Success Criteria**:
- Multi-harmonic Bjerknes force calculator with <10% error vs. Doinikov (2021)
- Shock wave Rankine-Hugoniot solver validated against Cleveland (2022) HIFU data
- Non-spherical bubble dynamics with shape oscillations matching Shaw (2023)
- All implementations <500 lines (GRASP compliance)
- Zero placeholders, complete theorem documentation

### Task Breakdown

#### Week 1: Gap A1 - Multi-Bubble Interactions (6 hours) - 🔴 NOT STARTED
- [ ] Hour 1-2: Literature review (Lauterborn 2023, Doinikov 2021, Zhang & Li 2022)
- [ ] Hour 3-5: Implement multi-harmonic Bjerknes force calculator
  - [ ] Multi-frequency driving force coupling
  - [ ] Phase-coherent interaction topology
  - [ ] Polydisperse bubble cloud models
- [ ] Hour 6-7: Spatial clustering (octree) for O(N log N) scaling
- [ ] Hour 8-10: Validate against Doinikov 2-bubble analytical solutions
- [ ] Hour 11-12: Property-based tests (phase coherence, energy conservation)
- [ ] **Deliverable**: `src/physics/acoustics/nonlinear/multi_bubble_interactions.rs`

**Mathematical Requirements**:
```
Secondary Bjerknes Force (Multi-Frequency):
F₁₂ = -(ρ/(4πr₁₂)) ∑ₙ ∑ₘ V̇₁ⁿ V̇₂ᵐ cos(φₙ - φₘ)
```

#### Week 2: Gap A5 - Shock Wave Physics (4 hours) - 🔴 NOT STARTED
- [ ] Hour 1-2: Literature review (Cleveland 2022, Coulouvrat 2020)
- [ ] Hour 3-4: Implement Rankine-Hugoniot jump conditions
  - [ ] Shock detection algorithm (pressure gradient threshold)
  - [ ] Entropy fix for rarefaction shocks
- [ ] Hour 5-6: Adaptive mesh refinement near shocks
- [ ] Hour 7-8: Validate against HIFU experimental data (Cleveland 2022)
- [ ] Hour 9-10: Integration tests with existing FDTD solver
- [ ] **Deliverable**: `src/physics/acoustics/nonlinear/shock_physics.rs`

**Mathematical Requirements**:
```
Rankine-Hugoniot Conditions:
[ρu] = 0  (mass)
[p + ρu²] = 0  (momentum)
[E + pu/ρ] = 0  (energy)
```

#### Week 3: Gap A2 - Non-Spherical Bubble Dynamics (6 hours) - 🔴 NOT STARTED
- [ ] Hour 1-2: Literature review (Lohse & Prosperetti 2021, Shaw 2023)
- [ ] Hour 3-5: Implement spherical harmonic decomposition (n=2-10 modes)
- [ ] Hour 6-8: Mode coupling coefficients (Prosperetti 1977)
- [ ] Hour 9-10: Instability detection (Rayleigh-Taylor criteria)
- [ ] Hour 11-12: Validate against Shaw (2023) jet formation data
- [ ] **Deliverable**: `src/physics/acoustics/nonlinear/shape_oscillations.rs`

**Mathematical Requirements**:
```
Shape Perturbation Equation (Prosperetti 1977):
d²aₙ/dt² + bₙ(daₙ/dt) + ωₙ²aₙ = fₙ(t)
```

### Quality Gates - Sprint 185
- [ ] All tests passing (maintain >95% pass rate)
- [ ] Validation error <10% RMS vs. literature
- [ ] All modules <500 lines (GRASP compliance)
- [ ] Complete Rustdoc with literature references
- [ ] Zero clippy warnings
- [ ] Property-based tests for invariants

### Literature References - Sprint 185
1. Lauterborn et al. (2023). "Multi-bubble systems with collective dynamics." *Ultrasonics Sonochemistry*
2. Doinikov (2021). "Translational dynamics of bubbles." *Physics of Fluids*
3. Zhang & Li (2022). "Phase-dependent bubble interaction." *J Fluid Mechanics*
4. Cleveland et al. (2022). "Shock waves in medical ultrasound." *J Therapeutic Ultrasound*
5. Shaw (2023). "Jetting and fragmentation in sonoluminescence." *Physical Review E*
6. Lohse & Prosperetti (2021). "Shape oscillations and instabilities." *Annual Review of Fluid Mechanics*

---

## Sprint 186-190: Advanced Physics Pipeline - PLANNED

### Sprint 186: Thermal Effects & Fractional Acoustics (8 hours)
- Gap A3: Thermal effects in dense bubble clouds (3h)
- Gap A4: Fractional nonlinear acoustics (5h)

### Sprint 187: Multi-Wavelength Sonoluminescence (6 hours)
- Gap O1: Wavelength-resolved spectroscopy with Stark broadening (6h)

### Sprint 188: Photon Transport & Nonlinear Optics (8 hours)
- Gap O2: Monte Carlo photon transport (6h)
- Gap O3: Nonlinear optical effects (2h)

### Sprint 189: Interdisciplinary Coupling (6 hours)
- Gap I1: Photoacoustic feedback mechanisms (4h)
- Gap O4: Plasmonic enhancement (2h)

### Sprint 190: Validation & Documentation (12 hours)
- Comprehensive validation suite (6h)
- Property-based testing (3h)
- Documentation completion (3h)

---

---

## Sprint 186 Success Criteria

### Quantitative Metrics
- ✅ 0 dead documentation files (65 removed)
- ✅ 0 layer violations detected
- 🟡 0 files >500 lines (17 remaining, 1 in progress)
- ⚠️ 0 circular dependencies (to be verified)
- ⚠️ 100% test pass rate (to be verified)
- ⚠️ <60s full rebuild time (currently ~45s)

### Qualitative Metrics
- ✅ Clean repository root (only living docs)
- ✅ Modular solver structure established
- 🟡 Self-documenting architecture (in progress)
- ⚠️ Research-competitive feature analysis (planned)

### Risk Mitigation
- **High**: GRASP refactoring breaks tests → Using backward-compatible re-exports
- **Medium**: Time overrun → Focusing on critical violations first (Priority 1-2)
- **Low**: Documentation drift → Updating docs before commits

---

## LEGACY: Phase 1 Sprint 4: Beamforming Consolidation (COMPLETED ✅)

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

## Advanced Physics Research Audit Reference

**Primary Document**: `ACOUSTICS_OPTICS_RESEARCH_GAP_AUDIT_2025.md`
**Backlog Updates**: `docs/backlog.md` (Sprints 185-190 roadmap added)
**Gap Analysis**: 15 critical gaps identified across acoustics, optics, and interdisciplinary domains

### Gap Priority Matrix
**High Priority (Sprints 185-187)**:
- A1: Multi-bubble interactions
- A5: Shock wave physics
- O1: Multi-wavelength sonoluminescence
- O2: Photon transport

**Medium Priority (Sprints 188-189)**:
- A2: Non-spherical bubble dynamics
- A3: Thermal effects in clouds
- O3: Nonlinear optics
- I1: Photoacoustic feedback

**Low Priority (Sprint 190+)**:
- A4: Fractional acoustics
- O4: Plasmonic enhancement
- O5: Dispersive Čerenkov

---

## LEGACY: Sensor Consolidation Micro-Sprint — Array Processing Unification

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
