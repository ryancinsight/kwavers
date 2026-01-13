# Architecture Audit: Cross-Contamination Analysis

**Date**: 2025-01-27  
**Sprint**: Architecture Cleanup  
**Status**: 🔴 CRITICAL - Multiple Layer Violations Detected  
**Priority**: P0 - Blocks clean architecture goals

---

## Executive Summary

**Problem**: Significant cross-contamination exists between `math/`, `physics/`, `solver/`, and `domain/` modules, violating clean architecture principles and creating circular dependencies.

**Impact**: 
- Circular dependencies between `physics/` ↔ `solver/`
- Redundant physics specifications in `domain/physics/` vs `physics/`
- Unclear module boundaries causing confusion
- Difficult to determine where new code should live

**Root Cause**: Incremental feature additions without architectural governance led to:
1. Physics specifications duplicated in domain layer
2. Solvers depending on physics implementations (correct)
3. Physics depending on solvers (incorrect - circular dependency)
4. Domain containing application-level concerns (imaging, therapy)

---

## 🎯 Desired Architecture (Target State)

### Clean Dependency Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                        │
│                 (clinical/, simulation/)                    │
│                                                             │
│  Clinical workflows, imaging protocols, therapy planning   │
└────────────────────────┬────────────────────────────────────┘
                         │ uses
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    ANALYSIS LAYER                           │
│              (analysis/, beamforming, ML)                   │
│                                                             │
│  Signal processing, image reconstruction, AI/ML             │
└────────────────────────┬────────────────────────────────────┘
                         │ uses
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     SOLVER LAYER                            │
│        (solver/forward/, solver/inverse/)                   │
│                                                             │
│  FDTD, PSTD, DG, PINN, analytical solvers                  │
│  Shared interfaces for simulation/ and clinical/           │
└────────────────────────┬────────────────────────────────────┘
                         │ uses
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    PHYSICS LAYER                            │
│    (physics/acoustics/, physics/optics/, etc.)             │
│                                                             │
│  Wave equations, conservation laws, material models         │
│  NO solver dependencies, only defines physics               │
└────────────────────────┬────────────────────────────────────┘
                         │ uses
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    DOMAIN LAYER                             │
│  (domain/grid/, domain/medium/, domain/sensor/, etc.)      │
│                                                             │
│  Pure domain entities: spatial grids, materials, geometry   │
│  NO physics equations, NO solvers, NO applications          │
└────────────────────────┬────────────────────────────────────┘
                         │ uses
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      MATH LAYER                             │
│      (math/fft/, math/linear_algebra/, etc.)               │
│                                                             │
│  Pure mathematical operations, no domain knowledge          │
└─────────────────────────────────────────────────────────────┘
                         │ uses
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      CORE LAYER                             │
│              (core/error/, core/types/)                     │
│                                                             │
│  Fundamental types, error handling, constants               │
└─────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | Responsibility | What It Contains | What It MUST NOT Contain |
|-------|---------------|------------------|-------------------------|
| **math/** | Pure mathematics | FFT, linear algebra, geometry, numerical methods | Domain concepts, physics equations |
| **physics/** | Physics models | Wave equations, material models, conservation laws | Solver implementations, numerical discretization |
| **solver/** | Numerical methods | FDTD, PSTD, PINN, analytical solvers, shared interfaces | Physics definitions (uses physics layer) |
| **domain/** | Domain entities | Grid, medium, sensors, sources, boundaries | Physics equations, solvers, applications |
| **analysis/** | Signal processing | Beamforming, image reconstruction, ML/AI | Simulation orchestration |
| **simulation/** | Orchestration | Simulation setup, execution, coordination | Clinical workflows |
| **clinical/** | Applications | Medical imaging, therapy planning, safety | Low-level physics/solver details |

---

## 🔴 Current State: Violations Detected

### Violation 1: `domain/physics/` Redundancy

**Problem**: `domain/physics/` contains physics specifications that overlap with `physics/`.

**Evidence**:
```
domain/physics/
├── wave_equation.rs     ← Wave equation traits and specs
├── coupled.rs           ← Multi-physics coupling
├── electromagnetic.rs   ← EM wave specs
├── nonlinear.rs         ← Nonlinear wave specs
└── plasma.rs            ← Plasma physics specs

physics/
├── acoustics/           ← Acoustic wave implementations
├── electromagnetic/     ← EM wave implementations  
├── nonlinear/           ← Nonlinear implementations
└── optics/              ← Optical physics
```

**Analysis**:
- `domain/physics/wave_equation.rs` defines `WaveEquation`, `AcousticWaveEquation`, `ElasticWaveEquation` traits
- These are physics specifications, not domain entities
- Should live in `physics/` as the canonical source
- Domain layer should only reference physics types, not define them

**Impact**: 
- Confusion about where to implement new wave physics
- Potential for drift between specifications and implementations
- Violates Single Source of Truth (SSOT) principle

### Violation 2: Physics ↔ Solver Circular Dependency

**Problem**: `physics/` and `solver/` have bidirectional dependencies.

**Evidence**:
```rust
// physics/electromagnetic/solvers.rs (VIOLATION)
use crate::solver::forward::fdtd::ElectromagneticFdtdSolver;

// solver/forward/elastic/plugin.rs (CORRECT)
use crate::physics::mechanics::elastic_wave::ElasticWave;

// solver/forward/poroelastic.rs (CORRECT)
use crate::physics::mechanics::poroelastic::{BiotTheory, PoroelasticMaterial};
```

**Analysis**:
- ✅ **Correct**: Solvers importing physics models (17 instances)
- ❌ **Incorrect**: Physics importing solvers (2 instances)
- Physics layer should define equations and models only
- Solver layer should implement numerical methods using physics definitions

**Impact**:
- Prevents independent physics/solver evolution
- Creates compilation ordering issues
- Violates Dependency Inversion Principle (DIP)

### Violation 3: Domain Layer Scope Creep

**Problem**: `domain/` contains application-level concerns that should be in higher layers.

**Evidence**:
```
domain/
├── imaging/          ← Should be in analysis/ or simulation/
├── signal/           ← Should be in analysis/
├── therapy/          ← Should be in clinical/
├── physics/          ← Should be in physics/ (as shown above)
└── sensor/beamforming/  ← Partially migrated to analysis/, still has remnants
```

**Analysis**:
- **imaging/**: Image reconstruction algorithms are analysis-layer concerns
- **signal/**: Signal processing is analysis-layer functionality
- **therapy/**: Therapy planning is application-layer (clinical/)
- Domain should contain only: grid, medium, sensors (hardware), sources, boundaries, field containers

**Impact**:
- Blurred layer boundaries
- Difficult to determine correct location for new features
- Violates Single Responsibility Principle (SRP)

### Violation 4: Inconsistent Import Patterns

**Problem**: Mixed import patterns indicate unclear architectural boundaries.

**Evidence**:
```rust
// GOOD: domain uses math (12 instances found)
use crate::math::linear_algebra::LinearAlgebra;
use crate::math::fft::kspace;
use crate::math::geometry::distance3;

// GOOD: solver uses physics (17 instances)
use crate::physics::mechanics::elastic_wave::ElasticWave;
use crate::physics::thermal::diffusion::ThermalDiffusionConfig;

// BAD: physics uses solver (2 instances - CIRCULAR DEPENDENCY)
use crate::solver::forward::fdtd::ElectromagneticFdtdSolver;

// UNCLEAR: domain/physics exists alongside physics/
use crate::domain::physics::{WaveEquation, BoundaryCondition};
use crate::physics::acoustics::mechanics::absorption::AbsorptionMode;
```

---

## 📊 Quantitative Analysis

### Dependency Metrics

| Relationship | Count | Status | Notes |
|-------------|-------|--------|-------|
| `domain/` → `math/` | 12 | ✅ Correct | Clean mathematical dependencies |
| `solver/` → `physics/` | 17 | ✅ Correct | Solvers use physics models |
| `physics/` → `solver/` | 2 | ❌ Violation | Creates circular dependency |
| `domain/physics/` existence | 1 | ❌ Violation | Redundant with `physics/` |
| `domain/imaging/` existence | 1 | ⚠️ Questionable | Should be analysis layer |
| `domain/therapy/` existence | 1 | ⚠️ Questionable | Should be clinical layer |

### Module Size Analysis (GRASP Compliance)

Files requiring refactoring due to >500 line GRASP violations:
- 15 files exceed 500 lines (documented in checklist.md)
- Priority 1: `math/linear_algebra/mod.rs` (1,889 lines)
- Related to architectural issues: Large files often span multiple responsibilities

---

## 🎯 Refactoring Strategy

### Phase 1: Merge `domain/physics/` → `physics/` (4 hours)

**Goal**: Eliminate redundant physics specifications in domain layer.

**Tasks**:
1. ✅ Audit `domain/physics/` modules:
   - `wave_equation.rs` - Core physics trait specifications
   - `coupled.rs` - Multi-physics coupling interfaces
   - `electromagnetic.rs` - EM wave specifications
   - `nonlinear.rs` - Nonlinear physics specs
   - `plasma.rs` - Plasma physics specs

2. **Merge Strategy**:
   ```
   domain/physics/wave_equation.rs → physics/foundations/wave_equation.rs
   domain/physics/coupled.rs       → physics/foundations/coupling.rs
   domain/physics/electromagnetic.rs → physics/electromagnetic/equations.rs
   domain/physics/nonlinear.rs     → physics/nonlinear/equations.rs
   domain/physics/plasma.rs        → physics/optics/plasma.rs
   ```

3. **Update all imports**:
   - Find: `use crate::domain::physics::`
   - Replace: `use crate::physics::foundations::`
   - Verify: Zero compilation errors

4. **Delete `domain/physics/` directory**

**Success Criteria**:
- ✅ All physics specifications in `physics/` only
- ✅ No `domain/physics/` module exists
- ✅ All tests pass
- ✅ Zero compilation errors
- ✅ Documentation updated (ADR entry)

### Phase 2: Break Physics → Solver Circular Dependency (2 hours)

**Goal**: Ensure unidirectional dependency: `solver/` → `physics/` only.

**Tasks**:
1. **Identify violations**:
   - `physics/electromagnetic/solvers.rs` imports `solver/forward/fdtd/`
   - `physics/acoustics/mechanics/poroelastic/mod.rs` references solver

2. **Refactor approach**:
   ```rust
   // BEFORE (VIOLATION):
   // physics/electromagnetic/solvers.rs
   use crate::solver::forward::fdtd::ElectromagneticFdtdSolver;
   
   // AFTER (CORRECT):
   // solver/forward/fdtd/electromagnetic.rs
   use crate::physics::electromagnetic::{EMFields, EMMaterialProperties};
   
   // physics/electromagnetic/mod.rs
   // NO solver imports - only define physics models
   ```

3. **Move solver-specific code**:
   - `physics/electromagnetic/solvers.rs` → `solver/forward/fdtd/electromagnetic.rs`
   - Update public API to expose solver in correct layer

4. **Verify dependency flow**:
   ```bash
   cargo tree --edges normal --depth 1 | grep "physics ->"
   # Should NOT show: physics -> solver
   ```

**Success Criteria**:
- ✅ Zero `use crate::solver::` in `physics/` modules
- ✅ All solver implementations in `solver/` layer
- ✅ Physics layer defines models only
- ✅ Dependency graph is acyclic

### Phase 3: Domain Layer Cleanup (4 hours)

**Goal**: Move application concerns out of domain layer.

**Tasks**:
1. **Audit domain submodules**:
   ```
   domain/imaging/   → analysis/imaging/
   domain/signal/    → analysis/signal_processing/
   domain/therapy/   → clinical/therapy/
   ```

2. **Retain pure domain entities**:
   ```
   domain/
   ├── grid/         ✅ Keep - spatial discretization
   ├── medium/       ✅ Keep - material properties
   ├── sensor/       ✅ Keep - sensor hardware (NOT beamforming algorithms)
   ├── source/       ✅ Keep - acoustic sources
   ├── boundary/     ✅ Keep - boundary conditions
   ├── field/        ✅ Keep - field data containers
   ├── tensor/       ✅ Keep - data storage abstractions
   ├── mesh/         ✅ Keep - computational meshes
   └── geometry/     ✅ Keep - geometric primitives
   ```

3. **Migration plan**:
   - Move `domain/imaging/` to `analysis/imaging/` (or deprecate if duplicate)
   - Move `domain/signal/` to `analysis/signal_processing/` (already exists - check for duplication)
   - Move `domain/therapy/` to `clinical/therapy/` (check if already exists)
   - Clean up `domain/sensor/beamforming/` (migrate remaining to `analysis/signal_processing/beamforming/`)

4. **Update imports and deprecations**

**Success Criteria**:
- ✅ Domain contains only domain entities (grid, medium, sensors, sources, boundaries)
- ✅ No application logic in domain layer
- ✅ Clear deprecation notices for moved modules
- ✅ Migration guide created

### Phase 4: Shared Solver Interfaces (3 hours)

**Goal**: Create clean, shared solver interfaces for `simulation/` and `clinical/` consumers.

**Tasks**:
1. **Define canonical solver traits** in `solver/interface/`:
   ```rust
   // solver/interface/acoustic.rs
   pub trait AcousticSolver {
       fn step(&mut self, dt: f64) -> KwaversResult<()>;
       fn get_pressure_field(&self) -> &Array3<f64>;
       fn set_source(&mut self, source: &dyn Source);
   }
   
   // solver/interface/elastic.rs
   pub trait ElasticSolver {
       fn step(&mut self, dt: f64) -> KwaversResult<()>;
       fn get_displacement_field(&self) -> &Array3<Vector3<f64>>;
   }
   ```

2. **Implement traits for all solvers**:
   - FDTD, PSTD, DG solvers implement `AcousticSolver`
   - Elastic wave solvers implement `ElasticSolver`
   - PINN solvers implement appropriate trait

3. **Create solver factory** for high-level consumers:
   ```rust
   // solver/factory.rs
   pub fn create_acoustic_solver(
       config: SolverConfig,
       grid: &Grid,
       medium: &dyn Medium,
   ) -> Box<dyn AcousticSolver>;
   ```

4. **Update simulation/ and clinical/ to use interfaces**:
   - Replace direct solver instantiation with factory
   - Use trait objects for solver abstraction

**Success Criteria**:
- ✅ Common solver traits defined
- ✅ All solvers implement appropriate traits
- ✅ `simulation/` and `clinical/` use shared interfaces
- ✅ Easy to add new solver implementations

### Phase 5: Documentation & Validation (2 hours)

**Goal**: Document architecture decisions and validate correctness.

**Tasks**:
1. **Create ADR entries**:
   - ADR-024: Physics Layer Consolidation
   - ADR-025: Unidirectional Solver Dependencies
   - ADR-026: Domain Layer Scope Definition
   - ADR-027: Shared Solver Interfaces

2. **Update architecture documentation**:
   - `docs/architecture.md` with layer diagrams
   - `README.md` architecture section
   - Module-level rustdoc with layer positioning

3. **Validation**:
   - Cargo tree analysis (verify no circular deps)
   - Full test suite pass
   - Documentation build pass
   - Clippy clean

4. **Create migration guide** for external consumers

**Success Criteria**:
- ✅ Complete ADR documentation
- ✅ Architecture diagrams updated
- ✅ All tests pass
- ✅ Zero clippy warnings
- ✅ Migration guide published

---

## 📋 Implementation Checklist

### Phase 1: Physics Consolidation (4h)
- [ ] Audit `domain/physics/` modules
- [ ] Create `physics/foundations/` module
- [ ] Move `domain/physics/wave_equation.rs` → `physics/foundations/wave_equation.rs`
- [ ] Move `domain/physics/coupled.rs` → `physics/foundations/coupling.rs`
- [ ] Move `domain/physics/electromagnetic.rs` → `physics/electromagnetic/equations.rs`
- [ ] Move `domain/physics/nonlinear.rs` → `physics/nonlinear/equations.rs`
- [ ] Move `domain/physics/plasma.rs` → `physics/optics/plasma.rs`
- [ ] Update all `use crate::domain::physics::` imports
- [ ] Delete `domain/physics/` directory
- [ ] Run full test suite
- [ ] Update documentation

### Phase 2: Break Circular Dependencies (2h)
- [ ] Identify all `physics/` → `solver/` imports
- [ ] Move `physics/electromagnetic/solvers.rs` → `solver/forward/fdtd/electromagnetic.rs`
- [ ] Remove solver references from `physics/acoustics/mechanics/poroelastic/`
- [ ] Verify zero `use crate::solver::` in `physics/`
- [ ] Run dependency analysis: `cargo tree`
- [ ] Run full test suite
- [ ] Create ADR-025

### Phase 3: Domain Cleanup (4h)
- [ ] Audit `domain/imaging/` - migrate or deprecate
- [ ] Audit `domain/signal/` - migrate or deprecate
- [ ] Audit `domain/therapy/` - migrate to `clinical/therapy/`
- [ ] Clean up `domain/sensor/beamforming/` remnants
- [ ] Update imports across codebase
- [ ] Add deprecation warnings
- [ ] Run full test suite
- [ ] Create migration guide

### Phase 4: Shared Interfaces (3h)
- [ ] Define `solver/interface/acoustic.rs` trait
- [ ] Define `solver/interface/elastic.rs` trait
- [ ] Implement traits for FDTD solver
- [ ] Implement traits for PSTD solver
- [ ] Implement traits for elastic solver
- [ ] Create `solver/factory.rs`
- [ ] Update `simulation/` to use interfaces
- [ ] Update `clinical/` to use interfaces
- [ ] Run full test suite

### Phase 5: Documentation (2h)
- [ ] Write ADR-024: Physics Layer Consolidation
- [ ] Write ADR-025: Unidirectional Solver Dependencies
- [ ] Write ADR-026: Domain Layer Scope Definition
- [ ] Write ADR-027: Shared Solver Interfaces
- [ ] Update `docs/architecture.md`
- [ ] Update `README.md`
- [ ] Update module rustdoc
- [ ] Create migration guide
- [ ] Verify documentation builds
- [ ] Run final validation suite

---

## 🎯 Success Metrics

### Architectural Purity
- ✅ Zero circular dependencies (verified by cargo tree)
- ✅ All physics in `physics/` layer only
- ✅ All solvers in `solver/` layer only
- ✅ Domain contains only entities, no logic
- ✅ Clear layer boundaries in module structure

### Code Quality
- ✅ All tests pass (867/867 current baseline)
- ✅ Zero clippy warnings with `-D warnings`
- ✅ Zero compilation errors
- ✅ Documentation builds without errors
- ✅ GRASP compliance maintained

### Developer Experience
- ✅ Clear architectural documentation
- ✅ Obvious module placement rules
- ✅ Migration guide for consumers
- ✅ ADR entries for future reference
- ✅ Improved compile times (fewer dependencies)

---

## 📚 References

### Architectural Principles
- **Clean Architecture** (Robert C. Martin): Dependency rule - outer layers depend on inner
- **Domain-Driven Design** (Eric Evans): Bounded contexts, ubiquitous language
- **SOLID Principles**: Especially Dependency Inversion Principle (DIP)
- **GRASP Patterns**: Information Expert, Low Coupling, High Cohesion

### Rust-Specific
- **Cargo Book**: Dependency management, workspace organization
- **API Guidelines**: Module organization, re-exports
- **Rustonomicon**: Unsafe code guidelines (for validation)

### Project Documents
- `docs/adr.md`: Architecture Decision Records
- `docs/prd.md`: Product Requirements
- `docs/srs.md`: Software Requirements Specification
- `docs/checklist.md`: Sprint progress tracking
- `docs/backlog.md`: Strategic planning

---

## 🚀 Estimated Effort

| Phase | Duration | Priority | Dependencies |
|-------|----------|----------|--------------|
| Phase 1: Physics Consolidation | 4 hours | P0 | None |
| Phase 2: Break Circular Deps | 2 hours | P0 | Phase 1 |
| Phase 3: Domain Cleanup | 4 hours | P1 | Phase 1 |
| Phase 4: Shared Interfaces | 3 hours | P1 | Phase 2, 3 |
| Phase 5: Documentation | 2 hours | P0 | All phases |
| **Total** | **15 hours** | - | - |

**Suggested Sprint**: 2 week sprint with daily 1-2 hour focused sessions

---

## 🔄 Risk Mitigation

### Risk 1: Breaking Changes
**Mitigation**: 
- Use deprecation warnings before deletion
- Create comprehensive migration guide
- Keep backward compatibility shims for 1 version

### Risk 2: Test Failures
**Mitigation**:
- Run tests after each phase
- Use feature flags for gradual rollout
- Keep baseline test count (867 tests)

### Risk 3: Import Resolution Errors
**Mitigation**:
- Use IDE refactoring tools
- Grep-based verification of import patterns
- Staged commits per module migration

### Risk 4: Performance Regression
**Mitigation**:
- Run benchmark suite after changes
- Monitor compilation times
- Profile critical paths

---

## 📝 Notes

- This audit identified **4 major architectural violations**
- Estimated **15 hours** to achieve architectural purity
- High ROI: Improved developer experience, faster compile times, clearer boundaries
- Aligns with Sprint 4 beamforming consolidation methodology (SSOT enforcement)
- Builds foundation for future feature additions with clear placement rules

**Next Steps**: Review this audit with stakeholders, prioritize phases, begin Phase 1.