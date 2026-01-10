# Architectural Refactoring Plan: Deep Vertical Hierarchy with SSOT

**Date**: 2025-01-11  
**Status**: ACTIVE  
**Priority**: P0 - Critical for maintainability and scalability  

## Executive Summary

This document outlines the architectural refactoring required to eliminate code duplication, establish proper layer separation, and create a deep vertical hierarchical file tree that embodies Single Source of Truth (SSOT) principles.

### Current State
- ✅ **Build Status**: All builds pass (library, examples, benchmarks)
- ✅ **Test Status**: 960 tests passing, 10 ignored
- ❌ **Architecture**: Significant code duplication across layers
- ❌ **SSOT Violations**: Duplicate modules in multiple locations

### Target State
- Eliminate all duplicate modules
- Establish clear layer boundaries with unidirectional dependencies
- Create self-documenting deep vertical file hierarchy
- Zero namespace bleeding through proper module boundaries
- Shared components accessed through well-defined interfaces

---

## Critical Issues Identified

### 1. Duplicate Math Modules (P0 - CRITICAL)

**Problem**: Identical math code exists in TWO locations:
- `src/math/` (17 files) - ✅ Correct location
- `src/domain/math/` (17 files) - ❌ Architectural violation

**Files Affected**: 17 identical files including:
- `fft/mod.rs`, `fft/fft_processor.rs`, `fft/kspace.rs`, `fft/utils.rs`
- `geometry/mod.rs`, `geometry/primitives.rs`, `geometry/spatial.rs`
- `linear_algebra/mod.rs`, `linear_algebra/sparse/*.rs`
- `numerics/mod.rs`, `numerics/integration/*.rs`, `numerics/operators/*.rs`, `numerics/transforms/*.rs`

**Impact**: 40+ files import from `domain::math::*`, causing maintenance burden and confusion

**Resolution**:
1. Delete `src/domain/math/` entirely
2. Update all imports from `crate::domain::math::*` to `crate::math::*`
3. Re-export math types at domain boundary ONLY if needed by domain

### 2. Duplicate Core Modules (P0 - CRITICAL)

**Problem**: Core infrastructure duplicated:
- `src/core/` - ✅ Correct location
- `src/domain/core/` - ❌ Architectural violation

**Modules Duplicated**:
- `error/` - Error handling and result types
- `constants/` - Physical and numerical constants
- `time/` - Time representation
- `utils/` - Utility functions

**Impact**: Inconsistent error handling, multiple sources of truth for constants

**Resolution**:
1. Delete `src/domain/core/` entirely
2. Update all imports from `crate::domain::core::*` to `crate::core::*`
3. Domain layer depends on core layer directly

### 3. Beamforming Layer Violations (P1 - HIGH)

**Problem**: Beamforming logic exists in multiple layers:
- `src/analysis/signal_processing/beamforming/` - ✅ Correct (analysis layer)
- `src/domain/sensor/beamforming/` - ⚠️ Deprecated/legacy code

**Status**: According to README, beamforming consolidation (Sprint 4) already moved core algorithms to analysis layer

**Resolution**:
1. Verify `domain/sensor/beamforming/` only contains thin configuration wrappers
2. If contains algorithm implementations, move to analysis layer
3. Delete any deprecated/unused beamforming code from domain

### 4. Therapy Logic Fragmentation (P2 - MEDIUM)

**Problem**: Therapy concerns scattered across:
- `src/clinical/therapy/` - Application workflows
- `src/domain/therapy/` - Domain types (parameters, metrics)
- `src/physics/acoustics/therapy/` - Physics models
- `src/simulation/therapy/` - Simulation coordination

**Resolution**:
1. Keep domain types (parameters, metrics) in `domain/therapy/`
2. Keep physics models in `physics/acoustics/therapy/`
3. Keep simulation coordination in `simulation/therapy/`
4. Keep clinical workflows in `clinical/therapy/`
5. Each layer accesses only layers below it

### 5. Imaging Logic Fragmentation (P2 - MEDIUM)

**Problem**: Imaging concerns scattered across:
- `src/clinical/imaging/` - Application workflows
- `src/domain/imaging/` - Domain types
- `src/physics/acoustics/imaging/` - Physics models
- `src/simulation/imaging/` - Simulation coordination

**Resolution**: Similar to therapy - maintain layer separation

---

## Architectural Layers: Correct Hierarchy

### Layer 0: Core (Foundation)
**Path**: `src/core/`  
**Dependencies**: NONE  
**Purpose**: Foundational infrastructure

```
core/
├── error/           # Error types, Result, validation
├── constants/       # Physical and numerical constants
├── time/            # Time representation
└── utils/           # Generic utilities
```

### Layer 1: Math (Pure Mathematics)
**Path**: `src/math/`  
**Dependencies**: `core`  
**Purpose**: Pure mathematical operations

```
math/
├── fft/                    # Fast Fourier Transform
│   ├── fft_processor.rs   # FFT implementations
│   ├── kspace.rs          # K-space calculations
│   └── utils.rs           # FFT utilities
├── geometry/              # Geometric primitives
├── linear_algebra/        # Linear algebra operations
│   └── sparse/           # Sparse matrix operations
└── numerics/             # Numerical methods
    ├── integration/      # Numerical integration
    ├── operators/        # Differential operators
    └── transforms/       # Mathematical transforms
```

### Layer 2: Domain (Domain Model)
**Path**: `src/domain/`  
**Dependencies**: `core`, `math`  
**Purpose**: Core domain entities and business logic

```
domain/
├── grid/              # Computational grid
├── medium/            # Material properties
│   ├── homogeneous/
│   ├── heterogeneous/
│   └── anisotropic/
├── source/            # Acoustic sources
│   └── transducers/
├── sensor/            # Sensors and recording
│   └── recorder/
├── signal/            # Signal definitions
├── boundary/          # Boundary conditions
│   └── cpml/         # CPML implementation
├── field/             # Field management
├── therapy/           # Therapy parameters
└── imaging/           # Imaging parameters
```

**Note**: NO math subdirectory, NO core subdirectory, NO beamforming algorithms

### Layer 3: Physics (Physics Models)
**Path**: `src/physics/`  
**Dependencies**: `core`, `math`, `domain`  
**Purpose**: Physical models and equations

```
physics/
├── acoustics/        # Acoustic physics
│   ├── mechanics/   # Wave mechanics
│   ├── imaging/     # Imaging physics
│   └── therapy/     # Therapy physics
├── thermal/          # Heat transfer
├── optics/           # Light physics
└── chemistry/        # Chemical reactions
```

### Layer 4: Solver (Numerical Solvers)
**Path**: `src/solver/`  
**Dependencies**: `core`, `math`, `domain`, `physics`  
**Purpose**: Numerical solution methods

```
solver/
├── forward/          # Forward solvers
│   ├── fdtd/        # FDTD method
│   ├── pstd/        # PSTD method
│   └── nonlinear/   # Nonlinear solvers
├── inverse/          # Inverse problems
└── utilities/        # Solver utilities
```

### Layer 5: Analysis (Analysis & Processing)
**Path**: `src/analysis/`  
**Dependencies**: All layers below  
**Purpose**: Signal processing, ML, analysis tools

```
analysis/
├── signal_processing/    # Signal processing
│   └── beamforming/     # ✅ SSOT for beamforming
│       ├── adaptive/    # Adaptive beamforming
│       ├── narrowband/  # Narrowband methods
│       ├── neural/      # Neural beamforming
│       └── utils/       # Shared utilities
├── ml/                  # Machine learning
│   ├── models/
│   ├── pinn/
│   └── optimization/
├── performance/         # Performance analysis
└── validation/          # Validation tools
```

### Layer 6: Simulation (Simulation Orchestration)
**Path**: `src/simulation/`  
**Dependencies**: All layers below  
**Purpose**: Coordinate multi-physics simulations

```
simulation/
├── imaging/         # Imaging simulation
├── therapy/         # Therapy simulation
└── modalities/      # Simulation modalities
```

### Layer 7: Clinical (Clinical Applications)
**Path**: `src/clinical/`  
**Dependencies**: All layers below  
**Purpose**: Clinical workflows and applications

```
clinical/
├── imaging/         # Clinical imaging workflows
└── therapy/         # Clinical therapy workflows
```

### Layer 8: Infrastructure (External Interfaces)
**Path**: `src/infra/`  
**Dependencies**: Can access any layer  
**Purpose**: I/O, API, cloud services

```
infra/
├── api/             # REST API
├── io/              # File I/O
├── cloud/           # Cloud services
└── runtime/         # Runtime management
```

### Layer 9: GPU (GPU Acceleration)
**Path**: `src/gpu/`  
**Dependencies**: Can access computational layers  
**Purpose**: GPU-accelerated operations

```
gpu/
├── memory/          # GPU memory management
└── shaders/         # Compute shaders
```

---

## Migration Strategy

### Phase 1: Eliminate Duplicate Math Module (Immediate)

**Status**: READY TO EXECUTE  
**Risk**: LOW (well-contained changes)  
**Estimated Time**: 2-3 hours

#### Steps:
1. ✅ Verify `src/math/` and `src/domain/math/` are identical (CONFIRMED)
2. Create migration script to update imports:
   ```rust
   // OLD (incorrect):
   use crate::domain::math::fft::{fft_3d_array, ifft_3d_array};
   use crate::domain::math::numerics::operators::*;
   
   // NEW (correct):
   use crate::math::fft::{fft_3d_array, ifft_3d_array};
   use crate::math::numerics::operators::*;
   ```
3. Run automated find-replace across codebase
4. Delete `src/domain/math/` directory
5. Update `src/domain/mod.rs` to remove `pub mod math;`
6. Verify all tests pass

#### Files to Update (40+ files):
- `src/infra/api/job_manager.rs`
- `src/physics/acoustics/mechanics/acoustic_wave/nonlinear/numerical_methods.rs`
- `src/physics/acoustics/mechanics/elastic_wave/spectral_fields.rs`
- `src/physics/acoustics/transducer/field_calculator.rs`
- `src/solver/analytical/transducer/fast_nearfield.rs`
- `src/solver/forward/fdtd/solver.rs`
- `src/solver/forward/hybrid/adaptive_selection/metrics.rs`
- `src/solver/forward/hybrid/mixed_domain.rs`
- `src/solver/forward/nonlinear/*/` (multiple files)
- `src/solver/forward/pstd/data.rs`
- ...and 30+ more files

### Phase 2: Eliminate Duplicate Core Module (Immediate)

**Status**: READY TO EXECUTE  
**Risk**: LOW (core is stable)  
**Estimated Time**: 1-2 hours

#### Steps:
1. Verify `src/core/` and `src/domain/core/` differences
2. If identical, delete `src/domain/core/` entirely
3. Update imports from `crate::domain::core::*` to `crate::core::*`
4. Update `src/domain/mod.rs` to remove `pub mod core;`
5. Verify all tests pass

### Phase 3: Consolidate Beamforming (Verification)

**Status**: NEEDS AUDIT  
**Risk**: LOW (mostly done in Sprint 4)  
**Estimated Time**: 1-2 hours

#### Steps:
1. Audit `src/domain/sensor/beamforming/` contents
2. Verify only thin configuration wrappers remain
3. If algorithm code found, migrate to `src/analysis/signal_processing/beamforming/`
4. Remove deprecated code
5. Document migration path in deprecation notices

### Phase 4: Clean Up Obsolete Files (Ongoing)

**Status**: CONTINUOUS  
**Risk**: VERY LOW  
**Estimated Time**: Ongoing

#### Targets:
- Remove all `.md` audit/report files from root (move to `docs/audits/`)
- Remove stale test outputs
- Remove deprecated code paths
- Clean up temporary files

---

## Dependency Rules

### Strict Unidirectional Dependencies

```
Clinical    →  Simulation  →  Analysis  →  Solver  →  Physics  →  Domain  →  Math  →  Core
   ↑              ↑             ↑           ↑          ↑          ↑         ↑        ↑
   └──────────────┴─────────────┴───────────┴──────────┴──────────┴─────────┴────────┘
                            Infrastructure & GPU (cross-cutting)
```

### Rules:
1. **Lower layers NEVER import from higher layers**
2. **Core has NO dependencies**
3. **Math only depends on Core**
4. **Domain only depends on Core + Math**
5. **Physics depends on Core + Math + Domain**
6. **Solver depends on Core + Math + Domain + Physics**
7. **Analysis depends on all computational layers**
8. **Simulation depends on all layers below**
9. **Clinical depends on all layers below**
10. **Infrastructure can cross-cut but should minimize dependencies**

### Import Patterns (Correct):
```rust
// ✅ CORRECT: Lower layer importing from even lower layer
// physics/acoustics/mechanics.rs
use crate::core::error::KwaversResult;
use crate::math::numerics::operators::Laplacian;
use crate::domain::grid::Grid;

// ✅ CORRECT: Analysis importing from solver
// analysis/signal_processing/beamforming/mod.rs
use crate::solver::forward::pstd::PSTDSolver;
use crate::domain::sensor::GridSensorSet;

// ❌ WRONG: Lower layer importing from higher layer
// domain/grid/mod.rs
use crate::analysis::ml::ModelMetadata;  // NEVER!

// ❌ WRONG: Circular dependencies
// physics/acoustics.rs imports solver::pstd
// solver::pstd imports physics::acoustics
```

---

## Shared Component Access Pattern

### Problem: Common Logic Reuse

Instead of duplicating code or creating circular dependencies, use **accessor interfaces**:

### Pattern: Shared Accessors in Lower Layers

```rust
// domain/medium/accessors.rs (correct location: shared domain logic)
pub trait MediumAccessor {
    fn sound_speed_at(&self, x: f64, y: f64, z: f64) -> f64;
    fn density_at(&self, x: f64, y: f64, z: f64) -> f64;
    fn attenuation_at(&self, x: f64, y: f64, z: f64, freq: f64) -> f64;
}

// Upper layers use accessor, don't reimplement
// physics/acoustics/wave.rs
fn compute_impedance<M: MediumAccessor>(medium: &M, x: f64, y: f64, z: f64) -> f64 {
    medium.sound_speed_at(x, y, z) * medium.density_at(x, y, z)
}
```

### Anti-Pattern: Duplication

```rust
// ❌ WRONG: Reimplementing shared logic in multiple places
// physics/acoustics/wave.rs
fn get_sound_speed(medium: &dyn Medium, x: f64, y: f64, z: f64) -> f64 {
    // Duplicated logic from domain layer
}

// therapy/calculator.rs
fn get_sound_speed(medium: &dyn Medium, x: f64, y: f64, z: f64) -> f64 {
    // Same duplicated logic again!
}
```

---

## File Size Constraints

### GRASP Compliance: Modules < 500 lines

**Current Violations**: To be audited  
**Target**: All modules under 500 lines

#### Enforcement:
1. CI check: Fail build if any `.rs` file > 500 lines
2. Split large modules using vertical decomposition
3. Use subdirectories for complex modules

#### Example Refactoring:
```
# BEFORE: Large monolithic file
solver/forward/pstd/mod.rs (850 lines) ❌

# AFTER: Vertical decomposition
solver/forward/pstd/
├── mod.rs (50 lines) ✅           # Public API
├── solver.rs (300 lines) ✅       # Core solver
├── propagator.rs (250 lines) ✅   # Propagation logic
├── numerics/ (250 lines) ✅       # Numerical methods
└── physics/ (200 lines) ✅        # Physics models
```

---

## Testing Strategy

### Test Organization Mirrors Source Structure

```
tests/
├── unit/                    # Mirror src/ structure
│   ├── core/
│   ├── math/
│   ├── domain/
│   └── ...
├── integration/             # Cross-layer integration
│   ├── beamforming_integration.rs
│   └── therapy_workflow.rs
├── validation/              # Physics validation
│   ├── literature_benchmarks.rs
│   └── analytical_solutions.rs
└── performance/             # Performance tests
    └── critical_paths.rs
```

### Test Classification:
- **Tier 1**: Fast unit tests (<10s) - Always run
- **Tier 2**: Standard tests (<30s) - Run on PR
- **Tier 3**: Comprehensive validation (>30s) - Run on release

---

## Success Metrics

### Quantitative Targets:
- ✅ Zero duplicate modules (0 files with identical content in different locations)
- ✅ Zero circular dependencies (verified by cargo build)
- ✅ Zero cross-layer violations (enforced by module visibility)
- ✅ 100% GRASP compliance (all files < 500 lines)
- ✅ 100% test pass rate (maintain current 960 passing tests)

### Qualitative Targets:
- Self-documenting file tree (directory structure reveals architecture)
- Clear separation of concerns (each module has single responsibility)
- Minimal cognitive load (developers know where to find/place code)
- Zero namespace bleeding (explicit imports, no glob re-exports)

---

## Rollback Plan

### If Issues Arise:
1. All changes in feature branches
2. Atomic commits for each phase
3. Full test suite run before merge
4. Git tags for each phase completion
5. Easy rollback: `git revert <phase-tag>`

### Risk Mitigation:
- Phase 1 & 2 are LOW RISK (pure refactoring, no logic changes)
- All tests must pass before proceeding to next phase
- Deprecation warnings guide migration for external users

---

## Timeline

### Immediate (This Sprint):
- ✅ Phase 1: Eliminate duplicate math module (2-3 hours)
- ✅ Phase 2: Eliminate duplicate core module (1-2 hours)
- ⏳ Phase 3: Verify beamforming consolidation (1-2 hours)

### Short-term (Next Sprint):
- Audit file sizes, split >500 line modules
- Document accessor patterns for shared logic
- Create CI checks for architecture compliance

### Medium-term (2-3 Sprints):
- Complete therapy/imaging consolidation
- Establish architecture decision records (ADR) for all patterns
- Create developer onboarding guide for architecture

---

## Appendix: Import Migration Commands

### Automated Migration Script (Phase 1):

```bash
# Find all files importing from domain::math
find src -name "*.rs" -exec grep -l "use crate::domain::math::" {} \;

# Replace imports (dry run first)
find src -name "*.rs" -exec sed -i 's/use crate::domain::math::/use crate::math::/g' {} \;

# Verify no remaining references
grep -r "domain::math" src/

# Delete duplicate module
rm -rf src/domain/math

# Update domain mod.rs
sed -i '/pub mod math;/d' src/domain/mod.rs
```

### Verification:
```bash
# Build and test
cargo clean
cargo build --all-features
cargo test --all-features
cargo clippy --all-features -- -D warnings
```

---

## Conclusion

This refactoring establishes a mathematically rigorous architectural foundation with:
1. **Zero duplication**: Single source of truth for all components
2. **Clear hierarchy**: Self-documenting vertical file tree
3. **Strict boundaries**: Unidirectional dependencies enforced by module system
4. **Maintainability**: GRASP-compliant module sizes
5. **Scalability**: Room for growth within established patterns

The refactoring is LOW RISK due to:
- No logic changes (pure structural refactoring)
- Comprehensive test coverage catches regressions
- Atomic changes allow easy rollback
- Compiler enforces correctness

**Recommendation**: Proceed with Phase 1 immediately.

---

**End of Document**