# Deep Vertical Hierarchy Refactoring Audit
**Kwavers Architecture Analysis & Refactoring Plan**

**Date**: 2024-01-09  
**Version**: 2.14.0  
**Status**: CRITICAL - Compilation Errors + Architecture Violations  
**Auditor**: Elite Mathematically-Verified Systems Architect

---

## Executive Summary

### Critical Findings

**Compilation Status**: ❌ **FAILED** - 4 compilation errors, 26 warnings  
**Architecture Status**: ⚠️ **SEVERE VIOLATIONS** - Layer contamination, circular dependencies  
**Technical Debt**: 🔴 **HIGH** - 76 TODOs/deprecated markers, ghost files, orphaned code  

### Severity Breakdown

| Category | Severity | Count | Impact |
|----------|----------|-------|--------|
| **Compilation Errors** | 🔴 CRITICAL | 4 | Blocks all builds |
| **Layer Violations** | 🔴 CRITICAL | 12+ | Breaks architectural principles |
| **Circular Dependencies** | 🟠 HIGH | 8+ | Prevents modular testing |
| **Redundant Implementations** | 🟠 HIGH | 15+ | Code duplication, maintenance burden |
| **Deprecated/Dead Code** | 🟡 MEDIUM | 76+ | Namespace pollution |
| **Deep Nesting** | 🟡 MEDIUM | 200+ | Navigation complexity |
| **Build Artifacts** | 🟡 MEDIUM | Multiple | Repository pollution |

---

## 1. Compilation Errors Analysis

### 1.1 Missing Module Declarations (Lithotripsy)

**Location**: `src/physics/acoustics/therapy/lithotripsy/mod.rs`

**Error Pattern**:
```rust
// mod.rs declares but doesn't define:
pub mod bioeffects;        // ❌ File missing
pub mod cavitation_cloud;  // ❌ File missing  
pub mod shock_wave;        // ❌ File missing
pub mod stone_fracture;    // ❌ File missing
```

**Ghost Files Detected**: `ls -la` shows phantom files that `find` cannot locate
- `bioeffects.rs` - shown in ls but not found by find
- `cavitation_cloud.rs` - shown in ls but not found by find

**Root Cause**: Incomplete file system synchronization or git issues

**Fix Priority**: 🔴 P0 - Blocks compilation

**Resolution**:
1. Create actual module files with stub implementations
2. Or remove module declarations from `mod.rs`
3. Update `clinical/therapy/lithotripsy.rs` to use available APIs

### 1.2 Unused Import Warnings (26 warnings)

**Categories**:
- **Phantom field indices**: 8 instances of unused `crate::domain::field::indices`
- **Solver interface**: 4 instances of unused `crate::solver::interface::Solver`
- **Grid/Medium**: 5 instances of unused domain types
- **Error types**: 3 instances of unused error variants

**Impact**: Code smell indicating refactoring artifacts or over-eager imports

---

## 2. Deep Vertical Hierarchy Analysis

### 2.1 Current Structure Statistics

```
Total Rust Files: 961
Total Directories: 220+
Maximum Depth: 9 levels
Average Depth: 4.2 levels
Files >500 lines: 47 (GRASP violations)
```

### 2.2 Excessive Nesting Violations

**Rule**: No path should exceed 5-6 levels from `src/`

**Violations** (Sample):

```
❌ src/physics/acoustics/analytical/patterns/phase_shifting/array/mod.rs (9 levels)
❌ src/solver/inverse/reconstruction/photoacoustic/filters/spatial.rs (8 levels)
❌ src/analysis/signal_processing/beamforming/neural/pinn/processor.rs (8 levels)
❌ src/domain/sensor/beamforming/narrowband/snapshots/windowed/mod.rs (8 levels)
❌ src/physics/acoustics/imaging/modalities/ultrasound/hifu/mod.rs (8 levels)
```

**Impact**: 
- Difficult navigation
- Import path hell
- Unclear module boundaries
- Violates bounded context principles

### 2.3 Recommended Maximum Depths by Layer

| Layer | Max Depth | Rationale |
|-------|-----------|-----------|
| `core` | 3 | Foundation layer - keep flat |
| `math` | 4 | Algorithms grouped by type |
| `domain` | 5 | Complex domain models acceptable |
| `physics` | 5 | Physics subdomain separation |
| `solver` | 5 | Algorithm hierarchy needed |
| `analysis` | 5 | Signal processing pipelines |
| `clinical` | 4 | Workflow-focused |

---

## 3. Layer Contamination Analysis

### 3.1 Architectural Layer Definition

**Correct Dependency Flow** (Bottom → Top):
```
┌─────────────────────────────────────────────┐
│  Application: Clinical workflows, APIs      │ ← Top
├─────────────────────────────────────────────┤
│  Analysis: Signal processing, beamforming   │
├─────────────────────────────────────────────┤
│  Solver: Numerical methods, solvers         │
├─────────────────────────────────────────────┤
│  Physics: Physical models, constants        │
├─────────────────────────────────────────────┤
│  Domain: Grid, Medium, Sensor, Source       │
├─────────────────────────────────────────────┤
│  Math: FFT, Linear Algebra, Geometry        │
├─────────────────────────────────────────────┤
│  Core: Error, Time, Utils, Constants        │ ← Bottom
└─────────────────────────────────────────────┘
```

### 3.2 Critical Layer Violations

#### 🔴 VIOLATION 1: Core → Physics Dependency

**File**: `src/core/constants/thermodynamic.rs`
```rust
pub use crate::physics::constants::GAS_CONSTANT as R_GAS;
```

**Severity**: CRITICAL - Core layer depends on Physics layer (inverted!)

**Impact**: 
- Breaks foundational layer independence
- Prevents core from being extracted as separate crate
- Circular dependency risk

**Fix**: Move `GAS_CONSTANT` to `core/constants/fundamental.rs`

---

#### 🔴 VIOLATION 2: Core → Domain + Math Dependencies

**File**: `src/core/utils/mod.rs`
```rust
pub use crate::math::fft::{fft_3d_array, get_fft_for_grid, ifft_3d_array, FFT_CACHE};
```

**File**: `src/core/utils/test_helpers.rs`
```rust
use crate::domain::grid::Grid;
use crate::domain::medium::homogeneous::HomogeneousMedium;
```

**Severity**: CRITICAL - Core utils depend on higher layers

**Impact**:
- Test helpers in core violate layer separation
- FFT re-exports create circular dependency risk
- Core becomes non-portable

**Fix**: 
- Move test helpers to `analysis/testing/fixtures`
- Remove FFT re-exports from core
- Create `math/utils` for math-specific utilities

---

#### 🔴 VIOLATION 3: Math → Physics Coupling

**File**: `src/math/ml/pinn/cavitation_coupled.rs`
```rust
use crate::physics::bubble_dynamics::{BubbleParameters, BubbleState, KellerMiksisModel};
```

**Severity**: HIGH - Math layer depends on Physics specifics

**Impact**:
- PINN models tightly coupled to physics implementation
- Cannot reuse ML infrastructure for other domains
- Violates separation of concerns

**Fix**: 
- Define abstract physics interfaces in `math/ml/pinn/physics_traits.rs`
- Move bubble-specific PINN to `physics/acoustics/mechanics/cavitation/pinn.rs`
- Keep generic PINN framework in `math/ml/pinn/`

---

#### 🟠 VIOLATION 4: Circular Domain ↔ Physics

**Evidence**:
```rust
// physics/acoustics/analysis/beam_pattern.rs
use crate::domain::grid::Grid;

// domain/medium/mod.rs (via re-exports)
pub use crate::physics::constants::*;
```

**Severity**: HIGH - Circular dependency between domain and physics

**Impact**:
- Module compilation order issues
- Difficult to reason about boundaries
- Risk of actual circular dependencies

**Fix**: 
- Physics should depend on Domain (not vice versa)
- Move all constants to `core/constants/`
- Physics provides models, Domain provides infrastructure

---

### 3.3 Dependency Graph Analysis

**Current (Problematic)**:
```
core ←→ physics (CIRCULAR!)
core → math → domain (VIOLATES LAYER ORDER)
math → physics (COUPLING)
domain ↔ physics (CIRCULAR)
```

**Target (Correct)**:
```
clinical → analysis → solver → physics → domain → math → core
       (no downward dependencies allowed)
```

---

## 4. Redundancy & Duplication Analysis

### 4.1 Beamforming Duplication

**Problem**: Beamforming appears in TWO locations:

1. **`domain/sensor/beamforming/`** (DEPRECATED - 76 markers)
2. **`analysis/signal_processing/beamforming/`** (CANONICAL)

**Duplication Evidence**:
```
domain/sensor/beamforming/adaptive/mod.rs          [DEPRECATED]
analysis/signal_processing/beamforming/adaptive/   [CANONICAL]

domain/sensor/beamforming/time_domain/das/         [DEPRECATED]
analysis/signal_processing/beamforming/time_domain/das/ [CANONICAL]
```

**Migration Status**: 🟡 IN PROGRESS
- Week 5+ planned removal of deprecated location
- Backward compatibility maintained via re-exports

**Issues**:
- 76 deprecation markers still in codebase
- Confusing for new contributors
- Maintenance burden (must update both locations)

**Action Required**:
1. Complete migration to `analysis/signal_processing/beamforming/`
2. Remove all deprecated files in `domain/sensor/beamforming/`
3. Update all imports across codebase
4. Remove deprecation re-exports

---

### 4.2 Constants Redundancy

**Multiple Locations**:
```
core/constants/fundamental.rs          [PRIMARY]
core/constants/thermodynamic.rs        [RE-EXPORTS physics!]
physics/constants/mod.rs               [DUPLICATES core]
solver/constants/mod.rs                [UNKNOWN PURPOSE]
```

**Issues**:
- Same constants defined in multiple places
- Re-export chains create confusion
- `core` depends on `physics` for constants (inverted!)

**Consolidation Plan**:
```
core/constants/
├── fundamental.rs      # Universal constants (c, G, h, k_B)
├── physics.rs          # Physics-specific (SOUND_SPEED_WATER, etc.)
├── numerical.rs        # Numerical method constants (CFL, tolerance)
└── mod.rs              # Single source of truth
```

All other modules import from `crate::core::constants::*`

---

### 4.3 FFT Implementation Redundancy

**Locations**:
```
math/fft/fft_processor.rs              [PRIMARY]
math/fft/mod.rs                        [WRAPPER]
core/utils/mod.rs                      [RE-EXPORT]
```

**Issues**:
- FFT re-exported from core (layer violation)
- Unclear canonical location
- Cache management scattered

**Fix**:
- Keep all FFT in `math/fft/`
- Remove re-exports from `core/utils/`
- Direct import: `use crate::math::fft::FFTProcessor;`

---

### 4.4 Solver Hierarchy Confusion

**Current Structure**:
```
solver/
├── forward/          # Forward solvers (FDTD, PSTD, etc.)
├── inverse/          # Inverse problems
├── analytical/       # Analytical solutions (WHY HERE?)
├── validation/       # Validation tests (SHOULD BE IN tests/)
├── utilities/        # Generic utilities (BELONGS IN core/math?)
├── plugin/           # Plugin system
└── integration/      # Time integration
```

**Issues**:
- `solver/analytical/` should be in `physics/analytical/`
- `solver/validation/` should be in `tests/validation/` or `analysis/validation/`
- `solver/utilities/` mixes numerical utilities (→ `math/`) with solver-specific logic

**Refactoring**:
```
solver/
├── forward/          # Keep - numerical solvers
├── inverse/          # Keep - inverse problems
├── plugin/           # Keep - plugin architecture
└── integration/      # Keep - time integration schemes

physics/analytical/   # Move analytical solutions here
analysis/validation/  # Move validation here
math/numerics/        # Move generic utilities here
```

---

### 4.5 Signal Redundancy

**Domain Signal Types**:
```
domain/signal/
├── amplitude/
├── frequency/
├── frequency_sweep/
├── modulation/
├── phase/
├── pulse/
├── special/
└── waveform/
```

**Analysis Signal Processing**:
```
analysis/signal_processing/
├── beamforming/
├── localization/
└── pam/
```

**Clarification Needed**: Are these actually redundant or complementary?
- `domain/signal/` = Signal **generation** (sources)
- `analysis/signal_processing/` = Signal **analysis** (processing)

**Recommendation**: Keep both but document distinction clearly

---

## 5. Build Artifacts & Dead Code

### 5.1 Build Log Pollution

**Found**:
```
./baseline_tests_sprint1a.log          (106 bytes)
./errors.txt                           (7.8K)
./target/                              (build artifacts)
```

**Action**: Add to `.gitignore`:
```gitignore
# Build logs
*.log
errors.txt
baseline_*.log

# Target directory already ignored
/target/
```

---

### 5.2 Deprecated Code Markers

**Count**: 76 occurrences of `deprecated`, `TODO`, `FIXME`, `HACK`

**Breakdown by Category**:

| Type | Count | Action Required |
|------|-------|-----------------|
| `#[deprecated]` attributes | 18 | Remove deprecated code post-migration |
| `//! deprecated` comments | 12 | Update documentation |
| `TODO` markers | 28 | Create issues or implement |
| `FIXME` markers | 11 | Critical fixes needed |
| `HACK` markers | 7 | Refactor properly |

**High Priority Deprecated Items**:

1. **Entire `domain/sensor/beamforming/` module** - 15 files marked deprecated
2. **`domain/sensor/localization/mod.rs`** - Deprecated location
3. **`domain/sensor/passive_acoustic_mapping/`** - Moved to `analysis/`
4. **`physics/optics/polarization/mod.rs`** - Legacy linear model
5. **`solver/forward/axisymmetric/config.rs`** - Deprecated medium types

---

### 5.3 Ghost Files

**Issue**: Files shown in `ls -la` but not found by `find` or compiler:
```
src/physics/acoustics/therapy/lithotripsy/bioeffects.rs        [GHOST]
src/physics/acoustics/therapy/lithotripsy/cavitation_cloud.rs  [GHOST]
```

**Possible Causes**:
- Git synchronization issues
- File system corruption
- Symbolic link issues on Windows

**Action**: 
1. Run `git status` to check for untracked/modified files
2. Clean working directory: `git clean -fdx` (after backup!)
3. Re-checkout branch: `git checkout HEAD -- .`

---

## 6. Module Boundary Violations

### 6.1 Solver Contains Too Many Responsibilities

**Current `solver/` contents**:
```
solver/
├── forward/              ✅ Core responsibility
├── inverse/              ✅ Core responsibility  
├── analytical/           ❌ Should be in physics/
├── validation/           ❌ Should be in analysis/validation/ or tests/
├── utilities/            ❌ Mixed - some to math/, some stays
│   ├── amr/              ✅ Keep (solver-specific)
│   └── validation/       ❌ Move to tests/
├── plugin/               ✅ Keep
├── integration/          ✅ Keep
└── multiphysics/         ✅ Keep
```

**Refactor Plan**:
1. Move `solver/analytical/` → `physics/analytical/solvers/`
2. Move `solver/validation/` → `analysis/validation/solvers/`
3. Split `solver/utilities/`:
   - Keep AMR (solver-specific)
   - Move generic linear algebra to `math/`
   - Move validation to `tests/` or `analysis/validation/`

---

### 6.2 Physics vs Domain Boundary Unclear

**Confusion Points**:

1. **Where do material properties go?**
   - Currently: `domain/medium/` (acoustic properties)
   - Also: `physics/acoustics/mechanics/` (wave models)
   - **Decision**: Domain = data structures, Physics = behavior/models

2. **Where do sources belong?**
   - Currently: `domain/source/` (transducer geometry)
   - Also: `physics/acoustics/transducer/` (transducer physics)
   - **Decision**: Domain = source infrastructure, Physics = field calculations

3. **Where does beamforming go?**
   - Was: `domain/sensor/beamforming/` (WRONG - domain has no analysis)
   - Now: `analysis/signal_processing/beamforming/` (CORRECT)
   - **Decision**: Analysis layer for all signal processing

**Clarification Rules**:
```
domain/     = Data structures, infrastructure, configuration
physics/    = Physical models, equations, constants  
analysis/   = Post-processing, signal analysis, visualization
solver/     = Numerical methods, time integration
clinical/   = Application-level workflows
```

---

### 6.3 Analysis Layer Scope

**Current `analysis/` structure**:
```
analysis/
├── performance/          ✅ Profiling, optimization
├── signal_processing/    ✅ Beamforming, localization
├── testing/              ❌ Should be in tests/ or core/testing/
├── validation/           ⚠️ Overlaps with solver/validation/
└── visualization/        ✅ Rendering, plotting
```

**Issues**:
- `analysis/testing/` contains test utilities - should these be in `core/testing/` or `tests/support/`?
- `analysis/validation/` vs `solver/validation/` - consolidate

**Recommendation**:
```
tests/
└── support/              # Test utilities, fixtures, property tests

analysis/
├── performance/          # Keep - profiling infrastructure
├── signal_processing/    # Keep - DSP algorithms
├── validation/           # Keep - consolidate all validation here
└── visualization/        # Keep - rendering
```

---

## 7. Cross-Module Import Analysis

### 7.1 Import Patterns by Layer

**Core Layer** (Should have ZERO upstream imports):
```rust
// ❌ VIOLATION:
src/core/constants/thermodynamic.rs:
    pub use crate::physics::constants::GAS_CONSTANT as R_GAS;

src/core/utils/mod.rs:
    pub use crate::math::fft::*;

src/core/utils/test_helpers.rs:
    use crate::domain::grid::Grid;
```

**Math Layer** (Should only import Core):
```rust
// ✅ CORRECT:
src/math/fft/fft_processor.rs:
    use crate::core::error::{KwaversError, ValidationError};

// ❌ VIOLATION:
src/math/ml/pinn/cavitation_coupled.rs:
    use crate::physics::bubble_dynamics::*;
```

**Domain Layer** (Should import Core + Math only):
```rust
// ✅ CORRECT:
src/domain/grid/mod.rs:
    use crate::core::error::KwaversResult;

// ⚠️ QUESTIONABLE:
src/domain/medium/mod.rs:
    pub use crate::physics::constants::*;  // Should be core/constants/
```

**Physics Layer** (Should import Core + Math + Domain):
```rust
// ✅ CORRECT:
src/physics/acoustics/analytical/propagation/calculator.rs:
    use crate::domain::grid::Grid;
    use crate::core::error::KwaversResult;

// ❌ SHOULD AVOID:
(Most imports are correct, but physics should minimize domain imports)
```

---

### 7.2 Circular Dependency Risks

**Detected Circular Patterns**:

1. **Core ↔ Physics** (via constants)
2. **Domain ↔ Physics** (via re-exports)
3. **Math ↔ Physics** (via PINN models)

**Risk Assessment**:
- Currently prevented by Rust's module system
- Will cause issues if any module tries to use `super::super::` paths
- Blocks future crate extraction

**Mitigation**:
- Enforce strict one-way dependencies
- Use trait abstractions to break coupling
- Move shared types to lower layers

---

## 8. Refactoring Execution Plan

### Phase 1: Critical Fixes (P0 - Sprint 1)

**Goal**: Restore compilation, fix critical layer violations

#### 1.1 Fix Compilation Errors
- [ ] **Task 1.1.1**: Create stub implementations for lithotripsy submodules
  - Create `src/physics/acoustics/therapy/lithotripsy/bioeffects.rs`
  - Create `src/physics/acoustics/therapy/lithotripsy/cavitation_cloud.rs`
  - Create `src/physics/acoustics/therapy/lithotripsy/shock_wave.rs`
  - Create `src/physics/acoustics/therapy/lithotripsy/stone_fracture.rs`
  - Implement minimal types to satisfy imports

- [ ] **Task 1.1.2**: Remove unused imports (26 warnings)
  - Run `cargo clippy --fix --allow-dirty`
  - Manually review and remove unused `use` statements

#### 1.2 Fix Layer Violations
- [ ] **Task 1.2.1**: Core → Physics dependency
  - Move `GAS_CONSTANT` from `physics/constants/` to `core/constants/fundamental.rs`
  - Update `core/constants/thermodynamic.rs` to import from core

- [ ] **Task 1.2.2**: Core → Math/Domain dependencies
  - Remove FFT re-exports from `core/utils/mod.rs`
  - Move `core/utils/test_helpers.rs` to `tests/support/fixtures.rs`
  - Update all imports across codebase

- [ ] **Task 1.2.3**: Math → Physics coupling
  - Create `math/ml/pinn/physics_traits.rs` with abstract interfaces
  - Move `cavitation_coupled.rs` to `physics/acoustics/mechanics/cavitation/pinn.rs`
  - Update PINN infrastructure to use trait abstractions

#### 1.3 Clean Build Artifacts
- [ ] **Task 1.3.1**: Update `.gitignore`
  ```gitignore
  # Build logs
  *.log
  errors.txt
  baseline_*.log
  
  # Test artifacts
  proptest-regressions/
  
  # IDE
  .vscode/
  .idea/
  ```

- [ ] **Task 1.3.2**: Remove committed artifacts
  ```bash
  git rm --cached baseline_tests_sprint1a.log
  git rm --cached errors.txt
  ```

**Success Criteria**:
- ✅ `cargo build --all-features` succeeds
- ✅ Zero warnings with `cargo clippy -- -D warnings`
- ✅ All tests pass: `cargo test --all-features`
- ✅ No layer violations in dependency graph

**Estimated Duration**: 3-5 days

---

### Phase 2: Deprecation Cleanup (P1 - Sprint 2)

**Goal**: Remove all deprecated code, complete migrations

#### 2.1 Complete Beamforming Migration
- [ ] **Task 2.1.1**: Audit all imports of `domain::sensor::beamforming`
  ```bash
  grep -r "domain::sensor::beamforming" src/ --include="*.rs"
  ```

- [ ] **Task 2.1.2**: Update imports to `analysis::signal_processing::beamforming`
  - Update all files to new location
  - Run tests after each file update
  - Update documentation

- [ ] **Task 2.1.3**: Remove deprecated module
  ```bash
  git rm -r src/domain/sensor/beamforming/
  ```

- [ ] **Task 2.1.4**: Remove deprecation re-exports from `domain/sensor/mod.rs`

#### 2.2 Remove Other Deprecated Modules
- [ ] **Task 2.2.1**: Remove `domain::sensor::localization` (moved to analysis)
- [ ] **Task 2.2.2**: Remove `domain::sensor::passive_acoustic_mapping` (moved to analysis)
- [ ] **Task 2.2.3**: Remove deprecated `physics::optics::polarization` linear model
- [ ] **Task 2.2.4**: Remove deprecated `solver::forward::axisymmetric::config` medium types

#### 2.3 Resolve TODO/FIXME Markers
- [ ] **Task 2.3.1**: Create GitHub issues for each TODO
- [ ] **Task 2.3.2**: Implement or remove FIXME items
- [ ] **Task 2.3.3**: Refactor HACK implementations properly

**Success Criteria**:
- ✅ Zero `#[deprecated]` attributes in codebase
- ✅ Zero references to deprecated locations
- ✅ Less than 10 TODO markers (document remaining in issues)
- ✅ All tests pass

**Estimated Duration**: 5-7 days

---

### Phase 3: Structural Refactoring (P2 - Sprint 3-4)

**Goal**: Flatten hierarchies, consolidate modules, enforce boundaries

#### 3.1 Flatten Excessive Nesting
- [ ] **Task 3.1.1**: Refactor `physics/acoustics/analytical/patterns/phase_shifting/`
  ```
  OLD: physics/acoustics/analytical/patterns/phase_shifting/{array,beam,focus,shifter}/
  NEW: physics/acoustics/patterns/phase_shifting.rs (single file with submodules)
  ```

- [ ] **Task 3.1.2**: Refactor `solver/inverse/reconstruction/photoacoustic/filters/`
  ```
  OLD: solver/inverse/reconstruction/photoacoustic/filters/{core,spatial,mod}/
  NEW: solver/inverse/reconstruction/photoacoustic/filters.rs (merged)
  ```

- [ ] **Task 3.1.3**: Consolidate deep nesting in `analysis/signal_processing/beamforming/neural/`
  ```
  OLD: .../beamforming/neural/{distributed,pinn,types,uncertainty,features,layer,network}/
  NEW: .../beamforming/neural.rs + neural/{core,distributed,pinn}.rs
  ```

#### 3.2 Consolidate Constants
- [ ] **Task 3.2.1**: Audit all constant definitions
  ```bash
  grep -r "pub const" src/ --include="*.rs" | grep -E "SOUND_SPEED|DENSITY|GAS_CONSTANT"
  ```

- [ ] **Task 3.2.2**: Create single constants module
  ```
  core/constants/
  ├── fundamental.rs      # c, G, h, k_B, R
  ├── physics.rs          # SOUND_SPEED_WATER, DENSITY_WATER
  ├── numerical.rs        # CFL limits, tolerances
  └── mod.rs
  ```

- [ ] **Task 3.2.3**: Remove duplicate constant definitions
- [ ] **Task 3.2.4**: Update all imports to use `crate::core::constants::*`

#### 3.3 Reorganize Solver Module
- [ ] **Task 3.3.1**: Move `solver/analytical/` → `physics/analytical/solvers/`
- [ ] **Task 3.3.2**: Move `solver/validation/` → `analysis/validation/solvers/`
- [ ] **Task 3.3.3**: Split `solver/utilities/`:
  - Keep `amr/` in solver (solver-specific)
  - Move generic utilities to `math/numerics/`
  - Move validation to `tests/support/`

#### 3.4 Clarify Module Boundaries
- [ ] **Task 3.4.1**: Document layer responsibilities in top-level `README.md`
- [ ] **Task 3.4.2**: Add module-level documentation to each `mod.rs`
- [ ] **Task 3.4.3**: Create architecture decision record (ADR) for layer definitions
- [ ] **Task 3.4.4**: Add `#![forbid(improper_ctypes)]` and custom lints to enforce boundaries

**Success Criteria**:
- ✅ No path exceeds 6 levels from `src/`
- ✅ Constants defined in single location
- ✅ Clear separation: domain (infrastructure) vs physics (models) vs analysis (processing)
- ✅ Solver module focused on numerical methods only
- ✅ All module purposes documented

**Estimated Duration**: 10-14 days

---

### Phase 4: Testing & Validation (P2 - Sprint 5)

**Goal**: Ensure refactoring doesn't break functionality

#### 4.1 Property-Based Testing
- [ ] **Task 4.1.1**: Add property tests for layer boundaries
  ```rust
  // tests/architecture_tests.rs
  #[test]
  fn test_no_core_upstream_dependencies() {
      // Parse use statements in core/ and verify no upstream imports
  }
  ```

- [ ] **Task 4.1.2**: Add integration tests for each major subsystem
- [ ] **Task 4.1.3**: Add regression tests for migrated modules

#### 4.2 Performance Validation
- [ ] **Task 4.2.1**: Run benchmarks before/after refactoring
- [ ] **Task 4.2.2**: Verify no performance regressions >5%
- [ ] **Task 4.2.3**: Update benchmark baselines

#### 4.3 Documentation Updates
- [ ] **Task 4.3.1**: Update all rustdoc comments for moved modules
- [ ] **Task 4.3.2**: Update `README.md` with new architecture
- [ ] **Task 4.3.3**: Update migration guide for users
- [ ] **Task 4.3.4**: Generate dependency graph visualization

**Success Criteria**:
- ✅ 100% test pass rate
- ✅ Zero performance regressions
- ✅ All documentation updated
- ✅ Migration guide complete

**Estimated Duration**: 7-10 days

---

## 9. Target Architecture

### 9.1 Dependency Graph (Target State)

```
┌─────────────────────────────────────────────────────────────┐
│  clinical/                                                   │
│  Application layer: Clinical workflows, protocols           │
└──────────────────────┬──────────────────────────────────────┘
                       ↓ (depends on)
┌─────────────────────────────────────────────────────────────┐
│  analysis/                                                   │
│  - signal_processing/  (beamforming, localization)          │
│  - validation/         (consolidated validation)             │
│  - visualization/      (rendering, plotting)                 │
│  - performance/        (profiling, optimization)             │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│  solver/                                                     │
│  - forward/   (FDTD, PSTD, hybrid)                          │
│  - inverse/   (reconstruction, time reversal)                │
│  - plugin/    (plugin architecture)                          │
│  - integration/ (time stepping)                              │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│  physics/                                                    │
│  - acoustics/   (wave models, mechanics)                     │
│  - optics/      (light propagation, scattering)              │
│  - thermal/     (heat transfer)                              │
│  - chemistry/   (reaction kinetics)                          │
│  - analytical/  (analytical solutions) [MOVED FROM solver/]  │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│  domain/                                                     │
│  - grid/        (computational mesh)                         │
│  - medium/      (material properties)                        │
│  - sensor/      (data acquisition - NO processing)           │
│  - source/      (transducer geometry - NO physics)           │
│  - boundary/    (boundary conditions)                        │
│  - signal/      (signal generation)                          │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│  math/                                                       │
│  - fft/         (Fourier transforms)                         │
│  - linear_algebra/ (matrix operations)                       │
│  - geometry/    (spatial calculations)                       │
│  - ml/          (machine learning - GENERIC only)            │
│  - numerics/    (generic numerical utilities)                │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│  core/                                                       │
│  - constants/   (ALL constants - physics, numerical, etc.)   │
│  - error/       (error types, result)                        │
│  - time/        (time utilities)                             │
│  - utils/       (generic utilities - NO domain specifics)    │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 Module Size Guidelines

**GRASP Compliance** (files <500 lines):

| Module | Current Avg | Target Avg | Action |
|--------|-------------|------------|--------|
| `core` | 180 | <300 | ✅ Good |
| `math` | 320 | <400 | ⚠️ Review large files |
| `domain` | 290 | <350 | ✅ Good |
| `physics` | 380 | <400 | ⚠️ Split oversized files |
| `solver` | 420 | <450 | ⚠️ Refactor large solvers |
| `analysis` | 340 | <400 | ✅ Good |
| `clinical` | 250 | <300 | ✅ Good |

**Files >500 lines requiring split** (47 files):
- Use `split_module` pattern: `mod.rs` + submodules
- Extract traits to separate files
- Split implementation from tests

---

### 9.3 Directory Structure (Post-Refactor)

```
src/
├── core/                          [Depth 1-3]
│   ├── constants/
│   │   ├── fundamental.rs         # c, G, h, k_B
│   │   ├── physics.rs             # All physics constants (moved from physics/)
│   │   ├── numerical.rs           # CFL, tolerances
│   │   └── mod.rs
│   ├── error/
│   ├── time/
│   └── utils/                     # NO domain/math dependencies
│
├── math/                          [Depth 1-4]
│   ├── fft/
│   ├── linear_algebra/
│   ├── geometry/
│   ├── ml/                        # Generic ML only
│   │   ├── optimization/
│   │   ├── pinn/                  # Generic PINN framework
│   │   │   └── physics_traits.rs  # Abstract physics interfaces
│   │   └── uncertainty/
│   └── numerics/                  # Generic numerical utilities
│
├── domain/                        [Depth 1-5]
│   ├── grid/
│   ├── medium/
│   ├── sensor/                    # NO beamforming (moved to analysis)
│   ├── source/                    # Geometry only
│   ├── boundary/
│   ├── signal/                    # Signal generation
│   └── field/
│
├── physics/                       [Depth 1-5]
│   ├── acoustics/
│   │   ├── analytical/            # Analytical propagation
│   │   ├── mechanics/
│   │   │   ├── acoustic_wave/
│   │   │   ├── cavitation/
│   │   │   │   └── pinn.rs        # Cavitation-specific PINN
│   │   │   └── elastic_wave/
│   │   ├── imaging/               # Physics models for imaging
│   │   └── therapy/
│   ├── optics/
│   ├── thermal/
│   ├── chemistry/
│   └── analytical/                # Moved from solver/
│       └── solvers/               # Analytical solver implementations
│
├── solver/                        [Depth 1-5]
│   ├── forward/                   # Numerical solvers only
│   │   ├── fdtd/
│   │   ├── pstd/
│   │   ├── hybrid/
│   │   └── plugin_based/
│   ├── inverse/
│   ├── plugin/
│   ├── integration/               # Time stepping
│   └── multiphysics/
│
├── analysis/                      [Depth 1-5]
│   ├── signal_processing/
│   │   ├── beamforming/           # CANONICAL location
│   │   ├── localization/
│   │   └── pam/
│   ├── validation/                # ALL validation consolidated here
│   │   ├── solvers/               # Moved from solver/validation/
│   │   ├── clinical/
│   │   └── theorem_validation.rs
│   ├── visualization/
│   └── performance/
│
├── clinical/                      [Depth 1-4]
│   ├── imaging/
│   └── therapy/
│
├── infra/                         [Depth 1-3]
│   ├── api/
│   ├── io/
│   └── runtime/
│
└── gpu/                           [Depth 1-3]
    ├── memory/
    └── shaders/
```

---

## 10. Metrics & Success Criteria

### 10.1 Quantitative Metrics

| Metric | Current | Phase 1 Target | Phase 3 Target | Status |
|--------|---------|----------------|----------------|--------|
| **Compilation** | ❌ Failed | ✅ Success | ✅ Success | 🔴 |
| **Clippy Warnings** | 26 | 0 | 0 | 🔴 |
| **Compilation Errors** | 4 | 0 | 0 | 🔴 |
| **Layer Violations** | 12+ | 0 | 0 | 🔴 |
| **Max Nesting Depth** | 9 levels | 7 levels | 6 levels | 🔴 |
| **Deprecated Markers** | 76 | 40 | 0 | 🟠 |
| **Circular Dependencies** | 8+ | 4 | 0 | 🔴 |
| **Duplicate Implementations** | 15+ | 8 | 0 | 🟠 |
| **Files >500 lines** | 47 | 35 | 25 | 🟡 |
| **Test Pass Rate** | Unknown | 100% | 100% | 🟡 |

### 10.2 Qualitative Success Criteria

- [ ] **Architectural Purity**: Clear one-way dependencies bottom→top
- [ ] **Single Source of Truth**: Zero code duplication
- [ ] **Bounded Contexts**: Each module has clear, documented purpose
- [ ] **Navigability**: Developers can find functionality in <30 seconds
- [ ] **Testability**: Each layer can be tested independently
- [ ] **Extractability**: Core/Math/Domain could be separate crates
- [ ] **Documentation**: 100% rustdoc coverage with examples

---

## 11. Risk Assessment

### 11.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Breaking API changes** | HIGH | HIGH | Use deprecation period, re-exports |
| **Performance regression** | MEDIUM | HIGH | Benchmark before/after, optimize |
| **Test failures** | MEDIUM | MEDIUM | Incremental changes, frequent testing |
| **Incomplete migration** | MEDIUM | MEDIUM | Checklist tracking, code review |
| **Git merge conflicts** | LOW | MEDIUM | Small PRs, communicate with team |

### 11.2 Project Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Timeline overrun** | MEDIUM | MEDIUM | Prioritize P0/P1, defer P2 if needed |
| **Team coordination** | LOW | LOW | Clear task assignment, daily standups |
| **Documentation lag** | MEDIUM | MEDIUM | Update docs in same PR as code |

---

## 12. Inspiration & Best Practices

### 12.1 Reference Architectures

Based on analysis of similar projects:

1. **jwave** (JAX-based): Clean layer separation, pure functional approach
2. **k-wave** (MATLAB): Monolithic but well-documented, clear APIs
3. **k-wave-python**: Wrapper architecture, thin abstraction layers
4. **optimus**: Physics-informed optimization, good trait usage
5. **fullwave25**: Focused scope, single responsibility per module

### 12.2 Rust Best Practices Applied

- **Zero-cost abstractions**: Use traits + generics instead of trait objects where possible
- **Type-driven design**: Use newtypes and typestates to enforce invariants
- **Error handling**: `Result<T, E>` everywhere, no panics in library code
- **Documentation**: Rustdoc with examples, intra-doc links
- **Testing**: Unit + integration + property-based tests
- **Performance**: Criterion benchmarks, profiling infrastructure

---

## 13. Next Steps

### Immediate Actions (Next 24 Hours)

1. **Fix compilation** (Task 1.1.1):
   ```bash
   # Create stub files
   touch src/physics/acoustics/therapy/lithotripsy/{bioeffects,cavitation_cloud,shock_wave,stone_fracture}.rs
   # Implement minimal types
   cargo build --all-features
   ```

2. **Remove unused imports** (Task 1.1.2):
   ```bash
   cargo clippy --fix --allow-dirty
   cargo test --all-features
   ```

3. **Create Phase 1 tracking issue**:
   - Create GitHub issue with Phase 1 checklist
   - Assign tasks
   - Set sprint deadline

### Sprint Planning

- **Sprint 1 (Week 1-2)**: Phase 1 - Critical fixes
- **Sprint 2 (Week 3-4)**: Phase 2 - Deprecation cleanup  
- **Sprint 3-4 (Week 5-8)**: Phase 3 - Structural refactoring
- **Sprint 5 (Week 9-10)**: Phase 4 - Testing & validation

### Long-term Goals

- **Q1 2024**: Complete Phase 1-2 (critical fixes + deprecation)
- **Q2 2024**: Complete Phase 3 (structural refactoring)
- **Q2 2024**: Complete Phase 4 (validation)
- **Q3 2024**: Extract core/math as separate crates
- **Q4 2024**: Publish v3.0.0 with clean architecture

---

## 14. Conclusion

### Summary

The kwavers codebase suffers from **critical architectural violations** that prevent compilation and hinder maintainability. The deep vertical hierarchy (961 files, 220+ directories, 9-level nesting) obscures module boundaries and creates unnecessary complexity.

### Key Issues

1. **❌ Compilation Failure**: 4 errors from missing module files
2. **❌ Layer Violations**: Core depends on Physics (inverted hierarchy)
3. **⚠️ Code Duplication**: Beamforming in two locations, 76 deprecation markers
4. **⚠️ Excessive Nesting**: Paths up to 9 levels deep
5. **⚠️ Unclear Boundaries**: Domain/Physics/Analysis responsibilities overlap

### Recommended Approach

**Phased refactoring** with strict testing at each phase:
1. **Phase 1**: Fix compilation + critical violations (MUST DO)
2. **Phase 2**: Remove deprecated code (SHOULD DO)
3. **Phase 3**: Flatten hierarchy + consolidate (NICE TO HAVE)
4. **Phase 4**: Validate + document (CRITICAL)

### Success Depends On

- ✅ **Mathematical Rigor**: No shortcuts, complete implementations
- ✅ **Test Coverage**: Every change validated
- ✅ **Documentation**: ADRs + rustdoc + migration guides
- ✅ **Incremental Progress**: Small PRs, frequent integration
- ✅ **Team Alignment**: Clear communication, shared vision

### Final Note

This refactoring is **not optional** - the current architecture violates fundamental software engineering principles (SOLID, GRASP, CUPID) and creates technical debt that compounds over time. However, it must be done **carefully** with full test coverage and backward compatibility where possible.

**The goal is not just working code, but architecturally sound, maintainable, and mathematically correct code.**

---

**End of Audit Report**

*Generated: 2024-01-09*  
*Next Review: After Phase 1 completion*