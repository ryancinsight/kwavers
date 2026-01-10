# Architectural Audit: Deep Vertical Hierarchy Analysis
## Sprint: Architecture Refactoring - Cross-Contamination & Redundancy Elimination

**Date**: 2024  
**Auditor**: Elite Mathematically-Verified Systems Architect  
**Scope**: Complete codebase structural analysis (944 Rust files)  
**Grade**: B+ (85%) - Significant cross-contamination requiring refactoring  

---

## Executive Summary

### Critical Findings

**Current State**: 944 Rust source files across 10 top-level modules with **significant cross-contamination and redundancy** between:
- Grid operations (domain/grid vs solver/*/numerics)
- Boundary conditions (domain/boundary vs solver implementations)
- Medium properties (domain/medium vs physics/acoustics vs solver)
- Beamforming (analysis/signal_processing vs domain/sensor)
- Math/numerics (math/numerics vs solver/forward/*/numerics)
- Clinical workflows (clinical vs physics/acoustics/therapy vs physics/acoustics/imaging)
- Solver abstractions (solver/interface vs individual solver implementations)

**Primary Issue**: Violation of deep vertical hierarchy principles - horizontal spreading of functionality across peer modules instead of clear bottom-up dependency flow.

---

## 1. Module Dependency Architecture Analysis

### 1.1 Current Module Structure

```
src/
├── core/              [6 subdirs]  - Fundamental types, errors, constants
├── infra/             [5 subdirs]  - Infrastructure (API, IO, cloud, runtime)
├── domain/            [10 subdirs] - Domain primitives (grid, medium, boundary, source, sensor, signal, field)
├── math/              [6 subdirs]  - Mathematical operations (FFT, linear algebra, numerics, ML)
├── physics/           [6 subdirs]  - Physics models (acoustics, optics, thermal, chemistry)
├── solver/            [10 subdirs] - Numerical solvers (forward, inverse, analytical)
├── simulation/        [2 subdirs]  - Simulation orchestration
├── clinical/          [2 subdirs]  - Clinical applications (imaging, therapy)
├── analysis/          [7 subdirs]  - Post-processing (beamforming, visualization, performance)
├── gpu/               [2 subdirs]  - GPU acceleration (optional)
```

### 1.2 Expected Dependency Flow (Deep Vertical)

**Correct hierarchy (bottom to top)**:
```
core → infra → domain → math → physics → solver → simulation → clinical → analysis
                                                                          ↓
                                                                         gpu
```

### 1.3 Actual Dependency Violations

**CRITICAL VIOLATIONS**:

1. **Solver → Domain circular dependency**
   - `solver/forward/fdtd/numerics/staggered_grid.rs` reimplements grid operations
   - `solver/forward/axisymmetric/coordinates.rs` has `CylindricalGrid` (should use `domain/grid`)
   - `solver/utilities/cpml_integration.rs` duplicates `domain/boundary/cpml`

2. **Physics ↔ Solver contamination**
   - `physics/acoustics/mechanics/acoustic_wave` has solver logic
   - `solver/forward/pstd/physics` has physics logic
   - Unclear separation of "what to compute" vs "how to compute"

3. **Analysis ↔ Domain duplication**
   - `analysis/signal_processing/beamforming` (3 subdirs, ~15 files)
   - `domain/sensor/beamforming` (6 subdirs, ~20 files)
   - Duplicate beamforming algorithms in both locations

4. **Clinical scattered across multiple modules**
   - `clinical/therapy` (4 files)
   - `physics/acoustics/therapy` (5 subdirs)
   - `physics/acoustics/imaging` (3 subdirs)
   - Clinical workflows split between 3 different top-level modules

5. **Math/Numerics duplication**
   - `math/numerics/operators` (differential operators)
   - `solver/forward/fdtd/numerics` (finite difference operators)
   - `solver/forward/pstd/numerics` (spectral operators)
   - Each solver reimplements basic numerical operations

---

## 2. Cross-Contamination Detailed Analysis

### 2.1 Grid Operations - SEVERITY: HIGH

**Primary Location**: `domain/grid/` (5 files, 1 subdir)

**Contaminated Locations**:
1. `solver/forward/axisymmetric/coordinates.rs` - `CylindricalGrid` (146 lines)
2. `solver/forward/fdtd/numerics/staggered_grid.rs` - `StaggeredGrid` (200+ lines)
3. `math/numerics/operators/differential.rs` - `StaggeredGridOperator` (150+ lines)
4. `domain/sensor/grid_sampling.rs` - `GridPoint`, `GridSensorSet` (partial duplication)

**Evidence of Duplication**:
```rust
// domain/grid/structure.rs
impl Grid {
    pub fn idx(&self, i: usize, j: usize, k: usize) -> usize { ... }
}

// solver/forward/fdtd/numerics/staggered_grid.rs
impl StaggeredGrid {
    pub fn idx(&self, i: usize, j: usize, k: usize) -> usize { ... }
}

// solver/forward/axisymmetric/coordinates.rs
impl CylindricalGrid {
    pub fn linear_index(&self, r: usize, z: usize) -> usize { ... }
}
```

**Recommended Refactoring**:
- Consolidate all grid types into `domain/grid/`
- Create `domain/grid/cartesian.rs`, `domain/grid/cylindrical.rs`, `domain/grid/staggered.rs`
- Solvers consume grid through trait `GridTopology`
- Estimated effort: 40 hours

---

### 2.2 Boundary Conditions - SEVERITY: HIGH

**Primary Location**: `domain/boundary/` (1 subdir: `cpml/`)

**Contaminated Locations**:
1. `solver/forward/fdtd/` - Inline CPML implementation
2. `solver/forward/pstd/` - Inline boundary handling
3. `solver/utilities/cpml_integration.rs` - Duplicate CPML logic (300+ lines)
4. `solver/forward/fdtd/numerics/boundary_stencils.rs` - Boundary-specific stencils

**Issue**: Each solver reimplements boundary condition application instead of consuming from `domain/boundary/cpml`.

**Recommended Refactoring**:
- Define trait `BoundaryCondition` in `domain/boundary/traits.rs`
- Implement `CPMLBoundary`, `PMLBoundary`, `ABCBoundary` in `domain/boundary/`
- Solvers call `boundary.apply(field, grid, dt)` - no internal implementation
- Move `solver/utilities/cpml_integration.rs` logic into `domain/boundary/cpml/`
- Estimated effort: 32 hours

---

### 2.3 Medium Properties - SEVERITY: CRITICAL

**Primary Location**: `domain/medium/` (8 subdirs, 30+ files)

**Contaminated Locations**:
1. `physics/acoustics/` - Uses medium but also defines acoustic-specific properties
2. `physics/optics/` - Defines optical medium properties separately
3. `physics/thermal/` - Defines thermal medium properties separately
4. `solver/forward/axisymmetric/config.rs` - `AxisymmetricMedium` struct
5. `simulation/core.rs` - `CoreSimulation<'a, M: Medium>` - medium logic in simulation layer

**Issue**: Medium properties scattered across physics domains instead of unified in `domain/medium/traits/`.

**Current Structure** (partially correct):
```
domain/medium/
├── heterogeneous/
│   └── traits/
│       ├── acoustic/
│       ├── bubble/
│       ├── elastic/
│       ├── optical/
│       ├── thermal/
│       └── viscous/
```

**Problem**: Physics modules redefine these properties instead of importing from `domain/medium/heterogeneous/traits/`.

**Recommended Refactoring**:
- All medium traits MUST live in `domain/medium/heterogeneous/traits/`
- Physics modules consume traits, never define them
- Remove `AxisymmetricMedium` from solver - use `domain/medium` types
- Estimated effort: 48 hours

---

### 2.4 Beamforming - SEVERITY: HIGH

**Duplication**: Complete algorithm duplication between analysis and domain layers.

**Location 1**: `analysis/signal_processing/beamforming/` (3 subdirs)
```
analysis/signal_processing/beamforming/
├── adaptive/
│   ├── mvdr.rs
│   └── subspace.rs
├── time_domain/
│   ├── das.rs
│   └── delay_reference.rs
└── test_utilities.rs
```

**Location 2**: `domain/sensor/beamforming/` (6 subdirs)
```
domain/sensor/beamforming/
├── adaptive/
│   └── algorithms/
├── experimental/
├── narrowband/
├── time_domain/
│   └── das/
└── shaders/
```

**Additional Contamination**:
3. `domain/sensor/passive_acoustic_mapping/beamforming_config.rs`
4. `domain/source/transducers/phased_array/beamforming.rs`
5. `core/utils/sparse_matrix/beamforming.rs` (!!!)

**Issue**: Beamforming appears in 5 different module contexts:
- Analysis (post-processing)
- Domain/Sensor (observation)
- Domain/Source (transmission)
- Core/Utils (???)
- PAM-specific

**Recommended Refactoring**:
- **Primary**: `domain/sensor/beamforming/` owns all beamforming algorithms
- **Analysis**: Only visualization and performance analysis of beamforming results
- **Source**: Use `domain/sensor/beamforming` for transmit beamforming (same algorithms)
- **Remove**: `core/utils/sparse_matrix/beamforming.rs` - inappropriate location
- **PAM**: Import from `domain/sensor/beamforming`, configure for PAM use case
- Estimated effort: 56 hours

---

### 2.5 Math/Numerics - SEVERITY: MEDIUM

**Primary Location**: `math/numerics/` (3 subdirs)
```
math/numerics/
├── integration/
├── operators/
└── transforms/
```

**Contaminated Locations**:
1. `solver/forward/fdtd/numerics/` (4 files)
   - `finite_difference.rs` - should use `math/numerics/operators`
2. `solver/forward/pstd/numerics/` (1 subdir)
   - `operators/` - duplicate of `math/numerics/operators`
3. `domain/grid/operators/` - differential operators (should be in `math/numerics`)

**Issue**: Each solver reimplements basic numerical operators (gradients, divergence, Laplacian).

**Recommended Refactoring**:
- Consolidate all operators into `math/numerics/operators/`
- Create `DifferentialOperator` trait
- Implementations: `FiniteDifference`, `SpectralDerivative`, `DGDerivative`
- Solvers consume operators, never implement them
- Move `domain/grid/operators/` to `math/numerics/operators/grid_based.rs`
- Estimated effort: 40 hours

---

### 2.6 Clinical Workflows - SEVERITY: MEDIUM

**Scattered Across**:
1. `clinical/imaging/workflows.rs` (orchestration)
2. `clinical/therapy/` (4 files)
3. `physics/acoustics/imaging/modalities/` (3 subdirs)
4. `physics/acoustics/therapy/` (5 subdirs)

**Issue**: Clinical domain split between `clinical/` (orchestration) and `physics/acoustics/` (implementation).

**Current Structure**:
```
clinical/
├── imaging/
│   ├── workflows.rs
│   └── mod.rs
└── therapy/
    ├── lithotripsy.rs
    ├── swe_3d_workflows.rs
    ├── therapy_integration.rs
    └── mod.rs

physics/acoustics/
├── imaging/
│   ├── modalities/
│   │   ├── ceus/
│   │   ├── elastography/
│   │   └── ultrasound/hifu/
│   ├── registration/
│   └── seismic/
└── therapy/
    ├── cavitation/
    ├── lithotripsy/
    ├── metrics/
    ├── modalities/
    └── parameters/
```

**Recommended Refactoring**:
- **Keep in `clinical/`**: Workflows, protocols, clinical decision logic
- **Move to `clinical/` from `physics/acoustics/`**:
  - `physics/acoustics/imaging/modalities/` → `clinical/imaging/modalities/`
  - `physics/acoustics/therapy/` → `clinical/therapy/techniques/`
- **Keep in `physics/acoustics/`**: Core physics (wave propagation, cavitation models)
- Clear separation: Clinical = "what to do medically", Physics = "how waves behave"
- Estimated effort: 64 hours

---

### 2.7 Solver Architecture - SEVERITY: MEDIUM

**Issue**: Each solver reimplements common functionality instead of using shared abstractions.

**Current Solvers**:
```
solver/
├── forward/
│   ├── acoustic/
│   ├── axisymmetric/
│   ├── elastic/
│   ├── fdtd/
│   ├── hybrid/
│   ├── imex/
│   ├── nonlinear/
│   ├── plugin_based/
│   ├── pstd/
│   └── thermal_diffusion/
├── inverse/
├── analytical/
└── interface/  ← Should be primary abstraction
```

**Problem**: `solver/interface/` exists but is underutilized. Each solver in `solver/forward/*` implements its own:
- Time stepping logic (duplicated in 8+ solvers)
- Field update patterns
- Boundary application
- Source injection

**Recommended Refactoring**:
- Strengthen `solver/interface/` with comprehensive traits:
  - `TimeIntegrator`
  - `SpatialOperator`
  - `SourceInjector`
  - `BoundaryApplicator`
- Each solver in `solver/forward/` becomes a composition of these traits
- Move shared logic to `solver/utilities/`
- Estimated effort: 72 hours

---

## 3. Dead Code & Build Artifacts

### 3.1 Build Artifacts in Source Control

**Found**: `kwavers/errors.txt` in root directory  
**Issue**: Build logs should never be committed to version control  
**Action**: Remove immediately

### 3.2 Deprecated Code Markers

**Search Required**: Grep for:
- `#[deprecated]` attributes
- `// TODO: Remove` comments
- `// DEPRECATED` comments
- `_legacy`, `_old`, `_v1`, `_v2` suffixes in module names

**Feature Flag**: `legacy_algorithms` exists in `Cargo.toml`
```toml
legacy_algorithms = []  # Legacy beamforming algorithms (deprecated)
```

**Action Required**: 
1. Audit all code behind `#[cfg(feature = "legacy_algorithms")]`
2. Document migration path or remove entirely
3. Set deprecation timeline

### 3.3 Markdown Documentation Proliferation

**Found in root** (28 markdown files):
```
ACCURATE_MODULE_ARCHITECTURE.md
ARCHITECTURE_IMPROVEMENT_PLAN.md
ARCHITECTURE_REFACTORING_AUDIT.md
CHERNKOV_SONOLUMINESCENCE_ANALYSIS.md
COMPREHENSIVE_MODULE_REFACTORING_PLAN.md
DEPENDENCY_ANALYSIS.md
DEPLOYMENT_GUIDE.md
PERFORMANCE_OPTIMIZATION_ANALYSIS.md
PERFORMANCE_OPTIMIZATION_SUMMARY.md
PINN_ECOSYSTEM_SUMMARY.md
REFACTORING_EXECUTIVE_SUMMARY.md
REFACTORING_PROGRESS.md
REFACTORING_QUICK_REFERENCE.md
REFACTOR_PHASE_1_CHECKLIST.md
SIMULATION_REFACTORING_PLAN.md
SOLVER_REFACTORING_PLAN.md
SOURCE_IMPLEMENTATION_COMPLETE.md
SOURCE_MODULE_AUDIT_SUMMARY.md
SOURCE_SIGNAL_ARCHITECTURE.md
```

**Issue**: 19 architecture/refactoring documents in root - indicates iterative refactoring without cleanup.

**Action Required**:
1. Consolidate into `docs/architecture/` directory
2. Keep only: README.md, LICENSE, gap_audit.md, prompt.yaml in root
3. Move all technical docs to `docs/`

---

## 4. Inspiration Repositories Analysis

### 4.1 k-Wave (MATLAB) Architecture

**Repository**: https://github.com/ucl-bug/k-wave

**Key Strengths**:
- Clear separation: k-Wave core (propagation) vs k-Wave-Toolbox (pre/post processing)
- Modular functions: `kspaceFirstOrder2D`, `kspaceFirstOrder3D` (no cross-contamination)
- Grid abstraction: `kWaveGrid` class owns all spatial operations

**Lessons for Kwavers**:
- Separate solver kernel from simulation setup
- Grid should be self-contained (no leakage into solvers)
- Post-processing (beamforming, visualization) completely separate from solvers

### 4.2 jwave (JAX) Architecture

**Repository**: https://github.com/ucl-bug/jwave

**Key Strengths**:
- Functional approach: Pure functions for operators
- Clear layers: `jwave.acoustics` (physics) → `jwave.geometry` (domain) → `jwave.signal_processing` (analysis)
- No circular dependencies (enforced by JAX functional paradigm)

**Lessons for Kwavers**:
- Physics models should be stateless where possible
- Operators as pure functions (input → output, no side effects)
- Dependency flow strictly bottom-up

### 4.3 k-Wave-Python Architecture

**Repository**: https://github.com/waltsims/k-wave-python

**Key Strengths**:
- Thin wrapper maintaining k-Wave architecture
- No reimplementation of algorithms - delegates to core

**Lessons for Kwavers**:
- Don't reimplement algorithms in multiple modules
- Thin adapters over thick reimplementations

### 4.4 Fullwave2.5 (C++) Architecture

**Repository**: https://github.com/pinton-lab/fullwave25

**Key Strengths**:
- Single solver implementation (no duplication)
- CUDA kernels completely separate from CPU code
- Grid, medium, solver as distinct compilation units

**Lessons for Kwavers**:
- GPU code (`kwavers/src/gpu/`) should mirror CPU structure
- Each solver type should be its own crate (optional)

---

## 5. Refactoring Priority Matrix

### Phase 1: Critical Path (Weeks 1-4)

| Module | Issue | Severity | Effort | Files Affected |
|--------|-------|----------|--------|----------------|
| Medium Properties | Cross-contamination | CRITICAL | 48h | 30+ files |
| Grid Operations | Duplication | HIGH | 40h | 15+ files |
| Boundary Conditions | Duplication | HIGH | 32h | 12+ files |
| Beamforming | 5-way duplication | HIGH | 56h | 35+ files |

**Total Phase 1**: 176 hours (4.4 weeks @ 40h/week)

### Phase 2: Architectural Cleanup (Weeks 5-8)

| Module | Issue | Severity | Effort | Files Affected |
|--------|-------|----------|--------|----------------|
| Math/Numerics | Operator duplication | MEDIUM | 40h | 20+ files |
| Clinical Workflows | Scattered modules | MEDIUM | 64h | 25+ files |
| Solver Abstractions | Weak interface | MEDIUM | 72h | 40+ files |
| Dead Code | Legacy features | MEDIUM | 24h | 50+ files |

**Total Phase 2**: 200 hours (5 weeks @ 40h/week)

### Phase 3: Documentation & Polish (Weeks 9-10)

| Task | Effort |
|------|--------|
| Consolidate root markdown docs | 8h |
| Update architecture diagrams | 16h |
| API documentation audit | 24h |
| Migration guide | 16h |
| Update examples | 16h |

**Total Phase 3**: 80 hours (2 weeks @ 40h/week)

---

## 6. Proposed Deep Vertical Architecture

### 6.1 Corrected Module Hierarchy

```
src/
├── core/                    [LAYER 0: Primitives]
│   ├── constants/          (physics, numerical, medical)
│   ├── error/              (typed error hierarchy)
│   ├── time/               (time representation)
│   └── utils/              (iterators, helpers)
│
├── infra/                   [LAYER 1: Infrastructure]
│   ├── io/                 (file I/O, serialization)
│   ├── runtime/            (async, zero-copy)
│   ├── api/                (REST API - optional)
│   └── cloud/              (cloud deployment - optional)
│
├── domain/                  [LAYER 2: Domain Primitives]
│   ├── grid/               (ALL grid types: cartesian, cylindrical, staggered)
│   ├── field/              (field representation, unified accessor)
│   ├── boundary/           (ALL boundary conditions: CPML, PML, ABC)
│   ├── medium/             (ALL medium traits: acoustic, optical, thermal, elastic)
│   ├── source/             (source definitions, transducers)
│   ├── sensor/             (sensor placement, recording - NO beamforming)
│   └── signal/             (signal generation, modulation)
│
├── math/                    [LAYER 3: Mathematical Operations]
│   ├── fft/                (FFT implementations)
│   ├── linear_algebra/     (matrix operations, solvers)
│   ├── numerics/           (ALL differential operators, integration schemes)
│   ├── geometry/           (geometric operations)
│   └── ml/                 (ML/PINN - optional)
│
├── physics/                 [LAYER 4: Physics Models]
│   ├── acoustics/          
│   │   ├── propagation/    (wave equations, nonlinearity)
│   │   ├── cavitation/     (bubble dynamics)
│   │   └── analytical/     (analytical solutions)
│   ├── optics/             (light propagation, scattering)
│   ├── thermal/            (heat diffusion, bioheat)
│   └── chemistry/          (sonochemistry, ROS)
│
├── solver/                  [LAYER 5: Numerical Solvers]
│   ├── interface/          (STRONG trait abstractions)
│   ├── forward/            
│   │   ├── fdtd/           (uses math/numerics, domain/boundary)
│   │   ├── pstd/           (uses math/fft, domain/boundary)
│   │   ├── hybrid/         (composes fdtd + pstd)
│   │   └── ...             (other solvers)
│   ├── inverse/            (reconstruction, time-reversal)
│   └── utilities/          (shared solver utilities)
│
├── simulation/              [LAYER 6: Simulation Orchestration]
│   ├── configuration/      (simulation setup)
│   ├── builder/            (simulation construction)
│   └── modalities/         (photoacoustic, elastography setups)
│
├── clinical/                [LAYER 7: Clinical Applications]
│   ├── imaging/            
│   │   ├── workflows/      (clinical imaging protocols)
│   │   └── modalities/     (CEUS, elastography, HIFU)
│   └── therapy/            
│       ├── workflows/      (therapy protocols)
│       └── techniques/     (lithotripsy, SDT, ablation)
│
├── analysis/                [LAYER 8: Post-Processing]
│   ├── beamforming/        (ALL beamforming: uses domain/sensor results)
│   ├── signal_processing/  (filtering, transforms - NO beamforming)
│   ├── visualization/      (rendering, plotting)
│   ├── validation/         (physics validation)
│   └── performance/        (profiling, benchmarking)
│
└── gpu/                     [LAYER 9: Hardware Acceleration - Optional]
    ├── compute/            (GPU kernels)
    ├── memory/             (GPU memory management)
    └── backend/            (WGPU/CUDA abstraction)
```

### 6.2 Dependency Rules (Enforcement Required)

**Strict Rules**:
1. Lower layers NEVER import from higher layers
2. Each layer only imports from layer N-1 (immediate predecessor)
3. Cross-layer imports allowed ONLY for `core` (accessible to all)
4. Optional features (GPU, API) cannot be dependencies of core functionality

**Enforcement**:
- Create `xtask/check_dependencies.rs` script
- CI pipeline checks: `cargo xtask check-deps`
- Fail build on violation

### 6.3 Module Size Compliance

**Current Status**: 944 files, most <500 lines (GRASP compliant)

**7 Known Violations** (from ADR-019):
1. (File paths require audit to identify current >500 line files)

**Action**: Re-audit all files >500 lines, justify or split.

---

## 7. Migration Strategy

### 7.1 Phase 1: Grid Consolidation (Week 1)

**Step 1.1**: Create canonical grid types in `domain/grid/`
```rust
// domain/grid/topologies.rs
pub trait GridTopology {
    fn dims(&self) -> (usize, usize, usize);
    fn linear_index(&self, i: usize, j: usize, k: usize) -> usize;
    fn spatial_indices(&self, idx: usize) -> (usize, usize, usize);
}

// domain/grid/cartesian.rs
pub struct CartesianGrid { ... }
impl GridTopology for CartesianGrid { ... }

// domain/grid/cylindrical.rs  
pub struct CylindricalGrid { ... }
impl GridTopology for CylindricalGrid { ... }

// domain/grid/staggered.rs
pub struct StaggeredGrid { ... }
impl GridTopology for StaggeredGrid { ... }
```

**Step 1.2**: Update all solvers to import from `domain/grid/`
- Remove `solver/forward/axisymmetric/coordinates.rs`
- Remove `solver/forward/fdtd/numerics/staggered_grid.rs`
- Update imports: `use crate::domain::grid::{CartesianGrid, StaggeredGrid};`

**Step 1.3**: Remove `domain/grid/operators/` (move to `math/numerics/operators/`)

### 7.2 Phase 1: Boundary Consolidation (Week 2)

**Step 2.1**: Define `BoundaryCondition` trait in `domain/boundary/traits.rs`
```rust
pub trait BoundaryCondition {
    fn apply(&mut self, field: &mut Field, grid: &dyn GridTopology, dt: f64) -> KwaversResult<()>;
}
```

**Step 2.2**: Implement all boundary types in `domain/boundary/`
- `domain/boundary/cpml.rs` (from existing `domain/boundary/cpml/`)
- `domain/boundary/pml.rs`
- `domain/boundary/abc.rs`

**Step 2.3**: Remove boundary implementations from solvers
- Delete `solver/utilities/cpml_integration.rs`
- Delete `solver/forward/fdtd/numerics/boundary_stencils.rs`
- Solvers call `boundary.apply(field, grid, dt)`

### 7.3 Phase 1: Medium Consolidation (Week 3-4)

**Step 3.1**: Audit all medium trait definitions
- Ensure ALL medium traits live in `domain/medium/heterogeneous/traits/`
- Submodules: `acoustic/`, `optical/`, `thermal/`, `elastic/`, `bubble/`, `viscous/`

**Step 3.2**: Remove medium definitions from physics modules
- `physics/acoustics/` → imports from `domain/medium/heterogeneous/traits/acoustic`
- `physics/optics/` → imports from `domain/medium/heterogeneous/traits/optical`
- `physics/thermal/` → imports from `domain/medium/heterogeneous/traits/thermal`

**Step 3.3**: Remove `AxisymmetricMedium` from `solver/forward/axisymmetric/config.rs`
- Use `domain/medium` types with cylindrical grid projection

### 7.4 Phase 2: Beamforming Consolidation (Week 5)

**Step 4.1**: Audit beamforming algorithms
- List all unique algorithms in both locations
- Identify true duplicates vs. variants

**Step 4.2**: Consolidate to `analysis/beamforming/`
```
analysis/beamforming/
├── time_domain/
│   ├── das.rs
│   └── delay_and_sum.rs
├── frequency_domain/
│   ├── mvdr.rs
│   ├── music.rs
│   └── capon.rs
├── adaptive/
│   └── subspace.rs
└── transmit/  ← NEW: transmit beamforming
    └── focused.rs
```

**Step 4.3**: Remove beamforming from domain/sensor
- `domain/sensor/beamforming/` → Delete (move algorithms to `analysis/`)
- Keep only: `domain/sensor/recording/` (sensor data capture)

**Step 4.4**: Update references
- `domain/source/transducers/phased_array/` → imports from `analysis/beamforming/transmit/`
- `clinical/imaging/` → imports from `analysis/beamforming/`
- Delete `core/utils/sparse_matrix/beamforming.rs`

### 7.5 Phase 2: Clinical Consolidation (Week 6-7)

**Step 5.1**: Move imaging modalities
- `physics/acoustics/imaging/modalities/ceus/` → `clinical/imaging/modalities/ceus/`
- `physics/acoustics/imaging/modalities/elastography/` → `clinical/imaging/modalities/elastography/`
- `physics/acoustics/imaging/modalities/ultrasound/hifu/` → `clinical/therapy/hifu/`

**Step 5.2**: Move therapy techniques
- `physics/acoustics/therapy/` → `clinical/therapy/techniques/`
- Keep in `physics/acoustics/`: Core cavitation models, wave propagation

**Step 5.3**: Clarify separation
- **Physics**: "How does ultrasound propagate and interact with tissue?" (domain-agnostic)
- **Clinical**: "How do we use ultrasound to treat/diagnose patients?" (application-specific)

### 7.6 Phase 2: Math/Numerics Consolidation (Week 8)

**Step 6.1**: Move all operators to `math/numerics/operators/`
- `domain/grid/operators/` → `math/numerics/operators/differential.rs`
- `solver/forward/fdtd/numerics/finite_difference.rs` → `math/numerics/operators/finite_difference.rs`
- `solver/forward/pstd/numerics/operators/` → `math/numerics/operators/spectral.rs`

**Step 6.2**: Define operator traits
```rust
// math/numerics/operators/traits.rs
pub trait DifferentialOperator {
    fn gradient(&self, field: &Field, grid: &dyn GridTopology) -> Field;
    fn divergence(&self, field: &Field, grid: &dyn GridTopology) -> Field;
    fn laplacian(&self, field: &Field, grid: &dyn GridTopology) -> Field;
}

pub struct FiniteDifferenceOperator { order: usize }
pub struct SpectralOperator;
pub struct DGOperator { polynomial_order: usize }
```

**Step 6.3**: Update solvers to consume operators
- FDTD: `let op = FiniteDifferenceOperator::new(2);`
- PSTD: `let op = SpectralOperator::new();`
- DG: `let op = DGOperator::new(4);`

### 7.7 Phase 3: Documentation Consolidation (Week 9)

**Step 7.1**: Root cleanup
```bash
# Keep in root
- README.md
- LICENSE
- Cargo.toml
- Cargo.lock
- gap_audit.md
- prompt.yaml
- build.rs
- clippy.toml
- deny.toml

# Move to docs/architecture/
- ACCURATE_MODULE_ARCHITECTURE.md → docs/architecture/module_structure.md
- ARCHITECTURE_IMPROVEMENT_PLAN.md → docs/architecture/improvement_plan.md
- (etc. for all 19 architecture docs)

# Delete
- errors.txt (build artifact)
```

**Step 7.2**: Create architecture overview
- `docs/architecture/README.md` - Links to all architecture documents
- `docs/architecture/dependency_graph.md` - Visual dependency flow
- `docs/architecture/layer_descriptions.md` - Detailed layer explanations

---

## 8. Testing Strategy During Refactoring

### 8.1 Regression Prevention

**Approach**: Property-based testing for invariants

```rust
// tests/refactoring_invariants.rs
#[test]
fn grid_indexing_invariant() {
    // Property: All grid implementations must satisfy idx/spatial_indices round-trip
    proptest!(|(i in 0usize..100, j in 0usize..100, k in 0usize..100)| {
        let grid = CartesianGrid::new(100, 100, 100, 1e-3, 1e-3, 1e-3)?;
        let idx = grid.linear_index(i, j, k);
        let (i2, j2, k2) = grid.spatial_indices(idx);
        assert_eq!((i, j, k), (i2, j2, k2));
    });
}

#[test]
fn boundary_reflection_invariant() {
    // Property: Reflection coefficient must be < -40dB
    // (Test CPML before and after refactoring)
}
```

### 8.2 Module Isolation Tests

**Per Module**: Create `mod_tests.rs` for each refactored module
- Test public API only (black-box)
- Document expected behavior
- Run before and after refactoring (results must match)

### 8.3 Integration Tests

**Critical Paths**:
1. Grid creation → Medium assignment → Solver step → Boundary application
2. Source → Propagation → Sensor recording
3. Clinical workflow end-to-end

**Run continuously**: On every commit during refactoring

---

## 9. Risk Assessment

### 9.1 High-Risk Areas

| Area | Risk | Mitigation |
|------|------|------------|
| Medium refactoring | Breaking API changes for all physics modules | Create adapter layer during transition |
| Beamforming consolidation | 35+ files, potential algorithm variants | Extensive testing, gradual migration |
| Solver interface strengthening | Could break all 10+ solver implementations | Implement default trait methods for backward compatibility |

### 9.2 Breaking Changes

**Expected Breaking Changes**:
1. Grid imports: `domain::grid::Grid` → `domain::grid::{CartesianGrid, CylindricalGrid}`
2. Medium imports: `domain::medium::Medium` → `domain::medium::heterogeneous::traits::Medium`
3. Beamforming: `domain::sensor::beamforming::*` → `analysis::beamforming::*`
4. Boundary: `solver::*::cpml` → `domain::boundary::cpml`

**Migration Path**: Provide deprecated re-exports for 2-3 releases
```rust
// domain/medium/mod.rs
#[deprecated(since = "2.15.0", note = "Use domain::medium::heterogeneous::traits::Medium")]
pub use heterogeneous::traits::Medium as Medium;
```

---

## 10. Success Metrics

### 10.1 Quantitative Metrics

| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| Module cross-references | ~250 (estimated) | <50 | `cargo-modules` dependency graph |
| Duplicate LOC | ~5000 (estimated) | <500 | Clone detection tools |
| Files >500 lines | 7 known | 0 | `find src -name "*.rs" -exec wc -l {} \; | awk '$1 > 500'` |
| Circular dependencies | ~10 (estimated) | 0 | `cargo-depgraph` |
| Build time (clean) | 71s (from Sprint 96) | <60s | `cargo clean && time cargo build` |
| Build time (incremental) | Unknown | <5s | `touch src/lib.rs && time cargo build` |

### 10.2 Qualitative Metrics

**Code Quality**:
- [ ] Zero clippy warnings (already achieved)
- [ ] Zero unsafe blocks without documentation (already achieved)
- [ ] All modules follow GRASP principles
- [ ] Clear dependency flow (bottom-up only)

**Documentation**:
- [ ] Architecture diagram matches reality
- [ ] All public APIs documented
- [ ] Migration guide complete
- [ ] Example code updated

**Developer Experience**:
- [ ] New contributors can understand module structure in <1 hour
- [ ] Adding new solver requires touching only `solver/forward/<new_solver>/`
- [ ] Adding new physics requires touching only `physics/<new_physics>/`

---

## 11. Recommended Actions (Sprint Plan)

### Immediate (Sprint 1-2)

**Priority 1**: Stop contamination from spreading
1. ✅ Create this audit document
2. [ ] Add `xtask/check_dependencies.rs` - Fail CI on layer violations
3. [ ] Document all current cross-references (automated script)
4. [ ] Freeze feature development until architecture stabilizes

**Priority 2**: Remove build artifacts
1. [ ] Delete `kwavers/errors.txt`
2. [ ] Add to `.gitignore`: `*.txt` (in root), `target/`, `*.log`
3. [ ] Move all markdown docs from root to `docs/architecture/`

### Short-term (Sprint 3-8)

**Phase 1 Execution** (Weeks 1-4):
- Week 1: Grid consolidation
- Week 2: Boundary consolidation
- Week 3-4: Medium consolidation

**Phase 2 Execution** (Weeks 5-8):
- Week 5: Beamforming consolidation
- Week 6-7: Clinical consolidation
- Week 8: Math/Numerics consolidation

### Medium-term (Sprint 9-12)

**Phase 3 Execution** (Weeks 9-10):
- Documentation consolidation
- Architecture diagrams
- Migration guide

**Stabilization** (Weeks 11-12):
- Run full test suite (TIER 3 comprehensive tests)
- Performance benchmarking (compare before/after)
- Update all examples
- Release v2.15.0 with deprecation warnings

---

## 12. Appendix: File Count by Module

```
Module                                Files   Subdirs
====================================================
src/core/                            15      6
src/infra/                           12      5
src/domain/                          120+    10 (deep hierarchy)
├── grid/                            8       1
├── boundary/                        6       1
├── medium/                          35+     8
├── sensor/                          25+     7
├── source/                          30+     7
├── field/                           5       0
├── signal/                          10+     8
└── imaging/                         3       0

src/math/                            40+     6
├── numerics/                        15+     3
├── fft/                             5       0
├── linear_algebra/                  8       0
├── ml/                              10+     4
└── geometry/                        5       0

src/physics/                         100+    6
├── acoustics/                       70+     10
├── optics/                          10+     3
├── thermal/                         8       2
├── chemistry/                       10+     4
└── plugin/                          5       0

src/solver/                          150+    10
├── forward/                         100+    10
├── inverse/                         30+     3
├── analytical/                      10+     1
└── utilities/                       10+     3

src/simulation/                      15+     2
src/clinical/                        10+     2
src/analysis/                        100+    7
├── beamforming (signal_processing)  15+     3
├── visualization/                   30+     8
├── performance/                     40+     10
└── validation/                      10+     3

src/gpu/                             20+     2
====================================================
TOTAL:                               944     ~200 subdirectories
```

---

## 13. Conclusion

**Current Grade**: B+ (85%)
- **Strengths**: GRASP-compliant file sizes, comprehensive test suite, good documentation
- **Weaknesses**: Significant cross-contamination, unclear layer boundaries, duplicated functionality

**Post-Refactoring Target**: A+ (98%)
- Clear vertical hierarchy
- Zero circular dependencies
- No duplicated algorithms
- Fast incremental builds (<5s)
- Maintainable by new contributors

**Estimated Total Effort**: 456 hours (11.4 weeks @ 40h/week, or 22.8 weeks @ 20h/week)

**Recommendation**: **PROCEED with phased refactoring**. The architecture has solid foundations but needs consolidation to scale. Current cross-contamination will compound with each new feature. Refactoring now will pay dividends in maintainability, build times, and developer onboarding.

---

**Audit Complete**  
**Next Step**: Executive approval and resource allocation for Phase 1 execution.