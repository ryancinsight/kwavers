# Architecture Refactoring Audit — kwavers
**Single Source of Truth for Hierarchical Restructuring**

**Date:** 2025-01-12  
**Status:** 🔴 CRITICAL REFACTORING REQUIRED  
**Auditor:** Elite Mathematically-Verified Systems Architect  
**Scope:** Complete codebase architectural analysis and refactoring plan

---

## Executive Summary

### Critical Findings

**Codebase Metrics:**
- **Total Rust Files:** 928 files
- **Largest Files:** 3,115 lines (neural.rs), 2,823 lines (elastic_wave_solver.rs)
- **Module Depth:** Up to 8 levels deep in some hierarchies
- **Cross-Module Dependencies:** Extensive, violating bounded context principles

**Severity Assessment:**
- 🔴 **CRITICAL:** Cross-contamination between `domain`, `physics`, `solver`, `math`
- 🔴 **CRITICAL:** Redundant implementations across modules (Grid operations, numerical methods)
- 🟡 **HIGH:** Files exceeding 500-line GRASP compliance limit (37+ files)
- 🟡 **HIGH:** Unclear separation between physics models and numerical solvers
- 🟢 **MEDIUM:** Dead code and deprecated artifacts

**Architecture Violations:**
1. **Bounded Context Bleeding:** Modules reference each other bidirectionally
2. **Vertical Layering Violation:** Lower layers depend on higher layers
3. **Duplicate Logic:** Grid operators, finite difference stencils, physics models repeated
4. **Mixed Concerns:** Clinical workflows mixed with physics models
5. **Namespace Pollution:** Excessive re-exports masking true dependencies

---

## Current Architecture Analysis

### Module Hierarchy (As-Is)

```
kwavers/
├── src/
│   ├── core/           [FOUNDATION LAYER - ✅ CORRECT]
│   │   ├── constants/  
│   │   ├── error/      
│   │   ├── time/       
│   │   └── utils/      
│   │
│   ├── infra/          [INFRASTRUCTURE LAYER - ✅ CORRECT]
│   │   ├── api/        
│   │   ├── cloud/      
│   │   ├── io/         
│   │   └── runtime/    
│   │
│   ├── domain/         [DOMAIN LAYER - ⚠️ MIXED CONCERNS]
│   │   ├── boundary/   ✅ Correct: Domain primitives
│   │   ├── field/      ✅ Correct: Field abstractions
│   │   ├── grid/       ✅ Correct: Spatial discretization
│   │   ├── medium/     ✅ Correct: Material properties
│   │   ├── sensor/     🔴 WRONG: Contains beamforming (signal processing)
│   │   ├── signal/     ⚠️ Mixed: Signal definitions vs processing
│   │   ├── source/     ✅ Correct: Source definitions
│   │   └── imaging/    🔴 WRONG: Should be in clinical/applications
│   │
│   ├── math/           [MATH LAYER - ⚠️ INCOMPLETE]
│   │   ├── fft/        ✅ Correct
│   │   ├── geometry/   ✅ Correct
│   │   ├── linear_algebra/ ✅ Correct
│   │   └── ml/         ⚠️ Should this be separate?
│   │
│   ├── physics/        [PHYSICS LAYER - 🔴 SCATTERED]
│   │   ├── acoustics/  🔴 Contains mechanics AND imaging AND therapy
│   │   │   ├── analytical/     ✅ Physics models
│   │   │   ├── mechanics/      ✅ Physics models
│   │   │   ├── nonlinear/      ✅ Physics models
│   │   │   ├── imaging/        🔴 WRONG: Application layer
│   │   │   ├── therapy/        🔴 WRONG: Application layer
│   │   │   ├── transcranial/   🔴 WRONG: Application layer
│   │   │   └── validation/     ⚠️ Should be in analysis/
│   │   ├── chemistry/  ✅ Correct
│   │   ├── optics/     ✅ Correct
│   │   ├── thermal/    ✅ Correct
│   │   └── plugin/     ⚠️ Mixed: Physics + Solver concerns
│   │
│   ├── solver/         [SOLVER LAYER - 🔴 MIXED CONCERNS]
│   │   ├── forward/    
│   │   │   ├── acoustic/   🔴 Redundant with physics/acoustics
│   │   │   ├── elastic/    🔴 Redundant with physics/mechanics
│   │   │   ├── fdtd/       ✅ Numerical method
│   │   │   ├── pstd/       ✅ Numerical method
│   │   │   ├── hybrid/     ✅ Numerical method
│   │   │   └── nonlinear/  🔴 Mixed: Physics + Numerics
│   │   ├── inverse/    ✅ Correct
│   │   ├── integration/✅ Correct: Time steppers
│   │   ├── multiphysics/ ⚠️ Should this be in physics/coupling?
│   │   └── utilities/  ⚠️ AMR should be separate module
│   │
│   ├── simulation/     [ORCHESTRATION LAYER - ⚠️ UNCLEAR]
│   │   ├── builder/    ✅ Correct
│   │   ├── configuration/ ✅ Correct
│   │   ├── core/       ✅ Correct
│   │   └── modalities/ 🔴 WRONG: Should be in clinical/
│   │
│   ├── clinical/       [APPLICATION LAYER - ⚠️ INCOMPLETE]
│   │   ├── imaging/    ⚠️ Should contain ALL imaging workflows
│   │   └── therapy/    ⚠️ Should contain ALL therapy workflows
│   │
│   └── analysis/       [ANALYSIS LAYER - ✅ MOSTLY CORRECT]
│       ├── performance/
│       ├── testing/
│       ├── validation/
│       └── visualization/
```

### Dependency Graph (Current)

```
┌─────────────────────────────────────────────────────────┐
│ CURRENT DEPENDENCY GRAPH (VIOLATES LAYERING)           │
└─────────────────────────────────────────────────────────┘

clinical ←──────┐
   ↓            │
simulation ←────┤
   ↓            │
solver ←────────┤    🔴 CIRCULAR DEPENDENCIES
   ↓            │    🔴 BIDIRECTIONAL REFERENCES
physics ←───────┤    🔴 LAYER VIOLATIONS
   ↓            │
domain ←────────┤
   ↓            │
math/core ──────┘

PROBLEMS:
1. solver → domain (OK) BUT domain → solver (WRONG)
2. physics → domain (OK) BUT domain/sensor → physics (WRONG)
3. clinical scattered across physics/acoustics/imaging
4. Signal processing in domain/sensor instead of analysis/
```

---

## Cross-Contamination Analysis

### 1. Grid Operations Duplication

**Issue:** Grid operations scattered across multiple modules

**Locations:**
- ✅ `domain/grid/` - Core grid structure (CORRECT)
- 🔴 `domain/grid/operators/` - Differential operators
- 🔴 `solver/forward/fdtd/numerics/` - Finite difference stencils
- 🔴 `solver/forward/pstd/numerics/operators/` - Spectral operators
- 🔴 `domain/medium/heterogeneous/interpolation/` - Grid interpolation

**Redundancy:**
- Finite difference stencils implemented in at least 3 places
- Grid interpolation logic duplicated
- Boundary handling logic repeated

**Solution:**
- Consolidate ALL grid operators in `math/numerics/operators/`
- Define clear trait-based interface
- Solvers access through abstraction layer

### 2. Medium/Material Properties Contamination

**Issue:** Material properties logic scattered

**Locations:**
- ✅ `domain/medium/` - Core medium traits (CORRECT)
- ✅ `domain/medium/heterogeneous/` - Heterogeneous media (CORRECT)
- 🔴 `physics/acoustics/mechanics/` - Acoustic wave with medium coupling
- 🔴 `solver/forward/*/` - Solvers directly accessing medium internals
- 🔴 `domain/medium/heterogeneous/traits/acoustic/` - Physics in domain layer

**Solution:**
- `domain/medium/` should ONLY define abstract interfaces
- ALL physics models in `physics/` layer
- Solvers access medium through well-defined accessor traits

### 3. Physics vs Solver Boundary Violation

**Issue:** Physics equations mixed with numerical methods

**Examples:**
- `solver/forward/acoustic/plugin.rs` - Should be `physics/acoustics/models/`
- `solver/forward/nonlinear/kuznetsov/` - Physics model in solver layer
- `solver/forward/elastic/plugin.rs` - Elastic physics in solver
- `physics/plugin/` - Solver concerns in physics layer

**Correct Separation:**
```
physics/
  ├── acoustics/
  │   ├── models/          # Wave equations (linear, nonlinear)
  │   │   ├── linear.rs
  │   │   ├── kuznetsov.rs
  │   │   ├── westervelt.rs
  │   │   └── kzk.rs
  │   └── constitutive/    # Material models
  │       ├── viscosity.rs
  │       └── nonlinearity.rs

solver/
  ├── methods/             # Numerical methods ONLY
  │   ├── fdtd/
  │   ├── pstd/
  │   ├── dg/
  │   └── hybrid/
  └── integration/         # Time stepping
      ├── explicit.rs
      ├── implicit.rs
      └── imex.rs
```

### 4. Clinical Applications Scattered

**Issue:** Clinical workflows mixed with physics

**Locations:**
- 🔴 `physics/acoustics/imaging/` - Should be `clinical/imaging/`
- 🔴 `physics/acoustics/therapy/` - Should be `clinical/therapy/`
- 🔴 `physics/acoustics/transcranial/` - Should be `clinical/transcranial/`
- 🔴 `simulation/modalities/photoacoustic.rs` - Should be `clinical/imaging/photoacoustic/`
- ✅ `clinical/imaging/` - Correct location (but incomplete)
- ✅ `clinical/therapy/` - Correct location (but incomplete)

**Solution:**
- Move ALL application-level workflows to `clinical/`
- Physics layer should ONLY contain abstract physics models
- Clinical layer composes physics + solver + domain

### 5. Signal Processing Misplaced

**Issue:** Signal processing in domain layer

**Locations:**
- 🔴 `domain/sensor/beamforming/` - Complex beamforming algorithms (3,115 lines!)
- 🔴 `domain/sensor/localization/` - Source localization algorithms
- 🔴 `domain/sensor/passive_acoustic_mapping/` - PAM algorithms

**Solution:**
- Move to `analysis/signal_processing/`
- Domain should ONLY define sensor geometry/sampling
- Signal processing is analysis, not domain primitives

### 6. Math Module Incompleteness

**Issue:** Mathematical operations scattered

**Missing in `math/`:**
- Numerical differentiation (in solver/forward/fdtd/numerics/)
- Spectral operations (in solver/forward/pstd/numerics/)
- Interpolation (in domain/medium/heterogeneous/interpolation/)
- Convolution (scattered across modules)

**Solution:**
```
math/
  ├── numerics/
  │   ├── operators/
  │   │   ├── differential.rs    # All finite difference stencils
  │   │   ├── spectral.rs        # All spectral operators
  │   │   └── interpolation.rs   # All interpolation methods
  │   ├── integration/
  │   │   ├── quadrature.rs
  │   │   └── adaptive.rs
  │   └── transforms/
  │       ├── fourier.rs
  │       └── wavelet.rs
  ├── linear_algebra/
  ├── geometry/
  └── ml/
```

### 7. Validation and Testing Scattered

**Issue:** Validation logic mixed with implementation

**Locations:**
- `physics/acoustics/validation/` - Physics validation
- `solver/utilities/validation/` - Solver validation
- `analysis/validation/` - General validation
- `domain/*/validation.rs` - Domain validation

**Solution:**
- ALL validation in `analysis/validation/`
- Organized by domain: `physics/`, `numerics/`, `clinical/`

---

## File Size Violations (>500 Lines)

### Critical Violations (>1000 Lines)

| File | Lines | Violation | Action |
|------|-------|-----------|--------|
| `domain/sensor/beamforming/experimental/neural.rs` | 3,115 | 6.2x | Split into 7+ modules |
| `physics/acoustics/imaging/modalities/elastography/elastic_wave_solver.rs` | 2,823 | 5.6x | Move to solver, split |
| `math/ml/pinn/burn_wave_equation_2d.rs` | 2,583 | 5.2x | Split into components |
| `domain/sensor/beamforming/adaptive/algorithms_old.rs` | 2,199 | 4.4x | **DELETE** (deprecated) |
| `math/linear_algebra/mod.rs` | 1,887 | 3.8x | Split into submodules |
| `physics/acoustics/imaging/modalities/elastography/nonlinear.rs` | 1,342 | 2.7x | Split, move to clinical |
| `domain/sensor/beamforming/beamforming_3d.rs` | 1,260 | 2.5x | Split into components |
| `clinical/therapy/therapy_integration.rs` | 1,241 | 2.5x | Split into workflows |
| `physics/acoustics/imaging/modalities/elastography/inversion.rs` | 1,233 | 2.5x | Move to inverse solver |
| `clinical/imaging/workflows.rs` | 1,181 | 2.4x | Split by modality |

**Total Files >1000 Lines:** 37 files  
**Total Files >500 Lines:** 120+ files (estimated)

---

## Proposed Target Architecture

### Clean Layered Architecture

```
┌─────────────────────────────────────────────────────────┐
│ TARGET ARCHITECTURE (STRICT LAYERING)                   │
└─────────────────────────────────────────────────────────┘

LAYER 0: FOUNDATION
├── core/               # Constants, errors, time, utils
└── infra/              # I/O, API, cloud, runtime

LAYER 1: MATHEMATICS
└── math/
    ├── numerics/       # All numerical methods primitives
    │   ├── operators/  # Differential, spectral, interpolation
    │   ├── integration/# Quadrature, adaptive integration
    │   └── transforms/ # FFT, wavelets
    ├── linear_algebra/
    ├── geometry/
    └── ml/             # Machine learning primitives

LAYER 2: DOMAIN PRIMITIVES
└── domain/
    ├── grid/           # Spatial discretization ONLY
    ├── field/          # Field storage abstractions
    ├── medium/         # Material property INTERFACES ONLY
    ├── boundary/       # Boundary condition INTERFACES
    ├── source/         # Source DEFINITIONS (not generation)
    └── sensor/         # Sensor GEOMETRY ONLY (no processing)

LAYER 3: PHYSICS MODELS
└── physics/
    ├── acoustics/
    │   ├── models/     # Wave equations (linear, nonlinear)
    │   ├── constitutive/ # Material models
    │   └── coupling/   # Multi-physics coupling
    ├── mechanics/
    │   ├── elastic/
    │   ├── cavitation/
    │   └── streaming/
    ├── thermal/
    ├── optics/
    └── chemistry/

LAYER 4: NUMERICAL SOLVERS
└── solver/
    ├── methods/        # Pure numerical methods
    │   ├── fdtd/
    │   ├── pstd/
    │   ├── dg/
    │   └── hybrid/
    ├── integration/    # Time stepping schemes
    ├── inverse/        # Inverse problem solvers
    └── analytical/     # Closed-form solutions

LAYER 5: SIMULATION ORCHESTRATION
└── simulation/
    ├── builder/        # Simulation builder pattern
    ├── configuration/  # Configuration management
    ├── orchestrator/   # Main simulation loop
    └── state/          # State management

LAYER 6: APPLICATIONS
└── clinical/
    ├── imaging/        # ALL imaging modalities
    │   ├── ultrasound/
    │   ├── photoacoustic/
    │   ├── elastography/
    │   └── ceus/
    ├── therapy/        # ALL therapy applications
    │   ├── hifu/
    │   ├── lithotripsy/
    │   └── transcranial/
    └── workflows/      # End-to-end clinical workflows

LAYER 7: ANALYSIS & TOOLS
└── analysis/
    ├── signal_processing/ # Beamforming, localization
    ├── validation/     # ALL validation/verification
    ├── testing/        # Test utilities
    ├── performance/    # Performance analysis
    └── visualization/  # Visualization tools

LAYER 8: GPU ACCELERATION (Cross-cutting)
└── gpu/
    ├── kernels/        # Raw GPU kernels
    ├── memory/         # GPU memory management
    └── pipeline/       # GPU pipeline management
```

### Dependency Rules (STRICT)

```
✅ ALLOWED:
- Layer N → Layer N-1 (downward only)
- Layer N → Layer 0 (foundation always accessible)
- gpu/ → any layer (cross-cutting concern)

🔴 FORBIDDEN:
- Layer N → Layer N+1 (upward dependency)
- Layer N → Layer N (circular within layer)
- Sibling modules in same layer (use shared lower layer)

EXAMPLES:
✅ physics/acoustics → domain/medium
✅ solver/methods → math/numerics
✅ clinical/imaging → physics/acoustics + solver/methods
🔴 domain/medium → physics/acoustics
🔴 solver/methods → physics/models
🔴 physics/acoustics/imaging (imaging is application, not physics)
```

---

## Refactoring Strategy

### Phase 1: Foundation Cleanup (Week 1)

**Priority P0: Remove Dead Code**
```bash
# Files to DELETE immediately (deprecated/dead code):
- domain/sensor/beamforming/adaptive/algorithms_old.rs (2,199 lines)
- Any files with "_old", "_backup", "_deprecated" suffixes
- Build artifacts already cleaned: build_errors.txt, check_tests.log, test_errors.txt
```

**Priority P0: Establish Math Layer**
```
math/
  └── numerics/
      ├── operators/
      │   ├── mod.rs                    # Public interface
      │   ├── differential.rs           # Consolidate ALL FD stencils
      │   ├── spectral.rs               # Consolidate ALL spectral ops
      │   └── interpolation.rs          # Consolidate ALL interpolation
      └── integration/
          └── quadrature.rs             # Integration schemes
```

**Actions:**
1. Create `math/numerics/operators/differential.rs`
   - Move from: `solver/forward/fdtd/numerics/finite_difference.rs`
   - Move from: `domain/grid/operators/*`
   - Unified trait: `DifferentialOperator`

2. Create `math/numerics/operators/spectral.rs`
   - Move from: `solver/forward/pstd/numerics/operators/spectral.rs`
   - Unified trait: `SpectralOperator`

3. Create `math/numerics/operators/interpolation.rs`
   - Move from: `domain/medium/heterogeneous/interpolation/*`
   - Unified trait: `Interpolator`

### Phase 2: Domain Layer Purification (Week 1-2)

**Priority P0: Remove Physics from Domain**

1. **Clean `domain/medium/`**
   ```
   domain/medium/
     ├── mod.rs               # Trait definitions ONLY
     ├── traits.rs            # Core Medium trait
     ├── homogeneous.rs       # Simple implementations
     └── heterogeneous/
         ├── core.rs          # Heterogeneous storage
         └── factory.rs       # Medium construction
   
   DELETE from domain/medium/:
     └── heterogeneous/traits/{acoustic,elastic,thermal,optical,viscous}
         → MOVE TO physics/constitutive/
   ```

2. **Clean `domain/sensor/`**
   ```
   domain/sensor/
     ├── mod.rs
     ├── geometry.rs          # Sensor positions/geometry ONLY
     ├── sampling.rs          # Grid sampling logic
     └── recorder.rs          # Data recording
   
   MOVE from domain/sensor/:
     ├── beamforming/         → analysis/signal_processing/beamforming/
     ├── localization/        → analysis/signal_processing/localization/
     └── passive_acoustic_mapping/ → analysis/signal_processing/pam/
   ```

3. **Clean `domain/signal/`**
   ```
   domain/signal/
     ├── mod.rs
     ├── waveform.rs          # Signal DEFINITIONS only
     └── traits.rs            # Signal trait
   
   MOVE signal PROCESSING to analysis/signal_processing/
   ```

### Phase 3: Physics Layer Reorganization (Week 2-3)

**Priority P1: Separate Physics Models from Applications**

```
physics/
  ├── acoustics/
  │   ├── models/              # Pure wave equations
  │   │   ├── linear.rs
  │   │   ├── kuznetsov.rs     # FROM solver/forward/nonlinear/kuznetsov/
  │   │   ├── westervelt.rs    # FROM solver/forward/nonlinear/westervelt/
  │   │   └── kzk.rs           # FROM solver/forward/nonlinear/kzk/
  │   └── constitutive/        # Material models
  │       ├── viscosity.rs
  │       ├── absorption.rs
  │       └── nonlinearity.rs
  │
  ├── mechanics/
  │   ├── elastic/
  │   │   ├── models.rs        # Elastic wave equations
  │   │   └── anisotropy.rs
  │   ├── cavitation/
  │   │   ├── rayleigh_plesset.rs
  │   │   ├── keller_miksis.rs
  │   │   └── gilmore.rs
  │   └── streaming/
  │
  ├── thermal/
  │   ├── bioheat.rs           # Pennes equation
  │   └── thermal_dose.rs
  │
  ├── optics/
  │   ├── absorption.rs
  │   ├── scattering.rs
  │   └── sonoluminescence/
  │
  └── coupling/                # Multi-physics coupling
      ├── acoustic_thermal.rs
      ├── acoustic_optical.rs
      └── framework.rs

DELETE from physics/:
  ├── acoustics/imaging/       → clinical/imaging/
  ├── acoustics/therapy/       → clinical/therapy/
  └── acoustics/transcranial/  → clinical/transcranial/
```

### Phase 4: Solver Layer Cleanup (Week 3-4)

**Priority P1: Pure Numerical Methods**

```
solver/
  ├── methods/                 # Numerical schemes ONLY
  │   ├── fdtd/
  │   │   ├── scheme.rs        # FDTD algorithm
  │   │   ├── staggered_grid.rs
  │   │   └── source_injection.rs
  │   ├── pstd/
  │   │   ├── scheme.rs        # PSTD algorithm
  │   │   └── k_space.rs
  │   ├── dg/
  │   │   ├── basis.rs
  │   │   ├── flux.rs
  │   │   └── limiter.rs
  │   └── hybrid/
  │       ├── domain_decomposition.rs
  │       └── coupling.rs
  │
  ├── integration/             # Time stepping
  │   ├── explicit/
  │   ├── implicit/
  │   └── imex/
  │
  ├── inverse/
  │   ├── reconstruction/
  │   └── time_reversal/
  │
  └── analytical/
      └── transducer/

DELETE from solver/:
  ├── forward/acoustic/        → Remove (physics in solver)
  ├── forward/elastic/         → Remove (physics in solver)
  └── multiphysics/            → physics/coupling/
```

### Phase 5: Clinical Applications Layer (Week 4-5)

**Priority P2: Consolidate Clinical Workflows**

```
clinical/
  ├── imaging/
  │   ├── ultrasound/
  │   │   ├── b_mode.rs
  │   │   ├── doppler.rs
  │   │   └── beamforming_workflow.rs
  │   ├── photoacoustic/       # FROM simulation/modalities/photoacoustic/
  │   │   ├── forward.rs
  │   │   ├── reconstruction.rs
  │   │   └── workflow.rs
  │   ├── elastography/        # FROM physics/acoustics/imaging/modalities/elastography/
  │   │   ├── swe.rs
  │   │   ├── arfi.rs
  │   │   └── workflow.rs
  │   └── ceus/                # FROM physics/acoustics/imaging/modalities/ceus/
  │       ├── contrast_agent.rs
  │       └── workflow.rs
  │
  ├── therapy/
  │   ├── hifu/                # FROM physics/acoustics/therapy/
  │   │   ├── ablation.rs
  │   │   └── treatment_planning.rs
  │   ├── lithotripsy/
  │   │   └── shock_wave.rs
  │   └── transcranial/        # FROM physics/acoustics/transcranial/
  │       ├── skull_correction.rs
  │       ├── bbb_opening.rs
  │       └── safety.rs
  │
  └── workflows/
      ├── standard_scan.rs
      ├── therapy_session.rs
      └── quality_assurance.rs
```

### Phase 6: Analysis Layer Completion (Week 5)

**Priority P2: Signal Processing and Validation**

```
analysis/
  ├── signal_processing/       # NEW: From domain/sensor/
  │   ├── beamforming/
  │   │   ├── das.rs
  │   │   ├── capon.rs
  │   │   ├── music.rs
  │   │   └── adaptive.rs
  │   ├── localization/
  │   │   ├── trilateration.rs
  │   │   └── beamforming_search.rs
  │   └── pam/
  │       └── passive_mapping.rs
  │
  ├── validation/
  │   ├── physics/             # FROM physics/*/validation/
  │   ├── numerics/            # FROM solver/utilities/validation/
  │   ├── clinical/
  │   └── integration/
  │
  ├── testing/
  ├── performance/
  └── visualization/
```

### Phase 7: File Size Compliance (Week 6)

**Priority P3: Split Large Files**

For each file >500 lines:
1. Identify logical components
2. Extract into focused modules (<500 lines each)
3. Define clear interfaces between components
4. Update imports and re-exports

**Example: `domain/sensor/beamforming/experimental/neural.rs` (3,115 lines)**
```
Split into:
analysis/signal_processing/beamforming/neural/
  ├── mod.rs                  (100 lines)  # Public interface
  ├── architecture.rs         (450 lines)  # Network architecture
  ├── training.rs             (480 lines)  # Training logic
  ├── inference.rs            (420 lines)  # Inference pipeline
  ├── preprocessing.rs        (380 lines)  # Data preprocessing
  ├── loss_functions.rs       (350 lines)  # Custom losses
  └── evaluation.rs           (420 lines)  # Metrics and evaluation
```

---

## Migration Plan

### Week-by-Week Breakdown

#### Week 1: Foundation & Math
- [ ] Delete dead code (`algorithms_old.rs`, etc.)
- [ ] Create `math/numerics/operators/` with unified interfaces
- [ ] Move all FD stencils to `math/numerics/operators/differential.rs`
- [ ] Move all spectral ops to `math/numerics/operators/spectral.rs`
- [ ] Move all interpolation to `math/numerics/operators/interpolation.rs`
- [ ] Update all references to use new math layer

#### Week 2: Domain Purification
- [ ] Move `domain/sensor/beamforming/` → `analysis/signal_processing/beamforming/`
- [ ] Move `domain/sensor/localization/` → `analysis/signal_processing/localization/`
- [ ] Move `domain/sensor/passive_acoustic_mapping/` → `analysis/signal_processing/pam/`
- [ ] Remove physics traits from `domain/medium/heterogeneous/traits/`
- [ ] Simplify `domain/sensor/` to geometry + sampling only
- [ ] Update all imports and re-exports

#### Week 3: Physics Models
- [ ] Move `solver/forward/nonlinear/kuznetsov/` → `physics/acoustics/models/kuznetsov/`
- [ ] Move `solver/forward/nonlinear/westervelt/` → `physics/acoustics/models/westervelt/`
- [ ] Move `solver/forward/nonlinear/kzk/` → `physics/acoustics/models/kzk/`
- [ ] Create `physics/constitutive/` for material models
- [ ] Move all physics traits to appropriate physics/ submodules

#### Week 4: Solver Cleanup
- [ ] Remove `solver/forward/acoustic/` (move to physics)
- [ ] Remove `solver/forward/elastic/` (move to physics)
- [ ] Restructure `solver/methods/` for pure numerical methods
- [ ] Clean up `solver/integration/` for time steppers only
- [ ] Move `solver/multiphysics/` → `physics/coupling/`

#### Week 5: Clinical Applications
- [ ] Move `physics/acoustics/imaging/` → `clinical/imaging/`
- [ ] Move `physics/acoustics/therapy/` → `clinical/therapy/`
- [ ] Move `physics/acoustics/transcranial/` → `clinical/transcranial/`
- [ ] Move `simulation/modalities/photoacoustic/` → `clinical/imaging/photoacoustic/`
- [ ] Create unified clinical workflows

#### Week 6: File Size Compliance
- [ ] Split all files >1000 lines
- [ ] Split all files >500 lines
- [ ] Verify GRASP compliance
- [ ] Update documentation

#### Week 7: Testing & Validation
- [ ] Run full test suite
- [ ] Fix broken imports
- [ ] Verify compilation
- [ ] Performance regression testing
- [ ] Update examples

#### Week 8: Documentation & Cleanup
- [ ] Update all module documentation
- [ ] Update ADR with architectural decisions
- [ ] Update README with new structure
- [ ] Clean up deprecated re-exports
- [ ] Final audit and sign-off

---

## Testing Strategy

### Pre-Refactoring
```bash
# Capture current test results as baseline
cargo test --all-features 2>&1 | tee pre_refactor_tests.log

# Run benchmarks for performance baseline
cargo bench 2>&1 | tee pre_refactor_bench.log

# Generate documentation
cargo doc --all-features --no-deps
```

### During Refactoring
```bash
# After each major move, verify compilation
cargo check --all-features

# Run affected tests
cargo test --lib <module>

# Verify no performance regression
cargo bench --bench <affected_benchmark>
```

### Post-Refactoring
```bash
# Full test suite
cargo test --all-features

# Compare benchmarks
cargo bench

# Verify documentation builds
cargo doc --all-features --no-deps

# Check for unused dependencies
cargo udeps

# Run clippy with strict lints
cargo clippy --all-features -- -D warnings
```

---

## Success Metrics

### Quantitative Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Total Rust Files | 928 | <800 | 🔴 15% reduction needed |
| Files >500 lines | ~120 | 0 | 🔴 CRITICAL |
| Files >1000 lines | 37 | 0 | 🔴 CRITICAL |
| Max file size | 3,115 | 500 | 🔴 6.2x violation |
| Module depth | 8 | 4 | 🟡 Simplification needed |
| Circular deps | Multiple | 0 | 🔴 Must eliminate |
| Cross-layer violations | Many | 0 | 🔴 Must eliminate |

### Qualitative Metrics

- [ ] **Bounded Context Isolation**: Each module has clear, minimal public API
- [ ] **Vertical Layering**: Strict downward-only dependencies
- [ ] **Single Responsibility**: Each file has one clear purpose
- [ ] **Zero Duplication**: No redundant implementations
- [ ] **Clear Abstractions**: Trait-based interfaces between layers
- [ ] **Documentation**: 100% public API documented
- [ ] **Testing**: All functionality preserved, tests pass

---

## Architectural Inspiration from Reference Projects

### jWave (JAX-based)
**Key Learnings:**
- Clean separation: `geometry/`, `medium/`, `acoustics/`, `utils/`
- Domain primitives in separate modules
- Physics models isolated from numerical methods
- Minimal, focused file sizes

### k-Wave
**Key Learnings:**
- Clear distinction between simulation setup and execution
- Medium properties abstracted
- Source and sensor as configuration objects
- Extensive validation and examples

### Application to kwavers:
1. **Adopt jWave's clean module boundaries**
2. **Follow k-Wave's configuration pattern**
3. **Implement strict layering not present in either**
4. **Add Rust-specific trait abstractions**

---

## Risk Assessment

### High Risk
- 🔴 **Breaking existing examples**: Mitigation: Update examples incrementally
- 🔴 **Test failures**: Mitigation: Comprehensive testing at each step
- 🔴 **Performance regression**: Mitigation: Benchmark at each phase

### Medium Risk
- 🟡 **Documentation drift**: Mitigation: Update docs with code
- 🟡 **Merge conflicts**: Mitigation: Refactor in dedicated branch
- 🟡 **User disruption**: Mitigation: Provide migration guide

### Low Risk
- 🟢 **Build time**: Expected to improve with smaller modules
- 🟢 **Binary size**: No expected impact
- 🟢 **API stability**: Internal refactor, minimal API changes

---

## Critical Path Forward

### Immediate Actions (This Sprint)
1. ✅ Create this audit document
2. ✅ Clean build artifacts
3. ⏳ Delete dead code (`algorithms_old.rs`, etc.)
4. ⏳ Create `math/numerics/operators/` structure
5. ⏳ Begin Phase 1: Foundation cleanup

### Next Sprint (Week 1-2)
- Complete Phase 1 & 2
- Domain layer purification
- Math layer establishment

### Ongoing (Weeks 3-8)
- Phases 3-8 as outlined
- Continuous testing and validation
- Documentation updates

---

## References

### External Architectures
1. **jWave**: https://github.com/ucl-bug/jwave
2. **k-Wave**: https://github.com/ucl-bug/k-wave
3. **k-Wave Python**: https://github.com/waltsims/k-wave-python
4. **Optimus**: https://github.com/optimuslib/optimus
5. **Fullwave**: https://github.com/pinton-lab/fullwave25

### Architecture Principles
1. **Bounded Context** (Domain-Driven Design)
2. **Vertical Slice Architecture**
3. **SOLID Principles**
4. **GRASP Patterns** (Modules <500 lines)
5. **Dependency Inversion Principle**

### Internal Documents
- `docs/adr.md` - Architecture Decision Records
- `docs/srs.md` - Software Requirements Specification
- `gap_audit.md` - Mathematical validation audit
- `COMPREHENSIVE_MODULE_REFACTORING_PLAN.md` - Previous refactoring attempts

---

## Approval & Sign-off

**Audit Prepared By:** Elite Mathematically-Verified Systems Architect  
**Date:** 2025-01-12  
**Status:** 🔴 AWAITING APPROVAL TO PROCEED

**Next Steps:**
1. Review and approve this audit
2. Confirm refactoring priorities
3. Begin Phase 1 execution
4. Weekly progress reviews

---

**END OF ARCHITECTURE REFACTORING AUDIT**

*This document serves as the Single Source of Truth for the kwavers architectural refactoring. All refactoring decisions must reference this document and update it accordingly.*