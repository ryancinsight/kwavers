# Kwavers Module Architecture Map

**Quick Reference Guide for Developers**  
**Last Updated**: 2024-01-09  
**Status**: Target Architecture (Post-Refactor)

---

## Layer Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 7: APPLICATION (clinical/)                               │
│  Purpose: Clinical workflows, protocols, user-facing APIs       │
│  Dependencies: ALL layers below                                 │
├─────────────────────────────────────────────────────────────────┤
│  Layer 6: ANALYSIS (analysis/)                                  │
│  Purpose: Post-processing, signal analysis, visualization       │
│  Dependencies: Layers 1-5                                       │
├─────────────────────────────────────────────────────────────────┤
│  Layer 5: SOLVER (solver/)                                      │
│  Purpose: Numerical methods, time integration, solvers          │
│  Dependencies: Layers 1-4                                       │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4: PHYSICS (physics/)                                    │
│  Purpose: Physical models, equations, material behavior         │
│  Dependencies: Layers 1-3                                       │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: DOMAIN (domain/)                                      │
│  Purpose: Domain entities (grid, medium, sensor, source)        │
│  Dependencies: Layers 1-2                                       │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: MATH (math/)                                          │
│  Purpose: Mathematical operations (FFT, linear algebra, ML)     │
│  Dependencies: Layer 1 only                                     │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: CORE (core/)                                          │
│  Purpose: Foundation (errors, constants, time, utils)           │
│  Dependencies: NONE (foundation layer)                          │
└─────────────────────────────────────────────────────────────────┘

RULE: Each layer may only depend on layers BELOW it (no upward dependencies)
```

---

## Module Directory Structure

### Layer 1: Core (Foundation)

```
core/
├── constants/              # ALL constants (physics, numerical, fundamental)
│   ├── fundamental.rs      # Universal: c, G, h, k_B, R
│   ├── physics.rs          # SOUND_SPEED_WATER, DENSITY_WATER, etc.
│   ├── numerical.rs        # CFL limits, tolerances, convergence criteria
│   └── mod.rs              # SINGLE SOURCE OF TRUTH
├── error/                  # Error types, Result<T, E>
│   ├── types/              # Error variants
│   └── mod.rs
├── time/                   # Time utilities, duration handling
└── utils/                  # Generic utilities ONLY (no domain logic)
    └── mod.rs

Dependencies: std, external crates only
Purpose: Zero-dependency foundation that could be extracted as separate crate
```

**What Goes Here**:
- ✅ Universal constants
- ✅ Error types
- ✅ Generic utilities (string manipulation, etc.)

**What Doesn't**:
- ❌ Domain-specific logic (Grid, Medium, etc.)
- ❌ Math operations (FFT, linear algebra)
- ❌ Any imports from higher layers

---

### Layer 2: Math (Mathematical Operations)

```
math/
├── fft/                    # Fast Fourier Transform
│   ├── fft_processor.rs    # Core FFT implementation
│   └── mod.rs
├── linear_algebra/         # Matrix operations, solvers
│   ├── sparse.rs           # Sparse matrix utilities
│   └── mod.rs
├── geometry/               # Spatial calculations, transformations
├── ml/                     # Machine learning (GENERIC ONLY)
│   ├── optimization/       # Generic optimizers
│   ├── pinn/               # Generic PINN framework
│   │   └── physics_traits.rs  # Abstract physics interfaces
│   └── uncertainty/        # Uncertainty quantification
└── numerics/               # Generic numerical utilities
    ├── integration/        # Quadrature, ODE solvers
    └── operators/          # Differential operators

Dependencies: core/ only
Purpose: Reusable mathematical operations, no physics knowledge
```

**What Goes Here**:
- ✅ FFT, wavelet transforms
- ✅ Linear algebra operations
- ✅ Generic ML infrastructure
- ✅ Numerical integration methods

**What Doesn't**:
- ❌ Physics-specific implementations (e.g., cavitation PINN)
- ❌ Domain entities (Grid, Medium)
- ❌ Solver-specific logic

---

### Layer 3: Domain (Domain Entities)

```
domain/
├── grid/                   # Computational mesh
│   ├── operators/          # Grid-specific operators
│   └── mod.rs
├── medium/                 # Material properties (data structures)
│   ├── homogeneous/        # Uniform media
│   ├── heterogeneous/      # Spatially varying media
│   ├── absorption/         # Absorption models
│   └── mod.rs
├── sensor/                 # Data acquisition infrastructure
│   ├── recorder/           # Data recording (NO signal processing)
│   └── mod.rs              # NO beamforming (moved to analysis/)
├── source/                 # Transducer geometry (NO physics calculations)
│   ├── transducers/        # Array geometry, element positions
│   └── mod.rs
├── signal/                 # Signal generation (waveforms, pulses)
│   ├── waveform/
│   ├── pulse/
│   └── mod.rs
├── boundary/               # Boundary conditions (PML, CPML)
│   └── cpml/
└── field/                  # Field management, indices

Dependencies: core/, math/
Purpose: Infrastructure and data structures, NO behavior/physics
```

**What Goes Here**:
- ✅ Grid structure, coordinate systems
- ✅ Material property storage
- ✅ Transducer geometry
- ✅ Signal waveform definitions

**What Doesn't**:
- ❌ Wave propagation physics (goes in physics/)
- ❌ Beamforming (goes in analysis/)
- ❌ Numerical solvers (goes in solver/)

**Key Principle**: Domain = "WHAT" (data), not "HOW" (algorithms)

---

### Layer 4: Physics (Physical Models)

```
physics/
├── acoustics/              # Acoustic wave physics
│   ├── analytical/         # Analytical solutions
│   │   ├── patterns/       # Beam patterns, focusing
│   │   └── propagation/    # Wave propagation models
│   ├── mechanics/          # Physical phenomena
│   │   ├── acoustic_wave/  # Linear/nonlinear acoustics
│   │   ├── cavitation/     # Bubble dynamics
│   │   │   └── pinn.rs     # Cavitation-specific PINN
│   │   ├── elastic_wave/   # Elastic wave propagation
│   │   └── streaming/      # Acoustic streaming
│   ├── imaging/            # Imaging physics models
│   └── therapy/            # Therapeutic physics
│       └── lithotripsy/    # Lithotripsy physics
├── optics/                 # Light propagation
│   ├── scattering/         # Light scattering
│   └── sonoluminescence/   # Light from cavitation
├── thermal/                # Heat transfer
│   └── diffusion/          # Thermal diffusion
├── chemistry/              # Chemical reactions
│   └── reaction_kinetics/  # Reaction rate models
└── analytical/             # Analytical solutions (moved from solver/)
    └── solvers/            # Closed-form solutions

Dependencies: core/, math/, domain/
Purpose: Physical models, equations, material behavior
```

**What Goes Here**:
- ✅ Wave equations (acoustic, elastic)
- ✅ Cavitation models (Rayleigh-Plesset, Keller-Miksis)
- ✅ Material constitutive relations
- ✅ Physical constants usage

**What Doesn't**:
- ❌ Numerical discretization (goes in solver/)
- ❌ Time integration (goes in solver/)
- ❌ Signal processing (goes in analysis/)

**Key Principle**: Physics = "WHY" (governing equations), not "HOW" (numerical methods)

---

### Layer 5: Solver (Numerical Methods)

```
solver/
├── forward/                # Forward problem solvers
│   ├── fdtd/               # Finite Difference Time Domain
│   ├── pstd/               # Pseudospectral Time Domain
│   │   └── dg/             # Discontinuous Galerkin
│   ├── hybrid/             # Hybrid methods
│   ├── elastic/            # Elastic wave solvers
│   └── acoustic/           # Acoustic wave solvers
├── inverse/                # Inverse problems
│   ├── reconstruction/     # Image reconstruction
│   │   ├── photoacoustic/  # Photoacoustic reconstruction
│   │   └── seismic/        # Seismic inversion
│   └── time_reversal/      # Time reversal methods
├── integration/            # Time integration schemes
│   └── time_integration/   # RK, IMEX, multi-rate
├── plugin/                 # Plugin architecture
│   ├── execution.rs
│   └── manager.rs
└── multiphysics/           # Coupled physics solvers

Dependencies: core/, math/, domain/, physics/
Purpose: Discretization, time stepping, numerical solvers
```

**What Goes Here**:
- ✅ FDTD, PSTD implementations
- ✅ Time integration schemes (RK4, IMEX)
- ✅ Grid-to-grid interpolation
- ✅ Solver orchestration

**What Doesn't**:
- ❌ Physics models (goes in physics/)
- ❌ Validation tests (goes in analysis/validation/)
- ❌ Signal processing (goes in analysis/)

**Moved Out**:
- `solver/analytical/` → `physics/analytical/solvers/`
- `solver/validation/` → `analysis/validation/solvers/`

---

### Layer 6: Analysis (Post-Processing)

```
analysis/
├── signal_processing/      # Signal analysis
│   ├── beamforming/        # ✅ CANONICAL LOCATION
│   │   ├── adaptive/       # MVDR, Capon
│   │   ├── narrowband/     # MUSIC, ESPRIT
│   │   ├── time_domain/    # DAS, DMAS
│   │   ├── neural/         # Neural beamforming
│   │   └── utils/          # Delay calculation, covariance
│   ├── localization/       # Source localization
│   └── pam/                # Passive acoustic mapping
├── validation/             # ALL validation (consolidated)
│   ├── solvers/            # Numerical method validation
│   ├── clinical/           # Clinical validation
│   └── theorem_validation.rs
├── visualization/          # Rendering, plotting
│   ├── renderer/           # GPU rendering
│   └── data_pipeline/      # Data processing for viz
└── performance/            # Profiling, optimization
    ├── profiling/          # Performance profiling
    └── optimization/       # Optimization strategies

Dependencies: core/, math/, domain/, physics/, solver/
Purpose: Post-simulation processing, analysis, visualization
```

**What Goes Here**:
- ✅ Beamforming algorithms (ALL types)
- ✅ Signal analysis, filtering
- ✅ Validation suites
- ✅ Visualization pipelines

**What Doesn't**:
- ❌ Domain entities (goes in domain/)
- ❌ Numerical solvers (goes in solver/)

**Key Migration**:
- ❌ `domain/sensor/beamforming/` → ✅ `analysis/signal_processing/beamforming/`

---

### Layer 7: Clinical (Application)

```
clinical/
├── imaging/                # Clinical imaging workflows
│   └── workflows.rs        # Ultrasound imaging protocols
└── therapy/                # Clinical therapy workflows
    ├── cavitation/         # Cavitation-enhanced therapy
    ├── lithotripsy/        # Lithotripsy application
    │   ├── bioeffects.rs   # Safety assessment
    │   ├── cavitation_cloud.rs
    │   ├── shock_wave.rs
    │   └── stone_fracture.rs
    └── modalities/         # Therapy modalities

Dependencies: ALL layers
Purpose: End-to-end clinical workflows, user-facing APIs
```

**What Goes Here**:
- ✅ Complete clinical workflows
- ✅ Treatment planning
- ✅ Safety monitoring
- ✅ Protocol management

**What Doesn't**:
- ❌ Low-level physics (goes in physics/)
- ❌ Numerical methods (goes in solver/)

---

## Cross-Cutting Concerns

### Infrastructure (infra/)

```
infra/
├── api/                    # REST API (if feature enabled)
├── io/                     # File I/O, serialization
├── cloud/                  # Cloud deployment
└── runtime/                # Runtime infrastructure

Purpose: External integrations, I/O, deployment
Dependencies: Depends on layer as needed
```

### GPU Acceleration (gpu/)

```
gpu/
├── memory/                 # GPU memory management
└── shaders/                # WGSL shaders

Purpose: GPU compute acceleration
Dependencies: Depends on layer as needed
```

### Testing (tests/)

```
tests/
├── support/                # Test utilities, fixtures
│   └── fixtures.rs         # Test data generators
├── integration/            # Integration tests
└── validation/             # Physics validation tests

Purpose: Testing infrastructure
Dependencies: ALL layers
```

---

## Import Guidelines

### ✅ CORRECT Import Patterns

```rust
// In core/ - NO upstream imports
use std::collections::HashMap;

// In math/ - core only
use crate::core::error::KwaversResult;
use crate::core::constants::*;

// In domain/ - core, math
use crate::core::error::KwaversResult;
use crate::math::fft::FFTProcessor;

// In physics/ - core, math, domain
use crate::core::constants::*;
use crate::domain::grid::Grid;
use crate::math::geometry::Vector3;

// In solver/ - core, math, domain, physics
use crate::physics::acoustics::acoustic_wave::WaveEquation;
use crate::domain::medium::Medium;

// In analysis/ - ALL lower layers
use crate::solver::forward::fdtd::FdtdSolver;
use crate::physics::acoustics::*;

// In clinical/ - ALL layers
use crate::analysis::signal_processing::beamforming::*;
use crate::solver::forward::pstd::PSTDSolver;
```

### ❌ FORBIDDEN Import Patterns

```rust
// In core/ - NO UPSTREAM IMPORTS!
use crate::physics::constants::*;  // ❌ WRONG - core cannot depend on physics
use crate::math::fft::*;           // ❌ WRONG - core cannot depend on math

// In math/ - NO PHYSICS/DOMAIN
use crate::physics::bubble_dynamics::*;  // ❌ WRONG - math should be generic
use crate::domain::grid::Grid;           // ❌ WRONG - math cannot depend on domain

// In domain/ - NO PHYSICS/SOLVER/ANALYSIS
use crate::physics::acoustics::*;        // ❌ WRONG - domain is infrastructure only
use crate::analysis::signal_processing::*; // ❌ WRONG - domain cannot do analysis

// In physics/ - NO SOLVER/ANALYSIS
use crate::solver::fdtd::*;              // ❌ WRONG - physics independent of numerics
use crate::analysis::validation::*;      // ❌ WRONG - physics cannot do validation

// ANYWHERE - NO CIRCULAR DEPENDENCIES
use crate::foo::bar;  // if bar also imports from crate::foo - ❌ CIRCULAR!
```

---

## Quick Lookup: "Where Does X Go?"

| Functionality | Correct Location | Why |
|---------------|------------------|-----|
| **Constants** | `core/constants/` | Single source of truth |
| **FFT** | `math/fft/` | Mathematical operation |
| **Grid** | `domain/grid/` | Domain entity |
| **Medium properties** | `domain/medium/` | Domain data structure |
| **Wave equation** | `physics/acoustics/` | Physical model |
| **FDTD solver** | `solver/forward/fdtd/` | Numerical method |
| **Beamforming** | `analysis/signal_processing/beamforming/` | Signal processing |
| **Validation** | `analysis/validation/` | Post-processing |
| **Clinical workflow** | `clinical/` | Application layer |
| **Transducer geometry** | `domain/source/` | Domain infrastructure |
| **Transducer physics** | `physics/acoustics/transducer/` | Physical calculations |
| **Time integration** | `solver/integration/` | Numerical method |
| **Visualization** | `analysis/visualization/` | Post-processing |
| **Test utilities** | `tests/support/` | Testing infrastructure |
| **Error types** | `core/error/` | Foundation |
| **ML infrastructure** | `math/ml/` | Mathematical operation |
| **Cavitation PINN** | `physics/acoustics/mechanics/cavitation/pinn.rs` | Physics-specific ML |

---

## Decision Tree: Module Placement

```
START: Where should this code go?
│
├─ Is it a universal constant or error type?
│  └─ YES → core/
│
├─ Is it a mathematical operation (FFT, linear algebra)?
│  └─ YES → math/
│
├─ Is it a domain entity (Grid, Medium, Sensor)?
│  └─ YES → domain/
│
├─ Is it a physical model or equation?
│  └─ YES → physics/
│
├─ Is it a numerical solver or time integrator?
│  └─ YES → solver/
│
├─ Is it signal processing, validation, or visualization?
│  └─ YES → analysis/
│
├─ Is it a clinical workflow or application?
│  └─ YES → clinical/
│
└─ Is it I/O, API, or infrastructure?
   └─ YES → infra/
```

---

## Common Mistakes & Fixes

### Mistake 1: Beamforming in Domain Layer

```rust
// ❌ WRONG:
use crate::domain::sensor::beamforming::DelayAndSum;

// ✅ CORRECT:
use crate::analysis::signal_processing::beamforming::time_domain::DelayAndSum;
```

**Reason**: Beamforming is signal processing (analysis), not domain infrastructure.

### Mistake 2: Constants in Physics Layer

```rust
// ❌ WRONG:
use crate::physics::constants::SOUND_SPEED_WATER;

// ✅ CORRECT:
use crate::core::constants::SOUND_SPEED_WATER;
```

**Reason**: Constants belong in core layer (single source of truth).

### Mistake 3: FFT in Core Utils

```rust
// ❌ WRONG:
use crate::core::utils::fft_3d_array;

// ✅ CORRECT:
use crate::math::fft::fft_3d_array;
```

**Reason**: Core cannot depend on math layer.

### Mistake 4: Test Helpers in Core

```rust
// ❌ WRONG:
use crate::core::utils::test_helpers::create_test_grid;

// ✅ CORRECT:
use crate::tests::support::fixtures::create_test_grid;
```

**Reason**: Test utilities belong in tests/, not core.

---

## Enforcement

### CI/CD Checks

```bash
# Check for layer violations
cargo deny check
cargo clippy -- -D warnings

# Custom checks (add to CI)
./scripts/check_architecture.sh  # Verify no upward dependencies
./scripts/check_nesting.sh       # Verify max depth ≤6
./scripts/check_file_sizes.sh    # Verify files <500 lines
```

### Pre-commit Hooks

```bash
# .git/hooks/pre-commit
#!/bin/bash
cargo clippy -- -D warnings || exit 1
cargo test --lib || exit 1
./scripts/check_architecture.sh || exit 1
```

---

## Migration Checklist

When moving code between layers:

1. [ ] Update module declaration in source `mod.rs`
2. [ ] Update module declaration in destination `mod.rs`
3. [ ] Move file to new location
4. [ ] Update all imports across codebase
5. [ ] Add deprecation notice in old location (if public API)
6. [ ] Update documentation (rustdoc, README)
7. [ ] Run full test suite
8. [ ] Update this architecture map if needed
9. [ ] Create ADR documenting the change
10. [ ] Remove deprecated code after grace period

---

## References

- **Detailed Audit**: `DEEP_VERTICAL_HIERARCHY_REFACTORING_AUDIT.md`
- **Executive Summary**: `AUDIT_EXECUTIVE_SUMMARY.md`
- **Immediate Fixes**: `IMMEDIATE_FIXES_CHECKLIST.md`
- **ADRs**: `docs/adr.md`
- **README**: `README.md` (architecture section)

---

**Last Updated**: 2024-01-09  
**Maintained By**: Architecture team  
**Review Frequency**: After each major refactor

---

*Keep this map updated as the architecture evolves.*