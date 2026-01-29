# Kwavers Architecture

## Overview

Kwavers is a comprehensive ultrasound and optics simulation library built with a strict **8-layer clean architecture** that enforces separation of concerns and prevents circular dependencies.

## Table of Contents

- [Architectural Principles](#architectural-principles)
- [Layer Structure](#layer-structure)
- [Dependency Rules](#dependency-rules)
- [Module Organization](#module-organization)
- [Design Patterns](#design-patterns)
- [Quality Gates](#quality-gates)
- [Development Guidelines](#development-guidelines)

---

## Architectural Principles

### 1. Clean Architecture with Unidirectional Dependencies

```
┌─────────────────────────────────────────────┐
│          Layer 8: Infrastructure            │  ← API, Cloud, GPU, I/O
├─────────────────────────────────────────────┤
│          Layer 7: Clinical                  │  ← Imaging, Therapy, Safety
├─────────────────────────────────────────────┤
│          Layer 6: Analysis                  │  ← Signal Processing, ML
├─────────────────────────────────────────────┤
│          Layer 5: Simulation                │  ← Orchestration, Config
├─────────────────────────────────────────────┤
│          Layer 4: Solver                    │  ← FDTD, PSTD, PINN
├─────────────────────────────────────────────┤
│          Layer 3: Domain                    │  ← Grid, Medium, Sensor
├─────────────────────────────────────────────┤
│          Layer 2: Physics                   │  ← Acoustics, Thermal, Optics
├─────────────────────────────────────────────┤
│          Layer 1: Math                      │  ← FFT, Linear Algebra
├─────────────────────────────────────────────┤
│          Layer 0: Core                      │  ← Error, Constants, Time
└─────────────────────────────────────────────┘
```

**Key Rule:** Higher layers may depend on lower layers, but **never the reverse**.

### 2. Single Source of Truth (SSOT)

- Every algorithm has **one canonical implementation**
- Constants and types defined once at the appropriate layer
- Re-exports used for backward compatibility without duplication

### 3. Domain-Driven Design (DDD)

- Modules represent **bounded contexts** with clear responsibilities
- Strong module boundaries prevent cross-contamination
- Ubiquitous language reflected in naming

### 4. Feature-Gated Optional Functionality

```toml
[features]
minimal = []                    # Core functionality only
gpu = ["wgpu", "bytemuck"]      # GPU acceleration
plotting = ["plotly"]           # Visualization
pinn = ["burn"]                 # Machine learning solvers
api = ["axum", "tower"]         # REST API
cloud = ["reqwest"]             # Cloud deployment
parallel = ["ndarray/rayon"]    # Parallel computing
full = ["gpu", "plotting", "pinn", "api", "cloud", "parallel"]
```

---

## Layer Structure

### Layer 0: Core (Foundation)

**Path:** `src/core/`

**Purpose:** Foundational types and utilities used throughout the library.

**Modules:**
- `error/` - Error types (`KwaversError`, `KwaversResult`)
- `constants/` - Physical and numerical constants
- `log/` - Logging infrastructure
- `time/` - Time management utilities
- `utils/` - General-purpose utilities
- `arena/` - Memory arena allocators

**Dependencies:** None (foundation layer)

**Key Exports:**
```rust
pub use crate::core::error::{KwaversError, KwaversResult};
pub use crate::core::constants::SOUND_SPEED_WATER;
pub use crate::core::time::Time;
```

---

### Layer 1: Math (Primitives)

**Path:** `src/math/`

**Purpose:** Pure mathematical algorithms and numerical methods.

**Modules:**
- `fft/` - Fast Fourier Transform (1D, 2D, 3D, k-space)
- `geometry/` - Geometric primitives and spatial operations
- `linear_algebra/` - Matrix operations, SVD, eigendecomposition
- `numerics/` - Numerical integration and optimization
- `simd/` - SIMD-accelerated operations
- `simd_safe/` - Architecture-safe SIMD fallbacks

**Dependencies:** Core (Layer 0)

**Key Exports:**
```rust
pub use crate::math::fft::{Fft1d, Fft2d, KSpaceCalculator};
pub use crate::math::linear_algebra::LinearAlgebra;
pub use crate::math::geometry::GeometricOperations;
```

---

### Layer 2: Physics (Domain Logic)

**Path:** `src/physics/`

**Purpose:** Physical models and wave equations.

**Modules:**
- `acoustics/` - Wave propagation, bubble dynamics, cavitation
- `thermal/` - Heat diffusion, bioheat equation
- `optics/` - Optical physics, sonoluminescence
- `chemistry/` - Reaction kinetics, free radical production
- `electromagnetic/` - EM wave interactions
- `foundations/` - Wave equation traits, boundary conditions

**Dependencies:** Core (Layer 0), Math (Layer 1)

**Key Exports:**
```rust
pub use crate::physics::acoustics::WaveEquation;
pub use crate::physics::acoustics::bubble_dynamics::KellerMiksis;
pub use crate::physics::thermal::BioheatEquation;
```

---

### Layer 3: Domain (Business Logic)

**Path:** `src/domain/`

**Purpose:** Simulation entities and specifications.

**Modules:**
- `grid/` - Computational grids (rectangular, spherical)
- `medium/` - Material properties (homogeneous, heterogeneous)
- `geometry/` - Domain geometry definitions
- `sensor/` - Sensor arrays, recording, hardware interface
- `source/` - Acoustic sources, time-varying signals
- `boundary/` - PML, CPML, boundary conditions
- `signal/` - Signal primitives, filters
- `field/` - Unified field indexing and access
- `plugin/` - Domain plugin system

**Dependencies:** Core, Math, Physics

**Key Exports:**
```rust
pub use crate::domain::grid::Grid;
pub use crate::domain::medium::Medium;
pub use crate::domain::sensor::GridSensorSet;
pub use crate::domain::source::TimeVaryingSource;
```

---

### Layer 4: Solver (Numerical Methods)

**Path:** `src/solver/`

**Purpose:** Numerical solution algorithms.

**Modules:**
- `forward/fdtd/` - Finite-Difference Time-Domain
- `forward/pstd/` - Pseudo-Spectral Time-Domain
- `forward/hybrid/` - Hybrid methods with domain decomposition
- `forward/nonlinear/` - Kuznetsov, Westervelt, KZK equations
- `forward/analytical/` - Analytical solutions
- `forward/plugin_based/` - Plugin-based architecture
- `inverse/pinn/` - Physics-Informed Neural Networks
- `inverse/time_reversal/` - Time-reversal focusing
- `inverse/reconstruction/` - PAT, seismic reconstruction
- `plugin/` - Plugin interface and execution
- `integration/` - Time integration methods (RK4, IMEX)
- `multiphysics/` - Multi-physics coupling
- `utilities/` - Grid utilities, AMR, validation

**Dependencies:** Core, Math, Physics, Domain

**Key Exports:**
```rust
pub use crate::solver::forward::fdtd::FDTDSolver;
pub use crate::solver::forward::pstd::PSTDSolver;
pub use crate::solver::inverse::pinn::PINNSolver;
```

---

### Layer 5: Simulation (Orchestration)

**Path:** `src/simulation/`

**Purpose:** Configuration and simulation coordination.

**Modules:**
- `configuration/` - Simulation parameters
- `setup/` - Grid and medium setup
- `factory/` - Physics factory patterns
- `parameters/` - Parameter definitions

**Dependencies:** Core, Math, Physics, Domain, Solver

**Key Exports:**
```rust
pub use crate::simulation::configuration::SimulationConfig;
pub use crate::simulation::setup::SimulationSetup;
```

---

### Layer 6: Analysis (Post-Processing)

**Path:** `src/analysis/`

**Purpose:** Signal processing and machine learning analysis.

**Modules:**
- `signal_processing/beamforming/` - DAS, MVDR, MUSIC, adaptive, neural, 3D
- `signal_processing/filtering/` - IIR, FIR, frequency-domain
- `signal_processing/clutter_filter/` - Polynomial, SVD, adaptive
- `signal_processing/localization/` - Beamforming search, multilateration, MUSIC
- `signal_processing/pam/` - Passive acoustic mapping
- `signal_processing/modulation/` - Frequency/amplitude modulation analysis
- `ml/` - Machine learning models and uncertainty quantification
- `performance/` - Benchmarks, profiling, optimization
- `visualization/` - 3D rendering, metrics
- `plotting/` - Visualization tools (feature-gated)
- `testing/` - Property-based testing support
- `validation/` - Physics validation against literature

**Dependencies:** Core, Math, Physics, Domain, Solver, Simulation

**Key Exports:**
```rust
pub use crate::analysis::signal_processing::beamforming::adaptive::MinimumVariance;
pub use crate::analysis::signal_processing::beamforming::time_domain::delay_and_sum;
pub use crate::analysis::signal_processing::clutter_filter::AdaptiveSVDFilter;
```

---

### Layer 7: Clinical (Applications)

**Path:** `src/clinical/`

**Purpose:** Clinical workflows and decision support.

**Modules:**
- `imaging/` - B-mode, Doppler, elastography, functional ultrasound
- `therapy/` - HIFU, ablation, safety planning
- `safety/` - Safety compliance and monitoring
- `workflows/` - Clinical decision support and orchestration

**Dependencies:** All lower layers

**Key Exports:**
```rust
pub use crate::clinical::imaging::workflows::ImagingWorkflow;
pub use crate::clinical::therapy::HIFUPlanner;
pub use crate::clinical::safety::SafetyMonitor;
```

---

### Layer 8: Infrastructure (Cross-Cutting)

**Path:** `src/infra/`

**Purpose:** External interfaces and services.

**Modules:**
- `api/` - REST API (Axum-based, feature-gated)
- `cloud/` - Cloud deployment (AWS, feature-gated)
- `io/` - I/O operations (DICOM, NIfTI, HDF5)
- `runtime/` - Async runtime, distributed computing
- `gpu/` - GPU acceleration (WGPU, experimental, feature-gated)

**Dependencies:** Independent (cross-cutting concerns)

**Key Exports:**
```rust
#[cfg(feature = "api")]
pub use crate::infra::api::KwaversAPI;

#[cfg(feature = "gpu")]
pub use crate::infra::gpu::GPUBackend;
```

---

## Dependency Rules

### ✅ Allowed Dependencies

Higher layers may depend on lower layers:

```rust
// ✅ Solver (Layer 4) depending on Domain (Layer 3)
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;

// ✅ Analysis (Layer 6) depending on Solver (Layer 4)
use crate::solver::forward::fdtd::FDTDSolver;

// ✅ Clinical (Layer 7) depending on Analysis (Layer 6)
use crate::analysis::signal_processing::beamforming::adaptive::MinimumVariance;
```

### ❌ Forbidden Dependencies

Lower layers **MUST NOT** depend on higher layers:

```rust
// ❌ Domain (Layer 3) depending on Solver (Layer 4)
use crate::solver::forward::fdtd::FDTDSolver;  // VIOLATION

// ❌ Physics (Layer 2) depending on Domain (Layer 3)
use crate::domain::grid::Grid;  // VIOLATION

// ❌ Solver (Layer 4) depending on Analysis (Layer 6)
use crate::analysis::signal_processing::beamforming::adaptive::MinimumVariance;  // VIOLATION
```

### Enforcement Mechanism

Violations are detected automatically via:

1. **CI/CD Pipeline:** `.github/workflows/architecture-validation.yml`
2. **Validation Script:** `scripts/validate_architecture.sh`
3. **Pre-commit Hooks:** (recommended for local development)

---

## Module Organization

### Naming Conventions

- **snake_case** for all module names
- **PascalCase** for types and traits
- **SCREAMING_SNAKE_CASE** for constants
- **lowercase** for functions and methods

### File Structure

```
src/
├── core/                    # Layer 0: Foundation
│   ├── error/
│   │   ├── mod.rs
│   │   └── types.rs
│   ├── constants/
│   │   ├── mod.rs
│   │   └── physical.rs
│   └── time/
│       └── mod.rs
├── math/                    # Layer 1: Primitives
│   ├── fft/
│   │   ├── mod.rs
│   │   ├── fft_1d.rs
│   │   ├── fft_2d.rs
│   │   └── fft_3d.rs
│   └── linear_algebra/
│       ├── mod.rs
│       └── svd.rs
├── physics/                 # Layer 2: Domain Logic
│   ├── acoustics/
│   │   ├── mod.rs
│   │   ├── wave_propagation.rs
│   │   └── bubble_dynamics/
│   │       ├── mod.rs
│   │       └── keller_miksis.rs
│   └── thermal/
│       ├── mod.rs
│       └── bioheat.rs
├── domain/                  # Layer 3: Business Logic
│   ├── grid/
│   │   ├── mod.rs
│   │   └── structure.rs
│   ├── medium/
│   │   ├── mod.rs
│   │   ├── homogeneous.rs
│   │   └── heterogeneous.rs
│   └── sensor/
│       ├── mod.rs
│       └── beamforming/
│           └── sensor_beamformer.rs
├── solver/                  # Layer 4: Numerical Methods
│   ├── forward/
│   │   ├── fdtd/
│   │   │   ├── mod.rs
│   │   │   └── solver.rs
│   │   └── pstd/
│   │       ├── mod.rs
│   │       └── solver.rs
│   └── inverse/
│       └── pinn/
│           ├── mod.rs
│           └── burn_adapter.rs
├── simulation/              # Layer 5: Orchestration
│   ├── configuration/
│   │   └── mod.rs
│   └── setup/
│       └── mod.rs
├── analysis/                # Layer 6: Post-Processing
│   ├── signal_processing/
│   │   ├── beamforming/
│   │   │   ├── mod.rs
│   │   │   ├── adaptive/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── mvdr.rs
│   │   │   │   └── music.rs
│   │   │   └── time_domain/
│   │   │       ├── mod.rs
│   │   │       └── das.rs
│   │   └── clutter_filter/
│   │       ├── mod.rs
│   │       └── svd.rs
│   └── ml/
│       └── mod.rs
├── clinical/                # Layer 7: Applications
│   ├── imaging/
│   │   └── workflows/
│   │       ├── mod.rs
│   │       └── neural.rs
│   └── therapy/
│       └── hifu.rs
└── infra/                   # Layer 8: Infrastructure
    ├── api/
    │   └── mod.rs
    └── gpu/
        └── mod.rs
```

### Module Size Guidelines

- **Core modules:** ≤ 500 lines
- **Complex algorithms:** ≤ 800 lines (justified)
- **Files > 800 lines:** Split into focused submodules
- **Deep nesting:** Maximum 5-6 levels preferred

---

## Design Patterns

### 1. Factory Pattern

Used for creating physics solvers and configurations:

```rust
pub struct PhysicsFactory;

impl PhysicsFactory {
    pub fn create_fdtd_solver(config: &SimulationConfig) -> KwaversResult<FDTDSolver> {
        // Auto-calculate CFL condition, grid spacing
        let solver = FDTDSolver::new(config)?;
        Ok(solver)
    }
}
```

### 2. Builder Pattern

Used for complex configuration:

```rust
let config = SimulationConfig::builder()
    .grid_size(100, 100, 100)
    .time_step(1e-6)
    .medium(HomogeneousMedium::water())
    .build()?;
```

### 3. Strategy Pattern

Used for interchangeable algorithms:

```rust
pub trait BeamformingAlgorithm {
    fn process(&self, data: &Array2<f64>) -> KwaversResult<Array1<f64>>;
}

pub struct DelayAndSum;
impl BeamformingAlgorithm for DelayAndSum { /* ... */ }

pub struct MinimumVariance;
impl BeamformingAlgorithm for MinimumVariance { /* ... */ }
```

### 4. Plugin Pattern

Used for extensible solver architecture:

```rust
pub trait SolverPlugin {
    fn name(&self) -> &str;
    fn initialize(&mut self, domain: &Domain) -> KwaversResult<()>;
    fn step(&mut self, dt: f64) -> KwaversResult<()>;
}
```

### 5. Adapter Pattern

Used for integrating external libraries (e.g., Burn for PINN):

```rust
pub struct BurnPinnBeamformingAdapter {
    model: BurnModel,
}

impl BeamformingAlgorithm for BurnPinnBeamformingAdapter {
    fn process(&self, data: &Array2<f64>) -> KwaversResult<Array1<f64>> {
        // Adapt ndarray to Burn tensors
        let tensor = self.convert_to_burn(data)?;
        let output = self.model.forward(tensor);
        Ok(self.convert_from_burn(output))
    }
}
```

---

## Quality Gates

### Continuous Integration Checks

The following checks run on every commit:

1. **Layer Boundary Enforcement**
   - Zero `domain → solver` violations
   - Zero `physics → domain` violations
   - Zero `solver → analysis` violations
   - `domain → analysis` violations ≤ 10 (documented technical debt)

2. **Build Quality**
   - Zero build errors
   - Zero build warnings
   - Successful builds with all feature combinations

3. **Test Suite**
   - 100% test pass rate
   - > 95% coverage for critical paths

4. **Code Quality**
   - `cargo fmt --check` passes
   - `cargo clippy -- -D warnings` passes
   - Documentation builds without warnings

5. **Module Organization**
   - Files > 1000 LOC ≤ 10
   - Deep nesting (>6 levels) ≤ 50 files
   - Dead code allowances ≤ 100

### Local Validation

Run before committing:

```bash
# Full architecture validation
./scripts/validate_architecture.sh

# Quick checks
cargo fmt --check
cargo clippy -- -D warnings
cargo test --lib
```

---

## Development Guidelines

### Adding New Features

1. **Identify Correct Layer**
   - Is it a primitive algorithm? → Math layer
   - Is it a physics model? → Physics layer
   - Is it domain configuration? → Domain layer
   - Is it a solver? → Solver layer
   - Is it post-processing? → Analysis layer
   - Is it clinical application? → Clinical layer

2. **Check for Existing Implementation**
   - Search for similar functionality
   - Avoid duplication (SSOT principle)
   - Reuse existing components

3. **Follow Module Organization**
   - Place in appropriate submodule
   - Use consistent naming conventions
   - Document module purpose

4. **Write Tests**
   - Unit tests for algorithms
   - Integration tests for workflows
   - Property-based tests for mathematical correctness

5. **Document Thoroughly**
   - Module-level documentation (`//!`)
   - Function-level documentation (`///`)
   - Examples in documentation
   - Mathematical foundations where applicable

### Refactoring Guidelines

1. **Preserve Backward Compatibility**
   - Use re-exports for old paths
   - Add deprecation warnings with migration guides
   - Document breaking changes in CHANGELOG

2. **Maintain SSOT**
   - Move, don't duplicate
   - Update all references
   - Remove deprecated code after grace period

3. **Validate Architecture**
   - Run `./scripts/validate_architecture.sh`
   - Check CI pipeline status
   - Update documentation

### Code Review Checklist

- [ ] Correct layer placement
- [ ] No circular dependencies
- [ ] SSOT maintained
- [ ] Tests included
- [ ] Documentation complete
- [ ] No build warnings
- [ ] Clippy passes
- [ ] Formatting applied
- [ ] Backward compatibility preserved
- [ ] Migration guide provided (if breaking)

---

## References

### Architectural Patterns

- **Clean Architecture:** Robert C. Martin (Uncle Bob)
- **Domain-Driven Design:** Eric Evans
- **Hexagonal Architecture:** Alistair Cockburn

### Ultrasound Simulation Architectures

- **k-Wave:** Backend abstraction, k-space methods
- **jWave:** Functional composition, auto-differentiation
- **Fullwave25:** Clinical workflows, high-order FDTD
- **OptimUS:** Multi-domain coupling, BEM optimization
- **SimSonic:** Elastodynamics, heterogeneous media

### Documentation

- [Architecture Analysis Report](./ULTRASOUND_SIMULATION_ARCHITECTURE_ANALYSIS.md)
- [Implementation Quick Reference](./docs/IMPLEMENTATION_QUICK_REFERENCE.md)
- [Migration Guides](./docs/migration/)
- [API Documentation](https://docs.rs/kwavers)

---

## Version History

- **v3.1.0** (2026-01-28): Comprehensive architecture audit and validation
- **v3.0.0** (2026-01-25): Fixed layer violations, established clean architecture
- **v2.x**: Legacy beamforming migration, plugin system implementation
- **v1.x**: Initial release with basic solver infrastructure

---

**Last Updated:** January 28, 2026  
**Maintained By:** Kwavers Development Team  
**License:** See LICENSE file
