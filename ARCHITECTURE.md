# Kwavers Architecture Documentation

## Overview

Kwavers is a comprehensive ultrasound and optics simulation library built in Rust with a clean, layered Domain-Driven Design (DDD) architecture. The codebase emphasizes **deep vertical hierarchical organization** with **strict separation of concerns** and **single source of truth (SSOT)** principles.

## Design Principles

1. **Deep Vertical Tree**: Clear hierarchical module structure with well-defined layer boundaries
2. **Separation of Concerns**: Each layer has specific responsibilities with minimal cross-contamination
3. **Single Source of Truth**: Canonical implementations with thin wrappers elsewhere
4. **No Circular Dependencies**: Strict unidirectional dependency flow from top to bottom
5. **Clean Builds**: Zero warnings, zero deprecated code, zero dead code in production

## 8-Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Layer 8: Infrastructure (Cross-Cutting)                │
│ - API (REST), Cloud (AWS), I/O (DICOM, NIfTI)         │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 7: Clinical (Applications)                       │
│ - Imaging Workflows, Therapy Planning, Safety          │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 6: Analysis (Post-Processing)                    │
│ - Beamforming, Signal Processing, ML, Visualization    │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 5: Simulation (Orchestration)                    │
│ - Configuration, Core Loop, Multi-Physics Coordination │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 4: Solvers (Numerical Methods)                   │
│ - Forward (FDTD, PSTD, Hybrid, Helmholtz)             │
│ - Inverse (PINN, Reconstruction, Time Reversal)        │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 3: Domain (Business Logic)                       │
│ - Boundary, Grid, Medium, Sensors, Sources             │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 2: Physics (Domain Logic)                        │
│ - Acoustics, Optics, Thermal, Electromagnetic          │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 1: Math (Primitives)                             │
│ - FFT, Linear Algebra, Geometry, Numerics              │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 0: Core (Foundation)                             │
│ - Constants, Errors, Logging, Time, Utils              │
└─────────────────────────────────────────────────────────┘
```

## Key Module Responsibilities

### Core (Layer 0)
- **Purpose**: Foundation types and utilities
- **Key Modules**: `constants`, `error`, `log`, `time`, `utils`
- **Dependencies**: None (leaf layer)
- **SSOT Examples**: Physical constants (`SOUND_SPEED_WATER`), error types

### Math (Layer 1)
- **Purpose**: Mathematical primitives
- **Key Modules**: `fft`, `linear_algebra`, `geometry`, `numerics`
- **Dependencies**: Core
- **SSOT Examples**: FFT algorithms, sparse matrix solvers, geometric transformations

### Physics (Layer 2)
- **Purpose**: Physical models and equations
- **Key Modules**: `acoustics`, `optics`, `thermal`, `electromagnetic`, `foundations`
- **Dependencies**: Core, Math
- **SSOT Examples**: Wave equations, bubble dynamics (Keller-Miksis), cavitation models

### Domain (Layer 3)
- **Purpose**: Simulation entities and configurations
- **Key Modules**: `boundary`, `grid`, `medium`, `sensor`, `source`, `field`
- **Dependencies**: Core, Math, Physics
- **SSOT Examples**: Grid operators, PML boundaries, sensor arrays

### Solvers (Layer 4)
- **Purpose**: Numerical solution methods
- **Key Modules**: `forward` (FDTD, PSTD, hybrid), `inverse` (PINN, reconstruction)
- **Dependencies**: Core, Math, Physics, Domain
- **SSOT Examples**: PSTD k-space operators, FDTD stencils, hybrid coupling

### Simulation (Layer 5)
- **Purpose**: Orchestration and coordination
- **Key Modules**: `configuration`, `core`, `factory`, `modalities`
- **Dependencies**: All lower layers
- **SSOT Examples**: Simulation loop, multi-physics scheduling

### Analysis (Layer 6)
- **Purpose**: Post-processing and signal analysis
- **Key Modules**: `signal_processing/beamforming`, `ml`, `performance`, `validation`
- **Dependencies**: All lower layers
- **SSOT Examples**: 
  - DAS beamforming: `analysis::signal_processing::beamforming::time_domain::das`
  - Steering vectors: `analysis::signal_processing::beamforming::utils::steering`
  - Covariance estimation: `analysis::signal_processing::beamforming::covariance`

### Clinical (Layer 7)
- **Purpose**: Clinical applications and workflows
- **Key Modules**: `imaging`, `therapy`, `safety`
- **Dependencies**: All lower layers
- **SSOT Examples**: Clinical decision support, IEC safety compliance

### Infrastructure (Layer 8)
- **Purpose**: Cross-cutting concerns
- **Key Modules**: `api`, `cloud`, `io`, `runtime`
- **Dependencies**: All layers
- **SSOT Examples**: DICOM I/O, AWS deployment, REST API

## Recent Architectural Improvements (2026-01)

### Fixed Issues
1. ✅ **Compilation Errors**: Fixed 6 critical build errors by adding proper re-exports
2. ✅ **Circular Dependencies**: Eliminated Physics → Domain dependency via constant relocation
3. ✅ **Module Path Issues**: Corrected `solver::hybrid` → `solver::forward::hybrid` paths
4. ✅ **Dead Code**: Removed deprecated files (`beamforming.rs`, `domain_time.rs`)
5. ✅ **Clean Comments**: Eliminated commented-out module declarations

### Established SSOT Patterns

#### Beamforming Algorithms
- **Canonical Location**: `src/analysis/signal_processing/beamforming/`
- **Time-Domain DAS**: `time_domain/das.rs` (SSOT)
- **Adaptive Methods**: `adaptive/` (MVDR, MUSIC, Eigenspace-MV)
- **3D Implementations**: Feature-gated GPU/CPU versions (NOT duplicates)
- **Steering Vectors**: `utils/steering.rs` (SSOT) with specialized wrappers for narrowband/3D

#### Delay Reference Policy
```rust
pub enum DelayReference {
    SensorIndex(usize),      // Fixed reference sensor (recommended default)
    EarliestArrival,         // Data-dependent, min delay
    LatestArrival,           // Data-dependent, max delay
}
```

## Dependency Rules

### Allowed Dependencies
- Any layer may depend on lower-numbered layers
- Feature-gated code may have conditional dependencies
- Test code may depend on production code

### Forbidden Patterns
- ❌ **Circular Dependencies**: Lower layers depending on higher layers
- ❌ **Cross-Contamination**: Domain logic in analysis layer, or vice versa
- ❌ **Duplicate Implementations**: Multiple canonical sources for same algorithm
- ❌ **Leaky Abstractions**: Implementation details escaping module boundaries

## Code Quality Standards

### Build Requirements
```bash
# Must pass without warnings or errors
cargo build --lib
cargo test --lib
cargo clippy -- -D warnings
```

### Testing Strategy
- Unit tests in same file as implementation (`#[cfg(test)] mod tests`)
- Integration tests in `tests/` directory
- Property-based tests for numerical algorithms
- Validation against reference implementations (k-wave, jwave)

### Documentation Standards
- Module-level doc comments (`//!`) explaining purpose and architecture
- Function-level doc comments (`///`) for all public APIs
- Mathematical foundations with equations and references
- Migration guides for architectural changes

## Feature Flags

### Available Features
- `gpu`: GPU acceleration via WGPU
- `advanced-visualization`: 3D rendering and volume visualization
- `structured-logging`: Tracing and telemetry
- `python-bindings`: PyO3 Python interoperability
- `cloud-deployment`: AWS/cloud infrastructure

### Feature Flag Patterns
```rust
#[cfg(feature = "gpu")]
pub fn process_gpu(...) { ... }

#[cfg(not(feature = "gpu"))]
pub fn process_cpu(...) { ... }  // Fallback implementation
```

## Performance Considerations

### Optimization Hierarchy
1. **Algorithm Selection**: Choose right algorithm (PSTD vs FDTD)
2. **Memory Layout**: Cache-friendly data structures
3. **SIMD**: Vectorized operations where applicable
4. **Parallelism**: Rayon for CPU, WGPU for GPU
5. **JIT/Compilation**: Inline hints, LTO, PGO

### Benchmarking
```bash
cargo bench --features gpu
```

## Comparison with Reference Libraries

### vs k-Wave (MATLAB)
- ✅ **Rust Safety**: Memory-safe, thread-safe
- ✅ **Performance**: Native compilation, SIMD
- ✅ **Modularity**: Clean layer separation
- 🔄 **Feature Parity**: Implementing k-space pseudospectral methods

### vs jWave (JAX/Python)
- ✅ **Differentiability**: PINN solvers with autodiff
- ✅ **GPU Support**: WGPU cross-platform
- ✅ **Type Safety**: Compile-time guarantees
- 🔄 **ML Integration**: Expanding Burn integration

### vs mSOUND (MATLAB)
- ✅ **Multi-Physics**: Coupled acoustic-thermal-optical
- ✅ **Clinical Workflows**: Imaging and therapy pipelines
- ✅ **Scalability**: Cloud deployment support

## Future Directions

### Planned Enhancements
1. **K-Space Methods**: Full k-Wave pseudospectral parity
2. **JAX-like Differentiability**: Seamless gradient-based optimization
3. **Distributed Computing**: MPI/cluster support
4. **Real-Time Processing**: CUDA streams, async pipelines
5. **Clinical Validation**: FDA/IEC compliance tooling

### Research Integration
- Functional ultrasound (fUS) brain imaging (Nature neuroscience methods)
- Photoacoustic tomography with learned reconstruction
- Therapeutic ultrasound planning with PINN optimization
- Sonoluminescence simulation and detection

## References

### Core Papers
- Van Trees (2002): *Optimum Array Processing*
- Jensen (1996): *Field: A Program for Simulating Ultrasound Systems*
- Treeby & Cox (2010): *k-Wave: MATLAB toolbox for simulation and reconstruction*

### Architectural Patterns
- Evans (2003): *Domain-Driven Design*
- Martin (2017): *Clean Architecture*
- Nygard (2007): *Release It!*

## Maintenance

### Code Ownership
- **Core/Math**: Foundation team
- **Physics**: Acoustics research team
- **Solvers**: Numerical methods team
- **Clinical**: Applications team

### Review Process
1. All changes require clean build (`cargo build`)
2. New features require tests and documentation
3. Breaking changes need migration guide
4. Performance regressions trigger investigation

---

**Last Updated**: 2026-01-24  
**Version**: 3.0.0  
**Status**: Production-ready, actively maintained
