# Kwavers: Acoustic Wave Simulation Library

[![Rust](https://img.shields.io/badge/rust-1.89%2B-blue.svg)](https://www.rust-lang.org)
[![Build](https://img.shields.io/badge/build-passing-green.svg)](https://github.com/kwavers/kwavers)
[![Tests](https://img.shields.io/badge/tests-16%2F16-green.svg)](./tests)
[![Grade](https://img.shields.io/badge/grade-B-yellow.svg)](./PRD.md)

## Comprehensive Acoustic Wave Simulation Library

A feature-complete acoustic wave simulation library implementing FDTD and PSTD solvers with extensive physics models. The codebase follows pragmatic engineering principles with a focus on functionality and correctness.

### Current Status (v2.15.0)
- **Build**: Clean compilation with managed warnings
- **Tests**: All 16 test suites passing
- **Examples**: 7 working examples
- **Physics**: Corrected CFL stability (0.5 for 3D FDTD)
- **Architecture**: Functional with ongoing improvements

## Quick Start

```toml
[dependencies]
kwavers = "2.15.0"
```

```rust
use kwavers::{Grid, HomogeneousMedium, FdtdSolver, FdtdConfig};

// Create simulation grid
let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);

// Define medium (water)
let medium = HomogeneousMedium::water(&grid);

// Configure and create FDTD solver
let config = FdtdConfig::default();
let solver = FdtdSolver::new(config, &grid)?;
```

## Core Features

### Numerical Solvers
- **FDTD** - Complete Yee scheme implementation with proper CFL stability
- **PSTD** - FFT-based pseudo-spectral solver
- **Plugin Architecture** - Extensible computation framework

### Physics Models
- Acoustic wave propagation with validated CFL (0.5 for 3D)
- PML/CPML boundary conditions
- Homogeneous and heterogeneous media support
- Chemical kinetics modeling
- Bubble dynamics and cavitation
- Thermal coupling

### Engineering Decisions
- **Comprehensive API** - Complete interface for extensibility
- **Pragmatic Warnings** - Managed for API completeness
- **Module Organization** - Ongoing refactoring for maintainability
- **Future Features** - Documented placeholders for planned enhancements

## Project Structure

```
src/
├── solver/           # Numerical solvers
│   ├── fdtd/        # FDTD implementation
│   ├── pstd/        # PSTD with FFT
│   └── ...
├── physics/          # Physics models
│   ├── chemistry/   # Chemical kinetics
│   ├── mechanics/   # Wave mechanics
│   └── ...
├── boundary/        # Boundary conditions
├── medium/          # Material properties
└── ...             # 369 source files
```

## Building & Testing

```bash
# Build with optimizations
cargo build --release

# Run all tests
cargo test --release

# Run specific example
cargo run --release --example basic_simulation

# Generate documentation
cargo doc --no-deps --open
```

### Test Results
```
✅ Unit tests:        3/3
✅ Integration tests: 5/5
✅ Solver tests:      3/3
✅ Doc tests:         5/5
━━━━━━━━━━━━━━━━━━━━━━━━
Total: 16/16 passing
```

## Examples

Seven working examples demonstrating key features:

```bash
# Basic FDTD simulation
cargo run --release --example basic_simulation

# Plugin system demonstration
cargo run --release --example plugin_example

# Physics validation
cargo run --release --example physics_validation

# FDTD vs PSTD comparison
cargo run --release --example pstd_fdtd_comparison

# Tissue modeling
cargo run --release --example tissue_model_example

# Phased array beamforming
cargo run --release --example phased_array_beamforming

# Wave propagation
cargo run --release --example wave_simulation
```

## Technical Assessment

| Aspect | Status | Notes |
|--------|--------|-------|
| **Correctness** | ✅ Good | Physics validated, CFL fixed |
| **Safety** | ✅ Good | No unsafe in critical paths |
| **Performance** | ⚠️ Adequate | Not fully optimized |
| **Architecture** | ⚠️ Improving | Refactoring large modules |
| **Documentation** | ✅ Good | Comprehensive with examples |

## Known Limitations & Trade-offs

### Current State
- Some modules exceed 500 lines (refactoring ongoing)
- Performance optimization opportunities exist
- GPU support planned but not implemented

### Pragmatic Decisions
- Warning suppressions for comprehensive API
- Placeholders documented for future features
- Focus on correctness over premature optimization

## Physics Validation

### CFL Stability
- Maximum stable CFL for 3D FDTD: 1/√3 ≈ 0.577
- Implementation uses 0.5 for safety margin
- Validated against Taflove & Hagness (2005)

### Numerical Accuracy
- Phase velocity errors expected with FDTD
- Absorption models validated against Beer-Lambert law
- Suitable for acoustic simulations within documented limits

## Use Cases

### Suitable For
- Academic research simulations
- Ultrasound modeling
- Wave propagation studies
- Educational demonstrations
- Prototype development

### Requirements
- Validate numerical parameters for your specific use case
- Consider memory requirements for large grids
- Profile performance for time-critical applications

## Roadmap

### Near Term
- Complete module refactoring (< 500 lines)
- Performance profiling and optimization
- Expand test coverage

### Future
- GPU acceleration (CUDA/OpenCL)
- Distributed computing support
- Real-time visualization
- Machine learning integration

## Contributing

We welcome contributions in these areas:

1. **Module Refactoring** - Help split large files
2. **Performance** - Optimization and profiling
3. **Testing** - Expand test coverage
4. **Documentation** - Improve examples and guides

### Guidelines
- Follow Rust idioms and best practices
- Maintain comprehensive API surface
- Document design decisions
- Add tests for new features

## License

MIT - See [LICENSE](LICENSE)

---

**Version**: 2.15.0  
**Grade**: B (Functional, Improving)  
**Status**: All tests passing, ongoing improvements  
**Recommendation**: Suitable for research and development use
