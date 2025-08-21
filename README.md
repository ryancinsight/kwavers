# Kwavers: Acoustic Wave Simulation Library

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)
[![Build](https://img.shields.io/badge/build-passing-green.svg)](https://github.com/kwavers/kwavers)
[![Status](https://img.shields.io/badge/status-alpha-yellow.svg)](./src)

## Project Status

| Component | Status | Details |
|-----------|--------|---------|
| **Library** | ✅ **BUILDS** | Compiles without errors |
| **Examples** | ✅ **WORKING** | Basic examples run successfully |
| **Tests** | ⚠️ **PARTIAL** | Some compilation issues remain |
| **Warnings** | ⚠️ **502** | Stable, not blocking functionality |
| **Architecture** | ✅ **SOLID** | Clean, modular, extensible |

## Overview

Kwavers is a high-performance acoustic wave simulation library in Rust, implementing state-of-the-art numerical methods with a focus on safety, performance, and extensibility.

### Key Features
- **Physics Models**: Linear/nonlinear acoustics, elastic waves, thermal effects
- **Numerical Methods**: FDTD, PSTD, Spectral-DG, AMR
- **Architecture**: Plugin-based, following SOLID/CUPID/GRASP principles
- **Performance**: Zero-cost abstractions, parallel processing ready
- **Safety**: Memory-safe, type-safe, thread-safe

## Quick Start

```bash
# Clone and build
git clone https://github.com/kwavers/kwavers
cd kwavers
cargo build --release

# Run working example
cargo run --example basic_simulation

# Output:
# Grid properties:
#   CFL timestep: 1.15e-7 s
#   Grid points: 262144
#   Memory estimate: 21.0 MB
```

## Working Examples

### ✅ Verified Working
- `basic_simulation` - Grid setup and basic wave propagation
- More examples being updated

### Example Code
```rust
use kwavers::{Grid, HomogeneousMedium, Time};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create computational grid
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
    
    // Define medium properties
    let medium = HomogeneousMedium::water(&grid);
    
    // Setup time stepping
    let dt = grid.cfl_timestep_default(1500.0);
    let time = Time::new(dt, 100);
    
    println!("Simulation ready!");
    println!("Grid: {}x{}x{}", grid.nx, grid.ny, grid.nz);
    println!("Time step: {:.2e} s", dt);
    
    Ok(())
}
```

## Architecture

The library follows enterprise-grade design principles:

### Applied Design Principles ✅
- **SOLID**: Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion
- **CUPID**: Composable, Unix Philosophy, Predictable, Idiomatic, Domain-based
- **GRASP**: General Responsibility Assignment Software Patterns
- **CLEAN**: Clear, Lean, Efficient, Adaptable, Neat
- **SSOT/SPOT**: Single Source/Point of Truth

### Module Structure
```
kwavers/
├── physics/          # Physics models and traits
├── solver/           # Numerical methods
├── medium/           # Material properties
├── boundary/         # Boundary conditions
├── source/           # Wave sources
├── grid/            # Grid management
└── utils/           # Utilities and helpers
```

## Current Metrics

| Metric | Value | Status | Target |
|--------|-------|--------|--------|
| Build Errors | 0 | ✅ Excellent | 0 |
| Test Compilation | Partial | ⚠️ In Progress | All Pass |
| Example Errors | Few | 🔄 Being Fixed | 0 |
| Warnings | 502 | ⚠️ Stable | <50 |
| Code Coverage | TBD | 📅 Planned | >80% |

## Module Refactoring Success

Successfully transformed monolithic modules into clean, focused components:

**Before**: `nonlinear/core.rs` (1172 lines) - Mixed concerns  
**After**:
- `wave_model.rs` (262 lines) - Data structures
- `multi_frequency.rs` (135 lines) - Frequency handling
- `numerical_methods.rs` (352 lines) - Core algorithms
- `trait_impl.rs` (134 lines) - Trait implementations

## Roadmap

### ✅ Completed
- Fix all build errors
- Establish clean architecture
- Apply design principles
- Get basic examples working

### 🔄 In Progress
- Fix test compilation issues
- Update remaining examples
- Reduce warnings

### 📅 Planned
- Complete test coverage
- Add benchmarks
- GPU acceleration
- Physics validation
- Production optimization

## Contributing

We welcome contributions! Priority areas:

1. **Test Fixes** - Help resolve compilation issues
2. **Example Updates** - Ensure all examples work
3. **Warning Reduction** - Clean up code
4. **Documentation** - Add missing docs
5. **Benchmarks** - Performance testing

## Dependencies

```toml
[dependencies]
ndarray = "0.15"     # N-dimensional arrays
rustfft = "6.1"      # FFT operations
rayon = "1.7"        # Parallel processing
nalgebra = "0.32"    # Linear algebra
```

## License

MIT License - See [LICENSE](LICENSE) for details

## Assessment

**Current State**: Functional alpha with working core features and examples.  
**Timeline to Production**: 2-3 months with continued development.  
**Key Achievement**: Clean, maintainable architecture following Rust best practices.