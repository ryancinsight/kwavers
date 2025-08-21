# Kwavers: Acoustic Wave Simulation Library

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)
[![Build](https://img.shields.io/badge/build-passing-green.svg)](https://github.com/kwavers/kwavers)
[![Warnings](https://img.shields.io/badge/warnings-502-yellow.svg)](./src)
[![Examples](https://img.shields.io/badge/examples-working-green.svg)](./examples)

## Project Status

**Build Status**: ✅ **PASSING**  
**Examples**: ✅ **WORKING** (basic_simulation runs successfully)  
**Warnings**: ⚠️ 502 (stable, manageable)  
**Tests**: ⚠️ Compilation issues remain  
**Code Quality**: 📈 Improving  

## Overview

Kwavers is an acoustic wave simulation library written in Rust, designed for:
- Medical ultrasound simulation
- Nonlinear acoustic wave propagation
- Photoacoustic imaging
- Computational acoustics research

The library implements various numerical methods including FDTD, PSTD, and spectral methods for solving acoustic wave equations.

## Recent Improvements

### ✅ Completed
- **Build Success**: All compilation errors fixed
- **Module Refactoring**: Large modules split into focused components
- **Code Modernization**: Started replacing C-style loops with iterators
- **Examples Working**: Basic simulation example runs successfully
- **Error Handling**: Proper error types throughout

### 🔄 In Progress
- Warning reduction (stable at ~500)
- Test suite fixes
- Documentation improvements
- Performance optimization

## Quick Start

```bash
# Clone and build
git clone https://github.com/kwavers/kwavers
cd kwavers
cargo build --release

# Run working example
cargo run --example basic_simulation

# Output shows successful simulation:
# Grid properties:
#   CFL timestep: 1.15e-7 s
#   Grid points: 262144
#   Memory estimate: 21.0 MB
```

## Working Example

```rust
use kwavers::{Grid, Time};
use kwavers::medium::homogeneous::HomogeneousMedium;
use kwavers::source::{PointSource, GaussianPulse};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create 64x64x64 grid with 1mm spacing
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
    
    // Water medium
    let medium = HomogeneousMedium::water(&grid);
    
    // Point source at center
    let source = PointSource::new(
        32.0e-3, 32.0e-3, 32.0e-3,  // position
        GaussianPulse::new(1e6, 1e-6) // 1 MHz, 1 μs pulse
    );
    
    // Time stepping
    let dt = grid.cfl_timestep_default(1500.0);
    let time = Time::new(dt, 100);
    
    println!("Simulation configured successfully!");
    println!("Grid: {}x{}x{}", grid.nx, grid.ny, grid.nz);
    println!("Time step: {:.2e} s", dt);
    
    Ok(())
}
```

## Architecture

```
kwavers/
├── src/
│   ├── physics/           # Physics models
│   │   ├── mechanics/     # Wave mechanics
│   │   │   └── acoustic_wave/
│   │   │       └── nonlinear/ # ✅ Refactored
│   │   ├── chemistry/     # Chemical models
│   │   └── optics/        # Optical models
│   ├── solver/            # Numerical solvers
│   │   ├── fdtd/         # Finite-difference time-domain
│   │   ├── pstd/         # Pseudo-spectral time-domain
│   │   └── spectral_dg/  # Spectral methods
│   ├── medium/           # Material properties
│   ├── source/           # Acoustic sources
│   ├── boundary/         # Boundary conditions
│   ├── grid/            # ✅ Modernized with iterators
│   └── fft/             # FFT operations
├── examples/            # ✅ Working examples
└── tests/              # Test suite (needs work)
```

## Code Quality Metrics

| Metric | Current | Target | Progress |
|--------|---------|--------|----------|
| Build | ✅ Passing | ✅ Passing | Complete |
| Warnings | 502 | <100 | 🔄 Ongoing |
| Examples | Working | All working | 🔄 Testing |
| Tests | Issues | All passing | ⚠️ TODO |
| C-style loops | ~850 | 0 | 🔄 Started |
| Documentation | Partial | Complete | 🔄 Ongoing |

## Module Refactoring Achievement

Successfully refactored the 1172-line `nonlinear/core.rs` into:
- `wave_model.rs` (262 lines) - Core data structures
- `multi_frequency.rs` (135 lines) - Frequency configuration  
- `numerical_methods.rs` (352 lines) - Algorithms
- `trait_impl.rs` (134 lines) - Trait implementations

This demonstrates proper Rust patterns:
- **Single Responsibility**: Each module has one clear purpose
- **Separation of Concerns**: Data, algorithms, and traits separated
- **Manageable Size**: All modules under 400 lines

## Features

### ✅ Working
- Basic acoustic wave simulation
- Grid creation and management
- Homogeneous medium modeling
- Point and Gaussian sources
- FDTD and PSTD solvers (basic)
- FFT operations

### 🔄 Partial
- Nonlinear acoustics
- Boundary conditions
- Heterogeneous media
- Multi-frequency simulations

### ❌ Not Implemented
- GPU acceleration (stubs only)
- Machine learning integration
- Advanced visualization

## Performance

Current performance characteristics:
- Grid operations: Using iterators for better optimization
- Memory usage: ~21 MB for 64³ grid
- Compilation: Release mode recommended for performance

## Development Roadmap

### Immediate (This Week)
- [x] Fix compilation errors
- [x] Get examples working
- [ ] Fix test compilation
- [ ] Reduce warnings to <200

### Short Term (2 Weeks)
- [ ] Complete test suite
- [ ] Modernize remaining loops
- [ ] Add benchmarks
- [ ] Improve documentation

### Medium Term (1 Month)
- [ ] Refactor large modules
- [ ] Validate physics
- [ ] Performance optimization
- [ ] Add more examples

### Long Term (3 Months)
- [ ] GPU implementation
- [ ] ML integration
- [ ] Production readiness
- [ ] Comprehensive validation

## Dependencies

Core dependencies:
```toml
ndarray = "0.15"      # N-dimensional arrays
rustfft = "6.1"       # FFT operations
rayon = "1.7"         # Parallel processing
nalgebra = "0.32"     # Linear algebra
```

## Building and Testing

```bash
# Build library
cargo build --release

# Run tests (currently has issues)
cargo test

# Run specific example
cargo run --example basic_simulation

# Check code quality
cargo clippy

# Format code
cargo fmt
```

## Contributing

Contributions are welcome! Priority areas:

1. **Test Fixes**: Help resolve test compilation issues
2. **Warning Reduction**: Clean up the remaining warnings
3. **Loop Modernization**: Convert C-style loops to iterators
4. **Documentation**: Add missing documentation
5. **Examples**: Create more working examples

Please ensure:
- Code follows Rust best practices
- New features include tests
- Documentation is updated
- Examples demonstrate usage

## License

MIT License - See [LICENSE](LICENSE)

## Acknowledgments

This project demonstrates modern Rust patterns for scientific computing, including:
- Zero-cost abstractions
- Iterator-based algorithms
- Trait-based design
- Safe concurrency with Rayon

## Current Assessment

**Positive Achievements:**
- ✅ Project compiles and runs
- ✅ Basic examples work
- ✅ Architecture is sound
- ✅ Refactoring patterns established

**Areas for Improvement:**
- ⚠️ Test suite needs fixes
- ⚠️ High warning count
- ⚠️ Some features incomplete
- ⚠️ Physics validation needed

**Overall**: The library is functional for basic use cases and development. While not production-ready, it provides a solid foundation for acoustic wave simulation in Rust.