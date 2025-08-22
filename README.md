# Kwavers: Acoustic Wave Simulation Library

[![Rust](https://img.shields.io/badge/rust-1.89%2B-blue.svg)](https://www.rust-lang.org)
[![Build](https://img.shields.io/badge/build-passing-green.svg)](https://github.com/kwavers/kwavers)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](./tests)
[![Examples](https://img.shields.io/badge/examples-working-green.svg)](./examples)
[![Status](https://img.shields.io/badge/status-production-green.svg)](./src)

## Production-Ready Acoustic Simulation

A high-performance acoustic wave simulation library in Rust with zero unsafe code issues.

### Key Features
- ✅ **Memory safe** - No unsafe code in critical paths
- ✅ **Zero warnings** - Clean compilation
- ✅ **Plugin system** - Extensible architecture
- ✅ **Working examples** - All demonstrations functional
- ✅ **Production ready** - Suitable for real applications

## Installation

```toml
[dependencies]
kwavers = "1.0.0"
```

## Quick Start

```rust
use kwavers::{
    grid::Grid,
    medium::HomogeneousMedium,
    solver::fdtd::FdtdSolver,
};

// Create simulation grid
let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);

// Define medium properties
let medium = HomogeneousMedium::water(&grid);

// Run simulation
// ... solver configuration and execution
```

## Examples

All examples are working and demonstrate various features:

```bash
# Basic simulation
cargo run --release --example basic_simulation

# Plugin system demonstration
cargo run --release --example plugin_example

# Phased array beamforming
cargo run --release --example phased_array_beamforming

# Physics validation
cargo run --release --example physics_validation

# Wave propagation
cargo run --release --example wave_simulation
```

## Architecture

### Core Components
- **Grid System** - Flexible 3D grid management
- **Medium Modeling** - Homogeneous and heterogeneous media
- **Solver Framework** - FDTD, PSTD (simplified), and plugin-based
- **Boundary Conditions** - PML, CPML implementations
- **Plugin System** - Safe, extensible architecture

### Design Principles
- **SOLID** - Single responsibility, dependency inversion
- **Zero-cost abstractions** - Rust's compile-time optimizations
- **Memory safety** - No unsafe code in core functionality
- **Type safety** - Strong typing throughout

## Performance

- Optimized for release builds
- Zero-copy operations where possible
- Parallel processing support via Rayon
- Efficient memory layout

## Testing

```bash
# Run all tests
cargo test --release

# Run integration tests
cargo test --test integration_test

# Run with optimizations
cargo test --release
```

## Documentation

```bash
# Generate documentation
cargo doc --open

# Run doctests
cargo test --doc
```

## Current Limitations

### Not Implemented
- GPU acceleration (stub interfaces only)
- Some advanced physics models

### Simplified
- PSTD uses finite differences instead of spectral methods (stability reasons)

## Contributing

Contributions welcome! Priority areas:
1. GPU implementation
2. Performance optimizations
3. Additional physics models
4. More examples

## License

MIT - See [LICENSE](LICENSE)

## Support

- Issues: [GitHub Issues](https://github.com/kwavers/kwavers/issues)
- Documentation: Generated via `cargo doc`

---

**Status: Production Ready** ✅

The library is stable, safe, and suitable for production use in acoustic wave simulation applications.