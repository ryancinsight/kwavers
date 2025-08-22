# Kwavers: Acoustic Wave Simulation Library

[![Rust](https://img.shields.io/badge/rust-1.89%2B-blue.svg)](https://www.rust-lang.org)
[![Build](https://img.shields.io/badge/build-passing-green.svg)](https://github.com/kwavers/kwavers)
[![Tests](https://img.shields.io/badge/tests-5_of_5_core-yellow.svg)](./tests)
[![Examples](https://img.shields.io/badge/examples-5_of_7-yellow.svg)](./examples)
[![Warnings](https://img.shields.io/badge/warnings-0-green.svg)](./src)
[![Status](https://img.shields.io/badge/status-partial_production-yellow.svg)](./src)

## ⚠️ Production Status - Please Read

**Core features are stable. Advanced features have issues.**

| Feature | Status | Safe to Use |
|---------|--------|-------------|
| **Core Simulation** | ✅ Stable | Yes |
| **FDTD Solver** | ✅ Working | Yes |
| **Plugin System** | ✅ Functional | Yes |
| **PSTD Solver** | ⚠️ Segfaults | Test carefully |
| **GPU Support** | ❌ Not implemented | No |
| **RTM/FWI** | ❌ Broken APIs | No |

## Quick Start (Stable Features Only)

```bash
# Build
cargo build --release

# Run working examples
cargo run --release --example basic_simulation
cargo run --release --example plugin_example
cargo run --release --example phased_array_beamforming

# Run core tests (advanced tests disabled)
cargo test --test integration_test
```

## What Works ✅

### Core Features
- Grid and medium abstractions
- Basic FDTD solver
- Plugin architecture
- PML/CPML boundaries
- Homogeneous media
- Basic wave propagation

### Working Examples
- `basic_simulation` - Core functionality demo
- `plugin_example` - Plugin system demo
- `phased_array_beamforming` - Array control
- `physics_validation` - Validation tests
- `wave_simulation` - Works but slow

## Known Issues ⚠️

### Critical Problems
1. **PSTD solver causes segmentation faults** - FFT buffer issues
2. **Advanced tests disabled** - 4 test files don't compile/crash
3. **Examples fail** - tissue_model_example doesn't work
4. **GPU is stub code only** - Not implemented

### Do Not Use
- GPU acceleration (not implemented)
- RTM/FWI reconstruction (broken APIs)
- PSTD solver in production (segfaults)

## Usage Example (Safe)

```rust
use kwavers::{
    grid::Grid,
    medium::HomogeneousMedium,
    solver::fdtd::{FdtdConfig, FdtdSolver},
    error::KwaversResult,
};

fn main() -> KwaversResult<()> {
    // This is safe and works
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
    let medium = HomogeneousMedium::water(&grid);
    
    let config = FdtdConfig {
        spatial_order: 2,
        staggered_grid: true,
        cfl_factor: 0.5,
        subgridding: false,
        subgrid_factor: 1,
    };
    
    // Basic simulation works fine
    // ... 
    
    Ok(())
}
```

## Installation

```toml
[dependencies]
# Use with caution - not all features work
kwavers = "1.0.0-rc1"
```

## Development Status

This is a **release candidate** with:
- ✅ Stable core (production-ready)
- ⚠️ Experimental advanced features (use with caution)
- ❌ Incomplete GPU support (do not use)

### Roadmap
- **v1.0** - Current release candidate
- **v1.1** - Fix segfaults and test issues
- **v2.0** - Complete GPU implementation

## Contributing

We need help with:
1. Fixing PSTD segmentation faults
2. Updating test APIs
3. Implementing GPU support
4. Performance optimization

## Testing

```bash
# Run only stable tests
cargo test --test integration_test

# Disabled tests (will crash):
# - solver_test.rs
# - fdtd_pstd_comparison.rs
# - rtm_validation_tests.rs
# - fwi_validation_tests.rs
```

## Documentation

- Core features are well documented
- Advanced features may have incomplete docs
- See examples for usage patterns

## License

MIT License - See [LICENSE](LICENSE) for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/kwavers/kwavers/issues)
- **Known Problems**: See [CHECKLIST.md](CHECKLIST.md)
- **Status**: See [PRD.md](PRD.md)

## ⚠️ Important Notice

This library is partially production-ready. Core acoustic simulation features work well, but advanced imaging and GPU features are broken or unimplemented. Please test thoroughly before production use.

---

**Status: Partial Production** ⚠️

Use core features with confidence. Avoid advanced features until v2.0.