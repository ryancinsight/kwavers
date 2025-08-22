# Kwavers: Acoustic Wave Simulation Library (Beta)

[![Rust](https://img.shields.io/badge/rust-1.89%2B-blue.svg)](https://www.rust-lang.org)
[![Build](https://img.shields.io/badge/build-passing-green.svg)](https://github.com/kwavers/kwavers)
[![Core Tests](https://img.shields.io/badge/core_tests-passing-green.svg)](./tests)
[![Examples](https://img.shields.io/badge/examples-5_of_7-yellow.svg)](./examples)
[![Status](https://img.shields.io/badge/status-beta-orange.svg)](./src)

## ⚠️ Beta Software - Read Before Use

This library has a **stable core** but **experimental features**. The plugin system has known issues.

### Quick Assessment
- ✅ **Core simulation**: Production ready
- ⚠️ **Plugin system**: Has segfault issues
- ❌ **GPU support**: Not implemented

## Installation

```toml
[dependencies]
kwavers = "0.9.0-beta"  # Beta release
```

## Safe Usage Pattern

```rust
// ✅ SAFE: Direct solver usage
use kwavers::{
    grid::Grid,
    medium::HomogeneousMedium,
    solver::fdtd::FdtdSolver,
};

let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
let medium = HomogeneousMedium::water(&grid);
// Direct solver usage is stable

// ⚠️ CAUTION: Plugin system may segfault
// Test thoroughly before production use
```

## What Works ✅

- Grid and medium abstractions
- Basic FDTD solver (direct usage)
- PML/CPML boundaries
- 5 of 7 examples
- Integration tests

## Known Issues ⚠️

1. **Plugin system segfaults** - Memory management issues
2. **PSTD uses finite differences** - Spectral methods removed due to bugs
3. **2 examples fail** - tissue_model and wave_simulation have issues
4. **GPU is stub code** - Not implemented

## Testing

```bash
# Safe tests
cargo test --test integration_test

# These may crash:
# - solver_test.rs (plugin issues)
# - fdtd_pstd_comparison.rs (plugin issues)
```

## Examples

```bash
# Working examples
cargo run --example basic_simulation
cargo run --example plugin_example  # May crash
cargo run --example phased_array_beamforming
cargo run --example physics_validation

# Broken examples
# - tissue_model_example (config issues)
# - wave_simulation (performance issues)
```

## Development Status

| Version | Status | Focus |
|---------|--------|-------|
| 0.9.0-beta | Current | Core features stable, plugins experimental |
| 1.0.0 | Planned | Fix plugin architecture |
| 2.0.0 | Future | GPU implementation |

## Contributing

Priority areas:
1. Fix plugin system memory management
2. Optimize spectral methods
3. Implement GPU support
4. Fix failing examples

## Support

- Issues: [GitHub Issues](https://github.com/kwavers/kwavers/issues)
- Status: [CHECKLIST.md](CHECKLIST.md)

## License

MIT - See [LICENSE](LICENSE)

---

**⚠️ Beta Software**: Core is stable, plugins are experimental. Test thoroughly before production use.