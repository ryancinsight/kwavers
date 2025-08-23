# Kwavers: Acoustic Wave Simulation Library

[![Rust](https://img.shields.io/badge/rust-1.89%2B-blue.svg)](https://www.rust-lang.org)
[![Build](https://img.shields.io/badge/build-passing-green.svg)](https://github.com/kwavers/kwavers)
[![Tests](https://img.shields.io/badge/tests-16%2F16-green.svg)](./tests)
[![Grade](https://img.shields.io/badge/grade-B%2B-yellow.svg)](./PRD.md)

## Acoustic Wave Simulation in Rust

A comprehensive acoustic wave simulation library implementing FDTD and simplified PSTD solvers with extensive physics models. The codebase is functional and well-documented but requires structural refinement.

### Current Status
- ✅ **All Tests Passing** - 16/16 test suites successful
- ✅ **Clean Compilation** - Builds without errors
- ✅ **Physics Validated** - Algorithms match literature
- ⚠️ **Refactoring Needed** - Some modules exceed 1000 lines
- ⚠️ **Technical Debt** - 4 TODOs, some underscored variables

## Quick Start

```toml
[dependencies]
kwavers = "2.15.0"
```

```rust
use kwavers::{Grid, HomogeneousMedium, FdtdSolver};

// Create simulation grid
let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);

// Define medium (water)
let medium = HomogeneousMedium::water(&grid);

// Run simulation
let solver = FdtdSolver::new(config, &grid)?;
```

## Core Features

### Solvers
- **FDTD** (Finite-Difference Time-Domain) - Complete Yee scheme implementation
- **PSTD** (Pseudo-Spectral Time-Domain) - Simplified finite-difference version

### Physics Models
- Wave propagation with proper CFL conditions
- PML/CPML boundary conditions
- Homogeneous and heterogeneous media
- Chemical kinetics and bubble dynamics
- Thermal diffusion coupling

### Architecture
- Plugin-based computation pipeline
- Trait-based abstractions
- Zero-copy where possible
- Comprehensive error handling

## Code Quality

| Metric | Status | Notes |
|--------|--------|-------|
| **Lines of Code** | ~50,000 | 369 source files |
| **Test Coverage** | Good | Core paths covered |
| **Documentation** | Extensive | Literature references included |
| **Module Size** | Mixed | 8 files > 900 lines |
| **Design Patterns** | B+ | SOLID mostly followed |

## Known Issues

### Needs Immediate Attention
1. **Large Modules** - fdtd/mod.rs (1138 lines), chemistry/mod.rs (964 lines)
2. **Magic Numbers** - Some constants not properly named
3. **TODO Comments** - 4 unresolved items
4. **Excessive Examples** - 30 examples (5-10 would suffice)

### Limitations
- PSTD uses finite differences, not true spectral methods
- GPU support is stub implementation only
- Some underscored variables indicate incomplete features

## Building

```bash
# Standard build
cargo build --release

# Run all tests
cargo test --release

# Build examples
cargo build --release --examples

# Generate documentation
cargo doc --open
```

## Testing

All test suites currently passing:

```bash
cargo test --release

✅ Integration tests: 5/5
✅ Solver tests: 3/3  
✅ Comparison tests: 3/3
✅ Doc tests: 5/5
━━━━━━━━━━━━━━━━━━━━━
Total: 16/16 (100%)
```

## Examples

Key examples demonstrating core functionality:

```bash
# Basic FDTD simulation
cargo run --release --example basic_simulation

# Plugin architecture demo
cargo run --release --example plugin_example

# Physics validation
cargo run --release --example physics_validation
```

Note: Currently 30 examples exist; consolidation to 5-10 focused demos recommended.

## Project Structure

```
src/
├── solver/          # Numerical solvers (needs splitting)
│   ├── fdtd/       # 1138 lines (too large)
│   ├── pstd/       # Simplified implementation
│   └── ...
├── physics/         # Physics models
│   ├── chemistry/   # Recently split into submodules
│   ├── mechanics/   
│   └── ...
├── boundary/        # Boundary conditions
├── medium/          # Material properties
└── ...             # 369 total source files
```

## Design Principles

| Principle | Grade | Notes |
|-----------|-------|-------|
| **SOLID** | B | Large modules violate SRP |
| **CUPID** | B | Overly complex in places |
| **GRASP** | B- | Some modules have too many responsibilities |
| **SSOT/SPOT** | B+ | Improved after removing redundant docs |
| **DRY** | B | Some test code duplication |

## Roadmap

### Immediate (This Week)
- [ ] Split modules > 500 lines
- [ ] Convert magic numbers to constants
- [ ] Address TODO comments
- [ ] Remove/implement underscored variables

### Short-term (This Month)
- [ ] Reduce examples from 30 to 5-10
- [ ] Implement true spectral methods for PSTD
- [ ] Add CI/CD pipeline
- [ ] Create dependency graph

### Long-term
- [ ] GPU acceleration
- [ ] Distributed computing
- [ ] Performance optimization
- [ ] Comprehensive benchmarks

## Contributing

Priority areas for contribution:
1. Module refactoring (split large files)
2. True spectral PSTD implementation
3. GPU kernel implementation
4. Performance profiling and optimization

Please ensure:
- No files > 500 lines
- All magic numbers are named constants
- No adjectives in component names
- Comprehensive tests for new features

## License

MIT - See [LICENSE](LICENSE)

---

**Version**: 2.15.0  
**Grade**: B+ (Good Quality, Needs Refinement)  
**Status**: Functional with known limitations  
**Recommendation**: Use with awareness of limitations; contribute to refactoring efforts
