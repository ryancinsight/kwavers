# Kwavers: Acoustic Wave Simulation Library

[![Rust](https://img.shields.io/badge/rust-1.89%2B-blue.svg)](https://www.rust-lang.org)
[![Build](https://img.shields.io/badge/build-passing-green.svg)](https://github.com/kwavers/kwavers)
[![Tests](https://img.shields.io/badge/tests-16%2F16-green.svg)](./tests)
[![Warnings](https://img.shields.io/badge/warnings-475-red.svg)](./src)
[![Grade](https://img.shields.io/badge/grade-C%2B-yellow.svg)](./PRD.md)

## Acoustic Wave Simulation Library - Under Development

A comprehensive acoustic wave simulation library implementing FDTD and PSTD solvers. **Currently in active development with significant issues to address.**

### ⚠️ Current Status (v2.15.0)
- **475 Warnings** - Significant unused code and imports
- **Tests Pass** - 16/16 test suites successful
- **Examples Work** - 7 examples run but may have issues
- **Module Structure** - 19 files exceed 500 lines (violation)
- **Physics Issues** - CFL constants corrected, some hardcoded values remain

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
- **FDTD** - Yee scheme with corrected CFL stability (was using incorrect 0.95)
- **PSTD** - Proper FFT-based spectral implementation
- **Plugin Architecture** - Extensible but complex

### Physics Models
- Acoustic wave propagation with corrected CFL (now 0.5 for 3D)
- PML/CPML boundary conditions
- Homogeneous and heterogeneous media
- Chemical kinetics (modularized but has placeholders)
- Bubble dynamics and cavitation
- Thermal coupling

### Known Issues
- **Code Quality** - 475 warnings from unused code
- **Module Size** - 19 files > 500 lines, some > 1000 lines
- **Incomplete Features** - Several TODOs and placeholders remain
- **Adjective Naming** - Fixed several violations but more may exist
- **Misleading Claims** - Previous "Grade A-" was incorrect

## Project Structure

```
src/
├── solver/           # Numerical solvers
│   ├── fdtd/        # 1138 lines - NEEDS SPLITTING
│   ├── pstd/        # Properly implemented with FFT
│   └── ...
├── physics/          # Physics models
│   ├── chemistry/   # Has placeholder implementations
│   ├── mechanics/   # 830+ lines - needs refactoring
│   └── ...
├── boundary/        # 918 lines - needs splitting
├── medium/          # Material properties
└── ...             # 369 source files, many too large
```

## Building & Testing

```bash
# Build (will show 475 warnings)
cargo build --release

# Run all tests (they pass but don't validate everything)
cargo test --release

# Generate documentation
cargo doc --no-deps --open
```

### Test Coverage
```
✅ Unit tests:        3/3
✅ Integration tests: 5/5
✅ Solver tests:      3/3
✅ Doc tests:         5/5
━━━━━━━━━━━━━━━━━━━━━━━━
Total: 16/16 (but limited coverage)
```

## Examples

Seven examples that run but may not fully validate physics:

```bash
# Basic FDTD simulation
cargo run --release --example basic_simulation

# Plugin system demonstration
cargo run --release --example plugin_example

# Physics validation (needs verification)
cargo run --release --example physics_validation

# Others...
```

## Code Quality - Real Assessment

| Metric | Status | Grade |
|--------|--------|-------|
| **Correctness** | Tests pass, physics partially fixed | C+ |
| **Safety** | No unsafe in critical paths | B+ |
| **Warnings** | 475 warnings present | D |
| **Documentation** | Comprehensive but misleading | C |
| **Architecture** | Major violations of 500-line rule | D |
| **Performance** | Not properly profiled | C |

## Critical Issues Found

### Physics Corrections Made
- ✅ Fixed CFL constant from unsafe 0.95 to safe 0.5 for 3D FDTD
- ✅ Added proper literature references for CFL stability

### Remaining Problems
- ❌ 475 compilation warnings (hidden by suppressions)
- ❌ 19 modules exceed 500 lines (some > 1000)
- ❌ Multiple placeholder/stub implementations
- ❌ Adjective-based naming in comments/docs
- ❌ Incomplete error handling in places
- ❌ Hardcoded physical constants

## Honest Assessment

This codebase is **NOT production ready** despite previous claims. Key issues:

1. **Module Organization** - Massive violations of SOLID principles with 1000+ line files
2. **Code Cleanliness** - 475 warnings indicate significant dead code
3. **Incomplete Implementation** - Multiple stubs and placeholders
4. **Physics Issues** - Had critical CFL stability bug (now fixed)
5. **Misleading Documentation** - Previous "Grade A-" was dishonest

## What Needs to Be Done

### Immediate Priority
1. Split all modules > 500 lines
2. Remove dead code causing 475 warnings
3. Complete stub implementations or remove them
4. Validate all physics against literature

### Architecture Fixes
1. Apply proper domain-driven design
2. Enforce single responsibility principle
3. Remove coupling between modules
4. Implement proper error propagation

### Quality Improvements
1. Add comprehensive unit tests
2. Benchmark performance properly
3. Profile memory usage
4. Document actual limitations

## Use Cases

### Currently Suitable For
- ⚠️ Research prototypes (with careful validation)
- ⚠️ Educational demonstrations (with caveats)
- ❌ NOT ready for production use
- ❌ NOT ready for medical applications

## Contributing

This project needs significant work. Priority areas:

1. **Module Refactoring** - Split large files urgently
2. **Warning Resolution** - Clean up dead code
3. **Physics Validation** - Verify all implementations
4. **Complete Features** - Replace stubs with real code

### Guidelines
- No warning suppressions
- Modules must be < 500 lines
- No adjectives in naming
- Complete implementations only
- Validate against literature

## License

MIT - See [LICENSE](LICENSE)

---

**Version**: 2.15.0  
**Grade**: C+ (Significant Issues)  
**Status**: 475 warnings, module structure violations, incomplete features  
**Recommendation**: NOT production ready - requires major refactoring
