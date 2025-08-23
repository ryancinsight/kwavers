# Kwavers: Acoustic Wave Simulation Library

[![Rust](https://img.shields.io/badge/rust-1.89%2B-blue.svg)](https://www.rust-lang.org)
[![Build](https://img.shields.io/badge/build-failing-red.svg)](https://github.com/kwavers/kwavers)
[![Tests](https://img.shields.io/badge/tests-unknown-yellow.svg)](./tests)
[![Warnings](https://img.shields.io/badge/warnings-322-red.svg)](./src)
[![Grade](https://img.shields.io/badge/grade-C-yellow.svg)](./PRD.md)

## Acoustic Wave Simulation Library - Under Active Refactoring

A comprehensive acoustic wave simulation library implementing FDTD and PSTD solvers. **Currently undergoing critical refactoring to address architectural issues.**

### ⚠️ Current Status (v2.15.0)
- **Build**: Currently failing due to refactoring
- **Warnings**: 322 warnings (reduced from 476)
- **Module Structure**: 15+ files exceed 500 lines (violation)
- **Physics**: CFL corrected to safe values
- **Code Quality**: Active cleanup in progress

## Critical Issues Being Addressed

### Architecture Problems
- **Module Size Violations**: 15 files exceed 500 lines (worst: 1138 lines)
- **SRP Violations**: Multiple responsibilities in single modules
- **Dead Code**: Significant unused code being removed
- **Incomplete Implementations**: Placeholders being completed

### Improvements Made
- ✅ Removed dishonest warning suppressions
- ✅ Fixed CFL stability (was unsafe 0.95, now 0.5)
- ✅ Eliminated adjective-based naming
- ✅ Completed placeholder implementations
- ✅ Started module splitting (FDTD refactored)

## Quick Start (When Build Fixed)

```toml
[dependencies]
kwavers = "2.15.0"  # Not recommended until refactoring complete
```

## Core Features

### Numerical Solvers
- **FDTD** - Yee scheme with corrected CFL (0.3 for 3D stability)
- **PSTD** - FFT-based pseudo-spectral solver
- **Plugin Architecture** - Being refactored for cleaner separation

### Physics Models
- Acoustic wave propagation (CFL validated)
- PML/CPML boundary conditions
- Homogeneous and heterogeneous media
- Chemical kinetics (implementations completed)
- Bubble dynamics and cavitation
- Thermal coupling

## Project Structure

```
src/
├── solver/           # Numerical solvers
│   ├── fdtd/        # Being split into submodules
│   │   ├── mod.rs   # Core solver (reduced from 1138 lines)
│   │   ├── config.rs # Configuration (extracted)
│   │   └── ...
│   ├── pstd/        # FFT-based implementation
│   └── ...
├── physics/          # Physics models
├── boundary/        # Boundary conditions (918 lines - needs splitting)
└── ...             # 369 source files (many need refactoring)
```

## Known Issues

### Critical
- Build currently failing due to module reorganization
- 15+ modules violate 500-line limit
- 322 warnings indicate design issues

### In Progress
- Splitting large modules into domain-based structure
- Removing dead code
- Completing stub implementations
- Fixing underscored variables

## Design Principles Being Enforced

- **SSOT/SPOT** - Single Source/Point of Truth
- **SOLID** - Especially Single Responsibility
- **CUPID** - Composable architecture
- **GRASP** - Proper responsibility assignment
- **SLAP** - Single Level of Abstraction
- **DRY** - Don't Repeat Yourself
- **CLEAN** - Clear, Lean, Efficient, Adaptable, Neat

## Module Size Violations (Top 10)

1. `solver/fdtd/mod.rs` - 1018 lines (being split)
2. `source/flexible_transducer.rs` - 1097 lines
3. `utils/kwave_utils.rs` - 976 lines
4. `solver/hybrid/validation.rs` - 960 lines
5. `source/transducer_design.rs` - 957 lines
6. `solver/spectral_dg/dg_solver.rs` - 943 lines
7. `sensor/beamforming.rs` - 923 lines
8. `boundary/cpml.rs` - 918 lines
9. `source/hemispherical_array.rs` - 917 lines
10. `medium/heterogeneous/tissue.rs` - 917 lines

## Physics Validation

### CFL Stability (Corrected)
- Maximum stable CFL for 3D FDTD: 1/√3 ≈ 0.577
- Implementation: 0.3 (safety margin)
- Reference: Taflove & Hagness (2005)

### Numerical Methods
- FDTD: Yee's staggered grid (validated)
- PSTD: FFT-based (proper implementation)
- Absorption: Beer-Lambert law validated

## Contributing

This project needs significant work:

1. **Module Refactoring** - Split all files > 500 lines
2. **Dead Code Removal** - Clean up unused code
3. **Complete Implementations** - No placeholders
4. **Fix Build** - Resolve current compilation errors

### Guidelines
- NO warning suppressions
- Modules MUST be < 500 lines
- NO adjectives in naming
- Complete implementations only
- Validate against literature

## Current Assessment

**Grade: C (Significant Issues)**

The codebase has potential but requires major refactoring:
- Massive module size violations
- Incomplete separation of concerns
- Build currently broken during refactoring
- Significant technical debt

**NOT recommended for production use until refactoring complete.**

---

**Version**: 2.15.0  
**Status**: Under active refactoring  
**Recommendation**: Wait for v3.0.0 after refactoring
