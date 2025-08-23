# Kwavers: Acoustic Wave Simulation Library

[![Rust](https://img.shields.io/badge/rust-1.89%2B-blue.svg)](https://www.rust-lang.org)
[![Build](https://img.shields.io/badge/build-passing-green.svg)](https://github.com/kwavers/kwavers)
[![Tests](https://img.shields.io/badge/tests-16%2F16-green.svg)](./tests)
[![Warnings](https://img.shields.io/badge/warnings-324-red.svg)](./src)
[![Grade](https://img.shields.io/badge/grade-D-red.svg)](./PRD.md)

## ⚠️ NOT PRODUCTION READY - Major Refactoring Required

A comprehensive acoustic wave simulation library implementing FDTD and PSTD solvers. **Currently has significant architectural issues that must be addressed before production use.**

### 🔴 Critical Status (v2.15.0)
- **Build**: Passes with 324 warnings
- **Tests**: 16/16 pass (insufficient coverage)
- **Architecture**: SEVERE violations - 20+ modules exceed 500 lines
- **Code Quality**: Grade D - Major technical debt
- **Production Ready**: ❌ NO

## Major Issues Requiring Resolution

### Architecture Violations (Critical)
- **20+ modules exceed 500 lines** (worst: 1097 lines)
  - `flexible_transducer.rs`: 1097 lines
  - `kwave_utils.rs`: 976 lines
  - `hybrid/validation.rs`: 960 lines
  - `fdtd/mod.rs`: 949 lines
  - And 16+ more...
- **SRP violations throughout** - modules mixing multiple concerns
- **324 warnings** indicating incomplete implementations
- **21 TODOs/placeholders** found in critical paths

### Physics Implementation Issues
- ✅ CFL corrected to 0.5 (was unsafe 0.95)
- ⚠️ Multiple placeholder implementations in hybrid solver
- ⚠️ Incomplete interpolation methods returning zeros
- ⚠️ Chemistry module using placeholder concentrations

### Code Quality Problems
- **Adjective naming violations** partially fixed
- **Underscored variables** indicating unused code
- **Dead code** throughout (324 warnings)
- **Missing implementations** masked as "placeholders"

## Quick Start ⚠️ NOT RECOMMENDED

```toml
[dependencies]
# DO NOT USE IN PRODUCTION
kwavers = "2.15.0"  # Has major architectural issues
```

## Core Features (With Caveats)

### Numerical Solvers
- **FDTD** - Working but module too large (949 lines)
- **PSTD** - Functional but uses finite differences (not true spectral)
- **Plugin Architecture** - Over-engineered, needs simplification

### Physics Models
- Acoustic wave propagation (CFL validated ✅)
- Chemistry (placeholder implementations ⚠️)
- Bubble dynamics (incomplete thermal models ⚠️)
- Elastic waves (830 lines - needs splitting ⚠️)

## Required Refactoring

### Immediate Actions Needed
1. **Split all 20+ modules >500 lines**
2. **Remove 324 warnings** - fix unused code
3. **Complete 21 placeholder implementations**
4. **Add comprehensive test coverage** (current: minimal)
5. **Implement proper error handling** throughout

### Architecture Redesign
- Apply SOLID principles properly
- Implement true SRP - one responsibility per module
- Remove over-engineering in plugin system
- Simplify complex inheritance hierarchies

## Documentation

- [API Documentation](https://docs.rs/kwavers) - ⚠️ Incomplete
- [Examples](./examples) - 7 examples (some may not reflect best practices)
- [PRD](./PRD.md) - Grade D assessment
- [CHECKLIST](./CHECKLIST.md) - Major items incomplete

## Performance

**Not benchmarked** - Architecture issues must be fixed first.

## Contributing

This project needs significant refactoring before accepting contributions.
Priority should be on:
1. Splitting large modules
2. Fixing architectural violations
3. Completing placeholder implementations
4. Adding comprehensive tests

## License

MIT License - Use at your own risk given current state.

## Disclaimer

**This library is NOT production ready.** It has significant architectural issues, incomplete implementations, and technical debt that must be addressed before any serious use. The code passes tests but has major structural problems that will cause maintenance nightmares and potential runtime issues.

### Known Risks
- Large modules make debugging difficult
- Placeholder implementations may cause unexpected behavior
- 324 warnings indicate significant dead code
- Insufficient test coverage for production use

**Grade: D** - Requires major refactoring before production use.
