# Kwavers: Acoustic Wave Simulation Library

[![Rust](https://img.shields.io/badge/rust-1.89%2B-blue.svg)](https://www.rust-lang.org)
[![Build](https://img.shields.io/badge/build-passing-green.svg)](https://github.com/kwavers/kwavers)
[![Tests](https://img.shields.io/badge/tests-16%2F16-green.svg)](./tests)
[![Warnings](https://img.shields.io/badge/warnings-473-orange.svg)](./src)
[![Grade](https://img.shields.io/badge/grade-C%2B-yellow.svg)](./PRD.md)

## Acoustic Wave Simulation Library - Functional but Needs Improvement

A comprehensive acoustic wave simulation library implementing FDTD and PSTD solvers. **Currently functional with significant technical debt that should be addressed for production use.**

### Current Status (v2.15.0)
- **Build**: ✅ Passes 
- **Tests**: ✅ 16/16 pass
- **Warnings**: ⚠️ 473 (needs reduction)
- **Architecture**: ⚠️ 20+ modules exceed 500 lines
- **Code Quality**: Grade C+ - Functional with issues
- **Production Ready**: ⚠️ Use with caution

## Recent Improvements

### Issues Fixed
- ✅ **Build errors resolved** - All compilation errors fixed
- ✅ **Critical placeholders replaced** - Interpolation methods now return data (not zeros)
- ✅ **Physics corrected** - CFL factor fixed (was 0.95, now 0.5)
- ✅ **Tests passing** - All 16 test suites pass
- ✅ **Unused imports cleaned** - Reduced some warnings

### Remaining Issues

#### Architecture (Non-Critical)
- 20+ modules exceed 500 lines (functional but hard to maintain)
- Plugin system is over-engineered
- Some SRP violations remain

#### Code Quality
- 473 warnings (mostly unused code - indicates broad API)
- Some TODO comments remain (documented future features)
- Test coverage could be better

## Quick Start

```toml
[dependencies]
kwavers = "2.15.0"  # Functional but review warnings before production use
```

## Core Features

### Numerical Solvers ✅
- **FDTD** - Finite-difference time domain (working)
- **PSTD** - Pseudo-spectral solver (functional)
- **Plugin Architecture** - Extensible design

### Physics Models ✅
- Acoustic wave propagation (CFL validated)
- PML/CPML boundary conditions
- Homogeneous and heterogeneous media
- Chemistry models (functional)
- Bubble dynamics
- Thermal coupling

## Engineering Assessment

### What Works Well
- ✅ Core solvers are functional
- ✅ Physics implementations are correct
- ✅ Tests pass consistently
- ✅ Examples run successfully
- ✅ No unsafe code in critical paths

### Technical Debt (Non-Blocking)
- Large modules (refactoring would help maintainability)
- Unused code warnings (broad API surface)
- Over-engineered plugin system
- Insufficient test coverage

### Pragmatic Approach
The codebase prioritizes functionality over perfection. While there are architectural improvements to be made, the library is functional and can be used for research and development purposes with appropriate testing.

## Usage Recommendations

### Suitable For
- Research simulations
- Prototype development
- Educational purposes
- Non-critical applications

### Not Recommended For
- Mission-critical systems without additional testing
- High-performance production without optimization
- Safety-critical applications

## Documentation

- [API Documentation](https://docs.rs/kwavers)
- [Examples](./examples) - 7 working examples
- [PRD](./PRD.md) - Product requirements
- [CHECKLIST](./CHECKLIST.md) - Development status

## Performance

The library is functional but not optimized. Performance profiling and optimization should be done based on specific use cases.

## Contributing

Contributions welcome, particularly for:
1. Reducing module sizes (currently 20+ files >500 lines)
2. Improving test coverage
3. Reducing warnings
4. Performance optimization

## Engineering Notes

This is a pragmatic implementation that works but has technical debt. The approach taken was to ensure functionality first, with the understanding that refactoring can be done incrementally based on actual usage patterns and requirements.

### Known Limitations
- Module size violations make some code hard to navigate
- Warning count is high but mostly benign (unused code)
- Test coverage is minimal but critical paths are tested
- Some placeholder implementations remain for future features

## License

MIT License

## Summary

**Grade: C+** - Functional library with technical debt. Suitable for research and development use with appropriate validation. Production use requires careful consideration of the technical debt and additional testing for specific use cases.

The library works and passes tests, but would benefit from architectural improvements for long-term maintainability.
