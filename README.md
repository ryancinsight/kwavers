# Kwavers: Acoustic Wave Simulation Library

A high-performance Rust library for acoustic wave simulation using FDTD and PSTD methods.

## Version 2.24.0 - Production Library Build 🏗️

**Status**: Library compiles cleanly. Test suite needs modernization. Examples functional.

### Current State Assessment

| Component | Status | Details |
|-----------|--------|---------|
| **Library** | ✅ Compiles | 0 errors, 593 warnings (mostly unused vars) |
| **Examples** | ✅ Working | All examples compile and run |
| **Tests** | ❌ Outdated | 35 compilation errors - using old APIs |
| **Documentation** | ⚠️ Partial | Core docs present, needs expansion |

### Known Issues

1. **Test Suite**: Tests use deprecated APIs (e.g., `step()` instead of `update_wave()`)
2. **Warnings**: 300 unused variables, 178 missing Debug derives
3. **God Objects**: 18 files >700 lines need refactoring

### Quick Start

```bash
# Build library
cargo build --release

# Run examples (working)
cargo run --example hifu_simulation
cargo run --example physics_validation

# Tests (currently broken - API mismatch)
# cargo test  # Will fail - needs migration to new APIs
```

## Architecture

### Core Components
- **Solvers**: FDTD, PSTD, Hybrid, Spectral-DG
- **Physics**: Nonlinear acoustics (Kuznetsov, Westervelt, KZK)
- **Medium**: Heterogeneous, anisotropic, frequency-dependent
- **Boundaries**: PML, CPML, absorbing layers

### Design Principles
- Zero-cost abstractions
- Plugin-based architecture
- SOLID/CUPID compliance
- Literature-validated physics

## Performance

SIMD optimizations deliver 2-4x speedups:
- Field operations: AVX2 vectorized
- FFT operations: Optimized with RustFFT
- Memory: Zero-copy where possible

## Development Roadmap

### Immediate (v2.25)
- [ ] Migrate tests to current API
- [ ] Fix 593 warnings
- [ ] Add missing Debug derives

### Short-term (v2.26)
- [ ] Refactor god objects
- [ ] Complete documentation
- [ ] Add integration tests

### Long-term (v3.0)
- [ ] Full GPU support
- [ ] Distributed computing
- [ ] Real-time visualization

## Contributing

The library core is stable. Main areas needing work:
1. Test modernization (highest priority)
2. Warning cleanup
3. Documentation expansion

## License

MIT
