# Kwavers: Acoustic Wave Simulation Library

A high-performance Rust library for acoustic wave simulation using FDTD and PSTD methods.

## Version 3.0.0 - Clean Architecture Refactor

**Status**: Production-ready library with clean, modular architecture

### Major Improvements in v3.0

| Component | Changes | Impact |
|-----------|---------|--------|
| **Architecture** | Refactored large modules (>500 lines) into domain-based subdirectories | Improved maintainability |
| **FDTD Module** | Split 943-line module into 7 focused submodules | Better separation of concerns |
| **Naming** | Removed all adjective-based names, enforced neutral descriptive naming | Cleaner API |
| **Constants** | Replaced magic numbers with named constants | Single Source of Truth |
| **Physics** | Validated implementations against literature references | Scientific accuracy |
| **Tests** | All tests passing, examples functional | Verified correctness |

### Current Status

| Metric | Status | Notes |
|--------|--------|-------|
| **Build** | ✅ PASSING | Zero compilation errors |
| **Tests** | ✅ PASSING | All unit and integration tests pass |
| **Examples** | ✅ WORKING | All examples run successfully |
| **Warnings** | 186 | Mostly missing Debug derives (non-critical) |
| **Architecture** | ✅ CLEAN | SOLID/CUPID principles applied |

## Quick Start

```bash
# Build the library
cargo build --release

# Run tests
cargo test

# Run examples
cargo run --example physics_validation
cargo run --example wave_simulation
cargo run --example phased_array_beamforming
```

## Architecture Highlights

### Design Principles Applied
- **SOLID**: Single responsibility, proper abstractions
- **CUPID**: Composable plugins, clear interfaces
- **SSOT/SPOT**: Single source/point of truth
- **Zero-copy**: Optimized for performance where possible
- **Literature-validated**: Physics implementations cross-referenced with academic sources

### Module Organization
```
src/
├── solver/
│   ├── fdtd/           # Finite-difference time-domain solver
│   │   ├── mod.rs      # Module documentation and exports
│   │   ├── solver.rs   # Core solver implementation
│   │   ├── finite_difference.rs  # Spatial derivatives
│   │   ├── staggered_grid.rs    # Yee cell implementation
│   │   ├── subgrid.rs  # Local mesh refinement
│   │   └── ...
│   ├── pstd/           # Pseudospectral time-domain solver
│   └── ...
├── physics/
│   ├── wave_propagation/  # Wave physics
│   ├── mechanics/         # Acoustic mechanics
│   └── validation/        # Physics validation tests
└── ...
```

## Key Features

- **Multiple Solvers**: FDTD, PSTD, spectral methods
- **Physics Models**: Linear/nonlinear acoustics, thermal effects
- **Boundary Conditions**: PML, CPML, absorbing boundaries
- **Performance**: SIMD optimizations, parallel processing
- **Validation**: Comprehensive physics validation against analytical solutions

## Documentation

- Each module includes literature references
- Physics implementations cite relevant papers
- API documentation available via `cargo doc`

## Testing

```bash
# Run all tests
cargo test

# Run with verbose output
cargo test -- --nocapture

# Run specific test
cargo test test_wave_propagation
```

## Examples

Available examples demonstrate various features:
- `basic_simulation`: Simple wave propagation
- `physics_validation`: Validation against analytical solutions
- `phased_array_beamforming`: Array beamforming demonstration
- `tissue_model_example`: Biological tissue modeling
- `plugin_example`: Plugin architecture usage

## License

MIT
