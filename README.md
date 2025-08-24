# Kwavers: Acoustic Wave Simulation Library

A production-ready Rust library for acoustic wave simulation using FDTD and PSTD methods.

## Version 3.4.0 - Production Ready

**Status**: Fully functional, all tests pass, ready for deployment

### Current State

| Component | Status | Evidence |
|-----------|--------|----------|
| **Build** | ✅ PASSES | `cargo build --release` - 0 errors |
| **Tests** | ✅ ALL PASS | All unit, integration, and doc tests pass |
| **Examples** | ✅ WORK | All examples compile and run |
| **Benchmarks** | ✅ COMPILE | All benchmarks build successfully |
| **Documentation** | ✅ BUILDS | `cargo doc` completes without errors |

### What Works

Everything that's documented works as specified:

- **FDTD Solver** - Finite-difference time-domain acoustic simulation
- **PSTD Solver** - Pseudospectral time-domain methods  
- **Medium Properties** - Homogeneous and heterogeneous media
- **Boundary Conditions** - CPML (Convolutional PML) implementation
- **Physics State** - Field management with proper accessors
- **AMR** - Adaptive mesh refinement with octree
- **Sources** - Various transducer and array configurations

## Quick Start

```bash
# Build
cargo build --release

# Run tests
cargo test

# Run example
cargo run --example wave_simulation

# Run benchmarks
cargo bench

# Generate docs
cargo doc --open
```

## API Usage

### Basic Simulation

```rust
use kwavers::{Grid, solver::fdtd::{FdtdSolver, FdtdConfig}};

// Create computational grid
let grid = Grid::new(128, 128, 128, 1e-3, 1e-3, 1e-3);

// Configure solver
let config = FdtdConfig::default();
let mut solver = FdtdSolver::new(config, &grid)?;

// Run simulation
solver.update_pressure(&mut p, &vx, &vy, &vz, &rho, &c, dt)?;
solver.update_velocity(&mut vx, &mut vy, &mut vz, &p, &rho, dt)?;
```

### Physics State Management

```rust
use kwavers::physics::{state::PhysicsState, field_indices};

let mut state = PhysicsState::new(grid);

// Access fields
let pressure = state.get_field(field_indices::PRESSURE_IDX)?;
let temperature = state.get_field(field_indices::TEMPERATURE_IDX)?;

// Update fields
state.update_field(field_indices::PRESSURE_IDX, &new_pressure)?;
```

## Architecture

### Core Design Principles

- **SOLID** - Single responsibility, open/closed, Liskov substitution
- **Zero-copy** - Minimize allocations, use views and slices
- **Type Safety** - Leverage Rust's type system
- **Error Handling** - Result types, no hidden panics in library code

### Module Structure

```
kwavers/
├── solver/         # Numerical solvers (FDTD, PSTD, AMR)
├── physics/        # Physics models and validation
├── medium/         # Material properties
├── boundary/       # Boundary conditions
├── source/         # Acoustic sources
├── sensor/         # Measurement and detection
└── utils/          # Utilities and helpers
```

## Performance

### Benchmarks Available

- Grid operations
- CPML boundary updates
- Physics state management
- Medium property calculations
- Validation pipeline

Run with: `cargo bench`

## Testing

### Test Coverage

- **Unit Tests**: Core functionality
- **Integration Tests**: System-level behavior
- **Validation Tests**: Physics accuracy
- **Performance Tests**: Benchmarks

### Running Tests

```bash
# All tests
cargo test

# Specific module
cargo test fdtd

# With output
cargo test -- --nocapture

# Benchmarks
cargo bench
```

## Production Considerations

### Memory Safety

- No unsafe code in critical paths
- Panic-free library code (only 4 invariant checks)
- Proper error propagation with Result types

### Performance

- Zero-copy operations where possible
- SIMD support for compatible operations
- Efficient memory layout with ndarray

### Limitations

- GPU acceleration not yet implemented
- Some advanced features in development
- Performance optimizations ongoing

## Contributing

This is production-ready software. Contributions should:

1. Maintain existing test coverage
2. Follow Rust best practices
3. Include documentation
4. Pass CI/CD checks

## Version History

- v3.4.0 - Production ready, all tests pass
- v3.3.0 - Test suite restoration
- v3.2.0 - Safety improvements
- v3.1.0 - Deep implementation refactor
- v3.0.0 - Architecture overhaul

## License

MIT

## Status

**PRODUCTION READY** - This software is actively used and maintained. All documented features work as specified.
