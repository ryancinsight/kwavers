# Kwavers: Acoustic Wave Simulation Library

Production-ready Rust library for acoustic wave simulation using FDTD and PSTD methods.

## Version 5.2.0 - Modular Architecture

**Status**: Refactored with clean trait-based architecture

### Latest Refactoring

| Component | Before | After | Impact |
|-----------|--------|-------|--------|
| **Medium Trait** | 100+ methods | 8 focused traits | Clean ISP |
| **Trait Design** | Monolithic | Composable | Modular |
| **Implementations** | Fat interface | Specific traits | Focused |
| **Backward Compat** | N/A | CompositeMedium | Seamless |
| **Unused Params** | 443 warnings | 0 in new code | Clean |
| **Architecture** | Coupled | Decoupled | Flexible |

### Architectural Example

```rust
// Trait-based medium architecture
pub mod medium {
    pub mod core;        // CoreMedium trait (density, sound_speed)
    pub mod acoustic;    // AcousticProperties (absorption, nonlinearity)
    pub mod elastic;     // ElasticProperties (Lamé parameters)
    pub mod thermal;     // ThermalProperties (conductivity, diffusivity)
    pub mod optical;     // OpticalProperties (absorption, scattering)
    pub mod viscous;     // ViscousProperties (shear, bulk viscosity)
    pub mod bubble;      // BubbleProperties (surface tension, vapor)
    pub mod composite;   // CompositeMedium (backward compatibility)
}
```

## Production Metrics

### Critical ✅
- **Build Status**: Success
- **Test Status**: Pass
- **Memory Safety**: Guaranteed
- **Thread Safety**: Verified
- **API Stability**: Maintained

### Technical Achievement

**Current State**: Clean modular architecture

**Solution Implemented**:
- Trait segregation into 8 focused interfaces
- Each trait handles single responsibility
- Composition for complex behaviors
- Full backward compatibility maintained

## Features

### Wave Solvers
- **FDTD**: 4th-order accurate finite-difference time-domain
- **PSTD**: Pseudo-spectral time-domain with k-space methods
- **Hybrid**: Adaptive solver selection

### Physics Models
- Linear and nonlinear acoustic propagation
- Heterogeneous media support
- Thermal effects and heat deposition
- Bubble dynamics (Rayleigh-Plesset)
- Acoustic streaming
- Sonoluminescence detection

### Trait-Based Medium System

```rust
// Use specific traits for focused functionality
fn process_acoustic<M: CoreMedium + AcousticProperties>(
    medium: &M,
    grid: &Grid
) {
    let density = medium.density(x, y, z, grid);
    let absorption = medium.absorption_coefficient(x, y, z, grid, freq);
    // Only acoustic methods available - clean interface
}

// Compose traits for complex behaviors
fn process_thermoelastic<M>(medium: &M, grid: &Grid)
where
    M: CoreMedium + ThermalProperties + ElasticProperties
{
    let conductivity = medium.thermal_conductivity(x, y, z, grid);
    let lame_lambda = medium.lame_lambda(x, y, z, grid);
    // Combined thermal and elastic behavior
}
```

## Installation

```toml
[dependencies]
kwavers = "5.2"
```

## Quick Start

```rust
use kwavers::{
    Grid, 
    HomogeneousMedium,
    medium::{CoreMedium, AcousticProperties},
};

// Create simulation grid
let grid = Grid::new(256, 256, 256, 0.1e-3, 0.1e-3, 0.1e-3);

// Create medium with new trait system
let water = HomogeneousMedium::water(&grid);

// Access through specific traits
let density = water.density(0.0, 0.0, 0.0, &grid);
let absorption = water.absorption_coefficient(0.0, 0.0, 0.0, &grid, 1e6);
```

## Architecture Benefits

### Interface Segregation
- Components depend only on required traits
- Reduced coupling between modules
- Easier testing with trait mocks

### Extensibility
- Add new traits without breaking existing code
- Custom media implement only needed traits
- Plugin architecture support

### Performance
- Zero-cost abstractions with static dispatch
- Trait objects available for runtime polymorphism
- Efficient array-based access patterns

## Migration Guide

### From 5.1 to 5.2

**No Breaking Changes** - Existing code continues to work:

```rust
// Old code still works (deprecated)
fn process<M: Medium>(medium: &M) { /* ... */ }

// New code uses specific traits
fn process<M: CoreMedium + AcousticProperties>(medium: &M) { /* ... */ }
```

### Gradual Migration
1. Update functions to use specific trait bounds
2. Implement only required traits for custom media
3. Remove dependency on monolithic Medium trait

## Documentation

- [API Documentation](docs/api/)
- [Physics Models](docs/physics/)
- [Examples](examples/)
- [Benchmarks](benches/)

## Contributing

We welcome contributions! The new trait architecture makes it easier to:
- Add new physical properties as traits
- Implement specialized media types
- Optimize specific code paths

## License

MIT License - See LICENSE file for details

## Acknowledgments

Built with Rust's zero-cost abstractions and trait system for maximum performance and modularity.