# Kwavers Architecture Documentation

## Overview

Kwavers is a high-performance acoustic wave simulation library built on Rust's zero-cost abstractions and safety guarantees. The architecture emphasizes composability through plugins, scientific accuracy through literature-validated implementations, and performance through safe SIMD and optional GPU acceleration.

## Core Design Principles

### SSOT (Single Source of Truth)
- **Configuration**: Unified `Configuration` struct in `src/configuration.rs` replaces 80+ redundant config structs
- **Physical Constants**: All constants consolidated in `src/physics/constants_physics.rs`
- **Field Indices**: Unified field mapping in `src/physics/field_mapping.rs`

### SOLID Principles
- **Single Responsibility**: Each module handles one domain (e.g., `bubble_dynamics`, `thermal`, `optics`)
- **Open/Closed**: Plugin system allows extension without modifying core
- **Liskov Substitution**: All plugins implement common `Plugin` trait
- **Interface Segregation**: Traits split by concern (`CoreMedium`, `AcousticProperties`, etc.)
- **Dependency Inversion**: Core depends on abstractions, not concrete implementations

### Zero-Cost Abstractions
- Iterator-based operations for automatic vectorization
- Generic programming with monomorphization
- Compile-time optimization through const generics

## Module Structure

```
src/
├── physics/               # Domain-specific physics implementations
│   ├── bubble_dynamics/   # Rayleigh-Plesset, Keller-Miksis models
│   ├── mechanics/         # Wave propagation (acoustic, elastic)
│   ├── thermal/           # Heat transfer, bioheat equation
│   ├── optics/            # Sonoluminescence, light emission
│   ├── chemistry/         # Chemical reactions, cavitation
│   └── plugin/            # Plugin system for composability
├── solver/                # Numerical methods
│   ├── fdtd/              # Finite Difference Time Domain
│   ├── pstd/              # Pseudospectral Time Domain
│   ├── spectral_dg/       # Discontinuous Galerkin
│   └── reconstruction/    # Image reconstruction algorithms
├── medium/                # Material properties
│   ├── homogeneous/       # Uniform media
│   ├── heterogeneous/     # Spatially varying properties
│   └── anisotropic/       # Directional properties
├── boundary/              # Boundary conditions
│   ├── cpml/              # Convolutional PML (Roden & Gedney 2000)
│   └── pml/               # Standard PML
├── gpu/                   # GPU acceleration (wgpu-rs)
├── ml/                    # Machine learning integration
└── visualization/         # Real-time visualization
```

## Plugin Architecture

The plugin system enables composable physics simulations:

```rust
pub trait Plugin: Send + Sync {
    fn metadata(&self) -> &PluginMetadata;
    fn initialize(&mut self, context: &PluginContext) -> Result<()>;
    fn execute(&mut self, state: &mut PhysicsState, dt: f64) -> Result<()>;
    fn finalize(&mut self) -> Result<()>;
}
```

Plugins can be composed to build complex simulations:
- **AcousticWavePlugin**: Linear/nonlinear wave propagation
- **ElasticWavePlugin**: Solid mechanics, mode conversion
- **ThermalPlugin**: Heat transfer, thermal dose
- **CavitationPlugin**: Bubble dynamics, cavitation detection

## Performance Optimizations

### Safe SIMD
- Replaced unsafe pointer arithmetic with safe array indexing
- Compiler auto-vectorization through iterator chains
- Architecture-conditional SIMD via `cfg(target_arch)`

### Memory Management
- Zero-copy operations where possible
- Copy-on-write (CoW) for shared data
- In-place operations to minimize allocations

### Parallelization
- Rayon for data parallelism
- Thread-safe plugin execution
- Lock-free data structures where applicable

## Scientific Validation

All physics implementations are validated against literature:

### Bubble Dynamics
- **Keller-Miksis (1980)**: Compressible bubble dynamics
- **Prosperetti & Lezzi (1986)**: Thermal effects
- **Gilmore equation**: High-amplitude oscillations

### Wave Propagation
- **Westervelt equation**: Nonlinear acoustics
- **Kuznetsov equation**: Combined nonlinearity and diffusivity
- **KZK equation**: Parabolic approximation

### Boundary Conditions
- **Roden & Gedney (2000)**: CPML formulation
- **Komatitsch & Martin (2007)**: Unsplit CPML

## Testing Strategy

### Unit Tests
- Module-level functionality
- Edge cases and error conditions
- ~280 tests across all modules

### Integration Tests
- Plugin composition
- Solver convergence
- Energy conservation

### Validation Tests
- Comparison with analytical solutions
- Literature-based benchmarks
- FDA regulatory compliance (MI, TI)

## Configuration System

Hierarchical configuration with validation:

```rust
pub struct Configuration {
    pub simulation: SimulationParameters,
    pub grid: GridParameters,
    pub medium: MediumParameters,
    pub source: SourceParameters,
    pub boundary: BoundaryParameters,
    pub solver: SolverParameters,
    pub output: OutputParameters,
    pub performance: PerformanceParameters,
    pub validation: ValidationParameters,
}
```

## Error Handling

Comprehensive error types with context:
- `PhysicsError`: Domain-specific errors
- `NumericalError`: Convergence, stability issues
- `ConfigError`: Invalid parameters
- `SystemError`: Resource limitations

## Future Enhancements

### Planned Features
- Adaptive mesh refinement (AMR)
- Multi-GPU support
- Real-time streaming processing
- Cloud deployment capabilities

### Research Directions
- Machine learning for parameter optimization
- Quantum acoustic effects
- Metamaterial modeling
- Nonlocal elasticity

## Contributing

See CONTRIBUTING.md for guidelines on:
- Code style and formatting
- Testing requirements
- Documentation standards
- Review process

## References

Key papers implemented in this library:

1. Keller, J. B., & Miksis, M. (1980). Bubble oscillations of large amplitude. JASA, 68(2), 628-633.
2. Roden, J. A., & Gedney, S. D. (2000). Convolutional PML (CPML). Microwave Opt. Technol. Lett., 27(5), 334-339.
3. Westervelt, P. J. (1963). Parametric acoustic array. JASA, 35(4), 535-537.
4. Szabo, T. L. (2004). Diagnostic ultrasound imaging. Academic Press.
5. Hamilton, M. F., & Blackstock, D. T. (1998). Nonlinear acoustics. Academic Press.