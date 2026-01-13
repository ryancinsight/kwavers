# Kwavers 🌀

[![Version](https://img.shields.io/badge/version-2.15.0-blue.svg)](https://github.com/kwavers/kwavers)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-available-blue.svg)](https://docs.rs/kwavers)
[![Rust](https://img.shields.io/badge/rust-2021+-orange.svg)](https://www.rust-lang.org/)

**An interdisciplinary ultrasound-light physics simulation library.** Kwavers models acoustic wave propagation, cavitation dynamics, and sonoluminescence for multi-modal imaging research and physics studies.

![Physics Pipeline](https://via.placeholder.com/800x200/4A90E2/FFFFFF?text=Ultrasound+→+Cavitation+→+Sonoluminescence+→+Multi-modal+Imaging)

## 📋 Library Components

### Physics Models
- **Acoustic Wave Propagation**: Linear and nonlinear wave equations
- **Cavitation Dynamics**: Bubble physics implementations
- **Multi-Physics Coupling**: Basic acoustic-thermal interactions
- **Electromagnetic Models**: Wave propagation in various media

### Numerical Methods
- **FDTD Solver**: Finite difference time domain implementation
- **PSTD Solver**: Pseudospectral time domain method
- **PINN Support**: Physics-informed neural networks (experimental)
- **Boundary Conditions**: Various absorbing and reflecting boundaries

### Application Areas
- **Research Simulations**: Acoustic wave propagation studies
- **Imaging Algorithms**: Basic beamforming and reconstruction
- **Material Modeling**: Acoustic properties of different media
- **Signal Processing**: Filtering and analysis utilities

## 📊 Current Development Status

**Current Focus**: PINN Phase 4 - Validation & Benchmarking

The library is under active development with ongoing work to improve physics-informed neural network implementations and validation frameworks.

| Component | Status | Notes |
|-----------|--------|-------|
| **Core Library** | ✅ Compiles | Basic functionality operational |
| **Architecture** | 🟡 Evolving | Layer separation implemented, ongoing refinement |
| **Test Suite** | 🟡 Active | Tests exist, ongoing improvements and fixes |
| **Documentation** | 🟡 Developing | API docs available, guides in progress |
| **Physics Models** | 🟡 Implemented | Core models present, validation ongoing |

### Architecture Overview

Kwavers follows a layered architecture designed for scientific computing:

```
Clinical Layer     → Research applications, safety compliance
Analysis Layer     → Signal processing, imaging algorithms
Simulation Layer   → Multi-physics orchestration
Solver Layer       → Numerical methods (FDTD, PSTD, PINN)
Physics Layer      → Mathematical specifications
Domain Layer       → Problem geometry, materials, sources
Math Layer         → Linear algebra, FFT, numerical primitives
Core Layer         → Fundamental types, error handling
```

Key architectural decisions:
- **Layer Separation**: Unidirectional dependencies prevent circular imports
- **Domain Purity**: Core entities remain free of application logic
- **Trait-Based Design**: Physics specifications defined as traits for testability
- **Feature Flags**: Optional components (GPU, PINN, API) can be enabled as needed

## 🚀 Quick Start

### Installation

Add Kwavers to your `Cargo.toml`:

```toml
[dependencies]
kwavers = "2.15.0"
```

For GPU acceleration and advanced features:

```toml
[dependencies]
kwavers = { version = "2.15.0", features = ["gpu", "pinn"] }
```

### Example 1: Basic Grid Setup

```rust
use kwavers::domain::grid::Grid;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a 3D computational grid
    let grid = Grid::new(100, 100, 100, 0.001, 0.001, 0.001)?;
    println!("Created grid: {}×{}×{} points", grid.nx, grid.ny, grid.nz);
    println!("Grid spacing: {} m", grid.dx);
    Ok(())
}
```

### Example 2: Material Properties

```rust
use kwavers::domain::medium::HomogeneousMedium;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define acoustic properties for water
    let density = 1000.0;      // kg/m³
    let sound_speed = 1500.0;  // m/s
    let absorption = 0.0;      // dB/cm/MHz (water)
    let nonlinearity = 0.0;    // B/A parameter

    // Note: Medium creation requires a grid reference
    // This is a simplified example showing property values
    println!("Water properties:");
    println!("  Density: {} kg/m³", density);
    println!("  Sound speed: {} m/s", sound_speed);
    println!("  Acoustic impedance: {} MPa·s/m", density * sound_speed / 1e6);

    Ok(())
}
```

### Example 3: Basic Acoustic Calculations

```rust
// Basic acoustic property calculations
fn main() {
    // Acoustic impedance calculation: Z = ρc
    let density_water = 1000.0;     // kg/m³
    let speed_water = 1500.0;       // m/s
    let impedance_water = density_water * speed_water; // Pa·s/m

    println!("Water acoustic impedance: {:.0} Pa·s/m", impedance_water);

    // Reflection coefficient: R = (Z2 - Z1)/(Z2 + Z1)
    let density_air = 1.2;          // kg/m³
    let speed_air = 343.0;          // m/s
    let impedance_air = density_air * speed_air;

    let reflection_coeff = (impedance_air - impedance_water) /
                          (impedance_air + impedance_water);

    println!("Air-water reflection coefficient: {:.4}", reflection_coeff);
}
```

## 📚 Documentation

### 📖 Documentation

- **[API Reference](https://docs.rs/kwavers)** - Generated Rust documentation
- **[Examples](examples/)** - Basic usage examples
- **Development Docs** - See `docs/` directory for planning and design documents

### 🎯 Basic Usage

See the `examples/` directory for basic usage patterns:

```bash
# List available examples
cargo run --example

# Run a basic example (if available)
cargo run --example basic_simulation
```

**Basic Test**: Check compilation
```bash
cargo check
```

### 🏗️ Architecture

Kwavers is structured with layered separation intended to support scientific computing workflows:

```
Clinical Applications    → Research use cases, safety monitoring
Analysis & Imaging       → Signal processing, reconstruction algorithms
Simulation Orchestration → Multi-physics coupling, time integration
Numerical Solvers        → FDTD, PSTD, PINN, spectral methods
Physics Specifications   → Wave equations, constitutive relations
Problem Domain           → Geometry, materials, boundary conditions
Mathematical Primitives  → Linear algebra, FFT, interpolation
Core Infrastructure      → Error handling, memory management
```

The architecture aims to separate concerns while maintaining flexibility for different research applications. Layer boundaries help organize code but are not strictly enforced in all areas during active development.


## 🤝 Contributing

This is an active research project under development. Contributions are welcome but the codebase is evolving.

### 📝 Development Notes

- **Architecture**: Layered design with separation between physics, solvers, and applications
- **Testing**: Unit tests exist for core functionality
- **Documentation**: API documentation available, guides are developing
- **Code Style**: Standard Rust conventions

### 🚀 Getting Started

1. Check the current sprint status in `checklist.md`
2. Review `backlog.md` for planned work
3. Run tests: `cargo test`
4. Check compilation: `cargo check`

### 📊 Current Development

The project uses sprint-based development with focus on:
- Physics model implementation and validation
- Architecture refinement
- Test coverage improvement
- Documentation development

## 📄 License

Kwavers is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

## 📚 References

### Key Physics Texts
- Hamilton, M.F. & Blackstock, D.T. - Nonlinear Acoustics
- Szabo, T.L. - Diagnostic Ultrasound Imaging
- Duck, F.A. - Physical Properties of Tissues

### Numerical Methods
- Yee, K.S. (1966) - FDTD method
- Liu, Q.H. (1997) - PSTD method
- Hesthaven, J.S. (2007) - DG methods

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/kwavers/kwavers/issues)
- **Discussions**: [GitHub Discussions](https://github.com/kwavers/kwavers/discussions)
- **Documentation**: [docs.rs/kwavers](https://docs.rs/kwavers)

---

**A research library for acoustic and optical physics simulations.**
