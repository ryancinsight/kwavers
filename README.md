# Kwavers - Advanced Ultrasound Simulation Toolbox

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://github.com/kwavers/kwavers/workflows/Rust/badge.svg)](https://github.com/kwavers/kwavers/actions)
[![Crates.io](https://img.shields.io/crates/v/kwavers)](https://crates.io/crates/kwavers)
[![Documentation](https://docs.rs/kwavers/badge.svg)](https://docs.rs/kwavers)

A modern, high-performance, open-source computational toolbox for simulating ultrasound wave propagation and its interactions with complex biological media, with advanced physics capabilities including cavitation dynamics, sonoluminescence, and light-tissue interactions.

## 🚀 Features

### Core Physics Capabilities
- **Nonlinear Wave Propagation**: Westervelt equation, KZK equation, k-space pseudospectral methods
- **Elastic Wave Propagation**: Linear isotropic media, P and S waves
- **Cavitation Dynamics**: Multi-bubble interactions, bubble cloud effects, acoustic emissions
- **Sonoluminescence**: Light emission from collapsing bubbles, spectral analysis
- **Light-Tissue Interactions**: Photothermal effects, light diffusion, polarization
- **Thermal Effects**: Bioheat equation, heat diffusion, perfusion, metabolic heat
- **Acoustic Streaming**: Fluid motion due to acoustic radiation forces
- **Chemical Reaction Kinetics**: Multi-species reactions, temperature-dependent rates

### Advanced Design Principles
This project implements comprehensive software engineering principles:

- **SOLID**: Single responsibility, open/closed, Liskov substitution, interface segregation, dependency inversion
- **CUPID**: Composable, Unix-like, Predictable, Idiomatic, Domain-focused
- **GRASP**: Information expert, creator, controller, low coupling, high cohesion
- **ACID**: Atomicity, consistency, isolation, durability for simulation operations
- **DRY**: Don't repeat yourself - shared patterns and utilities
- **KISS**: Keep it simple, stupid - clear, straightforward interfaces
- **YAGNI**: You aren't gonna need it - only implement necessary features
- **SSOT**: Single source of truth for all data and configuration
- **CCP**: Common closure principle for related functionality
- **CRP**: Common reuse principle for shared components
- **ADP**: Acyclic dependency principle for clean architecture

### Performance Optimizations
- **SIMD-friendly data layouts** for vectorized operations
- **Parallel processing** with Rayon for multi-core utilization
- **Memory-efficient algorithms** with minimal allocations
- **Optimized FFT operations** using rustfft
- **Cache-friendly memory access patterns**
- **Performance monitoring** and profiling tools

## 📦 Installation

### Prerequisites
- Rust 1.70+ (stable)
- Cargo package manager
- 4GB+ RAM (8GB+ recommended for large simulations)
- Multi-core CPU (8+ cores recommended)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/kwavers/kwavers.git
cd kwavers

# Build with basic physics features
cargo build --release

# Build with advanced physics features
cargo build --release --features advanced-physics

# Run tests
cargo test

# Run benchmarks
cargo bench
```

### Cargo.toml Dependencies
```toml
[dependencies]
kwavers = { version = "0.1.0", features = ["advanced-physics"] }
```

## 🎯 Usage Examples

### Basic Acoustic Wave Simulation
```rust
use kwavers::*;

fn main() -> KwaversResult<()> {
    // Initialize logging
    init_logging()?;
    
    // Create default configuration
    let config = create_default_config();
    
    // Run simulation
    run_advanced_simulation(config)?;
    
    Ok(())
}
```

### Advanced Sonoluminescence Simulation
```rust
use kwavers::*;

fn main() -> KwaversResult<()> {
    // Create advanced configuration
    let config = ConfigBuilder::new()
        .with_string("simulation_name".to_string(), "sonoluminescence".to_string())
        .with_float("frequency".to_string(), 2e6)  // 2 MHz
        .with_float("amplitude".to_string(), 3e6)  // 3 MPa
        .with_integer("grid_nx".to_string(), 160)
        .with_integer("grid_ny".to_string(), 120)
        .with_integer("grid_nz".to_string(), 120)
        .with_float("time_duration".to_string(), 2e-3)  // 2 ms
        .build();
    
    // Run advanced simulation with cavitation and light physics
    run_advanced_simulation(config)?;
    
    Ok(())
}
```

### Custom Physics Pipeline
```rust
use kwavers::*;

fn main() -> KwaversResult<()> {
    // Create physics pipeline
    let mut pipeline = PhysicsPipeline::new();
    
    // Add custom components
    pipeline.add_component(Box::new(
        physics::composable::AcousticWaveComponent::new("acoustic".to_string())
    ))?;
    
    pipeline.add_component(Box::new(
        physics::composable::ThermalDiffusionComponent::new("thermal".to_string())
    ))?;
    
    // Add custom cavitation component
    pipeline.add_component(Box::new(
        AdvancedCavitationComponent::new("cavitation".to_string())
    ))?;
    
    // Run simulation with custom pipeline
    // ... implementation details
    
    Ok(())
}
```

### Configuration Management
```rust
use kwavers::*;

fn main() -> KwaversResult<()> {
    // Create configuration manager
    let config_manager = ConfigManager::new();
    
    // Load configuration from file
    config_manager.load_config(
        "simulation_config".to_string(),
        "config/simulation.toml".into()
    )?;
    
    // Validate configuration
    let config = config_manager.get_config("simulation_config").unwrap();
    let validation_result = validate_simulation_config(&config)?;
    
    if !validation_result.is_valid {
        println!("Configuration validation failed: {}", validation_result.summary());
        return Ok(());
    }
    
    // Run simulation with validated configuration
    run_advanced_simulation(config)?;
    
    Ok(())
}
```

## 🏗️ Architecture

### Core Components

#### Physics System
- **Composable Physics Pipeline**: Modular physics components that can be combined
- **Field Management**: Efficient handling of pressure, temperature, light, and cavitation fields
- **Boundary Conditions**: PML (Perfectly Matched Layers) for absorbing boundaries
- **Source Modeling**: Complex transducer arrays with beamforming

#### Configuration System
- **Hierarchical Configuration**: Nested configuration structures
- **Schema Validation**: Type-safe configuration validation
- **Configuration Management**: Global configuration state management
- **File I/O**: TOML configuration file support

#### Validation System
- **Composable Validators**: Reusable validation rules
- **Validation Pipelines**: Complex validation workflows
- **Performance Tracking**: Validation performance metrics
- **Error Context**: Detailed error information and recovery strategies

#### Error Handling
- **Comprehensive Error Types**: Domain-specific error categories
- **Error Context**: Rich error information with stack traces
- **Recovery Strategies**: Automatic error recovery mechanisms
- **Error Severity**: Prioritized error handling

### Design Patterns

#### SOLID Principles
- **Single Responsibility**: Each component has one clear purpose
- **Open/Closed**: Extensible without modification
- **Liskov Substitution**: Interchangeable components
- **Interface Segregation**: Focused, minimal interfaces
- **Dependency Inversion**: Depend on abstractions, not concretions

#### CUPID Principles
- **Composable**: Components can be combined flexibly
- **Unix-like**: Each tool does one thing well
- **Predictable**: Same inputs always produce same outputs
- **Idiomatic**: Uses Rust's type system effectively
- **Domain-focused**: Clear separation of physics domains

#### GRASP Patterns
- **Information Expert**: Components know about their own data
- **Creator**: Components responsible for creating related objects
- **Controller**: Centralized control flow management
- **Low Coupling**: Minimal dependencies between components
- **High Cohesion**: Related functionality grouped together

## 📊 Performance

### Benchmarks
```bash
# Run performance benchmarks
cargo bench

# Run specific benchmark suites
cargo bench physics_benchmarks
cargo bench grid_benchmarks
cargo bench validation_benchmarks
```

### Performance Targets
- **Grid Size**: Support up to 1000³ grid points
- **Time Steps**: 10,000+ time steps per simulation
- **Memory Usage**: < 8GB for typical simulations
- **Execution Time**: < 1 hour for standard simulations
- **Parallel Efficiency**: > 80% on 8+ cores

### Optimization Features
- **SIMD Operations**: Vectorized numerical computations
- **Memory Pooling**: Efficient memory allocation patterns
- **Cache Optimization**: Cache-friendly data layouts
- **Parallel Processing**: Multi-threaded execution
- **Lazy Evaluation**: On-demand computation

## 🔧 Configuration

### Configuration File Format (TOML)
```toml
[simulation]
name = "advanced_cavitation_simulation"
version = "1.0.0"
enable_visualization = true

[physics]
frequency = 2e6  # 2 MHz
amplitude = 3e6  # 3 MPa
enable_cavitation = true
enable_sonoluminescence = true
enable_thermal_effects = true

[grid]
nx = 160
ny = 120
nz = 120
dx = 1e-3  # 1 mm
dy = 1e-3
dz = 1e-3

[time]
duration = 2e-3  # 2 ms
cfl_number = 0.3

[output]
directory = "simulation_output"
snapshot_interval = 10
save_raw_data = true
enable_visualization = true
```

### Environment Variables
```bash
# Enable debug logging
export RUST_LOG=debug

# Set number of threads
export RAYON_NUM_THREADS=8

# Enable performance profiling
export KWAVERS_PROFILE=true
```

## 🧪 Testing

### Test Suite
```bash
# Run all tests
cargo test

# Run specific test modules
cargo test physics
cargo test validation
cargo test config

# Run tests with output
cargo test -- --nocapture

# Run integration tests
cargo test --test integration_tests
```

### Test Coverage
```bash
# Install cargo-tarpaulin
cargo install cargo-tarpaulin

# Generate coverage report
cargo tarpaulin --out Html
```

## 📈 Benchmarks

### Performance Comparison
| Feature | Grid Size | Time Steps | Memory | Time | Cores |
|---------|-----------|------------|--------|------|-------|
| Basic Acoustic | 100³ | 1,000 | 100MB | 30s | 1 |
| Advanced Physics | 200³ | 2,000 | 500MB | 2m | 4 |
| Full Simulation | 400³ | 5,000 | 2GB | 15m | 8 |
| Large Scale | 800³ | 10,000 | 8GB | 1h | 16 |

### Scalability
- **Linear scaling** with grid size
- **Near-linear scaling** with number of cores
- **Memory-efficient** algorithms
- **Cache-friendly** data access patterns

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone https://github.com/kwavers/kwavers.git
cd kwavers

# Install development dependencies
cargo install cargo-watch
cargo install cargo-tarpaulin
cargo install cargo-audit

# Set up pre-commit hooks
cargo install cargo-husky
cargo husky install
```

### Code Style
- Follow Rust coding conventions
- Use `cargo fmt` for formatting
- Use `cargo clippy` for linting
- Write comprehensive tests
- Document all public APIs

### Pull Request Process
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Update documentation
6. Run the full test suite
7. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **k-Wave Toolbox**: Inspiration and reference implementation
- **Rust Community**: Excellent language and ecosystem
- **Scientific Computing Community**: Research and validation
- **Open Source Contributors**: Code, documentation, and feedback

## 📚 Documentation

- [API Documentation](https://docs.rs/kwavers)
- [User Guide](docs/user-guide.md)
- [Developer Guide](docs/developer-guide.md)
- [Physics Reference](docs/physics-reference.md)
- [Performance Guide](docs/performance-guide.md)

## 🐛 Issues and Support

- **Bug Reports**: [GitHub Issues](https://github.com/kwavers/kwavers/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/kwavers/kwavers/discussions)
- **Documentation**: [GitHub Wiki](https://github.com/kwavers/kwavers/wiki)
- **Community**: [Discord Server](https://discord.gg/kwavers)

## 🔮 Roadmap

### Short Term (3-6 months)
- [x] GPU acceleration support - **Phase 9 Architecture Complete** ✅
- [ ] CUDA/WebGPU kernel implementation 🚧
- [ ] Python bindings
- [ ] Advanced visualization features
- [ ] More physics models
- [ ] Performance optimizations

### Medium Term (6-12 months)
- [ ] Cloud deployment support
- [ ] Real-time simulation capabilities
- [ ] Machine learning integration
- [ ] Advanced post-processing tools
- [ ] Multi-physics coupling

### Long Term (1+ years)
- [ ] Web-based interface
- [ ] Collaborative simulation features
- [ ] Advanced AI/ML capabilities
- [ ] Industry-specific modules
- [ ] Commercial support options

---

**Kwavers** - Advancing ultrasound simulation technology with modern software engineering principles and cutting-edge physics capabilities.
