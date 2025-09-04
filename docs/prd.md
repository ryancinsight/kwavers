# Product Requirements Document (PRD) - Kwavers Physics Simulation Library

## Vision
Develop a production-ready, high-performance acoustic wave simulation library in Rust that provides scientific-grade accuracy for ultrasound, cavitation, and multi-physics simulations with GPU acceleration and cross-platform support.

## Goals

### Primary Objectives
1. **Scientific Accuracy**: Implement literature-validated physics models for acoustic wave propagation, nonlinear acoustics, cavitation dynamics, and thermal coupling
2. **High Performance**: Achieve >90% hardware utilization through SIMD optimization, GPU acceleration (wgpu-rs), and zero-copy data structures
3. **Production Quality**: Maintain >95% test coverage, <10 cyclomatic complexity per function, and comprehensive error handling
4. **Cross-Platform**: Support Linux, Windows, macOS with consistent API and performance characteristics

### Technical Requirements

#### Core Physics Models
- **Linear/Nonlinear Acoustics**: FDTD, PSTD, spectral methods with Westervelt and Kuznetsov equations
- **Cavitation Dynamics**: Rayleigh-Plesset equation with thermal effects and bubble cloud interactions  
- **Thermal Coupling**: Pennes bioheat equation with perfusion and acoustic heating
- **Elastic Wave Propagation**: Anisotropic media with mode conversion and attenuation
- **Sonoluminescence**: Chemistry-coupled bubble collapse with light emission

#### Numerical Methods
- **Finite Difference Time Domain (FDTD)**: 2nd/4th order accurate with CPML boundaries
- **Pseudospectral Time Domain (PSTD)**: k-space methods with dispersion correction
- **Spectral-DG Methods**: High-order accuracy with shock capturing for nonlinear problems
- **Adaptive Mesh Refinement**: Dynamic grid adaptation for multi-scale problems

#### Performance Targets
- **Memory Usage**: <4GB for 256³ grid simulations
- **Simulation Speed**: Real-time capability for 64³ grids on mid-range hardware
- **GPU Acceleration**: 10x+ speedup over CPU for large problems
- **SIMD Utilization**: Auto-vectorization with manual optimizations for critical paths

#### Architecture Requirements  
- **Modular Design**: SOLID/CUPID principles with <300 lines per module
- **Plugin System**: Extensible physics models and solvers
- **Zero-Cost Abstractions**: Trait-based design with compile-time optimization
- **Memory Safety**: No unsafe code except for performance-critical SIMD operations

### Functional Requirements

#### Solver Capabilities
1. **Multi-Physics Coupling**: Acoustic-thermal-chemical interactions
2. **Heterogeneous Media**: Arbitrary material property distributions
3. **Complex Geometries**: Curved boundaries with accurate reflection/transmission
4. **Source Modeling**: Point, line, plane, and phased array sources
5. **Real-Time Processing**: Live data input/output for experimental integration

#### API Design
- **Type Safety**: Compile-time units checking with uom crate
- **Error Handling**: Comprehensive Result types with structured error information
- **Configuration**: Single source of truth (SSOT) for simulation parameters
- **Interoperability**: C FFI and Python bindings for legacy integration

#### Validation & Testing
- **Analytical Benchmarks**: Exact solutions for simple geometries
- **Experimental Validation**: Published experimental data comparison
- **Cross-Validation**: k-Wave MATLAB toolbox parity
- **Performance Regression**: Automated benchmarking in CI/CD

## Success Criteria

### Quality Metrics
- **Test Coverage**: >95% line coverage with unit, integration, and property-based tests
- **Documentation**: Complete API documentation with examples for all public interfaces
- **Performance**: Benchmark suite with regression detection
- **Code Quality**: Zero clippy warnings, <10 cyclomatic complexity per function

### Delivery Milestones
1. **Phase 1**: Core FDTD solver with linear acoustics (Foundation)
2. **Phase 2**: Nonlinear acoustics and thermal coupling (Mid-development) 
3. **Phase 3**: GPU acceleration and cavitation dynamics (Advanced)
4. **Phase 4**: Production hardening and performance optimization (Production-ready)

## Non-Goals
- Real-time graphics rendering (use external visualization)
- Distributed computing (focus on single-node performance)
- Medical device certification (research/development tool)
- GUI applications (library-focused design)

## Dependencies & Constraints
- **Rust 2021 Edition**: Latest stable toolchain
- **External Libraries**: Minimal dependencies, prefer std when possible
- **Hardware Support**: x86_64 with AVX2, GPU with wgpu-rs compatibility
- **License**: MIT for maximum adoption

## Risk Assessment
- **Physics Complexity**: Mitigate through literature validation and expert review
- **Performance Requirements**: Address through profiling and iterative optimization  
- **Cross-Platform Support**: Validated through CI/CD on multiple platforms
- **API Stability**: Semantic versioning with careful breaking change management