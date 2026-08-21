# Kwavers Examples 🧪

This directory contains comprehensive examples demonstrating the interdisciplinary ultrasound-light physics simulation capabilities of the Kwavers library. Examples are organized by physics domain, complexity level, and application area.

## 📋 Example Categories

### 🔬 **Basic Simulations**
| Example | Description | Features |
|---------|-------------|----------|
| [`basic_simulation.rs`](basic_simulation.rs) | Simple acoustic wave propagation | FDTD solver, basic setup |
| [`minimal_demo.rs`](minimal_demo.rs) | Minimal working example | Core concepts, validation |

### 🩺 **Ultrasound Imaging**
| Example | Description | Features |
|---------|-------------|----------|
| [`advanced_ultrasound_imaging.rs`](advanced_ultrasound_imaging.rs) | Advanced imaging techniques | Synthetic aperture, plane wave, coded excitation |
| [`phased_array_beamforming.rs`](phased_array_beamforming.rs) | Phased array beamforming | Delay-and-sum, apodization |
| [`real_time_3d_beamforming.rs`](real_time_3d_beamforming.rs) | Real-time 3D beamforming | GPU acceleration, clinical workflows |

### 💥 **Therapeutic Ultrasound & Nonlinear Physics**
| Example | Description | Features |
|---------|-------------|----------|
| [`focused_ultrasound_water_tank.rs`](focused_ultrasound_water_tank.rs) | Focused-ultrasound solver comparison | Through-plane phased aperture, FDTD, PSTD, and DG diagnostics |
| [`multiphysics_sonoluminescence.rs`](multiphysics_sonoluminescence.rs) | Bounded multi-domain PINN training | Cavitation, sonoluminescence, electromagnetic domains (`pinn`) |
| [`single_bubble_sonoluminescence.rs`](single_bubble_sonoluminescence.rs) | Sonoluminescence modeling | Bubble dynamics, typed dimensioned emission |

### 🎯 **Advanced Applications**
| Example | Description | Features |
|---------|-------------|----------|
| [`photoacoustic_imaging.rs`](photoacoustic_imaging.rs) | Photoacoustic imaging | Light absorption, acoustic detection |
| [`elastography_simulation.rs`](elastography_simulation.rs) | Tissue elastography | Shear wave imaging, stiffness mapping |
| [`seismic_imaging_demo.rs`](seismic_imaging_demo.rs) | Seismic imaging | Full-waveform inversion, migration |

### 🤖 **AI & Machine Learning**
| Example | Description | Features | Requires |
|---------|-------------|----------|----------|
| [`pinn_2d_wave_equation.rs`](pinn_2d_wave_equation.rs) | PINN wave equation | Neural PDE solving | `pinn` feature |
| [`pinn_advanced_physics.rs`](pinn_advanced_physics.rs) | Advanced PINN physics | Multi-physics coupling | `pinn` feature |
| [`pinn_gpu_training.rs`](pinn_gpu_training.rs) | GPU-accelerated PINN | Real-time training | `pinn` + `gpu` |

### 🧬 **Medical & Biological**
| Example | Description | Features |
|---------|-------------|----------|
| [`heterogeneous_power_law_attenuation.rs`](heterogeneous_power_law_attenuation.rs) | Heterogeneous tissue attenuation | Spatially varying coefficient and exponent, analytical recovery oracles |
| [`skull_ct_phase_correction.rs`](skull_ct_phase_correction.rs) | Skull CT phase correction | RITK DICOM loading, 1024-element hemispherical array, three-plane phase image |
| [`swe_liver_fibrosis.rs`](swe_liver_fibrosis.rs) | Liver fibrosis assessment | SWE imaging, fibrosis staging |
| [`electromagnetic_simulation.rs`](electromagnetic_simulation.rs) | EM wave propagation | Maxwell equations, coupling |

### 🔬 **Research & Validation**
| Example | Description | Features |
|---------|-------------|----------|
| [`physics_validation.rs`](physics_validation.rs) | Physics validation suite | Literature validation, error analysis |
| [`literature_validation_safe.rs`](literature_validation_safe.rs) | Analytical validation | Green's functions, diffraction |
| [`pstd_fdtd_comparison.rs`](pstd_fdtd_comparison.rs) | Solver discrepancy diagnostics | FDTD, k-space FDTD, PSTD field metrics |
| [`dg_advection_diagnostics.rs`](dg_advection_diagnostics.rs) | DG scalar/acoustic readiness diagnostics | Periodic advection, one-way and bidirectional acoustic characteristics, mass, phase, amplitude metrics |
| [`dg_acoustic_1d_diagnostics.rs`](dg_acoustic_1d_diagnostics.rs) | Native DG acoustic diagnostics | Coupled pressure/velocity DG, analytical standing wave, characteristic cross-check, embedded FDTD/PSTD/DG Gaussian matrix |
| [`dg_acoustic_comparison_plot.rs`](dg_acoustic_comparison_plot.rs) | Acoustic solver comparison plots | PNG and CSV plots for native-grid, common-grid, and uniform-grid exact, DG, FDTD, k-space FDTD, and PSTD Gaussian pressure/error traces |
| [`dg_acoustic_convergence_plot.rs`](dg_acoustic_convergence_plot.rs) | DG acoustic p-refinement plots | PNG and CSV convergence diagnostics with nodal and common-quadrature DG Gaussian pressure errors |
| [`dg_acoustic_timestep_sweep.rs`](dg_acoustic_timestep_sweep.rs) | Acoustic timestep-refinement plots | PNG and CSV timestep sweep for DG, FDTD, k-space FDTD, and PSTD Gaussian pressure errors |
| [`theorem_validation_demo.rs`](theorem_validation_demo.rs) | Mathematical theorems | Formal verification, proofs |
| [`validate_2d_pinn.rs`](validate_2d_pinn.rs) | PINN validation | 2D wave equation, convergence |

### ⚡ **Performance & Benchmarks**
| Example | Description | Features |
|---------|-------------|----------|
| [`performance_validation.rs`](performance_validation.rs) | Performance analysis | Timing, scaling, optimization |
| [`safe_vectorization_benchmarks.rs`](safe_vectorization_benchmarks.rs) | Vectorization benchmarks | SIMD performance, safety |

### 🌊 **Specialized Physics**
| Example | Description | Features |
|---------|-------------|----------|
| [`electromagnetic_simulation.rs`](electromagnetic_simulation.rs) | EM wave simulation | Maxwell equations, antennas |
| [`adaptive_beamforming_refactored.rs`](adaptive_beamforming_refactored.rs) | Adaptive beamforming | MVDR, MUSIC, LCMV algorithms |
| [`comprehensive_pinn_demo.rs`](comprehensive_pinn_demo.rs) | Full PINN ecosystem | Training, inference, validation |

## 🚀 Running Examples

### Prerequisites
```bash
# Confirm the repository-pinned toolchain selected by rustup
rustup show active-toolchain

# Build with all features for maximum compatibility
cargo build --release --all-features
```

### Basic Usage
```bash
# Run a basic simulation
cargo run --example basic_simulation

# Run with specific features
cargo run --example multiphysics_sonoluminescence --features pinn

# Run performance benchmarks
cargo run --example safe_vectorization_benchmarks --release
```

### Feature Requirements

Some examples require specific feature flags:

| Feature | Examples | Description |
|---------|----------|-------------|
| `pinn` | AI/ML examples | Physics-Informed Neural Networks |
| `gpu` | GPU-accelerated examples | WGPU-based parallel computing |
| `full` | All examples | Complete feature set |

### Example: Complete Interdisciplinary Simulation

```bash
# Enable all features for maximum capability
cargo run --example multiphysics_sonoluminescence --features full
```

This trains the registered domains with a bounded demo workload and reports
the solver's returned final losses and convergence flags. It is a training
demonstration, not a validation claim for a physical experiment.

## 🧪 Testing Examples

```bash
# Test that all examples compile
cargo check --examples

# Run example-specific tests
cargo test --example <name>

# Compile PINN examples
cargo check -p kwavers --examples --features pinn
```

## 📚 Documentation

Each example includes:
- **Physics explanation** - Underlying mathematical models
- **Literature references** - Academic validation sources
- **Usage instructions** - How to run and modify
- **Performance notes** - Optimization and scaling information

## 🎯 Learning Path

**Beginners**: Start with `minimal_demo.rs` → `basic_simulation.rs`

**Ultrasound Imaging**: `phased_array_beamforming.rs` → `advanced_ultrasound_imaging.rs` → `real_time_3d_beamforming.rs`

**Multi-Physics**: `focused_ultrasound_water_tank.rs` → `multiphysics_sonoluminescence.rs` → AI examples

**Research**: `physics_validation.rs` → `literature_validation_safe.rs` → `theorem_validation_demo.rs`

## 🔬 Validation & Benchmarks

Examples include comprehensive validation against:
- **Analytical solutions** - Green's functions, diffraction theory
- **Literature benchmarks** - Published research results
- **Performance metrics** - Timing, accuracy, scaling analysis
- **Reference compatibility** - MATLAB toolbox validation
