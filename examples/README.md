# Kwavers Examples 🧪

This directory contains comprehensive examples demonstrating the interdisciplinary ultrasound-light physics simulation capabilities of the Kwavers library. Examples are organized by physics domain, complexity level, and application area.

## 📋 Example Categories

### 🔬 **Basic Simulations**
| Example | Description | Features |
|---------|-------------|----------|
| [`basic_simulation.rs`](basic_simulation.rs) | Simple acoustic wave propagation | FDTD solver, basic setup |
| [`minimal_demo.rs`](minimal_demo.rs) | Minimal working example | Core concepts, validation |
| [`wave_simulation.rs`](wave_simulation.rs) | Wave equation fundamentals | Linear/nonlinear propagation |

### 🩺 **Ultrasound Imaging**
| Example | Description | Features |
|---------|-------------|----------|
| [`advanced_ultrasound_imaging.rs`](advanced_ultrasound_imaging.rs) | Advanced imaging techniques | Synthetic aperture, plane wave, coded excitation |
| [`phased_array_beamforming.rs`](phased_array_beamforming.rs) | Phased array beamforming | Delay-and-sum, apodization |
| [`real_time_3d_beamforming.rs`](real_time_3d_beamforming.rs) | Real-time 3D beamforming | GPU acceleration, clinical workflows |

### 💥 **Cavitation & Nonlinear Physics**
| Example | Description | Features |
|---------|-------------|----------|
| [`hifu_tumor_ablation.rs`](hifu_tumor_ablation.rs) | HIFU therapy simulation | Thermal ablation, bioheat transfer |
| [`multiphysics_sonoluminescence.rs`](multiphysics_sonoluminescence.rs) | **Complete interdisciplinary** | Ultrasound → cavitation → light |
| [`sonoluminescence_simulation.rs`](sonoluminescence_simulation.rs) | Sonoluminescence modeling | Bubble dynamics, light emission |

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
| [`ai_integration_simple_test.rs`](ai_integration_simple_test.rs) | AI beamforming integration | Clinical AI workflows | `pinn` feature |

### 🧬 **Medical & Biological**
| Example | Description | Features |
|---------|-------------|----------|
| [`tissue_model_example.rs`](tissue_model_example.rs) | Realistic tissue modeling | Heterogeneous media, attenuation |
| [`swe_liver_fibrosis.rs`](swe_liver_fibrosis.rs) | Liver fibrosis assessment | SWE imaging, fibrosis staging |
| [`electromagnetic_simulation.rs`](electromagnetic_simulation.rs) | EM wave propagation | Maxwell equations, coupling |

### 🔬 **Research & Validation**
| Example | Description | Features |
|---------|-------------|----------|
| [`physics_validation.rs`](physics_validation.rs) | Physics validation suite | Literature validation, error analysis |
| [`literature_validation_safe.rs`](literature_validation_safe.rs) | Analytical validation | Green's functions, diffraction |
| [`theorem_validation_demo.rs`](theorem_validation_demo.rs) | Mathematical theorems | Formal verification, proofs |
| [`validate_2d_pinn.rs`](validate_2d_pinn.rs) | PINN validation | 2D wave equation, convergence |

### ⚡ **Performance & Benchmarks**
| Example | Description | Features |
|---------|-------------|----------|
| [`kwave_benchmarks.rs`](kwave_benchmarks.rs) | k-Wave compatibility | Performance comparison, migration |
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
# Ensure Rust toolchain is up to date
rustup update stable

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

This demonstrates the complete physics pipeline:
1. **Ultrasound excitation** → Acoustic wave propagation
2. **Cavitation physics** → Bubble oscillation and collapse
3. **Sonoluminescence** → Light emission from collapse
4. **Multi-modal detection** → Combined ultrasound + optical imaging

## 🧪 Testing Examples

```bash
# Test that all examples compile
cargo check --examples

# Run example-specific tests
cargo test --example <name>

# Test with specific features
cargo test --example ai_integration_test --features pinn
```

## 📚 Documentation

Each example includes:
- **Physics explanation** - Underlying mathematical models
- **Literature references** - Academic validation sources
- **Usage instructions** - How to run and modify
- **Performance notes** - Optimization and scaling information

## 🎯 Learning Path

**Beginners**: Start with `basic_simulation.rs` → `minimal_demo.rs` → `wave_simulation.rs`

**Ultrasound Imaging**: `phased_array_beamforming.rs` → `advanced_ultrasound_imaging.rs` → `real_time_3d_beamforming.rs`

**Multi-Physics**: `hifu_tumor_ablation.rs` → `multiphysics_sonoluminescence.rs` → AI examples

**Research**: `physics_validation.rs` → `literature_validation_safe.rs` → `theorem_validation_demo.rs`

## 🔬 Validation & Benchmarks

Examples include comprehensive validation against:
- **Analytical solutions** - Green's functions, diffraction theory
- **Literature benchmarks** - Published research results
- **Performance metrics** - Timing, accuracy, scaling analysis
- **k-Wave compatibility** - MATLAB toolbox validation