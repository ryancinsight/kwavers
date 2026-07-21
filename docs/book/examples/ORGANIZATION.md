# Example Organization

This document provides an overview of all kwavers examples and their organization for the book.

## Complete Example Index

### Core Simulation Examples (Book Docs Available)

| Example | Chapter | Description |
|---------|---------|-------------|
| [`basic_simulation`](../examples/basic_simulation.md) | Wave Physics Fundamentals | Simple acoustic wave propagation |
| [`minimal_demo`](../examples/minimal_demo.md) | Wave Physics Fundamentals | Minimal working example |
| [`pstd_fdtd_comparison`](../examples/pstd_fdtd_comparison.md) | Numerical Methods | FDTD vs PSTD comparison |
| [`dg_acoustic_1d`](../examples/dg_acoustic_1d.md) | Numerical Methods | Discontinuous Galerkin acoustic diagnostics |
| [`single_bubble_sonoluminescence`](../examples/single_bubble_sonoluminescence.md) | Cavitation | Single bubble sonoluminescence |
| [`multiphysics_sonoluminescence`](../examples/multiphysics_sonoluminescence.md) | Cavitation | Complete interdisciplinary simulation |
| [`spatially_varying_attenuation`](../examples/spatially_varying_attenuation.md) | Media | Spatially varying attenuation |
| [`focused_ultrasound_water_tank`](../examples/focused_ultrasound_water_tank.md) | Sources | Focused ultrasound water tank |
| [`phantom_builder_demo`](../examples/phantom_builder_demo.md) | Sources | Phantom builder demonstration |
| [`phased_array_beamforming`](../examples/phased_array_beamforming.md) | Beamforming | Phased array beamforming |
| [`adaptive_beamforming`](../examples/adaptive_beamforming.md) | Beamforming | Adaptive beamforming |
| [`real_time_3d_beamforming`](../examples/real_time_3d_beamforming.md) | Beamforming | Real-time 3D beamforming |
| [`doppler_velocity_estimation`](../examples/doppler_velocity_estimation.md) | Sensors | Doppler velocity estimation |
| [`advanced_ultrasound_imaging`](../examples/advanced_ultrasound_imaging.md) | Imaging | Advanced ultrasound imaging |
| [`photoacoustic_imaging`](../examples/photoacoustic_imaging.md) | Photoacoustics | Photoacoustic imaging |
| [`photoacoustic_blood_oxygenation`](../examples/photoacoustic_blood_oxygenation.md) | Photoacoustics | Photoacoustic blood oxygenation |
| [`elastography_simulation`](../examples/elastography_simulation.md) | Elastography | Tissue elastography simulation |
| [`swe_liver_fibrosis`](../examples/swe_liver_fibrosis.md) | Elastography | SWE liver fibrosis assessment |
| [`comprehensive_clinical_workflow`](../examples/comprehensive_clinical_workflow.md) | Therapy | Comprehensive clinical workflow |
| [`brain_theranostic_monitor`](../examples/brain_theranostic_monitor.md) | Theranostics | Brain theranostic monitor |
| [`liver_theranostic_reconstruction`](../examples/liver_theranostic_reconstruction.md) | Theranostics | Liver theranostic reconstruction |
| [`hybrid_lesion_monitor`](../examples/hybrid_lesion_monitor.md) | Theranostics | Hybrid lesion monitor |
| [`skull_ct_phase_correction`](../examples/skull_ct_phase_correction.md) | Transcranial | Skull CT phase correction |
| [`transcranial_fwi`](../examples/transcranial_fwi.md) | Transcranial | Transcranial full-waveform inversion |
| [`pinn_training_convergence`](../examples/pinn_training_convergence.md) | PINN | PINN training convergence |
| [`pinn_2d_wave_equation`](../examples/pinn_2d_wave_equation.md) | PINN | PINN 2D wave equation |
| [`pinn_advanced_physics`](../examples/pinn_advanced_physics.md) | PINN | PINN advanced physics |
| [`transfer_learning_pinn`](../examples/transfer_learning_pinn.md) | PINN | Transfer learning PINN |
| [`literature_validation`](../examples/literature_validation.md) | Validation | Literature validation |
| [`physics_validation`](../examples/physics_validation.md) | Validation | Physics validation |
| [`performance_validation`](../examples/performance_validation.md) | Validation | Performance validation |
| [`theorem_validation_demo`](../examples/theorem_validation_demo.md) | Validation | Theorem validation demo |
| [`safe_vectorization_benchmarks`](../examples/safe_vectorization_benchmarks.md) | Performance | Safe vectorization benchmarks |
| [`transcranial_ct_mri`](../examples/transcranial_ct_mri.md) | Transcranial | Transcranial CT/MRI reconstruction |

### Architecture and Utility Examples (No Book Docs)

These examples demonstrate architecture, refactoring, or utility functions rather than physics simulations.

| Example | Description |
|---------|-------------|
| `adaptive_beamforming_refactored` | Adaptive beamforming architecture refactoring demonstration |
| `boundary_smoothing` | Boundary smoothing implementation details |
| `comprehensive_pinn_demo` | Full PINN ecosystem demonstration |
| `dg_acoustic_timestep_sweep` | DG acoustic timestep refinement plots |
| `diagnostics_3d` | 3D diagnostics utilities |
| `electromagnetic_simulation` | EM wave propagation (Maxwell equations) |
| `dg_acoustic_common` | Common DG acoustic utilities |
| `dg_acoustic_comparison_plot` | Acoustic solver comparison plots |
| `dg_acoustic_convergence_plot` | DG convergence plots |
| `dg_advection_diagnostics` | DG advection diagnostics |
| `field_surrogate_demo` | Field surrogate modeling |
| `lines` | Line drawing utilities |
| `literature_validation_safe` | Analytical validation (safe/Green's functions) |
| `metrics` | Metrics collection utilities |
| `mod` | Module utilities |
| `monte_carlo_validation` | Monte Carlo validation methods |
| `nl_swe_convergence_validation` | Nonlinear SWE convergence validation |
| `physics` | Physics utilities |
| `plot` | Plotting utilities |
| `plugin_example` | Plugin architecture example |
| `sampling` | Sampling utilities |
| `seismic_imaging_3d_demo` | Seismic imaging 3D |
| `seismic_imaging_demo` | Seismic imaging demo |
| `simulation` | Simulation utilities |
| `swe_3d_liver_fibrosis` | SWE 3D liver fibrosis |
| `tau_sweep` | Tau parameter sweeps |
| `transcranial_ct_mri_reconstruction` | Transcranial CT/MRI reconstruction variant |
| `validate_2d_pinn` | 2D PINN validation |
| `pinn_gpu_training` | GPU PINN training |
| `pinn_meta_uncertainty` | PINN meta-uncertainty |
| `pinn_multi_gpu_training` | Multi-GPU PINN training |
| `pinn_real_time_inference` | Real-time PINN inference |
| `pinn_2d_heterogeneous` | 2D heterogeneous PINN |

### Helper Modules

These are internal helper modules used by examples, not standalone examples:

| Module | Description |
|--------|-------------|
| `dg_common` | Common DG utilities (lines, sampling, etc.) |
| `focused_water_tank_common` | Common focused water tank utilities |

## Running Examples

### Prerequisites

```bash
# Build with all features
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

| Feature | Examples | Description |
|---------|----------|-------------|
| `pinn` | AI/ML examples | Physics-Informed Neural Networks |
| `gpu` | GPU-accelerated examples | WGPU-based parallel computing |
| `ritk` | `skull_ct_phase_correction` | RITK DICOM series loading |
| `full` | All examples | Complete feature set |

## Documentation Template

Each example should have a book doc following this template:

1. **Title**: `# Example: <Name>`
2. **Metadata**: Crate, run command, source path
3. **Description**: What the example demonstrates
4. **Key Code Snippet**: Core code showing the main functionality
5. **Expected Output**: Sample output
6. **Book Chapter**: Link to the relevant chapter

## Future Work

- Create book docs for architecture/utility examples
- Add visual parity exports for comparative examples
- Add static regression guards for visual outputs
- Complete the multichapter book from examples
