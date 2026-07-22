# Summary

[Introduction](README.md)

# Part I — Foundations

- [1. Wave Physics Fundamentals](foundations.md)
  - [Example: Basic Simulation](examples/basic_simulation.md)
  - [Example: Minimal Demo](examples/minimal_demo.md)
  - [Example: Electromagnetic Simulation](examples/electromagnetic_simulation.md)
- [2. Numerical Methods: FDTD and PSTD](numerical_methods.md)
  - [Example: PSTD vs FDTD Comparison](examples/pstd_fdtd_comparison.md)
  - [Example: DG Acoustic 1D Diagnostics](examples/dg_acoustic_1d.md)
  - [Example: DG Acoustic Comparison Plot](examples/dg_acoustic_comparison_plot.md)
  - [Example: DG Acoustic Convergence Plot](examples/dg_acoustic_convergence_plot.md)
  - [Example: DG Acoustic Timestep Sweep](examples/dg_acoustic_timestep_sweep.md)
  - [Example: DG Advection Diagnostics](examples/dg_advection_diagnostics.md)
- [3. Nonlinear Acoustics](nonlinear_acoustics.md)
  - [Example: Sonoluminescence](examples/single_bubble_sonoluminescence.md)
  - [Example: Multiphysics Sonoluminescence](examples/multiphysics_sonoluminescence.md)
  - [Example: Sonoluminescence Comparison](examples/sonoluminescence_comparison.md)
- [4. Media and Tissue Models](media_and_tissue_models.md)
  - [Example: Spatially Varying Attenuation](examples/spatially_varying_attenuation.md)
  - [Example: Tau Parameter Sweep](examples/tau_sweep.md)
- [5. Cavitation and Bubble Dynamics](cavitation_and_bubbles.md)
- [6. Sources and Transducers](sources_and_transducers.md)
  - [Example: Focused Ultrasound Water Tank](examples/focused_ultrasound_water_tank.md)
  - [Example: Phantom Builder Demo](examples/phantom_builder_demo.md)
  - [Example: Boundary Smoothing](examples/boundary_smoothing.md)

# Part II — Imaging and Sensing

- [7. Transducer Arrays and Beamforming](beamforming_and_image_formation.md)
  - [Example: Phased Array Beamforming](examples/phased_array_beamforming.md)
  - [Example: Adaptive Beamforming](examples/adaptive_beamforming.md)
  - [Example: Adaptive Beamforming Refactored](examples/adaptive_beamforming_refactored.md)
  - [Example: Real-Time 3D Beamforming](examples/real_time_3d_beamforming.md)
- [8. Sensors and Measurements](sensors_and_measurements.md)
  - [Example: Doppler Velocity Estimation](examples/doppler_velocity_estimation.md)
- [9. Diagnostic Ultrasound Imaging](diagnostics.md)
  - [Example: Advanced Ultrasound Imaging](examples/advanced_ultrasound_imaging.md)
- [10. Photoacoustic Imaging](photoacoustics.md)
  - [Example: Photoacoustic Imaging](examples/photoacoustic_imaging.md)
  - [Example: Photoacoustic Blood Oxygenation](examples/photoacoustic_blood_oxygenation.md)
- [11. Elastography: Imaging Tissue Mechanical Properties](elastography.md)
  - [Example: Elastography Simulation](examples/elastography_simulation.md)
  - [Example: SWE Liver Fibrosis](examples/swe_liver_fibrosis.md)
  - [Example: 3D SWE Liver Fibrosis](examples/swe_3d_liver_fibrosis.md)
  - [Example: NL-SWE Convergence Validation](examples/nl_swe_convergence_validation.md)

# Part III — Therapy and Theranostics

- [12. Therapeutic Ultrasound](therapy.md)
  - [Example: Comprehensive Clinical Workflow](examples/comprehensive_clinical_workflow.md)
- [13. Theranostics: Combined Imaging and Therapy](theranostics.md)
  - [Example: Brain Theranostic Monitor](examples/brain_theranostic_monitor.md)
  - [Example: Liver Theranostic Reconstruction](examples/liver_theranostic_reconstruction.md)
  - [Example: Hybrid Lesion Monitor](examples/hybrid_lesion_monitor.md)
- [14. Histotripsy: Classical vs Millisecond-Pulse Regimes](histotripsy.md)
- [15. Transcranial Ultrasound: Physics, Aberration Correction, and Therapy](transcranial_ultrasound.md)
  - [Example: Skull CT Phase Correction](examples/skull_ct_phase_correction.md)
  - [Example: Transcranial FWI](examples/transcranial_fwi.md)
- [16. Ultrasound Safety and Dosimetry](safety_and_dosimetry.md)
- [17. Sonogenetics: Acoustic Control of Mechanosensitive Systems](sonogenetics.md)

# Part IV — Inverse Problems and Computation

- [18. Inverse Problems and Physics-Informed Neural Networks](inverse_problems_and_pinns.md)
  - [Example: PINN Training Convergence](examples/pinn_training_convergence.md)
  - [Example: PINN 2D Wave Equation](examples/pinn_2d_wave_equation.md)
  - [Example: PINN Advanced Physics](examples/pinn_advanced_physics.md)
  - [Example: Transfer Learning PINN](examples/transfer_learning_pinn.md)
  - [Example: Comprehensive PINN Demo](examples/comprehensive_pinn_demo.md)
  - [Example: PINN 2D Heterogeneous Media](examples/pinn_2d_heterogeneous.md)
  - [Example: PINN GPU Training](examples/pinn_gpu_training.md)
  - [Example: PINN Meta-Learning and Uncertainty](examples/pinn_meta_uncertainty.md)
  - [Example: PINN Multi-GPU Training](examples/pinn_multi_gpu_training.md)
  - [Example: PINN Real-Time Inference](examples/pinn_real_time_inference.md)
  - [Example: Validate 2D PINN](examples/validate_2d_pinn.md)
- [19. Validation and Benchmarking](validation_and_benchmarking.md)
  - [Example: Literature Validation](examples/literature_validation.md)
  - [Example: Literature Validation Safe Vectorization](examples/literature_validation_safe.md)
  - [Example: Physics Validation](examples/physics_validation.md)
  - [Example: Performance Validation](examples/performance_validation.md)
  - [Example: Theorem Validation Demo](examples/theorem_validation_demo.md)
  - [Example: Monte Carlo Validation](examples/monte_carlo_validation.md)
- [20. Performance and Memory](performance_and_memory.md)
  - [Example: Safe Vectorization Benchmarks](examples/safe_vectorization_benchmarks.md)
- [21. Simulation Orchestration: The Capability Catalog](simulation_orchestration.md)
  - [Example: Plugin Architecture](examples/plugin_example.md)

# Part V — Advanced Applications

- [22. Passive Acoustic Mapping](passive_acoustic_mapping.md)
- [23. LIFU-Mediated Blood–Brain Barrier Opening](bbb_lifu_opening.md)
- [24. Transcranial HIFU and BBB Treatment Planning](hifu_transcranial_ablation.md)
  - [Example: Transcranial CT/MRI Reconstruction](examples/transcranial_ct_mri.md)
- [25. Low-Intensity Ultrasound Neuromodulation](neuromodulation.md)
- [26. Transcranial UST Brain Imaging](transcranial_ust_brain_imaging.md)
- [27. Abdominal Histotripsy FWI Targeting and Lesion Monitoring](abdominal_histotripsy_fwi.md)
- [28. Same-Device Therapeutic Ultrasound, Finite-Frequency Inverse, and RTM Monitoring](theranostic_fwi_platforms.md)
- [29. Intravascular Ultrasound Imaging and Therapy](intravascular_ultrasound.md)
- [30. Clinical Theranostic Device Geometries](clinical_device_geometry.md)
- [31. Segmented Tissue Transducer Planning](segmented_tissue_transducer_planning.md)
- [32. Pancreatic Cancer Histotripsy: PDAC Treatment Planning](pancreatic_histotripsy.md)
- [33. CMUT vs PMUT: Micromachined and Flexible Transducers for IVUS](cmut_vs_pmut.md)
- [34. Optically-Generated Focused Ultrasound for Ultrahigh-Precision Neuromodulation](optoacoustic_focused_ultrasound.md)

---

# Part VI — Atlas Stack Integration (Migration Reference)

This part documents the migration from ndarray/nalgebra to the Atlas stack crates:

- [35. Migration Overview: ndarray/nalgebra → Leto](migration_overview.md)
- [36. Linear Algebra: Leto and Leto-Ops](migration_linalg.md)
- [37. Geometry: Leto for Point, Vector, Isometry](migration_geometry.md)
- [38. SIMD: Hermes for Vectorized Operations](migration_simd.md)
  - [Example: SIMD Wave Kernel](examples/simd_wave_kernel.md)
- [39. Memory: Mnemosyne and Themis](migration_memory.md)
- [40. Concurrency: Moirai for Parallel Execution](migration_concurrency.md)
- [41. FFT: Apollo for Spectral Methods](migration_fft.md)
- [42. Python Integration: PyO3 and NumPy Boundary](migration_python.md)
- [43. GAT Tiling: LendingIterator and Tiles](migration_gat_tiles.md)
  - [Example: Tiled K-Space Processing](examples/tiled_kspace_processing.md)

---

# Appendix

- [A. Migration Quick Reference](migration_quick_reference.md)
- [B. Atlas Crate Dependencies](atlas_dependencies.md)
- [C. Glossary](appendix_glossary.md)

