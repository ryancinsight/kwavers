# Kwavers Ultrasound Physics Book

This directory is the source tree for the in-repository book on ultrasound physics for simulation, diagnostics, therapy, and theranostics. Chapters are organized in pedagogical order — physics foundations first, then transduction, imaging, therapy, and clinical case studies — and reference the production modules that implement each model.

> **Note.** All chapters (1–32) now carry unified, unique, contiguous numbers matching
> their file headers — the duplicate-numbering collisions (two "Chapter 4/5/6/7") and the
> separate case-study scheme have been resolved. See
> [REFACTOR_PLAN.md](REFACTOR_PLAN.md) for the campaign tracker.

## Chapters

### Part I — Physics Foundations
- **1.** [Wave Physics Fundamentals](foundations.md) — the linear acoustic wave equation from first principles: conservation laws, plane/spherical waves, impedance, energy and intensity, power-law absorption, and the $B/A$ nonlinearity parameter.
- **2.** [Numerical Methods: FDTD and PSTD](numerical_methods.md) — staggered-grid FDTD and pseudospectral PSTD solvers: CFL stability, numerical dispersion, the k-space temporal correction, spectral filtering, and CPML boundaries.
- **3.** [Nonlinear Acoustics](nonlinear_acoustics.md) — the Westervelt, Kuznetsov, KZK, and Burgers equations; harmonic generation, shock formation, and thermoviscous losses.
- **4.** [Media and Tissue Models](media_and_tissue_models.md) — tissue acoustic, thermal, and viscoelastic parameters; power-law/fractional-Laplacian absorption; skull and fat-layer aberration; phantom materials.
- **5.** [Cavitation and Bubble Dynamics](cavitation_and_bubbles.md) — Rayleigh–Plesset and Keller–Miksis dynamics, Blake threshold, Minnaert resonance, passive cavitation detection, and inertial collapse.

### Part II — Transduction and Sensing
- **6.** [Sources and Transducers](sources_and_transducers.md) — transmit-side physics: piezoelectric sources, piston/bowl directivity, focusing gain, phased-array delay laws, and bandlimited-interpolation source rasterization.
- **7.** [Beamforming and Image Formation](beamforming_and_image_formation.md) — receive-side physics: array factor and grating lobes, apodization, transmit focusing, delay-and-sum, and coherent plane-wave compounding.
- **8.** [Sensors and Measurements](sensors_and_measurements.md) — hydrophone directivity, the spatial Nyquist criterion, pressure–velocity sensing, and time-reversal reconstruction.

### Part III — Imaging Modalities
- **9.** [Diagnostic Ultrasound Imaging](diagnostics.md) — B-mode pulse-echo, plane-wave compounding, Doppler, contrast-enhanced imaging, and image-quality metrics.
- **10.** [Photoacoustic Imaging](photoacoustics.md) — thermoelastic generation, the Grüneisen parameter, universal back-projection, and spectroscopic oxygen-saturation unmixing.
- **11.** [Elastography](elastography.md) — linear elasticity, shear-wave speed → modulus, strain/shear-wave/MR elastography, and viscoelastic tissue models.

### Part IV — Therapy and Theranostics
- **12.** [Therapeutic Ultrasound](therapy.md) — intensity and energy deposition, the Pennes bioheat equation, CEM43 thermal dose, radiation force, sonoporation, and lithotripsy.
- **13.** [Theranostics](theranostics.md) — closed-loop imaging + therapy: passive cavitation detection, MR thermometry, and microbubble-mediated drug-delivery feedback.
- **14.** [Histotripsy: Classical vs Millisecond-Pulse](histotripsy.md) — mechanical (non-thermal) tissue ablation regimes.

### Part V — Specialized Topics
- **15.** [Transcranial Ultrasound](transcranial_ultrasound.md) — skull aberration, CT-based phase correction, time-reversal focusing, blood–brain-barrier opening, and neuromodulation.
- **16.** [Safety and Dosimetry](safety_and_dosimetry.md) — mechanical/thermal indices, CEM43, the Arrhenius damage integral, FDA limits, and the ALARA principle.
- **17.** [Sonogenetics](sonogenetics.md) — acoustic control of genetically encoded mechanosensitive channels.
- **18.** [Inverse Problems and PINNs](inverse_problems_and_pinns.md) — full-waveform inversion, the adjoint-state method, the Born approximation, and physics-informed neural networks.

### Part VI — Implementation and Validation
- **19.** [Performance and Memory](performance_and_memory.md) — the roofline model, FFT complexity, cache-optimal field layouts, Rayon work-stealing, and GPU dispatch.
- **20.** [Validation and Benchmarking](validation_and_benchmarking.md) — Pearson/PSNR fidelity metrics, k-Wave parity, convergence, and the regression suite.
- **21.** [Simulation Orchestration: The Capability Catalog](simulation_orchestration.md) — the plugin contract, capability lattice, and dispatch/scheduling.

### Part VII — Clinical Case Studies
- **22.** [Passive Acoustic Mapping](passive_acoustic_mapping.md) — real-time cavitation localization.
- **23.** [LIFU-Mediated Blood–Brain Barrier Opening](bbb_lifu_opening.md)
- **24.** [Transcranial HIFU and BBB Treatment Planning](hifu_transcranial_ablation.md)
- **25.** [Low-Intensity Ultrasound Neuromodulation](neuromodulation.md)
- **26.** [Transcranial UST Brain Imaging](transcranial_ust_brain_imaging.md)
- **27.** [Abdominal Histotripsy FWI Targeting and Lesion Monitoring](abdominal_histotripsy_fwi.md)
- **28.** [Same-Device Therapeutic Ultrasound, Finite-Frequency Inverse, and RTM Monitoring](theranostic_fwi_platforms.md)
- **29.** [Intravascular Ultrasound Imaging and Therapy](intravascular_ultrasound.md)
- **30.** [Clinical Theranostic Device Geometries](clinical_device_geometry.md)
- **31.** [Segmented Tissue Transducer Planning](segmented_tissue_transducer_planning.md)
- **32.** [Pancreatic Cancer Histotripsy (PDAC)](pancreatic_histotripsy.md)

## Figure Sources

Figures are generated SVG assets committed with the chapter text so documentation builds remain reproducible without network access.

- [Focused therapy field](figures/therapy_focused_field.svg)
- [Ultrafast diagnostic pipeline](figures/diagnostics_ultrafast_pipeline.svg)
- [Theranostic feedback loop](figures/theranostics_feedback_loop.svg)
- [Wave energy flow](figures/wave_energy_flow.svg)
- [Solver validation stack](figures/solver_validation_stack.svg)
- [Transcranial path model](figures/transcranial_path_model.svg)
- [Inverse-problem loop](figures/inverse_problem_loop.svg)
