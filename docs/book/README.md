# Kwavers Ultrasound Physics Book

This directory is the source tree for the in-repository book on ultrasound physics for simulation, diagnostics, therapy, and theranostics. Chapters are organized in pedagogical order — physics foundations first, then transduction, imaging, therapy, and clinical case studies — and reference the production modules that implement each model.

> **Note (refactor in progress).** The book is being consolidated to remove
> cross-chapter redundancy; chapter numbering in the files is being unified in a
> final pass. The order below is the canonical target. See
> [REFACTOR_PLAN.md](REFACTOR_PLAN.md) for the campaign tracker.

## Chapters

### Part I — Physics Foundations
- [Wave Physics Fundamentals](foundations.md) — the linear acoustic wave equation from first principles: conservation laws, plane/spherical waves, impedance, energy and intensity, power-law absorption, and the $B/A$ nonlinearity parameter.
- [Numerical Methods: FDTD and PSTD](numerical_methods.md) — staggered-grid FDTD and pseudospectral PSTD solvers: CFL stability, numerical dispersion, the k-space temporal correction, spectral filtering, and CPML boundaries.
- [Nonlinear Acoustics](nonlinear_acoustics.md) — the Westervelt, Kuznetsov, KZK, and Burgers equations; harmonic generation, shock formation, and thermoviscous losses.
- [Media and Tissue Models](media_and_tissue_models.md) — tissue acoustic, thermal, and viscoelastic parameters; power-law/fractional-Laplacian absorption; skull and fat-layer aberration; phantom materials.
- [Cavitation and Bubble Dynamics](cavitation_and_bubbles.md) — Rayleigh–Plesset and Keller–Miksis dynamics, Blake threshold, Minnaert resonance, passive cavitation detection, and inertial collapse.

### Part II — Transduction and Sensing
- [Sources and Transducers](sources_and_transducers.md) — transmit-side physics: piezoelectric sources, piston/bowl directivity, focusing gain, phased-array delay laws, and bandlimited-interpolation source rasterization.
- [Beamforming and Image Formation](beamforming_and_image_formation.md) — receive-side physics: array factor and grating lobes, apodization, transmit focusing, delay-and-sum, and coherent plane-wave compounding.
- [Sensors and Measurements](sensors_and_measurements.md) — hydrophone directivity, the spatial Nyquist criterion, pressure–velocity sensing, and time-reversal reconstruction.

### Part III — Imaging Modalities
- [Diagnostic Ultrasound Imaging](diagnostics.md) — B-mode pulse-echo, plane-wave compounding, Doppler, contrast-enhanced imaging, and image-quality metrics.
- [Photoacoustic Imaging](photoacoustics.md) — thermoelastic generation, the Grüneisen parameter, universal back-projection, and spectroscopic oxygen-saturation unmixing.
- [Elastography](elastography.md) — linear elasticity, shear-wave speed → modulus, strain/shear-wave/MR elastography, and viscoelastic tissue models.

### Part IV — Therapy and Theranostics
- [Therapeutic Ultrasound](therapy.md) — intensity and energy deposition, the Pennes bioheat equation, CEM43 thermal dose, radiation force, sonoporation, and lithotripsy.
- [Theranostics](theranostics.md) — closed-loop imaging + therapy: passive cavitation detection, MR thermometry, and microbubble-mediated drug-delivery feedback.
- [Histotripsy: Classical vs Millisecond-Pulse](histotripsy.md) — mechanical (non-thermal) tissue ablation regimes.

### Part V — Specialized Topics
- [Transcranial Ultrasound](transcranial_ultrasound.md) — skull aberration, CT-based phase correction, time-reversal focusing, blood–brain-barrier opening, and neuromodulation.
- [Safety and Dosimetry](safety_and_dosimetry.md) — mechanical/thermal indices, CEM43, the Arrhenius damage integral, FDA limits, and the ALARA principle.
- [Sonogenetics](sonogenetics.md) — acoustic control of genetically encoded mechanosensitive channels.
- [Inverse Problems and PINNs](inverse_problems_and_pinns.md) — full-waveform inversion, the adjoint-state method, the Born approximation, and physics-informed neural networks.

### Part VI — Implementation and Validation
- [Performance and Memory](performance_and_memory.md) — the roofline model, FFT complexity, cache-optimal field layouts, Rayon work-stealing, and GPU dispatch.
- [Validation and Benchmarking](validation_and_benchmarking.md) — Pearson/PSNR fidelity metrics, k-Wave parity, convergence, and the regression suite.
- [Simulation Orchestration: The Capability Catalog](simulation_orchestration.md) — the plugin contract, capability lattice, and dispatch/scheduling.

### Part VII — Clinical Case Studies
- [Passive Acoustic Mapping](passive_acoustic_mapping.md) — real-time cavitation localization.
- [LIFU-Mediated Blood–Brain Barrier Opening](bbb_lifu_opening.md)
- [Transcranial HIFU and BBB Treatment Planning](hifu_transcranial_ablation.md)
- [Low-Intensity Ultrasound Neuromodulation](neuromodulation.md)
- [Transcranial UST Brain Imaging](transcranial_ust_brain_imaging.md)
- [Abdominal Histotripsy FWI Targeting and Lesion Monitoring](abdominal_histotripsy_fwi.md)
- [Same-Device Therapeutic Ultrasound, Finite-Frequency Inverse, and RTM Monitoring](theranostic_fwi_platforms.md)
- [Intravascular Ultrasound Imaging and Therapy](intravascular_ultrasound.md)
- [Clinical Theranostic Device Geometries](clinical_device_geometry.md)
- [Segmented Tissue Transducer Planning](segmented_tissue_transducer_planning.md)
- [Pancreatic Cancer Histotripsy (PDAC)](pancreatic_histotripsy.md)

## Figure Sources

Figures are generated SVG assets committed with the chapter text so documentation builds remain reproducible without network access.

- [Focused therapy field](figures/therapy_focused_field.svg)
- [Ultrafast diagnostic pipeline](figures/diagnostics_ultrafast_pipeline.svg)
- [Theranostic feedback loop](figures/theranostics_feedback_loop.svg)
- [Wave energy flow](figures/wave_energy_flow.svg)
- [Solver validation stack](figures/solver_validation_stack.svg)
- [Transcranial path model](figures/transcranial_path_model.svg)
- [Inverse-problem loop](figures/inverse_problem_loop.svg)
