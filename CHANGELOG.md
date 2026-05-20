# Changelog

## Unreleased

### Changed (2026-05-20) - Focused Bowl Geometry SSOT

- [minor] Route abdominal 3-D and nonlinear 3-D focused-bowl placement through
  `domain::source::transducers::focused::cap`. `SphericalCapConfig` now exposes
  explicit angular-span and vertex/focus constructors, while clinical bowl
  adapters reuse the source-domain equal-area layout and reject degenerate axes
  instead of fabricating repeated skin-contact elements.

### Added (2026-05-20) - Focused Spherical-Cap Source Layout

- [minor] Add `domain::source::transducers::focused::cap` as the reusable
  equal-area spherical-cap layout generator for focused bowl sources. The new
  `SphericalCapConfig`/`SphericalCapLayout` API covers hemispherical and partial
  bowl apertures through focus, axis, radius, and angular span parameters,
  keeping transcranial or abdominal placement policy out of solver-specific
  geometry names.

### Fixed (2026-05-20) - Broadband Cavitation Detection Domain Guards

- [patch] Guard broadband cavitation detection against empty or nonfinite
  signals so invalid inputs return finite zero metrics and cannot poison the
  adaptive baseline energy used for inertial-cavitation classification.

### Fixed (2026-05-20) - CEUS Microbubble Harmonic Domain Guards

- [patch] Guard microbubble harmonic-content analysis against invalid physical
  domains: zero harmonic index, invalid sample rate, mismatched time/pressure
  vectors, and nonfinite samples now return finite zero content instead of
  panicking or propagating NaN through the harmonic projection.

### Fixed (2026-05-20) - Analytical Plane-Wave Domain Guards

- [patch] Guard the analytical plane-wave generator against invalid physical
  domains: nonpositive frequency or sound speed, nonfinite amplitude or time,
  invalid grid spacing, and zero or nonfinite propagation direction now return
  a finite zero field instead of emitting NaN values through normalization or
  phase evaluation.

### Fixed (2026-05-20) - Sonogenetics Analytical Domain Guards

- [patch] Guard sonogenetics analytical helpers against invalid physical
  domains: Hill activation now rejects invalid thresholds/exponents and
  nonfinite pressure samples, acoustic radiation force and streaming reject
  negative/nonfinite intensity or material domains, and ISPTA rejects empty or
  invalid waveform/material inputs while ignoring nonfinite pressure samples.

### Fixed (2026-05-20) - Acoustic Analysis Field Validation and Directivity

- [patch] Add a shared acoustic analysis pressure-field validation module and
  route beam-pattern, focus, and field-metric routines through the same
  field/grid shape plus finite-pressure contract. Beam-pattern configuration
  now rejects invalid frequency, sound speed, and angular resolution before
  angular allocation, and directivity now averages squared pressure magnitude
  proportional to intensity instead of signed samples to prevent phase
  cancellation from inflating DI.

### Fixed (2026-05-20) - Acoustic Field Metrics Domain Validation

- [patch] Route acoustic field metrics through the canonical impedance and
  harmonic peak-intensity helpers, reject field/grid shape mismatches,
  nonfinite pressure samples, and invalid impedance domains before metric
  accumulation, and add value-semantic tests for signed peak magnitude,
  stored-energy/intensity formulas, shape mismatch, and invalid-domain
  rejection.

### Fixed (2026-05-20) - Cavitation Core Mechanical-Index Consolidation

- [patch] Complete the cavitation core MI consolidation by routing
  `CavitationModel` and core tests to
  `physics::acoustics::analysis::calculate_mechanical_index` after the local
  duplicate MI helper was removed. This restores compilation without
  reintroducing a compatibility alias and keeps cavitation MI on the canonical
  `|p_r|_MPa / sqrt(f_MHz)` contract.
- [patch] Complete the same no-alias MI consolidation in the nonlinear 3-D
  theranostic cavitation tests by importing the canonical helper directly after
  the local helper was removed from the forward map.

### Fixed (2026-05-20) - Transcranial Planning Acoustic Simulation Domains

- [patch] Correct transcranial treatment-planning acoustic simulation to apply
  per-element amplitudes in coherent pressure summation, evaluate documented
  millimeter transducer positions in SI meters, validate transducer frequency
  and vector domains, and reject negative/nonfinite acoustic intensity in the
  Pennes thermal response. Zero or invalid acoustic heating now yields an
  infinite treatment-time estimate instead of a false zero-duration plan. Added
  value-semantic tests for amplitude-squared intensity scaling, millimeter
  position conversion, Pennes source balance, and invalid-domain rejection.

### Fixed (2026-05-20) - Transcranial Treatment-Planning Safety Validation

- [patch] Correct transcranial treatment-planning MI validation to derive peak
  pressure from harmonic average intensity with `p_peak = sqrt(2 rho c I)` and
  route MI through the canonical acoustic pressure-analysis helper. Safety
  validation now rejects nonfinite brain temperature, invalid Hz frequency,
  negative/nonfinite intensity fields, and nonfinite MI instead of allowing
  invalid inputs to bypass constraints. Added focused value tests for the
  pressure-intensity theorem, valid safe fields, and invalid-domain rejection.

### Fixed (2026-05-20) - Mechanical Index Safety Path Consolidation

- [patch] Route cavitation power-modulation and transcranial safety-monitoring
  MI calculations through the canonical acoustic pressure-analysis helper
  `MI = |p_r|_MPa / sqrt(f_MHz)`. The power-modulation limiter now fails
  closed for nonfinite pressure, nonpositive/nonfinite MHz frequency, and
  signed rarefactional pressure. Transcranial monitoring now fails closed for
  invalid Hz frequency or nonfinite pressure fields, records finite-ratio
  safety margins only when MI is finite, and pins signed-pressure and
  invalid-domain tests.

### Fixed (2026-05-20) - Acoustic Pressure Analysis Domains

- [patch] Guard `physics::acoustics::analysis::pressure` against invalid
  acoustic impedance, nonfinite pressure samples, invalid MI frequency,
  negative/nonfinite thermal deposition inputs, negative derating distance or
  frequency, and out-of-range ISPTA duty cycle. Shared intensity helpers now
  preserve the harmonic peak-pressure contract `I = p^2 / (2 rho c)` while
  preventing NaN, infinity, negative exposure ratios, and attenuation-as-gain
  from safety-index calculations. Added value-semantic tests for valid formulas
  and invalid-domain rejection.

### Fixed (2026-05-20) - HIFU Field and Thermal-Dose Physics

- [patch] Split `physics::acoustics::imaging::modalities::ultrasound::hifu`
  into a facade plus `field`, `thermal_dose`, and `tests` submodules. Replaced
  the corner-focused Gaussian/spherical shortcut with a centered
  Rayleigh-Sommerfeld aperture integral using O'Neil focused-aperture phase
  delays, and corrected HIFU intensity to `p_peak^2 / (2 rho c)`.
- [patch] Correct CEM43 accumulation to the Sapareto-Dewey contract using
  seconds-to-minutes conversion and `R = 0.5` at/above 43 deg C, `R = 0.25`
  below 43 deg C. Added value-semantic tests for lateral focus centering,
  symmetry, intensity, CEM43 reference temperatures, and ablation threshold.

### Changed (2026-05-20) - Analytical Physics Boundary Rename

- [arch] Rename the artifact-owned analytical physics boundary to
  `kwavers::physics::analytical`. The former artifact-named domain module is
  removed rather than re-exported. PyO3 bindings now import the corrected
  analytical boundary, and Rust-side source references in the changelog use
  `physics::analytical`.

### Fixed (2026-05-20) - Cavitation Closed-Form Domain Guards

- [patch] Add physical-domain guards for book cavitation closed-form
  estimators. Minnaert resonance, Blake threshold, Rayleigh collapse time, and
  histotripsy lesion radius now return `0.0` for nonfinite or nonpositive
  domains where the formulas are undefined, preventing negative radii,
  negative frequencies, NaN, or infinite scalar estimates from propagating.
  Added value-semantic tests for invalid-domain rejection.

### Fixed (2026-05-20) - Mechanical Index Contract Unification

- [patch] Align the book histotripsy and transcranial BBB-opening mechanical
  index helpers with the canonical `MI = |p_r|_MPa / sqrt(f_MHz)` contract.
  Both paths now reject nonfinite pressure and nonpositive/nonfinite frequency
  with `0.0`, preventing negative MI, NaN, or infinite output from safety and
  therapeutic-window calculations. Added value tests for signed pressure and
  invalid frequency domains.

### Fixed (2026-05-20) - Clinical Safety Thermal Indices

- [patch] Clamp invalid thermal-index input domains in
  `physics::analytical::safety`: soft-tissue TI now rejects nonfinite/negative
  acoustic power and nonpositive/nonfinite MHz frequency, and bone TI rejects
  nonfinite/negative acoustic power or MHz frequency. Added value tests for
  unit-ratio examples and invalid-domain rejection. Corrected the stale FDA
  output-limit reference text to the diagnostic-ultrasound guidance table.

### Fixed (2026-05-20) - Clinical Safety Mechanical Index

- [patch] Correct `physics::analytical::safety::mechanical_index` to use the
  magnitude of rarefactional pressure and to reject nonpositive or nonfinite
  MHz frequency with `0.0`, matching the clinical safety contract
  `MI = |p_r,min|_MPa / sqrt(f_MHz)`. Added value tests for signed pressure
  input and invalid frequency.

### Changed (2026-05-18) - Type Collision Disambiguation Session 5

- [patch] Close residual trait-struct and same-name pairs deferred from
  session 4. `cargo check --lib` exit 0. Renames:
  `SpectralOperator` → `SpectralOperatorTrait` (math) /
  `KuznetsovSpectralOperator` (kuznetsov solver);
  `SourceParameters` → `DomainSourceParameters` (struct) /
  `FactorySourceParameters` (trait); `MediumParameters` →
  `DomainMediumParameters` (struct) / `FactoryMediumParameters` (trait);
  `GridParameters` → `DomainGridParameters` (struct) /
  `FactoryGridParameters` (trait); `Preconditioner` →
  `SparsePreconditioner` (math enum) / `HelmholtzPreconditioner` (trait);
  `PlasmonicEnhancement` → `PlasmonicEnhancementEquation` (trait) /
  `PlasmonicEnhancementCalculator` (struct); `OpticalProperties` →
  `MediumOpticalProperties` (trait) / `DiffusionOpticalProperties` (struct);
  `Backend` (domain tensor enum) → `TensorBackend` (canonical; removed the
  `pub use Backend as TensorBackend` alias since the type itself is now the
  canonical name); `Backend` (solver trait) → `ComputeBackend`;
  `AdaptiveBeamformer` → `AdaptiveTimeDomainBeamformer` /
  `AdaptiveFrequencyBeamformer`; `EMSource` → `DomainEMSource` (domain trait) /
  `PhysicsEMSource` (physics trait); `AcousticBoundaryType` →
  `DomainAcousticBoundaryType` (domain enum) / `PinnAcousticBoundaryType`
  (pinn enum); `TransducerGeometry` → `HifuTransducerGeometry` (hifu enum) /
  `FieldCalculatorTransducerGeometry` (struct); `SolverFactory` →
  `SolverFactoryRegistry` (struct) / `SolverFactoryTrait` (trait).

### Changed (2026-05-18) - Type Collision Disambiguation Session 4

- [patch] Disambiguate ~50 type collision clusters across kwavers via
  domain-specific renames. Every collision resolved by giving each definition
  a globally unique, module-path-prefixed name; no backward-compat aliases,
  no `pub use X as Y` shims; every caller updated in the same change.
  Verified by `cargo check --lib` exit 0.
- [patch] Representative renames (definition-side prefix per module scope):
  `LossComponents` → `PhysicsInformedLossComponents` / `ElasticPinnLossComponents`;
  `ActivationFunction` → `PinnBeamforming*` / `ElasticPinn*`;
  `Avx512{Config,StencilProcessor,Metrics}` → `Fdtd*` / `Simd*`;
  `BayesianPINN` → `Ml*` / `Pinn*`; `BjerknesForce` → `BjerknesForceData` /
  `BjerknesForceComputer`; `BoundaryComponent` → `Face*` / `Pinn*`;
  `BoundaryConditionSpec` → `Api*` / `Pinn*`; `BufferManager` →
  `GpuBufferManager` / `GpuBackendBufferManager`; `CavitationEvent` → `Pam*` /
  `Recorder*`; `CavitationState` → `CavitationDetectionState` /
  `CavitationMechanicsState`; `ClinicalDecisionSupport` → `Neural*` / `Swe3d*`;
  `CommunicationChannel` → `Gpu*` / `PinnMultiGpu*`; `ComplianceReport` →
  `Safety*` / `Validator*`; `ConformalPredictor` → `Ml*` / `Pinn*`;
  `ConservationEnforcer` → `MultiPhysics*` / `HybridCoupling*`;
  `ConservationMetrics` → `Acoustic*` / `Hybrid*`; `ConservativeInterpolator`
  → `Amr*` / `Util*`; `ConvergenceInfo` → `Gmres*` / `UniversalSolver*`;
  `DeviceInfo` → `Gpu*` / `Api*`; `DeviceLostRecovery` → `ErrorGpu*` / `Gpu*`;
  `DeviceStatus` → `Api*` / `Transducer*`; `DiffractionOperator` → `HybridAs*`
  / `KzkParabolic*`; `DirectivityPattern` → `Pam*` / `Transducer*`;
  `DispersionCorrection` → `Absorption*` / `Freq*`; `DomainInfo` → `Hybrid*` /
  `UniversalSolver*`; `ErrorMetrics` → `HybridValidation*` / `Kwave*`;
  `FaultInjector` → `GpuRecovery*` / `Core*` / `GpuInjector*`; `FieldCoupler`
  → `MultiPhysics*` / `Multiphysics*`; `FocalProperties` → `Source*` /
  `PinnSource*`; `Geometry` → `SolverGeometry` / `PinnTrainerGeometry`;
  `Geometry2D` → `BurnWave2dGeometry` / `UniversalSolver*`; `GpuBuffer` →
  `Perf*` / `GpuBufferData`; `GpuCapabilities` → `Core*` / `Pinn*`;
  `GpuContext` → `Renderer*` / `Core*`; `GpuOomRecovery` → `ErrorRecovery*` /
  `GpuRecoveryOom`; `GpuRecoveryManager` → `ErrorGpu*` /
  `GpuRecoveryManagerImpl`; `GradientComputer` → `Iterator*` / `Seismic*`;
  `InterfacePoint` → `Medium*` / `Iterator*`; `ModelMetadata` → `Ml*` / `Api*`
  / `Quantization*`; `MonitoringConfig` → `Clinical*` / `Hifu*` / `Cloud*`;
  `MultiPhysicsInterface` → `Boundary*` / `Simulation*`; `PMLBoundary` →
  `DomainPMLBoundary` / `ElasticSwe*`; `PerfusionModel` → `Ceus*` /
  `Thermal*`; `PhysicsDomain` → `BoundaryCoupling*` / `Simulation*` /
  `GpuKernel*`; `PinnBeamformingResult` → `Neural*` / `Interface*`;
  `PlaneWave` → `Ultrafast*` / `KwaveAnalytical*`; `PredictionWithUncertainty`
  → `Ml*` / `Pinn*`; `Predictor` (elastic_2d inference) → `ElasticPinnPredictor`;
  `ProcessingConfig` → `Visualization*` / `Mobile*`; `Quantizer` →
  `BurnWave2d*` / `Ml*`; `RateLimitConfig` → `Api*` / `RateLimiter*`;
  `RecoveryStats` → `GpuRecoveryFaultStats` / `CoreRecoveryStats`;
  `SizeDistribution` → `Ceus*` / `BubbleField*`; `StagePipeline` → `Stream*` /
  `Flat*`; `SvdClutterFilter` → `Signal*` / `Ulm*`; `TargetVolume` →
  `Therapy*` / `Transcranial*`; `TimeReversal` → `Aberration*` /
  `Photoacoustic*`; `TimeoutRecovery` → `Error*` / `Gpu*`;
  `TrilinearInterpolator` → `Het*` / `Numerics*`; `MemoryPool` (4-way) →
  `PerfMemoryPool` / `GpuMemoryPool` / `BurnWave2dInferenceMemoryPool` /
  `JitMemoryPool` (+ `PinnGpuAcceleratorMemoryPool` for the pub(crate) struct);
  sonochemistry `BubbleState` → `SonochemBubbleState`; pinn-side
  `GpuDeviceInfo` → `PinnMultiGpuDeviceInfo`; `KzkDiffractionOperator`
  parabolic → `KzkParabolicDiffractionOperator`.

### Fixed (2026-05-18) - DG Open-Boundary Policy

- [patch] Add `DgBoundaryCondition::{Periodic, AbsorbingCharacteristic}`
  for tensor acoustic DG face fluxes in per-axis `[x, y, z]` form. `Periodic`
  remains the default conservation baseline; focused water-tank DG tensor runs
  now use one-way acoustic characteristic exterior states on in-plane physical
  tank boundaries while preserving the periodic out-of-plane invariant axis of
  the embedded 2-D slab.
- [patch] Split tensor DG face-state selection into
  `acoustic/tensor/boundary.rs`, document the characteristic invariant
  `w+ = p + Z u_n`, `w- = p - Z u_n`, and validate outgoing-preservation /
  incoming-rejection plus periodic weighted-mass conservation in 2-D and 3-D.
- [patch] Repair normal-target verification blockers encountered in the
  requested test path: restore the local `apollo-fft` Stockham scalar fallback
  imports and correct the clinical safety `SafetyComplianceReport` re-export.
- [patch] Regenerate the focused water-tank artifacts under
  `target/focused_water_tank/`. Current focused-map metrics: FDTD vs PSTD
  normalized-L2 `1.142732e-1`, FDTD vs DG-2D `1.616039e-1`, PSTD vs DG-2D
  `1.635862e-1`, DG-2D vs analytic `1.933581e-1`, and DG-2D peak focus at
  `(8 mm, 9 mm)`, matching FDTD/PSTD/analytic. Axis-aware boundaries close the
  thin-slab 3-D DG discrepancy: FDTD vs DG-3D normalized-L2 is now
  `1.616039e-1`, and DG-2D vs DG-3D normalized-L2 is `1.756510e-8` with
  correlation `1.000000`.

### Fixed (2026-05-18) - DG Focused-Water-Tank Alignment

- [patch] Add uniform-grid interpolation for tensor acoustic DG pressure and
  velocity projection. Focused-water-tank metrics now evaluate DG GLL
  polynomials at FDTD/PSTD grid coordinates instead of treating GLL nodes as
  uniformly spaced samples.
- [patch] Add RK-stage-consistent acoustic tensor source injection with weak
  cell-source weights. The focused-water-tank DG source now enters each SSP-RK3
  stage RHS at the stage time and distributes one uniform source cell through
  the local GLL mass weights.
- [patch] Route the simulation DG adapter through the same uniform-grid field
  projection used by the comparison fixture.
- [patch] Add value-semantic tests for DG uniform interpolation, weak
  cell-source measure conservation, and SSP-RK3 source stage timing.
- [patch] Repair stale compile-blocking public-name drift encountered during
  verification: hybrid validation config naming, 3-D beamforming apodization
  naming, signal window naming, domain/imaging re-exports, SIMD level naming,
  transcranial safety monitor naming, photoacoustic reconstruction config
  naming, and bubble-dynamics Bjerknes re-export paths.
- [patch] Regenerate `target/focused_water_tank/focused_water_tank.png`,
  `focused_water_tank_metrics.csv`, and `focused_water_tank_profiles.csv`.
  Current focused-map metrics: FDTD vs PSTD normalized-L2 `1.142732e-1`,
  FDTD vs DG-2D `4.091354e-1`, PSTD vs DG-2D `3.872639e-1`, DG-2D vs DG-3D
  `1.037810e-9`, with DG-2D/DG-3D peak focus error `0.0 mm`. Superseded by
  the DG open-boundary policy metrics above.

### Added (2026-05-18) - Segmented Tissue Transducer Optimization

- [patch] Add Chapter 32 as a segmentation-driven transducer-placement and
  focal-spot optimization example. The chapter uses the local LiTS17 liver CT
  sample by default, maps native liver/tumor labels to normal/tumor planning
  compartments, derives air/fat/bone/vascular-avoid masks from CT HU thresholds,
  targets the largest connected lesion on the selected slice, forms a three
  angle crossfire plan around the safest central aperture, screens candidate
  apertures by segmented path fractions, solves complex per-element phase and
  amplitude weights with a weighted ridge system, and exports figures, metrics,
  plus value-semantic tests for both the real liver adapter and the deterministic
  analytic phantom. The default LiTS17 plan now includes dense hotspot
  refinement and regenerates Figure 2 with `target_dominant=true`, body
  sidelobe peak ratio `0.7395404024847666`, body sidelobe P99 ratio
  `0.3297347520675772`, tumor coverage `0.7837837837837838`, and protected
  peak ratio `0.2958651403757349`.

### Fixed (2026-05-18) - Book Chapter Verification

- [patch] Restore the Chapter 29 book-helper test contract by renaming the
  elastic shear display label from FWI terminology to iterative elastic inverse
  and accepting Python test stubs with empty `__text_signature__` while still
  rejecting stale nonlinear extension signatures.
- [patch] Repair PyO3 binding drift found during Chapter 32 verification by
  updating the Python array apodization binding to `KwaveApodizationWindow` and
  the signal binding to `SignalWindowType`, then updating release-only FDTD and
  PSTD Python solver wrappers to `SolverGeometry`; `cargo check -p pykwavers`,
  the development `pykwavers` cdylib build, and `cargo build -p pykwavers
  --release -j 1` now pass.

### Changed (2026-05-18) - Chapter 29 Elastic Shear Reconstruction

- [patch] Replace the Chapter 29 reduced-branch geometric exposure shortcut
  with a heterogeneous scalar acoustic peak-pressure solve. The exposure path
  now reuses the RTM finite-difference grid, CPML, attenuation, source encoding,
  and electronic steering delays, stores only three rolling pressure fields,
  two CPML fields, and one peak accumulator, and exports raw peak pressure plus
  time-step/workspace diagnostics through PyO3.
- [patch] Add a static theranostic exposure backend contract and pin the active
  backend to `reference_fdtd_cpml_2d`. The PyO3 payload now reports
  `exposure_backend` and `exposure_uses_hybrid_pstd_fdtd`, while tests keep the
  hybrid PSTD/FDTD path unselectable until source/receiver/medium/peak-map
  parity and memory-accounting checks exist. The reference peak-pressure loop
  now fuses attenuation with peak accumulation and clears only the FD halo after
  buffer rotation instead of zeroing the full destination field every step.
- [patch] Replace the Chapter 29 elastic shear comparator with iterative
  nonlinear ElasticPSTD FWI. The channel now runs baseline, observed-lesion,
  and current-estimate shear simulations from the commanded target focus,
  records same-aperture velocity traces, migrates residual trace energy for the
  update direction, accepts only objective-decreasing nonlinear shear-map
  updates, exports objective-history diagnostics through PyO3, and prints the
  CT-frame comparison theorem plus caption on Figure 6.
- [patch] Complete the mesh-boundary rename from `BoundaryType` to
  `MeshBoundaryType` at FEM and hybrid call sites so the kwavers lib tests can
  compile without compatibility aliases.

### Fixed (2026-05-18) - Chapter 29 Figure 6 Liver Targeting

- [patch] Correct the controlled Figure 6 brain linear target frame by
  resolving the canonical 3-D CT target once, mapping that index through the
  actual resampled head crop, and exporting the resampled brain crop bounds for
  CT-frame linear-field projection. Regenerated metrics show the brain linear
  exposure, linear fusion, and elastic shear hotspots inside the full-CT target
  mask, with `linear_focus_to_common_target_centroid_m = 0.0004366`.
- [patch] Preserve enclosed internal gas in the nonlinear 3-D material path.
  The body mask now flood-fills exterior air before material mapping, keeps
  enclosed HU `< -700` label-0 voxels inside the patient support, and maps them
  to gas sound speed, gas density, gas nonlinearity, and high attenuation while
  leaving boundary-connected CT background as coupling fluid.
- [patch] Correct the controlled Figure 6 liver alignment by exporting the
  linear inverse crop bounds, source dimensions, and source spacing from
  `PreparedTheranosticSlice` through PyO3, then using that metadata for
  full-CT-frame linear exposure/fusion resampling instead of assuming the
  cropped solver grid is centered on the CT placement slice.
- [patch] Make abdominal nonlinear target preparation select the same connected
  single treatment component as the 2-D linear path and demote non-selected
  label-2 lesions to organ before 3-D resampling, so liver linear and nonlinear
  comparisons use one shared target.
- [patch] Replace finite-area nonlinear source sum normalization with
  pressure-boundary peak normalization. The previous point-source integral
  model made target pressure fall as element support gained cells; the corrected
  model preserves configured surface pressure under grid refinement.
- [patch] Add a bounded measured electronic-steering calibration pass for
  abdominal nonlinear solves. The pass evaluates nominal, same-direction, and
  opposite-direction delay foci with real Westervelt calibration forwards and
  selects the focus with the highest target/window pressure ratio; liver
  calibration selected zero correction, localizing the residual pressure spread
  to treatment-window/source-region physics rather than a reversed steering
  sign.
- [patch] Update Figure 6 to display simulated nonlinear target pressure inside
  the matched lesion mask while archiving treatment-window pressure as
  `nonlinear_pressure_window`, raw body/coupling pressure as
  `nonlinear_pressure_raw`, and prefixed localization metrics. Regenerated
  `docs/book/figures/ch29/fig06_controlled_linear_nonlinear_comparison.*`,
  `controlled_comparison_metrics.json`, and
  `controlled_comparison_fields.npz`.
- [patch] Correct Figure 5 pressure targeting display by changing the visible
  Westervelt pressure panel from raw body/coupling peak pressure to target-mask
  pressure on the matched CT frame. The raw pressure evidence remains in
  diagnostics; the visible panel no longer lets source/coupling peaks appear as
  lesion-targeting failure.

### Added (2026-05-18) - Native DG Acoustic RHS and Diagnostics

- [patch] `solver::forward::pstd::dg::dg_solver::acoustic`: add a native
  1-D first-order acoustic DG RHS and SSP-RK3 stepper over pressure and
  particle velocity with caller-owned workspace reuse. The coupled RHS uses
  Lax-Friedrichs/Rusanov flux with face-normal strong-form corrections
  `F* - F_left` and `F_right - F*`, preserving quadrature-weighted pressure
  and velocity component masses on periodic line elements.
- [patch] `examples/dg_acoustic_1d_diagnostics`: add a native coupled
  acoustic comparison against the analytical standing-wave solution and the
  scalar characteristic reconstruction path. Verified metrics:
  `pressure_relative_l2 = 1.651618e-4`,
  `velocity_relative_l2 = 1.547224e-2`,
  `pressure_characteristic_l2 = 4.571134e-16`,
  `velocity_characteristic_l2 = 5.365939e-15`,
  `pressure_mass_error = 8.046975e-16`,
  `velocity_mass_error = 3.816392e-17`, and `energy_ratio = 1.0`.
- [patch] `physics::mod`: remove the stale `foundations::BoundaryCondition`
  re-export from the physics prelude; the canonical boundary trait remains
  `domain::boundary::BoundaryCondition`.
- [patch] `examples/dg_acoustic_1d_diagnostics`: extend the native diagnostic
  with an embedded-line Gaussian pressure IVP that runs native DG, classical
  FDTD, k-space FDTD, and PSTD against the same analytical d'Alembert pressure
  reference. Verified matrix metrics: `DG vs exact = 4.305350e-4`,
  `FDTD vs exact = 5.416002e-5`,
  `FDTD+k-space vs exact = 8.204688e-6`,
  `PSTD vs exact = 1.201431e-5`,
  `FDTD vs PSTD = 5.348865e-5`,
  `FDTD+k-space vs PSTD = 1.405561e-5`, and
  `DG pressure mass error = 1.865175e-14`.
- [patch] `examples/dg_acoustic_comparison_plot`: add plotted comparison
  output for the same shared fixture. The executable writes
  `target/dg_acoustic_comparison/gaussian_pressure.png` and
  `target/dg_acoustic_comparison/gaussian_pressure.csv`, plotting final
  pressure traces plus absolute error against the analytical reference for DG,
  classical FDTD, k-space FDTD, and PSTD. The plot/CSV now include both native
  solver-grid rows and common p4-quadrature rows so DG, FDTD, k-space FDTD, and
  PSTD are also compared on one physical sampling grid. Verified common-grid
  metrics: DG vs exact `1.992925e-3`, FDTD vs exact `7.912123e-3`,
  FDTD+k-space vs exact `7.943160e-3`, PSTD vs exact `7.943194e-3`,
  FDTD vs PSTD `5.197703e-5`, FDTD+k-space vs PSTD `1.097571e-5`,
  DG vs FDTD `7.700342e-3`, and DG vs PSTD `7.729329e-3`.
- [patch] `examples/dg_acoustic_comparison_plot`: add a uniform-grid DG
  resampling view on the native FDTD/PSTD grid. The DG trace averages left/right
  values at shared element interfaces, so the CSV now includes
  `uniform_pressure` and `uniform_absolute_error` rows without interpolating the
  FDTD/PSTD traces. Verified uniform-grid metrics: DG vs exact `4.661959e-5`,
  FDTD vs exact `5.416002e-5`, FDTD+k-space vs exact `8.204688e-6`,
  PSTD vs exact `1.201431e-5`, DG vs FDTD `7.735854e-5`, and DG vs PSTD
  `4.567891e-5`.
- [patch] `examples/dg_acoustic_convergence_plot`: add DG p-refinement
  diagnostics for the Gaussian pressure fixture. The executable writes
  `target/dg_acoustic_comparison/dg_order_convergence.png` and
  `target/dg_acoustic_comparison/dg_order_convergence.csv`. The CSV now reports
  both per-order nodal-quadrature error and common p4-quadrature error, proving
  that the apparent p1→p2 regression came from comparing different quadrature
  node sets. Verified common pressure relative-L2 by polynomial order: p1
  `3.402122e-2`, p2 `1.992925e-3`, p3 `1.807932e-4`, p4 `1.398263e-5`;
  mass errors remain bounded by `1.865175e-14`.
- [patch] `examples/dg_acoustic_timestep_sweep`: add a fixed-final-time
  timestep-refinement matrix and plot for DG, classical FDTD, k-space FDTD, and
  PSTD on the same Gaussian pressure IVP. The executable writes
  `target/dg_acoustic_comparison/timestep_sweep.png` and
  `target/dg_acoustic_comparison/timestep_sweep.csv`, using the native uniform
  grid with interface-averaged DG. Verified relative-L2 at steps 20/40/80:
  DG `4.661914e-5`/`4.661959e-5`/`4.661964e-5`, FDTD
  `5.478178e-5`/`5.416002e-5`/`5.384838e-5`, k-space FDTD
  `8.204427e-6`/`8.204688e-6`/`8.204890e-6`, and PSTD
  `1.206625e-5`/`1.201431e-5`/`1.198838e-5`.

### Added (2026-05-18) - Focused Water-Tank Solver Comparison

- [patch] `examples/focused_ultrasound_water_tank`: add a homogeneous-water
  focused-ultrasound comparison with a phased Hamming-apodized line aperture,
  FDTD+CPML and PSTD+CPML gated peak-pressure maps, analytical focused-array
  envelope reference, pairwise normalized-L2/correlation metrics, and
  axial/lateral profile exports. The example writes
  `target/focused_water_tank/focused_water_tank.png`,
  `target/focused_water_tank/focused_water_tank_metrics.csv`, and
  `target/focused_water_tank/focused_water_tank_profiles.csv`.
- [patch] `examples/focused_ultrasound_water_tank`: correct the source
  dimensionality for the embedded 2-D water-tank fixture by extending the
  phased aperture through every z-slice of the thin slab. The prior center-slice
  source radiated a 3-D problem while the analytical reference models a
  z-invariant line aperture. Verified metrics after correction: FDTD vs PSTD
  normalized-L2 `1.142732e-1`, correlation `0.979759`; FDTD vs analytical
  normalized-L2 `1.546687e-1`, correlation `0.962772`; PSTD vs analytical
  normalized-L2 `5.851104e-2`, correlation `0.995336`.
- [patch] `examples/focused_ultrasound_water_tank`: include native DG in the
  focused water-tank comparison as a scoped 1-D axial acoustic line solve. The
  DG path uses the coupled pressure/velocity DG RHS with SSP-RK3 substeps,
  source projection on the axial line, and a longer periodic line domain to
  avoid wraparound during the gated measurement. CSV output now records
  `axial_solver` and `axial_pair` rows. Current axial metrics: FDTD vs
  `DG-1D axial` normalized-L2 `2.218071e-1`, correlation `0.918299`; PSTD vs
  `DG-1D axial` normalized-L2 `2.199460e-1`, correlation `0.862900`;
  analytical vs `DG-1D axial` normalized-L2 `2.273648e-1`, correlation
  `0.823690`.
- [patch] `solver::forward::pstd::dg::dg_solver::acoustic`: add native
  tensor-product 2-D/3-D acoustic DG state evolution over `[p, u_x, u_y, u_z]`
  with Rusanov face fluxes, SSP-RK3 workspace reuse, direct grid projection,
  and value-semantic constant-state plus 2-D/3-D quadrature-mass conservation
  tests. `examples/focused_ultrasound_water_tank` now includes DG-2D and DG-3D
  gated peak-pressure map solvers beside FDTD, PSTD, and the analytical
  reference. Current focused-map metrics: FDTD vs DG-2D normalized-L2
  `3.079168e-1`, correlation `0.862685`; PSTD vs DG-2D normalized-L2
  `2.867320e-1`, correlation `0.880993`; DG-2D vs DG-3D normalized-L2
  `3.511912e-16`, correlation `1.000000` for the z-invariant homogeneous slab.
- [patch] `simulation::solver_adapters::dg`: route
  `SolverType::DiscontinuousGalerkin` through the native acoustic tensor state
  instead of the scalar DG advection operator. The simulation adapter now
  advances pressure and particle velocity components in 1-D/2-D/3-D active
  grids, projects `[p, u_x, u_y, u_z]` back to the simulation field layout, and
  reports nonzero velocity statistics for input-sensitive acoustic evolution.

### Added (2026-05-18) - OpenPros Speed-Shift Benchmark

- [patch] `clinical::imaging::reconstruction::sound_speed_shift`: add an
  OpenPros-style limited-view prostate SOS benchmark fixture and Criterion
  harness. The fixture builds a decimated 2-D SOS-shift phantom, top/bottom
  body-surface and rectal probe channels, finite-frequency 1 MHz sensitivity,
  and one shared fixed-acquisition frame, then compares dense and sparse
  reconstructions through the existing `SoundSpeedShiftPlan` API.

### Added (2026-05-18) - TFUScapes One-Case Import

- [patch] `pykwavers/examples/book/transcranial_planning/tfuscapes.py`: add a
  reproducible TFUScapes train-row-0 import path that validates `ct`, `pmap`,
  and `tr_coords`, derives the target from the paper pressure-map peak, fits
  paper transducer coordinates to the shared scene radius, routes the case
  through the existing skull-adaptive transcranial benchmark, and records
  structural geometry/output comparison metadata.

### Added (2026-05-18) - DG Bidirectional Acoustic Diagnostics

- [patch] `examples/dg_advection_diagnostics`: add a bidirectional linear
  acoustic characteristic fixture by evolving `w+` directly and evolving the
  reflected `w-` field through the same positive-advection DG core. The
  reconstructed standing-wave state verifies pressure, velocity, and acoustic
  energy against the analytical solution without changing the scalar DG RHS
  contract. Verified metrics: `pressure_relative_l2 = 1.651615e-4`,
  `velocity_relative_l2 = 1.547223e-2`, and `energy_ratio = 1.0`.

### Added (2026-05-18) - Chapter 29 Patient-Adaptive Transmit Scheduling

- [patch] `clinical::therapy::theranostic_guidance`: add a focused transmit
  schedule control surface with `full`, `uniform`, and `patient_adaptive`
  strategies plus an explicit transmit budget. The patient-adaptive schedule
  ranks CT-derived aperture elements by target sensitivity and greedy aperture
  diversity, then feeds the selected transmit subset through the existing
  matrix-free same-aperture inverse path.
- [patch] `pykwavers/examples/book/ch29_adaptive_transmit.py`: add an
  `adaptive_transmit` render scope that reuses the Chapter 29 brain, kidney,
  and liver cases and writes active Dice/CNR versus transmit budget metrics.

### Added (2026-05-18) - DG Acoustic Characteristic Diagnostics

- [patch] `examples/dg_advection_diagnostics`: extend the DG scalar fixture
  with the exact one-way linear-acoustic characteristic map
  `w+ = p + rho*c*u`, `w- = p - rho*c*u = 0`. The diagnostic reconstructs
  pressure and particle velocity from the evolved DG characteristic and checks
  the exact shifted acoustic solution. Verified metrics:
  `pressure_relative_l2 = 8.263806e-4`,
  `velocity_relative_l2 = 8.263806e-4`, `left_invariant_error = 0`, and
  `energy_ratio = 1.0`. This is the first acoustically valid DG comparison
  slice without claiming full bidirectional acoustic-system parity.

### Added (2026-05-18) - DG Scalar Discrepancy Diagnostics

- [patch] `examples/dg_advection_diagnostics`: add a real DG scalar-advection
  readiness diagnostic that advances the current nodal DG core against the
  analytical periodic solution and reports relative L2, quadrature-weighted
  mass error, phase error, and amplitude ratio. The bounded fixture records
  `relative_l2 = 8.263806e-4`, `mass_error = 4.873462e-16`,
  `phase_error_rad = 8.129815e-6`, and `amplitude_ratio = 9.999997e-1`.
  The example documents why DG is compared to scalar advection while
  `pstd_fdtd_comparison.rs` remains the acoustic FDTD/PSTD pressure-field
  comparison.

### Fixed (2026-05-18) - Spectral-DG Dimensional Completion

- [patch] `solver::forward::pstd::dg`: complete embedded 1-D, 2-D, and 3-D
  Spectral-DG execution by adding an explicit tensor-product DG topology,
  reusable physical-grid projection/reconstruction, tensor-product volume and
  periodic face RHS assembly, lower-dimensional discontinuity detection, and
  executable hybrid `solve_step_into`/`solve_step` APIs with reused spectral,
  DG, mask, and coupling buffers. The simulation DG adapter now accepts scalar
  physical grids whose active dimensions are divisible by `p + 1` instead of
  requiring the old line coefficient layout. Added dimensional projection,
  discontinuity, hybrid workspace, DG convergence, and adapter regressions.

### Fixed (2026-05-18) - DG Periodic RHS Conservation

- [patch] `solver::forward::pstd::dg`: correct the left-face
  Lax-Friedrichs residual sign in the scalar DG RHS, route DG time stepping
  through one extracted line/tensor-product RHS module, and preserve periodic
  quadrature-weighted global mass for both one-dimensional and tensor-product
  coefficient layouts. Added co-located conservation documentation with
  Hesthaven-Warburton, Cockburn-Shu, and Kopriva references, plus manufactured
  p=2 line and tensor-product regressions.

### Fixed (2026-05-17) - DG Shock-Capture Mass Conservation

- [patch] `solver::forward::pstd::dg`: make the DG troubled-cell limiter
  preserve the quadrature-weighted element mean represented by the diagonal
  mass matrix instead of the arithmetic node mean. Limited reconstructions now
  subtract the quadrature-weighted node centroid before applying the TVD slope,
  so flagged elements preserve the DG cell integral for nonuniform GLL weights.
  Updated DG limiter documentation and replaced the limiter regression with a
  polynomial-order-2 case that would fail under arithmetic-mean preservation.

### Fixed (2026-05-17) - Hybrid Coupling Quality and Compile-Blocker Cleanup

- [patch] `solver::forward::hybrid::coupling`: restrict region-coupling
  conservation and quality diagnostics to the active interface plane that is
  actually written to the target region. This closes the discrepancy where the
  affine conservation repair could place target integral into inactive
  region-buffer planes that were later discarded by `apply_to_target`. Added a
  manufactured two-region test that preserves the target pressure-plane
  integral, isolates non-pressure components, and verifies zero conservation
  error against the target interface trace.

- [patch] `clinical::therapy::theranostic_guidance::nonlinear3d`: update
  direct test fixtures for the explicit `SourceDomain` and `source_body_mask`
  contracts introduced by exterior-coupling source support. Source-plan tests
  now exercise both tissue-boundary and exterior-coupling source domains
  through the current constructor surface.

- [patch] `clinical::therapy::theranostic_guidance::nonlinear3d`: extend the
  abdominal nonlinear crop to include the focused-bowl standoff outside the
  planned skin contact and replace abdominal point injection with finite-area
  non-body exterior-coupling source patches. Chapter pressure diagnostics now
  record raw/body hotspot coordinates, hotspot-to-target distance,
  points-per-wavelength status, and source-plan support/delay metrics; the
  reduced KiTS19 histotripsy check reports target MI `2.55` with target/body
  peak ratio `0.513` while flagging the `0.290` PPW grid as a diagnostic-only
  under-resolved run.

- [patch] `clinical::imaging::reconstruction::real_time_sirt`: remove the
  duplicate acoustic row-norm helper left by overlapping edits while retaining
  the cached, documented row-norm preconditioner used by the acoustic SIRT
  update.

- [patch] `pykwavers/examples/book/ch29_theranostic_fwi_platforms.py`: render
  Figure 5 nonlinear pressure, FWI, cavitation, and fusion panels with the
  actual nonlinear 3-D aperture projection and target centroid rather than the
  Figure 2 planned 2-D beam overlay. The planned exposure panel still uses the
  Figure 2 planned aperture, so visual overlays now distinguish setup intent
  from the realized nonlinear source set.

- [patch] `clinical::therapy::theranostic_guidance::nonlinear3d::forward`:
  add source-delay regressions proving the focused delay law aligns arrivals
  at the target and delays high-speed skull paths relative to slower
  soft-tissue paths under the implemented scalar straight-ray slowness model.

- [patch] Chapter 29 controlled comparison metrics now report CT-frame
  nonlinear pressure-hotspot localization in meters, decomposed into planned
  beam-axis and cross-axis offsets. The geometry block also records planned
  versus realized nonlinear aperture axis angle and source-to-target distance
  statistics, so liver pressure error can be classified as prefocal/postfocal
  gain error or lateral aperture/phase error on the shared CT field of view.
  The pressure diagnostic helper now accepts both native 3-D volumes and
  projected 2-D target-slab fields with the same metric schema.

- [patch] Chapter 29 book generation now registers Python, extension, and
  MSYS UCRT DLL directories before loading `pykwavers.dll`, rejects stale PyO3
  extensions whose nonlinear signature lacks the current
  `treatment_window_radius_m` and `min_points_per_wavelength` controls, and
  supports `KWAVERS_CH29_OUT_DIR` for scratch artifact generation. A bounded
  comparison-scope smoke run completed brain, kidney, and liver nonlinear
  cases into `target/ch29-smoke` without overwriting production figures.

### Fixed (2026-05-17) - DG Shock-Capture Execution

- [patch] `solver::forward::pstd::dg`: route enabled shock capture through
  `SspRk3` and `ForwardEuler` time stepping. Each enabled SSP-RK stage now
  applies a conservative troubled-cell projection using solver-owned scratch:
  element means are preserved, intra-element oscillations are damped through a
  TVD limited linear reconstruction, and disabled shock capture leaves modal
  oscillations unchanged. Updated DG shock-capture documentation and added
  value-semantic tests for enabled and disabled paths.

### Fixed (2026-05-17) - Hybrid Conservation and DG Trait Solve

- [patch] `solver::forward::hybrid::coupling`: replace unit-sum pressure
  normalization with an affine interface projection that matches the target
  pressure integral exactly and matches target quadratic energy when the
  interpolated trace has nonzero variance. Shape mismatches now return a typed
  validation error. Added conservation tests for integral/energy matching,
  identical-interface idempotence, and mismatch rejection.

- [patch] `solver::forward::pstd::dg`: complete the `NumericalSolver` adapter
  by reconstructing updated modal coefficients back to the returned grid field
  after `solve_step`. Added a regression test comparing the trait-level result
  against explicit project/step/reconstruct execution and proving the returned
  field advances.

### Fixed (2026-05-17) - Hybrid Coupling and DG RK Workspace

- [patch] `solver::forward::hybrid::coupling`: enforce the canonical
  component-first field layout (`[field, x, y, z]`) for pressure-interface
  extraction and target writes. Single-region hybrid runs now skip coupling
  instead of fabricating a self-interface. Added value-semantic tests that
  reject component-last pressure reads/writes and preserve non-pressure fields.

- [patch] `solver::forward::pstd::dg`: add reusable SSP-RK workspaces for the
  original state, active stage, and RHS register. `SspRk3` and `ForwardEuler`
  now update modal coefficients in place after the initial projection, and the
  DG RHS path no longer allocates a face-residual vector per element/variable.
  Added a pointer-stability and constant-state invariant test.

### Fixed (2026-05-17) — Hybrid FDTD/PSTD and DG Audit

- [patch] `solver::forward::hybrid`: correct the hybrid transition weight from
  a PSTD-heavy boundary blend to a raised-cosine partition that gives FDTD full
  weight at the interface boundary and PSTD full weight in the smooth interior.
  `DomainRegion` is now `Copy`, allowing hybrid stepping and update paths to
  iterate by value instead of cloning the full region vector each step. Added a
  value-semantic test for the FDTD-to-PSTD weight contract.

- [patch] `solver::forward::pstd::dg`: remove redundant dense mass-matrix
  inversion from every DG `solve_step`; the differentiation and lift matrices
  already encode the inverse-mass action (`D = M^-1 S`, `LIFT = M^-1 E`).
  `RegionPSTDSolver` now preallocates `prev_field` at construction and uses a
  `has_prev_field` flag, eliminating the first-step `field.clone()` allocation.
  Added a pointer-stability test for previous-field reuse.

### Added (2026-05-17) — Solver Comparison Diagnostics

- [patch] `examples/pstd_fdtd_comparison`: replace the placeholder
  plugin-only demonstration with a real three-way pressure-field comparison
  between classical FDTD, k-space corrected FDTD, and PSTD on the same
  homogeneous Gaussian initial-value problem. The example documents the
  shared linear acoustic Cauchy problem, the half-step velocity alignment
  contract, and references CFL/FDTD/PSTD/k-space literature. It reports
  runtime, final-field L2 energy, relative L2 error, normalized max error,
  correlation, energy ratio, and centroid shift without writing
  output files. On the bounded debug fixture, classical FDTD differs from
  PSTD by `5.60099e-2` relative L2, while k-space FDTD matches PSTD at
  `7.25746e-16` relative L2.

### Fixed (2026-05-17) — Chapter 29 Histotripsy Pressure Path

- [patch] `clinical::therapy::theranostic_guidance::nonlinear3d`: replace the
  abdominal target-only nonlinear propagation crop with a target-to-skin path
  crop, reuse the planned abdominal skin contact as the nonlinear focused-bowl
  vertex, place abdominal sources in exterior coupling cells on the
  HistoSonics-like bowl, and steer each source by straight-ray slowness through
  the CT-derived sound-speed map. Chapter 29 pressure diagnostics now split raw
  global peak pressure from body-masked pressure so source/coupling drive peaks
  no longer masquerade as off-target focal pressure.

- [patch] `clinical::therapy::theranostic_guidance::nonlinear3d`: replace the
  explicit Westervelt `p*dtt(p)` feedback update with a finite-amplitude
  numerator/denominator cell update, bound additive source injection per source
  cell, preserve abdominal target-facing aperture order, and export raw
  pressure/MI diagnostics through the Chapter 29 artifacts. Regenerated Figure 5,
  Figure 6, `metrics.json`, `controlled_comparison_metrics.json`, and
  `controlled_comparison_fields.npz`; all three nonlinear target masks now exceed
  the inertial-cavitation MI threshold at finite pressure.

- [patch] `clinical::therapy::theranostic_guidance::nonlinear3d::cavitation`:
  normalize the Rayleigh-Plesset source density by the active treatment-window
  peak pressure instead of the global pressure peak. Excluded source/boundary
  lobes can no longer downscale valid in-window cavitation evidence.

- [patch] `clinical::therapy::theranostic_guidance::nonlinear3d::optimization`:
  record per-iteration FWI line-search diagnostics and allow a bounded
  single-parameter fallback at the smallest accepted scale after the coupled
  multiparameter update fails. The Chapter 29 metrics writer now serializes
  objective-before/objective-after, gradient norms, accepted scale, and accepted
  parameter block for nonlinear full runs.

### Added (2026-05-17) — Chapter 29 Elastic Shear Comparison

- [patch] `solver::inverse::same_aperture`: parameterize finite-frequency
  pitch-catch rows by `phase_speed_m_s` instead of hard-wiring the acoustic
  tissue speed. The same matrix-free operator now supports an explicit
  low-frequency shear comparison without cloning the inverse implementation.

- [patch] `clinical::therapy::theranostic_guidance`: add an exported
  `same_aperture_low_frequency_shear_wave_inverse` reconstruction channel with
  default frequencies `250/500/750 Hz` and shear speed `2.5 m/s`. PyO3 exports
  the reconstruction, metrics, frequencies, shear speed, and comparison-model
  metadata. Figure 2 and the controlled comparison scripts render the new
  elastic shear panel on the same full CT placement grid as the acoustic
  channels.

- [patch] Chapter 29 artifacts: regenerated Figure 1, Figure 2, and Figure 4
  with the rebuilt release PyO3 extension. Full Figure 6 regeneration remains
  blocked by a nonlinear brain-case process exit during
  `run_theranostic_nonlinear_3d_from_ritk`; no Python traceback was emitted.

### Added (2026-05-17) — Book Physics Structural Splits

- [patch] `physics::analytical::wave` directory split: replace monolithic `wave.rs` (641 lines) with
  `wave/{mod,bessel,dispersion,linear,nonlinear,tests}`. `bessel.rs` (120 lines) — `bessel_j0`,
  `bessel_j1_clean`, `bessel_jn` (Miller downward recurrence, two-buffer normalisation),
  `pub(crate) jn`; `dispersion.rs` (84 lines) — CFL, phase-error, k-space correction functions;
  `linear.rs` (118 lines) — plane-wave, spherical-wave, reflection/transmission, attenuation
  functions; `nonlinear.rs` (110 lines) — Fubini harmonic series, Westervelt harmonic evolution,
  shock-formation distance; `mod.rs` (27 lines) — re-exports; `tests.rs` (57 lines) — all wave tests.
  Dead code removed: `bessel_j1` (double-sign error in negative-x path) and `bessel_j1_n`
  (normalisation bookkeeping bug, marked "not quite right" in original) both eliminated.
  `fubini_harmonic_amplitude` now routes exclusively through the clean `jn` Miller-recurrence
  driver, activating the correct two-buffer normalisation path.

- [patch] `physics::analytical::cavitation` directory split: replace monolithic `cavitation.rs`
  (586 lines) with `cavitation/{mod,dynamics,histotripsy,power_spectrum,tests}`.
  `dynamics.rs` (218 lines) — Minnaert resonance, Blake threshold, Rayleigh collapse time,
  Rayleigh-Plesset and Keller-Miksis RK4 integrators; `histotripsy.rs` (85 lines) —
  `mechanical_index`, `inertial_cavitation_dose`, `histotripsy_lesion_radius_m`;
  `power_spectrum.rs` (88 lines) — `bubble_power_spectrum` (O(N²) DFT), `period_doubling_ratio`;
  `mod.rs` (18 lines) — re-exports; `tests.rs` (142 lines) — all 12 cavitation tests.

- [patch] `physics::analytical::rtm` directory split: replace monolithic `rtm.rs` (526 lines) with
  `rtm/{mod,backprop,beam,condition,temporal,tests}`. `backprop.rs` (91 lines) — 2-D and 3-D
  Green's function back-propagators; `beam.rs` (67 lines) — Gaussian beam with skull transmission
  and standing-wave factor; `condition.rs` (142 lines) — Claerbout cross-correlation, multi-frequency
  fusion, Guitton source-normalized condition, aperture-weighted fusion; `temporal.rs` (41 lines) —
  modulation frequencies and suppression gain; `mod.rs` (21 lines) — re-exports; `tests.rs`
  (135 lines) — all 13 RTM tests.

### Fixed (2026-05-17) — Book Physics and Test Correctness

- [patch] `physics::analytical::wave::nonlinear::fubini_harmonic_amplitude`: eliminate dead
  `bessel_j1_n` call (known-buggy normalisation); route through `jn` (Miller two-buffer
  recurrence). Fixes silent incorrect Bessel output for harmonic orders n ≥ 2.

- [patch] `solver::forward::nonlinear::westervelt::tests::absorption_causes_amplitude_decay_not_growth`:
  replace inaccessible direct field assignment `medium.absorption = 5.0` with public setter
  `medium.set_acoustic_properties(5.0, 1.0, medium.nonlinearity).unwrap()`.

- [patch] `physics::analytical::imaging::compounding_narrower_than_single`: fix assertion from
  `psf4[0] > psf1[0]` (wrong direction) to `psf4[0] < psf1[0]`. Plane-wave compounding
  narrows the PSF: eff_width_4 = 0.886·F#·λ/√4 = 0.665 mm; sinc²(u₄) ≈ 0.088 < sinc²(u₁) ≈ 0.613
  at x = 0.5 mm.

### Added (2026-05-17) — Image Reconstruction Optimization

- [patch] `solver::inverse::seismic::brain_helmet::volume_born::pcg`:
  replace `smooth_active_values_3d` 3-D neighbourhood scan (O(N·(2r+1)³)) with
  three separable 1-D prefix-sum box-filter passes (O(N + 6·NX·NY·NZ)).  The
  Z-axis pass uses Rayon `par_chunks_mut` for cache-friendly parallelism.  For
  r=2 on a 56³ active region the inner-loop work falls ~12×.  A parallel
  indicator array preserves the existing semantics: only active-voxel values
  contribute to each neighbourhood average.  Removed the `ndarray::Array3`
  index table that required O(NX·NY·NZ) initialisation.

- [patch] `math::linear_algebra::iterative::lsqr::matfree`: new module adding
  `MatFreeOperator` trait (`rows`, `cols`, `matvec`, `t_matvec`) and
  `solve_lsqr_matfree` — damped LSQR (Paige & Saunders 1982) over any
  operator implementing the trait.  Uses only `Vec<f64>` scratch buffers (no
  ndarray); carries `objective_history` (0.5·φ̄² per iteration) for convergence
  monitoring.

- [patch] `clinical::imaging::reconstruction::sound_speed_shift`: add
  `ShiftPrior::Lsqr { damping: f64 }` as a third solver option alongside Dense
  (PCG) and Sparse (ISTA).  The new `solver::lsqr::solve_shift_lsqr` wraps
  `SoundSpeedShiftOperator` via `ShiftOperatorAdapter: MatFreeOperator` and
  calls `solve_lsqr_matfree`.  All match arms on `ShiftPrior` updated; `Eq`
  derive removed from `SoundSpeedShiftBatchStreamSummary` (f64 field).

- [patch] removed stale monolithic `physics::analytical::cavitation.rs` (superseded
  by `cavitation/` split in a prior session) which was causing an E0761
  module-conflict error.

### Added (2026-05-17)

- [patch] `nonlinear3d::stencil`: replace `nonlinear_term` (3-level, used
  P^n-2/older) with `westervelt_cell_terms` returning `WesterveltCellTerms`
  struct. New form: `D[n] = 1 − 2βp[n]/(ρc²)`, `N[n] = c²dt²Lp[n] +
  2β(p[n]−p[n-1])²/(ρc²)`. Denominator D[n] > 0 at histotripsy drives;
  reduces to linear acoustic update when β=0; eliminates need for older buffer.

- [patch] `nonlinear3d::adjoint`: remove `adj_older` from `AccumulateInput`
  and `accumulate_step`; update gradient computation to `westervelt_cell_terms`;
  remove `older_for_step` usage; consistent with 2-level forward stencil.

- [patch] `nonlinear3d::checkpoint`: remove dead `older_for_step` from
  `HistorySegment` (pub(super), never called).

- [patch] `nonlinear3d::forward::stencil`: remove dead `older` field from
  `UpdateCells` struct; Westervelt 2-level scheme reads only current and previous.

### Fixed (2026-05-17)

- [patch] `solver::forward::nonlinear::westervelt`: corrected the 4th-order
  finite-difference Laplacian coefficient placement, implemented the documented
  6th-order stencil, changed unsupported `spatial_order` from silent downgrade
  to typed validation error, and reused the nonlinear-term and next-pressure
  workspaces during updates. Added quadratic-field stencil exactness,
  unsupported-order rejection, and workspace-reuse tests.

- [patch] `solver::forward::thermal_diffusion::solver`: reject unsupported
  finite-difference `spatial_order` values with a typed validation error instead
  of silently mutating the configuration to second order. Documented the
  centered-stencil quadratic exactness theorem and added value tests for O4
  quadratic exactness plus invalid-order state preservation.

- [patch] `solver::forward::nonlinear::westervelt_spectral`: remove per-step
  pressure-history clones from `update_wave`, remove the unused `B/A` field
  allocation, and document the borrowed three-buffer leapfrog role invariant.
  Added ring-buffer permutation and pointer-stability zero-state update tests.

- [patch] `solver::forward::nonlinear::westervelt_spectral`: add solver-owned
  nonlinear and damping workspaces, route nonlinear and viscoelastic terms
  through caller-owned `_into` kernels, compute the damping stencil without a
  full `dp_dt` temporary, fold source amplitude into the final update loop
  instead of allocating `src_term`, and stop charging source-mask time to
  k-space metrics.

- [patch] `solver::forward::nonlinear::westervelt_spectral`: remove per-step
  shear/bulk viscosity coefficient array allocation from the damping path by
  reading `Medium::shear_viscosity` and `Medium::bulk_viscosity` pointwise in
  the stencil. This also restores homogeneous-medium damping, which the
  inherited zero-valued `ElasticArrayAccess` defaults had suppressed.

- [patch] `solver::forward::nonlinear::westervelt_spectral` and
  `domain::source`: add caller-owned source-mask sinks
  `Source::create_mask_into` / `Source::add_mask_into`, route the spectral
  Westervelt update through solver-owned `source_mask_scratch`, and implement
  allocation-free direct mask writes for all in-crate source implementations.
  Added owned-vs-reused mask equivalence and source-mask pointer-stability
  tests.

- [patch] `domain::source`: override `get_source_term` for point,
  time-varying, composite, and null sources so per-cell wave-equation RHS
  assembly no longer inherits the allocating full-mask fallback. Composite
  source terms now sum child-local terms instead of multiplying the summed mask
  by the summed amplitude. `SimpleCustomSource::get_source_term` now matches its
  grid-cell mask semantics rather than selecting the nearest source position,
  and `TimeVaryingSource` no longer stores its waveform vector twice.

- [patch] `solver::forward::hybrid`: add solver-owned
  `source_mask_scratch` and route update-time pressure source injection through
  `Source::create_mask_into`, removing the per-step mask allocation from the
  hybrid PSTD/FDTD update path. Added a full update-path pointer-stability test.

- [patch] `docs/book/cavitation_and_bubbles.md` Theorem 7.6 proof: integrand
  was `R dR` but correct form from energy integral ṙ²=(2p∞/3ρL)(R₀³/R³−1) is
  `R^(3/2) dR`; beta function corrected from `B(2/3, 1/2)` to `B(5/6, 1/2) ≈
  2.241`; gamma values updated (Γ(5/6)≈1.129, Γ(4/3)≈0.893); coefficient
  updated 0.9146→0.9147 matching Rayleigh (1917).

- [patch] `docs/book/cavitation_and_bubbles.md` §7.4.2 Blake threshold derivation:
  exponent typo `R₀^0` → `R₀^3` (dimensional consistency).

- [patch] `docs/book/cavitation_and_bubbles.md` §7.15 Minnaert summary table:
  `p_g0` → `p_0` with "(large-bubble limit)" note; §7.16 validation claim
  corrected from "matches (7.6)" to "matches large-bubble approx (7.7); valid
  for R₀ ≳ 1 μm; (7.6) required below 1 μm".

- [patch] `docs/book/abdominal_histotripsy_fwi.md`: Rayleigh-Plesset subharmonic
  source sign error — vapor pressure was `+p_v` (unphysical, promotes collapse
  acceleration) → `−p_v` (correct: vapor pressure opposes bubble contraction).

- [patch] `kwavers/src/physics/analytical/cavitation.rs` `rayleigh_collapse_time_s`:
  add derivation comment `B(5/6,1/2)·√(3/2)/3 = Γ(5/6)Γ(1/2)/Γ(4/3) ≈ 2.241
  → 0.9147`.

### Added

- [patch] `physics::analytical::rtm`: add `backprop_green_function_3d` (3-D spherical
  1/(4πr) amplitude law), `rtm_source_normalized_condition` (Guitton 2007
  source-amplitude-free imaging condition for skull shadow zones), and
  `rtm_aperture_weighted_fusion` (per-element solid-angle/transmission-weighted
  migration fusion). Eight new value-semantic tests covering exact amplitude
  ratios, shadow-zone bias removal, equal-weight equivalence, and zero-weight
  fallback. Applies to 1024-element transcranial RTM brain imaging.

- [patch] `physics::analytical::cavitation`: add `mechanical_index` (FDA MI = |P−| [MPa]
  / √f [MHz]; Apfel & Holland 1991), `inertial_cavitation_dose` (collapse-event
  weighted sum (R_max/R₀)³; Duryea et al. 2015), `histotripsy_lesion_radius_m`
  (energy-balance Rayleigh-collapse model R_L = R₀·(P₀·ICD/σ_y)^(1/3); Maxwell
  et al. 2011, Vlaisavljevich et al. 2015), and `period_doubling_ratio`
  (subharmonic S(f₀/2)/S(f₀) via ±1-bin spectral integration; Cramer et al. 2021).
  Fix dead-code double-assignment in `bubble_power_spectrum`. Seven new
  value-semantic tests including zero-ICD guard, strong-drive collapse detection,
  cube-root ICD scaling, and spectral ratio monotonicity.

- [patch] `nonlinear3d::aperture`: replace azimuth-only `sort_by_spherical_angle`
  + 1-D stride `select_evenly` in `brain_candidates` with Fibonacci golden-angle
  sphere lattice (`fibonacci_sphere_select`). Theorem: golden angle φ_g ≈ 2.400
  rad minimises maximum nearest-neighbour angular gap over N points on a sphere
  (Álvarez 2001). For 1024 elements this reduces coherent grating lobes in RTM/FWI
  reconstructions vs the prior azimuth-sorted stride. Calvarium HU threshold
  relaxed from 250 to 200 to include cancellous bone.

- [patch] `synthetic::brain`: add stratified brain anatomy — gray matter cortex
  (HU 37, r_inner > 0.92), white matter (HU 28, interior), CSF lateral ventricles
  (HU 10, bilateral ellipsoidal), thalamic nuclei (HU 38, bilateral deep-brain
  histotripsy target at stereotaxic centre). Replaces single-HU brain interior with
  a five-class tissue model that correctly represents c and α variations relevant
  to transcranial wave propagation and RTM imaging.

### Changed

- [patch] Ch27 (seismic_fwi_brain_imaging): add Theorem 27.1 Born Linearity with
  formal proof (single-scatter assumption linearizes p_s = Am + O(m²); AᵀA+λI
  strictly positive-definite for λ>0); add Born validity scope and skull exclusion
  rationale; add §27.7 Minimal usage example for run_seismic_helmet_fwi_volume_from_ritk_ct;
  replace "Pending" elastic FWI table entry with explicit deferral note; renumber
  §27.8→27.9.

- [patch] Ch29 (theranostic_fwi_platforms): add Definition: Same-Device Aperture
  Contract (E={e_k}, all tx/rx indices in {1,…,N}); add Theorem: Same-Aperture
  Operator Rank (m ≤ N²); add Period-doubling observable derivation via Floquet
  theory (δ_PD = max_t|R(t)−R(t−T)|/R₀ isolates inertial cavitation); add minimal
  PyO3 usage example.

- [patch] Ch5 (diagnostics): add complete Cramér-Rao Jacobian (Φ=πv/v_max →
  dv/dΦ=v_max/π → Var(v̂)=v_max²(1−|R(1)|²)/(π²M|R(1)|²)); add explicit kr≫1
  far-field condition (r≳1mm for R₀=2μm, f=2MHz); add shell stiffness scope note.

- [patch] Ch7 (theranostics): inline CEM43 dose rate definition and R^x>0 proof
  in Theorem 7.6; rename Corollary 7.1 to "Irreversibility"; specify 4-state
  KalmanFilter vector [T,D,ρ_b,c_s]; add dilute-bubble scope in Theorem 7.7.

- [patch] Ch31 (clinical_device_geometry): fix bowl radius theorem to
  R = d_f/cos(θ_max) (was 1.15×‖F-S‖); rewrite algorithm section to match actual
  exterior flood-fill BFS + approach-angle penalty (W_z=4.0, W_y=6.0);
  fix ch31 metrics channel_keys to match Rust dict keys without _reconstruction suffix.

- [patch] Ch6 (therapy): fix ΔT unit (°C total, not °C/s); simplify focal gain
  formula to G = ka²/(2R_f); correct 479→60-80°C explanation (thermal conduction
  + perfusion, not "thermal diffusion"); fix CEM43 time units (1s = 1/60 min →
  CEM43 ≈ 2184 min, not 131,072 min).

- [patch] Ch3 (nonlinear_acoustics): expand Theorem 3.6 Burgers proof (explicit
  τ integration with radiation BC, 2p∂p/∂τ identity); add Jacobi–Anger–Kepler
  identity proof sketch (3.23); expand second-harmonic growth proof (P₂ ∝ z via
  driven wave equation); fix Kuznetsov Eq. 3A.1 non-standard "·2" notation.

- [patch] Added a Chapter 29 controlled linear-vs-nonlinear comparison path:
  the generator now runs a matched linear case at the nonlinear grid, element
  count, frequency, and pressure, evaluates both branches on the nonlinear
  crop projection, writes `fig06_controlled_linear_nonlinear_comparison`,
  `controlled_comparison_metrics.json`, and `controlled_comparison_fields.npz`,
  and records the measured pressure-spread/aperture-residual explanation in the
  chapter text and metrics manifest. The nonlinear histotripsy path now gates
  Rayleigh-Plesset cavitation by mechanical index, preserves per-element source
  drive before calibration, reports actual realized aperture counts, expands the
  brain cap aperture to the requested element count when skull-labeled grid
  cells are sparse, and uses histotripsy-scale drive for all three Chapter 29
  targets.

- [patch] Refined the Chapter 29 passive cavitation inverse and controlled
  diagnostics: the passive subharmonic Green inverse now solves only over the
  MI-gated Rayleigh-Plesset source support, handles empty source support
  explicitly, and removes an unused nonlinear aperture helper. The controlled
  comparison now archives the projected `nonlinear_cavitation_source` field and
  records hotspot distance-to-target plus source-to-reconstruction metrics.
  Regenerated metrics show mean common-grid nonlinear fusion Dice `0.565` vs
  linear `0.331`, while mean MI-gated source outside-target energy `0.949`
  localizes the remaining cavitation failure before passive inversion.

- [patch] Updated Chapter 29 Figure 6 to match the Figure 2/Figure 5 context
  layout: the controlled comparison now renders every row in the full-resolution
  CT placement frame, including target/body contours, therapy tx/rx elements,
  central imaging receivers, focus, and skin-contact marker. The compressed
  controlled field archive now stores `ct_frame_*` display fields, so linear
  and nonlinear reconstructions share the same CT pixel grid instead of a
  smaller nonlinear subsection.

- [patch] Added `CANONICAL_BRAIN_SCENE` as the CT-aligned source of truth for
  the Chapter 25 brain target and transducer pose. Figure 2 phase correction,
  Chapter 29 Figure 5 brain nonlinear simulation, 3-D helmet placement, and the
  skull-adaptive benchmark now consume the same target fraction, 1024-element
  helmet geometry, cap angles, acoustic speeds, pressure scale, and HU
  thresholds instead of deriving independent centroids or defaults.

- [patch] Optimized the Chapter 29 nonlinear 3-D FWI iteration workspace:
  `run_fwi` now reuses one residual trace buffer across source-encoded shots,
  and `apply_line_search` owns a `LineSearchWorkspace` that overwrites
  candidate `c`/`beta` buffers in place for each backtracking scale instead of
  allocating two model vectors per candidate. Removed stale pykwavers checkpoint
  imports so the PyO3 crate checks without warning noise on this path.

- [patch] Hardened the Chapter 25 GBM modality bridge for imperfect modality
  sets: CT-space segmentation no longer fabricates T1-Gd/FLAIR paths, CT-backed
  BBB planning accepts real CT plus segmentation only, MRI-space planning
  accepts segmentation plus any real in-space MRI reference, and the manifest
  records the 2025 Holder-MI incomplete-MRI and TextBraTS references as
  contract-level design inputs. Focused transcranial planning tests pass 24/24
  and the Chapter 25 executable completes with CT-space GBM BBB metrics.

- [major] Replaced the public P-STD
  `PSTDSolver::run_orchestrated_with_thermal(...)` positional argument list
  with `ThermalOrchestrationInput<'_>`. The equations and loop order are
  unchanged; the public call site now names acoustic step count, thermal
  solver, thermal medium, center angular frequency, thermal timestep, coupling
  ratio, volumetric heat capacity, and background heat rate explicitly.

- [patch] Corrected and optimized the Chapter 29 nonlinear 3-D cavitation path:
  passive cavitation data now uses the active-voxel source vector required by
  `PassiveOperator` column indexing, the Rayleigh-Plesset period-doubling
  observable uses a one-period radius ring buffer equivalent to full-history
  indexing, and the passive Tikhonov inverse reuses prediction/residual/gradient
  workspaces through `apply_into`/`normal_gradient_into` instead of allocating
  per iteration. Completed the split-module cleanup exposed by fresh Rust
  diagnostics so canonical directory modules are the only module definitions.

- [patch] Optimized same-aperture inverse solver (`solver::inverse::same_aperture`):
  `dot` and `axpy` kernels use 8-lane unrolled accumulators enabling compiler AVX2
  auto-vectorization (`vmulps`/`vfmadd231ps`); `EncodedOperator::normal_diag`
  parallelized via Rayon fold-reduce (one `cols`-length acc per thread, no per-row
  allocation); PCG inner loop eliminates `collect::<Vec<_>>()` for `r`/`z`
  initialization in favour of in-place writes; diagonal preconditioner factored
  into a 4-unrolled `apply_preconditioner` helper. All 6 same_aperture unit tests
  pass; zero new warnings.

- [patch] Removed stale monolithic `pykwavers/src/simulation_py/gpu/session.rs`
  (superseded by `session/` directory split into `construction.rs`, `control.rs`,
  `run.rs`). Removed phantom `PlacementPoint3` type from `theranostic_bindings/helpers.rs`
  and updated `inverse.rs` callers to use the canonical `points3_to_array`. Both
  crates compile with zero errors.

- [patch] `helmet3d.rs`: extracted named constants (`CAP_UNIT_Z_MIN`, `CAP_UNIT_Z_MAX`,
  `HELMET_SKIN_MARGIN_M`, `HELMET_RADIUS_MIN_M`, `BEAM_SAMPLE_COUNT`, `BEAM_TRACE_STEPS`,
  `ORIENTATION_PROBE_FRACTION`); replaced fragile single-slice superior-orientation
  heuristic with bottom-quartile vs top-quartile body-area comparison; added empty-calvarium
  guard and module-level theorem documenting assumptions; documented `body_radius` 3-D
  Euclidean semantics and `calvarium_cap_elements` Fibonacci distribution invariant.

- [patch] `solver.rs`: extracted `FUSED_WEIGHT_ACTIVE` (0.40), `FUSED_WEIGHT_SUBHARMONIC`
  (0.25), `FUSED_WEIGHT_HARMONIC` (0.20), `FUSED_WEIGHT_ULTRAHARMONIC` (0.15), and
  `FUSED_GATE_FLOOR` (0.25) as named constants with derivation documented in module
  header; removed anonymous floating-point literals from `fuse_maps`.

- [patch] Removed unused import `flat_index` from
  `clinical/therapy/theranostic_guidance/nonlinear3d/westervelt/fwi.rs`.
  Zero warnings in kwavers library.

- [patch] Split `pykwavers/src/bubble_bindings.rs` (593 lines) into
  `bubble_bindings/{mod,rayleigh_plesset,keller_miksis,cem43,arrhenius,hodgkin_huxley}.rs`;
  wired `register_bubble` into `lib.rs`; all six physics functions now exposed in pykwavers.
  pykwavers compiles clean with zero errors.

- [patch] Split `apollo-fft` 500-line structural violations: `dimension_3d.rs` (1,899 lines)
  into `dimension_3d/{mod,precision_bridge,axis_pass_f64,axis_pass_f32,r2c,r2c_row,
  r2c_axis_pass,tests}.rs`; `radix_composite.rs` (868 lines) into
  `radix_composite/{mod,stage,tests}.rs`; `dimension_2d.rs` (739 lines) into
  `dimension_2d/{mod,axis_pass,tests}.rs`. All files ≤ 500 lines. `cargo check -p apollo-fft`
  and `cargo check -p kwavers` both clean.

- [patch] Parallelized all sequential nested loops in acoustic-thermal pipeline:
  `acoustic_heat_source()` in `physics::acoustics::conservation::heat` now uses two
  `Zip::par_for_each` passes (kinetic energy + heat source), replacing 3-nested
  sequential loops; `ThermalAcousticCoupler::update_material_properties()`,
  `compute_acoustic_heating()`, `step_thermal()`, and `step_acoustic()` in
  `solver::forward::coupled::thermal_acoustic` replaced with `Zip::par_for_each`
  and field-destructured `Zip::indexed().par_for_each` patterns. All 15
  heat-source and thermal-acoustic tests pass.

- [minor] Implemented PSTD → thermal coupling API: added `alpha_si: Array3<f64>`
  to `AbsorptionKernel` (stored per-cell during PowerLaw/Stokes initialization);
  added `alpha_np_m: Array3<f64>` to `PSTDSolver`; added
  `populate_alpha_np_m_at_frequency(omega_c)` to scale α_SI → α(ω_c) [Np/m],
  `set_alpha_np_m(alpha)` for external override, and
  `compute_acoustic_heat_source() → Array3<f64>` that calls the existing
  `acoustic_heat_source()` function — all in new
  `orchestrator/thermal.rs`. Callers pass the result to
  `ThermalDiffusionSolver::update(medium, grid, dt, Some(q.view()))`.

- [minor] Wired acoustic→thermal coupled time loop into pykwavers `Simulation`:
  added `PSTDSolver::run_orchestrated_with_thermal(ThermalOrchestrationInput {
  ... })` to `orchestrator/thermal.rs`; added
  `ThermalCouplingConfig` and `run_pstd_with_thermal_impl()` to
  `pykwavers/src/simulation_py/thermal_coupling.rs`; exposed
  `Simulation.set_thermal(center_frequency, n_acoustic_per_thermal, …)`,
  `Simulation.clear_thermal()`, and `Simulation.has_thermal` to Python;
  `Simulation.run()` with `SolverType.PSTD` and thermal config attached now
  runs the coupled loop and populates `result.thermal_temperature` (°C,
  nx×ny×nz) and `result.thermal_dose` (CEM43 min, nx×ny×nz). Both crates build
  clean with zero errors.

- [patch] Fixed pre-existing `Nonlinear3dResult` struct initializer in
  `nonlinear3d/mod.rs` missing `source_dimensions`, `source_spacing_m`,
  `crop_bounds_index` fields (values taken from `volume`).

- [patch] Parallelized axisymmetric velocity update in PSTD solver
  (`propagator/velocity.rs`): changed two `Zip::for_each` to `Zip::par_for_each`
  for the `ux` and `uz` updates in `update_velocity_as`, consistent with the 84
  parallel loops in all other PSTD spatial operations.

- [patch] Closed the nonlinear 3-D volume and absorption 500-line structural-limit
  gaps by splitting `clinical::therapy::theranostic_guidance::nonlinear3d::volume`
  (521 lines) into `volume/{mod,validation,bbox,mask,resample,material,
  attenuation,centroid}` and `clinical::therapy::theranostic_guidance::nonlinear3d::absorption`
  (555 lines) into `absorption/{mod,construction,spectrum,apply,tests}` while
  preserving the public `prepare_volume`/`centroid_index` paths and the
  `FractionalLaplacianAbsorption`/`AbsorptionBuilder` API. Every nonlinear3d
  source file is now `<= 500` lines. Focused verification: 20/20 default
  nonlinear3d tests pass, 3 Tier-2 ignored tests remain ignored, the heavy
  `nonlinear_3d_westervelt_fwi_and_cavitation_inverse_are_input_sensitive`
  integration test passes, and `cargo clippy -p kwavers --lib --no-deps
  -- -D warnings` is clean.

### Added

- [minor] Added a Chapter 25 skull-adaptive transcranial benchmark grounded in
  the existing brain FUS workflow. `kwavers` now exposes
  `run_skull_adaptive_transcranial_benchmark` for CT-conditioned helmet
  aperture selection, skull-aware reference pressure, uncorrected baseline
  pressure, and TFUScapes-aligned relative-L2, focal-position, and max-pressure
  metrics. `pykwavers` exposes
  `run_transcranial_skull_adaptive_benchmark_from_ritk_ct`, plus a book helper
  that summarizes the benchmark without duplicating the planning pipeline.

- [minor] Added Treeby-Cox 2010 fractional-Laplacian power-law absorption
  to the nonlinear 3-D theranostic Westervelt FDTD
  (`kwavers::clinical::therapy::theranostic_guidance::nonlinear3d::absorption`).
  The operator is precomputed per simulation via Apollo R2C FFTs with a
  per-voxel `dt·τ = dt·2·α₀_ω·c^(y+1)` array and a half-spectrum `|k|^y`
  filter (median-`y` global exponent). The η Kramers-Kronig dispersion
  term is dropped because explicit `−dt²·η·L_{y+1}(p[n])` is conditionally
  unstable for `y < 2` (von-Neumann analysis); the τ magnitude alone
  matches Treeby-Cox 2010 exactly (verified by a Tier-2 plane-wave decay
  test that fits `α(f) = α₀·f_MHz^y` against the discrete forward).
  Plumbed through `ForwardInput` and `ReplayInput` and applied bit-for-bit
  identically in `forward_with_schedule`, `forward_dense_history_for_test`,
  and `replay_history_segment_into` so checkpoint replay reproduces the
  lossy forward state exactly. Closed under the self-adjoint
  `apply_transpose` in `adjoint::gradient` so the discrete adjoint of the
  Westervelt FWI remains exact. Inner-product transpose identity verified
  to 1e-9. Brain helmet and abdominal FWI integration tests pass with the
  realistic skull and soft-tissue attenuation now active in the forward.

- [patch] Added Chapter 27, "Seismic Full-Waveform Brain Imaging", with a
  CT-derived 1024-element helmet FWI core in `kwavers`, a RITK-backed
  `pykwavers` wrapper, an executable RIRE CT reconstruction script, generated
  single-slice and multi-slice book figures, and value-semantic Rust
  verification.
- [minor] Extended Chapter 27 with a bounded `56^3` volume default,
  five-frequency continuation, eight encoded receiver offsets,
  weak-Westervelt second-harmonic encoded rows, Sobolev update conditioning,
  a matrix-free 3-D helmet operator, a regularized FWI display image,
  twelve nonempty reconstructed volume slices, centroid ROI visualization,
  seven generated figure pairs, and synchronized metrics for the expanded
  nonlinear simulation.
- [minor] Replaced the Chapter 27 2-D acquisition ring with a deterministic
  1024-element hemispherical cap, 3-D source/receiver path lengths, slice axial
  offsets from CT spacing, and figure metadata that labels the geometry as a
  hemispherical-cap simulation.
- [minor] Updated Chapter 27 figures and acquisition physics so the multi-slice
  stack shows raw CT HU above the CT-derived acoustic target, and the encoded
  sensitivity matrix includes CT-derived skull/soft-tissue attenuation.
- [minor] Replaced Chapter 27 slice-wise inversion with one matrix-free 3-D
  helmet inversion over a resampled CT volume, exposed
  `pykwavers::run_seismic_helmet_fwi_volume_from_ritk_ct`, sliced the returned
  3-D arrays for the multislice and centroid ROI figures, and regenerated
  metrics with `inversion_dimensionality = "3d_volume"`.
- [patch] Improved Chapter 27 3-D reconstruction by replacing the volume
  Landweber loop with projected diagonal-preconditioned CG, caching matrix-free
  row norms and row constants across objective/gradient/normal-operator
  applications, adding stage-boundary Charbonnier edge-preserving proximal
  regularization, and making `fig06` display a target-independent
  mask-regularized FWI row derived from the reconstructed 3-D volume.
- [patch] Added Chapter 27 histotripsy-monitoring subchapters that separate
  RTM, linear time-lapse FWI, multiparameter attenuation FWI, tissue-harmonic
  FWI, passive cavitation source inversion, bubble-dynamics nonlinear FWI, and
  elastic/shear FWI by data stream, inverted quantity, and implementation
  status.
- [minor] Added a custom Chapter 27 histotripsy monitoring simulation that
  uses the RITK-backed CT baseline, a 1024-element receive aperture,
  110/160/220 kHz frequency continuation, deterministic receiver noise,
  Huber-robust normal FWI, multiparameter speed/attenuation FWI, weak nonlinear
  harmonic FWI, passive 110/220/440 kHz RTM, 110 kHz subharmonic
  cavitation-source FWI, and frequency-gated fusion, generating figures 8-10
  plus metrics.
- [minor] Added Chapter 28 abdominal histotripsy FWI analysis for the existing
  KiTS19 kidney and LiTS liver CT examples, with a CT-textured anatomical
  support mask, finite-frequency Born operator, H1-regularized
  diagonal-preconditioned CG targeting reconstruction, time-lapse lesion-state
  reconstruction, generated kidney/liver figure panels, metrics, and
  value-semantic Python tests.
- [minor] Expanded Chapter 28 to use the largest tumor-centered CT field of
  view, a HistoSonics-like 256-element therapy aperture with a central imaging
  receiver line, a 750 kHz upper continuation frequency, hybrid
  therapy/imaging FWI rows, 18 PCG iterations, and
  regenerated FOV/geometry-aware metrics and figures.
- [minor] Added Chapter 28 subharmonic cavitation-source FWI and weak-nonlinear
  harmonic FWI channels, with shared operator assembly, shared PCG inversion,
  second figure panels per case, metrics, documentation, and value-semantic
  tests.
- [minor] Replaced Chapter 28 reduced nonlinear source maps with bounded 2-D
  Westervelt FDTD pressure spectra and Rayleigh-Plesset subharmonic bubble
  response driven by simulated lesion pressure; synchronized figures, metrics,
  and documentation to state that the receiver inversion remains reduced 2-D
  synthetic FWI rather than measured hardware data or full 3-D adjoint
  nonlinear FWI.
- [minor] Added Chapter 29 same-device therapeutic ultrasound finite-frequency
  inverse/RTM
  simulations for brain, kidney, and liver CT scenarios through the
  RITK-backed `pykwavers.run_theranostic_inverse_from_ritk` wrapper. The Rust
  clinical theranostic-guidance workflow now exports INSIGHTEC-like helmet and
  HistoSonics-like skin-coupled geometries, pressure-calibrated exposure
  fields, active finite-frequency Born inversion, passive subharmonic
  inversion, harmonic and ultraharmonic rows, linear acoustic RTM, uncropped
  full-patient placement context for
  abdominal CT slices, a calvarium-limited 3-D helmet placement figure with
  sampled skull beam intersections, aperture placement metrics, generated CT
  placement and reconstruction figures, and research-aligned documentation.
- [minor] Added a separated Chapter 29 nonlinear 3-D branch exposed as
  `pykwavers.run_theranostic_nonlinear_3d_from_ritk`, with CT-derived bounded
  3-D volume preparation, skin/calvarium source-receiver placement,
  heterogeneous Westervelt FDTD propagation, discrete-adjoint sound-speed FWI,
  Rayleigh-Plesset cavitation source simulation, passive subharmonic
  nonnegative inversion, value-semantic Rust verification, and
  `fig05_nonlinear_3d_westervelt_rp_cavitation`.
- [minor] Improved the Chapter 29 nonlinear 3-D branch with deterministic
  source-encoded Westervelt shots, multiparameter `c` and `beta`
  discrete-adjoint updates, CT/segmentation-derived target-ROI inversion masks,
  body-restricted `H1` regularization,
  Sobolev-smoothed gradients, Rust-side nonlinear FWI/cavitation fusion
  metrics, a higher-resolution Figure 5 layout aligned with Figure 2, and PyO3
  controls for source encoding and regularization.
- [patch] Regenerated Chapter 29 Figure 5 with the same per-case simulation
  grids as Figure 2: `48^3` for brain and `52^3` for kidney/liver nonlinear
  Westervelt/Rayleigh-Plesset runs. The Figure 5 script now inherits each case
  grid unless an explicit nonlinear-grid environment override is present.
- [patch] Optimized the Chapter 29 nonlinear 3-D Westervelt adjoint by
  replacing the dense `(steps + 1)` adjoint-state allocation with four rolling
  adjoint volumes. The reverse sweep remains algebraically equivalent for the
  three-step temporal stencil and is covered by a dense-adjoint differential
  test, while adjoint-state memory drops from `O(steps * cells)` to `O(cells)`.
- [patch] Optimized the Chapter 29 nonlinear 3-D Westervelt forward-history
  memory by replacing the retained dense pressure history with exact sparse
  checkpoints plus interval replay. Each checkpoint stores the three states
  required by the Westervelt recurrence, the adjoint replays bounded segments
  with the production forward update, PyO3 exposes `checkpoint_interval_steps`,
  and tests verify bitwise replay equivalence plus checkpoint-interval-invariant
  `c/beta` gradients. The Chapter 29 metrics now record
  `checkpoint_interval_steps = 128`; replay runtime is the remaining
  performance target for the default full-resolution figure run.
- [patch] Corrected Chapter 29 abdominal target handling so one single-focus
  kidney/liver sonication selects one connected label-2 treatment component
  instead of plotting every disconnected tumor object as if one exposure could
  target them all; regenerated placement, exposure, reconstruction, and metrics
  artifacts.
- [patch] Added Chapter 29 reconstruction dynamic-range diagnostics with a
  shared `[-40, 0] dB` display scale and outside-target peak/energy metrics so
  finite-frequency aperture sidelobes are not conflated with treated tissue.
- [minor] Added Chapter 30 intravascular ultrasound imaging and therapy, with
  a public IVUS segmentation dataset contract, a dual-frequency 64-element
  catheter design, deterministic coronary wall phantom, radial B-mode
  simulation, localized microbubble delivery field, five generated figures,
  metrics, manifest registration, README link, and value-semantic Python
  tests.
- [patch] Added clinical ultrasonic speed-of-sound shift imaging under
  `clinical::imaging::reconstruction::sound_speed_shift`, with dense
  Tikhonov/H1 reconstruction, deterministic sparse row selection, sparse L1
  reconstruction, forward travel-time prediction, Chapter 5 documentation,
  and focused value-semantic Rust tests.
- [minor] Added 2-D curved-array acquisition support for the clinical
  straight-ray speed-of-sound shift model. `CurvedArray2d` defines circular-arc
  element coordinates, and `CurvedArrayShiftScan` emits deterministic
  same-aperture pitch-catch `SoundSpeedShiftSample` rows with measured
  time-shift attachment.
- [minor] Added curved-ray propagation and finite-frequency sensitivity to the
  clinical 2-D speed-of-sound shift operator. `ShiftPropagation::CircularArc`
  assembles circular arcs as exact grid-traversed subsegments, while
  `ShiftSensitivity::FiniteFrequency` builds compact normalized Fresnel tubes
  around the propagation path.
- [minor] Added `SoundSpeedShiftPlan` for fixed-acquisition speed-of-sound
  shift imaging. The plan caches fixed geometry samples plus the assembled
  operator, reconstructs repeated frames from raw time-shift slices, predicts
  selected-row shifts through the cached operator, and supports curved-array,
  curved-ray, and finite-frequency configurations without a parallel inverse
  path.
- [minor] Added fixed-acquisition batch reconstruction for clinical
  speed-of-sound shift imaging. `SoundSpeedShiftPlan` now reconstructs frame
  batches through one cached operator and one workspace, returns compact
  per-frame objective summaries by default, and retains full histories only
  when `SoundSpeedShiftObjectiveHistoryPolicy::Full` is selected.

### Changed

- [patch] Optimized clinical speed-of-sound shift ray assembly by replacing
  per-row active-pixel scans with exact grid traversal over crossed cells.
  The operator stores only nonzero segment lengths and focused tests compare
  traversal output against the prior per-cell clipping oracle.
- [patch] Modernized the clinical speed-of-sound shift operator into
  `operator/{construction,algebra,graph,row_storage,validation}.rs` and
  replaced per-row segment vectors with flat row-offset, column, and length
  arrays. Focused tests pin flat storage semantics and crossed-cell nonzero
  scaling.
- [patch] Modernized the clinical speed-of-sound shift solver into
  `solver/{dense,sparse,normal,linear_algebra,workspace}.rs`, added
  `SoundSpeedShiftWorkspace` plus
  `reconstruct_sound_speed_shift_with_workspace`, and changed normal-diagonal
  assembly to fill caller-owned buffers. Focused tests prove repeated
  reconstructions retain workspace capacity while preserving reconstructed
  values.
- [patch] Optimized the Chapter 29 theranostic inverse solve by precomputing
  the active-tissue graph Laplacian once per CT support and reusing CG row,
  normal-operator, prediction, and Laplacian work buffers instead of allocating
  image-sized Laplacian state inside every iteration. The PyO3 metrics now
  report `finite_frequency_same_aperture_graph_laplacian_pcg` as the operator
  model. The same cleanup pass removes current kwavers no-deps clippy blockers
  in the seismic/FWI path, including the Chapter 27 composite-objective
  argument fanout.
- [patch] Replaced Chapter 29 reduced same-aperture dense-row use in the
  theranostic workflow with matrix-free `FiniteFrequencyOperator` instances
  over a generic `LinearOperator` PCG contract. Dense `RowMatrix`
  materialization remains available for equivalence testing and bounded
  diagnostics, and PyO3 now reports `operator_backend`,
  `operator_storage_values`, and `dense_operator_values`.
- [patch] Added deterministic source/row encoding to the Chapter 29 reduced
  same-aperture inverse. `EncodedOperator` implements the exact compressed
  operator `B = C A` for active, passive, harmonic, and ultraharmonic channels,
  clinical PCG solves now use encoded matrix-free rows by default, PyO3 reports
  `inverse_encoding_rows_per_code`, `encoded_measurements`, and
  `unencoded_measurements`, and tests verify encoded forward, adjoint,
  diagonal, and measurement products against materialized rows without
  relabeling the reduced branch as nonlinear FWI.
- [patch] Corrected the Chapter 29 public API and figure metadata so the
  workflow is exposed as a theranostic inverse rather than full-waveform
  inversion. The PyO3 entry point is now
  `run_theranostic_inverse_from_ritk`, metrics report
  `inverse_model_family`, `is_full_wave_inversion = false`, and
  `uses_nonlinear_wave_propagation = false`, and the source-encoded
  time-domain RTM helper simulates baseline/lesion receiver traces with
  pascal-scale source injection, a CT-domain travel-time horizon,
  Charbonnier/L2 residual selection, and one flat forward-history buffer
  instead of allocating one vector per timestep.
- [patch] Added a documented robust waveform-misfit strategy for the Chapter
  29 linear acoustic RTM channel. The default receiver adjoint source now uses
  the Charbonnier derivative scaled from observed-trace RMS and configured
  receiver-noise fraction, PyO3 accepts `waveform_misfit` and
  `waveform_misfit_scale_fraction`, and metrics report the chosen misfit,
  scale, and objective value without changing the workflow into nonlinear FWI.
- [minor] Migrated the Chapter 29 same-device therapeutic ultrasound workflow
  from `kwavers::solver::inverse::seismic::theranostic` to
  `kwavers::clinical::therapy::theranostic_guidance`, removed the old
  solver-layer module, updated the PyO3 bindings to call the clinical workflow,
  and synchronized the book contract so clinical/device orchestration no
  longer lives under inverse solvers.
- [minor] Extracted the Chapter 29 reduced same-aperture inverse kernels into
  `kwavers::solver::inverse::same_aperture`, making finite-frequency
  active/passive row assembly, harmonic row assembly, active-support graph
  indexing, and graph-Laplacian PCG the solver-owned SSOT while keeping
  clinical exposure rendering in `clinical::therapy::theranostic_guidance`.
- [patch] Updated Chapter 29 Figure 2 so every brain, kidney, and liver row
  starts with the CT placement slice, target/body overlays, and transducer
  coordinates before the exposure and reconstruction channels, preserving
  anatomical targeting and device-placement context in the same panel grid.
- [patch] Optimized the Chapter 29 matrix-free `FiniteFrequencyOperator` for the
  inverse-PCG hot path: per-row source/receiver/wavenumber/frequency-MHz/
  nonlinear-path-weight or sine-phase metadata is precomputed once at
  construction so `matvec`, `t_matvec`, `normal_diag`, `compute_row_norms`, and
  `materialize` never recompute the row `divmod` or the variant dispatch on a
  per-cell basis. Inverse row norms are cached so the inner loops never recompute
  `1 / norm`. The outer row and column loops dispatch through rayon for cache-
  aware parallelism on the SPD normal equations driven by PCG. Verified bit-
  identical against the dense `RowMatrix` oracle by the existing
  `matrix_free_operator_matches_materialized_rows` regression test.
- [patch] Optimized the Chapter 29 nonlinear 3-D Westervelt forward path and
  Rayleigh-Plesset passive operator. The retained FWI forward history now uses
  exact sparse checkpoints plus bounded interval replay instead of one stored
  volume per timestep; the prior fragmented `Vec<Vec<f64>>` history is gone.
  The four rotating forward buffers (older, previous, current, next) are
  `mem::swap`-rotated; no `vec![0.0; cells]` allocation occurs inside the time
  loop. The cell update is rayon-parallel: each cell writes only to its own
  `next[i]`, so the outer 3-D loop dispatches through
  `par_iter_mut().enumerate()` without coloring, atomics, or locks. The
  Rayleigh-Plesset passive operator now builds its dense Green's matrix
  row-parallel via `par_chunks_mut().zip(receivers.par_iter())`, runs `apply`
  through `par_chunks().map().collect()`, and runs `normal_gradient`
  column-parallel through `(0..cols).into_par_iter()`. The dead `rows` field
  on `PassiveOperator` is removed. The reverse-mode adjoint is unchanged
  mathematically and continues to pass nonlinear 3-D fixture tests.
- [patch] Split the Chapter 29 nonlinear volume-preparation module by moving
  CT-derived attenuation and centroid utilities into `volume/attenuation.rs`
  and `volume/centroid.rs`, keeping every nonlinear 3-D Rust source file under
  the 500-line structural limit.

### Fixed

- [patch] **Westervelt FDTD nonlinear-term sign correction.** The explicit
  leapfrog update for the Westervelt equation
  `∇²p − (1/c²)·p_tt + (β/(ρ₀·c⁴))·∂²(p²)/∂t² = 0` solves for `p_tt` as
  `p_tt = c²·∇²p + (β/(ρ₀·c²))·∂²(p²)/∂t²`, so the discrete recurrence
  `p[n+1] = 2·p[n] − p[n−1] + (c·Δt)²·∇²p + q·∂²(p²)/∂t²|_n` with
  `q = β·Δt²/(ρ₀·c²) > 0` must apply the nonlinear contribution with a
  **positive** sign (forward steepening; peaks at fixed `x` arrive earlier
  than linear). Both the canonical
  `solver::forward::nonlinear::westervelt` FDTD and its Chapter 29
  derivative `clinical::therapy::theranostic_guidance::nonlinear3d::westervelt`
  were applying it with a negative sign, producing non-physical reverse
  steepening. The fix flips both code paths to `+ q·∂²(p²)/∂t²` and updates
  the Chapter 29 discrete adjoint so the reverse-mode chain rule matches
  the corrected forward (the `add_nonlinear_transpose` adjoint contributions
  and the `d_update_dc` sound-speed sensitivity flip sign on the nonlinear
  term). The Kuznetsov solver at
  `solver/forward/nonlinear/kuznetsov/solver/rhs.rs` already used the
  positive convention (`*r += nl`) and required no change. A
  literature-derivable sign-sensitive regression test
  `forward_westervelt_exhibits_physical_forward_steepening_with_corrected_sign`
  drives a 1 MHz / 5 MPa source through a homogeneous β = 10 cube and
  asserts `max(∂p/∂t) > |min(∂p/∂t)|` on the steady-state receiver trace —
  the sign-flipped form fails this check. References: Westervelt (1963)
  Eq. 24; Hamilton & Blackstock (1998) Eq. 3.10.
- [patch] **Chapter 29 Rayleigh-Plesset passive Green's-function frequency-
  dependent absorption.** The passive subharmonic operator
  `PassiveOperator::new` in
  `clinical::therapy::theranostic_guidance::nonlinear3d::cavitation`
  previously used a hardcoded `exp(−2·r)` attenuation factor — a single
  number with no frequency or anatomy dependence. The 2 Np/m value happens
  to match brain at a 325 kHz subharmonic (650 kHz drive) but is wrong for
  the abdominal 250 kHz subharmonic (500 kHz drive) by ≈40 %. The fix
  derives `α [Np/m]` from a soft-tissue power-law baseline
  `α₀ = 0.5 dB/(cm·MHz)` scaled by the actual subharmonic frequency
  `f_s = f₀/2`, with the dB→Np conversion `α [Np] = α [dB]/8.685889638…`
  and the cm→m factor of 100. The Green's-function kernel is now
  `exp(−α_s·r) · cos(k_s·r) / (4π·r)` with both `α_s` and `k_s` tied to the
  subharmonic. Reference: Hamilton & Blackstock 1998 Table 4.1
  (soft-tissue median).
- [patch] **Canonical Westervelt FDTD absorption comment honesty.** The
  comment I added previously claimed `(p_n − 2 p_{n-1} + p_{n-2})/dt` is
  `−dt²·(δ/c²)·p_ttt`. That is dimensionally wrong: the expression is the
  3-point backward second derivative of `p` (i.e., `dt²·p_tt(n-1)`), not a
  third derivative. The corrected comment now states honestly that the
  FDTD absorption is a Kelvin-Voigt-like lagged-`p_tt` proxy that
  approximates Stokes-Kirchhoff in the small-`δ/(c²·dt)` limit (so the
  plane-wave dispersion still has `Im(ω) > 0` and the wave decays), but is
  NOT a strict `p_ttt` discretization and is NOT a frequency-dependent
  power-law absorption. For physically accurate power-law absorption the
  comment directs users to the PSTD fractional-Laplacian path (Treeby &
  Cox 2010). The code itself is unchanged — the existing implementation is
  functionally correct in the small-`δ` limit; only the comment was
  misleading.
- [patch] **Chapter 29 heterogeneous CT-derived attenuation for the cavitation
  Green's function.** The previous fix derived absorption from frequency but
  used a single soft-tissue-typical scalar everywhere — underestimating
  absorption by ~26× on transcranial paths because cortical skull is ~13
  dB/(cm·MHz) vs. soft tissue ~0.5 (Hamilton & Blackstock 1998 Table 4.1;
  Connor & Hynynen 2002). Fix: (1) `Nonlinear3dVolume` now carries an
  `attenuation_np_per_m_mhz: Array3<f64>` field derived in `material_maps`
  from CT HU with explicit tissue classes — cortical bone (HU ≥ 300) at
  13 → 20 dB/(cm·MHz) interpolated by HU density, air pockets
  (HU < −700, label = 0) as nearly opaque, segmented organs at
  0.6 dB/(cm·MHz), and generic soft tissue at 0.5 dB/(cm·MHz). (2)
  `cavitation::PassiveOperator::new` computes a **path-integrated**
  absorption by sampling the attenuation field along the straight line from
  source voxel to receiver with trilinear interpolation and a trapezoidal-
  rule integral, then scaling by the subharmonic frequency for the `y = 1`
  tissue power law. The Green's kernel is `exp(−∫ α_s(s)·ds) · cos(k_s·r) /
  (4π·r)`. For brain cases this correctly tracks the skull's order-of-
  magnitude attenuation on every ray. The existing
  `nonlinear_3d_westervelt_fwi_and_cavitation_inverse_are_input_sensitive`
  integration test passes (9/9 nonlinear3d + Westervelt tests green),
  locking that the new path-integral Green's matrix is still SPD via the
  projected-gradient inverse.
- [patch] **Chapter 29 tissue-class power-law `y` exponent for heterogeneous
  attenuation.** Added a per-voxel `attenuation_power_law_y` field on
  `Nonlinear3dVolume` and wired it through `cavitation::PassiveOperator::new`'s
  path integral. The frequency dependence is now `α(f) = α(1MHz) · f_MHz^y`
  per voxel sample, matching Treeby & Cox 2010 Table I / Szabo 1995 for soft
  tissue (`y ≈ 1.05`) and Connor & Hynynen 2002 / classical Stokes-Kirchhoff
  for cortical skull bone (`y ≈ 2.0`, measured 1.9 - 2.0 across 0.5 - 3.5
  MHz). The `y = 2` skull behavior gives **3.07× less attenuation at the
  325 kHz subharmonic** (650 kHz brain drive) than a naive `y = 1`
  extrapolation would predict — without this correction the transcranial
  passive cavitation receive path would be over-attenuated by a factor of
  3, starving the cavitation inverse. Added 9 literature-anchored unit
  tests in `volume::attenuation_tests` that lock each tissue-class value
  against its citation (e.g. `soft_tissue_attenuation_matches_hamilton_
  blackstock_1998_table_4_1_median`, `skull_bone_attenuation_at_lower_hu_
  bound_matches_connor_hynynen_2002`, `skull_power_law_y_matches_connor_
  hynynen_2002_stokes_kirchhoff`, `skull_subharmonic_attenuation_with_y2_
  is_three_times_less_than_y1`). 19/19 nonlinear3d + Westervelt tests
  pass; `cargo clippy --no-deps -- -D warnings` clean.
- [patch] **Chapter 29 brain-helmet end-to-end integration test.** Closed the
  brain-anatomy coverage gap. Added
  `nonlinear_3d_brain_helmet_pipeline_is_input_sensitive_through_skull` plus
  a synthetic `brain_fixture()` (28³ ellipsoidal cortical-bone shell at
  HU = 600 wrapping a brain interior at HU = 40, surrounded by air at
  HU = -1000). The test exercises the full INSIGHTEC-like transcranial
  pipeline: `AnatomyKind::Brain` volume preparation (no segmentation
  required — synthetic ellipsoidal target built from the body centroid),
  calvarium-cap helmet aperture, source-encoded Westervelt forward through
  skull voxels, discrete-adjoint FWI for `c` and `β`, Rayleigh-Plesset
  cavitation source from the resulting peak pressure field, and passive
  subharmonic inverse with the **heterogeneous CT-derived path-integrated
  attenuation Green's function including `y = 2` Stokes-Kirchhoff skull
  power-law absorption**. The test asserts the pipeline runs, the aperture
  model is `insightec_like_calvarium_helmet_3d_westervelt_sources`,
  Westervelt peak pressure is positive, cavitation density responds to the
  pressure field, the FWI and cavitation objectives are non-increasing, and
  the helmet places ≥ 16 therapy points / ≥ 4 receivers. This is the only
  test in the suite that actually places skull voxels (HU > 300) between
  source and receiver, so it is the only test that exercises the 3.07× `y = 2`
  vs `y = 1` skull-attenuation correction in the cavitation operator path
  integral. 20/20 nonlinear3d + Westervelt tests pass;
  `cargo clippy --no-deps -- -D warnings` clean.
- [patch] **Chapter 29 Westervelt physics-scaling regression tests.** Added
  two negative-control / scaling regressions that close gaps left by the
  single-β sign test:
  (1) `linear_westervelt_with_beta_zero_produces_symmetric_pressure_trace_
  within_fdtd_tolerance` — runs the same forward fixture at `β = 0` and
  asserts the asymmetry ratio `R = max(∂p/∂t) / |min(∂p/∂t)|` stays in
  `[0.80, 1.20]`. This catches numerical-dispersion artifacts masquerading
  as nonlinearity: at β = 0 the Westervelt recurrence reduces to the linear
  wave equation and must produce a (near-)symmetric receiver trace.
  (2) `westervelt_steepening_signature_scales_linearly_with_beta_per_weak_
  nonlinear_theory` — runs the same fixture at β = 0, β = 5, β = 10 and
  verifies the **excess-over-linear** asymmetry `δ(β) = R(β) − R(0)` scales
  linearly with β per leading-order weak-nonlinear Born/Fubini theory
  (Hamilton & Blackstock 1998 §4.3): `δ(10) / δ(5)` must fall in `[1.3, 3.0]`
  with target `2.0`. This catches β-coefficient sign or magnitude errors —
  a ratio near 4 would suggest `β²` instead of `β` in the recurrence; a
  ratio near 1 would suggest β is not entering at all. The excess-over-
  linear formulation isolates the β-dependent nonlinear contribution from
  the β-independent FDTD dispersion bias floor. 22/22 nonlinear3d +
  Westervelt tests pass; `cargo clippy --no-deps -- -D warnings` clean.
- [patch] **Chapter 29 Westervelt harmonic-generation presence test**
  (Tier-2, `#[ignore]`'d, ~53 s runtime). Added
  `westervelt_fdtd_point_source_generates_measurable_second_harmonic_content`
  which extracts fundamental and 2nd-harmonic amplitudes via discrete
  sine/cosine projection at known frequencies (exact for harmonics, no FFT
  needed) and asserts `|P_2|/|P_1| ∈ [0.03, 0.40]` for a 5 MPa / β = 10
  point source. Catches a nonlinear term that propagates as just a phase
  shift (ratio ≈ 0), a spuriously-high 2nd harmonic from `β²` coefficient
  error (ratio > 0.5), and DC-only or NaN output. **Why not Fubini-
  absolute**: Aanonsen-1984 / Fubini analytical assumes 1-D plane-wave
  propagation with constant amplitude; a 3-D point-source FDTD has 1/r
  spreading so local Γ varies along the path. The KZK solver carries the
  literature-validated Fubini-absolute test because KZK parabolically
  reduces 3-D to 1-D-along-z with constant-amplitude planar shots. The
  Westervelt FDTD cannot drive that configuration without API changes.
  Measured ratio at fixture: `0.133` (within [0.03, 0.40]). 22/22 default
  and 1 Tier-2 ignored test pass; `cargo clippy --no-deps -- -D warnings`
  clean.
- [patch] **Chapter 29 Westervelt Aanonsen-1984 Fubini-absolute test on a
  1-D harness** (Tier-2, `#[ignore]`'d, < 1 s runtime). Closed the last
  harmonic-amplitude validation gap by introducing a clean 1-D Westervelt
  FDTD harness inline in the test file. The 1-D recurrence is
  algebraically identical to the 3-D `update_cells`:
  `p[n+1] = sponge·(2 p[n] − p[n−1] + (c·dt)²·∇²p + q·∂²(p²)/∂t²)` with
  `q = β·dt²/(ρ·c²)` and the product-rule `∂²(p²)/∂t² ≈ 2 p·d²p/dt² + 2·(dp/dt)²`,
  using a 3-point 1-D Laplacian instead of the 7-point 3-D stencil. **Hard
  sinusoidal source** at `x = 4` clamps the source-cell pressure;
  absorbing sponge at the far boundary prevents reflections. Resolution:
  `dx = 0.05 mm` → 30 pts/wavelength fundamental, 15 pts/wavelength 2nd
  harmonic. Discrete sine/cosine projection (exact for harmonics on
  integer-period windows) extracts `|P_1|` and `|P_2|`. The Westervelt
  recurrence algebra must match Fubini
  `|P_2|/|P_1| = J_2(2Γ)/(2·J_1(Γ))` at the **empirical Γ** computed from
  the observed `|P_1|`. Empirical Γ is required because a 1-D FDTD hard
  source radiates ≈ 0.57 × the nominal `P_0` (radiation coupling
  determined by the discrete Laplacian / CFL) — the physically meaningful
  Γ is the one carried by the propagating wave, not the source-clamp
  nominal. The Bessel `J_0`, `J_1`, `J_2` analytical values are computed
  via convergent power series inline. **Tolerance: 15 %**. **Measured at
  the fixture**: `|P_1| = 5.70e5 Pa`, `|P_2| = 7.92e4 Pa`, `|P_2|/|P_1| =
  0.139`, empirical `Γ = 0.286`, Fubini at empirical Γ = 0.148 →
  relative error 6 %. This validates that the Westervelt `q·∂²(p²)/∂t²`
  algebra matches Fubini analytical at the empirical wave Γ to within
  numerical-dispersion tolerance — closing the last harmonic-amplitude
  literature-validation gap. 22/22 default + 2 Tier-2 ignored tests pass
  with `cargo test --lib -- --ignored`; `cargo clippy --no-deps -- -D
  warnings` clean. References: Aanonsen et al. 1984 Eq. 6; Hamilton &
  Blackstock 1998 §4.3.2.
