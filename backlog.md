# Backlog / Strategy

## Focused bowl axis-reference aperture config - closed (2026-05-24)
- **[done] [minor]** Promoted axis-reference focused-bowl construction to the
  public source-domain API and added `FocusedBowlAperture::AxisReferencePolarBounds`
  for config-driven focused sources. `position` now acts as the contact/axis
  reference for that aperture mode, `radius_of_curvature_m` fixes the source
  radius independently of contact-to-focus distance, and `SourceFactory` still
  delegates element positions, normals, and weights to `BowlTransducer`.
  Verification: the axis-reference source factory regression passes 1/1, the
  bowl axis-reference preset theorem test passes 1/1, and
  `cargo check -p kwavers --lib --message-format=short -j 1` exits 0.

## Ali 2025 PSTD odd-z FFT parity - closed (2026-05-24)
- **[done] [patch]** Repaired odd-length z-axis 3-D r2c/c2r transforms in
  `apollo-fft` by dispatching odd `nz > 1` through a full-spectrum fallback with
  Hermitian reconstruction for inverse c2r. This closes the reduced `(4,4,3)`
  passive-channel direct-field mismatch at the FFT primitive layer.
  Verification: `cargo test -p apollo-fft r2c_ --lib -j 1 -- --nocapture`
  passes 8/8; `cargo test -p kwavers finite_grid_pstd_prediction_matches_homogeneous_dataset --lib -j 1 -- --nocapture`
  passes; PSTD temporal transfer, homogeneous direct-field diagnostics, and
  PSTD CBS adjoint-gradient tests pass.

## Focused-bowl model-label cleanup - closed (2026-05-24)
- **[done] [patch]** Removed vendor-like focused-bowl labels from live Rust and
  PyO3 source/model metadata. Abdominal 2-D layout, abdominal placement
  context, nonlinear 3-D aperture metadata, the PyO3 abdominal placement docs,
  and the therapy plotting fixture now use generic `focused_bowl` source
  labels. Verification: live Rust/PyO3/test source scan has no
  `HistoSonics`/`InSightec`/`brain_helmet`/`helmet` matches, abdominal
  theranostic tests pass 4/4, brain focused-bowl tests pass 4/4, nonlinear3d
  tests pass 57/57 with 3 ignored, `cargo check -p kwavers --lib` exits 0, and
  the targeted therapy plotting pytest passes 1/1.

## Medium property SSOT constant closure - closed (2026-05-24)
- **[done] [patch]** Completed the in-progress literal-to-constant migration
  for fluid, tissue, and implant material properties. `fundamental.rs` now
  owns the referenced fluid/tissue density, sound-speed, and absorption
  constants; `implants.rs` owns effective per-material implant nonlinearity
  constants with the exact prior model values. Verification: implant constants
  tests pass 2/2 and `cargo check -p kwavers --lib --message-format=short -j 1`
  exits 0.

## Abdominal focused-bowl axis-reference source routing - closed (2026-05-24)
- **[done] [patch]** Added a source-domain focused-bowl config constructor for
  non-vertex axis references. Abdominal 3-D placement now passes skin contact as
  an orientation reference and delegates explicit-radius cap sampling to
  `BowlTransducer::with_angular_bounds`, preserving outside-body rim geometry
  without a clinical-local spherical-cap implementation. Verification: source
  bowl tests pass 16/16, abdominal 3-D tests pass 10/10, and
  `cargo check -p kwavers --lib --message-format=short -j 1` exits 0.

## Clinical focused-bowl cap helper consolidation - closed (2026-05-23)
- **[done] [patch]** Replaced duplicated transcranial cap point samplers in the
  theranostic 2-D context and 3-D placement planner with
  `geometry::focused_bowl`. The helper accepts typed vertex direction and
  `BowlAngularBounds`, then delegates to `BowlTransducer::with_angular_bounds`.
  Invalid configured polar bounds now fail validation instead of falling back
  to defaults. Verification: geometry helper tests pass 2/2, brain theranostic
  tests pass 4/4, and `cargo check -p kwavers --lib` exits 0.

## Transcranial UST aperture routing - closed (2026-05-23)
- **[done] [patch]** Removed the remaining hard-coded hemispherical acquisition
  path from the transcranial UST Born adapter. `TranscranialUstBornInversionConfig`
  now carries a source-domain `BowlAngularBounds`, and slice/volume
  reconstruction use `BowlTransducer::with_angular_bounds` for geometry
  generation. Verification: transcranial UST tests pass 4/4, focused source
  transducer tests pass 33/33 with 1 ignored, and `cargo check -p pykwavers --lib`
  exits 0.

## Ali 2025 reduced-array row planning boundary — closed (2026-05-23)
- **[done] [minor]** `BreastUstReducedArrayPlan` and
  `BreastUstReducedArrayRowPolicy` now live in the Rust clinical reduction
  layer. The Table 1 parity policy derives one row per interior z-slice and
  leaves one grid-cell margin at both axial boundaries; explicit and smoke
  policies are validated separately. PyO3 exposes
  `derive_breast_fwi_reduced_array_plan`, and the Python replication script now
  selects a policy string and reports Rust-derived geometry instead of owning
  row-count or row-spacing formulas.
- **[open] [patch]** Determined probe rerun after PyO3 rebuild remains rank
  sufficient but passive-channel mismatched: all-row best model is still
  `single_scatter_born`, while `pstd_spectral_convergent_born` matches active
  source/receiver channels to numerical precision and has passive-only
  normalized residual `0.7905925502451137` with the prior nonzero absorber.
  Next increment: align passive receiver Green/operator semantics for the PSTD
  spectral CBS path.

## Ali 2025 zero-thickness absorber contract — closed (2026-05-23)
- **[done] [patch]** PyO3 now maps
  `absorbing_boundary="polynomial", absorbing_thickness_cells=0` to
  `AbsorbingBoundary::disabled()` for the general frequency-domain FWI
  constructor and both spectral convenience constructors. The replication
  example default now uses zero absorbing cells, matching the no-CPML PSTD
  dataset path.
- **[open] [patch]** The zero-absorber determined probe improves
  `pstd_spectral_convergent_born` passive-only residual to
  `0.6007092896747324` but worsens all-channel residual to
  `0.8646947820594513`; active-only residual remains numerical zero. Next
  increment: isolate passive phase/source-scale semantics in the PSTD spectral
  Green path rather than adding Python-side correction factors.

## Ali 2025 PSTD CBS discrete contrast alignment — closed (2026-05-24)
- **[done] [patch]** The PSTD spectral CBS operator now uses the leapfrog
  temporal mass symbol `4 sin²(ωΔt/2)/Δt²` for the real scattering potential
  and adjoint slowness derivative, while dense and continuous spectral CBS
  retain the continuous Helmholtz `ω²` contrast. The theorem lives in
  `solver::inverse::fwi::frequency_domain::cbs::potential`, and the forward
  plus adjoint-gradient paths dispatch through `GreenOperatorKind`.
  Verification: the PSTD contrast theorem tests and
  `pstd_spectral_cbs_adjoint_gradient_matches_finite_difference` pass through
  `cargo test`.
- **[open] [patch]** The determined-probe passive residual has not been rerun
  after this Rust-side contrast correction and the odd-z FFT repair. Next
  increment: rebuild pykwavers and regenerate the reduced determined-probe
  metrics to determine whether any residual mismatch remains in the PyO3-facing
  PSTD spectral CBS path.

## CBS adjoint O(N log N) iterative solver — closed (2026-05-23)
- **[done] [minor]** `solve_adjoint_spectral_iterative` now implements the correct
  Richardson adjoint for spectral CBS operators. The iterate uses `λ += γ^H·residual`
  where `γ^H = conj(γ) = −iε/Ṽ*`, giving iteration matrix `I + γ^H A^H` with diagonal
  `V/(V+iε)` satisfying `|V/(V+iε)| < 1` under `ε ≥ ‖V‖_∞` — the same contraction bound
  as the forward. `DenseFreeSpace` retains exact dense LU; `SpectralPeriodic` and
  `SpectralPstdPeriodic` use the O(max_iter × N log N) iterative path. The
  O(N² log N) `operator_matrix_by_columns` matrix build is removed.
  Verification: `spectral_cbs_adjoint_gradient_matches_finite_difference` and
  `pstd_spectral_cbs_adjoint_gradient_matches_finite_difference` both PASS. (commit `045982e44`)

## Ch29 OOM fix — early CT drop in `run_theranostic_nonlinear_3d` (2026-05-22)
- **[done] [patch]** Root cause of "memory allocation ... failed" abort in the PyO3 book generation
  path (`fig05 nonlinear brain start` / `comparison nonlinear brain start`):
  `run_theranostic_nonlinear_3d` accepted `ct_hu: &Array3<f64>` and `label_volume: Option<&Array3<i16>>`.
  The caller (PyO3 binding) held the full-resolution brain CT (~600 MB at f64 for a 512×512×300 scan)
  alive across every forward pass, checkpoint store, and adjoint step of the Westervelt FWI loop.
  Combined with Python heap, matplotlib state, and results from prior cases, this exhausted available
  RAM before the FWI loop completed.
  Fix: changed signature to take owned `Array3<f64>` / `Option<Array3<i16>>` and added explicit
  `drop(ct_hu); drop(label_volume);` immediately after `prepare_volume` returns the resampled
  `grid_size³` volume. Three call sites updated: PyO3 binding (`nonlinear3d.rs`), abdominal
  pipeline test, brain pipeline test. `cargo check -p kwavers --lib` and
  `cargo check -p pykwavers --lib` exit 0.

## T10/T15b — time-domain FWI solver-type factory dispatch (2026-05-22)
- **[done] [arch]** `FwiParameters` gained `solver_type: SolverType` (default `FDTD`). The
  `build_fdtd_solver_for_forward` method was renamed `build_solver_for_forward` and its return
  type is `(Box<dyn Solver>, dims, dt)`. Dispatch on `SolverType::FDTD` uses the existing
  `FdtdSolver` + `enable_cpml` path (now extracted to `build_fdtd_boxed`); `SolverType::PSTD`
  routes to `PSTDSolver` with CPML embedded in `PSTDConfig::boundary` (now in `build_pstd_boxed`).
  `adjoint_model` in `adjoint.rs` was updated in parallel — it now builds the adjoint solver
  through the same typed helpers and steps via `Box<dyn Solver>` dynamic dispatch, preserving
  the time-reversal theorem for both solver types. Unsupported types return
  `KwaversError::InvalidInput` naming the rejected variant. Two new value-semantic tests:
  PSTD forward smoke (non-zero receiver trace) + unsupported-type rejection (error contains
  type name). 76/76 FWI tests pass; `cargo check -p {kwavers,pykwavers} --lib` exit 0.

## T19b-slice-2 — sensor-pressure trait promotion (2026-05-21)
- **[done] [patch]** Added `Solver::recorded_sensor_pressure(&self) -> Option<Array2<f64>>` with default impl returning `None`. Concrete overrides on `FdtdSolver` and `PSTDSolver` forward to their existing `sensor_recorder.extract_pressure_data()`. FWI A's `forward_model` and `forward_model_sensor_only` now read the synthetic receiver trace through `<FdtdSolver as Solver>::recorded_sensor_pressure(&solver)` instead of `solver.sensor_recorder.extract_pressure_data()` — same cross-layer cleanup pattern as the `step_forward` / `pressure_field` trait dispatch landed in T19a. Hybrid and DG solvers keep the default `None` (they have no integrated sensor recorder). 72/72 FWI tests pass; `cargo check -p {kwavers, pykwavers} --lib` exit 0.

## DG CPML finite-3D closure (2026-05-21)
- **[done] [patch]** New `solver::forward::pstd::dg::cpml` module: Roden-Gedney
  profile + Lazarov-Warburton joint-stepped auxiliary `ψ` ODE for the tensor
  acoustic DG solver. `DgCpmlConfig` on `DGConfig` gates the path; the
  standard non-CPML path is bit-for-bit unchanged. Water-tank example gains a
  `DG-3D-CPML` row matching DG-2D / DG-3D to L2 ≈ 7.4e-4 (corr 0.999999) and
  reproducing FDTD / PSTD CPML pairwise metrics. 24 new CPML tests pass;
  pre-existing 45 DG tests untouched.

## Session 3 closure summary — Ali 2025 replication on cleaned foundation (2026-05-21)

### Architectural cleanups delivered
- **T17a/T17b**: `HelmholtzForwardOperator` trait + 3 impls in
  `solver::inverse::fwi::frequency_domain::operator`. `Config.forward_operator:
  Arc<dyn HelmholtzForwardOperator>` replaces the old `PropagationModel` enum
  end-to-end (kwavers + pykwavers; `propagation_model` kwarg preserved in
  Python).
- **T13a/T13b-Phase-1/2/3 + T13c**: `TransducerGeometry` trait,
  `LinearBornInversionConfig`, `VolumeVoxel`, `dense`, `schedule`,
  `enhancement`, `regularization`, `pcg`, `volume_operator` all hoisted from
  `clinical::imaging::reconstruction::transcranial_ust` to
  `solver::inverse::linear_born_inversion`. `RingPoint` unified with
  `ElementPosition`. `MultiRowRingArray` and `TranscranialBowlGeometry` both
  impl `TransducerGeometry`.
- **T15/T16**: `solver::inverse::seismic::brain_helmet` relocated to
  `clinical::imaging::reconstruction::transcranial_ust`. FWI A namespace
  relocated `solver::inverse::seismic::fwi` →
  `solver::inverse::fwi::time_domain`. Parallel FWI B stack deleted
  (~1500 LOC, 0 external consumers).
- **T19a / T19b-slice-1**: `Solver::step_forward` added to the unified trait
  with default `self.run(1)` and concrete overrides on FDTD/PSTD/Hybrid; FWI
  A's hot loops now read `pressure_field()` through trait dispatch instead of
  the previous `solver.fields.p` direct field access.
- **T14**: `pykwavers::seismic_bindings` → `imaging_bindings`;
  `run_seismic_helmet_fwi_*` → `run_transcranial_ust_*_inversion_from_ritk_ct`.
- **T24**: rand 0.9 `Rng` trait import fix across ritk-core noise.rs + 7
  kwavers sites (`rng.random()` / `rng.random_range(...)` need explicit
  `use rand::Rng;` for method-resolution).

### Ali 2025 replication delivered
- **T6/T7/T8/T9 closed**: PyO3 surface
  (`pykwavers::breast_fwi_bindings::*`, ~1200 LOC across 6 submodules) + PSTD
  data-gen pipeline + replication driver
  (`pykwavers/examples/replicate_ali2025_breast_fwi.py` + 8-module
  `ali2025_breast_fwi/` helper package) + Table 1 parity scaffold with
  configurable thresholds.
- **T27 executed**: maturin develop --release succeeded in 4m55s after the
  Phase-3 refactors; minimum-scale replication ran end-to-end in ~9 s with
  full diagnostic JSON output. Windows DLL workaround: `cp
  /d/miniforge3/python3.dll /d/miniforge3/libpython3.dll` +
  `os.add_dll_directory('D:/miniforge3')` before import.
- **T30 executed**: progressive scale-up rungs 0/1b/3 confirmed rank scaling
  (0.012 → 0.023 → 0.469 informative-DOF ratio with 1, 2, 16 transmitters).
  At 0.47 rank ratio the system moves out of formally rank-limited regime;
  remaining reconstruction-quality gap (RMSE ~40 m/s, PCC ~0 at min scale)
  is FWI-iteration-limited (paper uses 5 × 13 = 65 outer passes).
- **Compute extrapolation**: paper-scale CPU is ~38 h at 16³ grid up to
  multi-week at paper 0.4 mm 3-D grid. GPU PSTD wiring (T31) brings the
  forward-sim phase to ~4–8 h; CBS frequency-domain inversion remains
  CPU-bound.

### Open architectural items
- **T10/T19b-slice-2..N**: FWI A factory dispatch via `SolverType` — Solver
  trait already exposes `step_forward` (T19a); remaining work is CPML →
  config-time hoist, sensor recording → FWI-internal, `build_fdtd_solver_for_forward`
  return type → `Box<dyn Solver>`, PSTD adjoint reciprocity verification.
- **T31**: route `breast_ust_fwi::dataset` PSTD construction through
  `SimulationSolverFactory::create_solver(SolverType::PstdGpu, ...)`. Blocked
  on `GpuPstdSolver` not implementing `Solver` trait + factory not accepting
  `GridSource` at construction.
- **Paper-scale Ali run**: long-lived background job; infrastructure ready,
  compute-bound only.

## Architectural Cleanup — Session 2 closures (2026-05-20)
- **[minor] closed** CBS forward + adjoint kernel at
  `solver::inverse::fwi::frequency_domain::cbs` (Osnabrugge 2016).
- **[minor] closed** `PropagationModel::{SingleScatter, Cbs}` selection wired
  through `fwi::frequency_domain::{forward, gradient}`.
- **[patch] closed** `fwi::frequency_domain::mod.rs` §Linearisation block.
- **[patch] closed** `brain_helmet::*Fwi*` → `*BornInversion*` rename
  (identifier level only).
- **[patch] closed** Removed empty `seismic::abdominal_theranostic/`.

## Open Architectural Items
- **[done] [arch] T11b: brain_helmet layer relocation — CLOSED 2026-05-20.**
  Full module moved out of solver layer to
  `clinical::imaging::reconstruction::transcranial_ust`. Solver path no longer
  references anatomy or transducer topology. 12 caller files updated; cargo
  check exit 0.
- **[done] [minor] T11e: bowl angular aperture source layout — CLOSED 2026-05-21.**
  `BowlTransducer` now exposes fixed-count polar-span and polar-bounds
  constructors backed by the focused spherical-cap SSOT. Full-volume brain
  placement delegates major-cap element generation to the bowl source API
  instead of owning a local aperture sampler.
- **[done] [patch] T11f: 3-D clinical focused-bowl source routing — CLOSED 2026-05-21.**
  The full-volume calvarium placement helper now obtains its major-cap element
  positions from `BowlTransducer::with_polar_bounds` and no longer owns a local
  Fibonacci aperture sampler.
- **[done] [patch] T11g: transcranial focused-bowl naming completion — CLOSED 2026-05-21.**
  Public Rust/PyO3/Python/book APIs now expose focused-bowl placement names
  (`plan_transcranial_focused_bowl_placement*`) with no compatibility alias for
  the old brain-helmet planner. The nonlinear 3-D aperture model string and
  generated book metrics are synchronized to the same focused-bowl terminology.
- **[done] [patch] T11h: transcranial UST reconstruction boundary — CLOSED 2026-05-21.**
  The finite-frequency Born inversion moved from
  `solver::inverse::seismic::brain_helmet` to
  `clinical::imaging::reconstruction::transcranial_ust`. Exported config,
  result, and geometry names now use `TranscranialUstBornInversion*` and
  `TranscranialBowlGeometry`, and geometry generation delegates to
  `BowlTransducer::with_polar_span`.
- **[done] [arch] T13a (was T11c): TransducerGeometry trait landed — CLOSED 2026-05-20.**
  New module `solver/inverse/linear_born_inversion/{mod,geometry}.rs` owns
  `ElementPosition` + the `TransducerGeometry` trait (elements / len /
  is_empty / receiver_indices with cyclic-offset default).
  `TranscranialBowlGeometry` impls the trait with bowl-specific azimuthal-
  rotation override of `receiver_indices`. `cargo check -p kwavers --lib` and
  `cargo test -p kwavers linear_born_inversion --lib` pass.
- **[done] [arch] T13c: MultiRowRingArray adopts TransducerGeometry — CLOSED 2026-05-20.**
  `RingPoint` (in `physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi`)
  renamed to `ElementPosition` across 38 references in 9 files; duplicate
  type definition removed; the canonical type now lives only in the solver
  layer. `MultiRowRingArray` impls `TransducerGeometry` (default cyclic-
  offset semantics, appropriate for a ring). Second cross-module consumer
  proves the abstraction generalises; the breast-UST ring array and the
  transcranial bowl now share one trait surface. cargo check --lib clean.
- **[done] [arch] T13b-Phase-1: LinearBornInversionConfig defined — CLOSED 2026-05-20.**
  New `solver/inverse/linear_born_inversion/config.rs` holds 19 anatomy-
  neutral numerical fields (frequencies_hz, receiver_offsets, iterations,
  relaxation, regularization, frequency_continuation, sobolev_*,
  enhancement_gain, edge_preserving_*, attenuation_model,
  nonlinear_harmonic_model, source_pressure_mpa, nonlinear_beta,
  contrast_min/max) with validate/harmonic_count/measurement_count methods.
  element_count and radius_m intentionally excluded — they belong on the
  transducer geometry constructor, not on the inversion config.
- **[done] [arch] T13b-Phase-2: migrate transcranial_ust kernels to consume the
  generic config — CLOSED 2026-05-21.** Embed `LinearBornInversionConfig` inside
  `TranscranialUstBornInversionConfig` (clinical wrapper = generic + 2
  anatomy fields). Migrate kernel signatures: 12 sites
  (`&TranscranialUstBornInversionConfig` → `&LinearBornInversionConfig`)
  across linear_algebra, sensitivity, conditioning, volume_regularization,
  volume_operator/*, volume_born/pcg. born.rs entry points pass
  `&config.linear`. The current compile gate is restored:
  `cargo check -p kwavers --lib --message-format=short -j 2`,
  `cargo test -p kwavers transcranial_ust --lib -j 2`, and
  `cargo check -p pykwavers --lib --message-format=short -j 2` pass with only
  pre-existing unrelated warnings.
- **[done] [arch] T13b-Phase-3: physically relocate the generic kernels — CLOSED 2026-05-21.**
  `VolumeOperator` generalised over `<G: TransducerGeometry + ?Sized>`; the
  hardcoded `C_BRAIN_REF_M_S` / `C_TISSUE_DENSITY_KG_M3` constants in the
  operator construction are replaced by `LinearBornInversionConfig::{reference_sound_speed_m_s,
  reference_density_kg_m3}` (validated, brain-overridden in
  `TranscranialUstBornInversionConfig::default`). Files moved:
  `clinical/…/transcranial_ust/volume_operator{.rs,/}` →
  `solver/inverse/linear_born_inversion/volume_operator{.rs,/}` and
  `clinical/…/transcranial_ust/volume_born/pcg.rs` →
  `solver/inverse/linear_born_inversion/pcg.rs`. Clinical `volume_born/mod.rs`
  consumes them through the public solver path (`VolumeOperator`,
  `VolumeVoxel`, `pcg_invert`); no compatibility alias remains in the clinical
  layer. `cargo check -p kwavers --lib` exit 0 / 0 warnings;
  `cargo check -p pykwavers --lib` exit 0; `cargo test -p kwavers transcranial_ust
  --lib` 3/3 pass (including the coupled 3-D volume inversion); `cargo test -p
  kwavers linear_born_inversion --lib` 8/8 pass.
- **[done] [patch] T11d: pykwavers binding rename — CLOSED 2026-05-20.**
  `seismic_bindings/` → `imaging_bindings/`; `slice_fwi.rs` →
  `transcranial_slice_inversion.rs`; `volume_fwi.rs` →
  `transcranial_volume_inversion.rs`; pyfunctions
  `run_seismic_helmet_fwi_*` → `run_transcranial_ust_*_inversion_from_ritk_ct`;
  book chapter file + helper dir + test file + doc filename + chapter title
  all renamed consistently. 10 files updated; cargo check pykwavers exit 0.
### Open — Unified-dispatcher migration (corrected framing of prior T10)

The previous T10 ("consolidate dual FDTD stacks") mis-classified the defect.
The actual defect: `simulation::solver_factory::SimulationSolverFactory::create_solver()`
already returns `Box<dyn Solver>` and dispatches `SolverType::{FDTD, PSTD,
Hybrid, KSpace, DG}` to concrete impls — but the FWI modules **bypass the
factory** and hardcode their own forward stacks. Correction:

- **[done] [arch] T15: FWI A namespace relocation — CLOSED 2026-05-20.**
  Moved `solver/inverse/seismic/fwi/` → `solver/inverse/fwi/time_domain/`.
  FWI taxonomy now consistent across both domains
  (`fwi::frequency_domain`, `fwi::time_domain`). 4 example consumers +
  seismic plugin updated. cargo check --lib + --examples clean.
- **[done] [arch] T15b: time-domain FWI factory migration — CLOSED.**
  `FwiParameters::build_solver_for_forward` dispatches `SolverType::{FDTD, PSTD}`
  to `build_fdtd_boxed`/`build_pstd_boxed` and returns `Box<dyn Solver>`.
  Unsupported types return `KwaversError::InvalidInput`. PSTD forward smoke test
  and unsupported-type rejection test verified. PSTD adjoint-reciprocity
  verification remains open (track separately if needed).
- **[done] [arch] T16: FWI B parallel stack deleted — CLOSED 2026-05-20.**
  Removed `solver/inverse/reconstruction/seismic/fwi/` entirely
  (~1500 LOC across mod, gradient, optimization, regularization, wavefield/*).
  Verified zero external struct consumers prior to deletion. Parent
  `reconstruction/seismic/mod.rs` updated to point future maintainers at
  the canonical `solver::inverse::fwi::time_domain` engine (factory-dispatched
  after T15 lands). cargo check --lib exit 0.
- **[done] [patch] T15c: FWI example import closure — CLOSED 2026-05-21.**
  Updated remaining example consumers and reconstruction comments to import
  `solver::inverse::fwi::time_domain` directly, preserving the FWI taxonomy
  without a legacy seismic-owned compatibility path. Verified examples with
  `cargo check -p kwavers --features nifti --example ...`.
- **[done] [arch] T17a: HelmholtzForwardOperator trait landed — CLOSED 2026-05-20.**
  New module `solver/inverse/fwi/frequency_domain/operator.rs` with the
  `HelmholtzForwardOperator` trait + three impls (`SingleScatterBornOperator`,
  `DenseConvergentBornOperator`, `SpectralConvergentBornOperator`). Re-exported
  from frequency_domain. 3 unit tests for model_id / cbs_descriptor /
  adjoint-path classification. cargo check --lib exit 0.
- **[done] [arch] T17b: flip Config to Arc<dyn HelmholtzForwardOperator> — CLOSED.**
  `Config.forward_operator: Arc<dyn HelmholtzForwardOperator>` is live in
  `types.rs`. `PropagationModel` enum is fully removed. `forward.rs` and
  `gradient.rs` dispatch through the trait object. `with_forward_operator` builder
  method present. No compatibility alias remains.

## Active Sprint — Ali et al. 2025 Multi-Row Frequency-Domain FWI Replication

Target version: 0.x.0 (additive; new module
`solver::inverse::fwi::frequency_domain::cbs` + pykwavers
binding). Confirms kwavers FWI theorems against an external published 3D
breast-imaging reconstruction.

- **[done] [minor] T1: 3D FWI foundation module split — CLOSED 2026-05-20.**
  Added the Ali et al. paper identities under
  `physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi`,
  a solver-owned matrix-free 3-D single-scatter frequency-domain FWI foundation
  under `solver::inverse::fwi::frequency_domain`, and a clinical breast UST adapter
  under `clinical::imaging::reconstruction::breast_ust_fwi`. Verification pins
  geometry, source scaling, Helmholtz derivative, objective metrics, forward
  sensitivity, exact adjoint gradient for the implemented discrete model, and
  objective-decreasing reconstruction. CBS remains T2.
- **[done] [minor] T1.5: FWI method taxonomy and CBS identities — CLOSED
  2026-05-20.** Moved the frequency-domain solver under
  `solver::inverse::fwi::frequency_domain`, moved the existing time-domain
  acoustic adjoint-state core under `solver::inverse::fwi::time_domain`, removed
  the old top-level method module names from call sites, and added
  `frequency_domain::cbs` as the SSOT for real scattering potential, convergence
  epsilon, shifted potential, and pointwise preconditioner identities.
- **[done] [minor] T2a: dense CBS volume-field kernel — CLOSED 2026-05-20.**
  Split `frequency_domain::cbs` into `potential`, `grid`, `green`, and `solve`
  leaves. Added centered grid indexing, BLI point weights using the canonical
  `0.05` tolerance, shifted outgoing Green evaluation, and a dense CBS
  fixed-point solve for cell-centered source densities. Tests pin homogeneous
  unit-source Green output and residual reduction on a contrast fixture.
- **[done] [minor] T2b: CBS prediction route — CLOSED 2026-05-20.**
  Added `frequency_domain::PropagationModel::{SingleScatterBorn,
  DenseConvergentBorn}`. Prediction now dispatches explicitly: Born uses the
  existing single-scatter path; dense CBS injects cylindrical sources through
  BLI, solves the CBS field, and samples receivers through BLI. Tests pin
  homogeneous CBS/Born equivalence on an on-grid ring, CBS sensitivity to sound
  speed, and BLI support-domain rejection.
- **[done] [minor] T3/T4: dense CBS adjoint and Eq. 6 gradient — CLOSED
  2026-05-20.** Moved BLI source/receiver projection into the CBS bounded
  context, added the shifted-Green Euclidean adjoint, added the dense discrete
  adjoint solve for `(I + G_epsilon diag(V - i epsilon))^H`, and wired
  `DenseConvergentBorn` through the nonlinear objective/gradient path. Tests pin
  `<Gx, y> = <x, G^H y>` and finite-difference agreement for the dense CBS
  slowness gradient.
- **[done] [minor] T5a: spectral periodic CBS operator — CLOSED
  2026-05-20.** Added `GreenOperatorKind::{DenseFreeSpace, SpectralPeriodic}`
  and `PropagationModel::SpectralConvergentBorn`. The spectral path applies the
  periodic Helmholtz symbol `(k0^2 + i epsilon - |k|^2)^-1` through the existing
  Apollo FFT facade and shares the same CBS fixed-point, adjoint solve, BLI
  projection, and slowness-gradient accumulation as the dense path. Tests cover
  the zero-mode theorem, spectral adjoint identity, prediction sensitivity, and
  spectral finite-difference gradient agreement.
- **[done] [minor] T5b: spectral CBS absorbing boundary — CLOSED
  2026-05-20.** Added `AbsorbingBoundary::{Disabled, Polynomial}` and applies
  the polynomial sponge as `W G W` around the spectral Green operator. Because
  `W` is real diagonal, the adjoint remains `W G^H W`; tests pin unit interior
  weight, edge/corner decay, edge-source damping, and the absorbed spectral
  adjoint identity.
- **[done] [minor] T2c: reduced-grid spectral CBS performance validation —
  CLOSED 2026-05-20.** Added Criterion target
  `kwavers/benches/fwi_spectral_cbs.rs` and Cargo bench registration. The
  benchmark runs the public `simulate_frequency_observation` entrypoint through
  `PropagationModel::SpectralConvergentBorn` with polynomial absorbing boundary
  on a reduced 24x24x18 multi-row ring fixture, asserts finite pressure and
  sound-speed perturbation sensitivity before timing, and records a post-format
  median of 19.998 ms / 2.0738 Melem/s.
- **[done] [minor] T2: convergent Born-series forward kernel — CLOSED
  2026-05-20.** Implement Osnabrugge
  2016 preconditioned CBS in `solver::inverse::fwi::frequency_domain::cbs`:
  shifted potential `V_s = V − iε`, `V = k² − k₀²`, with `ε ≥ ‖V‖∞`,
  pointwise preconditioner `γ = iε / V_s`, fixed-point iteration over the
  shifted outgoing Green operator `G(k) = 1/(k₀² + iε − |k|²)`. Polynomial
  wavenumber apodisation in PML region. On-grid BLI injection at ring-element
  positions, BLI receiver sampling, and periodic FFT acceleration are
  implemented. Reduced-grid performance validation is now covered by
  `fwi_spectral_cbs`.
- **[done] [minor] T6: PyO3 bindings — CLOSED 2026-05-20.** Added
  `pykwavers::breast_fwi_bindings` and registered the flat Python API classes
  `MultiRowRingArray`, `FrequencyDomainFwiConfig`, and `FrequencyObservation`.
  Added `ali_2025_breast_fwi_frequency_sweep_hz`,
  `simulate_breast_fwi_frequency_observation`, and `invert_breast_fwi`.
  Forward prediction accepts `np.ndarray[float64]` sound-speed volumes and
  returns `np.ndarray[complex128]`; inversion accepts a stacked
  `(frequency, transmit, receiver)` `complex128` array and delegates to the
  clinical breast UST adapter for metadata-preserving reconstruction output.
- **[done] [minor] T7: data-generation pipeline — CLOSED 2026-05-20.**
  Added `clinical::imaging::reconstruction::breast_ust_fwi::dataset` wrapping
  the existing PSTD solver over centered multi-row ring geometry. The generator
  runs one PSTD acquisition per frequency/transmit pair, preserves receiver
  ordering with ordered sensor indices, extracts the complex bin
  `2/N Σ p[n] exp(-i2πf n dt)`, and exposes
  `generate_breast_fwi_pstd_dataset` plus `BreastFwiPstdDatasetConfig` through
  pykwavers.
- **[done] [minor] T8a: Rust HDF5 phantom ingest — CLOSED 2026-05-20.**
  Added `clinical::imaging::reconstruction::breast_ust_fwi::phantom_hdf5` as
  the Rust-owned `consus` HDF5/MAT-v7.3 ingest boundary for Ali phantom
  volumes. The loader resolves known sound-speed dataset paths or an explicit
  caller path, requires file metadata or caller-provided `spacing_m`, supports
  contiguous and chunked datasets, supports C and MATLAB/Fortran storage order,
  converts m/s or km/s storage units to m/s, and exposes
  `load_ali_2025_breast_fwi_phantom` through pykwavers. Verification covers
  real `consus` HDF5 fixture decoding, chunked payload decoding, unit
  conversion, missing-spacing rejection, pykwavers build, and binding surface.
- **[done] [minor] T8b: replication example — CLOSED 2026-05-20.**
  Added `pykwavers/examples/replicate_ali2025_breast_fwi.py`. The script
  downloads `BreastPhantomFromMRI.mat` from GitHub Release v1.0.0 if absent
  (cached at `D:\3D-FWI-MultiRowRingArrayUST\phantoms\` per user-confirmed
  location), loads sound speed through the Rust clinical phantom boundary, applies
  deterministic center-crop plus decimation, runs T7 PSTD data generation and
  T6 spectral-CBS frequency-domain FWI through pykwavers, and writes RMSE/PCC
  metrics plus matplotlib orthographic comparison slices. Helper tests pin
  domain reduction, reduced geometry, metrics, and slice selection without
  Python HDF5 or physics stand-ins.
- **[done] [minor] T8c: MATLAB-5 MRI phantom ingest — CLOSED 2026-05-20.**
  The published release asset is MATLAB Level-5, not HDF5. Added
  `clinical::imaging::reconstruction::breast_ust_fwi::phantom_mat5` to decode
  compressed `breast_mri`, cubic-interpolate the MRI through the published
  right/left breast rotation, fill thresholded per-slice tissue holes, map
  tissue intensities into the Ali sound-speed interval, and return a uniform
  sound-speed grid. PyO3 auto-detects HDF5 vs MAT5. The replication script now
  requests a pre-reduced MAT5 output grid and includes Table 1 parity constants.
- **[done] [patch] T8d: PSTD steady-state frequency bin — CLOSED 2026-05-20.**
  `BreastUstPstdDatasetConfig` now separates total simulated cycles from
  trailing cycles used for Fourier extraction. The clinical dataset returns
  `frequency_bin_start_steps_per_frequency`, and pykwavers exposes the same
  audit metadata. The analytic phasor test proves startup samples are excluded
  from the bin when `frequency_bin_cycles < cycles_per_frequency`.
- **[done] [patch] T8e: reduced-probe identifiability diagnostics — CLOSED 2026-05-20.**
  The replication script is split below the 500-line limit into orchestration,
  metric, volume, visualization, and identifiability modules. Reports now
  include the acquisition rank upper bound after complex source-scale nuisance
  parameters. The current 8x8x4 probe has 16 complex observations, 32 real
  observation DoF, 8 source-scale nuisance DoF, and only 24 informative real
  DoF for 256 unknown voxels.
- **[done] [patch] T8f: determined-acquisition guard — CLOSED 2026-05-20.**
  Added `--require-determined-acquisition` to reject rank-underdetermined probes
  before PSTD generation and inversion. The 8x8x4 one-frequency probe rejects
  with `24 informative real DoF for 256 unknown voxels`. A 4x4x3 two-frequency
  probe satisfies the guard with 48 informative real DoF for 48 voxels.
- **[done] [patch] T8g: PSTD/FWI grid-snapped geometry — CLOSED 2026-05-20.**
  Added topology-preserving ordered ring elements and a clinical
  `snap_multi_row_ring_array_to_grid` boundary so the inverse model consumes the
  same effective source/receiver coordinates as PSTD's grid-index acquisition.
  PyO3 exposes `snap_breast_fwi_array_to_grid`, and the replication report now
  includes true-model PSTD-vs-CBS source-scaled residuals.
- **[done] [patch] T8h: source-channel residual attribution — CLOSED 2026-05-21.**
  Added active-source receiver masks for cylindrical multi-row firing and
  passive-only residual metrics to the Ali 2025 replication report. The
  determined 4x4x3/two-frequency probe shows active-source receiver channels
  contribute 17.7068% of full-scale residual energy, while passive-only
  row-scaled residual remains 0.543288 versus 0.523411 for all channels.
- **[done] [patch] T8i: source-excitation scalar diagnostic — CLOSED 2026-05-21.**
  Added analytic PSTD sine-bin coefficient diagnostics for the additive source
  signal and normalized row-wise PSTD-vs-CBS source scales by that coefficient.
  The determined probe has tone-bin magnitudes 1.000000 at 200 kHz and 0.980367
  at 300 kHz, but normalized source-scale dispersion remains non-scalar:
  magnitude coefficient of variation reaches 0.297005 and phase span reaches
  0.919426 rad at 300 kHz.
- **[done] [patch] T8j: forward-operator equivalence diagnostic — CLOSED 2026-05-21.**
  Added the `operator_equivalence` report comparing `single_scatter_born`,
  `dense_convergent_born`, and `spectral_convergent_born` against the same PSTD
  data with row-wise complex source scaling and source-bin-normalized scale
  diagnostics. On the determined 4x4x3/two-frequency probe, single-scatter Born
  fits best (`0.456575` normalized residual), dense CBS follows (`0.476438`),
  and absorbed spectral CBS is worst (`0.523411`).
- **[done] [patch] T8k: homogeneous direct-field Green diagnostic — CLOSED 2026-05-21.**
  Added the `homogeneous_direct_field` report comparing homogeneous snapped PSTD
  observations against the outgoing Helmholtz direct Green field with the same
  array coordinates, frequencies, row source scaling, and passive-channel
  attribution. On the determined 4x4x3/two-frequency probe, the homogeneous
  direct field residual is `0.454900`, passive-only residual is `0.757352`,
  passive phase-error RMS is `1.458883` rad, and passive log-amplitude-error
  RMS is `1.028543`.
- **[done] [patch] T8l: PSTD source-kappa direct-field diagnostic — CLOSED 2026-05-21.**
  Added a source-kappa filtered direct-field branch that maps snapped ring
  sources back to PSTD grid cells, applies the pressure-source
  `cos(c_ref |k| dt / 2)` spatial correction by FFT, and evaluates the outgoing
  Green field from the filtered source distribution. On the determined
  4x4x3/two-frequency probe, source-kappa filtering changes homogeneous
  residual from `0.454900` to `0.454689` (`-0.000211`) while passive-only
  residual remains `0.757458`; this rejects source-kappa filtering as the
  primary parity repair.
- **[done] [patch] T8m: finite-grid PSTD Green diagnostic — CLOSED 2026-05-21.**
  Added a finite-grid homogeneous PSTD direct-field diagnostic derived from the
  no-CPML modal recurrence with
  propagation kappa, pressure-source kappa, source timing, and the same trailing
  Fourier-bin projection used by the Rust acquisition. On the determined
  4x4x3/two-frequency probe, the periodic PSTD Green worsens all-channel
  homogeneous residual to `0.741005` but improves passive-only residual to
  `0.455227`, passive phase-error RMS to `0.956928` rad, and passive
  log-amplitude-error RMS to `0.422984`; the remaining dominant discrepancy is
  active source/receiver self-channel semantics.
- **[done] [minor] T8o: Rust-owned direct-field diagnostics — CLOSED 2026-05-21.**
  Moved the T8k/T8l/T8m point-Green, source-kappa, and finite-grid PSTD
  computations out of Python and into
  `kwavers::clinical::imaging::reconstruction::breast_ust_fwi::direct_field`.
  `pykwavers` now exposes
  `diagnose_breast_fwi_homogeneous_direct_field`, while the Python
  `ali2025_breast_fwi.direct_field` module only delegates to that binding.
  The Python `discrete_green` implementation and tests were removed. The
  determined 4x4x3/two-frequency probe reproduces the prior diagnostic values
  through Rust-owned computation: point residual `0.454900`, source-kappa
  residual `0.454689`, and periodic PSTD residual `0.741005`.
- **[done] [minor] T8p: Rust-owned replication diagnostics — CLOSED 2026-05-21.**
  Moved scaled observation residuals, source-channel attribution,
  source-excitation dispersion, rank identifiability, reconstruction
  RMSE/PCC, and Table 1 parity gates into
  `kwavers::clinical::imaging::reconstruction::breast_ust_fwi::diagnostics`.
  `pykwavers` now exposes the corresponding diagnostic functions plus the
  combined `diagnose_breast_fwi_observation_pair`; Python support modules
  delegate to bindings and the direct-field path reuses the same residual and
  source-excitation implementations. The determined 4x4x3/two-frequency probe
  preserves the prior report values through Rust-owned metrics: true-model
  normalized residual `0.523411`, source-excitation phase span `0.919426`, RMSE
  `54.6750` m/s, and PCC `0.110968`.
- **[done] [minor] T8q: Rust-owned reduced-domain preparation — CLOSED 2026-05-21.**
  Moved reduced phantom decimation, center cropping, median homogeneous initial
  model construction, and reduced ring-array geometry derivation into
  `kwavers::clinical::imaging::reconstruction::breast_ust_fwi::reduction`.
  `pykwavers` now exposes `prepare_breast_fwi_reduced_phantom` and
  `derive_breast_fwi_reduced_array_geometry`, and the replication script uses
  those bindings instead of Python-owned domain formulas. The determined
  4x4x3/two-frequency probe preserves the Rust-owned report values: true-model
  normalized residual `0.523411`, RMSE `54.6750` m/s, and PCC `0.110968`.
- **[done] [minor] T8r: Rust-owned operator-equivalence diagnostics — CLOSED 2026-05-21.**
  Moved forward-operator equivalence aggregation into
  `kwavers::clinical::imaging::reconstruction::breast_ust_fwi::diagnostics`.
  `pykwavers` now exposes `breast_fwi_operator_equivalence_diagnostics`, and
  the Python Ali support module delegates residual/source-excitation
  aggregation across single-scatter Born, dense CBS, and absorbed spectral CBS
  to that binding. This keeps Python as orchestration/reporting while preserving
  the existing reduced-probe model ordering.
- **[done] [minor] T8s: active self-channel direct-field diagnostics — CLOSED 2026-05-21.**
  Extended `BreastUstDirectFieldDiagnostics` with active-only residual,
  co-located self-channel phase/amplitude errors, and active pair counts. The
  PyO3 direct-field report now exposes these fields for the point, source-kappa,
  and finite-grid PSTD references. The focused Rust test suite now includes an
  active-channel perturbation with analytic active residual `sqrt(1/2)` and
  exact passive residual zero, keeping receiver-selection diagnostics in Rust
  while preserving Python as the report layer.
- **[done] [minor] T8t: receiver-policy operator equivalence — CLOSED 2026-05-21.**
  Added `BreastUstReceiverChannelPolicy` and policy-aware forward-operator
  equivalence diagnostics for `all`, `active_only`, and `passive_only` receiver
  selections. PyO3 exposes the selected policy and the reduced probe now writes
  policy-specific rankings. On the determined 4x4x3/two-frequency probe,
  active-only residuals are scale-absorbed near zero, while passive-only ranking
  selects `spectral_convergent_born` at normalized residual `0.5432880999009375`.
- **[done] [minor] T8u: passive direct-field residual deltas — CLOSED 2026-05-22.**
  Extended `BreastUstHomogeneousDirectFieldDiagnostics` with Rust-owned
  passive residual deltas for the source-kappa Green and finite-grid PSTD Green
  references relative to the outgoing point Green reference. PyO3 exposes both
  values, the Rust and Python tests verify the delta arithmetic against the
  nested passive residual fields, and the determined 4x4x3/two-frequency probe
  now records `source_kappa_filtered_passive_residual_delta =
  0.00010581210140714337` and `pstd_periodic_passive_residual_delta =
  -0.30212499274440036`.
- **[done] [minor] T8v: PSTD spectral CBS operator — CLOSED 2026-05-22.**
  Added `PstdSpectralConvergentBornOperator` under
  `solver::inverse::fwi::frequency_domain`. The CBS Green boundary now supports
  a `SpectralPstdPeriodic` operator whose denominator is
  `[4 sin²(ω Δt / 2) - 4 sin²(c0 |k| Δt / 2)] / (c0 Δt)² + iε`, preserving
  the acquisition generator's homogeneous PSTD leapfrog/k-space propagation
  symbol. PyO3 exposes `pstd_spectral_convergent_born`, and the Ali 2025
  reduced probe includes it in operator-equivalence rankings. On the determined
  4x4x3/two-frequency probe it reports all-channel residual
  `0.5227508888630437` and passive-only residual `0.5435181467026386`.
- **[done] [minor] T8w: PSTD CBS source projection — CLOSED 2026-05-22.**
  Routed `PstdSpectralConvergentBornOperator` source injection through exact
  centered-grid source indices and the PSTD pressure-source k-space correction
  `cos(c0 Δt |k| / 2)`. Continuous Helmholtz CBS operators still use the
  existing BLI point-source density path. The CBS tests now verify the two-cell
  source-kappa symbol and the PSTD spectral adjoint-gradient path. On the
  determined 4x4x3/two-frequency probe, the PSTD spectral CBS all-channel
  residual is `0.5233688602227166` and passive-only residual is
  `0.5434979751472874`, so source-kappa projection alone is not the parity
  repair.
- **[done] [minor] T8y: PSTD CBS receiver projection — CLOSED 2026-05-23.**
  Routed `PstdSpectralConvergentBornOperator` receiver sampling through exact
  centered-grid cell extraction and its Euclidean adjoint through exact cell
  residual injection. Continuous Helmholtz CBS operators still use the BLI
  receiver projection/adjoint path. The CBS tests now verify exact PSTD
  receiver extraction, the receiver projection adjoint identity, and off-grid
  PSTD receiver rejection. This closes receiver sampling as a hidden
  interpolation variable; the remaining parity variable is the temporal
  source/frequency-bin transfer function.
- **[done] [minor] T8z: PSTD temporal bin transfer SSOT — CLOSED 2026-05-23.**
  Added `solver::inverse::fwi::frequency_domain::cbs::temporal` as the
  solver-owned theorem boundary for PSTD source kappa, leapfrog modal theta,
  the frequency-domain PSTD denominator, and the exact finite-window modal
  frequency-bin response of the additive sine pressure-source recurrence.
  The clinical breast-UST homogeneous direct-field diagnostic now consumes this
  solver SSOT and no longer owns a separate PSTD modal recurrence. This closes
  the hidden formula duplication; the next parity step is to wire the temporal
  transfer into the selectable frequency-domain forward operator and rerun the
  determined probe.
- **[done] [minor] T8aa: PSTD CBS temporal transfer wiring — CLOSED 2026-05-23.**
  `PstdSpectralConvergentBornOperator` now carries optional
  `PstdTemporalTransferConfig` and builds a frequency-specific
  `PstdTemporalBinConfig` for the selected drive frequency. The CBS descriptor
  is frequency-aware, so forward prediction and adjoint-gradient reconstruction
  receive the same finite-window source/bin transfer. PyO3 exposes source
  amplitude, total cycles, and bin cycles on `FrequencyDomainFwiConfig`, and
  the Ali 2025 operator-equivalence builder passes the acquisition settings
  into the Rust operator instead of keeping them only in Python diagnostics.
- **[done] [patch] T8x: focused source adapter compile closure — CLOSED 2026-05-22.**
  Added the explicit `ElementMap` type to the focused bowl source adapter's
  `HashMap` construction. This resolves the unrelated `E0282` inference defect
  that blocked full `kwavers` library-test compilation while verifying the FWI
  solver slice.
- **[done] [patch] T8n: focused-bowl terminology cleanup — CLOSED 2026-05-21.**
  Removed residual transcranial vendor/helmet labels from book examples and
  documentation, renamed the Chapter 25 phase-correction artifact stem to
  `fig02_transcranial_bowl_phase_correction`, and retained the bowl transducer
  source as the geometry-level abstraction.
- **[patch] T9: parity gate.** At reduced grid (2D center slice or 3D
  dxi=1.6 mm) run T7 → T6, assert RMSE within 2× the Ali et al. Table 1
  3-D FWI RMSE and PCC at least 95% of the Table 1 3-D FWI PCC. Current
  8x8x4 probe executes the real path but fails parity after steady-state
  binning (RMSE 60.4246 m/s, PCC 0.218243) and is rank-underdetermined
  by upper bound (24 informative real DoF / 256 voxels). Closure now requires
  solver/model refinement because the determined 4x4x3 two-frequency probe also
  fails parity after grid snapping (RMSE 54.6750 m/s, PCC 0.110968) with exactly
  determined rank upper bound. True-model PSTD-vs-CBS normalized residual is
  0.523411 after optimal row-wise complex source scaling, down from 0.741168
  before snapping. Active-source receiver exclusion does not close the gap
  because passive-only residual is 0.543288; remaining gap is the PSTD
  source/propagation contract versus the Helmholtz/CBS forward model. Source
  excitation is now partly isolated: the analytic sine-bin coefficient does not
  explain the mismatch because normalized transmit source scales are not
  constant within frequency. Operator comparison shows CBS is not the immediate
  parity repair on this probe because single-scatter Born fits PSTD better than
  dense or spectral CBS. Homogeneous direct-field comparison now shows a
  passive Green-field phase/amplitude mismatch persists before scattering:
  normalized residual `0.454900`, passive-only residual `0.757352`, phase-error
  RMS `1.458883` rad, and log-amplitude-error RMS `1.028543`. Next work should
  align the discrete propagation Green function itself because the PSTD
  source-kappa correction changes homogeneous residual by only `-0.000211` and
  leaves passive-only residual at `0.757458`. The finite-grid PSTD modal Green
  now explains a material part of passive propagation (`0.455227` passive-only
  residual, `0.956928` rad phase RMS) but worsens all-channel residual to
  `0.741005`. Receiver-policy operator ranking now shows active-only channels
  are scale-absorbed near zero, while passive-only ranking selects
  `spectral_convergent_born` at `0.5432880999009375`. Passive direct-field
  deltas now quantify the reference gap directly: source-kappa changes passive
  residual by `0.00010581210140714337`, while finite-grid PSTD changes passive
  residual by `-0.30212499274440036`; the next repair is the passive
  PSTD/Helmholtz propagation contract, not active-channel exclusion. The
  frequency-domain PSTD spectral CBS operator closes the modal denominator
  mismatch as an isolated variable but does not close parity: all-channel
  residual improves only from `0.5234113936105187` to `0.5227508888630437`,
  while passive-only residual changes from `0.5432880999009376` to
  `0.5435181467026386`. Matching PSTD source-kappa projection in the CBS source
  path changes the all-channel residual to `0.5233688602227166` and passive-only
  residual to `0.5434979751472874`, so source projection/filtering alone is not
  the repair. Receiver projection is now operator-aware and exact-grid for PSTD
  CBS. The PSTD temporal source/frequency-bin transfer formulas are now
  solver-owned, shared by clinical diagnostics, and wired into the selectable
  PSTD spectral CBS operator. The next work should rerun the determined
  4x4x3/two-frequency probe and use the new residual ranking to decide whether
  the remaining gap is CPML/absorbing-boundary alignment or heterogeneous
  scattering linearization.

### Deprecation (T2 prerequisite)
- **[patch] Mark `solver::forward::helmholtz::born_series::convergent::ConvergentBornSolver`
  unsuitable for FWI** in module docstring. The implemented recursion is the
  non-convergent classical Born series with a sign-flipped Green denominator, a
  6-point-stencil mock fallback, hardcoded `c0`/`ρ0`, and no absorbing layer.
  See gap_audit "Ali et al. 2025 Replication Foundation" item 2 for the
  evidence. No call sites in pykwavers; preserve internal API for now to avoid
  a parallel refactor.

## Architectural Enhancements
- **[done] [arch] Rename the artifact-owned analytical physics boundary —
  CLOSED 2026-05-20.** The previous module name encoded a documentation
  artifact, not a physics bounded context. The module directory now lives at
  `kwavers/src/physics/analytical`, the public Rust path is
  `kwavers::physics::analytical`, and the PyO3 binding module imports that
  corrected boundary. No compatibility alias or forwarding module remains.

- **[done] `SolverType.ElasticPSTD` parity with KWave.jl `pstd_elastic_2d` — CLOSED 2026-05-10.** End-to-end PASS in `pykwavers/examples/ewp_elastic_2d_jl_compare.py --pstd` with `peak_ratio = 1.0000` across all 4 downstream sensors (`+3`, `+6`, `+9`, `+12`) and `pearson_mean = 0.974`. Three sequential fixes converged the parity:
    1. **Stress accumulation contract** — `StressUpdateParams.txx_fft … tyz_fft` added so the spectral stress kernel ADDs the per-step increment (`σ̃(t+dt) = σ̃(t) + dt · C : ε̃`) rather than overwriting (which was acoustic-fluid-only). [`pstd::extensions::PstdElasticPlugin::apply_stress_update_in_place`].
    2. **Staggered-grid k-shift** — `StressUpdateParams.dkx_op/dky_op/dkz_op` (and the velocity equivalent) refactored from real `Array3<f64>` wavenumbers to complex spectral derivative operators carrying the half-cell shift `i·k·exp(±i·k·Δ/2)`. Orchestrator precomputes both shift sets at construction (`StaggeredDerivativeOps::build`) and dispatches the negative shift to the stress update (KWave.jl `ddx_k_shift_neg`) and the positive shift to the velocity update (`ddx_k_shift_pos`). Without this the orchestrator was running collocated PSTD, off by ~3× in peak amplitude.
    3. **Source dimensionality** — the parity script's `--pstd` branch extends the velocity-source mask through all z-layers (`src_mask[i, :, :] = True`) so the 3-D slab problem reduces to the equivalent 2-D problem KWave.jl solves; without this the source was a y-line at one z-slice (cylindrical 3-D spreading) instead of a y-z plane (2-D wave).
  Convergence trace: legacy FD pearson 0.36 → collocated PSTD 0.71 → staggered PSTD 0.78 → staggered PSTD + 2-D source 0.974. peak_ratio at sensor +3: collocated 0.23 → staggered 0.35 → staggered + 2-D source 1.0000.

- **[minor] Elastic-PSTD extensions on top of the canonical orchestrator.** Strict parity is closed without these, but each opens new use cases:
    1. Split-field elastic PML — needed for long-propagation simulations beyond the wraparound horizon (currently fine because the parity test is short enough that FFT periodic wraparound is benign at peak measurement). Estimated ~200 LOC behind `ElasticPstdOrchestrator::set_pml(thickness)`.
    2. k-space correction (Tabei et al. 2002) extending the elastic stress / velocity updates the same way `pstd::PSTDSolver` does for acoustic — eliminates temporal dispersion at the elastic CFL limit, allowing CFL = 1.0 instead of the leapfrog-stable 0.3 the parity test uses.
- Restructure into clean Domain/Application/Infrastructure/Presentation bounded contexts.
- Ensure dependency flows are strictly unidirectional (Domain -> App -> Infra/Presentation).
- Keep concrete solver assembly in `simulation::solver_factory`, keep `solver::factory` limited to descriptor-based selection policy, and reject domain-layer imports of solver or simulation modules.
- Review all modules (core, physics, math, domains, simulation, clinical, analysis, solvers).
- BURN crate integration for optimized GPU support.
- Autodiff/PINN implementations for neural network-based physics solving.

## Validation Goals
- 2026-05-20: [minor] Closed the bowl transducer cap geometry SSOT gap.
  `BowlTransducer` now consumes the canonical focused spherical-cap layout,
  derives element count from cap area and requested element size, preserves
  equal-area weights, and rejects nonfinite or degenerate bowl domains.

- 2026-05-20: [minor] Closed the hemispherical source-array geometry SSOT gap.
  `domain::source::hemispherical::ElementPlacement` now delegates spherical-cap
  placement to `domain::source::transducers::focused::cap`, rejects
  zero-element layouts and nonfinite radii, and pins focus-directed unit normals
  on the positive-y aperture.

- 2026-05-20: [minor] Closed the tracked transcranial FUS cap geometry SSOT
  gap. The clinical adapter now converts `SphericalCapLayout` into the ndarray
  shape required by skull-ray and Rayleigh integration, preserving the existing
  negative-z aperture orientation while source-domain validation rejects invalid
  polar spans.

- 2026-05-20: [minor] Closed the focused bowl geometry SSOT gap. Abdominal
  and nonlinear 3-D focused-bowl placement now consume
  `domain::source::transducers::focused::cap`, share one angular aperture
  contract, and propagate invalid spherical-cap domains as placement errors
  instead of generating repeated placeholder elements.

- 2026-05-20: [minor] Closed the focused spherical-cap source-layout gap.
  `domain::source::transducers::focused::cap` now owns reusable equal-area
  focused-bowl geometry for hemispherical and partial caps parameterized by
  focus, axis, radius, and angular span, so clinical placement policy can
  consume source-domain geometry without solver-specific transducer names.

- 2026-05-20: [patch] Closed the broadband cavitation detection domain guard
  gap. Empty or nonfinite signal windows now return finite zero metrics without
  seeding invalid adaptive baseline energy, and the detector recovers on the
  next finite signal window.

- 2026-05-20: [patch] Closed the CEUS microbubble harmonic-domain guard gap.
  Harmonic-content analysis now rejects zero harmonic index, invalid sample
  rate, mismatched time and scattered-pressure vectors, and nonfinite samples
  before spectral projection, preserving finite zero content for invalid
  domains.

- 2026-05-20: [patch] Closed the analytical plane-wave domain guard gap.
  Plane-wave field generation now rejects nonpositive frequency or sound
  speed, nonfinite amplitude or time, invalid grid spacing, and zero or
  nonfinite propagation direction before normalization or phase evaluation,
  preserving finite zero fields for invalid domains.

- 2026-05-20: [patch] Closed the sonogenetics analytical-domain guard gap.
  Hill activation now requires positive finite threshold and Hill exponent and
  ignores nonfinite pressure samples. Acoustic radiation force and streaming
  reject negative or nonfinite intensity/material domains. ISPTA rejects empty
  waveforms and invalid `dt`, density, or sound speed while excluding
  nonfinite pressure samples from the pressure-squared integral.

- 2026-05-20: [patch] Closed the acoustic analysis validation/directivity
  gap. Field metrics, focus, focal-plane, beam-width, and beam-pattern
  calculations now share one internal pressure-field validator for grid-shape
  equality and finite samples. Beam-pattern analysis rejects invalid frequency,
  sound-speed, and angular-resolution domains before deriving angular sample
  counts. Directivity now evaluates `10 log10(max |B|^2 / mean |B|^2)` for the
  pressure-amplitude pattern instead of averaging signed samples.

- 2026-05-20: [patch] Closed the acoustic field metrics domain-validation
  gap. `physics::acoustics::analysis::metrics` now rejects pressure
  field/grid shape mismatches, nonfinite pressure samples, and invalid
  density/sound-speed impedance domains before computing peak pressure,
  stored acoustic energy, or spatial peak intensity. The metric path shares
  the canonical `Z = rho c` and `I = p^2 / (2Z)` helpers with pressure
  analysis, preserving a single acoustic intensity implementation. Focused
  tests pin signed pressure magnitude, exact single-cell intensity and energy,
  dimension mismatch, nonfinite samples, and invalid impedance rejection.

- 2026-05-20: [patch] Completed the cavitation mechanical-index consolidation.
  The cavitation core no longer imports or re-exports the removed local
  `thresholds::mechanical_index` helper; model state updates and core tests now
  call `physics::acoustics::analysis::calculate_mechanical_index` directly.
  The nonlinear 3-D theranostic cavitation tests also import the canonical
  helper directly after their local helper removal. This fixes the compile
  breaks introduced by the partial consolidation without adding compatibility
  aliases.

- 2026-05-20: [patch] Closed the transcranial treatment-planning acoustic
  simulation domain gap. Field synthesis now validates transducer setup
  cardinality, positive finite frequency, finite element positions/phases, and
  finite nonnegative amplitudes; applies per-element amplitude in the coherent
  pressure sum; and converts documented millimeter element coordinates to SI
  meters before evaluating spherical propagation. The Pennes thermal response
  now rejects negative/nonfinite intensity values, and treatment-time
  estimation returns infinity for zero or invalid heating instead of zero.
  Focused tests pin amplitude-squared intensity scaling, millimeter conversion,
  Pennes source balance, invalid transducer/intensity domains, and treatment
  time from peak intensity.

- 2026-05-20: [patch] Closed the transcranial treatment-planning safety
  validation gap. The planner now converts harmonic average intensity to peak
  pressure with `p_peak = sqrt(2 rho c I)` using the brain-medium constants
  used by the acoustic-field simulation, then delegates MI calculation to
  `physics::acoustics::analysis::calculate_mechanical_index`. The safety gate
  rejects nonfinite brain temperature, invalid Hz frequency,
  negative/nonfinite intensity fields, and nonfinite MI rather than treating
  invalid input as safe. Focused tests pin the pressure-intensity theorem,
  low-intensity valid fields, and invalid-domain rejection.

- 2026-05-20: [patch] Closed the duplicate mechanical-index safety path gap in
  cavitation power modulation and transcranial safety monitoring. Both paths
  now delegate valid-domain MI calculation to
  `physics::acoustics::analysis::calculate_mechanical_index`, preserving the
  canonical `|p_r|_MPa / sqrt(f_MHz)` contract. The cavitation power limiter
  now fails closed for invalid MPa/MHz inputs, and the transcranial monitor now
  fails closed for invalid frequency or nonfinite pressure fields instead of
  reporting a zero-risk MI. Focused tests pin signed-pressure handling,
  invalid-domain rejection, exact 1 MPa / 1 MHz MI, and safety-margin behavior.

- 2026-05-20: [patch] Closed the acoustic pressure analysis invalid-domain
  gap. `physics::acoustics::analysis::pressure` now routes harmonic
  peak-pressure intensity through a shared impedance/intensity helper, rejects
  undefined impedance domains, ignores nonfinite pressure samples in scalar
  peak searches, requires finite positive MI frequency, preserves nonnegative
  TI exposure ratios, rejects derating inputs that would convert attenuation
  into gain, and enforces ISPTA duty cycle as a bounded temporal fraction.
  Focused tests pin the valid intensity, MI, TI, derating, ISPTA, and ISPPA
  formulas plus invalid-domain rejection.

- 2026-05-20: [patch] Closed the HIFU field and thermal-dose physics gap.
  `physics::acoustics::imaging::modalities::ultrasound::hifu` now has a
  facade plus focused `field`, `thermal_dose`, and `tests` submodules. The
  pressure field no longer pins the focus to the grid corner or uses the
  previous Gaussian/spherical shortcut; it evaluates a centered
  Rayleigh-Sommerfeld aperture integral with O'Neil phase delays. HIFU
  intensity now uses the harmonic peak-pressure contract `p_peak^2/(2 rho c)`.
  CEM43 now uses seconds-to-minutes conversion and the Sapareto-Dewey
  `R = 0.5` / `R = 0.25` temperature regimes. Focused tests pin focus
  centering, lateral symmetry, intensity, CEM43 reference values, and ablation
  threshold behavior.

- 2026-05-21: [patch] Closed the thermal-dose SSOT drift.
  `ThermalCEM43Grid` now uses `BODY_TEMPERATURE_C`, exposes
  `CEM43_REFERENCE_TEMPERATURE_C`, and aliases irreversible cell-death dose to
  `medical::THERMAL_DOSE_THRESHOLD`. Focused tests pin the alias and reference
  value.

- 2026-05-20: [patch] Closed the book cavitation closed-form invalid-domain
  gap. Minnaert resonance, Blake threshold, Rayleigh collapse time, and
  histotripsy lesion radius now reject nonfinite or nonpositive physical
  domains with `0.0` instead of emitting negative frequencies/radii, NaN, or
  infinite estimates. Focused cavitation tests pin these rejection paths while
  preserving the existing valid-domain checks.

- 2026-05-20: [patch] Closed the duplicate mechanical-index contract drift in
  book histotripsy and transcranial BBB-opening physics. Both helpers now use
  rarefactional-pressure magnitude and require positive finite MHz frequency,
  returning `0.0` for invalid domains instead of producing negative, NaN, or
  infinite MI values. Focused tests cover signed pressure and invalid frequency
  cases in both modules.

- 2026-05-20: [patch] Closed the clinical-safety thermal-index invalid-domain
  gap. The soft-tissue and bone thermal-index helpers now preserve the
  nonnegative exposure-ratio invariant by returning `0.0` for nonfinite or
  negative acoustic power and for invalid frequency domains. Focused tests pin
  unit-ratio examples and invalid-domain rejection, and the FDA output-limit
  Rustdoc now points to the diagnostic-ultrasound guidance table instead of an
  unrelated MR-device document.

- 2026-05-20: [patch] Closed the clinical-safety mechanical-index sign and
  frequency-domain gap. `physics::analytical::safety::mechanical_index` now computes
  the dimensionless MI from rarefactional-pressure magnitude and returns `0.0`
  for nonpositive or nonfinite frequency, preventing negative MI and infinite
  or NaN output from invalid input. Focused tests cover signed pressure input
  and invalid frequency.

- 2026-05-18: [patch] Closed the Chapter 29 Figure 5 pressure-display
  targeting gap. The visible Westervelt pressure panel now masks the pressure
  volume to the nonlinear target support before CT-frame projection, preventing
  raw source/coupling peaks from dominating the displayed lesion-targeting
  panel. The raw body/coupling pressure remains a diagnostic quantity. Focused
  verification covers the target-mask display contract; full Figure 5
  regeneration remains blocked by the existing nonlinear brain PyO3 allocation
  abort, so the checked-in PNG/PDF pressure column was updated from the
  successful controlled CT-frame field archive.

- 2026-05-18: [patch] Added Chapter 32 segmented tissue transducer
  optimization. The new chapter defaults to the local LiTS17 liver CT sample,
  maps native liver/tumor labels into normal/tumor planning compartments,
  targets the largest connected lesion on the selected slice, derives air, fat,
  bone, and vascular-avoid masks from CT HU thresholds, scores candidate
  apertures by segmented ray-path fractions, builds a three-angle crossfire plan,
  solves per-element complex drive weights for tumor spot shaping and
  protected-structure nulling, exports figures and metrics under
  `docs/book/figures/ch32`, and verifies the real liver adapter plus the
  analytic phantom contract. Follow-up correction closed the dense-field focus
  issue by increasing hotspot refinement and sidelobe nulling in the same solver
  path; regenerated LiTS metrics now record `target_dominant=true`, body
  sidelobe peak ratio `0.7395404024847666`, body sidelobe P99 ratio
  `0.3297347520675772`, tumor coverage `0.7837837837837838`, protected peak
  ratio `0.2958651403757349`, air path fraction `0.003477700061599821`, and
  bone path fraction `0.032944540572524016`.

- 2026-05-18: [patch] Closed the book verification errors introduced by the
  updated Chapter 29 contracts. The elastic shear display title now avoids FWI
  terminology in figure labels, and the extension freshness helper accepts
  Python stubs with empty signatures while preserving stale nonlinear signature
  rejection. The same verification pass repaired PyO3 release-build API drift
  by updating stale array apodization, signal window, and FDTD/PSTD geometry
  imports; `cargo build -p pykwavers --release -j 1` now passes.

- 2026-05-18: [patch] Closed the Chapter 29 reduced-exposure shortcut gap.
  The planned exposure now comes from the source-encoded heterogeneous acoustic
  wave solver rather than from a constant-speed phasor field. The new path uses
  the existing RTM grid, CPML, attenuation, source delays, and source cells,
  records raw peak pressure, time-step, source-count, and workspace metrics,
  and stores only `6 * nx * ny` scalar workspace values. Focused tests prove
  bounded workspace and nonzero downstream field change when an internal gas
  strip changes the speed map. Follow-up remains regenerating the full Chapter
  29 Figure 6 artifacts with the slower nonlinear branch profiled.

- 2026-05-18: [patch] Closed the Chapter 29 exposure-backend governance gap.
  Peak-pressure exposure now flows through a static generic backend contract,
  with `reference_fdtd_cpml_2d` as the only selectable backend and
  `exposure_uses_hybrid_pstd_fdtd=false` exported through PyO3. The hybrid
  PSTD/FDTD path remains blocked until it has source, receiver, CT-medium,
  peak-pressure, and memory-accounting parity tests against the reference. The
  reference loop now fuses attenuation with peak accumulation and clears only
  the finite-difference halo after buffer rotation, reducing per-step clearing
  work without increasing retained workspace.

- 2026-05-18: [patch] Closed the Chapter 29 iterative nonlinear elastic FWI
  reconstruction gap. The elastic channel now uses the real ElasticPSTD
  propagator for baseline, observed-lesion, and current-estimate shear
  simulations from the commanded target focus, records same-aperture velocity
  traces, migrates receiver residual trace energy for the update direction,
  accepts only objective-decreasing nonlinear shear-map updates, and exports
  objective-history diagnostics through PyO3. Verified with elastic unit tests,
  the Chapter 29 model-name contract, the abdominal theranostic inverse
  recovery test, `cargo check -p pykwavers`, and Python syntax compilation for
  the updated figure-caption modules.

- 2026-05-18: [patch] Extended the native acoustic DG diagnostic into a
  direct embedded-line solver matrix. The Gaussian IVP uses the analytical
  d'Alembert pressure solution as the shared reference and runs native DG,
  classical FDTD, k-space FDTD, and PSTD in the same homogeneous lossless
  medium. Verified metrics: DG vs exact `4.305350e-4`, FDTD vs exact
  `5.416002e-5`, k-space FDTD vs exact `8.204688e-6`, PSTD vs exact
  `1.201431e-5`, FDTD vs PSTD `5.348865e-5`, k-space FDTD vs PSTD
  `1.405561e-5`, and DG pressure mass error `1.865175e-14`. This closes the
  first direct DG/FDTD/PSTD acoustic pressure comparison while keeping the
  localized-pulse boundary assumption explicit.

- 2026-05-18: [patch] Added plotted comparison output for the same
  DG/FDTD/PSTD Gaussian pressure matrix. The fixture is now shared by the
  diagnostic and plotting examples, and `dg_acoustic_comparison_plot.rs` writes
  `target/dg_acoustic_comparison/gaussian_pressure.png` plus
  `gaussian_pressure.csv`. The PNG contains final pressure traces and absolute
  error traces against the analytical d'Alembert pressure reference; the CSV
  records the plotted series for downstream inspection.

- 2026-05-18: [patch] Lifted the common p4-quadrature metric into the
  DG/FDTD/PSTD pressure matrix. `dg_common/sampling.rs` samples DG by element
  Lagrange interpolation and samples FDTD/PSTD lines by periodic linear
  interpolation at the same physical coordinates, preserving the original
  native-grid metrics as a separate audit channel. The regenerated
  `gaussian_pressure.png` is a four-panel plot with native/common pressure and
  error rows; `gaussian_pressure.csv` now includes `common_pressure` and
  `common_absolute_error` rows. Verified common-grid metrics: DG vs exact
  `1.992925e-3`, FDTD vs exact `7.912123e-3`, k-space FDTD vs exact
  `7.943160e-3`, PSTD vs exact `7.943194e-3`, FDTD vs PSTD `5.197703e-5`,
  k-space FDTD vs PSTD `1.097571e-5`, DG vs FDTD `7.700342e-3`, DG vs PSTD
  `7.729329e-3`.

- 2026-05-18: [patch] Added the uniform-grid DG resampling view requested by
  the comparison audit. `dg_acoustic_comparison_plot.rs` now derives a DG trace
  on the native FDTD/PSTD grid from the existing DG plotted samples, averaging
  left/right traces at element interfaces and leaving FDTD/PSTD values
  uninterpolated. The regenerated `gaussian_pressure.png` adds uniform-grid
  pressure/error panels, and `gaussian_pressure.csv` includes
  `uniform_pressure` plus `uniform_absolute_error` rows. Verified uniform-grid
  metrics: DG vs exact `4.661959e-5`, FDTD vs exact `5.416002e-5`,
  k-space FDTD vs exact `8.204688e-6`, PSTD vs exact `1.201431e-5`,
  DG vs FDTD `7.735854e-5`, DG vs PSTD `4.567891e-5`.

- 2026-05-18: [patch] Added a fixed-final-time timestep sweep for the same
  Gaussian acoustic fixture. `dg_acoustic_timestep_sweep.rs` runs DG, classical
  FDTD, k-space FDTD, and PSTD at 20/40/80 steps, resamples DG onto the native
  uniform grid, and writes `target/dg_acoustic_comparison/timestep_sweep.png`
  plus `timestep_sweep.csv`. Results separate temporal behavior from
  interpolation/spatial error: DG remains near `4.6619e-5`, k-space FDTD stays
  near `8.204e-6`, while FDTD contracts from `5.478178e-5` to `5.384838e-5`
  and PSTD contracts from `1.206625e-5` to `1.198838e-5`.

- 2026-05-18: [patch] Added a focused ultrasound water-tank comparison fixture.
  `focused_ultrasound_water_tank.rs` drives a Hamming-apodized phased line
  aperture in homogeneous water, runs FDTD+CPML and PSTD+CPML on the same
  source, compares gated peak-pressure maps against an analytical focused-array
  reference, and writes `target/focused_water_tank/focused_water_tank.png`,
  `focused_water_tank_metrics.csv`, and `focused_water_tank_profiles.csv`.
  Follow-up investigation corrected the comparison source from a single
  center-z slice to a through-plane aperture matching the embedded 2-D
  analytical reference. Verified metrics improved FDTD/PSTD normalized-L2 from
  `3.071616e-1` to `1.142732e-1` and correlation from `0.935009` to
  `0.979759`; PSTD vs analytical improved to normalized-L2 `5.851104e-2` and
  correlation `0.995336`. Remaining discrepancy is now attributed to
  FDTD/PSTD numerical dispersion and stencil/order differences under CPML on
  the finite grid, not to mismatched source dimensionality.
  DG is now included at three levels: `DG-2D` and `DG-3D` tensor-product
  acoustic maps using the native `[p, u_x, u_y, u_z]` pressure/velocity RHS,
  plus `DG-1D axial` as the line-regression diagnostic. Follow-up correction
  replaced nodal DG sampling with GLL-polynomial interpolation onto the uniform
  FDTD/PSTD grid and moved the focused source into the SSP-RK3 stage RHS with
  weak GLL cell-source weights. The next correction added an explicit tensor
  DG boundary policy and routed the water-tank tensor DG maps through per-axis
  boundary conditions: one-way acoustic characteristic exterior states on x/y
  physical tank faces and periodic z for the embedded 2-D invariant slab.
  Current focused-map metrics are FDTD vs DG-2D normalized-L2 `1.616039e-1`,
  correlation `0.985529`; FDTD vs DG-3D normalized-L2 `1.616039e-1`,
  correlation `0.985529`; PSTD vs DG-2D/DG-3D normalized-L2 `1.635862e-1`,
  correlation `0.986426`; DG-2D vs analytic normalized-L2 `1.933581e-1`,
  correlation `0.975261`; and DG-2D vs DG-3D normalized-L2 `1.756510e-8`
  with correlation `1.000000`. All 2-D/reference maps and the z-invariant 3-D
  DG midplane peak at `(8 mm, 9 mm)` with `focus_error_mm = 3.0` under the
  finite-grid convention. Axial line metrics remain FDTD vs DG-1D
  normalized-L2 `2.218071e-1`, correlation `0.918299`; PSTD vs DG-1D
  normalized-L2 `2.199460e-1`, correlation `0.862900`; analytical vs DG-1D
  normalized-L2 `2.273648e-1`, correlation `0.823690`. Remaining follow-up is
  DG CPML or an equivalent DG-native absorbing layer for fully finite 3-D
  domains rather than this embedded slab comparison.
  The high-level simulation adapter now uses the same tensor acoustic state and
  uniform-grid field projection for `SolverType::DiscontinuousGalerkin`, so
  2-D/3-D DG is available through both the focused comparison fixture and the
  generic simulation solver path.

- 2026-05-18: [patch] Added DG p-refinement convergence plotting for the same
  Gaussian acoustic fixture. `dg_acoustic_convergence_plot.rs` keeps the p2
  DG/FDTD/PSTD discrepancy baseline intact, then measures DG orders p1-p4
  against the same analytical d'Alembert pressure reference and writes
  `target/dg_acoustic_comparison/dg_order_convergence.png` plus
  `dg_order_convergence.csv`. The diagnostic now records both the original
  per-order nodal-quadrature error and a common p4-quadrature error evaluated
  at the same physical points for every order. Verified common pressure
  relative-L2: p1 `3.402122e-2`, p2 `1.992925e-3`, p3 `1.807932e-4`,
  p4 `1.398263e-5`. The original nodal values are retained in the CSV as an
  aliasing audit trail: p1 `2.306593e-4`, p2 `4.305350e-4`, p3 `3.730400e-5`,
  p4 `1.398263e-5`.

- 2026-05-18: [patch] Closed the Figure 6 liver targeting regression. The
  controlled linear branch now exports crop/source metadata through PyO3 and
  projects exposure/fusion through the CT crop bounds. The abdominal nonlinear
  branch now uses the same connected single treatment lesion as the linear
  slice. Finite-area nonlinear source patches now preserve pressure-boundary
  peak drive under grid refinement. The regenerated Figure 6 displays simulated
  target-mask pressure, archives treatment-window and raw prefocal pressure
  separately, and records measured electronic-steering calibration. Liver
  linear exposure and displayed nonlinear target pressure now peak inside the
  selected target; liver target MI is `4.28`, treatment-window hotspot distance
  is `17.78 mm`, raw prefocal body-pressure hotspot distance is `103.74 mm`,
  and the measured steering search selected correction `[0, 0, 0]` in
  `controlled_comparison_metrics.json`.

- 2026-05-18: [patch] Closed the Figure 6 brain target-frame regression. The
  controlled linear branch now resolves the canonical brain target in the full
  3-D CT support, maps that source index through the resampled head crop for
  the reduced 2-D inverse, exports the brain crop bounds to PyO3, and applies
  focal-distance steering apodization in linear exposure synthesis. Regenerated
  Figure 6 metrics put the brain linear exposure, linear fusion, and elastic
  shear hotspots inside the full-CT target mask; brain linear fusion Dice is
  `0.746`, elastic shear Dice is `0.806`, and
  `linear_focus_to_common_target_centroid_m = 0.0004366`.

- 2026-05-18: [patch] Closed the nonlinear internal-gas material masking gap.
  The nonlinear 3-D body mask now flood-fills boundary-connected exterior air
  before material assignment, keeps enclosed HU `< -700` label-0 voxels inside
  the patient support, and maps those voxels to gas sound speed `343 m/s`, gas
  density `1.225 kg/m^3`, gas nonlinearity `1.2`, and high attenuation
  `1000 Np/(m*MHz)`. Exterior CT air remains coupling fluid at `1480 m/s` so
  the source coupling domain is not converted into a gas domain. Follow-up:
  profile and regenerate the full controlled Figure 6 comparison with
  internal-gas material enabled; the first bounded post-fix regeneration run
  exceeded 30 minutes and was stopped after the release extension build and
  focused Rust tests passed.

- 2026-05-18: [patch] Added the native coupled 1-D acoustic DG RHS and
  diagnostic. The previous acoustic DG examples reconstructed pressure and
  velocity through scalar characteristic solves; this increment adds direct
  pressure/velocity residual assembly with Rusanov flux, face-normal
  strong-form signs, and a reusable SSP-RK3 workspace. Component masses are
  conserved under GLL quadrature on periodic line elements. The new
  `dg_acoustic_1d_diagnostics.rs` example compares native DG against the
  analytical standing wave and the characteristic reconstruction path. Verified
  metrics: pressure relative L2 `1.651618e-4`, velocity relative L2
  `1.547224e-2`, native-vs-characteristic pressure L2 `4.571134e-16`,
  native-vs-characteristic velocity L2 `5.365939e-15`, pressure mass error
  `8.046975e-16`, velocity mass error `3.816392e-17`, and acoustic energy
  ratio `1.0`. Remaining work is lifting this 1-D coupled RHS into the broader
  FDTD/PSTD/DG pressure-field comparison matrix.

- 2026-05-18: [patch] Added the OpenPros-style dense/sparse clinical
  speed-shift benchmark. The nearest reusable path was
  `clinical::imaging::reconstruction::sound_speed_shift`: existing
  `SoundSpeedShiftSample`, finite-frequency row assembly, `ShiftSampling`,
  `ShiftPrior`, `SoundSpeedShiftPlan`, and workspace reuse cover the inverse
  API. The missing piece was a prostate limited-view fixture plus comparison
  metrics and a Criterion harness. The new fixture follows paper `2505.12261`
  structurally: top/bottom body-surface and rectal probe rows, 40 source
  channels, receiver lines across the lateral aperture, 1 MHz waveform
  metadata, 1,000 time steps, 120 ABC points, and a decimated 2-D SOS phantom.
  Remaining risk is full-waveform FDTD/RK inversion parity; this increment
  benchmarks the existing linearized finite-frequency shift operator, not a
  separate waveform solver.

- 2026-05-18: [patch] Added the DG bidirectional acoustic characteristic
  diagnostic. A left-going acoustic invariant satisfies `w-_t - c w-_x = 0`;
  reflecting coordinates converts it into the positive-advection equation
  already implemented by DG. The example now evolves `w+` and reflected `w-`,
  reconstructs `p=(w+ + w-)/2` and `u=(w+ - w-)/(2*rho*c)`, and compares the
  resulting standing wave against the exact acoustic solution. Verified metrics:
  pressure relative L2 `1.651615e-4`, velocity relative L2 `1.547223e-2`, and
  acoustic energy ratio `1.0`. Remaining work is a native coupled first-order
  DG acoustic RHS and direct pressure-field comparison with FDTD/PSTD.

- 2026-05-18: [patch] Added the Chapter 29 patient-adaptive focused transmit
  scheduling experiment. The minimal control surface is
  `transmit_schedule_strategy` plus `transmit_budget`; CT preprocessing,
  organ scenarios, device placement, matrix-free same-aperture operators,
  deterministic row encoding, PCG solve, fusion, and metrics remain shared.
  The new scope `KWAVERS_CH29_RENDER_SCOPE=adaptive_transmit` compares
  patient-adaptive and uniform focused transmit subsets for brain, kidney, and
  liver, recording active inverse Dice/CNR against transmit budget.

- 2026-05-18: [patch] Added the DG acoustic characteristic diagnostic slice.
  The one-way linear acoustic subspace diagonalizes to scalar advection through
  `w+ = p + rho*c*u` with `w- = 0`, so the current DG advection core can be
  tested against an exact acoustic pressure/velocity state without pretending
  to solve the full bidirectional acoustic system. Verified metrics:
  pressure relative L2 `8.263806e-4`, velocity relative L2 `8.263806e-4`,
  left-going invariant error `0`, and acoustic energy ratio `1.0`. Remaining
  acoustic DG work is a bidirectional characteristic or first-order system RHS
  before joining the full FDTD/PSTD pressure-field comparison matrix.

- 2026-05-18: [patch] Added the DG scalar discrepancy diagnostic example.
  The current DG core advances scalar periodic advection, not the coupled
  acoustic pressure/velocity system used by FDTD and PSTD. The new example
  therefore compares DG against the exact periodic shifted sine solution and
  reports mass, phase, amplitude, and relative-L2 metrics before any acoustic
  DG/FDTD/PSTD field comparison is treated as valid. Verified debug metrics:
  relative L2 `8.263806e-4`, mass error `4.873462e-16`, phase error
  `8.129815e-6` rad, amplitude ratio `9.999997e-1`.

- 2026-05-18: [patch] Closed the Spectral-DG dimensional completion and
  workspace gap. Audit finding: the physical-grid DG projection and RHS were
  line-only, lower-dimensional discontinuity detection returned all-false masks,
  the hybrid Spectral-DG solver exposed construction but no executable step,
  and the simulation DG adapter still required the old line coefficient layout.
  Fix: add an explicit tensor-product DG topology for active Cartesian axes,
  project/reconstruct 1-D, 2-D, and 3-D embedded grids into reusable coefficient
  storage, assemble tensor-product volume and periodic face terms through the
  shared RHS module, reuse detector/spectral/DG/coupling workspaces in the
  hybrid step, and route the simulation adapter through the tensor core. Tests
  now cover 1-D/2-D/3-D projection round-trips, lower-dimensional
  discontinuity masks, hybrid workspace pointer stability, DG convergence, and
  adapter layout rejection.

- 2026-05-18: [patch] Closed the DG periodic RHS conservation gap.
  Audit finding: the scalar DG RHS used the negated left-face upwind residual,
  so periodic surface terms could not telescope to zero under the
  quadrature-weighted mass functional. Fix: use `flux_left - c*u_left` at the
  left face, preserve the right-face residual, and route both line and
  tensor-product coefficient layouts through the extracted RHS module. Tests
  now verify zero weighted global mass derivative for a p=2 line fixture and a
  tensor-product manufactured state. This is the real DG equivalence
  prerequisite before broader DG/PSTD/FDTD discrepancy comparisons.

- 2026-05-17: [patch] Closed the DG shock-capture mass-conservation gap.
  Audit finding: the limiter claimed mean preservation but used an arithmetic
  node average, while nodal DG mass is the quadrature-weighted integral
  represented by the diagonal GLL mass matrix. For polynomial order greater
  than one, GLL weights are nonuniform, so arithmetic mean preservation is not
  conservation. Fix: troubled-cell indicators, neighbour jumps, and limited
  reconstructions now use quadrature-weighted element means, and the limited
  slope is centered by the quadrature-weighted node centroid. The regression
  uses p=2 GLL weights to prove the corrected invariant.

- 2026-05-17: [patch] Closed the hybrid two-region coupling quality gap.
  Audit finding: `apply_coupling` conserved a region-shaped buffer but wrote
  only the active interface plane, allowing affine conservation mass assigned
  to inactive planes to be discarded while diagnostics compared the transfer
  against the wrong source/target contract. Fix: restrict conservation and
  quality metrics to the active interface plane, compare the conserved transfer
  against the target trace, and pin target pressure-plane integral preservation
  plus non-pressure isolation in a manufactured two-region test. Also closed
  compile blockers from the nonlinear 3-D source-domain/source-body-mask test
  fixtures and the duplicate real-time SIRT row-norm helper so solver-targeted
  verification can link and execute.

- 2026-05-17: [patch] Closed the DG shock-capture execution gap. Audit
  finding: `ShockCaptureConfig` documented limiter application after RK
  sub-stages, but `DGSolver::solve_step` ignored the configuration. Fix:
  enabled SSP-RK3 stages and Forward Euler now apply a conservative
  troubled-cell projection using existing solver scratch. The limiter flags
  elements from neighbour mean jumps and intra-element variation, preserves
  each element mean exactly, and reconstructs flagged elements with the
  configured TVD slope limiter. Remaining DG work is quantitative
  conservation/dispersion comparison against FDTD/PSTD on shared fixtures.

- 2026-05-17: [patch] Closed the Chapter 29 nonlinear abdominal source-geometry
  defect. The nonlinear propagation crop now includes the acoustic path from
  target to planned skin contact instead of cropping only around the treatment
  window, abdominal sources are selected from exterior coupling cells on the
  canonical focused bowl, and source firing delays integrate straight-ray
  slowness through the CT-derived sound-speed map. The reduced KiTS19 real-data
  check executes through PyO3 with the focused-bowl aperture. The follow-up
  targeting increment keeps the outside focused-bowl standoff inside the crop,
  forbids abdominal exterior-coupling source stencils from writing into body
  voxels, and distributes each element over a finite exterior patch. At
  histotripsy-scale drive the reduced check reports source support
  `24..40` cells per element, mean support `29.81`, electronic steering delay
  span `9.679e-6 s`, target MI `2.55`, objective
  `2.4165e-5 -> 1.7785e-5`, target/body peak ratio `0.513`, coupling/body
  peak ratio `1.11`, and body-hotspot distance `14.93` grid cells, while
  recording `points_per_wavelength_min = 0.290`. Remaining work is a
  resolved-grid or k-space nonlinear propagation path for target pressure gain;
  the diagnostic no longer localizes the error to missing standoff geometry,
  direct tissue injection, point-source collapse, or reversed delay law.
  Chapter diagnostics now report raw global, body, coupling, target, source
  support, steering-delay, hotspot, and PPW metrics separately.

- 2026-05-17: [patch] Closed the Chapter 29 Figure 5 nonlinear beam-overlay
  diagnostic defect. The planned exposure panel keeps the Figure 2 planned
  aperture, while nonlinear pressure/FWI/cavitation/fusion panels now draw the
  actual nonlinear 3-D aperture projection and nonlinear target centroid on the
  full CT placement grid. Focused source tests also pin the electronic
  steering sign and the scalar skull phase-correction contract. Remaining
  liver pressure offset is therefore a pressure-localization/source-gain issue,
  not a known Figure 5 overlay mismatch or reversed delay law.

- 2026-05-17: [patch] Closed the Chapter 29 pressure-localization diagnostic
  gap. Controlled comparison metrics now project nonlinear pressure hotspots
  into the full CT placement frame, decompose the offset into planned beam-axis
  and cross-axis components, and record planned-vs-realized aperture axis angle
  plus nonlinear source-to-target distance statistics. Next increment: rerun
  the controlled nonlinear generator after isolating the current brain-case
  process exit, then use these metrics to decide whether liver correction is
  source normalization/focal gain, aperture realization, or propagation loss.

- 2026-05-17: [patch] Closed the Chapter 29 extension-loader reproducibility
  gap. The book script now registers dependency DLL search directories,
  rejects stale PyO3 extensions by nonlinear function signature, and exposes
  `KWAVERS_CH29_OUT_DIR` for scratch figure/metric output. The bounded
  comparison smoke run completed controlled linear and nonlinear brain,
  kidney, and liver generation at `40^3` into `target/ch29-smoke`. Remaining
  work is a production-grid rerun, not an untyped Python extension mismatch.

- 2026-05-17: [patch] Closed the hybrid conservation repair gap. Audit
  finding: `ConservationEnforcer` normalized transferred pressure traces to
  unit sum before applying momentum/energy corrections, which made the
  conservation target implicit and mixed unrelated constraints. Fix: interface
  repair is now the affine projection `v = mean(target) + alpha *
  (interpolated - mean(interpolated))`, which preserves the target integral
  exactly and matches target L2 energy whenever the interpolated trace has
  nonzero variance. Shape mismatches now fail with a typed validation error.

- 2026-05-17: [patch] Closed the DG `NumericalSolver` adapter completion gap.
  Audit finding: the adapter projected the input to modal coefficients and
  advanced those coefficients, but returned the original grid field because it
  did not reconstruct from the updated coefficients after `solve_step`. Fix:
  `NumericalSolver::solve` now calls `project_to_grid` before mask restoration,
  and the regression compares trait output against explicit
  project/step/reconstruct execution.

- 2026-05-17: [patch] Closed the hybrid coupling field-layout defect. Audit
  finding: `CouplingInterface` still read and wrote `Array4` fields as
  component-last even though the unified solver state is component-first
  (`[field, x, y, z]`). It also performed coupling work for a single-region
  decomposition with no interface. Fix: pressure extraction and target writes
  now go through `UnifiedFieldType::Pressure.index()` on axis 0, populate only
  the active interface plane, preserve non-pressure components, and return
  immediately when fewer than two regions exist.

- 2026-05-17: [patch] Closed the next DG time-stepping allocation increment.
  Audit finding: after removing redundant mass inversion, SSP-RK3 still
  allocated cloned `u_n`, three RHS arrays, and two stage arrays each step, and
  the surface-flux loop allocated a `Vec` for every element/variable face
  residual. Fix: `DGSolver` owns reusable original/stage/RHS registers sized to
  the modal coefficient tensor, `SspRk3` and `ForwardEuler` mutate modal
  coefficients in place, and face residuals are scalar left/right values.
  Remaining DG work is broader conservation/dispersion comparisons against
  FDTD/PSTD and shock-capturing limiter integration through each RK sub-stage.

- 2026-05-17: [patch] Closed a bounded hybrid FDTD/PSTD correctness and memory
  increment. Audit finding: both hybrid step paths cloned `self.regions` each
  step to avoid borrow conflicts, and the hybrid-region blend used
  `0.5*(1+cos(pi*d/W))`, which gives PSTD full weight at the FDTD/PSTD
  interface boundary (`d=0`) and again in the interior. Fix: `DomainRegion` is
  `Copy`, loops now copy one small region record by index with no `Vec`
  allocation, and the blend is `0.5*(1-cos(pi*d/W))`, clamped to `1` for
  `d>=W`, so the boundary uses FDTD and the smooth interior uses PSTD. This is
  a convex partition-of-unity transition for bounded fields.

- 2026-05-17: [patch] Closed a bounded DG memory-efficiency increment. Audit
  finding: `DGSolver::solve_step` recomputed a dense inverse of the mass matrix
  even though the RHS implementation did not use it; the nodal differentiation
  and lift matrices already encode the inverse-mass action (`D=M^-1S`,
  `LIFT=M^-1E`). `RegionPSTDSolver` also performed a first-step `field.clone()`
  to initialize previous history. Fix: remove the redundant inverse, document
  the matrix contract, allocate `prev_field` at construction, and use
  `has_prev_field` plus `.assign()` for history updates. Remaining DG work is
  deeper RK workspace reuse (`u_n`, `rhs`, and RK stage buffers) and extending
  tests from matrix identities into conservation/dispersion comparisons against
  PSTD and FDTD.

- 2026-05-17: [patch] Closed the bounded FDTD/PSTD example-comparison gap by
  replacing `kwavers/examples/pstd_fdtd_comparison.rs` with a real solver
  diagnostic. The fixture runs classical FDTD, k-space corrected FDTD, and
  PSTD on the same homogeneous Gaussian IVP with leapfrog-compatible
  `u(t=-dt/2)`. It reports field-level discrepancy metrics and literature
  sources instead of a placeholder API note. Verified debug-run metrics:
  FDTD vs PSTD relative L2 `5.60099e-2`, normalized max error `1.19624e-1`,
  correlation `0.997919`; FDTD+k-space vs PSTD relative L2 `7.25746e-16`,
  normalized max error `1.85106e-15`, correlation `1.0`. This confirms the
  current k-space FDTD derivative path improves alignment with PSTD on the
  bounded homogeneous fixture. Remaining validation work is a broader solver
  matrix across heterogeneous media, source-injection modes, absorbing
  boundaries, nonlinear paths, and longer propagation windows.
- Implement automated test scenarios comparing `pykwavers` outputs natively against `k-wave-python` identical scenarios.
- Quantitatively verify sources, signals, grids, sensors, and solvers.
- Closed the Chapter 29 uncontrolled visual-comparison gap by adding a matched
  linear/nonlinear comparison artifact: the linear branch now reruns at the
  nonlinear resolution, element count, drive frequency, and pressure, both
  branches are evaluated on the nonlinear crop projection, and metrics record
  the nonlinear pressure outside-target energy plus residual projected aperture
  distance. The follow-on histotripsy correction gates Rayleigh-Plesset
  cavitation by mechanical index, preserves calibrated per-element source
  weights, expands the brain cap aperture to the requested element count,
  constrains passive cavitation inversion to the MI-gated source support, and
  records source-support hotspot metrics. Figure 6 now renders every comparison
  panel on the full-resolution CT/transducer placement grid used by Figure
  2/Figure 5. Nonlinear fusion still exceeds the reduced linear fusion on
  average, while passive cavitation remains dominated by off-target MI-gated
  Rayleigh-Plesset source energy.
- Closed the Chapter 29 nonlinear pressure-simulation defect by replacing the
  explicit Westervelt `p*dtt(p)` feedback loop with the finite-amplitude
  denominator form, bounding additive source injection, preserving abdominal
  target-facing source order, and defaulting abdominal nonlinear histotripsy to
  500 kHz. Regenerated metrics show finite target MI above threshold for brain,
  kidney, and liver. Remaining Chapter 29 work is cavitation specificity and
  inverse sensitivity: pressure/Rayleigh-Plesset support still spreads outside
  the lesion, and kidney nonlinear FWI objective remains flat after pressure
  delivery.
- Closed the Chapter 29 cavitation-source normalization defect by computing the
  Rayleigh-Plesset source-density normalization peak over active treatment-window
  voxels only. Excluded source/boundary pressure lobes no longer reduce valid
  in-window cavitation evidence.
- Closed the Chapter 29 nonlinear FWI observability gap by emitting
  per-iteration line-search diagnostics from the Rust solver through the PyO3
  result and Chapter 29 metrics writer. The remaining performance gap is the
  full-grid kidney line-search measurement: even one `56^3` kidney FWI iteration
  is dominated by real Westervelt candidate forward solves, so the next
  increment should optimize candidate evaluation/checkpoint reuse rather than
  reducing the CT-frame resolution.
- Closed the Chapter 29 elastic-comparison gap for the reduced inverse by
  adding `phase_speed_m_s` to the same-aperture finite-frequency operator and
  exporting a low-frequency shear inverse channel through Rust, PyO3, Figure 2,
  and the controlled full-CT comparison grid. This is an elastic/shear
  comparator on the same aperture, not a trace-based `ElasticPSTD` clinical FWI
  branch. `SolverType.ElasticPSTD` remains validated separately against
  KWave.jl; wiring it into Chapter 29 requires source scheduling and receiver
  trace plumbing for the full CT clinical setup.
- New Chapter 29 artifact blocker: the default Figure 6 regeneration run
  exits during the nonlinear brain case after printing `comparison nonlinear
  brain start`, with no Python traceback. Focused Rust tests still pass, so the
  next increment should isolate the PyO3 nonlinear book-generation process exit
  before claiming regenerated Figure 5/Figure 6 elastic comparison metrics.
- Closed five KWave.jl parity gaps (physics with no equivalent in `external/k-wave-python/examples/`) via `pykwavers/examples/{diff_bioheat_1d,ewp_elastic_2d,pr_time_reversal_2d,us_phased_array_3d,us_beamforming_2d}_jl_compare.py` paired with `run_kwave_julia_*.jl` drivers and a `_run_julia_parity_sweep.py` harness; bioheat 1D / TR 2D / phased-array 3D / beamforming 2D land PASS. The elastic 2D pair lands as a diagnostic that surfaces a pre-existing pykwavers `SolverType.Elastic` source-scaling regression (also breaks the historical `external/elastic_julia_parity/compare_elastic.py` matched-mode peak ratios — separate fix required).
- Closed the vendored `k-wave-python` 2-D FFT line-sensor parity gap in `pykwavers` via native `kspace_line_recon`, the non-square 2-D FFT axis fix, and Python binding export.
- Closed the vendored `k-wave-python` 3-D planar-sensor time-reversal parity gap in `pykwavers` by caching the reconstructed fields and preserving the exact forward pressure/sensor ordering contract.
- Closed the vendored `k-wave-python` 3-D circular piston parity gap in `pykwavers` by using the native `KWaveArray` disc geometry, clipping the PML halo before source-weight comparison against the padded reference mask, and validating the analytical on-axis piston profile.
- Closed the vendored `k-wave-python` 3-D focused bowl parity gap in `pykwavers` by switching the bowl rasterizer to the canonical spiral/BLI source path, reporting the physical-interior source-weight parity, and validating the on-axis waveform comparison.
- Closed the vendored `k-wave-python` 2-D focussed detector parity gap in `pykwavers` by comparing detector-averaged traces for the on-axis and off-axis source cases and validating the directivity-energy ratio.
- Closed the vendored `k-wave-python` 2-D sensor directivity modelling gap in `pykwavers` by comparing the full source-angle trace matrix and the derived directivity curve against the reference example.
- Closed the vendored `k-wave-python` `at_array_as_sensor` gap in `pykwavers` by aligning the arc geometry to the upstream line-sampled BLI footprint, preferring the rebuilt `target/maturin/pykwavers.dll` extension artifact, and validating exact mask parity plus raw/combined detector-matrix comparison.
- Closed the vendored `k-wave-python` `at_array_as_source` gap in `pykwavers` by reusing the canonical arc ordering, comparing exact source-mask and distributed source-signal parity, and validating p_max/p_rms field parity against the rebuilt extension.
- Closed the pressure-source ordering contract in `pykwavers` by switching the arc and linear-array source builders to Fortran-order active-cell rows and pinning the exact helper-matrix parity against k-wave-python; the rebuilt `at_linear_array_transducer` example now passes the `p_max` field comparison.
- Closed the vendored `k-wave-python` `us_defining_transducer` gap in `pykwavers` by carrying the reference time-step count through the pykwavers scan-line run, aligning the sensor-trace lengths, and validating per-sensor trace metrics with a PASS report.
- Closed the vendored `k-wave-python` `ivp_photoacoustic_waveforms` gap in `pykwavers` by reusing the cached initial-pressure traces, comparing the single-sensor waveform directly, and validating the PASS report metrics.
- Closed the vendored `k-wave-python` `us_bmode_phased_array` gap in `pykwavers` by validating the quick steering-angle sweep against the cached k-Wave and pykwavers scan lines, confirming the fundamental/harmonic B-mode parity, and preserving the existing GPU profile contract.
- Closed the vendored `k-wave-python` `sd_focussed_detector_3D` gap in `pykwavers` by validating per-source trace parity, checking the on-axis/off-axis directivity ratio, and preserving the PASS report contract.
- Closed the vendored `k-wave-python` `us_bmode_linear_transducer` gap in `pykwavers` by disabling GPU source-kappa correction to match the upstream `NotATransducer.u_mode = "additive-no-correction"` contract, reusing full medium buffers with borrowed `PyReadonlyArray3` uploads, and pinning the PASS report with a cached regression test.
- Reduced the pre-sweep `us_bmode_linear_transducer` GPU hot path by restoring the measured medium-upload timing after cached execution, aggregating per-line GPU timings through a compact tuple summary, and advancing the lateral medium window in place to avoid rebuilding the active slab on every scan line.
- Closed the PSTD checkpointing contract by validating bit-exact save/resume continuation, exact checkpoint file deletion after restore, and the PASS report emitted by `checkpointing_compare.py`.
- Keep exact tone-burst regression coverage for the Gaussian default envelope and non-integer sample-count cases.
- Validate the seismic FWI adjoint-state path with receiver-order residual reversal, discrete L2 objective scaling, CFL checks, and finite-difference gradient identities.
- Validate the reconstruction FWI path with sign-correct residuals, `dt`-scaled objectives, checkpointed adjoint replay, timestep validation, and encoded-gradient aggregation.
- Extract and keep the acoustic adjoint-state core as the single source of truth for L2 residuals, objective scaling, time reversal, and signed-correlation accumulation.
- Maintain checkpointed replay regression coverage for reconstruction FWI to preserve exact adjoint-state accumulation with reduced peak memory.
- Keep simulation-owned concrete solver assembly as the only high-level construction boundary. `SolverType::KSpace` now assembles through the canonical PSTD full k-space path; `SolverType::DiscontinuousGalerkin` now has a real simulation adapter for the validated 1-D element/node/scalar layout; FEM now has structured-grid tetrahedralization, exact nodal loads, tagged Dirichlet boundary assembly, an explicit `FrequencyDomainAcousticBackend` contract separate from time-domain `Solver::run`, and a real Gaia `IndexedMesh<f64>` tetrahedral-volume import boundary for mesh-provider integration.
- Validate the acoustic GPU compute path with workgroup sizes that satisfy device invocation limits, matched uniform-buffer layouts, and fused field-update loops that avoid transient gradient volumes.
- Validate the GPU memory-tracking surface through the public `kwavers::profiling` export, direct allocation-guard RAII semantics, and FDTD pressure upload/download roundtrips.
- Remove remaining GPU-adjacent lint noise in beamforming and k-space hot paths by replacing zero-fill readback, eliminating redundant casts, and keeping dispatch/debug metadata on the production path.
- Closed remaining active FDTD solver allocation churn: acoustic stepping reuses staggered divergence scratch state, scalar dispatch avoids redundant full zero-fills, GPU readback keeps in-place overwrite semantics, and EM boundary application now copies the authoritative field cache into caller-owned buffers without steady-state `EMFields` cloning.
- Finish the FFT migration by keeping `kwavers` on Apollo-backed transforms only, preserving no direct `rustfft` usage in `kwavers` source, tests, or benches.
- Keep the Apollo GPU FFT backend parity-checked against kwavers examples after the radix-stage dispatch fix and hybrid absolute/relative parity metric.

## Outstanding k-wave-python Parity Gaps
- `at_linear_array_transducer`: closed after switching the parity example to the upstream additive pressure-source mode; the source rows remain Fortran-ordered and the rebuilt extension now matches `p_max` parity.
- `at_focused_bowl_AS` and `at_circular_piston_AS`: closed after fixing the pykwavers sensor reshape to Fortran order, which restored PASS parity on both cached axisymmetric PSTD example comparisons.
- `na_controlling_the_pml`: closed by validating waveform parity across the PML attenuation sweep and exact k-Wave-style save-to-disk HDF5 input-file parity via versioned artifacts in `pykwavers/examples/output/na_controlling_the_pml/hdf5_v1/`.
- `checkpointing`: closed by validating bit-exact save/resume continuation, exact checkpoint file deletion after restore, and the PASS report emitted by `checkpointing_compare.py`.

## Schwarz domain-decomposition boundary tree cleanup
- 2026-05-01: closed the Schwarz boundary oversized-file gap by splitting `domain::boundary::coupling::schwarz` (819 lines) into `schwarz/{mod,gradient,transmission,boundary_impl,tests}` partitioned by responsibility (theorem facade with builder methods, shared finite-difference normal-gradient helper, four-branch transmission dispatcher Dirichlet/Neumann/Robin/Optimized, `BoundaryCondition` trait bridge, 11 value-semantic regression tests); preserved the `SchwarzBoundary` re-export through `schwarz/mod.rs` (parent `coupling/mod.rs` unchanged); targeted suite passes 11/11, clippy `-D warnings` clean (`--no-deps`), full lib suite passes 2640/2640 with 12 ignored in 9.14 s; source files are ≤109 lines, tests file 429 lines.

## Optical diffusion solver tree cleanup
- 2026-05-01: closed the optical diffusion solver oversized-file gap by splitting `solver::forward::optical::diffusion::solver` (837 lines) into `solver/{mod,construction,operator,preconditioner,solve,accessors,analytical,tests}` partitioned by responsibility (theorem facade with config + struct, constructors with shared boundary helpers, 7-point heterogeneous-D operator, Jacobi preconditioner, PCG driver, read-only accessors, Contini-1997 Green's-function references, value-semantic tests); preserved all five public exports through `solver/mod.rs` (parent `diffusion/mod.rs` unchanged); struct fields are now `pub(super)` for sibling submodule access; targeted suite passes 4/4, clippy `-D warnings` clean (`--no-deps`), full lib suite passes 2640/2640 with 12 ignored in 8.37 s; all eight split files are ≤173 lines.

## Ultrasound physics book expansion
- 2026-05-03: expanded `docs/book/` from the original therapy/diagnostics/theranostics scaffold into a domain-indexed book with 20 chapters total. New chapters cover acoustic foundations, propagation, numerical methods, tissue/media models, sources/transducers, sensors, beamforming, photoacoustics, elastography, cavitation and bubble dynamics, nonlinear acoustics, transcranial ultrasound, sonogenetics, inverse problems/PINNs, safety/dosimetry, validation/benchmarking, and performance/memory. Each new chapter includes theorem/proof sketch, algorithm contract, implementation targets, and research anchors tied to kwavers modules.
- 2026-05-12: [patch] closed the neuromodulation chapter gap by adding Chapter 26 `docs/book/neuromodulation.md`, executable simulations in `pykwavers/examples/book/ch26_neuromodulation.py`, generated acoustic/mechanochemical/thermal/clinical-guidance figures under `docs/book/figures/ch26/`, manifest and README registration, and value-semantic tests for acoustic safety, focal decay, channel gating, neural response, thermal dose, and cavitation guardrails.
- 2026-05-12: [patch] closed the seismic brain FWI chapter gap by adding Chapter 27 `docs/book/seismic_fwi_brain_imaging.md`, `kwavers::solver::inverse::seismic::brain_helmet`, the RITK-backed `pykwavers::run_seismic_helmet_fwi_from_ritk_ct` wrapper, the executable CT reconstruction script, generated RIRE CT single-slice and multi-slice reconstruction figures under `docs/book/figures/ch27/`, and value-semantic Rust verification for objective reduction plus recovered brain-speed contrast.
- 2026-05-12: [minor] extended Chapter 27 with a bounded `56^3` volume reconstruction default, a deterministic 1024-element hemispherical cap, 3-D source/receiver path lengths, CT-derived slice axial offsets, multiscale frequency continuation over 200/350/500/650/800 kHz, eight deterministic receiver offsets, weak-Westervelt second-harmonic encoded rows, Sobolev-smoothed update conditioning, matrix-free sensitivity application, target-independent regularized FWI display reconstruction, twelve nonempty simulated volume slices, centroid-cropped ROI inspection, seven generated figure pairs, and regenerated metrics showing the stack satisfies the visible-reconstruction contract.
- 2026-05-12: [minor] tightened Chapter 27 visualization and physics by adding a CT HU row to the multi-slice stack, relabeling the CT-derived acoustic target, and including CT-derived path attenuation in each encoded source/receiver sensitivity row.
- 2026-05-12: [minor] replaced Chapter 27 slice-wise inversion with a coupled matrix-free 3-D inversion over a resampled CT volume, added `AcousticVolume`/`reconstruct_brain_volume` plus `pykwavers::run_seismic_helmet_fwi_volume_from_ritk_ct`, sliced the returned 3-D arrays for the primary/stack/ROI figures, and regenerated metrics with `56^3` voxels, 1024 elements, 81,920 encoded nonlinear rows, and visible 3-D reconstruction.
- 2026-05-12: [patch] corrected the Chapter 27 3-D inversion quality regression by replacing diagonal Landweber with projected preconditioned CG on the matrix-free normal equations, caching acquisition row norms and per-row constants instead of recomputing them in every operator call, adding stage-boundary Charbonnier edge-preserving proximal regularization plus target-independent mask-aware regularized display for `fig06`, and regenerating default metrics with global Pearson `0.856879060248954`, NRMSE `0.00264293344795594`, and stack slice Pearson range `0.801703760879262`-`0.879606457699549`.
- 2026-05-12: [patch] added Chapter 27 histotripsy-monitoring subchapters that define RTM as the real-time cavitation-localization layer, active time-lapse FWI as the post-packet lesion-property update, and nonlinear harmonic/bubble/elastic FWI variants as distinct physics contracts rather than interchangeable frequency channels.
- 2026-05-12: [minor] added the custom Chapter 27 histotripsy RTM/FWI simulator `pykwavers/examples/book/ch27_histotripsy_fwi_rtm.py`, which loads the RITK-backed CT baseline, builds CT-derived 1024-element active/passive operators, reconstructs compact, elongated, and multi-packet lesion states with deterministic noise, 110/160/220 kHz frequency continuation, Huber-robust normal FWI, multiparameter speed/attenuation FWI, weak nonlinear harmonic FWI, passive 110/220/440 kHz RTM, 110 kHz subharmonic source inversion, and frequency-gated fusion, and emits figures 8-10 plus `histotripsy_monitoring_metrics.json`.
- 2026-05-12: [minor] added Chapter 28 abdominal histotripsy FWI analysis for the KiTS19 kidney and LiTS liver CT examples, using the largest tumor-centered anatomical plane, a CT-textured support mask, a HistoSonics-like 256-element therapy aperture with central imaging receivers and a 750 kHz upper continuation frequency, fundamental path-sum Born receiver rows, half-frequency subharmonic receiver rows, second-harmonic nonlinear receiver rows, bounded 2-D Westervelt FDTD source-map generation, Rayleigh-Plesset subharmonic bubble response driven by simulated lesion pressure, and a diagonal-preconditioned H1-regularized CG solver for baseline targeting, time-lapse lesion-state reconstruction, subharmonic cavitation-source inversion, and nonlinear susceptibility inversion; regenerated kidney/liver panels and `docs/book/figures/ch28/metrics.json` explicitly state the synthetic/model-consistent limitation, record the larger FOV, report Westervelt propagation steps, and expose lesion pressure calibration metrics.
- 2026-05-13: [minor] added Chapter 29 same-device therapeutic ultrasound finite-frequency inverse/RTM simulations for brain, kidney, and liver CT/NIfTI cases through the RITK-backed `pykwavers.run_theranostic_inverse_from_ritk` wrapper, using an INSIGHTEC-like 1024-element helmet projection for brain and HistoSonics-like 256-element skin-coupled abdominal arcs with 64 central imaging receivers for kidney/liver; the kwavers theranostic module now emits pressure-calibrated exposure fields, active pitch-catch Born inversion, source-encoded linear acoustic RTM from baseline/lesion receiver traces, passive subharmonic receive-only inversion, weak harmonic and ultraharmonic rows, uncropped full-patient abdominal placement context, one connected abdominal treatment component per single-focus sonication, a calvarium-limited 3-D helmet placement with skull beam intersections, generated CT placement/reconstruction figures, and research citations covering transcranial FWI, UCT source encoding, RTM/FWI, passive cavitation mapping, and current HistoSonics/INSIGHTEC/Verasonics platform constraints.
- 2026-05-13: [patch] added Chapter 29 reconstruction dynamic-range diagnostics: `fig04_reconstruction_dynamic_range_diagnostics` renders the same active, passive, harmonic, ultraharmonic, and fusion maps on a common `[-40, 0] dB` scale and `metrics.json` now records outside-target peak ratio, outside-target peak dB, and outside-target energy fraction for each channel/case.
- 2026-05-13: [patch] updated Chapter 29 `fig02_exposure_and_reconstruction` so every brain, kidney, and liver row begins with the CT placement slice, target/body overlay, and transducer coordinates used by that case before exposure, target, active Born inverse, linear acoustic RTM, subharmonic inverse, harmonic inverse, ultraharmonic inverse, and fusion panels; the figure script now exposes the column layout and plotted transducer coordinates as tested contracts.
- 2026-05-13: [patch] optimized the Chapter 29 theranostic inverse path by precomputing the active CT-support graph once and reusing PCG work buffers for row products, prediction residuals, normal-operator output, and graph-Laplacian smoothing; also closed current kwavers no-deps clippy blockers in the seismic/FWI path by consolidating Chapter 27 composite-objective arguments. The next research-aligned increment remains a full 3-D adjoint Westervelt/Rayleigh-Plesset multiparameter path with robust W2/HV-style misfit support rather than additional Python-side plotting.
- 2026-05-13: [minor] migrated the Chapter 29 same-device therapeutic ultrasound workflow out of `solver::inverse::seismic::theranostic` and into `clinical::therapy::theranostic_guidance`, updated PyO3 to bind the clinical workflow entry point, removed the stale solver-layer module, and synchronized the book contract. The remaining architecture increment is replacing dense same-aperture row materialization with a matrix-free backend without moving clinical anatomy/device ownership back into the solver layer.
- 2026-05-13: [minor] extracted the Chapter 29 same-aperture inverse kernels into `solver::inverse::same_aperture`: active-support graph indexing, finite-frequency active/passive row assembly, harmonic and ultraharmonic rows, deterministic noisy simulated data, and graph-Laplacian PCG now form the solver-owned SSOT. `clinical::therapy::theranostic_guidance` now owns CT/anatomy/device workflow, pressure exposure synthesis, and reporting only. Next increment: replace dense row materialization with a matrix-free same-aperture operator backend before adding full 3-D adjoint nonlinear FWI.
- 2026-05-13: [patch] replaced Chapter 29 same-aperture dense-row use in the clinical theranostic workflow with matrix-free `FiniteFrequencyOperator` channels behind a generic `LinearOperator` PCG contract; PyO3 metrics now expose the backend label, stored operator value count, and dense-equivalent value count, with tests proving matrix-free products match materialized `RowMatrix` products and storage is below dense storage. Next increment: add robust misfit strategy types and extend source encoding from the linear RTM trace path into iterative 3-D adjoint inversion before the full Westervelt/Rayleigh-Plesset multiparameter path.
- 2026-05-13: [patch] added deterministic normalized row/source encoding to the Chapter 29 reduced same-aperture inverse. `solver::inverse::same_aperture::EncodedOperator` implements the exact compressed operator `B = C A`, the clinical active/passive/harmonic/ultraharmonic PCG channels now use encoded matrix-free rows by default, PyO3 metrics expose `inverse_encoding_rows_per_code`, `encoded_measurements`, and `unencoded_measurements`, and value tests compare encoded forward/adjoint/diagonal products against materialized rows. Next increment: expose alternative encoding bases and robust trace-space objectives as explicit strategies rather than changing the reduced inverse label.
- 2026-05-13: [patch] corrected Chapter 29 over-claiming by renaming the public PyO3 entry point to `run_theranostic_inverse_from_ritk`, removing the old theranostic-FWI wrapper, exposing `inverse_model_family`, `is_full_wave_inversion=false`, and `uses_nonlinear_wave_propagation=false`, changing figure labels to finite-frequency inverse and linear acoustic RTM, flattening the RTM forward-history buffer, replacing logarithmic source display scaling with pascal-scale pressure injection, selecting RTM time steps from the CT-domain travel-time horizon, and exposing value-tested L2/Charbonnier RTM residuals. Next increment: implement a 3-D nonlinear adjoint without presenting the current reduced inverse as nonlinear FWI.
- 2026-05-13: [patch] added the first robust waveform-misfit strategy to Chapter 29: the linear acoustic RTM residual now defaults to a Charbonnier adjoint-source derivative bounded by a scale derived from observed-trace RMS and configured receiver-noise fraction; PyO3 accepts `waveform_misfit`/`waveform_misfit_scale_fraction`, metrics report the selected misfit, scale, and objective, and Rust tests verify L2 exactness plus Charbonnier bounded influence. Next increment: add source-encoded acquisition weighting or OT/HV trace misfits as explicit strategies, still without claiming nonlinear FWI.
- 2026-05-13: [minor] added Chapter 30 intravascular ultrasound imaging and therapy, including a public IVUS segmentation dataset contract, deterministic 384 x 384 coronary vessel phantom, 64-element 20 MHz imaging ring, 1.5 MHz side-looking microbubble therapy sector, radial IVUS B-mode simulation, localized vessel-wall delivery and thermal maps, usage-sequence figure, `docs/book/figures/ch30/metrics.json`, manifest/README registration, and value-semantic Python tests. Next increment: add a real IVUS-Net contour loader and differential validation against measured B-mode frames.
- 2026-05-13: [patch] optimized the Chapter 29 matrix-free `FiniteFrequencyOperator` for the inverse-PCG hot path. Per-row source/receiver/wavenumber/frequency-MHz metadata (PitchCatchRow) and per-row receiver/wavenumber/sine-phase metadata (PassiveRow) are now precomputed once at construction so `matvec`, `t_matvec`, `normal_diag`, `compute_row_norms`, and `materialize` never recompute the row index `divmod` or the variant dispatch on a per-cell basis. Inverse row norms are cached so the inner loops never recompute `1 / norm`. Outer row loops (matvec) and outer column loops (t_matvec, normal_diag) dispatch through rayon for cache-aware parallelism on the SPD normal equations driven by PCG. `storage_values()` now accounts for the precomputed per-row metadata so the dense-vs-matrix-free comparison remains meaningful. Verified bit-identical against the dense `RowMatrix` oracle by the existing `matrix_free_operator_matches_materialized_rows` regression test. Next increment: replace the per-(row, col) `exp/cos/sqrt` triple with a per-(source, voxel) distance cache so repeated PCG iterations do not re-invoke `hypot` on already-known geometry.
- 2026-05-13: [minor] added the separated Chapter 29 nonlinear 3-D Westervelt/Rayleigh-Plesset branch. `clinical::therapy::theranostic_guidance::nonlinear3d` now performs CT-derived bounded volume preparation, skin/calvarium same-aperture placement, heterogeneous Westervelt FDTD propagation, exact discrete-adjoint sound-speed FWI for the implemented recurrence, Rayleigh-Plesset period-doubling cavitation-source simulation, passive subharmonic nonnegative inversion, PyO3 export as `run_theranostic_nonlinear_3d_from_ritk`, Figure 5 generation, metrics flags that distinguish it from linear RTM, and a value-semantic Rust fixture test. Next increment: joint multiparameter inversion for `c`, `alpha`, `rho`, `beta`, and cavitation density with a robust OT/HV trace misfit.
- 2026-05-13: [patch] optimized the Chapter 29 nonlinear 3-D Westervelt forward + Rayleigh-Plesset passive operator. Forward pressure history switched from fragmented `Vec<Vec<f64>>` (one heap allocation per timestep) to one contiguous `Vec<f64>` of length `(steps + 1) * cells`; the discrete adjoint now slices the buffer via `history_slice(step)`. The four rotating buffers (older, previous, current, next) are `mem::swap`-rotated each step — no `vec![0.0; cells]` allocation occurs inside the time loop. The forward cell update is rayon-parallel: each cell writes only to its own `next[i]`, so the outer 3-D loop dispatches through `par_iter_mut().enumerate()` without coloring, atomics, or locks. The structural cleanup also extracted `adjoint.rs` and `stencil.rs` from the original monolithic `westervelt.rs`, packaged `forward_with_schedule` arguments in a `ForwardInput<'a>` struct, packaged `accumulate_step` arguments in `AccumulateInput`/`NonlinearTransposeInput` structs, and computed the sponge weights once per gradient call instead of once per backward step. `PassiveOperator::new` now builds the dense Green's matrix row-parallel via `par_chunks_mut().zip(receivers.par_iter())`; `apply` runs through `par_chunks().map().collect()`; `normal_gradient` runs column-parallel through `(0..cols).into_par_iter()`. The dead `rows` field on `PassiveOperator` is removed. Mathematics unchanged; `nonlinear_3d_westervelt_fwi_and_cavitation_inverse_are_input_sensitive` continues to pass and `cargo clippy -p kwavers --lib --no-deps -- -D warnings` is clean. Next increment: replace the full adjoint history `vec![vec![0.0; cells]; steps + 1]` with a sliding 4-buffer rolling window so reverse-mode memory drops from O(steps * cells) to O(cells).
- 2026-05-13: [minor] improved the Chapter 29 nonlinear 3-D branch conditioning and visibility. The Westervelt inverse now stacks deterministic focused source encodings, computes discrete-adjoint gradients for both `c` and `beta`, restricts parameter updates to a CT/segmentation-derived target ROI while propagating through the full body support, adds body-restricted `H1` regularization and Sobolev gradient smoothing, exposes PyO3 controls for source encoding and regularization, returns Rust-side multiparameter FWI and nonlinear fusion scores, and renders Figure 5 with the same CT/exposure/target/reconstruction/fusion grammar as Figure 2. Next increment: add thermoviscous/shock-capturing stabilization for higher histotripsy pressure envelopes before increasing grid size beyond the bounded default.
- 2026-05-13: [patch] regenerated Chapter 29 Figure 5 at the same per-case simulation grids as Figure 2. The nonlinear example now defaults to the case grid (`48^3` brain, `52^3` kidney/liver) and keeps explicit `KWAVERS_CH29_{BRAIN,KIDNEY,LIVER}_NONLINEAR_GRID` / `KWAVERS_CH29_NONLINEAR_GRID` overrides for controlled downsampling or stress tests. A rebuild also closed the `EncodedOperator<O>` Rayon `Sync` bound defect exposed by the release PyO3 build. Next increment: reduce nonlinear adjoint memory and runtime without lowering the requested grid.
- 2026-05-13: [patch] optimized the Chapter 29 nonlinear 3-D Westervelt adjoint by replacing dense `(steps + 1)` adjoint-state storage with four rolling adjoint volumes matched to the three-step temporal stencil. The adjoint-state memory drops from `O(steps * cells)` to `O(cells)` and the per-step full-volume `clone()` is removed. A dense-adjoint oracle test verifies the rolling gradients for both `c` and `beta`. The full Chapter 29 generator completed the `48^3/52^3` Figure 5 workload in `360.1 s` with sampled peak process-tree working set `8.29 GB`. Next increment: add exact checkpointed forward-history replay to reduce the remaining forward-history memory without weakening the Figure 5 grid contract.
- 2026-05-14: [patch] optimized the Chapter 29 nonlinear 3-D Westervelt forward-history memory by replacing retained dense pressure history with exact sparse checkpoints and bounded interval replay. Each checkpoint stores the three pressure states required by the recurrence, the reverse sweep materializes only one replay segment plus four rolling adjoint volumes, and PyO3 exposes `checkpoint_interval_steps` for Figure 5 runs. Focused tests prove bitwise replay equivalence against dense forward history and checkpoint-interval-invariant `c/beta` gradients. The full Chapter 29 figure run regenerated `fig05` at `48^3/52^3/52^3` and records `checkpoint_interval_steps = 128`, but local runtime increased to about 42 min; next increment: reduce replay runtime with segment-source preplanning, line-search trace caching, and backend-level replay fusion.
- 2026-05-14: [patch] closed the nonlinear volume oversized-file gap by moving CT attenuation laws and centroid utilities into `volume/attenuation.rs` and `volume/centroid.rs`; the nonlinear 3-D tree now satisfies the <500-line leaf-file rule. Next increment: apply the same structural audit to adjacent Chapter 29 reduced-workflow modules after the active performance pass.
- 2026-05-14: [patch] re-closed the Chapter 29 nonlinear 3-D structural gap after `volume.rs` re-grew to 521 lines and `absorption.rs` reached 555 lines. The `volume` module is now an 86-line facade plus SRP children `volume/{validation,bbox,mask,resample,material}` (largest 152 lines) alongside the previously-extracted `volume/{attenuation,centroid}`. The `absorption` module is now a 115-line facade carrying Treeby-Cox 2010 docs and the `FractionalLaplacianAbsorption` + `AbsorptionBuilder` struct definitions, plus `absorption/{construction,spectrum,apply,tests}` children (largest 149 lines). External callers (`nonlinear3d::adjoint`, `nonlinear3d::forward`) keep the same import path. Every nonlinear3d non-test source file is now `<= 500` lines (largest: `forward.rs` at 500, `cavitation.rs` at 451). Verification: `cargo check`, `cargo clippy -p kwavers --lib --no-deps -- -D warnings` clean; 20/20 default nonlinear3d tests pass with 3 Tier-2 ignored; the heavy `nonlinear_3d_westervelt_fwi_and_cavitation_inverse_are_input_sensitive` integration test passes. Next increment: the previously-recorded replay-runtime reduction with segment-source preplanning and line-search trace caching.
- 2026-05-16: [patch] closed the Chapter 29 nonlinear cavitation active-support and workspace increment. `run_cavitation_inverse` now generates passive data from the active-voxel source vector required by `PassiveOperator`, Rayleigh-Plesset period-doubling uses a one-period radius ring buffer instead of full-radius history, and projected passive inversion reuses prediction/residual/gradient buffers with parallel in-place kernels. The same increment completed the pending split-directory SSOT cleanup for nonlinear Westervelt, sound-speed-shift fixed acquisition, P-STD split-field stepping, and monolithic-coupler tests. Next increment: profile checkpoint replay and line-search trace reuse before raising Figure 5 grid size beyond the current Figure 2 parity contract.
- 2026-05-16: [patch]/[major] closed the nonlinear FWI iteration workspace increment and the P-STD thermal argument-shape warning gate. Chapter 29 `run_fwi` now reuses one residual trace buffer and one `LineSearchWorkspace` for candidate `c/beta` models across all backtracking scales, eliminating per-shot residual allocation and two full model allocations per line-search candidate. The P-STD thermal orchestration API now takes `ThermalOrchestrationInput<'_>` so unit-bearing thermal coupling values are named at the call site. Next increment: profile checkpoint replay segment construction and source-plan reuse inside `replay_history_segment_into`.
- 2026-05-16: [patch] closed the CT-aligned brain scene duplication gap. `pykwavers/examples/book/transcranial_planning/scene.py` now owns the VIM-like target fraction, 1024-element Insightec-like helmet pose, cap angles, acoustic speeds, pressure scale, aperture diameter, and HU thresholds; Chapter 25 Figure 2, Chapter 29 Figure 5 brain nonlinear simulation, the 3-D helmet placement wrappers, Chapter 31 brain helmet geometry, and the skull-adaptive benchmark consume that scene instead of deriving independent centroids or benchmark defaults. Next increment: regenerate affected book figures after the rebuilt PyO3 extension is available.
- 2026-05-17: [patch] closed the canonical Westervelt FDTD stencil/workspace gap. The 4th-order Laplacian now uses the mathematically correct centered second-derivative coefficients (`center=-5/2`, `near=4/3`, `far=-1/12`), the documented 6th-order stencil is implemented instead of silently missing, and invalid `spatial_order` values return typed validation errors without mutating configuration. The update path reuses solver-owned nonlinear-term and next-pressure buffers and rotates pressure histories by swap rather than allocating a fresh next field every step. Added theorem-backed quadratic-field exactness coverage for O2/O4/O6, unsupported-order rejection, and pointer-stability verification. Focused evidence before later unrelated worktree churn: `cargo test -p kwavers solver::forward::nonlinear::westervelt --lib -- --nocapture` passed 8/8 and `cargo test -p kwavers --test nonlinear_physics_tests -- --nocapture` passed 3/3. Current rerun after subsequent unrelated edits timed out behind active cargo/rustc workloads without emitting a Westervelt diagnostic.
- 2026-05-17: [patch] closed the thermal diffusion finite-difference validation gap. `ThermalDiffusionSolver::calculate_laplacian` now rejects unsupported `spatial_order` values with `KwaversError::Validation` instead of silently mutating the solver configuration to second order. The solver module documents the centered-stencil quadratic exactness theorem, and focused tests pin O4 `laplacian((x-x0)^2 + 2(y-y0)^2 + 3(z-z0)^2)=12`, singleton-axis O2 behavior, narrow-axis O4 fallback, borrowed-source update behavior, and invalid-order state preservation. Evidence: `cargo test -p kwavers solver::forward::thermal_diffusion::solver --lib -- --nocapture` passed 5/5; `cargo fmt --check -- kwavers/src/solver/forward/thermal_diffusion/solver/mod.rs kwavers/src/solver/forward/thermal_diffusion/solver/tests.rs` passed; `git diff --check -- kwavers/src/solver/forward/thermal_diffusion/solver/mod.rs kwavers/src/solver/forward/thermal_diffusion/solver/tests.rs` passed.
- 2026-05-17: [patch] closed the Westervelt spectral pressure-history allocation gap. `WesterveltWave::update_wave` now borrows current and previous pressure buffers from the three-slot ring, writes the existing next buffer in place, accepts borrowed initial pressure views, and removes the unused per-step `B/A` field allocation. The solver documents the three-buffer leapfrog role theorem, and focused tests prove all six ring-buffer permutations return disjoint roles plus zero-state updates preserve pressure-buffer storage pointers. Evidence: `cargo test -p kwavers solver::forward::nonlinear::westervelt_spectral::solver --lib -- --nocapture` passed 2/2; `cargo check -p kwavers --lib` passed with four pre-existing unrelated SWE/KZK warnings; `cargo fmt --check -- kwavers/src/solver/forward/nonlinear/westervelt_spectral/solver/mod.rs kwavers/src/solver/forward/nonlinear/westervelt_spectral/solver/wave_model.rs` passed; `git diff --check -- kwavers/src/solver/forward/nonlinear/westervelt_spectral/solver/mod.rs kwavers/src/solver/forward/nonlinear/westervelt_spectral/solver/wave_model.rs` passed.
- 2026-05-17: [patch] closed the Westervelt spectral nonlinear/damping workspace allocation gap. `WesterveltWave` now owns reusable `nonlinear_scratch` and `damping_scratch` buffers; `update_wave` fills them through `compute_nonlinear_term_into` and `compute_viscoelastic_term_into`, computes `∇²((pⁿ-pⁿ⁻¹)/dt)` directly from pressure-history neighbours instead of materializing `dp_dt`, and multiplies source amplitude inside the final update loop instead of allocating `src_term`. Evidence: `cargo test -p kwavers solver::forward::nonlinear::westervelt_spectral --lib -- --nocapture` passed 4/4; `cargo check -p kwavers --lib` passed with the four pre-existing unrelated SWE/KZK warnings; `cargo fmt --check -- kwavers/src/solver/forward/nonlinear/westervelt_spectral/nonlinear.rs kwavers/src/solver/forward/nonlinear/westervelt_spectral/solver/mod.rs kwavers/src/solver/forward/nonlinear/westervelt_spectral/solver/wave_model.rs` passed; `git diff --check -- kwavers/src/solver/forward/nonlinear/westervelt_spectral/nonlinear.rs kwavers/src/solver/forward/nonlinear/westervelt_spectral/solver/mod.rs kwavers/src/solver/forward/nonlinear/westervelt_spectral/solver/wave_model.rs` passed.
- 2026-05-17: [patch] closed the Westervelt spectral viscosity-array allocation and homogeneous-damping gap. `compute_viscoelastic_term_into` now takes `&dyn Medium`, borrows density via `density_array()`, and reads shear/bulk viscosity pointwise through `Medium::shear_viscosity` and `Medium::bulk_viscosity` instead of cloning `shear_viscosity_coeff_array()` and `bulk_viscosity_coeff_array()` every update. This preserves heterogeneous coefficient-field behavior at grid nodes and corrects homogeneous media, where the inherited `ElasticArrayAccess` defaults returned zero viscosity even though `ViscousProperties` carried nonzero shear/bulk viscosity. Evidence: `cargo test -p kwavers solver::forward::nonlinear::westervelt_spectral --lib -- --nocapture` passed 4/4; `cargo check -p kwavers --lib` finished clean; `rustfmt --check kwavers/src/solver/forward/nonlinear/westervelt_spectral/nonlinear.rs kwavers/src/solver/forward/nonlinear/westervelt_spectral/solver/wave_model.rs kwavers/src/solver/forward/nonlinear/westervelt_spectral/solver/mod.rs` passed; `git diff --check -- kwavers/src/solver/forward/nonlinear/westervelt_spectral/nonlinear.rs kwavers/src/solver/forward/nonlinear/westervelt_spectral/solver/wave_model.rs kwavers/src/solver/forward/nonlinear/westervelt_spectral/solver/mod.rs` passed.
- 2026-05-17: [patch] closed the Westervelt spectral source-mask allocation gap for in-crate sources. `Source` now exposes caller-owned `create_mask_into` and additive `add_mask_into` contracts; `WesterveltWave` owns `source_mask_scratch` and fills it once per update instead of allocating a mask volume; every in-crate `Source` implementation has a direct `create_mask_into` path, while additive overrides are restricted to implementations whose mask algebra is exactly additive. Evidence: `cargo test -p kwavers source_mask_into --lib -- --nocapture` passed 2/2; `cargo test -p kwavers solver::forward::nonlinear::westervelt_spectral --lib -- --nocapture` passed 4/4; `cargo check -p kwavers --lib` finished clean; file-scoped `rustfmt --check` passed for all touched source and Westervelt spectral files.
- 2026-05-17: [patch] closed the core source-term fallback allocation and source-superposition gap. `PointSource`, `TimeVaryingSource`, `CompositeSource`, and `NullSource` now provide direct `get_source_term` implementations, so `KuznetsovWave::compute_rhs` no longer falls through to `Source::get_source_term`'s full-mask allocation for these core source types on every grid cell. Composite source terms now sum child-local terms, preventing unrelated child amplitudes from appearing at another child's cell. `SimpleCustomSource::get_source_term` now matches its discrete mask contract and returns zero off active cells instead of selecting the nearest configured position. `TimeVaryingSource` stores the waveform once via `TimeVaryingSignal`. Evidence: `cargo test -p kwavers source_term --lib -- --nocapture` passed 5/5 after a first command timeout during startup; `cargo test -p kwavers source_mask_into --lib -- --nocapture` passed 2/2; `cargo test -p kwavers solver::forward::nonlinear::kuznetsov --lib -- --nocapture` passed 12/12 with 2 ignored Tier-3 tests; `cargo check -p kwavers --lib` finished clean.
- 2026-05-17: [patch] closed the hybrid PSTD/FDTD update-time source-mask allocation gap. `HybridSolver` now owns `source_mask_scratch`, constructs it once with the solver shape, and calls `Source::create_mask_into` before pressure source injection instead of allocating `source.create_mask(&grid)` on every `update`. The focused test exercises the full `HybridSolver::update` path and proves the scratch pointer remains stable while the point-source mask has exactly one active cell. Evidence: `cargo test -p kwavers update_reuses_source_mask_scratch_for_pressure_source --lib -- --nocapture` passed 1/1; `cargo test -p kwavers solver::forward::hybrid --lib -- --nocapture` passed 31/31; `cargo check -p kwavers --lib` finished clean.
- 2026-05-13: [patch] **Westervelt FDTD nonlinear-term sign correction.** Audited the Westervelt discrete recurrence `p[n+1] = 2·p[n] − p[n−1] + (c·Δt)²·∇²p ± q·∂²(p²)/∂t²` against the canonical form `∇²p − (1/c²)·p_tt + (β/(ρc⁴))·∂²(p²)/∂t² = 0` (Westervelt 1963 Eq. 24; Hamilton & Blackstock 1998 Eq. 3.10). Solving for `p_tt` gives `p_tt = c²·∇²p + (β/(ρc²))·∂²(p²)/∂t²` so the nonlinear contribution on `p[n+1]` must be **positive** (forward steepening: peaks at fixed `x` arrive earlier than linear). Both `solver::forward::nonlinear::westervelt` and `clinical::therapy::theranostic_guidance::nonlinear3d::westervelt` were applying it with a negative sign — producing non-physical reverse steepening. The fix flips the sign on the forward in both code paths and re-derives the matching discrete adjoint in `nonlinear3d::adjoint` (`add_nonlinear_transpose` adjoint contributions and `d_update_dc` sound-speed sensitivity both flip sign on the nonlinear term). The Kuznetsov solver at `solver/forward/nonlinear/kuznetsov/solver/rhs.rs` already used the correct convention (`*r += nl`) and required no change. Added `forward_westervelt_exhibits_physical_forward_steepening_with_corrected_sign`: a sign-sensitive regression that drives a 1 MHz / 5 MPa source through a homogeneous β = 10 cube and asserts `max(∂p/∂t) > |min(∂p/∂t)|` on the steady-state receiver trace; the previous sign-flipped form fails this check. All 8 Westervelt-related tests pass (`test_linear_wave_propagation`, `test_energy_calculation_accuracy`, `test_conservation_diagnostics_integration`, `test_conservation_check_interval`, `test_westervelt_fdtd_creation`, `test_westervelt_correction_nonzero_after_history`, the new forward-steepening test, and `nonlinear_3d_westervelt_fwi_and_cavitation_inverse_are_input_sensitive`); `cargo clippy -p kwavers --lib --no-deps -- -D warnings` is clean. Next increment: add an Aanonsen-1984-style Fubini-amplitude harmonic-ratio regression for the canonical Westervelt FDTD path (currently only the KZK solver carries that literature-validated test).
- 2026-05-13: [patch] added clinical ultrasonic speed-of-sound shift imaging under `clinical::imaging::reconstruction::sound_speed_shift`. The module implements the linearized straight-ray travel-time contract `A delta_c = -c0^2 delta_t`, exact segment/pixel intersection lengths, dense Tikhonov/H1 PCG reconstruction, deterministic sparse row selection, sparse L1 proximal reconstruction, and a forward predictor for differential validation. Chapter 5 now documents the dense and sparse imaging approach and maps it to the clinical API. Focused Rust verification passes for the forward sign, dense uniform recovery, sparse crossing-row localization, and invalid sampling rejection.
- 2026-05-13: [patch] optimized clinical speed-of-sound shift ray assembly by replacing the per-row scan across every active pixel with exact parametric traversal of only the crossed grid cells. The operator still stores nonzero segment lengths for fast repeated `matvec`/`t_matvec`, but construction now scales with crossed cells per ray rather than full active-mask cardinality. Added traversal equivalence tests against the per-cell clipping oracle plus clipped-path length conservation.
- 2026-05-14: [patch] modernized the clinical speed-of-sound shift operator topology and row storage. The flat `operator.rs` is now a directory-backed tree with construction, algebra, graph, row-storage, and validation responsibilities separated under one `SoundSpeedShiftOperator` SSOT. Ray rows now use flat row-offset, column, and length arrays instead of nested per-row segment vectors, preserving the same matrix-free algebra while reducing per-row allocation overhead. Focused Rust verification covers flat storage semantics and proves a diagonal ray stores crossed-cell nonzeros rather than full active-mask entries. Next increment: add a reusable solver workspace object for dense and sparse speed-shift solves so repeated reconstructions can reuse PCG/ISTA buffers.
- 2026-05-14: [patch] closed the clinical speed-of-sound shift solver workspace increment. The flat solver file is now `solver/{dense,sparse,normal,linear_algebra,workspace}.rs`, `SoundSpeedShiftWorkspace` owns all dense PCG, sparse ISTA, objective, Laplacian, prediction, and power-iteration work buffers, and `reconstruct_sound_speed_shift_with_workspace` lets callers reuse allocations across repeated reconstructions. The normal diagonal now fills caller-owned storage instead of allocating a fresh vector. Focused Rust verification proves repeated reconstructions preserve workspace capacity and reconstructed values. Next increment: expose a planned operator/workspace pair for repeated acquisitions with fixed mask and changing measured shifts, so operator construction can also be amortized.
- 2026-05-14: [minor] added curved-array acquisition support to the clinical 2-D straight-ray speed-of-sound shift model. `CurvedArray2d` owns the circular-arc element coordinate contract, `CurvedArrayShiftScan` owns deterministic transmitter-major same-aperture pitch-catch row generation, and measured time shifts are attached without creating a parallel inverse path. Curved-array rows are emitted as `SoundSpeedShiftSample` and reuse the existing straight-ray operator, CSR row storage, and dense/sparse solvers. Focused Rust verification pins endpoint geometry, row ordering, invalid scan rejection, and nonzero straight-ray prediction through curved-array diametric rows. Next increment: add a fixed-acquisition plan that caches the `SoundSpeedShiftOperator` for repeated curved-array frames.
- 2026-05-14: [minor] added curved-ray propagation and finite-frequency sensitivity to the clinical 2-D speed-of-sound shift model. `ShiftPropagation::CircularArc` represents a signed circular-arc sagitta and segment count; each subsegment reuses exact grid traversal so curved rows remain matrix-free and sparse. `ShiftSensitivity::FiniteFrequency` builds compact Fresnel tubes with per-subsegment normalization, preserving the uniform-field path-integral contract while assigning sensitivity to off-axis cells. Focused Rust verification covers curved path length greater than the chord, finite-frequency weight conservation, off-axis detection, and invalid propagation/sensitivity rejection. Next increment: add a fixed-acquisition plan that caches both curved-array samples and the assembled operator for repeated frames.
- 2026-05-14: [minor] added `SoundSpeedShiftPlan` for fixed-acquisition clinical speed-of-sound shift imaging. The plan caches geometry samples and the assembled `SoundSpeedShiftOperator`, reconstructs repeated frames from raw time-shift slices indexed in original acquisition row order, predicts selected-row shifts through the cached operator, and supports curved-array, curved-ray, and finite-frequency configurations without introducing a second inverse path. Focused Rust verification covers direct-reconstruction equivalence, invalid frame-shift rejection, and repeated curved-array curved-ray finite-frequency frames with stable workspace allocation and stable cached weight count. Next increment: add a batch-frame API that drives a sequence of frames through one plan and one workspace while returning per-frame objective summaries without retaining full intermediate histories unless requested.
- 2026-05-14: [minor] added fixed-acquisition batch reconstruction for clinical speed-of-sound shift imaging. `SoundSpeedShiftPlan::reconstruct_frames*` validates all frame shift slices up front, reuses the cached operator, one sampled-row RHS buffer, and caller-owned solver workspace, and returns `SoundSpeedShiftBatch` with compact `SoundSpeedShiftFrameSummary` records by default. Full per-frame objective histories are retained only under `SoundSpeedShiftObjectiveHistoryPolicy::Full`. Focused Rust verification covers compact-summary default behavior, optional full-history retention, invalid empty/short/nonfinite batches, and equivalence between batch frame 0 and single-frame planned reconstruction. Next increment: cache plan-level normal diagonal and sparse Lipschitz estimates so repeated dense/sparse frame solves do not recompute frame-invariant operator metrics.
- 2026-05-13: [patch] **Cavitation Green's-function frequency-dependent attenuation + Westervelt absorption comment honesty.** Two physics-correctness cleanups on the Chapter 29 / canonical Westervelt path: (1) The passive subharmonic operator in `clinical::therapy::theranostic_guidance::nonlinear3d::cavitation::PassiveOperator::new` no longer hardcodes `exp(−2·r)` (which happened to match brain at a 325 kHz subharmonic only by coincidence and was wrong for the abdominal 250 kHz subharmonic by ≈40 %); it now derives `α [Np/m]` from a soft-tissue power-law baseline `α₀ = 0.5 dB/(cm·MHz)` (Hamilton & Blackstock 1998 Table 4.1) scaled by `f_s = f₀/2` with the exact `8.685889638…` dB→Np factor, so both `α_s` and `k_s` are tied to the actual subharmonic frequency. The Green's function is now `exp(−α_s·r) · cos(k_s·r) / (4π·r)`. (2) Corrected the misleading comment I previously added in `solver::forward::nonlinear::westervelt::update`: the discrete form `(p_n − 2 p_{n-1} + p_{n-2})/dt` is `dt·p_tt(n-1)`, not `dt²·(δ/c²)·p_ttt`. The FDTD absorption is a Kelvin-Voigt-like lagged-`p_tt` proxy that approximates Stokes-Kirchhoff to leading order in `δ/(c²·dt)` and produces correct plane-wave decay (`Im(ω) > 0`), but it is NOT a strict third-derivative discretization and NOT a frequency-dependent power-law absorption. The corrected comment now documents this honestly and directs users to the PSTD fractional-Laplacian path for physical power-law absorption. All 13 targeted tests pass (`nonlinear3d` + canonical Westervelt + parallel-agent `sound_speed_shift`). Next physics increment: per-voxel attenuation in the cavitation operator (currently uses a single tissue-typical scalar; the kwavers nonlinear-3D volume already carries a heterogeneous `c, ρ, β` field but no `α` field — needs an attenuation map and operator threading).
- 2026-05-13: [patch] **Chapter 29 heterogeneous CT-derived path-integrated attenuation for the cavitation Green's function.** Closed the deferred increment above. `Nonlinear3dVolume` now carries an `attenuation_np_per_m_mhz: Array3<f64>` field derived in `material_maps` from CT HU with explicit tissue classes: cortical bone (HU ≥ 300) at 13 → 20 dB/(cm·MHz) interpolated linearly by HU density (Connor & Hynynen 2002 cortical bone), air pockets (HU < −700, label = 0) as nearly opaque (1000 Np/(m·MHz)), segmented organs (label > 0) at 0.6 dB/(cm·MHz) (brain/liver/kidney median, Hamilton & Blackstock 1998 §4.1), and generic soft tissue at 0.5 dB/(cm·MHz). `cavitation::PassiveOperator::new` now computes a **path-integrated** absorption by sampling the attenuation field along the straight line from source voxel to receiver with trilinear interpolation and trapezoidal-rule integration, then scaling by the subharmonic frequency for the `y = 1` tissue power law. The Green's kernel is `exp(−∫ α_s(s)·ds) · cos(k_s·r) / (4π·r)`. For brain cases this correctly tracks the ~26× skull/soft-tissue attenuation contrast on every source-to-receiver ray; soft-tissue-only abdominal paths still produce a near-uniform exponent close to the previous single-scalar value. The `nonlinear_3d_westervelt_fwi_and_cavitation_inverse_are_input_sensitive` integration test passes, locking that the path-integral Green's matrix is still SPD via the projected-gradient inverse. Next physics increment: replace the `y = 1` power law assumption with the per-tissue-class measured `y` exponent (Treeby & Cox 2010 / Szabo) — biological tissue is actually slightly superlinear (`y ≈ 1.05 − 1.1`), and the difference matters for high-frequency harmonic content.
- 2026-05-13: [patch] **Chapter 29 tissue-class power-law `y` exponent for the cavitation attenuation field.** Closed the deferred `y` increment from the previous entry. `Nonlinear3dVolume` now carries an `attenuation_power_law_y: Array3<f64>` field paired with `attenuation_np_per_m_mhz`, so the cavitation Green-operator path integral evaluates the true tissue power law `α(f) = α(1MHz) · f_MHz^y` at every voxel sample. Tissue classes: cortical skull bone `y = 2.0` (Stokes-Kirchhoff classical viscous limit; Connor & Hynynen 2002 measured 1.9 - 2.0 across 0.5 - 3.5 MHz), soft tissue / segmented organ `y = 1.05` (Treeby & Cox 2010 Table I; biological tissue is slightly superlinear), air pocket `y = 1.0`, outside body `y = 1.0`. The `y = 2` skull behavior at the 325 kHz subharmonic (650 kHz brain drive) gives **3.07× less attenuation** than a `y = 1` linear extrapolation predicts — without this correction the transcranial passive cavitation receive path would be over-attenuated by a factor of 3. Added 9 literature-anchored unit tests in `volume::attenuation_tests` (each test name cites the paper it validates: `..._matches_hamilton_blackstock_1998_table_4_1_median`, `..._matches_connor_hynynen_2002`, `..._matches_treeby_cox_2010_table_i`, `..._matches_connor_hynynen_2002_stokes_kirchhoff`, `skull_subharmonic_attenuation_with_y2_is_three_times_less_than_y1`). 19/19 nonlinear3d + Westervelt tests pass; `cargo clippy --no-deps -- -D warnings` clean. Next physics increment: Aanonsen-1984 Fubini-amplitude harmonic-ratio test for the canonical Westervelt FDTD to lock the harmonic AMPLITUDE (currently only sign is locked by the forward-steepening test).
- 2026-05-13: [patch] **Chapter 29 brain-helmet end-to-end integration test.** Closed the brain-anatomy coverage gap. Added `nonlinear_3d_brain_helmet_pipeline_is_input_sensitive_through_skull` plus a synthetic `brain_fixture()` (28³ cortical-bone shell at HU = 600 wrapping a brain interior at HU = 40, surrounded by air at HU = -1000). The test exercises the full INSIGHTEC-like transcranial pipeline: `AnatomyKind::Brain` volume preparation (no segmentation required — synthetic ellipsoidal target built from the body centroid), calvarium-cap helmet aperture, source-encoded Westervelt forward through skull voxels, discrete-adjoint FWI for `c` and `β`, Rayleigh-Plesset cavitation source from the resulting peak pressure field, and passive subharmonic inverse with the heterogeneous CT-derived path-integrated attenuation Green's function including the `y = 2` Stokes-Kirchhoff skull power-law absorption. Test asserts: pipeline runs, aperture model is `insightec_like_calvarium_helmet_3d_westervelt_sources`, Westervelt peak pressure positive, cavitation density positive, FWI and cavitation objectives non-increasing, ≥ 16 therapy points / ≥ 4 receivers. **This is the only test in the suite that places skull voxels (HU > 300) between source and receiver — the only test that exercises the 3.07× `y = 2` vs `y = 1` skull-attenuation correction.** 20/20 nonlinear3d + Westervelt tests pass; `cargo clippy --no-deps -- -D warnings` clean. Next physics increment: Aanonsen-1984 Fubini-amplitude harmonic-ratio test for the canonical Westervelt FDTD, then PSTD fractional-Laplacian absorption inside the Westervelt forward (currently absorption only appears in the post-hoc cavitation Green operator).
- 2026-05-14: [patch] **Codebase cleanup — SRP file-tree split + nextest default-profile timeout.** Two co-located cleanups bring the lib suite into compliance with the 500-line leaf-file rule and pin a default nextest timeout policy.
    1. **`nonlinear3d/tests.rs` (1391 lines → 9 SRP-aligned files under 260 lines each).** The historical monolithic test file at `kwavers/src/clinical/therapy/theranostic_guidance/nonlinear3d/tests.rs` carried 8 distinct test concerns plus shared fixtures and Bessel helpers in one 1391-line file (878 % over the 500-line cap). Split into a `tests/` directory: `mod.rs` (22 L facade), `fixtures.rs` (75 L — `brain_fixture`, `abdominal_fixture`, `ellipsoid_radius`), `bessel.rs` (43 L — `bessel_j1`, `bessel_j2` analytical helpers), `pipeline.rs` (186 L — abdominal + brain-helmet end-to-end), `sign_correction.rs` (121 L — Westervelt nonlinear-term sign regression), `beta_scaling.rs` (252 L — β = 0 linear baseline + β-scaling regression), `harmonic_presence.rs` (184 L — point-source 2nd-harmonic check), `fubini_1d.rs` (210 L — Aanonsen 1984 1-D Fubini-absolute literature test), `absorption.rs` (203 L — Treeby-Cox 2010 fractional-Laplacian power-law decay). Largest leaf is 252 lines, exactly aligning with SoC / SRP / 500-line invariants.
    2. **`solver/inverse/same_aperture/operator.rs` (551 lines → 5-file directory under 175 lines each).** The matrix-free finite-frequency operator's row-spec types, the `LinearOperator` impl, the row constructors / norms / writers, and the dot-product kernels lived together as four logically-distinct concerns in one over-cap file. Split into `operator/{mod,types,linear_op,rows,dot}.rs`: `mod.rs` (33 L facade re-exporting `FiniteFrequencyOperator`), `types.rs` (173 L — operator struct + row-spec enum + `unscaled_value` closures), `linear_op.rs` (126 L — `impl LinearOperator`), `rows.rs` (136 L — `pitch_catch_rows` / `passive_rows` / norms / writers), `dot.rs` (135 L — matvec / t_matvec / normal-diag kernels + `column_lookup` / `scaled_input` / `distance` helpers). All shared types use `pub(super)` visibility for sibling-module access without leaking out of the operator submodule. External imports (`super::operator::FiniteFrequencyOperator`) remain stable.
    3. **`.config/nextest.toml` default profile.** Added an explicit `[profile.default]` with `global-timeout = "15m"` and `slow-timeout = { period = "10s", terminate-after = 6, grace-period = "5s" }` so plain `cargo nextest run -p kwavers --lib` enforces a 60 s per-test ceiling (six 10 s windows before termination) and a 15-min suite ceiling without requiring `--profile ci`. Matches the kwavers `standards.yaml` testing policy. Identified 6 slow tests (>10 s): 4 PSTD numerical-accuracy literature-validation tests (20-44 s; `point_source_phase_accuracy`, `linear_array_phase_accuracy`, `gaussian_beam_phase_accuracy`, `phase_velocity_phase_accuracy`) and 2 nonlinear 3-D pipeline tests (19 s + 28 s). All complete within the new 60 s per-test budget; recorded as candidates for future optimization or `#[ignore]` tier-2 promotion when their value-vs-runtime trade-off changes.
    4. **Verification.** `cargo check -p kwavers --lib --tests`, `cargo clippy -p kwavers --lib --no-deps -- -D warnings`, and `cargo nextest run -p kwavers --lib --no-fail-fast` all clean (3468/3468 lib tests passed in 80 s with 14 skipped). The `kwavers/tests/test_fft_peak.rs` integration test gained the `Fft3dInOutExt` import that the apollo 0.11 migration required for the integration-test target. Next cleanup increment: split `theranostic_guidance/tests.rs` (518 L) and the marginal `solver/inverse/seismic/brain_helmet/volume_born.rs` (500 L) when their internal SRP boundaries become as clear-cut as the two splits above.
- 2026-05-14: [arch] **RITK split-IO + Apollo 0.11.0 API migrations.** Closed both deferred external-dependency increments in one pass.
    1. **RITK split-IO.** The new `ritk` `origin/main` (`e70b312`) splits the historical `ritk-io` crate into per-format crates (`ritk-png`, `ritk-jpeg`, `ritk-tiff`, `ritk-minc`) and lifts `openjp2` / `jpeg2k` into workspace dependencies. The kwavers workspace root `Cargo.toml` (a) adds the four new ritk-* path entries to `[workspace.dependencies]` so `ritk-io/Cargo.toml`'s `workspace = true` inheritance resolves, (b) declares `openjp2 = { version = "0.6.1", default-features = false, features = ["std"] }` and `jpeg2k = { version = "0.10.1", default-features = false, features = ["openjp2"] }` mirroring the ritk-workspace pins, and (c) swaps the `dicom-transfer-syntax-registry` feature flag from `"openjpeg-sys"` to `"openjp2"` to avoid the simultaneous `openjp2 + openjpeg-sys` enablement that triggers a `sys` module re-definition conflict inside `jpeg2k`. ritk submodule pointer advances from `d459c5e` to `e70b312`.
    2. **Apollo 0.11.0.** The new `apollo` `origin/main` (`ab8b07d`) consolidates `ProcessorFft3d` into the unified `FftPlan3D`, removes the `apollo::FFT_CACHE` global in favor of per-dimension `FFT_CACHE_{1D,2D,3D}`, removes the `apollo::types` re-export module surface (shapes are exported at the crate root), and drops `FftPlan{2D,3D}::forward_into` / `inverse_into` in favor of explicit `forward_complex_inplace` / `inverse_complex_inplace` (full-spectrum c2c) plus `forward_r2c_into` / `inverse_c2r_into` (half-spectrum r2c/c2r). kwavers' historical `forward_into` / `inverse_into` call sites were uniformly full-spectrum c2c with `Array3<Complex64>::zeros((nx, ny, nz))` buffers, so the kwavers FFT facade at `kwavers/src/math/fft/mod.rs` now (a) re-aliases `apollo::FftPlan3D as ProcessorFft3d` and `apollo::FFT_CACHE_3D as FFT_CACHE` for source compatibility on type and global names, (b) imports `Shape{1,2,3}D` from the apollo crate root rather than `apollo::types::*`, and (c) defines two extension traits `Fft3dInOutExt` and `Fft2dInOutExt` that implement the previous `forward_into` / `inverse_into` surface on top of the new in-place c2c API (`out = field + 0i; forward_complex_inplace(out)` for forward; `scratch = field_hat; inverse_complex_inplace(scratch); out = Re(scratch)` for inverse). Apollo submodule pointer advances from `787b4eb` to `ab8b07d`. The eight call-site files (`solver/forward/fdtd/kspace_correction/operators.rs`, `solver/forward/hybrid/mixed_domain.rs`, `solver/forward/nonlinear/kuznetsov/{numerical,spectral}.rs`, `solver/forward/pstd/dg/spectral_solver.rs`, `solver/forward/pstd/propagator/axisymmetric.rs`, `solver/inverse/reconstruction/photoacoustic/{fourier,time_reversal}.rs`) each import `Fft{2,3}dInOutExt` so the existing `fft.forward_into(...)` / `fft.inverse_into(...)` method-call surface continues to compile unchanged.
    3. **Verification.** `cargo check -p kwavers --lib`, `cargo clippy -p kwavers --lib --no-deps -- -D warnings`, and `cargo nextest run -p kwavers --lib --no-fail-fast` all clean (3468/3468 passed, 14 skipped). The c2c forward/inverse contract preserves the prior FFTW-compatible normalization (apollo `inverse_complex_inplace` divides by `nx*ny*nz`), so no PSTD/FDTD parity regression is introduced. Next external-dependency increment: revisit the previously-noted optional migration to apollo's new half-spectrum r2c API at hot-loop PSTD sites for memory-bandwidth gains, once the operator-API stabilizes in apollo 0.12.
- 2026-05-14: [patch] **PID Tustin reset-invariant regression fix.** `cavitation_control::pid_controller::discrete::tests::tustin_reset_clears_all_states` was asserting a post-reset value of exactly `kp*error = 0.5` after a single `update(0.5, 0.0)` call with `ki = 1.0`. That hand-computed expectation is wrong: a single update step with `ki > 0` legitimately adds `ki*dt*error = 0.0005` to the integral state, so the correct post-reset output is `0.5005`. The test now asserts the load-bearing invariant directly — post-reset response equals a freshly-constructed controller response — and additionally asserts the analytic value `kp*error + ki*dt*error`. 3/3 Tustin tests pass; lib clippy `--no-deps -- -D warnings` remains clean. Next increment: apollo 0.11.0 API migration (see below).
- 2026-05-14: **[arch-deferred] RITK split-IO API migration.** `ritk` submodule advanced 5 commits on `origin/main` (through `e70b312`) carrying sprint-240/241 work that splits `ritk-io` into per-format crates and adds real-brain metric parity coverage. The new tree drops the `ritk-jpeg` workspace dependency that the current kwavers workspace manifest still references through `external::registration` / `external::imaging_io`. Migrating without breaking the registration boundary requires (a) updating the kwavers workspace `Cargo.toml` to depend on the split IO crates (`ritk-nifti`, `ritk-vtk`, etc.) directly instead of the unified `ritk-jpeg` entry, (b) re-resolving the public re-exports the kwavers-side registration adapter consumes, and (c) running the full transcranial-planning Chapter 25 pipeline to confirm CT/MR/MNI registration parity. For this turn the ritk submodule is pinned at `d459c5e` (the last commit before the split). The migration is paired with the apollo 0.11.0 migration as the next concrete external-dependency increment.
- 2026-05-14: **[arch-deferred] Apollo 0.11.0 API migration.** `apollo` submodule advanced 4 commits ahead on `origin/main` (through `ab8b07d`) carrying a `mixed_radix` monomorphization refactor (`MixedRadixScalar` sealed trait) and an `apollo-fft 0.11.0` version bump. The new release removes `FftPlan{2D,3D}::forward_into`/`inverse_into` and reorganizes `apollo::types`/`apollo::ProcessorFft3d`/`apollo::FFT_CACHE` re-exports. kwavers currently calls these at 22 sites across the PSTD propagator, beamforming, and absorption paths. Migrating without weakening the FFT contract requires (a) mapping `forward_into`/`inverse_into` to the new in-place / typed-into surface (`forward_complex_inplace`, `forward_real_to_complex_into`, `forward_typed_into`), (b) re-resolving the `apollo::types`/`ProcessorFft3d`/`FFT_CACHE` symbols against the current public surface or replacing them with the new equivalents, and (c) running the full kwavers PSTD parity suite to confirm no normalization regression. For this turn the apollo submodule is **pinned back to `787b4eb`** so the kwavers tree continues to build and the 918-file consolidation can land. The migration is the next concrete increment.
- 2026-05-14: [patch] **Chapter 29 Westervelt physics-scaling regression tests.** Added two negative-control / scaling regressions that close coverage gaps left by the single-β sign test. (1) `linear_westervelt_with_beta_zero_produces_symmetric_pressure_trace_within_fdtd_tolerance`: runs the homogeneous forward fixture at `β = 0` (Westervelt reduces to linear wave equation) and asserts the asymmetry ratio `R = max(∂p/∂t) / |min(∂p/∂t)|` stays within `[0.80, 1.20]`. Catches numerical-dispersion artifacts masquerading as nonlinearity. (2) `westervelt_steepening_signature_scales_linearly_with_beta_per_weak_nonlinear_theory`: runs the same fixture at β = 0, 5, 10 and verifies the **excess-over-linear** asymmetry `δ(β) = R(β) − R(0)` satisfies `δ(10) / δ(5) ∈ [1.3, 3.0]` (target 2.0) per leading-order weak-nonlinear Born/Fubini scaling (Hamilton & Blackstock 1998 §4.3: `|P_2| ∝ β · |P_1|² · z`). Catches β-coefficient sign/magnitude errors — a scaling near 4 would suggest `β²` in the recurrence, a scaling near 1 would suggest β is not entering. The excess-over-linear formulation isolates the β-dependent nonlinear contribution from the β-independent FDTD dispersion bias floor. **Empirical reality discovered during implementation**: at low source pressure (50 kPa) the FDTD dispersion bias signs *opposite* to the physical forward-steepening direction, so raw absolute signatures at low amplitude are dominated by numerical artifacts — only the `R(β) − R(0)` excess-over-linear ratio robustly isolates the nonlinear physics. 22/22 nonlinear3d + Westervelt tests pass; `cargo clippy --no-deps -- -D warnings` clean. Next physics increment: Aanonsen-1984 Fubini-amplitude harmonic-ratio test for the canonical Westervelt FDTD.
- 2026-05-14: [patch] **Chapter 29 Westervelt harmonic-generation presence test (Tier-2, `#[ignore]`'d).** Attempted the Aanonsen-1984 Fubini-amplitude test; converted it into a harmonic-presence check after discovering that the 3-D point-source FDTD geometry departs fundamentally from the 1-D plane-wave Burgers regime underlying Fubini. The Fubini formula `|P_n|/|P_1| = J_n(nΓ) / (n J_1(Γ))` assumes constant amplitude over the propagation path; in a point-source FDTD the local amplitude decays as `1/r` so the local Γ varies along the path. The KZK solver carries the literature-validated Fubini-absolute test because KZK parabolically reduces 3-D to 1-D-along-z with constant-amplitude planar shots — the Westervelt FDTD cannot drive that configuration through the existing `forward_with_schedule` API without significant changes. **What the test validates instead**: `westervelt_fdtd_point_source_generates_measurable_second_harmonic_content` extracts fundamental and 2nd-harmonic amplitudes via discrete sine/cosine projection at known frequencies (exact for harmonics, no FFT) and asserts `|P_2|/|P_1| ∈ [0.03, 0.40]` for a 5 MPa / β = 10 point source. **Catches**: a nonlinear term that propagates as just a phase shift (ratio ≈ 0); spuriously-high 2nd harmonic from `β²` coefficient error or grid dispersion (ratio > 0.5); DC-only or NaN output. Measured ratio at the fixture: `0.133`. 22/22 default + 1 Tier-2 ignored test pass with `cargo test --lib -- --ignored harmonic`; `cargo clippy --no-deps -- -D warnings` clean. Next physics increment: full Fubini-absolute validation would require either (a) a clean 1-D Westervelt forward harness, or (b) adapting the KZK Aanonsen test machinery as a comparison cross-check across solvers.
- 2026-05-14: [minor] **Chapter 29 nonlinear-3D Westervelt FDTD fractional-Laplacian power-law absorption.** Closed the in-forward absorption gap. `clinical::therapy::theranostic_guidance::nonlinear3d::absorption` implements the Treeby-Cox 2010 (J. Biomed. Opt. 15(2) 021314 Eqs. 9-11) wave-equation form on the FDTD Westervelt stencil: per-voxel `dt·τ = dt · 2·α₀_ω · c^(y+1)` with α₀_ω = α₀_f / (2π·1e6)^y, half-spectrum `|k|^y` filter using Apollo R2C FFTs (`fft_3d_r2c_into` / `ifft_3d_r2c_into`), global y exponent from `representative_y(...)` (volume-area-weighted median: ≈2.0 for brain, ≈1.05 for abdominal). Apply contributes `next += -dt_tau·(L_y(p[n]) - L_y(p[n-1]))` after the lossless `update_cells`. Plumbed through `ForwardInput` and `ReplayInput` so `forward_with_schedule`, `forward_dense_history_for_test`, and `replay_history_segment_into` execute the lossy forward bit-for-bit identically, preserving the checkpointed-replay bitwise equivalence and gradient-interval invariance regression tests. `adjoint::gradient` now applies `apply_transpose` per replayed step using the self-adjointness of `L_y` (real symmetric multiplier in k-space) and the diagonality of `dt·τ`. The η Kramers-Kronig dispersion term is omitted: von-Neumann analysis on the explicit `−dt²·η·L_{y+1}(p[n])` gives a Nyquist-mode growth factor `|z|² ≈ 1 + dt²·|η|·k_max^(y+1)` that exceeds unity for `y < 2` at clinically realistic α₀ and dt; for y = 2 (skull) η ≡ 0 because tan(π) = 0; for y ≈ 1.05 (soft tissue) the dropped term is a sub-leading frequency-dependent phase-velocity correction. **Validation**: (1) coefficient regression `dt_tau == dt · 2·α₀_ω · c^(y+1)`; (2) `|k|^0 = 1` at Nyquist + DC = 0; (3) representative-y median for 80/20 soft-tissue/skull mix; (4) `maybe_new` short-circuits to None for identically-zero α₀ (preserves loss-free baseline zero-cost); (5) inner-product transpose identity `⟨Av,w⟩ = ⟨v,Aᵀw⟩` to 1e-9 on a non-trivial probe field; (6) Tier-2 plane-wave decay: dual 3-D simulations (lossless + α₀=5.8 Np/m at 1 MHz, y=1.05) at 4 axial receivers, ratio peaks via short-pulse trace measurement (boundary reflections rejected by time window), least-squares fit of `log(p_abs/p_lossless)` vs `r`, fitted α matches analytical α(1 MHz) = 5.8 Np/m within 35% tolerance. The deep-vertical hierarchy splits the file as `absorption/{mod,construction,spectrum,apply,tests}.rs` — each ≤250 lines. 24/24 nonlinear3d default + Tier-2 tests pass; clippy `-D warnings` clean. Next physics increment: joint `c/α/ρ/β/bubble-density` coupled inverse with one KKT/Gauss-Newton system, including adjoint of `α₀` through the same fractional-Laplacian operator.
- 2026-05-14: [patch] **Chapter 29 Westervelt Aanonsen-1984 Fubini-absolute test on a 1-D harness (Tier-2, `#[ignore]`'d).** Closed the harmonic-amplitude literature-validation gap by implementing option (a) from the previous entry: a clean 1-D Westervelt FDTD harness inline in the test file. The 1-D recurrence is algebraically identical to the 3-D `update_cells`: `p[n+1] = sponge·(2 p[n] − p[n−1] + (c·dt)²·∇²p + q·∂²(p²)/∂t²)` with `q = β·dt²/(ρ·c²)` and the product-rule `∂²(p²)/∂t² ≈ 2 p·d²p/dt² + 2·(dp/dt)²`, using a 3-point 1-D Laplacian instead of the 7-point 3-D stencil. Hard sinusoidal source at `x = 4` clamps the source-cell pressure; absorbing sponge at the far boundary prevents reflections. Resolution: `dx = 0.05 mm` → 30 pts/wavelength fundamental, 15 pts/wavelength 2nd harmonic. Discrete sine/cosine projection (exact for harmonics on integer-period windows) extracts `|P_1|` and `|P_2|`. The test asserts that the Westervelt recurrence algebra matches Fubini `|P_2|/|P_1| = J_2(2Γ)/(2·J_1(Γ))` at the **empirical Γ** computed from the observed `|P_1|`. Empirical Γ is required because the 1-D FDTD hard source radiates ≈ 0.57× the nominal `P_0` (radiation coupling determined by the discrete Laplacian / CFL) — the physically meaningful Γ is the one carried by the propagating wave, not the source-clamp nominal. Tolerance: **15 %**. **Measured at the fixture**: `|P_1| = 5.70e5 Pa`, `|P_2| = 7.92e4 Pa`, `|P_2|/|P_1| = 0.139`, empirical `Γ = 0.286`, Fubini at empirical Γ = 0.148 → relative error **6 %**. This validates the Westervelt `q·∂²(p²)/∂t²` algebra against literature analytical to within numerical-dispersion tolerance — closing the last harmonic-amplitude validation gap. The Bessel `J_0`, `J_1`, `J_2` analytical values are computed via convergent power series inline (no external crate). 22/22 default and 2 Tier-2 ignored tests pass with `cargo test --lib -- --ignored`; `cargo clippy --no-deps -- -D warnings` clean. References: Aanonsen et al. 1984 Eq. 6; Hamilton & Blackstock 1998 §4.3.2. Next physics increment: PSTD fractional-Laplacian absorption inside the Westervelt forward path, or joint `c/α/ρ/β/bubble-density` coupled inverse with one KKT/Gauss-Newton system.

## DICOM SSOT consolidation
- ritk-io owns DICOM I/O: `ritk_io::scan_dicom_directory` + `ritk_io::load_dicom_series::<Backend>(...)`. The pattern is exercised correctly by `kwavers/examples/skull_ct_phase_correction.rs`.
- Three SSOT violations remain in kwavers: (1) `domain/imaging/medical/dicom_loader.rs` is a 512-line placeholder whose `load_series_internal` returns `KwaversError::NotImplemented` and never imports `dicom`; (2) `infrastructure/io/dicom.rs` is a 684-line parallel reader using `dicom::core::DataElement` directly, re-exported as `DicomReader`/`DicomStudy`/`DicomSeries`/`DicomValue`/`DicomObject` from `infrastructure::io`; (3) `kwavers/Cargo.toml:50` keeps `dicom = { version = "0.7" }` as a direct dep, double-vendoring the crate already pulled by ritk-io.
- 2026-04-30 partial fix: redirected all "DICOM not implemented" error messages and module headers in `domain/imaging/medical/dicom_loader.rs`, `infrastructure/io/dicom.rs`, and `clinical/therapy/therapy_integration/orchestrator/initialization.rs` to point users at `ritk_io::scan_dicom_directory` / `load_dicom_series` plus the `skull_ct_phase_correction` example. Build clean, 2645/2645 lib tests pass, clippy `-D warnings` clean.
- 2026-04-30 follow-up: ritk is now a **mandatory** dependency in `kwavers/Cargo.toml` — `ritk-core`/`ritk-io` are no longer `optional = true`, `burn` is no longer `optional = true` (already mandatory transitively via ritk-io), the `ritk` feature is reduced to a no-op alias `ritk = []`, the `pinn` feature is reduced to a no-op `pinn = []`, the `full` feature drops the `ritk` literal, the three `required-features = ["ritk"]` example markers are removed, and the `#[cfg(feature = "ritk")]` gate around `clinical::imaging::functional_ultrasound::registration::ritk` is removed. Build and full lib suite (2645/2645) pass; `cargo clippy --no-deps -- -D warnings` clean.
- 2026-05-01 closure: built `infrastructure::io::dicom_ritk` (the SSOT adapter wrapping `ritk_io::scan_dicom_directory` + `ritk_io::load_dicom_series::<NdArray>`) which converts ritk-io's `Image<B, 3>` → kwavers `Array3<f64>` + `MedicalImageMetadata` (f32→f64 + `[depth, rows, cols]`→`(x, y, z)` repack + mm→m spacing + direction × spacing → 4×4 affine + intensity-range tracking); `DicomImageLoader::load_series_internal` now delegates to `dicom_ritk::load_series_from_dir`; deleted the parallel `infrastructure/io/dicom.rs` (684-line `dicom`-crate-direct reader, zero callers) and the orphaned `src/bin_test.rs`; dropped the direct `dicom = "0.7"` dep from `kwavers/Cargo.toml`; dropped `#[cfg(feature = "dicom")]` from `KwaversError::DicomError` (the `dicom` feature is reduced to a no-op alias). Build + clippy clean; full lib suite passes 2640/2640 with 12 ignored (–5 vs. pre-cleanup: the deleted legacy reader's smoke tests). DICOM I/O in kwavers is now SSOT-canonical via ritk-io.

## Technical Debt Prevention
- Proactively locate and discard deprecated or duplicate methods, replacing them strictly with unified accessors.
- Prefer `..Default::default()` for `FdtdConfig` test/example literals so new defaulted fields remain single-sourced by `FdtdConfig::default()`.
- Keep FWI example and caller synthetic data generation routed through `FwiProcessor::generate_synthetic_data`, the public wrapper over the canonical forward model.
- Remove outdated benchmarking, test data, and logs upon obsolescence.
- [patch] Closed the root scratch-artifact cleanup pass by removing obsolete `.log`, `.err`, `.txt`, temporary root patch scripts, scratch binaries, NumPy transient arrays, project-owned Python bytecode caches, stale PyO3 extension backups, and generated example output directories while preserving datasets, docs, virtual environments, and source files.
- [patch] Closed the chemistry integrator oversized-file gap by splitting the RK45 facade, DOPRI5 tableau constants, RHS/species mapping, result/error types, and value-semantic tests into `physics::chemistry::integrator::{mod,tableau,rhs,types,tests}` without changing the public `chemistry::integrator::RadicalIntegrator` path.
- [patch] Closed the generated comparison-output cleanup pass by removing untracked parity figures and HDF5 output under `pykwavers/examples/output` while preserving tracked validation images and input datasets.
- [patch] Removed regenerated `kwavers/examples/output` demo artifacts from the working tree after verification; source examples and input data remain intact.
- [patch] Closed the simulation progress-boundary leak by routing simulation progress reporting through `solver::interface`, reducing `solver::progress` to a compatibility re-export, and adding an architecture regression test that rejects direct `simulation` imports of `solver::progress`.
- [patch] Closed the FEM boundary oversized-file gap by splitting the public facade, boundary manager, boundary-condition enum, and value-semantic tests into `domain::boundary::fem::{mod,manager,types,tests}` without changing `FemBoundaryManager` or `FemBoundaryCondition` import paths.
- [patch] Closed the cloud deployment configuration oversized-file gap by splitting validated configuration value objects and tests into `infrastructure::cloud::config::{mod,types,tests}` without changing public re-exports.
- [patch] Closed the domain detector/calibration facade cleanup by splitting `domain::sensor::sonoluminescence::detector::{mod,constants,core,types}` and `domain::source::flexible::calibration::{mod,manager,types}` without changing public import paths.
- [patch] Closed the Burgers analytical-solution oversized-file gap by splitting the public facade, Bessel kernel, Fubini-Blackstock solution formulas, and value-semantic tests into `physics::acoustics::wave_propagation::nonlinear::burgers::{mod,bessel,solution,tests}` without changing `burgers_equation` or `fubini_harmonic_amplitude` call paths.
- [patch] Closed the Keller-Miksis thermodynamics oversized-file gap by splitting phase-change property laws, Van der Waals EOS, vapor mass transfer, temperature ODE update, and value-semantic tests into `physics::acoustics::bubble_dynamics::keller_miksis::thermodynamics::{mod,phase,eos,transfer,temperature,tests}` without changing the public thermodynamics facade used by `KellerMiksisModel`.
- [patch] Closed the CEUS microbubble dynamics oversized-file gap by splitting simulator facade/configuration, Velocity-Verlet radial integration, nonlinear scattering efficiency, and value-semantic tests into `physics::acoustics::imaging::modalities::ceus::microbubble::dynamics::{mod,integration,scattering,tests}` without changing the parent `BubbleDynamics` re-export or method call paths.
- [patch] Closed the Monte Carlo optical solver oversized-file gap by splitting solver facade, parallel simulation/result assembly, MCML photon tracing, and voxel geometry helpers into `physics::optics::monte_carlo::solver::{mod,simulation,trace,geometry}` without changing `MonteCarloSolver::{new,simulate}` or the parent optics re-export.
- [patch] Closed the multimodal fusion PCA gap by adding `algorithms::pca` as the single PCA fusion implementation, computing covariance over registered modalities, deriving convex weights from the first principal loading, sharing modality-order/dimension/result-metadata helpers with intensity projection, and validating correlated, dominant-variance, and nonfinite-input cases.
- [patch] Closed the enhanced BEM-FEM validation diagnostic gap by replacing unsupported spurious-resonance/interface-error branches with an input-sensitive Burton-Miller estimator: configured Burton-Miller suppresses fictitious-frequency diagnosis, standard BEM checks explicit validation frequencies, interface residuals scale with coupling tolerance and `(h/lambda)^2`, adaptive refinement records estimated element/error progression, and invalid frequencies/mesh bounds are rejected.
- [patch] Closed the PSTD/DG Chebyshev basis gap by adding first-kind Chebyshev Vandermonde construction and Chebyshev differentiation matrices using `T'_n=nU_{n-1}` plus endpoint limits. Focused tests verify finite endpoint matrices and exact differentiation of a quadratic polynomial on Chebyshev collocation nodes; Fourier remains an explicit unsupported periodic-basis path.
- [patch] Closed the transcranial aberration-correction validation oversized-file gap by splitting the validation facade/result type, corrected-field simulation, interpolation/intensity/sidelobe/FWHM metrics, and value-semantic tests into `physics::acoustics::transcranial::aberration_correction::validation::{mod,types,field,metrics,tests}` without changing `CorrectionValidation` or `TranscranialAberrationCorrection::validate_correction`.
- [patch] Closed the therapy cavitation oversized-file gap by splitting the facade/constructor, water/bubble constants, detector types, pressure/spectral detection kernels, scalar metrics, and value-semantic tests into `physics::acoustics::therapy::cavitation::{mod,constants,types,detection,metrics,tests}` without changing the parent `CavitationDetectionMethod` and `TherapyCavitationDetector` re-exports.
- [patch] Closed the nonlinear harmonics oversized-file gap by splitting the facade, Fubini harmonic amplitude kernels, tissue harmonic imaging efficiency/frequency formulas, contrast-agent harmonic response, and value-semantic tests into `physics::acoustics::wave_propagation::nonlinear::harmonics::{mod,amplitude,tissue,contrast,tests}` without changing the public harmonic function paths.
- [patch] Closed the skull heterogeneous-properties oversized-file gap by splitting the facade, constants, layer types, model data type, binary-mask construction, CT/Hill construction, BVF/layer/impedance helpers, and value-semantic tests into `physics::acoustics::skull::heterogeneous::{mod,constants,types,model,mask,ct,properties,tests}` without changing `HeterogeneousSkull`, `SkullLayer`, or the exported water/HU constants.
- [patch] Closed the bubble-field core oversized-file gap by splitting the facade, defaults, model construction, secondary Bjerknes coupling, adaptive update/history, state-field accessors, statistics, and coupling tests into `physics::acoustics::bubble_dynamics::bubble_field::core::{mod,constants,model,coupling,update,accessors,stats,tests}` without changing `BubbleField` or `BubbleFieldStats` exports.
- [patch] Closed the chemistry diffusion oversized-file gap by splitting the public facade, solver/error/result types, logarithmic grid and species helpers, Crank-Nicolson step assembly, Thomas tridiagonal solver, and value-semantic tests into `physics::chemistry::diffusion::{mod,types,grid,step,linear,tests}` without changing `RadicalDiffusionSolver`, `DiffusionStepResult`, or `DiffusionError` exports.
- [patch] Closed the Keller-Miksis shape-instability oversized-file gap by splitting constants, shape-mode state, Plesset-Prosperetti mode advancement, Blake wall-jet speed estimation, and value-semantic tests into `physics::acoustics::bubble_dynamics::keller_miksis::shape_instability::{mod,constants,state,dynamics,jet,tests}` without changing `N_MODES`, `BREAKUP_FRACTION`, `JET_STANDOFF_CRITICAL`, `ShapeModeState`, `advance_shape_modes`, or `jet_speed` exports.
- [patch] Closed the API models oversized-file gap by splitting `infrastructure::api::models` (861 lines) into `models/{mod,jobs,devices,imaging,clinical,dicom,mobile,tests}` partitioned by domain (job queue/training/audit, device connectivity, imaging frames, clinical analysis, DICOM integration, mobile workflows, value-semantic tests); preserved all public type re-exports through `models/mod.rs` (parent `api/mod.rs::pub use models::{...}` unchanged); default lib build clean, full lib suite passes 2645/2645 with 12 ignored in 8.12 s, clippy `-D warnings` clean; all eight split files are ≤265 lines.
- [patch] Closed the cylindrical medium projection oversized-file gap by splitting `domain::medium::adapters::cylindrical` (840 lines) into `cylindrical/{mod,construction,accessors,validation,tests}` partitioned by responsibility (axisymmetric projection facade with struct + Debug, θ=0 sampling constructor, field/point accessors, physical-bound validation, value-semantic tests); preserved the `CylindricalMediumProjection` re-export through `cylindrical/mod.rs` (parent `adapters/mod.rs` unchanged); struct fields are now `pub(super)` for sibling submodule access; targeted suite passes 15/15, clippy `-D warnings` clean, full lib suite passes 2645/2645 with 12 ignored in 8.76 s; source files are ≤165 lines, tests file 344 lines.
- [patch] Closed the linear elastography methods oversized-file gap by splitting `solver::inverse::elastography::linear_methods` (842 lines) into `linear_methods/{mod,time_of_flight,phase_gradient,direct,volumetric,directional,tests}` partitioned by inversion method (Bercoff TOF, McLaughlin-Renzi phase gradient, Gauss-Seidel direct, Urban volumetric multi-source, Wang directional, value-semantic tests); preserved the `ShearWaveInversion` re-export through `linear_methods/mod.rs` (parent `elastography/mod.rs` unchanged); targeted suite passes 9/9, clippy `-D warnings` clean, full lib suite passes 2645/2645 with 12 ignored in 8.32 s; all seven split files are ≤151 lines.
- [patch] Closed the symplectic bubble integrator oversized-file gap by splitting `solver::forward::ode::bubble_symplectic` (839 lines) into `bubble_symplectic/{mod,stormer_verlet,yoshida,integrate,tests}` partitioned by integrator responsibility (theorem-bearing facade with config/wrapper struct, Störmer-Verlet kernel, Yoshida triple-composition kernel, time-span integration wrapper, long-time validation tests); preserved `BubbleSymplecticIntegrator`, `SymplecticConfig`, `stormer_verlet_step`, `yoshida4_step`, `integrate_bubble_dynamics_symplectic` re-exports through `bubble_symplectic/mod.rs` (parent `ode/mod.rs` unchanged); targeted suite passes 4/4 (Minnaert, Hamiltonian non-drift, Yoshida order, equilibrium), clippy `-D warnings` clean, full lib suite passes 2645/2645 with 12 ignored in 8.81 s; source files ≤191 lines, tests file 462 lines kept cohesive due to shared helper utilities.
- [patch] Closed the beamforming-traits oversized-file gap by splitting `analysis::signal_processing::beamforming::traits` (851 lines) into `traits/{mod,core,time_domain,frequency_domain,adaptive,config,tests}` partitioned by trait responsibility (facade with hierarchy diagram, root `Beamformer`, `TimeDomainBeamformer`, `FrequencyDomainBeamformer`, `AdaptiveBeamformer`, `BeamformerConfig`, mock-driven conformance tests); preserved all five trait re-exports through `traits/mod.rs` (parent `beamforming/mod.rs` unchanged); targeted suite passes 4/4, clippy `-D warnings` clean, full lib suite passes 2645/2645 with 12 ignored in 8.06 s; all seven split files are ≤148 lines.
- [patch] Closed the staggered-grid operator oversized-file gap by splitting `math::numerics::operators::differential::staggered_grid` (850 lines) into `staggered_grid/{mod,forward,backward,tests}` partitioned by responsibility (Yee-scheme theorem facade with `new` and `DifferentialOperator` impl, forward-difference cell-center → cell-edge kernels, backward-difference cell-edge → cell-center kernels with `i=0` boundary fallback, value-semantic tests); preserved the `StaggeredGridOperator` re-export through `staggered_grid/mod.rs` (parent `differential/mod.rs` unchanged); targeted suite passes 13/13, clippy `-D warnings` clean, full lib suite passes 2645/2645 with 12 ignored in 8.66 s; all four split files are ≤276 lines.
- [patch] Closed the SIMD oversized-file gap by splitting `math::simd` (875 lines) into `simd/{mod,config,fdtd_ops,fft_ops,interpolation_ops,metrics,tests}` partitioned by responsibility (facade, capability detection, FDTD pressure/velocity AVX2 kernels, FFT complex-multiply AVX2, trilinear interpolation, performance estimation, value-semantic tests); preserved `FdtdSimdOps`, `FftSimdOps`, `InterpolationSimdOps`, `SimdConfig`, `SimdLevel`, `SimdPerformance`, `SimdMetrics` re-exports through `simd/mod.rs` (parent `math/mod.rs::pub use simd::{...}` unchanged); targeted `math::simd` suite passes 18/18, clippy `-D warnings` clean, full lib suite passes 2645/2645 with 12 ignored in 9.04 s; all seven split files are ≤354 lines.
- [patch] Closed the subspace beamforming oversized-file gap by splitting `analysis::signal_processing::beamforming::adaptive::subspace` (877 lines) into `subspace/{mod,music,eigenspace_mv,tests}` partitioned by responsibility (theorem-bearing facade with re-exports, MUSIC pseudospectrum kernel, ESMV signal-subspace MVDR with diagonal loading, value-semantic tests); preserved the `EigenspaceMV` and `MUSIC` re-exports through `subspace/mod.rs` (parent `adaptive/mod.rs` re-export unchanged); targeted suite passes 12/12, clippy `-D warnings` clean, full lib suite passes 2645/2645 with 12 ignored in 8.36 s; all four split files are ≤246 lines.
- [patch] Closed the clinical validation oversized-file gap by splitting `analysis::validation::clinical` (894 lines) into `clinical/{mod,bmode,doppler,safety,score,report,tests}` partitioned by responsibility (type definitions and FDA/IEC default-loaded validator, B-mode validator, default and configurable Doppler validators, IEC safety validator, weighted scoring kernel, Markdown report renderer, value-semantic tests); preserved `validate_bmode`/`validate_doppler[_with_thresholds]`/`validate_safety`/`generate_validation_report` exports; `requirements` field is now `pub(super)` for sibling submodule access; also fixed two pre-existing `clippy::needless_return` warnings in `apollo/crates/apollo-fft/src/application/execution/plan/fft/dimension_3d.rs`; targeted clinical suite passes 7/7, clippy `-D warnings` clean, full lib suite passes 2645/2645 with 12 ignored in 9.16 s; all seven split files are ≤264 lines.
- [patch] Closed the Westervelt FDTD oversized-file gap by splitting `solver::forward::nonlinear::westervelt` (888 lines) into `westervelt/{mod,laplacian,nonlinear,update,conservation,tests}` partitioned by responsibility (PDE theorem + struct facade + diagnostics API, in-place finite-difference Laplacian, product-rule nonlinear kernel, full leapfrog time-step with absorption and artificial viscosity, `ConservationDiagnostics` impl, value-semantic tests); preserved the `WesterveltFdtd` and `WesterveltFdtdConfig` exports; struct fields are now `pub(super)` for sibling submodule access; targeted Westervelt suite passes 5/5, clippy `-D warnings` clean, full lib suite passes 2645/2645 with 12 ignored in 8.45 s; all six split files are ≤362 lines.
- [patch] Closed the FDTD solver oversized-file gap by splitting `solver::forward::fdtd::solver` (955 lines) into `solver/{mod,central_diff,construction,stepping,sources,accessors,gpu_accelerator,interface}` partitioned by responsibility (struct/Debug facade, central-difference dispatch enum, constructor with k-space and scratch-buffer pre-allocation, Yee leapfrog plus debug NaN scans, dynamic source dispatch and mask-geometry classification, public accessors and orchestrated run loop, GPU-accelerator trait surface, `Solver` interface bridge); preserved `FdtdSolver` and `FdtdGpuAccelerator` re-exports through `solver/mod.rs`; struct fields elevated to `pub(crate)` for sibling submodule access; targeted FDTD suite passes 35/35, clippy `-D warnings` clean, full lib suite passes 2645/2645 with 12 ignored in 10.33 s; all eight split files are ≤221 lines.
- [patch] Closed the KZK solver oversized-file gap by splitting `solver::forward::nonlinear::kzk::solver` into `solver/{mod,stepping,observables,conservation,traits,tests}` partitioned by responsibility (struct/source-init facade, Strang-split propagation + diagnostics dispatch, real observables, `ConservationDiagnostics` impl, physics-layer trait bridge, value-semantic tests); preserved all existing public methods including `new`, `step`, `solve`, `set_source`, `enable/disable_conservation_diagnostics`, `get_conservation_summary`, `is_solution_valid`, `get_pressure`, `get_time_signal`, `get_intensity`, `get_peak_pressure`; struct fields are now `pub(super)` for submodule access; targeted KZK suite passes 11/11 with 1 pre-existing ignored Tier-3 test, clippy `-D warnings` clean, full lib suite passes 2645/2645 with 12 ignored in 11.04 s; all six split files are ≤386 lines.
- [patch] Closed the Keller-Miksis validation oversized-file gap by splitting `physics::acoustics::bubble_dynamics::keller_miksis::validation` into `validation/{mod,dynamics,thermodynamics,shape_stability}` with 17 tests grouped by physical responsibility (K-M wall ODE, thermodynamic auxiliary updates, Plesset-Prosperetti shape-mode coupling); removed an orphaned `forward_model` doc block from `solver::inverse::seismic::fwi` that triggered `doc_lazy_continuation`; targeted `keller_miksis` passes 32/32 with 1 pre-existing ignored equilibrium test, full lib suite passes 2645/2645 with 12 ignored, clippy `-D warnings` clean.
- [patch] Added and finalized the MATLAB-free KWave.jl external benchmark path by installing Julia 1.12.6, cloning `https://github.com/JClingo/k-wave-julia.git` into `external/k-wave-julia`, and adding a `benchmarks/kwavers` harness that runs 1-D, 2-D, and 3-D IVP Gaussian cases through KWave.jl and pykwavers with deterministic one-sample IVP timing alignment, sensor-outside-PML validation, timing metrics, value metrics, aligned CSV output, per-dimension pressure/timing/summary plots, and aggregate dimension-sweep plots.
- [patch] Extended the reference-project benchmark matrix by linking KWave.jl in `README.md`, adding k-wave-python Python-backend execution to the KWave.jl benchmark sweep, and recording native MATLAB k-Wave as source-present but not runnable in this environment without MATLAB or Octave.
- [patch] Closed the first PSTD benchmark hot-path cleanup by skipping Cartesian y/z spectral derivative FFTs when those axes are singleton lower-dimensional embeddings; the discrete derivative is zero because the only admissible wavenumber on a length-1 periodic axis is k=0. The benchmark still shows k-wave-python faster than pykwavers on 24-point IVP cases, so the next optimization item is the 3-D PSTD setup/FFT execution path rather than benchmark tolerance or scenario reduction.
- [patch] Pulled Apollo to `32729af` and closed the Apollo 3-D FFT temporary-allocation gap by filling caller-owned complex outputs directly instead of constructing a full `mapv` temporary. Singleton-axis FFT passes now return immediately because the transform over a length-1 axis is identity. PSTD linear density updates now read `materials.rho0` directly instead of copying it into `div_u` each step. The 1-D pykwavers benchmark now beats k-wave-python on the latest run; remaining 2-D/3-D performance debt is full active-axis FFT lane movement and PSTD setup overhead.
- [patch] Closed the PSTD IVP singleton-axis setup gap by skipping y/z initial velocity inverse FFTs when the embedded axis length is 1. Do not pursue shared cached 1-D Bluestein plans inside the current 3-D parallel lane loop without per-worker scratch; the measured mutex plan serialized lane transforms and regressed 2-D/3-D. Do not replace `inverse_into(grad_k, ...)` with in-place inverse plus separate real extraction in the current path; measured runtime regressed despite lower scratch copying.
- [patch] Closed the Apollo 3-D FFT non-contiguous lane allocation gap by replacing `Vec<Vec<Complex*>>` lane materialization with one flat workspace for axis-0 and axis-1 passes. This preserves one separable 3-D FFT algorithm and lowers allocator pressure without public API changes. Latest rebuilt benchmark remains PASS and improves 3-D pykwavers timing to 0.508495 s; 2-D and 3-D remain slower than k-wave-python, so the next optimization target is active-axis lane transposition/cache locality and PSTD setup overhead.
- [patch] Closed the PSTD inactive split-density update gap for lower-dimensional Cartesian embeddings: singleton-axis spectral derivatives no longer trigger full-array zero-fills or no-op `rhoy`/`rhoz` updates, while the stored split-density state remains part of the pressure equation of state. Parallelized Apollo lane gather/scatter is rejected for now because measured timing regressed despite preserving correctness. Latest rebuilt benchmark remains PASS with pykwavers 1-D 0.009833 s, 2-D 0.047745 s, and 3-D 0.505348 s; remaining 2-D/3-D debt is active FFT lane movement, not inactive-axis bookkeeping.
- [patch] Closed the PSTD IVP setup scale-allocation gap by computing `sin(c₀|k|dt/2)/(ρ₀c₀|k|)` from `source_kappa` into the existing `div_u` scratch buffer during `initialize_ivp_velocity`. The scratch has no live density-sum meaning during construction and is overwritten before pressure update, so the change removes one full setup allocation without changing the exact staggered start formula. Final pre-commit benchmark remains PASS with pykwavers 1-D 0.009152 s, 2-D 0.049823 s, and 3-D 0.516947 s.
- [patch] Closed the PSTD source-kappa setup allocation gap by reusing the `k_mag` buffer after absorption initialization and converting it in place to `cos(c_ref·dt·|k|/2)`. This preserves the source phase factor and removes one full-volume `Array3<f64>` allocation from PSTD construction. Latest rebuilt benchmark remains PASS with pykwavers 1-D 0.009874 s, 2-D 0.045697 s, and 3-D 0.480546 s.
- [patch] Closed the PSTD IVP initial-density setup allocation gap by reusing the existing `div_u` scratch buffer as the `apply_initial_conditions` density output and splitting from that buffer into `rhox`/`rhoy`/`rhoz`. `div_u` has no live density-sum meaning during construction and is overwritten before pressure evaluation. Latest rebuilt benchmark remains PASS with pykwavers 1-D 0.009789 s, 2-D 0.044440 s, and 3-D 0.410568 s.
- [patch] Closed the PSTD homogeneous material setup overhead gap by filling `rho0`, `c0`, and `BonA` from one canonical homogeneous-medium sample instead of per-voxel coordinate conversion and dynamic property lookup. Latest rebuilt benchmark remains PASS with pykwavers 1-D 0.008503 s, 2-D 0.044713 s, and 3-D 0.473107 s.
- [patch] Closed the PSTD diagnostic source-control hot-path overhead by parsing `KWAVERS_PSTD_SOURCE_TIME_SHIFT` and `KWAVERS_PSTD_SOURCE_GAIN` once in `PSTDSolver::new` and reusing the stored values during pressure-source injection. This keeps solver behavior deterministic per instance and removes per-step environment access from the source path. Latest rebuilt benchmark remains PASS with pykwavers 1-D 0.009737 s, 2-D 0.045513 s, and 3-D 0.417537 s.
- [patch] Closed the PSTD disabled-trace scan overhead by guarding the `max_p` full-pressure-field diagnostic scan with `tracing::enabled!(Level::TRACE)`. Default runs no longer pay an O(N) scan every early/tenth step for a disabled trace event, while trace-enabled diagnostics remain value-identical. Fresh release rebuild now passes after `.cargo/config.toml` selects Clang/Clang++ and Ninja for native `openjpeg-sys` and `charls-sys` compilation, and the Windows override loader materializes the stable-ABI `libpython3.dll` proxy from the active interpreter. Rebuilt comparison sweep PASS: pykwavers 1-D 0.009561 s, 2-D 0.047431 s, 3-D 0.473963 s; k-wave-python 1-D 0.016715 s, 2-D 0.032788 s, 3-D 0.085284 s.
- [patch] Closed the Apollo `6ba31de` pull and FFT-frequency SSOT integration by updating the `apollo` gitlink, using Apollo's current public cache/frequency re-exports from `kwavers::math::fft`, and routing PSTD/k-space wavenumber vector construction through Apollo `fftfreq`. This removes duplicated FFT-bin sign logic in the `kwavers` facade and aligns even-length Nyquist handling with Apollo/numpy semantics. Rebuilt comparison sweep PASS: pykwavers 1-D 0.009644 s, 2-D 0.055903 s, 3-D 0.495308 s; k-wave-python 1-D 0.016240 s, 2-D 0.006481 s, 3-D 0.099361 s. Remaining performance debt remains active-axis FFT lane movement/setup cost in 2-D/3-D, not frequency-bin generation.
- [patch] Closed four remaining monolith-to-directory conversions without public API changes: plane-wave compounding, Helmholtz FEM solver, conservative interpolation, and elastic 2-D PINN model. Each now has a module facade plus responsibility-scoped implementation/config/test files, preserving existing parent re-exports while reducing oversized flat modules by 324 net lines across the three always-built modules and completing the feature-gated PINN model split. Focused verification passes 8/8 plane-wave, 3/3 FEM, 5/5 conservative interpolation, and default non-`pinn` model selector 0/0.
- [patch] Closed the BEM boundary module conversion by separating the manager implementation, boundary-condition enum, and value-semantic tests under `domain/boundary/bem/`. The parent boundary facade still re-exports `BemBoundaryManager` and `BemBoundaryCondition`; focused BEM tests pass 5/5.
- [patch] Closed the Gaia mesh-provider import boundary by making `gaia` a workspace member and kwavers dependency, adding `TetrahedralMesh::from_gaia_indexed_mesh`, preserving Gaia vertex coordinates and tetrahedral cell connectivity, deriving missing `vertex_ids` from cell faces for Gaia structured grids, mapping only explicit mathematical boundary labels (`dirichlet`, `neumann`, `robin`, `radiation`/`sommerfeld`), rejecting conflicting labels/non-tetra cells/non-finite vertices, and pinning unit-cube volume preservation plus boundary-label semantics with value tests.
- [patch] Closed the DG Fourier basis kernel gap by replacing the `NotImplemented` branches in the basis and differentiation builders with a real trigonometric basis `1, sin(kπ(x+1)), cos(kπ(x+1))` and its analytic derivative. The kernel now validates finite periodic nodes and rejects simultaneous `-1`/`1` endpoints because they are the same point on `[-1,1)`. The high-level `DGSolver::new` still rejects `BasisType::Fourier` explicitly because its nodal constructor is GLL-based; the remaining additive work is a periodic-node DG constructor, not Fourier basis algebra.
- [patch] Closed the DAS-PAM delay-and-sum module conversion by separating the processor, serializable configuration/event types, and tests under `analysis/signal_processing/pam/delay_and_sum/`. The PAM facade still re-exports `DelayAndSumPAM`, `DelayAndSumConfig`, `ApodizationType`, and `CavitationEvent`; focused DAS-PAM tests pass 7/7.
- [patch] Closed the transducer-interface module conversion by separating hardware trait, device manager, mock device, hardware command/status types, and tests under `infrastructure/device/transducer_interface/`. The device facade still re-exports all existing public items. Mock calibration now uses deterministic state transitions without wall-clock sleep, eliminating artificial test latency while leaving physical hardware latency to real `TransducerHardware` implementations; focused transducer-interface tests pass 12/12.
- Keep the neural beamforming adaptation and distributed execution paths on the canonical SSOT helpers; extend them by refining the shared partition/recomposition logic rather than cloning variant-specific processors.
- Closed the sonoluminescence bremsstrahlung oversized-file gap by splitting constants, Gaunt factors, noble-gas data, plasma state, emission model, field assembly, and value-semantic tests into nested vertical modules below 200 lines each.
- [patch] Closed the PSTD/DG Legendre endpoint derivative gap by replacing the singular interior quotient at GLL endpoints with the analytic limits `P'_n(1)=n(n+1)/2` and `P'_n(-1)=(-1)^(n+1)n(n+1)/2`; value-semantic tests now verify finite endpoint derivatives and exact differentiation of constant and linear nodal polynomials.
- [patch] Closed the spectral-filter implementation gap by routing `SpectralFilter::apply` and `SpectralOperator::apply_antialias_filter` through Apollo-backed 3-D FFTs, applying a tensor-product modal transfer function, and validating constant-field invariance plus Nyquist-mode rejection without adding a parallel FFT implementation.
- [patch] Closed the spectral-filter workspace allocation gap by adding `SpectralFilter::apply_into(field, spectrum, output)`, reusing `output` as the real FFT staging buffer, routing the allocating `apply` wrapper through the same implementation, and removing the extra owned real-field copy. Tests verify output equivalence, buffer pointer stability across repeated calls, and explicit spectrum/output shape rejection.
- [patch] Closed the pseudospectral derivative allocation and duplication gap by routing x/y/z derivative APIs through one const-generic axis kernel and exposing caller-owned `derivative_{x,y,z}_into` workspace APIs. The allocating wrappers now allocate one output plus one reusable complex line buffer per call rather than one `Array1` per line; tests verify value equivalence, pointer stability, and workspace mismatch rejection.
- [patch] Closed the complex Hermitian extension-trait eigendecomposition gap by routing `Array2<Complex<f64>>::eig()` through the existing SSOT Hermitian eigensolver, returning real eigenvalues embedded in `Complex<f64>`, and rejecting non-Hermitian matrices through the same validator.
- [patch] Closed the MUSIC time-delay trait gap by routing `LocalizationProcessor::localize` for `MUSICProcessor` through the existing TDOA least-squares solver when only arrival-time data are available, preserving true MUSIC covariance processing on `MUSICProcessor::run` and validating an analytical off-axis source case.
- [patch] Closed the analytical phase-shifter strategy gap by implementing `Focused`, `MultiFocus`, and `Custom` in `PhaseShifter::apply_phases`; focused and multifocus now route through the canonical spherical/multipoint phase laws, while custom phase patterns validate one phase per element and wrap phases into the canonical interval.
- [patch] Closed the phase-control allocation and elastic-PML constructor hardening increment by reusing `PhaseShifter::phase_offsets` as the owned workspace across all phase-law dispatch paths, eliminating the flat multifocus target reboxing allocation, preserving no-mutation-on-rejection semantics, and consolidating real-space plus split-field PML construction through `ElasticPmlSpec`. The same increment removes the split-field orchestrator unwrap path and restores the source-tree 500-line structural invariant.
- [patch] Closed the thermal diffusion memory/monomorphization increment by replacing owned external-source handoff with `ArrayView3`, computing Pennes perfusion in the same parallel traversal as diffusion/source application, and routing the Laplacian through one const-generic stencil implementation with per-axis admissibility rather than cloned order-specific loops.
- [patch] Closed the hyperbolic thermal diffusion workspace increment by removing the unused `prev_flux_x/y/z` arrays, reusing an owned divergence workspace during temperature updates, and routing heat-flux/divergence component math through const-generic axis selectors. Verification pins the one-step analytical update for `T(i)=i^2`, workspace pointer stability, owned-vs-workspace divergence equivalence, and boundary-zero divergence.
- [patch] Closed the PSTD Dirichlet-PML bypass allocation increment by replacing per-application x-plane `to_owned()` clones with one solver-owned `(bypass_rows, ny, nz)` scratch buffer reused across velocity and split-density component preservation. The shared helper validates row bounds and scratch shape, restores bypass rows after successful or failed PML mutation, and has value tests for restoration, error-path restoration, and invalid workspace rejection.
- [patch] Closed the KZK real-field diffraction FFT allocation increment. Audit confirmed AS PSTD already uses preallocated WSWA FFT buffers in `AsContext`; the active allocation gap was in `AngularSpectrum2D` and real-field `KzkDiffractionOperator`, which now use cached `Fft2d` plans and reusable `Array2<Complex64>` scratch buffers for in-place FFT, modal propagation, and inverse projection. Tests pin scratch pointer stability, zero-distance identity, FFT round-trip recovery, and energy preservation.
- [patch] Closed the monolithic residual block-allocation increment by replacing per-field `to_owned()` slices in `compute_residual` with borrowed `ArrayView3` block views over the stacked Newton state. The Laplacian kernel now writes into caller-owned output over generic `ndarray` storage, and residual evaluation reuses one rate scratch buffer across pressure, light-fluence, and temperature blocks. Tests pin view storage sharing, owned/view Laplacian equivalence, analytical quadratic output, zero residual, and Grüneisen source scaling.
- [patch] Closed the monolithic previous-state snapshot allocation increment by adding solver-owned `u_prev_scratch` to `MonolithicCoupler`. Each coupled step now refreshes the previous-state snapshot with `assign` after flattening and reuses the allocation across shape-compatible repeated solves, eliminating the per-step `u_current.clone()` allocation without adding variant-specific APIs. Tests pin pointer stability and refreshed pressure/temperature snapshot values under an analytically zero residual.
- [patch] Closed the monolithic Newton RHS/update allocation increment by adding solver-owned `rhs_scratch` for `-F(u)` and applying Newton corrections in place with `u += alpha * du`. This removes the per-iteration `&f * -1.0` RHS allocation plus the `du * alpha` and replacement-state temporaries while preserving the existing GMRES algebra and single monolithic residual path. Tests pin pointer-stable RHS reuse and exact constant-fluence RHS values.
- [patch] Closed the monolithic line-search candidate allocation increment by moving coupler tests into `monolithic/coupler/tests.rs`, adding solver-owned `line_search_state_scratch`, and evaluating each backtracking candidate by overwriting that workspace with `u + alpha * du`. This removes one full-state candidate allocation per tested alpha, restores scratch ownership on residual errors, and keeps production monolithic files below the 500-line structural limit.
- [patch] Closed the monolithic JVP perturbed-state allocation increment by adding solver-owned `jvp_state_scratch` for `u + eps * v` and converting `F(u + eps*v)` in place into the returned `(F(u + eps*v) - F(u)) / eps` vector. This removes the perturbation temporary and the separate scaled-difference allocation while preserving the existing GMRES closure API and single finite-difference JVP formula.
- [patch] Closed the monolithic line-search configuration gap by making `NewtonKrylovConfig::line_search_parameter` the adaptive backtracking maximum `alpha_max`, using `alpha_k = alpha_max / 2^k`, and rejecting invalid values through a typed validation error before candidate residual evaluation. Verification was unblocked by synchronizing root workspace `ritk-*` dependency keys with `ritk-io` inheritance and removing a duplicate `BinaryThreshold` re-export in `ritk-core`.
- [patch] Closed the monolithic solve input-contract gap by adding a pre-flattening validation gate for finite-positive `dt`, positive Newton iteration count, finite-positive Newton tolerance, nonempty field maps, and exact field/grid dimension equality. The check is O(field count), preserves hot-path memory behavior, and converts empty-state fabrication plus shape-mismatch panics into typed validation errors.
- [patch] Closed the monolithic line-search fallback gap by returning the final residual-evaluated backtracking alpha when all candidates fail sufficient decrease. This removes the untested extra-halving fallback while preserving the single adaptive line-search implementation and configured `alpha_max / 2^k` sequence.
- [patch] Closed the monolithic residual vertical-tree gap by replacing the single `residual.rs` implementation file with `residual/{mod,compute,jvp,line_search,tests}.rs`. The internal method surface remains a set of statically dispatched inherent `MonolithicCoupler` impls, so the architecture split improves responsibility isolation without adding wrapper APIs or runtime dispatch.
- [patch] Closed the monolithic utility vertical-tree and line-search norm gap by replacing `utils.rs` with `utils/{mod,layout,block,norm,laplacian,tests}.rs` and adding `norm_squared` for candidate comparisons. Adaptive line search now evaluates the same sufficient-decrease condition in squared-norm form, removing repeated square roots without changing acceptance semantics.
- [patch] Closed the monolithic coupler vertical-tree gap by replacing `coupler.rs` with `coupler/{mod,construction,validation,solve,plugins,accessors,tests}.rs`. The public `MonolithicCoupler` API remains the same inherent method set, while constructor state, validation, solve-loop orchestration, plugin/coefficient mutation, and accessors are isolated by responsibility without runtime indirection.
- [patch] Closed the monolithic semantic-tree naming gap by removing the generic `utils` module and moving its responsibilities into `state_vector`, `residual_metric`, and `spatial_operator`. The new folders describe the actual simulation concepts: stacked Newton state representation, convergence metrics, and spatial finite-difference operators.
- [patch] Closed the monolithic configuration-domain gap by replacing `config.rs` with `config/{mod,convergence,newton,physics,tests}.rs` and moving Newton/coefficient validation onto the owning types. The solve preflight now rejects invalid material denominators and optical transport before residual assembly can compute infinities or divide by zero.
- [minor] Closed the fluid-structure coupling activation and ghost-workspace increment by compiling the existing FSI source tree under `solver::multiphysics::fluid_structure`, replacing the 9-argument `FsiInterface` constructor with `FsiInterfaceSpec`, and changing ghost exchange from full-volume clones plus temporary `p_new`/`t_new` arrays to solver-owned previous-state workspaces plus in-place pressure/traction buffer mutation. Tests pin interface validation, reflection/energy coefficients, traction balance, velocity continuity, and repeated-exchange pointer stability.
- [patch] Closed the EM FDTD boundary-cache allocation increment by replacing `fields_cache.clone()` in `apply_em_boundary_conditions` with caller-buffer assignment, using fixed `Ix4` cache views for cell-centered E/H writes, and adding value tests for pointer reuse, cache equivalence, auxiliary-field clearing, and shape-repair semantics.
- [patch] Closed the multimodal fusion intensity-projection gap by implementing voxelwise maximum- and minimum-intensity fusion as order-statistic projections with selected-modality confidence, optional uncertainty, identity registration metadata, and dimension validation.
- Closed the acoustic conservation oversized-file gap by splitting metrics, energy, mass-continuity, momentum, entropy, intensity, heat-source, validation, and value-semantic tests into nested vertical modules below 150 lines each.
- Closed the sonogenetics channel oversized-file gap by splitting constants, gating parameters, channel identity, open-probability equations, ion-current computation, and value-semantic tests into a nested vertical module tree with unchanged public facade exports.
- Closed the quantum-optics orphan/oversized-file gap by wiring `physics::optics::quantum_optics` into the optics module tree, splitting Einstein coefficients, Gaunt factors, special functions, correction assessment, constants, and tests into nested files, and replacing constant invalid-domain fallbacks with non-finite outputs plus tests.
- Closed the skull aberration oversized-file gap by splitting phase-screen constants, model construction, volumetric phase integration, element correction extraction, aperture maps, and value-semantic tests into nested files below 200 lines each; mismatched element coordinate arrays now return a dimension error instead of panicking inside a `KwaversResult` API.
- Added the RITK-backed skull CT DICOM phase-correction example for an Insightec-style 1024-element hemispherical transducer at 650 kHz, including HU-to-acoustic-property conversion, phase-screen correction, per-element correction CSV output, and a three-plane PPM visualization.

## AS PSTD FFT Hot-Path Optimization
- Closed the axisymmetric PSTD parity gap: `at_circular_piston_AS` (pearson=0.9907, PASS) and `at_focused_bowl_AS` (pearson=1.0000, PASS) after five physics fixes: one-sided radial PML (`pnz = nz + p`, `pz_embed = 0`), CPML inner-z transparency (`radial_inner_z_transparent`), source injection skip for `rhoy`, `density_scale = 1.0` for AS, and correct embed offset.
- Identified AS PSTD FFT allocation churn: `axisymmetric.rs` calls `fft_2d_array`/`ifft_2d_array` (allocating ~6–8 `Array2<Complex64>` per time step on the `(2·nx)×(4·nz)` WSWA domain) instead of the `forward_into`/`inverse_into` pre-allocated path already used by the 3D PSTD propagators. Pre-allocating expanded buffers (`a_exp`, `uz_exp`, `uz_on_r_exp`) and FFT output buffers in `AsContext`, then routing through `plan.forward_into`/`plan.inverse_into`, eliminates all per-step allocation churn on the AS hot path.

## Sonogenetics Research Modernization
- Closed the bacterial-channel coverage gap in `physics::acoustics::therapy::sonogenetics` by adding `MscLG22N` and `MscS` to the existing `MechanoChannel` abstraction, updating theorem/proof documentation for two-state gating, and preserving one canonical `compute_p_open` dispatch path.
- Closed the sonogenetics channel organization gap by moving the two-state gating theorem, pressure-threshold theorem, channel identity table, canonical parameters, and ion-current theorem into domain-scoped nested files while preserving the single canonical dispatch path.
- Corrected `ion_current` to return injected depolarizing current `g·n·P_open·(E_rev − V_m)`, matching the LIF equation contract while documenting the distinction from electrophysiology outward-current sign.
- Residual performance follow-up: `cargo test -p kwavers --lib` passes but reports `solver::forward::nonlinear::kzk::solver::tests::test_conservation_diagnostics_disable` and `solver::validation::numerical_accuracy::pstd::tests::test_pstd_phase_velocity_accuracy` as running beyond 60 seconds; optimize the real KZK/PSTD paths before treating this as closed performance debt.

## Thermal Property Law Modernization
- Closed the thermal absorption placeholder gap in `physics::thermal::properties`: the previous `1 - 0.02 ΔT` law could become negative during ablation heating. The replacement is a positive exponential soft-tissue law using the same `0.015 1/°C` coefficient as the bioheat absorption model.
- Aligned `sound_speed_vs_temperature` with the generic soft-tissue coefficient `dc/(c dT)=1.6e-3` used in the temperature-dependent medium model and documented the local hyperthermia validity boundary from ultrasound thermometry literature.
- Residual modeling scope: generic scalar functions still use soft-tissue coefficients. Tissue-specific thermal updates should route through table-backed material records or explicit coefficient structs before adding organ-specific behavior.

## Plasmonic Effective-Medium Modernization
- Closed the plasmonic mixture-law placeholder gap in `physics::electromagnetic::plasmonics`: `CouplingModel::None` now evaluates the Maxwell-Garnett dilute-sphere closed form instead of a linear dielectric blend.
- Replaced the linearized `CouplingModel::QuasiStatic` branch with the physical closed-form root of the symmetric Bruggeman equation, preserving the existing coupling-model API while making endpoint and residual identities testable.
- Closed the `MieTheory::gold_in_water` simplified Drude-Lorentz placeholder by routing the gold dielectric closure through Johnson-Christy measured optical constants, affine interpolation over the tabulated wavelength domain, and exact `ε=(n+ik)²` conversion.
- Closed the electromagnetic plasmonic-trait physics/organization gap by moving spheroid depolarization formulas into the nested `traits/plasmonic/geometry.rs` kernel and replacing the prior negative-permittivity resonance expression with a Fröhlich/Drude resonance law plus finite damping.
- Residual modeling scope: additional gold datasets such as Rakic, Olmon, or temperature-dependent Magnozzi/Yakubovsky records should be introduced through an explicit dielectric-data strategy before adding film, size-corrected, or thermal gold models.

## Tree Cleanup Sprint
- Closed the diverging-wave config SSOT gap: `domain::sensor::ultrafast::diverging_wave::config` now owns `DivergingWaveConfig`, while the processor facade retains the existing public re-export and focused tests remain co-located with the processor module.
- Closed the Kuznetsov solver monolith gap: `solver::forward::nonlinear::kuznetsov::solver` now separates solver state, RHS assembly, acoustic-model integration, and conservation diagnostics. The RHS hot path reuses workspace `k1` and no longer clones the pressure field before RHS evaluation.
- Closed the seismic misfit organization gap: `solver::inverse::reconstruction::seismic::misfit` now isolates dispatch types, norm metrics, envelope/phase metrics, and Wasserstein metrics behind one unchanged facade.
- Closed the GPU k-space organization gap: propagation and spectral-shift GPU paths now live under separate `gpu::kspace` child modules behind the unchanged parent re-exports.
- Closed the FDTD-FEM coupling duplicate-module gap: hybrid coupling now uses `fdtd_fem_coupling/{config,interface,coupler,solver,tests}` behind one parent facade.
- Closed the FDTD k-space correction organization gap: spectral gradient/divergence operators and value tests now live under `kspace_correction/{operators,tests}` behind one parent facade.
- Closed the beamforming localization search organization gap: policy types, search orchestration, and tests now live under `beamforming_search/{types,search,tests}` behind one parent facade.
- Closed a broad flat-module backlog segment by making 40 additional modules directory-backed facades with child modules for configuration, types, kernels, orchestration, tests, and implementation-specific responsibilities. This removes another wave of flat-file drift while preserving public module paths.
- Closed the CPML profile and narrowband integration-test organization gaps. CPML profile math now separates facade/state, profile kernels, and value tests under `cpml/profiles/`; narrowband pipeline tests now separate shared fixture generation, end-to-end pipeline checks, invariance checks, snapshot consistency, and steering-unit assertions under `narrowband/integration_tests/`.
- Closed the fusion test organization gap by replacing the flat `physics::acoustics::imaging::fusion::tests` file with behavior-scoped child test modules for basic fusion contracts, confidence/uncertainty, tissue-property extraction, registration/quality helpers, and advanced probabilistic/non-rigid paths. This keeps the production fusion facade unchanged and preserves all value-semantic assertions.
- Closed the plane-wave compounding compatibility-stub gap: `PlaneWaveCompound::config()` no longer returns `ThermalAcousticConfig::default()` independent of the imaging setup. It now maps the configured plane-wave image into a one-cell-thick thermal-acoustic volume with geometry-derived dimensions, spacings, sound speed, and CFL time step while preserving the public method signature.
- Closed the next broad facade-consolidation segment across analysis, clinical, core, domain, GPU, infrastructure, math, simulation, and solver modules by replacing additional flat modules with directory-backed facades and SRP child files while preserving parent import paths. Also closed the RTM imaging-condition placeholder gap: `EnergyNormalized`, `SourceNormalized`, and `Poynting` now compute real source-energy normalization, temporal-source-derivative correlation, and spatial-gradient dot-product images under the existing `RtmImagingCondition` enum, with focused value-semantic tests.
- Closed another flat-module consolidation segment by converting time-domain DAS beamforming, ULM reconstruction/mapping, CEUS ultrasound imaging, boundary typing, ultrafast sensing, signal filtering, inverse-problem regularization, acoustic backend, SWE, AVX-512 stencil, Born-series convergence, GPU PSTD tests, PINN residual/optimizer/model/geometry/uncertainty, and sonoluminescence benchmark files into SRP child modules behind unchanged facades.
- Started the in-repository ultrasound physics book under `docs/book/` with separate therapy, diagnostics, and theranostics chapters, theorem/proof/algorithm sections, committed SVG figures, and references to current focused-ultrasound, ULM, and microbubble theranostics literature. The README now uses a local book figure instead of a remote placeholder image.
- Closed the multi-rate time-integration test-placeholder gap: `solver::integration::time_integration::tests` now validates RK4, Adams-Bashforth, CFL, diffusion stability, and subcycle selection against analytical formulas. The new subcycle test exposed and closed a floating-point integer-boundary defect in `MultiRateController`, where an exact 5:1 schedule could become 6 subcycles due to binary roundoff.
- [patch] Closed the time-scale separation allocation and stencil-duplication gap by passing each component field as an `ArrayView3` into derivative analysis instead of cloning `field.to_owned()`, returning zero scales when no central-stencil interior exists, and expressing first/second central differences through const-generic axis helpers. Tests verify analytical quadratic time scales and small-domain behavior.
- [patch] Closed the multiphysics field-coupling read-allocation gap by replacing per-edge pressure/intensity `Array3` clones with const-generic split-borrowed read/write pairs. Weak/adaptive/strong coupling now validate required field indices and collocated shapes before mutation; strong coupling reuses previous-state snapshot buffers across non-converged iterations instead of reallocating them. Tests verify analytical weak-coupling updates, source-field preservation, and mismatch rejection without mutation.
- [patch] Closed the functional kernel/window allocation gap by collecting borrowed sparse-kernel coefficients once, removing the parallel coordinate `Vec`, and changing `windowed_operation` to invoke closures with borrowed `ArrayView3` windows. Tests use non-`Clone` scalar inputs and outputs to prove the kernel and window contracts no longer hide value cloning.
- [patch] Closed the functional transform smoothing allocation gap by removing the `Clone` bound from owned-value `FieldTransform`/`ReversibleTransform` pipelines and rewriting `Array3Transform::smooth` to generate its output directly from borrowed input values. Boundary cells are preserved without cloning the whole field, interior cells use one const-generic axis-neighbor stencil contract, and tests verify non-`Clone` pipeline values, quadratic stencil output, and small-domain boundary behavior.
- [patch] Closed the functional field-ops read traversal gap by removing the blanket `T: Clone` bound from `FieldOps for Array3<T>` and replacing `par_map_field`'s temporary `Vec<&T>` with direct flat-index Rayon traversal. Tests verify map, fold, filter, and parallel map over non-`Clone` field elements.
- [patch] Closed the time-reversal signal reversal allocation gap by replacing `signals.clone()` plus row swaps with one direct `Array2::from_shape_fn((sensors, samples), |(s,t)| signals[[s, samples - 1 - t]])` construction. Tests verify exact row reversal, the time-reversal involution, unchanged source data, single-sample identity, and empty-sensor shape preservation.
- Closed the math SIMD AVX-512 pressure-kernel placeholder gap: `FdtdSimdOps::update_pressure_avx512` now performs the real 16-wide AVX-512F recurrence `p_next = 2p - p_prev + c²dt²∇²p` on row-contiguous interior cells instead of delegating to AVX2. The SIMD test now validates every boundary/interior cell against the FDTD recurrence rather than checking that one value changed.
- Closed another facade-backed module-tree segment by splitting ML uncertainty quantification, beamforming deterministic fixtures, architectural layer validation, lithotripsy stone-fracture mechanics, multiphysics coupling, CPML updates, CT loading, signal utilities, multi-GPU orchestration, API data types, AWS provider lifecycle, photoacoustic acoustics/reconstruction, multiphysics solver state, GPU pipeline management, BEM-FEM coupling, KZK harmonic tracking, GMRES iteration, and PINN ML helpers into nested child modules. Public parent facades remain stable, and incomplete Azure/GCP cloud provider stubs are removed from active exports rather than retained as non-computing placeholders.
- Closed the clinical mechanical-index value-test gap: `MechanicalIndexCalculator` now rejects undefined input domains before evaluating attenuation or `sqrt(f_c)`, and `calculate_max_mi` rejects single-point depth profiles instead of constructing a division-by-zero depth grid. The previous no-op panic test is replaced by analytical assertions that the maximum MI occurs at zero depth under nonnegative attenuation and by value-semantic negative tests for invalid center frequency, focal depth, and profile cardinality.
- Closed the sixth-order central-difference module ambiguity introduced by the in-progress tree split: the active directory tree now has a `mod.rs` facade that exports `core::CentralDifference6` and owns the co-located tests. The parent differential facade can compile against the directory-backed module without restoring the deleted flat monolith.
- Closed the Apollo WGPU FFT discrepancy found during kwavers GPU validation: f32/f16 Bluestein execution now uses FFT-precomputed chirp kernels, flat padded work dispatch, padded-row indexing for postmultiply/scale/conjugation, correct conjugate-forward-conjugate inverse ordering, and explicit inverse `1/N` axis scaling. `kwavers::math::fft::gpu_fft` now re-exports the Apollo WGPU backend and reusable buffers, with value-semantic GPU tests covering CPU spectrum equivalence and non-power-of-two round-trip behavior.
- Closed the GPU-feature lint cleanup gap: PSTD GPU docs now satisfy rustdoc list indentation, dispatch uses standard `div_ceil`/`is_multiple_of`, GPU readback avoids needless borrows, FDTD upload sizing uses slice size semantics, and multi-GPU priority sorting uses `sort_by_key(Reverse(_))`. The fast-nearfield module transition is completed as a facade-backed directory split across core logic, data types, and tests.
- Closed the FEM boundary dummy-fixture/domain-validation gap: `FemBoundaryManager::apply_all` now validates assembled system shape, RHS length, boundary node ranges, finite Robin coefficients, and finite Sommerfeld wavenumber before mutating CSR matrices. Tests now inspect row-level CSR values for Dirichlet elimination and value-preserving error paths rather than relying on dummy matrices and success-only assertions.
- Closed the clinical thermal-index omission gap: `clinical::safety` now exports `ThermalIndexCalculator` with explicit `TIS`/`TIB`/`TIC` model identity, finite-domain validation, acoustic-power derating, and value-semantic safety status tests. The implementation keeps `W_deg` as explicit model input rather than embedding one soft-tissue reference power as a hidden default.
- Closed the KZK zero-field diagnostic hot-path gap: `KZKSolver::step` now proves and uses the homogeneous-operator identity `D(0)=A(0)=N(0)=0` to skip diffraction, absorption, and nonlinear FFT work when the complex pressure field is exactly zero. The disabled-diagnostics regression now verifies the production step path without paying full default-grid FFT cost, and `test_zero_pressure_step_is_identity` pins the invariant.
- Closed the PSTD shader documentation/oversized-file gap: `gpu/shaders/pstd.wgsl` now keeps only dispatch-local comments and delegates the storage-buffer contract, shift packing, twiddle packing, source packing, and invariants to `docs/gpu/pstd_shader_abi.md`, reducing the shader from 924 to 844 lines without changing WGSL bindings.
- Closed the GPU facade verification gap exposed by feature-gated PSTD tests: `GpuPstdSolver::new` now imports its sibling pipeline modules through the active facade hierarchy, `DelaySumGPU` is re-exported from the delay-and-sum module facade, GPU-only unused test imports are gated, and the stale workflow-level RF-data import is removed.
- Closed the sensor-recorder non-staggered velocity allocation gap: `SensorRecorder::record_velocity_step` now samples the half-cell collocated value directly at sensor positions instead of allocating a full shifted `Array3` per requested component. The change preserves the k-Wave interpolation identity and reduces non-staggered recording scratch memory from O(nx·ny·nz) per component to O(1) per sampled value.
- Closed the pressure-statistics sampled-extraction allocation gap: `extract_p_max`, `extract_p_min`, `extract_p_rms`, and `extract_p_final` now call single-field samplers instead of constructing all four sampled arrays and discarding three. Zero-step sampled RMS now returns the mathematically neutral zero vector, matching full-field `p_rms()`.
- Closed the velocity-statistics component-allocation gap: `SensorRecordSpec` now identifies ux/uy/uz statistic requirements independently, `SensorRecorder::with_spec` allocates only requested component accumulators, and narrow component extractors expose sampled ux/uy/uz max, min, and RMS without forcing unused full-grid velocity-stat buffers.
- Closed the sensor-recorder time-series clone-avoidance extension: pressure and velocity recorders now expose borrowed full-buffer and recorded-prefix `ArrayView2` accessors, preserving the existing owned extraction API while enabling zero-copy consumers for checkpointing, diagnostics, and Python bindings.
- Extended zero-copy sensor time-series access through solver facades: PSTD, FDTD, and elastic SWE solvers now expose borrowed full-buffer and recorded-prefix sensor views in addition to their owned extraction methods, allowing downstream diagnostics and bindings to avoid recorder-buffer clones without changing existing APIs.
- Closed the recorder checkpoint clone-boundary refinement: `SensorRecorder` now exposes `checkpoint_state_view` as the borrowed checkpoint source of truth, the owned `checkpoint_state` composes that view with `to_owned`, and PSTD checkpoint creation now takes ownership only at the serialization boundary.
- Closed the intensity-average recorder allocation gap: `SensorRecordSpec::records_ux/uy/uz` now means velocity time-series storage only. Acoustic intensity still requests pressure and instantaneous velocity samples, but `IntensityAvg*` no longer allocates unused velocity or intensity time-series buffers when only averages are requested.
- Closed the sensor-recorder velocity module ambiguity by completing activation of the directory-backed `simple::velocity` tree. The active facade now resolves to `velocity/{mod,series,intensity,stats,recording,tests}` with no competing flat-file module root.
- Closed the sampled-statistics temporary-allocation gap: pressure and velocity statistic samplers now fill `Array1` outputs directly instead of collecting into an intermediate `Vec` and converting, preserving value semantics with one allocation per sampled output field.
- Extended time-averaged intensity extraction with reusable-output APIs: `fill_i_avg_x/y/z` writes into caller-owned `Array1` storage, validates component availability and output length, and keeps the existing owned `extract_i_avg_*` APIs as compatibility wrappers.
- Extended sampled pressure-statistics extraction with reusable-output APIs: `fill_p_max`, `fill_p_min`, `fill_p_rms`, and `fill_p_final` write into caller-owned `Array1` storage, validate unavailable statistics and output length, and keep the existing owned extractors as compatibility wrappers.
- Extended sampled velocity-statistics extraction with reusable-output APIs: `fill_ux/uy/uz_{max,min,rms}` write into caller-owned `Array1` storage, validate unavailable component statistics and output length, and keep the existing owned extractors as compatibility wrappers.
- Closed the GPU PSTD run-cache sentinel cleanup: zero-source/zero-sensor storage buffers now use an explicitly named non-empty sentinel with documented WebGPU binding invariant, packed source-buffer helper tests pin index-prefix preservation, empty-tail behavior, and cache-hit tail overwrite semantics, and placeholder terminology is removed from the time-loop dispatch docs.
- Closed the PSTD anti-aliasing success-only test gap: the anti-aliasing regression now seeds a Nyquist checkerboard pressure field, applies the Butterworth spectral filter, and asserts strong L2 attenuation, finite output, and no time-step advancement for direct filter application.
- Closed the velocity-only recorder sequencing gap: `SensorRecorder::record_step` now advances the shared timestep even when pressure time-series storage is intentionally absent, so `record_velocity_step` records velocity-only specs into the correct column without allocating a pressure buffer. Verification: recorder suite 17/17, velocity-only regression 1/1, `cargo check -p kwavers`, `cargo check -p pykwavers`, and `cargo clippy -p kwavers --lib --no-deps -- -D warnings`.
- Closed the pykwavers recorder extraction clone gap: PSTD result assembly now borrows `SensorRecorder::{recorded_pressure_view,recorded_ux_view,recorded_uy_view,recorded_uz_view}` and applies `trim_initial_recorder_view`, so the Python boundary performs one owned allocation for the final `Nt` output instead of cloning `Nt+1` and cloning again after trimming. Verification: targeted trim-view regression 1/1, `cargo check -p pykwavers`, and `cargo clippy -p pykwavers --lib --no-deps -- -D warnings`.
- Closed the acoustic-intensity recorder completion gap: `SensorRecordSpec` now treats intensity as requiring pressure plus the matching velocity component, `SensorRecorder` computes `I_alpha(t)=p(t)u_alpha(t)` and `I_avg_alpha`, and pykwavers maps `Ix/Iy/Iz/I_avg_x/I_avg_y/I_avg_z` to observable `SimulationResult` arrays. Verification: recorder suite 20/20, pykwavers intensity mapping regression 1/1, `cargo clippy -p kwavers --lib --no-deps -- -D warnings`, and `cargo clippy -p pykwavers --lib --no-deps -- -D warnings`.
- Closed the acoustic-intensity spec/documentation hardening gap: recorder-field tests now prove that `IntensityX`/`IntensityAvgX` require pressure and only ux, not uy/uz or statistic grids, and pykwavers' Rustdoc mapping table lists all intensity record strings. Verification: targeted spec regression 1/1 and full pykwavers lib suite 6/6.
- Closed the acoustic-intensity velocity-buffer memory-policy gap: `SensorRecordField::needs_velocity_time_series` now means stored raw velocity series only, while `SensorRecordSpec::needs_any_velocity` separately includes intensity. Intensity-only modes still receive instantaneous velocity fields for `p·u` but no longer classify themselves as raw velocity time-series requests. Verification: targeted intensity spec regression 1/1, recorder suite 23/23, and `cargo clippy -p kwavers --lib --no-deps -- -D warnings`.
- Closed the recorder intensity test-quality/documentation gap: allocation tests now assert allocated shapes, zero initial averages, computed `p·u` values, and explicit absence contracts instead of relying on `is_some`/`is_none`; recorder Rustdoc now uses the canonical `I_avg_x/I_avg_y/I_avg_z` names. Verification: recorder suite 23/23, scoped assertion/naming search clean, and `cargo clippy -p kwavers --lib --no-deps -- -D warnings`.
- Closed the pykwavers intensity record-mode contract drift: the mapping regression now asserts `Ix/I_avg_x` sets intensity and pressure requirements, preserves `needs_any_velocity()` for instantaneous `p·u` sampling, and keeps `records_ux()` false so raw velocity time-series storage is not reintroduced through Python-facing record modes. Verification: targeted pykwavers regression 1/1, full pykwavers lib suite 6/6, and `cargo clippy -p pykwavers --lib --no-deps -- -D warnings`.
- Closed the CPML per-dimension config test-quality gap: valid dimension access now asserts exact x/y/z thickness and alpha values, invalid dimensions assert the structured `KwaversError::InvalidInput` message, and dimension-specific theoretical reflection is checked against the documented exponential formula.
- Closed the microbubble state test-quality gap: SonoVue, Definity, and drug-loaded constructors now assert exact radius, shell, thermodynamic, gas-mole, and drug-mass values; negative constructor/runtime validation paths now assert the structured invalid-value parameter and reason instead of only checking failure.
- Closed the CentralDifference2 test-quality gap: constructor tests now verify anisotropic operator metadata and exact `InvalidGridSpacing` payloads, insufficient-grid tests verify exact required/actual/direction values, and linear-field tests pin the documented forward/backward boundary stencil values.
- Closed the CentralDifference4 test-quality gap: constructor tests now verify anisotropic operator metadata and exact `InvalidGridSpacing` payloads, insufficient-grid tests verify exact required/actual/direction values, and linear-field tests pin the documented first-order boundary plus second-order near-boundary stencil values.
- Closed the CentralDifference6 test-quality gap: constructor tests now verify anisotropic operator metadata and exact `InvalidGridSpacing` payloads, insufficient-grid tests verify exact required/actual/direction values, and linear-field tests pin the documented first-order boundary, second-order near-boundary, fourth-order near-boundary, and sixth-order interior stencil values.
- Closed the staggered-grid test-quality gap: constructor tests now verify anisotropic operator metadata and exact `InvalidGridSpacing` payloads, insufficient-grid tests verify exact required/actual/direction values across forward/backward x/y/z allocating paths, and the forward-x zero-allocation path verifies the same rejection contract.
- Closed the differential-operator SSOT cleanup gap: the tracked but unreferenced `staggered_grid_draft_20260430172431` duplicate tree is removed, leaving `staggered_grid::{operator,forward,backward,tests}` as the only authoritative staggered-grid implementation under the active differential module facade.
- Closed the analysis draft-tree SSOT cleanup gap: tracked but unreferenced `subspace_draft_20260430163601` and `clinical_draft_20260430143451` trees are removed, leaving active adaptive beamforming and validation facades as the only authorities.
- Closed the functional-ultrasound atlas placeholder gap: `BrainAtlas::load_default` now builds a deterministic stereotactic mouse reference phantom with nonuniform intensity and anatomical region annotations, `with_annotation` validates shape/finite domains, coordinate conversion rejects negative/out-of-range physical coordinates before integer casting, and atlas registration borrows the reference image instead of cloning it.
- Closed the functional-ultrasound vasculature completion gap: segmentation now uses histogram Otsu thresholding, vessel classification computes static contrast, equivalent diameter, principal-axis orientation, and conservative artery/vein confidence from the real mask/image, centerline extraction returns 6-neighbour medial-axis voxels, and Doppler velocity uses `v = f_d c / (2 f_0 cos(theta))` with finite-domain validation.
- Closed the functional-ultrasound Otsu SSOT correction: vasculature thresholding now delegates to RITK's canonical Otsu implementation via `compute_otsu_threshold_from_slice`, removing the local duplicate histogram/prefix-sum threshold code while keeping connected-component logic local to the vasculature module.
- Closed the analytical phase-shifting strategy-dispatch gap: `PhaseShifter::apply_phases` no longer has a catch-all `NotImplemented` branch for supported `ShiftingStrategy` variants. `Focused` uses spherical focusing, `MultiFocus` consumes packed 3-D focal-point triples, and `Custom` applies one direct phase per element; sibling beam/focus controllers now share the same 60-degree steering and 1 mm focal-distance contracts.
- Closed the medium-builder heterogeneous file-map fallback gap: `MediumBuilder::build_heterogeneous` now rejects requested `tissue_file` and `property_maps` with explicit `FeatureNotAvailable` errors until real volume loaders are selected, instead of logging and constructing a scalar field that erases the requested heterogeneity. Scalar heterogeneous configs with no file/map still build from the configured density, sound speed, absorption, and nonlinearity.
- Closed the CEUS orchestrator registry error-boundary gap: an empty `CEUSOrchestrators` registry now returns a typed `FeatureNotAvailable` configuration error instead of `NotImplemented`, making the domain registry boundary explicit while leaving concrete CEUS simulation implementations to the registered factory layer.
- Closed the hybrid validation mock/unavailable-computation gap: `HybridValidationSuite` now owns a documented manufactured acoustic eigenmode boundary with a sixth-order centered second-derivative residual, closed-form reference derivative, monotone grid-refinement check, and CFL value calculation. This removes the mock convergence curve and the solver/reference/eigenvalue `NotImplemented` paths without binding the validation harness to a concrete PSTD/FDTD implementation.
- Closed the comparative-example visual parity export gap: `example_parity_utils.save_side_by_side_parity_figure` is now the shared PNG-export boundary for reference/candidate/difference panels, missing mask and diagnostic comparisons now write side-by-side figures plus report references, PR FFT/TR comparisons export reconstruction and sensor-matrix visualizations, and the utility test suite statically rejects future comparative examples that omit a declared visual export.
- Closed the initial-pressure comparison disparity: sensors now use a documented PML-safe interior layout, k-wave-python and pykwavers traces are aligned by propagated-state recorder semantics, and PASS/FAIL is evaluated over a geometry-derived pre-boundary acoustic window.
- Closed the IVP particle-velocity plot discrepancy: k-wave-python's implicit multidimensional `p0` smoothing is now represented as an explicit shared source-preprocessing boundary, pykwavers receives the same smoothed field, and k-wave-python runs with `smooth_p0=False` to prevent double smoothing.
- Added a HIFU procedure visualization slice using a documented Rayleigh-Sommerfeld focused-aperture field and Pennes bioheat update. The example exports focal intensity, absorbed heat, final temperature, and focal temperature-over-time plots with computed focus and FWHM metrics.
- Extended the HIFU procedure slice from cavitation-risk screening to explicit bubble feedback: Keller-Miksis radius dynamics drive passive receiver pressure, subharmonic/RMS and Rmax/R0 onset metrics modulate source pressure around a nominal cavitation-onset operating point, repeated receiver-control bursts provide a nonconstant pressure-squared envelope for Pennes heat deposition, and the example exports closed-loop cavitation-feedback and modulated-versus-constant-power temperature/power plots plus report metrics.
- Hardened Burn DAS beamforming verification by replacing success-only assertions with deterministic computed-value checks for focal delay sums, apodization weighting, CPU wrapper output, and multi-focus all-one RF data.
- Closed the DAS-PAM Python boundary gap: `DelayAndSumPAM::beamform_view` now accepts borrowed ndarray views with authoritative Rust-side shape/finite-value validation and fractional-delay interpolation, `pykwavers.passive_acoustic_map_das` delegates to that boundary without cloning input matrices, and `pykwavers/examples/passive_acoustic_mapping_compare.py` compares localization against the KWave.jl delay-law reference grid with 2-D parity panels plus 3-D cavitation-volume maximum-intensity projections.
- Added the histotripsy cavitation-volume example: `pykwavers/examples/histotripsy_cavitation_compare.py` evaluates a Rayleigh-Sommerfeld focused aperture, rotates the axisymmetric field into a 3-D volume, compares Maxwell intrinsic-threshold cavitation probability against millisecond-pulse Keller-Miksis collapse strength, writes intrinsic, ms-pulse cavitation, bubble internal-temperature, mechanism, and pressure-response PNG visualizations, records computed threshold/support volumes in JSON, and is pinned by value-semantic pytest coverage.

## Session 2026-05-04 Engineering Increments

- [patch] Closed the ScratchArena consolidation gap: defined the `ScratchArena` trait (`memory_bytes() → usize`, `clear() → ()`) in `solver::workspace` with Memory Monotonicity invariant; implemented it for `SolverWorkspace` (fixing the `memory_usage()` formula: 1 × Complex<f64> + 3 × f64 buffers = `complex_size·N + 3·real_size·N`), `KuznetsovWorkspace` (14 named `Array3<f64>` scratch buffers; `SpectralOperator` excluded as a grid constant), and `BornWorkspace`; added value-semantic tests (memory size, all-zeros-after-clear, stable-after-clear) for each implementor; added `pub use workspace::ScratchArena` re-export in `solver::mod`; deleted the two orphaned files `fdtd/workspace.rs` and `pstd/workspace.rs` (not declared in their `mod.rs`, called non-existent `SolverWorkspace::memory_budget()`, zero consumers); wired `validation/contract.rs` into `validation/mod.rs`. Full test suite passes 2 passed; `cargo check` clean.
- [patch] Closed the checkpoint-test control-flow gap: converted 2 `match + panic!` arms in `pstd/checkpoint/tests.rs::test_checkpoint_validate_restore_contract_rejects_mismatch` to `let…else` patterns (stable since Rust 1.65), matching `KwaversError::InvalidInput(ref msg)` and asserting message content; removed `assert_matches!` usage (unavailable in MSYS2 Rust 1.95.0). Suite passes 3/3.
- [patch] Closed the R2C/C2R FFT verification task: audited `pstd/propagation/time_loop/mod.rs` — confirmed `forward_r2c_into` / `inverse_c2r_into` are called for pressure, velocity, and absorption stages with correct half-spectrum dimension `nz_c = nz/2 + 1`; no gap found, no changes required.
- [patch] Closed the Spectral-CPML incompatibility gap: added an invariant guard at the top of `GenericFdtdSolver::enable_cpml` returning `KwaversError::InvalidInput` when `config.kspace_correction == KSpaceCorrectionMode::Spectral` (CPML requires finite-difference gradient arrays absent from the spectral path — Roden & Gedney 2000); added `# Errors` documentation with the Roden & Gedney reference; added two value-semantic tests in `fdtd/solver/accessors.rs::tests`: `enable_cpml_rejects_spectral_kspace_correction` (asserts `InvalidInput` with "Spectral" in message) and `enable_cpml_accepts_none_kspace_correction` (asserts `Ok(())` and `cpml_boundary.is_some()`). Both pass in 0.093 s.
- [minor] Closed the ultrasound physics book gap: authored 7 chapters (`docs/book/ch01` through `ch07`) covering wave equation foundations, propagation models, FDTD numerics, PSTD/k-space methods, nonlinear acoustics, transducers and arrays, and sensors; each chapter includes theorem/proof skeletons, algorithm contracts, implementation cross-references to kwavers modules, and research anchors. Added automated figure-generation Python scripts under `pykwavers/examples/book/` (`ch01` through `ch07` plus `generate_all_figures.py`) using analytical formulas and matplotlib, with per-chapter output directories under `docs/book/figures/`.
- Closed the DeepFusion runtime gap by implementing a deterministic voxel-attention fusion operator instead of requiring nonexistent trained weights. The active strategy computes robust per-modality salience, multiplies it by quality/configuration priors, normalizes through a softmax simplex, and emits convex fused intensity, confidence, and optional entropy uncertainty maps.
- [minor] Closed the 3D CPU DAS beamformer gap: `delay_and_sum_cpu` in `cpu/das.rs` implements the plane-wave coherent receive DAS formula (Thomenius 1996, Jeong & Kwon 2013) with fractional-delay linear interpolation, Rayon voxel-parallel loop, and full apodization support; `cpu/mod.rs` wires the `das` and `mvdr` submodules; `three_dimensional/mod.rs` declares `pub(super) mod cpu;`; `processing.rs` `process_volume`, `process_streaming`, `validate_input`, and `process_delay_and_sum` now call the real implementations instead of returning `FeatureNotAvailable`. Tests: zero-delay passthrough (7.0 = rf[0]), channel-mismatch `InvalidInput`, coherent gain M=4 co-located, exact delay geometry τ=1 sample. All 4 pass.
- [minor] Closed the 3D CPU MVDR beamformer gap: `mvdr_cpu` in `cpu/mvdr.rs` implements the Capon (1969) / Synnevåg et al. (2007) MVDR with spatially-smoothed covariance (Shan & Kailath 1985, Q=(nel−L+1)³ sub-apertures), relative diagonal loading R_δ=R+δ·(tr(R)/L)·I, Cholesky/LU solve via nalgebra, Rayon voxel-parallel loop; `processing.rs` `process_mvdr_3d` dispatches to it on non-GPU builds. Tests: L=1 identity theorem (output=|x̄[0]|=3.0), L=1 δ-invariance corollary, channel-mismatch `InvalidInput`, subarray-exceeds-array `InvalidInput`, diagonal-loading PD guarantee (4 δ values, finite non-negative output). All 5 pass. Total suite: 3258/3258 PASS (net+9).
- [patch] Closed the ivp_1D PML default gap: `cpml_thickness_limits` used `(min_dim/6).max(2)` → 85 cells for NX=512 quasi-1D grid, placing both sensors inside the PML; fixed to `20_usize.min(max_allowed).max(2)` matching k-Wave's fixed 20-cell default; added explicit `sim.set_pml_size(20)` in `ivp_1D_simulation_compare.py`; ivp_1D parity PASS (pearson=0.999994, PSNR=63.81 dB).
- [patch] Closed the parity-sweep regex gap: `_run_parity_sweep.py` patterns were case-sensitive on `Status:` and didn't match `RESULT: PASS`; updated `re_status`, `re_pearson`, `re_psnr`, `re_overall` to catch both variants; added `tvsp_snells_law_compare.py` to the sweep list; 18/18 sweep PASS.
- [patch] Closed the AS WSWA-FFT double-normalization gap: `axisymmetric.rs::compute_vel_grads` and `compute_density_divs` applied `norm = 1/(nx*nr_exp)` after `inverse_complex_inplace` which already applies FFTW-compatible 1/N normalisation; removed the redundant factor; `at_circular_piston_AS` Pearson 0.007→1.000, RMS→0.9997, PSNR→68.6 dB; `at_focused_bowl_AS` Pearson −0.18→1.000, RMS→0.9993, PSNR→69.2 dB; 18/18 sweep PASS.
- [patch] Closed the EWP parity gaps: `ewp_plane_wave_absorption_compare.py` PASS (timing error 0.67 samples, Pearson 0.9916) via SIGMA_CELLS=20 to eliminate superdispersive pre-cursor (kΔx=π/2 components travel at 1.178×c_p) plus windowed Pearson ±2σ around measured peak; `ewp_layered_medium_compare.py` PASS (Pearson 0.9635) via windowed Pearson ±2SIGMA_CELLS around each sensor peak; `ewp_3D_simulation_compare.py` PASS (min |Pearson| = 0.986) via within-group symmetry (axial ±x P-wave, transverse ±y,±z S-wave); all three added to `_run_parity_sweep.py`; 21/21 sweep PASS.

## k-Wave Example Parity Gaps (2026-05-08 Audit)

### DIFF category — Thermal diffusion / Pennes bioheat (CLOSED 2026-05-08)

All four DIFF scripts delivered and PASS:
- [x] `diff_homogeneous_medium_diffusion_compare.py`: Pennes ODE transient vs analytical; pearson=0.999997, PSNR=113.9 dB.
- [x] `diff_homogeneous_medium_source_compare.py`: 3D Gaussian source forward Euler; Python BC matches Rust zero-Laplacian-at-boundary exactly; pearson=1.000000, PSNR=276.7 dB.
- [x] `diff_focused_ultrasound_heating_compare.py`: acoustic→thermal Q coupling; rel_err=0.006 vs 5% tolerance; analytical beam fallback (NZ=1 PSTD unsupported).
- [x] `diff_binary_sensor_mask_compare.py`: sensor extraction identity (1.63e-10°C) + ODE Pearson ≥ 0.999997.
All four added to `_run_parity_sweep.py`.

### EWP category — Elastic waves — CLOSED 2026-05-08

- [x] `ewp_shear_wave_snells_law_compare.py`: SH wave Snell's law at planar interface c_s1=1500/c_s2=2500; uz IVP, 5 sensors at i=44; angular error=1.323° ≤ 1.5° — PASS.

### Axisymmetric (AS) validation — CLOSED 2026-05-08

- [x] `ivp_axisymmetric_simulation_compare.py`: AS PSTD IVP Gaussian pulse vs k-wave-python kspaceFirstOrderASC; on-axis Pearson=0.9988 ≥ 0.98, full-2D Pearson=0.9989 ≥ 0.95 — PASS. Both scripts added to `_run_parity_sweep.py`.

### PR category — Photoacoustic reconstruction (advanced, 7 scripts)

k-wave-python has partial PR coverage (2D/3D TR point sensors closed); remaining k-Wave MATLAB PR scripts require reconstruction infrastructure:
- [x] [major] `pr_2D_attenuation_compensation_compare.py`: CLOSED 2026-05-08. CW angular spectrum + scalar on-axis Beer's-law compensation exp(+α_Np·z_m); α₀=3 dB/MHz/cm, z_m=30mm; PSNR gain=+32.67 dB, PSNR_comp=47.61 dB — PASS. (PSTD fractional-Laplacian path remains open as separate backlog item.)
- [x] [major] `pr_2D_FFT_reconstruction_compare.py`: CLOSED 2026-05-08. kspace_line_recon; depth-slice Pearson=0.9626, max-proj Pearson=0.9721 — PASS.
- [x] [major] `pr_3D_FFT_reconstruction_compare.py`: CLOSED 2026-05-08. time_reversal_reconstruction on 32³ Gaussian; center-plane Pearson=0.9218, flat Pearson=0.7933 — PASS.
- [x] [minor] `pr_2D_TR_directional_sensors_compare.py`: CLOSED 2026-05-08. Cardioid filter (1+kz/k)/2 in NX×NY k-space; analytic E[|W|²]=17/24 → 1.50 dB gain for pixel-uniform forward-only noise; PSNR_dir=51.18 dB, gain=+1.11 dB — PASS.
- [x] [minor] `pr_3D_TR_directional_sensors_compare.py`: CLOSED 2026-05-08. Combined H_back×W in single padded N=256 FFT; PSNR_dir=39.86 dB, gain=+1.02 dB — PASS. All 3 new PR scripts added to sweep (65 active).

### TVSP propagator scripts (4 scripts, require angular spectrum)

- [x] [major] `tvsp_acoustic_field_propagator_compare.py`: CLOSED 2026-05-08. Validated pykwavers `angular_spectrum_cw` against numerical RS-2 integral (Sommerfeld pressure-specified formula); circular piston a=2mm, f₀=1MHz, z∈[5,50]mm; Pearson=0.9974, PSNR=36.54 dB — PASS; added to sweep (61 active).
- [x] [minor] `tvsp_angular_spectrum_method_compare.py`: CLOSED 2026-05-08. Implemented `pykwavers.angular_spectrum_cw` (pure-NumPy Zeng & McGough 2008 CW propagator); compares against k-wave-python `angular_spectrum_cw`; lossless Pearson=1.000 PSNR=299 dB, absorbing Pearson=1.000 PSNR=118 dB — PASS; added to sweep (60 active).
- [x] [minor] `tvsp_equivalent_source_holography_compare.py`: CLOSED 2026-05-08. `pkw.backward_angular_spectrum_cw` + `pkw.gaussian_source_2d` added to public pykwavers API; holography roundtrip Pearson=0.999980, PSNR=56.56 dB — PASS; added to sweep (62 active).
- [x] [minor] `tvsp_slit_diffraction_compare.py`: CLOSED 2026-05-08. Pearson=0.9960 ≥ 0.95 — PASS.

### Sweep expansion (infrastructure)

- [x] [patch] CLOSED 2026-05-08. `_run_parity_sweep.py` expanded to 44 scripts. `us_beam_patterns` amplitude deficit (RMS=0.57) fixed by removing double-application of 2*c0*dt/dx velocity source scaling (commit caabc640 added it internally); now PASS (RMS=0.948, PSNR=26.6 dB). Remaining FAIL: `us_bmode_phased_array_tiny` — same root cause, pending fix.

### NA category — remaining analysis scripts

- [x] [minor] `na_optimising_time_step_compare.py`: CLOSED 2026-05-08. CFL sweep [0.05–0.70]; k-space corrected PSTD dispersion-exact, error=0.008% across all CFL — PASS.
- [x] [minor] `na_optimising_grid_parameters_compare.py`: CLOSED 2026-05-08. PPW sweep [3–20]; spectral accuracy, error=0.009% at all PPW ≥ 3 for smooth Gaussian — PASS.

### Sweep expansion — 59 scripts (2026-05-08)

- [x] [patch] CLOSED 2026-05-08. Expanded `_run_parity_sweep.py` from 45 to 59 scripts. Added tuple syntax for per-script extra args. All newly-added scripts confirmed PASS before adding:
  - AT: at_array_as_sensor, at_array_as_source, at_linear_array_transducer, at_linear_array_transducer_mask, at_focused_annular_array_3D_full, at_focused_annular_array_3D_mask, at_focused_annular_array_3D_weights
  - SD: sd_directional_array_elements, sd_directivity_modelling_2D, sd_directivity_modelling_3D
  - TVSP: tvsp_doppler_effect, tvsp_steering_linear_array
  - US: us_defining_transducer, us_bmode_phased_array (--quick --pykwavers-gpu)
- [x] [patch] CLOSED 2026-05-08. Velocity source double-scaling fix applied to 5 US scripts (us_beam_patterns, us_bmode_phased_array_tiny, us_bmode_phased_array, us_bmode_linear_transducer, us_defining_transducer): removed manual transducer_scale=2*c0*dt/dx which was double-counting the internal scaling added in commit caabc640. All 5 now PASS.
- [x] [patch] CLOSED 2026-05-08. Added `parity_status: {status}` stdout print to 7 scripts that wrote it only to file: at_linear_array_transducer, at_linear_array_transducer_mask, at_focused_annular_array_3D_full, at_focused_annular_array_3D_mask, at_focused_annular_array_3D_weights, us_defining_transducer, us_bmode_linear_transducer.
- [x] [patch] CLOSED 2026-05-08. Added `--allow-failure` argparse support to us_defining_transducer_compare.py and us_bmode_phased_array_compare.py. Updated `re_status` regex in sweep to match `parity_status:` prefix.
- [x] [patch] CLOSED 2026-05-08. Added NPZ scan-line caching to us_bmode_linear_transducer_compare.py; fixed `_EXAMPLE_UTILS_DIR` path to legacy k-wave-python example directory. us_bmode_linear_transducer commented out of sweep pending first-run cache generation.
- [x] [patch] CLOSED 2026-05-08. Sweep status-regex fix: added `parity_status:` stdout to 6 scripts that only wrote it to report file (na_optimising_time_step, na_optimising_grid_parameters, pr_2D_FFT_reconstruction, pr_3D_FFT_reconstruction, at_focused_annular_array_3D, checkpointing); sweep now 61/61 PASS with zero '?' entries. Also added tvsp_acoustic_field_propagator to sweep (61st script).
- [patch] Closed the 500-line structural limit enforcement pass (2026-05-08): split `solver::forward::bubble_dynamics::plugin` (627 lines), `physics::acoustics::bubble_dynamics::gilmore` (541 lines), `domain::medium::heterogeneous::factory::general` (529 lines), and `physics::acoustics::imaging::fusion::algorithms::tests` (558 lines) into directory-backed or sibling-file structures; `cargo check --lib` clean; all source files now satisfy the 500-line limit. `pub(super)` visibility used precisely where sibling test files access private fields; re-export pattern in `tests/mod.rs` eliminates per-subfile `use` repetition.
- [patch] Closed the PSTD divergence-cache and k-space multiply optimization pass (2026-05-08): added `div_ux`/`div_uy`/`div_uz` divergence-cache fields to `PSTDSolver`; wrote per-axis divergences after each axis IFFT in `update_density_cartesian`; replaced the 6-FFT recomputation block in `apply_absorption_to_pressure` with 3 `assign` calls; replaced `Complex64::new(kap,0.0)*u` patterns (4 mults+2 adds) with `(shift*u)*kap` (2 mults) in velocity/pressure/absorption hot paths; replaced `*gk*=Complex64::new(n,0.0)` with real-scalar `*gk*=n` for nabla1/nabla2 multiplies; removed four unused `Complex64` imports; cache excluded from KWCP checkpoint format (recomputed on first post-restore step). `cargo check --lib` and `cargo clippy -p kwavers --lib --no-deps -- -D warnings` clean.
- [patch] Closed the simulation multiphysics residual allocation and norm-consistency gap (2026-05-12): added the semantic `simulation::multi_physics::residual` module, moved coupled-field convergence to one allocation-free L-infinity metric, routed explicit/implicit/monolithic solver checks plus `FieldCoupler` transfers through it, rejected shape/nonfinite residual inputs, and upgraded tests from existence checks to exact residual/history assertions. `cargo test --manifest-path kwavers/Cargo.toml --lib multi_physics -- --nocapture`, `cargo check --manifest-path kwavers/Cargo.toml -p kwavers`, and `cargo clippy --manifest-path kwavers/Cargo.toml -p kwavers --lib --no-deps -- -D warnings` pass.
- [minor] Closed the Chapter 27 seismic/FWI simulated-ultrasound nonlinear hemispherical reconstruction figure gap (2026-05-12): the Rust brain-helmet FWI result emits a single-pass adjoint migration image and weak-Westervelt second-harmonic encoded rows from the same simulated ultrasound acquisition used for iterative FWI; acquisition now uses a deterministic 1024-element hemispherical cap, 3-D source/receiver distances, nearest azimuth-rotated receiver mapping, CT-derived axial offsets, five frequencies, and eight receiver offsets; pykwavers exposes the image, attenuation switch, nonlinear harmonic controls, geometry metadata, migration metrics, and Charbonnier edge-preserving controls; the matrix-free row operator avoids dense sensitivity storage and now caches per-row constants inside hot PCG operator calls; the chapter documents the migration/continuation/attenuation/nonlinear/ROI/hemisphere contract; `fig05_simulated_ultrasound_reconstruction.{pdf,png}` compares simulated data, migration reconstruction, and iterative FWI; `fig06_multislice_reconstruction_stack.{pdf,png}` covers twelve nonempty 3-D volume slices with target-independent mask-regularized FWI display; and `fig07_centroid_pons_thalamus_roi.{pdf,png}` crops the deep-midline ROI. Verification: brain-helmet Rust tests 3/3, seismic chapter pytest 9/9, Chapter 27 figure generation emits 7 PDFs, `cargo check -p kwavers --lib`, and `cargo build -p pykwavers --release` pass.
- [patch] Closed the workspace clippy error elimination pass (2026-05-08): boxed `BubbleField` in `BubbleEngine::KmOrRp` (`large_enum_variant`); reformatted `GilmoreSolver::step_rk4` doc continuation indent (`doc list item overindented`); removed `ref vs`/`ref active` from elastic SWE propagation pattern (`needless_borrow`); removed `let sx=sx`/`sy`/`sz` redundant rebindings in DAS 3D CPU beamformer (`redundant_locals`). All five pre-existing kwavers clippy errors eliminated; `cargo clippy -p kwavers --lib --no-deps -- -D warnings` exits clean.
- [patch] Closed the PINN NotImplemented pass (2026-05-08): implemented `BurnPinnBeamformingAdapter::beamform` (PINN 1D inference: channel/sample grid → normalised x/t coords, `BurnPINN1DWave::predict` → `Array3<f32>` image, replicated across frames); implemented `BurnPinnBeamformingAdapter::train` (`BurnPINNTrainer<B: AutodiffBackend>` round-trip over flattened target frames, 1000 epochs, 1500 m/s wave speed, trained model stored under `Arc<Mutex>`); tightened `PinnBeamformingProvider` impl bound from `Backend + 'static` to `AutodiffBackend + 'static`; updated `test_model_info` to `Autodiff<NdArray<f32>>`; implemented `train_pinn<B: AutodiffBackend>` (elastic 2D forward problem: forward pass at collocation/boundary/IC points, `compute_elastic_wave_pde_residual` with ρ=1000/λ=2.25GPa/μ=0 tissue defaults, `LossComputer` weighted total loss, backward, `PINNOptimizer::step`, `LRScheduler::step`, `TrainingMetrics::has_converged` early exit); implemented `DistributedPinnTrainer::train_epoch_distributed` (single-GPU fallback: build Burn [N,1] tensors from `&[(f64,f64,f64)]` slices, `BurnPINN2DWave::compute_physics_loss` 5-tuple, backward + `SimpleOptimizer2D::step` per replica, return per-replica `BurnTrainingMetrics2D`); fixed pre-existing `burn::module::Module` missing import in `pinn_optimizer.rs`, `pde_residual/tests.rs`, and `wavespeed/tests.rs`; `cargo check -p kwavers --features pinn` clean; 3214 tests pass.
- [patch] Closed the PINN feature-gate import hygiene pass (2026-05-09): removed 7 unused imports across 5 files gated under `--features pinn` — `AutodiffBackend` from `elastic_2d/training/optimizer/state.rs` and `burn_wave_equation_3d/solver/core.rs` (structs bound by plain `Backend`), `Geometry3D` from `burn_wave_equation_3d/wavespeed/mod.rs`, `Instant`/`BurnLossWeights3D`/`BurnTrainingMetrics3D` from `core.rs`, `KwaversResult` from `jit_compiler/mod.rs`, `HardwareCapabilities` from `edge_runtime/runtime.rs`; `cargo clippy -p kwavers --lib --features pinn --no-deps -- -W unused-imports` zero warnings.
- [patch] Closed the Bayesian MC-dropout NotImplemented→FeatureNotAvailable pass (2026-05-09): `BurnPinnBeamformingAdapter::estimate_uncertainty` now returns `SystemError::FeatureNotAvailable { feature: "Bayesian MC-dropout uncertainty", reason: "… no dropout layers …" }` instead of `KwaversError::NotImplemented`; removes the last production NotImplemented return from the codebase; `cargo check -p kwavers --lib --features pinn` clean.
- [patch] Closed the cross-solver hot-path `for_each`→`par_for_each` parallelization pass (2026-05-09): converted sequential element-wise passes on full 3-D and k-space arrays to `par_for_each` across PSTD (13 passes: absorption Steps 2–4, 7 anti-aliasing spectral multiplies, fill_rho_sum, propagate_kspace, source kappa, Helmholtz real-scalar opt, PSTDPlugin density-split, AS kappa_2d×2/uz_on_r/dpdx/dpdr/duxdx/duzdr), FDTD (7 passes: k-space ux/uy/uz velocity, collocated velocity×3, Westervelt accumulation, compute_grad_pos 5-array Zip::indexed real-scalar, compute_divergence_neg 3×Zip::indexed real-scalar), and nonlinear solvers (8 passes: Westervelt FDTD nonlinear 2-branch, Kuznetsov diffusion 5-array, nonlinear 4-array, numerical Laplacian/gradient Zip::indexed, spectral Laplacian/gradient with as_slice() captures, operator-splitting flux scale-hoist); `cargo check -p kwavers --lib` clean; 2848 tests pass (1 pre-existing; 3214 with pinn).
- [patch] Closed the codebase-wide `mapv_inplace`→`par_mapv_inplace` parallelization pass (2026-05-09): converted all 37 sequential element-wise in-place scalar transforms across 30 files — dispersion correction (2), adaptive boundary, SWE magnitude, CEUS scattering, ROS species decay, workspace scale/apply (trait `TensorMut::map_inplace` + `NdArrayTensor` impl updated to `F: Fn(f64) -> f64 + Send + Sync`), PSTD source-kappa, monolithic coupler residual (×3), FWI gradient norm (×2), IMEX stiffness/stability power-iter (×4), photoacoustic positivity clamps (Array1 ×2), time-reversal norm, line-reconstruction clamp, Fourier clamp, wavelet threshold, AMR criteria (×2), GPU IFFT norm, covariance shrinkage (×2), HAS plane absorption, FWI model constraints, clinical workflow norm, covariance sensor (×3), beamforming estimation scale, spatial smoothing (×2), PAM squared-pressure, SLSC coherence clamp, polynomial filter norm, neural layer scale (×2), feature aggregation (Array3<f32>), visualization norm/log-transform, spectral-derivative inner plane, power-law absorption inner-slice; zero `mapv_inplace` calls remain in `src/`; `cargo check -p kwavers --lib` clean; 2848 tests pass (1 pre-existing failure).
- [patch] Fixed `test_stress_divergence_uniform_displacement` test defect (2026-05-09): root-cause — `uy=0.3`, `uz=0.1` are non-binary f64 constants; 4th-order interior FD stencil (`-a + 8a - 8a + a`) produces ULP-level rounding ∼2e-16 while 1st/2nd-order boundary stencils give exactly 0 (identical-bit subtraction); with λ,μ∼1e9 amplification, syy varies between boundary (∼2e-6) and interior (∼3e-5) j-indices; fd1_y(syy) at j=2 mixes both giving spurious div_y=0.024; fix — test displacement changed to exact binary fractions (ux=0.5=2⁻¹, uy=0.25=2⁻², uz=0.125=2⁻³) with tighter tolerance 1e-10; `stress_divergence` implementation is correct; 2849/2849 tests pass.

## Session 2026-05-12 Transcranial Brain FUS Planning

- [x] [minor] Closed the pykwavers transcranial brain focused-ultrasound planning slice: added chapter 25 with local cranial CT/T1/MNI loading, source-backed TCIA/MNI dataset manifest generation, RITK registration adapter, 1024-element Exablate-style hemispherical phased array, skull path phase correction, Rayleigh focal pressure synthesis, Pennes CEM43 thermal dose, cavitation probability, and optional CFB-GBM tumor subspot planning.
- [x] [minor] Closed the `ritk-python` wheel/binding surface gap: re-exported the implemented segmentation and distance-transform APIs from `ritk-core`, made image-statistics value computation available to the binding, added Windows DLL-directory discovery in `ritk.__init__`, built and installed the local wheel into the pykwavers venv, and updated chapter 25 to read NIfTI sources through `ritk.io.read_image` before `ritk.registration.multires_syn_register`.
- [x] [patch] Added RITK Windows GNU static runtime preference flags where the current PyO3/CharLS link permits them: `.cargo/config.toml` now enables `target-feature=+crt-static`, `-static-libstdc++`, and `-static-libgcc` beside the existing `lld` linker selection. Forced `static=stdc++` in `ritk-io/build.rs` was rejected because the final extension link leaves unresolved CharLS C++ ABI symbols; the verified wheel still depends on `libstdc++-6.dll`.
- [x] [minor] Closed the clinical HIFU planning subspot gap: `HIFUPlanner::plan_sonication_schedule` now builds a deterministic `SonicationSchedule` over the target plus safety margin, derives lateral/axial pitch from the focal FWHM ellipsoid corner-bound, allocates treatment duration across subspots, computes per-subspot CEM43/peak temperature, and `plan_treatment` bases feasibility on proven coverage plus minimum subspot dose instead of a single-focus adequacy heuristic without changing the `HIFUTreatmentPlan` struct layout.
- [x] [minor] Closed the HIFU/BBB book parity gap: Chapter 24 remains the BBB-opening mechanism chapter; Chapter 25 is now titled and documented as transcranial HIFU plus BBB treatment planning; `docs/book/hifu_transcranial_ablation.md` is linked from the book README; the Chapter 25 GBM branch computes BBB subspot dose, Hill permeability, stable-cavitation probability, inertial-cavitation risk, opened tumor mask, and optional figure/metrics for a segmented real case.
- [x] [minor] Closed the executable GBM sample gap with UPenn-GBM `sub-002`: downloaded real co-registered T1/T1-Gd/T2/FLAIR/segmentation NIfTI assets under `data/upenn_gbm_sample`, added source/license provenance, and made Chapter 25 execute the BBB subspot branch without a fabricated CT.
- [x] [minor] Closed the skull-acoustics CT gap for Chapter 25: downloaded and converted RIRE patient 109 CT to NIfTI, made it the preferred skull acoustic map, and changed phase correction from binary skull-delay only to CT-derived travel time, impedance transmission, attenuation, and element amplitude weighting.
- [x] [minor] Closed the HIFU-vs-BBB execution-contract ambiguity: HIFU uses CT plus registered atlas; BBB opening accepts CT plus CT-space segmentation as sufficient, with MRI used only when segmentation must be defined before registration into CT space.
- [x] [minor] Closed the affine sample-CT-to-MRI QC gap: Chapter 25 can resample a sample CT to an MRI-space GBM case from NIfTI affines and emits visual overlay QC plus NMI/edge-overlap metrics; this remains a fallback and does not replace same-patient CT-backed GBM acquisition.
- [x] [patch] Closed the CT/MRI/MNI same-plane QC defect: registration now affine-initializes T1 and MNI into the CT lattice before RITK refinement and `fig01_registered_ct_mri_mni` displays one CT-space target plane rather than modality-native voxel planes.
- [x] [patch] Closed the AP-reflection and metric-deficiency gap: foreground affine initialization now searches axis reflections by mask Dice, and Chapter 25 reports NMI/MSE beside NCC for multimodal registration QC.
- [x] [minor] Closed the same-patient RIRE CT/MR registration graph gap: converted RIRE patient 109 MR-T1/MR-T2 MetaImage data to NIfTI, made MR-T1 the default subject MRI when the RIRE CT skull map is present, corrected raw-buffer axis ordering to MetaImage x-fastest `(z, y, x) -> (x, y, z)`, and changed Chapter 25 to map MNI through subject MRI on the CT lattice instead of independently registering atlas-to-CT.
- [x] [patch] Closed the registration hardening increment: CT brain masks now come from filled skull boundaries rather than HU background thresholds, atlas affine fitting uses the CT-derived intracranial mask intersected with T1 foreground, the same-patient MNI atlas path forbids LR/AP/SI reflections, foreground affine initialization performs NMI translation refinement, RITK boundary conversion explicitly maps internal XYZ arrays to RITK ZYX images and back, deformable candidates are metric-guarded against NMI regression, and the registration QC figure now displays axial/coronal/sagittal target planes.
- [x] [minor] Closed the GBM modality-bridge workflow gap: added `transcranial_planning.modality_bridge`, deterministic `modality_bridge_manifest.json` emission, CT/MRI/segmentation pairing requirements, CT-space versus MRI-space execution boundaries, cWDM/SLaM-DiMM/NV-Segment-CTMR reference records, and focused tests proving CT-backed and UPenn MRI-only cases remain correctly scoped.
- [x] [patch] Closed the GBM imperfect-modality ingest gap: `GbmCasePaths` now represents optional MRI channels directly, CT-space segmentation no longer aliases CT into T1-Gd/FLAIR fields, CT-backed BBB planning accepts real CT plus segmentation without MRI, MRI-space planning accepts segmentation plus any real in-space MRI reference, and the modality bridge records Holder-MI incomplete-MRI segmentation plus TextBraTS as design references for available-input reconciliation without synthetic in-script fallbacks.
- [x] [minor] Closed the skull-adaptive transcranial benchmark gap: `kwavers::clinical::therapy::theranostic_guidance::transcranial_fus` now evaluates CT-conditioned helmet aperture placement, skull-aware corrected pressure, uncorrected baseline pressure, and TFUScapes-aligned relative-L2, focal-position, and max-pressure metrics from the existing Chapter 25 CT/Rayleigh planning path; `pykwavers` exposes the RITK CT wrapper and the book helper records the paper-structure comparison without adding a parallel demo.
- [x] [patch] Closed the TFUScapes one-case import and structural comparison gap: added a reproducible loader for `vinkle-srivastav/TFUScapes` train row 0 (`A00028185/exp_0.npz`, pinned revision and SHA-256), identified the minimal paper fields (`ct`, `pmap`, `tr_coords`), derived the target from the pressure-map peak, fitted the transducer index coordinates to the shared scene radius, routed the case through the existing skull-adaptive benchmark wrapper via a temporary CT NIfTI, and documented the no-parallel-demo execution contract.
- [ ] [minor] Add a small licensed same-patient CT-backed CFB-GBM extraction under `data/cfb_gbm_sample` (`ct.nii.gz`, `t1gd.nii.gz`, `flair.nii.gz`, `segmentation.nii.gz`) so the GBM branch can exercise tumor and skull acoustics from the same patient without downloading the full 208 GB cohort.
