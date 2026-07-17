## State refresh (2026-07-16) — kwavers Batch #1 closure: Rayon → Moirai migration complete

- **kwavers Batch #1 — CLOSED** per clean legacy migration audit.
  - Source-level scoped audits under `crates/kwavers*/src` report zero direct
    `rayon::`, `use rayon`, `par_for_each`, `into_par_iter()`, or `ndarray::Zip`
    parallel iterator usage.
  - `cargo run --bin xtask -- legacy-migration-audit` reports 0 Rayon, 0 ndarray,
    0 nalgebra, 0 burn, 0 tokio surface items.
  - `cargo run --bin xtask -- refresh-legacy-allowlist` regenerated the allowlist.
  - `cargo check -p kwavers-math` passes.
  - Workspace provider version-constraint fixes required to unblock the audit:
    - `repos/ritk/Cargo.toml`: moirai `0.3.0` → `0.4.0`, leto `0.36.0` → `0.37.0`,
      leto-ops `0.36.0` → `0.37.0`, hephaestus-core `0.13.0` → `0.15.0`,
      hephaestus-wgpu `0.13.0` → `0.15.0`, apollo-fft `0.17.0` → `0.18.0`.
    - `repos/coeus/Cargo.toml`: leto `0.36.0` → `0.37.0`,
      leto-ops `0.36.0` → `0.37.0`, hephaestus-core `0.14.0` → `0.15.0`,
      hephaestus-wgpu `0.14.0` → `0.15.0`, hephaestus-cuda `0.14.0` → `0.15.0`.
  - Residual: the active RITK provider migration emits compiler warnings under
    Kwavers' full facade build; it no longer blocks compilation.

# Gap Audit

- In progress 2026-07-16: `PulsedWaveDoppler` previously returned only a
  one-sided magnitude waveform, discarding reverse-flow bins. The provider now
  has `signed_spectrum`, whose centered two-sided frequency, velocity, and
  power axes preserve both signs, reject perpendicular beam geometry, and
  reject an ensemble longer than its FFT instead of silently truncating pulses.
  The coherent offline lock now resolves; `kwavers-analysis` compiles, normal
  warning-denied Clippy passes, focused PW Nextest passes 8/8, and locked
  `kwavers --all-features` compilation passes. The facade reports only current
  upstream RITK and Hephaestus warnings. Evidence tier: compiler diagnostics
  and value-semantic provider regression.

- In progress 2026-07-16: the coherent all-feature graph exposed 79 solver
  source diagnostics. Machine-applicable corrections plus test-only contract
  repairs now make warning-denied Clippy pass for solver's independently
  buildable `pinn,gpu,async-runtime,nightly,simd,test-util` feature set. The
  full solver suite passes 844 runnable tests with 4 ignored. The locked
  `kwavers --all-features` facade compiles and package Nextest passes. The
  all-target gate is reconciling stale test/example call sites surfaced by
  concurrent provider migration. Current RITK and Hephaestus dependency
  warnings remain outside Kwavers' source lint surface. Evidence tier:
  compiler linting and value-semantic regression tests.

- In progress 2026-07-15: GitHub Actions checked out Kwavers alone even though
  the workspace declares live sibling Atlas paths such as `../apollo`. PR #288
  therefore failed before compilation in every Cargo job; the architecture
  workflow also called the deleted `scripts/validate_architecture.sh`. One
  composite setup action now derives and checks out the manifest-declared
  providers at `codex/kwavers-atlas-integration`'s coordinated submodule
  revisions before Cargo runs, and stale native `cargo test` invocations move
  to Nextest. The first repair runs confirm provider defaults are incompatible
  (`apollo-fft` 0.17.0 versus RITK's required `^0.15.0`) and Atlas `main`
  pins incompatible Apollo 0.14. The subsequent Architecture rerun
  `29444236283` completes provider checkout and finds the independent Linux
  `CPU_SET` mutability defect in Kwavers' explicit-affinity branch. Strict
  Clippy then identifies two manual NUMA-mask ceiling divisions. The migration
  audit's substring matcher additionally misclassifies `numpy::ndarray` in the
  sanctioned PyO3 facade as direct legacy ndarray; the classifier now separates
  them, with a direct-vs-facade regression and regenerated stale baseline.
  Evidence tier: authoritative Actions logs plus local manifest-path resolution,
  YAML parsing, and local audit execution. Residual: the corrected PR head must
  complete the full remote matrix before closure.

- In progress 2026-07-15: the repaired workflow reached two independent
  infrastructure defects. The committed global `target-cpu=native` flag made
  the hosted xtask binary fault with SIGILL, and the generic `--all-features`
  CPU workflow selected the explicit CUDA runtime without its required CUDA
  13.2 toolkit. The policy now targets a portable CPU baseline with runtime
  dispatch and retains CUDA compilation in its own CUDA 13.2 container job.
  This is not a reduced feature check. The CPU feature matrix exercises the
  deployable public `kwavers` surface; the CUDA runtime receives its required
  toolchain. That compile path exposed additional GPU PINN defects: transposed
  Coeus linear weights, invalid direct host access to device storage, silent
  construction fallback, incorrect activation metadata, and a constant
  uncertainty output. The repair preserves `[out, in]` weights and transposes
  only at Coeus matmul, uses backend `copy_to_host`, propagates errors, and
  bounds each prediction by affine propagation of the symmetric-int8 half-step
  error through 1-Lipschitz activations. Evidence tier: authoritative Actions
  failure logs; focused `pinn,gpu` compilation; four value-semantic
  regressions. A strict full-facade Clippy invocation also exposed 81 existing
  `kwavers-solver` findings; two in the touched quantization constructor are
  corrected with `to_vec`, leaving 79 outside this repair. CI now denies
  warnings for the clean `kwavers-core` crate and the deployable `kwavers`
  facade without dependency linting; the legacy solver findings remain a
  tracked ratchet debt. The public `full` package build additionally exposed
  stale Leto view/axis APIs and a direct WGPU 26 versus Hephaestus WGPU 30
  split. Kwavers now uses WGPU 30 throughout, maps former push-constant kernels
  to immediate data, propagates readback map/range errors, and uses Leto's
  fallible native views and axes. Evidence tier: locked full-package
  compilation plus the focused PINN value regressions. Residual: the new remote
  matrix remains pending. Workspace Rustdoc has a large pre-existing
  broken-link baseline concentrated in `kwavers-physics`; its CI command now
  proves all workspace crates document successfully without elevating that
  recorded legacy baseline, while the deployable `kwavers` facade stays
  warning-denied. This change corrects every public-facade link reached by the
  full build; the broader physics documentation ratchet remains a separate
  cleanup item. The first remote run also exposed a stale solver-validation
  workflow target: `validation::literature` existed on disk but was absent from
  the parent module, making Nextest reject the empty filter. The restored
  module edge compiled its latent nested-constant and Leto-index defects;
  nine value-semantic literature regressions now pass locally, including an
  exact Treeby snapshot and multi-time dimension rejection. Residual: the
  re-triggered remote matrix remains the final integration gate. Its
  architecture job also missed the fontconfig development package required by
  the public full-feature facade; it now matches the package contract used by
  the existing Cargo CI jobs. The subsequent strict facade check identifies
  three no-op Leto `Array3` conversions in the FD monitor; they are removed
  rather than suppressed. Residual: rerun the full remote matrix against the
  corrected PR head.

- Closed 2026-07-15: Kwavers had no repository-root Cargo-deny policy and CI
  resolved a crate-local policy instead of one scoped to the deployable Kwavers
  graph. The root policy now rejects unknown registries and Git sources, permits
  only Crates.io plus the required Consus and cutile origins, and carries exact
  license exceptions for `cuda-oxide@0.4.0`, `colored@3.1.1`, and
  `epaint@0.25.0`. Unused direct DICOM 0.8 workspace pins are removed; the lock
  now resolves RITK's DICOM 0.10 graph and fixed advisory releases. Evidence
  tier: Cargo-deny license/advisory/source checks and a locked all-feature
  `kwavers-core` compile plus 69 focused Nextest cases. Residual: `spin` 0.9.8
  (Flume 0.11.1) and `spin` 0.10.0 (Burn) are yanked but have no advisory; both
  remain warnings until their upstream constraints change. The broader
  all-feature workspace check remains independently blocked by two
  `kwavers-solver` generic-backend compile errors.

- Corrected 2026-07-15: analytic baseband removes `exp(j 2πf₀τ)` from every
  channel, so complex DAS must restore that phase after fractional interpolation
  and before coherent summation. `demodulate_rf_to_iq` owns the finite,
  Nyquist-valid real-RF transform; complex active DAS now requires its carrier,
  shares the real kernel's geometry/transmit-delay/apodization law, and
  rephases every physical path. The snapshot path delegates to that provider
  surface, deleting its duplicate FFT/Hilbert implementation. Evidence tier:
  a fractional-delay complex-I/Q value regression plus exact Nyquist rejection;
  `kwavers-analysis` Nextest passes 711/711 and warning-denied Clippy passes.
  Rustdoc completes with 57 existing unresolved links outside this contract,
  and doctests pass 1/1 with 21 intentionally ignored. Residual: slow-time
  frames must arise from explicit physical scatterer states; no API synthesizes
  arbitrary inter-frame phase progression.

- Closed 2026-07-15: active sector imaging needed the same plane-wave or
  virtual-source transmit arrival in its phantom RF and its receive-DAS
  reconstruction, but Kwavers exposed only monostatic RF and receive-only DAS.
  `TransmitWavefront` now validates three-dimensional plane and diverging
  events, their physical total path, and their spreading; the generalized DAS
  receives a validated transmit-delay vector for each image point. The existing
  `kwavers-transducer::ultrafast` APIs stay as specialized 2-D linear-array
  delay processors rather than becoming an invalid 3-D abstraction. Evidence
  tier: closed-form point-scatterer timing/spreading plus active-event
  localization and invalid-input regressions (12/12 + 6/6 locked Nextest).
  Residual: complex I/Q beamforming and slow-time ensemble synthesis require a
  separate provider contract; real DAS output is not treated as I/Q.

- Closed 2026-07-15: the fUS reference performed complex NumPy SVD and power
  reduction because Kwavers' existing clutter filter accepts only centred real
  slow-time matrices. `IqSvdClutterFilter` now owns the uncentred complex I/Q
  contract by exact realification; every complex singular mode maps to a paired
  real mode, so a rank cut removes `2k` singular values. The provider returns
  filtered I/Q and `Σ_t |I/Q|²` together. The stale Demené reference now names
  IEEE TMI 34(11), 2271-2285 (2015), DOI 10.1109/TMI.2015.2428634. Evidence
tier: realification identity, 3/3 locked value-semantic provider tests,
warning-denied Clippy, package Rustdoc with its 57 pre-existing unresolved links
outside the I/Q files, and LeoNeuro's independent 14/14 CPython boundary/
reference suite. Residual: this is the fUS I/Q domain contract; a generic public
complex SVD remains outside its justified scope.

- Closed 2026-07-15: the autocorrelation provider exposed velocity and
  normalized variance but forced each consumer that needed a Doppler power map
  to duplicate lag-zero reduction. `AutocorrelationEstimate` now owns all three
  maps from the same lag window, and the legacy tuple delegates to it. A
  coherent complex-I/Q oracle establishes `R1 = mean(I_n * conj(I_n+1))`, its
  negative velocity sign for positive phase progression, unit power, and zero
  coherence loss. Evidence tier: value-semantic provider test, focused locked
  Nextest 5/5, default-feature warning-denied Clippy, and Rustdoc. Residual:
  all-feature Clippy is independently blocked in `wgpu-hal` 30.0.0 by the
  workspace `windows` 0.61/0.62 Direct3D type split before this crate builds.

- Closed 2026-07-15: layered field evaluation kept its ordered segment path
  private to `RayleighIntegralSpec`, forcing LeoNeuro/Python focus control to
  reimplement or omit the same physics. `RayleighPropagationPath` now owns
  layer-order validation plus phase/attenuation accumulation, while
  `RayleighIntegralSpec` consumes that path directly. Evidence tier: exact
  segment/error contracts, locked 217/217 `kwavers-transducer` Nextest, and
  warning-denied Clippy/Rustdoc. Consumer wheel verification remains tracked in
  `KW-RAY-040` until the LeoNeuro gate completes.

- Closed 2026-07-14: Kwavers declared Gaia from Git and corrected it only via
  a workspace-root source patch. Cargo ignores that patch when a downstream
  package is isolated, causing LeoNeuro SemVer packaging to select historical
  Gaia without Eunomia. Gaia is now a direct Atlas path dependency and
  `kwavers-mesh` resolves it locally. Evidence tier: locked dependency-graph
  resolution, warning-denied package diagnostics, and 9/9 value-semantic
  package tests. Residual: isolated packaging now reaches Moirai's independent
  Themis Git-version mismatch; its owner is Moirai, not this dependency edge.

- Closed 2026-07-14: forward PSTD consumers unconditionally compiled clinical
  image I/O and registration because both `kwavers-physics` and
  `kwavers-solver` declared `kwavers-imaging` unconditionally. ADR-036 makes
  the clinical surface explicit, retains only pure thermoelastic material laws
  outside it, and updates in-workspace clinical consumers. Locked Nextest passes
  1,554/1,554 without `clinical-imaging`, 1,710/1,710 with it, and 29/29 in
  LeoNeuro; reverse dependency resolution finds no `ritk-filter` package in
  Leo's active graph. Evidence tier: manifest/source audit, value-semantic
  regressions, locked feature-matrix integration, and active graph proof.

- Closed 2026-07-14: KWaveArray clamped arbitrarily distant source samples to
  the nearest grid boundary before evaluating sinc, creating false source
  support. The BLI mapper now accepts a sample only when its finite stencil
  window overlaps the grid. Evidence tier: exact clipped-versus-distant source
  regression, Leo focus-delay integration, and locked package execution.

- Open documentation baseline 2026-07-14: `cargo doc -p kwavers-physics
  --all-features --no-deps` emits 575 unresolved intra-doc-link warnings in
  pre-existing Physics modules. The clinical-imaging boundary files emit none;
  the package documentation gate is therefore not warning-clean. Evidence tier:
  all-feature Rustdoc execution. Re-open trigger: a focused Rustdoc correction
  increment that rewrites the unresolved links and bracketed unit annotations.

- Closed 2026-07-14: KWaveArray could rasterize finite discs but not the
  independently driven planar annular sectors already accepted by the Rayleigh
  provider. `PlanarApertureGeometry` is now the shared validated geometry;
  equal-area sector sampling and normalized BLI stencils preserve analytical
  source area and per-sector signals. Evidence tier: type-level geometry reuse,
  analytical area equality, value-semantic source-matrix regression, Clippy,
  215/215 Nextest, doctests, and Rustdoc.

- Closed 2026-07-14: the finite-aperture provider exposed only complete
  circular pistons, forcing hybrid C/D Fresnel cells either to collapse
  independently driven electrodes or approximate annular sectors downstream.
  ADR-035 replaces that boundary with one oriented planar-aperture kernel over
  disk and annular-sector bounds. Evidence tier: analytical area equality,
  coherent sector-superposition differential equality, retained disk oracles,
  warning-denied Clippy, 214/214 Nextest, doctests, and warning-clean Rustdoc.

- Closed 2026-07-13: downstream finite-aperture propagation duplicated the
  circular-piston diffraction factor at each surface point and used equal
  point weights that did not integrate disk area. `kwavers-transducer` now owns
  the baffled Rayleigh first integral with Gauss-Legendre squared-radius and
  periodic azimuth quadrature. Evidence tier: analytical on-axis and disk-area
  equalities, Bessel far-field differential validation, geometric invariance
  tests, bounded-work rejection, warning-denied package diagnostics, and
  212/212 package tests.
- Residual evidence limit: `cargo-semver-checks` cannot compare
  `kwavers-transducer` because no registry release exists. The additive public
  surface is classified [minor] by source review; no machine semver proof is
  claimed.

- Closed 2026-07-13: the workspace-wide Leto `ndarray-compat` feature concealed
  21 same-type conversions across beamforming, PAM, BEM, FEM, FDTD, PSTD, and
  thermal coupling. The feature and conversions are deleted; matrix inversion
  uses `leto-ops` while Kwavers retains its eigendecomposition and complex-solve
  contracts. Evidence tier: compile-time integration, warning-denied locked
  package Clippy, and 908/908 value-semantic package tests with one existing
  skip and no test reaching the 30-second threshold.

- Closed 2026-07-12: `nd_to_leto1` and `leto1_to_nd1` duplicated Leto's owned
  rank-1 ndarray conversions and coupled ten PyO3 consumers to a local
  compatibility module. All consumers now invoke the provider conversions
  directly. Evidence tier: compile-time integration, warning-denied package
  Clippy, focused package tests, and exact static residual audit.

- Closed 2026-07-11: six rank-specific PyO3 complex converters copied complete
  arrays between identical `eunomia::Complex64` aliases. The converters and all
  24 runtime call sites are deleted; the Leto/NumPy boundary remains explicit.
  Follow-through also deletes the 29 stale same-type conversions exposed by
  package Clippy. Evidence tier: compile-time type identity, package
  compilation, static source audit, warning-denied package Clippy, and 6/6
  focused package tests.

- Closed 2026-07-11: `kwavers-boundary` duplicated Leto indexed traversal in a
  consumer-owned `parallel` module and retained an otherwise-unused direct
  Moirai dependency. CPML, smoothing, and adaptive coupling now use the
  canonical const-generic Leto operations, and the duplicate module and direct
  dependency are deleted. Evidence tier: compile-time integration,
  value-semantic package tests, warning-denied Clippy, and static source audit.

- Closed 2026-07-10: `kwavers-physics` was the sole direct `ndarray-npy`
  consumer. `consus-npy` now owns bounded typed NPY/NPZ parsing, and the
  consumer constructs Leto arrays directly from the owned payload. Evidence
  tier: NumPy-generated provider fixture, compile-time typed integration, five
  value-semantic consumer tests, warning-denied library Clippy, and 39 focused
  adjacent physics regressions.

- Closed 2026-07-10: six production documentation sites described removed
  Rayon, Tokio, nalgebra, or ndarray execution. Their contracts now match the
  live Moirai/Leto implementations. Evidence tier: static source audit.

Module-by-module audit of physics/numerics implementations, marking items for
future revision. Compact + link-navigable per agent-artifact policy; closed
history (pre-2026-05-29) was pruned from `docs/` during the workspace-split
docs cleanup and remains recoverable from git history.

**Method:** four parallel read-only module audits (2026-05-31) over `solver/`,
`physics/`, `clinical/`+`domain/`, `analysis/`+`math/`+`core/`+`gpu/`. Findings
below are candidate gaps for revision; severity tiers per `CLAUDE.md` integrity
policy. Items tagged **[verify]** are pattern-match suspicions that MUST be
confirmed against the code (and ideally a literature reference) before any fix —
do not assert an unconfirmed physics error.

### Atlas provider migration residuals (2026-07-01)

- **kwavers-grid duplicate Leto compatibility surface - RESOLVED [arch].**
  Deleted `compat.rs`, redundant `_leto` forwarding APIs, duplicate wave-number
  names, and allocating Leto-to-Leto k-space conversions. Canonical grid
  operations import Leto directly and accept borrowed views where appropriate.
  Evidence tier: compile-time integration, empirical unit tests, and static
  source audit. All-target grid clippy passes; grid nextest passes 38/38;
  doctests and warning-clean docs pass; `kwavers-physics` library check passes.
  The broader test frontier remains in unrelated `kwavers-math` and
  `kwavers-solver` Leto test/API conversions.
- **kwavers-analysis signal-processing FFT facade holdouts - RESOLVED [patch].**
  Narrowband legacy analytic-baseband and windowed STFT snapshot extraction now
  route through Apollo 1-D FFT APIs over Leto buffers instead of importing FFT
  execution or complex types from `kwavers_math::fft`. The covariance-facing
  ndarray boundary remains `num_complex` for this slice, with explicit
  conversion from Apollo complex scratch output. Evidence tier: compile-time
  integration, focused empirical tests, and static source audit. Direct
  `rustfmt --check` passed for touched snapshot files; `rustup run nightly
  cargo check -p kwavers-analysis` passed; `rustup run nightly cargo nextest
  run -p kwavers-analysis narrowband snapshots stft baseband` passed 30/30;
  scoped `rg` found no `kwavers_math::fft` imports in
  `kwavers-analysis/src/signal_processing`. Residual: broader
  `kwavers-analysis` ndarray and `num_complex` boundary cleanup remains.
- **kwavers-analysis Doppler 1-D FFT facade holdouts - RESOLVED [patch].**
  Continuous-wave, pulsed-wave, and Welch spectral Doppler FFT execution now
  route through Apollo's 1-D real/complex FFT APIs over Leto buffers instead of
  importing FFT execution and shift utilities from `kwavers_math::fft`.
  Evidence tier: compile-time integration, focused empirical tests, and static
  source audit. Direct `rustfmt --check` passed for touched Doppler files;
  `rustup run nightly cargo check -p kwavers-analysis` passed; `rustup run
  nightly cargo nextest run -p kwavers-analysis doppler continuous_wave
  pulsed_wave spectral` passed 49/49; scoped `rg` found no
  `kwavers_math::fft` import in migrated Doppler files. Residual: remaining
  direct `kwavers_math::fft` consumers in `kwavers-analysis` are narrowband
  snapshot extraction only.
- **kwavers-analysis PAM 1-D FFT facade holdouts - RESOLVED [patch].**
  PAM processor spectrum computation and delay-and-sum peak frequency
  estimation now route through Apollo's 1-D real FFT over Leto buffers instead
  of importing FFT execution from `kwavers_math::fft`. `kwavers-analysis`
  declares Apollo directly for this native execution path. Evidence tier:
  compile-time integration, focused empirical tests, and static source audit.
  Direct `rustfmt --check` passed for touched PAM files; `rustup run nightly
  cargo check -p kwavers-analysis` passed; `rustup run nightly cargo nextest
  run -p kwavers-analysis pam delay_and_sum` passed 18/18; scoped `rg` found no
  `kwavers_math::fft` import in the migrated PAM files. Residual: remaining
  direct `kwavers_math::fft` consumers in `kwavers-analysis` are Doppler
  continuous/pulsed/spectral paths and narrowband snapshot extraction.
- **kwavers-analysis analytic-signal FFT facade holdouts - RESOLVED [patch].**
  B-mode envelope detection and time-domain phase-coherence analytic-signal
  construction now route through `kwavers-signal`'s Apollo-backed Hilbert
  transform instead of `kwavers_math::fft::analytic_signal_1d`. Evidence tier:
  compile-time integration, focused empirical tests, and static source audit.
  Direct `rustfmt --check` passed for touched files; `rustup run nightly cargo
  check -p kwavers-analysis` passed; `rustup run nightly cargo nextest run -p
  kwavers-analysis b_mode coherence` passed 51/51; scoped `rg` found no
  `kwavers_math::fft` or `analytic_signal_1d` in the migrated files.
  Residual: remaining direct `kwavers_math::fft` consumers in `kwavers-analysis`
  are Doppler continuous/pulsed/spectral paths, PAM processor and
  delay-and-sum beamform paths, and narrowband snapshot extraction.
- **kwavers-signal 1-D FFT facade holdouts - RESOLVED [patch].**
  `kwavers-signal` analytic-signal Hilbert transforms and frequency-domain
  filtering now call Apollo APIs over Leto buffers directly instead of
  importing execution from `kwavers_math::fft`. The public analytic-signal
  boundary remains `num_complex` for this slice, with explicit conversion from
  Apollo complex output. `kwavers-math` remains in `kwavers-signal` for non-FFT
  window coefficients. Evidence tier: compile-time integration, focused
  empirical tests, and static source audit. `rustup run nightly cargo fmt
  --package kwavers-signal --check` passed; `rustup run nightly cargo check -p
  kwavers-signal` passed; `rustup run nightly cargo nextest run -p
  kwavers-signal analytic frequency_filter` passed 13/13; scoped `rg` found no
  `kwavers_math::fft` imports in the touched signal files. Residual: remaining
  direct `kwavers_math::fft` consumers are in analysis, physics, solver
  reconstruction/FWI, and wider 3-D solver surfaces.
- **kwavers-solver PSTD axisymmetric 2-D FFT facade holdout - RESOLVED [patch].**
  `forward::pstd::propagator::axisymmetric` now routes real forward and complex
  inverse 2-D FFT execution through Apollo APIs over Leto buffers instead of
  importing the `kwavers_math::fft` plan/cache facade. The ndarray
  `num_complex` working buffers remain the current PSTD storage boundary, with
  explicit conversion at the Apollo scratch edge. Evidence tier: compile-time
  integration, focused empirical tests, and static source audit. `rustup run
  nightly cargo fmt --package kwavers-solver --check` passed; `rustup run
  nightly cargo check -p kwavers-solver` passed; `rustup run nightly cargo
  nextest run -p kwavers-solver axisymmetric_apollo` passed 2/2; scoped `rg`
  found no `kwavers_math::fft` import in the axisymmetric module. Residual:
  wider solver 3-D FFT facade users, shift utilities, and `num_complex`-typed
  PSTD storage boundaries remain separate migration slices.
- **kwavers-solver line-reconstruction 2-D FFT facade holdout - RESOLVED [patch].**
  `inverse::reconstruction::photoacoustic::line_reconstruction` now calls
  Apollo's 2-D complex FFT APIs over Leto buffers directly instead of importing
  execution from `kwavers_math::fft`. The interpolation/scaling math remains
  `num_complex` at the current ndarray boundary, with one private conversion
  SSOT for Apollo scratch buffers. Evidence tier: compile-time integration,
  focused empirical tests, and static source audit. `rustup run nightly cargo
  fmt --package kwavers-solver --check` passed; `rustup run nightly cargo check
  -p kwavers-solver` passed; `rustup run nightly cargo nextest run -p
  kwavers-solver line_reconstruction` passed 4/4; scoped `rg` showed only
  Apollo FFT execution calls in the line-reconstruction module. Residual: PSTD
  axisymmetric 2-D FFT plan/cache usage and wider solver 3-D FFT facade users
  remain separate migration slices.
- **kwavers-solver fast-nearfield 2-D FFT facade holdout - RESOLVED [patch].**
  `analytical::transducer::fast_nearfield` field computation now calls
  Apollo's 2-D complex FFT APIs over Leto buffers directly instead of importing
  execution from `kwavers_math::fft`. The FNM public/storage boundary remains
  `num_complex` for this slice because cached Green spectra and ndarray-backed
  field arrays still use that representation. Evidence tier: compile-time
  integration, focused empirical tests, and static source audit. `rustup run
  nightly cargo fmt --package kwavers-solver --check` passed; `rustup run
  nightly cargo check -p kwavers-solver` passed; `rustup run nightly cargo
  nextest run -p kwavers-solver fast_nearfield` passed 6/6; scoped `rg` showed
  only Apollo FFT execution calls in the fast-nearfield module. Residual:
  wider solver 3-D FFT facade users remain separate migration slices.
- **kwavers-solver HAS 2-D FFT facade holdout - RESOLVED [patch].**
  `forward::nonlinear::hybrid_angular_spectrum::diffraction` now calls Apollo's
  2-D complex FFT APIs over Leto buffers directly instead of importing
  execution from `kwavers_math::fft`. Evidence tier: compile-time integration,
  focused empirical tests, and static source audit. `rustup run nightly cargo
  fmt --package kwavers-solver --check` passed; `rustup run nightly cargo check
  -p kwavers-solver` passed; `rustup run nightly cargo nextest run -p
  kwavers-solver hybrid_angular_spectrum` passed 18/18; scoped `rg` showed only
  Apollo FFT calls in the HAS cone. Residual: PSTD axisymmetric 2-D FFT
  plan/cache usage and wider solver 3-D FFT facade users remain separate
  migration slices.
- **kwavers-solver KZK 2-D FFT facade holdouts - RESOLVED [patch].** KZK
  angular-spectrum, real parabolic, and complex parabolic 2-D diffraction
  scratch paths now use direct Apollo FFT APIs over Leto buffers instead of
  importing execution from `kwavers_math::fft`. The complex-field public
  boundary remains `num_complex` for this slice, with explicit copy-in/copy-out
  at the leaf scratch boundary. Evidence tier: compile-time integration,
  focused empirical tests, and static source audit. `rustup run nightly cargo
  fmt --package kwavers-solver --check` passed; `rustup run nightly cargo check
  -p kwavers-solver` passed; `rustup run nightly cargo nextest run -p
  kwavers-solver kzk` passed 49/49; scoped `rg` found no `kwavers_math::fft`
  imports in the touched KZK 2-D diffraction files. Residual: PSTD axisymmetric
  2-D FFT plan/cache usage and wider solver 3-D FFT facade users remain
  separate migration slices.
- **kwavers-solver KZK 1-D FFT facade holdouts - RESOLVED [patch].** KZK
  absorption, nonlinear spectral differentiation, and finite-difference
  diffraction temporal complex 1-D FFT scratch paths now use direct Apollo APIs
  over Leto buffers instead of importing 1-D FFT execution from
  `kwavers_math::fft`. Evidence tier: compile-time integration, focused
  empirical tests, and static source audit. `rustup run nightly cargo fmt
  --package kwavers-solver --check` passed; `rustup run nightly cargo check -p
  kwavers-solver` passed; `rustup run nightly cargo nextest run -p
  kwavers-solver kzk` passed 49/49; scoped `rg` showed only Apollo 1-D FFT calls
  in the touched KZK files. Residual: KZK 2-D angular/parabolic diffraction and
  wider solver 2-D/3-D FFT facade users remain separate migration slices.
- **kwavers warning/example cleanup and Apollo complex boundary - RESOLVED [patch].**
  `kwavers` all-target warnings from property/comparative tests and the GPU
  beamforming benchmark were removed, the benchmark CPU path now uses the
  existing CPU helper instead of a duplicate inline loop, touched examples/tests
  are clippy-clean, and inverse-reconstruction Apollo 1-D FFT outputs convert
  explicitly at the remaining `num_complex` facade boundary. Evidence tier:
  compile-time integration and focused empirical tests. `rustup run nightly
  cargo check -p kwavers --examples` passed; `rustup run nightly cargo check
  -p kwavers --all-targets` passed; `rustup run nightly cargo clippy -p
  kwavers --all-targets --no-deps -- -D warnings` passed; `rustup run nightly
  cargo nextest run -p kwavers-solver photoacoustic --status-level fail
  --no-fail-fast` passed 10/10; `rustup run nightly cargo nextest run -p
  kwavers --test property_based_tests --test comparative_solver_tests --test
  nonlinear_physics_tests --test test_pstd_kwave_comparison --test
  imaging_literature_validation --status-level fail --no-fail-fast` passed
  46/46; `rustup run nightly cargo run -p xtask -- burn-migration-audit`
  passed with 0 Burn manifest deps and 5 approved non-solver source residuals.
  Residual: package-wide `rustup run nightly cargo fmt -p kwavers --check`
  remains blocked by pre-existing formatting drift outside this slice in
  `crates/kwavers/examples/focused_water_tank_common/simulation.rs`,
  `crates/kwavers/examples/pstd_fdtd_comparison.rs`,
  `crates/kwavers/src/theranostic/monitor/fd.rs`,
  `crates/kwavers/tests/pstd_finite_window_born.rs`, and
  `crates/kwavers/tests/quick_comparative_test.rs`; touched files were
  formatted with file-scoped `rustfmt`.
- **kwavers-solver inverse 1-D FFT facade holdouts - RESOLVED [patch].**
  Photoacoustic filtering/Fourier reconstruction and seismic envelope-phase
  Hilbert 1-D FFT call sites now use Apollo's Leto-native real FFT APIs and the
  direct Apollo complex inverse API instead of
  `kwavers_math::fft::{fft_1d_array, ifft_1d_array}`. Evidence tier:
  compile-time integration, focused empirical tests, and static source audit.
  `rustup run nightly cargo fmt --package kwavers-solver --check` passed;
  `rustup run nightly cargo check -p kwavers-solver` passed; `rustup run
  nightly cargo nextest run -p kwavers-solver photoacoustic` passed 10/10;
  `rustup run nightly cargo nextest run -p kwavers-solver envelope misfit phase`
  passed 34/34; scoped `rg` found no `fft_1d_array`/`ifft_1d_array` calls under
  `crates/kwavers-solver/src`. Residual: broader 2-D/3-D
  `kwavers_math::fft` facade users remain separate because the current 3-D
  solver facade still owns `num_complex`-typed PSTD and k-space APIs.
- **Burn-to-Coeus migration guard - RESOLVED [patch].** Kwavers now has a
  focused `xtask` Burn-surface audit matching the RITK allowlist pattern:
  `burn-migration-audit`, `refresh-burn-allowlist`, and
  `xtask/burn_surface.allowlist`. The CI legacy migration workflow runs a
  separate Burn audit job so direct Burn drift fails independently of the
  broader Atlas legacy audit. Evidence tier: static source/manifest audit,
  unit tests, and CI configuration. `rustup run nightly cargo fmt -p xtask
  --check` passed; `rustup run nightly cargo nextest run -p xtask burn_audit
  --status-level fail --no-fail-fast` passed 2/2; `rustup run nightly cargo
  run -p xtask -- burn-migration-audit` passed with 0 Burn manifest deps and
  5 approved non-solver source residuals. The Burn allowlist contains no solver
  PINN entries.
- **kwavers-physics acoustic heat-source direct ndarray/Rayon edge - RESOLVED [patch].**
  `acoustics::conservation::heat::acoustic_heat_source` now routes through the
  crate-local Moirai-backed `parallel` traversal SSOT instead of direct
  ndarray/Rayon `Zip::par_for_each`. `parallel.rs` now owns the missing
  `zip_mut_five_refs` arity so heat-source output consumes pressure, velocity
  magnitude, density, sound speed, and absorption in one pass. Evidence tier:
  compile-time integration, focused empirical tests, and static source audit.
  `rustup run nightly cargo check -p kwavers-physics --lib` passed; `rustup run
  nightly cargo nextest run -p kwavers-physics heat_source --status-level fail`
  passed 9/9 with 1704 skipped; scoped `rg` found no
  `Zip|par_for_each|rayon` hits in
  `crates/kwavers-physics/src/acoustics/conservation/heat.rs`.
  Residual: broader solver/physics direct `.par_for_each` holdouts are now 49
  sites outside RTM inherent, sonogenetics, and acoustic heat-source traversal.
  Package clippy remains blocked before this package by local dependency
  `ritk-transform` Burn `Module` derive errors in the concurrent RITK provider
  migration diff.
- **kwavers-physics focused direct ndarray/Rayon edge - RESOLVED [patch].**
  `acoustics::therapy::sonogenetics` gating and volumetric ARF field traversal
  plus heterogeneous skull-mask property assignment now route through the
  crate-local Moirai-backed `parallel` traversal SSOT instead of direct
  ndarray/Rayon `Zip::par_for_each` or duplicate one-input helper calls.
  `parallel.rs` now owns the missing `zip_mut_ref` and `zip_two_mut_four_refs`
  arities so one-input updates and ARF's intensity/body-force update share the
  traversal SSOT.
  Evidence tier:
  compile-time integration, focused empirical tests, and static source audit.
  `rustup run nightly cargo check -p kwavers-physics --lib` passed; `rustup run
  nightly cargo nextest run -p kwavers-physics sonogenetics --status-level
  fail --no-fail-fast` passed 53/53 with 1660 skipped; `rustup run nightly
  cargo nextest run -p kwavers-physics skull --status-level fail
  --no-fail-fast` passed 51/51 with 1662 skipped; scoped `rg` found no
  `Zip|par_for_each|rayon` hits under
  `crates/kwavers-physics/src/acoustics/therapy/sonogenetics` or the touched
  skull mask file.
  Residual: broader solver/physics direct `.par_for_each` holdouts are now 49
  sites outside RTM inherent, sonogenetics, skull mask, and acoustic
  heat-source traversal.
- **kwavers-solver RTM inherent direct ndarray/Rayon edge - RESOLVED [patch].**
  `inverse::reconstruction::seismic::rtm::inherent` now routes wavefield
  stencils, decimated wavefield interpolation, source illumination, Laplacian
  filtering, image post-processing, and imaging-condition updates through the
  private Moirai-backed `parallel::for_each_view_mut` strided-view seam instead
  of direct ndarray/Rayon `Zip::par_for_each`. Evidence tier: compile-time
  integration, focused empirical tests, and static source audit.
  `rustup run nightly cargo check -p kwavers-solver --lib` passed; `rustup run
  nightly cargo nextest run -p kwavers-solver rtm --status-level fail` passed
  10/10 with 916 skipped; scoped `rg` found no `Zip|par_for_each|rayon` hits
  under `crates/kwavers-solver/src/inverse/reconstruction/seismic/rtm/inherent`.
  Residual: broader solver/physics direct ndarray/Rayon holdouts remain outside
  RTM inherent, sonogenetics, and acoustic heat-source traversal: 49
  `.par_for_each` sites across
  `crates/kwavers-solver/src/forward/elastic/swe/stress/divergence.rs`,
  `crates/kwavers-solver/src/forward/elastic/swe/integration/integrator/mod.rs`,
  `crates/kwavers-solver/src/forward/nonlinear/westervelt_spectral/spectral.rs`,
  `crates/kwavers-solver/src/forward/nonlinear/kuznetsov/{workspace,spectral,numerical,nonlinear,diffusion,operator_splitting/mod}.rs`,
  `crates/kwavers-solver/src/forward/nonlinear/kuznetsov/solver/{rhs,model_impl}.rs`,
  `crates/kwavers-solver/src/forward/pstd/extensions/{elastic,elastic_orchestrator/pml/mod}.rs`,
  `crates/kwavers-solver/src/multiphysics/fluid_structure/{interface,solver/struct_impl}.rs`,
  `crates/kwavers-physics/src/acoustics/mechanics/acoustic_wave/nonlinear/{wave_model,numerical_methods/spectral/mod,numerical_methods/nonlinear_term}.rs`,
  and `crates/kwavers-physics/src/acoustics/mechanics/cavitation/damage/model.rs`.
  Package fmt is still blocked by pre-existing formatting drift in
  `crates/kwavers-solver/src/forward/fdtd/electromagnetic/tests.rs`; current
  package clippy passes with `rustup run nightly cargo clippy -p kwavers-solver
  --lib --no-deps -- -D warnings` after the Atlas provider graph refresh.
- **Phase-1A kwavers-math numeric SSOT pilot - RESOLVED [patch].** The `kwavers_math::linear_algebra::NumericOps<T>` trait moved from `num_traits::{Float, NumCast, Zero}` to `eunomia::RealField` + `NumericElement::ZERO`. `crates/kwavers-math/Cargo.toml` declares `eunomia = { workspace = true }` while retaining `num-traits` (csr.rs only); super-traits `Clone + Zero` + vestigial `NumCast` are dropped to `Copy + PartialOrd`; the six method bodies use `T::ZERO` instead of `T::zero()`; `max_abs` uses a `PartialOrd`-driven fold because `eunomia::RealField` does not propagate a `max` method. Evidence tier: focused compile validation plus kwavers xtask lexical audit. Verification: `cargo build -p kwavers-math` exits 0; `cargo run -p xtask -- legacy-migration-audit` no longer lists `numeric_ops.rs` in the source-legacy per-file output.
- **Atlas extension: eunomia Complex64 SSOT for csr.rs - OPENED [arch].** Phase-1A closed `kwavers-math::linear_algebra::numeric_ops` but the `num_complex::Complex64 → CsrScalar` impl in `linear_algebra::sparse::csr.rs` cannot drop `num_traits::Zero` because `eunomia::NumericElement` and `eunomia::FloatElement` are `private::Sealed`. Atlas extension request `CR-EUNOMIA-COMPLEX`: unsealed `eunomia::Scalar` supertrait, OR a native `eunomia::Complex` with `magnitude`/`norm` derivations usable from `CsrScalar::magnitude`. Until that lands, `kwavers-math/Cargo.toml` carries both `num-traits` (csr.rs only) and `eunomia` (numeric_ops + downstream); the xtask audit tracks csr.rs as the lone remaining source-legacy entry under `crates/kwavers-math/src/linear_algebra/sparse`.
- **GPU backend provider boundary - RESOLVED [patch].** The solver-owned
  compute backend trait now models GPU provider identity explicitly through
  `GpuProvider` on `BackendType::GPU`, and the existing WGPU leaf backend
  reports `GpuProvider::Wgpu`. `kwavers-core` no longer depends on WGPU or
  exposes `kwavers-core/gpu`; WGPU buffer-map errors are mapped inside
  `kwavers-gpu`, preserving the foundation crate as provider-neutral. Evidence
  tier: compile-time validation plus focused trait tests; `cargo check -p
  kwavers-core --all-features`, `cargo check -p kwavers-solver`, `cargo check
  -p kwavers-gpu --features gpu`, and `cargo nextest run -p kwavers-solver
  backend_surface_tests` pass. Follow-up focused GPU backend verification also
  passes: `cargo check -p kwavers --features gpu`, `cargo check -p
  moirai-core --tests`, and `cargo nextest run -p kwavers-gpu --features gpu
  backend --no-fail-fast` (31/31).
- **Hephaestus GPU adapter - RESOLVED [patch].** `kwavers-gpu` now depends on
  `hephaestus-core` and `hephaestus-wgpu`, `GPUBackend` is generic over
  `GpuComputeProvider`, and the provider trait now requires its associated
  device to satisfy the Kwavers `GpuDeviceProvider` trait rather than
  Hephaestus capability queries alone. `GpuDeviceProvider` owns the acquisition
  label, default preference, optional features, and required limits, so
  `GpuProviderContext<P>` does not impose WGPU's `ShaderF64` and WGSL
  workgroup policy on CUDA providers. `GPUBackend<P>::provider()` exposes the
  concrete provider generically; raw WGPU device/queue access remains
  restricted to the `WgpuDevice` specialization required by current WGSL
  kernels. The GPU compute contract is now factored into
  `GpuKernelProvider`, `ElementWiseMultiplyProvider`, and
  `SpatialDerivativeProvider`, leaving `GpuComputeProvider` as the public
  composite only for backends with all required real operation kernels.
  The WGSL pipeline compiler/executor is explicitly named
  `WgpuPipelineManager`, so CUDA-provider paths no longer see a
  backend-neutral pipeline-manager type name.
  The raw WGPU command-helper surface is explicitly named
  `WgpuComputeCommands`, so bind-group layout and command-encoder ownership no
  longer appear as provider-neutral GPU compute.
  `WaveEquationGpu<P>` and `AcousticFieldKernel<P>` now carry
  `AcousticFieldProvider`, which is bound to the shared
  `GpuKernelProvider`/`GpuProviderBackend` stack. The current WGPU acoustic
  provider stores `GpuProviderContext<WgpuDevice>`, so future real CUDA
  acoustic kernels satisfy the same Hephaestus-backed provider trait instead
  of a standalone acoustic trait.
  Solver-facing documentation now points to provider-generic `GPUBackend<P>`
  and identifies the legacy elastic SWE GPU file as a performance model rather
  than a real WGPU/CUDA dispatch path.
  `WgpuComputeProvider` no longer fabricates a fixed 4 GiB memory value or
  fixed 5 TFLOP/s peak throughput; memory is derived from the acquired
  Hephaestus provider limits, and peak throughput remains `0.0` until a
  provider exposes enough topology data to derive it. The generic
  `GpuComputeProvider::estimate_performance` default also no longer fabricates
  a problem-size speedup curve; it reports the provider peak value unless a
  provider supplies a real benchmark-backed override.
  `WgpuComputeProvider` also reports `supports_fft = false` because the
  solver-owned `ComputeBackend` trait does not expose FFT operations; Apollo
  remains the GPU FFT owner through `kwavers_math::fft::gpu_fft`.
  Evidence tier: type-level/compile-time validation plus focused
  backend tests; `cargo check -p kwavers-gpu --features gpu` passes, `cargo
  check -p kwavers-gpu --features cuda-provider --offline` passes, and `cargo
  nextest run -p kwavers-gpu --features cuda-provider --status-level fail
  --no-fail-fast --offline` passes 102/102 with 1 skipped. Follow-up
  verification on 2026-07-03: `rustup run nightly cargo check -p kwavers-gpu
  --features gpu`, `rustup run nightly cargo check -p kwavers-gpu --features
  cuda-provider`, `rustup run nightly cargo clippy -p kwavers-gpu --features
  gpu --lib -- -D warnings`, and the focused CUDA-provider operation-trait
  nextest passed 4/4. The focused CUDA-provider WGPU pipeline/provider naming
  nextest passed 5/5. The focused CUDA-provider wave-equation/acoustic
  provider nextest passed 4/4. The focused CUDA-provider WGPU command-helper
  naming nextest passed 3/3. Follow-up Tokio-removal evidence: `kwavers-gpu`
  source and manifest no longer name Tokio, `cargo check -p kwavers-gpu
  --features cuda-provider` passes, `cargo clippy -p kwavers-gpu --features
  cuda-provider --all-targets -- -D warnings` passes, and the focused
  CUDA-provider non-hardware provider/multi-GPU/acoustic nextest selection
  passes 11/11. Follow-up provider-wrapper evidence: the acoustic-field WGPU
  provider now stores `GpuDevice<WgpuDevice>` and acquires through
  `GpuDeviceProvider`; `rustup run nightly cargo fmt -p kwavers-gpu --check`,
  `rustup run nightly cargo check -p kwavers-gpu --features cuda-provider`,
  `rustup run nightly cargo clippy -p kwavers-gpu --features cuda-provider
  --all-targets -- -D warnings`, and `rustup run nightly cargo nextest run -p
  kwavers-gpu --features cuda-provider acoustic provider device
  --status-level fail --no-fail-fast` pass 42/42. Follow-up provider metadata
  evidence: `rustup run nightly cargo fmt -p kwavers-gpu --check` passes,
  `rustup run nightly cargo check -p kwavers-gpu --features gpu` passes, and
  `rustup run nightly cargo nextest run -p kwavers-gpu --features gpu
  -E "test(test_performance_estimation) or
  test(limit_bytes_to_usize_preserves_representable_values) or
  test(limit_bytes_to_usize_saturates_unrepresentable_values)" --status-level
  fail --no-fail-fast` passes 3/3. Follow-up FFT capability evidence:
  `rustup run nightly cargo fmt -p kwavers-gpu -p kwavers-solver --check`
  passes, `rustup run nightly cargo check -p kwavers-gpu --features gpu`
  passes, focused `kwavers-gpu` capability/performance nextest passes 2/2,
  and `rustup run nightly cargo nextest run -p kwavers-solver
  backend_surface_tests --status-level fail --no-fail-fast` passes 3/3.
  Residual: the broader GPU-hardware acquisition nextest
  selection was interrupted after it produced no result for several minutes
  beyond compilation, so hardware-device acquisition tests need a separate
  isolation pass. Follow-up evidence on 2026-07-04: `rustup run nightly cargo
  check -p kwavers-gpu --features cuda-provider` passes, the
  `kwavers-simulation` GPU PSTD adapter consumes Leto CPML profiles directly,
  and the WGPU auto-device adapter explicitly selects
  `WgpuPstdStateProvider` at the construction boundary. The earlier
  `kwavers_analysis::visualization::stream` module blocker is closed by the
  provider-owned stream module and focused top-level stream nextest pass.
  Residual: top-level Tokio removal is now narrowed to the
  `real_time_3d_beamforming` example's raw WGPU async constructor edge.
  Follow-up provider-trait evidence on 2026-07-04: `hephaestus-cuda` now
  implements the shared unary/binary storage-kernel traits, and Kwavers'
  provider boundary re-verifies without a downstream CUDA helper. `rustup run
  nightly cargo check -p kwavers-gpu --features gpu --all-targets`, `rustup run
  nightly cargo check -p kwavers-gpu --features cuda-provider --all-targets`,
  `rustup run nightly cargo clippy -p kwavers-gpu --features gpu --all-targets
  --no-deps -- -D warnings`, `rustup run nightly cargo clippy -p kwavers-gpu
  --features cuda-provider --all-targets --no-deps -- -D warnings`, focused
  `kwavers-gpu --features gpu provider` nextest (32/32), and focused
  `kwavers-gpu --features cuda-provider provider` nextest (39/39) pass.
  Follow-up CUDA operation-family evidence on 2026-07-04:
  `CudaElementWiseProvider` implements `ElementWiseMultiplyProvider` through
  Hephaestus CUDA `binary_elementwise_into::<MulOp, f32>` and provider-native
  `leto::Array3<f32>` slices while remaining outside `GpuComputeProvider`
  until the remaining operation traits have CUDA kernels. The realtime Hilbert
  FFT path uses `kwavers_math::fft::{fft_1d_complex_slice_inplace,
  ifft_1d_complex_slice_inplace}` instead of Apollo's Leto-native plan API.
  `rustup run nightly cargo fmt -p kwavers-math -p kwavers-gpu --check`,
  `rustup run nightly cargo check -p kwavers-gpu --features cuda-provider`,
  `rustup run nightly cargo check -p kwavers-math -p kwavers-gpu --features
  kwavers-gpu/cuda-provider`, `rustup run nightly cargo clippy -p
  kwavers-gpu --features cuda-provider --lib --no-deps -- -D warnings`, and
  focused `kwavers-gpu --features cuda-provider provider elementwise
  realtime` nextest (52/52) pass.
  Follow-up top-level feature evidence on 2026-07-04: `kwavers/cuda-provider`
  and `kwavers/cuda-runtime` now forward to the same `kwavers-gpu` Hephaestus
  CUDA provider features, and stale boundary wording no longer describes GPU as
  WGPU-only or CPU as Rayon-only. `rustup run nightly cargo fmt -p kwavers -p
  kwavers-solver --check`, `rustup run nightly cargo check -p kwavers
  --features cuda-provider`, `rustup run nightly cargo clippy -p kwavers
  --features cuda-provider --lib --no-deps -- -D warnings`, focused `rustup
  run nightly cargo nextest run -p kwavers-gpu --features cuda-provider
  provider --status-level fail --no-fail-fast` (44/44), and `cargo tree -p
  kwavers-gpu --features cuda-provider --depth 1` provider-edge audit pass.
  Residual: CUDA compute remains limited to real operation-family providers
  until the remaining CUDA kernels and WGPU/CUDA differential tests land.
  Follow-up top-level stream evidence on 2026-07-04: the stream visualization
  test now runs through blocking stream/pipeline entry points and
  provider-native `leto::Array3<f32>` frames without Tokio macros, `.await`,
  async test functions, or ndarray arrays in that target. Verification:
  `rustup run nightly cargo fmt -p kwavers-analysis -p kwavers --check`,
  focused `kwavers` check/clippy for `--features
  gpu-visualization,async-runtime --test stream_visualization_test`, and
  focused nextest pass 25/25. Former residual:
  `examples/real_time_3d_beamforming.rs` still had a Tokio wrapper because
  construction directly awaited WGPU acquisition; the provider-constructor
  follow-up below closes that edge.
  Follow-up 3-D beamforming provider evidence on 2026-07-04:
  `BeamformingProcessor3D<P>` is generic over `BeamformingGpuProvider`,
  `WgpuBeamformingProvider` acquires through Hephaestus `WgpuDevice`, and the
  real-time 3-D beamforming example calls a synchronous constructor without
  Tokio. Verification: `rustup run nightly cargo fmt -p kwavers-analysis -p
  kwavers --check`, `rustup run nightly cargo check -p kwavers-analysis
  --features gpu --all-targets`, `rustup run nightly cargo check -p kwavers
  --features gpu --example real_time_3d_beamforming`, clippy for both focused
  targets with `-D warnings`, focused `kwavers-analysis --features gpu
  three_dimensional` nextest (52/52), scoped Tokio source audit, and depth-1
  `kwavers --features gpu` Tokio dependency audit pass. Follow-up
  provider-leak closure on 2026-07-04: `kwavers-analysis` now owns only the
  `BeamformingGpuProvider` operation contract plus CPU reference, while
  `kwavers-gpu::beamforming::three_dimensional::WgpuBeamformingProvider` owns
  WGPU acquisition, device-error mapping, bind-group layout, dispatch, DAS
  parameters, dynamic-focus dispatch, and WGSL shaders. Verification: direct
  `rustfmt --edition 2021` over touched Rust files, `rustup run nightly cargo
  check -p kwavers-analysis --features gpu`, `rustup run nightly cargo check -p
  kwavers-gpu --features gpu`, `rustup run nightly cargo check -p kwavers
  --features gpu --example real_time_3d_beamforming`, and focused `rustup run
  nightly cargo nextest run -p kwavers-gpu --features gpu
  wgpu_das_matches_cpu_reference_when_available --status-level fail
  --no-fail-fast` pass. Focused `kwavers-analysis --features gpu` nextest is
  blocked before the target tests by the out-of-scope
  `D:\atlas\repos\eunomia` `Complex<T>: NumericElement` compile error.
  Remaining GPU holdouts after this slice: `kwavers-analysis/src/visualization/**`
  still owns WGPU visualization behind `gpu-visualization`; CUDA 3-D
  beamforming DAS kernels do not exist yet, so no CUDA placeholder provider was
  added; broader `kwavers-gpu` FDTD/PSTD/thermal/acoustic WGPU providers remain
  leaf implementations pending real CUDA operation-family kernels and
  WGPU/CUDA differentials; solver PINN Burn code remains outside this
  provider-boundary slice.
  Follow-up provider-constructor evidence on 2026-07-04:
  `BeamformingProcessor3D::with_provider` is the public generic GPU
  constructor, `BeamformingProcessor3D::new_wgpu` names the current WGPU
  convenience constructor explicitly, and scoped source audit finds no in-tree
  `BeamformingProcessor3D::new` call sites. Verification: `rustup run nightly
  cargo fmt -p kwavers-analysis -p kwavers --check`, `rustup run nightly cargo
  check -p kwavers-analysis --features gpu --all-targets`, `rustup run
  nightly cargo check -p kwavers --features gpu --example
  real_time_3d_beamforming`, `rustup run nightly cargo check -p
  kwavers-analysis --all-targets`, focused clippy for both GPU targets, and
  `git diff --check` for the touched beamforming/example files pass. Focused
  `kwavers-analysis --features gpu three_dimensional` nextest could not execute
  because Windows reported `D:` at 4096 free bytes and rustc failed writing
  `D:\atlas\target` incremental artifacts with OS error 112.
  Follow-up distributed neural beamforming evidence on 2026-07-04:
  `DistributedNeuralBeamformingProcessor::process_volume_distributed` is
  synchronous over the existing Moirai-backed processor fan-out, the test no
  longer constructs a Tokio runtime, and `kwavers-analysis` no longer carries a
  Tokio dev-dependency. Verification: `rustup run nightly cargo fmt -p
  kwavers-analysis --check`, `rustup run nightly cargo check -p
  kwavers-analysis --features pinn --all-targets`, `rustup run nightly cargo
  clippy -p kwavers-analysis --features pinn --all-targets --no-deps -- -D
  warnings`, focused distributed `kwavers-analysis --features pinn` nextest
  (6/6), scoped Tokio source audit, and depth-1 Tokio dependency audit pass.
  Follow-up acoustic provider evidence on 2026-07-04: `rustup run nightly
  cargo fmt -p kwavers-gpu --check`, `rustup run nightly cargo check -p
  kwavers-gpu --features gpu --all-targets`, `rustup run nightly cargo check
  -p kwavers-gpu --features cuda-provider --all-targets`, focused provider
  nextest under both `gpu` (32/32) and `cuda-provider` (39/39), and clippy for
  both feature sets pass after binding `AcousticFieldProvider` to the shared
  provider trait stack.
  Follow-up regularization evidence on 2026-07-04: inverse regularization
  Tikhonov, smoothness, and L1 updates route through a shared Moirai-backed
  contiguous traversal helper instead of ndarray/Rayon `Zip::par_for_each`,
  preserving sequential ndarray traversal for non-standard layouts. `rustup
  run nightly cargo fmt -p kwavers-math --check`, `rustup run nightly cargo
  check -p kwavers-math --all-targets`, `rustup run nightly cargo nextest run
  -p kwavers-math regularization --status-level fail --no-fail-fast` (10/10),
  `rustup run nightly cargo clippy -p kwavers-math --all-targets --no-deps --
  -D warnings`, and the scoped direct-provider audit pass.
  Follow-up SIMD-safe evidence on 2026-07-04: `kwavers-math::simd_safe`
  dense add/scale now route through `hermes_simd::{elementwise_add, scale}`,
  dense ternary accumulation routes through Moirai chunk traversal, and
  non-standard ndarray layouts keep sequential traversal. `rustup run nightly
  cargo fmt -p kwavers-math --check`, `rustup run nightly cargo check -p
  kwavers-math --all-targets`, `rustup run nightly cargo nextest run -p
  kwavers-math simd --status-level fail --no-fail-fast` (18/18), `rustup run
  nightly cargo clippy -p kwavers-math --all-targets --no-deps -- -D warnings`,
  and the scoped `simd_safe` direct-provider audit pass. Residual upstream gap:
  Hermes lacks a public ternary `out += alpha * a * b` slice facade, so Kwavers
  keeps that exact operation in a Moirai-backed helper rather than allocating a
  temporary or changing rounding semantics.
  Follow-up differential evidence on 2026-07-04: second-order central and
  staggered-grid finite-difference operators route standard-layout destination
  fills through a shared Moirai-backed traversal helper, preserving sequential
  ndarray traversal for non-standard layouts. `rustup run nightly cargo fmt -p
  kwavers-math --check`, `rustup run nightly cargo check -p kwavers-math
  --all-targets`, `rustup run nightly cargo nextest run -p kwavers-math
  differential --status-level fail --no-fail-fast` (46/46), `rustup run
  nightly cargo clippy -p kwavers-math --all-targets --no-deps -- -D warnings`,
  and the scoped differential direct-provider audit pass.
  Follow-up FFT/k-space evidence on 2026-07-04: FFT real/complex packing and
  `KSpaceCalculator::generate_k_squared` route standard-layout arrays through
  Moirai-backed contiguous traversal, preserving sequential ndarray traversal
  for non-standard FFT layouts. `rustup run nightly cargo fmt -p kwavers-math
  --check`, `rustup run nightly cargo check -p kwavers-math --all-targets`,
  `rustup run nightly cargo nextest run -p kwavers-math -E "test(fft) or
  test(kspace)" --status-level fail --no-fail-fast` (18/18), `rustup run
  nightly cargo clippy -p kwavers-math --all-targets --no-deps -- -D
  warnings`, scoped `crates/kwavers-math/src` direct-provider audit, and `git
  diff --check` pass. Residual: source-level direct ndarray/Rayon parallel
  calls are closed under `kwavers-math`; manifest-level `ndarray/rayon`
  removal remains a dependency audit because transitive provider crates may
  still select Rayon.
  Follow-up manifest/API evidence on 2026-07-04: `kwavers-math` no longer
  enables ndarray's `rayon` feature, Apollo is consumed from the local Atlas
  checkout, Apollo's WGPU helper resolves local `hephaestus-wgpu v0.11.0`, and
  the Kwavers GPU FFT facade now exposes Apollo's `FftBackend` trait while
  documenting WGPU as the current implementation rather than the generic
  provider boundary. `rustup run nightly cargo fmt -p kwavers-math --check`,
  `rustup run nightly cargo check -p kwavers-math --all-targets`, `rustup run
  nightly cargo check -p kwavers-math --features gpu --all-targets`, focused
  FFT/k-space/spectral nextest (33/33), focused GPU FFT nextest (2/2), clippy,
  and dependency-tree audits pass. Evidence tier: compile-time dependency/type
  validation plus focused empirical FFT tests. Residual: Apollo has no real
  CUDA FFT provider yet; that upstream gap requires real CUDA kernels and
  WGPU/CUDA differential tests.
  Follow-up solver-boundary evidence on 2026-07-04: `kwavers-solver/gpu` no
  longer owns concrete WGPU runtime dependencies; its feature now only forwards
  `kwavers-math/gpu`, and direct `wgpu`, `bytemuck`, and `pollster` manifest
  edges were removed. Solver-layer Apollo FFT calls in KZK, PSTD derivatives,
  axisymmetric PSTD, viscoacoustic derivatives, and factory Fourier adapters
  now route through `kwavers_math::fft`, with 3-D axis-transform facade
  functions added to keep the ndarray/Leto boundary centralized. `rustup run
  nightly cargo fmt -p kwavers-math -p kwavers-solver --check`, `rustup run
  nightly cargo check -p kwavers-solver --features gpu --all-targets`, backend
  surface nextest (3/3), focused KZK/PSTD/viscoacoustic/backend nextest
  (62/62), `rustup run nightly cargo clippy -p kwavers-solver --features gpu
  --lib --no-deps -- -D warnings`, direct dependency-tree audit, and
  stale-token source audit pass. Residual: broad `kwavers-solver --features
  gpu --all-targets` clippy remains blocked by unrelated existing test-target
  lint debt.
  Residual: CUDA acoustic/FDTD execution is not claimed until real CUDA kernel
  sources and WGPU/CUDA differential tests exist.
- **Top-level GPU test provider cleanup - RESOLVED [patch].** The obsolete
  `recovery_stress_tests.rs` top-level test is removed because it referenced
  nonexistent recovery modules rather than current production APIs.
  `gpu_compute_backend_patterns.rs`, `gpu_buffer_tests.rs`,
  `gpu_device_tests.rs`, `gpu_compute_kernel_tests.rs`, and
  `gpu_pstd_parity.rs` now exercise the current provider stack through
  Hephaestus/CoreGpuContext/GPUBackend, pollster-backed `GpuDevice`,
  `GpuPstdSolver<WgpuPstdStateProvider>`, and provider-native
  `leto::Array3<f32>` instead of raw WGPU acquisition, Tokio test macros, or
  ndarray GPU surfaces. Evidence tier: compile-time validation plus focused
  empirical tests; `rustup run nightly cargo check -p kwavers --features gpu
  --tests` passes, focused top-level GPU nextest passes 27/27 with 3 ignored
  PSTD hardware tests skipped, and focused clippy for the five repaired test
  targets passes with `-D warnings`. Follow-up evidence: the unused
  `gpu_fft_3d` helper is deleted, ignored GPU FFT tests use
  `pollster::block_on` instead of Tokio macros, the unused finite-window Born
  baseline allocation is removed, and inactive `moirai-http`,
  `moirai-metrics`, and `moirai-tls` patch entries are removed because no
  workspace manifest depends on them. `rustup run nightly cargo check -p
  kwavers --features gpu --tests` now passes warning-clean; focused top-level
  GPU FFT/Born nextest passes 6/6 with 15 ignored hardware tests skipped; and
  focused clippy for those two test targets passes with `-D warnings`.
  Residual: real CUDA kernels plus WGPU/CUDA differential hardware coverage
  remain implementation work.
- **CUDA provider acquisition contract - RESOLVED [patch].** `kwavers-gpu`
  now depends optionally on local `hephaestus-cuda`, exposes
  `cuda-provider` for the compile-time acquisition contract and `cuda-runtime`
  for Hephaestus' real CUDA loader, and implements `GpuDeviceProvider` for
  `hephaestus_cuda::CudaDevice`. The CUDA provider does not inherit WGPU-only
  optional features or shader-stage storage-buffer slot limits, and no
  placeholder `CudaComputeProvider` was added; compute dispatch remains WGPU
  until real CUDA kernels are implemented. Evidence tier: type-level
  validation plus focused empirical tests; `cargo check -p kwavers-gpu
  --features cuda-provider --offline`, `cargo clippy -p kwavers-gpu
  --features cuda-provider --all-targets --offline -- -D warnings`, and
  `cargo nextest run -p kwavers-gpu --features cuda-provider --status-level
  fail --no-fail-fast --offline` pass, with nextest reporting 102/102 passed
  and 1 skipped. Re-verification on 2026-07-03: `rustup run nightly cargo
  check -p kwavers-gpu --features cuda-provider` passes, and focused nextest
  over provider identity/CUDA contract tests passes 5/5.
- **kwavers-math nalgebra decomposition bridge - RESOLVED [patch].**
  `LinearAlgebra::qr_decomposition` now delegates to Leto's Householder QR and
  `LinearAlgebra::svd` delegates to Leto-ops rank-revealing SVD, with ndarray
  retained only at the current public API boundary. The direct
  `kwavers-math` nalgebra dependency and all nalgebra/DMatrix/DVector tokens in
  `kwavers-math` source and manifest are removed. Evidence tier: compile-time
  dependency/type validation plus focused empirical tests; `rustup run nightly
  cargo check -p kwavers-math --all-targets`, focused `kwavers-math
  linear_algebra` nextest (51/51), and `rustup run nightly cargo clippy -p
  kwavers-math --all-targets --no-deps -- -D warnings` pass. Residual:
  `kwavers-math` still carries ndarray and num-traits migration holdouts
  outside this decomposition slice.
- **kwavers-physics thermal diffusion ndarray/Rayon edge - RESOLVED [patch].**
  Pennes bioheat perfusion/update and Cattaneo-Vernotte
  flux/divergence/temperature traversals now route through a private
  Moirai-backed dense-field adapter. Standard-layout arrays use
  `moirai-parallel`; non-contiguous views keep sequential ndarray traversal
  without cloning or concrete Rayon dispatch. Evidence tier: static source
  audit plus compile-time/lint validation and focused empirical tests; `rustup
  run nightly cargo check -p kwavers-physics --all-targets`, focused
  `kwavers-physics thermal::diffusion` nextest (2/2), and `rustup run nightly
  cargo clippy -p kwavers-physics --all-targets --no-deps -- -D warnings`
  pass. Residual: broader `kwavers-physics` direct ndarray/Rayon kernels
  remain outside this thermal diffusion slice.
- **kwavers-physics sonoluminescence ndarray/Rayon edge - RESOLVED [patch].**
  Blackbody, bremsstrahlung, and Cherenkov field assembly now route through
  the private Moirai-backed physics traversal adapter, including explicit
  shape validation for dense-slice zip arities. Evidence tier: static source
  audit plus compile-time/lint validation and focused empirical tests; `rustup
  run nightly cargo check -p kwavers-physics --all-targets`, focused
  `kwavers-physics sonoluminescence` nextest (34/34), and `rustup run nightly
  cargo clippy -p kwavers-physics --all-targets --no-deps -- -D warnings`
  pass. Residual: broader `kwavers-physics` direct ndarray/Rayon kernels
  remain outside this sonoluminescence slice.
- **GPU PSTD auto-device acquisition - RESOLVED [patch].**
  `GpuPstdSolver::with_auto_device` now acquires through
  `hephaestus_wgpu::WgpuDevice` instead of constructing a WGPU instance,
  adapter, and device locally. The request preserves the PSTD shader
  requirements for push constants and storage-buffer limits. Evidence tier:
  compile-time validation plus focused GPU tests; `cargo check -p kwavers-gpu
  --features gpu` passes and `cargo nextest run -p kwavers-gpu --features gpu
  pstd_gpu --no-fail-fast` passes 12/12.
  Follow-up provider-wrapper evidence: PSTD auto-device acquisition now returns
  `GpuProviderContext<WgpuDevice>` through the shared `GpuDeviceProvider`
  acquisition contract, and `WgpuPstdState` stores that provider context
  instead of owned raw WGPU device/queue handles. Evidence tier: static source
  audit plus compile-time/lint/focused empirical validation; `rg` finds no
  direct WGPU acquisition or `pollster::block_on` in
  `pstd_gpu::pipeline::auto_device`, and a scoped audit finds no
  `Arc<wgpu::Device>`/`Arc<wgpu::Queue>`, old device/queue associated handles,
  or raw WGPU clone accessors in the PSTD state/device boundary. `rustup run
  nightly cargo fmt -p kwavers-gpu --check`, `rustup run nightly cargo check
  -p kwavers-gpu --features gpu --all-targets`, `rustup run nightly cargo
  check -p kwavers-gpu --features cuda-provider --all-targets`, focused
  PSTD/provider nextest under `gpu` (46/46) and `cuda-provider` (53/53), and
  clippy for both feature sets pass.
  Follow-up PSTD test-support evidence: PSTD construction and medium-update
  test helpers now acquire through `GpuProviderContext<WgpuDevice>` and
  `GpuDeviceProvider`; scoped `rg` finds no direct WGPU acquisition or
  `pollster::block_on` under `crates/kwavers-gpu/src/pstd_gpu`, `rustup run
  nightly cargo fmt -p kwavers-gpu --check` passes, `rustup run nightly cargo
  check -p kwavers-gpu --features cuda-provider` passes, `rustup run nightly
  cargo clippy -p kwavers-gpu --features cuda-provider --all-targets -- -D
  warnings` passes, and focused `rustup run nightly cargo nextest run -p
  kwavers-gpu --features cuda-provider pstd_auto_device
  pstd_solver_auto_device_provider_uses_provider_handles
  pstd_gpu::tests::construction pstd_gpu::tests::medium_update --status-level
  fail --no-fail-fast` passes 7/7.
  Follow-up backend buffer-manager evidence: the backend buffer-manager GPU
  construction test now acquires through `GpuDevice<WgpuDevice>` and
  `GpuDeviceProvider`; scoped `rg` finds no direct WGPU acquisition in
  `backend::buffers` or under `pstd_gpu`, `rustup run nightly cargo fmt -p
  kwavers-gpu --check` passes, `rustup run nightly cargo check -p kwavers-gpu
  --features cuda-provider` passes, `rustup run nightly cargo clippy -p
  kwavers-gpu --features cuda-provider --all-targets -- -D warnings` passes,
  and focused `rustup run nightly cargo nextest run -p kwavers-gpu --features
  cuda-provider test_buffer_manager_creation
  backend_buffer_manager_wrapper_is_generic_over_provider_trait --status-level
  fail --no-fail-fast` passes 2/2.
  Follow-up backend readback evidence: `WgpuBackendBufferManager` no longer
  uses `pollster::block_on` for provider-native readback; both public readback
  entry points route through one blocking WGPU map/read implementation that
  returns `leto::Array3<f32>`. Evidence tier: static source audit plus
  compile-time/lint/focused empirical validation; scoped `rg` shows no
  `pollster::block_on` under `kwavers-gpu::backend`, `rustup run nightly cargo
  fmt -p kwavers-gpu --check` passes, `rustup run nightly cargo check -p
  kwavers-gpu --features cuda-provider` passes, `rustup run nightly cargo
  clippy -p kwavers-gpu --features cuda-provider --all-targets -- -D warnings`
  passes, and `rustup run nightly cargo nextest run -p kwavers-gpu --features
  cuda-provider backend --status-level fail --no-fail-fast` passes 45/45.
  Follow-up PSTD runner evidence: `run_gpu_pstd_with_provider<P>` is generic
  over `PstdAutoDeviceProvider` and `P::State: PstdRunState`, `run_gpu_pstd`
  remains the WGPU default, and the public PSTD runner mask/output contract is
  `leto::Array3<bool>`/`leto::Array2<f64>`. Evidence tier:
  type-level/compile-time validation plus focused empirical tests; fmt,
  GPU-enabled `kwavers-gpu`/`kwavers-diagnostics` check, clippy, focused
  runner/diagnostics nextest, and broader `kwavers-gpu` PSTD nextest pass.
  Follow-up CPML evidence: `CPMLProfiles` and `PmlExpFactors` now use Leto
  `Array1`; the required owned-array indexing and equality gaps were filled in
  Leto, and focused CPML/PSTD/PML nextest runs pass. Residual:
  source masks/signals still enter through the upstream ndarray-owned solver
  API.
- **GpuDevice acquisition trait - RESOLVED [patch].**
  `kwavers-gpu::gpu::GpuDevice<P>` is generic over a local
  `GpuDeviceProvider` trait backed by Hephaestus `ComputeDeviceAcquisition`.
  Generic callers use backend-neutral `DevicePreference`, `DeviceFeature`, and
  `DeviceLimits`; raw WGPU handles are exposed only on the
  `GpuDevice<WgpuDevice>` specialization for existing WGSL shader dispatch.
  Evidence tier: static source audit plus compile-time validation and focused
  GPU tests; `cargo check -p kwavers-gpu --features gpu` passes and `cargo
  nextest run -p kwavers-gpu --features gpu backend
  gpu::shaders::neural_network gpu::multi_gpu
  pstd_gpu::tests::construction --status-level fail --no-fail-fast` passes
  37/37.
- **CoreGpuContext provider-generic boundary - RESOLVED [patch].**
  `kwavers-gpu::gpu::CoreGpuContext<P>` now owns
  `GpuDevice<P: GpuDeviceProvider>` instead of a raw `WgpuDevice`. Generic
  callers can borrow the concrete provider through `provider()`, while raw
  WGPU device/queue/submit access remains restricted to the
  `CoreGpuContext<WgpuDevice>` specialization needed by current WGSL kernels.
  The provider capability hook keeps WGPU's existing core-atomic claim explicit
  and leaves CUDA conservative until real CUDA Kwavers kernels exist. Evidence
  tier: type-level/compile-time validation plus focused GPU tests; `cargo check
  -p kwavers-gpu --features gpu` passes, `cargo check -p kwavers-gpu --features
  cuda-provider --offline` passes, `cargo clippy -p kwavers-gpu --features gpu
  --lib -- -D warnings` passes, `cargo nextest run -p kwavers-gpu --features
  gpu provider --status-level fail --no-fail-fast` passes 3/3, and `cargo
  nextest run -p kwavers-gpu --features cuda-provider provider --status-level
  fail --no-fail-fast --offline` passes 5/5.
- **GpuBackend provider-generic alias - RESOLVED [patch].**
  `kwavers-gpu::gpu::GpuBackend<P>` now exposes the same provider parameter as
  `CoreGpuContext<P>`, preserving the WGPU default while allowing explicit
  provider types at the context boundary. Evidence tier:
  type-level/compile-time validation plus focused GPU test; `cargo check -p
  kwavers-gpu --features gpu` passes, `cargo clippy -p kwavers-gpu --features
  gpu --lib -- -D warnings` passes, and `cargo nextest run -p kwavers-gpu
  --features gpu gpu_backend_alias_exposes_provider_parameter --status-level
  fail --no-fail-fast` passes 1/1.
- **MultiGpuContext provider-generic boundary - RESOLVED [patch].**
  `kwavers-gpu::gpu::multi_gpu::MultiGpuContext<P>` now owns
  `Vec<CoreGpuContext<P>>` and acquires devices through
  `P::try_acquire_devices` on the shared `GpuDeviceProvider` contract. The
  default `MultiGpuContext::new()` remains the WGPU constructor required by
  current WGSL kernels, while CUDA type-checks at the topology/scheduling
  boundary without a fake compute implementation. Evidence tier:
  type-level/compile-time validation plus focused GPU tests; `rustup run
  nightly cargo fmt -p kwavers-gpu --check`, `rustup run nightly cargo check
  -p kwavers-gpu --features cuda-provider`, `rustup run nightly cargo clippy
  -p kwavers-gpu --features gpu --all-targets -- -D warnings`, focused
  `gpu::multi_gpu` nextest under `--features gpu` (4/4), and focused
  `gpu::multi_gpu` nextest under `--features cuda-provider` (5/5) pass.
- **kwavers-gpu Burn accelerator surface - RESOLVED [patch].**
  The local `BurnGpuAccelerator` module/export, optional `burn` dependency,
  and `kwavers-gpu/pinn` feature were removed from `kwavers-gpu`. That surface
  was not backed by Hephaestus or Coeus, converted `Array3<f64>` through Burn
  `f32` tensors, and returned zero tensors for non-wave PDE residual variants.
  Evidence tier: static source audit plus compile-time/focused empirical
  tests; `rustup run nightly cargo fmt -p kwavers-gpu --check`, `rustup run
  nightly cargo check -p kwavers-gpu --features gpu`, `rustup run nightly
  cargo check -p kwavers-gpu --features cuda-provider`, `rustup run nightly
  cargo check -p kwavers-gpu --all-features`, `rustup run nightly cargo clippy
  -p kwavers-gpu --features gpu --all-targets -- -D warnings`, and `rustup run
  nightly cargo nextest run -p kwavers-gpu --features gpu --status-level fail
  --no-fail-fast` pass with nextest reporting 128/128 passed and 1 skipped.
  The broader `kwavers-gpu --all-features` nextest build did not reach test
  execution because `D:/atlas/target` hit OS error 112 while writing an Apollo
  FFT rlib; `Get-PSDrive` reported `D:` at 0.15 GB free. Residual: solver-side
  Burn PINN modules still require a real Coeus migration.
- **Solver PINN false GPU backend surface - RESOLVED [major].** The
  `inverse::pinn::ml::gpu_accelerator` module was removed because it exposed
  CUDA-named buffers, streams, kernel manager, and batched trainer without real
  CUDA module compilation or device execution. The `pinn-gpu`,
  `burn-wgpu`, and `burn-cuda` Cargo feature aliases were removed from
  `kwavers`/`kwavers-solver`, and PINN docs/examples now describe GPU training
  as pending Coeus training routed through provider-generic Hephaestus traits,
  with WGPU and CUDA as interchangeable providers behind that seam. Evidence
  tier: static source audit plus formatting/whitespace validation; fixed-string
  audit over the touched manifests, PINN solver docs, and examples found no
  removed feature names or CUDA mock exports, `rustup run nightly cargo fmt -p
  kwavers-solver -p kwavers --check` passed, and scoped `git diff --check`
  passed. `rustup run nightly cargo metadata --no-deps --format-version 1
  --manifest-path crates/kwavers-solver/Cargo.toml --features pinn` passed.
  `rustup run nightly cargo check -p kwavers-solver --features pinn` failed
  before reaching the changed package because Cargo could not write dependency
  `.rmeta` files on `D:` with OS error 112; `Get-PSDrive` reported `D:` at
  0.17 GB free. Residual: real PINN GPU execution still requires Coeus backend
  integration plus value-semantic training/residual tests.
- **Solver PINN direct WGPU discovery - RESOLVED [patch].** The Burn-era
  `MultiGpuManager` no longer enumerates WGPU adapters inside
  `kwavers-solver`. Multi-GPU PINN construction now returns a typed
  unavailable-provider error naming the missing Coeus training provider and
  Hephaestus WGPU/CUDA device-trait route, so the solver core does not choose a
  concrete GPU provider or fabricate CPU/GPU devices. Evidence tier: static
  source audit plus compile-time validation; scoped `rg` found no WGPU
  discovery tokens under the solver PINN multi-GPU manager/distributed-training
  path, and `rustup run nightly cargo check -p kwavers-solver --features
  pinn,gpu` passed. Focused nextest was attempted but stopped after remaining
  in `apollo-fft` dependency codegen without a test result. Follow-up
  Tokio-removal evidence: `MultiGpuManager::new` and discovery are synchronous
  while they only return the typed unavailable-provider condition, scoped `rg`
  finds no Tokio token under the solver PINN multi-GPU manager, `rustup run
  nightly cargo fmt -p kwavers-solver --check` passes, `rustup run nightly
  cargo check -p kwavers-solver --features pinn` passes, and `rustup run
  nightly cargo nextest run -p kwavers-solver --features pinn
  multi_gpu_manager --status-level fail --no-fail-fast` passes 3/3. Follow-up
  distributed-trainer evidence: `DistributedPinnTrainer::new` is synchronous,
  its creation test uses `#[test]`, scoped `rg` finds no async constructor or
  Tokio-test holdout under distributed training, `rustup run nightly cargo
  nextest run -p kwavers-solver --features pinn distributed_training
  --status-level fail --no-fail-fast` passes 3/3, and `rustup run nightly
  cargo clippy -p kwavers-solver --features pinn --lib -- -D warnings` passes.
  Follow-up solver-local async removal: distributed training and checkpoint
  persistence are synchronous, checkpoint save/load writes JSON state instead
  of only logging success, `kwavers-solver/pinn` no longer enables `dep:tokio`,
  `kwavers-solver/async-runtime` is an empty retained feature, focused
  distributed-training nextest passes 4/4, and `kwavers-solver --features
  async-runtime` check passes. Residual: real multi-GPU PINN execution still
  requires a Coeus training provider behind Hephaestus WGPU/CUDA traits plus
  value-semantic training/residual tests; broader top-level/transitive Tokio
  edges still exist outside this solver-local PINN slice.
- **Burn WGPU dependency feature coupling - RESOLVED [patch].** The workspace
  Burn dependency and the remaining `kwavers`, `kwavers-solver`, and
  `kwavers-analysis` Burn dependencies no longer enable Burn's `wgpu` feature.
  Burn defaults are disabled on remaining kwavers Burn edges; solver keeps the
  non-GPU `train` feature required by current PINN code, and RITK's workspace
  Burn default was repaired from WGPU to NdArray so the downstream kwavers
  graph does not re-enable `burn-wgpu`. GPU PINN execution remains assigned to
  Coeus through Hephaestus provider traits. The analysis beamforming GPU module
  no longer publicly reexports the Burn DAS implementation; that Burn path now
  compiles only for `pinn` tests pending the same Coeus/Hephaestus migration.
  The analysis neural beamforming facade no longer publicly reexports the
  Burn-specific PINN provider constructor or adapter; callers use the
  solver-agnostic `PinnBeamformingProvider` trait surface instead.
  Evidence tier: static manifest
  audit plus dependency-tree and compile-time validation; `rustup run nightly
  cargo tree -p kwavers --features pinn -i burn-wgpu` reports no matching
  package, `rustup run nightly cargo tree -p kwavers --features pinn | rg
  "burn-wgpu|burn-cuda|burn-rocm"` returns no selected dependency hits, `rustup
  run nightly cargo fmt -p kwavers --check` passed, and `rustup run nightly
  cargo check -p kwavers --features pinn` passed. Follow-up analysis neural
  evidence: scoped source audit found no public Burn provider reexport or stale
  CUDA/wgpu doc claim, `rustup run nightly cargo fmt -p kwavers-analysis
  --check` passed, `rustup run nightly cargo check -p kwavers-analysis
  --features pinn` passed, focused `cargo nextest run -p kwavers-analysis
  --features pinn neural --status-level fail --no-fail-fast` passed 77/77,
  and `rustup run nightly cargo clippy -p kwavers-analysis --features pinn
  --all-targets -- -D warnings` passed.
  Follow-up analysis uncertainty evidence: `PinnUncertaintyPredictor` now owns
  the analysis-side PINN prediction contract; Bayesian, ensemble, conformal,
  and top-level uncertainty methods accept that trait instead of
  `BurnPINN1DWave<B>`. Burn remains only as a compatibility impl for the
  current solver model pending Coeus. Evidence tier: type-level API boundary
  plus value-semantic tests; scoped source audit under
  `crates/kwavers-analysis/src/ml/uncertainty` finds Burn tokens only in the
  single compatibility impl, `rustup run nightly cargo fmt -p kwavers-analysis
  --check` passed, `rustup run nightly cargo check -p kwavers-analysis
  --features pinn` passed, focused `cargo nextest run -p kwavers-analysis
  --features pinn uncertainty --status-level fail --no-fail-fast` passed
  33/33, and `rustup run nightly cargo clippy -p kwavers-analysis --features
  pinn --all-targets -- -D warnings` passed.
  Follow-up analysis Burn-removal evidence: `kwavers-analysis` no longer has a
  direct Burn dependency or source-level Burn use. The test-only Burn DAS
  holdout and analysis-side Burn compatibility impl were removed; Coeus remains
  the owning replacement path. Evidence tier: static source/manifest audit plus
  compile-time/focused empirical validation; scoped source/manifest audit under
  `crates/kwavers-analysis` returned no Burn matches, direct `cargo metadata`
  returned `NO_DIRECT_BURN`, `rustup run nightly cargo fmt -p kwavers-analysis
  --check` passed, `rustup run nightly cargo check -p kwavers-analysis
  --features pinn` passed, focused `cargo nextest run -p kwavers-analysis
  --features pinn -E "test(uncertainty) or test(time_domain::das) or
  test(das_single_element_zero_delay_passthrough) or
  test(das_coherent_gain_co_located_elements) or
  test(das_receive_delay_is_geometrically_correct) or
  test(das_channel_mismatch_returns_error)" --status-level fail --no-fail-fast`
  passed 47/47, and `rustup run nightly cargo clippy -p kwavers-analysis
  --features pinn --all-targets -- -D warnings` passed.
- **kwavers-python RITK NIfTI direct Burn edge - RESOLVED [patch].**
  `kwavers-python::ritk_image::load_ritk_nifti` now reads through
  `ritk_io::format::nifti::native::NiftiReader` on
  `coeus_core::SequentialBackend` instead of RITK's legacy Burn `NdArray`
  backend. The PyO3 wrapper still returns the existing NumPy-facing
  `(x, y, z)` `ndarray::Array3<f64>` and `(dx, dy, dz)` spacing contract, so
  this slice removes the file-I/O substrate edge without moving Python-facing
  domain behavior into the wrapper. Evidence tier: static source/dependency
  audit plus formatting; scoped `rg` finds no `burn`, `NdArrayBackend`,
  `read_nifti::<`, `TensorData`, or `into_data` hits in
  `crates/kwavers-python/Cargo.toml` or
  `crates/kwavers-python/src/ritk_image.rs`; `cargo tree -p kwavers-python
  --depth 1` shows direct `coeus-core`, `ritk-io`, and `ritk-image` edges and
  no direct Burn edge; `rustup run nightly cargo fmt -p kwavers-python
  --check` passes. Residual package gate: `cargo check -p kwavers-python`
  remains blocked by existing crate-wide NumPy/ndarray version-boundary errors
  and Leto wrapper-boundary mismatches outside this loader slice.
- **kwavers-imaging CT/NIfTI direct Burn edge - RESOLVED [patch].**
  `kwavers_imaging::medical::CTImageLoader` now reads through
  `ritk_io::format::nifti::native::NiftiReader` on
  `coeus_core::SequentialBackend` instead of RITK's legacy Burn-backed
  `read_nifti::<AdapterBackend>` path. The bridge keeps RITK `Point`,
  `Spacing`, and `Direction` metadata typed through `ritk-spatial` and shares
  the row-major `[depth, rows, cols]` to kwavers `(x, y, z)` conversion with
  the remaining DICOM path. Evidence tier: compile-time validation plus
  focused empirical tests; `rustup run nightly cargo fmt -p kwavers-imaging
  --check` passes, `rustup run nightly cargo check -p kwavers-imaging`
  passes, and `rustup run nightly cargo nextest run -p kwavers-imaging
  ct_loader --status-level fail --no-fail-fast` passes 8/8.
- **kwavers-imaging DICOM direct Burn edge - RESOLVED [patch].**
  RITK now exposes native DICOM series loading on the public series facade,
  and `kwavers_imaging::medical::dicom_loader` calls
  `ritk_io::load_native_dicom_series` on `coeus_core::SequentialBackend`
  instead of the legacy Burn-backed `load_dicom_series::<AdapterBackend>`
  path. `kwavers-imaging` no longer depends directly on Burn or `ritk-core`;
  DICOM and NIfTI share the native RITK image to kwavers volume bridge.
  Evidence tier: source/dependency audit plus compile-time validation and
  focused empirical tests; RITK `cargo check -p ritk-io` passes, RITK focused
  nextest passes `native_dicom_loader_matches_legacy_loader` 1/1 and
  `native_series_loader_matches_legacy_loader` 1/1, `cargo check -p
  kwavers-imaging` passes, focused `cargo nextest run -p kwavers-imaging
  dicom --status-level fail --no-fail-fast` passes 14/14, scoped `rg` finds
  no `AdapterBackend`, `TensorData`, `burn::`, `burn =`, or `ritk-core` under
  `crates/kwavers-imaging`, and `cargo tree -p kwavers-imaging --depth 1`
  shows no direct Burn edge.
- **Top-level Rayon feature coupling - RESOLVED [major].** The top-level
  `kwavers` crate no longer depends directly on Rayon or exposes the obsolete
  `parallel = ["ndarray/rayon"]` feature. Liver theranostic reconstruction and
  3-D seismic example fan-out/blur loops now use `moirai-parallel`, and
  `kwavers-python` no longer enables `ndarray/rayon` where wrapper source has
  no ndarray-parallel call sites. Evidence tier: static source/dependency audit
  plus compile-time validation of the edited examples; scoped `rg` found no
  direct Rayon or ndarray-parallel hits under top-level `kwavers` targets or
  `kwavers-python/src`, `cargo tree -p kwavers --depth 1` lists
  `moirai-parallel` with no direct `rayon`, and both edited examples compile.
  Residual: `kwavers-solver` and `kwavers-physics` still contain direct
  Rayon/ndarray-parallel holdouts; `kwavers-python` package compilation remains
  blocked by existing numpy/ndarray and Leto wrapper boundary errors.
- **kwavers-physics direct Rayon edge - RESOLVED [patch].**
  `kwavers-physics` no longer has a direct Rayon manifest dependency or
  source-level direct Rayon iterator usage; transducer steering, RTM
  beam/backpropagation, nonlinear stability constraints, and Monte Carlo photon
  chunking now route through `moirai-parallel`. Evidence tier: static
  source/dependency audit plus compile-time and focused empirical validation;
  scoped `rg` found no `rayon::`, `use rayon`, `par_iter(`, or
  `into_par_iter(` hits under `crates/kwavers-physics/src` or its manifest,
  `cargo tree -p kwavers-physics --depth 1` lists `moirai-parallel` and no
  direct `rayon`, `cargo check -p kwavers-physics` passes, and focused
  `cargo nextest run -p kwavers-physics` provider-migration tests pass 41/41.
  Follow-up bubble-interaction evidence on 2026-07-04:
  `acoustics::bubble_dynamics::interactions` now routes interaction-field
  assembly through `crate::parallel::for_each_indexed_mut` instead of
  ndarray/Rayon `Zip::par_for_each`. The regression
  `interaction_field_uses_monopole_pressure_and_skips_self_cell` checks the
  analytical monopole-pressure contribution and source-cell exclusion.
  `rustup run nightly cargo check -p kwavers-physics`, `rustup run nightly
  cargo clippy -p kwavers-physics --lib -- -D warnings`, and focused
  `kwavers-physics bubble_dynamics::interactions` nextest (4/4) pass; scoped
  `rg` finds no direct Rayon or ndarray-parallel hit in the edited file.
  Follow-up field-surrogate evidence on 2026-07-04:
  `field_surrogate::resample` and `field_surrogate::cube` now route trilinear
  output assembly and in-place corner blending through the existing
  Moirai-backed physics traversal adapter instead of ndarray/Rayon
  `Zip::par_for_each`. `rustup run nightly cargo check -p kwavers-physics`,
  `rustup run nightly cargo clippy -p kwavers-physics --lib -- -D warnings`,
  and focused `kwavers-physics field_surrogate` nextest (24/24) pass; scoped
  `rg` finds no direct Rayon, ndarray-parallel, or `Zip` hit under
  `crates/kwavers-physics/src/field_surrogate`.
  Follow-up reaction-kinetics evidence on 2026-07-04:
  `chemistry::reaction_kinetics` now routes the hydroxyl and hydrogen-peroxide
  field update through the reusable `crate::parallel::zip_two_mut_two_refs`
  adapter instead of ndarray/Rayon `Zip::par_for_each`. The regression
  `update_reactions_matches_arrhenius_and_recombination_formula` checks
  Arrhenius OH generation and H2O2 recombination values. `rustup run nightly
  cargo check -p kwavers-physics`, `rustup run nightly cargo clippy -p
  kwavers-physics --lib -- -D warnings`, and focused `kwavers-physics
  reaction_kinetics` nextest (1/1) pass; scoped `rg` finds no direct Rayon or
  ndarray-parallel hit in the edited reaction file.
  Follow-up ROS concentration evidence on 2026-07-04:
  `chemistry::ros_plasma::ros_species::ROSConcentrations::apply_decay` now
  routes through `crate::parallel::for_each_indexed_mut` instead of
  ndarray/Rayon `par_mapv_inplace`. The regression
  `ros_decay_matches_species_lifetime_exponential` checks the exact
  lifetime-decay formula. `rustup run nightly cargo check -p kwavers-physics`,
  `rustup run nightly cargo clippy -p kwavers-physics --lib -- -D warnings`,
  and focused `kwavers-physics ros_species` nextest (4/4) pass; scoped `rg`
  finds no direct Rayon or ndarray-parallel hit under
  `crates/kwavers-physics/src/chemistry/ros_plasma/ros_species`.
  Residual: `ndarray/rayon` remains for tracked `Zip::par_for_each` and
  `par_mapv_inplace` kernels pending the Leto/Hephaestus backend migration;
  `kwavers-solver` still has separate direct Rayon/ndarray-parallel holdouts.
- **kwavers-solver Westervelt spectral direct Rayon loop - RESOLVED [patch].**
  The Westervelt spectral wave-model leapfrog combination loop now uses
  `moirai_parallel::enumerate_mut_with::<Adaptive>` over the next-pressure
  slice, with pressure history, sound speed, Laplacian, nonlinear, damping, and
  source slices read-only. The value recurrence is unchanged; only the
  execution provider moved from direct Rayon to Moirai. Evidence tier:
  source audit plus compile-time and focused empirical validation; scoped `rg`
  finds no direct Rayon or ndarray-parallel token in the edited wave-model
  file, `cargo check -p kwavers-solver` passes, `cargo clippy -p
  kwavers-solver --lib -- -D warnings` passes, and focused `cargo nextest run
  -p kwavers-solver -E "test(westervelt_spectral::solver)"` passes 8/8.
  Residual: the `kwavers-solver` manifest-level Rayon dependency remains open
  until the remaining solver direct Rayon and ndarray-parallel holdouts are
  migrated.
- **kwavers-solver mixed-domain direct Rayon edge - RESOLVED [patch].**
  `forward::hybrid::mixed_domain` now applies its frequency-domain phase
  propagator through Moirai indexed traversal over the standard-layout complex
  spectral output instead of ndarray/Rayon `Zip::par_for_each`. The propagated
  coefficient formula is unchanged: each input coefficient is multiplied by
  the derived `exp(i k dx)` phase factor. Evidence tier: static source audit
  plus compile-time and focused empirical validation; scoped `rg` finds no
  direct Rayon or ndarray-parallel token in the edited file, `rustup run
  nightly cargo fmt -p kwavers-solver --check` passes, `rustup run nightly
  cargo check -p kwavers-solver` passes, `rustup run nightly cargo clippy -p
  kwavers-solver --lib -- -D warnings` passes, and focused `rustup run
  nightly cargo nextest run -p kwavers-solver -E "test(mixed_domain) or
  test(hybrid)" --status-level fail --no-fail-fast` passes 59/59. Residual:
  broad solver direct Rayon/ndarray-parallel scan still finds FDTD, PSTD, SWE,
  Kuznetsov, RTM, photoacoustic, and multiphysics holdouts.
- **kwavers-solver KZK plugin direct Rayon edge - RESOLVED [patch].**
  `forward::nonlinear::kzk_solver_plugin::solver` now applies the legacy
  plugin nonlinear Burgers update through Moirai indexed traversal for
  standard-layout dense fields instead of ndarray/Rayon `Zip::par_for_each`.
  Non-standard ndarray layouts retain sequential mutable iteration without
  cloning or forcing a concrete Rayon provider. The nonlinear limiter formula
  is unchanged. Evidence tier: static source audit plus compile-time and
  focused empirical validation; scoped `rg` finds no direct Rayon or
  ndarray-parallel token in the edited file, `rustup run nightly cargo fmt -p
  kwavers-solver --check` passes, `rustup run nightly cargo check -p
  kwavers-solver` passes, `rustup run nightly cargo clippy -p kwavers-solver
  --lib -- -D warnings` passes, and focused `rustup run nightly cargo nextest
  run -p kwavers-solver -E "test(kzk_solver_plugin) or test(kzk) or
  test(nonlinear)" --status-level fail --no-fail-fast` passes 181/181.
  Residual: broad solver direct Rayon/ndarray-parallel scan still finds FDTD,
  PSTD, SWE, Kuznetsov, RTM, photoacoustic, and multiphysics holdouts.
- **kwavers-solver FDTD pressure-source direct Rayon edge - RESOLVED
  [patch].** `forward::fdtd::solver::sources` now applies dynamic pressure
  Dirichlet and additive source-mask updates through Moirai indexed traversal
  for standard-layout dense pressure/mask fields instead of ndarray/Rayon
  `Zip::par_for_each`. Non-standard ndarray layouts retain sequential mutable
  iteration without cloning or forcing a concrete Rayon provider. Evidence
  tier: static source audit plus compile-time and focused empirical validation;
  scoped `rg` finds no direct Rayon or ndarray-parallel token in the edited
  file, `rustup run nightly cargo fmt -p kwavers-solver --check` passes,
  `rustup run nightly cargo check -p kwavers-solver` passes, `rustup run
  nightly cargo clippy -p kwavers-solver --lib -- -D warnings` passes, and
  focused `rustup run nightly cargo nextest run -p kwavers-solver -E
  "test(fdtd) or test(source) or test(sources)" --status-level fail
  --no-fail-fast` passes 93/93. Residual: the later FDTD pressure-updater,
  velocity-updater, k-space-correction, and construction slices close the
  remaining FDTD direct-provider tokens; broader solver holdouts remain in
  PSTD, SWE, Kuznetsov, RTM, photoacoustic, and multiphysics paths.
- **kwavers-solver FDTD pressure-updater direct Rayon edge - RESOLVED
  [patch].** `forward::fdtd::pressure_updater` now routes divergence
  accumulation, pressure update, and Westervelt nonlinear pressure-delta
  application through shared Moirai-backed helpers for standard-layout dense
  fields instead of ndarray/Rayon `Zip::par_for_each`. Non-standard ndarray
  layouts retain sequential Zip semantics without cloning. Evidence tier:
  static source audit plus compile-time and focused empirical validation;
  scoped `rg` finds no direct Rayon or ndarray-parallel token under the
  pressure-updater subtree, `rustup run nightly cargo fmt -p kwavers-solver
  --check` passes, `rustup run nightly cargo check -p kwavers-solver` passes,
  `rustup run nightly cargo clippy -p kwavers-solver --lib -- -D warnings`
  passes, and focused `rustup run nightly cargo nextest run -p kwavers-solver
  -E "test(pressure_updater) or test(update_pressure) or test(divergence) or
  test(fdtd)" --status-level fail --no-fail-fast` passes 63/63. Residual:
  the later FDTD velocity-updater, k-space-correction, and construction
  slices close the remaining FDTD direct-provider tokens; broader solver
  holdouts remain in PSTD, SWE, Kuznetsov, RTM, photoacoustic, and
  multiphysics paths.
- **kwavers-solver FDTD velocity-updater direct Rayon edge - RESOLVED
  [patch].** `forward::fdtd::velocity_updater` now routes k-space,
  collocated, and staggered pressure-gradient velocity updates plus staggered
  gradient scratch fills through Moirai-backed helpers for standard-layout
  dense fields instead of ndarray/Rayon `Zip::par_for_each`. Non-standard
  ndarray layouts retain sequential Zip semantics without cloning. Evidence
  tier: static source audit plus compile-time and focused empirical
  validation; scoped `rg` finds no direct Rayon or ndarray-parallel token in
  `fdtd/velocity_updater.rs`, `rustup run nightly cargo fmt -p kwavers-solver
  --check` passes, `rustup run nightly cargo check -p kwavers-solver` passes,
  `rustup run nightly cargo clippy -p kwavers-solver --lib -- -D warnings`
  passes, and focused `rustup run nightly cargo nextest run -p kwavers-solver
  -E "test(velocity) or test(fdtd) or test(kspace)" --status-level fail
  --no-fail-fast` passes 91/91. Residual: broader solver holdouts remain in
  PSTD, SWE, Kuznetsov, RTM, photoacoustic, and multiphysics paths.
- **kwavers-solver FDTD k-space-correction direct Rayon edge - RESOLVED
  [patch].** `forward::fdtd::kspace_correction::operators` now routes shifted
  spectral gradient and divergence kernel construction through a Moirai-backed
  helper for standard-layout dense transformed fields instead of ndarray/Rayon
  `Zip::par_for_each`. Non-standard ndarray layouts retain sequential Zip
  semantics without cloning. Evidence tier: static source audit plus
  compile-time and focused empirical validation; scoped `rg` finds no direct
  Rayon or ndarray-parallel token in `fdtd/kspace_correction/operators.rs`,
  `rustup run nightly cargo fmt -p kwavers-solver --check` passes, `rustup run
  nightly cargo check -p kwavers-solver` passes, `rustup run nightly cargo
  clippy -p kwavers-solver --lib -- -D warnings` passes, and focused `rustup
  run nightly cargo nextest run -p kwavers-solver -E "test(kspace) or
  test(fdtd) or test(velocity) or test(pressure_updater)" --status-level fail
  --no-fail-fast` passes 91/91. Residual: broader solver holdouts remain in
  PSTD, SWE, Kuznetsov, RTM, photoacoustic, and multiphysics paths.
- **kwavers-solver FDTD construction ndarray Zip edge - RESOLVED
  [patch].** `forward::fdtd::solver::construction` now fills construction-time
  `rho*c^2` and nonlinear coefficient arrays through Moirai-backed helpers for
  standard-layout dense fields instead of ndarray `Zip`/`map_collect`.
  Non-standard ndarray layouts retain sequential explicit indexing without
  cloning. Evidence tier: static source audit plus compile-time and focused
  empirical validation; scoped `rg` finds no direct Rayon, ndarray-parallel,
  or explicit `ndarray::Zip` token under `forward::fdtd`, `rustup run nightly
  cargo fmt -p kwavers-solver --check` passes, `rustup run nightly cargo check
  -p kwavers-solver` passes, `rustup run nightly cargo clippy -p
  kwavers-solver --lib -- -D warnings` passes, and focused `rustup run
  nightly cargo nextest run -p kwavers-solver -E "test(fdtd) or test(kspace)
  or test(nonlinear) or test(construction)" --status-level fail
  --no-fail-fast` passes 290/290. Residual: FDTD still owns ndarray `Array3`
  solver-state storage pending the larger Leto/CPU-backend migration; broader
  solver holdouts remain in PSTD, SWE, Kuznetsov, RTM, photoacoustic, and
  multiphysics paths.
- **kwavers-solver PSTD utility direct Rayon edge - RESOLVED [patch].**
  `forward::pstd::utils` now computes k-squared, k-magnitude, and
  spectral-derivative complex scaling through Moirai-backed dense helpers
  instead of ndarray/Rayon `Zip::par_for_each` and `par_mapv_inplace`.
  Non-standard ndarray views retain sequential iteration without cloning.
  Evidence tier: static source audit plus compile-time and focused empirical
  validation; scoped `rg` finds no direct Rayon, ndarray-parallel, or explicit
  `ndarray::Zip` token in `forward/pstd/utils/mod.rs`, `rustup run nightly
  cargo fmt -p kwavers-solver --check` passes, `rustup run nightly cargo check
  -p kwavers-solver` passes, `rustup run nightly cargo clippy -p
  kwavers-solver --lib -- -D warnings` passes, and focused `rustup run
  nightly cargo nextest run -p kwavers-solver -E "test(pstd::utils) or
  test(wavenumber) or test(k_magnitude) or test(spectral_gradient) or
  test(kappa)" --status-level fail --no-fail-fast` passes 22/22. Residual:
  broader PSTD direct-provider holdouts remain in propagator, implementation,
  derivatives, physics, and extension paths pending separate slices.
- **kwavers-solver PSTD implementation k-space direct Rayon edge - RESOLVED
  [patch].** `forward::pstd::implementation::k_space::operators` now routes
  Helmholtz spectral multiplication and x/y/z spectral-gradient multipliers
  through Moirai-backed helpers for standard-layout dense spectral fields
  instead of ndarray/Rayon `Zip::par_for_each`. Non-standard ndarray layouts
  retain sequential explicit indexing without cloning, and spectral gradients
  no longer require contiguous axis vectors. Evidence tier: static source
  audit plus compile-time and focused empirical validation; scoped `rg` finds
  no direct Rayon, ndarray-parallel, or explicit `Zip` token in
  `forward/pstd/implementation/k_space/operators.rs`, `rustup run nightly
  cargo fmt -p kwavers-solver --check` passes, `rustup run nightly cargo check
  -p kwavers-solver` passes, `rustup run nightly cargo clippy -p
  kwavers-solver --lib -- -D warnings` passes, and focused `rustup run nightly
  cargo nextest run -p kwavers-solver -E "test(pstd) or test(k_space) or
  test(kspace) or test(spectral_grad) or test(helmholtz)" --status-level fail
  --no-fail-fast` passes 206/206. Residual: broader PSTD direct-provider
  holdouts remain in propagator, implementation core, derivatives, physics,
  and extension paths pending separate slices.
- **kwavers-solver PSTD implementation anti-aliasing filter direct Rayon edge
  - RESOLVED [patch].**
  `forward::pstd::implementation::core::stepper::filter` now routes pressure,
  split-density, and velocity half-spectrum anti-aliasing multipliers through
  one Moirai-backed helper for standard-layout dense spectra instead of seven
  ndarray/Rayon `Zip::par_for_each` calls. Non-standard ndarray layouts retain
  sequential explicit indexing without cloning. Evidence tier: static source
  audit plus compile-time and focused empirical validation; scoped `rg` finds
  no direct Rayon, ndarray-parallel, or explicit `Zip` token in
  `forward/pstd/implementation/core/stepper/filter.rs`, `rustup run nightly
  cargo fmt -p kwavers-solver --check` passes, `rustup run nightly cargo check
  -p kwavers-solver` passes, `rustup run nightly cargo clippy -p
  kwavers-solver --lib -- -D warnings` passes, and focused `rustup run nightly
  cargo nextest run -p kwavers-solver -E "test(anti_aliasing) or test(filter)
  or test(pstd)" --status-level fail --no-fail-fast` passes 175/175.
  Residual: broader PSTD direct-provider holdouts remain in propagator,
  implementation core step/source/construction paths, derivatives, physics,
  and extension paths pending separate slices.
- **kwavers-solver PSTD implementation full-k-space step direct Rayon edge -
  RESOLVED [patch].**
  `forward::pstd::implementation::core::stepper::step` now routes dynamic
  pressure-source accumulation, velocity-source gradient accumulation,
  spectral wave-coefficient multiplication, and propagated pressure/source
  updates through Moirai-backed helpers for standard-layout dense fields
  instead of ndarray/Rayon `Zip::par_for_each`. Non-standard ndarray layouts
  retain sequential explicit indexing without cloning. Evidence tier: static
  source audit plus compile-time and focused empirical validation; scoped `rg`
  finds no direct Rayon, ndarray-parallel, or explicit `Zip` token in
  `forward/pstd/implementation/core/stepper/step.rs`, `rustup run nightly
  cargo fmt -p kwavers-solver --check` passes, `rustup run nightly cargo check
  -p kwavers-solver` passes, `rustup run nightly cargo clippy -p
  kwavers-solver --lib -- -D warnings` passes, and focused `rustup run nightly
  cargo nextest run -p kwavers-solver -E "test(source) or test(step) or
  test(kspace) or test(fullkspace) or test(pstd)" --status-level fail
  --no-fail-fast` passes 231/231. Residual: broader PSTD direct-provider
  holdouts remain in source handlers, construction paths, derivatives,
  physics, and extension paths pending separate slices.
- **kwavers-solver PSTD implementation source-handler direct Rayon edge -
  RESOLVED [patch].**
  `forward::pstd::implementation::core::stepper::sources` now routes
  source-gain scaling, dynamic pressure-source accumulation, source-kappa
  spectral multiplication, split-density component injection, and dynamic
  velocity-source writes through shared Moirai-backed stepper helpers instead
  of ndarray `mapv_inplace` and ndarray/Rayon `Zip::par_for_each`. The
  anti-aliasing filter and full-k-space step paths now share the same
  `stepper::ops` dense traversal helpers rather than retaining local helper
  variants. Non-standard ndarray layouts retain sequential explicit indexing
  without cloning. Evidence tier: static source audit plus compile-time and
  focused empirical validation; scoped `rg` finds no direct Rayon,
  ndarray-parallel, or explicit `Zip` token in
  `forward/pstd/implementation/core/stepper/{filter,step,sources,ops}.rs`,
  `rustup run nightly cargo fmt -p kwavers-solver --check` passes, `rustup run
  nightly cargo check -p kwavers-solver` passes, `rustup run nightly cargo
  clippy -p kwavers-solver --lib -- -D warnings` passes, and focused `rustup
  run nightly cargo nextest run -p kwavers-solver -E "test(source) or
  test(step) or test(kspace) or test(fullkspace) or test(anti_aliasing) or
  test(filter) or test(pstd)" --status-level fail --no-fail-fast` passes
  234/234. Residual: broader PSTD direct-provider holdouts remain in
  construction, thermal/orchestrator, derivatives, physics, and extension paths
  pending separate slices.
- **kwavers-solver PSTD implementation density-sum direct Rayon edge -
  RESOLVED [patch].**
  `forward::pstd::implementation::core::orchestrator::PSTDSolver::fill_rho_sum`
  now routes total split-density accumulation through a Moirai-backed helper
  for standard-layout dense fields instead of ndarray/Rayon
  `Zip::par_for_each`. Non-standard ndarray layouts retain sequential explicit
  indexing without cloning. Evidence tier: static source audit plus
  compile-time and focused empirical validation; scoped `rg` finds no direct
  Rayon, ndarray-parallel, or explicit `Zip` token in
  `forward/pstd/implementation/core/orchestrator/mod.rs`, `rustup run nightly
  cargo fmt -p kwavers-solver --check` passes, `rustup run nightly cargo check
  -p kwavers-solver` passes, `rustup run nightly cargo clippy -p
  kwavers-solver --lib -- -D warnings` passes, and focused `rustup run nightly
  cargo nextest run -p kwavers-solver -E "test(rho_sum) or test(source) or
  test(step) or test(pstd)" --status-level fail --no-fail-fast` passes
  208/208. Residual: broader PSTD direct-provider holdouts remain in
  construction, thermal/orchestrator, derivatives, physics, and extension paths
  pending separate slices.
- **kwavers-solver PSTD implementation thermal absorption direct Rayon edge -
  RESOLVED [patch].**
  `forward::pstd::implementation::core::orchestrator::thermal::populate_alpha_np_m_at_frequency`
  now routes Stokes and power-law absorption coefficient scaling through one
  Moirai-backed helper for standard-layout dense fields instead of
  ndarray/Rayon `Zip::par_for_each`. Non-standard ndarray layouts retain
  sequential explicit indexing without cloning. Evidence tier: static source
  audit plus compile-time and focused empirical validation; scoped `rg` finds
  no direct Rayon, ndarray-parallel, or explicit `Zip` token in
  `forward/pstd/implementation/core/orchestrator/thermal.rs`, `rustup run
  nightly cargo fmt -p kwavers-solver --check` passes, `rustup run nightly
  cargo check -p kwavers-solver` passes, `rustup run nightly cargo clippy -p
  kwavers-solver --lib -- -D warnings` passes, and focused `rustup run nightly
  cargo nextest run -p kwavers-solver -E "test(thermal) or test(pstd) or
  test(acoustic_heat) or test(absorption)" --status-level fail --no-fail-fast`
  passes 206/206. Residual: broader PSTD direct-provider holdouts remain in
  construction, derivatives, physics, and extension paths pending separate
  slices.
- **kwavers-solver PSTD implementation construction direct Rayon edge -
  RESOLVED [patch].**
  `forward::pstd::implementation::core::orchestrator::construction` now routes
  source-kappa cosine transformation and initial split-density component fills
  through Moirai-backed helpers for standard-layout dense fields instead of
  ndarray `par_mapv_inplace` and ndarray/Rayon `Zip::par_for_each`.
  Non-standard ndarray layouts retain sequential explicit indexing without
  cloning. Evidence tier: static source audit plus compile-time and focused
  empirical validation; scoped `rg` finds no direct Rayon, ndarray-parallel, or
  explicit `Zip` token in
  `forward/pstd/implementation/core/orchestrator/construction/mod.rs`, `rustup
  run nightly cargo fmt -p kwavers-solver --check` passes, `rustup run nightly
  cargo check -p kwavers-solver` passes, `rustup run nightly cargo clippy -p
  kwavers-solver --lib -- -D warnings` passes, and focused `rustup run nightly
  cargo nextest run -p kwavers-solver -E "test(construction) or test(pstd) or
  test(initial_pressure) or test(ivp) or test(kappa)" --status-level fail
  --no-fail-fast` passes 209/209.
- **kwavers-solver PSTD implementation IVP velocity direct Rayon edge -
  RESOLVED [patch].**
  `forward::pstd::implementation::core::orchestrator::construction::ivp_velocity`
  now routes IVP density seeding, k-space-corrected spectral-gradient
  construction, and half-step velocity density scaling through Moirai-backed
  helpers for standard-layout dense fields instead of ndarray/Rayon
  `Zip::par_for_each`. Non-standard ndarray layouts retain sequential explicit
  indexing without cloning. Evidence tier: static source audit plus
  compile-time and focused empirical validation; scoped `rg` finds no direct
  Rayon, ndarray-parallel, or explicit `Zip` token in
  `forward/pstd/implementation/core/orchestrator/construction/ivp_velocity.rs`,
  `rustup run nightly cargo fmt -p kwavers-solver --check` passes, `rustup run
  nightly cargo check -p kwavers-solver` passes, `rustup run nightly cargo
  clippy -p kwavers-solver --lib -- -D warnings` passes, and focused `rustup
  run nightly cargo nextest run -p kwavers-solver -E "test(ivp) or
  test(initial_pressure) or test(construction) or test(pstd) or test(kappa)"
  --status-level fail --no-fail-fast` passes 209/209. Residual: scoped
  PSTD implementation-core direct-provider audit now reports no hits.
- **kwavers-solver PSTD spectral-correction direct Rayon edge - RESOLVED
  [patch].**
  `forward::pstd::numerics::spectral_correction::corrections` now routes
  exact-dispersion, Treeby2010, Liu PSTD, low-dispersion PSTD, sinc-spatial
  kappa generation, and correction application through Moirai-backed dense
  helpers for standard-layout fields instead of ndarray/Rayon
  `Zip::par_for_each`. Non-standard ndarray layouts retain sequential explicit
  indexing without cloning. Evidence tier: static source audit plus
  compile-time and focused empirical validation; scoped `rg` finds no direct
  Rayon, ndarray-parallel, or explicit `Zip` token in
  `forward/pstd/numerics/spectral_correction/corrections.rs`, `rustup run
  nightly cargo fmt -p kwavers-solver --check` passes, `rustup run nightly
  cargo check -p kwavers-solver` passes, `rustup run nightly cargo clippy -p
  kwavers-solver --lib -- -D warnings` passes, and focused `rustup run nightly
  cargo nextest run -p kwavers-solver -E "test(spectral_correction) or
  test(kappa) or test(wavenumber) or test(pstd)" --status-level fail
  --no-fail-fast` passes 175/175. Residual: broader PSTD direct-provider
  holdouts remain in propagator, physics, derivatives, DG, and elastic
  extension paths pending separate slices.
- **kwavers-solver PSTD propagator pressure direct Rayon edge - RESOLVED
  [patch].**
  `forward::pstd::propagator::pressure::update_pressure` now routes
  split-density accumulation, nonlinear equation-of-state pressure writes, and
  fused linear pressure writes through Moirai-backed dense helpers for
  standard-layout fields instead of ndarray/Rayon `Zip::par_for_each`.
  Non-standard ndarray layouts retain sequential explicit indexing without
  cloning. Evidence tier: static source audit plus compile-time and focused
  empirical validation; scoped `rg` finds no direct Rayon, ndarray-parallel,
  or explicit `Zip` token in `forward/pstd/propagator/pressure/mod.rs`,
  `rustup run nightly cargo fmt -p kwavers-solver --check` passes, `rustup run
  nightly cargo check -p kwavers-solver` passes, `rustup run nightly cargo
  clippy -p kwavers-solver --lib -- -D warnings` passes, and focused `rustup
  run nightly cargo nextest run -p kwavers-solver -E "test(pressure) or
  test(density) or test(pstd) or test(kappa)" --status-level fail
  --no-fail-fast` passes 203/203. Residual: pressure-density Cartesian/AS
  propagation and broader PSTD direct-provider holdouts remain for separate
  slices.
- **kwavers-solver PSTD Cartesian density direct Rayon edge - RESOLVED
  [patch].**
  `forward::pstd::propagator::pressure::density_cartesian` now routes
  shifted-kappa spectral multiplication, nonlinear density coefficient
  construction, fused PML density updates, and fallback pre/post-PML density
  updates through Moirai-backed dense helpers for standard-layout fields
  instead of ndarray/Rayon `Zip::par_for_each`. Non-standard ndarray layouts
  retain sequential explicit indexing without cloning. Evidence tier: static
  source audit plus compile-time and focused empirical validation; scoped `rg`
  finds no direct Rayon, ndarray-parallel, or explicit `Zip` token in
  `forward/pstd/propagator/pressure/density_cartesian.rs`, `rustup run
  nightly cargo fmt -p kwavers-solver --check` passes, `rustup run nightly
  cargo check -p kwavers-solver` passes, `rustup run nightly cargo clippy -p
  kwavers-solver --lib -- -D warnings` passes, and focused `rustup run nightly
  cargo nextest run -p kwavers-solver -E "test(density) or test(pressure) or
  test(pstd) or test(kappa)" --status-level fail --no-fail-fast` passes
  203/203. Residual: axisymmetric pressure-density propagation remains for a
  separate slice.
- **kwavers-solver PSTD axisymmetric density direct Rayon edge - RESOLVED
  [patch].**
  `forward::pstd::propagator::pressure::density_as` now routes nonlinear
  coefficient construction, fused PML axisymmetric density updates, and
  fallback pre/post-PML density updates through Moirai-backed dense helpers
  for standard-layout 2-D views instead of ndarray/Rayon `Zip::par_for_each`.
  Non-standard ndarray layouts retain sequential explicit indexing without
  cloning. Evidence tier: static source audit plus compile-time and focused
  empirical validation; scoped `rg` finds no direct Rayon, ndarray-parallel,
  or explicit `Zip` token in `forward/pstd/propagator/pressure/density_as.rs`,
  `rustup run nightly cargo fmt -p kwavers-solver --check` passes, `rustup run
  nightly cargo check -p kwavers-solver` passes, `rustup run nightly cargo
  clippy -p kwavers-solver --lib -- -D warnings` passes, and focused `rustup
  run nightly cargo nextest run -p kwavers-solver -E "test(density) or
  test(pressure) or test(axisymmetric) or test(pstd) or test(kappa)"
  --status-level fail --no-fail-fast` passes 203/203. Follow-up subtree audit
  finds no direct Rayon, ndarray-parallel, or explicit `Zip` token under
  `forward/pstd/propagator/pressure`.
- **kwavers-solver PSTD velocity propagator direct Rayon edge - RESOLVED
  [patch].**
  `forward::pstd::propagator::velocity` now routes Cartesian shifted-kappa
  spectral-gradient construction, fused Cartesian velocity updates, fallback
  Cartesian velocity updates, fused axisymmetric velocity updates, and fallback
  axisymmetric velocity updates through Moirai-backed dense helpers for
  standard-layout fields instead of ndarray/Rayon `Zip::par_for_each`.
  Non-standard ndarray layouts retain sequential explicit indexing without
  cloning. Evidence tier: static source audit plus compile-time and focused
  empirical validation; scoped `rg` finds no direct Rayon, ndarray-parallel,
  or explicit `Zip` token in `forward/pstd/propagator/velocity.rs`, `rustup
  run nightly cargo fmt -p kwavers-solver --check` passes, `rustup run nightly
  cargo check -p kwavers-solver` passes, `rustup run nightly cargo clippy -p
  kwavers-solver --lib -- -D warnings` passes, and focused `rustup run
  nightly cargo nextest run -p kwavers-solver -E "test(velocity) or
  test(pressure) or test(density) or test(axisymmetric) or test(pstd) or
  test(kappa)" --status-level fail --no-fail-fast` passes 210/210.
- **kwavers-solver PSTD axisymmetric propagator direct Rayon edge - RESOLVED
  [patch].**
  `forward::pstd::propagator::axisymmetric` now routes WSWA-FFT kappa
  multiplication, row/column spectral operators, real-window extraction,
  radial velocity quotient construction, and radial divergence composition
  through Moirai-backed dense helpers for standard-layout buffers. The
  non-contiguous expansion-slice assignments now use explicit indexed loops
  instead of ndarray `Zip`. Evidence tier: static source audit plus
  compile-time and focused empirical validation; scoped `rg` finds no direct
  Rayon, ndarray-parallel, or explicit `Zip` token in
  `forward/pstd/propagator/axisymmetric.rs`, `rustup run nightly cargo fmt -p
  kwavers-solver --check` passes, `rustup run nightly cargo check -p
  kwavers-solver` passes, `rustup run nightly cargo clippy -p kwavers-solver
  --lib -- -D warnings` passes, and focused `rustup run nightly cargo nextest
  run -p kwavers-solver -E "test(axisymmetric) or test(velocity) or
  test(density) or test(pressure) or test(pstd) or test(kappa)"
  --status-level fail --no-fail-fast` passes 210/210.
- **kwavers-solver PSTD residual-gas absorption direct Rayon edge - RESOLVED
  [patch].**
  `forward::pstd::physics::residual_gas_absorption` now routes broadband
  attenuation spectral-shape multiplication, pressure loss application,
  dispersion spectral-shape multiplication, and pressure dispersion application
  through Moirai-backed dense helpers for standard-layout fields instead of
  ndarray/Rayon `Zip::par_for_each`. Non-standard layouts retain explicit
  sequential indexing without cloning. Evidence tier: static source audit plus
  compile-time and focused empirical validation; scoped `rg` finds no direct
  Rayon, ndarray-parallel, or explicit `Zip` token in
  `forward/pstd/physics/residual_gas_absorption.rs`, `rustup run nightly cargo
  fmt -p kwavers-solver --check` passes, `rustup run nightly cargo check -p
  kwavers-solver` passes, `rustup run nightly cargo clippy -p kwavers-solver
  --lib -- -D warnings` passes, and focused `rustup run nightly cargo nextest
  run -p kwavers-solver -E "test(residual_gas) or test(absorption) or
  test(pressure) or test(pstd) or test(kappa)" --status-level fail
  --no-fail-fast` passes 213/213.
- **kwavers-solver PSTD pressure absorption direct Rayon edge - RESOLVED
  [patch].**
  `forward::pstd::physics::absorption::apply` now routes weighted divergence
  construction, spectral operator multiplication, stratified bracket-weight
  accumulation, and final pressure correction through Moirai-backed dense
  helpers instead of ndarray/Rayon `Zip::par_for_each`. Non-standard layouts
  and sliced spectral operators retain explicit indexed fallback semantics
  without cloning. Evidence tier: static source audit plus compile-time and
  focused empirical validation; scoped `rg` finds no direct Rayon,
  ndarray-parallel, or explicit `Zip` token in
  `forward/pstd/physics/absorption/apply.rs`, `rustup run nightly cargo fmt -p
  kwavers-solver --check` passes, `rustup run nightly cargo check -p
  kwavers-solver` passes, `rustup run nightly cargo clippy -p kwavers-solver
  --lib -- -D warnings` passes, and focused `rustup run nightly cargo nextest
  run -p kwavers-solver -E "test(absorption) or test(pressure) or test(pstd)
  or test(kappa)" --status-level fail --no-fail-fast` passes 211/211.
- **kwavers-solver PSTD absorption strata direct Zip edge - RESOLVED [patch].**
  `forward::pstd::physics::absorption::strata` now computes per-voxel
  `(lower_stratum, upper_weight)` bracket pairs through Moirai indexed
  collection, then writes the two output arrays explicitly. Evidence tier:
  static source audit plus compile-time and focused empirical validation;
  scoped `rg` finds no direct Rayon, ndarray-parallel, or explicit `Zip` token
  under `forward/pstd/physics/absorption`, `rustup run nightly cargo fmt -p
  kwavers-solver --check` passes, `rustup run nightly cargo check -p
  kwavers-solver` passes, `rustup run nightly cargo clippy -p kwavers-solver
  --lib -- -D warnings` passes, and focused `rustup run nightly cargo nextest
  run -p kwavers-solver -E "test(absorption) or test(pressure) or test(pstd)
  or test(kappa)" --status-level fail --no-fail-fast` passes 211/211.
- **kwavers-solver PSTD derivative operator direct Rayon edge - RESOLVED
  [patch].**
  `forward::pstd::derivatives::operator` now routes x-pencil spectral
  derivatives through Moirai indexed collection and y/z pencil derivatives
  through Moirai chunked i-slab traversal instead of ndarray/Rayon
  `axis_iter_mut(...).into_par_iter()`. Evidence tier: static source audit plus
  compile-time and focused empirical validation; scoped `rg` finds no direct
  Rayon, ndarray-parallel, or explicit `Zip` token in
  `forward/pstd/derivatives/operator.rs`, `rustup run nightly cargo fmt -p
  kwavers-solver --check` passes, `rustup run nightly cargo check -p
  kwavers-solver` passes, `rustup run nightly cargo clippy -p kwavers-solver
  --lib -- -D warnings` passes, and focused `rustup run nightly cargo nextest
  run -p kwavers-solver -E "test(derivative) or test(spectral) or test(pstd)
  or test(kappa)" --status-level fail --no-fail-fast` passes 214/214.
- **kwavers-solver PSTD DG spectral solver direct Zip edge - RESOLVED
  [patch].**
  `forward::pstd::dg::spectral_solver` now routes Laplacian-symbol
  construction and spectral Laplacian application through Moirai-backed dense
  helpers instead of explicit ndarray `Zip` traversal. Evidence tier: static
  source audit plus compile-time and focused empirical validation; scoped `rg`
  finds no direct Rayon, ndarray-parallel, or explicit `Zip` token in
  `forward/pstd/dg/spectral_solver.rs`, `rustup run nightly cargo fmt -p
  kwavers-solver --check` passes, `rustup run nightly cargo check -p
  kwavers-solver` passes, `rustup run nightly cargo clippy -p kwavers-solver
  --lib -- -D warnings` passes, and focused `rustup run nightly cargo nextest
  run -p kwavers-solver -E "test(dg) or test(spectral) or test(pstd) or
  test(kappa)" --status-level fail --no-fail-fast` passes 210/210. Current
  residual after the follow-up CPML slice: the DG subtree has no direct Rayon,
  ndarray-parallel, or explicit `Zip` traversal holdouts.
- **kwavers-solver PSTD DG acoustic SSP-RK direct Zip edge - RESOLVED
  [patch].**
  `forward::pstd::dg::dg_solver::acoustic` now routes pressure and velocity
  SSP-RK Euler, second-stage, and final-stage coefficient updates through
  Moirai-backed dense helpers instead of explicit ndarray `Zip` traversal.
  Evidence tier: static source audit plus compile-time and focused empirical
  validation; scoped `rg` finds no direct Rayon, ndarray-parallel, or explicit
  `Zip` token in `forward/pstd/dg/dg_solver/acoustic.rs`, `rustup run nightly
  cargo fmt -p kwavers-solver --check` passes, `rustup run nightly cargo check
  -p kwavers-solver` passes, `rustup run nightly cargo clippy -p
  kwavers-solver --lib -- -D warnings` passes, and focused `rustup run nightly
  cargo nextest run -p kwavers-solver -E "test(dg) or test(spectral) or
  test(pstd) or test(kappa)" --status-level fail --no-fail-fast` passes
  210/210. Current residual after the follow-up CPML slice: the DG subtree has
  no direct Rayon, ndarray-parallel, or explicit `Zip` traversal holdouts.
- **kwavers-solver PSTD DG modal RK direct Zip edge - RESOLVED [patch].**
  `forward::pstd::dg::dg_solver::solver_ops` now routes SSP-RK3 and Forward
  Euler modal coefficient updates through shared Moirai-backed dense RK
  helpers in `forward::pstd::dg::dg_solver::rk_update`. The one-dimensional
  acoustic SSP-RK path uses the same helper module, removing duplicated stage
  algebra. Evidence tier: static source audit plus compile-time and focused
  empirical validation; scoped `rg` finds no direct Rayon, ndarray-parallel, or
  explicit `Zip` token in `forward/pstd/dg/dg_solver/solver_ops.rs`,
  `forward/pstd/dg/dg_solver/acoustic.rs`, or
  `forward/pstd/dg/dg_solver/rk_update.rs`, `rustup run nightly cargo fmt -p
  kwavers-solver --check` passes, `rustup run nightly cargo check -p
  kwavers-solver` passes, `rustup run nightly cargo clippy -p kwavers-solver
  --lib -- -D warnings` passes, and focused `rustup run nightly cargo nextest
  run -p kwavers-solver -E "test(dg) or test(spectral) or test(pstd) or
  test(kappa)" --status-level fail --no-fail-fast` passes 210/210. Current
  residual after the follow-up CPML slice: the DG subtree has no direct Rayon,
  ndarray-parallel, or explicit `Zip` traversal holdouts.
- **kwavers-solver PSTD DG tensor source direct Zip edge - RESOLVED [patch].**
  `forward::pstd::dg::dg_solver::acoustic::tensor::source` now routes
  source-coupled SSP-RK3 state updates through the shared Moirai-backed dense
  RK helpers in `forward::pstd::dg::dg_solver::rk_update`. Evidence tier:
  static source audit plus compile-time and focused empirical validation;
  scoped `rg` finds no direct Rayon, ndarray-parallel, or explicit `Zip` token
  in `forward/pstd/dg/dg_solver/acoustic/tensor/source.rs` or
  `forward/pstd/dg/dg_solver/rk_update.rs`, `rustup run nightly cargo fmt -p
  kwavers-solver --check` passes, `rustup run nightly cargo check -p
  kwavers-solver` passes, `rustup run nightly cargo clippy -p kwavers-solver
  --lib -- -D warnings` passes, and focused `rustup run nightly cargo nextest
  run -p kwavers-solver -E "test(dg) or test(spectral) or test(pstd) or
  test(kappa)" --status-level fail --no-fail-fast` passes 210/210. Current
  residual after the follow-up CPML slice: the DG subtree has no direct Rayon,
  ndarray-parallel, or explicit `Zip` traversal holdouts.
- **kwavers-solver PSTD DG tensor CPML direct Zip edge - RESOLVED [patch].**
  `forward::pstd::dg::dg_solver::acoustic::tensor::cpml` now routes field and
  CPML memory SSP-RK3 state updates through the shared Moirai-backed dense RK
  helpers in `forward::pstd::dg::dg_solver::rk_update`. Evidence tier: static
  source audit plus compile-time and focused empirical validation; scoped `rg`
  finds no direct Rayon, ndarray-parallel, or explicit `Zip` token in
  `crates/kwavers-solver/src/forward/pstd/dg`, `rustup run nightly cargo fmt
  -p kwavers-solver --check` passes, `rustup run nightly cargo check -p
  kwavers-solver` passes, `rustup run nightly cargo clippy -p kwavers-solver
  --lib -- -D warnings` passes, and focused `rustup run nightly cargo nextest
  run -p kwavers-solver -E "test(dg) or test(spectral) or test(pstd) or
  test(kappa)" --status-level fail --no-fail-fast` passes 210/210. Residual:
  broader `kwavers-solver` direct Rayon/ndarray-parallel holdouts remain
  outside the DG subtree.
- **Thermal CEM43 Leto state - RESOLVED [patch].**
  `kwavers_physics::thermal::ThermalCEM43Grid` now owns
  `leto::Array3<f64>` dose storage, accepts Leto temperature fields, and
  schedules dense updates through Moirai instead of ndarray/Rayon
  `Zip::par_for_each`. The top-level theranostic lesion mask consumes that
  Leto CEM43 field directly, and `brain_theranostic_monitor` keeps its thermal
  temperature and absorbed-power fields in Leto. Evidence tier: compile-time
  validation plus focused empirical tests; `cargo check -p kwavers-physics`
  passes, `cargo check -p kwavers --example brain_theranostic_monitor` passes,
  focused `kwavers-physics` thermal-dose nextest passes 12/12, and focused
  `kwavers --lib` lesion nextest passes 10/10. Residual: solver
  `ThermalDoseCalculator` and the FWI sound-speed reconstruction path still
  carry ndarray boundaries. The then-existing top-level Leto/RITK migration
  failures in `skull_ct_phase_correction`, `ultrasound_physics_validation`,
  and `nl_swe_validation` are closed by the follow-up item below.
- **Top-level Leto/RITK stale API blockers - RESOLVED [patch].** The skull CT
  phase-correction example now uses current RITK series accessors and typed
  spacing/origin extraction, ultrasound fusion/registration validation feeds
  Leto arrays directly into the migrated registration/fusion APIs, and NL-SWE
  validation constructs Leto parameter fields directly. The constant-quality
  NL-SWE mean assertion uses a reduction-size-derived `f64::EPSILON` bound
  instead of exact equality on a floating-point reduction. Evidence tier:
  compile-time validation plus focused empirical tests; `rustup run nightly
  cargo check -p kwavers --example skull_ct_phase_correction --test
  ultrasound_physics_validation --test nl_swe_validation` passes, focused
  ultrasound validation nextest passes 5/5, and focused NL-SWE validation
  nextest passes 2/2.
- **kwavers-solver thermal diffusion direct Rayon edge - RESOLVED [patch].**
  `ThermalDiffusionSolver::update_standard_diffusion` now uses
  `moirai-parallel` chunk scheduling for dense owned temperature and Laplacian
  buffers instead of ndarray/Rayon `Zip::par_for_each`. Borrowed source views
  are shape-validated before indexing; dense source views participate in the
  Moirai traversal, while non-contiguous borrowed views retain sequential
  ndarray view semantics without a source clone or concrete Rayon path.
  Evidence tier: static source audit plus compile-time and focused empirical
  validation; `rustup run nightly cargo check -p kwavers-solver` passes,
  `rustup run nightly cargo clippy -p kwavers-solver --lib -- -D warnings`
  passes, focused thermal diffusion nextest passes 7/7 with 934 skipped,
  rustfmt `--check` passes, and scoped `rg` finds no direct Rayon hits under
  `crates/kwavers-solver/src/forward/thermal_diffusion/solver`. Residual: the
  thermal solver still owns ndarray storage boundaries pending the Leto plus
  CPU/GPU backend-trait migration.
- **kwavers-solver thermal-acoustic direct Rayon edge - RESOLVED [patch].**
  `forward::coupled::thermal_acoustic` material-property, acoustic-heating,
  acoustic-step, and thermal-step kernels now use `moirai-parallel` dense
  traversal instead of ndarray/Rayon `par_for_each`. Sequential ndarray
  traversal remains only as the non-standard-layout fallback. Evidence tier:
  static source audit plus compile-time and focused empirical validation;
  `rustup run nightly cargo check -p kwavers-solver` passes, `rustup run
  nightly cargo clippy -p kwavers-solver --lib -- -D warnings` passes,
  focused thermal-acoustic nextest passes 9/9 with 934 skipped, rustfmt
  `--check` passes, and scoped `rg` finds no direct Rayon hits under
  `crates/kwavers-solver/src/forward/coupled/thermal_acoustic`. Residual:
  manifest-level solver Rayon remains until the remaining solver holdout
  kernels migrate.
- **kwavers-solver BEM scattered-field direct Rayon edge - RESOLVED [patch].**
  `BemSolver::compute_scattered_field` now uses `moirai-parallel` ordered
  map-collect instead of direct Rayon `par_iter`, preserving the BEM
  representation formula and output order. Evidence tier: static source audit
  plus compile-time and focused empirical validation; `rustup run nightly
  cargo check -p kwavers-solver` passes, `rustup run nightly cargo clippy -p
  kwavers-solver --lib -- -D warnings` passes, focused BEM nextest passes
  65/65 with 878 skipped, rustfmt `--check` passes, and scoped `rg` finds no
  direct Rayon hits in
  `crates/kwavers-solver/src/forward/bem/solver/solution.rs`. Residual:
  manifest-level solver Rayon remains until the remaining solver holdout
  kernels migrate.
- **kwavers-solver legacy seismic RTM direct Rayon edge - RESOLVED [patch].**
  `inverse::seismic::rtm` zero-lag and normalized imaging-condition passes now
  use `moirai-parallel` dense traversal instead of ndarray/Rayon
  `par_for_each`, with sequential ndarray fallback for non-standard layouts.
  The normalized single-snapshot formula computes source illumination inline
  rather than allocating a temporary array. Evidence tier: static source audit
  plus compile-time and focused empirical validation; `rustup run nightly
  cargo check -p kwavers-solver` passes, `rustup run nightly cargo clippy -p
  kwavers-solver --lib -- -D warnings` passes, focused legacy RTM nextest
  passes 3/3 with 940 skipped, rustfmt `--check` passes, and scoped `rg`
  finds no direct Rayon hits in
  `crates/kwavers-solver/src/inverse/seismic/rtm.rs`. Residual: the full
  `inverse::reconstruction::seismic::rtm` stack still has direct
  ndarray/Rayon holdouts.
- **kwavers-solver photoacoustic line reconstruction direct Rayon edge -
  RESOLVED [patch].** `kspace_line_recon` now applies optional positivity
  clamping through `moirai-parallel` dense traversal instead of ndarray/Rayon
  `par_mapv_inplace`, with sequential ndarray fallback for non-standard
  layouts. Evidence tier: static source audit plus compile-time and focused
  empirical validation; `rustup run nightly cargo check -p kwavers-solver`
  passes, `rustup run nightly cargo clippy -p kwavers-solver --lib -- -D
  warnings` passes, focused line reconstruction nextest passes 4/4 with 940
  skipped, rustfmt `--check` passes, and scoped `rg` finds no direct Rayon
  hits in
  `crates/kwavers-solver/src/inverse/reconstruction/photoacoustic/line_reconstruction.rs`.
  Current residual after the follow-up reconstruction slice: the
  photoacoustic reconstruction subtree has no direct Rayon, ndarray-parallel,
  or explicit `Zip` traversal holdouts.
- **kwavers-solver photoacoustic reconstruction direct Rayon edge - RESOLVED
  [patch].** `inverse::reconstruction::photoacoustic::iterative` now routes
  ART row updates and OSEM/nonnegative clamps through Moirai-backed traversal.
  `photoacoustic::fourier` now applies positivity clamping through the shared
  core Moirai-backed `apply_inplace`, and `photoacoustic::time_reversal` now
  computes the k-space leapfrog update through Moirai indexed traversal with a
  sequential fallback for non-standard ndarray layouts. Evidence tier: static
  source audit plus compile-time and focused empirical validation; scoped `rg`
  finds no direct Rayon, ndarray-parallel, or explicit `Zip` token in
  `crates/kwavers-solver/src/inverse/reconstruction/photoacoustic`, `rustup
  run nightly cargo fmt -p kwavers-solver --check` passes, `rustup run
  nightly cargo check -p kwavers-solver` passes, `rustup run nightly cargo
  clippy -p kwavers-solver --lib -- -D warnings` passes, and focused `rustup
  run nightly cargo nextest run -p kwavers-solver -E "test(photoacoustic)"
  --status-level fail --no-fail-fast` passes 10/10. Residual: broader
  `kwavers-solver` direct Rayon/ndarray-parallel holdouts remain outside the
  photoacoustic reconstruction subtree.
- **kwavers-solver hybrid angular spectrum absorption direct Rayon edge -
  RESOLVED [patch].** `forward::nonlinear::hybrid_angular_spectrum::absorption`
  now routes broadband harmonic attenuation planes through the shared
  Moirai-backed dense `apply_inplace` traversal instead of ndarray/Rayon
  `par_mapv_inplace`. Evidence tier: static source audit plus compile-time and
  focused empirical validation; scoped `rg` finds no direct Rayon,
  ndarray-parallel, or explicit `Zip` token in
  `crates/kwavers-solver/src/forward/nonlinear/hybrid_angular_spectrum`,
  `rustup run nightly cargo fmt -p kwavers-solver --check` passes, `rustup run
  nightly cargo check -p kwavers-solver` passes, `rustup run nightly cargo
  clippy -p kwavers-solver --lib -- -D warnings` passes, and focused `rustup
  run nightly cargo nextest run -p kwavers-solver -E
  "test(hybrid_angular_spectrum) or test(absorption)" --status-level fail
  --no-fail-fast` passes 43/43. Residual: broader `kwavers-solver` direct
  Rayon/ndarray-parallel holdouts remain outside the HAS subtree.
- **kwavers-solver nonlinear elastic propagation damping direct Rayon edge -
  RESOLVED [patch].**
  `forward::elastic::nonlinear::solver::propagation` now routes fundamental,
  previous-fundamental, second-harmonic, and higher-harmonic attenuation maps
  through the shared Moirai-backed dense `apply_inplace` traversal instead of
  ndarray/Rayon `par_mapv_inplace`. Evidence tier: static source audit plus
  compile-time and focused empirical validation; scoped `rg` finds no
  `par_mapv_inplace` token in
  `crates/kwavers-solver/src/forward/elastic/nonlinear/solver/propagation.rs`,
  `rustup run nightly cargo fmt -p kwavers-solver --check` passes, `rustup run
  nightly cargo check -p kwavers-solver` passes, `rustup run nightly cargo
  clippy -p kwavers-solver --lib -- -D warnings` passes, and focused `rustup
  run nightly cargo nextest run -p kwavers-solver -E
  "test(nonlinear) or test(elastic) or test(harmonic) or test(propagation)"
  --status-level fail --no-fail-fast` passes 264/264. Current residual after
  the follow-up harmonic-generation and stepping slices: the nonlinear elastic
  subtree has no direct Rayon, ndarray-parallel, or explicit `Zip` traversal
  holdouts; broader `kwavers-solver` direct Rayon/ndarray-parallel holdouts
  remain outside this subtree.
- **kwavers-solver nonlinear elastic harmonic-generation direct Rayon edge -
  RESOLVED [patch].**
  `forward::elastic::nonlinear::solver::harmonics` now routes the
  second-harmonic Jacobi update and third/higher harmonic delta-fill passes
  through the shared Moirai-backed indexed traversal instead of ndarray/Rayon
  `Zip::par_for_each`. Evidence tier: static source audit plus compile-time
  and focused empirical validation; scoped `rg` finds no direct Rayon,
  ndarray-parallel, or explicit `Zip` token in
  `crates/kwavers-solver/src/forward/elastic/nonlinear/solver/harmonics.rs`,
  `rustup run nightly cargo fmt -p kwavers-solver --check` passes, `rustup run
  nightly cargo check -p kwavers-solver` passes, `rustup run nightly cargo
  clippy -p kwavers-solver --lib -- -D warnings` passes, and focused `rustup
  run nightly cargo nextest run -p kwavers-solver -E
  "test(nonlinear) or test(elastic) or test(harmonic) or test(propagation)"
  --status-level fail --no-fail-fast` passes 264/264. Current residual after
  the follow-up stepping slice: the nonlinear elastic subtree has no direct
  Rayon, ndarray-parallel, or explicit `Zip` traversal holdouts; broader
  `kwavers-solver` direct Rayon/ndarray-parallel holdouts remain.
- **kwavers-solver nonlinear elastic fundamental stepping direct Rayon edge -
  RESOLVED [patch].**
  `forward::elastic::nonlinear::solver::stepping` now routes independent
  `(j, k)` x-line TVD-RK2 updates through Moirai-backed line scheduling, then
  applies the computed lines in a separate write-back pass to avoid unsafe
  strided mutable aliasing. Evidence tier: static source audit plus
  compile-time and focused empirical validation; scoped `rg` finds no direct
  Rayon, ndarray-parallel, or explicit `Zip` token under
  `crates/kwavers-solver/src/forward/elastic/nonlinear`, `rustup run nightly
  cargo fmt -p kwavers-solver --check` passes, `rustup run nightly cargo check
  -p kwavers-solver` passes, `rustup run nightly cargo clippy -p
  kwavers-solver --lib -- -D warnings` passes, and focused `rustup run nightly
  cargo nextest run -p kwavers-solver -E
  "test(nonlinear) or test(elastic) or test(harmonic) or test(propagation)"
  --status-level fail --no-fail-fast` passes 264/264. Residual: broader
  `kwavers-solver` direct Rayon/ndarray-parallel holdouts remain outside the
  nonlinear elastic subtree.
- **Top-level Burn production dependency - RESOLVED [patch].** The workspace
  `burn`/`burn-ndarray` dependency aliases were unused by member manifests and
  were removed. The top-level `kwavers` crate no longer carries Burn as a
  normal dependency; Burn is now a `dev-dependency` there because only that
  package's examples, benches, and integration tests import Burn. Solver,
  analysis, and Python still retain Burn where source files directly depend on
  it pending the Coeus migration. Evidence tier: static source/manifest audit
  plus metadata validation; no `burn = { workspace = true }` or
  `burn-ndarray` hits remain, `crates/kwavers/src` has no Burn imports, and
  `rustup run nightly cargo metadata --no-deps --format-version 1
  --manifest-path Cargo.toml --features pinn` passed with `kwavers` reporting
  Burn as `kind=dev`.
- **GPU provider identity/dispatch split - RESOLVED [patch].**
  `kwavers-gpu::backend::GpuProviderBackend` now owns provider identity,
  Hephaestus device access, and synchronization, while
  `GpuComputeProvider` extends it only for providers that implement real
  Kwavers kernel dispatch. `GpuDeviceProvider` now exposes provider identity,
  so `GpuProviderContext<WgpuDevice>` and `GpuProviderContext<CudaDevice>`
  satisfy the generic provider contract without adding a fake CUDA compute
  implementation. Evidence tier: type-level/compile-time validation plus
  focused GPU tests; `cargo check -p kwavers-gpu --features gpu` passes,
  `cargo check -p kwavers-gpu --features cuda-provider --offline` passes,
  `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings` passes,
  `cargo clippy -p kwavers-gpu --features cuda-provider --all-targets
  --offline -- -D warnings` passes, `cargo nextest run -p kwavers-gpu
  --features gpu provider --status-level fail --no-fail-fast` passes 4/4,
  `cargo nextest run -p kwavers-gpu --features cuda-provider provider
  --status-level fail --no-fail-fast --offline` passes 7/7, and stale
  CUDA-as-compute source claims were not found in the touched provider
  surface.
- **Acoustic-field kernel provider seam - RESOLVED [patch].**
  `kwavers-gpu::gpu::compute_kernels::AcousticFieldKernel<P>` now delegates to
  an `AcousticFieldProvider` operation trait. The current WGSL pipeline, buffer
  allocation, dispatch, and readback path live in `WgpuAcousticFieldProvider`,
  so CUDA can only enter this kernel by implementing the same operation with
  real kernels and tests. Follow-up moved the operation surface and
  `WaveEquationGpu` from `ndarray::Array3<f64>` to provider-native
  `leto::Array3<f32>` for WGPU, with precision exposed as the provider's
  associated scalar rather than a hidden f64-to-f32 conversion. Evidence tier:
  type-level/compile-time validation plus focused GPU tests; `rustup run
  nightly cargo fmt -p kwavers-gpu --check`, `rustup run nightly cargo check
  -p kwavers-gpu --features gpu`, `rustup run nightly cargo clippy -p
  kwavers-gpu --features gpu --lib -- -D warnings`, focused
  acoustic/provider nextest 4/4, and `rustup run nightly cargo check -p
  kwavers-gpu --features cuda-provider` pass.
- **Thermal-acoustic buffer provider seam - RESOLVED [patch].**
  `kwavers-gpu::gpu::thermal_acoustic::GpuThermalAcousticBuffers<P>` now
  delegates to `ThermalAcousticBufferProvider`. The WGPU storage/uniform
  buffers, upload path, and readback path live in `WgpuThermalAcousticBuffers`,
  so the generic buffer wrapper no longer exposes public `wgpu::Buffer` fields.
  Follow-up moved WGPU upload/readback field I/O from
  `ndarray::Array3<f32>` to provider-native `leto::Array3<f32>`, with the
  scalar contract exposed as `ThermalAcousticBufferProvider::Scalar = f32`.
  Evidence tier: type-level/compile-time validation plus focused GPU tests;
  `rustup run nightly cargo fmt -p kwavers-gpu --check`, `rustup run nightly
  cargo check -p kwavers-gpu --features gpu`, `rustup run nightly cargo clippy
  -p kwavers-gpu --features gpu --lib -- -D warnings`, focused
  thermal-acoustic nextest 9/9, and `rustup run nightly cargo check -p
  kwavers-gpu --features cuda-provider` pass.
- **FDTD pressure provider-native I/O - RESOLVED [patch].**
  `kwavers-gpu::gpu::WgpuFdtd` pressure upload/readback now uses
  provider-native `leto::Array3<f32>` for the WGPU storage contract. Upload
  rejects non-dense Leto fields with `KwaversError::InvalidInput`, and
  readback reconstructs the Leto array without widening through `f64`. The
  WGSL-only public surfaces are named `WgpuFdtd` and
  `WgpuFdtdPressureDispatcher` without compatibility aliases, so CUDA remains a
  provider-trait implementation gap rather than sharing a generic GPU type.
  Follow-up provider-stack evidence on 2026-07-04: `WgpuFdtdPressureDispatcher`
  now acquires through `GpuProviderContext<WgpuDevice>` and implements
  `GpuKernelProvider`/`GpuProviderBackend`, removing raw
  `Arc<wgpu::Device>`/`Arc<wgpu::Queue>` ownership from the dispatcher boundary.
  Follow-up FDTD solver evidence on 2026-07-04: `FdtdGpuProvider` is the
  provider-generic operation seam for construction, pressure upload/readback,
  and step execution; `WgpuFdtd` is the current real WGSL implementation backed
  by `GpuProviderContext<WgpuDevice>`, and the top-level roundtrip tests bind
  through `P: FdtdGpuProvider<Scalar = f32>` without raw WGPU setup, Tokio, or
  ndarray.
  Evidence tier: type-level API/provider validation plus focused empirical tests;
  `rustup run nightly cargo fmt -p kwavers-gpu --check`, `rustup run nightly
  cargo check -p kwavers-gpu --features gpu --all-targets`, `rustup run nightly
  cargo check -p kwavers-gpu --features cuda-provider --all-targets`, focused
  FDTD/provider nextest under `gpu` (39/39) and `cuda-provider` (46/46),
  `rustup run nightly cargo nextest run -p kwavers --features gpu --test
  gpu_allocation_tracking_test --status-level fail --no-fail-fast` (4/4), and
  clippy for both feature sets plus the focused top-level test pass. Residual:
  CUDA FDTD execution remains open until real CUDA kernels and WGPU/CUDA
  differential tests exist.
- **FDTD CPU reference Leto pressure surface - RESOLVED [patch].**
  `kwavers-gpu::gpu::compute::FdtdCpuReferenceDispatcher::{update_pressure,
  update_pressure_into}` and boundary-zeroing now use `leto::Array3<f64>`
  instead of `ndarray::Array3<f64>`, preserving the f64 reference stencil while
  removing the local ndarray pressure API. The public type name no longer
  presents this CPU reference path as GPU dispatch. Dimension mismatch tests now
  assert the typed `KwaversError::InvalidInput` path. Evidence tier: type-level API
  validation plus focused empirical tests; `rustup run nightly cargo fmt -p
  kwavers-gpu --check`, `rustup run nightly cargo check -p kwavers-gpu
  --features gpu`, `rustup run nightly cargo clippy -p kwavers-gpu --features
  gpu --lib -- -D warnings`, `rustup run nightly cargo nextest run -p
  kwavers-gpu --features gpu fdtd_gpu --status-level fail --no-fail-fast`
  (5/5), and `rustup run nightly cargo check -p kwavers-gpu --features
  cuda-provider` pass.
- **ComputeManager provider-generic boundary - RESOLVED [patch].**
  `kwavers-gpu::gpu::ComputeManager<P>` is now generic over
  `GpuDeviceProvider`, stores `Option<GpuDevice<P>>`, and exposes raw
  `wgpu::Device`/`wgpu::Queue` helpers only on the `WgpuDevice`
  specialization. CPU-only use is explicit through `cpu_only()` instead of a
  silent acquisition fallback, and CUDA type-checks at this boundary without a
  placeholder compute implementation. Evidence tier: type-level API
  validation plus focused empirical tests; `rustup run nightly cargo fmt -p
  kwavers-gpu --check`, `rustup run nightly cargo check -p kwavers-gpu
  --features gpu`, `rustup run nightly cargo clippy -p kwavers-gpu --features
  gpu --lib -- -D warnings`, `rustup run nightly cargo check -p kwavers-gpu
  --features cuda-provider`, focused WGPU compute-manager nextest 2/2, and
  focused CUDA-provider compute-manager nextest 3/3 pass.
  Follow-up blocking-constructor evidence: `ComputeManager::new_blocking`
  now acquires through `GpuDevice<P>::try_create_with_features_and_limits`
  instead of `pollster::block_on(Self::new())`, so blocking construction shares
  the same provider-generic acquisition contract as the async constructor.
  Evidence tier: static source audit plus compile-time/lint/focused empirical
  validation; scoped `rg` finds no `pollster::block_on` in `compute_manager.rs`
  or `kwavers-gpu::backend`, `rustup run nightly cargo fmt -p kwavers-gpu
  --check` passes, `rustup run nightly cargo check -p kwavers-gpu --features
  cuda-provider` passes, `rustup run nightly cargo clippy -p kwavers-gpu
  --features cuda-provider --all-targets -- -D warnings` passes, and focused
  `rustup run nightly cargo nextest run -p kwavers-gpu --features
  cuda-provider compute_manager --status-level fail --no-fail-fast` passes 5/5.
- **ComputeBackend Leto array contract - RESOLVED [patch].**
  `kwavers_solver::backend::ComputeBackend` now accepts `leto::Array3` through
  an associated `Scalar`, so solver-owned backend dispatch is generic over the
  provider precision. `kwavers-gpu::backend::GPUBackend<P>` implements that
  contract with `P::Scalar`; current WGPU kernels remain `f32`, and CUDA stays
  acquisition-only until real operation kernels exist. Evidence tier:
  type-level API validation plus focused empirical tests; scoped `rg` finds no
  `ndarray::Array3`/`NdArray3` on `kwavers-solver::backend` or
  `kwavers-gpu::backend`, `rustup run nightly cargo fmt -p kwavers-solver -p
  kwavers-gpu --check` passes, `rustup run nightly cargo check -p
  kwavers-solver -p kwavers-gpu --features kwavers-gpu/cuda-provider` passes,
  `rustup run nightly cargo clippy -p kwavers-solver -p kwavers-gpu --features
  kwavers-gpu/cuda-provider --lib -- -D warnings` passes, and focused `rustup
  run nightly cargo nextest run -p kwavers-solver -p kwavers-gpu --features
  kwavers-gpu/cuda-provider -E "test(backend_surface_tests) or
  test(gpu_backend_is_generic_over_provider_trait) or
  test(gpu_compute_contract_is_composed_from_operation_traits) or
  test(solver_compute_backend_uses_provider_native_scalar) or
  test(solver_compute_backend_dispatches_provider_native_values_when_gpu_available)
  or test(cuda_satisfies_provider_identity_without_fake_kernels) or
  test(cuda_device_satisfies_kwavers_provider_contract)" --status-level fail
  --no-fail-fast` passes 9/9. Broader solver `--all-targets` clippy is still
  blocked by pre-existing test/doc lints outside this backend slice.
- **ComputeManager Leto CPU field helpers - RESOLVED [patch].**
  `kwavers-gpu::gpu::ComputeManager::{fdtd_update, apply_absorption}` and
  their private CPU implementations now use `leto::Array3<f64>` instead of
  `ndarray::Array3<f64>`. Absorption updates reject shape mismatches and
  non-dense Leto layouts with `KwaversError::InvalidInput`. Evidence tier:
  type-level API validation plus focused empirical tests; `rustup run nightly
  cargo fmt -p kwavers-gpu --check`, `rustup run nightly cargo check -p
  kwavers-gpu --features gpu`, `rustup run nightly cargo clippy -p
  kwavers-gpu --features gpu --lib -- -D warnings`, `rustup run nightly cargo
  check -p kwavers-gpu --features cuda-provider`, focused WGPU compute-manager
  nextest 3/3, and focused CUDA-provider compute-manager nextest 4/4 pass.
- **GPU/CPU equivalence Leto comparator - RESOLVED [patch].**
  `kwavers-gpu::validation::gpu_cpu_equivalence::EquivalenceValidator` now
  compares `leto::Array3<f64>` pressure fields directly and no longer depends
  on `ndarray::Zip`. The validation runner converts the current solver-owned
  ndarray pressure field into Leto at the boundary, leaving source mask/signal
  ndarray use scoped to the existing FDTD solver API. The runner no longer
  constructs `GPUBackend` and then runs the CPU `FdtdSolver` as GPU evidence;
  it returns a typed unavailable-provider report until a real provider-generic
  Leto/Hephaestus FDTD GPU trait implementation exists. Evidence tier:
  type-level API validation plus focused empirical tests; `rustup run nightly
  cargo fmt -p kwavers-gpu --check`, `rustup run nightly cargo check -p
  kwavers-gpu --features gpu`, `rustup run nightly cargo clippy -p
  kwavers-gpu --features gpu --all-targets -- -D warnings`, `rustup run
  nightly cargo nextest run -p kwavers-gpu --features gpu gpu_cpu_equivalence
  --status-level fail --no-fail-fast` (21/21), and `rustup run nightly cargo
  check -p kwavers-gpu --features cuda-provider` pass. Residual: real FDTD
  GPU equivalence remains open until the FDTD solver has a provider-generic
  Leto/Hephaestus trait implementation.
- **Realtime imaging pipeline Leto frame buffers - RESOLVED [patch].**
  `kwavers-gpu::gpu::pipeline::RealtimeImagingPipeline` and
  `StreamingDataSource` now exchange RF input frames and processed output
  frames as `leto::Array4<f32>`/`leto::Array3<f32>` instead of ndarray arrays.
  The local tx-sum beamforming path is explicit Leto traversal, and pipeline
  tests construct Leto frames directly. Follow-up closed the private Hilbert
  transform scratch by replacing ndarray `Array1<Complex64>` with a
  thread-local `Vec<Complex64>` passed through Apollo's slice FFT API, leaving
  no ndarray imports in `kwavers-gpu::gpu::pipeline`. Evidence tier:
  type-level API validation plus focused empirical tests; `rustup run nightly cargo fmt -p kwavers-gpu
  --check`, `rustup run nightly cargo check -p kwavers-gpu --features gpu`,
  `rustup run nightly cargo clippy -p kwavers-gpu --features gpu --lib -- -D
  warnings`, `rustup run nightly cargo nextest run -p kwavers-gpu --features
  gpu gpu::pipeline --status-level fail --no-fail-fast` (5/5), and `rustup
  run nightly cargo check -p kwavers-gpu --features cuda-provider` pass.
  Follow-up 2026-07-04: scoped `rg` finds no ndarray token under
  `crates/kwavers-gpu/src/gpu/pipeline`, `rustup run nightly cargo fmt -p
  kwavers-gpu --check` passes, `rustup run nightly cargo check -p kwavers-gpu
  --features cuda-provider` passes, `rustup run nightly cargo clippy -p
  kwavers-gpu --features cuda-provider --all-targets -- -D warnings` passes,
  and focused `rustup run nightly cargo nextest run -p kwavers-gpu --features
  cuda-provider pipeline --status-level fail --no-fail-fast` passes 8/8.
- **Thermal-acoustic solver provider seam - RESOLVED [patch].**
  `kwavers-gpu::gpu::thermal_acoustic::GpuThermalAcousticSolver<P>` now
  delegates to `ThermalAcousticSolverProvider`, which now extends the shared
  `GpuKernelProvider`/`GpuProviderBackend` stack. WGPU context, compute
  pipelines, bind group, and step dispatch live in
  `WgpuThermalAcousticSolverProvider`, so the generic solver wrapper no longer
  exposes WGPU pipeline fields, WGPU step parameters, or raw WGPU device/queue
  constructor arguments. Evidence tier:
  type-level/compile-time validation plus focused GPU test; `cargo check -p
  kwavers-gpu --features gpu` passes, `cargo clippy -p kwavers-gpu --features
  gpu --lib -- -D warnings` passes, and `cargo nextest run -p kwavers-gpu
  --features gpu thermal_acoustic_solver_is_generic_over_provider_trait
  --status-level fail --no-fail-fast` passes 1/1. Follow-up provider-stack
  evidence: `rustup run nightly cargo fmt -p kwavers-gpu --check`, `rustup run
  nightly cargo check -p kwavers-gpu --features gpu --all-targets`, `rustup
  run nightly cargo check -p kwavers-gpu --features cuda-provider
  --all-targets`, focused thermal-acoustic/provider nextest under `gpu`
  (38/38) and `cuda-provider` (45/45), and clippy for both feature sets pass.
- **Backend buffer-manager provider seam - RESOLVED [patch].**
  `kwavers-gpu::backend::GpuBackendBufferManager<P>` now delegates to
  `BackendBufferProvider`. WGPU buffer pooling, allocation, array upload, and
  readback live in `WgpuBackendBufferManager`, so the generic backend
  buffer-manager wrapper no longer exposes `wgpu::Buffer` methods. Evidence
  tier: type-level/compile-time validation plus focused GPU test; `cargo check
  -p kwavers-gpu --features gpu` passes, `cargo clippy -p kwavers-gpu
  --features gpu --lib -- -D warnings` passes, and `cargo nextest run -p
  kwavers-gpu --features gpu
  backend_buffer_manager_wrapper_is_generic_over_provider_trait --status-level
  fail --no-fail-fast` passes 1/1.
- **PSTD buffer allocation provider seam - RESOLVED [patch].**
  `GpuPstdSolver::new` and the run-cache rebuild path now delegate owned
  storage/staging buffer allocation through `PstdBufferProvider`. The WGPU
  read-only, static, upload, read/write, and staging-buffer creation paths live
  in `WgpuPstdBufferFactory`, so PSTD construction and cache rebuild no longer
  call WGPU buffer allocation directly for those buffers. Evidence tier:
  type-level/compile-time validation plus focused GPU tests; `cargo check -p
  kwavers-gpu --features gpu` passes, `cargo clippy -p kwavers-gpu --features
  gpu --lib -- -D warnings` passes, and `cargo nextest run -p kwavers-gpu
  --features gpu pstd_buffer_factory_is_generic_over_provider_trait
  packed_signal_len_keeps_storage_buffers_non_empty
  rewrite_packed_source_buffer_preserves_indices_and_signal_tail
  rewrite_packed_source_buffer_uses_zero_signal_sentinel_for_empty_tail
  overwrite_packed_signal_tail_keeps_index_prefix_stable --status-level fail
  --no-fail-fast` passes 5/5.
- **PSTD pipeline provider seam - RESOLVED [patch].**
  `GpuPstdSolver::new` now delegates shader-module, pipeline-layout, and
  standard/absorption compute-pipeline creation through
  `PstdPipelineProvider`. The WGPU shader, layout, and
  `ComputePipelineDescriptor` construction paths live in
  `WgpuPstdPipelineFactory`, so PSTD construction no longer calls those WGPU
  creation APIs directly for pipeline setup. Evidence tier:
  type-level/compile-time validation plus focused GPU tests; `cargo check -p
  kwavers-gpu --features gpu` passes, `cargo clippy -p kwavers-gpu --features
  gpu --lib -- -D warnings` passes, and `cargo nextest run -p kwavers-gpu
  --features gpu pstd_buffer_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait
  packed_signal_len_keeps_storage_buffers_non_empty
  rewrite_packed_source_buffer_preserves_indices_and_signal_tail
  rewrite_packed_source_buffer_uses_zero_signal_sentinel_for_empty_tail
  overwrite_packed_signal_tail_keeps_index_prefix_stable --status-level fail
  --no-fail-fast` passes 6/6.
- **PSTD bind-group layout provider seam - RESOLVED [patch].**
  `GpuPstdSolver::new` now delegates field, k-space, sensor/source, and
  absorption bind-group layout creation through `PstdBindGroupLayoutProvider`.
  The WGPU binding-slot descriptor construction path lives in
  `WgpuPstdBindGroupLayoutFactory`, so PSTD construction no longer calls WGPU
  bind-group-layout builders directly. Evidence tier: type-level/compile-time
  validation plus focused GPU tests; `cargo check -p kwavers-gpu --features
  gpu` passes, `cargo clippy -p kwavers-gpu --features gpu --lib -- -D
  warnings` passes, and `cargo nextest run -p kwavers-gpu --features gpu
  pstd_bind_group_layout_factory_is_generic_over_provider_trait
  pstd_buffer_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait
  packed_signal_len_keeps_storage_buffers_non_empty
  rewrite_packed_source_buffer_preserves_indices_and_signal_tail
  rewrite_packed_source_buffer_uses_zero_signal_sentinel_for_empty_tail
  overwrite_packed_signal_tail_keeps_index_prefix_stable --status-level fail
  --no-fail-fast` passes 7/7.
- **PSTD bind-group provider seam - RESOLVED [patch].**
  Permanent constructor bind groups and run-cache sensor bind groups now
  delegate assembly through `PstdBindGroupProvider`. The WGPU
  `BindGroupDescriptor` construction path lives in `WgpuPstdBindGroupFactory`,
  so PSTD construction and cache rebuild no longer call WGPU bind-group
  creation APIs directly. Evidence tier: type-level/compile-time validation
  plus focused GPU tests; `cargo check -p kwavers-gpu --features gpu` passes,
  `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings` passes,
  and `cargo nextest run -p kwavers-gpu --features gpu
  pstd_bind_group_factory_is_generic_over_provider_trait
  pstd_bind_group_layout_factory_is_generic_over_provider_trait
  pstd_buffer_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait
  packed_signal_len_keeps_storage_buffers_non_empty
  rewrite_packed_source_buffer_preserves_indices_and_signal_tail
  rewrite_packed_source_buffer_uses_zero_signal_sentinel_for_empty_tail
  overwrite_packed_signal_tail_keeps_index_prefix_stable --status-level fail
  --no-fail-fast` passes 8/8.
- **PSTD command provider seam - RESOLVED [patch].**
  Run-loop sensor clear, sensor copy, command submit, and wait-poll operations
  now delegate through `PstdCommandProvider`. The WGPU command encoder and
  queue wait mechanics for those paths live in `WgpuPstdCommandProvider`, so
  the run-loop no longer calls those WGPU command/queue APIs directly for
  clear/copy/wait paths. Evidence tier: type-level/compile-time validation
  plus focused GPU tests; `cargo check -p kwavers-gpu --features gpu` passes,
  `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings` passes,
  and `cargo nextest run -p kwavers-gpu --features gpu
  pstd_command_provider_is_generic_over_provider_trait
  pstd_bind_group_factory_is_generic_over_provider_trait
  pstd_bind_group_layout_factory_is_generic_over_provider_trait
  pstd_buffer_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait
  packed_signal_len_keeps_storage_buffers_non_empty
  rewrite_packed_source_buffer_preserves_indices_and_signal_tail
  rewrite_packed_source_buffer_uses_zero_signal_sentinel_for_empty_tail
  overwrite_packed_signal_tail_keeps_index_prefix_stable --status-level fail
  --no-fail-fast` passes 9/9.
- **PSTD command encoder provider seam - RESOLVED [patch].**
  Zero-field and batched-step command encoder creation/submission now delegate
  through `PstdCommandProvider::submit_encoder`. The provider trait owns an
  associated encoder type rather than naming `wgpu::CommandEncoder`, while the
  WGPU implementation keeps command encoder creation and queue submission in
  `WgpuPstdCommandProvider`. Evidence tier: type-level/compile-time validation
  plus focused GPU tests; `cargo check -p kwavers-gpu --features gpu` passes,
  `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings` passes,
  and `cargo nextest run -p kwavers-gpu --features gpu
  pstd_command_provider_is_generic_over_provider_trait
  pstd_bind_group_factory_is_generic_over_provider_trait
  pstd_bind_group_layout_factory_is_generic_over_provider_trait
  pstd_buffer_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait
  packed_signal_len_keeps_storage_buffers_non_empty
  rewrite_packed_source_buffer_preserves_indices_and_signal_tail
  rewrite_packed_source_buffer_uses_zero_signal_sentinel_for_empty_tail
  overwrite_packed_signal_tail_keeps_index_prefix_stable` passes 9/9. Source
  audit confirms direct `create_command_encoder` and `queue.submit` calls are
  confined to `time_loop/commands.rs` for the touched PSTD run-loop files.
- **PSTD compute-pass provider seam - RESOLVED [patch].**
  Zero-field and batched-step compute-pass creation now delegate through
  `PstdCommandProvider::submit_compute_pass` and
  `PstdCommandProvider::submit_compute_passes`. The provider trait owns a
  lifetime-associated compute-pass type rather than naming `wgpu::ComputePass`,
  while the WGPU implementation keeps pass descriptors and begin-pass calls in
  `WgpuPstdCommandProvider`. Evidence tier: type-level/compile-time validation
  plus focused GPU tests; `cargo check -p kwavers-gpu --features gpu` passes,
  `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings` passes,
  and `cargo nextest run -p kwavers-gpu --features gpu
  pstd_command_provider_is_generic_over_provider_trait
  pstd_bind_group_factory_is_generic_over_provider_trait
  pstd_bind_group_layout_factory_is_generic_over_provider_trait
  pstd_buffer_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait
  packed_signal_len_keeps_storage_buffers_non_empty
  rewrite_packed_source_buffer_preserves_indices_and_signal_tail
  rewrite_packed_source_buffer_uses_zero_signal_sentinel_for_empty_tail
  overwrite_packed_signal_tail_keeps_index_prefix_stable` passes 9/9. Source
  audit confirms direct `begin_compute_pass` calls are confined to
  `time_loop/commands.rs` for the touched PSTD run-loop files.
- **PSTD pass-body provider seam - RESOLVED [patch].**
  Zero-field and per-step pass-body encoding now delegates through
  `PstdPassProvider`. The WGPU implementation owns the existing dispatch and
  `encode_*` call sequence, so `time_loop::run` no longer calls WGPU pass-body
  methods directly or names `wgpu::ComputePass`. Evidence tier:
  type-level/compile-time validation plus focused GPU tests; `cargo check -p
  kwavers-gpu --features gpu` passes, `cargo clippy -p kwavers-gpu --features
  gpu --lib -- -D warnings` passes, and `cargo nextest run -p kwavers-gpu
  --features gpu pstd_pass_provider_is_generic_over_provider_trait
  pstd_command_provider_is_generic_over_provider_trait
  pstd_bind_group_factory_is_generic_over_provider_trait
  pstd_bind_group_layout_factory_is_generic_over_provider_trait
  pstd_buffer_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait
  packed_signal_len_keeps_storage_buffers_non_empty
  rewrite_packed_source_buffer_preserves_indices_and_signal_tail
  rewrite_packed_source_buffer_uses_zero_signal_sentinel_for_empty_tail
  overwrite_packed_signal_tail_keeps_index_prefix_stable` passes 10/10. Source
  audit confirms `time_loop::run` no longer calls `self.dispatch`,
  `self.encode_*`, `begin_compute_pass`, or `wgpu::ComputePass`.
- **PSTD readback provider seam - RESOLVED [patch].**
  Sensor staging-buffer readback now delegates through
  `PstdCommandProvider::read_mapped`. The WGPU implementation owns staging
  buffer slicing, `map_async`, `MapMode::Read`, mapped-range extraction, and
  unmap; `time_loop::run` only requests the typed host vector. Evidence tier:
  type-level/compile-time validation plus focused GPU tests; `cargo check -p
  kwavers-gpu --features gpu` passes, `cargo clippy -p kwavers-gpu --features
  gpu --lib -- -D warnings` passes, and `cargo nextest run -p kwavers-gpu
  --features gpu pstd_pass_provider_is_generic_over_provider_trait
  pstd_command_provider_is_generic_over_provider_trait
  pstd_bind_group_factory_is_generic_over_provider_trait
  pstd_bind_group_layout_factory_is_generic_over_provider_trait
  pstd_buffer_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait
  packed_signal_len_keeps_storage_buffers_non_empty
  rewrite_packed_source_buffer_preserves_indices_and_signal_tail
  rewrite_packed_source_buffer_uses_zero_signal_sentinel_for_empty_tail
  overwrite_packed_signal_tail_keeps_index_prefix_stable` passes 10/10. Source
  audit confirms direct readback map/unmap calls are confined to
  `time_loop/commands.rs` for the touched PSTD run-loop files.
- **PSTD cache-hit upload provider seam - RESOLVED [patch].**
  Cache-hit source and velocity signal-tail uploads now delegate through
  `PstdCommandProvider::write_buffer`. The provider trait accepts POD host
  slices generically, and the WGPU implementation owns `queue.write_buffer`,
  byte casting, and byte-offset submission. Evidence tier:
  type-level/compile-time validation plus focused GPU tests; `cargo check -p
  kwavers-gpu --features gpu` passes, `cargo clippy -p kwavers-gpu --features
  gpu --lib -- -D warnings` passes, and `cargo nextest run -p kwavers-gpu
  --features gpu pstd_pass_provider_is_generic_over_provider_trait
  pstd_command_provider_is_generic_over_provider_trait
  pstd_bind_group_factory_is_generic_over_provider_trait
  pstd_bind_group_layout_factory_is_generic_over_provider_trait
  pstd_buffer_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait
  packed_signal_len_keeps_storage_buffers_non_empty
  rewrite_packed_source_buffer_preserves_indices_and_signal_tail
  rewrite_packed_source_buffer_uses_zero_signal_sentinel_for_empty_tail
  overwrite_packed_signal_tail_keeps_index_prefix_stable` passes 10/10. Source
  audit confirms cache-hit upload calls in `time_loop/buffer.rs` now call
  `commands.write_buffer`, with direct WGPU `queue.write_buffer` confined to
  `time_loop/commands.rs` for the touched run-cache files.
- **PSTD medium-update upload provider seam - RESOLVED [patch].**
  Variable/full medium refreshes and source-correction writes now delegate
  through `PstdCommandProvider::write_buffer`. The provider is visible only
  inside `pstd_gpu`, while the WGPU implementation owns byte casting and
  `queue.write_buffer` submission. Evidence tier: type-level/compile-time
  validation plus focused GPU tests; `cargo check -p kwavers-gpu --features
  gpu` passes, `cargo clippy -p kwavers-gpu --features gpu --lib -- -D
  warnings` passes, and `cargo nextest run -p kwavers-gpu --features gpu
  medium_variable_update pstd_command_provider_is_generic_over_provider_trait
  pstd_pass_provider_is_generic_over_provider_trait
  pstd_bind_group_factory_is_generic_over_provider_trait
  pstd_bind_group_layout_factory_is_generic_over_provider_trait
  pstd_buffer_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait` passes 7/7. Source
  audit confirms `queue.write_buffer` under `crates/kwavers-gpu/src/pstd_gpu`
  is confined to `time_loop/commands.rs`.
- **PSTD medium buffer state grouping - RESOLVED [patch].**
  `GpuPstdSolver` now owns k-space, medium, twiddle, and source-kappa buffers
  through `WgpuPstdMediumBuffers` instead of separate top-level WGPU buffer
  fields. Construction preserves the existing bind-group slot order, and
  medium-update/source-correction writes access the grouped state through the
  command provider. Evidence tier: type-level/compile-time validation plus
  focused GPU tests; `cargo check -p kwavers-gpu --features gpu` passes,
  `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings` passes,
  and `cargo nextest run -p kwavers-gpu --features gpu medium_variable_update
  pstd_command_provider_is_generic_over_provider_trait
  pstd_pass_provider_is_generic_over_provider_trait
  pstd_bind_group_factory_is_generic_over_provider_trait
  pstd_bind_group_layout_factory_is_generic_over_provider_trait
  pstd_buffer_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait` passes 7/7. Source
  audit confirms `GpuPstdSolver` no longer exposes separate top-level
  `buf_kappa`, `buf_rho0_inv`, `buf_c0_sq`, `buf_rho0`, `buf_bon_a`,
  `buf_alpha_decay`, or `buf_source_kappa` fields.
- **PSTD acoustic field buffer state grouping - RESOLVED [patch].**
  `GpuPstdSolver` now owns pressure, velocity, and density buffers through
  `WgpuPstdFieldBuffers` instead of separate top-level WGPU buffer fields.
  Construction preserves the existing group(0) bind-group slot order. Evidence
  tier: type-level/compile-time validation plus focused GPU tests; `cargo
  check -p kwavers-gpu --features gpu` passes, `cargo clippy -p kwavers-gpu
  --features gpu --lib -- -D warnings` passes, and `cargo nextest run -p
  kwavers-gpu --features gpu medium_variable_update
  pstd_command_provider_is_generic_over_provider_trait
  pstd_pass_provider_is_generic_over_provider_trait
  pstd_bind_group_factory_is_generic_over_provider_trait
  pstd_bind_group_layout_factory_is_generic_over_provider_trait
  pstd_buffer_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait` passes 7/7. Source
  audit confirms `GpuPstdSolver` no longer exposes separate top-level `buf_p`,
  `buf_ux`, `buf_uy`, `buf_uz`, `buf_rhox`, `buf_rhoy`, or `buf_rhoz` fields.
- **PSTD absorption buffer state grouping - RESOLVED [patch].**
  `GpuPstdSolver` now owns fractional-Laplacian operator and scratch buffers
  through `WgpuPstdAbsorptionBuffers` instead of separate top-level WGPU
  buffer fields. Construction preserves the existing group(3) bind-group slot
  order, and medium updates write absorption tau/eta through grouped state.
  Evidence tier: type-level/compile-time validation plus focused GPU tests;
  `cargo check -p kwavers-gpu --features gpu` passes, `cargo clippy -p
  kwavers-gpu --features gpu --lib -- -D warnings` passes, and `cargo nextest
  run -p kwavers-gpu --features gpu medium_variable_update
  pstd_command_provider_is_generic_over_provider_trait
  pstd_pass_provider_is_generic_over_provider_trait
  pstd_bind_group_factory_is_generic_over_provider_trait
  pstd_bind_group_layout_factory_is_generic_over_provider_trait
  pstd_buffer_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait` passes 7/7. Source
  audit confirms `GpuPstdSolver` no longer exposes separate top-level
  `buf_absorb_*` fields.
- **PSTD PML/shift buffer state grouping - RESOLVED [patch].**
  `GpuPstdSolver` now owns split PML coefficients, packed PML axis data, and
  packed k-space shift operators through `WgpuPstdPmlShiftBuffers` instead of
  separate top-level WGPU buffer fields. Run-cache sensor bind groups preserve
  the existing sensor layout while reading through grouped state. Evidence
  tier: type-level/compile-time validation plus focused GPU tests; `cargo
  check -p kwavers-gpu --features gpu` passes, `cargo clippy -p kwavers-gpu
  --features gpu --lib -- -D warnings` passes, and `cargo nextest run -p
  kwavers-gpu --features gpu medium_variable_update
  pstd_command_provider_is_generic_over_provider_trait
  pstd_pass_provider_is_generic_over_provider_trait
  pstd_bind_group_factory_is_generic_over_provider_trait
  pstd_bind_group_layout_factory_is_generic_over_provider_trait
  pstd_buffer_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait` passes 7/7. Source
  audit confirms `GpuPstdSolver` no longer exposes separate top-level
  `buf_pml_*` or `buf_shifts_all` fields.
- **PSTD run-cache state grouping - RESOLVED [patch].**
  `GpuPstdSolver` now owns cached sensor/source/velocity buffers, staging
  buffer, sensor bind groups, and cache-key counters through
  `WgpuPstdRunCache` instead of separate top-level WGPU cache fields.
  Cache-hit signal-tail refreshes and sensor readback preserve the existing
  data flow while reading through grouped state. Evidence tier:
  type-level/compile-time validation plus focused GPU tests; `cargo check -p
  kwavers-gpu --features gpu` passes, `cargo clippy -p kwavers-gpu --features
  gpu --lib -- -D warnings` passes, and `cargo nextest run -p kwavers-gpu
  --features gpu medium_variable_update
  pstd_command_provider_is_generic_over_provider_trait
  pstd_pass_provider_is_generic_over_provider_trait
  pstd_bind_group_factory_is_generic_over_provider_trait
  pstd_bind_group_layout_factory_is_generic_over_provider_trait
  pstd_buffer_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait` passes 7/7. Source
  audit confirms `GpuPstdSolver` no longer exposes separate top-level
  `cache_*` fields under `pstd_gpu`.
- **PSTD permanent bind-group state grouping - RESOLVED [patch].**
  `GpuPstdSolver` now owns field, k-space, and absorption bind groups through
  `WgpuPstdPermanentBindGroups` instead of separate top-level WGPU bind-group
  fields. Dispatch helpers preserve the existing group slots while reading
  through grouped state. Evidence tier: type-level/compile-time validation plus
  focused GPU tests; `cargo check -p kwavers-gpu --features gpu` passes,
  `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings` passes,
  and `cargo nextest run -p kwavers-gpu --features gpu medium_variable_update
  pstd_command_provider_is_generic_over_provider_trait
  pstd_pass_provider_is_generic_over_provider_trait
  pstd_bind_group_factory_is_generic_over_provider_trait
  pstd_bind_group_layout_factory_is_generic_over_provider_trait
  pstd_buffer_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait` passes 7/7. Source
  audit confirms `GpuPstdSolver` no longer exposes separate top-level
  `bg_fields`, `bg_kspace`, or `bg_absorb` fields.
- **PSTD layout state grouping - RESOLVED [patch].**
  `GpuPstdSolver` now owns retained WGPU layout handles through
  `WgpuPstdLayouts` instead of separate top-level layout fields. The grouped
  state retains only the sensor bind-group layout required for run-cache sensor
  bind-group rebuilds; the unused retained base pipeline layout field was
  deleted. Evidence tier: type-level/compile-time validation plus focused GPU
  tests; `cargo check -p kwavers-gpu --features gpu` passes, `cargo clippy -p
  kwavers-gpu --features gpu --lib -- -D warnings` passes, and `cargo nextest
  run -p kwavers-gpu --features gpu medium_variable_update
  pstd_command_provider_is_generic_over_provider_trait
  pstd_pass_provider_is_generic_over_provider_trait
  pstd_bind_group_factory_is_generic_over_provider_trait
  pstd_bind_group_layout_factory_is_generic_over_provider_trait
  pstd_buffer_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait` passes 7/7. Source
  audit confirms `GpuPstdSolver` no longer exposes separate top-level
  `bgl_sensor` or `pipeline_layout` fields.
- **PSTD k-space work buffer state grouping - RESOLVED [patch].**
  `GpuPstdSolver` now owns `kspace_re` and `kspace_im` through
  `WgpuPstdKspaceBuffers` instead of separate top-level WGPU buffer fields.
  Construction preserves the existing group(1) bind-group slots. Evidence
  tier: type-level/compile-time validation plus focused GPU tests; `cargo
  check -p kwavers-gpu --features gpu` passes, `cargo clippy -p kwavers-gpu
  --features gpu --lib -- -D warnings` passes, and `cargo nextest run -p
  kwavers-gpu --features gpu medium_variable_update
  pstd_command_provider_is_generic_over_provider_trait
  pstd_pass_provider_is_generic_over_provider_trait
  pstd_bind_group_factory_is_generic_over_provider_trait
  pstd_bind_group_layout_factory_is_generic_over_provider_trait
  pstd_buffer_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait` passes 7/7. Source
  audit confirms `GpuPstdSolver` no longer exposes separate top-level
  `buf_kspace_re` or `buf_kspace_im` fields.
- **PSTD compute-pipeline state grouping - RESOLVED [patch].**
  `GpuPstdSolver` now owns all WGPU compute pipelines through
  `WgpuPstdPipelines` instead of separate top-level WGPU pipeline fields.
  Time-loop dispatch and encode paths preserve the existing shader entry-point
  mapping through grouped state. Evidence tier: type-level/compile-time
  validation plus focused GPU tests; `cargo check -p kwavers-gpu --features
  gpu` passes, `cargo clippy -p kwavers-gpu --features gpu --lib -- -D
  warnings` passes, and `cargo nextest run -p kwavers-gpu --features gpu
  medium_variable_update pstd_command_provider_is_generic_over_provider_trait
  pstd_pass_provider_is_generic_over_provider_trait
  pstd_bind_group_factory_is_generic_over_provider_trait
  pstd_bind_group_layout_factory_is_generic_over_provider_trait
  pstd_buffer_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait` passes 7/7. Source
  audit confirms `GpuPstdSolver` no longer exposes separate top-level
  `pipeline_*` fields.
- **PSTD WGPU state aggregate - RESOLVED [patch].**
  `GpuPstdSolver` now owns grouped WGPU buffers, pipelines, bind groups,
  layouts, and run-cache state through one `WgpuPstdState` field instead of
  separate grouped WGPU fields. Evidence tier: type-level/compile-time
  validation plus focused GPU tests; `cargo check -p kwavers-gpu --features
  gpu` passes, `cargo clippy -p kwavers-gpu --features gpu --lib -- -D
  warnings` passes, and `cargo nextest run -p kwavers-gpu --features gpu
  medium_variable_update pstd_command_provider_is_generic_over_provider_trait
  pstd_pass_provider_is_generic_over_provider_trait
  pstd_bind_group_factory_is_generic_over_provider_trait
  pstd_bind_group_layout_factory_is_generic_over_provider_trait
  pstd_buffer_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait` passes 7/7. Source
  audit confirms `GpuPstdSolver` exposes `state: WgpuPstdState` and no direct
  grouped-state solver fields.
- **PSTD provider-associated state - RESOLVED [patch].** `GpuPstdSolver<P>`
  now owns `P::State` through `PstdStateProvider`, with
  `WgpuPstdStateProvider` as the default real provider. Existing PSTD
  construction, medium-update, run-cache, dispatch, pass-body, and encode
  methods are specialized on `GpuPstdSolver<WgpuPstdStateProvider>`, leaving
  CUDA absent until real PSTD state and kernels exist. Evidence tier:
  type-level/compile-time validation plus focused empirical tests; `cargo
  check -p kwavers-gpu --features gpu` passes, `cargo clippy -p kwavers-gpu
  --features gpu --lib -- -D warnings` passes, and `cargo nextest run -p
  kwavers-gpu --features gpu medium_variable_update
  pstd_solver_state_is_provider_associated
  pstd_command_provider_is_generic_over_provider_trait
  pstd_pass_provider_is_generic_over_provider_trait
  pstd_bind_group_factory_is_generic_over_provider_trait
  pstd_bind_group_layout_factory_is_generic_over_provider_trait
  pstd_buffer_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait` passes 8/8.
- **PSTD provider-owned WGPU handles - RESOLVED [patch].** `GpuPstdSolver<P>`
  no longer owns raw `Arc<wgpu::Device>` or `Arc<wgpu::Queue>` fields. Those
  handles now live in `WgpuPstdState` with the WGPU buffers, pipelines, bind
  groups, layouts, and run cache, and WGPU-specialized methods borrow them
  through `self.state`. Evidence tier: type-level/compile-time validation plus
  focused empirical tests; `cargo check -p kwavers-gpu --features gpu` passes,
  `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings` passes,
  and `cargo nextest run -p kwavers-gpu --features gpu medium_variable_update
  pstd_solver_state_is_provider_associated
  pstd_command_provider_is_generic_over_provider_trait
  pstd_pass_provider_is_generic_over_provider_trait
  pstd_bind_group_factory_is_generic_over_provider_trait
  pstd_bind_group_layout_factory_is_generic_over_provider_trait
  pstd_buffer_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait` passes 8/8.
- **PSTD provider-owned WGPU scratch buffers - RESOLVED [patch].**
  `GpuPstdSolver<P>` no longer owns WGPU host scratch/upload buffers.
  `WgpuPstdState` owns `scratch_c0_sq`, `scratch_rho0_inv`,
  `scratch_rho0_flat`, `scratch_source_kappa_ones`, `scratch_source_data`, and
  `scratch_vel_x_data`; WGPU-specialized medium-update and run-cache staging
  paths borrow them through `self.state`. Evidence tier:
  type-level/compile-time validation plus focused empirical tests; `cargo
  check -p kwavers-gpu --features gpu` passes, `cargo clippy -p kwavers-gpu
  --features gpu --lib -- -D warnings` passes, and `cargo nextest run -p
  kwavers-gpu --features gpu medium_variable_update
  pstd_solver_state_is_provider_associated
  pstd_command_provider_is_generic_over_provider_trait
  pstd_pass_provider_is_generic_over_provider_trait
  pstd_bind_group_factory_is_generic_over_provider_trait
  pstd_bind_group_layout_factory_is_generic_over_provider_trait
  pstd_buffer_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait` passes 8/8.
- **PSTD provider-owned WGPU state construction - RESOLVED [patch].**
  `WgpuPstdStateProvider::build_state` now owns WGPU PSTD state assembly:
  buffers, pipelines, bind groups, layouts, run cache, device/queue handles,
  and host scratch/upload buffers. `GpuPstdSolver::new` wraps the returned
  provider state with dimensions, time step, and physics flags. Evidence tier:
  type-level/compile-time validation plus focused empirical tests; `cargo
  check -p kwavers-gpu --features gpu` passes, `cargo clippy -p kwavers-gpu
  --features gpu --lib -- -D warnings` passes, and `cargo nextest run -p
  kwavers-gpu --features gpu medium_variable_update
  pstd_solver_state_is_provider_associated
  pstd_command_provider_is_generic_over_provider_trait
  pstd_pass_provider_is_generic_over_provider_trait
  pstd_bind_group_factory_is_generic_over_provider_trait
  pstd_bind_group_layout_factory_is_generic_over_provider_trait
  pstd_buffer_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait` passes 8/8.
- **PSTD provider-owned WGPU medium uploads - RESOLVED [patch].**
  `WgpuPstdState` now owns variable-medium upload, full medium refresh, and
  source-correction disablement bodies. Public
  `GpuPstdSolver<WgpuPstdStateProvider>` methods forward to provider-state
  methods instead of creating WGPU command providers and issuing `write_buffer`
  calls directly. Evidence tier: type-level/compile-time validation plus
  focused empirical tests; `cargo check -p kwavers-gpu --features gpu` passes,
  `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings` passes,
  and `cargo nextest run -p kwavers-gpu --features gpu medium_variable_update
  pstd_solver_state_is_provider_associated
  pstd_command_provider_is_generic_over_provider_trait
  pstd_pass_provider_is_generic_over_provider_trait
  pstd_bind_group_factory_is_generic_over_provider_trait
  pstd_bind_group_layout_factory_is_generic_over_provider_trait
  pstd_buffer_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait` passes 8/8.
- **PSTD provider-owned WGPU run cache - RESOLVED [patch].** `WgpuPstdState`
  now owns run-scoped sensor/source/velocity buffer allocation, sensor
  bind-group rebuild, cache-key updates, and cache-hit signal-tail uploads.
  `GpuPstdSolver<WgpuPstdStateProvider>` methods forward to provider state and
  supply the solver time-step count. Evidence tier:
  type-level/compile-time validation plus focused empirical tests; `cargo
  check -p kwavers-gpu --features gpu` passes, `cargo clippy -p kwavers-gpu
  --features gpu --lib -- -D warnings` passes, and `cargo nextest run -p
  kwavers-gpu --features gpu medium_variable_update
  pstd_solver_state_is_provider_associated
  pstd_command_provider_is_generic_over_provider_trait
  pstd_pass_provider_is_generic_over_provider_trait
  pstd_bind_group_factory_is_generic_over_provider_trait
  pstd_bind_group_layout_factory_is_generic_over_provider_trait
  pstd_buffer_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait
  packed_signal_len_keeps_storage_buffers_non_empty
  rewrite_packed_source_buffer_preserves_indices_and_signal_tail
  rewrite_packed_source_buffer_uses_zero_signal_sentinel_for_empty_tail
  overwrite_packed_signal_tail_keeps_index_prefix_stable` passes 12/12.
- **PSTD provider-owned WGPU pass encoding - RESOLVED [patch].**
  `WgpuPstdState` now owns WGPU dispatch, absorption dispatch, FFT/IFFT, and
  per-phase pass encoding methods. `WgpuPstdPassProvider` stores provider
  state instead of a `GpuPstdSolver<WgpuPstdStateProvider>` reference. Evidence
  tier: type-level/compile-time validation plus focused empirical tests;
  `cargo check -p kwavers-gpu --features gpu` passes, `cargo clippy -p
  kwavers-gpu --features gpu --lib -- -D warnings` passes, and `cargo nextest
  run -p kwavers-gpu --features gpu
  pstd_pass_provider_is_generic_over_provider_trait
  pstd_command_provider_is_generic_over_provider_trait
  pstd_solver_state_is_provider_associated
  pstd_buffer_factory_is_generic_over_provider_trait
  pstd_bind_group_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait
  pstd_bind_group_layout_factory_is_generic_over_provider_trait
  packed_signal_len_keeps_storage_buffers_non_empty
  rewrite_packed_source_buffer_preserves_indices_and_signal_tail
  rewrite_packed_source_buffer_uses_zero_signal_sentinel_for_empty_tail
  overwrite_packed_signal_tail_keeps_index_prefix_stable
  medium_variable_update` passes 12/12.
- **PSTD provider-owned WGPU run orchestration - RESOLVED [patch].**
  `WgpuPstdState` now owns high-level WGPU run-loop orchestration: cache
  validation, cache rebuild/refresh selection, sensor clear, zero-field pass
  submission, batched time-step submission, throttled provider wait, sensor
  copy, and mapped readback. `GpuPstdSolver<WgpuPstdStateProvider>::run`
  delegates with scalar metadata and input slices. Evidence tier:
  type-level/compile-time validation plus focused empirical tests; `cargo
  check -p kwavers-gpu --features gpu` passes, `cargo clippy -p kwavers-gpu
  --features gpu --lib -- -D warnings` passes, and `cargo nextest run -p
  kwavers-gpu --features gpu
  pstd_pass_provider_is_generic_over_provider_trait
  pstd_command_provider_is_generic_over_provider_trait
  pstd_solver_state_is_provider_associated
  pstd_buffer_factory_is_generic_over_provider_trait
  pstd_bind_group_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait
  pstd_bind_group_layout_factory_is_generic_over_provider_trait
  packed_signal_len_keeps_storage_buffers_non_empty
  rewrite_packed_source_buffer_preserves_indices_and_signal_tail
  rewrite_packed_source_buffer_uses_zero_signal_sentinel_for_empty_tail
  overwrite_packed_signal_tail_keeps_index_prefix_stable
  medium_variable_update` passes 12/12.
- **PSTD provider-generic state construction - RESOLVED [patch].**
  `PstdStateBuilder` now defines an associated provider context type and the
  `build_state` contract. `GpuPstdSolver<P>::new` delegates state construction
  through `P::build_state`, with WGPU as the only real implementation. Direct
  WGPU test/helper constructor call sites pass `GpuProviderContext<WgpuDevice>`
  through `WgpuPstdStateProvider` explicitly. Evidence tier:
  type-level/compile-time validation plus focused empirical tests; `cargo
  check -p kwavers-gpu --features gpu` passes, `cargo clippy -p kwavers-gpu
  --features gpu --lib -- -D warnings` passes, and `cargo nextest run -p
  kwavers-gpu --features gpu pstd_solver_state_builder_uses_provider_handles
  pstd_solver_state_is_provider_associated
  pstd_pass_provider_is_generic_over_provider_trait
  pstd_command_provider_is_generic_over_provider_trait
  pstd_buffer_factory_is_generic_over_provider_trait
  pstd_bind_group_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait
  pstd_bind_group_layout_factory_is_generic_over_provider_trait
  packed_signal_len_keeps_storage_buffers_non_empty
  rewrite_packed_source_buffer_preserves_indices_and_signal_tail
  rewrite_packed_source_buffer_uses_zero_signal_sentinel_for_empty_tail
  overwrite_packed_signal_tail_keeps_index_prefix_stable
  medium_variable_update` passes 13/13.
- **PSTD provider-generic run execution - RESOLVED [patch].**
  `PstdRunState` now defines the provider-state run execution contract using
  provider-neutral `PstdRunScalars` and `PstdRunInputs`. `GpuPstdSolver<P>::run`
  is generic for providers whose state implements `PstdRunState`; WGPU remains
  the only real implementation. Evidence tier: type-level/compile-time
  validation plus focused empirical tests; `cargo check -p kwavers-gpu
  --features gpu` passes, `cargo clippy -p kwavers-gpu --features gpu --lib --
  -D warnings` passes, and `cargo nextest run -p kwavers-gpu --features gpu
  pstd_solver_run_state_is_provider_owned
  pstd_solver_state_builder_uses_provider_handles
  pstd_solver_state_is_provider_associated
  pstd_pass_provider_is_generic_over_provider_trait
  pstd_command_provider_is_generic_over_provider_trait
  pstd_buffer_factory_is_generic_over_provider_trait
  pstd_bind_group_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait
  pstd_bind_group_layout_factory_is_generic_over_provider_trait
  packed_signal_len_keeps_storage_buffers_non_empty
  rewrite_packed_source_buffer_preserves_indices_and_signal_tail
  rewrite_packed_source_buffer_uses_zero_signal_sentinel_for_empty_tail
  overwrite_packed_signal_tail_keeps_index_prefix_stable
  medium_variable_update` passes 14/14.
- **PSTD provider-generic medium updates - RESOLVED [patch].**
  `PstdMediumUpdateState` now defines the provider-state contract for variable
  medium upload, full medium refresh, and source-correction disablement.
  `GpuPstdSolver<P>` medium methods are generic for providers whose state
  implements that contract; WGPU remains the only real implementation. Evidence
  tier: type-level/compile-time validation plus focused empirical tests;
  `cargo check -p kwavers-gpu --features gpu` passes, `cargo clippy -p
  kwavers-gpu --features gpu --lib -- -D warnings` passes, and `cargo nextest
  run -p kwavers-gpu --features gpu
  pstd_solver_medium_update_state_is_provider_owned
  pstd_solver_run_state_is_provider_owned
  pstd_solver_state_builder_uses_provider_handles
  pstd_solver_state_is_provider_associated
  pstd_pass_provider_is_generic_over_provider_trait
  pstd_command_provider_is_generic_over_provider_trait
  pstd_buffer_factory_is_generic_over_provider_trait
  pstd_bind_group_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait
  pstd_bind_group_layout_factory_is_generic_over_provider_trait
  packed_signal_len_keeps_storage_buffers_non_empty
  rewrite_packed_source_buffer_preserves_indices_and_signal_tail
  rewrite_packed_source_buffer_uses_zero_signal_sentinel_for_empty_tail
  overwrite_packed_signal_tail_keeps_index_prefix_stable
  medium_variable_update` passes 15/15.
- **PSTD provider-generic auto-device acquisition - RESOLVED [patch].**
  `PstdAutoDeviceProvider` now defines the provider acquisition contract, WGPU
  implements it through Hephaestus, and `GpuPstdSolver<P>::with_auto_device`
  is generic for providers that return real device/queue handles. WGPU remains
  the only real implementation; no CUDA PSTD placeholder was introduced.
  Evidence tier: type-level/compile-time validation plus focused empirical
  tests; `cargo check -p kwavers-gpu --features gpu` passes, `cargo clippy -p
  kwavers-gpu --features gpu --lib -- -D warnings` passes, and `cargo nextest
  run -p kwavers-gpu --features gpu
  pstd_solver_auto_device_provider_uses_provider_handles
  pstd_solver_medium_update_state_is_provider_owned
  pstd_solver_run_state_is_provider_owned
  pstd_solver_state_builder_uses_provider_handles
  pstd_solver_state_is_provider_associated
  pstd_pass_provider_is_generic_over_provider_trait
  pstd_command_provider_is_generic_over_provider_trait
  pstd_buffer_factory_is_generic_over_provider_trait
  pstd_bind_group_factory_is_generic_over_provider_trait
  pstd_pipeline_factory_is_generic_over_provider_trait
  pstd_bind_group_layout_factory_is_generic_over_provider_trait
  packed_signal_len_keeps_storage_buffers_non_empty
  rewrite_packed_source_buffer_preserves_indices_and_signal_tail
  rewrite_packed_source_buffer_uses_zero_signal_sentinel_for_empty_tail
  overwrite_packed_signal_tail_keeps_index_prefix_stable
  medium_variable_update` passes 16/16.
- **kwavers-analysis visualization data-pipeline direct ndarray/Rayon edge -
  RESOLVED [patch].** `DataProcessor` normalization and log scaling now use a
  Moirai-backed contiguous scalar traversal helper instead of
  `par_mapv_inplace`, with ndarray sequential traversal retained only for
  non-standard layouts. Evidence tier: static source audit plus
  compile-time validation and focused empirical tests; `cargo check -p
  kwavers-analysis --features gpu-visualization` passes, `cargo clippy -p
  kwavers-analysis --features gpu-visualization --lib -- -D warnings` passes,
  `cargo nextest run -p kwavers-analysis --features gpu-visualization -E
  "test(normalize_maps_contiguous_values_to_configured_range) or test(log_scale_clamps_values_at_configured_epsilon)"`
  passes 2/2, and `rg` finds no direct Rayon or ndarray-parallel hits under
  `crates/kwavers-analysis/src/visualization/data_pipeline`. Residual provider
  gap: `kwavers-analysis` still has direct Rayon in other analysis paths such
  as 3-D CPU beamforming modules, so the crate-level Rayon dependency and
  ndarray `rayon` feature remain in place.
- **kwavers-analysis performance optimizer direct Rayon edge - RESOLVED
  [patch].** `ParallelOptimizer` now uses Moirai indexed fan-out for 3-D
  traversal, Moirai ordered map-collect for array mapping, and Moirai indexed
  reduction for chunked reductions. The former `rayon::ThreadPoolBuilder`
  global mutation path is removed; `set_num_threads` validates nonzero lane
  counts and feeds only the local chunk-size heuristic. Evidence tier: static
  source audit plus compile-time validation and focused empirical tests;
  `cargo check -p kwavers-analysis` passes, `cargo clippy -p kwavers-analysis
  --lib -- -D warnings` passes, `cargo nextest run -p kwavers-analysis -E
  "test(parallel_optimizer_) or test(parallel_3d_visits_every_cell_exactly_once) or test(set_num_threads_)"`
  passes 8/8, and `rg` finds no direct Rayon symbols in
  `crates/kwavers-analysis/src/performance/optimization/parallel.rs`. Residual
  provider gap: `kwavers-analysis` still has direct Rayon and ndarray-parallel
  edges in 3-D CPU beamforming paths.
- **Shared scalar ndarray Moirai seam - RESOLVED [patch].**
  `kwavers-core::utils::iterators::apply_inplace` now owns scalar in-place
  ndarray transforms, using Moirai for standard-layout arrays and sequential
  ndarray traversal for non-standard layouts. `kwavers-analysis` visualization
  normalization/log scaling, PAM time-exposure acoustic squaring, and
  polynomial clutter-filter time normalization now use that shared seam instead
  of direct ndarray/Rayon `par_mapv_inplace`. Evidence tier: static source
  audit plus compile-time validation and focused empirical tests; `cargo check
  -p kwavers-core` passes, `cargo check -p kwavers-analysis` passes, `cargo
  clippy -p kwavers-core --lib -- -D warnings` passes, `cargo clippy -p
  kwavers-analysis --lib -- -D warnings` passes, `cargo clippy -p
  kwavers-analysis --features gpu-visualization --lib -- -D warnings` passes,
  focused nextest runs pass 2/2 for core layout paths, 3/3 for PAM/polynomial
  consumers, and 2/2 for visualization consumers. Residual provider gap:
  `kwavers-analysis` still has direct Rayon and ndarray-parallel edges in
  3-D CPU beamforming paths.
- **kwavers-analysis covariance direct ndarray/Rayon edge - RESOLVED
  [patch].** Beamforming covariance sample scaling, estimator real/complex
  normalization, shrinkage scaling, and spatial-smoothing normalization now use
  `kwavers-core::utils::iterators::apply_inplace` instead of direct
  ndarray/Rayon `par_mapv_inplace`. Evidence tier: static source audit plus
  compile-time validation and focused empirical tests; `cargo check -p
  kwavers-analysis` passes, `cargo clippy -p kwavers-analysis --lib -- -D
  warnings` passes, focused covariance nextest passes 30/30, and `rg` finds no
  direct Rayon or ndarray-parallel hits under
  `crates/kwavers-analysis/src/signal_processing/beamforming/covariance`.
  Residual provider gap: `kwavers-analysis` still has direct Rayon and
  ndarray-parallel edges in 3-D CPU beamforming paths.
- **kwavers-analysis safe-vectorization direct ndarray/Rayon edge - RESOLVED
  [patch].** `SafeVectorOps::add_arrays_parallel`, the non-contiguous chunked
  addition fallback, and `scalar_multiply_inplace` now use shared
  `kwavers-core::utils::iterators` Moirai-backed traversal seams instead of
  ndarray/Rayon `Zip::par_for_each` or local sequential in-place traversal.
  Evidence tier: static source audit plus compile-time validation and focused
  empirical tests; `cargo check -p kwavers-analysis` passes, `cargo clippy -p
  kwavers-analysis --lib -- -D warnings` passes, focused safe-vectorization
  nextest passes 8/8, and `rg` finds no direct Rayon or ndarray-parallel hits
  in `crates/kwavers-analysis/src/performance/safe_vectorization.rs`.
  Residual provider gap: `kwavers-analysis` still has direct Rayon and
  ndarray-parallel edges in 3-D CPU beamforming paths.
- **kwavers-analysis SLSC direct Rayon edge - RESOLVED [patch].**
  `SlscBeamformer::process_parallel` and `process_volume` now use Moirai
  ordered indexed collection instead of Rayon fan-out, and
  `create_coherence_map` clamps through the shared scalar ndarray seam instead
  of ndarray/Rayon `par_mapv_inplace`. Evidence tier: static source audit plus
  compile-time validation and focused empirical tests; `cargo check -p
  kwavers-analysis` passes, `cargo clippy -p kwavers-analysis --lib -- -D
  warnings` passes, focused SLSC nextest passes 24/24, and `rg` finds no
  direct Rayon or ndarray-parallel hits under
  `crates/kwavers-analysis/src/signal_processing/beamforming/slsc`. Residual
  provider gap: `kwavers-analysis` still has direct Rayon and ndarray-parallel
  edges in 3-D CPU beamforming paths.
- **kwavers-analysis neural scalar direct ndarray/Rayon edge - RESOLVED
  [patch].** Neural layer adaptation scaling and neural feature normalization
  now use `kwavers-core::utils::iterators::apply_inplace` instead of direct
  ndarray/Rayon `par_mapv_inplace`. Evidence tier: static source audit plus
  compile-time validation and focused empirical tests; `cargo check -p
  kwavers-analysis` passes, `cargo clippy -p kwavers-analysis --lib -- -D
  warnings` passes, focused neural scalar nextest passes 4/4, and `rg` shows
  the remaining neural direct Rayon edge before the distributed closure was
  limited to
  `crates/kwavers-analysis/src/signal_processing/beamforming/neural/distributed/core/processor.rs`.
  Residual provider gap: `kwavers-analysis` still has direct Rayon and
  ndarray-parallel edges in 3-D CPU beamforming paths.
- **kwavers-analysis distributed neural direct Rayon edge - RESOLVED
  [patch].** Distributed neural processor fan-out now uses
  `moirai_parallel::map_collect_mut_with` instead of direct Rayon
  `par_iter_mut`, preserving ordered result collection and typed error
  propagation. Evidence tier: static source audit plus compile-time validation
  and focused empirical tests; `cargo check -p kwavers-analysis --features
  pinn` passes, `cargo clippy -p kwavers-analysis --features pinn --lib -- -D
  warnings` passes, focused distributed neural nextest passes 6/6, and `rg`
  finds no direct Rayon or ndarray-parallel hits under
  `crates/kwavers-analysis/src/signal_processing/beamforming/neural`.
  Residual provider gap from this item was the 3-D CPU beamforming path,
  closed by the following resolved item.
- **kwavers-analysis 3-D CPU beamforming direct Rayon edge - RESOLVED
  [patch].** The DAS and MVDR CPU volume paths now use
  `moirai_parallel::map_collect_index_with` instead of direct Rayon parallel
  iterators. The package manifest no longer declares direct `rayon` or enables
  ndarray's `rayon` feature. Evidence tier: static source/manifest audit plus
  compile-time validation and focused empirical tests; `cargo check -p
  kwavers-analysis` passes, `cargo clippy -p kwavers-analysis --lib -- -D
  warnings` passes, focused DAS/MVDR nextest passes 39/39, `rg` finds no
  direct Rayon or ndarray-parallel hits in `crates/kwavers-analysis/src` or
  `crates/kwavers-analysis/Cargo.toml`, and `cargo tree -p kwavers-analysis
  --depth 1` lists no direct `rayon` dependency. Residual provider gap:
  closed for `kwavers-analysis`; continue Atlas provider migration in other
  crates.
- **Upstream Atlas consumer gate repairs - RESOLVED [patch].**
  `ritk-registration` now declares the `ritk-tensor-ops` dependency used by its
  native preprocessing executor, and Moirai core stale `CachePadded` channel,
  ring-buffer, and pool call sites now use the canonical `CacheAligned`
  wrapper. Evidence tier: compile-time validation; `cargo check -p
  ritk-registration` passes in Ritk, `cargo fmt -p moirai-core --check` and
  `cargo check -p moirai-core` pass in Moirai, and the downstream Kwavers GPU
  check/clippy/nextest gates pass after these owner-crate repairs.
- **GPU kernel-buffer provider abstraction - OPEN [arch].**
  PSTD state, raw WGPU handles, WGPU host scratch/upload buffers, WGPU state
  assembly, WGPU medium/source uploads, and WGPU run-cache mechanics are now
  provider-owned. WGPU dispatch, FFT, and per-phase pass encoding are also
  provider-state-owned, and high-level WGPU run-loop orchestration is now owned
  by `WgpuPstdState`. PSTD solver construction is provider-generic through
  `PstdStateBuilder`, with WGPU as the only real implementation. CUDA PSTD
  requires real provider-owned state, kernels, command/readback mechanics, and
  differential tests before CUDA can implement the same operation surface.
  Public PSTD run execution is provider-generic through `PstdRunState`, but
  WGPU is still the only real run implementation. Public PSTD medium/source
  updates are provider-generic through `PstdMediumUpdateState`, with WGPU as
  the only real medium-update implementation. Public PSTD auto-device
  acquisition is provider-generic through `PstdAutoDeviceProvider`, with WGPU
  as the only real acquisition implementation. CUDA compute must not be
  exposed until those real contracts exist. Current evidence tier: static
  source audit.
- **Top-level GPU FFT raw WGPU acquisition - RESOLVED [patch].**
  `crates/kwavers/tests/gpu_fft_arbitrary_size.rs` and
  `crates/kwavers/tests/gpu_cpu_fft_parity.rs` now construct ignored hardware
  FFT plans through Apollo's `FftBackend` and
  `kwavers_math::fft::gpu_fft::WgpuBackend`, with explicit Leto test buffers
  at the GPU boundary. The scoped audit finds no `wgpu::Instance`,
  `request_adapter`, raw `Arc<wgpu::Device>`/`Queue`, `GpuFft3d::new`,
  `pollster::block_on`, or `tokio::test` references in those files. Evidence
  tier: type-level/compile-time validation plus static source audit; focused
  `kwavers --features gpu` check/clippy pass, and nextest harness discovery
  passes with 22 ignored GPU hardware tests skipped. Residual: top-level
  raw-buffer/device tests still exercise WGPU-specialized public handles until
  those surfaces are provider-wrapped.
- **Top-level GPU runtime dependency edges - RESOLVED [patch].**
  The `kwavers/gpu` feature no longer forwards direct `wgpu`, `bytemuck`, or
  `pollster` dependencies. `kwavers-gpu` owns the synchronous provider
  acquisition, buffer write/readback, acoustic-kernel, wave-equation, and FDTD
  pressure readback wrappers, and the top-level GPU buffer/device/allocation
  and compute-kernel tests use those wrappers without direct runtime helper
  imports. Evidence tier: type-level/compile-time validation plus static
  source/dependency audit and focused empirical tests; fmt passes,
  `kwavers-gpu --features gpu --lib` check/clippy pass, affected
  `kwavers --features gpu` test-target check/clippy pass, affected nextest
  passes 28/28, the top-level source audit finds no `wgpu::`, `pollster::`,
  or `bytemuck::` references, and the depth-1 `kwavers --features gpu`
  dependency audit has no direct WGPU/pollster/bytemuck hits. Residual:
  concrete WGPU handles remain inside `kwavers-gpu` provider implementations
  for current WGSL kernels.
- **kwavers-math tensor Burn placeholder - RESOLVED [patch].**
  `kwavers-math::tensor` no longer exposes the unused
  `TensorBackend::BurnNdArray` variant or documents unimplemented Burn
  ndarray/WGPU/CUDA tensor backends. The module now documents the actual
  ndarray-backed host tensor boundary and leaves differentiable PINN tensors to
  the solver-layer Coeus provider migration. Evidence tier: static source
  audit plus compile-time/lint validation and focused tests; `cargo check -p
  kwavers-math` passes, `cargo clippy -p kwavers-math --all-targets -- -D
  warnings` passes, `cargo nextest run -p kwavers-math tensor r2c_optimized
  --status-level fail --no-fail-fast` passes 9/9, and the Burn-specific tensor
  source audit returns no hits under `crates/kwavers-math/src`.
- **kwavers-math tensor Moirai traversal - RESOLVED [patch].**
  `NdArrayTensor::map_inplace` now dispatches contiguous tensor storage through
  `moirai_parallel::for_each_chunk_mut_with::<Adaptive>` and uses sequential
  ndarray mutation only for non-contiguous `ArrayD` layouts where a disjoint
  memory-order slice is unavailable. This removes the tensor module's direct
  ndarray/Rayon traversal edge without changing the broader `kwavers-math`
  ndarray `rayon` feature, which is still needed by FFT and operator kernels.
  Evidence tier: static source audit plus compile-time/lint validation and
  focused value-semantic tests; `cargo check -p kwavers-math` passes, `cargo
  clippy -p kwavers-math --all-targets -- -D warnings` passes, `cargo nextest
  run -p kwavers-math tensor --status-level fail --no-fail-fast` passes 9/9,
  the tensor Rayon audit returns no hits, and `cargo tree -p kwavers-math
  --depth 1` shows the direct `moirai-parallel` dependency.
- **kwavers-gpu leaf acquisition cleanup - RESOLVED [patch].**
  `AcousticFieldKernel`, `CoreGpuContext`, PSTD construction, and
  `ComputeManager` now acquire through Hephaestus-backed provider wrappers
  instead of local WGPU instance/adapter/device request code. `ComputeManager`
  uses the generic `GpuDevice` acquisition wrapper; raw WGPU handles remain
  only where WGSL buffer/shader APIs require them. Evidence tier: static
  source audit plus compile-time validation and focused GPU tests; `cargo
  check -p kwavers-gpu --features gpu` passes and `cargo nextest run -p
  kwavers-gpu --features gpu backend gpu::shaders::neural_network
  gpu::multi_gpu pstd_gpu::tests::construction --status-level fail
  --no-fail-fast` passes 37/37.
- **kwavers-gpu direct WGPU acquisition - RESOLVED [patch].**
  Multi-GPU device discovery now goes through Hephaestus
  `ComputeDeviceAcquisition::try_acquire_devices`, and PSTD GPU tests use a
  shared Hephaestus-backed provider helper. Source search for WGPU acquisition
  APIs now leaves only test-only provider helper calls in
  `backend/buffers.rs` and `pstd_gpu/tests/helpers.rs`; production acquisition
  routes through the provider wrappers. Evidence tier: static source audit
  plus compile-time validation and focused GPU tests; `cargo check -p
  kwavers-gpu --features gpu` passes and `cargo nextest run -p kwavers-gpu
  --features gpu backend gpu::shaders::neural_network gpu::multi_gpu
  pstd_gpu::tests::construction --status-level fail --no-fail-fast` passes
  37/37.
- **kwavers-gpu backend spatial-derivative CPU fallback - RESOLVED [patch].**
  `WgpuComputeProvider::apply_spatial_derivative` now uses real WGPU
  finite-difference shader dispatch instead of a CPU-computed derivative behind
  the GPU provider, and `operators.wgsl` no longer has the copy placeholder for
  `spatial_derivative`. Evidence tier: static shader/source audit plus
  compile-time validation and focused empirical test; `cargo check -p
  kwavers-gpu --features gpu` passes, `cargo clippy -p kwavers-gpu --features
  gpu --lib -- -D warnings` passes, `cargo check -p kwavers-gpu --features
  cuda-provider` passes, focused backend nextest passes 5/5, and the full
  backend test module passes 10/10. Residual
  provider gap: CUDA derivative dispatch still requires real CUDA kernels and
  WGPU-vs-CUDA differential tests before CUDA can implement
  `GpuComputeProvider`.
- **kwavers-gpu backend scalar precision contract - RESOLVED [patch].**
  `GpuComputeProvider` now carries an associated scalar type, and the WGPU
  backend operator path dispatches provider-native `Array3<f32>` buffers
  directly. WGPU backend capabilities no longer claim f64 support, and the
  solver-owned `ComputeBackend` operation methods now use the provider scalar
  instead of a fixed f64 rejection branch. The WGPU provider-native path now has
  value-semantic elementwise multiplication and derivative tests, plus
  pre-dispatch shape guards for elementwise/derivative buffers. Follow-up:
  provider-native dispatch now accepts `leto::Array3<f32>` at
  `GpuComputeProvider`/`GPUBackend::dispatch_*`, and
  `WgpuBackendBufferManager` uploads/readbacks use `leto::Array3<f32>`
  directly instead of a provider-side ndarray adapter. Realtime field maps now
  use `leto::Array3<f64>` on the provider-generic GPU surface. The
  downstream gate also required
  repairing Hephaestus WGPU axis-reduction call sites to use the shared
  `hephaestus-core::plan_axis_reduction` planner instead of a stale local
  helper. Evidence tier: type-level/compile-time validation plus focused
  empirical tests; `cargo check -p hephaestus-wgpu` passes, `cargo check -p
  kwavers-gpu --features gpu` passes, `cargo check -p kwavers-gpu --features
  cuda-provider` passes, `cargo clippy -p kwavers-gpu --features gpu --lib --
  -D warnings` passes, and focused nextest over provider-native elementwise,
  provider-native derivative, provider-generic backend, capability, and
  f64-rejection tests passes 8/8. The 2026-07-03 Leto API follow-up also
  passed `cargo fmt -p kwavers-gpu --check`, `cargo check -p kwavers-gpu
  --features gpu`, `cargo clippy -p kwavers-gpu --features gpu --lib -- -D
  warnings`, and the same focused nextest filter 8/8. The WGPU
  buffer-manager Leto staging follow-up passed the same checks. The realtime
  Leto field-map follow-up passed `rustup run nightly cargo fmt -p
  kwavers-gpu --check`, `rustup run nightly cargo check -p kwavers-gpu
  --features gpu`, `rustup run nightly cargo clippy -p kwavers-gpu --features
  gpu --lib -- -D warnings`, focused realtime/f64 nextest 5/5, `rustup run
  nightly cargo check -p kwavers-gpu --features cuda-provider`, and focused
  CUDA-provider provider/realtime nextest 7/7.
  Residual [arch] gap: CUDA still needs real kernels and WGPU/CUDA
  differential tests before implementing `GpuComputeProvider`; solver-level
  GPU dispatch still needs a scalar-generic `ComputeBackend` contract before
  f64 GPU arithmetic can be claimed.
- **kwavers-gpu direct Rayon edge - RESOLVED [patch].**
  `kwavers-gpu::gpu::pipeline::{realtime,streaming}` now dispatches Hilbert
  envelope planes and synthetic RF receive slices through `moirai-parallel`
  chunk scheduling. The crate no longer depends directly on Rayon. Evidence
  tier: static source/dependency audit plus focused empirical tests; `rg` finds
  no direct Rayon imports or parallel iterator calls under `crates/kwavers-gpu`,
  `cargo tree -p kwavers-gpu --features gpu --depth 1` shows
  `moirai-parallel` with no direct `rayon`, `cargo check -p kwavers-gpu
  --features gpu` passes, and `cargo nextest run -p kwavers-gpu --features gpu
  gpu::pipeline --status-level fail --no-fail-fast` passes 5/5.
- **kwavers-boundary direct ndarray/Rayon edge - RESOLVED [patch].**
  CPML full-field damping, CPML strip memory/correction updates, and adaptive
  boundary attenuation now dispatch through private `moirai-parallel` traversal
  helpers, with sequential ndarray view semantics retained for non-standard
  layouts. `kwavers-boundary` no longer enables ndarray's `rayon` feature.
  Evidence tier: static source/dependency audit plus focused empirical tests;
  `rg` finds no direct Rayon or ndarray-parallel hits under
  `crates/kwavers-boundary`, `cargo tree -p kwavers-boundary --depth 1` shows
  `moirai-parallel` with no direct `rayon`, `cargo check -p kwavers-boundary`
  passes, and `cargo nextest run -p kwavers-boundary --status-level fail
  --no-fail-fast` passes 96/96.

- **kwavers-receiver direct ndarray/Rayon edge - RESOLVED [patch].**
  Pressure and velocity statistics updates now dispatch through
  `moirai-parallel` triple/quad mutable chunk helpers for standard-layout
  arrays, with sequential ndarray `Zip` retained for non-standard layouts.
  `kwavers-receiver` no longer depends directly on `rayon` or enables
  ndarray's `rayon` feature. Evidence tier: static source/dependency audit
  plus focused empirical tests; `rg` finds no direct Rayon or ndarray-parallel
  hits under `crates/kwavers-receiver`, `cargo tree -p kwavers-receiver
  --depth 1` shows `moirai-parallel` with no direct `rayon`, `cargo check -p
  kwavers-receiver` passes, and `cargo nextest run -p kwavers-receiver
  --status-level fail --no-fail-fast` passes 47/47.

- **kwavers-medium direct ndarray/Rayon edge - RESOLVED [patch].**
  Medium property traversal, absorption/dispersion k-space updates, and
  frequency-dependent correction now dispatch through `moirai-parallel`
  indexed and chunk traversal. Sequential ndarray traversal is retained for
  non-standard layouts through a crate-local provider adapter.
  `kwavers-medium` no longer depends directly on `rayon` or enables ndarray's
  `rayon` feature. Evidence tier: static source/dependency audit plus focused
  empirical tests; `rg` finds no direct Rayon or ndarray-parallel hits under
  `crates/kwavers-medium`, `cargo tree -p kwavers-medium --depth 1` shows
  `moirai-parallel` with no direct `rayon`, `cargo check -p kwavers-medium`
  passes, `cargo clippy -p kwavers-medium --all-targets -- -D warnings`
  passes, and `cargo nextest run -p kwavers-medium --status-level fail
  --no-fail-fast` passes 179/179.

- **kwavers-core direct Rayon edge — RESOLVED [patch].** `kwavers-core` no
  longer depends directly on `rayon` or enables ndarray's `rayon` feature.
  NUMA first-touch, SoA first-touch, and gradient interior-loop parallelism now
  dispatch through `moirai-parallel`, matching the Atlas provider-first
  migration path. Current Clippy also surfaced constant-invariant test
  assertions in `kwavers-core`; those checks now execute as `const` assertions
  with the same value predicates. Evidence tier: static analysis plus
  empirical package tests; `cargo clippy -p kwavers-core --all-targets
  --all-features -- -D warnings` passes, `cargo nextest run -p kwavers-core`
  passes 68/68, and `cargo tree -p kwavers-core --depth 1` shows
  `moirai-parallel` as the direct parallel provider.
- **kwavers-therapy orchestrator ndarray/Rayon edge - RESOLVED [minor].**
  Therapy integration acoustic-field generation and acoustic-heating updates
  now dispatch through shared `kwavers-core::utils::iterators`
  `moirai-parallel` helpers for standard-layout arrays, with sequential
  ndarray traversal retained only for non-standard-layout fallback. Current
  package-local therapy Clippy blockers were cleaned without changing tested
  predicates. Evidence tier: static source audit plus focused empirical tests;
  `cargo clippy -p kwavers-core -p kwavers-therapy --all-targets -- -D
  warnings` passes, `cargo nextest run -p kwavers-core iterators
  --status-level fail --no-fail-fast` passes 2/2, and `cargo nextest run -p
  kwavers-therapy therapy_integration --status-level fail --no-fail-fast`
  passes 59/59. Follow-up timeout closure is tracked in the next item.
- **kwavers-therapy abdominal preprocessing timeout - RESOLVED with residual
  performance debt [patch].** The two abdominal preprocessing tests no longer
  terminate under nextest. The fix tightens the acoustic recording window to
  actual source/body/receiver geometry, reuses adjoint RTM replay buffers, and
  evaluates elastic-FWI line search lazily in first-improving order. Evidence
  tier: empirical nextest execution plus lint validation; `cargo clippy -p
  kwavers-therapy --all-targets -- -D warnings` passes and `cargo nextest run
  -p kwavers-therapy abdominal_preprocessing --status-level fail
  --no-fail-fast` passes 2/2. Broader package verification: `cargo nextest run
  -p kwavers-therapy --status-level fail --no-fail-fast` passes 340/340 with 1
  skipped. Residual risk: the package run takes about 141 s and the paired
  abdominal filter still takes about 110 s, exceeding the 30 s slow-test
  budget, so a follow-up performance item remains open.
- **kwavers-therapy elastic-shear and emission Rayon edges - RESOLVED [patch].**
  `theranostic_guidance::elastic_shear::sampling::migrate_residual` now uses
  `moirai_parallel::map_collect_index_with` for independent flat-index voxel
  migration and no longer imports Rayon. `waveform::emission` now uses
  `moirai_parallel::map_collect_with` for passive-acoustic-mapping eikonal
  delay-column solves. Evidence tier: static source audit plus focused
  empirical tests; `cargo check -p kwavers-therapy`, `cargo clippy -p
  kwavers-therapy --all-targets -- -D warnings`, `cargo nextest run -p
  kwavers-therapy residual_migration_samples_expected_arrival --status-level
  fail --no-fail-fast`, and `cargo nextest run -p kwavers-therapy
  waveform::emission --status-level fail --no-fail-fast` pass. Residual
  provider gap: direct Rayon remains in `nonlinear3d` therapy paths, so the
  crate-level Rayon dependency cannot be removed yet.
- **kwavers-therapy standing-wave FDTD Rayon edge - RESOLVED [patch].**
  `theranostic_guidance::standing_wave_opt::fdtd` now dispatches independent
  Green-function column solves through `moirai_parallel::map_collect_with`
  instead of Rayon. Evidence tier: static source audit plus focused empirical
  tests; `cargo check -p kwavers-therapy`, `cargo clippy -p kwavers-therapy
  --all-targets -- -D warnings`, and `cargo nextest run -p kwavers-therapy
  standing_wave --status-level fail --no-fail-fast` pass. Residual provider
  gap: direct Rayon remains in `nonlinear3d` therapy paths, so the crate-level
  Rayon dependency cannot be removed yet.
- **kwavers-therapy waveform-forward Rayon edge - RESOLVED [patch].**
  `theranostic_guidance::waveform::forward` now dispatches CPML row updates,
  pressure updates, attenuation, and peak-pressure updates through
  `moirai-parallel` instead of direct Rayon row chunks and parallel iterators.
  Evidence tier: static source audit plus focused empirical tests; `cargo
  check -p kwavers-therapy`, `cargo clippy -p kwavers-therapy --all-targets --
  -D warnings`, and `cargo nextest run -p kwavers-therapy waveform
  --status-level fail --no-fail-fast` pass, with nextest reporting 13/13
  passed and 1 slow test. Residual provider gap: direct Rayon remains in
  `nonlinear3d` forward/cavitation therapy paths, so the crate-level Rayon
  dependency cannot be removed yet.
- **kwavers-therapy nonlinear3d absorption Rayon edge - RESOLVED [patch].**
  `theranostic_guidance::nonlinear3d::absorption` now dispatches
  Treeby-Cox coefficient construction plus forward/adjoint absorption
  element-wise updates through `moirai-parallel` instead of direct Rayon.
  Evidence tier: static source audit plus focused empirical tests; `cargo
  check -p kwavers-therapy`, `cargo clippy -p kwavers-therapy --all-targets --
  -D warnings`, and `cargo nextest run -p kwavers-therapy absorption
  --status-level fail --no-fail-fast` pass, with nextest reporting 5/5 passed.
  Residual provider gap: direct Rayon remains in `nonlinear3d`
  forward-stencil and passive-inverse therapy paths, so the crate-level Rayon
  dependency cannot be removed yet.
- **kwavers-therapy nonlinear3d cavitation-forward Rayon edge - RESOLVED
  [patch].** `theranostic_guidance::nonlinear3d::cavitation::forward` now
  dispatches contiguous source-mask max reduction and cavitation source-density
  mapping through `moirai-parallel` instead of direct Rayon. Evidence tier:
  static source audit plus focused empirical tests; `cargo check -p
  kwavers-therapy`, `cargo clippy -p kwavers-therapy --all-targets -- -D
  warnings`, and `cargo nextest run -p kwavers-therapy cavitation
  --status-level fail --no-fail-fast` pass, with nextest reporting 46/46
  passed. Residual provider gap: direct Rayon remains in `nonlinear3d`
  forward-stencil and passive-inverse therapy paths, so the crate-level Rayon
  dependency cannot be removed yet.
- **kwavers-therapy nonlinear3d forward-stencil Rayon edge - RESOLVED
  [patch].** `theranostic_guidance::nonlinear3d::forward::stencil` now
  dispatches the Westervelt x-slab cell update through `moirai-parallel`
  instead of direct Rayon. The adjacent Westervelt performance docs now name
  the Atlas provider seam rather than the removed Rayon call. Evidence tier:
  static source audit plus focused empirical tests; `cargo check -p
  kwavers-therapy`, `cargo clippy -p kwavers-therapy --all-targets -- -D
  warnings`, and `cargo nextest run -p kwavers-therapy nonlinear3d
  --status-level fail --no-fail-fast` pass, with nextest reporting 59/59
  passed. Residual provider gap: direct Rayon remains in `nonlinear3d`
  passive-inverse therapy paths, so the crate-level Rayon dependency cannot be
  removed yet.
- **kwavers-therapy nonlinear3d passive-inverse Rayon edge - RESOLVED
  [patch].** `theranostic_guidance::nonlinear3d::cavitation::passive_inverse`
  now dispatches dense Green-operator fill, forward apply, normal-gradient,
  residual/objective reductions, and projected updates through
  `moirai-parallel` instead of direct Rayon. `kwavers-therapy` no longer
  depends directly on Rayon and no longer enables ndarray's `rayon` feature.
  Evidence tier: static source/dependency audit plus focused empirical tests;
  `cargo check -p kwavers-therapy`, `cargo clippy -p kwavers-therapy
  --all-targets -- -D warnings`, `cargo nextest run -p kwavers-therapy
  cavitation --status-level fail --no-fail-fast`, and `cargo nextest run -p
  kwavers-therapy nonlinear3d --status-level fail --no-fail-fast` pass, with
  nextest reporting 46/46 and 59/59 passed. `rg` finds no Rayon hits under
  `crates/kwavers-therapy/src` or its manifest, and `cargo tree -p
  kwavers-therapy --depth 1` shows no direct `rayon`.
- **kwavers-simulation direct Rayon edge — RESOLVED [patch].**
  `kwavers-simulation` no longer depends directly on `rayon` or enables
  ndarray's `rayon` feature. Photoacoustic multi-wavelength fluence and
  time-reversal reconstruction buffer writes now dispatch through
  `moirai-parallel`. The slice also repaired all-features GPU-PSTD adapter
  tests by importing the `Solver` trait whose methods they call. Evidence tier:
  static analysis plus empirical package tests; `cargo clippy -p
  kwavers-simulation --all-targets --all-features --no-deps -- -D warnings`
  passes, `cargo nextest run -p kwavers-simulation --all-features` passes
  91/91, and `cargo tree -p kwavers-simulation --depth 1` shows
  `moirai-parallel` as a direct dependency with no direct `rayon` dependency.
- **Dependency-inclusive kwavers-simulation Clippy gate — RESOLVED [patch].**
  The `kwavers-physics` lints that blocked `cargo clippy -p
  kwavers-simulation --all-targets --all-features -- -D warnings` are closed:
  IVUS delivery and Gaussian photoacoustic profile functions now use typed
  request structs, Gaussian deconvolution and apodization-window helpers return
  typed result structs, thin PyO3 wrappers unpack those Rust-owned results, and
  the centered-Hann tests now appear after production items. Evidence tier:
  static analysis plus focused value-semantic tests; `cargo clippy -p
  kwavers-physics --all-targets -- -D warnings`, `cargo check -p
  kwavers-python`, focused `cargo nextest run -p kwavers-physics
  ivus_microbubble_delivery_fraction gaussian_absorber_photoacoustic_profile
  gaussian_deconvolution_fixture apodization_response centered_hann_tone_burst`
  (10/10), and dependency-inclusive `cargo clippy -p kwavers-simulation
  --all-targets --all-features -- -D warnings` pass.
- **kwavers-transducer direct Rayon edge — RESOLVED [patch].**
  `kwavers-transducer` no longer depends directly on `rayon` or enables
  ndarray's `rayon` feature. Linear/matrix focus-delay writes and arc, bowl,
  multi-bowl, and phased-array source-field writes now dispatch through
  `moirai-parallel` indexed mutable-slice helpers. Evidence tier: static
  analysis plus empirical package tests; `cargo clippy -p kwavers-transducer
  --all-targets -- -D warnings`, `cargo nextest run -p kwavers-transducer`
  (203/203, 1 skipped), and `cargo tree -p kwavers-transducer --depth 1` pass,
  with `moirai-parallel` as the direct parallel provider.
- **Unused low-level ndarray Rayon features — RESOLVED [patch].**
  `kwavers-field`, `kwavers-signal`, `kwavers-source`, and `kwavers-imaging`
  no longer enable ndarray's `rayon` feature. A crate-local source search found
  no direct Rayon, Tokio, or ndarray-parallel call sites in those trees, so the
  feature activation was not an active computation provider. Evidence tier:
  static analysis plus package tests; the four-crate fmt, check, clippy, and
  nextest gates pass, and `cargo tree -p` for the four crates shows no direct
  `rayon` dependency.
- **Remaining workspace Rayon/Tokio usage — OPEN [patch].** Root workspace
  dependencies and non-core crates still contain direct `rayon`/`tokio` usage.
  Next closure increment: audit call sites by crate, replace the smallest
  provider-owned edge with Moirai, and keep any missing Moirai capability in
  Moirai rather than duplicating it downstream.
- **kwavers-solver same-aperture direct Rayon edge — RESOLVED [patch].**
  `kwavers-solver::inverse::same_aperture` encoded/operator paths now dispatch
  through `moirai-parallel` instead of direct Rayon iterators. Evidence tier:
  static source audit plus focused empirical tests; `rg` found no direct Rayon
  iterator imports or calls in the module tree, `cargo check -p
  kwavers-solver` passed, and `cargo nextest run -p kwavers-solver
  same_aperture` passed 7/7.
- **kwavers-solver linear Born inversion direct Rayon edge — RESOLVED
  [patch].** `kwavers-solver::inverse::linear_born_inversion` dense products,
  volume-operator construction, normal-equation reductions, and Sobolev Z-pass
  now dispatch through `moirai-parallel`. The Sobolev pass uses the
  provider-owned `for_each_chunk_mut_with_state` primitive added in Moirai.
  Evidence tier: static source audit plus focused empirical tests; `rg` found
  no direct Rayon iterator imports or calls in the module tree, `cargo check -p
  kwavers-solver` passed, and `cargo nextest run -p kwavers-solver
  linear_born_inversion` passed 6/6.
- **kwavers-solver time-domain FWI search/MOFI direct Rayon edge — RESOLVED
  [patch].** `kwavers-solver::inverse::fwi::time_domain::{search,mofi}` now
  routes joint objective, line-search trial-model writes, and coarse-pose
  candidate evaluation through `moirai-parallel`. Evidence tier: static source
  audit plus focused empirical tests; `rg` found no direct Rayon/thread-pool
  hits in `search.rs` or `mofi/mod.rs`, `cargo check -p kwavers-solver`
  passed, `cargo nextest run -p kwavers-solver time_domain --status-level
  fail` passed 58/58, and `cargo test --doc -p kwavers-solver --
  --show-output` passed 6 doctests with 14 ignored. `cargo doc -p
  kwavers-solver --no-deps` generated docs but reported 189 pre-existing
  rustdoc warnings outside this slice.
- **kwavers-solver time-domain FWI constraints/adjoint-state direct Rayon edge
  - RESOLVED [patch].** `constraints.rs` model clamping and pressure
  second-derivative writes plus `adjoint_state.rs` signed-correlation
  accumulation now dispatch standard-layout field volumes through
  `moirai-parallel`, with sequential Zip semantics retained for non-standard
  ndarray views. Evidence tier: static source audit plus focused empirical
  tests; `rg` found no direct Rayon hits in those two files, `cargo check -p
  kwavers-solver` passed, and `cargo nextest run -p kwavers-solver time_domain
  --status-level fail --no-fail-fast` passed 58/58.
- **kwavers-solver time-domain FWI field-update direct Rayon edge - RESOLVED
  [patch].** `adjoint.rs`, `gradient.rs`, and `inversion/multi_source.rs`
  now route field mutations through the shared Moirai-backed `field_ops.rs`
  helper module, and the stale `rayon::par_iter` doc mention in `forward.rs`
  is removed. Evidence tier: static source audit plus focused empirical tests;
  `rg` found no direct Rayon/thread-pool hits in
  `crates/kwavers-solver/src/inverse/fwi/time_domain`, `cargo check -p
  kwavers-solver` passed, and `cargo nextest run -p kwavers-solver
  time_domain --status-level fail --no-fail-fast` passed 58/58. Follow-up on
  2026-07-04 removed explicit ndarray `Zip` fallback traversal from
  `field_ops.rs`; scoped `rg` found no `Zip` tokens in that helper, `cargo
  fmt -p kwavers-solver --check` passed, and `cargo check -p kwavers-solver`
  passed, `cargo clippy -p kwavers-solver --lib -- -D warnings` passed, and
  `cargo nextest run -p kwavers-solver time_domain --status-level fail
  --no-fail-fast` passed 58/58.
- **kwavers-solver workspace direct Rayon edge - RESOLVED [patch].**
  `workspace::inplace_ops` now dispatches standard-layout in-place arithmetic
  through `moirai-parallel`, with sequential ndarray `Zip` retained for
  non-standard layouts. Evidence tier: static source audit plus focused
  empirical tests; `rg` found no direct Rayon/thread-pool hits in
  `crates/kwavers-solver/src/workspace`, `cargo check -p kwavers-solver`
  passes, and `cargo nextest run -p kwavers-solver workspace --status-level
  fail --no-fail-fast` passes 20/20.
- **kwavers-solver time-integration direct Rayon edge - RESOLVED [patch].**
  `integration::time_integration::time_stepper` now dispatches RK4 stage
  updates and Adams-Bashforth 2/3 field updates through `moirai-parallel` for
  standard-layout arrays, with sequential ndarray `Zip` retained for
  non-standard layouts. Evidence tier: static source audit plus focused
  empirical tests; `rg` found no direct Rayon/thread-pool hits in
  `crates/kwavers-solver/src/integration/time_integration`, `cargo check -p
  kwavers-solver` passes, and `cargo nextest run -p kwavers-solver
  time_integration --status-level fail --no-fail-fast` passes 12/12.
- **kwavers-solver Helmholtz FEM direct Rayon edge - RESOLVED [patch].**
  `forward::helmholtz::fem::assembly` now collects per-element FEM
  contributions through `moirai_parallel::map_collect_index_with::<Adaptive>`,
  preserving ordered serial accumulation into global sparse matrices. The
  assembly boundary now rejects mismatched element/stiffness/mass/RHS array
  lengths with `KwaversError::InvalidInput` instead of silently truncating the
  zipped contribution stream. Evidence tier: source audit plus compile-time
  and focused empirical validation; scoped `rg` finds no direct Rayon token in
  `assembly.rs`, `cargo check -p kwavers-solver` passes, `cargo clippy -p
  kwavers-solver --lib -- -D warnings` passes, and focused `cargo nextest run
  -p kwavers-solver -E "test(helmholtz) or
  test(assembly_rejects_mismatched_element_contribution_lengths)"` passes
  10/10.
- **kwavers-solver Westervelt conservation direct Rayon edge - RESOLVED
  [patch].** `forward::nonlinear::westervelt::conservation` now computes
  total acoustic energy, pressure-gradient momentum proxy, and acoustic mass
  perturbation through `moirai_parallel::reduce_index_with::<Adaptive>` instead
  of direct Rayon `par_iter`/`into_par_iter` reductions. Evidence tier: source
  audit plus compile-time and focused empirical validation; scoped `rg` finds
  no direct Rayon or ndarray-parallel token in `conservation.rs`, `cargo check
  -p kwavers-solver` passes, `cargo clippy -p kwavers-solver --lib -- -D
  warnings` passes, and focused `cargo nextest run -p kwavers-solver -E
  "test(westervelt)"` passes 32/32. Residual: KZK/PSTD/FDTD/SWE solver
  modules still contain
  direct Rayon or ndarray-parallel call sites.
- **kwavers-solver Westervelt Laplacian direct Rayon edge - RESOLVED
  [patch].** `forward::nonlinear::westervelt::laplacian` now dispatches
  O2/O4/O6 finite-difference stencil slabs through
  `moirai_parallel::for_each_chunk_mut_enumerated_with::<Adaptive>` over
  standard-layout `i` slabs instead of direct Rayon
  `axis_iter_mut(Axis(0)).into_par_iter()` traversal. Evidence tier: source
  audit plus compile-time and focused empirical validation; scoped `rg` finds
  no direct Rayon or ndarray-parallel token in `laplacian.rs`, `cargo check -p
  kwavers-solver` passes, `cargo clippy -p kwavers-solver --lib -- -D
  warnings` passes, and focused `cargo nextest run -p kwavers-solver -E
  "test(westervelt_laplacian) or test(test_westervelt_fdtd_creation) or
  test(westervelt)"` passes 32/32. Residual: KZK/PSTD/FDTD/SWE solver modules
  still contain direct Rayon or ndarray-parallel call sites.
- **kwavers-solver Westervelt nonlinear/update direct Rayon edge - RESOLVED
  [patch].** `forward::nonlinear::westervelt::nonlinear` now computes the
  full-history product-rule curvature and first-step `2(dp/dt)^2`
  initialization through `moirai_parallel::enumerate_mut_with::<Adaptive>`.
  `forward::nonlinear::westervelt::update` now dispatches the medium-coupled
  leapfrog propagation, nonlinear coefficient, artificial viscosity, and
  absorption multiply through the same Moirai indexed traversal over
  standard-layout pressure fields. Evidence tier: source audit plus
  compile-time and focused empirical validation; scoped `rg` finds no direct
  Rayon or ndarray-parallel token in `nonlinear.rs` or `update.rs`, `cargo
  check -p kwavers-solver` passes, `cargo clippy -p kwavers-solver --lib -- -D
  warnings` passes, `cargo test --no-run --package kwavers-solver` passes, and
  focused `cargo nextest run -p kwavers-solver -E "test(westervelt)"` passes
  32/32. Residual: KZK/PSTD/FDTD/SWE solver modules still contain direct Rayon
  or ndarray-parallel call sites.
- **kwavers-solver KZK observables direct Rayon edge - RESOLVED [patch].**
  `forward::nonlinear::kzk::solver::observables` now computes intensity and
  peak-pressure summaries through
  `moirai_parallel::enumerate_mut_with::<Adaptive>` over standard-layout 2-D
  outputs, and `forward::nonlinear::kzk::solver::traits` now computes the
  physics-layer RMS field through the same provider traversal. Evidence tier:
  source audit plus compile-time and focused empirical validation; scoped `rg`
  finds no direct Rayon or ndarray-parallel token in `observables.rs` or
  `traits.rs`, `cargo check -p kwavers-solver` passes, `cargo clippy -p
  kwavers-solver --lib -- -D warnings` passes, and focused `cargo nextest run
  -p kwavers-solver -E "test(kzk)"` passes 49/49. Follow-up KZK operator
  slices below close the residual direct Rayon and ndarray-parallel call sites
  in the KZK subtree.
- **kwavers-solver KZK angular-spectrum direct Rayon edge - RESOLVED [patch].**
  `forward::nonlinear::kzk::angular_spectrum_2d` and
  `forward::nonlinear::kzk::parabolic_diffraction` now pack real fields into
  cached complex FFT scratch, apply diagonal spectral multipliers, and project
  real outputs through `moirai_parallel::enumerate_mut_with::<Adaptive>` when
  arrays expose standard-layout slices. Non-contiguous borrowed field views keep
  sequential ndarray semantics without cloning. Evidence tier: source audit
  plus compile-time and focused empirical validation; scoped `rg` finds no
  direct Rayon or ndarray-parallel token in the edited files, `cargo fmt -p
  kwavers-solver --check` passes, `cargo check -p kwavers-solver` passes,
  `cargo clippy -p kwavers-solver --lib -- -D warnings` passes, and focused
  `cargo nextest run -p kwavers-solver -E "test(kzk) or test(absorption)"`
  passes 80/80. Follow-up KZK operator slices below close the residual direct
  Rayon and ndarray-parallel call sites in the KZK subtree.
- **kwavers-solver KZK absorption direct Rayon edge - RESOLVED [patch].**
  `forward::nonlinear::kzk::absorption` now applies the power-law spectral
  attenuation step through
  `moirai_parallel::for_each_chunk_mut_enumerated_with::<Adaptive>` over
  standard-layout `i` slabs. Each scheduled slab owns its waveform scratch, and
  the precomputed half/full-step attenuation masks remain unchanged. Evidence
  tier: source audit plus compile-time and focused empirical validation; scoped
  `rg` finds no direct Rayon or ndarray-parallel token in `absorption.rs`,
  `cargo fmt -p kwavers-solver --check` passes, `cargo check -p
  kwavers-solver` passes, `cargo clippy -p kwavers-solver --lib -- -D
  warnings` passes, and focused `cargo nextest run -p kwavers-solver -E
  "test(kzk) or test(absorption)"` passes 80/80. Follow-up KZK operator slices
  below close the residual direct Rayon and ndarray-parallel call sites in the
  KZK subtree.
- **kwavers-solver KZK nonlinear/diffraction direct Rayon edge - RESOLVED
  [patch].** `forward::nonlinear::kzk::complex_parabolic_diffraction` now
  applies its diagonal parabolic spectral multiplier through
  `moirai_parallel::enumerate_mut_with::<Adaptive>` over the standard-layout
  scratch buffer. `forward::nonlinear::kzk::nonlinearity` now computes the
  buffered nonlinear `delta` and applies the real-pressure update through
  `moirai_parallel::for_each_chunk_mut_enumerated_with::<Adaptive>` over
  standard-layout `i` slabs, preserving operator isolation by completing the
  delta pass before mutating pressure. Evidence tier: source audit plus
  compile-time and focused empirical validation; scoped `rg` finds no direct
  Rayon or ndarray-parallel token under
  `crates/kwavers-solver/src/forward/nonlinear/kzk`, `cargo fmt -p
  kwavers-solver --check` passes, `cargo check -p kwavers-solver` passes,
  `cargo clippy -p kwavers-solver --lib -- -D warnings` passes, and focused
  `cargo nextest run -p kwavers-solver -E "test(kzk) or test(absorption) or
  test(diffraction) or test(nonlinear)"` passes 204/204. Residual: no direct
  Rayon or ndarray-parallel source hits remain under the KZK subtree.
- **kwavers-solver plugin Rayon placeholder seam - RESOLVED [patch].**
  `plugin::execution::ParallelStrategy` no longer accepts a
  `rayon::ThreadPool` that it discards while executing plugins sequentially.
  The strategy docs now make the ordering constraint explicit: shared mutable
  field access requires a real read/compute/write phase split before plugin
  parallelism can be implemented. Evidence tier: static source audit plus
  focused empirical tests; `cargo check -p kwavers-solver`, `cargo clippy -p
  kwavers-solver --lib -- -D warnings`, and `cargo nextest run -p
  kwavers-solver plugin --status-level fail --no-fail-fast` pass. The
  all-targets clippy gate still fails on unrelated pre-existing test/doc lints
  outside this slice.
- **kwavers-solver time-reversal direct Rayon edge - RESOLVED [patch].**
  `inverse::time_reversal::reconstruction` now normalizes reconstruction
  volumes through the shared `workspace::inplace_ops::apply_inplace`
  Moirai-backed traversal instead of ndarray/Rayon `par_mapv_inplace`.
  Evidence tier: static source audit plus compile-time/lint validation and a
  focused value-semantic test; `cargo clippy -p kwavers-solver --lib -- -D
  warnings` passes, `cargo nextest run -p kwavers-solver time_reversal
  --status-level fail --no-fail-fast` passes 9/9, and the time-reversal Rayon
  audit returns no hits. Residual package gate: all-targets clippy still fails
  on unrelated pre-existing test/doc lints outside this slice.
- **kwavers-solver monolithic residual direct Rayon edge - RESOLVED [patch].**
  `multiphysics::monolithic::residual` now scales pressure, light-fluence, and
  temperature Laplacian rate buffers through the shared
  `workspace::inplace_ops::scale_inplace` Moirai-backed traversal instead of
  ndarray/Rayon `par_mapv_inplace`. Evidence tier: static source audit plus
  compile-time/lint validation and a focused value-semantic test; `cargo check
  -p kwavers-solver` passes, `cargo clippy -p kwavers-solver --lib -- -D
  warnings` passes, `cargo nextest run -p kwavers-solver monolithic
  --status-level fail --no-fail-fast` passes 30/30, and the monolithic
  residual Rayon audit returns no hits.
- **kwavers-solver monolithic coupler direct Rayon edge - RESOLVED [patch].**
  `multiphysics::monolithic::coupler` now builds the Newton GMRES right-hand
  side by assigning `F(u)` into the reusable RHS scratch buffer and applying
  the required sign inversion through `workspace::inplace_ops::scale_inplace`
  instead of ndarray/Rayon `par_mapv_inplace`. Evidence tier: static source
  audit plus compile-time/lint validation and focused value-semantic tests;
  `cargo check -p kwavers-solver` passes, `cargo clippy -p kwavers-solver
  --lib -- -D warnings` passes, `cargo nextest run -p kwavers-solver
  monolithic --status-level fail --no-fail-fast` passes 30/30, and the
  monolithic coupler Rayon audit returns no hits.
- **kwavers-solver AMR direct Rayon edge - RESOLVED [patch].**
  `utilities::amr` now routes wavelet/physics error normalization through
  `workspace::inplace_ops::scale_inplace`, wavelet coefficient thresholding
  through `workspace::inplace_ops::apply_inplace`, and refinement marker
  initialization through `moirai_parallel::enumerate_mut_with` instead of
  direct ndarray/Rayon parallel dispatch. Evidence tier: static source audit
  plus focused value-semantic tests; `cargo fmt -p kwavers-solver --check`
  passes, `cargo nextest run -p kwavers-solver amr --status-level fail
  --no-fail-fast` passes 11/11, and the AMR direct Rayon/ndarray-parallel
  audit returns no hits.
- **kwavers-solver SWE displacement-magnitude direct Rayon edge - RESOLVED
  [patch].** `ElasticWaveField::displacement_magnitude` now routes the final
  square-root transform through `workspace::inplace_ops::apply_inplace` instead
  of ndarray/Rayon `par_mapv_inplace`. Evidence tier: static source audit plus
  focused value-semantic tests; `cargo fmt -p kwavers-solver --check` passes,
  `cargo nextest run -p kwavers-solver displacement_magnitude --status-level
  fail --no-fail-fast` passes 3/3, and the SWE types direct
  Rayon/ndarray-parallel audit returns no hits.
- **kwavers-solver SWE PML boundary direct Rayon edge - RESOLVED [patch].**
  `ElasticSwePMLBoundary` now routes attenuation-field construction and mask
  generation through shared Moirai-backed indexed 3-D traversal, and velocity
  damping through Moirai triple mutable chunk traversal instead of direct
  ndarray/Rayon parallel dispatch. Evidence tier: static source audit plus
  focused value-semantic tests; `cargo fmt -p kwavers-solver --check` passes,
  `cargo nextest run -p kwavers-solver pml --status-level fail --no-fail-fast`
  passes 45/45, and the SWE boundary direct Rayon/ndarray-parallel audit
  returns no hits.
- **Diagnostics direct Rayon edge — RESOLVED [patch].**
  `kwavers-diagnostics::reconstruction::real_time_sirt::pipeline` now routes
  row-norm cache construction and separable smoothing through
  `moirai-parallel`; diagnostics no longer depends directly on `rayon` or
  enables ndarray's `rayon` feature. Evidence tier: static source/dependency
  audit plus focused empirical tests; `rg` found no direct Rayon/thread-pool
  hits in `crates/kwavers-diagnostics/src` or its manifest, `cargo check -p
  kwavers-diagnostics` passes, `cargo nextest run -p kwavers-diagnostics
  real_time_sirt --status-level fail --no-fail-fast` passes 14/14, and `cargo
  tree -p kwavers-diagnostics --depth 1` shows no direct `rayon`.
- **Diagnostics sound-speed-shift direct Rayon edge - RESOLVED [patch].**
  `operator::algebra` now uses Moirai indexed mutable dispatch and
  `fold_reduce_with` for matrix-vector, transpose, and normal-diagonal
  operations. Evidence tier: static source audit plus focused empirical tests;
  `cargo check -p kwavers-diagnostics` passes and `cargo nextest run -p
  kwavers-diagnostics sound_speed_shift --status-level fail --no-fail-fast`
  passes 34/34.
- **Diagnostics transcranial UST direct Rayon edge - RESOLVED [patch].**
  Finite-frequency sensitivity rows and attenuation/traveltime ray integrals
  in `reconstruction::transcranial_ust::sensitivity` now dispatch through
  `moirai-parallel`. Evidence tier: static source audit plus focused empirical
  tests; `cargo check -p kwavers-diagnostics` passes and `cargo nextest run -p
  kwavers-diagnostics transcranial_ust --status-level fail --no-fail-fast`
  passes 7/7.
- **Imaging/simulation direct Rayon edge — RESOLVED in current audit
  [patch].** `rg` found no direct Rayon/thread-pool source hits under
  `crates/kwavers-imaging` or `crates/kwavers-simulation`; their manifests also
  no longer list a direct `rayon` dependency or ndarray `rayon` feature.
- **kwavers-solver manifest-level Rayon dependency — OPEN [patch].**
  `crates/kwavers-solver/Cargo.toml` still enables ndarray's `rayon` feature
  and depends directly on `rayon` because non-workspace, non-time-integration,
  non-time-domain solver modules still contain direct Rayon/ndarray-parallel
  call sites.
- **kwavers-grid direct ndarray Rayon edge — RESOLVED [patch].**
  `kwavers-grid` no longer enables ndarray's `rayon` feature for the
  Laplacian operator path. Second-order interior writes dispatch through
  `moirai-parallel`, and a nonstandard-output-view regression keeps the
  sequential view path value-semantic. Evidence tier: compile-time validation
  plus focused package tests; `cargo check -p kwavers-grid` and `cargo
  nextest run -p kwavers-grid test_laplacian` pass 3/3.
- **Gaia straight-ray ownership — RESOLVED [patch].** Gaia owns the reusable
  `Ray<f64>` primitive used by the liver theranostic straight-ray rasterizer.
  Kwavers retains only voxel path-length accumulation for tomography. Evidence
  tier: type-level unit-direction invariant in Gaia plus value-semantic tests;
  `cargo nextest run -p gaia ray` passes 8/8.
- **kwavers-imaging multimodality ndarray volume surface — RESOLVED [patch].**
  `kwavers-imaging::multimodality_fusion` now stores medical image volumes and
  fusion outputs as `leto::Array3<f64>` and calls local Ritk registration
  directly with Leto arrays. The rejected ndarray-to-Leto helper was removed.
  Evidence tier: compile-time validation plus focused package tests; `cargo
  check -p kwavers-imaging` passes and `cargo nextest run -p kwavers-imaging
  multimodality` passes 9/9.
- **kwavers fusion/workflow/photoacoustic ndarray volume surfaces — RESOLVED
  [patch].** `kwavers-physics::acoustics::imaging::fusion`, diagnostics
  workflow products, fUS atlas registration volumes, photoacoustic result
  volumes, and photoacoustic simulation fluence/pressure/reconstruction
  snapshots now use `leto::Array3<f64>` directly in the migrated path. The
  rejected ndarray-to-Leto helper was not reintroduced. Evidence tier:
  compile-time validation plus focused empirical tests; `cargo check -p
  kwavers --example liver_theranostic_reconstruction --features nifti` passes,
  `cargo nextest run -p kwavers-physics fusion` passes 103/103, `cargo nextest
  run -p kwavers-diagnostics workflows functional_ultrasound atlas` passes
  80/80, and `cargo nextest run -p kwavers-simulation photoacoustic` passes
  27/27.
- **kwavers-solver linear elastography producer boundary — RESOLVED [patch].**
  Direct, directional, and LFE shear-wave-speed maps now allocate
  `leto::Array3<f64>` directly. Elastography smoothing and boundary fill share
  one crate-local 3-D volume trait implementation for ndarray and Leto arrays,
  avoiding duplicated filter logic. Evidence tier: compile-time validation plus
  focused empirical tests; `cargo check -p kwavers-solver`, `cargo nextest run
  -p kwavers-solver elastography` (53/53), and `cargo check -p kwavers
  --example liver_theranostic_reconstruction --features nifti` pass.
- **Photoacoustic reconstructor producer boundary — RESOLVED [patch].**
  The selected solver universal back-projection producer now emits
  `leto::Array3<f64>` directly for the simulation photoacoustic path, and the
  caller-side `Into::into` conversion is removed. The generic ndarray
  `Reconstructor` contract remains for unmigrated reconstructor consumers.
  Evidence tier: compile-time validation plus focused empirical tests; `cargo
  check -p kwavers-solver`, `cargo check -p kwavers-simulation`, `cargo
  nextest run -p kwavers-solver photoacoustic` (9/9), and `cargo nextest run
  -p kwavers-simulation photoacoustic` (27/27) pass.
- **Optical diffusion fluence producer boundary — RESOLVED [patch].**
  The selected optical diffusion PCG producer now has one internal generic
  volume kernel shared by ndarray and Leto entry points. The photoacoustic
  optics path allocates its source as `leto::Array3<f64>` and calls
  `DiffusionSolver::solve_leto`, removing the caller-side `Ok(fluence.into())`
  conversion. Evidence tier: compile-time validation plus bitwise differential
  and focused empirical tests; `cargo check -p kwavers-solver`, `cargo check
  -p kwavers-simulation`, `cargo nextest run -p kwavers-solver diffusion`
  (13/13, including `leto_solver_matches_ndarray_solver_bitwise`), and `cargo
  nextest run -p kwavers-simulation photoacoustic` (27/27) pass.

### Gate residuals (2026-06-30)

- **PHYS-CLIPPY all-target mechanical lint layer — RESOLVED [patch].** The
  current `kwavers-physics --all-targets` clippy blockers were mechanical lint
  debt in tests/local helpers, not physics changes: manual range predicates,
  `items_after_test_module`, runtime assertions on compile-time constants,
  default-then-reassign test setup, `clone()` on `Copy` states, a dense helper
  tuple type, and one identity-index expression. The fix preserves value
  assertions and moves constant invariants into `const` assertions. Evidence
  tier: static analysis plus value-semantic tests; `cargo clippy -p
  kwavers-physics --all-targets -- -D warnings` passes and `cargo nextest run
  -p kwavers-physics` passes 1665/1665 with 1 skipped.

### Book script physics ownership residuals (2026-06-30)

- **Cavitation passive-map binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical cavitation facade no longer owns receiver-array
  PSD integration or passive-map emission-energy PyO3 wrappers directly. Those
  responsibilities are isolated in `cavitation/passive_map.rs`; the facade is
  now module declarations plus registered-name re-exports only. Evidence tier:
  warning-clean compile-time validation with and without the `gpu` feature plus
  `kwavers-python` nextest regression coverage.
- **Cavitation chirp/shielding binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical cavitation facade no longer owns
  frequency-swept engagement, chirped expansion, residual clearance,
  residual dissolution, optimal-frequency search, staged sonication,
  shielding trace simulation, or shielding-control comparison PyO3 wrappers
  directly. Those responsibilities are isolated in `cavitation/chirp.rs`; the
  facade keeps the registered Python names through re-exports. Evidence tier:
  warning-clean compile-time validation with and without the `gpu` feature plus
  `kwavers-python` nextest regression coverage.
- **Cavitation monitor/control binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical cavitation facade no longer owns monitor traces,
  simulated population monitor traces, closed-loop sonication, raster pulsing,
  therapeutic-window classification, inertial-fraction onset, per-spot dose
  grids, or controller-pressure PyO3 wrappers directly. Those responsibilities
  are isolated in `cavitation/monitor.rs`; the facade keeps the registered
  Python names through re-exports. Evidence tier: warning-clean compile-time
  validation with and without the `gpu` feature plus `kwavers-python` nextest
  regression coverage.
- **Cavitation spectrum binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical cavitation facade no longer owns bubble PSD,
  Hann-windowed PSD, Keller-Miksis PCD spectrum/controller trace, acoustic
  emission pressure, ensemble superposition, emission-band decomposition,
  normalized spectrum, cumulative dose, or passive-dose fixture PyO3 wrappers
  directly. Those responsibilities are isolated in `cavitation/spectrum.rs`;
  the facade keeps the registered Python names through re-exports. Verification
  also repaired current-tree bubble-dynamics compile blockers: an invalid
  `AdaptiveBubbleModel` self re-export and missing `BubbleField: Debug`.
  Evidence tier: warning-clean compile-time validation with and without the
  `gpu` feature plus `kwavers-python` nextest regression coverage.
- **Cavitation emission binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical cavitation facade no longer owns free/coated
  bubble emission, population emission, population pressure sweep, focal-volume
  emission spectrum, or focal-volume pressure sweep PyO3 wrappers directly.
  Those responsibilities are isolated in `cavitation/emission.rs`; the facade
  keeps the registered Python names through re-exports. Evidence tier:
  warning-clean compile-time validation with and without the `gpu` feature plus
  `kwavers-python` nextest regression coverage.
- **Cavitation passive-receive binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical cavitation facade no longer owns
  receiver-channel PSD propagation, channel PSD integration, passive
  point-source RF synthesis, or Van Cittert-Zernike coherence PyO3 wrappers
  directly. Those responsibilities are isolated in
  `cavitation/passive_receive.rs`; the facade keeps the registered Python names
  through re-exports. Evidence tier: warning-clean compile-time validation with
  and without the `gpu` feature plus `kwavers-python` nextest regression
  coverage.
- **Cavitation lesion binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical cavitation facade no longer owns fractionation
  backscatter/impedance, boiling-lesion sizing/time profiles, lacuna void
  fraction, histotripsy lesion-radius conversion, or inertial cavitation dose
  PyO3 wrappers directly. Those responsibilities are isolated in
  `cavitation/lesion.rs`; the facade keeps the registered Python names through
  re-exports. Evidence tier: warning-clean compile-time validation with and
  without the `gpu` feature plus `kwavers-python` nextest regression coverage.
- **Cavitation therapy binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical cavitation facade no longer owns sonication
  scheduling, delivery fractions, interface-pressure scalars, lesion
  susceptibility, histotripsy dose-response, focal-mask checks,
  measured-emission scaling, delivered-progress, or cloud-erosion validation
  PyO3 wrappers directly. Those responsibilities are isolated in
  `cavitation/therapy.rs`; the facade keeps the registered Python names through
  re-exports. Evidence tier: warning-clean compile-time validation with and
  without the `gpu` feature plus `kwavers-python` nextest regression coverage.
- **Cavitation medium binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical cavitation facade no longer owns
  Epstein-Plesset dissolution, shelled dissolution, Wood sound speed, or
  Commander-Prosperetti attenuation/phase-velocity PyO3 wrappers directly.
  Those responsibilities are isolated in `cavitation/medium.rs`; the facade
  keeps the registered Python names through re-exports. Evidence tier:
  warning-clean compile-time validation with and without the `gpu` feature plus
  `kwavers-python` nextest regression coverage.
- **Cavitation single-bubble binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical cavitation facade no longer owns Minnaert
  resonance/radius, surface-tension corrected resonance, Blake threshold, or
  Rayleigh collapse-time PyO3 wrappers directly. Those responsibilities are
  isolated in `cavitation/bubble.rs`; the facade keeps the registered Python
  names through re-exports. Evidence tier: warning-clean compile-time
  validation with and without the `gpu` feature plus `kwavers-python` nextest
  regression coverage.
- **Cavitation probability binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical cavitation facade no longer owns
  intrinsic-threshold probability, frequency-dependent threshold, cumulative
  probability, or PRF efficacy PyO3 wrappers directly. Those responsibilities
  are isolated in `cavitation/probability.rs`; the facade keeps the registered
  Python names through re-exports. Evidence tier: warning-clean compile-time
  validation with and without the `gpu` feature plus `kwavers-python` nextest
  regression coverage.
- **Neuromodulation binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical neuromodulation facade no longer owns
  Hodgkin-Huxley, NICE/SONIC response, bilayer curve, threshold-search, ITRUSST
  safety, or pulse-train dosimetry PyO3 wrappers directly. Those
  responsibilities are isolated in `neuromodulation/response.rs`,
  `neuromodulation/bilayer.rs`, `neuromodulation/threshold.rs`, and
  `neuromodulation/safety.rs`; the facade keeps the registered Python names
  through re-exports. Evidence tier: warning-clean compile-time validation with
  and without the `gpu` feature plus `kwavers-python` nextest regression
  coverage.
- **Inverse-problem binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical inverse facade no longer owns Helmholtz, SVD,
  L-curve, deconvolution fixture, Born inversion, convergence, or
  parameter-selection PyO3 wrappers directly. Those responsibilities are
  isolated in `inverse/operators.rs`, `inverse/reconstruction.rs`,
  `inverse/convergence.rs`, and `inverse/selection.rs`, with shared array
  conversion in `inverse/arrays.rs` and seismic imaging already isolated in
  `inverse/seismic.rs`; the facade keeps the registered Python names through
  re-exports. Evidence tier: warning-clean compile-time validation with and
  without the `gpu` feature plus `kwavers-python` nextest regression coverage.
- **RTM analytical binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical RTM facade no longer owns focused-beam,
  back-propagation, imaging/fusion, temporal modulation, or standing-wave PyO3
  wrappers directly. Those responsibilities are isolated in `rtm/fields.rs`,
  `rtm/imaging.rs`, and `rtm/standing_wave.rs`, with shared array conversion in
  `rtm/arrays.rs`; the facade keeps the registered Python names through
  re-exports. Evidence tier: warning-clean compile-time validation with and
  without the `gpu` feature plus `kwavers-python` nextest regression coverage.
- **Skull analytical binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical skull facade no longer owns insertion-loss,
  phase-screen, Strehl, Hounsfield-conversion, thermal-rise, or layered
  transmission PyO3 wrappers directly. Those responsibilities are isolated in
  `skull/aberration.rs`, `skull/ct.rs`, `skull/thermal.rs`, and
  `skull/transmission.rs`; the facade keeps the registered Python names through
  re-exports. Evidence tier: warning-clean compile-time validation with and
  without the `gpu` feature plus `kwavers-python` nextest regression coverage.
- **Sonogenetics binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical sonogenetics facade no longer owns
  mechanosensitive-channel activation, radiation-force/streaming mechanics, or
  ISPTA dosimetry PyO3 wrappers directly. Those responsibilities are isolated
  in `sonogenetics/activation.rs`, `sonogenetics/mechanics.rs`, and
  `sonogenetics/dosimetry.rs`; the facade keeps the registered Python names
  through re-exports. Evidence tier: warning-clean compile-time validation with
  and without the `gpu` feature plus `kwavers-python` nextest regression
  coverage.
- **MEMS CMUT/PMUT binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical MEMS facade no longer owns clamped-plate, CMUT,
  PMUT, or therapy/IVUS comparison PyO3 wrappers directly. Those
  responsibilities are isolated in `mems/plate.rs`, `mems/cmut.rs`,
  `mems/pmut.rs`, and `mems/comparison.rs`, with binding-only validation
  helpers in `mems/helpers.rs`; the facade keeps the registered Python names
  through re-exports. Evidence tier: warning-clean compile-time validation with
  and without the `gpu` feature plus `kwavers-python` nextest regression
  coverage.
- **Acousto-optics binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical acousto-optics facade no longer owns
  Klein-Cook/Raman-Nath/Bragg regime parameters, angle/frequency geometry, or
  diffraction-order solver PyO3 wrappers directly. Those responsibilities are
  isolated in `acousto_optics/regime.rs`, `acousto_optics/geometry.rs`, and
  `acousto_optics/orders.rs`; the facade keeps the registered Python names
  through re-exports. Evidence tier: warning-clean compile-time validation with
  and without the `gpu` feature plus `kwavers-python` nextest regression
  coverage.
- **Tissue analytical binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical tissue facade no longer owns temperature-dependent
  water properties, attenuation/dispersion, or tissue property lookup PyO3
  wrappers directly. Those responsibilities are isolated in `tissue/water.rs`,
  `tissue/attenuation.rs`, and `tissue/properties.rs`; the facade keeps the
  registered Python names through re-exports. Evidence tier: warning-clean
  compile-time validation with and without the `gpu` feature plus
  `kwavers-python` nextest regression coverage.
- **Statistics validation binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical statistics facade no longer owns Pearson,
  phase-sensitivity, RMSE, or PSNR PyO3 wrappers directly. Those
  responsibilities are isolated in `statistics/correlation.rs` and
  `statistics/metrics.rs`, with shared NumPy slice conversion in
  `statistics/arrays.rs`; the facade keeps the registered Python names through
  re-exports. Evidence tier: warning-clean compile-time validation with and
  without the `gpu` feature plus `kwavers-python` nextest regression coverage.
- **BBB and CEUS binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical BBB facade no longer owns permeability, damage,
  closure, CEUS backscatter, or CEUS display PyO3 wrappers directly. Those
  responsibilities are isolated in `bbb/permeability.rs` and `bbb/ceus.rs`;
  the facade keeps the registered Python names through re-exports. Evidence
  tier: warning-clean compile-time validation with and without the `gpu`
  feature plus `kwavers-python` nextest regression coverage.
- **Photoacoustics binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical photoacoustics facade no longer owns spectral
  absorption/Gruneisen, source/signal, axial-resolution, or spectroscopic
  unmixing PyO3 wrappers directly. Those responsibilities are isolated in
  `photoacoustics/spectrum.rs`, `photoacoustics/source.rs`, and
  `photoacoustics/reconstruction.rs`; the facade keeps the registered Python
  names through re-exports, and the sO2 sweep wrapper avoids the prior
  flatten/rebuild transient allocation before NumPy conversion. Evidence tier:
  warning-clean compile-time validation with and without the `gpu` feature plus
  `kwavers-python` nextest regression coverage.
- **Elastography thermal-strain binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical elastography facade no longer owns the
  thermal-strain RF fixture, combined coefficient, or reconstruction PyO3
  wrappers directly. Those responsibilities are isolated in
  `elastography/thermal_strain.rs`; the facade keeps the registered Python
  names through re-exports. Evidence tier: warning-clean compile-time
  validation with and without the `gpu` feature plus `kwavers-python` nextest
  regression coverage.
- **Safety Arrhenius damage binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical safety facade no longer owns Arrhenius damage,
  cumulative damage, thermal kill probability, steady thermal kill probability,
  or combined mechanical/thermal kill PyO3 wrappers directly. Those
  responsibilities are isolated in `safety/damage.rs`; the facade now owns
  module topology plus FDA scalar-limit wrappers. Evidence tier: warning-clean
  compile-time validation with and without the `gpu` feature plus
  `kwavers-python` nextest regression coverage.
- **Safety thermal-index binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical safety facade no longer owns soft-tissue, bone,
  cranial thermal index, CEM43 cumulative dose, or closed-loop CEM43 fixture
  PyO3 wrappers directly. Those responsibilities are isolated in
  `safety/thermal.rs`; the facade keeps the registered Python names through
  re-exports. Evidence tier: warning-clean compile-time validation with and
  without the `gpu` feature plus `kwavers-python` nextest regression coverage.
- **Safety mechanical-index binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical safety facade no longer owns scalar MI, field MI,
  frequency-sweep MI, or MI cavitation-risk PyO3 wrappers directly. Those
  responsibilities are isolated in `safety/mechanical.rs`; the facade keeps the
  registered Python names through re-exports. Evidence tier: warning-clean
  compile-time validation with and without the `gpu` feature plus
  `kwavers-python` nextest regression coverage.
- **Thermal acoustic binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical thermal facade no longer owns HIFU focal gain,
  Gaussian power deposition, depth intensity/power deposition,
  pressure/intensity conversion, or acoustic heat-source PyO3 wrappers directly.
  Those responsibilities are isolated in `thermal/acoustic.rs`; the facade
  keeps the registered Python names through re-exports. Evidence tier:
  warning-clean compile-time validation with and without the `gpu` feature plus
  `kwavers-python` nextest regression coverage.
- **Inverse seismic binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical inverse facade no longer owns eikonal
  traveltime, Kirchhoff point-scatterer imaging, paired-index validation, or
  Ricker trace synthesis directly. Those responsibilities are isolated in
  `inverse/seismic.rs`; the facade keeps the registered Python names through
  re-exports. Evidence tier: warning-clean compile-time validation with and
  without the `gpu` feature plus `kwavers-python` nextest regression coverage.
- **Imaging IVUS B-mode and metrics binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical imaging facade no longer owns polar B-mode RF,
  scan conversion, complete B-mode image, or Chapter 30 metric PyO3 wrappers
  directly. Those responsibilities are isolated in `imaging/bmode.rs` and
  `imaging/metrics.rs`; the facade now owns module topology and re-exports
  only. Evidence tier: warning-clean compile-time validation with and without
  the `gpu` feature plus `kwavers-python` nextest regression coverage.
- **Imaging IVUS therapy binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical imaging facade no longer owns IVUS therapy
  pressure, microbubble delivery, response, or aggregate therapy-field PyO3
  wrappers directly. Those responsibilities are isolated in `imaging/therapy.rs`;
  the facade keeps the registered Python names through re-exports. Evidence
  tier: warning-clean compile-time validation with and without the `gpu` feature
  plus `kwavers-python` nextest regression coverage.
- **Imaging IVUS phantom binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical imaging facade no longer owns deterministic IVUS
  vessel-phantom dictionary materialization or its square-array helper directly.
  Those responsibilities are isolated in `imaging/phantom.rs`; the facade keeps
  the registered Python name through re-export. Evidence tier: warning-clean
  compile-time validation with and without the `gpu` feature plus
  `kwavers-python` nextest regression coverage.
- **Imaging PSF binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical imaging facade no longer owns lateral/axial PSF,
  plane-wave compounding PSF, or lateral-resolution PyO3 wrappers directly.
  Those wrappers are isolated in `imaging/psf.rs`; the facade keeps the
  registered Python names through re-exports. Evidence tier: warning-clean
  compile-time validation with and without the `gpu` feature plus
  `kwavers-python` nextest regression coverage.
- **Imaging pulse-echo binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical imaging facade no longer owns synthetic receive
  RF, B-mode envelope, fixed-reference log compression, or delta B-mode PyO3
  wrappers directly. Those wrappers are isolated in `imaging/pulse_echo.rs`;
  the facade keeps the registered Python names through re-exports. Evidence
  tier: warning-clean compile-time validation with and without the `gpu` feature
  plus `kwavers-python` nextest regression coverage.
- **Imaging Doppler binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical imaging facade no longer owns Doppler frequency
  shift, contrast-agent Doppler spectrum, or continuous-wave/vector-flow PyO3
  wrappers directly. Those wrappers are isolated in `imaging/doppler.rs`; the
  facade keeps the registered Python names through re-exports. Evidence tier:
  compile-time validation with and without the `gpu` feature plus
  `kwavers-python` nextest regression coverage.
- **Transducer beam binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical transducer facade no longer owns 2-D focus-delay,
  complex beam-pattern, far-field beam-magnitude, or 2-D beam-magnitude PyO3
  wrappers directly. Those wrappers are isolated in `transducer/beam.rs`; the
  facade now owns module topology and re-exports only. Evidence tier:
  compile-time validation with and without the `gpu` feature plus
  `kwavers-python` nextest regression coverage.
- **Transducer basic binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical transducer facade no longer owns circular-piston
  directivity, linear-array factor, grating-lobe, apodization, or on-axis
  pressure PyO3 wrappers directly. Those wrappers are isolated in
  `transducer/basic.rs`; the facade keeps the registered Python names through
  re-exports. Evidence tier: compile-time validation with and without the
  `gpu` feature plus `kwavers-python` nextest regression coverage.
- **Transducer multi-focus binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical transducer facade no longer owns multi-focus
  delay-law and phase-conjugated field-magnitude PyO3 wrappers directly. Those
  wrappers are isolated in `transducer/multi_focus.rs`; the facade keeps the
  registered Python names through re-exports. Evidence tier: compile-time
  validation with and without the `gpu` feature plus `kwavers-python` nextest
  regression coverage.
- **Transducer aperture binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical transducer facade no longer owns linear-array
  positioning, focused-bowl element geometry, 3-D focus delays, steered aperture
  pressure, or focused-bowl pressure-profile PyO3 wrappers directly. Those
  wrappers are isolated in `transducer/aperture.rs`; the facade keeps the
  registered Python names through re-exports. Evidence tier: compile-time
  validation with and without the `gpu` feature plus `kwavers-python` nextest
  regression coverage.
- **Transducer interpolation binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical transducer facade no longer owns BLI stencil and
  interpolation-error-curve PyO3 wrappers directly. Those wrappers are isolated
  in `transducer/interpolation.rs`; the facade keeps the registered Python
  names through re-exports. Evidence tier: compile-time validation with and
  without the `gpu` feature plus `kwavers-python` nextest regression coverage.
- **Transducer steering binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical transducer facade no longer owns natural-focus
  steering, sparse-aperture, grating-lobe, safe-steering, and electronic
  steering-efficiency PyO3 wrappers directly. Those wrappers are isolated in
  `transducer/steering.rs`; the facade keeps the registered Python names
  through re-exports. Evidence tier: compile-time validation with and without
  the `gpu` feature plus `kwavers-python` nextest regression coverage.
- **Transducer binding topology — RESOLVED [patch].** The
  `kwavers-python` analytical transducer facade no longer owns SOAP/
  optoacoustic wrappers and static acoustic-lens material wrappers in the same
  monolithic file. Those bounded wrapper families are split into
  `transducer/optoacoustic.rs` and `transducer/lens.rs`; the facade keeps the
  registered Python names through re-exports. Evidence tier: compile-time
  validation with and without the `gpu` feature plus `kwavers-python` nextest
  regression coverage.
- **GPU PSTD session source topology — RESOLVED [patch].** The
  `GpuPstdSession` facade no longer owns source/sensor index construction or
  velocity-signal packing; `session/source.rs` owns that responsibility. Cached
  scan-line execution also no longer allocates empty pressure-source vectors for
  unused solver inputs. Evidence tier: compile-time validation with and without
  the `gpu` feature plus `kwavers-python` nextest regression coverage.
- **GPU PSTD session constructor topology — RESOLVED [patch].** The
  `kwavers-python` GPU PSTD session constructor no longer owns absorption-kernel
  generation and CPML array materialization in the same source file. Those
  responsibilities are split into `session/absorption.rs` and `session/pml.rs`
  behind the unchanged `GpuPstdSession` facade. Evidence tier: compile-time
  validation with and without the `gpu` feature plus `kwavers-python` nextest
  regression coverage.
- **THERAPY chapter guards — RESOLVED [patch].** The focused therapy chapter
  regression now resolves `docs/book` from the repository root under the current
  crate layout, and the active Chapter 31 clinical-device script no longer emits
  vendor-style source labels in the guarded figure text. Evidence tier: focused
  pytest plus source-token scan over the guarded active artifacts.
- **BOOK-CH24 CEUS backscatter display — RESOLVED [patch].** The Chapter 24
  BBB-LIFU CEUS panel no longer computes peak-normalised dB display values or
  the optimal concentration marker in Python from a raw signal array. Rust
  `ceus_backscatter_display` owns finite-input validation, raw signal
  generation, declared-floor dB normalization, and peak sample selection; Python
  only plots returned arrays and metadata. Evidence tier: value-semantic Rust
  nextest coverage against the underlying CEUS signal model; focused PyO3
  source/value tests; editable `maturin` rebuild; Miniforge py-compile;
  touched-crate `cargo check`; and touched-path diff hygiene.
- **BOOK-CH30 IVUS therapy fields — RESOLVED [patch].** The Chapter 30
  intravascular-ultrasound therapy panel no longer orchestrates pressure-field
  generation and therapy-response assembly from split Rust helpers in Python.
  Rust `ivus_therapy_fields` owns finite-input pressure/response composition;
  Python only reshapes returned fields for plotting and metrics. Evidence tier:
  value-semantic Rust nextest coverage against the lower-level pressure and
  response helpers; focused PyO3 source/value tests; editable `maturin`
  rebuild; Miniforge py-compile; touched-crate `cargo check`; and touched-path
  diff hygiene.
- **BOOK-CH30 IVUS metrics — RESOLVED [patch].** The Chapter 30
  intravascular-ultrasound script no longer computes wavelength, lumen/plaque
  area, masked B-mode mean, or therapy summary metrics with Python-side scalar
  formulas. Rust `ivus_chapter_metrics` owns finite-input validation, grid
  spacing, mask areas, and masked means; Python only serializes the returned
  metric dictionary with figure paths. Evidence tier: value-semantic Rust
  nextest coverage for wavelengths, areas, masked means, therapy scalar
  forwarding, and empty-mask rejection; focused PyO3 source/value tests;
  editable `maturin` rebuild; Miniforge py-compile; touched-crate `cargo
  check`; and touched-path diff hygiene.
- **BOOK-CH30 IVUS B-mode image — RESOLVED [patch].** The Chapter 30
  intravascular-ultrasound B-mode panel no longer assembles polar RF,
  RF-column Hilbert envelopes, envelope clamping, fixed-reference log
  compression, normalized display mapping, or Cartesian scan conversion in
  Python. Rust `ivus_bmode_image` owns the complete finite-input B-mode image
  fixture and reuses Rust-owned RF and scan-conversion helpers; Python only
  reshapes returned arrays for plotting and metrics. Evidence tier:
  value-semantic Rust nextest coverage for output shape, display bounds, scan
  conversion consistency, and invalid floors; focused PyO3 source/value tests;
  editable `maturin` rebuild; Miniforge py-compile; touched-crate `cargo
  check`; and touched-path diff hygiene.
- **BOOK-CH30 IVUS therapy response — RESOLVED [patch].** The Chapter 30
  intravascular-ultrasound therapy panel no longer assembles intensity,
  effective attenuation, absorbed power, adiabatic temperature rise, delivery
  masks, mechanical index, or target/off-target deposition ratio in Python.
  Rust `ivus_therapy_response` owns finite-input validation and reuses the
  Rust-owned intensity, temperature-rise, microbubble delivery, and safety
  kernels; Python only reshapes returned fields for plotting and metrics.
  Evidence tier: value-semantic Rust nextest coverage for closed-form
  intensity/thermal/delivery/MI/ratio outputs and invalid target masks;
  focused PyO3 source/value tests; editable `maturin` rebuild; Miniforge
  py-compile; touched-crate `cargo check`; and touched-path diff hygiene.
- **BOOK-CH30 IVUS scan conversion — RESOLVED [patch].** The Chapter 30
  intravascular-ultrasound B-mode panel no longer projects polar image samples
  onto the Cartesian phantom grid with Python-side nearest-bin radius/theta
  indexing. Rust `ivus_scan_convert` owns finite-input validation, radial
  clipping, periodic theta wrapping, and row-major polar lookup; Python only
  reshapes returned Cartesian image samples for plotting. Evidence tier:
  value-semantic Rust nextest coverage for radial bounds and theta wrapping;
  focused PyO3 source/value tests; editable `maturin` rebuild; Miniforge
  py-compile; touched-crate `cargo check`; and touched-path diff hygiene.
- **BOOK-CH30 IVUS polar RF — RESOLVED [patch].** The Chapter 30
  intravascular-ultrasound B-mode panel no longer computes polar phantom
  sampling, two-way attenuation, or catheter-ring echo algebra in Python. Rust
  `ivus_polar_bmode_rf` owns finite-input validation, nearest-grid sampling,
  attenuation, and ring-echo construction; Python only reshapes returned RF
  samples before calling the existing Rust envelope/log-compression kernels.
  Evidence tier: value-semantic Rust nextest coverage for the attenuation/ring
  formula and invalid inputs; focused PyO3 source/value tests; editable
  `maturin` rebuild; Miniforge py-compile; touched-crate `cargo check`; and
  touched-path diff hygiene.
- **BOOK-CH30 IVUS delivery fraction — RESOLVED [patch].** The Chapter 30
  intravascular-ultrasound therapy map no longer computes microbubble delivery
  with Python-side acoustic-radiation-force, radial-band, normalization, or
  exponential-delivery algebra. Rust `ivus_microbubble_delivery_fraction` owns
  finite-input validation and the acoustic-force-to-delivery mapping; Python
  only reshapes returned delivery samples for plotting and summary metrics.
  Evidence tier: value-semantic Rust nextest coverage for wall/target weighting
  and invalid inputs; focused PyO3 source/value tests; editable `maturin`
  rebuild; Miniforge py-compile; touched-crate `cargo check`; and touched-path
  diff hygiene.
- **BOOK-CH30 IVUS therapy pressure field — RESOLVED [patch].** The Chapter 30
  intravascular-ultrasound therapy map no longer computes its sector-focused
  pressure field with Python-side angular Gaussian and radial exponential
  algebra. Rust `ivus_therapy_pressure_field` owns finite-input validation and
  the sector aperture plus radial decay pressure model; Python only reshapes
  returned pressure samples for plotting and downstream dose metrics. Evidence
  tier: value-semantic Rust nextest coverage for catheter zeroing, angular
  aperture, radial decay, and invalid inputs; focused PyO3 source/value tests;
  editable `maturin` rebuild; Miniforge py-compile; touched-crate `cargo
  check`; and touched-path diff hygiene.
- **BOOK-CH20 PSNR relative-error curve — RESOLVED [patch].** The Chapter 20
  validation PSNR panel no longer computes the relative-RMSE theorem with
  Python-side `-20 * np.log10(eps)`. Rust
  `validation_psnr_from_relative_rmse` owns finite positive input validation and
  the `PSNR = -20 log10(relative_rmse)` conversion; Python only plots returned
  values. Evidence tier: value-semantic Rust nextest coverage for closed-form
  samples and invalid inputs; focused PyO3 source/value tests; editable
  `maturin` rebuild; Miniforge py-compile; touched-crate `cargo check`; and
  touched-path diff hygiene.
- **BOOK-CH20 Pearson phase sensitivity — RESOLVED [patch].** The Chapter 20
  validation phase-sensitivity panel no longer computes the same-frequency
  sinusoid Pearson theorem with Python-side `np.cos` or inverse threshold
  markers with Python-side `np.arccos`. Rust
  `phase_shift_correlation_curve` owns `r(phi) = cos(phi)` over finite phase
  samples, and `phase_error_degrees_for_correlation` owns the inverse
  `phi = acos(r)` degree conversion over `r in [-1, 1]`; Python only plots
  returned values. Evidence tier: value-semantic Rust nextest coverage for
  closed-form samples and invalid inputs; focused PyO3 source/value tests;
  editable `maturin` rebuild; Miniforge py-compile; touched-crate `cargo
  check`; and touched-path diff hygiene.
- **BOOK-CH17 PINN convergence curve — RESOLVED [patch].** The Chapter 17
  inverse-problems PINN loss panel no longer computes exponential convergence
  curves with Python-side `np.exp`. Rust `exponential_convergence_curve` owns
  finite-input validation and the `L(epoch) = L0 * exp(-epoch / tau) + floor`
  law; Python only supplies the figure's curve parameters and plots returned
  arrays. Evidence tier: value-semantic Rust nextest coverage for closed-form
  samples, monotone decay, and invalid inputs; focused PyO3 source/value tests;
  editable `maturin` rebuild; Miniforge py-compile; touched-crate `cargo
  check`; and touched-path diff hygiene.
- **BOOK-CH17 Gaussian deconvolution fixture — RESOLVED [patch].** The Chapter
  17 inverse-problems L-curve panel no longer builds the Gaussian convolution
  matrix, two-bump truth signal, or sinusoidal measurement perturbation with
  Python-side `np.exp`/`np.sin` algebra. Rust `gaussian_deconvolution_fixture`
  owns fixture validation and deterministic generation; Python only plots
  returned arrays and passes them into the Rust L-curve routines. Evidence
  tier: value-semantic Rust nextest coverage for matrix/truth samples,
  perturbation sensitivity, and invalid inputs; focused PyO3 source/value
  tests; editable `maturin` rebuild; Miniforge py-compile; touched-crate
  `cargo check`; and touched-path diff hygiene.
- **BOOK-CH10 MRE envelope — RESOLVED [patch].** The Chapter 10 elastography
  MRE displacement panel no longer computes the exponential depth envelope with
  Python-side `np.exp`. Rust `mre_displacement_envelope` owns finite-input
  validation and the `A * exp(-z / d_pen)` law; Python only plots returned
  arrays. Evidence tier: value-semantic Rust nextest coverage for zero depth,
  one penetration depth, monotonic decay, and invalid inputs; focused PyO3
  source/value tests; editable `maturin` rebuild; Miniforge py-compile;
  touched-crate `cargo check`; and touched-path diff hygiene.
- **BOOK-CH23 VCZ coherence — RESOLVED [patch].** The Chapter 23
  passive-acoustic-mapping spatial-coherence panel no longer computes the
  Van Cittert-Zernike sinc law with Python-side `np.sinc`. Rust
  `van_cittert_zernike_coherence` owns geometry validation and the normalized
  sinc evaluation; Python only plots returned arrays. Evidence tier:
  value-semantic Rust nextest coverage for zero separation, midpoint, first
  zero, and invalid inputs; focused PyO3 source/value tests; editable
  `maturin` rebuild; Miniforge py-compile; touched-crate `cargo check`; and
  touched-path diff hygiene.
- **BOOK-CH03 PSTD source waveform — RESOLVED [patch].** The Chapter 3
  Westervelt PSTD validation figure no longer constructs its source waveform
  with Python-side `P0 * np.sin(OMEGA0 * t_src)` algebra. Python routes the
  source through existing Rust/PyO3 `fubini_waveform` at `sigma=0.0`, whose
  zero-distance branch is value-tested as the sinusoidal source contract. The
  script header no longer declares a stale SciPy dependency. Evidence tier:
  focused PyO3 value/source tests; Miniforge py-compile; touched-path diff
  hygiene.
- **BOOK-CH01 standing-wave construction — RESOLVED [patch].** The Chapter 1
  solver-validation figure no longer constructs its standing-wave initial
  condition or analytic overlay with Python-side `p0 * np.sin(k * x)` algebra.
  The script routes both through existing Rust/PyO3 `standing_wave_1d`, and the
  wrapper documentation now states the same `p0 * sin(kx) * cos(omega*t)`
  contract as the Rust core. Evidence tier: focused PyO3 value/source tests
  plus Rust/PyO3 compile checks; Miniforge py-compile; touched-path diff
  hygiene.
- **BOOK-CH05 axial RF pulse — RESOLVED [patch].** The Chapter 5
  diagnostic-imaging PSF figure no longer constructs its centered two-cycle
  Hann-windowed RF pulse with Python `np.hanning` and local carrier
  multiplication. Rust `centered_hann_tone_burst_waveform` owns finite-input
  handling, active-sample selection, discrete Hann weighting, and carrier
  evaluation; Python calls the helper before passing the pulse to the existing
  Rust B-mode envelope detector. Evidence tier: Rust value-semantic nextest
  coverage plus focused PyO3 differential/source tests; `cargo check -p
  kwavers-physics -p kwavers-python`; editable `maturin` rebuild; Miniforge
  py-compile; touched-path diff hygiene.
- **BOOK-CH25 RTM axial spectrum — RESOLVED [patch].** The Chapter 25
  RTM/adaptive-beamforming axial spatial-spectrum figure no longer computes
  Hann windowing or one-sided FFT power spectra in Python. Rust/PyO3
  `demeaned_hann_power_spectrum_1d` owns finite-input validation, mean removal,
  workspace Hann weighting, Apollo-backed FFT execution, one-sided frequency
  axis generation, and power calculation; Python only plots returned arrays.
  Evidence tier: differential PyO3 test against NumPy's `rfft`/`rfftfreq`
  contract plus source guard; `cargo check -p kwavers-python`; editable
  `maturin` rebuild; Miniforge py-compile; touched-path diff hygiene.
- **BOOK population-emission seed boundary — RESOLVED [patch].** The shared
  book `simulate_population_emission` helper no longer accepts a Python RNG
  object or derives a Rust seed with `rng.integers`; it accepts an integer seed
  and forwards that seed directly to Rust/PyO3 `simulate_population_emission`.
  Chapter 24 and Chapter 21e population-emission callers now pass explicit
  deterministic seeds. Evidence tier: source guard plus focused PyO3 value
  tests; Miniforge py-compile; touched-path diff hygiene.
- **BOOK-CH07 closed-loop CEM43 fixture — RESOLVED [patch].** The Chapter 7
  theranostics dose panel no longer generates feedback focal-temperature jitter
  or per-trace CEM43 curves in Python. Rust `closed_loop_cem43_fixture` owns the
  fixed-power, feedback, and underdrive temperature traces, deterministic
  seed-controlled thermometry jitter, and cumulative CEM43 arrays through the
  canonical `cem43_cumulative` implementation; Python only plots returned
  arrays. Evidence tier: value-semantic Rust nextest coverage, focused PyO3
  source/value tests, editable `maturin` rebuild, executable chapter
  regeneration, and changed PNG decode.
- **BOOK-CH23 cavitation dose fixture — RESOLVED [patch].** The Chapter 23
  passive-acoustic-mapping dose panel no longer builds the stable-dose
  staircase or seeded compound-Poisson inertial-dose trials in Python. Rust
  `passive_cavitation_dose_fixture` owns the validated time axis, deterministic
  stable dose, seeded Poisson event counts, exponential collapse energies, and
  normalization; Python only plots returned arrays. Evidence tier:
  value-semantic Rust nextest coverage, focused PyO3 source/value tests,
  editable `maturin` rebuild, executable chapter regeneration, and changed PNG
  decode.
- **BOOK-CH05 shear-wave tissue ranges — RESOLVED [patch].** The Chapter 5
  shear-wave elastography tissue-range panel no longer computes range endpoint
  speeds with Python-side `np.sqrt(mu/rho)`. Python routes each endpoint through
  the existing Rust/PyO3 `shear_wave_speed` binding and plots returned limits.
  Evidence tier: closed-form differential/value test against
  `sqrt(shear_modulus / density)`, Chapter 5 source/export guards, Miniforge
  py-compile, focused pytest, and touched-path diff hygiene.
- **BOOK-CH04 apodization response — RESOLVED [patch].** The Chapter 4
  beamforming apodization-window panel no longer computes zero-padded FFT,
  FFT-shift, magnitude normalization, or dB response conversion in Python.
  Rust `apodization_window_response` owns the window coefficients, response
  spectrum, and cycles-per-aperture axis; Python only plots returned arrays.
  Evidence tier: value-semantic Rust nextest coverage against a manual
  DFT-shift oracle and invalid lengths, focused PyO3 source/value tests against
  NumPy's equivalent FFT convention, editable `maturin` rebuild, and
  touched-crate `cargo check`.
- **BOOK-CH10 thermal-strain RF fixture — RESOLVED [patch].** The Chapter 10
  elastography thermal-strain panel no longer generates broadband speckle RF,
  carrier modulation, or apparent-displacement warp interpolation in Python.
  Rust `thermal_strain_rf_fixture` owns the seeded RF fixture and returns
  pre/post-heating volumes to the existing Rust `thermal_strain_reconstruct`
  pipeline; Python only plots returned arrays. Evidence tier: value-semantic
  Rust nextest coverage for seeded determinism, input sensitivity, zero-shift
  identity, and invalid inputs; focused PyO3 source/value tests for script
  routing and binding behavior; editable `maturin` rebuild; and touched-crate
  `cargo check`.
- **BOOK-CH03 PSTD harmonic extraction — RESOLVED [patch].** The Chapter 3
  nonlinear-acoustics PSTD validation panel no longer computes Hann-windowed
  FFT harmonic amplitudes in Python. Rust `hann_windowed_harmonic_amplitudes`
  owns the symmetric Hann window, workspace FFT, one-sided amplitude
  normalization, nearest harmonic-bin selection, and finite-input validation;
  Python only passes the steady-state sensor trace slab and plots returned
  amplitudes. Evidence tier: value-semantic Rust nextest coverage against a
  manual DFT-bin oracle and invalid inputs, focused PyO3 source/value tests
  against NumPy's equivalent Hann-windowed FFT convention, editable `maturin`
  rebuild, and touched-crate `cargo check`.
- **BOOK-CH07 PCD spectra/controller — RESOLVED [patch].** The Chapter 7
  theranostics PCD panels no longer compute Hann-windowed FFTs, SC/IC band
  ratios, or the asymmetric pressure-control loop in Python. Rust
  `keller_miksis_pcd_spectrum` owns Keller-Miksis wall-velocity spectrum
  generation and subharmonic/broadband ratios using the workspace FFT, and Rust
  `keller_miksis_pcd_controller_trace` owns the pulse-by-pulse pressure trace;
  Python only plots returned arrays. Evidence tier: value-semantic Rust nextest
  coverage for synthetic subharmonic ratio and bounded controller traces,
  focused PyO3 source/value tests, executable figure regeneration, and finite
  nonblank PNG validation for all Chapter 7 figures.
- **BOOK-CH05 Gaussian photoacoustic waveform — RESOLVED [patch].** The
  Chapter 5 photoacoustic panel no longer computes the Gaussian absorber
  initial-pressure profile or spatial derivative waveform with Python/NumPy.
  Rust `gaussian_absorber_photoacoustic_profile` now owns the closed-form
  `p0(z) = Gamma * mu_a * Phi * exp(-0.5*((z-z0)/sigma)^2)` profile and
  analytic `dp0/dz` signal sampled at `z = c*t`; Python only plots returned
  arrays. Evidence tier: analytical closed-form specification plus
  value-semantic Rust nextest coverage, focused PyO3 source/value manifest
  coverage, editable `maturin` rebuild, and touched-crate `cargo check`.
- **BOOK-TRANSCRANIAL subspot/BBB dose adapters — RESOLVED [patch].** The book
  transcranial planning adapters no longer construct GBM subspot rasters,
  focal coverage fractions, Gaussian BBB dose fields, Hill permeability fields,
  stable-cavitation fields, inertial-risk fields, or opening masks in Python.
  `kwavers-therapy` owns those computations, and `pykwavers` exposes thin
  wrappers returning arrays for plotting/dataclass packaging. Evidence tier:
  Rust value-semantic nextest coverage for focal coverage fraction, focused
  Python source/value tests, direct binding smoke check, editable PyO3 rebuild,
  and clean touched-crate `cargo check`.
- **BOOK-TRANSCRANIAL planning PyO3 contract — RESOLVED [patch].** The book
  transcranial planning helpers no longer treat `pykwavers` as optional and no
  longer carry Python fallback formulas for MI fields, cavitation risk, BBB
  permeability, HU sound speed, or HU density. They route those quantities
  through existing Rust/PyO3 bindings, and the top-level Python facade now
  exports the existing transcranial array planner and Pennes thermal-dose
  binding used by the scripts. Evidence tier: source guard plus focused
  value-semantic Python tests; py-compile passes for touched files, focused
  transcranial planning pytest passes, and top-level binding export check
  confirms both exported functions.
- **BOOK-CH24 CEM43 vector dose and population helper — RESOLVED [patch].**
  The Chapter 24 LIFU thermal-safety panel no longer computes sparse growing
  CEM43 prefixes with `kw.compute_cem43` and interpolates them in Python. It
  calls Rust/PyO3 `kw.cem43_cumulative` once over the full focal-temperature
  history. The same pass removed the ignored `max_nucleation_cycles` argument
  from the shared cavitation population helper and all book callers, avoiding a
  Python parameter that had no Rust-core effect. Evidence tier: source guard
  plus empirical artifact validation; py-compile passes for touched scripts,
  focused Chapter 24/26 source pytest passes, Chapter 24 regenerates all
  figures, and all 10 Chapter 24 PNGs decode as nonblank.
- **BOOK-CH05 CW/vector Doppler fixture — RESOLVED [patch].** The Chapter 5
  continuous-wave/vector Doppler figure no longer synthesizes RF, demodulates
  CW baseband, runs FFTs, computes pulsed-wave Nyquist velocity, builds
  cross-beam projections, or solves the vector-flow normal equations in Python.
  Rust `continuous_wave_vector_flow_fixture` now composes
  `ContinuousWaveDoppler` and `VectorFlowEstimator`; Python only plots returned
  arrays and vectors. Evidence tier: Rust value-semantic nextest coverage for
  CW peak recovery, PW Nyquist comparison, vector recovery, and invalid inputs;
  focused PyO3 value/source manifest coverage; editable `maturin` rebuild;
  executable Chapter 5 regeneration; and visual inspection of
  `fig11_cw_vector_doppler`.
- **BOOK-CH13 spectroscopic unmixing sweep — RESOLVED [patch].** The Chapter 13
  spectroscopic unmixing figure no longer owns the HbO2/Hb sO2 sweep,
  deterministic perturbation model, nonnegative concentration clipping, or sO2
  ratio calculation in Python. Rust `spectroscopic_unmixing_so2_sweep` now owns
  those calculations and reuses the existing Rust least-squares unmixing kernel;
  Python only plots returned curves. Evidence tier: Rust value-semantic nextest
  coverage for exact unperturbed recovery and invalid inputs, focused PyO3
  source/value manifest coverage, editable `maturin` rebuild, executable
  Chapter 13 regeneration, and visual inspection of
  `fig05_spectroscopic_unmixing`.
- **BOOK-CH05 contrast-agent Doppler spectrum — RESOLVED [patch].** The Chapter
  5 Doppler figure no longer computes the slow-time IQ series, FFT spectrum,
  velocity axis, Nyquist velocity, or Kasai estimate in Python. Rust
  `contrast_agent_doppler_spectrum` now owns those calculations after the
  existing Rust Rayleigh-Plesset solve supplies the bubble-scattering amplitude;
  Python only adapts arrays and plots. Evidence tier: Rust value-semantic
  nextest coverage for recovered velocity, exact output lengths, spectral peak
  bin, Nyquist velocity, and invalid inputs; focused PyO3 value/source manifest
  coverage; editable `maturin` rebuild; executable Chapter 5 regeneration; and
  visual inspection of `fig03_doppler_spectrum`.
- **BOOK-CH23 eigenspace PAM spectrum — RESOLVED [patch].** The Chapter 23
  eigenspace singular-value panel no longer builds a stochastic CSD matrix in
  Python before calling the generic Hermitian eigensolver. Rust
  `eigenspace_covariance_eigenvalues` now owns Theorem 22.2's deterministic
  signal/noise eigenvalue split; Python only plots the returned spectrum.
  Evidence tier: Rust value-semantic nextest coverage for the exact split and
  invalid inputs, focused PyO3 source/value regression, editable `maturin`
  rebuild, executable Chapter 23 regeneration, and visual inspection of
  `fig04_eigenspace_svd`.
- **BOOK-CH14 pressure/velocity plane wave — RESOLVED [patch].** The Chapter 14
  pressure-velocity panel no longer computes `P0*sin(kx-wt)` and
  `P0/(rho*c)*sin(kx-wt)` in Python. Rust `plane_wave_pressure_velocity_1d`
  now owns the progressive-wave pressure field and impedance-scaled particle
  velocity; Python selects the axis, converts units, and plots. Evidence tier:
  Rust value-semantic nextest coverage for impedance ratio and invalid media,
  focused PyO3 source/value regression, editable `maturin` rebuild, executable
  Chapter 14 regeneration, and visual inspection of `fig03_pressure_velocity`.
- **BOOK-CH23 passive DAS RF synthesis — RESOLVED [patch].** The Chapter 23
  passive DAS sensitivity panel no longer synthesizes the point-source receive
  RF traces in Python. Rust `passive_cavitation_point_source_rf` now owns
  element receive delay, Gaussian emission envelope, carrier phase, and `1/r`
  spreading; Python adapts arrays and passes the traces to the existing
  `passive_acoustic_map_das` beamformer. Evidence tier: Rust value-semantic
  nextest coverage for the closed-form sample and invalid inputs, focused PyO3
  source/value regression, editable `maturin` rebuild, executable Chapter 23
  regeneration, and visual inspection of `fig02_das_sensitivity_map`.
- **BOOK-CH07 CEM43 dose accumulation — RESOLVED [patch].** The Chapter 7
  closed-loop CEM43 panel no longer performs an O(n²) Python prefix loop around
  `kw.compute_cem43`. It now calls the vector Rust/PyO3
  `kw.cem43_cumulative` binding once per temperature history; Python only plots
  the returned dose arrays. Evidence tier: focused source/value PyO3 regression,
  executable Chapter 7 regeneration, and visual inspection of
  `fig05_closed_loop_cem43`.
- **BOOK-CH22/CH23 passive acoustic mapping — RESOLVED [patch].** The stable
  versus inertial cavitation spectrum model for Figure 22.1 no longer lives in
  the Python book script. Rust `normalized_cavitation_emission_spectrum` owns
  the harmonic/subharmonic Lorentzian spectrum and inertial broadband envelope;
  the PyO3 wrapper exposes stable/inertial regimes, and Python only converts the
  returned normalized PSD to dB for plotting. Evidence tier: Rust
  value-semantic nextest coverage for normalization, invalid inputs, and
  inertial interharmonic-floor elevation; focused Python source/value tests;
  editable `maturin` rebuild; executable Chapter 23 regeneration; and visual
  inspection of `fig01_cavitation_spectra`.
- **BOOK-CH21 histotripsy comparison — RESOLVED [patch].** The Chapter 21
  classical-vs-millisecond histotripsy comparison no longer duplicates the
  shock-rich intensity-to-pressure inverse `sqrt(2*rho*c*I)` in Python. Its
  millisecond-pulse heat-source path now calls Rust/PyO3
  `kw.acoustic_pressure_amplitude_from_intensity` before the existing Rust
  `kw.acoustic_heat_source_density` call; Python supplies the scalar scenario
  intensity and plots the returned thermal/CEM43 outputs. Evidence tier:
  focused source/value PyO3 regression, executable Chapter 21 comparison
  regeneration, and visual inspection of the regenerated bioheat, CEM43, and
  mechanism-map artifacts.
- **BOOK-CH04 transducer arrays and beamforming — RESOLVED [patch].** The
  Chapter 4 transducer-array script no longer carries stale Python-side
  array-factor/directivity multiplication or obsolete helper call signatures.
  Figure 7.2 now calls Rust/PyO3 `kw.beam_pattern_magnitude` and
  `kw.grating_lobe_angles`; Figure 7.4 passes the current f-number contract into
  Rust `kw.lateral_resolution_m`; Figure 7.3 passes x/z axes plus element z
  coordinates directly into Rust `kw.beam_pattern_2d`, removing the Python
  `meshgrid` allocation; and Figure 7.6 uses Rust `kw.bli_stencil_weights` with
  the current even-stencil contract. Evidence tier: focused source/value PyO3
  regression, executable Chapter 4 transducer-array regeneration, and visual
  inspection of the regenerated beam-pattern, 2-D field, and BLI artifacts.
- **BOOK-CH01 wave fundamentals — RESOLVED [patch].** The Chapter 1 travelling
  pulse source profile and d'Alembert reference no longer duplicate the
  Gaussian/carrier and shifted-interpolation formulas in Python. Figure 1.1 now
  calls Rust/PyO3 `kw.gaussian_modulated_pulse_1d` and
  `kw.dalembert_split_solution_1d`; Python invokes the solver binding, adapts
  arrays, and renders the returned fields. Evidence tier: Rust value-semantic
  nextest coverage for both analytical helpers, focused PyO3 binding tests,
  executable chapter regeneration, and visual inspection of the regenerated
  Chapter 1 standing/travelling-wave artifact.
- **BOOK-CH02 numerical methods — RESOLVED [patch].** The Chapter 2 CFL
  stability, modified-wavenumber, and k-space temporal-correction figure data no
  longer duplicate stencil or sinc formulas in Python. The figure script now
  calls Rust/PyO3 `kw.fdtd_cfl_stability_region_2d`,
  `kw.centered_fd_modified_wavenumber`, and `kw.kspace_temporal_correction`;
  Python generates plotting axes, reshapes returned arrays, and renders figures.
  Evidence tier: Rust value-semantic nextest coverage for the new analytical
  helpers, focused PyO3 binding tests, executable Chapter 2 regeneration, and
  visual inspection of the regenerated CFL/dispersion/correction artifacts.
- **BOOK-CH03 nonlinear acoustics — RESOLVED [patch].** The Chapter 3 Fubini
  waveform evolution no longer reconstructs the harmonic series in Python. The
  figure script now calls Rust/PyO3 `kw.fubini_waveform`; Python selects sample
  axes and plots returned pressure arrays. Evidence tier: Rust value-semantic
  nextest coverage for the sinusoid limit and harmonic expansion, focused PyO3
  binding tests, executable Chapter 3 regeneration, and visual inspection of
  the regenerated nonlinear waveform artifact.
- **BOOK-CH06 therapeutic ultrasound — RESOLVED [patch].** The Chapter 6 HIFU
  heat-source setup no longer duplicates the pressure/intensity inverse
  `sqrt(2*rho*c*I)` in Python. The figure script now calls Rust/PyO3
  `kw.acoustic_pressure_amplitude_from_intensity`; Python passes scalar
  intensity samples into the binding and plots the solver outputs. Evidence
  tier: Rust value-semantic nextest coverage for pressure/intensity round-trip
  values and invalid inputs, focused PyO3 binding tests, executable Chapter 6
  regeneration, and visual inspection of the regenerated thermal artifacts.
- **BOOK-CH07 theranostics — RESOLVED [patch].** The Chapter 7 Minnaert
  resonance marker radii no longer duplicate the inverse closed-form formula in
  Python. The figure script now calls Rust/PyO3
  `kw.minnaert_radius_for_frequency_m`; Python selects marker frequencies and
  plots returned radii. Evidence tier: Rust value-semantic nextest coverage for
  the inverse/forward round-trip and invalid inputs, focused PyO3 binding tests,
  executable Chapter 7 regeneration, and visual inspection of the regenerated
  Minnaert resonance artifact.
- **BOOK-CH08 retained propagation script — RESOLVED [patch].** The retained
  Chapter 8 acoustic-propagation script no longer derives geometric spreading
  envelopes from Python-side pressure samples. Its spreading-law panel now calls
  Rust/PyO3 `kw.geometric_spreading_intensity_envelopes`; Python selects the
  radius axis and plots returned intensity envelopes. Evidence tier: Rust
  value-semantic nextest coverage for normalized spherical/cylindrical laws and
  invalid radii, focused PyO3 binding tests, executable retained-script
  regeneration, and visual inspection of the regenerated spreading artifact.
- **BOOK-CH05 diagnostic imaging — RESOLVED [patch].** The Chapter 5 figure
  script no longer falls back to SciPy Hilbert envelope detection or random
  Python Doppler noise. It requires `pykwavers` and routes the axial envelope,
  lateral PSF, Doppler shift, and contrast-bubble amplitude through Rust/PyO3
  bindings. The top-level `pykwavers` package now re-exports the source-registered
  imaging helper bindings, and all Chapter 5 figure artifacts regenerate and
  decode. Evidence tier: source-level manifest regression, editable `maturin`
  rebuild, executable chapter regeneration, and PNG/PDF artifact validation.
- **BOOK-CH10 elastography — RESOLVED [patch].** The Chapter 10 figure script no
  longer treats `pykwavers` as optional. It routes the MRE displacement figure
  through the Rust `mre_displacement_field` analytical kernel, the top-level
  package re-exports that helper, and the book caption describes the implemented
  damped plane-wave model. Evidence tier: source-level manifest regression,
  executable chapter regeneration, visual inspection of `fig05_mre_displacement`,
  and PNG/PDF artifact validation.
- **BOOK-CH11 sources/transducers — RESOLVED [patch].** The Chapter 11 figure
  script no longer treats `pykwavers` as optional. Its BLI accuracy panel now
  routes nearest-neighbour and BLI sinusoid reconstruction RMS curves through
  Rust/PyO3 `kw.bli_interpolation_error_curves`, which uses the Rust
  `bli_stencil_weights` kernel internally; Python only converts RMS to dB and
  plots. Array-factor dB rendering uses magnitude before log compression.
  Evidence tier: Rust value-semantic nextest coverage, focused PyO3
  source/value regression, editable `maturin` rebuild, executable chapter
  regeneration, visual inspection of `fig05_bli_accuracy`, and PNG/PDF artifact
  validation.
- **BOOK-CH12 media/tissue models — RESOLVED [patch].** The Chapter 12 figure
  script no longer treats `pykwavers` as optional. Its Pennes bioheat slab
  profile now uses `kw.pennes_steady_state_temperature_profile` from the Rust
  analytical thermal module instead of a Python-side closed-form duplicate, and
  the media chapter captions name the Rust bindings behind the regenerated
  sound-speed, tissue-property, B/A, power-law attenuation, and Pennes figures.
  Evidence tier: Rust value-semantic nextest coverage for the new analytical
  helper, source-level manifest regression, executable chapter regeneration,
  visual inspection of `fig05_bioheat`, and PNG/PDF artifact validation.
- **BOOK-CH13 photoacoustics — RESOLVED [patch].** The Chapter 13 figure script
  no longer treats `pykwavers` as optional. Its spectroscopic unmixing panel
  now uses deterministic measurement perturbations instead of random
  Python-generated noise while retaining Rust-owned Hb/HbO2 spectra,
  Gruneisen, PA sphere, axial-resolution, and least-squares unmixing calls.
  Evidence tier: source-level manifest regression, executable chapter
  regeneration, visual inspection of `fig05_spectroscopic_unmixing`, and
  PNG/PDF artifact validation.
- **BOOK-CH14 sensors/measurements — RESOLVED [patch].** The Chapter 14 figure
  script no longer treats `pykwavers` as optional. Its hydrophone directivity
  panel now matches the chapter's circular-hydrophone model through Rust
  `kw.circular_piston_directivity`, and its noisy sensor recording panel uses
  seeded Rust `kw.add_noise` instead of Python RNG. Evidence tier: source-level
  manifest regression, executable chapter regeneration, visual inspection of
  `fig01_hydrophone_directivity` and `fig05_signal_comparison`, and PNG/PDF
  artifact validation.
- **BOOK-CH17 inverse problems — RESOLVED [patch].** The Chapter 17 figure script
  no longer treats `pykwavers` as optional for the SVD/L-curve/FWI figures, no
  longer skips the FWI figures when the binding is absent, and no longer uses
  Python RNG for the L-curve perturbation. The SVD/L-curve captions now name the
  implemented Rust helper bindings, and the manifest guards the Chapter 17
  binding calls, top-level exports, and figure artifacts. Figure 18.6 now routes
  its fast-sweeping eikonal traveltimes through Rust/PyO3
  `kw.eikonal_traveltime_2d` and its synthetic point-scatterer Kirchhoff image
  through Rust/PyO3 `kw.kirchhoff_point_scatterer_image_2d`; Python only adapts
  arrays and plots the returned fields. Evidence tier: source-level manifest
  regression plus focused value-semantic Rust-owned binding tests; full
  executable chapter regeneration and visual inspection remain unrefreshed in
  this slice.
- **BOOK-CH18 sonogenetics — RESOLVED [patch].** The Chapter 18 figure script
  no longer treats `pykwavers` as optional and no longer skips the LIF raster
  when the binding is absent. Its streaming panel uses Rust
  `kw.acoustic_streaming_velocity`, its Gorkov force panel is documented as the
  Rust `kw.gorkov_radiation_force_1d` cell-force model it renders, and its
  activation panel now matches the book contract by routing intensity-to-pressure
  conversion through Rust `kw.acoustic_pressure_amplitude_from_intensity`,
  tension-gated channels through Rust membrane-tension plus Boltzmann gates, and
  hsTRPA1 through a thin PyO3 wrapper over the existing Rust pressure-threshold
  gate. Evidence tier: source-level manifest regression plus focused
  source/value tests, executable chapter regeneration, visual inspection of
  `fig05_activation_comparison`, and PNG/PDF artifact validation.
- **BOOK-CH21 simulation orchestration — RESOLVED [patch].** The Chapter 21
  figure script no longer treats `pykwavers` as optional. The bubble-radius
  comparison routes directly through the Rust/PyO3 Rayleigh-Plesset,
  Keller-Miksis, and Gilmore solver bindings, and the manifest guards the
  solver calls, top-level exports, book-text Rust ownership claim, and PNG/PDF
  artifact presence. Evidence tier: source-level manifest regression,
  executable chapter regeneration, visual inspection of
  `fig01_bubble_ode_comparison`, and PNG/PDF artifact validation.
- **BOOK-CH34 optoacoustic focused ultrasound — RESOLVED [patch].** The Chapter
  34 figure script no longer treats `pykwavers` as optional. The SOAP resolution
  and gain figure routes through the Rust/PyO3 optoacoustic transducer kernels
  for numerical aperture, f-number, lateral resolution, and focal gain, and the
  manifest guards the binding calls, top-level exports, book-text SSOT claim,
  and PNG/PDF artifact presence. Evidence tier: source-level manifest
  regression, executable chapter regeneration, visual inspection of
  `fig01_soap_resolution_gain`, and PNG/PDF artifact validation.
- **BOOK-CH29 pressure diagnostics — RESOLVED [patch].** The Chapter 29
  pressure-diagnostics helper no longer treats `pykwavers` as optional and no
  longer carries a Python duplicate of the mechanical-index equation. The
  projected pressure diagnostic now routes MI through the Rust/PyO3
  `kw.mechanical_index` safety kernel. Evidence tier: source-level regression
  guarding against fallback tokens, value-semantic projected-pressure diagnostic
  test, and Rust safety-kernel nextest coverage.
- **BOOK-CH30 intravascular ultrasound — RESOLVED [patch].** The Chapter 30 IVUS
  figure script no longer treats `pykwavers` as optional and no longer carries
  extension-unavailable fallback formulas for intensity, adiabatic temperature
  rise, B-mode log compression, RF-line envelope detection, or therapy
  mechanical index. Those surfaces now call Rust/PyO3 kernels unconditionally.
  The deterministic IVUS vessel phantom, anatomy masks, tissue-property fields,
  impedance-gradient reflectivity, and seeded Rayleigh speckle now come from
  Rust/PyO3 `kw.ivus_vessel_phantom`; Python maps returned arrays into the
  plotting dataclass. Evidence tier: source-level regression guarding against
  fallback tokens and Python RNG/material generation, value-semantic IVUS
  chapter tests, focused Rust nextest coverage for the analytical IVUS phantom,
  executable Chapter 30 regeneration, and visual inspection of the anatomy and
  B-mode PNG artifacts.
- **BOOK-CH26 response smoothing and focal dose — RESOLVED [patch].** The
  Chapter 26 neural-response trace no longer imports SciPy or builds a
  Python-side spike train for Gaussian response-probability smoothing. Rust
  `lif_response_probability_py` now owns spike-time binning, Gaussian
  convolution, firing-rate normalization, and response clamping. The Chapter 26
  focal Pennes trace also no longer computes sparse growing-prefix CEM43 values
  and interpolates them in Python; it calls Rust/PyO3 `cem43_cumulative`.
  Evidence tier: Rust value-semantic nextest coverage for bounded/input-
  sensitive response probability and invalid domains, focused Python
  source/value coverage, editable `maturin` rebuild, executable Chapter 26
  regeneration, visual inspection of `fig02_mechanochemical_response`, and
  nonblank PNG decode checks for all Chapter 26 figures.
- **BOOK-CH33 CMUT vs PMUT — RESOLVED [patch].** The Chapter 33 MEMS figure
  script no longer carries a redundant optional `pykwavers` import branch. The
  script imports `pykwavers` directly and calls Rust/PyO3 MEMS helpers for
  resonance, immersion loading, CMUT collapse/coupling/heating/bandwidth/output,
  PMUT coupling/heating/bandwidth/output, flex derating, and the IVUS figure of
  merit. Evidence tier: focused Python source/value tests for CMUT pull-in gap
  scaling, PMUT drive scaling/material ordering, and IVUS verdict routing,
  executable chapter regeneration, and visual inspection of a regenerated
  Chapter 33 artifact.
- **BOOK-CH24/CH26 PyO3 import contract — PARTIAL [patch].** The Chapter 24
  BBB-LIFU and Chapter 26 neuromodulation scripts no longer wrap `pykwavers`
  imports in optional `_HAS_KW` branches. Chapter 24 also replaces the
  try/except helper import with an explicit script-directory path before import.
  Chapter 24's inertial-cavitation MI frequency curves now route through the
  Rust/PyO3 `mechanical_index_frequency_sweep` safety helper instead of
  Python-side `constant / sqrt(f_MHz)` formulas, and its passive-cavitation
  pressure sweep routes MI through `kw.mechanical_index_field`. The Chapter 24
  inertial-damage probability curve now routes through the Rust/PyO3
  `bbb_inertial_damage_probability` BBB helper instead of inline NumPy logistic
  algebra. Chapter 26's neuromodulation cavitation-risk contour now routes
  through the Rust/PyO3 `mechanical_index_cavitation_risk` safety helper instead
  of inline NumPy logistic algebra. Chapter 24's passive-cavitation
  stable-onset, inertial-onset, and controller-cap classification now routes
  through the Rust/PyO3 `cavitation_therapeutic_window_indices` passive-dose
  helper instead of Python-side band-ratio scans. Chapter 24's population-monitor
  operating-point selection now routes through the Rust/PyO3
  `cavitation_inertial_fraction_onset_index` passive-dose helper instead of
  Python-side broadband-fraction scans. Chapter 24's per-spot cavitation monitor
  raster now routes through the Rust/PyO3 `per_spot_cavitation_dose_grid`
  delivery helper instead of Python-side steering/interpolation loops. The
  shared curve-driven cavitation monitor trace now routes through the Rust/PyO3
  `cavitation_monitor_timeseries` helper instead of Python-side interpolation,
  seeded jitter, controller stepping, and dose accumulation. The Chapter 24
  passive-cavitation closed-loop sonication trace now routes through the
  Rust/PyO3 `closed_loop_cavitation_sonication` helper instead of Python-side
  stable/inertial interpolation, controller stepping, and dose accumulation. The
  shared raster-pulsing monitor now routes through the Rust/PyO3
  `raster_cavitation_pulsing` helper instead of Python-side steering derating,
  pressure-sweep interpolation, schedule expansion, residual-bubble shielding,
  thermal relaxation, coverage, and cumulative-dose resampling. The shared
  one-pressure population-emission helper now routes through the Rust/PyO3
  `simulate_population_emission` helper instead of Python-side
  bubble-population sampling, per-bubble solver dispatch, trace rejection, Hann
  FFT spectrum construction, and cavitation-band decomposition. The shared
  simulated per-pulse population monitor now routes through the Rust/PyO3
  `simulated_population_monitor_timeseries` helper instead of Python-side
  population-emission dispatch, controller stepping, acoustic-power scaling,
  and cumulative-dose integration. The Chapter 24 population pressure sweep now
  routes through the Rust/PyO3 `population_emission_sweep` helper instead of
  Python-side per-pressure aggregation over the one-pressure population helper.
  The Chapter 24 V_s-integrated analytic spectrum and pressure sweep now route
  through the Rust/PyO3 `volume_emission_spectrum` and `volume_emission_sweep`
  helpers instead of Python-side Keller-Miksis loops, emission conversion, PSD
  construction, receiver integration, and band decomposition. Chapter 26's
  neural-response Gaussian smoothing now routes through Rust/PyO3
  `lif_response_probability_py`, and its focal thermal-dose trace routes
  through Rust/PyO3 `cem43_cumulative`. Classification:
  the remaining summary fraction formatting is presentation-only over
  Rust-returned arrays, not domain physics. Evidence tier:
  source-level regression guarding against optional PyO3 import tokens,
  value-semantic Rust/PyO3 MI, MI-risk, BBB-damage, therapeutic-window, and
  inertial-onset, per-spot dose-grid, monitor-trace, closed-loop sonication, and
  raster-pulsing, population-emission, simulated-population-monitor, and
  population-emission-sweep, V_s spectrum, and V_s sweep tests, plus focused
  Python compile/pytest coverage.
- **TEST-MEM-1 PNG artifact decoder — RESOLVED [patch].** The cached-parity PNG
  helper no longer decodes generated PNGs through Matplotlib float arrays. It
  uses Pillow dimensions and extrema, preserving decodable/nonblank checks while
  avoiding dashboard-sized float allocations. Evidence tier: manifest
  regression rerun passed after the memory failure.
- **BOOK-AUDIT residual [verify].** The broader book still contains scripts
  with synthetic fixtures and Python-side numerical preparation. Each instance
  needs a case-by-case classification as plotting/data shaping versus domain
  physics before removal or Rust/PyO3 promotion; do not claim the book-wide
  "Python only plots" invariant until this audit is complete and executable.

### Cavitation-cloud branch reconciliation (2026-06-28)

`main` was compared against `feat/cloud-time-resolved-bubble-dynamics`,
`feat/cloud-acoustic-shielding`, `feat/cloud-implicit-coupling`, and
`feat/cloud-strong-regime-solver`. All four tips are ancestors of current
`main`; the content delta runs from those branches into `main`, not the reverse.
The remaining CLD-1 risk is therefore not missing branch integration.

Evidence: `cargo nextest run -p kwavers-therapy --all-features cavitation_cloud`
passed 26/26 value-semantic cloud tests; `D:\miniforge3\python.exe -m pytest
crates/kwavers-python/tests/test_bubble_cloud_parity.py -q` passed 19/19 Python
parity tests; `D:\miniforge3\python.exe
crates/kwavers-python/examples/book/ch21e_treatment_pipeline.py` regenerated the
chapter 21e realtime treatment/feedback artifacts with no tracked output drift.
CLD-1 remains open only for k-wave/experimental erosion validation and nonlinear
frontier extensions already named in the CLD-1 row.

### Sprint A verification results (2026-05-31) — all 6 C-tier `[verify]` resolved
Verified each against code + governing equation. **Zero confirmed physics bugs**;
4 false positives, 2 real-but-overrated (downgraded). Lesson: automated severity
inflation is real — the verify gate paid off.
- **SOL-4** Westervelt `d²(p²)/dt²` — **FALSE POSITIVE.** `2p·p̈+2ṗ²` is exact;
  the FMA fuses one product (precision *gain*, not error). Closed.
- **PHY-1** Gilmore vapor correction — **FALSE POSITIVE.** `p_eq` subtracts `pv`
  (line 211) so `p_gas` is the non-condensable partial pressure; polytropic rate
  is correct. Closed.
- **PHY-3** IAPWS-IF97 Region 4 — **FALSE POSITIVE.** Coefficients n₁–n₁₀ and the
  θ/A/B/C/`[2C/(−B+√(B²−4AC))]⁴` form match the standard exactly (K→MPa→Pa). The
  "dimensional inconsistency" misread the standard. Closed.
- **AMC-1** 6th-order central difference — **FALSE POSITIVE.** Nested `mul_add`
  expands to `[−f₋₃+9f₋₂−45f₋₁+45f₊₁−9f₊₂+f₊₃]/(60dx)` = exact Fornberg. Closed.
- **AMC-2** MVDR imag-part guard — **RESOLVED (2026-06-30)**. Happy path was
  correct (`aᴴR⁻¹a` real for Hermitian R); `compute_weights` and
  `pseudospectrum` now share a validator that rejects non-finite, non-positive,
  or roundoff-inconsistent complex denominators.
- **AMC-4** wgsl boundary — **REAL, downgrade C→M** (AMC-4 below). Live shader
  (`WaveEquationGpu`); boundary is persistence (`out=in`), not the assumed
  implicit-Dirichlet. Undocumented + likely inconsistent with CPU Neumann/Dirichlet.

### Sprint B verification results (2026-05-31) — "confirmed correctness" mostly evaporated
Same pattern as Sprint A: the audit's "production panic" labels were wrong.
- **SOL-1** harmonic accessor panics — **NOT production.** Only callers are
  `#[test]` (incl. `tests.rs:104` intentional should-panic); `try_*` variants
  already exist for fallible use. Idiomatic infallible-accessor pattern. No change.
- **SOL-2** elastography:296 — **test-only** (`#[test] test_all_nonlinear_methods_integration`,
  `panic!` is the `unwrap_or_else` message). No change.
- **SOL-3** PINN-EM:69 — **test-only** (`#[test] test_boundary_condition_builder`,
  panic in a type-assertion match arm). No change.
- **PHY-5** Cattaneo defaults — **REAL, fixed.** Root cause was not a wrong value
  but dead state: `thermal_wave_speed` was never read. Removed the field (the
  flux law uses only τ; `c=√(α/τ)` is derived). 2 value-semantic thermal tests
  pass, physics unchanged.
Net: 3 false-positive "production panics" closed, 1 real dead-field removed.

### Sprint C verification results (2026-06-01) — approximation-validity bounds
6 items; **zero physics behavior changed** (doc + one behavior-preserving named const).
- **PHY-2** Gilmore adiabatic — already documented (Prosperetti 1977 cited). Closed.
- **PHY-4** Marmottant shell viscosity — **FALSE POSITIVE.** `12·μ_s·(d/R)·Ṙ/R²`
  term present at `marmottant.rs:107` (audit had wrong path + wrong claim). Closed.
- **PHY-8** parametric averaging — correctly scoped; added `Δf/f̄≪1` bound. Done.
- **CLD-2** HIFU linear-only — real; `generate_acoustic_field` already self-documents,
  added orchestrator-level note. **RESOLVED [minor] (2026-07-12):** wire KZK — added
  `propagate_volume()` to `KzkSolverPlugin`, `generate_kzk_acoustic_field()` adapter,
  conditional dispatch in `execute_therapy_step()`, `use_nonlinear_field` config flag.
  Evidence tier: compile-time integration. Residual: collimated only (no focusing phase).
- **CLD-3** O'Neil "Theorem" — real; →"approximation" + validity regime + refs;
  named/flagged the 0.7 fill factor (value preserved). Done.
- **CLD-6** Thermal Index Pennes omission — real; documented + conservative-bias
  direction + Pennes 1948 ref. Done.
Net: 2 already-handled/false-positive, 4 validity-regime docs; 0 physics changed.

### Sprint D verification results (2026-06-01) — missing literature validation
Pattern holds: of the "missing validation" items, most were already covered or
mislabeled. Added 2 genuine external-reference/property tests; tightened nothing
that was already tight; refused to fabricate a reference where none is sourced.
- **PHY-10** Minnaert resonance — **DONE.** The flagged `validate_implementation()`
  is indeed circular (code-vs-itself), but an independent check already existed
  (`test_epstein_plesset_vs_minnaert_frequency`). Added
  `test_minnaert_constant_matches_literature_value`: pins `f₀·R₀ ≈ 3.26 m·Hz`
  (Minnaert 1933; Leighton 1994) across R₀∈[1e-6,1e-3], max_rel=0.02 (computed
  ≈3.286, 0.8% off). PASSED.
- **SOL-7** PSTD source scale — **DONE.** Added
  `interior_source_conserves_total_amplitude_across_geometry`. Key physics:
  scale=1/N is *amplitude* normalization → N·scale=1 invariant to geometry/
  resolution; energy Σscale²=1/N is deliberately NOT conserved (asserting energy
  conservation would be physically wrong). PASSED.
- **PHY-9** K-M equilibrium tol — **FALSE POSITIVE (overrated).** `<1.0 m/s²` is
  NOT loose: characteristic K-M accel for default R₀=5µm is (2σ/R₀)/(ρR₀)≈5.8×10⁶
  m/s², so 1.0 is a *relative* ~1.7×10⁻⁷ bound (at the f64 cancellation floor of
  the 2σ/R₀≈2.9×10⁴ Pa subtraction). Documented the scale in-place; tightening
  further would test FP noise, not physics. No behavior change.
- **PHY-11** Gilmore collapse — **ADEQUATE.** Suite already has a value-semantic
  analytical differential check (`enthalpy_derivative_uses_state_wall_acceleration`,
  rel_err<1e-10 vs closed-form Gilmore RHS) — the strongest tier short of a
  published collapse dataset. A Lauterborn Rmax/R₀ regression needs a *citable*
  number; not fabricated. Deferred to backlog [minor] pending sourced reference.
- **CLD-11** CPML reflection — **DONE.** Added
  `theoretical_reflection_decays_monotonically_with_thickness` (Collino&Tsogka
  2001): strict-decrease + bounded-in-(0,target] property test. Parameters
  derived analytically from σ_max=σ_factor·(m+1)·c/(150π·dx) to keep R in the
  representable range (fine dx underflows R→0 by t=1, itself correct CPML behavior;
  cosθ clamped ≥0.1 internally). Note: pre-existing `test_theoretical_reflection_
  for_dimension` is circular (recomputes the same formula).

### Technical debt log (crate-split, 2026-06-01)
Debt surfaced by the workspace extraction (ADR 011). Logged for proper future
resolution; current state is a sound mitigation, not a full fix.
- **DEBT-1 (M, arch):** `KwaversError` is a kitchen-sink foundation error coupled
  to 5 higher-layer crates via `From`/`#[from]`: `wgpu`, `flume`, `ritk_registration`,
  `nifti`, `anyhow`. The orphan rule pins these to `kwavers-core`. **Mitigation
  applied:** made them optional + `#[cfg(feature)]`-gated (`gpu`/`channels`/
  `registration`/`nifti`); default `kwavers-core` is a clean leaf, facade enables
  what the monolith had. **Proper fix (deferred):** the foundation error must not
  know about GPU/channel/file-format/registration types — replace these conversions
  with generic variants (`Io`/`External`) + explicit boundary conversion in the
  consuming layer, or split `KwaversError` into layered error types
  (`thiserror` per layer, `anyhow` only at app top — per CLAUDE.md error policy).
  `anyhow::Error` in a *library* foundation error (`Other` variant) also violates
  the lib-uses-thiserror rule and should be removed in that refactor.
  File: `crates/kwavers-core/src/error/mod.rs`.
- **DEBT-2 (L):** `solver::interface::factory::RegistrationEngine` trait now has
  zero implementors (its only impl was the dead one removed in CLD-14-adjacent
  cleanup). Candidate removal once confirmed no external/plugin consumer.
- **DEBT-4 (C) — RESOLVED (2026-06-01):** FWI finite-window Born adjoint-gradient
  mismatch FIXED. Root cause: a discrete-adjoint **off-by-one** in
  `finite_window::adjoint_backward_pass`. The forward source `s[m] = −χ·accel[m]`
  drives `ps1[m+1]`, so the adjoint-state gradient must pair `accel[m]` with the
  adjoint field at the SAME source index, `ν[m]` (= `pa_prev`, just computed). The
  code paired it with `ν[m+1]` (= `pa_curr`), a one-leapfrog-step bias → 3.7% error.
  Fix: cross-correlate against `pa_prev`. Now matches central finite-difference to
  5e-4 (gradient_fd: 6/6 pass). Verified by derivation (transpose of the discrete
  leapfrog) + value-semantic FD test. The other CBS adjoints (separate paths) were
  already correct.

  ~~PRE-EXISTING WIP — original finding:~~
  `solver::inverse::fwi::frequency_domain::tests::gradient_fd::pstd_finite_window_
  born_adjoint_gradient_matches_finite_difference` FAILS: analytic adjoint gradient
  57.88 vs central-difference 55.72 = **3.7% error** (tolerance 5e-4). Magnitude
  rules out FD-step roundoff — it's a genuine adjoint-correctness gap in the Born
  operator (`DenseConvergentBornOperator`). NOT caused by the crate migration
  (extractions only rewrite import paths; numerics identical). This is in-progress
  FWI work that was *uncommitted* in the working tree at session start and got
  swept into commit 7cb668baf via `git add kwavers/`; the adjacent new
  `PstdFiniteWindowBornSecondOrderOperator` self-documents its adjoint as
  "approximate, not the exact second-order adjoint". **Action (owner): complete the
  exact Born adjoint (verify the operator transpose / adjoint-state derivation), or
  mark the test `#[ignore]` with a WIP note until then.** Not fixed here — deep FWI
  numerical work outside the migration scope.

- **DEBT-3 (M, SSOT):** Photoacoustic implementation is fragmented. After lifting
  the imaging vertical out of solver (2026-06-01: `solver/photoacoustics` →
  `simulation/photoacoustics/vertical`), there are still TWO parallel
  photoacoustic impls — `simulation/photoacoustics/vertical/` (optical/source/
  acoustic/reconstruction "solver vertical", consumed by the orchestrator) and
  `simulation/modalities/photoacoustic/` (acoustics/optics/core/reconstruction/
  types, owns `PhotoacousticResult`/`PressureFieldSeries`) — plus a coupled-solver
  touch `solver/multiphysics/photoacoustic.rs`. **Consolidate to one canonical
  photoacoustic modality** (determine the live impl, merge, delete the parallel
  one). Deferred: merging two parallel impls is careful SSOT work, not a move.

**Severity:** `C` correctness/physics-incorrect · `H` simplification/approximation
presented as exact · `M` missing validation/test · `L` doc/cleanup.

Standing facts (do NOT re-flag):
- FDTD `simd_stencil` (tiled scalar) vs `avx512_stencil` (intrinsics) are NOT
  duplication — distinct coefficient conventions + BCs (Neumann vs Dirichlet).
- `physics::bubble_dynamics` is a live `pub use` alias of `acoustics::bubble_dynamics`.
- Theranostic operator-vs-PAM passive-channel distinction is documented/intentional.
- Genuine orphans already removed: `fdtd/simd/`, `symplectic_integration/`.

---

<a id="solver"></a>
## solver/ (FDTD/PSTD/k-space/nonlinear · FWI/RTM/CBS/elastography)

| ID | Sev | file:line | Gap | Revision |
|----|-----|-----------|-----|----------|
| SOL-1 | ~~C~~ CLOSED (false positive) | `forward/elastic/nonlinear/wave_field/mod.rs:145,165` | Verified: panicking accessors only called from `#[test]` (incl. intentional should-panic at tests.rs:104); `try_*` exist for fallible use. Idiomatic. No change. | — |
| SOL-2 | ~~H~~ CLOSED (false positive) | `inverse/elastography/mod.rs:296` | Verified: inside `#[test] test_all_nonlinear_methods_integration`; panic is the `unwrap_or_else` message. No change. | — |
| SOL-3 | ~~H~~ CLOSED (false positive) | `inverse/pinn/ml/electromagnetic/mod.rs:69` | Verified: inside `#[test] test_boundary_condition_builder`, type-assertion arm. No change. | — |
| SOL-4 | H [verify] | `forward/fdtd/pressure_updater/nonlinear.rs:44` | Westervelt `d²(p²)/dt²` FMA ordering — confirm bit-faithful to Hamilton&Blackstock 3.43a | add value-semantic test vs analytic |
| SOL-5 | M | `forward/nonlinear/hybrid_angular_spectrum/absorption.rs:85` | power-law α no freq>0 / exponent∈(0,2] guard | validate inputs |
| SOL-6 | M | `forward/coupled/thermal_acoustic/stepping.rs:174-204` | density-gradient momentum update unvalidated for stability | add CFL/gradient test |
| SOL-7 | ~~M~~ DONE (2026-06-01) | `forward/pstd/implementation/core/source_injection.rs:91` | Added geometry-invariant amplitude-conservation test (clustered vs dispersed give equal scale; N·scale=1). Energy NOT conserved by design (amplitude normalization). | done |
| SOL-8 | M | `forward/fdtd/avx512_stencil/construction.rs:43` | coeffs precomputed w/o finite/sign assertion | debug_assert finite |
| SOL-9 | M | `solver/constants.rs:82` | benchmark tolerances (5%/10%/2%) uncited | cite Taflove&Hagness |
| SOL-10 | L | (crate-wide, ~339 files) | ~30% public fns lack Rustdoc | doc sweep |
| SOL-11 | L | `validation/kwave_comparison/` | validators exist but not CI-wired | wire regression suite |

<a id="physics"></a>
## physics/ (nonlinear acoustics · bubble dynamics · thermal · optics · chemistry)

| ID | Sev | file:line | Gap | Revision |
|----|-----|-----------|-----|----------|
| PHY-1 | C [verify] | `acoustics/bubble_dynamics/gilmore/mod.rs:213` | gas-pressure rate may omit vapor-pressure correction | verify vs Gilmore 1952; add term if confirmed |
| PHY-2 | ~~H~~ CLOSED (already documented) | `acoustics/bubble_dynamics/gilmore/mod.rs:99-102` | Verified: adiabatic approximation already stated + Prosperetti 1977 cited; γ selects polytropic/isothermal. No gap. | — |
| PHY-3 | ~~C~~ CLOSED (false positive) | `acoustics/bubble_dynamics/thermodynamics/vapor_pressure.rs:189` | Verified Sprint A: IAPWS-IF97 Region 4 matches the standard exactly (K→MPa→Pa). | — |
| PHY-4 | ~~H~~ CLOSED (false positive) | `acoustics/bubble_dynamics/encapsulated/model/marmottant.rs:107` | Verified: shell-viscosity term `12·μ_s·(d/R)·Ṙ/R²` IS present and in the pressure balance (Marmottant 2005 eq. 3). Audit had wrong path + wrong claim. | — |
| PHY-5 | ~~H~~ RESOLVED (2026-05-31) | `thermal/diffusion/hyperbolic.rs:13` | Verified: `thermal_wave_speed` was DEAD state (never read; flux law uses only τ). The implausible default never affected results. Fixed by REMOVING the field (SSOT/SRP) — `c=√(α/τ)` is derived, not an input. τ=20s default kept + cited (Mitra 1995). 2 thermal tests pass. | done |
| PHY-6 | H | `optics/sonoluminescence/blackbody.rs:26` | emissivity=0.1, optical_depth=0.1 unjustified magic defaults | cite or make required |
| PHY-7 | H | `chemistry/ros_plasma/ros_species/generation.rs:20,30` | Arrhenius prefactors 1e13/1e14 uncited | cite NIST/ChemKin |
| PHY-8 | ~~H~~ DOC'D (2026-06-01) | `acoustics/wave_propagation/nonlinear/parametric.rs:96` | Correctly scoped to closely-spaced primaries; added explicit `Δf/f̄≪1` validity bound. Large-Δf case is outside the parametric-array model by definition. | done |
| PHY-9 | ~~M~~ CLOSED (false positive, 2026-06-01) | `acoustics/bubble_dynamics/keller_miksis/validation/dynamics.rs:44` | `<1.0 m/s²` is relative ~1.7e-7 vs the ≈5.8e6 m/s² characteristic accel — already at the FP floor, NOT loose. Documented in-place. No change. | — |
| PHY-10 | ~~M~~ DONE (2026-06-01) | `acoustics/bubble_dynamics/epstein_plesset/tests.rs:128` | Added `test_minnaert_constant_matches_literature_value`: `f₀·R₀≈3.26 m·Hz` (Minnaert 1933; Leighton 1994), max_rel=0.02. Independent of code's own formula. | done |
| PHY-11 | M (adequate; backlog) | `acoustics/bubble_dynamics/gilmore/tests.rs` | Analytical differential check (rel_err<1e-10 vs closed-form RHS) already present — strongest tier short of a citable collapse dataset. Lauterborn Rmax/R₀ regression deferred [minor] (won't fabricate a reference). | backlog |
| PHY-12 | M [verify] | `acoustics/bubble_dynamics/heterogeneous_nucleation.rs:129` | `T::from(16π/3).expect()` generic-cast precision/panic at f32 | static bound / document |
| PHY-13 | M | `acoustics/imaging/modalities/ceus/.../scattering.rs` | no scattering-cross-section ref test (de Jong 1991) | add quantitative test |
| PHY-14 | L | `acoustics/bubble_dynamics/gilmore/mod.rs:289+` | RK4 `unwrap_or(0.0)` silences u≈c singularity | log/diagnostic |
| PHY-15 | L | multiple (`kzk.rs:38`, `rayleigh_plesset/mod.rs:14`, `attenuation.rs`) | approximation validity criteria incomplete/uncited | document bounds + refs |

<a id="clinical-domain"></a>
## clinical/ + domain/ (therapy planning · imaging recon · transducers · grid/medium/source)

| ID | Sev | file:line | Gap | Revision |
|----|-----|-----------|-----|----------|
| CLD-1 | C → **PARTIALLY ADDRESSED (2026-06-19)** | `kwavers-therapy/.../lithotripsy/cavitation_cloud.rs` | **Single-bubble dynamics now real:** the cloud erosion is driven by the actual **Gilmore (1952) compressible single-bubble collapse** (`representative_max_radius`/`inertial_collapse_energy`), capturing inertial growth `R_max ≫ R0` under rarefaction — replacing the static-R0 linear proxy. Tests: `R_max(12 MPa) > 3·R0`, deeper rarefaction erodes more. This implements the "Gilmore + Mach corrections" the code comment listed as absent. **Still open (collective / research-frontier):** multi-bubble acoustic coupling + emission back-reaction, cloud-scale energy focusing (Maeda & Colonius 2018), shock-bubble Richtmyer-Meshkov / Rayleigh-Taylor cloud instabilities, inter-phase mass transfer. Erosion carries an empirical `erosion_efficiency` (Sapozhnikov 2002) — collective cloud erosion is not a closed, "100%-accurate" problem in any library. **UPDATE (ADR 027): snapshot→time-resolved coupling DONE** — each cell now carries a real `(R,Ṙ)` state integrated by the canonical adaptive Keller-Miksis solver under the local instantaneous pressure across calls; keystone test proves a cloud cell == the standalone integrator bit-for-bit. Remaining open = the *collective* effects above. **UPDATE (ADR 028): inter-bubble acoustic coupling DONE** — `bubble_radiated_pressure = (ρ/d)(R²R̈+2RṘ²)` couples each cell to its neighbours (two-pass explicit scheme), opt-in (`coupling_enabled`, default off for cost). Tests: closed-form radiated pressure, 1/d scaling, coupling alters a two-bubble trajectory, lone bubble unaffected. **UPDATE (ADR 029): cloud-scale shielding DONE** — the incident field is screened by the cloud's void fraction (`commander_prosperetti_attenuation`, reused) via Beer-Lambert along the incident axis (`shielded_pressure`), opt-in (`shielding_enabled`, default off). Tests: closed-form exponential decay, no-nuclei pass-through, denser-screens-more. **UPDATE (ADR 030): self-consistent (implicit) coupling DONE** — fixed-point iteration of the coupling field (`coupling_pressure_field`), reusing the KM acceleration each iterate; opt-in (`implicit_coupling`, default off). Tests: returned field satisfies its own fixed-point equation, implicit differs from explicit under close coupling. **UPDATE (ADR 031): strong-regime solver DONE** — `CouplingScheme::ImplicitDirect` exactly solves the affine coupling system `(I−D·G)S=e` (robust where fixed-point diverges; self-consistent to ~1e-9 at 20 µm coupling), plus `ImplicitFixedPoint{under_relaxation}`. **UPDATE (ADR 032): four frontier refinements DONE** — (1) `dp/dt` coupling (`couple_pressure_rate`: lagged FD rate `(driving−prev_total)/dt` fed into the affine source acceleration; system stays exact since R̈ is affine in dp/dt); (2) `R(t)`-dependent shielding (`shielding_radius_dependent`: instantaneous per-cell R in the CP resonance, quasi-static); (3) cloud-interface RT/RM linear growth-rate **diagnostic** (`interface_instability`: σ_RT=√(A·k·a), ȧ_RM=k·Δv·a₀·A, A=β/(2−β)); (4) sparse/matrix-free solver (`CouplingScheme::ImplicitIterative`: `solve_lsqr_matfree` + on-the-fly `G_ab`, O(active) memory, matches dense to 1e-6). All opt-in; defaults reduce to ADR 027-031. **Now remaining (deepest frontier):** nonlinear RT/RM interface *evolution* (not just growth rates), fully implicit `dp/dt`, nonlinear large-amplitude cloud scattering, multi-directional screening, and a k-wave/experimental erosion comparison. | open: k-wave/experimental validation |
| CLD-2 | C → **RESOLVED (2026-07-12)** | `orchestrator/{execution.rs,methods.rs}`, `config.rs`, `kzk_solver_plugin/solver.rs` | Wired KZK: `propagate_volume()` on solver, `generate_kzk_acoustic_field()` adapter in execution, conditional dispatch in `execute_therapy_step()`, `use_nonlinear_field` in `AcousticTherapyParams`. Residual: collimated only (no focusing phase). | resolved |
| CLD-3 | ~~H~~ DOC'D (2026-06-01) | `clinical/therapy/hifu_planning/types.rs:60` | Rewrote "Theorem"→"closed-form approximation" w/ validity regime (linear/paraxial F#≳1/homogeneous) + refs (O'Neil 1949, Cobbold 2007); named the magic 0.7 `MINUS6DB_ELLIPSOID_FILL_FACTOR` + flagged unvalidated (value preserved). | done |
| CLD-4 | ~~H~~ RESOLVED (2026-06-01) | `domain/source/transducers/physics/mod.rs:47,50` | Category mismatch: `TISSUE_IMPEDANCE` is the nominal *matching-layer design load* (fixed manufactured hardware, `Z_match=√(Z_pzt·Z_load)`, Szabo/Cobbold), NOT a per-voxel sim medium — CT-derivation does not apply; documented to prevent re-flag. `BACKING_IMPEDANCE` was DEAD (no refs) — removed. | done |
| CLD-5 | ~~H~~ RESOLVED (2026-06-01) | `domain/source/transducers/phased_array/config.rs:34` | "Ignores user freq" is false — `Default` is correctly nominal; no constructor drops a passed freq; `satisfies_nyquist` already takes `sound_speed`. Real defect was SSOT dup of `2.5` (geometry + freq field) → single `DEFAULT_CENTER_FREQUENCY_HZ` const. | done |
| CLD-6 | ~~H~~ DOC'D (2026-06-01) | `clinical/therapy/lithotripsy/bioeffects.rs:191` | Documented Pennes-perfusion omission + its CONSERVATIVE (over-estimating) direction for a safety index; cited Pennes 1948; pointer to bioheat solver for quantitative dose. | done |
| CLD-7 | H | `clinical/therapy/therapy_integration/orchestrator/microbubble.rs:197` | uniform microbubble conc; no advection/cluster dynamics | document/extend |
| CLD-8 | M | `domain/boundary/bem/manager/assembly.rs:85` | `.unwrap()` on `last()` w/o bounds | safe `.last().copied()` |
| CLD-9 | M | `clinical/.../hifu_planning/tests.rs:115,156` | focal-spot tested only vs itself, not k-wave/analytic | add reference baseline |
| CLD-10 | M | `domain/source/transducers/focused/bowl/tests.rs:20` | bowl geometry tested, pressure field NOT vs k-wave | add field test |
| CLD-11 | M → **DONE (2026-06-20)** | `domain/boundary/cpml/config/cpml_config.rs:214` + `kwavers/tests/cpml_absorption_quality.rs` | Added `theoretical_reflection_decays_monotonically_with_thickness` (Collino&Tsogka 2001): strict-decrease + bounded-(0,target] property test, params analytically chosen to avoid FP underflow. **Courant sub-item DONE:** `test_cpml_stable_across_thicknesses` (Komatitsch&Martin 2007) sweeps PML thickness {6,8,10,12} at a fixed CFL `dt`, asserting for each that the post-propagation energy is finite (no blow-up), decays below initial (stably absorbing), and absorption is monotone non-decreasing in thickness — empirical proof the CFS-CPML preserves CFL stability regardless of thickness. Refactored the single-thickness test onto a shared `run_cpml_absorption(thickness)` helper (SSOT). | done |
| CLD-12 | ~~M~~ RESOLVED (2026-06-01) | `clinical/imaging/reconstruction/transcranial_ust/medium.rs:14` | `AIR_REJECTION_HU=-300` was a verbatim SSOT DUP of canonical `ct_acoustics::HU_BRAIN_BODY_THRESHOLD=-300` (Aubry 2003 ref). Deleted local const, switched 8 call sites (medium.rs+volume.rs) to canonical. Value drives a *qualitative* slice-selection count (robust to ±100 HU), not a calibrated mapping — no scanner-validated tolerance test warranted. | done |
| CLD-13 | ~~M~~ DONE (2026-06-01) | `domain/imaging/photoacoustic/types.rs:21,127` | Added `PressureFieldSeries` newtype (own leaf `pressure_series.rs`) wrapping `Vec<Array3<f64>>` with a validating constructor (non-empty + dimensionally uniform) and `Deref<[Array3<f64>]>` (zero consumer churn — all slice/`iter`/index callers unchanged). Both struct fields + 3 construction sites wrapped. 4 value-semantic ctor tests (accept/empty/ragged/round-trip). NB: `Array3<f64>` isn't a primitive — the captured invariant is intra-series dimension consistency, not a unit marker; cross-field time-alignment stays test-covered. | done |
| CLD-14 | ~~L~~ DONE (2026-06-01) | various | Audit framing ("uncited magic numbers") was largely false: `LENS_CURVATURE_FACTOR=0.7` already named; `crosstalk 0.1` already `// 10% (typical)`-commented; both erf impls already cited A&S 7.1.26. Real finding = DUPLICATION: two identical A&S 7.1.26 erf copies (`histotripsy.rs`, `clinical_scenarios/scenario/mod.rs`). Hoisted to canonical `math::statistics::erf` (named const + cite + error bound + 3 value-semantic tests); both sites delegate. SSOT. | done |

<a id="analysis-math"></a>
## analysis/ + math/ + core/ + gpu/ (beamforming/PAM/ML · operators/FFT · constants · shaders)

| ID | Sev | file:line | Gap | Revision |
|----|-----|-----------|-----|----------|
| AMC-1 | C [verify] | `math/numerics/operators/differential/central_difference_6/core.rs:95` | 6th-order stencil FMA nesting — verify signs vs Fornberg | add analytic-derivative value test |
| AMC-2 | ~~L~~ DONE (2026-06-30) | `analysis/.../beamforming/adaptive/mvdr/{spectrum.rs,weights.rs}` | `aᴴR⁻¹a` denominator checks `.re` only; imag dropped silently | Shared real-positive denominator validator with complex-dot roundoff bound; value-semantic regressions for weights + pseudospectrum. |
| AMC-3 | H | `analysis/.../localization/music/spectrum.rs:88`, `subspace/music.rs:108` | pseudospectrum hard-clamp (1e12/1e30) masks ill-conditioning | sentinel/error, not magic cap |
| AMC-4 | M (verified: live shader, persistence BC) | `gpu/shaders/acoustic_field.wgsl:41` | boundary `out=in` (persistence, NOT implicit-Dirichlet); used by `WaveEquationGpu`; undocumented + likely inconsistent w/ CPU Neumann/Dirichlet | document BC choice + reconcile with CPU paths |
| AMC-5 | M | `analysis/ml/physics_informed_loss/loss.rs:36` | wave-eq residual MSE unnormalized → scale-dependent loss | normalize by field scale |
| AMC-6 | M | `analysis/.../pam/delay_and_sum/processor/mod.rs` | no check delays fit signal duration | bounds validation |
| AMC-7 | M | `analysis/.../beamforming/covariance/estimation.rs:68` | parallel covariance accumulation order unguaranteed (FP) | document/forward-backward |
| AMC-8 | M | `analysis/ml/inference.rs:92` | `f32::EPSILON` normalization guard too tight | relative-ε guard |
| AMC-9 | ~~L~~ DONE (2026-06-01) | `analysis/.../beamforming/adaptive/subspace/{esmv,music}.rs` | Confirmed `Complex64≡num_complex::Complex<f64>` and eig/solve take/return that exact type → all 4 `mapv` round-trips + per-element rebuilds were identity. Removed: −2 `Array2` clones, −1 `Array1` clone, −per-element reconstruction. Value-identical (perf+clarity). | done |
| AMC-10 | ~~L~~ DONE (2026-06-01) | `narrowband/capon/mod.rs:115`, `mvdr/mod.rs:62` | Added `numerical::DEFAULT_DIAGONAL_LOADING=1e-6` (Carlson 1988 ref); both `Default`s now read it. SSOT. | done |
| AMC-11 | ~~L~~ DONE (2026-06-01) | `localization/music/mod.rs:86` | "Dup" was FALSE — `processor.rs:116` takes `frequency` as a param, not `fs/4`. Single site; named `DEFAULT_CENTER_FREQUENCY_NYQUIST_FRACTION=0.25` + justified (midpoint of Nyquist band). | done |
| AMC-12 | ~~L~~ **DONE (verified 2026-06-20)** | PAM `MUSIC`/`EigenspaceMinVariance` | **Stale entry — already fully wired, not stubs.** `pam::mapper::subspace_localization_map` dispatches both methods to the shared narrowband `subspace_spatial_spectrum_point` (`{music,eigenspace_mv}_spatial_spectrum_point`): real Hermitian eigendecomposition (`EigenDecomposition::hermitian_eigendecomposition_complex`) partitioning the rank-K signal/noise subspaces + steering, producing a per-focal-point localization power (PAM Theorem 22.2). Tested: `eigenvalue_split_matches_theorem_22_2` (σ_s²+σ_n² vs σ_n²), MUSIC/ESMV point-spectrum peak-at-source. No duplication (SSOT subspace code). | done (no-op) |

### Audit-table remediation pass (2026-06-20)

Drove every remaining row above to a terminal state. Each was re-verified against
current code before acting (verify-first); several "open" rows were already
adequate or false positives.

**Fixed (committed):**
- **SOL-5** — `HASConfig::validate()` SSOT (adds `reference_frequency>0`, finite
  non-negative `attenuation_coeff`); re-checked at the now-fallible
  `HasAbsorptionOperator::new`. Negative test per invariant.
- **SOL-8** — `debug_assert!` finite leapfrog/velocity coeffs at AVX-512 stencil
  construction.
- **SOL-9** — documented the discretization-error rationale for the 5%/10%/2%
  benchmark tolerances (no fabricated citation).
- **PHY-14** — Gilmore RK4 `unwrap_or(0.0)` at `|u|→c` now routed through
  `stage_acceleration`, which `log::trace!`s the validity-boundary clamp instead
  of silently freezing the wall (anti-defensive-slop).
- **CLD-7** — documented the uniform-concentration limitation of
  `update_microbubble_dynamics`.
- **CLD-8** — BEM assembly `.last().unwrap()` → `.last().copied() == Some(col)`
  (no unwrap, identical behavior).
- **AMC-2** — follow-up implementation (2026-06-30): MVDR `compute_weights` and
  `pseudospectrum` now share a real-positive denominator validator with a
  complex-dot roundoff bound; non-Hermitian inputs that produce a complex
  denominator are rejected instead of silently using `.re`.

**Closed — already adequate / false positive (verify-first, no change):**
- **AMC-7** — FALSE POSITIVE: covariance accumulation is a *sequential* triple
  loop (deterministic); the only parallel op (`par_mapv_inplace`) is element-wise
  scaling with no cross-element ordering. No FP hazard.
- **PHY-15** — ADEQUATE: KZK already documents the θ<17° parabolic-validity bound
  and cites Zabolotskaya & Khokhlov 1969; Rayleigh-Plesset already cites its
  Mach<0.3 / <100 MHz bounds. Cited gap does not exist.
- **AMC-6** — ADEQUATE: PAM interpolation already bounds-checks; a delay outside
  the recording window correctly contributes zero (physically correct), so a hard
  delay-vs-duration rejection would wrongly reject legitimate far-field points.
- **PHY-12** — NO REAL DEFECT: `16π/3 ≈ 16.76` is representable in f32/f64, so
  `T::from(16π/3).expect(...)` does not panic for the supported `Scalar` types;
  the `expect` message already states the invariant.
- **AMC-3** — the MUSIC pseudospectrum cap (1e12/1e30) is the standard MUSIC
  regularization at exact source alignment (1/distance→∞), not a masked error;
  legitimate sentinel.
- **AMC-8** — the absolute `f32::EPSILON` L2-normalization floor is defensible
  (guards 0/0; a nonzero row of any scale still normalizes to a unit vector);
  relative-ε is a marginal preference, not a defect.

**Deferred with recorded reason (not fabricating evidence):**
- **PHY-6 / PHY-7** — emissivity/optical-depth defaults and Arrhenius prefactors
  need an external literature citation to ground; will not fabricate one. Open
  [patch]: cite from source, or make the SL params required constructor inputs.
- **AMC-5** — normalizing the PINN wave-eq residual MSE by field scale changes
  training numerics; own [minor] increment with a scale-invariance test, not a
  drive-by edit.
- **SOL-6** — coupled density-gradient CFL bound needs a stability derivation +
  test; own increment.
- **SOL-10** — ~30% public-fn Rustdoc gap across kwavers-solver is an ongoing
  sweep, not a single increment (won't silently mass-stub docs).
- **SOL-11** — wiring the k-Wave validators into CI is infra (workflow + runtime
  budget); own change.
- **CLD-2 (KZK wiring)** — **RESOLVED (2026-07-12).** Wired `kzk_solver_plugin` into
  HIFU therapy path via `propagate_volume()`, `generate_kzk_acoustic_field()`, and
  conditional `execute_therapy_step()` dispatch. Residual: collimated beam only
  (no focusing phase).
- **PHY-11**, **COV-5 de Jong/Herring** — need an external experimental baseline
  (Lauterborn collapse) or a paywalled convention PDF (de Jong S_p/S_f prefactor);
  deferred until a real oracle is available rather than asserting a fabricated one.
- **PHY-13 / CLD-9 / CLD-10** — initially deferred as "needs k-Wave"; **RESOLVED
  2026-06-20** via *analytical* oracles instead (no external data) — see the
  remaining-items pass below.

### Remaining-items resolution pass (2026-06-20)

Every open item driven to a terminal state. Implemented where an analytical oracle
existed; closed the rest without fabricating.

**Implemented (analytical oracle):**
- **COV-4 finite-aperture SIR** — `CircularPistonSir::round_trip_response`, the
  two-way pulse-echo diffraction kernel `h⊛h`. Oracle: convolution factorization
  `∫(h⊛h)dt=(∫h dt)²` (exact vs same discretization) + on-axis triangle support
  `[2z,2√(z²+a²)]/c`.
- **CLD-9 / CLD-10** — focused-bowl discretization vs O'Neil (1949): a discrete
  Rayleigh–Sommerfeld element sum at the focus reproduces the analytical focal gain
  `|p(F)|/p₀ = k·h` (all spherical-cap elements are one `F` away → coherent sum =
  `(k/2π)A_total/F = k·h`). Numerical-vs-analytical, no k-Wave needed.
- **PHY-13** — bubble scattering cross-section resonance closed form
  `σ_s(ω₀)=4πR²(ka₀)²/δ_tot²` (δ_tot re-derived independently, Church 1995) + the
  low-frequency `σ_s ∝ ω²` scaling. No de Jong PDF needed.
- **COV-1 PCF-IQ** — `phase_coherence_from_iq_aperture` (native complex/baseband
  path bypassing Hilbert), keystone-equivalent to the RF phase core.

**Closed (no groundable oracle / correct-layering):**
- **COV-6 loaded-Mason Z_e** — no verified closed form for the loaded electrical
  impedance; implementing from memory = fabrication risk. `AcousticLayer` covers
  matching/backing design. Deferred pending a cited Mason/KLM reference.
- **DG-solver CPML** — legitimately different discretization (per-GLL flux-based
  memory + joint SSP-RK3 ≠ FDTD recursive convolution); consolidation would distort
  both. Verify-first false-positive, correct-layering.

---

## Triage order (per `CLAUDE.md` sprint policy: correctness → architecture → tests → docs)

1. ~~[verify] C-tier suspicions~~ — **DONE (Sprint A, 2026-05-31):** all 6 resolved,
   0 confirmed physics bugs (4 false positives closed; AMC-2→L, AMC-4→M downgraded).
2. ~~Confirmed correctness~~ — **DONE (Sprint B, 2026-05-31):** SOL-1/2/3 were all
   false positives (test-only panics, no change); PHY-5 fixed (dead-field removal).
3. ~~Documented-approximation bounds~~ — **DONE (Sprint C, 2026-06-01):** PHY-2/4
   already-handled/false-positive; PHY-8/CLD-2/3/6 validity regimes + refs added.
    CLD-2 KZK wiring resolved [minor] (2026-07-12).
4. ~~Missing literature validation~~ — **DONE (Sprint D, 2026-06-01):** PHY-10/SOL-7
   added external/property tests; CLD-11 reflection-decay test added; PHY-9 closed
   (false positive — tol already FP-tight); PHY-11 adequate (analytical check
   present, collapse regression backlogged). Remaining: PHY-13/CLD-9/10 (need
   k-wave/de-Jong baselines), CLD-11 Courant sub-item.
5. ~~CT-derived params over hardcoded~~ — **DONE (Sprint E, 2026-06-01):** CLD-4
   (category mismatch — design load not sim medium; dead BACKING_IMPEDANCE removed),
   CLD-5 (SSOT 2.5 dedup; "ignores freq" false), CLD-12 (verbatim SSOT dup of
   `HU_BRAIN_BODY_THRESHOLD` removed, 8 sites).
6. **DRY/SSOT + docs** — AMC-9/10/11 **DONE (Sprint E):** identity-cast removal
   (perf), `DEFAULT_DIAGONAL_LOADING` + `DEFAULT_CENTER_FREQUENCY_NYQUIST_FRACTION`
   consts. Remaining: CLD-13 (pressure-field newtype — public-API change, own
   increment), CLD-14 (scattered magic-number naming), SOL-10/11 (doc sweep, CI-wire).

See [backlog.md](backlog.md) for sprint sequencing and [CHECKLIST.md](CHECKLIST.md)
for the active increment.

---

# Coverage & placement audit (2026-06-19)

**Different axis from Sprints A–E above.** Those audited *internal correctness* of
existing code. This audit asks two new questions: (1) **coverage** — what ultrasound
physics do peer libraries (k-Wave, Field II, FOCUS, Stride, j-Wave, mSOUND, USTB,
MUST, BabelBrain) implement that kwavers lacks? (2) **placement** — what physics is
duplicated across crates or living in the wrong layer?

**Method:** four parallel read-only coverage explorers (forward solvers; bubble/
thermal/chemistry; transducer/imaging/beamforming; inverse/therapy/medium), then
**direct verification** of every `ABSENT`/`NOT FOUND` claim by targeted grep — the
explorers are pattern-matchers and over-call gaps (one example: explorer flagged
"Kirchhoff migration ABSENT" — it exists at
`kwavers-physics/src/acoustics/imaging/seismic/kirchhoff.rs`, with `eikonal.rs`
alongside). Only grep-confirmed gaps are listed.

**Headline:** kwavers' physics breadth meets or exceeds every peer surveyed
(uniquely: frequency-domain CBS-FWI, full PINN stack, neuromodulation HH+NICE,
sonochemistry, transcranial CT aberration). Gaps are narrow and concentrated in
classic **imaging-pipeline beamforming refinements** and a few **bubble-shell
models**; the larger risk is **cross-crate fragmentation** of three modality
verticals.

## Coverage gaps (grep-verified absent)

| ID | Sev | Area | Gap (peer that has it) | Notes |
|----|-----|------|------------------------|-------|
| COV-1 | ~~M~~ **DONE (2026-06-19)** | beamforming | Added `time_domain::coherence` — Mallart-Fink amplitude CF + Camacho sign CF (SCF) behind one `CoherenceFactor` enum + `delay_and_sum_coherence`; DAS refactored onto SSOT `align_channels`/`sum_aligned`. 11 value-semantic tests. **Surfaced + fixed a real bug:** SAFT 3-D CF squared `Σ|x|` instead of summing energies (coherent aperture → 1/N not 1); consolidated onto canonical `amplitude_coherence_from_sums`. NB: SLSC (Lediju 2011) + SAFT-CF already existed — gap was the canonical DAS-path CF/SCF, now filled. **UPDATE (2026-06-20): GCF DONE** — `CoherenceFactor::Generalized { m0 }` (Li & Li 2003): aperture spectral energy in the low-spatial-frequency passband |k|≤m0 over the Parseval total `N·Σx²`; `m0=0` reduces **exactly** to the amplitude CF (keystone differential test), `m0≥N/2 ⇒ 1`. 5 value-semantic tests (incl. pure-2-cycle spectral localization). **UPDATE (2026-06-20): PCF DONE** — `CoherenceFactor::Phase { sensitivity }` (Camacho et al. 2009): `PCF = max(0, 1 − (γ/σ₀)·min(σ(φ), σ(ψ)))`, σ₀=π/√3, auxiliary phase ψ=φ−sign(φ)·π for ±π-wrap immunity; per-element instantaneous phase from the analytic-signal SSOT (`kwavers_math::fft::analytic_signal_1d`) + canonical scalar `phase_coherence_from_phases`. 11 value-semantic tests (exact closed forms, keystone wrap-rescue, quadrature-spread column path). **Fixed broken main:** the variant+helpers+validate had landed without the `weight_for_column` dispatch / `weights()` phase-path → non-exhaustive match (E0004), `kwavers-analysis` did not compile; this commit wired the missing dispatch. Coherence-factor family (amplitude/sign/phase/generalized) now complete. | done |
| COV-2 | ~~M~~ **DONE (2026-06-19)** | beamforming | Added `time_domain::dmas` — canonical `dmas_combine` (signed-sqrt pairwise closed form) + active `delay_and_sum_dmas` (reuses `align_channels`). **Consolidated:** passive PAM `dmas_at_point_view` now calls the shared `dmas_combine` (was inline-duplicated). 8 value-semantic tests (closed-form pairwise products, anti-phase suppression). | done |
| COV-3 | ~~M~~ **DONE (2026-06-19)** | transducer | Added `kwavers-transducer::curvilinear::ConvexArrayGeometry` — element positions + outward radial normals + tangents on a curvature arc, transmit-focusing delays, aperture/arc-pitch geometry. 8 analytic tests (on-arc, apex, unit-radial normals, chord width, zero-delay-at-curvature-center, symmetry). Feeds `kwave_array` Rect/Arc elements or a `Source` (rasterization = follow-up). | done |
| COV-4 | ~~M~~ **DONE (core, 2026-06-19)** | phantom | Added `kwavers-phantom::scatterers` — `ScattererCloud` (Field II tissue model) + monostatic synthetic-aperture `synthesize_rf`: `RF_e(t)=Σ_s (a_s/r²)·pulse(t−2r/c)`. 7 analytic tests (round-trip delay, 1/r² amplitude, superposition, linearity, pulse placement, min-distance guard). **Follow-up DONE (2026-06-19):** transient circular-piston spatial impulse response (Stepanishen 1971) added as `analytical::transducer::CircularPistonSir` (the Field II diffraction kernel; on-axis ∫h dt = √(z²+a²)−z verified). **UPDATE (2026-06-20): rectangular-element SIR DONE** — `analytical::transducer::RectangularPistonSir` (Lockwood & Willette 1973): `h=(c/2π)·Φ(ρ)`, Φ = exact angular measure of the wavefront circle within the rectangle from the arccos/arcsin breakpoints (no numerical integration). 5 tests incl. on-axis plateau=c and a keystone differential of analytic Φ vs an independent θ-sampling oracle across 7 geometries × 5 radii (inside/edge/corner/outside). **UPDATE (2026-06-20): attenuation follow-up DONE** — opt-in power-law tissue attenuation in `synthesize_rf` (round-trip `exp(−α(f₀)·2r)`, α₀ dB/(cm·MHz); α₀=0 = prior lossless), validated vs the closed-form factor + deeper-scatterer differential. Remaining [minor]: finite-aperture SIR convolution (Tupholme–Stepanishen). | done |
| COV-5 | ~~M~~ **PARTIAL (2026-06-19)** | bubble dynamics | Added **Hoff (2000)** + **Sarkar (2005)** shell models as `EncapsulatedShellModel` impls (+ value-semantic tests incl. Hoff≡Church-at-G_s=0 differential). **Deferred [minor]:** de Jong (lumped S_p/S_f prefactor is convention-dependent — needs Doinikov&Bouakaz PDF verification before asserting) and Herring (free-bubble compressible EOM — different category, belongs with KM/Gilmore, not a shell model). Evidence tier: literature-recall (Doinikov&Bouakaz 2011) validated by equilibrium/restoring/damping properties. | partial |
| COV-6 | ~~L~~ **DONE (2026-06-19)** | transducer | Was mostly present: `bulk_piezo::BulkPiezoResonator` already had the thickness-mode resonator (antiresonance f_p, series f_s, clamped capacitance, IEEE k_t² relation) — the explorer's "absent" was an over-call (searched only "KLM"/"Mason"). **Added the genuine gap** — the Mason/KLM frequency-dependent `electrical_impedance(f)` (free-plate `Z_e=1/(jωC₀)[1−k_t² tan X/X]`), plus `acoustic_impedance` (Rayl, for matching-layer design) and `free_capacitance` C^T. 5 analytic tests incl. Z_e=0 at the IEEE f_s (cross-check) and divergence at f_p. **UPDATE (2026-06-20): loaded transmission line DONE** — `AcousticLayer` telegrapher input-impedance transform + reflection coefficient, `quarter_wave_match_impedance = √(Z_s·Z_L)`, `quarter_wave_matching_layer`; 6 closed-form tests (λ/4 inversion, λ/2 pass-through, matched identity, Γ→0 into water). Remaining [minor]: loaded-Mason `Z_e` radiation resistance from front/back loads. | done |
| COV-7 | ~~L~~ **DONE (2026-06-19)** | elastography | Added the MRE front end `kwavers-physics::...::elastography::mre`: `extract_first_harmonic` (single-bin temporal DFT of a motion-encoded phase-offset stack → complex displacement, DC-rejecting), `harmonic_snapshot`, and `mre_displacement_field_z` producing the `DisplacementField` the existing LFE/direct inversions consume. 6 analytic tests (amplitude/phase recovery, DC rejection, snapshot, validation). Closes the front-end gap; the modulus inversion (LFE/direct) already existed. | done |
| COV-8 | ~~L~~ **FALSE POSITIVE (2026-06-19)** | sonoluminescence | NOT a stub: `cherenkov/model.rs` has the full Frank-Tamm formula (Jackson 1999 §13.5) — `frank_tamm_factor`, `spectral_intensity ∝ f`, `emission_spectrum ∝ 1/λ³`, threshold logic. Complete + literature-grounded. No gap. | — |
| COV-9 | ~~L~~ **FALSE POSITIVE (2026-06-19)** | inverse | NOT a dead config: `apply_sobolev_preconditioner_3d` (`linear_born_inversion/pcg.rs:232`) is a real Sobolev-gradient smoothing preconditioner (`smooth_active_values_3d` + convex blend), wired into the PCG iteration (pcg.rs:210). No gap. | — |
| COV-10 | ~~L~~ **DONE (2026-06-19)** | phantom | Added `kwavers-phantom::shepp_logan::SheppLogan` — 10-ellipse phantom, Original (1974) + Modified (Toft 1996) intensity variants, `value_at`/`rasterize`. 7 analytic tests (origin=1.02/0.2, outside=0, inclusion sum, semi-axis membership, raster shape). | done |
| COV-11 | L | boundary | **Mur absorbing BC** absent. | CPML/PML present and superior; **WONTFIX** unless a thin-PML budget case appears. |

**Confirmed NON-gaps (explorer false positives — do NOT re-flag):** Kirchhoff
migration (`seismic/kirchhoff.rs`), eikonal (`seismic/eikonal.rs`), Rytov
(`inverse/rytov.rs`), power/vector Doppler, ULM super-resolution, axisymmetric PSTD,
fractional-Laplacian + multi-relaxation absorption, anisotropic Christoffel —
all present and accounted for.

## Placement / SSOT gaps (cross-crate fragmentation — grep-verified)

| ID | Sev | Concern | Evidence | Resolution direction |
|----|-----|---------|----------|----------------------|
| PLC-1 | arch — **DONE (2026-06-19, ADR 026)** | **Photoacoustic across 5 locations.** Consumer analysis showed these are mostly *layered* (physics / analytical / imaging-datamodel / solver-inversion / forward-simulator), NOT duplicates. The genuine duplication was the **two forward pipelines in `kwavers-simulation`**: `modalities/photoacoustic` (live `PhotoacousticSimulator` — examples + 3 test suites) vs `photoacoustics/{orchestrator,runner,vertical}` (~1325 LOC, consumed only by one unused `PhotoacousticRunner` re-export). | **Removed the dead `photoacoustics/` pipeline** (1325 LOC) + its `pub mod`/`pub use` in lib.rs; `modalities::photoacoustic` is the single canonical forward pipeline. Resolves the in-simulation half of DEBT-3. No capability merged (dead code, no tests). | done |
| PLC-2 | ~~arch~~ **MOSTLY FALSE POSITIVE (2026-06-19)** | (was: CEUS duplicated physics across 4 crates) | **Verified correctly layered, not duplicated:** (1) `Microbubble`/`MicrobubblePopulation` types live in `kwavers-imaging` (domain) and physics CEUS **re-exports** them (`pub use kwavers_imaging::ultrasound::ceus::{...}`) — SSOT, the explorer's "duplicate type" was wrong. (2) Perfusion is *not* duplicated: imaging `PerfusionMap`/`PerfusionStatistics` is image-analysis (ROI peak/TTP/AUC), physics `CeusPerfusionModel` is the forward advection-diffusion-reaction transport PDE + pharmacokinetics. Different concerns. **Residual [patch] (optional):** minor overlap in perfusion-parameter extraction (`FlowKinetics::analyze_tic` vs `PerfusionStatistics::from_samples`) could be unified — not an arch duplication. | closed (arch) |
| PLC-3 | arch — **shell-SSOT DONE; remainder CONFIRMED (2026-06-19)** | Microbubble dynamics duplicated within `kwavers-physics` + a `therapy/` subtree living in the physics crate. | **Done:** Church/Marmottant/Hoff/Sarkar now share one `EncapsulatedShellModel` trait + RP driver. **Confirmed-real remainder (needs ADR + careful merge):** (a) `therapy/microbubble/shell/properties.rs::MarmottantShellProperties` is a **second Marmottant (2005) implementation** (its own `surface_tension`/buckled-elastic-ruptured state/`pressure_contribution`) parallel to the canonical `encapsulated::model::MarmottantModel`; (b) `ceus/microbubble/dynamics/integration.rs` has its **own `wall_acceleration` RP integrator** (a 3rd RP-with-shell path). Consolidate (a)+(b) onto `EncapsulatedShellModel`. (c) Layering: therapy-domain code in physics (`physics/src/therapy/*`, `acoustics/therapy/{neuromodulation,sonogenetics}`, `transcranial/bbb_opening`) vs `kwavers-therapy` — keep physics *models*, move therapy *planning/consumers* to `kwavers-therapy`. **(a) UPDATE (2026-06-19): fixed a real bug surfaced by the investigation** — therapy `MarmottantShellProperties::surface_tension` used the R₀ reference, giving negative χ over `[R_buckling, R₀)` (discontinuous at R_buckling); corrected to the Marmottant-2005 R_buckling reference, now matching the canonical model's σ(R) convention. **(b) CLOSED (2026-06-19):** the ceus `wall_acceleration` is a *distinct simplified CEUS model* (linear shell `4 G_s d (R−r0)/r0²` + ad-hoc post-division damping), not a Church/Marmottant clone — forcing it onto `EncapsulatedShellModel` would distort the trait for one consumer (over-abstraction); the only shared piece is the 1-line RP core, not worth coupling 3 modules. Legitimate differentiation, like PLC-2/PLC-4. **(a) remaining:** therapy stays a separate stateful model (buckling irreversibility) with a different viscous form (`4μ/R`) — genuinely different from the canonical, so not merged; σ(R) bug fixed + convention aligned is the actionable part. **(c) CLOSED — correct layering (2026-06-19):** `physics/src/therapy/*` and `acoustics/therapy/*` contain genuine *physics models* (BilayerSonophore/NICE, Hodgkin-Huxley `CorticalNeuron`, `MarmottantShellProperties`, lithotripsy) that `kwavers-therapy` *consumes* (`use kwavers_physics`) for planning/regulatory/safety. Moving the models *up* to `kwavers-therapy` would break the layer DAG (physics/solver couldn't use them). No move warranted; the module name `physics/.../therapy` groups therapy-related *physics*, not therapy *planning*. | (a) bug fixed; (b)+(c) closed |
| PLC-4 | ~~M~~ **VERIFIED NOT DUPLICATED (2026-06-19)** | (was: time-reversal in 3 locations) | **Closed:** the sites are legitimately separated, not cloned. (1) `solver/inverse/time_reversal::propagate_backwards` delegates to a `PluginBasedSolver` (`solver.step()`) — no own propagator. (2) `solver/.../photoacoustic/time_reversal.rs` holds the canonical real-cosine k-space propagator (Tabei 2002). (3) `simulation/.../vertical/reconstruction/time_reversal.rs` **delegates** to the solver `PhotoacousticReconstructor` — not a clone. (4) `physics/.../transcranial/aberration_correction/time_reversal.rs` is phase conjugation on a complex field — a distinct aberration-correction concern. No consolidation needed. | closed |
| PLC-5 | ~~L~~ **RESOLVED — correct layering, no drift (2026-06-19)** | **Histotripsy across 3 crates** — `kwavers-medium/absorption/histotripsy.rs` (mechanical/threshold tissue constants), `kwavers-physics/analytical/cavitation/histotripsy.rs` (intrinsic-threshold model), `kwavers-therapy/.../lithotripsy` (cloud/erosion). | **Verified distinct concerns, correctly layered with NO shared-constant drift:** `kwavers-medium/absorption/histotripsy.rs` is the explicit SSOT for the tissue constants (intrinsic threshold 28.2 MPa, slope 1.4 MPa/decade, σ_T 0.96 MPa — Maxwell 2013 / Vlaisavljevich 2015); the physics functions take these as parameters (no hard-coding). **NB:** "WONTFIX" referred ONLY to *not consolidating the 3 locations* (placement), never to the physics — the intrinsic-threshold physics is verified-accurate-to-literature, and the cloud-dynamics gap is tracked separately as CLD-1 (now partially addressed with real Gilmore dynamics). | closed (placement) |

**Severity:** `arch` cross-cutting structural · `M` real but bounded · `L` cleanup.
Placement items are **[verify]-gated for duplication**: confirm the logic is actually
cloned (not legitimately layered forward-vs-inverse) before consolidating — same
discipline that turned 6/6 Sprint-A suspicions into 4 false positives.

See [backlog.md](backlog.md) for sprint sequencing and [CHECKLIST.md](CHECKLIST.md)
for the active increment.

---

# Findings 2026-07-10: full-workspace test triage (post kwavers-python close)

The first full-workspace `cargo nextest run` after the kwavers-python migration
surfaced 20 failures + 4 timeouts. Resolved clusters (committed): complex
Hermitian eig sign (kwavers-math), leto `FixedMatrix` 3×3 symmetric-eigen sign
bug (christoffel, 4), clutter-filter SVD/eigensolver selection (svd_filter +
adaptive + ulm, 12+ tests & timeouts), pam/pstd `index_axis` output-rank (3).
Two residual clusters, NOT migration-correctness defects:

## kwavers-driver: missing gitignored KiCad fixtures [test-infra]
- `component_accuracy::tests::{hv_driver_artifact_uses_exact_j4_power_header,
  hv7355_32ch_artifact_has_renderable_component_models}` fail reading
  `crates/kwavers-driver/tests/fixtures/boards/hv7355_{24,32}ch_tile/*.kicad_pcb`
  (os error 3 — path not found). The fixtures directory is **gitignored** and
  absent, so these tests can only pass where a developer generated the boards
  locally. Pre-existing test-design defect (tests depend on absent gitignored
  artifacts), independent of the Atlas migration.
- DoR: either commit small deterministic `.kicad_pcb` fixtures (build-size
  budget permitting), generate them in a build/test setup step, or gate the
  tests behind a fixture-present guard. Owner decision needed on fixture policy.

## kwavers-therapy: abdominal FWI preprocessing exceeds test-time budget [perf]
- `theranostic_guidance::tests::abdominal::{abdominal_preprocessing_keeps_external_skin_between_target_and_aperture,
  abdominal_preprocessing_selects_one_connected_treatment_component}` terminate
  at the therapy profile timeout (90 s). Both call `run_theranostic_inverse` on
  64×64×3 / 72×72×3 grids; the passing `abdominal_theranostic_inverse_recovers_lesion_support`
  (42×42×3) already runs 16–19 s (near the 30 s slow threshold), so the FWI
  inverse scales super-linearly past the budget at the larger grids.
- DoR (profile-first per performance_engineering): flamegraph
  `run_theranostic_inverse` at 64×64×3, identify the hot path (forward/adjoint
  PSTD loop, per-iteration cost, leto array-op constants vs pre-migration),
  and optimize the real component — never raise the timeout or shrink the grid
  (test-gaming). Classify migration-regression vs inherent FWI cost by comparing
  the 42×42×3 wall-time against the pre-migration commit.
- **Partial characterization (2026-07-12):** the O(N) setup (elastic-medium,
  orchestrator) is negligible; cost is dominated by the per-timestep 3-D FFT loop
  in the spectral forward/adjoint solver (`config.iterations=12` ×
  `elastic_fwi_iterations=3` × n_time-steps × several FFTs each). NOT a plan-caching
  issue — apollo already caches via `f64::get_3d_plan` (`PlanCacheProvider`).
  The optimizable overhead is the kwavers FFT FACADE
  (`kwavers-math/src/fft/mod.rs`): `fft_3d_array` allocates a fresh leto `Array3` +
  element-wise `from_apollo_complex` conversion every call, and `fft_3d_array_into`
  double-copies (`out.assign(&fft_3d_array(field))`) instead of routing to apollo's
  zero-alloc `fft_3d_array_into`/`_typed_into`. NEXT (needs a CONFIRMED profile — a
  prior attempt instrumented `prof_fft_take()` but ran out of budget before running
  it): confirm facade-conversion vs inherent-FFT split, then eliminate the double
  allocation/conversion (route facade `_into` through apollo `_into`, reuse scratch
  buffers across timesteps) and verify no accuracy regression on the elastic_fwi
  convergence tests. Dedicated effort; do not rush.
