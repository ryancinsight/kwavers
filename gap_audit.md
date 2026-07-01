# Gap Audit

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
- **Remaining workspace Rayon/Tokio usage — OPEN [patch].** Root workspace
  dependencies and non-core crates still contain direct `rayon`/`tokio` usage.
  Next closure increment: audit call sites by crate, replace the smallest
  provider-owned edge with Moirai, and keep any missing Moirai capability in
  Moirai rather than duplicating it downstream.

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
  added orchestrator-level note. **Open [minor]:** wire KZK into HIFU path. Partial.
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
| CLD-2 | C → **M DOC'D (2026-06-01)** | `orchestrator/{execution.rs:61,methods.rs:execute_therapy_step}` | `generate_acoustic_field` already documented its linear limitation; added orchestrator-level note (used by ALL modalities incl. HIFU; KZK exists, unwired). **Open follow-up:** wire `kzk_solver_plugin` into the HIFU therapy path [minor]. | partial |
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
- **CLD-2 (KZK wiring)** — routing the KZK plugin into the HIFU therapy path is a
  ~50–100 LOC [minor] with a return-type adapter; documented limitation already
  in place.
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
   Open follow-up: CLD-2 KZK wiring [minor].
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
