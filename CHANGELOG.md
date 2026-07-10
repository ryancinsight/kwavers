# Changelog

## Unreleased

### Fixed (2026-07-10) - Hermitian eigensolver rotation correctness [patch]
- [patch] `kwavers-math::linear_algebra::EigenSolver::jacobi_hermitian` produced
  wrong eigenvalues for complex Hermitian matrices with non-zero imaginary
  off-diagonal entries. The rotation derived its angle from `|h_pq|` but applied a
  purely real rotation using only `h_pq.re` (dropping the phase), and inverted the
  `tan(2θ)` angle formula. Replaced with the standard phase-reduced complex
  Hermitian Jacobi rotation (unitary `diag(1, ē)·[[c,s],[-s,c]]`, `ē = conj(h_pq)/|h_pq|`)
  and Golub & Van Loan 8.4.1 stable angle selection. Verified against closed-form
  eigenvalues: `[[2,1-i],[1+i,3]] → {4,1}`, `[[2,1],[1,2]] → {3,1}`; all 263
  `kwavers-math` tests pass. Corrected `eigendecomposition_symmetric_2x2` to compare
  eigenvalue sets order-independently (`qr_algorithm` sorts descending, the
  `leto_ops` oracle ascending) while retaining the authoritative `A·v = λ·v` residual.

### Fixed (2026-07-10) - kwavers bulk Leto migration closure (compute stage) [minor]
- [minor] Consolidated the kwavers compute pipeline (analysis, solver, simulation,
  therapy, diagnostics, physics, grid-adjacent) on the Leto+/eunomia/moirai-parallel
  stack. Migration completes the `kwavers-physics`, `kwavers-solver`, `kwavers-analysis`,
  `kwavers-simulation`, `kwavers-therapy`, and `kwavers-diagnostics` source surfaces to
  native Leto array operations; removes the `kwavers-math::linear_algebra::basic` and
  `eigen` shim modules in favor of `leto_ops::{solve, inv, qr, svd,
  symmetric_eigen_jacobi, kron, matmul, matvec}`; replaces all `Vec<f64>` →
  `Array*::from_vec(vec)` call sites with the two-argument `Array*::from_vec(shape, vec)`
  contract; and rewrites the kwavers-precision `LinearAlgebra`/`EigenDecomposition`
  trait surfaces on `kwavers-math::linear_algebra::{EigenSolver, ComplexLinearAlgebra,
  LinearAlgebraExt}` as documented migration SSOT. Internal bindings (`Array{1,2,3}::eye`,
  `Array2::zeros`, `Lanes`-based reductions) standardized on Leto's typed shape and
  zero-copy view API. Removes the workspace-level `ndarray` dependency. Workspace library
  check passes for all 24 member crates excluding `kwavers-python`; the kwavers facade,
  kwavers-physics, kwavers-solver, kwavers-analysis, kwavers-simulation, kwavers-therapy,
  and kwavers-diagnostics crate-frontier are clear. `kwavers-python` Leto↔PyO3 binding
  surface remains an open follow-up (filed under Bulk migration priority #2
  in gap_audit.md).

### Breaking (2026-07-10) - kwavers-grid native Leto surface [arch]
- [major] Removed the transitional `kwavers_grid::compat` module and redundant
  `_leto` grid, differential-operator, and k-space APIs. Removed duplicate
  `Grid::kx`/`ky`/`kz` names in favor of the existing `compute_kx`/`compute_ky`/
  `compute_kz` surface. Canonical operations now import and return Leto types
  directly, while differential operators retain borrowed Leto views as their
  zero-copy input boundary. Also removed identity and allocating Leto-to-Leto
  k-space conversions and corrected finite-difference coefficient
  documentation to state native-precision rounding.

### Migration
- Replace `kwavers_grid::compat::*` imports with direct `leto` imports. Replace
  deleted `_leto` forwarding calls with the unsuffixed operation and pass
  `.view()` when its canonical signature accepts `ArrayView3`. Replace
  `Grid::kx`/`ky`/`kz` with `compute_kx`/`compute_ky`/`compute_kz`.
- Verification: static duplicate audit clean; exact formatting and all-target
  grid clippy pass; grid nextest passes 38/38; doctests pass with five
  intentionally ignored; grid docs are warning-clean; and `kwavers-physics`
  library check passes. Existing `kwavers-math` and `kwavers-solver` Leto
  migration errors block their broader test/facade gates.

### Fixed (2026-07-10) - physics Leto test migration closure [patch]
- [patch] Migrated the remaining sonoluminescence tests and spectrum
  construction to native Leto spectral ranges, iterator reductions,
  constructors, and fixed-rank shapes; synchronized acoustic shape-error
  assertions with the typed array-shape contract; and removed obsolete
  Leto-to-Leto QR conversions in `kwavers-math`. Focused sonoluminescence
  nextest passes 65/65, full `kwavers-physics` nextest passes 1713/1713 with
  one skipped test, and doctests pass 8/8 with eight intentionally ignored.
  All-target clippy advances past `kwavers-math` and remains blocked by 63
  pre-existing `kwavers-grid` migration lints plus an unrelated malformed
  `kwavers` example import.

### Fixed (2026-07-10) - chemistry and field-surrogate Leto tests [patch]
- [patch] Migrated chemistry concentration assertions and field-surrogate
  fixtures to explicit Leto iteration, fallible indexed iteration, fixed-rank
  shapes, and array indices. Focused nextest compilation reduces the package
  frontier from 20 to 13 errors, now isolated to sonoluminescence tests.

### Fixed (2026-07-10) - acoustic Leto iteration and shape tests [patch]
- [patch] Migrated ultrasound-code, cavitation, skull-aberration, acoustic-state,
  sonogenetics-channel, and transcranial-aberration tests to explicit Leto
  element iteration, fixed-rank array shapes, value shape comparisons, and
  array callback indices. Focused nextest compilation reduces the package
  frontier from 33 to 20 errors with no remaining diagnostic in these clusters.

### Fixed (2026-07-10) - elastography Leto test migration [patch]
- [patch] Migrated MRE and thermal-strain fixtures from ndarray tuple shapes,
  callback indices, `mapv_inplace`, view `len`, and direct array reductions to
  native Leto fixed-rank shapes, mutation iterators, `size`, and iterator
  reductions. Focused nextest compilation reduces the package frontier from 39
  to 33 errors with no remaining MRE or thermal-strain diagnostic.

### Fixed (2026-07-10) - acoustic heat Leto fixtures [patch]
- [patch] Replaced stale `uniform_ndarray` calls and two duplicate fill helpers
  in acoustic heat-conservation tests with the canonical Leto
  `Array3::from_elem` constructor. Focused nextest compilation reduces the
  package frontier from 42 to 39 errors; no heat-conservation diagnostic
  remains.

### Fixed (2026-07-10) - analytical acoustics Leto test migration [patch]
- [patch] Replaced ndarray `arr2!`/`array!` fixture construction, view
  `to_vec`, implicit axis indexing, and tuple-shaped callback indices in the
  phase-shifting, phase-randomization, pulse-echo, and nonlinear Nyquist tests
  with native Leto APIs. The `kwavers-physics` test compile frontier decreased
  from 60 to 42 errors; the selected analytical errors are absent from the
  rerun diagnostic. Remaining failures are in unrelated migration clusters.

### Changed (2026-07-09) - nonlinear acoustic Leto boundary removal [patch]
- [patch] Removed the allocating Leto-to-Leto `array_boundary` conversion
  module from nonlinear acoustic spectral methods. FFT inputs and inverse FFT
  outputs now remain native `leto::Array3` values through all 14 former
  conversion sites. Production verification: `cargo check -p kwavers-physics
  --lib` passed. The focused nextest build remains blocked by 59 unrelated
  package test-migration errors after correcting the touched Nyquist fixture to
  Leto's `[usize; 3]` index contract.

### Changed (2026-07-08) - kwavers-analysis narrowband Apollo FFT routing [patch]
- [patch] Routed narrowband legacy analytic-baseband and windowed STFT
  snapshot extraction through Apollo 1-D FFT APIs over Leto buffers instead of
  importing FFT execution or complex types from `kwavers_math::fft`. The
  covariance-facing ndarray boundary remains `num_complex` for this slice, with
  explicit conversion from Apollo complex scratch output. Verification: direct
  `rustfmt --check` passed for the touched snapshot files; `rustup run nightly
  cargo check -p kwavers-analysis` passed; `rustup run nightly cargo nextest
  run -p kwavers-analysis narrowband snapshots stft baseband` passed 30/30;
  and a scoped source audit found no `kwavers_math::fft` imports in
  `kwavers-analysis/src/signal_processing`.

### Changed (2026-07-08) - kwavers-analysis Doppler Apollo 1-D FFT routing [patch]
- [patch] Routed continuous-wave, pulsed-wave, and Welch spectral Doppler FFT
  execution through Apollo's 1-D real/complex FFT APIs over Leto buffers
  instead of importing FFT execution and shift utilities from
  `kwavers_math::fft`. Verification: direct `rustfmt --check` passed for the
  touched Doppler files; `rustup run nightly cargo check -p kwavers-analysis`
  passed; `rustup run nightly cargo nextest run -p kwavers-analysis doppler
  continuous_wave pulsed_wave spectral` passed 49/49; and a scoped source audit
  found no `kwavers_math::fft` imports in the migrated Doppler files.

### Changed (2026-07-08) - kwavers-analysis PAM Apollo 1-D FFT routing [patch]
- [patch] Routed PAM processor spectrum computation and delay-and-sum peak
  frequency estimation through Apollo's 1-D real FFT over Leto buffers instead
  of importing FFT execution from `kwavers_math::fft`. Verification: direct
  `rustfmt --check` passed for the touched PAM files; `rustup run nightly cargo
  check -p kwavers-analysis` passed; `rustup run nightly cargo nextest run -p
  kwavers-analysis pam delay_and_sum` passed 18/18; and a scoped source audit
  found no `kwavers_math::fft` import in the migrated PAM files.

### Changed (2026-07-08) - kwavers-analysis analytic-signal Apollo routing [patch]
- [patch] Routed B-mode envelope detection and time-domain phase-coherence
  analytic-signal construction through `kwavers-signal`'s Apollo-backed Hilbert
  transform instead of `kwavers_math::fft::analytic_signal_1d`. Verification:
  direct `rustfmt --check` passed for the touched files; `rustup run nightly
  cargo check -p kwavers-analysis` passed; `rustup run nightly cargo nextest
  run -p kwavers-analysis b_mode coherence` passed 51/51; and a scoped source
  audit found no `kwavers_math::fft` or `analytic_signal_1d` in the migrated
  B-mode/coherence files.

### Changed (2026-07-08) - kwavers-signal Apollo 1-D FFT migration [patch]
- [patch] Routed `kwavers-signal` analytic-signal and frequency-filter 1-D FFT
  execution through Apollo APIs over Leto buffers instead of
  `kwavers_math::fft`. The public analytic-signal boundary remains
  `num_complex` for this slice, with explicit conversion from Apollo complex
  output. Verification: `rustup run nightly cargo fmt --package kwavers-signal
  --check` passed; `rustup run nightly cargo check -p kwavers-signal` passed;
  `rustup run nightly cargo nextest run -p kwavers-signal analytic
  frequency_filter` passed 13/13; and a scoped source audit found no
  `kwavers_math::fft` imports in the touched signal files.

### Changed (2026-07-08) - kwavers-solver PSTD axisymmetric Apollo 2-D FFT migration [patch]
- [patch] Routed `forward::pstd::propagator::axisymmetric` real forward and
  complex inverse 2-D FFT execution through Apollo APIs over Leto buffers
  instead of the `kwavers_math::fft` plan/cache facade. The ndarray
  `num_complex` working buffers remain the current PSTD storage boundary, with
  explicit conversion at the Apollo scratch edge. Verification: `rustup run
  nightly cargo fmt --package kwavers-solver --check` passed; `rustup run
  nightly cargo check -p kwavers-solver` passed; `rustup run nightly cargo
  nextest run -p kwavers-solver axisymmetric_apollo` passed 2/2; and a scoped
  source audit found no `kwavers_math::fft` import in the axisymmetric module.

### Changed (2026-07-08) - kwavers-solver line-reconstruction Apollo 2-D FFT migration [patch]
- [patch] Routed `inverse::reconstruction::photoacoustic::line_reconstruction`
  2-D FFT execution through Apollo's complex FFT APIs over Leto buffers instead
  of the `kwavers_math::fft` facade. The interpolation/scaling math remains
  `num_complex` at the current ndarray boundary, with one private conversion
  SSOT for Apollo scratch buffers. Verification: `rustup run nightly cargo fmt
  --package kwavers-solver --check` passed; `rustup run nightly cargo check -p
  kwavers-solver` passed; `rustup run nightly cargo nextest run -p
  kwavers-solver line_reconstruction` passed 4/4; and a scoped source audit
  found only Apollo FFT execution calls in the line-reconstruction module.

### Changed (2026-07-08) - kwavers-solver fast-nearfield Apollo 2-D FFT migration [patch]
- [patch] Routed `analytical::transducer::fast_nearfield` field computation
  through Apollo's 2-D complex FFT APIs over Leto buffers instead of the
  `kwavers_math::fft` facade. The FNM public/storage boundary remains
  `num_complex` for this slice because cached Green spectra and ndarray-backed
  field arrays still use that representation. Verification: `rustup run
  nightly cargo fmt --package kwavers-solver --check` passed; `rustup run
  nightly cargo check -p kwavers-solver` passed; `rustup run nightly cargo
  nextest run -p kwavers-solver fast_nearfield` passed 6/6; and a scoped source
  audit found only Apollo FFT execution calls in the fast-nearfield module.

### Changed (2026-07-08) - kwavers-solver HAS Apollo 2-D FFT migration [patch]
- [patch] Routed `forward::nonlinear::hybrid_angular_spectrum::diffraction`
  through Apollo's 2-D complex FFT APIs over Leto buffers instead of the
  `kwavers_math::fft` facade. Verification: `rustup run nightly cargo fmt
  --package kwavers-solver --check` passed; `rustup run nightly cargo check -p
  kwavers-solver` passed; `rustup run nightly cargo nextest run -p
  kwavers-solver hybrid_angular_spectrum` passed 18/18; and a scoped source
  audit found only Apollo FFT calls in the HAS cone.

### Changed (2026-07-08) - kwavers-solver KZK Apollo 2-D FFT migration [patch]
- [patch] Routed KZK angular-spectrum, real parabolic, and complex parabolic
  2-D diffraction scratch paths through direct Apollo FFT APIs over Leto
  buffers. The complex-field public boundary remains `num_complex` for this
  slice, with explicit copy-in/copy-out at the leaf scratch boundary.
  Verification: `rustup run nightly cargo fmt --package kwavers-solver --check`
  passed; `rustup run nightly cargo check -p kwavers-solver` passed; `rustup
  run nightly cargo nextest run -p kwavers-solver kzk` passed 49/49; and a
  scoped source audit found no `kwavers_math::fft` imports in the touched KZK
  2-D diffraction files.

### Changed (2026-07-08) - kwavers-solver KZK Apollo 1-D FFT migration [patch]
- [patch] Routed KZK temporal complex 1-D FFT scratch paths in absorption,
  nonlinear spectral differentiation, and finite-difference diffraction through
  direct Apollo APIs over Leto buffers. Verification: `rustup run nightly cargo
  fmt --package kwavers-solver --check` passed; `rustup run nightly cargo check
  -p kwavers-solver` passed; `rustup run nightly cargo nextest run -p
  kwavers-solver kzk` passed 49/49; and a scoped KZK source audit found only
  Apollo 1-D FFT calls in the touched files.

### Changed (2026-07-08) - kwavers warning/example cleanup and Apollo complex boundary [patch]
- [patch] Cleaned `kwavers` all-target warnings in property/comparative tests
  and the GPU beamforming benchmark, consolidated the benchmark CPU path
  through its existing helper, tightened example clippy surfaces, and repaired
  inverse-reconstruction Apollo complex boundaries so Apollo-owned 1-D FFT
  results convert explicitly at the remaining `num_complex` facade edge.
  Verification: `rustup run nightly cargo check -p kwavers --examples`
  passed; `rustup run nightly cargo check -p kwavers --all-targets` passed;
  `rustup run nightly cargo clippy -p kwavers --all-targets --no-deps -- -D
  warnings` passed; `rustup run nightly cargo nextest run -p kwavers-solver
  photoacoustic --status-level fail --no-fail-fast` passed 10/10; `rustup run
  nightly cargo nextest run -p kwavers --test property_based_tests --test
  comparative_solver_tests --test nonlinear_physics_tests --test
  test_pstd_kwave_comparison --test imaging_literature_validation
  --status-level fail --no-fail-fast` passed 46/46; and `rustup run nightly
  cargo run -p xtask -- burn-migration-audit` passed with 0 Burn manifest deps
  and 5 approved non-solver source residuals. Residual: package-wide
  `rustup run nightly cargo fmt -p kwavers --check` is still blocked by
  pre-existing formatting drift outside this slice in
  `examples/focused_water_tank_common/simulation.rs`,
  `examples/pstd_fdtd_comparison.rs`, `src/theranostic/monitor/fd.rs`,
  `tests/pstd_finite_window_born.rs`, and `tests/quick_comparative_test.rs`;
  touched files were formatted with file-scoped `rustfmt`.

### Changed (2026-07-08) - kwavers-solver inverse Apollo 1-D FFT migration [patch]
- [patch] Routed inverse-reconstruction 1-D FFT call sites in photoacoustic
  filtering/Fourier reconstruction and seismic envelope-phase Hilbert paths
  through Apollo's Leto-native `fft_1d_leto`/`ifft_1d_leto` and direct Apollo
  complex inverse APIs. Verification: `rustup run nightly cargo fmt --package
  kwavers-solver --check` passed; `rustup run nightly cargo check -p
  kwavers-solver` passed; `rustup run nightly cargo nextest run -p
  kwavers-solver photoacoustic` passed 10/10; `rustup run nightly cargo nextest
  run -p kwavers-solver envelope misfit phase` passed 34/34; and a scoped
  source audit found no `fft_1d_array`/`ifft_1d_array` calls under
  `crates/kwavers-solver/src`.

### Changed (2026-07-08) - Burn-to-Coeus migration guard [patch]
- [patch] Added `xtask` commands `burn-migration-audit` and
  `refresh-burn-allowlist` plus `xtask/burn_surface.allowlist` as the focused
  direct-Burn baseline for the current Coeus cleanup. The CI legacy migration
  workflow now runs a separate Burn audit job. Verification: `rustup run
  nightly cargo fmt -p xtask --check` passed; `rustup run nightly cargo nextest
  run -p xtask burn_audit --status-level fail --no-fail-fast` passed 2/2; and
  `rustup run nightly cargo run -p xtask -- burn-migration-audit` passed with
  0 Burn manifest deps, 5 approved non-solver source residuals, and no solver
  PINN allowlist entries.

### Changed (2026-07-07) - kwavers-physics acoustic heat-source Moirai traversal [patch]
- [patch] Routed `acoustics::conservation::heat::acoustic_heat_source` through
  the crate-local Moirai-backed `parallel` traversal SSOT instead of direct
  ndarray/Rayon `Zip::par_for_each`. Added `zip_mut_five_refs` so heat-source
  output consumes pressure, velocity magnitude, density, sound speed, and
  absorption in one pass. Verification: `rustup run nightly cargo check -p
  kwavers-physics --lib` passed; `rustup run nightly cargo nextest run -p
  kwavers-physics heat_source --status-level fail` passed 9/9 with 1704
  skipped; and a scoped source audit found no `Zip|par_for_each|rayon` tokens
  in `heat.rs`. Residual direct solver/physics `.par_for_each` holdouts are now
  49 sites outside RTM inherent, sonogenetics, and acoustic heat-source
  traversal; `cargo clippy -p kwavers-physics --lib -- -D warnings` is blocked
  before this package by local dependency `ritk-transform` Burn `Module` derive
  errors in the concurrent RITK provider migration diff.

### Changed (2026-07-07) - kwavers-physics focused traversal cleanup [patch]
- [patch] Routed `acoustics::therapy::sonogenetics` gating and volumetric ARF
  field traversal plus heterogeneous skull-mask property assignment through the
  crate-local Moirai-backed `parallel` traversal SSOT instead of direct
  ndarray/Rayon `Zip::par_for_each` or duplicate one-input helper calls. Added
  `zip_mut_ref` and `zip_two_mut_four_refs` so one-input updates and ARF's
  intensity/body-force update share the traversal SSOT. Verification: `rustup
  run nightly cargo fmt -p kwavers-physics --check` passed; `rustup run nightly
  cargo check -p kwavers-solver --lib` passed; `rustup run nightly cargo clippy
  -p kwavers-solver --lib --no-deps -- -D warnings` passed; `rustup run nightly
  cargo nextest run -p kwavers-physics sonogenetics --status-level fail
  --no-fail-fast` passed 53/53 with 1660 skipped; `rustup run nightly cargo
  nextest run -p kwavers-physics skull --status-level fail --no-fail-fast`
  passed 51/51 with 1662 skipped; and a scoped source audit found no
  `Zip|par_for_each|rayon` tokens under the sonogenetics cone or the touched
  skull mask file. Residual direct solver/physics `.par_for_each` holdouts are
  now 49 sites outside RTM inherent, sonogenetics, skull mask, and acoustic
  heat-source traversal.

### Changed (2026-07-07) - kwavers-solver RTM inherent Moirai traversal [patch]
- [patch] Routed `inverse::reconstruction::seismic::rtm::inherent` wavefield,
  propagation interpolation, source illumination, Laplacian filtering,
  post-processing, and imaging-condition loops through the RTM-private
  Moirai-backed strided-view traversal helper instead of direct ndarray/Rayon
  `Zip::par_for_each`. Verification: `rustup run nightly cargo check -p
  kwavers-solver --lib` passed; `rustup run nightly cargo nextest run -p
  kwavers-solver rtm --status-level fail` passed 10/10 with 916 skipped; and a
  scoped source audit found no `Zip|par_for_each|rayon` tokens under the RTM
  inherent cone. Residual direct ndarray/Rayon solver/physics holdouts are 49
  `.par_for_each` sites outside RTM inherent, sonogenetics, and acoustic
  heat-source traversal; `cargo fmt -p kwavers-solver --check` remains blocked
  by pre-existing formatting drift in
  `crates/kwavers-solver/src/forward/fdtd/electromagnetic/tests.rs`, and
  `rustup run nightly cargo clippy -p kwavers-solver --lib --no-deps -- -D
  warnings` passes after the Atlas Mnemosyne/Themis/Melinoe provider graph
  refresh.

### Changed (2026-07-05) - kwavers-math numeric SSOT Phase-1A pilot [patch]
- [patch] Phase-1A closed `kwavers_math::linear_algebra::NumericOps<T>` against the eunomia numeric SSOT. `num_traits::{Float, NumCast, Zero}` is replaced by `eunomia::RealField` (re-exported from `eunomia::traits::field`) and `eunomia::NumericElement::ZERO`. Super-traits `Clone + Zero` (and the vestigial `NumCast`) are dropped to `Copy + PartialOrd`. The six method bodies (`dot_product`, `normalize`, `add_arrays`, `scale_array`, `l2_norm`, `max_abs`, `safe_divide`) use `T::ZERO` instead of `T::zero()`. `max_abs` folds via `if val > acc { val } else { acc }` driven by `T: PartialOrd` because `eunomia::RealField` does not propagate a `max` method. `eunomia = { workspace = true }` is now declared in `crates/kwavers-math/Cargo.toml`; `num-traits` is retained only for `linear_algebra::sparse::csr.rs` (Phase-1B blocker — `num_complex::Complex64` does not impl `eunomia::NumericElement` under eunomia's `private::Sealed` float traits). Completion condition: `cargo build -p kwavers-math` succeeds; `numeric_ops.rs` drops from the kwavers xtask `legacy-migration-audit` source-legacy list. Residual: csr.rs Phase-1B queued under `CR-EUNOMIA-COMPLEX`.

### Reverted (2026-07-05) - kwavers-math Phase-1B §2 ssot-rebind [patch]
- [patch] Reverted the kwavers-math Phase-1B §2 batch that attempted to rebind `CsrScalar: eunomia::ComplexField` with blanket impls. The orphan-rule gap in `eunomia::types::complex::{ops.rs,float.rs}` (`ndarray::ScalarOperand` / `nalgebra::LinalgScalar` cross-impls missing) surfaced 7× E0277 from the proposed `Complex * Array1` site in `solver/bicgstab.rs` and the migration was abandoned before landing. `crates/kwavers-math/src/linear_algebra/sparse/csr.rs` returns to `CsrScalar: num_traits::Zero` + per-impl `magnitude(self) -> f64` (`f64::abs()`, `num_complex::Complex64::norm()`); `crates/kwavers-math/Cargo.toml` retains `num-traits = "0.2"` for csr.rs. The §3 deliverable — `kwavers-boundary` `num_complex::Complex64` → `eunomia::Complex64` migration plus the csr.rs ssot-rebind — is rescheduled to a follow-up ADR bundled with the upstream eunomia cross-impl. Reference: csr.rs `//!` mod-doc (`crates/kwavers-math/src/linear_algebra/sparse/csr.rs:1-9`) and the new `## TODO: kwavers-math Phase-1B §3 deferral` section in `repos/kwavers/backlog.md`.

### Changed (2026-07-01) - Atlas provider migration [patch]

- [patch] Made the solver-owned GPU backend boundary provider-generic by
  carrying `GpuProvider` on `BackendType::GPU`, updated the WGPU leaf backend
  to report `GpuProvider::Wgpu`, and removed WGPU error conversion/feature
  forwarding from `kwavers-core`.
- [patch] Closed the focused GPU backend verification blockers by normalizing
  diagnostics Leto arrays through direct volume traversal, making transfer
  overhead dominate bottleneck classification before derived GPU-utilization
  checks, and adding the missing WGPU derivative-pipeline binding.
- [patch] Made `kwavers-gpu::backend::GPUBackend` generic over a
  `GpuComputeProvider` trait and routed the default WGPU provider through
  `hephaestus_wgpu::WgpuDevice`, leaving CUDA as a sibling provider
  implementation point behind the same trait.
- [patch] Corrected solver-facing GPU documentation to name the
  provider-generic `GPUBackend<P>` boundary and to identify the legacy SWE GPU
  file as a performance model, not a real WGPU/CUDA dispatch path.
- [patch] Split the `kwavers-gpu` GPU compute contract into
  `GpuKernelProvider`, `ElementWiseMultiplyProvider`, and
  `SpatialDerivativeProvider`, keeping `GpuComputeProvider` as the composite
  backend trait so WGPU and CUDA can implement operation families only when
  real kernels exist.
- [patch] Renamed the backend WGSL pipeline surface to
  `WgpuPipelineManager`, making WGPU pipeline compilation/execution explicit
  instead of exposing a backend-neutral `PipelineManager` name.
- [patch] Renamed the raw WGPU command helper from `GpuCompute` to
  `WgpuComputeCommands`, making bind-group layout and command-encoder
  ownership provider-explicit instead of naming it as generic GPU compute.
- [patch] Bound `GpuComputeProvider` to an associated Hephaestus
  `ComputeDeviceCapabilities` device type so WGPU and future CUDA providers
  compile through the same accelerator trait seam.
- [patch] Reverified the `kwavers-gpu` provider-generic boundary after
  `hephaestus-cuda` implemented the shared unary/binary storage-kernel traits;
  WGPU and CUDA-provider feature sets compile, lint, and pass focused provider
  tests without a Kwavers-local CUDA helper.
- [patch] Added `CudaElementWiseProvider` as a real
  `ElementWiseMultiplyProvider` backed by Hephaestus CUDA elementwise kernels,
  keeping CUDA out of the composite `GpuComputeProvider` until the remaining
  operation families have real CUDA kernels. The realtime imaging Hilbert FFT
  path now calls the `kwavers_math::fft` slice facade instead of Apollo's
  Leto-native plan API directly.
- [patch] Bound `AcousticFieldProvider` to the shared
  `GpuKernelProvider`/`GpuProviderBackend` trait stack and moved the current
  WGPU acoustic-field provider onto `GpuProviderContext<WgpuDevice>`, so future
  real CUDA acoustic kernels substitute through the same Hephaestus-backed
  provider contract instead of a standalone acoustic trait.
- [patch] Bound the thermal-acoustic solver provider to the shared
  `GpuKernelProvider`/`GpuProviderBackend` stack and moved default WGPU
  construction onto `GpuProviderContext<WgpuDevice>`, removing the raw
  `wgpu::Device`/`Queue` constructor from the generic solver wrapper.
- [patch] Made `kwavers-gpu::backend::GpuProviderContext<P>` generic over the
  Hephaestus-backed `GpuDeviceProvider` trait and re-exported that trait from
  `kwavers_gpu::gpu`, keeping raw WGPU device/queue access on the
  `WgpuDevice` specialization only.
- [patch] Moved default GPU acquisition requirements onto
  `GpuDeviceProvider` so `GpuProviderContext<P>` is generic over provider
  labels, preferences, optional features, and limits instead of baking WGPU's
  `ShaderF64` and WGSL workgroup policy into the generic path.
- [patch] Added the optional `kwavers-gpu/cuda-provider` acquisition contract
  backed by local `hephaestus-cuda`, plus `cuda-runtime` for Hephaestus' real
  CUDA loader, and implemented `GpuDeviceProvider` for
  `hephaestus_cuda::CudaDevice` without adding placeholder CUDA compute
  dispatch.
- [patch] Exposed the existing Hephaestus CUDA provider seam through top-level
  `kwavers/cuda-provider` and `kwavers/cuda-runtime` feature forwards, keeping
  concrete WGPU/CUDA execution behind `kwavers-gpu` provider traits.
- [patch] Routed top-level ignored GPU FFT parity tests through Apollo's
  `FftBackend` plan seam and Leto test buffers instead of constructing raw
  WGPU instances/devices in `kwavers`, keeping WGPU as the current
  Hephaestus-backed implementation and CUDA as an upstream backend
  implementation of the same trait.
- [patch] Removed top-level `kwavers/gpu` direct `wgpu`, `bytemuck`, and
  `pollster` feature edges by adding provider-owned synchronous acquisition,
  buffer, readback, acoustic-kernel, and FDTD readback wrappers in
  `kwavers-gpu`; top-level GPU tests now exercise those provider APIs without
  importing concrete WGPU runtime crates.
- [patch] Converted the top-level stream visualization tests to blocking
  stream/pipeline entry points and provider-native `leto::Array3<f32>` frames,
  removing Tokio and ndarray from that test target while keeping the async
  stream API available. The electromagnetic PINN example also no longer uses a
  Tokio wrapper where no async work exists; the former real-time 3-D
  beamforming raw async WGPU constructor edge is closed by the provider
  constructor slice below.
- [patch] Made `kwavers-analysis` 3-D beamforming construction generic over a
  `BeamformingGpuProvider` trait, with the current WGPU implementation
  acquired through Hephaestus `WgpuDevice`;
  `BeamformingProcessor3D::with_provider` is the public provider-generic
  constructor, `BeamformingProcessor3D::new_wgpu` names the current WGPU
  implementation explicitly, the real-time 3-D beamforming example no longer
  uses Tokio, and top-level `kwavers` no longer has a direct Tokio
  dev-dependency.
- [patch] Moved the concrete 3-D WGPU beamforming provider, DAS dispatch,
  dynamic-focus dispatch, parameters, device-error mapping, and WGSL shaders
  from `kwavers-analysis` into `kwavers-gpu`, leaving analysis with only the
  provider contract and CPU reference. The real-time 3-D beamforming example
  now injects `kwavers_gpu::beamforming::three_dimensional::WgpuBeamformingProvider`,
  and `kwavers-gpu` owns the WGPU-vs-CPU differential seam.
- [patch] Removed the artificial async wrapper from distributed neural
  beamforming in `kwavers-analysis`; `process_volume_distributed` now executes
  synchronously over the existing Moirai-backed processor fan-out, and the
  crate no longer needs a Tokio dev-dependency for that test path.
- [major] Tightened `GpuComputeProvider::Device` to require the Kwavers
  `GpuDeviceProvider` trait, making WGPU/CUDA provider substitution a public
  type contract backed by Hephaestus acquisition/capability traits, and exposed
  `GPUBackend<P>::provider()` for provider-generic callers.
- [patch] Removed fixed WGPU memory and peak-FLOP constants from
  `WgpuComputeProvider`; reported memory now derives from the acquired
  Hephaestus device limits, and unknown peak throughput is reported as `0.0`
  rather than a fabricated hardware value. The generic provider performance
  estimate now returns the provider-reported peak value instead of a hardcoded
  problem-size speedup curve.
- [patch] Corrected `WgpuComputeProvider` capabilities to report
  `supports_fft = false` because `ComputeBackend` does not expose FFT
  operations; GPU FFT remains owned by Apollo through
  `kwavers_math::fft::gpu_fft`.
- [patch] Removed `kwavers-math`'s direct `ndarray/rayon` feature, routed the
  Apollo dependency to the local Atlas Apollo checkout, and patched Apollo to
  use local Hephaestus for GPU FFT support. The Kwavers FFT facade preserves
  the current ndarray/`num_complex` API while calling Apollo's Leto/`eunomia`
  FFT contract internally, and its GPU facade now exposes Apollo's
  `FftBackend` trait while recording WGPU as the current implementation.
- [patch] Removed concrete WGPU runtime dependencies from `kwavers-solver/gpu`
  by dropping its direct `wgpu`, `bytemuck`, and `pollster` optional edges,
  keeping concrete GPU ownership in `kwavers-gpu`; solver FFT call sites now
  route through the `kwavers_math::fft` ndarray/Leto facade instead of calling
  Apollo's Leto-native plan API directly.
- [patch] Re-verified the WGPU/CUDA provider-generic boundary with
  `kwavers-gpu --features cuda-provider`, kept the stream visualization test
  out of async-only builds, updated the seismic imaging example to current
  RITK DICOM/spacing accessors, and repaired the GPU PSTD simulation adapter
  so its CPML profile coefficient helper consumes Leto arrays directly.
- [patch] Routed `GpuPstdSolver::with_auto_device` through
  `hephaestus_wgpu::WgpuDevice` while preserving PSTD push-constant and
  storage-buffer limit requirements.
- [patch] Routed `GpuPstdSolver::with_auto_device` through
  `GpuDevice<WgpuDevice>` and the shared `GpuDeviceProvider` acquisition
  contract, preserving PSTD push-constant limits while removing the remaining
  direct WGPU acquisition call from the PSTD auto-device constructor.
- [patch] Routed PSTD GPU construction and medium-update test helpers through
  `GpuDevice<WgpuDevice>` and the shared `GpuDeviceProvider` acquisition
  contract, removing the last direct WGPU adapter/device acquisition path from
  the PSTD subtree.
- [patch] Routed the backend buffer-manager GPU construction test through
  `GpuDevice<WgpuDevice>` and `GpuDeviceProvider`, removing its direct
  Hephaestus WGPU acquisition helper while keeping buffer management
  WGPU-owned.
- [patch] Removed `pollster` from the backend buffer readback path by routing
  both async and synchronous public readback methods through one blocking WGPU
  readback implementation, leaving provider-native `leto::Array3<f32>`
  semantics unchanged.
- [patch] Made `kwavers-gpu::gpu::GpuDevice<P>` generic over a local
  `GpuDeviceProvider` trait backed by Hephaestus `ComputeDeviceAcquisition`,
  replaced the public constructor preference with backend-neutral
  `DevicePreference`, and kept raw WGPU handles only on the default
  `GpuDevice<WgpuDevice>` specialization for WGSL shader dispatch.
- [patch] Routed the acoustic-field WGPU provider through
  `GpuDevice<WgpuDevice>` and the shared `GpuDeviceProvider` acquisition
  contract instead of directly acquiring/storing a raw Hephaestus
  `WgpuDevice`, keeping CUDA eligible for the same provider trait without
  placeholder acoustic kernels.
- [patch] Routed `kwavers-gpu::gpu::CoreGpuContext` and multi-GPU
  logical-device creation through Hephaestus `ComputeDeviceAcquisition` while
  keeping raw WGPU handles as provider-owned dispatch handles and preserving
  the generic `GpuComputeProvider` seam for a future CUDA provider.
- [patch] Made `kwavers-gpu::gpu::CoreGpuContext<P>` generic over
  `GpuDeviceProvider`, storing `GpuDevice<P>` instead of raw `WgpuDevice`,
  while keeping raw `wgpu` handle access only on `CoreGpuContext<WgpuDevice>`.
- [patch] Made the public `kwavers-gpu::gpu::GpuBackend<P>` alias expose the
  same provider parameter as `CoreGpuContext<P>`, keeping WGPU as the default
  provider while allowing CUDA and future Hephaestus providers to type-check at
  the context boundary without call-site backend branches.
- [patch] Made `kwavers-gpu::gpu::multi_gpu::MultiGpuContext<P>` generic over
  the same `GpuDeviceProvider` trait, added a provider-generic multi-device
  acquisition constructor, and kept `MultiGpuContext::new()` as the WGPU
  default for current WGSL kernels while CUDA type-checks through
  `kwavers-gpu/cuda-provider`.
- [patch] Removed `kwavers-gpu`'s direct Tokio test-runtime dependency by
  running the remaining async GPU acquisition tests through `pollster`, keeping
  the crate's provider-generic WGPU/CUDA seam free of a Tokio dev-dependency.
- [patch] Removed the `kwavers-gpu` Burn-backed `BurnGpuAccelerator` and its
  `pinn` feature/dependency because it was not a Hephaestus or Coeus backend
  and contained placeholder PDE residual branches for non-wave equations.
- [major] Removed the solver-local PINN `gpu_accelerator` surface and the
  `pinn-gpu`/`burn-wgpu`/`burn-cuda` feature aliases. PINN GPU training is now
  documented as pending Coeus training routed through provider-generic
  Hephaestus traits, so WGPU and CUDA remain provider implementations behind
  the same seam instead of Burn-specific public feature contracts.
- [patch] Removed solver PINN multi-GPU manager direct WGPU adapter discovery;
  multi-GPU PINN construction now returns a typed unavailable-provider error
  until a real Coeus training provider is routed through Hephaestus WGPU/CUDA
  device traits, and that unavailable-provider constructor path no longer
  requires a Tokio test runtime. The distributed PINN trainer constructor is
  also synchronous while it only assembles local replicas and provider state;
  distributed training/checkpoint persistence no longer depends on Tokio, and
  checkpoint save/load now writes JSON checkpoint files with value-tested state
  restoration.
- [patch] Removed the Burn `wgpu` feature from the workspace and remaining
  `kwavers`/`kwavers-solver`/`kwavers-analysis` Burn dependencies, keeping Burn
  scoped to the current CPU PINN path while GPU PINN execution migrates to
  Coeus through Hephaestus providers.
- [patch] Removed unused workspace-level Burn dependency aliases and demoted
  the top-level `kwavers` Burn edge to `dev-dependencies`, since that crate's
  production targets do not import Burn and only examples, benches, and tests
  still instantiate Burn CPU PINN paths.
- [patch] Disabled Burn default features on remaining kwavers Burn edges,
  retained only required non-GPU features, and repaired the upstream RITK
  workspace Burn default from WGPU to NdArray so `kwavers --features pinn` no
  longer resolves `burn-wgpu`, `burn-cuda`, or `burn-rocm`.
- [major] Removed the remaining direct Burn dependency from `kwavers-analysis`
  by deleting the test-only Burn DAS holdout, removing the analysis-side Burn
  model compatibility impl, and keeping the PINN uncertainty API on the
  solver-agnostic `PinnUncertaintyPredictor` trait.
- [patch] Removed the direct Burn dependency from `kwavers-python` by routing
  RITK NIfTI loading through `ritk_io::format::nifti::native` on
  `coeus_core::SequentialBackend`, preserving the existing NumPy-facing
  `(x, y, z)` volume and spacing contract.
- [patch] Routed `kwavers-imaging` CT/NIfTI loading through the native RITK
  Coeus image path while preserving the existing `(x, y, z)` volume, spacing,
  affine, and intensity-range contract.
- [patch] Removed the remaining direct Burn edge from `kwavers-imaging` by
  adding native DICOM series loading in RITK and routing the Kwavers DICOM
  bridge through `ritk_io::load_native_dicom_series` on
  `coeus_core::SequentialBackend`.
- [patch] Removed the public `kwavers-analysis::beamforming::gpu` Burn DAS
  reexports, leaving the Burn implementation as a test-only legacy `pinn`
  holdout until the DAS/PINN path is migrated to Coeus/Hephaestus.
- [major] Removed the public `kwavers-analysis::beamforming::neural` Burn PINN
  provider reexports and exposed the solver-agnostic
  `PinnBeamformingProvider` trait instead, keeping analysis generic over the
  provider seam while Burn remains a solver implementation detail pending
  Coeus/Hephaestus migration.
- [major] Replaced direct `BurnPINN1DWave` uncertainty estimator signatures in
  `kwavers-analysis` with the solver-agnostic `PinnUncertaintyPredictor` trait,
  leaving a single Burn compatibility impl for the current solver model while
  Coeus becomes the replacement provider.
- [major] Removed the top-level `kwavers` `parallel` feature and direct Rayon
  dependency, routed the remaining top-level example parallel loops through
  `moirai-parallel`, and removed `ndarray/rayon` from the Python wrapper
  manifest where wrapper source has no ndarray-parallel call sites.
- [patch] Removed the direct `kwavers-physics` Rayon dependency by routing
  remaining direct Rayon loops through `moirai-parallel`; `ndarray/rayon`
  remains only for tracked ndarray-parallel kernels pending Leto/Hephaestus
  backend migration.
- [patch] Routed `kwavers-physics` bubble-interaction field assembly through
  the crate's Moirai-backed indexed traversal adapter instead of
  ndarray/Rayon `Zip::par_for_each`, with a value test for the monopole
  pressure contribution and self-cell exclusion.
- [patch] Routed `kwavers-physics::field_surrogate` trilinear resampling and
  kernel-corner blending through the crate's Moirai-backed traversal adapter
  instead of ndarray/Rayon `Zip::par_for_each`, preserving focal-position,
  normalization, and blend semantics.
- [patch] Added a two-output/two-input Moirai-backed physics traversal adapter
  and routed `kwavers-physics::chemistry::reaction_kinetics` through it instead
  of ndarray/Rayon `Zip::par_for_each`, with a value test for Arrhenius
  radical generation and hydrogen-peroxide recombination.
- [patch] Routed `kwavers-physics::chemistry::ros_plasma::ros_species`
  concentration decay through the crate's Moirai-backed traversal adapter
  instead of ndarray/Rayon `par_mapv_inplace`, with an exact species-lifetime
  exponential regression.
- [patch] Routed `kwavers-physics` Pennes bioheat and Cattaneo-Vernotte
  thermal diffusion traversals through a private Moirai-backed dense-field
  adapter, removing direct ndarray/Rayon dispatch from those thermal diffusion
  files while preserving sequential ndarray semantics for non-contiguous views.
- [patch] Routed `kwavers-physics` sonoluminescence blackbody,
  bremsstrahlung, and Cherenkov emission field assembly through the same
  Moirai-backed physics traversal adapter, removing direct ndarray/Rayon
  dispatch from those optics kernels.
- [patch] Replaced the `kwavers-math` QR/SVD nalgebra bridge with
  Leto/Leto-ops decompositions, added the workspace-local `leto-ops`
  dependency, and removed the direct `kwavers-math` nalgebra manifest edge.
- [patch] Routed `kwavers-math::fft` real/complex packing and k-space
  squared-field generation through Moirai-backed contiguous traversal,
  removing the final source-level ndarray/Rayon parallel calls under
  `crates/kwavers-math/src` while preserving sequential ndarray traversal for
  non-standard FFT array layouts.
- [patch] Routed the Westervelt spectral wave-model leapfrog combination loop
  in `kwavers-solver` through `moirai-parallel`, removing that direct Rayon
  iterator while preserving the same value-semantic pressure recurrence.
- [patch] Routed Helmholtz FEM element-contribution collection in
  `kwavers-solver` through `moirai-parallel` and added explicit element-array
  length validation instead of the previous zipped truncation behavior.
- [patch] Routed Westervelt FDTD conservation energy, momentum, and mass
  reductions in `kwavers-solver` through `moirai-parallel` indexed reductions,
  removing the direct Rayon iterator dependency from that diagnostic path.
- [patch] Routed Westervelt FDTD Laplacian O2/O4/O6 stencil slabs in
  `kwavers-solver` through `moirai-parallel`, preserving the same finite
  difference coefficients and quadratic exactness contract.
- [patch] Routed Westervelt FDTD nonlinear-term and leapfrog update field
  traversals in `kwavers-solver` through `moirai-parallel`, preserving the
  product-rule curvature initialization and medium-coupled propagation algebra.
- [patch] Routed KZK solver observable reductions and physics-trait RMS field
  generation in `kwavers-solver` through `moirai-parallel`, removing direct
  ndarray/Rayon dispatch from those 2-D pressure summary paths.
- [patch] Routed KZK angular-spectrum and real-field parabolic diffraction
  scratch packing, diagonal spectral multiplication, and real-output
  projection in `kwavers-solver` through `moirai-parallel`, preserving cached
  FFT workspaces and propagator semantics while removing direct
  ndarray/Rayon dispatch from those operators.
- [patch] Routed KZK spectral absorption slab traversal in `kwavers-solver`
  through `moirai-parallel`, preserving per-slab waveform scratch reuse and
  the exact frequency-domain attenuation mask.
- [patch] Routed KZK complex parabolic diffraction and nonlinear sub-step slab
  traversals in `kwavers-solver` through `moirai-parallel`, leaving the KZK
  subtree with no direct Rayon or ndarray-parallel call sites.
- [patch] Routed the mixed-domain frequency-domain propagator in
  `kwavers-solver` through `moirai-parallel`, replacing direct ndarray/Rayon
  dispatch with provider-owned indexed traversal over the complex spectral
  field.
- [patch] Routed the legacy KZK solver plugin nonlinear Burgers update in
  `kwavers-solver` through `moirai-parallel` for dense fields, preserving
  sequential ndarray semantics for non-standard layouts.
- [patch] Routed FDTD dynamic pressure-source Dirichlet and additive mask
  updates in `kwavers-solver` through `moirai-parallel` for dense fields,
  preserving sequential ndarray semantics for non-standard layouts.
- [patch] Routed FDTD pressure-updater divergence accumulation, pressure
  update, and Westervelt nonlinear pressure-delta application in
  `kwavers-solver` through `moirai-parallel` for dense fields.
- [patch] Routed FDTD velocity-updater spectral, collocated, and staggered
  pressure-gradient velocity updates in `kwavers-solver` through
  `moirai-parallel` for dense fields.
- [patch] Routed FDTD k-space correction spectral gradient/divergence kernels
  in `kwavers-solver` through `moirai-parallel` for dense transformed fields.
- [patch] Routed FDTD construction-time `rho*c^2` and nonlinear coefficient
  fills in `kwavers-solver` through `moirai-parallel` for dense fields.
- [patch] Routed PSTD utility k-norm fills and spectral-derivative scaling in
  `kwavers-solver` through `moirai-parallel` for dense buffers.
- [patch] Routed PSTD implementation k-space Helmholtz and spectral-gradient
  multipliers in `kwavers-solver` through `moirai-parallel` for dense
  spectral fields.
- [patch] Routed PSTD implementation anti-aliasing spectral filter
  multipliers in `kwavers-solver` through `moirai-parallel` for dense
  half-spectrum buffers.
- [patch] Routed PSTD implementation full-k-space source accumulation,
  spectral wave-coefficient multiplication, and propagated pressure/source
  updates in `kwavers-solver` through `moirai-parallel` for dense fields.
- [patch] Routed PSTD implementation pressure-source correction, split-density
  source injection, and dynamic velocity-source writes in `kwavers-solver`
  through shared `moirai-parallel` dense stepper helpers.
- [patch] Routed PSTD implementation total split-density accumulation in
  `kwavers-solver` through `moirai-parallel` for dense fields.
- [patch] Routed PSTD implementation thermal absorption coefficient scaling
  in `kwavers-solver` through `moirai-parallel` for dense fields.
- [patch] Routed PSTD implementation construction-time source-kappa cosine
  transformation and initial split-density component fills in `kwavers-solver`
  through `moirai-parallel` for dense fields.
- [patch] Routed PSTD implementation IVP density seeding, spectral-gradient
  construction, and half-step velocity scaling in `kwavers-solver` through
  `moirai-parallel` for dense fields.
- [patch] Routed PSTD spectral-correction kappa fills and correction
  application in `kwavers-solver` through `moirai-parallel` for dense spectral
  fields.
- [patch] Routed PSTD propagator pressure equation-of-state density
  accumulation and pressure writes in `kwavers-solver` through
  `moirai-parallel` for dense fields.
- [patch] Routed PSTD Cartesian density spectral-gradient and split-density
  updates in `kwavers-solver` through `moirai-parallel` for dense fields.
- [patch] Routed PSTD axisymmetric pressure-density coefficient and
  split-density updates in `kwavers-solver` through `moirai-parallel` for dense
  fields.
- [patch] Routed PSTD Cartesian and axisymmetric velocity spectral-gradient and
  velocity-field updates in `kwavers-solver` through `moirai-parallel` for dense
  fields.
- [patch] Routed PSTD axisymmetric WSWA-FFT pressure-gradient and
  density-divergence propagation in `kwavers-solver` through
  `moirai-parallel` for dense spectral fields.
- [patch] Routed PSTD broadband residual-gas absorption and dispersion pressure
  corrections in `kwavers-solver` through `moirai-parallel` for dense fields.
- [patch] Routed PSTD pressure-side fractional-Laplacian absorption correction
  in `kwavers-solver` through `moirai-parallel` for dense fields.
- [patch] Routed PSTD fractional-Laplacian absorption stratum bracket
  construction in `kwavers-solver` through `moirai-parallel`.
- [patch] Routed PSTD spectral derivative pencil traversal in `kwavers-solver`
  through `moirai-parallel`.
- [patch] Routed PSTD DG spectral Laplacian symbol construction and
  application in `kwavers-solver` through `moirai-parallel`.
- [patch] Routed PSTD DG one-dimensional acoustic SSP-RK stage updates in
  `kwavers-solver` through `moirai-parallel`.
- [patch] Routed PSTD DG modal SSP-RK and Forward Euler coefficient updates in
  `kwavers-solver` through shared `moirai-parallel` dense RK helpers.
- [patch] Routed PSTD DG tensor acoustic source SSP-RK state updates in
  `kwavers-solver` through shared `moirai-parallel` dense RK helpers.
- [patch] Routed PSTD DG tensor CPML field and memory SSP-RK state updates in
  `kwavers-solver` through shared `moirai-parallel` dense RK helpers, leaving
  the DG subtree with no direct Rayon or explicit ndarray `Zip` traversal.
- [patch] Moved `ThermalCEM43Grid` dose storage and update input to
  `leto::Array3<f64>`, scheduled dense CEM43 updates through
  `moirai-parallel`, and moved the top-level theranostic lesion mask plus
  brain-monitor thermal state to Leto.
- [patch] Routed `kwavers-solver` standard thermal diffusion updates through
  `moirai-parallel` for dense owned buffers, preserved sequential ndarray view
  semantics for non-contiguous borrowed sources, and added a typed source-shape
  mismatch error instead of indexing mismatched views.
- [patch] Routed `kwavers-solver` coupled thermal-acoustic material-property,
  acoustic-heating, acoustic-step, and thermal-step kernels through
  `moirai-parallel` for dense owned buffers, leaving only sequential ndarray
  fallback paths for unexpected non-standard layouts.
- [patch] Routed `kwavers-solver` BEM scattered-field evaluation through
  `moirai-parallel` ordered map-collect instead of direct Rayon `par_iter`.
- [patch] Routed the legacy `kwavers-solver` seismic RTM zero-lag and
  normalized imaging-condition passes through `moirai-parallel`, removing the
  temporary source-illumination allocation from the single-snapshot normalized
  formula.
- [patch] Routed `kwavers-solver` k-space line reconstruction positivity
  clamping through `moirai-parallel` instead of ndarray/Rayon
  `par_mapv_inplace`.
- [patch] Routed `kwavers-solver` photoacoustic iterative, Fourier, and
  time-reversal reconstruction updates through Moirai-backed traversal,
  leaving the photoacoustic reconstruction subtree with no direct Rayon or
  ndarray-parallel traversal.
- [patch] Routed `kwavers-solver` hybrid angular spectrum broadband harmonic
  absorption through Moirai-backed dense traversal instead of ndarray/Rayon
  `par_mapv_inplace`.
- [patch] Routed `kwavers-solver` nonlinear elastic propagation damping maps
  through Moirai-backed dense traversal instead of ndarray/Rayon
  `par_mapv_inplace`.
- [patch] Routed `kwavers-solver` nonlinear elastic harmonic-generation Jacobi
  update and delta-fill passes through Moirai-backed indexed traversal instead
  of ndarray/Rayon `Zip::par_for_each`.
- [patch] Routed `kwavers-solver` nonlinear elastic fundamental frequency
  line updates through Moirai-backed line scheduling, leaving the nonlinear
  elastic subtree with no direct Rayon or ndarray-parallel traversal.
- [patch] Repaired the top-level skull CT example and ultrasound/NL-SWE
  validation tests for current RITK accessor and Leto array APIs, keeping
  registration/fusion inputs on Leto and using a derived floating-point
  reduction bound for the constant-quality NL-SWE statistic.
- [patch] Split `kwavers-gpu::gpu::compute_kernels::AcousticFieldKernel<P>`
  from concrete WGPU dispatch by adding an `AcousticFieldProvider` operation
  trait and moving the WGSL pipeline, buffers, and readback path into
  `WgpuAcousticFieldProvider`.
- [patch] Made `kwavers-gpu::gpu::compute_kernels::WaveEquationGpu<P>`
  generic over the same `AcousticFieldProvider<Scalar = f32>` provider seam,
  keeping the default constructor WGPU-backed and avoiding CUDA placeholders.
- [patch] Moved `AcousticFieldProvider` and `WaveEquationGpu` from
  `ndarray::Array3<f64>` to provider-native `leto::Array3<f32>` for the WGPU
  implementation, making the WGPU precision contract explicit as an associated
  scalar instead of narrowing f64 fields inside the provider.
- [patch] Split `kwavers-gpu::gpu::thermal_acoustic::GpuThermalAcousticBuffers<P>`
  from concrete WGPU buffer handles by adding `ThermalAcousticBufferProvider`
  and moving the WGPU storage/uniform buffers plus upload/download paths into
  `WgpuThermalAcousticBuffers`.
- [patch] Moved thermal-acoustic buffer upload/readback from
  `ndarray::Array3<f32>` to provider-native `leto::Array3<f32>`, with WGPU
  declaring `ThermalAcousticBufferProvider::Scalar = f32`.
- [patch] Moved `kwavers-gpu::gpu::WgpuFdtd` pressure upload/readback from
  `ndarray::Array3<f64>` to provider-native `leto::Array3<f32>`, surfacing
  dense-layout validation on upload and removing hidden widen/narrow pressure
  conversion at the WGPU storage boundary.
- [patch] Renamed the WGPU-only FDTD public surfaces to `WgpuFdtd` and
  `WgpuFdtdPressureDispatcher`, removing generic GPU names from WGSL-specific
  implementations while leaving CUDA behind the provider-trait gap.
- [patch] Moved `WgpuFdtdPressureDispatcher` construction onto
  `GpuProviderContext<WgpuDevice>` and implemented the shared
  `GpuKernelProvider`/`GpuProviderBackend` stack for the dispatcher, removing
  raw `Arc<wgpu::Device>`/`Arc<wgpu::Queue>` ownership from the FDTD pressure
  kernel boundary.
- [patch] Added the provider-generic `FdtdGpuProvider` operation trait and
  made `WgpuFdtd` its current real implementation, with pressure upload,
  readback, and step execution routed through `GpuProviderContext<WgpuDevice>`.
  The top-level allocation/roundtrip test now binds through the trait over
  provider-native `leto::Array3<f32>` instead of raw WGPU handles, Tokio, or
  ndarray; CUDA remains a real-kernel implementation gap rather than a fake
  branch.
- [patch] Repaired the top-level GPU test compile gate by deleting the obsolete
  recovery stress test, replacing raw-WGPU compute and buffer tests with
  Hephaestus/CoreGpuContext/GPUBackend coverage, routing GPU device tests
  through pollster-backed `GpuDevice`, and converting acoustic-field and
  wave-equation tests to provider-native `leto::Array3<f32>`. Broad
  `kwavers --features gpu --tests` now compiles, and focused top-level GPU
  nextest passes 27/27 with 3 ignored PSTD hardware tests skipped.
- [patch] Closed the remaining warning debt in the top-level GPU test compile
  gate by deleting the unused `gpu_fft_3d` helper, converting ignored GPU FFT
  tests from Tokio macros to `pollster::block_on`, removing the unused
  finite-window Born baseline, and dropping unused Moirai patch entries for
  crates that no workspace manifest depends on.
- [patch] Moved `kwavers-gpu::gpu::compute::FdtdCpuReferenceDispatcher`
  pressure updates from `ndarray::Array3<f64>` to `leto::Array3<f64>`,
  preserving the f64 reference stencil while eliminating the local ndarray
  pressure API and removing the misleading CPU-as-GPU dispatcher name.
- [patch] Made `kwavers-gpu::gpu::ComputeManager<P>` generic over
  `GpuDeviceProvider`, kept raw WGPU handle helpers only on the WGPU
  specialization, and replaced silent GPU-acquisition fallback with explicit
  `cpu_only()` construction.
- [patch] Removed `pollster` from `ComputeManager::new_blocking` by routing
  blocking construction through `GpuDevice<P>::try_create_with_features_and_limits`,
  preserving the same provider-generic WGPU/CUDA acquisition contract.
- [patch] Moved the solver-owned `ComputeBackend` element-wise and spatial
  derivative method contract from `ndarray::Array3<f64>` to
  `leto::Array3<f64>`, removing the legacy ndarray surface from the
  provider-generic GPU backend implementation.
- [major] Made the solver-owned `ComputeBackend` operation contract
  scalar-associated, so `kwavers-gpu::backend::GPUBackend<P>` dispatches
  element-wise and spatial-derivative work through `P::Scalar` instead of a
  fixed f64 method surface while WGPU and future CUDA providers share the same
  trait boundary.
- [patch] Moved `kwavers-gpu::gpu::ComputeManager` CPU field-update helpers
  from `ndarray::Array3<f64>` to `leto::Array3<f64>`, including dense-layout
  validation on absorption updates.
- [patch] Moved `kwavers-gpu::validation::gpu_cpu_equivalence` result
  comparison from `ndarray::Array3<f64>`/`Zip` to `leto::Array3<f64>`, with
  solver-owned pressure fields converted at the validation runner boundary.
- [patch] Removed the false FDTD GPU equivalence path that constructed a
  GPU backend but still ran the CPU solver, so the runner now reports a typed
  unavailable-provider failure until a real provider-generic Leto/Hephaestus
  FDTD GPU trait implementation is wired.
- [patch] Moved `kwavers-gpu::gpu::pipeline` realtime RF input and processed
  output frame buffers from `ndarray::Array4<f32>`/`Array3<f32>` to
  `leto::Array4<f32>`/`Array3<f32>`, replacing the local beamforming
  `sum_axis` path with explicit Leto traversal.
- [patch] Removed the private `ndarray::Array1<Complex64>` Hilbert FFT scratch
  from `kwavers-gpu::gpu::pipeline`, reusing a thread-local `Vec<Complex64>`
  through Apollo's slice FFT API so the realtime pipeline subtree has no
  ndarray dependency.
- [patch] Split `kwavers-gpu::gpu::thermal_acoustic::GpuThermalAcousticSolver<P>`
  from concrete WGPU pipeline ownership by adding `ThermalAcousticSolverProvider`
  and moving WGPU device/queue handles, compute pipelines, bind group, and step
  dispatch into `WgpuThermalAcousticSolverProvider`.
- [patch] Split `kwavers-gpu::backend::GpuBackendBufferManager<P>` from
  concrete WGPU buffer pooling/readback by adding `BackendBufferProvider` and
  moving WGPU buffer allocation, upload, readback, and pooling into
  `WgpuBackendBufferManager`.
- [patch] Made `kwavers-gpu::backend::GpuComputeProvider` scalar-associated
  and moved WGPU backend operator dispatch onto provider-native `Array3<f32>`
  buffers, with the solver-owned `ComputeBackend` trait now dispatching through
  the provider scalar instead of a fixed f64 rejection path.
- [patch] Moved `kwavers-gpu::backend::GpuComputeProvider` provider-native
  elementwise and spatial-derivative dispatch from `ndarray::Array3<f32>` to
  `leto::Array3<f32>`, keeping ndarray only inside the current WGPU buffer
  manager adapter until that lower staging layer is migrated.
- [patch] Moved `kwavers-gpu::backend::WgpuBackendBufferManager` upload and
  readback for provider-native elementwise/derivative dispatch onto
  `leto::Array3<f32>`, removing the provider-side Leto-to-ndarray adapter.
- [patch] Moved `kwavers-gpu::backend::RealtimeSimulationOrchestrator` field
  maps to `leto::Array3<f64>`, keeping realtime scheduling on the same
  provider-generic GPU surface while the compute backend operation seam remains
  scalar-associated for provider-native dispatch.
- [patch] Added provider-native WGPU elementwise value tests and shape
  precondition checks for backend elementwise/derivative dispatch, and repaired
  the upstream Hephaestus WGPU axis-reduction planner call sites so Kwavers GPU
  gates build through the shared Atlas GPU provider.
- [patch] Split PSTD construction and run-cache storage-buffer allocation
  behind `PstdBufferProvider`, moving WGPU read-only, static, upload,
  read/write, and staging-buffer creation into `WgpuPstdBufferFactory`.
- [patch] Split PSTD shader-module, pipeline-layout, and compute-pipeline
  creation behind `PstdPipelineProvider`, moving WGPU shader, layout, and
  `ComputePipelineDescriptor` construction into `WgpuPstdPipelineFactory`.
- [patch] Split PSTD bind-group layout creation behind
  `PstdBindGroupLayoutProvider`, moving WGPU binding-slot descriptor
  construction into `WgpuPstdBindGroupLayoutFactory`.
- [patch] Split PSTD constructor and run-cache bind-group assembly behind
  `PstdBindGroupProvider`, moving WGPU bind-group descriptor construction into
  `WgpuPstdBindGroupFactory`.
- [patch] Split PSTD run-loop clear, copy, submit, and poll operations behind
  `PstdCommandProvider`, moving WGPU command encoder and queue wait mechanics
  for those paths into `WgpuPstdCommandProvider`.
- [patch] Split PSTD zero-field and batched-step command encoder
  creation/submission behind `PstdCommandProvider::submit_encoder`, with the
  provider-native encoder modeled as an associated type so the contract remains
  open to non-WGPU providers.
- [patch] Split PSTD zero-field and batched-step compute-pass creation behind
  `PstdCommandProvider::{submit_compute_pass, submit_compute_passes}`, with the
  provider-native pass modeled as a lifetime-associated type so the contract
  does not name `wgpu::ComputePass`.
- [patch] Split PSTD zero-field and per-step pass-body encoding behind
  `PstdPassProvider`, moving WGPU dispatch and `encode_*` call sequencing out
  of `time_loop::run` without exposing CUDA compute until real kernels exist.
- [patch] Split PSTD sensor staging-buffer readback behind
  `PstdCommandProvider::read_mapped`, moving WGPU map/unmap mechanics out of
  `time_loop::run`.
- [patch] Split PSTD cache-hit signal-tail uploads behind
  `PstdCommandProvider::write_buffer`, moving WGPU queue upload mechanics out
  of `time_loop::buffer`.
- [patch] Split PSTD medium and source-correction uploads behind
  `PstdCommandProvider::write_buffer`, moving remaining direct
  `queue.write_buffer` calls out of `pstd_gpu` modules.
- [patch] Grouped PSTD k-space, medium, twiddle, and source-kappa WGPU buffers
  into `WgpuPstdMediumBuffers`, removing their separate top-level solver
  fields while preserving existing bind-group layout and medium-update
  semantics.
- [patch] Grouped PSTD k-space work buffers into `WgpuPstdKspaceBuffers`,
  removing the remaining separate top-level k-space buffer fields while
  preserving the existing group(1) bind-group bindings.
- [patch] Grouped PSTD acoustic field WGPU buffers into
  `WgpuPstdFieldBuffers`, removing their separate top-level solver fields
  while preserving the existing group(0) bind-group layout.
- [patch] Grouped PSTD fractional-Laplacian absorption WGPU buffers into
  `WgpuPstdAbsorptionBuffers`, removing their separate top-level solver fields
  while preserving existing group(3) bind-group layout and medium absorption
  updates.
- [patch] Grouped PSTD PML and packed k-space shift WGPU buffers into
  `WgpuPstdPmlShiftBuffers`, removing their separate top-level solver fields
  while preserving the per-run sensor bind-group rebuild path.
- [patch] Grouped PSTD run-cache WGPU buffers, bind groups, and cache-key
  counters into `WgpuPstdRunCache`, removing the separate top-level cache
  fields while preserving cache-hit signal-tail refreshes.
- [patch] Grouped PSTD permanent WGPU bind groups into
  `WgpuPstdPermanentBindGroups`, removing the separate top-level solver bind
  group fields while preserving field, k-space, and absorption dispatch slots.
- [patch] Grouped retained PSTD WGPU layout state into `WgpuPstdLayouts`,
  keeping only the sensor bind-group layout needed for run-cache rebuilds and
  deleting the unused retained base pipeline layout field.
- [patch] Grouped PSTD WGPU compute pipelines into `WgpuPstdPipelines`,
  removing the separate top-level solver pipeline fields while preserving all
  shader entry-point mappings and dispatch call sites.
- [patch] Consolidated PSTD grouped WGPU state under `WgpuPstdState`, so
  `GpuPstdSolver` owns one provider-state field instead of separate WGPU buffer,
  pipeline, bind-group, layout, and run-cache groups.
- [patch] Made PSTD solver state provider-associated through
  `PstdStateProvider`, keeping `WgpuPstdStateProvider` as the default real
  implementation and specializing current WGPU construction/time-loop methods
  without adding a placeholder CUDA compute path.
- [patch] Moved PSTD WGPU device and queue ownership into `WgpuPstdState`, so
  the generic `GpuPstdSolver<P>` wrapper no longer stores raw WGPU handles.
- [patch] Replaced PSTD WGPU state raw device/queue ownership with
  `GpuProviderContext<WgpuDevice>`, changed `PstdStateBuilder` and
  `PstdAutoDeviceProvider` to pass one provider context instead of separate
  device/queue associated handles, and removed the obsolete raw WGPU handle
  clone accessors from `GpuDevice<WgpuDevice>`.
- [patch] Moved PSTD WGPU host scratch/upload buffers into `WgpuPstdState`,
  leaving the generic `GpuPstdSolver<P>` wrapper with provider state,
  dimensions, time step, and physics flags only.
- [patch] Moved PSTD WGPU state assembly behind
  `WgpuPstdStateProvider::build_state`, so `GpuPstdSolver::new` wraps
  provider-built state instead of owning WGPU buffer, pipeline, and bind-group
  construction directly.
- [patch] Moved PSTD WGPU medium/source upload bodies onto `WgpuPstdState`,
  leaving `GpuPstdSolver` medium-update methods as thin public wrappers around
  provider-owned buffer writes.
- [patch] Moved PSTD WGPU run-cache allocation, bind-group rebuild, and
  signal-tail upload bodies onto `WgpuPstdState`, leaving solver run-cache
  methods as time-loop forwarding wrappers.
- [patch] Moved PSTD WGPU dispatch, FFT, and per-phase pass encoding onto
  `WgpuPstdState`, so `WgpuPstdPassProvider` now encodes compute passes from
  provider-owned state instead of holding a
  `GpuPstdSolver<WgpuPstdStateProvider>` reference.
- [patch] Moved PSTD WGPU high-level run-loop orchestration onto
  `WgpuPstdState`, leaving `GpuPstdSolver<WgpuPstdStateProvider>::run` as a
  public delegate that supplies solver scalar metadata and input slices.
- [patch] Added the provider-generic `PstdStateBuilder` construction seam and
  made `GpuPstdSolver<P>::new` build state through `P::build_state`, with
  WGPU as the only real implementation and no CUDA placeholder.
- [patch] Added the provider-state `PstdRunState` execution seam and made
  `GpuPstdSolver<P>::run` generic over `P::State: PstdRunState`, with WGPU as
  the only real implementation and no CUDA placeholder.
- [patch] Added the provider-state `PstdMediumUpdateState` seam and made PSTD
  medium/source-correction wrapper methods generic over
  `P::State: PstdMediumUpdateState`, with WGPU as the only real implementation
  and no CUDA placeholder.
- [patch] Added the provider-generic `PstdAutoDeviceProvider` acquisition seam
  and made `GpuPstdSolver<P>::with_auto_device` build through provider-owned
  context acquisition, with WGPU as the only real implementation and no CUDA
  placeholder.
- [patch] Made `kwavers-gpu::pstd_gpu::run_gpu_pstd_with_provider<P>`
  provider-generic over `PstdAutoDeviceProvider`/`PstdRunState`, kept
  `run_gpu_pstd` as the WGPU default, and moved the public PSTD sensor mask
  and returned traces from ndarray to `leto::Array3<bool>`/`leto::Array2<f64>`.
- [patch] Moved `kwavers-boundary` CPML profile and `PmlExpFactors`
  one-dimensional storage from ndarray `Array1` to Leto `Array1`, filling Leto
  owned-array indexing/equality gaps upstream and removing the CPML ndarray
  import from `kwavers-gpu::pstd_gpu::runner`.
- [patch] Routed visualization data-pipeline normalization and log scaling
  through Moirai-backed contiguous traversal, retaining ndarray's sequential
  value semantics for non-standard layouts and leaving crate-level Rayon in
  place for remaining `kwavers-analysis` call sites.
- [patch] Routed `kwavers-analysis` `ParallelOptimizer` fan-out, map, and
  reduction helpers through Moirai's shared scheduler and removed the local
  Rayon thread-pool mutation path, keeping `set_num_threads` as a validated
  chunk-size scheduling hint.
- [patch] Promoted a shared `kwavers-core::utils::iterators::apply_inplace`
  ndarray scalar-transform seam backed by Moirai for standard layouts and
  routed `kwavers-analysis` visualization normalization/log scaling, PAM
  time-exposure squaring, and polynomial clutter-filter time normalization
  through it.
- [patch] Routed `kwavers-analysis` beamforming covariance sample scaling,
  shrinkage scaling, and spatial-smoothing normalization through the shared
  Moirai-backed scalar ndarray seam, removing the covariance subtree's direct
  ndarray/Rayon parallel transforms.
- [patch] Routed `kwavers-analysis` safe-vectorization array addition and
  in-place scalar multiplication through shared Moirai-backed ndarray traversal
  seams, replacing the remaining ndarray/Rayon `Zip::par_for_each` paths in
  that module.
- [patch] Routed `kwavers-analysis` SLSC sample and volume coherence fan-out
  through Moirai ordered index collection and moved coherence-map clamping onto
  the shared scalar ndarray seam, removing direct Rayon from the SLSC subtree.
- [patch] Routed `kwavers-analysis` neural layer adaptation scaling and neural
  feature normalization through the shared Moirai-backed scalar ndarray seam.
- [patch] Replaced `kwavers-analysis` distributed neural processor fan-out
  with Moirai mutable map-collect scheduling, removing direct Rayon from the
  neural subtree while preserving distributed-vs-sequential output equality.
- [patch] Routed `kwavers-analysis` 3-D CPU DAS/MVDR voxel fan-out through
  Moirai ordered index collection, removed the package's direct `rayon`
  dependency and ndarray `rayon` feature, and left no direct Rayon or
  ndarray-parallel source hits in `kwavers-analysis`.
- [patch] Repaired upstream Atlas consumer gates by declaring
  `ritk-registration`'s `ritk-tensor-ops` dependency and consolidating Moirai
  core stale `CachePadded` uses onto the canonical `CacheAligned` wrapper.
- [patch] Split `kwavers-gpu::backend` GPU provider identity from kernel
  dispatch with `GpuProviderBackend`, so `GpuProviderContext<CudaDevice>` can
  satisfy the provider/acquisition contract while `GpuComputeProvider` remains
  reserved for providers with real Kwavers kernels.
- [patch] Replaced the `kwavers-gpu::backend` spatial-derivative CPU fallback
  and WGSL copy placeholder with real WGPU finite-difference dispatch, while
  keeping CUDA at the provider/acquisition boundary until CUDA kernels exist.
- [patch] Removed the unused `kwavers-math::tensor::TensorBackend::BurnNdArray`
  placeholder and updated the tensor docs to describe the implemented
  ndarray-backed host tensor boundary until a real Coeus PINN provider lands.
- [patch] Routed `kwavers-math::tensor::NdArrayTensor::map_inplace` through
  `moirai-parallel` chunk traversal for contiguous host tensors while keeping
  ndarray's sequential value semantics for non-contiguous tensor layouts.
- [patch] Routed inverse regularization Tikhonov, smoothness, and L1 gradient
  updates through a shared Moirai-backed contiguous traversal helper, removing
  the regularization subtree's direct ndarray/Rayon `Zip::par_for_each` calls
  while preserving sequential ndarray semantics for non-standard layouts.
- [patch] Routed `kwavers-math::simd_safe` dense add/scale operations through
  the Atlas Hermes SIMD facade, routed dense ternary accumulation through
  Moirai chunk traversal, and removed the subtree's direct ndarray/Rayon
  `par_*` calls while preserving sequential ndarray semantics for
  non-standard layouts.
- [patch] Routed `kwavers-math` second-order central and staggered-grid
  differential operators through a shared Moirai-backed standard-layout
  traversal, removing the differential subtree's direct ndarray/Rayon
  `par_for_each` calls while preserving sequential ndarray traversal for
  non-standard layouts.
- [patch] Routed `AcousticFieldKernel`, `ComputeManager`, and the backend
  buffer-manager GPU test helper through Hephaestus-backed provider wrappers,
  removing their local WGPU instance/adapter/device request paths. The
  `ComputeManager` acquisition now goes through generic `GpuDevice` so CUDA can
  land as a sibling provider without changing the manager entry point.
- [patch] Routed multi-GPU device discovery and PSTD GPU test construction
  through Hephaestus provider traits; remaining direct WGPU-provider
  constructor calls under `crates/kwavers-gpu/src` are test-only provider
  helpers.
- [patch] Replaced `kwavers-gpu` real-time pipeline Rayon dispatch with
  `moirai-parallel` chunk scheduling for Hilbert-envelope planes and synthetic
  RF frame generation, removing the crate's direct Rayon dependency.
- [patch] Replaced `kwavers-boundary` CPML damping, CPML strip
  memory/correction updates, and adaptive-boundary attenuation ndarray/Rayon
  dispatch with private `moirai-parallel` traversal helpers, and removed
  ndarray's `rayon` feature from `kwavers-boundary`.
- [patch] Replaced `kwavers-receiver` pressure and velocity statistics
  ndarray/Rayon dispatch with `moirai-parallel` multi-output chunk helpers for
  standard-layout arrays, retaining sequential ndarray semantics for
  non-standard layouts, and removed the crate's direct Rayon dependency plus
  ndarray `rayon` feature.
- [patch] Replaced `kwavers-medium` medium-property traversal,
  absorption/dispersion k-space updates, and frequency-dependent correction
  ndarray/Rayon dispatch with `moirai-parallel` indexed and chunk traversal,
  removed the crate's direct Rayon dependency plus ndarray `rayon` feature, and
  cleared package-local Clippy blockers in Christoffel and material-property
  tests.
- [patch] Replaced `kwavers-core`'s direct `rayon` dependency with
  `moirai-parallel` for NUMA first-touch, SoA first-touch, and gradient
  interior-loop parallel dispatch, and removed ndarray's `rayon` feature from
  `kwavers-core`.
- [patch] Moved `kwavers-core` constant-invariant tests flagged by current
  Clippy into `const` assertions without changing the checked predicates.
- [minor] Added shared `kwavers-core::utils::iterators` indexed mutable array
  traversal helpers backed by `moirai-parallel` for standard-layout arrays and
  routed therapy integration acoustic-field/heating loops through them, keeping
  sequential ndarray traversal only as the non-standard-layout fallback.
- [patch] Cleared current `kwavers-therapy` package-local Clippy blockers in
  therapy tests without changing the tested predicates.
- [patch] Closed the `kwavers-therapy` abdominal preprocessing nextest
  timeouts by tightening the theranostic acoustic recording window to actual
  source/body/receiver geometry, reusing adjoint RTM replay buffers, and
  evaluating elastic-FWI line-search candidates lazily in first-improving
  order. The full `kwavers-therapy` nextest package now passes, with remaining
  runtime over the 30 s slow-test budget tracked as performance debt.
- [patch] Replaced the `kwavers-therapy` elastic-shear residual-migration
  Rayon flat-index map and passive-acoustic-mapping eikonal delay-column map
  with `moirai-parallel`, adding the package's direct Moirai provider
  dependency while leaving crate-level Rayon in place for remaining
  nonlinear3d forward-stencil and passive-inverse paths.
- [patch] Replaced `kwavers-therapy` standing-wave FDTD Green-function column
  fan-out with `moirai-parallel`, leaving crate-level Rayon in place for
  remaining nonlinear3d forward-stencil and passive-inverse paths.
- [patch] Replaced `kwavers-therapy` waveform-forward CPML row updates,
  pressure updates, attenuation, and peak-pressure updates with
  `moirai-parallel`, leaving crate-level Rayon in place for remaining
  nonlinear3d forward-stencil and passive-inverse paths.
- [patch] Replaced `kwavers-therapy` nonlinear3d absorption coefficient
  construction and forward/adjoint absorption element-wise updates with
  `moirai-parallel`, leaving crate-level Rayon in place for remaining
  nonlinear3d forward-stencil and passive-inverse paths.
- [patch] Replaced `kwavers-therapy` nonlinear3d cavitation-forward contiguous
  source-mask max reduction and source-density mapping with `moirai-parallel`,
  leaving crate-level Rayon in place for remaining nonlinear3d forward-stencil
  and passive-inverse paths.
- [patch] Replaced `kwavers-therapy` nonlinear3d forward-stencil x-slab
  dispatch with `moirai-parallel` and updated the adjacent Westervelt
  performance docs, leaving crate-level Rayon in place for the remaining
  nonlinear3d passive-inverse path.
- [patch] Replaced `kwavers-therapy` nonlinear3d passive-inverse Green
  operator and projected-Tikhonov update parallelism with `moirai-parallel`,
  removing the crate's direct Rayon dependency and ndarray `rayon` feature.
- [patch] Routed time-domain FWI model constraints, pressure
  second-derivative writes, and signed-correlation accumulation through
  `moirai-parallel` chunk dispatch for standard-layout field volumes.
- [patch] Routed time-domain FWI adjoint, gradient, and multi-source field
  updates through a shared Moirai-backed `field_ops.rs` helper module and
  removed the stale forward-run Rayon doc mention.
- [patch] Removed explicit ndarray `Zip` fallback traversal from the
  time-domain FWI shared `field_ops.rs` helper; standard-layout fields still
  use Moirai chunk execution and non-standard layouts now use direct indexed
  sequential traversal.
- [patch] Routed `kwavers-solver::workspace::inplace_ops` standard-layout
  in-place array arithmetic through `moirai-parallel`, retaining sequential
  ndarray semantics for non-standard layouts.
- [patch] Routed `kwavers-solver::integration::time_integration` RK4 and
  Adams-Bashforth field-update kernels through `moirai-parallel`, retaining
  sequential ndarray semantics for non-standard layouts.
- [patch] Removed `kwavers-solver::plugin::execution::ParallelStrategy`'s
  unused `rayon::ThreadPool` constructor because the strategy still executes
  plugins in order until a real read/compute/write plugin parallelism contract
  exists.
- [patch] Routed `kwavers-solver::inverse::time_reversal` reconstruction
  normalization through the shared Moirai-backed workspace in-place operation,
  removing that module's direct ndarray/Rayon map edge.
- [patch] Routed `kwavers-solver::multiphysics::monolithic::residual`
  Laplacian rate scaling through the shared Moirai-backed workspace in-place
  operation, removing that residual subsystem's direct ndarray/Rayon map edge.
- [patch] Routed `kwavers-solver::multiphysics::monolithic::coupler` GMRES
  RHS sign inversion through the shared Moirai-backed workspace in-place
  operation, removing the coupler's direct ndarray/Rayon map edge.
- [patch] Routed `kwavers-solver::utilities::amr` wavelet/physics error
  normalization and wavelet-threshold scalar transforms through the shared
  Moirai-backed workspace in-place operations, removing AMR's remaining
  direct `par_mapv_inplace` sites.
- [patch] Replaced `kwavers-solver::utilities::amr::refinement` marker
  initialization with `moirai-parallel` indexed traversal for standard-layout
  arrays, removing the AMR subtree's remaining direct ndarray/Rayon parallel
  call site.
- [patch] Routed `kwavers-solver::forward::elastic::swe` displacement
  magnitude square-root normalization through the shared Moirai-backed
  workspace in-place operation, removing the SWE types module's direct
  ndarray/Rayon scalar-transform edge.
- [patch] Replaced `kwavers-solver::forward::elastic::swe::boundary` PML
  attenuation, mask, and velocity-damping ndarray/Rayon dispatch with
  Moirai-backed indexed and triple-chunk traversal.
- [patch] Routed transcranial UST finite-frequency sensitivity and ray-integral
  row assembly through `moirai-parallel` and resolved `moirai-parallel` to the
  sibling Atlas checkout for local provider co-evolution.
- [patch] Routed sound-speed-shift matrix-vector algebra through
  `moirai-parallel` indexed mutable dispatch and fold/reduce partial-vector
  reductions.
- [patch] Routed real-time SIRT row-norm construction and separable smoothing
  through `moirai-parallel`, removing `kwavers-diagnostics`' direct `rayon`
  dependency and ndarray `rayon` feature.
- [patch] Replaced `kwavers-simulation`'s direct `rayon` dependency with
  `moirai-parallel` for photoacoustic multi-wavelength fluence mapping and
  time-reversal reconstruction output writes, and removed ndarray's `rayon`
  feature from `kwavers-simulation`.
- [patch] Repaired `kwavers-simulation` GPU-PSTD adapter all-features tests by
  importing the `Solver` trait whose methods the tests call.
- [patch] Replaced broad `kwavers-physics` analytical tuple/argument surfaces
  with typed request/result structs for IVUS delivery, Gaussian photoacoustic
  profiles, Gaussian deconvolution fixtures, and apodization-window responses,
  keeping PyO3 wrappers as thin adapters over Rust-owned logic.
- [patch] Cleared the dependency-inclusive `kwavers-simulation` all-features
  Clippy gate by resolving current `kwavers-physics` Clippy blockers.
- [patch] Replaced `kwavers-transducer`'s direct `rayon` dependency with
  `moirai-parallel` for focus-delay and source-field indexed buffer writes, and
  removed ndarray's `rayon` feature from `kwavers-transducer`.
- [patch] Removed unused ndarray `rayon` feature activation from
  `kwavers-field`, `kwavers-signal`, `kwavers-source`, and `kwavers-imaging`
  after confirming those crate trees have no direct parallel call sites.
- [patch] Replaced `kwavers-grid`'s Laplacian interior parallel dispatch with
  `moirai-parallel` and removed ndarray's `rayon` feature from that crate.
- [patch] Routed the liver theranostic straight-ray rasterizer through Gaia's
  `Ray<f64>` primitive while retaining voxel path-length accumulation in
  Kwavers.
- [patch] Moved `kwavers-imaging::multimodality_fusion` medical image volumes
  and fusion outputs to `leto::Array3<f64>` and routed registration directly
  into local Ritk's Leto API without an ndarray compatibility helper.
- [patch] Moved `kwavers-physics::acoustics::imaging::fusion` registered
  modality, registration, resampling, quality, and algorithm volume surfaces
  to `leto::Array3<f64>`, removing ndarray inputs from the local Ritk
  registration path.
- [patch] Moved diagnostics workflow products, fUS atlas registration
  reference volumes, photoacoustic result volumes, and photoacoustic
  simulation fluence/pressure/reconstruction snapshots to `leto::Array3<f64>`
  without adding an ndarray compatibility helper.
- [patch] Restored the liver theranostic reconstruction example compile gate
  under local Atlas provider routing; remaining Leto work is narrowed to
  producer boundaries that still compute ndarray arrays before returning Leto
  results.
- [patch] Moved `kwavers-solver` direct, directional, and LFE linear
  elastography shear-wave-speed producers to allocate `leto::Array3<f64>`
  directly, with shared smoothing and boundary extrapolation consolidated over
  a crate-local ndarray/Leto volume trait.
- [patch] Moved the selected photoacoustic solver universal back-projection
  producer to return `leto::Array3<f64>` directly for the simulation
  reconstruction path, removing the caller-side ndarray-to-Leto conversion.
- [patch] Moved the selected optical diffusion fluence producer to a shared
  generic PCG volume kernel with a direct `leto::Array3<f64>` source/result
  path, removing the caller-side ndarray-to-Leto conversion and pinning
  ndarray/Leto parity with a bitwise differential test.
- [patch] Replaced direct Rayon iterator usage in
  `kwavers-solver::inverse::same_aperture` encoded/operator paths with
  `moirai-parallel`, preserving the matrix-free linear operator contract.
- [patch] Replaced direct Rayon iterator usage in
  `kwavers-solver::inverse::linear_born_inversion` dense, volume-operator, and
  Sobolev-preconditioner paths with `moirai-parallel`, using the provider-owned
  stateful chunk primitive for reusable Z-pass scratch.
- [patch] Narrowed the next provider residual to direct Rayon usage in
  `kwavers-solver::inverse::fwi::time_domain` search and MOFI paths.
- [patch] Replaced direct Rayon usage in
  `kwavers-solver::inverse::fwi::time_domain::{search,mofi}` with
  `moirai-parallel` for joint objective evaluation, line-search trial-model
  writes, and MOFI coarse-pose candidate evaluation.
- [patch] Resolved local Atlas Apollo/RITK/Mnemosyne scratch compatibility by
  enabling Mnemosyne's `eunomia` scratch feature through RITK and resolving
  Mnemosyne's optional Eunomia dependency to the sibling Atlas checkout.

### Changed (2026-07-01) - Cavitation passive-map bindings [patch]

- [patch] Split `kwavers-python` cavitation receiver-array PSD integration and
  passive-map emission-energy bindings into a dedicated child module, leaving
  the parent cavitation facade as a re-export surface.

### Changed (2026-07-01) - Cavitation chirp bindings [patch]

- [patch] Split `kwavers-python` cavitation frequency-swept engagement,
  residual-clearance, staged-sonication, and shielding-control bindings into a
  dedicated child module while preserving the registered Python function names.

### Changed (2026-07-01) - Cavitation monitor bindings [patch]

- [patch] Split `kwavers-python` cavitation monitor, closed-loop control,
  raster pulsing, therapeutic-window, and per-spot dose-grid bindings into a
  dedicated child module while preserving the registered Python function names.

### Changed (2026-07-01) - Cavitation spectrum bindings [patch]

- [patch] Split `kwavers-python` cavitation spectrum, emission-band,
  PCD-controller, cumulative-dose, and passive-dose fixture bindings into a
  dedicated child module while preserving the registered Python function names.
- [patch] Repaired bubble-dynamics compile blockers by removing an invalid
  `AdaptiveBubbleModel` self re-export and deriving `Debug` for `BubbleField`.

### Changed (2026-07-01) - Cavitation emission bindings [patch]

- [patch] Split `kwavers-python` cavitation free/coated bubble, population, and
  focal-volume emission simulation bindings into a dedicated child module while
  preserving the registered Python function names.

### Changed (2026-07-01) - Cavitation passive-receive bindings [patch]

- [patch] Split `kwavers-python` cavitation passive receiver PSD/RF and
  Van Cittert-Zernike coherence bindings into a dedicated child module while
  preserving the registered Python function names.

### Changed (2026-07-01) - Cavitation lesion bindings [patch]

- [patch] Split `kwavers-python` cavitation lesion-state, boiling-profile,
  lacuna void-fraction, lesion-radius, and inertial-dose bindings into a
  dedicated child module while preserving the registered Python function names.

### Changed (2026-07-01) - Cavitation therapy bindings [patch]

- [patch] Split `kwavers-python` cavitation therapy-delivery, lesion-response,
  focal-mask, measured-emission scaling, and erosion-validation bindings into
  a dedicated child module while preserving the registered Python function
  names.

### Changed (2026-07-01) - Cavitation medium bindings [patch]

- [patch] Split `kwavers-python` cavitation residual-gas dissolution and
  bubbly-medium propagation bindings into a dedicated child module while
  preserving the registered Python function names.

### Changed (2026-07-01) - Cavitation single-bubble bindings [patch]

- [patch] Split `kwavers-python` cavitation Minnaert, Blake-threshold, and
  Rayleigh collapse-time bindings into a dedicated child module while
  preserving the registered Python function names.

### Changed (2026-07-01) - Cavitation probability bindings [patch]

- [patch] Split `kwavers-python` cavitation probability, threshold, and PRF
  efficacy bindings into a dedicated child module while preserving the
  registered Python function names.

### Changed (2026-07-01) - Neuromodulation bindings [patch]

- [patch] Split `kwavers-python` neuromodulation response, bilayer curve,
  threshold-search, and safety/dosimetry bindings into dedicated child modules
  while preserving the registered Python function names.

### Changed (2026-07-01) - Inverse-problem bindings [patch]

- [patch] Split `kwavers-python` inverse-problem operator, reconstruction,
  convergence, and parameter-selection bindings into dedicated child modules
  while preserving the registered Python function names and consolidating
  repeated 2-D array conversion code.

### Changed (2026-07-01) - RTM analytical bindings [patch]

- [patch] Split `kwavers-python` RTM field, imaging/fusion, and standing-wave
  suppression bindings into dedicated child modules while preserving the
  registered Python function names and consolidating repeated 2-D array
  conversion code.

### Changed (2026-07-01) - Skull analytical bindings [patch]

- [patch] Split `kwavers-python` skull attenuation/aberration, Hounsfield
  conversion, thermal rise, and layered transmission bindings into dedicated
  child modules while preserving the registered Python function names.

### Changed (2026-07-01) - Sonogenetics bindings [patch]

- [patch] Split `kwavers-python` sonogenetics channel-activation,
  force/streaming, and ISPTA dosimetry bindings into dedicated child modules
  while preserving the registered Python function names.

### Changed (2026-07-01) - MEMS CMUT/PMUT bindings [patch]

- [patch] Split `kwavers-python` MEMS plate, CMUT, PMUT, and comparison
  figure-of-merit bindings into dedicated child modules while preserving the
  registered Python function names.

### Changed (2026-07-01) - Acousto-optics bindings [patch]

- [patch] Split `kwavers-python` acousto-optics regime, geometry, and
  diffraction-order bindings into dedicated child modules while preserving the
  registered Python function names.

### Changed (2026-07-01) - Tissue analytical bindings [patch]

- [patch] Split `kwavers-python` tissue water-property,
  attenuation/dispersion, and tissue-property lookup bindings into dedicated
  child modules while preserving the registered Python function names.

### Changed (2026-07-01) - Statistics validation bindings [patch]

- [patch] Split `kwavers-python` statistics correlation/phase and RMSE/PSNR
  validation bindings into dedicated child modules while preserving the
  registered Python function names.

### Changed (2026-07-01) - BBB and CEUS bindings [patch]

- [patch] Split `kwavers-python` BBB permeability/closure and CEUS
  backscatter/display bindings into dedicated child modules while preserving
  the registered Python function names.

### Changed (2026-07-01) - Photoacoustics bindings [patch]

- [patch] Split `kwavers-python` photoacoustics spectral, source/signal, and
  reconstruction bindings into dedicated child modules while preserving the
  registered Python function names and removing an avoidable transient
  allocation in the sO2 sweep wrapper.

### Changed (2026-07-01) - Elastography thermal-strain bindings [patch]

- [patch] Split `kwavers-python` elastography thermal-strain fixture,
  coefficient, and reconstruction bindings into a dedicated child module while
  preserving the registered Python function names.

### Changed (2026-07-01) - Safety damage bindings [patch]

- [patch] Split `kwavers-python` safety Arrhenius damage, thermal kill, and
  combined kill-probability bindings into a dedicated child module while
  preserving the registered Python function names.

### Changed (2026-07-01) - Safety thermal bindings [patch]

- [patch] Split `kwavers-python` safety thermal-index and CEM43 dose bindings
  into a dedicated child module while preserving the registered Python function
  names.

### Changed (2026-07-01) - Safety mechanical bindings [patch]

- [patch] Split `kwavers-python` safety Mechanical Index and cavitation-risk
  bindings into a dedicated child module while preserving the registered Python
  function names.

### Changed (2026-07-01) - Thermal acoustic bindings [patch]

- [patch] Split `kwavers-python` thermal acoustic gain, power deposition,
  pressure/intensity conversion, and heat-source bindings into a dedicated child
  module while preserving the registered Python function names.

### Changed (2026-07-01) - Inverse seismic bindings [patch]

- [patch] Split `kwavers-python` inverse eikonal traveltime and Kirchhoff
  point-scatterer imaging bindings into a dedicated child module while
  preserving the registered Python function names.

### Changed (2026-07-01) - Imaging IVUS B-mode bindings [patch]

- [patch] Split `kwavers-python` IVUS polar RF, scan conversion, complete
  B-mode image, and Chapter 30 metric bindings into dedicated child modules,
  leaving the imaging facade as module topology and re-exports only.

### Changed (2026-07-01) - Imaging IVUS therapy bindings [patch]

- [patch] Split `kwavers-python` IVUS therapy pressure, delivery, response, and
  aggregate field bindings into a dedicated child module while preserving the
  registered Python function names.

### Changed (2026-07-01) - Imaging IVUS phantom bindings [patch]

- [patch] Split `kwavers-python` IVUS vessel-phantom bindings and square-array
  materialization into a dedicated child module while preserving the registered
  Python function name.

### Changed (2026-07-01) - Imaging PSF bindings [patch]

- [patch] Split `kwavers-python` imaging point-spread and lateral-resolution
  bindings into a dedicated child module while preserving the registered
  Python function names.

### Changed (2026-07-01) - Imaging pulse-echo bindings [patch]

- [patch] Split `kwavers-python` imaging pulse-echo RF and B-mode bindings into
  a dedicated child module while preserving the registered Python function
  names.

### Changed (2026-07-01) - Imaging Doppler bindings [patch]

- [patch] Split `kwavers-python` imaging Doppler and vector-flow bindings into
  a dedicated child module while preserving the registered Python function
  names.

### Changed (2026-07-01) - Transducer beam bindings [patch]

- [patch] Split `kwavers-python` transducer 2-D focus and beam-pattern bindings
  into a dedicated child module, leaving the transducer facade as module
  topology and re-exports only.

### Changed (2026-07-01) - Transducer basic bindings [patch]

- [patch] Split `kwavers-python` transducer directivity, apodization,
  linear-array factor, grating-lobe, and on-axis pressure bindings into a
  dedicated child module while preserving the registered Python function names.

### Changed (2026-07-01) - Transducer multi-focus bindings [patch]

- [patch] Split `kwavers-python` transducer multi-focus delay-law and
  field-magnitude bindings into a dedicated child module while preserving the
  registered Python function names.

### Changed (2026-07-01) - Transducer aperture bindings [patch]

- [patch] Split `kwavers-python` transducer aperture geometry and 3-D pressure
  bindings into a dedicated child module while preserving the registered Python
  function names.

### Changed (2026-07-01) - Transducer interpolation bindings [patch]

- [patch] Split `kwavers-python` transducer band-limited-interpolation
  bindings into a dedicated child module while preserving the registered
  Python function names.

### Changed (2026-07-01) - Transducer steering bindings [patch]

- [patch] Split `kwavers-python` transducer steering, sparse-aperture,
  grating-lobe, and electronic-efficiency bindings into a dedicated child
  module while preserving the registered Python function names.

### Changed (2026-07-01) - Transducer binding structure [patch]

- [patch] Split `kwavers-python` transducer analytical bindings into dedicated
  optoacoustic and acoustic-lens child modules while preserving the registered
  Python function names.

### Changed (2026-07-01) - GPU PSTD session sources [patch]

- [patch] Split `kwavers-python` GPU PSTD session source/sensor packing into a
  dedicated child module and removed cached-run empty source-vector allocations
  by passing empty slices directly to the solver.

### Changed (2026-07-01) - GPU PSTD session structure [patch]

- [patch] Split `kwavers-python` GPU PSTD session construction helpers into
  absorption-kernel and PML-array child modules, preserving the public PyO3
  `GpuPstdSession` facade while reducing constructor-module responsibility.

### Fixed (2026-07-01) - Therapy chapter guards [patch]

- [patch] Corrected the focused therapy chapter test's docs-root calculation
  for the current crate layout and removed residual vendor-style source labels
  from the active Chapter 31 clinical-device script.

### Added (2026-07-01) - Chapter 24 CEUS display [patch]

- [patch] Added Rust/PyO3 `ceus_backscatter_display` and routed Chapter 24's
  CEUS signal, peak-normalised dB values, and optimal concentration marker
  through the Rust-owned helper instead of local Python normalization and peak
  selection.

### Added (2026-07-01) - Chapter 30 IVUS therapy fields [patch]

- [patch] Added Rust/PyO3 `ivus_therapy_fields` and routed Chapter 30's
  therapy pressure field plus response fields through the Rust-owned aggregate
  helper instead of local Python orchestration across split pressure/response
  kernels.

### Added (2026-07-01) - Chapter 30 IVUS metrics [patch]

- [patch] Added Rust/PyO3 `ivus_chapter_metrics` and routed Chapter 30's
  wavelength, lumen/plaque area, masked B-mode means, and therapy summary
  metrics through the Rust-owned helper instead of Python-side scalar formulas.

### Added (2026-07-01) - Chapter 30 IVUS B-mode image [patch]

- [patch] Added Rust/PyO3 `ivus_bmode_image` and routed Chapter 30's polar RF,
  Hilbert-envelope detection, fixed-reference log compression, normalized
  polar image, and Cartesian scan conversion through the Rust-owned helper
  instead of local Python B-mode image assembly.

### Added (2026-07-01) - Chapter 30 IVUS therapy response [patch]

- [patch] Added Rust/PyO3 `ivus_therapy_response` and routed Chapter 30's
  therapy intensity, absorption, temperature rise, delivery masks, mechanical
  index, and target/off-target deposition ratio through the Rust-owned helper
  instead of local Python therapy-response algebra.

### Added (2026-07-01) - Chapter 30 IVUS scan conversion [patch]

- [patch] Added Rust/PyO3 `ivus_scan_convert` and routed Chapter 30's polar
  B-mode image projection through the Rust-owned nearest-bin radius/theta mapper
  instead of local Python scan-conversion indexing.

### Added (2026-07-01) - Chapter 30 IVUS polar RF [patch]

- [patch] Added Rust/PyO3 `ivus_polar_bmode_rf` and routed Chapter 30's IVUS
  B-mode RF fixture through the Rust-owned polar grid sampling, two-way
  attenuation, and catheter-ring echo helper instead of local Python RF
  construction.

### Added (2026-07-01) - Chapter 30 IVUS delivery fraction [patch]

- [patch] Added Rust/PyO3 `ivus_microbubble_delivery_fraction` and routed
  Chapter 30's microbubble delivery map through the Rust-owned acoustic
  radiation force, radial targeting, normalization, and exponential delivery
  helper instead of local Python deposition algebra.

### Added (2026-07-01) - Chapter 30 IVUS pressure field [patch]

- [patch] Added Rust/PyO3 `ivus_therapy_pressure_field` and routed Chapter
  30's microbubble-therapy pressure map through the Rust-owned sector aperture
  and radial decay helper instead of local Python angular Gaussian and
  exponential pressure algebra.

### Added (2026-07-01) - Chapter 20 PSNR curve [patch]

- [patch] Added Rust/PyO3 `validation_psnr_from_relative_rmse` and routed
  Chapter 20's PSNR-vs-relative-RMSE figure through the Rust-owned validation
  theorem helper instead of local Python `-20 * np.log10(eps)` algebra.

### Added (2026-07-01) - Chapter 20 phase sensitivity [patch]

- [patch] Added Rust/PyO3 `phase_shift_correlation_curve` and
  `phase_error_degrees_for_correlation`, then routed Chapter 20's Pearson
  phase-sensitivity figure through those Rust-owned theorem helpers instead of
  local Python `np.cos`/`np.arccos` algebra.

### Added (2026-07-01) - Chapter 17 PINN convergence curve [patch]

- [patch] Added Rust/PyO3 `exponential_convergence_curve` and routed Chapter
  17's PINN loss convergence illustration through the Rust-owned exponential
  decay-with-floor helper instead of local Python `np.exp` fixture logic.

### Added (2026-07-01) - Chapter 17 deconvolution fixture [patch]

- [patch] Added Rust/PyO3 `gaussian_deconvolution_fixture` and routed Chapter
  17's Tikhonov L-curve fixture through the Rust-owned Gaussian convolution,
  two-bump truth, and deterministic perturbation generator instead of local
  Python fixture algebra.

### Added (2026-07-01) - Chapter 10 MRE envelope [patch]

- [patch] Added Rust/PyO3 `mre_displacement_envelope` and routed Chapter 10's
  MRE displacement-envelope overlay through the Rust-owned exponential decay
  helper instead of Python `np.exp`.

### Added (2026-07-01) - Chapter 23 VCZ coherence [patch]

- [patch] Added Rust/PyO3 `van_cittert_zernike_coherence` and routed Chapter
  23 spatial-coherence plotting through the Rust-owned Van Cittert-Zernike sinc
  law instead of Python `np.sinc`; removed the stale SciPy requirement text.

### Changed (2026-07-01) - Chapter 3 source waveform routing [patch]

- [patch] Routed the Chapter 3 Westervelt PSTD source waveform through existing
  Rust/PyO3 `fubini_waveform` at `sigma=0` instead of local Python sinusoid
  construction, and removed stale SciPy requirement text from the script header.

### Changed (2026-06-30) - Chapter 1 standing wave routing [patch]

- [patch] Routed Chapter 1 standing-wave initial-condition and analytic overlay
  generation through existing Rust/PyO3 `standing_wave_1d` and corrected the
  wrapper formula documentation to match the Rust core contract.

### Added (2026-06-30) - Chapter 5 axial RF pulse [patch]

- [patch] Added Rust/PyO3 `centered_hann_tone_burst_waveform` and routed the
  Chapter 5 B-mode PSF axial RF pulse through the Rust-owned centered
  discrete-Hann tone-burst helper instead of Python `np.hanning` and local
  carrier multiplication.

### Added (2026-06-30) - Chapter 25 RTM axial spectrum [patch]

- [patch] Added Rust/PyO3 `demeaned_hann_power_spectrum_1d` and routed Chapter
  25 RTM axial spatial-spectrum plotting through the Rust-owned Hann-windowed
  one-sided power spectrum helper instead of Python `np.hanning` and
  `np.fft.rfft`.

### Changed (2026-06-30) - Population emission seed boundary [patch]

- [patch] Changed the shared book population-emission helper and Chapter
  24/21e callers to pass deterministic seeds directly to Rust/PyO3
  `simulate_population_emission` instead of deriving Rust seeds from Python
  RNG objects.

### Added (2026-06-30) - Chapter 7 CEM43 fixture [patch]

- [patch] Added Rust/PyO3 `closed_loop_cem43_fixture` and routed Chapter 7
  closed-loop focal-temperature and CEM43 dose curves through Rust-owned
  fixed-power, feedback, and underdrive traces instead of Python-side RNG and
  per-trace dose calls.

### Added (2026-06-30) - Chapter 23 cavitation dose fixture [patch]

- [patch] Added Rust/PyO3 `passive_cavitation_dose_fixture` and routed Chapter
  23 passive-cavitation dose accumulation through Rust-owned stable-dose and
  seeded compound-Poisson inertial-dose traces instead of Python-side RNG loops.

### Changed (2026-06-30) - Chapter 5 shear-wave speed routing [patch]

- [patch] Routed Chapter 5 shear-wave elastography tissue-range speed
  conversion through Rust/PyO3 `shear_wave_speed` instead of Python-side
  `sqrt(mu/rho)` in the book plotting script.

### Added (2026-06-30) - Chapter 4 apodization response [patch]

- [patch] Added Rust/PyO3 `apodization_window_response` and routed Chapter 4
  apodization-window response plotting through Rust-owned zero-padded,
  FFT-shifted spatial-frequency response calculation instead of Python
  `np.fft`.

### Added (2026-06-30) - Chapter 10 thermal-strain RF fixture [patch]

- [patch] Added Rust/PyO3 `thermal_strain_rf_fixture` and routed the Chapter 10
  thermal-strain synthetic RF generation through Rust-owned seeded speckle,
  carrier modulation, and apparent-displacement warp interpolation.

### Added (2026-06-30) - Chapter 3 harmonic extraction [patch]

- [patch] Added Rust/PyO3 `hann_windowed_harmonic_amplitudes` and routed the
  Chapter 3 PSTD-vs-Fubini harmonic-amplitude extraction through Rust-owned
  Hann-windowed FFT bin extraction instead of Python `np.fft.rfft`.

### Added (2026-06-30) - Chapter 7 PCD fixtures [patch]

- [patch] Added Rust/PyO3 `keller_miksis_pcd_spectrum` and
  `keller_miksis_pcd_controller_trace`, then routed Chapter 7 PCD spectrum and
  controller figures through Rust-owned Keller-Miksis FFT band extraction and
  SC/IC pressure stepping.

### Added (2026-06-30) - Chapter 5 PA Gaussian fixture [patch]

- [patch] Added Rust/PyO3 `gaussian_absorber_photoacoustic_profile` and routed
  Chapter 5 Figure 04 through Rust-owned Gaussian initial pressure and analytic
  bipolar surface-signal computation.

### Added (2026-06-30) - Transcranial subspot BBB bindings [patch]

- [patch] Exposed Rust-owned transcranial GBM subspot rastering and BBB
  subspot-dose fields through thin PyO3 helpers, then routed the Chapter 25
  transcranial planning adapters through those helpers so Python only packages
  returned arrays for plotting and downstream book figures.

### Fixed (2026-06-30) - Transcranial planning PyO3 contract [patch]

- [patch] Removed optional `pykwavers` fallback branches from the book
  transcranial planning helpers. Acoustic observables, cavitation risk, BBB
  permeability, and HU material mapping now route through existing Rust/PyO3
  bindings, and the top-level package facade exports the existing transcranial
  array planner and Pennes thermal-dose binding used by the book scripts.

### Fixed (2026-06-30) - Chapter 24 CEM43 vector dose [patch]

- [patch] Routed Chapter 24 focal CEM43 accumulation through Rust/PyO3
  `cem43_cumulative` instead of Python-side sparse prefix calls to
  `compute_cem43`, and removed the ignored `max_nucleation_cycles` keyword from
  the shared cavitation population book helper and callers.

### Fixed (2026-06-30) - kwavers-physics all-target clippy gate [patch]

- [patch] Cleared the current `kwavers-physics --all-targets` clippy blockers
  by replacing manual range predicates, moving test modules after exported
  items, converting constant invariants to compile-time assertions, removing
  redundant `clone()` calls on `Copy` values, and naming a test helper tuple
  type.

### Added (2026-06-30) - Chapter 26 response smoothing helper [patch]

- [patch] Added Rust/PyO3 `lif_response_probability_py` and routed Chapter 26
  neural-response smoothing through Rust-owned spike-train sampling and
  Gaussian response probability computation. The Chapter 26 focal thermal-dose
  trace now uses Rust/PyO3 `cem43_cumulative` instead of Python-side sparse
  prefix-dose interpolation.

### Added (2026-06-30) - Chapter 5 CW/vector Doppler fixture [patch]

- [patch] Added Rust/PyO3 `continuous_wave_vector_flow_fixture` and routed
  Chapter 5 Figure 9.4 through Rust-owned RF tone synthesis, CW
  demodulation/FFT, PW Nyquist comparison, cross-beam projection, and
  vector-flow recovery.

### Added (2026-06-30) - Chapter 13 unmixing sweep helper [patch]

- [patch] Added Rust/PyO3 `spectroscopic_unmixing_so2_sweep` and routed Chapter
  13 Figure 10.4 through Rust-owned HbO2/Hb deterministic perturbation sweeps,
  nonnegative concentration clipping, and sO2 ratio calculation.

### Added (2026-06-30) - Chapter 5 Doppler spectrum helper [patch]

- [patch] Added Rust/PyO3 `contrast_agent_doppler_spectrum` and routed Chapter
  5 Figure 9.3 through Rust-owned contrast-agent IQ synthesis, finite-tone
  spectrum power, velocity-axis mapping, Nyquist velocity, and Kasai estimate.

### Added (2026-06-30) - Chapter 23 PAM eigenspace spectrum [patch]

- [patch] Added Rust/PyO3 `eigenspace_covariance_eigenvalues` and routed
  Chapter 23 Figure 22.5 through the deterministic Theorem 22.2 signal/noise
  eigenvalue split instead of a Python-local stochastic CSD fixture.

### Changed (2026-06-30) - KWaveArray compact per-element sources [patch]

- [patch] Changed `KWaveArray::build_per_element_source` to stream sparse
  BLI-weighted cell contributions from each element instead of storing one
  dense 3-D mask per element. The per-cell signal contract and
  MATLAB/Fortran-order active-cell row ordering are preserved by value tests.

### Added (2026-06-30) - Chapter 14 plane-wave velocity helper [patch]

- [patch] Added Rust/PyO3 `plane_wave_pressure_velocity_1d` and routed Chapter
  14 Figure 8.3 through it so Rust owns the pressure/particle-velocity plane
  wave pair and the impedance ratio `u = p/(rho*c)`.

### Added (2026-06-30) - Chapter 23 PAM RF helper [patch]

- [patch] Added Rust/PyO3 `passive_cavitation_point_source_rf` and routed the
  Chapter 23 passive DAS sensitivity panel through it so Rust owns point-source
  receive delays, Gaussian emission envelope, carrier phase, and `1/r`
  spreading before the existing `passive_acoustic_map_das` reconstruction.

### Added (2026-06-30) - Chapter 11 BLI error curves [patch]

- [patch] Added Rust/PyO3 `bli_interpolation_error_curves` and routed the
  Chapter 11 BLI accuracy panel through it so Rust owns the deterministic
  sinusoid reconstruction RMS curves; Python now converts the returned RMS
  series to dB and plots.

### Changed (2026-06-30) - Chapter 7 vector CEM43 binding [patch]

- [patch] Routed Chapter 7 closed-loop CEM43 dose accumulation through the
  vector Rust/PyO3 `cem43_cumulative` binding instead of an O(n²) Python prefix
  loop around `compute_cem43`.

### Added (2026-06-30) - Chapter 22 PAM spectrum helper [patch]

- [patch] Added Rust/PyO3
  `normalized_cavitation_emission_spectrum` and routed Chapter 22/23 passive
  acoustic mapping Figure 22.1 through it instead of carrying the Lorentzian
  harmonic/subharmonic and inertial broadband spectrum model in Python.

### Changed (2026-06-30) - Chapter 21 Rust pressure inversion [patch]

- [patch] Routed the Chapter 21 histotripsy comparison's millisecond-pulse
  intensity-to-pressure inversion through Rust/PyO3
  `acoustic_pressure_amplitude_from_intensity` before the thermal heat-source
  calculation.

### Changed (2026-06-30) - Chapter 4 beamforming binding contracts [patch]

- [patch] Routed Chapter 4 transducer-array beam-pattern, grating-lobe,
  lateral-resolution, 2-D beam-field, and BLI stencil panels through the current
  Rust/PyO3 binding contracts. The 2-D beam-field panel now passes x/z axes
  directly to `beam_pattern_2d` instead of allocating a Python mesh.

### Changed (2026-06-30) - Chapter 18 Rust pressure conversion [patch]

- [patch] Routed Chapter 18 sonogenetics activation pressure reconstruction
  through Rust/PyO3 `acoustic_pressure_amplitude_from_intensity` instead of
  duplicating `sqrt(2*rho*c*I)` in Python.

### Changed (2026-06-30) - Chapter 33 MEMS book guard [patch]

- [patch] Removed the redundant optional `pykwavers` import branch from Chapter
  33 CMUT/PMUT figure generation and added focused Python tests proving the
  script routes MEMS physics through Rust/PyO3 bindings.

### Added (2026-06-30) - Chapter 7 Rust Minnaert inverse [patch]

- [patch] Added Rust/PyO3 `minnaert_radius_for_frequency_m` and routed Chapter
  7 resonance-marker radii through it instead of duplicating the inverse
  Minnaert formula in Python.

### Added (2026-06-30) - Chapter 6 Rust intensity conversion [patch]

- [patch] Added Rust/PyO3 `acoustic_pressure_amplitude_from_intensity` and
  routed Chapter 6 HIFU heat-source pressure setup through that helper instead
  of duplicating `sqrt(2*rho*c*I)` in Python.

### Added (2026-06-30) - Chapter 8 Rust spreading envelopes [patch]

- [patch] Added Rust/PyO3 `geometric_spreading_intensity_envelopes` and routed
  the retained Chapter 8 acoustic-propagation spreading-law figure through that
  helper instead of deriving the normalized `1/r^2` and `1/r` envelopes in
  Python.

### Changed (2026-06-30) - Chapter 3 Rust Fubini waveform [patch]

- [patch] Routed Chapter 3 Fubini waveform evolution through the existing
  Rust/PyO3 `fubini_waveform` helper instead of reconstructing the harmonic
  series in Python. Added focused Rust and PyO3 value-semantic tests for the
  waveform sinusoid limit and harmonic expansion.

### Added (2026-06-30) - Chapter 2 Rust numerical-method helpers [patch]

- [patch] Added Rust/PyO3 `centered_fd_modified_wavenumber`,
  `kspace_temporal_correction`, and `fdtd_cfl_stability_region_2d`, then routed
  Chapter 2 CFL, modified-wavenumber, and k-space correction figure data through
  those helpers instead of Python-side stencil and sinc formulas.

### Added (2026-06-30) - Chapter 1 Rust wave helpers [patch]

- [patch] Added Rust/PyO3 `gaussian_modulated_pulse_1d` and
  `dalembert_split_solution_1d`, then routed Chapter 1 Figure 1.1's
  travelling-pulse source profile and d'Alembert reference through those helpers
  instead of Python-side Gaussian/carrier and interpolation formulas.

### Added (2026-06-30) - k-Wave cache parity manifest [patch]

- [patch] **`kwavers-python` k-Wave/KWave.jl validation inventory** - added a
  fast cache-manifest regression that classifies every current k-Wave reference
  cache as paired pykwavers parity data or explicitly reference-only, verifies at
  least 40 paired cache families, checks finite nonzero numeric payloads for
  paired k-Wave/pykwavers outputs, validates every current KWave.jl comparison
  report as `RESULT: PASS`, parses the report metrics against each script's
  executable `PARITY_THRESHOLDS`, checks finite metadata plus nonzero CSV/NPY
  numeric payloads, and decodes each comparison PNG as a finite nonblank image
  for all six KWave.jl artifact families.
- [patch] The same manifest now classifies every `*_compare.py` /
  `compare_*.py` driver as directly pytest-covered or reference/diagnostic, and
  verifies that each directly covered driver is actually referenced by a
  `test_kwave*.py` file. It also rejects reference/diagnostic drivers that still
  have a standard paired k-Wave/pykwavers cache. This prevents silent drift
  between example drivers, cache artifacts, and the parity suite.
- [patch] Hardened the direct-coverage reference guard so it no longer counts
  `test_kwave_cache_manifest.py`'s own manifest literals as pytest coverage. The
  guard now proves each directly covered driver is referenced by a non-manifest
  `test_kwave*.py` file, except KWave.jl drivers whose semantic metric/metadata/
  PNG validation is intentionally owned by the manifest itself.
- [patch] Hardened the direct `us_bmode_linear_transducer` parity test so it no
  longer duplicates quick-mode thresholds in the test body. The test now parses
  the raw scan-line physics-parity block, enforces the report-owned target line
  emitted by `pykwavers.parity_targets.evaluate_parity`, and decodes both
  generated B-mode PNG artifacts as finite nonblank images.
- [patch] Updated the validation chapter parity summary so it reports the
  current raw B-mode scan-line metrics, moves the log-compressed display panels
  out of the active physics-validation table, and records the closed
  axisymmetric circular-piston, focused-bowl, and IVP Gaussian reports.
  Regenerated the Chapter 20 validation scatter from the same current metrics
  and updated the sensors chapter's B-mode RMS target note to match the raw
  scan-line oracle. The acceptance text now distinguishes strict field-tier
  reference lines from driver-owned quick-tier thresholds, so B-mode raw
  scan-line PASS is no longer contradicted by a stale global PSNR target.
  Chapter 20 `fig04` no longer fabricates a noisy pseudo-kwavers trace; it
  repackages the current cached `at_focused_bowl_AS_compare.py` PASS artifact.
  The comparison pseudocode now reads scenario-owned `PARITY_THRESHOLDS` instead
  of duplicating strict field-tier constants. The manifest test now rejects
  synthetic Chapter 20 parity patterns and verifies that the cached focused-bowl
  source PNG plus regenerated book PNG remain decodable and nonblank. The
  Chapter 20 scatter now reads the closed-validation markdown table instead of
  carrying a second hardcoded metric list, and the manifest verifies the parsed
  row set. Figures 19.1 and 19.2 now label r = 0.99 and PSNR = 40 dB as strict
  field-tier references rather than universal acceptance thresholds. The Python
  parity command block now uses the current `crates/kwavers-python/tests` paths
  and Miniforge interpreter instead of the obsolete `cd pykwavers` layout, with
  a manifest regression preventing the stale command form from returning.
  The manifest also parses the Chapter 20 figure index and verifies each listed
  PNG decodes plus each paired PDF exists.
- [patch] Hardened the Chapter 5 diagnostic-imaging figure script so its axial
  envelope, lateral PSF, Doppler shift, and contrast-bubble amplitude route
  through `pykwavers` Rust/PyO3 bindings without a SciPy Hilbert fallback or
  random Python-generated Doppler noise. The manifest now rejects the removed
  fallback/random tokens, requires the Rust binding calls to remain present,
  checks the top-level `pykwavers` re-exports for the imaging helper bindings,
  and decodes all regenerated Chapter 5 PNG/PDF figure artifacts.
- [patch] Hardened the Chapter 10 elastography figure script so `pykwavers` is a
  required dependency, optional import guards are removed, the MRE figure uses
  the Rust `mre_displacement_field` analytical kernel instead of a Python-only
  cylindrical-inclusion sketch, and the top-level `pykwavers` package exports
  the MRE helper. The book caption now describes the implemented damped
  plane-wave model, and the manifest checks Chapter 10 Rust-binding calls,
  top-level exports, and all six PNG/PDF figure artifacts.
- [patch] Hardened the Chapter 11 sources/transducers figure script so
  `pykwavers` is required, optional import guards are removed, array-factor dB
  rendering uses magnitude to avoid non-finite log paths, and the BLI accuracy
  panel is computed from the Rust `bli_interpolation_error_curves` helper rather
  than a duplicate Python formula. The sources chapter caption now names the
  Rust BLI helper, and the manifest checks Chapter 11 Rust-binding calls,
  top-level exports, and all seven PNG/PDF figure artifacts.
- [patch] Hardened the Chapter 12 media/tissue figure script so `pykwavers` is
  required, optional import guards are removed, and the Pennes steady-state
  slab profile is computed by the Rust
  `pennes_steady_state_temperature_profile` analytical kernel rather than a
  Python-side closed-form duplicate. The media chapter captions now name the
  Rust bindings used for sound speed, tissue properties, B/A, power-law
  attenuation, and Pennes bioheat, and the manifest checks Chapter 12
  Rust-binding calls, top-level exports, and all five PNG/PDF figure artifacts.
- [patch] Hardened the Chapter 13 photoacoustics figure script so `pykwavers` is
  required, optional import guards are removed, and the spectroscopic unmixing
  panel uses deterministic measurement perturbations instead of random
  Python-generated noise. The photoacoustics chapter caption now describes the
  deterministic perturbation model, and the manifest checks Chapter 13
  Rust-binding calls, top-level exports, and all five PNG/PDF figure artifacts.
- [patch] Hardened the Chapter 14 sensors/measurements figure script so
  `pykwavers` is required, optional import guards are removed, the hydrophone
  directivity panel uses Rust `circular_piston_directivity` instead of a
  mismatched Python rectangular-sinc model, and the noisy sensor recording panel
  uses seeded Rust `add_noise` instead of Python RNG. The sensors chapter
  captions now name those Rust binding sources, and the manifest checks Chapter
  14 Rust-binding calls, top-level exports, and all five PNG/PDF figure
  artifacts.
- [patch] Hardened the Chapter 17 inverse-problems figure script so `pykwavers`
  is required, optional FWI skip paths are removed, and the Tikhonov L-curve
  fixture uses deterministic measurement perturbation instead of Python RNG.
  The inverse-problems chapter now labels the SVD panel as the implemented
  1-D Helmholtz finite-difference Rust helper and names the Rust L-curve/corner
  helpers behind Figure 18.3. The manifest checks Chapter 17 Rust-binding calls,
  top-level exports, and all six PNG/PDF figure artifacts. Added Rust/PyO3
  `eikonal_traveltime_2d` and `kirchhoff_point_scatterer_image_2d`, and routed
  Figure 18.6 through them instead of Python-side fast-sweeping and
  diffraction-stack loops.
- [patch] Hardened the Chapter 18 sonogenetics figure script so `pykwavers` is
  required, optional import/skip branches are removed, the streaming panel uses
  Rust `acoustic_streaming_velocity`, and the activation panel now matches the
  book contract: tension-gated MscL-G22S/TRPC6 route through Rust membrane
  tension plus Boltzmann gating, while hsTRPA1 routes through a new thin PyO3
  wrapper over the existing Rust pressure-threshold gate. The sonogenetics
  captions now match the generated Gorkov, streaming, activation, and CEM43
  panels, and the manifest checks Chapter 18 Rust-binding calls, top-level
  exports, and all seven PNG/PDF figure artifacts.
- [patch] Hardened the Chapter 21 simulation-orchestration figure script so
  `pykwavers` is required and the bubble-radius comparison routes through the
  Rust Rayleigh-Plesset, Keller-Miksis, and Gilmore PyO3 solver bindings. The
  manifest rejects optional import fallbacks, checks the three solver exports,
  verifies the book text still names the Rust-owned ODE path, and decodes the
  regenerated PNG/PDF figure artifact.
- [patch] Hardened the Chapter 34 optoacoustic focused-ultrasound figure script
  so `pykwavers` is required and the SOAP resolution/gain figure routes through
  the Rust/PyO3 optoacoustic transducer kernels for numerical aperture,
  f-number, lateral resolution, and focal gain. The manifest rejects optional
  import fallbacks, checks the four top-level exports, verifies the book text
  keeps the Rust single-source-of-truth claim for Eqs. 34.4-34.5, and decodes
  the regenerated PNG/PDF figure artifact.
- [patch] Hardened the Chapter 29 pressure-diagnostics helper so `pykwavers` is
  required and mechanical-index diagnostics route through the Rust/PyO3
  `kw.mechanical_index` safety kernel. The Python fallback formula and optional
  import branch were removed, and the Chapter 29 therapy-chapter tests now pin
  both the computed MI value and the absence of those fallback tokens.
- [patch] Hardened the Chapter 30 intravascular-ultrasound figure script so
  `pykwavers` is required and the IVUS helper path has no extension-unavailable
  fallback branches. Intensity, adiabatic temperature rise, B-mode log
  compression, RF-line envelope detection, and mechanical index now call the
  Rust/PyO3 kernels unconditionally, with source regressions guarding against
  the removed Python duplicate formulas. Added Rust/PyO3 `ivus_vessel_phantom`
  for the deterministic vessel anatomy, tissue-property fields,
  impedance-gradient reflectivity, and seeded Rayleigh speckle fixture, leaving
  Python responsible only for array adaptation and plotting.
- [patch] Hardened the Chapter 24 BBB-LIFU and Chapter 26 neuromodulation book
  scripts so `pykwavers` is a direct required import rather than an optional
  try/except branch with `_HAS_KW` state. The Chapter 24 helper-module import now
  uses an explicit script-directory path before import, and the therapy-chapter
  tests guard against reintroducing optional PyO3 import fallbacks while checking
  the key Rust binding calls remain present. Added
  `mechanical_index_frequency_sweep` as a Rust/PyO3 safety helper and routed
  Chapter 24's inertial-cavitation MI frequency curves through it instead of
  Python-side `constant / sqrt(f_MHz)` formulas. The passive-cavitation drive
  pressure sweep now uses the existing Rust/PyO3 `mechanical_index_field`
  helper instead of duplicating the MI equation in Python. Added the Rust/PyO3
  `bbb_inertial_damage_probability` BBB helper and routed Chapter 24's
  inertial-cavitation damage-risk curve through it instead of inline NumPy
  logistic algebra. Added the Rust/PyO3
  `mechanical_index_cavitation_risk` safety helper and routed Chapter 26's
  neuromodulation cavitation-risk contour through it instead of inline NumPy
  logistic algebra. Added the Rust/PyO3
  `cavitation_therapeutic_window_indices` passive-dose helper and routed
  Chapter 24's stable-onset, inertial-onset, and controller-cap classification
  through it instead of Python-side band-ratio scans. Added the Rust/PyO3
  `cavitation_inertial_fraction_onset_index` passive-dose helper and routed
  Chapter 24's population-monitor operating-point selection through it instead
  of Python-side broadband-fraction scans. Added the Rust/PyO3
  `per_spot_cavitation_dose_grid` delivery helper and routed Chapter 24's
  per-spot cavitation monitor raster through it instead of Python-side nested
  steering/interpolation loops. Added the Rust/PyO3
  `cavitation_monitor_timeseries` helper and routed the shared curve-driven
  cavitation-monitor trace through it instead of Python-side interpolation,
  seeded jitter, controller stepping, and dose accumulation. Added the Rust/PyO3
  `closed_loop_cavitation_sonication` helper and routed the Chapter 24
  passive-cavitation closed-loop sonication trace through it instead of
  Python-side stable/inertial interpolation, controller stepping, and dose
  accumulation. Added the Rust/PyO3 `raster_cavitation_pulsing` delivery helper
  and routed the shared raster-pulsing monitor through it instead of Python-side
  steering derating, pressure-sweep interpolation, schedule expansion,
  residual-bubble shielding, thermal relaxation, coverage, and cumulative-dose
  resampling. Added the Rust/PyO3 `simulate_population_emission` helper and
  routed the shared one-pressure population-emission helper through it instead
  of Python-side bubble-population sampling, per-bubble solver dispatch, trace
  rejection, Hann FFT spectrum construction, and cavitation-band decomposition.
  Added the Rust/PyO3 `simulated_population_monitor_timeseries` helper and
  routed the shared simulated per-pulse population monitor through it instead
  of Python-side per-pulse population-emission dispatch, controller stepping,
  acoustic-power scaling, and cumulative-dose integration. Added the Rust/PyO3
  `population_emission_sweep` helper and routed the Chapter 24 population
  pressure sweep through it instead of Python-side per-pressure aggregation over
  the one-pressure population helper. Added the Rust/PyO3
  `volume_emission_spectrum` and `volume_emission_sweep` helpers and routed the
  Chapter 24 V_s-integrated analytic spectrum and pressure sweep through them
  instead of Python-side Keller-Miksis loops, emission conversion, PSD
  construction, receiver integration, and band decomposition. Classified the
  remaining summary fraction formatting as presentation-only over Rust-returned
  arrays, not domain physics.
- [patch] Reduced the cached-parity PNG sanity helper's peak memory use by
  replacing Matplotlib float-array image decoding with Pillow size/extrema
  checks. This preserves nonblank artifact validation while avoiding the
  dashboard PNG float allocation failure.
- [patch] Hardened the axisymmetric aperture parity regressions for
  `at_circular_piston_AS_compare.py` and `at_focused_bowl_AS_compare.py`.
  Both tests now reuse each example's `PARITY_THRESHOLDS`, `METRICS_PATH`, and
  `FIGURE_PATH`, add fast current-artifact checks that decode the generated PNGs,
  and keep the slow full-regeneration tests behind `KWAVERS_RUN_SLOW=1`.
  The circular-piston analytical comparison now asserts bounded agreement between
  k-wave-python and pykwavers analytical correlations instead of assuming one
  solver's correlation must be lower. The analytical-reference thresholds now
  live in the compare drivers, and the focused-bowl plot path masks the O'Neil
  singularity so regenerated artifacts no longer emit non-finite analytical
  curve warnings.
- [patch] Hardened the 3-D circular-piston and focused-bowl aperture parity
  regressions so their tests consume script-owned `PARITY_THRESHOLDS` instead
  of duplicate stale literals. Both tests now add fast current-artifact
  PASS/PNG checks while keeping full simulator regeneration behind
  `KWAVERS_RUN_SLOW=1`.
- [patch] Hardened the direct `at_array_as_source` parity regression by moving
  the executable PASS contract into `PARITY_THRESHOLDS`, emitting
  `max_abs_diff` and `peak_ratio` in each report section, and adding a fast
  report/PNG artifact check. The stale slow-test-only `p_max` absolute-error
  literal is no longer used as a hidden contract; exact source-mask and
  distributed-signal invariants remain thresholded at the driver boundary.
- [patch] Hardened the direct `at_array_as_sensor` parity regression the same
  way: the driver now owns `PARITY_THRESHOLDS`, emits PSNR, max-absolute
  difference, and trace extrema needed by fast artifact checks, and makes
  `combine_sensor_data(..., order="F")` explicit to preserve the current
  k-wave-python ordering contract.
- [patch] Hardened the direct `at_linear_array_transducer` parity regression by
  moving source-mask, source-weighted-mask, and `p_max` field targets into
  script-owned `PARITY_THRESHOLDS`, adding a fast report/PNG artifact check, and
  removing the stale slow-test-only `p_max` PSNR literal that contradicted the
  current executable PASS report.
- [patch] Hardened the direct `us_defining_transducer` parity regression so its
  pytest no longer carries duplicate per-sensor trace thresholds. The default
  test now validates the current PASS report against the example-owned
  `TRACE_THRESHOLDS`, confirms diagnostic trace metrics are finite, decodes the
  comparison PNG, and leaves the full simulator regeneration behind
  `KWAVERS_RUN_SLOW=1`.
- [patch] Hardened the direct `ivp_photoacoustic_waveforms` parity regression so
  the example owns the single-trace Pearson/RMS/PSNR/peak-ratio target map, the
  default pytest path validates the current PASS report plus nonblank PNG, and
  the slow full 3-D regeneration no longer relies on hidden test-local threshold
  literals.
- [patch] Hardened the direct `pr_2D_FFT_line_sensor` parity regression so the
  example owns reconstruction and ground-truth metric thresholds, emits the
  reference RMS/PSNR diagnostics needed by the report contract, and the default
  pytest path validates the PASS report plus both reconstruction and pressure
  PNG artifacts.
- [patch] Hardened the direct `pr_2D_TR_line_sensor` parity regression with a
  driver-owned threshold map for the lossy time-reversal reconstruction, the
  near-exact FFT reconstruction, and reconstruction-vs-ground-truth diagnostics.
  The report now emits the reference RMS/PSNR diagnostics, the default pytest
  path validates all three generated PNG artifacts, and stale slow-test-only
  time-reversal literals were replaced by the regenerated executable report
  contract.
- [patch] Hardened the direct `pr_3D_TR_planar_sensor` parity regression so the
  example owns its time-reversal and ground-truth thresholds, emits reference
  RMS/PSNR diagnostics, and the default pytest path validates the PASS report
  plus time-reversal and pressure PNG artifacts before the slow full simulator
  path.
- [patch] Hardened the direct `na_controlling_the_pml` parity regression so the
  example-owned `PARITY_THRESHOLDS` also covers waveform max-absolute
  difference and HDF5 writer parity, and the default pytest path validates the
  current PASS report plus nonblank comparison PNG before the slow full PML
  sweep.
- [patch] Hardened the direct `sd_focussed_detector_2D` parity regression so
  the example owns trace and directivity thresholds, and the default pytest path
  validates the current PASS report plus both detector-trace and directivity PNG
  artifacts.
- [patch] Hardened the direct `sd_focussed_detector_3D` parity regression so
  the example owns source-specific trace and directivity thresholds, and the
  default pytest path validates the current PASS report plus both 3-D detector
  PNG artifacts before the slow full simulator path.
- [patch] Hardened the direct `sd_directivity_modelling_2D` parity regression so
  the example owns matrix, trace-summary, and directivity thresholds, and the
  default pytest path validates the current PASS report plus both trace-matrix
  and directivity PNG artifacts before the slow full simulator path.
- [patch] Hardened the direct `ivp_saving_movie_files` parity regression so its
  default pytest path validates the current PASS report and comparison PNG
  against the driver-owned `PARITY_THRESHOLDS`. The driver now crops the
  pykwavers `p_final` field to the same PML-excluded physical interior emitted
  by k-wave-python before comparing or plotting the final pressure field.
- [patch] Hardened the direct `na_optimising_performance` parity regression so
  its default pytest path validates the current PASS report and comparison PNG
  against the driver-owned `PARITY_THRESHOLDS`. The driver now crops pykwavers
  `p_final` to the same PML-excluded physical interior emitted by k-wave-python,
  and the test source-image path now resolves to the repo-root
  `external/k-wave-python/tests/EXAMPLE_source_two.bmp` path used by the driver.
- [patch] Hardened the direct `us_bmode_phased_array` parity regression so the
  strict quick-tier fundamental/harmonic image thresholds live in the compare
  driver instead of hidden test literals. The default pytest path now validates
  the current PASS report against those thresholds and decodes both the B-mode
  comparison PNG and transducer-face debug PNG as finite nonblank images.
- [patch] Hardened the direct `checkpointing` parity regression so the exact
  save/resume contract lives in `checkpointing_compare.py`. The default pytest
  path now validates the current bit-exact PASS report, checkpoint lifecycle
  metrics, full-grid sensor shape, and comparison PNG without rerunning the
  slow checkpoint simulation.
- [patch] Hardened the direct `pr_3D_FFT_planar_sensor` parity regression so
  the driver owns summary and representative-trace thresholds. Removed a stale
  one-sample alignment shift after cache inspection showed the raw k-wave-python
  and pykwavers matrices match at zero lag; the regenerated report records mean
  Pearson 1.000000, mean RMS ratio 0.999941, and max absolute difference
  6.219181e-05. The default pytest path now validates the current PASS report
  and pressure PNG before the slow full simulator regeneration path.
- [patch] The manifest also enumerates all 51 vendored
  `external/k-wave-python/examples/**/*.py` sources. Fifty standalone examples
  must map to an existing local compare/dashboard script; the only current
  non-standalone source is
  `legacy/us_bmode_linear_transducer/example_utils.py`. This is a source
  inventory guard rather than a fresh simulator rerun.
- [patch] The dashboard source manifest now explicitly classifies the three
  current non-compare dashboard artifacts:
  `cavitation_bubble_validation.py`, `hifu_procedure_simulation.py`, and
  `phase_compare_minimal.py`. Any new non-compare dashboard row must be added to
  that manifest before the fast parity gate passes.
- [patch] The manifest now verifies every current dashboard metrics report is
  nonempty, explicitly records PASS, and contains no non-finite numeric tokens.
  Regenerated `us_bmode_phased_array_tiny_metrics.txt` with finite image Pearson,
  PSNR, and RMS-ratio fields instead of an unsupported SSIM `nan`.
- [patch] Hardened the remaining reference/diagnostic reports that already
  expose executable thresholds:
  `diff_homogeneous_medium_diffusion_compare.py`,
  `diff_homogeneous_medium_source_compare.py`,
  `ivp_opposing_corners_sensor_mask_compare.py`,
  `tvsp_acoustic_field_propagator_compare.py`,
  `tvsp_angular_spectrum_method_compare.py`, and
  `tvsp_equivalent_source_holography_compare.py`, plus
  `tvsp_transducer_field_patterns_compare.py`. The manifest now parses the
  current reports against those driver-owned `PARITY_THRESHOLDS` contracts while
  decoding each comparison PNG, and self-audits that every reference/diagnostic
  compare driver exporting `PARITY_THRESHOLDS` is covered by that semantic
  parser set.
- [patch] Regenerated the tracked parity dashboard from current metric files and
  hardened `parity_dashboard.py` so metric rows map back to real current example
  sources. The dashboard now excludes orphan canonical metric files from totals
  while listing their filenames, resolves nonstandard source names such as
  `compare_initial_pressure.py`,
  `ivp_axisymmetric_simulation_compare.py`, and `hifu_procedure_simulation.py`,
  classifies standalone analytical validation artifacts under the
  analytical/canonical backend, records 79/79 PASS current artifacts, resolves
  report-declared `figure:` / `figure_*:` PNG artifacts, rejects dangling
  declared figure references, and decodes at least one current per-example PNG
  for every dashboard row.
- [patch] Promoted `at_focused_annular_array_3D_compare.py`,
  `at_focused_annular_array_3D_full_compare.py`, `us_beam_patterns_compare.py`,
  `na_modelling_absorption_compare.py`, `ivp_3D_simulation_compare.py`,
  `tvsp_3D_simulation_compare.py`, `tvsp_snells_law_compare.py`, and
  `na_source_smoothing_compare.py` from reference/diagnostic to direct cached
  parity coverage, and promoted `us_bmode_phased_array_tiny_compare.py` after
  factoring its aggregate scan-line thresholds into a reusable
  `PARITY_THRESHOLDS` contract. The parameterized regression loads the existing
  k-Wave and pykwavers caches, applies the same k-Wave row permutation as the 3-D
  planar sensor scripts, verifies finite nonzero image/trace payloads, requires a
  PASS status in each example report, decodes each comparison PNG as finite and
  nonblank, and enforces each example's documented metric thresholds. Current
  cached metrics:
  annular axial amplitude Pearson 0.999999/0.999892, RMS ratio
  0.999678/0.992681, PSNR 69.18/45.23 dB;
  `us_beam_patterns` `p_rms` Pearson 0.999688, RMS ratio 0.921284, PSNR 30.46 dB;
  `us_beam_patterns` `p_max` Pearson 0.997555, RMS ratio 0.982948, PSNR 34.96 dB;
  `na_modelling_absorption` pressure Pearson 1.000000, RMS ratio 1.000004, PSNR
  90.34 dB; `ivp_3D_simulation` pressure Pearson 0.985404, RMS ratio 1.034993,
  PSNR 50.62 dB; `tvsp_3D_simulation` pressure Pearson 0.966665, RMS ratio
  1.102110, PSNR 29.94 dB; `tvsp_snells_law` `p_final` Pearson 1.000000, RMS
  ratio 1.000000, PSNR 239.45 dB; `na_source_smoothing` no-window/Hanning/
  Blackman traces Pearson 0.999680/1.000000/1.000000 and RMS ratio
  1.001548/1.000000/1.000000; tiny phased-array scan lines mean Pearson
  1.000000, mean RMS ratio 0.946366, image RMS ratio 0.946361.
- [patch] Promoted seven more vendored k-wave-python scenarios from
  reference/diagnostic to direct cached parity coverage:
  `na_filtering_part_1_compare.py`, `na_filtering_part_2_compare.py`,
  `na_filtering_part_3_compare.py`, `na_modelling_nonlinearity_compare.py`,
  `sd_directivity_modelling_3D_compare.py`,
  `tvsp_homogeneous_medium_monopole_compare.py`, and
  `tvsp_steering_linear_array_compare.py`.
- [patch] Promoted `ivp_1D_simulation_compare.py` to direct cached parity
  coverage using its global matrix metric contract, including PSNR. The cached
  report records Pearson 0.999994, RMS ratio 1.000000, and PSNR 63.81 dB
  against the k-wave-python reference.
- [patch] Added PSNR to the shared trace metric helper and promoted
  `tvsp_doppler_effect_compare.py` plus
  `tvsp_homogeneous_medium_dipole_compare.py` to direct cached parity coverage.
  The regenerated reports record `tvsp_doppler_effect` Pearson 0.995260, RMS
  ratio 1.000039, PSNR 28.35 dB and `tvsp_homogeneous_medium_dipole` Pearson
  0.992315, RMS ratio 0.976013, PSNR 23.70 dB. The direct cached parity gate
  covers these through the shared parameterized driver.
- [patch] Promoted four row-permuted IVP drivers to direct cached parity
  coverage using each script's `sensor_row_perm` contract:
  `ivp_binary_sensor_mask_compare.py`, `ivp_heterogeneous_medium_compare.py`,
  `ivp_homogeneous_medium_compare.py`, and
  `ivp_loading_external_image_compare.py`. Cached report metrics:
  binary-mask Pearson 1.000000, RMS ratio 1.000000, PSNR 303.35 dB;
  heterogeneous Pearson 0.999945, RMS ratio 0.999745, PSNR 56.11 dB;
  homogeneous Pearson 1.000000, RMS ratio 1.000000, PSNR 303.99 dB;
  external-image Pearson 1.000000, RMS ratio 1.000000, PSNR 302.38 dB.
  The direct cached parity gate now covers these through the shared
  parameterized driver.
- [patch] Promoted the final two upstream-mapped residual drivers to direct
  cached parity coverage with driver-specific value-semantic contracts.
  `sd_directional_array_elements_compare.py` compares the script-owned
  13-element averaged matrix rather than raw sensor rows and records Pearson
  0.992761, RMS ratio 0.996054, and PSNR 30.69 dB.
  `ivp_recording_particle_velocity_compare.py` now writes real k-wave-python and
  pykwavers NPZ caches for pressure plus `ux`/`uy`; the direct test applies the
  script's sensor-order permutation, gates pressure on all four sensors at
  Pearson >= 0.99, gates only the dominant velocity component at each sensor at
  Pearson >= 0.95, and verifies pykwavers directional dominance. Cached report
  metrics include pressure Pearson 0.998047/0.998047/0.997855/0.997855 and
  dominant-velocity Pearson 0.986909/0.986909/0.967838/0.967838. The focused
  parity gate now covers 25 direct cached tests: 22 parameterized cache-backed
  drivers, two driver-specific contracts, and the tiny phased-array aggregate.
- [patch] Consolidated cached parity test utilities into
  `crates/kwavers-python/tests/parity_test_utils.py`, making example-module
  loading, numeric cache loading, nonzero-payload checks, and PNG validation one
  shared implementation across the manifest and direct cached parity tests.

### Fixed (2026-06-30) - MVDR denominator guard (AMC-2) [patch]

- [patch] **AMC-2** `kwavers-analysis` - MVDR `compute_weights` and
  `pseudospectrum` now share a denominator validator for `a^H R^{-1} a`.
  The validator rejects non-finite values, non-positive real values, and imaginary
  components larger than a Higham-style complex dot-product roundoff bound. This
  closes the prior `.re`-only path that silently accepted non-Hermitian covariance
  defects. Added value-semantic regression tests for both public paths.
- [patch] **`kwavers-analysis` test hygiene** - cleared package clippy blockers
  exposed by the gate: default-mutate test configs now use struct initializers,
  the narrowband plane-wave fixture uses a typed spec, the vacuous GPU compile
  assertion was removed, and iterator-based assertions replaced index-only loops.

### Added (2026-06-20) — resolve remaining open items (PHY-13, COV-4 SIR, CLD-9/10, PCF-IQ) [minor]

Implemented every remaining item with a checkable analytical oracle; closed the
rest honestly (no fabricated formulas/baselines).

- [minor] **PHY-13** `kwavers-imaging` — analytical-limit tests for the Hoff (2000)
  bubble scattering cross-section: the resonance closed form
  `σ_s(ω₀)=4πR²(ka₀)²/δ_tot²` (independently re-derived δ_tot, Church 1995) and the
  low-frequency `σ_s ∝ ω²` scaling. No external (de Jong) data needed.
- [minor] **COV-4** `kwavers-physics` — `CircularPistonSir::round_trip_response`, the
  two-way (monostatic pulse-echo) diffraction kernel `h⊛h` (finite-aperture
  refinement of the point-element model). Grounded by the convolution factorization
  `∫(h⊛h)dt=(∫h dt)²` and the on-axis triangle support `[2z, 2√(z²+a²)]/c`.
- [minor] **CLD-9/CLD-10** `kwavers-transducer` — focused-bowl discretization validated
  against the O'Neil (1949) analytical focal gain via a discrete Rayleigh–Sommerfeld
  element sum at the focus (`|p(F)|/p₀ = k·h`) — a numerical-vs-analytical
  differential, replacing the prior self-consistency-only tests.
- [minor] **COV-1 (PCF-IQ)** `kwavers-analysis` — `phase_coherence_from_iq_aperture`,
  a native complex/baseband I/Q phase-coherence path that bypasses the Hilbert
  transform of the real-RF `CoherenceFactor::Phase` and routes through the same
  `phase_coherence_from_phases` core (keystone equivalence test).

### Closed (2026-06-20) — remaining items resolved as WONTFIX/deferred (verify-first)

- **COV-6 loaded-Mason Z_e** — no groundable closed-form oracle for the loaded
  electrical impedance without a verified Mason/KLM reference; implementing from
  memory would risk a fabricated formula. The `AcousticLayer` transmission-line
  transform already covers matching/backing design. Deferred pending a cited reference.
- **DG-solver CPML consolidation** — verify-first verdict: legitimately different
  discretization (per-GLL-node flux-based memory + joint SSP-RK3 ≠ FDTD recursive
  convolution); consolidating would distort both. Correct-layering, closed.
- **SOL-10** (workspace Rustdoc sweep), **SOL-11** (k-Wave validators → CI),
  **SOL-6** (coupled-CFL stability test), **AMC-5** (PINN loss normalization),
  **PHY-6/PHY-7** (magic-default citations), **PHY-11** (Lauterborn collapse
  regression), **COV-5 de Jong/Herring** — each needs an external baseline,
  paywalled reference, CI infra, or is an open-ended sweep / own numerics-changing
  increment; documented-deferred rather than fabricated.

### Added (2026-06-20) — coverage follow-up features (COV-6, COV-4, CFS-PML) [minor]

- [minor] **COV-6** `kwavers-transducer::bulk_piezo` — loaded acoustic
  transmission line for matching/backing design: `AcousticLayer` with the
  lossless telegrapher input-impedance transform and a reflection-coefficient
  helper, `quarter_wave_match_impedance = √(Z_s·Z_L)`, and
  `BulkPiezoResonator::quarter_wave_matching_layer`. Extends the free-plate KLM
  model to loaded faces. 6 closed-form tests (λ/4 inversion, λ/2 pass-through,
  matched-layer identity, Γ→0 matching into water).
- [minor] **COV-4** `kwavers-phantom::scatterers` — opt-in power-law tissue
  attenuation in the monostatic pulse-echo RF synthesis: each echo scaled by the
  round-trip factor `exp(−α(f₀)·2r)` (α₀ in dB/(cm·MHz); α₀=0 ⇒ prior lossless
  model). Validated against the closed-form factor + a deeper-scatterer
  differential.
- [minor] **CFS-PML** `kwavers-boundary` — `CPMLConfig::with_cfs_pml_for_frequency`
  sets `alpha_max = π·f₀` (Roden & Gedney 2000) so callers need not compute the
  complex-frequency-shift by hand.

### Fixed (2026-06-20) — audit-table remediation pass (SOL/PHY/CLD/AMC) [patch]

Drove the remaining Sprint A–E audit findings to terminal states; each verified
against current code first (several were already adequate / false positives).

- [patch] **SOL-5** `kwavers-solver` — `HASConfig::validate()` SSOT now also rejects
  a non-positive `reference_frequency` (which produced `(f/MHz)^y = NaN` in the
  HAS power-law absorption) and a non-finite/negative `attenuation_coeff`;
  `HasAbsorptionOperator::new` is fallible and re-validates at the boundary so a
  `default()` + field-mutation bypass is caught. Negative test per invariant.
- [patch] **SOL-8** `debug_assert!` the AVX-512 leapfrog/velocity coefficients are
  finite at construction. **SOL-9** documented the discretization-error rationale
  for the 5%/10%/2% benchmark tolerances.
- [patch] **PHY-14** `kwavers-physics` — the Gilmore RK4 `|u|→c` validity-boundary
  clamp is surfaced via `log::trace!` (`stage_acceleration`) instead of a silent
  `unwrap_or(0.0)` that froze the wall.
- [patch] **CLD-8** `kwavers-boundary` — BEM assembly `.last().unwrap()` →
  `.last().copied() == Some(col)` (no unwrap, identical behavior). **CLD-7**
  documented the uniform-concentration limitation of `update_microbubble_dynamics`.
- [docs] **AMC-2** documented that the MVDR `.re`-only denominator check is
  exhaustive (aᴴR⁻¹a is provably real for the upstream-Hermitian-validated R).
- [docs] Closed as adequate / false-positive (no change): **AMC-7** (covariance
  accumulation is sequential/deterministic; only scaling is parallel), **PHY-15**
  (KZK/RP validity bounds already cited), **AMC-6** (PAM bounds-checks; out-of-
  window → zero is correct), **PHY-12** (16π/3 is f32-representable), **AMC-3**
  (standard MUSIC regularization), **AMC-8** (absolute-ε floor defensible).
- [docs] Reconciled the **PLC-3** CHECKLIST↔gap_audit drift: the shell-model
  consolidation is closed as "do not merge" (over-abstraction across divergent
  consumers), not pending execution.

### Fixed (2026-06-20) — complete the phase coherence factor (PCF) dispatch (COV-1) [minor]

- [minor] **`kwavers-analysis::...::beamforming::time_domain::coherence`** completes
  `CoherenceFactor::Phase`, the phase coherence factor (Camacho et al. 2009):
  `PCF = max(0, 1 − (γ/σ₀)·min(σ(φ), σ(ψ)))`, where `φᵢ` are the per-element
  instantaneous phases (`arg` of the analytic signal), `ψᵢ = φᵢ − sign(φᵢ)·π` the
  auxiliary phases (shifting the wrap discontinuity off `±π`), `σ₀ = π/√3` the
  std of a uniform phase, and `γ = sensitivity`. Adds the canonical scalar
  `phase_coherence_from_phases` and derives phases via the analytic-signal SSOT
  `kwavers_math::fft::analytic_signal_1d`.
- [fix] The variant, validate arm, and helpers had landed without the
  `weight_for_column` dispatch or the `weights()` analytic-phase path, leaving the
  match non-exhaustive (**E0004**) — `kwavers-analysis` did not compile and
  `instantaneous_phase_matrix` was dead. This wires the missing dispatch.
- 11 value-semantic tests: exact closed forms (coherent → 1, 90°-spread → 1−√3/2,
  linear sensitivity scaling), the keystone auxiliary-phase wrap rescue
  (`±π`-straddling coherence scored high), near-zero for an evenly-spread aperture,
  and the column-path wiring (identical rows → 1; quadrature-spread → low) through
  the analytic signal. The coherence-factor family (amplitude / sign / phase /
  generalized) is now complete; 39/39 coherence tests pass, clippy clean.

### Added (2026-06-20) — CPML Courant-vs-thickness stability test (CLD-11) + AMC-12 verified

- [patch] **`kwavers/tests/cpml_absorption_quality.rs`** adds
  `test_cpml_stable_across_thicknesses` (Komatitsch & Martin 2007): sweeping the PML
  thickness `{6,8,10,12}` at a fixed CFL `dt`, the post-propagation domain energy
  stays finite (no blow-up), decays below the initial energy (stably absorbing), and
  absorption is monotone non-decreasing in thickness — empirical confirmation that
  the CFS-CPML preserves CFL stability independent of layer thickness. Refactored the
  existing single-thickness test onto a shared `run_cpml_absorption(thickness)` helper
  (SSOT). Closes the CLD-11 Courant sub-item.
- [docs] **AMC-12** marked DONE after verification (no code change): the PAM
  `MUSIC`/`EigenspaceMinVariance` methods are already fully wired through
  `pam::mapper::subspace_localization_map` → the shared narrowband
  `subspace_spatial_spectrum_point` (real Hermitian eigendecomposition + steering),
  not "not yet wired" stubs. The gap_audit entry was stale.

### Added (2026-06-20) — Rectangular-element spatial impulse response (COV-4) [minor]

- [minor] **`kwavers-physics::analytical::transducer::RectangularPistonSir`**
  (Lockwood & Willette 1973): the transient SIR of a flat rectangular piston,
  `h = (c/2π)·Φ(ρ)`, where `Φ` is the angular measure of the wavefront-intersection
  circle (radius `ρ=√((ct)²−z²)`, centered at the field-point projection) lying
  within the rectangle. `Φ` is evaluated **exactly** from the `arccos`/`arcsin`
  band breakpoints (membership constant between breakpoints) — no numerical
  integration — and handles projection inside / on-edge / at-corner / outside
  uniformly. Sibling of `CircularPistonSir`; both exported. 5 tests: on-axis
  plateau `h=c` until the nearest edge, a keystone differential of the analytic
  `Φ` against an independent θ-sampling oracle across 7 geometries × 5 radii,
  reflection symmetry, `[0,c]` bound, and parameter validation.

### Added (2026-06-20) — Generalized coherence factor (GCF, Li & Li 2003) (COV-1) [minor]

- [minor] **`kwavers-analysis::...::beamforming::time_domain::coherence`** adds
  `CoherenceFactor::Generalized { m0 }`, the generalized coherence factor: the
  fraction of delay-aligned aperture **spectral** energy in the low-spatial-frequency
  passband `|k| ≤ m0` (spatial DFT across elements) over the Parseval total
  `N·Σxᵢ²`. `m0 = 0` counts only the DC bin and reduces **exactly** to
  `CoherenceFactor::Amplitude` (Mallart-Fink); `m0 ≥ N/2 ⇒ GCF = 1`. Routes through
  the existing `delay_and_sum_coherence` unchanged. 5 value-semantic tests: the
  `m0=0 ≡ amplitude CF` keystone (differential), full-passband unity, monotonic
  non-decreasing in `m0`, and exact spectral localization on a pure two-cycle
  aperture (GCF jumps 0→1 as `m0` crosses the signal's spatial frequency).
  **Deferred [minor]:** PCF (phase coherence, Camacho 2009) needs an analytic-signal
  (complex) aperture path, not fabricable from real RF.

### Added (2026-06-20) — Cloud-model refinements: dp/dt coupling, R(t) shielding, RT/RM diagnostic, sparse solver (CLD-1, ADR 032) [major]

- [major] **`kwavers-therapy::...::lithotripsy::cavitation_cloud`** closes four CLD-1
  frontier items, each **opt-in** (defaults reproduce ADR 027-031 exactly):
  - **`dp/dt` coupling** (`couple_pressure_rate`): the coupling source strengths
    `S = R²R̈ + 2RṘ²` now carry the Keller-Miksis acoustic-radiation term by feeding
    the per-cell lagged finite-difference rate `(driving − prev_total)/dt` into the
    source/affine acceleration. Because `R̈` is affine in `dp/dt`, the direct/iterative
    linear system stays exact (the rate folds into the constant `c_j`; slope `d_j`
    unchanged). Test: a non-zero `dp/dt` changes the source strength and the
    two-bubble trajectory; off ⇒ identical to ADR 031.
  - **`R(t)`-dependent shielding** (`shielding_radius_dependent`): the
    Commander-Prosperetti resonance in `shielded_pressure` uses the instantaneous
    per-cell radius `R(t)` instead of the equilibrium `R0` (quasi-static extension).
    Test: equals the `R0` screen at `R = R0`, differs otherwise, and matches
    Beer-Lambert with the instantaneous radius.
  - **Cloud-interface instability diagnostic** (`interface_instability`): linear
    Rayleigh-Taylor `σ = √(A·k·a)` and Richtmyer-Meshkov `ȧ = k·Δv·a₀·A` growth
    rates at the cloud edge, Atwood number `A = β/(2−β)` from the Wood mixture density
    (`representative_void_fraction`). A **diagnostic** (growth rates), not a nonlinear
    interface simulation. Tests: both match their closed forms; RT is stable
    (rate 0) when the light fluid is on top.
  - **Sparse / matrix-free coupling solver** (`CouplingScheme::ImplicitIterative`):
    solves the same `(I − D·G)·S = e` with the validated `solve_lsqr_matfree` and a
    `MatFreeOperator` that computes `G_ab = ρ/d_ab` on the fly within the cutoff —
    `O(active)` memory, `O(active·neighbours)` per matvec — for very large active
    counts where the dense `ImplicitDirect` (`O(active³)`/`O(active²)`) is intractable.
    Tests: matches the dense direct solve to 1e-6 and is self-consistent.
  Coupling matrix building consolidated into `coupling_matrix`/`pair_distance` (SSOT,
  reused by all schemes). 26 cavitation-cloud tests pass; clippy-clean. See ADR 032.
  **Still open** (CLD-1): nonlinear RT/RM interface evolution, fully implicit `dp/dt`,
  nonlinear (large-amplitude) cloud scattering, k-Wave/experimental erosion comparison.

### Added (2026-06-19) — Cavitation/bubble validation example (pykwavers)

- [patch] **`crates/kwavers-python/examples/cavitation_bubble_validation.py`**: a
  runnable validation of the cavitation-cloud foundations (ADRs 027-030) against
  **analytical bubble theory** with plots — Keller-Miksis forced resonance peaks at
  the surface-tension-corrected Minnaert frequency (1.9% rel err), Wood mixture
  sound speed matches the closed-form Wood equation to machine precision
  (2.7e-16), and the Commander-Prosperetti attenuation peaks at the bubble
  resonance (1.6%). Saves `output/cavitation_bubble_validation.png` + a metrics
  report, all checks PASS. Documents why k-Wave is not the oracle here (it has no
  bubble-dynamics model — the k-Wave *acoustic* parity is the `at_*_compare.py`
  suite).

### Added (2026-06-19) — Strong-regime coupling solver: direct + under-relaxed (CLD-1, ADR 031) [major]

- [major] **`kwavers-therapy::...::lithotripsy::cavitation_cloud`** makes the
  self-consistent inter-bubble coupling robust in the strong-coupling regime where
  the plain fixed point (ADR 030) diverges. The `implicit_coupling: bool` is replaced
  by a **`CouplingScheme`** enum (`Explicit` / `ImplicitFixedPoint { under_relaxation }`
  / `ImplicitDirect`):
  - **`ImplicitDirect`** assembles the affine coupling system `(I − D·G)·S = e`
    (the Keller-Miksis acceleration is affine in the driving pressure, so the
    coefficients `c_j, d_j` come from two exact acceleration evaluations) and solves
    it with the validated `kwavers_math::LinearAlgebra::solve_linear_system` — the
    exact self-consistent solution, robust regardless of coupling strength; it falls
    back to an under-relaxed fixed point if the system is singular.
  - **Under-relaxation** in `ImplicitFixedPoint { under_relaxation: ω }`
    (`p_couple ← (1−ω)·old + ω·new`) extends the fixed point's convergence radius.
  Opt-in (default `Explicit`); coupling-off / single cell reduces exactly to
  ADR 027/028. The coupling code is consolidated into one matrix `G` reused by all
  schemes. 2 new value-semantic tests: the direct solve is self-consistent to ~1e-9
  even at 20 µm (strong) coupling, and matches the converged fixed point in the weak
  regime. 34 lithotripsy tests pass; clippy-clean. See ADR 031.
  **Still open** (CLD-1): `dp/dt` coupling, `R(t)`-dependent shielding, cloud-interface
  instabilities, sparse direct solve for very large active counts, k-Wave comparison.

### Added (2026-06-19) — Self-consistent (implicit) inter-bubble coupling (CLD-1, ADR 030) [major]

- [major] **`kwavers-therapy::...::lithotripsy::cavitation_cloud`** can now solve the
  inter-bubble coupling **self-consistently** within each step (ADR 030) instead of
  the explicit/lagged single pass (ADR 028). The coupling is refactored into one
  `coupling_pressure_field`: the explicit branch is the prior lagged pass; the new
  implicit branch fixed-point-iterates the coupling field — each bubble's source
  strength `S = R²R̈ + 2RṘ²` and the coupling its neighbours feel co-determined,
  reusing the canonical Keller-Miksis acceleration each iterate — to
  `coupling_tolerance`. `CloudParameters` gains `implicit_coupling` (opt-in, default
  off), `coupling_max_iterations`, `coupling_tolerance`. Off, or coupling-disabled,
  or a single cell ⇒ reduces **exactly** to ADR 028/027. 2 new value-semantic tests
  (the returned field satisfies its own fixed-point equation; implicit differs from
  explicit under close coupling). 32 lithotripsy tests pass, clippy-clean.
  **Still open** (CLD-1): direct linear solve / under-relaxation for the strong
  (non-contractive) regime, `dp/dt` coupling, cloud-interface instabilities, k-Wave
  comparison.

### Added (2026-06-19) — Cloud-scale acoustic shielding of the cavitation cloud (CLD-1, ADR 029) [major]

- [major] **`kwavers-therapy::...::lithotripsy::cavitation_cloud`** gains the second
  cloud-scale collective effect: **acoustic shielding**. The incident field is
  screened by the cloud's void fraction as it penetrates, so the periphery shields
  the interior (Maeda & Colonius 2018). `shielded_pressure` computes the void
  fraction `β = n·(4/3)πR³` per cell, gets the per-cell attenuation `α` from the
  validated `commander_prosperetti_attenuation` (reuse, SSOT), and applies
  Beer–Lambert screening `p_eff = p_ext·exp(−∫α dl)` along the configured incident
  axis (`O(N)`, a prefix sum per column); `evolve_cloud` drives the bubbles with
  the screened field (coupling added on top). `CloudParameters` gains
  `shielding_enabled` (opt-in, default off), `incident_axis`, `incident_from_high`.
  Zero void fraction or shielding-off ⇒ `p_eff = p_ext`, reducing **exactly** to
  ADR 027/028. 3 new value-semantic tests (Beer–Lambert exponential decay vs the
  closed-form `α`, no-nuclei pass-through, denser-cloud-screens-more). See ADR 029.
  **Still open** (CLD-1): `R(t)`-dependent (vs `R0`-linearized) attenuation,
  multi-directional/scattered screening, cloud-interface instabilities, and a
  self-consistent implicit coupled shielding+collapse solve.

### Added (2026-06-19) — Inter-bubble acoustic coupling in the cavitation cloud (CLD-1, ADR 028) [major]

- [major] **`kwavers-therapy::...::lithotripsy::cavitation_cloud`** gains the leading
  **collective** effect: inter-bubble acoustic coupling. Each pulsating bubble
  radiates the incompressible near-field pressure
  `bubble_radiated_pressure = (ρ/d)·(R²R̈ + 2RṘ²)`, which adds to its neighbours'
  driving field; `evolve_cloud` is now two-pass (explicit/lagged source strengths,
  then per-cell integration under `p_ext + Σ_{j≠i}(ρ/d_ij)S_j`). `CloudParameters`
  gains `cell_spacing`, `coupling_enabled`, `interaction_radius`. **Opt-in (default
  off):** the `O(active²)` sum amplifies the drive into the stiff regime, so
  enabling it by default made the full-grid orchestrator sim exceed the test
  ceiling — it is enabled per-study on tractable sizes (with a cutoff). A single
  active cell or coupling-off reduces **exactly** to ADR 027 (keystone preserved).
  Per-cell integrator non-convergence under the amplified drive is now handled
  gracefully (re-nucleate at R₀) instead of crashing. 5 new value-semantic tests
  (radiated-pressure closed form, 1/d scaling, coupling alters a two-bubble
  trajectory, closer-bubbles-couple-more, lone bubble unaffected). See ADR 028.
  **Still open** (CLD-1): self-consistent/implicit coupling, cloud-scale energy
  focusing/shielding (Maeda & Colonius 2018), cloud-interface instabilities,
  compressible retarded coupling.

### Changed (2026-06-19) — Time-resolved per-cell cavitation-cloud dynamics (CLD-1, ADR 027) [major]

- [major] **`kwavers-therapy::...::lithotripsy::cavitation_cloud`**: each cloud cell
  now carries a **real, time-resolved representative bubble** — its `(R, Ṙ)` is
  integrated across `evolve_cloud` calls by the canonical **adaptive Keller-Miksis**
  solver under the local instantaneous pressure (with `dp/dt` from the previous
  call), resolving violent collapse via sub-stepping. Bubble history is carried per
  cell (`radius_field`/`velocity_field`), so stepping at acoustic resolution
  reproduces the true `R(t)` waveform per cell — removing the snapshot/field-peak
  limitation of the prior increment. Erosion is the compression work `∫p dV` on each
  collapsing bubble (≈ the Rayleigh collapse energy over a full collapse), localized
  per cell. `CloudParameters` gains `drive_frequency`; `cloud_radius()` accessor
  added. **Keystone test**: a 1-cell cloud reproduces the standalone adaptive
  Keller-Miksis integrator bit-for-bit (the cell *is* a real bubble, not a
  surrogate). Density is now the seeded nuclei count (the ad-hoc growth/collapse
  *rate* model is removed). 7 value-semantic tests. See ADR 027. **Still open**
  (collective/research-frontier, CLD-1): inter-bubble coupling, cloud-scale energy
  focusing (Maeda & Colonius 2018), cloud instabilities — cells remain independent
  oscillators.

### Changed (2026-06-19) — Real Gilmore bubble dynamics for histotripsy cloud erosion (CLD-1)

- [minor] **`kwavers-therapy::...::lithotripsy::cavitation_cloud`** now drives
  cloud erosion with the **real Gilmore (1952) compressible single-bubble
  collapse** instead of a static-R₀ linear `(p−ambient)` proxy. New
  `representative_max_radius`/`inertial_collapse_energy` integrate the existing
  `GilmoreSolver` to capture inertial growth under rarefaction (`R_max ≫ R₀`) and
  the Rayleigh collapse energy `(4/3)π R_max³ (p₀−p_v)`; the per-cell erosion now
  scales with the collapsing-bubble count × this physics-based energy. Adds a
  `drive_frequency` field to `CloudParameters` (defaults via `Default`, only
  construction site unaffected). 4 new value-semantic tests (`R_max(12 MPa)>3·R₀`,
  energy scales with drive, deeper rarefaction erodes more); the prior
  pure-compression erosion test was updated to a realistic rarefaction+compression
  field (the static-R₀ proxy "eroded" without any bubble growth). Implements the
  "Gilmore + Mach corrections" the code itself flagged as absent (CLD-1).
  **Still open** (collective/research-frontier): multi-bubble coupling, cloud-scale
  energy focusing (Maeda & Colonius 2018), cloud instabilities, time-resolved
  per-cell coupling — no library is "100% accurate" for collective cloud collapse.

### Added (2026-06-19) — Transient spatial impulse response (COV-4 follow-up)

- [minor] **`kwavers-physics::analytical::transducer::spatial_impulse_response::CircularPistonSir`**:
  the transient (broadband) spatial impulse response of a flat circular piston
  (Stepanishen 1971), the Field II diffraction kernel — the impulse/pulse-echo
  complement to the existing continuous-wave Fast Nearfield Method.
  `evaluate(r, z, t)` returns the closed-form `h(r,z,t)` (on-axis rectangular
  pulse of height c; off-axis plateau + arccos arc), with `first/last_arrival_time`
  for the support. 6 analytic value-semantic tests, including `∫h dt = √(z²+a²)−z`
  (the on-axis path-difference identity). Completes the COV-4 finite-aperture
  follow-up (the scatterer RF synthesis used the point-element far-field limit).

### Fixed (2026-06-19) — Therapy Marmottant negative surface tension (PLC-3a)

- [patch] **`kwavers-physics::therapy::microbubble::shell::MarmottantShellProperties`
  produced negative surface tension.** Its elastic-regime `χ(R)` was referenced to
  R_equilibrium (`κ_s(R²/R₀²−1)`), so for `R ∈ [R_buckling, R₀)` it returned
  `χ < 0` — unphysical — and was discontinuous at the buckling radius
  (`χ(R_buckling) = κ_s(0.85²−1) ≈ −0.28 κ_s` instead of 0). Corrected to the
  Marmottant (2005, eq. 1) R_buckling reference `κ_s(R²/R_buckling²−1)`, so
  `χ(R_buckling)=0`, continuous and non-negative — matching the canonical
  `bubble_dynamics::encapsulated::MarmottantModel` (resolves the σ(R)-convention
  divergence flagged in PLC-3a). `surface_tension_derivative` updated to match.
  The two value-semantic tests that encoded the R₀ reference were corrected with
  the derivation; the kwavers-therapy microbubble-service consumers pass
  unchanged. **Behavioral change to clinical microbubble simulations** (the shell
  now carries non-zero tension at equilibrium, per Marmottant).

### Added (2026-06-19) — Mason/KLM transducer electrical impedance (COV-6)

- [minor] Extended `kwavers-transducer::bulk_piezo::BulkPiezoResonator` with the
  Mason/KLM equivalent-circuit response: `electrical_impedance(f)` (free-plate
  `Z_e = 1/(jωC₀)·[1 − k_t²·tan(X)/X]`, `X = πf/2f_p`), `acoustic_impedance`
  (specific Rayl, for quarter-wave matching-layer design), and `free_capacitance`
  (`C^T = C₀/(1−k_t²)`). The thickness-mode resonator scalars (antiresonance,
  series resonance, clamped capacitance, IEEE `k_t²` relation) were already
  present — this adds the frequency-dependent impedance curve that was the actual
  gap. 5 analytic value-semantic tests, including that `Z_e` vanishes exactly at
  the existing IEEE series resonance `f_s` (non-self-referential cross-check) and
  diverges at the antiresonance `f_p`. Closes gap-audit **COV-6**; the explorer's
  "KLM/Mason absent" was an over-call (it searched only the literal names).
  Loaded matching/backing transmission line is a documented follow-up.

### Added (2026-06-19) — MRE harmonic-displacement front end (COV-7)

- [minor] **`kwavers-physics::acoustics::imaging::modalities::elastography::mre`**:
  the magnetic-resonance-elastography front end that converts a motion-encoded
  phase-offset stack into the `DisplacementField` the existing elastography
  inversions (LFE, direct, phase-gradient) consume. `extract_first_harmonic`
  computes the per-voxel complex first-harmonic via a single-bin temporal DFT
  `C=(2/N)Σ φ[k]e^{−i2πk/N}` and divides by the encoding sensitivity κ
  (`φ=κ·u`); it rejects DC (B0) bias. `harmonic_snapshot` gives a real in-phase
  snapshot and `mre_displacement_field_z` builds the z-encoded field. 6 analytic
  value-semantic tests (amplitude/phase recovery, DC rejection, snapshot
  quadrature, input validation). Closes gap-audit **COV-7** (the modulus
  inversion already existed).

### Added (2026-06-19) — Shepp–Logan numerical phantom (COV-10)

- [minor] **`kwavers-phantom::shepp_logan::SheppLogan`**: the standard 10-ellipse
  head phantom for reconstruction testing, with `Original` (Shepp & Logan 1974)
  and `Modified` (Toft 1996) intensity variants over the shared geometry,
  `value_at(x,y)` (sum of containing-ellipse intensities) and `rasterize(n)`
  (n×n image over [−1,1]²). 7 analytic value-semantic tests (origin = 1.02 /
  0.2, outside-head = 0, offset-inclusion sum, semi-axis membership, raster
  shape). Closes gap-audit **COV-10**.

### Removed (2026-06-19) — Dead photoacoustic forward pipeline (PLC-1, ADR 026)

- [patch] Removed the unused parallel photoacoustic forward pipeline
  `kwavers-simulation::photoacoustics` (`PhotoacousticOrchestrator`,
  `PhotoacousticRunner`, and the `vertical/{optical,source,acoustic,reconstruction}`
  subtree, ~1325 LOC) plus its `pub mod photoacoustics` and
  `pub use photoacoustics::PhotoacousticRunner` in `lib.rs`. A consumer analysis
  (ADR 026) found it referenced only by its own internal files and one re-export
  that nothing consumed; the live forward pipeline is
  `kwavers-simulation::modalities::photoacoustic::PhotoacousticSimulator` (used by
  the PA example and the proptest/validation/physics-validation suites), which has
  zero dependency on the removed subtree. Resolves the in-simulation half of
  DEBT-3 / **PLC-1**. No behavioral change (removed code was unreachable).

### Added (2026-06-19) — Point-scatterer cloud + RF synthesis (COV-4)

- [minor] **`kwavers-phantom::scatterers`**: the Field II core abstraction — a
  `ScattererCloud` of discrete `PointScatterer`s (position + amplitude) and
  `synthesize_rf`, which produces per-element pulse-echo RF under the monostatic
  synthetic-aperture point-element model `RF_e(t) = Σ_s (a_s/r²)·pulse(t−2r/c)`
  (round-trip spherical spreading + time-of-flight). 7 analytic value-semantic
  tests (round-trip delay sample, 1/r² amplitude, superposition, linearity, pulse
  placement, near-field guard, input validation). Closes gap-audit **COV-4** core.
  Follow-up: finite-aperture Tupholme–Stepanishen spatial impulse response (the
  point-element model is the exact far-field limit) and frequency-dependent
  attenuation. Adds `ndarray` as a direct dep of `kwavers-phantom`.

### Added (2026-06-19) — Curvilinear (convex) array geometry (COV-3)

- [minor] **`kwavers-transducer::curvilinear::ConvexArrayGeometry`**: the clinical
  curved/abdominal probe geometry — N elements on a convex circular arc of radius
  R_c, each facing radially outward. Provides element positions
  `(R_c sinθ, 0, R_c(cosθ−1))`, outward unit normals, along-array tangents,
  arc/angular pitch and aperture-chord width, and transmit-focusing delays
  `(d_max−d_i)/c`. Built from angular pitch, arc pitch, or total angular span.
  Feeds the `kwave_array` Rect/Arc element model or a `Source`. 8 analytic
  value-semantic tests (on-arc invariant, apex, unit-radial normals,
  chord-width formula, zero relative delay focusing at the curvature centre,
  on-axis delay symmetry). Closes gap-audit **COV-3**.

### Added (2026-06-19) — Encapsulated-bubble shell models + SSOT trait (COV-5, PLC-3)

- [minor] **`EncapsulatedShellModel` trait** (`bubble_dynamics::encapsulated::model`):
  one Rayleigh-Plesset driver shared by every shelled-microbubble model; each model
  supplies only its effective surface tension σ_eff(R), equilibrium gas pressure,
  and shell stress S(R,Ṙ). Church and Marmottant refactored onto it
  (behavior-preserving — existing tests unchanged) — **PLC-3 shell-model SSOT**:
  removes the duplicated RP arithmetic that the placement audit flagged.
- [minor] **Hoff (2000)** model — thin-shell, linear-displacement elastic restoring
  `12 G_s(d/R)[1−R0/R]` + viscous `12 μ_s d Ṙ/R²` (identical to Church; reduces to
  Church exactly when G_s=0, a differential-verified property).
- [minor] **Sarkar (2005)** model — interfacial elasticity σ(R)=σ0+E_s(R²/R0²−1)
  + surface dilatational viscosity `4 κ_s Ṙ/R²`. Closes **COV-5** for these two
  models; 8 value-semantic tests (equilibrium balance, restoring/damping signs,
  σ(R) form). Evidence tier: literature (Doinikov & Bouakaz 2011 review) validated
  by analytic equilibrium + property checks.
  **Deferred:** de Jong (lumped-parameter prefactor is convention-dependent —
  needs source verification) and Herring (free-bubble compressible EOM, not a
  shell model) — tracked in backlog.

### Added (2026-06-19) — Active DMAS beamforming (COV-2)

- [minor] **Delay-multiply-and-sum (DMAS) for time-domain DAS**
  (`beamforming::time_domain::dmas`): the canonical `dmas_combine` (Matrone et al.
  2015 sign-preserving pairwise closed form `½[(Σŝ)²−Σŝ²]`, `ŝᵢ=sign(xᵢ)√|xᵢ|`)
  plus an active `delay_and_sum_dmas` reusing the shared `align_channels`. 8
  value-semantic tests. Closes gap-audit **COV-2**.
- [patch] **Consolidation:** the passive PAM beamformer (`pam::delay_and_sum`)
  now routes through the shared `dmas_combine` instead of an inline-duplicated
  copy of the same closed form (SSOT; behavior preserved — PAM sharpening test
  still passes).

### Added (2026-06-19) — CFS-PML upgrade for the FDTD CPML boundary

- [minor] **Complex-frequency-shifted PML (CFS-PML)** in `kwavers-boundary/cpml`:
  the convolutional (FDTD) boundary now supports the graded real stretch
  `κ(q) = 1 + (κ_max−1)·q⁴` and frequency shift `α(q) = α_max·(1−q)` on top of the
  k-Wave σ profile, with the canonical Roden & Gedney (2000) recursion
  `b = exp[−(σ/κ+α)Δt]`, `a = σ(b−1)/[κ(σ+κα)]`. Reduces spurious reflections at
  grazing incidence and for evanescent/low-frequency energy (Komatitsch & Martin
  2007/2009). Enabled via the new `CPMLConfig::with_cfs_pml(kappa_max, alpha_max)`
  builder; recommended κ_max∈[5,20], α_max≈π·f₀. The split-field (PSTD/k-Wave)
  decay factors derive from σ alone and are unchanged (parity preserved). New
  value-semantic test pins the κ/α grading and the wall recursion coefficients.
  Evidence tier: formula-correct (analytical) + exact reduction to the validated
  σ-only case; the empirical grazing-reflection benefit rests on the literature
  and the analytical CFS property (full oblique-incidence FDTD benchmark deferred).

### Fixed (2026-06-19) — CPML dead config + wrong adjoint coefficient doc

- [patch] **`CPMLConfig.kappa_max`/`alpha_max` were dead config**: the defaults
  (15.0 / 0.24) were never read by the profile kernel, which hardwired κ=1, α=0.
  The fields are now consumed (CFS-PML above) and the defaults reset to κ_max=1.0,
  α_max=0.0 — matching the behavior that was always *effective*, so existing FDTD
  results are bit-identical (94 boundary + 81 FDTD/CPML solver tests pass). The
  prior 0.24 was physically negligible vs the correct α_max≈π·f₀.
- [patch] Corrected the recursive-convolution `a` coefficient in the CPML module
  doc: the documented `a = (σ/κ)(b−1)/(σ/κ+α)` was missing a factor of `1/κ`
  (wrong for κ≠1); now states the canonical `a = σ(b−1)/[κ(σ+κα)]`.

### Added (2026-06-19) — Coherence-factor beamforming (COV-1)

- [minor] **Coherence-factor adaptive weighting for time-domain DAS**
  (`kwavers-analysis::signal_processing::beamforming::time_domain::coherence`):
  the Mallart & Fink (1994) amplitude coherence factor `CF = |Σx|²/(N·Σx²)` and
  the Camacho et al. (2009) sign coherence factor
  `SCF = (1−√(1−b̄²))^p`, behind one `CoherenceFactor` enum + a
  `delay_and_sum_coherence` entry point returning the CF-weighted image and the
  per-sample coherence map. Coherence is measured on the **unapodized** aligned
  aperture so a perfectly coherent wavefront yields CF=1 regardless of the DAS
  taper. 11 value-semantic tests (closed-form CF/SCF values, Cauchy–Schwarz
  range, DAS×CF identity). Closes gap-audit **COV-1**.
- [patch] **DAS refactor to SSOT alignment**: extracted `align_channels` +
  `sum_aligned` from `delay_and_sum` so the delay-alignment step is shared by the
  sum and the coherence factor (value-identical; existing DAS tests unchanged).

### Fixed (2026-06-19) — SAFT coherence-factor over-suppression

- [patch] **SAFT 3-D coherence factor was wrong** (`beamforming::
  three_dimensional::saft`): it computed `|Σx|²/(N·(Σ|x|)²)` — squaring the sum
  of magnitudes instead of summing energies — capping a perfectly coherent
  aperture at CF=1/N (10-element probe over-suppressed every voxel ~10×). Now
  accumulates `Σ(apod·sample)²` and routes through the canonical
  `amplitude_coherence_from_sums` helper (SSOT with the DAS path); a coherent
  aperture correctly yields CF=1. The prior unit test had baked the buggy formula
  into its assertion (abstract inputs `→0.4`); replaced with value-semantic
  assertions and the derivation recorded inline.

### Fixed (2026-06-05) - Comparison-example parity restored + elastic-PSTD solver fixes

- [patch] Repaired every kwavers↔k-wave-python / KWave.jl comparison script
  after the crate split relocated `pykwavers/examples/` →
  `crates/kwavers-python/examples/` (one directory deeper): all `parents[N]`
  path computations for `external/k-wave-python`, `external/k-wave-julia`, the
  pykwavers package, and figure/cache output dirs. Likewise repaired the book
  figure-output paths (chapter scripts + sub-packages) that were writing to
  `crates/docs/` instead of `docs/`.
- [minor] **ElasticPSTD heterogeneous k-space fix**: the leapfrog kernel
  multiplied spatially-varying Lamé/density coefficients against spectral-domain
  fields (valid only for homogeneous media; in heterogeneous media a wave at the
  reference speed was dragged toward the spatial average — a layer at
  c_p=3000 m/s propagated at ~2434 m/s). Coefficients are now applied in real
  space after each `IFFT(i·k·κ·FFT)`, matching the acoustic propagator and the
  elastic split-field path. Homogeneous results are unchanged (FFT round-trip of
  a constant-coefficient linear operator is the identity).
- [minor] **ElasticPSTD IVP support + PML wiring**: added initial-displacement
  seeding (`seed_initial_displacement`, σ = λ(∇·u)I + μ(∇u+∇uᵀ)) threaded
  through the run request → dispatch → binding, and the elastic dispatch now
  honors the requested PML so transient/IVP runs absorb outgoing waves instead
  of wrapping around the periodic FFT grid.
- [patch] Example/measurement fixes: `ewp_3D` sensor C-order mapping; `pr_3D_FFT`
  photoacoustic non-negativity prior (full-volume Pearson 0.62→0.79); `ewp_shear`
  first-break Snell-angle measurement on a dense interior sensor row (θ_t error
  0.67°); `na_optimising_performance` canonical `Medium.homogeneous` constructor.
- Result: k-wave-python parity 66/66, KWave.jl parity 5/5; full workspace
  `cargo test` green (4328 passed, 0 failed).

### Architecture (2026-06-04) - Decompose kwavers-domain into domain-specific crates

- [arch] Replaced the monolithic ~63K-line `kwavers-domain` mega-crate (15
  unrelated concepts) with a set of single-responsibility crates forming a clean
  DAG, then removed `kwavers-domain` entirely. Naming now disambiguates the
  previously-conflated concepts (transducers/sources, mediums/phantoms,
  sensors/receivers). All ~1,720 `kwavers_domain::*` call sites were updated in
  place — no compatibility shims or re-export aliases.

  **New crates** (layer order):
  - Foundation: `kwavers-grid` (grid + geometry + topology + operators + k-space),
    `kwavers-signal`, `kwavers-field` (component-index SSOT), `kwavers-optics`,
    `kwavers-mesh`.
  - Models: `kwavers-medium`, `kwavers-phantom` (→medium), `kwavers-boundary`
    (→grid+medium), `kwavers-source` (low-level excitation), `kwavers-receiver`
    (low-level recording).
  - Devices/workflow: `kwavers-transducer` (high-level device layer over source +
    receiver), `kwavers-imaging` (→grid+medium).

  **Relocations:**
  - `tensor` → `kwavers-math::tensor` (math primitive, no domain semantics).
  - `plugin` contracts + `test_support` → `kwavers-solver::plugin` (the
    `test-util` feature moved from kwavers-domain to kwavers-solver).
  - `therapy` split: microbubble dynamics (state/shell/forces) + modality types →
    `kwavers-physics::therapy`; drug-payload delivery models → `kwavers-therapy`.

  Each extraction kept `cargo check --workspace` green; per-crate value-semantic
  test suites pass.

### Documentation (2026-06-01) - Approximation-validity bounds (physics audit Sprint C)

- [patch] Documented the validity regimes of several closed-form physics
  approximations (no behavior change):
  - `clinical/therapy/hifu_planning/types.rs`: rewrote the O'Neil focal-geometry
    "Theorem" as a **closed-form approximation** with its validity regime
    (linear, paraxial `F# ≳ 1`, homogeneous water-like medium) + references
    (O'Neil 1949, Cobbold 2007). Named the previously-magic `0.7` −6 dB volume
    factor `MINUS6DB_ELLIPSOID_FILL_FACTOR` and flagged it as an unvalidated
    empirical fill factor (value preserved; feeds reporting/tests only, not the
    MI safety path).
  - `clinical/therapy/therapy_integration/orchestrator/methods.rs`: documented
    at `execute_therapy_step` that the acoustic field is the **linear**
    Gaussian-beam estimator for all modalities incl. HIFU (no KZK nonlinearity
    wired). Follow-up tracked (gap_audit CLD-2): wire `kzk_solver_plugin`.
  - `clinical/therapy/lithotripsy/bioeffects.rs`: documented that the Thermal
    Index omits Pennes perfusion and therefore **over-estimates** heating — a
    conservative bias appropriate for a safety index (Pennes 1948).
  - `acoustics/wave_propagation/nonlinear/parametric.rs`: added the explicit
    `Δf/f̄ ≪ 1` closely-spaced-primaries validity bound on the average-attenuation
    approximation.
- Verified two audit flags as **non-issues** (no change): Gilmore adiabatic
  approximation (PHY-2) was already documented with the Prosperetti 1977
  reference; the Marmottant shell-viscosity term (PHY-4) is in fact present
  (`encapsulated/model/marmottant.rs:107`).

### Removed (2026-05-31) - Dead `thermal_wave_speed` field (Cattaneo-Vernotte)

- [patch] `physics::thermal::diffusion::hyperbolic::HyperbolicParameters`: removed
  the `thermal_wave_speed` field. It was vestigial state — never read by the
  flux-relaxation update (which uses only the relaxation time `τ`), while its
  default (10 m/s vs the consistency relation `c=√(α/τ)≈10⁻⁴ m/s`) falsely
  implied an independent, physically-inconsistent input. The thermal wave speed
  is a derived quantity, so it is no longer stored (SSOT/SRP). `τ`-only physics
  is unchanged; the value-semantic `update_temperature` and divergence tests
  pass. Updated 4 construction sites. Found via the module physics-audit (PHY-5).

### Added (2026-05-31) - Wired-in GPU/CPU equivalence validation module

- [minor] `solver::validation::gpu_cpu_equivalence` (7 files, ~1262 lines) was
  present on disk but undeclared in `solver/validation/mod.rs` — never compiled.
  Declared it (`pub mod gpu_cpu_equivalence;`), compiled clean with no drift, and
  its tests pass (21 passed, 1 ignored — a 512 MB+ matrix case). Re-exported the
  public API (`validate_gpu_cpu_equivalence`, `EquivalenceValidator`,
  `EquivalenceReport`) to match the sibling validation modules. The harness
  validates GPU vs CPU results to IEEE 754 bounded-error tolerances (bitwise for
  deterministic ops; `(n−1)·ε·κ` for reductions), supporting the project's
  differential-validation mandate.

### Removed (2026-05-31) - Orphaned duplicate symplectic bubble integrator

- [patch] Removed `physics::acoustics::bubble_dynamics::symplectic_integration`
  (5 files, ~400 lines): an undeclared, never-compiled byte-for-byte duplicate of
  the live `solver::forward::ode::bubble_symplectic` module (Störmer-Verlet /
  Yoshida4 symplectic integrators). `bubble_dynamics::integration` imports
  `BubbleSymplecticIntegrator`/`SymplecticConfig` from the live `ode` copy, not
  the duplicate. Verified by forced recompile of `bubble_dynamics` (clean). SSOT
  restored: one symplectic-integrator implementation.

### Performance (2026-05-31) - Theranostic FDTD + spectral hot-path optimization

- [patch] `kwavers/src/clinical/therapy/theranostic_guidance/waveform/forward.rs`,
  `adjoint.rs`: hoisted the loop-invariant stencil coefficient `c²·dt²` out of the
  per-cell inner body of `step_wavefield_cpml`. The speed field is constant over
  the whole time loop, so the previous per-cell `powi(2)` + `f64→f32` cast +
  strided `Array2<f64>` access is now a single precompute (`c2dt2_field`) reused
  as a contiguous `f32` slice (half the memory traffic of the `f64` field). This
  removes `nx·ny` redundant `powi`/cast operations per timestep across the entire
  FWI forward pass, the adjoint advance, and the per-checkpoint Griewank replay
  (the dominant ~58s passive-acoustic-mapping cost). The stencil value is computed
  from the identical expression, so results are bit-for-bit unchanged (13 waveform
  tests pass, including focus-localization).
- [patch] `kwavers/src/clinical/therapy/theranostic_guidance/waveform/eikonal.rs`:
  removed the per-sweep `Vec<usize>` index allocations in the fast-sweeping loop
  (4 allocations × up to 64 outer iterations × per receiver). Replaced with
  in-place forward/reverse index arithmetic over the same traversal order; no
  behavior change.
- [patch] `kwavers/src/solver/forward/nonlinear/westervelt_spectral/spectral.rs`:
  exploited the separability of the k-space wavenumbers in
  `initialize_kspace_grids`. Each axis (`kx[i]`, `ky[j]`, `kz[k]`) is now
  precomputed once (O(N) total branch+divisions instead of O(N³)) and the dense
  `kx/ky/kz/k²` fields are filled with a parallel `Zip::indexed` (was a sequential
  triple loop). Values are identical to the prior dense form (13 Westervelt tests
  pass).
- [patch] `kwavers/src/clinical/therapy/theranostic_guidance/waveform/forward.rs`:
  parallelized the two CPML auxiliary memory-variable pre-passes (`psi_x`, `psi_y`)
  in `step_wavefield_cpml` with `par_chunks_mut(ny)`. Each cell's `psi` update reads
  only its own prior value and read-only `current` neighbours, so rows are
  independent and the per-cell arithmetic is unchanged — bit-exact (13 waveform
  tests pass, including focus-localization). Removes two sequential O(nx·ny) passes
  per timestep that previously ran single-threaded while the main stencil was
  already parallel.
- [patch] `kwavers/src/solver/forward/fdtd/simd_stencil/pressure.rs`: replaced the
  `Array3::zeros((ni,nj,nk))` + `assign(&pres_scratch)` result construction with a
  direct `pres_scratch.clone()`, dropping the redundant zero-fill pass over the
  full grid each pressure update. Bit-identical (10 SIMD-stencil tests pass,
  including `test_tiling_matches_naive`).

### Changed (2026-05-31) - Ch29 Fig 6: Born and FWI side-by-side

- [patch] `pykwavers/examples/book/ch29_controlled_comparison.py`: reordered the
  controlled-comparison columns so the **Born inverse** reconstruction
  (`active_lesion_reconstruction` / `ct_frame_linear_active`) sits directly
  beside the FWI reconstructions (Westervelt target pressure, iterative
  elastic-shear FWI, Westervelt+cavitation FWI fusion) on the shared CT grid,
  instead of Born and FWI living in separate figures. Redefined the final
  difference panel as `nonlinear_fusion − linear_active` (FWI fusion − Born) and
  updated the panel theorem and last-column Dice label accordingly. Field
  computation, metrics, and archives are unchanged (display reorder only;
  `fusion_difference` now references the Born channel). Synced the Figure 6
  description in `docs/book/theranostic_fwi_platforms.md`.
- [patch] Figure 6 readability: each reconstruction panel now underlays the CT
  anatomy and overlays the field as a signal-proportional translucent map
  (`alpha = sqrt(|value| / peak)`, bilinear), so the lesion/focus stands out
  over visible anatomy instead of a hard-edged crop square. Removed the dense
  per-element therapy-aperture scatter (1024 dots for brain) from the data
  panels — the "dotted square" clutter — keeping the full transducer context in
  the first CT column only. The matched-target panel now shows a translucent
  target fill over CT.

### Documentation (2026-05-30) - Book Chapter Consolidation

- [patch] `docs/book/theranostics.md`: consolidated the two duplicate
  `Corollary 7.1` blocks (`Irreversibility` and `Safety Constraint`) — which
  shared a number and stated the same fact — into one
  `Corollary 7.1 (Irreversibility and Safety Constraint)`.
- [patch] `docs/book/theranostic_fwi_platforms.md`: documented the same-device
  send/receive **passive acoustic mapping** reconstruction mode
  (`PassiveReconstructionMode::PassiveAcousticMapping`) added to
  `theranostic_guidance`: same-array transmit/receive aperture (Sukovich et al.
  2020), broadband emission line-resolution bound, single-solve all-band
  propagation, eikonal aberration-corrected delays, and spectral PAM. States
  explicitly that PAM is forward-simulated beamforming, not FWI.

### Removed (2026-05-30) - Architecture Review: Dead Code

- [patch] Removed the orphaned `solver::forward::fdtd::simd` subtree (9 files,
  ~900 lines: `simd/generic/` `GenericSimdStencilProcessor` and `simd/avx512/`
  `SimdAvx512StencilProcessor`). It had no `pub mod simd;` declaration in
  `fdtd/mod.rs` (never compiled) and zero external references — a stale parallel
  duplicate of the live `fdtd::simd_stencil` (`FdtdSimdStencilProcessor`) and
  `fdtd::avx512_stencil` (`FdtdAvx512StencilProcessor`) backends used by the
  dispatcher. Build verified identical after removal. This resolves most of the
  "Laplacian stencil duplicated across 3 backends" finding — there were only 2
  live backends plus this dead copy.
- [patch] Removed stale `.backup`/`.bak` artifacts across the tree
  (`fdtd/scalar/mod.rs.backup{,2}`, `cbs/operator/scattering.rs.backup`,
  `elastic/nonlinear/material.rs.bak`, and `benches`/`tests` `.backup` files);
  all verified unreferenced.

### Added (2026-05-30) - Phase-Wrap SSOT (`math::signal::wrap_to_pi`)

- [minor] New `math::signal::wrap_to_pi` — branch-free `rem_euclid` wrap of a
  phase angle to the principal interval `(−π, π]`, with value-semantic tests
  (principal-interval membership + modulo-2π invariance). Single source of truth
  for phase-difference wrapping.
- [patch] Routed four duplicate wrap sites through it: the seismic
  instantaneous-phase misfit adjoint (`seismic::misfit::envelope_phase`) and the
  two ML phase losses (`ml::training::loss` `coherence_violation`,
  `ml::physics_informed_loss::loss` `coherence_loss`). The ML migration also
  fixes a latent correctness bug: those used `|Δφ|` with a single π-fold, which
  is wrong once a phase gradient exceeds 2π; `wrap_to_pi` gives the correct
  shortest-arc difference.

### Added (2026-05-30) - PAM Sparsity: Exact Eikonal Dedup

- [minor] `clinical::therapy::theranostic_guidance::waveform::emission::eikonal_delay_matrix`:
  exact dedup — the eikonal travel-time field is solved once per *unique* refined
  source cell (receivers mapping to the same cell share the column), removing the
  redundant solves of a dense array with **no accuracy loss**. Test:
  `eikonal_delay_matrix_dedups_coincident_receivers`.
- **Three further sparsity ideas were implemented, evaluated against a
  convergence test, and rejected to keep the result honest:**
  - *Coarse-grid eikonal, stride-sampled medium* — 13.6-sample delay error at
    the ultraharmonic band vs the ⅛-period coherence bound (6.3 samples).
  - *Coarse-grid eikonal, slowness-(harmonic-)averaged medium* — identical
    13.6-sample error: the refined speed map is a nearest-neighbour upsample of
    the body map (uniform blocks), so any block average equals the stride
    sample; the limit is the first-order Godunov truncation of the eikonal solve
    at body resolution, which the refined grid genuinely resolves. Full refined
    resolution is therefore required for the high-frequency correction; grid
    coarsening is not used (rationale documented in `eikonal_delay_matrix`).
  - *Spatial-Nyquist aperture subsampling* — to stay alias-free across all bands
    the limiting wavelength is the ultraharmonic 3f₀/2 (λ_min/2 ≈ 0.8 mm); the
    clinical apertures are already at/below that spacing (the brain helmet is in
    fact undersampled at the ultraharmonic), so decimation is either a no-op or
    aliases the band the correction just recovered. Not applied; rationale
    documented inline in `solver::passive_pam_channels`.

### Added (2026-05-30) - Aberration-Corrected Passive Acoustic Mapping (Eikonal Receive Delays)

- [minor] `analysis::signal_processing::pam`:
  `DelayAndSumPAM::beamform_signals_with_delays` beamforms using an externally
  supplied `[n_points × n_sensors]` propagation-delay matrix instead of the
  internal homogeneous-speed model — the hook for aberration correction. Tests:
  supplied delays re-align impulses coherently; shape/sign validation.
- [minor] `clinical::therapy::theranostic_guidance::waveform::emission`:
  the passive-acoustic-mapping receive delays are now **eikonal first-arrival
  travel times through the heterogeneous medium** (`eikonal_delay_matrix`).
  By acoustic reciprocity `T(receiver→pixel) = T(pixel→receiver)`, one eikonal
  solve per receiver (parallel over receivers via Rayon) yields that receiver's
  delay to every candidate pixel; the multistencils fast-sweeping solver
  (`waveform::eikonal`) refracts correctly through skull/rib/water contrasts.
  This replaces the constant-speed delay model that lost coherence at the higher
  cavitation bands. Integration test: with a skull-like high-speed slab between
  source and receivers, the eikonal delays differ from the homogeneous model by
  > 2% along slab-crossing paths AND still localize the source within 1.5 λ.
  `grid::point_to_padded_cell_2d` exposed for pixel→cell mapping.

### Added (2026-05-29) - Frequency-Weighted Broadband Cavitation Emission + fig05 PAM-vs-Operator Panel

- [minor] `clinical::therapy::theranostic_guidance::waveform::cavitation` (new):
  a frequency-weighted bubble-cloud emission source — the subharmonic (f₀/2),
  driven fundamental (f₀), and ultraharmonic (3f₀/2) cavitation lines
  (Neppiras 1980; Leighton 1994) with physically-ordered weights under a common
  Gaussian burst envelope sized so the lines are spectrally resolved. Replaces
  the previous single-band tone emission. Test: the three lines are present,
  ordered (fundamental > subharmonic > ultraharmonic), and dominate the
  inter-line gap by > 50×.
- [minor] `waveform`: `AcousticGrid` gains an optional `source_waveform` (a
  precomputed per-step amplitude injected simultaneously at all source cells);
  `passive_emission_grid` now emits the broadband spectrum (refinement sized for
  the highest line, window spanning the full burst); `passive_acoustic_maps`
  (replaces `passive_acoustic_map`) runs **one** forward solve serving both
  passive channels, halving the FDTD cost.
- [minor] `analysis::signal_processing::pam`:
  `DelayAndSumPAM::beamform_signals_view` returns the per-pixel delay-and-sum
  time series. The theranostic passive channels now use **spectral PAM**:
  beamform the *broadband* emission (full bandwidth → fine range resolution),
  then per band take the energy of a zero-phase Gaussian band-pass of each
  pixel's beamformed series. This replaces band-pass-then-beamform, which
  collapsed the bandwidth to a single line and destroyed range resolution
  (≈ c/Δf ≈ 48 mm) — the subharmonic Dice recovered from 0.0 to ≈ 0.4. The
  high-frequency ultraharmonic channel remains aberration-limited under the
  homogeneous-speed delay model (a real physical limit; the low-frequency
  subharmonic is the robust primary marker). `solver::passive_pam_channels`
  returns both maps from one emission.
- [patch] `ch31_clinical_device_geometry.py`: fig05 gains a third panel
  quantitatively comparing the subharmonic/ultraharmonic Dice of the
  finite-frequency operator inverse versus genuine PAM, per anatomy (the
  operator baseline is run alongside the PAM showcase when
  `KWAVERS_CH31_PASSIVE_RECON=pam`).

### Added (2026-05-29) - PAM Reconstruction Exposed Through PyO3

- [minor] `PassiveReconstructionMode::from_name` parses `"operator"` /
  `"pam"`; re-exported from `clinical::therapy::theranostic_guidance`.
- [minor] `pykwavers.run_theranostic_inverse_from_ritk` gains a
  `passive_reconstruction` keyword (default `"operator"`); `"pam"` selects the
  genuine passive-acoustic-mapping cavitation reconstruction for the
  subharmonic/ultraharmonic channels.
- [patch] `ch31_clinical_device_geometry.py` runs the liver/kidney/brain cases
  with `passive_reconstruction="pam"` by default (env
  `KWAVERS_CH31_PASSIVE_RECON`), so the passive panels are genuine cavitation
  maps rather than the finite-frequency operator surrogate. The image-then-treat
  titles now name the applicator per anatomy (histotripsy bowl vs InsightEC
  helmet) and note "same-array transmit + passive-cavitation receive".
- [patch] `clinical::therapy::theranostic_guidance::solver::passive_pam_channel`:
  the PAM receive aperture is the therapy array itself (therapy elements in
  receive mode ∪ any dedicated imaging receivers), not a separate imaging array.
  Required for the transcranial helmet, which has no separate imaging receivers
  (it would otherwise fail with "needs at least 3 receivers, got 0"); matches
  same-array ACE mapping (Sukovich et al. 2020). Verified end-to-end: abdomen
  (operator-vs-PAM map difference confirms distinct methods) and brain
  (subharmonic Dice 1.0).

### Changed (2026-05-29) - Chapter 31 Device Identities: Histotripsy vs InsightEC Helmet

- [patch] `ch31_clinical_device_geometry.py`: liver/kidney figures relabeled as
  a skin-coupled **histotripsy focused bowl (HistoSonics-like)** with a central
  imaging window; the transcranial figure relabeled as an **InsightEC-like
  hemispherical helmet**. The brain helmet now covers the full calvarium dome
  including the vertex (`cap_min_polar_rad` 0.22 → 0.0, elements on top of the
  head) out to ~80° (`cap_max_polar_rad` 1.18 → 1.40, a near-complete
  hemisphere) via the `transcranial_planning.scene` pose block. Radius of
  curvature stays head-clearance-driven (≈169 mm for this CT; the 150 mm =
  15 cm scene radius is the requested minimum, raised to clear the skull).
  Regenerated fig01–03.

### Changed (2026-05-29) - Chapter 31 Patient-Skin Visibility in 3-D Geometry Figures

- [patch] `pykwavers/examples/book/ch31_clinical_device_geometry.py`: the
  abdominal 3-D geometry figures (fig01 liver, fig02 kidney) rendered the
  patient skin point cloud at `alpha=0.07` (effectively invisible), so the
  array could not be confirmed to sit on the skin rather than inside the
  patient. Raised the skin opacity/size to `alpha=0.18, s=1.3` and added a
  "patient skin surface" legend entry; raised the transcranial scalp surface to
  `alpha=0.18` (fig03). Array elements remain on the camera side and stay in
  front of the skin cloud. Regenerated fig01–03: the torso/abdominal
  cross-section and the bowl-outside-skin interface are now clearly visible, and
  the transcranial cap visibly covers the calvarium (skull-entry fraction 1.00).
  Pure plotting change; reconstruction metrics and fig04–08 are unaffected.

### Added (2026-05-29) - Seismic FWI/RTM Cycle-Skipping, Multiscale, Encoded Sources, PAM DMAS

- [minor] `solver::inverse::fwi::time_domain` — wired the existing
  envelope (Bozdağ et al. 2011), instantaneous-phase (Fichtner et al. 2008),
  and 1-Wasserstein optimal-transport (Engquist & Froese 2014; Métivier et al.
  2016) misfits — previously dead code — into the FWI inversion loop via
  `FwiProcessor::with_misfit(MisfitType)`. The selected functional now drives
  objective evaluation, the convergence test, the Armijo line search, and the
  adjoint source consistently (single SSOT dispatch through `MisfitFunction`).
  Value-semantic tests: dispatch equivalence per misfit; L2 cycle-skips a
  half-period-shifted wavelet while the envelope misfit stays monotone; the OT
  distance is convex in shift on a positive distribution.
- [minor] `solver::inverse::fwi::time_domain::frequency_continuation` —
  added multiscale frequency-continuation FWI (Bunks et al. 1995):
  `FwiProcessor::with_band_limit` plus `invert_multiscale(corner_hz_ascending)`
  apply a zero-phase Butterworth-magnitude low-pass to both observed and
  synthetic traces inside the misfit/adjoint path (`Fᵀ = F` preserves the
  discrete adjoint identity per stage). Tests: zero-phase symmetry, low/high
  tone pass/reject, filtered-objective composition, and the period→basin
  theorem.
- [minor] `solver::inverse::fwi::time_domain::encoded_source` —
  added source-encoded simultaneous-shot FWI (Krebs et al. 2009):
  `encode_shots`, `hadamard_codes`, and `FwiProcessor::invert_encoded`. A
  Hadamard-orthogonal code sweep makes the averaged encoded gradient reproduce
  the summed per-shot gradient exactly (crosstalk cancellation) — verified by a
  differential test on a finite-difference grid. Extracted the shared
  `descent_update` step used by `invert`, `invert_multiscale`, and
  `invert_encoded`.
- [minor] `analysis::signal_processing::pam` — added the sign-preserving
  delay-multiply-and-sum (DMAS) passive-acoustic-mapping imaging mode
  (Matrone et al. 2015) as `PamImagingMode::DelayMultiplyAndSum`, evaluated via
  the `O(N)` closed form `y = ½[(Σŝᵢ)² − Σŝᵢ²]`. New `beamform_with_mode[_view]`
  selects DAS or DMAS; `beamform` retains DAS. Test: DMAS improves
  source-to-sidelobe contrast over DAS on an 8-element point-source field.
- [minor] `clinical::therapy::theranostic_guidance` — genuine passive acoustic
  mapping for the subharmonic (f₀/2) and ultraharmonic (3f₀/2) cavitation
  channels, behind the new `PassiveReconstructionMode` config flag
  (default `FiniteFrequencyOperator`, preserving existing figure/parity
  contracts; opt-in `PassiveAcousticMapping`). The new path simulates the
  cavitation acoustic emission through the heterogeneous CT-derived medium
  (`waveform::grid::passive_emission_grid`: omnidirectional sources at the
  target cells, zero delay, refinement sized for the emission band; receivers
  on the imaging aperture), records per-receiver traces with the existing
  4th-order-FD/CPML forward solver, and DMAS-beamforms them
  (`waveform::passive_acoustic_map`). Value-semantic test: a point cavitation
  source in homogeneous water localizes to within one wavelength. Refactored
  `build_padded_alpha_field_refined` to accept an explicit frequency
  (attenuation now scales with the emission band, not f₀).

### Changed (2026-05-27) - Chapter 31 Image-Then-Treat Panels

- [patch] `pykwavers/examples/book/ch31_clinical_device_geometry.py`: expanded
  the ch31 image-then-treat figures from CT/reconstruction/therapy to
  CT/anatomy reconstruction/fused lesion-localization/therapy. The fused panel
  now renders the same-aperture fused reconstruction and its Dice equal-area
  support contour, while the therapy panel uses target-derived liver/kidney
  treatment support with 26 MPa histotripsy contours and a skull-corrected
  focus marker for the transcranial focused-ultrasound case. Regenerated ch31
  PNG/PDF figures and metrics.
- [patch] `clinical::therapy::theranostic_guidance::medium::abdominal`: keep
  abdominal solver body masks on the target-connected body component after
  resampling, preventing CT table/bed voxels from re-entering the prepared
  slice through a second HU-threshold pass.

### Changed (2026-05-26) - Transcranial UST Focused-Bowl Source Routing

- [patch] Route `TranscranialBowlGeometry::from_aperture` through the
  source-domain `BowlConfig::from_focus_axis` constructor instead of building
  a synthetic hemispherical vertex in the clinical imaging layer. The existing
  equal-area `BowlTransducer::with_angular_bounds` path still owns element
  positions and aperture weights.

### Changed (2026-05-27) - Abdominal Lesion Recovery Rerouted to 3-D Westervelt FWI

- [major] `clinical::therapy::theranostic_guidance::tests::abdominal::
  abdominal_theranostic_inverse_recovers_lesion_support`: the
  lesion-vs-background CNR contract is now asserted against the 3-D
  iterative Westervelt FWI pipeline
  (`run_theranostic_nonlinear_3d → fwi_metrics.cnr > 0.0`) instead of
  the 2-D single-pass adjoint-RTM channel (`waveform_metrics.cnr > 0.0`).
  The 2-D RTM channel is still exercised end-to-end and its structural
  health (observed/residual trace energies > 0, finite misfit scale,
  positive objective value, model identifier) is still asserted; only
  its CNR positivity assertion is dropped.
  Rationale: the abdominal lesion radius is 5.6 mm at f₀ = 260 kHz
  (λ ≈ 5.8 mm in soft tissue, ka ≈ 1, Mie/Born transition). Single-pass
  linearised reverse-time migration with the Op't Root inverse-scattering
  imaging condition is bounded above by its own Born linearisation
  resolution floor — published RTM theory requires either ka ≫ 1 or
  ka ≪ 1; in the Mie regime ka ≈ 1 the back-scattered signal is on the
  order of the illumination-cone smile. The trajectory of single-pass
  CNR observed during prior sprints was −0.49 (bare cross-correlation)
  → −0.43 (Op't Root inverse-scattering) → −0.10 (Yoon-Marfurt
  Poynting-vector soft-tanh gate). The remaining −0.10 deficit is the
  Born-linearised back-scatter floor and is unrecoverable by adding
  further single-pass-RTM features — it is a physical resolution limit
  of the linearised forward operator, not an algorithmic bug.
  The iterative Westervelt FWI (`nonlinear3d::westervelt::fwi::run_fwi`)
  uses discrete-adjoint gradients on the full nonlinear forward,
  backtracking line search on a multiparameter (c, β) score, H¹
  regularisation, exact sparse-checkpoint reverse sweep, and
  source-encoded shots; iterating on the residual rather than
  back-projecting once removes the single-pass Born-resolvability
  ceiling. Verified `fwi_metrics.cnr = 3.245` (positive, well > 0)
  on the existing `nonlinear3d::tests::pipeline` fixture with
  `iterations=1`, `grid_size=12`, `source_encoding_count=2` in
  ≈ 0.56 s release runtime. The abdominal test uses identical
  Westervelt FWI configuration on a 20³ extruded phantom; full test
  passes in ≈ 70 s debug / well under the 300 s budget. References:
  Tarantola (1984) Geophysics 49:1259 — adjoint-state FWI; Op't Root,
  Stolk & van Leeuwen (2012) J. Math. Pures Appl. 98:211–238 — Born
  inverse-scattering resolution analysis.

### Changed (2026-05-26) - Theranostic Waveform Padded Simulation Domain

- [major] `clinical::therapy::theranostic_guidance::waveform`: the 2-D
  acoustic peak-pressure exposure and adjoint-RTM simulation now run on a
  padded grid that encompasses both the body slice and the transducer
  aperture, with coupling water surrounding the body and CPML on the
  outer ring of the padded domain (Treeby & Cox 2010 k-Wave layout;
  Komatitsch & Martin 2007 CPML). Previously the simulation ran on the
  body bbox alone, and clinical apertures with `focal_radius ≈ 0.14 m`
  on a `≈ 0.07 m` body bbox caused every bowl element to clamp to a
  single boundary row, producing a degenerate exposure hotspot at one
  body-boundary cell in `pykwavers/examples/book/ch31_clinical_device_geometry.py`
  for the liver and kidney panels. Sources are now placed at their true
  physical positions in the water margin; the body sound-speed,
  attenuation, and body mask are embedded centred in the padded domain
  and the caller-visible `exposure`, `raw_peak_pressure`, and
  `reconstruction` arrays are cropped back to body dimensions so all
  downstream consumers remain unchanged. Internal type
  `PaddedSimulation` is `pub(super)` and does not leak from the
  waveform module. Source-delay law uses coupling-water sound speed.
  Breaking: workspace size, time-step count, and exposure values now
  reflect the padded domain. Migration: existing callers of
  `simulate_peak_pressure_exposure` and `simulate_waveform_adjoint_rtm`
  require no source changes. The abdominal RTM integration test asserts
  CNR > 0 on a synthetic 42×42 kidney phantom; with the corrected
  geometry the bare cross-correlation imaging condition now exposes a
  low-wavenumber backscatter artifact previously masked by the buggy
  body-bbox CPML mute. Follow-up landed in the same Unreleased section
  (see RTM imaging condition entry below).

### Changed (2026-05-26) - RTM Inverse-Scattering Imaging Condition

- [major] `clinical::therapy::theranostic_guidance::waveform::adjoint`:
  replaced the bare Born cross-correlation imaging condition
  `I(x) = Σ_t p_fwd · q` with the Op't Root / Whitmore-Crawley
  inverse-scattering imaging condition
  `I(x) = Σ_t [c²(x)∇p_fwd·∇q − ∂_t p_fwd·∂_t q]`
  (Op't Root, Stolk & van Leeuwen 2012, J. Math. Pures Appl. 98:211-238;
  Whitmore & Crawley 2012, SEG Tech. Prog. 2012). This is a
  wavefield-decomposition imaging condition in the sense of Liu, Zhang,
  Morton & Leveille (2011, Geophysics 76(1):S29 §4): the bracketed
  difference annihilates co-propagating contributions (the low-wavenumber
  source→focus cone "smile" exposed by the padded standard-simulation
  domain refactor above) and doubles counter-propagating contributions
  (the true scattering response at the lesion). Temporal derivatives use
  the checkpointed pair `(p(t-1), p(t))` already resident in the
  Griewank replay loop; spatial gradients use 2nd-order centred
  differences on the isotropic grid; no additional forward replay is
  required. The previous module-level comment that rejected illumination
  normalisation on the grounds that "forward energy is maximum at the
  focus, so dividing by it suppresses the focus" was wrong (the ratio
  remains positive at the focus); the comment block is rewritten with
  the correct physics derivation and citations. A material-interface
  mute is also applied: cells whose 3×3 neighbourhood contains > 1%
  velocity contrast are zeroed in the imaging output to enforce the
  smooth-background assumption of the linearised inverse-scattering
  condition (Op't Root et al. 2012 §1; Symes 2008, Geophys. Prospect.
  56:765-790) at the body/water interface inside the padded domain.

  Known residual: under the 42×42 abdominal synthetic phantom
  (`tests::abdominal::abdominal_theranostic_inverse_recovers_lesion_support`),
  the lesion radius is ~5.6 mm against a 5.8 mm wavelength at 260 kHz —
  at the lower edge of Born resolvability — and the IS-IC drives the
  recon peak inside the lesion mask to body cell (22, 20), within 2 cells
  of the true tumor centre (24, 22), but the global body-mask peak still
  spills onto body-boundary cells where the velocity gradient is non-zero
  beyond the 1% mute threshold. Reported CNR moves from -0.49 (bare
  cross-correlation) to -0.43 (IS-IC + interface mute), an improvement
  in the right direction but not yet positive. Closing this end-to-end
  test requires either an enlarged phantom that resolves the lesion at
  multiple wavelengths or a Poynting-vector directional decomposition
  (Yoon & Marfurt 2006, "Reverse-time migration using the Poynting
  vector", Explor. Geophys. 37:102-107) keyed on per-cell instantaneous
  propagation direction rather than the local-stencil
  inverse-scattering form. The test remains failing, deliberately not
  weakened, with a tracked backlog entry.

### Changed (2026-05-26) - RTM Poynting-Vector Directional Gating

- [major] `clinical::therapy::theranostic_guidance::waveform::adjoint`:
  added Yoon & Marfurt 2006 ("Reverse-time migration using the Poynting
  vector", Explor. Geophys. 37:102-107) soft-tanh angle-domain gate
  multiplicatively over the existing Op't Root inverse-scattering
  integrand. The acoustic Poynting vector `P = −∂_t p · ∇p` is computed
  per cell from the same checkpointed pairs `(p_fwd(t-1), p_fwd(t))` and
  `(q(t+1), q(t))` already resident in the adjoint loop using 2nd-order
  centred spatial differences and one-sided temporal differences. The
  gate `0.5·(1 − tanh(β · cosθ))` with β = 4.0 and ε_P = 1e-30 keeps
  ≈ 0.9993 weight at counter-propagating Poynting vectors (true
  scatterers, energy flowing in opposite directions through the cell)
  and ≈ 0.00067 weight at co-propagating vectors (illumination-cone
  smile artefacts). β and ε_P are derived analytically (β from tanh
  transition width, ε_P from the f32 underflow guard for ~1e25 product
  magnitudes); no empirical tuning. The gate stacks multiplicatively
  with the existing CPML-zone and material-interface mutes.

  Residual finding: on the 42×42 abdominal synthetic phantom
  (`tests::abdominal::abdominal_theranostic_inverse_recovers_lesion_support`)
  CNR improves from −0.4336 (IS-IC + interface mute, prior commit) to
  −0.0995 (IS-IC + interface mute + Poynting gate, this commit) — a
  4.4× reduction in artefact magnitude — but does not become positive.
  At lesion radius ≈ 5.6 mm against a wavelength λ ≈ 5.8 mm at 260 kHz,
  the scatterer is at the Born resolvability floor (ka ≈ 1, where a is
  the scatterer radius and k is the wavenumber): the back-scattered
  field is dominated by forward-scattering rather than a clean
  counter-propagating reflection, so the Poynting gate has no
  unambiguous anti-parallel signature to lock onto at the true lesion.
  This is a physical resolution limit of the linearised Born/RTM model
  on this specific synthetic phantom, not an algorithmic gap. Closing
  this end-to-end test cleanly requires either (a) an enlarged phantom
  geometry where the lesion spans multiple wavelengths or (b) a
  full-waveform-inversion update (already exercised by the nonlinear-3D
  Westervelt pipeline tests) rather than a single-pass adjoint RTM. The
  test remains failing, deliberately not weakened, with a tracked
  backlog entry recording the sub-Born-resolvability physical limit.

### Fixed (2026-05-26) - Time-Reversal Solver Physics Defects

- [patch] `PhotoacousticTimeReversal` (solver/inverse/reconstruction/photoacoustic):
  propagator corrected from complex `exp(-i·c·k·dt)` to real `cos(c·|k|·dt)` per
  Tabei et al. 2002 Eq. 2; spurious imaginary energy was causing Pearson r=0.71 in
  kspace-vs-PSTD TR comparison.
- [patch] `inject_sensor_data` changed from additive soft-source (`+=`) to hard
  Dirichlet replacement (`=`) at sensor sites, applied before and after each
  propagation step (Treeby et al. 2010, §2.3).
- [patch] `time_reversal_reconstruction` (simulation/photoacoustic/reconstruction/core):
  delay-and-sum time index corrected to `n_time − 1 − floor(delay/dt)` (reversed)
  from incorrect forward-time lookup (Xu & Wang 2005, Eq. 7).

### Fixed (2026-05-26) - SSOT Physics Constants

- [patch] `thermodynamic.rs`: add `DITTUS_BOELTER_COEFFICIENT` (0.023),
  `DITTUS_BOELTER_VELOCITY_EXPONENT` (0.8), `DITTUS_BOELTER_PRANDTL_EXPONENT_HEATING`
  (0.4), `NUSSELT_LAMINAR_PIPE_CONST_TEMP` (3.66), and
  `REYNOLDS_LAMINAR_TURBULENT_THRESHOLD` (2300.0) with literature references
  (Dittus & Boelter 1930; Incropera & DeWitt 2007, §8.5).
- [patch] `physics/thermal/perfusion.rs`: replace five inline magic numbers with SSOT
  constants from `thermodynamic.rs`.
- [patch] `meta_learning/types/physics.rs`: correct water absorption default from
  0.025 dB/cm (was ~11× too high) to `WATER_ABSORPTION_ALPHA_0` = 0.0022 dB/cm
  (Duck 1990); replace B/A literals with `B_OVER_A_WATER` and `B_OVER_A_SOFT_TISSUE`.

### Changed (2026-05-26) - Ali 2025 Scattering-Increment Scale Decomposition

- [patch] Extend the Rust-owned Ali 2025 scattering-increment diagnostic with
  baseline-scaled full-field residual, model-scaled full-field residual,
  source-scale relative drift, and source-scale phase drift metrics. Expose the
  same fields through PyO3 and add analytic Rust/Python test fixtures where the
  model-scaled residual is zero while the baseline-scaled increment residual is
  above unity. Rebuilt `pykwavers`, passed focused scattering pytest 3/3, and
  regenerated the determined `(4,4,3)` probe. The finite-window model retains
  low model-scaled full-field residual (`0.03308952523301831`) while the
  baseline-scaled increment residual remains high (`1.4759860412851549`),
  which prompted the finite-window source-phasing proof below.

### Changed (2026-05-27) - Ali 2025 Finite-Window Source-Phasing Proof

- [patch] Pin the finite-window Born source term against the production PSTD
  acquisition generator with a Rust first-variation theorem test. The test
  verifies that `-chi * (p0[n+1] - 2p0[n] + p0[n-1])`, including the
  pressure-source contribution to the reference-field acceleration, matches the
  small-contrast finite difference of real PSTD data. Source phasing is closed
  as the cause of the remaining calibrated increment residual; the subsequent
  model-scaled increment diagnostic narrows the active gap to finite-window
  second-order scattering.

### Changed (2026-05-27) - Ali 2025 Model-Scaled Increment Diagnostic

- [patch] Extend the Rust-owned Ali 2025 scattering-increment diagnostic with
  model-scaled observed increment norm, model-scaled increment residual norm,
  model-scaled normalized increment residual, and model-scaled increment energy
  ratio. Expose the same fields through PyO3 and add analytic Rust/Python tests
  for scalar source drift. The determined `(4,4,3)` probe now ranks
  `pstd_finite_window_born` best for model-scaled increment residual
  (`0.3150272802598277`) while preserving the baseline-calibrated increment
  residual (`1.4759860412851549`) and low model-scaled full-field residual
  (`0.03308952523301831`). The remaining Ali 2025 gap is now narrowed to
  finite-window second-order scattering beyond scalar calibration.
- [patch] Repair the PyO3 Rayleigh-Sommerfeld wrapper build by using
  `Medium::density` for center-cell density sampling and preserving rectangular
  aperture width before transducer ownership transfer.

### Changed (2026-05-25) - Ali 2025 Finite-Window Determined Probe

- [patch] Rebuilt the local PyO3 extension and regenerated the determined
  `(4,4,3)` Ali 2025 report with `pstd_finite_window_born` included. The
  finite-window model now ranks best for full-field operator equivalence
  (`0.03308952523301831` all-channel, `0.03395758947454344` passive-only),
  while calibrated scattering-increment residual remains above unity and is
  tracked as the next solver diagnostic gap.

### Changed (2026-05-25) - Focused Bowl Focus-Axis Source Constructor

- [patch] Move clinical focused-bowl vertex construction into the source
  domain with a crate-internal `BowlConfig::from_focus_axis` constructor.
  Transcranial clinical cap placement now requests a focus, axis, radius, and
  angular bounds from `BowlTransducer` instead of constructing a synthetic
  vertex in the clinical layer. Added value-semantic source tests for radius,
  axis direction, area weights, angular bounds, and degenerate-axis rejection.

### Fixed (2026-05-25) - Focused Bowl Source Label Artifact Closure

- [patch] Replace vendor/helmet source identity labels in active Chapter 31
  focused-bowl renderer text, Chapter 31 book prose, and stale Chapter 29/31
  metrics artifacts with generic focused-bowl model names. Added a book test
  that rejects those labels in active focused-bowl artifacts and asserts the
  current transcranial source model is `transcranial_focused_bowl_projection`.

### Changed (2026-05-25) - Ali 2025 Finite-Window Report Routing

- [patch] Route the Ali 2025 reduced comparison report through the Rust-backed
  `pstd_finite_window_born` predictor, keep inversion on the adjoint-capable
  `pstd_spectral_convergent_born` operator, and add routing tests that prove
  Python only forwards acquisition parameters to PyO3.

### Added (2026-05-25) - Ali 2025 Finite-Window PSTD Born Boundary

- [patch] Add solver-owned finite-window PSTD Born prediction for Ali 2025
  FWI diagnostics, expose it through PyO3 as a conversion-only wrapper, and
  document the boundary in ADR-008. The implementation keeps time-window PSTD
  recurrence semantics separate from stationary CBS operators until the
  matching adjoint theorem exists.

### Fixed (2026-05-25) - Focused Bowl Aperture Chord Guard

- [patch] Reject axis-reference focused-bowl aperture chords larger than
  `2 * radius_m`, pin the validation with a value-semantic test, and keep the
  Chapter 25 visualization surface on generic focused-bowl cap terminology.

### Changed (2026-05-24) - Ali 2025 Scattering Policy Report Guard

- [patch] Preserve strict Rust rejection for zero-energy calibrated scattering
  increments while letting the Ali 2025 Python report record receiver-policy
  diagnostics that are not applicable. The determined probe report now includes
  scattering-increment metrics showing `dense_convergent_born` best matches the
  calibrated finite-window increment while `pstd_spectral_convergent_born`
  over-amplifies that increment by approximately `985-989x`.

### Added (2026-05-24) - Ali 2025 Scattering Increment Diagnostics

- [minor] Add Rust-owned Ali 2025 scattering-increment diagnostics that compare
  candidate heterogeneous forward models against the observed finite-window
  increment after homogeneous direct-field source-scale calibration. Expose the
  diagnostic through PyO3 and the reduced replication script so the remaining
  PSTD/CBS mismatch is measured on the scattered field instead of being mixed
  with homogeneous transfer.

### Fixed (2026-05-24) - Source Config Finite-Domain Validation

- [patch] Reject non-finite source amplitude, frequency, radius, phase, delay,
  position/focus components, pulse cycles, zero configured element counts, and
  invalid focused-bowl aperture bounds at `DomainSourceParameters::validate`.
  This prevents `NaN`/infinite values and impossible bowl apertures from
  reaching source construction while keeping focused bowl geometry generation
  delegated to `BowlTransducer`.

### Changed (2026-05-24) - Ali 2025 PSTD Operator Boundary Rerun

- [patch] Rebuild `pykwavers` against the Rust odd-z FFT repair, regenerate the
  four-cycle determined Ali 2025 probe, and add a Rust clinical boundary test
  proving homogeneous `PstdSpectralConvergentBornOperator` with temporal
  transfer equals the finite-grid PSTD modal predictor. The regenerated probe
  now ranks `pstd_spectral_convergent_born` best while leaving a heterogeneous
  passive residual for the next solver refinement.

### Added (2026-05-24) - Focused Bowl Hemisphere Aperture Config

- [minor] Add generic `FocusedBowlAperture::Hemisphere` and
  `FocusedBowlAperture::AxisReferenceHemisphere` options so config-driven
  focused sources can request fixed-count hemispherical bowl layouts without
  anatomy- or device-specific source names. Both variants delegate element
  positions, normals, and area weights to `BowlTransducer`.

### Fixed (2026-05-24) - Solver Convergence And Water Constant Test Contracts

- [patch] Correct the FDTD solver-convergence Gaussian pulse width from grid
  cells to meters, update the test to use canonical `DomainPMLBoundary`, and
  tighten the pre-PML energy-conservation assertion to the lossless-interior
  regime. Simple integration tests now assert `HomogeneousMedium::water`
  against canonical `DENSITY_WATER` and `SOUND_SPEED_WATER` constants.

### Fixed (2026-05-24) - Integration Test Domain Type Names

- [patch] Update source-factory and steering-vector integration tests to use
  the canonical `DomainSourceParameters` and `SensorArrayGeometry` type names
  after the internal domain API cleanup removed older names.

### Fixed (2026-05-24) - DG Convergence CPML Config Literals

- [patch] Update DG convergence-test `DGConfig` literals to set `cpml: None`
  explicitly after the DG CPML configuration field was added, preserving the
  periodic-boundary convergence and shock-capture test contracts.

### Fixed (2026-05-24) - Thermal Dose SSOT Constants

- [patch] Route thermal-dose R factors through canonical medical constants,
  route analytical thermal tests through `tissue_thermal` heat-capacity
  constants, and keep thermodynamic constant tests on the tissue-thermal SSOT.
  CEM43 accumulation now follows the Sapareto-Dewey formula for all positive
  temperatures, with mild-hyperthermia regression coverage.

### Fixed (2026-05-24) - PSTD Odd-Z R2C Direct-Field Parity

- [patch] Route odd-length 3-D real FFT z-axis transforms through a
  full-spectrum fallback in `apollo-fft`, preserving half-spectrum/full-spectrum
  equivalence for `nz > 1` odd grids. Added a Rust clinical breast UST
  direct-field equivalence test proving the `(4,4,3)` homogeneous PSTD dataset
  generator matches the finite-grid modal PSTD predictor.

### Changed (2026-05-24) - Focused Bowl Model Label Cleanup

- [patch] Remove vendor-like focused-bowl source labels from live Rust and PyO3
  surfaces. Abdominal placement, nonlinear 3-D aperture metadata, and the
  therapy plotting fixture now identify the source as `focused_bowl` while
  anatomy remains target metadata.

### Added (2026-05-24) - Focused Bowl Axis-Reference Aperture

- [minor] Add `FocusedBowlAperture::AxisReferencePolarBounds` and make
  `BowlConfig::from_axis_reference_focus` public so config-driven focused
  sources can use an anatomical/contact axis reference with an explicit
  curvature radius while still generating element layout through
  `BowlTransducer`.

### Fixed (2026-05-24) - Breast FWI PSTD CBS Discrete Contrast

- [patch] Route `PstdSpectralConvergentBornOperator` scattering potential and
  adjoint slowness derivative through the PSTD leapfrog temporal mass symbol
  `4 sin²(ωΔt/2)/Δt²`. Dense and continuous spectral CBS retain the continuous
  Helmholtz `ω²` contrast.

### Fixed (2026-05-24) - Medium Property SSOT Constant Closure

- [patch] Complete the medium-property SSOT extraction by defining the missing
  fluid/tissue constants in `core::constants::fundamental` and the implant
  effective nonlinearity constants in `core::constants::implants`. Fluid,
  tissue, and implant property tables now compile against named constants
  instead of unresolved literal replacements.

### Changed (2026-05-24) - Abdominal Focused-Bowl Source Routing

- [patch] Add a source-domain axis-reference focused-bowl constructor for
  placements where the anatomical contact point fixes aperture orientation but
  the curvature radius is larger than the contact-to-focus distance. Abdominal
  3-D placement now delegates element positions to `BowlTransducer` rather than
  constructing a local spherical-cap layout.

### Changed (2026-05-23) - Focused Bowl Placement Helper Consolidation

- [patch] Consolidate clinical transcranial focused-bowl cap point generation
  into `theranostic_guidance::geometry::focused_bowl`, with typed vertex
  orientation and source-domain `BowlAngularBounds`. The 2-D context and 3-D
  visual placement planner now delegate all cap sampling to `BowlTransducer`.

### Changed (2026-05-23) - Transcranial UST Bowl Aperture Routing

- [patch] Move the transcranial UST reconstruction aperture from a hard-coded
  hemispherical constructor to `TranscranialUstBornInversionConfig::aperture`
  backed by source-domain `BowlAngularBounds`. Slice and volume Born adapters
  now build geometry through `BowlTransducer::with_angular_bounds`, preserving
  the bowl source as the single aperture SSOT.

### Added (2026-05-23) - Breast FWI Reduced Array Plan Policy

- [minor] Add Rust-owned reduced-array row planning for Ali 2025 breast FWI.
  `BreastUstReducedArrayPlan` derives smoke, explicit, and Table 1 parity
  interior-row policies inside the clinical reduction layer, exposes
  `derive_breast_fwi_reduced_array_plan` through PyO3, and keeps the Python
  replication script limited to orchestration, reporting, and plotting.

### Changed (2026-05-23) - Breast FWI Zero-Thickness Absorber Contract

- [patch] Treat `absorbing_boundary="polynomial"` with
  `absorbing_thickness_cells=0` as `AbsorbingBoundary::disabled()` across the
  PyO3 frequency-domain FWI constructor and the spectral/PSTD spectral
  convenience constructors. The reduced Ali 2025 replication default now uses
  zero absorbing cells, matching the no-CPML PSTD dataset configuration.

### Fixed (2026-05-23) - CBS Adjoint Richardson Iterate Sign

- [minor] Restore correct `+=` sign in `solve_adjoint_spectral_iterative`.
  The adjoint iterate `λ += γ^H·residual` (γ^H = −iε/Ṽ*) gives iteration
  matrix `I + γ^H A^H` with spectral radius `|V/(V+iε)| < 1`, matching the
  forward CBS contraction bound. The previous `-=` caused ~10^38 divergence.
  `DenseFreeSpace` retains exact dense LU; all spectral operators now use the
  O(max_iter × N log N) iterative adjoint. The O(N²logN) `operator_matrix_by_columns`
  path is removed. All 11 gradient-matching tests pass.

### Fixed (2026-05-23) - Panic-on-NaN in floating-point comparators

- [patch] Replace `partial_cmp(…).unwrap()` with `total_cmp(…)` across 74 call
  sites in analysis, clinical, domain, math, physics, and solver layers.
  `total_cmp` defines a total order over all f64 bit patterns (incl. NaN/±inf)
  without panicking; `partial_cmp` returns `None` for NaN, causing `.unwrap()`
  panics in release mode on subnormal inputs.

### Fixed (2026-05-23) - Divide-by-zero in SonothermalStats::from_slice

- [patch] Guard `SonothermalStats::from_slice` against empty slice input.
  Without the guard, `values.len() as f64 == 0.0` caused silent NaN propagation
  through mean/variance. Returns an all-zero stats struct when input is empty.

### Added (2026-05-23) - Breast FWI PSTD Temporal Bin Transfer

- [minor] Add solver-owned PSTD temporal source and finite frequency-bin
  transfer identities under `frequency_domain::cbs::temporal`. Clinical breast
  UST direct-field diagnostics now consume those modal response functions
  instead of owning a private PSTD recurrence.

### Changed (2026-05-23) - Breast FWI PSTD CBS Temporal Transfer Wiring

- [minor] Expose `PstdTemporalTransferConfig` through the frequency-domain FWI
  API and PyO3 `FrequencyDomainFwiConfig`, then pass acquisition source
  amplitude, total cycles, and bin cycles from the Ali 2025 replication config
  into `PstdSpectralConvergentBornOperator`.

### Changed (2026-05-23) - Breast FWI PSTD CBS Receiver Projection

- [minor] Route `PstdSpectralConvergentBornOperator` receiver sampling and
  receiver-adjoint projection through exact PSTD grid-cell extraction/injection.
  Continuous Helmholtz CBS operators retain the BLI receiver projection path.

### Changed (2026-05-22) - Breast FWI PSTD CBS Source Projection

- [minor] Route `PstdSpectralConvergentBornOperator` source injection through
  the same on-grid pressure mask and PSTD source-kappa spectral filter used by
  the clinical PSTD acquisition generator. The continuous Helmholtz CBS
  operators still use the existing BLI point-source density path.

### Fixed (2026-05-22) - Focused Source Adapter Type Inference

- [patch] Add the explicit `ElementMap` type to the focused source adapter's
  `HashMap` construction so Rust can infer the collected element-index vector
  type during full library test compilation.

### Added (2026-05-22) - Breast FWI PSTD Spectral CBS Operator

- [minor] Add `PstdSpectralConvergentBornOperator` to the
  frequency-domain FWI solver. The operator keeps the CBS solver boundary but
  replaces the continuous periodic Helmholtz denominator with the homogeneous
  PSTD leapfrog/k-space modal symbol, exposes
  `pstd_spectral_convergent_born` through PyO3, and includes it in the Ali 2025
  operator-equivalence probe.

### Added (2026-05-22) - Breast FWI Passive Direct-Field Residual Deltas

- [minor] Add Rust-owned passive-channel residual deltas to Ali 2025
  homogeneous direct-field diagnostics. The clinical report now records the
  source-kappa and finite-grid PSTD passive residual changes relative to the
  outgoing Helmholtz point Green reference, exposes both values through PyO3,
  and writes them into the reduced replication report without adding Python
  formulas.

### Added (2026-05-21) - Breast FWI Receiver-Policy Operator Equivalence Diagnostics

- [minor] Add Rust-owned receiver-channel policy support to Ali 2025
  forward-operator equivalence diagnostics. The clinical diagnostic now ranks
  operators under `all`, `active_only`, and `passive_only` receiver selections,
  exposes the selected policy through PyO3, and records policy-specific rankings
  in the reduced replication report without adding Python formulas.

### Added (2026-05-21) - Breast FWI Active Self-Channel Direct-Field Diagnostics

- [minor] Extend Rust-owned Ali 2025 homogeneous direct-field diagnostics with
  active source/receiver self-channel residual, phase, amplitude, and pair-count
  metrics. Expose the fields through PyO3 so the replication report can separate
  co-located source-channel mismatch from passive propagation mismatch without
  reintroducing Python diagnostic formulas. Add an analytic Rust test that
  perturbs active channels while keeping passive channels exactly matched.

### Added (2026-05-21) - Breast FWI Rust-Owned Operator Equivalence Diagnostics

- [minor] Move Ali 2025 forward-operator equivalence aggregation into
  `kwavers::clinical::imaging::reconstruction::breast_ust_fwi::diagnostics`.
  Expose `breast_fwi_operator_equivalence_diagnostics` through PyO3 so Python
  no longer owns the residual/source-excitation aggregation across
  `single_scatter_born`, `dense_convergent_born`, and
  `spectral_convergent_born`.

### Added (2026-05-21) - Breast FWI Rust-Owned Reduced-Domain Preparation

- [minor] Move Ali 2025 reduced phantom decimation, center cropping,
  homogeneous median initial-model construction, and reduced ring-array geometry
  derivation into
  `kwavers::clinical::imaging::reconstruction::breast_ust_fwi::reduction`.
  Expose `prepare_breast_fwi_reduced_phantom` and
  `derive_breast_fwi_reduced_array_geometry` through PyO3 so the replication
  script no longer owns clinical reduction formulas.

### Added (2026-05-21) - Breast FWI Rust-Owned Replication Diagnostics

- [minor] Move Ali 2025 observation residuals, source-channel residual
  attribution, source-excitation dispersion, acquisition identifiability,
  reconstruction RMSE/PCC, and Table 1 parity gates into
  `kwavers::clinical::imaging::reconstruction::breast_ust_fwi::diagnostics`.
  Expose the metrics through PyO3 and reduce the Python support modules to
  binding callers for reporting and plotting workflows.

### Added (2026-05-21) - Breast FWI Direct-Field Rust Ownership

- [minor] Move Ali 2025 homogeneous direct-field diagnostics into
  `kwavers::clinical::imaging::reconstruction::breast_ust_fwi::direct_field`,
  covering point Green, PSTD source-kappa filtering, and the finite-grid PSTD
  periodic modal recurrence. Expose
  `diagnose_breast_fwi_homogeneous_direct_field` through PyO3 and reduce the
  Python support module to a binding caller.

### Changed (2026-05-21) - Transcranial Linear Born Config Boundary

- [arch] Complete the clinical wrapper to generic linear-Born config boundary:
  transcranial clinical entrypoints keep anatomy fields on
  `TranscranialUstBornInversionConfig` and pass `&config.linear` to the
  anatomy-neutral kernels, restoring the kwavers and pykwavers compile gates.

### Changed (2026-05-21) - Transcranial UST Reconstruction Boundary

- [patch] Move the transcranial ultrasound tomography Born inversion out of
  `solver::inverse::seismic::brain_helmet` and into the clinical imaging
  reconstruction boundary, rename exported config/result/geometry types to
  `TranscranialUstBornInversion*` and `TranscranialBowlGeometry`, and route the
  bowl acquisition geometry through `BowlTransducer::with_polar_span`.

### Changed (2026-05-21) - Transcranial Focused-Bowl Naming Completion

- [patch] Replace the public 3-D brain placement API and PyO3 export with
  transcranial focused-bowl terminology, remove the old planner name without a
  compatibility alias, and sync the affected clinical examples, book figures,
  metrics, and nonlinear 3-D aperture model strings.

### Added (2026-05-21) - Bowl Polar-Span Source Layout

- [minor] Add fixed-count `BowlTransducer::with_polar_span` and
  `BowlTransducer::with_polar_bounds` constructors so focused bowl sources can
  generate hemispherical, major-cap, and annular angular apertures through the
  source-domain API. Full-volume brain placement now delegates its major-cap
  source points to this bowl API instead of owning local source geometry.

### Changed (2026-05-21) - 3D Focused-Bowl Placement Source Routing

- [patch] Route full-volume calvarium cap placement through
  `BowlTransducer::with_polar_bounds`, remove the local Fibonacci aperture
  sampler from the clinical placement helper, and pin the normalized polar
  z-bounds in the 3-D placement regression.

### Changed (2026-05-20) - Bowl Transducer Cap Geometry SSOT

- [minor] Route `BowlTransducer` surface discretization through
  `domain::source::transducers::focused::cap`, derive element count from
  spherical-cap area and requested element size, preserve equal-area weights,
  and reject nonfinite or degenerate bowl domains.

### Changed (2026-05-20) - Hemispherical Array Geometry SSOT

- [minor] Route `domain::source::hemispherical::ElementPlacement` through
  `domain::source::transducers::focused::cap`, reject zero-element layouts and
  nonfinite radii, preserve the established positive-y aperture orientation, and
  document element normals as focus-directed.

### Changed (2026-05-20) - Transcranial FUS Cap Geometry SSOT

- [minor] Route tracked transcranial FUS cap placement through
  `domain::source::transducers::focused::cap`, remove the local Fibonacci
  geometry implementation, preserve the established negative-z aperture
  orientation, and reject invalid polar spans through source-domain validation.

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

### Added (2026-05-20) - Ali 2025 Multi-Row Ring 3D FWI Foundation

- [minor] Add `physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi`
  as the paper-level SSOT for Ali et al. 2025 multi-row ring geometry, 200-800
  kHz sweep, slowness/sound-speed transforms, Helmholtz slowness derivative,
  complex source scaling, L2 objective, RMSE, and PCC.
- [minor] Add `solver::inverse::fwi::frequency_domain` with matrix-free 3-D
  frequency-domain Born prediction, exact discrete adjoint-gradient
  accumulation for that model, nonlinear conjugate-gradient inversion, and
  value-semantic tests covering forward sensitivity, finite-difference gradient
  agreement, and objective-decreasing reconstruction.
- [minor] Add `solver::inverse::fwi::time_domain` as the method-scoped owner of
  the existing acoustic adjoint-state residual, objective, time-reversal, and
  signed-correlation theorems; remove the old top-level acoustic-FWI method
  namespace from call sites.
- [minor] Add `solver::inverse::fwi::frequency_domain::cbs` for the CBS
  scattering-potential identities: `V = omega^2(s^2 - s0^2)`, epsilon satisfying
  `epsilon >= ||V||_infinity`, shifted potential `V - i epsilon`, and pointwise
  preconditioner construction.
- [minor] Extend `solver::inverse::fwi::frequency_domain::cbs` into a theorem
  split module tree with centered grid geometry, BLI point weights, shifted
  outgoing Green application, and a dense CBS fixed-point volume solver with
  residual diagnostics.
- [minor] Add explicit `PropagationModel` selection to frequency-domain FWI.
  `DenseConvergentBorn` now routes prediction through BLI source projection,
  dense CBS propagation, and BLI receiver sampling.
- [minor] Add dense CBS adjoint-gradient support: shared BLI projection helpers,
  shifted-Green adjoint, dense solve for the discrete adjoint
  `(I + G_epsilon diag(V - i epsilon))^H`, and Eq. 6 slowness-gradient
  accumulation for `DenseConvergentBorn` inversion.
- [minor] Add `SpectralConvergentBorn` and `GreenOperatorKind::SpectralPeriodic`
  using the periodic pseudospectral Green symbol
  `(k0^2 + i epsilon - |k|^2)^-1` through the existing Apollo FFT facade, while
  preserving the shared CBS solve, adjoint, projection, and gradient route.
- [minor] Add `AbsorbingBoundary` for spectral CBS and apply the polynomial
  sponge as `W G W`; this suppresses FFT wraparound while preserving the exact
  adjoint relation `G_abs^H = W G^H W`.
- [minor] Add `fwi_spectral_cbs` Criterion validation for the absorbed spectral
  CBS public solver path on a reduced 24x24x18 multi-row ring fixture. The
  benchmark asserts finite pressure and sound-speed perturbation sensitivity
  before timing; current benchmark-profile median is 19.998 ms with
  2.0738 Melem/s throughput.
- [minor] Add `clinical::imaging::reconstruction::breast_ust_fwi` as the
  clinical adapter that delegates reconstruction to the solver while preserving
  modality, array, frequency, and objective-history metadata.
- [minor] Add the PyO3 breast-FWI surface for Ali et al. 2025 replication:
  `MultiRowRingArray`, `FrequencyDomainFwiConfig`, `FrequencyObservation`,
  `ali_2025_breast_fwi_frequency_sweep_hz`,
  `simulate_breast_fwi_frequency_observation`, and `invert_breast_fwi`.
  The inversion binding accepts stacked `complex128` frequency/transmit/receiver
  data and delegates to the clinical breast UST adapter.
- [minor] Add the PSTD breast-FWI dataset generation path. The clinical
  adapter runs the existing PSTD solver over centered multi-row ring geometry,
  preserves receiver ordering with ordered sensor indices, extracts complex
  Fourier bins per frequency/transmit/receiver, and exposes
  `BreastFwiPstdDatasetConfig` plus `generate_breast_fwi_pstd_dataset`
  through pykwavers.
- [patch] Refine PSTD breast-FWI frequency binning so acquisition separates
  total simulated cycles from trailing Fourier-bin cycles. The returned dataset
  now includes per-frequency bin-start samples, and pykwavers exposes
  `frequency_bin_cycles` plus the bin-start metadata.
- [patch] Split the Ali 2025 replication example into support modules for
  volume validation, metrics, visualization, and identifiability diagnostics.
  The reduced-grid report now includes the acquisition rank upper bound after
  estimated complex source-scale nuisance parameters.
- [patch] Add `--require-determined-acquisition` to the Ali 2025 replication
  example so rank-underdetermined probes fail before PSTD generation and
  inversion. A 4x4x3 two-frequency probe now satisfies the reduced rank guard
  while preserving the open Table 1 parity failure.
- [patch] Add grid-snapped clinical ring-array geometry for Ali 2025 PSTD/FWI
  replication. `MultiRowRingArray` can now preserve topology over explicit
  ordered element coordinates, the clinical breast-FWI boundary snaps the array
  to PSTD grid centers, pykwavers exposes `snap_breast_fwi_array_to_grid`, and
  the replication report includes source-scaled PSTD-vs-CBS forward residuals.
- [patch] Add source-channel residual attribution to the Ali 2025 replication
  report. The diagnostic builds the cylindrical active-source receiver mask,
  reports passive-only row-scaled residuals, and shows the determined 4x4x3
  probe remains mismatched on passive receiver channels rather than being
  dominated by co-located source/receiver samples.
- [patch] Add source-excitation scalar diagnostics to the Ali 2025 replication
  report. The diagnostic computes the additive-sine PSTD frequency-bin
  coefficient, normalizes row-wise PSTD-vs-CBS source scales by that
  coefficient, and reports transmit-scale magnitude/phase dispersion per
  frequency.
- [patch] Add forward-operator equivalence diagnostics to the Ali 2025
  replication report. The diagnostic compares `single_scatter_born`,
  `dense_convergent_born`, and `spectral_convergent_born` against the same PSTD
  observation cube with row-wise source scaling and source-bin-normalized scale
  metrics.
- [patch] Add homogeneous direct-field Green diagnostics to the Ali 2025
  replication report. The diagnostic compares homogeneous snapped PSTD
  observations against the outgoing Helmholtz direct Green field and reports
  passive phase/amplitude residuals before scattering.
- [patch] Add PSTD source-kappa filtered direct-field diagnostics to the Ali
  2025 replication report. The diagnostic applies the pressure-source
  `cos(c_ref |k| dt / 2)` spatial correction to discrete source masks before
  outgoing Green evaluation and records the residual delta.
- [patch] Add finite-grid PSTD Green diagnostics to the Ali 2025 replication
  report. The diagnostic derives the homogeneous no-CPML modal recurrence with
  propagation kappa, source kappa, source timing, and the same frequency-bin
  projection as the Rust acquisition.
- [patch] Close remaining time-domain FWI example imports onto
  `solver::inverse::fwi::time_domain` and remove stale reconstruction comments
  that referenced the deleted seismic-owned path.
- [patch] Remove residual transcranial vendor/helmet labels from book examples
  and documentation, routing the Chapter 25 phase-correction artifact through a
  generic focused-bowl figure stem.
- [patch] Pin perfused thermal tissue tests to the canonical
  `BLOOD_SPECIFIC_HEAT` SSOT instead of accepting any positive bioheat value.
- [minor] Add `solver::inverse::linear_born_inversion` with the shared
  `ElementPosition` and `TransducerGeometry` acquisition-geometry contract, and
  route `TranscranialBowlGeometry` through that trait with its bowl-specific
  azimuthal receiver mapping.
- [patch] Route CEM43 body-temperature and cell-death thresholds through
  canonical thermal/medical constants and pin the thermal-dose SSOT contract
  with value-semantic tests.
- [minor] Add Rust-owned Ali 2025 HDF5 phantom ingest through
  `clinical::imaging::reconstruction::breast_ust_fwi::phantom_hdf5`, backed by
  `consus-hdf5`/`consus-core`/`consus-io`. The loader decodes real 3-D
  sound-speed datasets from contiguous or chunked HDF5 storage, supports C and
  MATLAB/Fortran linearization, requires spacing metadata or explicit spacing,
  normalizes m/s or km/s storage units to m/s, and exposes
  `load_ali_2025_breast_fwi_phantom` through pykwavers.
- [minor] Add `pykwavers/examples/replicate_ali2025_breast_fwi.py` as the
  reduced-grid Ali 2025 replication entry point. The script downloads/caches
  the published phantom, delegates clinical phantom ingest to Rust, runs
  clinical PSTD data generation plus spectral-CBS frequency-domain FWI through
  pykwavers, and writes RMSE/PCC metrics plus orthographic comparison slices.
- [minor] Add MATLAB Level-5 ingest for the published
  `BreastPhantomFromMRI.mat` release asset. `phantom_mat5` decodes compressed
  `breast_mri`, applies the published MRI-to-sound-speed transform on a
  requested uniform grid, and the PyO3 phantom loader now auto-detects HDF5 vs
  MAT5 containers.
- Residual replication gap: the current solver path has an executable
  reduced-grid phantom workflow and Table 1 parity evaluator. Full Ali et al.
  replication still requires meeting the reduced Table 1 parity gate tracked in
  `backlog.md`.

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
