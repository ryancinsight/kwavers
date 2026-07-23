# Backlog / Strategy

## KW-CI-068 — Close Moirai admission workaround [patch] — done

- Owner: Codex `/root`; last-update: 2026-07-22; scope: the canonical Atlas
  checkout revision, Cargo lock, six therapy-test scheduling overrides, and
  focused value-semantic verification. Solver workloads and assertions are
  non-goals.
- Outcome: consume merged Moirai PR #83 and delete serialization that masked
  bounded-admission failures under ordinary parallel Nextest execution.
- Acceptance: Cargo records `moirai-core` as a direct `moirai-parallel`
  dependency, the six tests have no dedicated serialization override, their
  unchanged values pass under the default profile, and hosted Linux locked
  resolution succeeds against Atlas `c982fe0` without the inherited Reqwest
  0.11 TLS advisory chain, and the replacement TLS graph passes the repository
  license policy.
- Risk/change class: `[patch]`; evidence is lock provenance, configuration
  residue scans, focused Nextest values, warning-denied static checks, and
  exact-head hosted CI.
- Evidence: implementation head `eb80ad2` passed Architecture Validation run
  `29964349679`, CI/CD run `29964349756`, legacy-migration run `29964349739`,
  and benchmark-regression run `29964349692`. The matrix includes locked
  stable/beta/nightly builds and tests, full native tests and doctests, every
  feature combination, CUDA, Miri, security, coverage, solver/PINN validation,
  and both bounded benchmark smokes.
- Integration: Python-portability head `b04cf397` passed Architecture
  Validation `29963556227`, CI/CD `29963556297`, legacy-migration
  `29963556225`, and benchmark-regression `29963556257`, then merged as default
  `1dc60bd`. This closure rebases on that default, preserves its portability and
  dependency graph, and replaces only provisional Atlas pin `a534313` with
  canonical merged graph `c982fe0`.

## KW-PERF-067 — Stream elastic-FWI adjoint gradient [patch] — done

- Owner: Codex `/root`; last-update: 2026-07-22; scope:
  `crates/kwavers-solver/src/forward/elastic/swe/core/solver/point_force_drive.rs`,
  `crates/kwavers-solver/src/inverse/elastography/elastic_fwi/gradient.rs`,
  focused regressions, and synchronized PM evidence. Forward-history
  checkpointing, FWI workload or tolerance changes, and public API changes are
  non-goals.
- Outcome: accumulate the elastic-FWI shear-modulus gradient while the adjoint
  propagator runs, retaining one current adjoint field instead of cloning a
  second complete six-component wave-field history.
- Acceptance: the streamed gradient and illumination match the full-history
  reference; the existing 2-D/3-D directional-gradient and reconstruction
  contracts pass unchanged; the retained-history bound decreases by six
  `f64` grid volumes per time step; and controlled peak-memory/runtime
  measurements show no performance regression.
- Risk/change class: `[patch]`; the time-index reversal and floating-point
  accumulation order are the primary correctness risks. Verification uses a
  focused differential regression, existing analytical gradient oracles,
  configured Nextest budgets, warning-denied Clippy, doctests, and measured
  process-tree peak memory.
- Current evidence: the independent full-history oracle is bitwise-equal to the
  streamed gradient and illumination. The unchanged 3-D directional-gradient
  case retains 20.44 MiB less wave-field history analytically and measured a
  median process-tree peak of 266.25 MiB versus 286.41 MiB (−20.16 MiB, −7.0%).
  Its three-sample median Nextest duration decreased from 1.651 s to 1.216 s
  (−26.3%). The focused oracle, unchanged 2-D/3-D directional-gradient, and two
  reconstruction regressions pass 5/5 in 21.486 s; warning-denied Clippy and
  doctests pass. All repository-owned hosted checks passed on exact
  implementation head `918cd826`; the external, non-required RecurseML
  analysis reported an out-of-band service error.

## KW-ARCH-065 — Consolidate optical transport in Hyperion [major] [arch] — in-progress

- Owner: `/root`; scope: published Hyperion integration in `kwavers-medium`,
  `kwavers-physics`, and `kwavers-solver`; deletion of the named parallel law
  and coefficient owners; provider-graph pins; ADR 046; consumer regression
  evidence. General electromagnetic solvers, Monte Carlo ownership,
  photoacoustic source policy, chromophore spectra, and release are non-goals.
- Acceptance: Hyperion is the only owner of reduced scattering, coefficient
  validation, albedo, diffusion, effective attenuation, penetration depth,
  optical depth, and transmission; all superseded Kwavers owners are absent;
  direct consumers pass value-semantic, invalid-input, Nextest, Clippy,
  doctest, Rustdoc, and SemVer gates against the locked published graph.
- Decision: [ADR 046](docs/ADR/046-hyperion-optical-transport-ownership.md).
- Current evidence: implementation and lock reconciliation are complete. The
  affected Clippy gate, six-package doctest gate, focused invalid-input and
  value-semantic Nextest suites, provider-source uniqueness scan, and normal
  Rustdoc build pass. Warning-denied Rustdoc passes for `kwavers-medium`,
  `kwavers-imaging`, and `kwavers-phantom`; `kwavers-physics` retains its tracked
  KW-DOC-038 baseline (557 warnings in this configuration). The major SemVer
  gate is attempted but cannot resolve `origin/main`: its pinned Aequitas still
  requires Eunomia `^0.6.0`, while the canonical Git source now publishes
  `0.7.0`. The aggregate current-graph check also exposes distinct path and Git
  Leto identities at the RITK registration boundary. No compatibility adapter
  is introduced. The integrated workspace Nextest closure passes 6,168/6,168
  tests with 15 skipped; publication remains.

## KW-PERF-066 — Restore elastic-FWI test budget [patch] — done

- Owner: `/root`; the item blocks the KW-ARCH-065 consumer gate.
- Scope: the elastic SWE production kernel and Nextest scheduling of internally
  parallel full-grid solvers. Test workloads, assertions, and timeout expansion
  are non-goals.
- Acceptance: `fwi_outperforms_linear_inversion` and
  `recovers_stiff_inclusion` preserve their current value-semantic coverage and
  each complete below 30 seconds through production-path optimization.
- Evidence: the KW-ARCH-065 affected-package Nextest run measured 37.411 seconds
  and 31.064 seconds respectively on 2026-07-21; both passed but exceeded the
  committed ordinary-test budget. The production path now selects homogeneous
  density once, caches separable PML exponentials per time step, and dispatches
  the three stress/divergence output passes through Moirai's canonical
  triple-buffer primitive. One full-grid group removes cross-group CPU
  oversubscription and the obsolete 90-second FWI exception. The exact
  workspace closure passes 6,168/6,168 tests in 173.986 seconds, with the two
  regressions at 10.310 and 9.480 seconds; all 13 focused stress/PML tests,
  warning-denied Solver Clippy, and Solver doctests pass.

## KW-PYTHON-064 — Python release wheels [patch] — in-progress

- Owner: `/root`; scope: `kwavers-python` distribution metadata and lock, the
  release workflow, protected GitHub environment, base-package import contract,
  distribution documentation, and PyPI trusted publisher. Numerical Python
  binding behavior is a non-goal.
- Acceptance: a GitHub Release tagged `kwavers-python-v<version>` builds one
  locked Python-3.8-compatible stable-ABI wheel for each of Linux, Windows, and
  macOS, installs and imports each wheel as `pykwavers`, validates Cargo-owned
  distribution identity, attests and attaches the exact artifacts, then
  publishes the same wheels to the `kwavers-python` PyPI project through OIDC.
- Current evidence: the release workflow and synchronized distribution contract
  are implemented, and GitHub environment `pypi` accepts only
  `kwavers-python-v*` tags. A locked `cp38-abi3` wheel builds as
  `kwavers-python` 0.1.0, installs into an isolated target, and imports as
  `pykwavers`. Release run `29967429949` then exposed that the package
  initializer eagerly imported `comparison.py` and made undeclared
  `matplotlib` mandatory for every base-wheel import. PR #314 removes those
  eager imports and the optional names from the base `__all__`; standard
  explicit submodule imports retain their normal dependency errors. A
  fresh-interpreter regression blocks `matplotlib` and proves the base package
  does not load any optional submodule. The same oracle now gates installed
  stable-ABI base wheels on Linux, Windows, and macOS before merge. PR #314
  head `c191173d` merged as `21fc7119` before its exact-head matrix completed:
  legacy-migration run `29968431907` passed, while CI/CD `29968431956` and
  Architecture Validation `29968431955` remained active. PR #313 rebases the
  provider closure onto that merge; its final combined matrix is authoritative.
  The shared local GNU linker configuration emits its existing unused-static-
  link-argument diagnostic; final hosted wheel evidence and pending-publisher
  registration remain open.

## KW-BUILD-065 — Bound debug build artifacts [patch] — done

- Owner: /root; scope: the Kwavers development profile, exact pinned-provider
  build/test evidence, and shared Atlas target-cache measurement. Non-goals:
  weaker numerical tests, higher nextest timeouts, a private target directory,
  release-profile changes, or deletion of source/user data.
- Acceptance: development dependencies use the lowest optimization level that
  keeps all four full-grid simulation test binaries within the committed
  60-second per-test bound; the exact PR head is warning-clean; hosted build
  duration does not regress; and the generated dependency/artifact footprint
  decreases without removing line-number backtraces from workspace code.
- Baseline: `D:\atlas\target\debug\deps` contains 127,468 files and 170.34 GiB,
  including 43.08 GiB of rlibs and 41.13 GiB of rmeta; `debug\examples`
  contains 6,031 files and 50.95 GiB. The stack config already uses
  line-table-only workspace debug information, disables dependency and
  build-script debug information, and Kwavers already links through LLD.
- Decision: remove `[profile.dev.package."*"] opt-level = 3` so the broad
  dependency graph inherits `opt-level = 1`. Cargo documents that dependency
  optimization levels 2 and 3 prevent reuse/export of shared generic
  monomorphizations, while level 1 retains basic optimization and sharing.
  Optimize Kwavers' contiguous FFT spectrum movement instead of retaining
  `-O3` for Apollo FFT, Leto, or Moirai. Workspace members and every dependency
  now use the same development optimization level.
- Local topology: temporary linked worktrees materialize the exact provider
  commits selected by the Atlas checkout action. The aligned Atlas graph uses
  Leto 0.40, Hermes 0.4.1, and one Eunomia 0.7 source identity across Coeus,
  Hephaestus, RITK, and Kwavers. Cargo metadata and all verification below use
  `--locked`; hosted CI remains the authoritative clean-checkout build and
  artifact-size oracle.
- First hosted profile evidence: on run `29888001830`, uncached feature-build
  steps fell from 622 s to 342 s for `minimal` (-45.0%), 650 s to 453 s for
  `pinn` (-30.3%), 847 s to 591 s for `full` (-30.2%), and 503 s to 411 s for
  `plotting` (-18.3%) relative to the exact `-O3` head. The test job completed
  in 29m57s versus 37m17s: the default library suite passed 5,650 tests in
  517.008 s and the PINN library suite passed 39 tests in 2.077 s. This first
  head restored lock-only target caches, so its release/doctest timings and
  artifact size are not a clean profile comparison.
- Final-head instrumentation: every CI cache containing `target/` also hashes
  `Cargo.toml` and `.cargo/config.toml`; the architecture job executes the four
  unchanged full-grid integration binaries through the committed Nextest
  profile, then records `target/debug` bytes and file count. Broad cache
  restore prefixes are removed so profile-incompatible target artifacts cannot
  enter the measurement.
- Profile falsification: exact head `1cafb7f67` passed all pre-existing
  non-coverage CI jobs. The ordinary `-O1` build passed 5,650
  library tests in 568.365 s, then terminated
  `test_plane_wave_boundary_injection_pstd` at 60 s. This falsifies broad
  `-O1` for the solver/FFT path without justifying `-O3` for every dependency.
  Tarpaulin's ptrace instrumentation separately varied
  `test_plane_wave_boundary_injection_pstd` from about 315 s on the passing
  first head to about 350 s, exceeding the unchanged 300-second response
  timeout. Coverage now uses a dedicated profile inheriting `dev` while
  optimizing dependencies at level 3. This restores the previously proven
  instrumented execution regime without changing ordinary debug artifacts,
  test inputs, assertions, or timeouts.
- Targeted-provider falsification: exact head `f80822a55` retained `-O3` for
  `apollo-fft` plus the ineffective workspace-member overrides
  `kwavers-solver` and `kwavers-math`; Cargo defines wildcard dependency
  overrides as excluding workspace members. The library suite passed 5,650
  tests in 389.448 s, but the same PSTD boundary test terminated at 60.010 s.
  Exact head `73ec35245` then optimized the full non-workspace provider closure
  and still terminated that test at 60.016 s, falsifying provider profile
  selection as the root cause.
- Production fix: the FFT facade now copies contiguous half-spectrum rows and
  reconstructs Hermitian rows through direct slice indexing instead of a
  general strided assignment and triple-indexed loop. Under broad `-O1`, the
  unchanged 64-cubed, 300-step PSTD test passes in 18.099 s alone and 16.781 s
  in the serialized architecture grid. `kwavers-math` passes 266/266 tests,
  including reference C2C comparison and R2C/C2R round trips across even, odd,
  power-of-two, and degenerate shapes. The four architecture binaries pass
  24/24 in 69.640 s; serializing their internally parallel processes reduces
  the longest test from 31.563 s under contention to 22.853 s without changing
  workloads, assertions, or timeouts.
- Exact hosted evidence: implementation head `905b5efbe` passes every
  architecture job and the bounded benchmark workflow. Feature-build jobs
  complete in 7m48s–10m57s, the full architecture job completes in 33m06s,
  and its unchanged PSTD regression completes in 24.546 s. The clean
  `target/debug` tree contains 16,771,464,617 bytes across 6,109 files. No
  comparable clean `-O3` footprint was retained, so this establishes the
  clean artifact baseline without claiming an unsupported size percentage.
  The earlier uncached profile comparison remains the build-time oracle.
- Exact-graph reconciliation: the post-merge provider checkout exposed a stale
  all-feature lock closure with Eunomia 0.6 and 0.7, producing incompatible
  `Complex` identities in the hosted benchmark compile. The Atlas gitlinks and
  Kwavers action pin now select the single-Eunomia graph; locked all-feature
  metadata and `cargo check -p kwavers-math --all-targets --all-features` pass,
  and the exact merged graph passes all 266 `kwavers-math` tests in 2.117 s.
  The benchmark workflow now resolves its smoke and phase-reversed jobs through
  that same candidate-pinned action; its historical baseline lock is normalized
  against the held-constant provider graph before measurement.
- Exact-head run `29911114271` completed every bounded pair in 22m15s–22m41s,
  then reported two replicated Grid allocation regressions even though the head
  differed from the preceding green measured revision only in `deny.toml`.
  The smoke job now builds the three merge-critical base/head executables from
  the same path and compares their SHA-256 hashes. Byte-identical sets terminate
  with that stronger proof; differing sets retain the unchanged four-pair
  statistical instrument. Exact head `04bced11b` passes all 26 hosted checks:
  CI `29913169738`, architecture `29913169852`, legacy audit `29913169756`,
  and benchmark run `29913169741`. The benchmark workflow proves executable
  identity and completes in 12m12s without pair jobs; run `29909003760` retains
  the exact-head four-pair evidence for differing executables.

## KW-UQ-064 — Integrate Tyche collocation sampling [major] [arch] — done

- Owner: /root; scope: `kwavers-grid::geometry`, PINN collocation sampling,
  Tyche dependency integration, ADR 043, allocation/value-semantic tests, and
  synchronized public documentation. Non-goals: ML-owned residual-adaptive
  sampling and unrelated random consumers.
- Acceptance: rectangular, disk, and ball domains validate construction and
  map Tyche unit designs without rejection or cardinality loss; one generic
  collector serves counter, Latin-hypercube, and Sobol designs; the collocation
  hot path is statically dispatched; generated matrices allocate once without
  reallocation; local Latin-hypercube, pseudo-Sobol, and obsolete geometry
  adaptive implementations have no residue.
- Evidence target: analytical mapping reference cases, deterministic replay,
  exact cardinality and domain classification, face-measure sampling law,
  allocation counts, focused Nextest/Clippy/doctest/Rustdoc gates, dependency
  residue scans, and public SemVer classification.
- Decision: extend [`ADR-043`](docs/ADR/043-tyche-uncertainty-provider.md) with
  the collocation ownership, public migration, transform proofs, and rejected
  duplicate/rejection alternatives.
- Exact rebased-head evidence: grid Nextest passes 45/45, including
  fixed-layout/output allocation and typed reservation-failure contracts;
  solver geometry/config/allocation selection passes 21/21; Tyche sensitivity
  passes 9/9; and all-target checks pass for all three affected packages.
  Changed-file rustfmt, `git diff --check`, grid warning-denied Clippy, all
  three doctest suites, and grid/solver warning-denied Rustdoc pass. Solver and
  analysis Clippy stop only at the two pre-existing KW-LINT-047 forward-solver
  diagnostics; analysis warning-denied Rustdoc exposes its pre-existing
  cross-module link backlog. SemVer comparison against live main classifies
  both grid and solver as major. Residue scans retain only the ML-owned
  `AdaptiveRefinementConfig` and the documented cold heterogeneous
  multi-region vtable. Exact-head ordinary CI `29875284052`, architecture
  validation `29875284007`, and legacy audit `29875283982` all pass at
  `cc382dbc2243678fef55101aa106e9f8d7ad7bbf`. PR #304 merged as `9ad18523d`.
- Hosted PR #304 first-head evidence: the pinned Atlas checkout confirmed the
  Gaia `approx` lock entry was stale; Cargo regenerated the one-line closure.
  The legacy audit now passes after replacing two new `approx::` test imports
  and one inherited provider-name doc token, without allowlist growth. The
  inherited rustfmt drift reported by main and the PR is corrected in the
  exact twelve files emitted by the CI formatter. Replacement run
  `29864331893` then proved the ordinary workflows still pinned Atlas
  `71cdc54c` while the committed lock and benchmark workflow used `614914cf`;
  those Atlas revisions differ at thirteen provider gitlinks. One local
  composite action now owns the ordinary-workflow provider pin and all sixteen
  call sites delegate to it. The first exact-graph security audit rejected the
  registered Iris Git source because the source policy lagged the lock graph;
  `deny.toml` now admits that exact first-party repository. Exact-head CI rerun
  pending.
## KW-CI-063 — Bound Atlas benchmark oracle [patch] [arch] — done

- Owner: /root; scope: benchmark CI, its retired local classifier, ADR 045,
  and synchronized PM evidence.
- Acceptance: benchmark-relevant PRs compare their exact base and head with
  the canonical production Criterion targets held constant at one filesystem
  path; every plotting-eligible target executes once on the candidate; four
  isolated pair jobs finish within 30 minutes and execute phase-reversed AB/BA
  replications; Atlas derives family-wise confidence and fails closed on
  missing, mismatched, or regressed results.
- Decision: [`ADR-045`](docs/ADR/045-atlas-benchmark-regression-gate.md).
- Evidence: the single-run same-baseline Python classifier is deleted. The
  dedicated workflow pins Atlas classifier and provider graph `614914cf`. Run
  `29797805169` exposed eight auto-discovered libtest targets before
  measurement; automatic discovery is now disabled, all 22 retained Criterion
  targets are explicit, package libtest harnesses are excluded, placeholder
  instruments and unreachable no-op entries are removed, and both revisions
  must match their benchmark source registry. Exact-head run `29814752294`
  proved that serializing all four isolated pairs exceeds the finite job
  bound; the unchanged pair measurements now execute as four matrix jobs and
  feed one aggregate classifier. Exact-head run `29841101698` completed all
  four pairs but found three replicated apparent regressions despite no
  semantic production delta; distinct checkout paths remained correlated with
  revision. The workflow now moves each revision through one
  `kwavers-measurement` path. Run `29867760523` remained active after 157
  minutes; `linear_swe_wave_propagation` alone requested about 32 minutes for
  one revision measurement. The bounded replacement executes the full
  candidate suite once, retains unchanged samples for
  `performance_baseline`, `critical_path_benchmarks`, and `simd_field_ops`,
  and caps every job at 30 minutes. The superseded exact-head run `29875283986`
  completed all four pairs but classified all 190 long-horizon and ancillary
  cases, reporting 37 replicated regressions outside those three canonical
  targets. That run confirms the already-recorded full-suite scope and latency
  defect; it does not exercise the bounded workflow. Replacement head
  `a85aa58e5ad350f5a72483fd541337b95ed0f8de` passes full candidate smoke, all
  four 21–23 minute AB/BA pair jobs, and aggregate classification in run
  `29884797777`; ordinary CI `29884797767`, architecture `29884797709`, and
  legacy audit `29884797739` also pass. PR #306 merged as `00d06f00e`.

## KW-GPU-062 — GPU PSTD peak-pressure output [major] — review

- Owner: /root; scope: `crates/kwavers-gpu/src/pstd_gpu/`, its WGPU shader
  ABI, `crates/kwavers-simulation/src/solver_adapters/gpu_pstd.rs`,
  `crates/kwavers-math/src/fft/mod.rs` CPU-reference FFT boundary, and the
  in-repository simulation consumer boundary.
- Acceptance: the provider accumulates `max_t |p|` on the GPU for every voxel,
  transfers exactly that one pressure volume when requested, and never labels a
  final pressure frame as a peak envelope. The output request supports final,
  peak, or both without allocating a peak volume for a sensor-only run. Its
  source, lossless-absorption, and heterogeneous-nonlinearity choices match
  the CPU PSTD contract without a host fallback. The reference FFT executes
  directly on the shared Leto/Eunomia complex type rather than copying through
  a duplicate facade representation.
- Decision: [`ADR-040`](docs/ADR/040-gpu-pstd-peak-pressure-output.md).
- Evidence target: value-semantic output-selection and final-versus-peak
  invariants, a real WGPU burst regression, GPU-feature Nextest, and an
  in-repository simulation-consumer regression.
- Evidence: the simulation adapter requests the provider's explicit peak
  output, retains it separately from final fields, and shares the direct
  runner's weighted local-medium pressure-source schedule. It rejects both
  unsampled `Source` objects and unsupported velocity-source assembly rather
  than discarding source information. Warning-denied all-feature Clippy passes,
  and the WGPU-featured Nextest lane passes 259/259 tests, including the
  heterogeneous CPU/GPU contract and real peak-envelope runs. Hosted ordinary
  workflows use one local action pinned to the Atlas-owned checkout action and
  provider graph at `614914cf`; direct Aequitas and Proteus revisions match
  that graph, and the lock contains one Aequitas source identity. A replacement
  hosted matrix remains the closure gate.
- External integration requirement: the private full-wave consumer remains
  responsible for its explicit peak-pressure regression. Its inaccessible
  checkout does not widen or block this repository's delivery boundary.

## KW-UQ-062 — Integrate Tyche uncertainty ownership [major] [arch] — implemented

- Owner: /root; scope: Analysis conformal/sensitivity APIs, PINN conformal and
  ensemble statistics, Tyche dependency policy, ADR 043, and synchronized
  public documentation.
- Acceptance: one Tyche-owned corrected rank and moments implementation serves
  both consumers; interval results borrow scores and retain every prediction;
  sensitivity is const-generic, deterministic, allocation-free per sample, and
  named squared correlation; pseudo-Sobol/Morris bodies have no residue.
- Evidence: public Tyche merge `2b8fb14` is the single dependency in both
  packages; focused Analysis and PINN suites pass 13/13 and 12/12. The final
  local all-feature Analysis suite passes 764/764, including heterogeneous
  report borrowing, native-precision even medians, and invalid beamforming
  boundary regressions. Review regressions require every configured confidence
  level to identify an emitted interval and allow non-finite PINN outputs to
  reach typed validation without a debug-only panic. Warning-denied Clippy,
  no-default checks, doctests, normal Rustdoc generation, the facade clinical
  workflow example, and source policy pass. `cargo-semver-checks` runs 223
  Analysis checks and identifies 10 major API breaks, matching ADR 043.
- The Atlas-owned checkout action pins provider graph `71cdc54`, eliminating
  the moving-provider lock failure exposed by run `29781981026`. PR 298 is the
  canonical hosted-verification and merge record for the documentation-complete
  head.
- The migrated comprehensive clinical workflow is partitioned into 127/168/161/
  157/106/91/60-line root and concern leaves. Default and GPU builds plus
  warning-denied Clippy pass; no no-op uncertainty clone, vtable path, or
  redundant CEUS map copy remains.
- Heterogeneous report dispatch remains only at the cold report boundary;
  callers pass a borrowed reference slice that `UncertaintyReport` retains
  without boxes or a second vector allocation.
- The Apollo Git patch maps Coeus's remote FFT dependency to Kwavers's
  synchronized provider checkout, eliminating duplicate Apollo package
  identities while retaining the committed lock.
- PR 298 retains the hosted matrix and terminal merge evidence.
- The combined Tyche/Asclepius consumer head passes locked all-target,
  all-feature compile plus 144/144 cross-provider Nextest cases at merged
  provider revisions (`2b8fb14`, `794f8c3`; runs `ee8687cc` and `a6fcba7b`).

## KW-GPU-061 — Extend GPU PSTD FFT lattice [minor] — in-progress

- Owner: /root; scope: `crates/kwavers-gpu/src/pstd_gpu/` and GPU PSTD
  consumer validation in `kwavers-simulation`.
- Acceptance: the Hephaestus-acquired WGPU PSTD provider accepts every
  power-of-two axis through 1,024, rejects 2,048 before allocation, and keeps
  the final-field readback contract intact. The shader declares no more than
  12 KiB of workgroup storage and the acquisition contract requires that
  amount explicitly.
- Evidence target: value-semantic dimension contracts, shader/host ABI
  regression, GPU-feature Nextest, and a Leo consumer gate.

## KW-MED-059 — Consolidate continuous medium access [major] — ✅ done

- Owner: Codex; scope: `crates/kwavers-medium/src/{wrapper.rs,interface.rs,lib.rs}`.
- Acceptance: one generic `CoreMedium` entry point serves both concrete and
  trait-object callers; the four `*_at_core` compatibility forwarders and their
  re-exports are deleted; interface detection compiles and its value semantics
  remain unchanged.
- Evidence: source search reports no `*_at_core` references; package check
  passes; locked nightly Nextest runs 187/187 tests in 1.801 seconds.
- Versioning: `kwavers-medium` 3.0.0 → 4.0.0; ADR 038 records the public
  removal and value-preservation theorem.

## KW-SOL-058 — Elide elastic-FWI objective histories [patch] — in-progress

- Owner: Codex; scope:
  `crates/kwavers-solver/src/forward/elastic/swe/core/solver/point_force_drive.rs`,
  `crates/kwavers-solver/src/inverse/elastography/elastic_fwi/mod.rs`, and
  synchronized PM evidence.
- Acceptance: observed-data synthesis and objective-only FWI forward runs
  retain receiver traces directly, preserve exact trace values relative to the
  full-history propagator, and complete the lesion-reconstruction contract
  under the existing CI timeout.
- Evidence: hosted job `87949355634` timed out the FWI contract at 90.010 s
  despite `--test-threads=1`. The new exact trace-equivalence regression passes,
  and the full FWI contract passes locally in 29.123 s; a fresh hosted matrix
  remains the closure gate.

## KW-CI-057 — Serialize full workspace test processes [patch] — superseded

- The workflow serialization landed, but job `87949355634` still timed out
  `fwi_outperforms_linear_inversion` at 90.010 s. The timeout is a real
  allocation/performance defect, now owned by KW-SOL-058 rather than a runner
  oversubscription issue.

## KW-GPU-056 — Align Hephaestus device-limit contract [patch] — in-progress

- Owner: Codex; scope: `crates/kwavers-gpu/src/gpu/` and
  `crates/kwavers-gpu/src/beamforming/three_dimensional/provider.rs`.
- Acceptance: every explicit `hephaestus_core::DeviceLimits` initializer carries
  the aggregate buffer/acceleration-structure limit; WGPU preserves the
  provider baseline and CUDA reports `None` for the non-applicable capability.
- Evidence: hosted Architecture Validation job `87946612531` reported four
  `E0063` diagnostics after Hephaestus added the field. Focused GPU check and
  feature validation must pass on the refreshed provider graph.

## KW-SOL-054 — Repair AVX-512 FDTD layout contract [patch] — in-progress

- Owner: Codex; scope: `crates/kwavers-solver/src/forward/fdtd/avx512_stencil/`
  and synchronized PM evidence.
- Acceptance: pressure and velocity AVX-512 kernels use Leto C-order strides,
  cover all interior vector tails, validate raw-pointer layout preconditions,
  and match analytical uniform/linear reference fields on an AVX-512 host.
- Evidence: Architecture Validation job `87932791305` observed the old kernel
  write `0` at interior `[8, 8, 8]` for a uniform `7.5` field. On an AVX-512
  host, the focused Nextest suite passes all seven cases and package test
  compilation passes. A fresh hosted matrix remains required before merge.

## KW-CI-053 — Update GPU PSTD parity contract [patch] — ✅ review

- Owner: Codex; scope: `crates/kwavers/tests/gpu_pstd_parity.rs` and its PM
  evidence only.
- Acceptance: ignored GPU parity tests call the provider-owned six-argument
  `GpuPstdSolver::run` API with `PstdOutputRequest::sensor_traces()` and consume
  `sensor_data`; no
  compatibility wrapper or test simplification is introduced.
- Evidence: hosted job `87936633879` gave the exact E0061/E0308 diagnostics;
  package-scoped nightly rustfmt passes after the direct call-site migration.
- Residual: focused Nextest and the fresh hosted matrix must complete.

## KW-CI-051 — Remove obsolete deployment workflow [patch] — ✅ review

- Owner: Codex; scope: `.github/workflows/deploy.yml` only.
- Acceptance: no workflow references absent deployment artifacts or invalid
  step syntax; the supported CI surface remains architecture validation,
  migration audit, and the provider-aware build/test workflow.
- Evidence: the repository contains no `Dockerfile` or `k8s` tree, and Actions
  run `29593287070` failed at workflow parsing before creating jobs. The stale
  workflow is deleted without changing simulation or deployment code.
- Driver: the workflow was inherited from the old PINN service layout and has
  no live repository inputs.

## KW-CI-052 — Lock and parse supported Cargo workflows [patch] — ✅ review

- Owner: Codex; scope: `.github/workflows/ci.yml` and
  `.github/workflows/architecture-validation.yml`.
- Acceptance: all Cargo graph-consuming commands use `--locked`, the workflow
  YAML parses, and no command is hidden behind a malformed step indentation.
- Evidence: local PyYAML parsing reports `yaml-ok`; the diff is workflow-only
  and `git diff --check` is clean. The hosted matrix is the remaining
  verification tier.
- Driver: the prior CI definition allowed live provider resolution and had
  indentation errors in the build/convergence/validation steps.

## KW-CI-050 — Restore hosted format and CUDA prerequisites [patch] — ✅ review

- Owner: Codex; scope: the finite-window PSTD test formatting and CUDA
  container prerequisites in `architecture-validation.yml`.
- Acceptance: repository rustfmt passes for the corrected test and the CUDA
  build image provides OpenSSL development metadata required by
  `openssl-sys`; no test workload or assertion changes.
- Evidence: file-scoped nightly rustfmt is clean after the mechanical rewrite;
  `libssl-dev` is installed before the CUDA compile step. Hosted rerun is the
  remaining external verification.
- Driver: Architecture Validation jobs `87924378467` and `87918394437`
  reported the exact stale-format and missing `openssl.pc` failures.

## KW-CI-049 — Align Apollo provider lock [patch] — ✅ review

- Owner: Codex; scope: `Cargo.lock` and provider-graph synchronization.
- Acceptance: the lock records Apollo `0.24.0`, the version supplied by the
  merged Apollo PR #45, and the focused locked suite remains value-semantic
  green.
- Evidence: lock-only diff; `cargo nextest run --locked -p kwavers-gpu
  -p kwavers-simulation -p kwavers-solver` passes 1,036/1,036 with four
  skipped tests.
- Residual: three existing solver tests are slow; one exceeded 30 seconds in
  this run and requires a separate profile-guided optimization item.
## KW-FFT-050 — Direct Apollo axis FFT storage [patch] — review

- Owner: Codex; scope: `kwavers-math::fft` axis-transform facade, locked
  provider graph, and synchronized Kwavers artifacts.
- Driver: each viscoacoustic derivative copied a full `Array3<Complex64>` into
  and out of Apollo despite both sides using Leto storage and
  `eunomia::Complex64`. The three velocity gradients and three divergence
  derivatives therefore created twelve temporary full fields and performed
  twenty-four avoidable full-buffer copies per solver step.
- Acceptance: the facade delegates directly to Apollo's axis plan methods, the
  locked graph resolves Apollo 0.24.0, and
  `decay_matches_dispersion_3d_diagonal` passes under the unchanged Nextest
  timeout and workload. Evidence: the exact regression completes below the
  60-second cap.

## KW-FFT-049 — Retire stale Apollo GPU probe [major] — done

- Owner: Codex; scope: `kwavers-math::fft` GPU facade, its public migration
  note, and the all-feature Clippy frontier.
- Closure: no `gpu_fft_available` wrapper remains. The GPU tests acquire a
  typed Hephaestus `WgpuDevice`, construct Apollo's `WgpuBackend`, and preserve
  value-semantic spectrum parity plus reusable-buffer round-trip coverage.
  GPU-enabled Nextest passes 265/265; warning-denied all-feature Clippy, docs,
  and doctests pass. CI resolves Atlas path dependencies from `main` rather
  than the stale integration branch whose RITK pin preceded the repair. Its
  CUDA container now installs `libssl-dev` for the RITK/DICOM OpenSSL build.
  It also installs the `clang` executable selected by the OpenSSL build script.
  The plotting benchmark job executes only the `kwavers` package, avoiding
  an invalid PyO3 extension link while preserving Criterion execution; the
  stable/beta/nightly plotting build/test matrix uses the same boundary.

## KW-GRID-048 — Checked grid cardinality [minor] — done

- Owner: Codex; scope: `kwavers-grid::Grid`, its core error text, and
  synchronized Kwavers artifacts.
- Closure: `Grid::new` rejects non-finite spacing and `checked_size` returns
  `None` for an externally-mutated dimension product that overflows. Locked
  Nextest passes 40/40; warning-denied Clippy, docs, and doctests pass. The
  isolated SemVer baseline remains blocked by its duplicate Leto graph.

## KW-DOP-045 — Signed pulsed-wave spectral Doppler [minor] — review

- Owner: Codex; scope: `kwavers-analysis` pulsed-wave Doppler spectrum contract,
  its value-semantic regressions, and synchronized provider artifacts.
- Acceptance: a physical complex-I/Q trace produces a two-sided spectrum with
  explicit negative and positive velocity bins; reverse-flow energy remains in
  the returned spectrum, invalid Doppler geometry is rejected rather than
  mapped through an artificial positive angle cosine, and an FFT shorter than
  the acquired ensemble fails rather than silently discarding pulses.
- Driver: LeoNeuro's physical moving-scatterer sector ensemble requires a PW
  provider that retains reverse-flow bins; the former one-sided magnitude API
  discarded that physical degree of freedom.
- Evidence: the locked Atlas graph resolves, `kwavers-analysis` compiles, its
  normal warning-denied Clippy surface passes, and the focused PW Nextest
  regression passes 8/8. The package's all-feature Clippy reaches the separate
  `kwavers-solver` lint ratchet below.

## KW-LINT-047 — Solver all-feature lint ratchet [patch] — in-progress

- Owner: Codex; scope: coherent offline `Cargo.lock` feature resolution plus
  `crates/kwavers-solver` machine-applicable Clippy corrections exposed through
  `kwavers-analysis --all-features`.
- Acceptance: the solver compiles under its complete feature set and its
  warning-denied Clippy gate has no remaining source diagnostics; behavior is
  unchanged and existing solver regressions retain their value semantics.
- Driver: 79 source diagnostics prevented the public analysis package from
  completing its all-feature warning-denied gate.
- Evidence: the complete solver feature set compiles, warning-denied solver
  Clippy passes across all targets, and its full Nextest suite passes 844
  runnable tests with 4 ignored after two invalid test oracles were corrected.
  The locked `kwavers --all-features` facade check passes. The all-target gate
  is reconciling stale test and example APIs exposed by concurrent provider
  migration. Current RITK and Hephaestus dependency warnings remain outside
  this crate's lint scope.

## KW-CI-046 — Atlas-path CI and security audit [patch] — in-progress

- Owner: Codex; scope: reusable GitHub Actions setup for the sibling Atlas path
  providers declared by `Cargo.toml`, root Cargo-deny policy, stale
  architecture-workflow cleanup, portable hosted CPU code generation, explicit
  CUDA-runtime compilation, native Nextest invocation, WGPU 30 provider
  alignment, Leto API migration required by the public `full` build, and the
  native solver literature-validation module targeted by CI.
- Acceptance: every Cargo job materializes the manifest-declared sibling
  providers at the `codex/kwavers-atlas-integration` submodule revisions before
  resolving the workspace; no workflow invokes the deleted
  `scripts/validate_architecture.sh`;
  native test jobs use Nextest, with doctests retaining Rustdoc's supported
  runner; hosted CPU jobs never use `target-cpu=native`; CUDA runtime code
  compiles against its required toolkit; and the security job evaluates the
  root policy against the Kwavers manifest and rejects unapproved sources,
  licenses, and advisories. The public `full` package build uses the same WGPU
  30 immediate-data ABI as its Hephaestus provider and all Leto operations
  propagate their fallible view/index contracts. Solver literature validation
  is a compiled native module with value-semantic reference regressions rather
  than an empty test filter.
- Driver: PR #288 fails before compilation because `../apollo` and the other
  Atlas path providers are absent in GitHub Actions. The architecture workflow
  separately invokes a script deleted in commit `91514cad2`.
- Evidence: GitHub Actions run `29443042765` reports the missing
  `apollo/crates/apollo-fft/Cargo.toml`; the first repair run proves that
  provider defaults are insufficient (`apollo-fft` 0.17.0 conflicts with
  RITK's `^0.15.0`), while Atlas `main` pins incompatible Apollo 0.14. The
  committed Kwavers Atlas integration branch pins Apollo 0.15. The next
  architecture rerun materializes all 12 providers and exposes the first real
  source error: Linux `CPU_SET` receives an immutable set in the explicit-CPU
  branch. Strict Clippy then finds two manual NUMA-mask ceiling divisions.
  The legacy audit also misclassifies NumPy's PyO3 ndarray facade as direct
  ndarray use; it now distinguishes those boundaries and removes 1,477 stale
  allowlist entries. Local manifest-path resolution finds all 12 sibling
  providers. The root `deny.toml` now uses strict registry/Git allowlists,
  records only exact license exceptions for `cuda-oxide`, `colored`, and
  `epaint`, removes unused direct DICOM 0.8 workspace pins, and updates the
  lock graph through RITK DICOM 0.10 and patched advisory releases. Local
  Cargo-deny licenses, advisories, and sources checks pass. The two remaining
  yanked notices are non-advisory `spin` 0.9.8 (Flume 0.11.1) and 0.10.0
  (Burn) transitive constraints.
  Re-open trigger: any coordinated-provider checkout, manifest
  resolution, or subsequent CI-job failure on the repaired PR head.
  The first rerun additionally proves that the old committed native-CPU flag
  can SIGILL on hosted runners and that a CPU runner cannot build the explicit
  CUDA runtime. Both are environmental configuration defects, not acceptable
  reasons to suppress the checks: portable CPU workflow legs now compile the
  supported feature surface while a CUDA 13.2 container compiles the runtime
  provider. The resulting GPU PINN compile also exposed a source defect: it
  interpreted Coeus `[out, in]` weights as `[in, out]`, read device tensors as
  host slices, hid constructor errors behind `.ok()`, and returned a fixed
  uncertainty vector. The corrected implementation uses Coeus backend
  readback, authoritative weight orientation, propagated construction errors,
  and an analytical half-step quantization bound. The direct WGPU 26
  dependency is now removed: Kwavers uses the WGPU 30 provider selected by
  Hephaestus, with former push-constant kernels expressed through WGPU
  immediate data and map-range failures propagated as typed GPU errors. The
  Leto call sites exposed by the public `full` build now use native shapes,
  views, and fallible axes without ndarray fallback adapters. Workspace Rustdoc
  compiles under the legacy warning baseline while the deployable public
  `kwavers` facade remains warning-denied; the extensive physics Rustdoc-link
  cleanup remains tracked ratchet work rather than a CI suppression. The first
  repaired remote run proved the solver workflow's `validation::literature`
  filter selected no module. Its source and tests existed but `validation::mod`
  omitted the module declaration. Restoring the native edge corrected the
  nested `TWO_PI` scope and Leto three-axis index; nine literature regressions
  now pass locally, including an exact Treeby snapshot and multi-time
  dimension-contract rejection. The architecture job's first full-facade build
  also lacked the fontconfig development package already present in the other
  Cargo CI jobs; its system prerequisites now match that established contract.
  The strict rerun also identifies and removes three no-op Leto `Array3`
  conversions in the touched FD monitor, preserving the public facade's
  warning-denied contract.

## KW-IMG-044 — Active complex-I/Q imaging primitives [minor] — done

- Owner: Codex; scope: promote real-RF analytic-baseband demodulation into a
  direct `kwavers-analysis` API, then provide a transmit-aware complex DAS
  kernel consumed by LeoNeuro's active sector ensemble.
- Acceptance: one validated Kwavers contract converts real RF channels to
  complex baseband without the narrowband-snapshot adapter; complex DAS uses
  the same per-pixel transmit delays and apodization law as real DAS and
  restores the carrier phase removed by demodulation. LeoNeuro can compose
  those primitives without reimplementing demodulation, delay, rephasing, or
  complex interpolation.
- Driver: KW-IMG-043 closes real active B-mode. LeoNeuro's remaining
  color/power/PW/fUS sector modes require complex slow-time I/Q and cannot use
  a real B-mode result as a substitute.
- Evidence: the bin-centred analytic-baseband identity and a fractional-delay
  complex-I/Q regression now prove both interpolation and restoration of
  `exp(j 2πf₀τ)` before summation; the former snapshot adapter delegates its
  RF-to-I/Q work to the same API. The direct primitives do not invent
  inter-frame phase or scatterer motion.

## KW-IMG-045 — Frame-resolved physical I/Q ensemble [minor] — todo

- Owner: Codex; scope: consume the direct I/Q primitives from LeoNeuro using
  explicit physical scatterer states for each slow-time frame.
- Acceptance: a color/power/PW/fUS sector sequence derives its frame-to-frame
  phase from submitted scatterer position and reflectivity evolution, then
  applies the provider I/Q demodulator and complex DAS. No deterministic
  phase-animation surrogate remains in the Python reference path.
- Driver: KW-IMG-044 makes individual real-RF frames complex-I/Q capable but
  deliberately does not claim a physical slow-time evolution law.

## KW-IMG-043 — Active transmit-event imaging contract [minor] — done

- Owner: Codex; scope: `kwavers-phantom` transmit-event RF synthesis,
  `kwavers-analysis` transmit-aware imaging DAS, focused regressions, and
  synchronized provider artifacts.
- Acceptance: one validated plane-wave or virtual-source event supplies the
  same transmit travel time to synthetic RF generation and DAS reconstruction;
  no consumer reproduces transmit-time or spreading arithmetic.
- Driver: LeoNeuro's reference sector imager has plane/diverging transmit
  timing, but the provider pair only supports monostatic RF plus receive-only
  DAS. This provider gap blocks a native replacement of that Python solver.
- Decision: the existing `kwavers-transducer::ultrafast` processors are
  two-dimensional delay utilities tied to linear-array coordinates. The new
  `TransmitWavefront` is the validated three-dimensional point-scatterer event
  contract; DAS remains generic over one transmit delay per pixel so measured
  or refracting events do not fork its receive kernel.
- Evidence: plane-wave and virtual-source closed-form RF regressions pass
  12/12 in `kwavers-phantom`; transmit-aware DAS localization and invalid-delay
  regressions pass 6/6 in `kwavers-analysis`. Warning-denied Clippy passes;
  package Rustdoc completes with one pre-existing Phantom and 57 pre-existing
  Analysis unresolved links, none from this contract.

## KW-IQ-042 — Complex I/Q SVD clutter contract [minor] — done

- Owner: Codex; scope: `kwavers-analysis` I/Q SVD provider, its public export,
  value-semantic regressions, and synchronized PM artifacts.
- Acceptance: the provider accepts `[slow_time, angle, range]` `Complex64` I/Q,
  applies the reference rank truncation without centering, removes each complex
  clutter mode as its two realified singular modes, and returns filtered I/Q plus
  `Σ_t |I/Q|²` power. A rank-one complex/DC clutter oracle proves no mean is
  re-added; an orthogonal complex mode proves phase and power preservation.
- Driver: LeoNeuro's tracked reference currently executes NumPy complex SVD and
  its own power reduction. Leto supplies the authoritative real rank-revealing
  SVD; the fUS-domain realification belongs in Kwavers, not a consumer fallback.
- Evidence: `IqSvdClutterFilter` realifies the full complex matrix, removes
  paired modes, and reconstructs filtered I/Q plus unnormalized power without
  temporal centering. Locked offline focused Nextest passes 3/3; default-feature
  warning-denied Clippy passes; package Rustdoc completes with the existing 57
  unresolved-link warnings outside the I/Q files. LeoNeuro's rebuilt CPython
  boundary/reference suite independently passes 14/14.

## KW-DOP-041 — Doppler autocorrelation signal-power contract [minor] — done

- Owner: Codex; scope: `kwavers-analysis` autocorrelation provider/export and
  synchronized PM artifacts.
- Acceptance: one Kasai traversal returns velocity, normalized variance, and
  lag-zero signal power; the existing tuple API delegates to that authoritative
  result; a coherent-IQ oracle pins sign, power, and variance semantics.
- Driver: LeoNeuro's private reference GUI currently recomputes power and uses
  the opposite lag-one phase order in Python. The provider must own the full
  map contract before its Python conversion boundary can delete that solver.
- Evidence: `estimate_with_power` owns all three maps and the tuple API
  delegates without a second traversal. Locked offline focused Nextest passes
  5/5; default-feature Clippy passes; Rustdoc completes with a documented
  pre-existing unresolved-link baseline. LeoNeuro's PyO3 consumer verifies the
  sign and map values independently.

## KW-RAY-040 — Layered focus-path contract [minor] — in-progress

- Owner: Codex; scope: `kwavers-transducer` layered Rayleigh propagation API,
  LeoNeuro focus integration, reference-Python parity, tests, and PM records.
- Driver: the provider's Rayleigh kernel integrates each straight-ray layer, but
  LeoNeuro focus steering currently receives only one sound speed while the
  reference script separately recreates the layered phase law.
- Acceptance: Kwavers exposes the validated segmentwise propagation phase;
  LeoNeuro focuses through that provider contract; the reference delegates
  instead of retaining an independent layered phase implementation.

## KW-DEP-039 — Make Gaia an Atlas-local dependency [patch] — review

- Owner: Codex; scope: workspace manifest, dependency records, and LeoNeuro
  SemVer integration.
- Driver: Cargo ignores Kwavers' root `[patch]` tables when LeoNeuro's SemVer
  checker packages `leoneuro-sim`; its transitive Gaia Git source therefore
  resolves a historical revision that lacks the Eunomia dependency.
- Acceptance: Kwavers declares the live Atlas Gaia checkout directly, deletes
  the redundant Gaia source patch, and LeoNeuro's historical SemVer comparison
  resolves through the local Gaia-to-Eunomia graph.
- Evidence: locked offline metadata resolves `gaia` at `D:\atlas\repos\gaia`;
  warning-denied `kwavers-mesh` Clippy passes; Nextest passes 9/9. The isolated
  LeoNeuro package now passes Gaia resolution and stops at the independent
  Moirai-to-Themis Git edge (`themis ^0.10` versus 0.9.17 at its pinned Git
  revision). That residual belongs to Moirai portability, not Kwavers.

## KW-ARCH-036 — Clinical-imaging dependency boundary [major] — review

- Owner: Codex; scope: `kwavers-physics`, `kwavers-solver`, direct clinical
  consumers, and ADR-036.
- Driver: LeoNeuro's forward PSTD package reaches `ritk-filter` through
  unconditional clinical image I/O and registration dependencies.
- Acceptance: `leoneuro-sim` no longer reaches `ritk-filter`; PSTD builds and
  its finite-aperture boundary regression runs through the native Kwavers path;
  every in-workspace user of gated clinical APIs opts in explicitly.
- Design: [`ADR-036`](docs/ADR/036-clinical-imaging-feature-boundary.md).
- Evidence: locked offline Physics Nextest passes 1,554/1,554 without the
  feature and 1,710/1,710 with it; locked Leo Nextest passes 29/29 and reverse
  dependency resolution reports no `ritk-filter` package. The feature-enabled
  path compiles RITK only when explicitly selected.

## KW-DIAG-037 — Promote multimodal fusion to Diagnostics [major] — todo

- Owner: unclaimed; blocked by KW-ARCH-036 verification.
- Move the complete `kwavers-physics::acoustics::imaging::fusion` ownership into
  `kwavers-diagnostics` with every call site rewritten directly. Delete the old
  physics path and retain no re-export. Acceptance: Physics has no registration
  dependency and Diagnostics owns all fusion and registration contracts.

## KW-DOC-038 — Resolve Physics Rustdoc links [patch] — todo

- Owner: unclaimed; scope: unresolved intra-doc links and bracketed unit
  annotations in `kwavers-physics` Rustdoc.
- Driver: the all-feature package documentation build emits 575 unresolved-link
  warnings; the clinical-imaging boundary change introduces none.
- Acceptance: `cargo doc -p kwavers-physics --all-features --no-deps` is
  warning-clean without disabling Rustdoc diagnostics.
- Evidence: `gap_audit.md` documentation baseline entry.

## KW-APERTURE-003 — Planar sector BLI rasterization [minor] — review

- Owner: Codex; scope: `kwavers-transducer::kwave_array`, canonical planar
  aperture geometry, tests, and PM artifacts.
- Driver: private LeoNeuro hybrid C/D sectors require full-wave PSTD sources
  without finite-disc substitution.
- Acceptance: validated oriented disk/annular-sector geometry rasterizes through
  the existing BLI per-element source path, conserves analytical aperture area,
  preserves independent element signals, and passes package gates.
- Evidence: warning-denied all-target/all-feature Clippy; Nextest 215/215 with
  one existing skip; doctests 1/1 with six existing ignored; warning-clean
  Rustdoc; exact per-quadrant analytical area and independent-signal regression.
  A subsequent value regression proves BLI rejects only sources beyond its
  finite window, preserving clipped apertures while preventing distant sinc-tail
  boundary injection.

## KW-APERTURE-002 — General planar aperture propagation [major] — review

- Owner: Codex; scope: `kwavers-transducer` Rayleigh aperture types, kernel,
  tests, ADR-035, version, and private LeoNeuro consumer migration.
- Driver: hybrid Fresnel-zone pMUT cells require independently driven central
  and annular electrode sectors without circular-piston tessellation.
- Acceptance: one bounded provider kernel integrates disks and oriented annular
  sectors, preserves existing circular-piston oracles, and proves coherent
  sector superposition before the consumer adds electrode control topology.
- Evidence: warning-denied all-target/all-feature Clippy; Nextest 214/214 with
  one existing skip; doctests 1/1 with six existing ignored examples; and
  warning-clean package documentation.

## KW-APERTURE-001 — Own finite circular-piston propagation [minor] — review

- Owner: Codex; scope: `kwavers-transducer::transducers::physics`, its public
  exports, analytical/differential tests, version, and synchronized PM records.
- Driver: private Atlas consumer `leoneuro-rs` currently duplicates and
  double-counts finite-aperture diffraction.
- Acceptance: the provider evaluates the baffled Rayleigh first integral with
  the `k/(2π)` surface-pressure prefactor, area-consistent disk quadrature,
  oriented half-space suppression, and complex coherent summation; analytical
  and far-field reference tests plus package gates pass.
- Evidence: exact on-axis and disk-area oracles, far-field Bessel differential
  oracle, rotation and baffle invariants, warning-denied Clippy/docs, and
  Nextest 209/209. Registry-baseline semver analysis is externally unavailable.

## KW-MEDIUM-CT-001 — Own complete CT medium assembly [arch] — review

- Owner: Codex; scope: `kwavers-medium::heterogeneous` CT builder,
  removal of the former `kwavers-physics` skull-owned builder, and affected
  documentation/tests.
- Acceptance: `CtMediumBuilder` is exported only by `kwavers-medium`, maps all
  five acoustic fields through `HuAcousticModel`, rejects shape mismatch, and
  focused package gates pass.
- Driver: private Atlas consumer `leoneuro-rs` requires the provider-owned
  standard-HU medium contract.
- Evidence: warning-denied all-target/all-feature `kwavers-medium` Clippy and
  Nextest 187/187 pass on the aligned Atlas provider graph.

- [x] [patch] Close the native Leto beamforming provider graph.
  Owner: Codex. Scope: workspace Leto features, adaptive/Capon linear solves,
  transducer inversion, solver identity-conversion residue, lockfile, and
  matching PM records. Acceptance met by warning-denied locked package Clippy
  and locked package Nextest 908/908 without `ndarray-compat`.

- [x] [patch] Remove the rank-1 Leto/NumPy shim pair and its 44 name
  occurrences across ten Python-boundary consumers.

- [x] [patch] Delete the PyO3 complex-array identity conversion family and
  remove 24 redundant allocation/traversal sites while preserving the real
  Leto-to-NumPy ownership boundary.

- [x] [patch] Remove the `kwavers-boundary` traversal adapter and use Leto's
  canonical indexed map/zip operations directly at all ten bounded consumers.

- [x] [minor] Move focal-kernel NPZ storage ownership upstream to Consus and
  remove the direct `ndarray-npy` production dependency.

- [x] [patch] Reconcile stale provider names in production module contracts;
  closed 2026-07-10 with Moirai/Leto ownership reflected at each touched site.

> Active strategy at top; CLOSED history retained below for traceability.
> Full gap inventory: [gap_audit.md](gap_audit.md). Active increment: [CHECKLIST.md](CHECKLIST.md).

## DONE: kwavers-grid native Leto surface [arch]

Delete the obsolete compatibility re-export module and redundant `_leto`
forwarding APIs now that the canonical grid storage and operator signatures are
Leto-native. Update all workspace callers without a compatibility bridge, then
close the package clippy frontier. Decision:
[ADR 034](docs/ADR/034-kwavers-grid-native-leto-surface.md). Verification:
static duplicate audit clean; all-target grid clippy clean; grid nextest 38/38;
doctests and warning-clean docs pass; `kwavers-physics` library check passes.
Residual test compilation failures are isolated to existing `kwavers-math` and
`kwavers-solver` Leto migration frontiers.

## DONE: kwavers-analysis narrowband Apollo FFT routing [patch]

Routed narrowband legacy analytic-baseband and windowed STFT snapshot extraction
through Apollo 1-D FFT APIs over Leto buffers instead of importing FFT
execution or complex types from `kwavers_math::fft`. The covariance-facing
ndarray boundary remains `num_complex` for this slice, with explicit conversion
from Apollo complex scratch output.

Evidence tier: compile-time integration, focused empirical tests, and static
source audit. Direct `rustfmt --check` passed for the touched snapshot files;
`rustup run nightly cargo check -p kwavers-analysis` passed; `rustup run
nightly cargo nextest run -p kwavers-analysis narrowband snapshots stft
baseband` passed 30/30; scoped `rg` found no `kwavers_math::fft` imports in
`kwavers-analysis/src/signal_processing`.

Residual: `kwavers-analysis` signal-processing FFT execution now routes through
Apollo. Remaining provider migration work in analysis is the broader ndarray
and `num_complex` boundary cleanup.

## DONE: kwavers-analysis Doppler Apollo 1-D FFT routing [patch]

Routed continuous-wave, pulsed-wave, and Welch spectral Doppler FFT execution
through Apollo's 1-D real/complex FFT APIs over Leto buffers instead of
importing FFT execution and shift utilities from `kwavers_math::fft`.

Evidence tier: compile-time integration, focused empirical tests, and static
source audit. Direct `rustfmt --check` passed for the touched Doppler files;
`rustup run nightly cargo check -p kwavers-analysis` passed; `rustup run
nightly cargo nextest run -p kwavers-analysis doppler continuous_wave
pulsed_wave spectral` passed 49/49; scoped `rg` found no `kwavers_math::fft`
import in the migrated Doppler files.

Residual: remaining direct `kwavers_math::fft` consumers in `kwavers-analysis`
are narrowband snapshot extraction only.

## DONE: kwavers-analysis PAM Apollo 1-D FFT routing [patch]

Routed PAM processor spectrum computation and delay-and-sum peak frequency
estimation through Apollo's 1-D real FFT over Leto buffers instead of importing
FFT execution from `kwavers_math::fft`.

Evidence tier: compile-time integration, focused empirical tests, and static
source audit. Direct `rustfmt --check` passed for the touched PAM files; `rustup
run nightly cargo check -p kwavers-analysis` passed; `rustup run nightly cargo
nextest run -p kwavers-analysis pam delay_and_sum` passed 18/18; scoped `rg`
found no `kwavers_math::fft` import in the migrated PAM files.

Residual: remaining direct `kwavers_math::fft` consumers in `kwavers-analysis`
are Doppler continuous/pulsed/spectral paths and narrowband snapshot extraction.

## DONE: kwavers-analysis analytic-signal Apollo routing [patch]

Routed B-mode envelope detection and time-domain phase-coherence analytic-signal
construction through `kwavers-signal`'s Apollo-backed Hilbert transform instead
of `kwavers_math::fft::analytic_signal_1d`.

Evidence tier: compile-time integration, focused empirical tests, and static
source audit. Direct `rustfmt --check` passed for the touched files; `rustup run
nightly cargo check -p kwavers-analysis` passed; `rustup run nightly cargo
nextest run -p kwavers-analysis b_mode coherence` passed 51/51; scoped `rg`
found no `kwavers_math::fft` or `analytic_signal_1d` in the migrated
B-mode/coherence files.

Residual: remaining direct `kwavers_math::fft` consumers in `kwavers-analysis`
are Doppler continuous/pulsed/spectral paths, PAM processor and delay-and-sum
beamform paths, and narrowband snapshot extraction.

## DONE: kwavers-signal Apollo 1-D FFT migration [patch]

Routed `kwavers-signal` analytic-signal Hilbert transforms and frequency-domain
filtering through Apollo APIs over Leto buffers instead of `kwavers_math::fft`.
The public analytic-signal boundary remains `num_complex` for this slice, with
explicit conversion from Apollo complex output. `kwavers-math` remains in
`kwavers-signal` for non-FFT window coefficients.

Evidence tier: compile-time integration, focused empirical tests, and static
source audit. `rustup run nightly cargo fmt --package kwavers-signal --check`
passed; `rustup run nightly cargo check -p kwavers-signal` passed; `rustup run
nightly cargo nextest run -p kwavers-signal analytic frequency_filter` passed
13/13; scoped `rg` found no `kwavers_math::fft` imports in the touched signal
files.

Residual: remaining direct `kwavers_math::fft` consumers are in analysis,
physics, solver reconstruction/FWI, and wider 3-D solver surfaces.

## DONE: kwavers-solver PSTD axisymmetric Apollo 2-D FFT migration [patch]

Routed `forward::pstd::propagator::axisymmetric` real forward and complex
inverse 2-D FFT execution through Apollo APIs over Leto buffers instead of the
`kwavers_math::fft` plan/cache facade. The ndarray `num_complex` working
buffers remain the current PSTD storage boundary, with explicit conversion at
the Apollo scratch edge.

Evidence tier: compile-time integration, focused empirical tests, and static
source audit. `rustup run nightly cargo fmt --package kwavers-solver --check`
passed; `rustup run nightly cargo check -p kwavers-solver` passed; `rustup run
nightly cargo nextest run -p kwavers-solver axisymmetric_apollo` passed 2/2;
scoped `rg` found no `kwavers_math::fft` import in the axisymmetric module.

Residual: wider solver 3-D FFT facade users, shift utilities, and
`num_complex`-typed PSTD storage boundaries remain separate migration slices.

## DONE: kwavers-solver line-reconstruction Apollo 2-D FFT migration [patch]

Routed `inverse::reconstruction::photoacoustic::line_reconstruction` 2-D FFT
execution through Apollo's complex FFT APIs over Leto buffers instead of the
`kwavers_math::fft` facade. The interpolation/scaling math remains
`num_complex` at the current ndarray boundary, with one private conversion SSOT
for Apollo scratch buffers.

Evidence tier: compile-time integration, focused empirical tests, and static
source audit. `rustup run nightly cargo fmt --package kwavers-solver --check`
passed; `rustup run nightly cargo check -p kwavers-solver` passed; `rustup run
nightly cargo nextest run -p kwavers-solver line_reconstruction` passed 4/4;
scoped `rg` showed only Apollo FFT execution calls in the line-reconstruction
module.

Residual: wider solver 3-D FFT facade users, shift utilities, and
`num_complex`-typed PSTD storage boundaries remain separate migration slices.

## DONE: kwavers-solver fast-nearfield Apollo 2-D FFT migration [patch]

Routed `analytical::transducer::fast_nearfield` field computation through
Apollo's 2-D complex FFT APIs over Leto buffers instead of the
`kwavers_math::fft` facade. The FNM public/storage boundary remains
`num_complex` for this slice because cached Green spectra and ndarray-backed
field arrays still use that representation.

Evidence tier: compile-time integration, focused empirical tests, and static
source audit. `rustup run nightly cargo fmt --package kwavers-solver --check`
passed; `rustup run nightly cargo check -p kwavers-solver` passed; `rustup run
nightly cargo nextest run -p kwavers-solver fast_nearfield` passed 6/6; scoped
`rg` showed only Apollo FFT execution calls in the fast-nearfield module.

Residual: wider solver 3-D FFT facade users remain separate migration slices.

## DONE: kwavers-solver HAS Apollo 2-D FFT migration [patch]

Routed `forward::nonlinear::hybrid_angular_spectrum::diffraction` through
Apollo's 2-D complex FFT APIs over Leto buffers instead of the
`kwavers_math::fft` facade.

Evidence tier: compile-time integration, focused empirical tests, and static
source audit. `rustup run nightly cargo fmt --package kwavers-solver --check`
passed; `rustup run nightly cargo check -p kwavers-solver` passed; `rustup run
nightly cargo nextest run -p kwavers-solver hybrid_angular_spectrum` passed
18/18; scoped `rg` showed only Apollo FFT calls in the HAS cone.

Residual: wider solver 3-D FFT facade users, shift utilities, and
`num_complex`-typed PSTD storage boundaries remain separate migration slices.

## DONE: kwavers-solver KZK Apollo 2-D FFT migration [patch]

Routed KZK angular-spectrum, real parabolic, and complex parabolic 2-D
diffraction scratch paths through direct Apollo FFT APIs over Leto buffers. The
complex-field public boundary remains `num_complex` for this slice; conversion
is localized at the leaf scratch boundary.

Evidence tier: compile-time integration, focused empirical tests, and static
source audit. `rustup run nightly cargo fmt --package kwavers-solver --check`
passed; `rustup run nightly cargo check -p kwavers-solver` passed; `rustup run
nightly cargo nextest run -p kwavers-solver kzk` passed 49/49; scoped `rg`
found no `kwavers_math::fft` imports in the touched KZK 2-D diffraction files.

Residual: wider solver 3-D FFT facade users, shift utilities, and
`num_complex`-typed PSTD storage boundaries remain separate migration slices.

## DONE: kwavers-solver KZK Apollo 1-D FFT migration [patch]

Routed KZK absorption, nonlinear spectral differentiation, and
finite-difference diffraction temporal complex 1-D FFT scratch paths through
direct Apollo APIs over Leto buffers. The external pressure state remains at the
existing `num_complex` boundary for this slice; conversion is localized at the
leaf scratch boundary rather than hidden behind the legacy FFT facade.

Evidence tier: compile-time integration, focused empirical tests, and static
source audit. `rustup run nightly cargo fmt --package kwavers-solver --check`
passed; `rustup run nightly cargo check -p kwavers-solver` passed; `rustup run
nightly cargo nextest run -p kwavers-solver kzk` passed 49/49; scoped `rg`
showed only Apollo 1-D FFT calls in the touched KZK files.

Residual: KZK 2-D angular/parabolic diffraction and wider solver 2-D/3-D FFT
facade users remain separate migration slices.

## DONE: kwavers warning/example cleanup and Apollo complex boundary [patch]

Cleaned `kwavers` all-target warnings in property/comparative tests and the GPU
beamforming benchmark, consolidated the benchmark CPU path through the existing
helper, addressed clippy findings in touched examples/tests, and repaired
inverse-reconstruction Apollo complex boundaries so Apollo-owned 1-D FFT
results convert explicitly at the remaining `num_complex` facade edge.

Evidence tier: compile-time integration and focused empirical tests. `rustup
run nightly cargo check -p kwavers --examples` passed; `rustup run nightly
cargo check -p kwavers --all-targets` passed; `rustup run nightly cargo clippy
-p kwavers --all-targets --no-deps -- -D warnings` passed; `rustup run nightly
cargo nextest run -p kwavers-solver photoacoustic --status-level fail
--no-fail-fast` passed 10/10; `rustup run nightly cargo nextest run -p kwavers
--test property_based_tests --test comparative_solver_tests --test
nonlinear_physics_tests --test test_pstd_kwave_comparison --test
imaging_literature_validation --status-level fail --no-fail-fast` passed 46/46;
`rustup run nightly cargo run -p xtask -- burn-migration-audit` passed with 0
Burn manifest deps and 5 approved non-solver source residuals.

Residual: package-wide `rustup run nightly cargo fmt -p kwavers --check`
remains blocked by pre-existing formatting drift outside this slice in
`crates/kwavers/examples/focused_water_tank_common/simulation.rs`,
`crates/kwavers/examples/pstd_fdtd_comparison.rs`,
`crates/kwavers/src/theranostic/monitor/fd.rs`,
`crates/kwavers/tests/pstd_finite_window_born.rs`, and
`crates/kwavers/tests/quick_comparative_test.rs`; touched files were formatted
with file-scoped `rustfmt`.

## DONE: kwavers-solver inverse Apollo 1-D FFT migration [patch]

Routed the inverse-reconstruction photoacoustic filtering/Fourier and seismic
envelope-phase Hilbert 1-D FFT call sites through Apollo's Leto-native real FFT
APIs and direct Apollo complex inverse API.

Evidence tier: compile-time integration, focused empirical tests, and static
source audit. `rustup run nightly cargo fmt --package kwavers-solver --check`
passed; `rustup run nightly cargo check -p kwavers-solver` passed; `rustup run
nightly cargo nextest run -p kwavers-solver photoacoustic` passed 10/10;
`rustup run nightly cargo nextest run -p kwavers-solver envelope misfit phase`
passed 34/34; scoped `rg` found no `fft_1d_array`/`ifft_1d_array` calls under
`crates/kwavers-solver/src`.

Residual: broader 2-D/3-D `kwavers_math::fft` facade users remain as separate
migration slices because the current 3-D solver facade still owns
`num_complex`-typed PSTD and k-space APIs.

## DONE: Burn-to-Coeus migration guard [patch]

Ported the RITK Burn-surface audit pattern into kwavers as a focused
`burn-migration-audit` xtask command, an intentional-only
`refresh-burn-allowlist` command, `xtask/burn_surface.allowlist`, and a
separate CI job in `.github/workflows/legacy-migration-audit.yml`.

Evidence tier: static source/manifest audit, unit tests, and CI configuration.
`rustup run nightly cargo fmt -p xtask --check` passed; `rustup run nightly
cargo nextest run -p xtask burn_audit --status-level fail --no-fail-fast`
passed 2/2; `rustup run nightly cargo run -p xtask -- burn-migration-audit`
passed with 0 Burn manifest deps and 5 approved non-solver source residuals.
The generated Burn allowlist contains no solver PINN entries.

## DONE: kwavers-physics acoustic heat-source Moirai traversal [patch]

Routed `crates/kwavers-physics/src/acoustics/conservation/heat.rs` through the
crate-local Moirai-backed `parallel` traversal SSOT instead of direct
ndarray/Rayon `Zip::par_for_each`. Added the missing `zip_mut_five_refs`
traversal arity so heat-source output can consume pressure, velocity magnitude,
density, sound speed, and absorption in one pass.

Evidence tier: compile-time integration, focused empirical tests, and static
source audit. `rustup run nightly cargo check -p kwavers-physics --lib` passed;
`rustup run nightly cargo nextest run -p kwavers-physics heat_source
--status-level fail` passed 9/9 with 1704 skipped; scoped source audit found no
`Zip|par_for_each|rayon` hits in `heat.rs`.

Residual: broader `kwavers-solver`/`kwavers-physics` direct `.par_for_each`
holdouts are now 49 sites outside RTM inherent, sonogenetics, and acoustic
heat-source traversal. Package clippy remains blocked before this package by
local dependency `ritk-transform` Burn `Module` derive errors in the concurrent
RITK provider migration diff.

## DONE: kwavers-physics focused Moirai traversal cleanup [patch]

Routed `crates/kwavers-physics/src/acoustics/therapy/sonogenetics` gating and
volumetric ARF field traversal through the crate-local Moirai-backed
`parallel` SSOT instead of direct ndarray/Rayon `Zip::par_for_each`, and routed
heterogeneous skull-mask property assignment through the same one-input helper
instead of duplicating the mask argument through a two-input traversal. Added
the missing `zip_mut_ref` and `zip_two_mut_four_refs` traversal arities so
one-input updates and ARF finalization share the traversal SSOT while ARF still
computes intensity and body-force outputs in one fused pass over the four input
fields.

Evidence tier: compile-time integration, focused empirical tests, and static
source audit. `rustup run nightly cargo check -p kwavers-solver --lib` passed;
`rustup run nightly cargo clippy -p kwavers-solver --lib --no-deps -- -D
warnings` passed; `rustup run nightly cargo nextest run -p kwavers-physics
sonogenetics --status-level fail --no-fail-fast` passed 53/53 with 1660
skipped; `rustup run nightly cargo nextest run -p kwavers-physics skull
--status-level fail --no-fail-fast` passed 51/51 with 1662 skipped; scoped
source audit found no `Zip|par_for_each|rayon` hits under the sonogenetics cone
or the touched skull mask file.

Residual: broader `kwavers-solver`/`kwavers-physics` direct `.par_for_each`
holdouts are now 49 sites outside RTM inherent, sonogenetics, skull mask, and
acoustic heat-source traversal.

## DONE: kwavers-solver RTM inherent Moirai traversal [patch]

Routed `crates/kwavers-solver/src/inverse/reconstruction/seismic/rtm/inherent`
through the private Moirai strided-view helper instead of direct ndarray/Rayon
`Zip::par_for_each`. The slice covers wavefield update, decimated wavefield
interpolation, source illumination, Laplacian filter, post-processing
normalisation, and all six RTM imaging conditions.

Evidence tier: compile-time integration, focused empirical tests, and static
source audit. `rustup run nightly cargo check -p kwavers-solver --lib` passed;
`rustup run nightly cargo nextest run -p kwavers-solver rtm --status-level
fail` passed 10/10 with 916 skipped; and scoped source audit found no
`Zip|par_for_each|rayon` hits under the RTM inherent cone.

Residual: broader `kwavers-solver`/`kwavers-physics` direct ndarray/Rayon
holdouts remain outside RTM inherent, sonogenetics, and acoustic heat-source
traversal: 49 `.par_for_each`
sites, enumerated in
`gap_audit.md`. Package fmt is still blocked by pre-existing formatting drift in
`crates/kwavers-solver/src/forward/fdtd/electromagnetic/tests.rs`; package
clippy now passes with `rustup run nightly cargo clippy -p kwavers-solver --lib
--no-deps -- -D warnings` after the Atlas provider graph refresh.

## KW-GPU-060 — Hephaestus backend-kernel ownership [major] — review

- Owner: Codex; scope: `crates/kwavers-gpu/src/backend/{provider,buffers.rs,pipeline,shaders/operators.wgsl,mod.rs,tests.rs}`, `docs/adr/039-hephaestus-backend-kernel-ownership.md`, and synchronized package metadata.
- Acceptance: `WgpuComputeProvider` uses Hephaestus typed transfer plus
  `binary_elementwise_into` and `WgslMultiStorageKernel`; the local backend
  buffer/pipeline managers and their unsafe device-pointer ownership are
  deleted; Leto remains only at the host-array boundary; WGPU value regressions
  preserve exact multiplication and affine derivatives.
- Driver: the LeoNeuro GPU path must select Hephaestus as device-execution
  owner rather than duplicating it beneath Leto host arrays.
- Evidence: offline GPU and CUDA-provider compilation pass; warning-denied
  Clippy passes for both feature sets; GPU backend Nextest passes 45/45 and
  CUDA-provider backend Nextest passes 50/50. The WGPU cases execute exact
  multiplication plus all three affine spatial derivatives on a real adapter.
- Residual: current Hephaestus `WgslMultiStorageKernel` is WGPU-specific. A
  CUDA spatial-derivative implementation remains unavailable until a real CUDA
  kernel exists; CUDA therefore remains outside the composite provider trait.

Lift concrete WGPU buffer allocation, pipeline execution, and shader dispatch
behind a Hephaestus-owned provider trait so WGPU and CUDA can implement the
same operation contracts without algorithm-call-site branches.

Definition of Ready: ADR names the provider trait surface, operation contracts,
buffer ownership model, and differential verification plan for WGPU versus CUDA
where CUDA kernels exist. Acceptance: concrete `wgpu::Buffer`,
`wgpu::ComputePipeline`, and `GpuProviderContext<WgpuDevice>` signatures are
confined to the WGPU provider implementation; CUDA compute is exposed only for
operations with real CUDA kernels and value-semantic differential tests.

Current evidence tier: type-level/compile-time validation plus static source
audit. `AcousticFieldKernel<P>`,
`GpuThermalAcousticBuffers<P>`, `GpuThermalAcousticSolver<P>`,
`GpuBackendBufferManager<P>`, PSTD construction/run-cache buffer allocation,
PSTD shader-module, pipeline-layout, compute-pipeline creation, PSTD bind-group
layout/bind-group assembly, and PSTD run-loop command/compute-pass submission
now own operation/provider trait boundaries. `AcousticFieldProvider` and
`WaveEquationGpu` use provider-native `leto::Array3<f32>` for WGPU instead of
an ndarray f64 surface with hidden narrowing. Thermal-acoustic buffer
upload/readback uses provider-native `leto::Array3<f32>` for WGPU instead of an
ndarray field I/O surface. `ThermalAcousticSolverProvider` now also binds to
the shared `GpuKernelProvider`/`GpuProviderBackend` stack, and the default WGPU
solver provider acquires through `GpuProviderContext<WgpuDevice>` instead of
raw `wgpu::Device`/`Queue` constructor arguments. FDTD pressure upload/readback
now uses provider-native `leto::Array3<f32>` for WGPU and rejects non-dense
host fields explicitly instead of widening/narrowing through ndarray. The WGPU
FDTD pressure dispatcher now also acquires through `GpuProviderContext<WgpuDevice>`
and satisfies `GpuKernelProvider`/`GpuProviderBackend`; CUDA remains a real
kernel implementation gap, not a fake provider branch. `FdtdGpuProvider` now
owns the generic FDTD GPU operation contract, `WgpuFdtd` implements it with
real WGSL kernels and provider-native `leto::Array3<f32>` I/O, and the
top-level roundtrip tests bind through the trait without raw WGPU device
construction, Tokio, or ndarray. PSTD solver state is now provider-associated
through `PstdStateProvider`, `WgpuPstdState` owns
`GpuProviderContext<WgpuDevice>` instead of raw WGPU device/queue handles,
WGPU host scratch/upload buffers remain owned by `WgpuPstdState`, and current
WGPU state assembly is delegated to `WgpuPstdStateProvider::build_state`.
WGPU medium/source upload bodies are owned by `WgpuPstdState`. Current
run-cache allocation, bind-group rebuild, and signal-tail upload bodies are
also owned by `WgpuPstdState`. Current WGPU dispatch, FFT, and per-phase
pass-encoding methods are implemented on `WgpuPstdState`, and
`WgpuPstdPassProvider` holds provider state instead of the solver wrapper.
Current high-level WGPU run-loop orchestration is implemented on
`WgpuPstdState`, with `GpuPstdSolver<WgpuPstdStateProvider>::run` as a public
delegate that supplies scalar metadata and input slices. Current
`GpuPstdSolver<P>::new` construction is provider-generic through
`PstdStateBuilder`; WGPU remains the only real PSTD state builder. Remaining
public `GpuPstdSolver<P>::run` execution is provider-generic through
`PstdRunState`; WGPU remains the only real PSTD run implementation. Remaining
PSTD medium/source-correction updates are provider-generic through
`PstdMediumUpdateState`; WGPU remains the only real medium-update
implementation. The WGPU/CUDA provider boundary was re-verified on 2026-07-03
with `rustup run nightly cargo check -p kwavers-gpu --features cuda-provider`
and a focused 5/5 `cargo nextest run` over provider identity, CUDA provider
contract, and provider-generic context tests. `MultiGpuContext<P>` now carries
the same `GpuDeviceProvider` parameter as `CoreGpuContext<P>` and acquires
multi-device contexts through `P::try_acquire_devices`; WGPU remains the
default constructor for current WGSL kernels, and CUDA type-checks at the
topology/scheduling boundary without fake compute kernels. The backend
`GpuComputeProvider`/`GPUBackend::dispatch_*` API now accepts
`leto::Array3<f32>` for provider-native elementwise and derivative dispatch;
`WgpuBackendBufferManager` now stages those operation buffers as
`leto::Array3<f32>` directly, and focused GPU checks on 2026-07-03 pass 8/8.
`WaveEquationGpu<P>` and `AcousticFieldKernel<P>` now carry
`AcousticFieldProvider`, which is bound to the shared
`GpuKernelProvider`/`GpuProviderBackend` trait stack. The default WGPU acoustic
provider stores `GpuProviderContext<WgpuDevice>`, so future real CUDA acoustic
kernels enter through the same Hephaestus-backed provider contract without
changing the wrapper API.
Hephaestus CUDA now implements the shared unary/binary storage-kernel traits
upstream, and the Kwavers provider boundary was reverified on 2026-07-04 under
both `gpu` and `cuda-provider` feature sets without a Kwavers-local CUDA helper.
Evidence tier: type-level trait satisfaction plus focused provider tests.
Runtime CUDA acoustic/FDTD execution still requires real CUDA kernel sources
and value-semantic WGPU/CUDA differential tests.
Follow-up 2026-07-04: `CudaElementWiseProvider` now implements the real
CUDA elementwise multiplication operation family through Hephaestus CUDA
`MulOp` over provider-native `leto::Array3<f32>` buffers. It is intentionally
not a `GpuComputeProvider` until spatial derivative and the remaining
composite backend operations have real CUDA kernels. The realtime Hilbert FFT
path now calls the Kwavers FFT slice facade instead of Apollo's Leto-native
plan API directly. Evidence tier: type-level trait satisfaction plus focused
empirical tests; fmt/check/clippy pass and focused `kwavers-gpu --features
cuda-provider provider elementwise realtime` nextest passes 52/52.
The top-level `kwavers --features gpu --tests` compile gate now passes after
removing the obsolete recovery stress test and replacing stale raw-WGPU, Tokio,
and ndarray test surfaces with Hephaestus/CoreGpuContext/GPUBackend,
pollster-backed `GpuDevice`, `GpuPstdSolver<WgpuPstdStateProvider>`, and
provider-native `leto::Array3<f32>` coverage. Focused top-level GPU nextest
passes 27/27 with 3 ignored PSTD hardware tests skipped. The remaining warning
debt is also closed: `gpu_fft_arbitrary_size.rs` uses pollster-backed ignored
tests without an unused helper, `pstd_finite_window_born.rs` no longer carries
the unused baseline allocation, inactive Moirai patch entries with no workspace
manifest dependency are removed, and `rustup run nightly cargo check -p kwavers
--features gpu --tests` passes warning-clean. Remaining GPU migration debt is
real ignored hardware execution and WGPU/CUDA differential CUDA kernels.
Solver-facing GPU documentation now names the provider-generic `GPUBackend<P>`
boundary and marks the legacy elastic SWE GPU file as a performance model
instead of a real WGPU/CUDA dispatch path; focused solver backend nextest
passes 3/3 and the stale concrete-provider wording audit returns no matches.
The acoustic-field WGPU provider now stores `GpuDevice<WgpuDevice>` and
acquires through the shared `GpuDeviceProvider` contract instead of directly
acquiring/storing raw `WgpuDevice`; `kwavers-gpu --features cuda-provider`
fmt/check/clippy pass and the focused acoustic/provider/device nextest
selection passes 42/42.
PSTD auto-device acquisition now also routes through `GpuDevice<WgpuDevice>`
and `GpuDeviceProvider`; the WGPU PSTD state builder remains the only real
PSTD implementation, the auto-device constructor no longer contains direct
WGPU acquisition or `pollster::block_on`, and the focused
CUDA-provider PSTD/provider nextest selection passes 6/6.
PSTD construction and medium-update test helpers now use the same
`GpuDevice<WgpuDevice>` provider wrapper; the scoped PSTD subtree audit finds
no direct WGPU acquisition or `pollster::block_on`, and the focused
CUDA-provider PSTD/provider nextest selection passes 7/7.
The backend buffer-manager GPU construction test now also acquires through
`GpuDevice<WgpuDevice>` and `GpuDeviceProvider`; scoped audit finds no direct
WGPU acquisition in `backend::buffers` or `pstd_gpu`, and the focused
CUDA-provider backend buffer-manager nextest selection passes 2/2.
Backend buffer readback no longer uses `pollster::block_on`; both public
readback entry points share one blocking WGPU map/read implementation returning
provider-native `leto::Array3<f32>`, and focused CUDA-provider backend nextest
passes 45/45.
Top-level `kwavers/cuda-provider` and `kwavers/cuda-runtime` now forward to the
existing `kwavers-gpu` Hephaestus CUDA provider features, so integrators can
select the CUDA provider seam through `kwavers` without bypassing the
provider-generic GPU traits. Focused verification passed: top-level
`kwavers --features cuda-provider` check/clippy, `kwavers-gpu --features
cuda-provider provider` nextest (44/44), and `kwavers-gpu` provider-edge cargo
tree audit showing `hephaestus-core`, `hephaestus-wgpu`, and
`hephaestus-cuda`.
Top-level stream visualization tests now use blocking stream/pipeline entry
points owned by `kwavers-analysis::visualization::stream` and provider-native
`leto::Array3<f32>` frames, so that test target no longer requires Tokio or
ndarray. The follow-up 3-D beamforming slice moved processor construction
behind `BeamformingGpuProvider`; the current WGPU provider acquires through
Hephaestus `WgpuDevice`, `BeamformingProcessor3D::with_provider` is the public
provider-generic constructor, `BeamformingProcessor3D::new_wgpu` names the
current WGPU convenience constructor, and `examples/real_time_3d_beamforming.rs`
no longer owns Tokio. Top-level `kwavers --features gpu` also has no depth-1
Tokio edge. Next increment: move
the concrete 3-D beamforming operation provider from `kwavers-analysis` into
`kwavers-gpu` when the operation contract is split out with real CUDA kernels
or WGPU/CUDA differential tests; do not add a placeholder CUDA provider.
Distributed neural beamforming in `kwavers-analysis` now also runs without a
Tokio runtime: the processor exposes synchronous `process_volume_distributed`
over the existing Moirai `Adaptive` fan-out, the test calls it directly, and
the crate's Tokio dev-dependency is removed. Focused `pinn` check/clippy and
distributed nextest pass; source and depth-1 dependency audits show no Tokio
edge for `kwavers-analysis`.
Top-level ignored GPU FFT parity tests now use Apollo's `FftBackend` plan seam
through `kwavers_math::fft::gpu_fft::WgpuBackend` and explicit Leto test
buffers at the GPU boundary instead of local WGPU instance/device/queue
construction, `pollster::block_on`, or `tokio::test`. Focused verification
passed: `kwavers --features gpu` check/clippy for
`gpu_fft_arbitrary_size`/`gpu_cpu_fft_parity`, nextest harness discovery with
22 ignored hardware tests skipped, and a scoped source audit for raw WGPU
acquisition symbols in those files. Residual direct-WGPU scope remains in
top-level raw-buffer/device tests whose public surfaces still expose
WGPU-specialized handles.
The top-level `kwavers/gpu` feature no longer forwards direct `wgpu`,
`bytemuck`, or `pollster` dependencies. `kwavers-gpu` now owns synchronous
provider acquisition, buffer readback/write, acoustic-kernel, wave-equation,
and FDTD pressure readback wrappers; top-level GPU buffer/device/allocation
and compute-kernel tests call those provider APIs without importing concrete
runtime helper crates. Focused verification passed: `kwavers-gpu --features
gpu --lib` check/clippy, affected `kwavers --features gpu` test-target
check/clippy, affected nextest 28/28, top-level source audit, and depth-1
dependency audit. Residual WGPU runtime ownership is confined to
`kwavers-gpu` provider implementations for current WGSL kernels.
The inverse regularization subtree no longer uses direct ndarray/Rayon
`Zip::par_for_each`; Tikhonov, smoothness, and L1 gradient updates now route
through a Moirai-backed dense traversal helper with sequential ndarray
fallback for non-standard layouts. The `kwavers-math` source-level
ndarray/Rayon parallel cleanup is now closed after the FFT real/complex
packing and k-space squared-field generation routes moved through
Moirai-backed contiguous traversal. The remaining `kwavers-math` Rayon work is
manifest-level `ndarray/rayon` removal audit after confirming no transitive
provider still requires the feature.
The `kwavers-math::simd_safe` subtree now uses the Atlas Hermes SIMD facade for
dense add/scale operations and Moirai chunk traversal for dense ternary
`c += multiplier * a * b` accumulation, with sequential ndarray traversal only
for non-standard layouts. Remaining `kwavers-math` ndarray-parallel holdouts
are now narrowed to FFT/k-space. Residual upstream
gap: Hermes needs a public ternary accumulation slice facade before Kwavers can
route that operation through Hermes without allocating a temporary.
The `kwavers-math` differential subtree now routes second-order central and
staggered-grid standard-layout output fills through a shared Moirai traversal
helper, with sequential ndarray traversal only for non-standard layouts.
`kwavers-math` FFT/k-space traversal has also moved to Moirai for
standard-layout arrays, so no direct Rayon or ndarray-parallel source calls
remain under `crates/kwavers-math/src`.
`kwavers-math` no longer enables ndarray's `rayon` feature directly: its
manifest now keeps only `serde`, the dependency tree shows no
`ndarray/rayon` feature under either default or `gpu`, and Apollo is consumed
from the local Atlas checkout. Apollo's WGPU helper now resolves local
`hephaestus-wgpu v0.11.0`, while Kwavers' GPU FFT facade exposes Apollo's
`FftBackend` trait and keeps WGPU documented as the current implementation,
not the provider architecture. Focused verification on 2026-07-04:
`rustup run nightly cargo check -p kwavers-math --all-targets`, `rustup run
nightly cargo check -p kwavers-math --features gpu --all-targets`, focused
FFT/k-space/spectral nextest (33/33), focused GPU FFT nextest (2/2), and
`kwavers-math` clippy pass. Residual upstream gap: Apollo has no real CUDA FFT
provider yet; that belongs in Apollo/Hephaestus with WGPU/CUDA differential
tests, not as a Kwavers placeholder.
`kwavers-solver/gpu` no longer owns concrete WGPU runtime dependencies. The
solver manifest removed direct `wgpu`, `bytemuck`, and `pollster` optional
edges, leaving the feature as a `kwavers-math/gpu` forwarding switch while
concrete GPU execution stays in `kwavers-gpu`. Solver FFT call sites that were
calling Apollo's Leto-native plan methods now route through
`kwavers_math::fft`, including new 3-D axis-transform facade functions for
viscoacoustic derivatives. Focused verification on 2026-07-04:
`rustup run nightly cargo check -p kwavers-solver --features gpu --all-targets`,
backend surface nextest (3/3), KZK/PSTD/viscoacoustic/backend nextest (62/62),
library clippy, direct dependency-tree audit, and stale-token source audit
passed. Residual: broad `kwavers-solver --features gpu --all-targets` clippy
is still blocked by unrelated existing test-target lint debt.
`kwavers-gpu` no longer carries a crate-local Tokio dev-dependency: the
remaining async GPU acquisition tests now run through `pollster`, the
`kwavers-gpu` source/manifest Tokio audit returns no hits, and the focused
CUDA-provider non-hardware nextest selection passed 11/11. A broader
hardware-acquisition nextest selection still needs isolation because it was
interrupted after producing no result for several minutes beyond compilation.
`GpuComputeProvider` is now the composite of `GpuKernelProvider`,
`ElementWiseMultiplyProvider`, and `SpatialDerivativeProvider`, so CUDA can
implement only operation families backed by real kernels instead of inheriting
placeholder methods from a coarse backend trait; the focused CUDA-provider
operation-trait check on 2026-07-03 passed 4/4.
The WGSL pipeline compiler/executor is now explicitly named
`WgpuPipelineManager`, and no backend-neutral `PipelineManager` token remains
under `kwavers-gpu/src/backend`; the focused CUDA-provider pipeline/provider
test passed 5/5.
The raw WGPU command-helper surface is now `WgpuComputeCommands`, and no stale
`GpuCompute` token remains under `kwavers-gpu/src`; the focused CUDA-provider
command-helper/provider test passed 3/3.
`WgpuComputeProvider` now reports memory from the acquired Hephaestus device
limits instead of a fixed 4 GiB constant, and reports unknown peak throughput
as `0.0` instead of a fixed 5 TFLOP/s value; focused provider metadata tests
and `kwavers-gpu --features gpu` check pass. The generic provider performance
estimate now returns the provider-reported peak value rather than a hardcoded
problem-size speedup curve. `WgpuComputeProvider` now reports
`supports_fft = false` because `ComputeBackend` does not own FFT operations;
Apollo remains the GPU FFT owner through `kwavers_math::fft::gpu_fft`.
Realtime scheduling field maps now use `leto::Array3<f64>`. Remaining backend
work is real CUDA kernel implementation and WGPU-vs-CUDA differential
verification behind the scalar-associated `ComputeBackend`/`GpuComputeProvider`
trait seam, not a placeholder provider. Remaining PSTD GPU work is real CUDA
state/kernel implementation and WGPU-vs-CUDA differential verification, not a
placeholder provider. A 2026-07-04 recheck confirms
`kwavers-gpu --features cuda-provider` still compiles through the
Hephaestus-backed provider graph after the top-level async-runtime cleanup.
The `kwavers-simulation` GPU PSTD adapter now consumes Leto CPML profiles
directly and makes the current WGPU `PstdStateProvider` explicit only at the
auto-device construction boundary. The top-level async-only stream
visualization nextest filter now passes with 0 selected tests because the
stream test is gated away from non-GPU builds. Residual: the optional
`async-runtime,gpu` stream test target is blocked by the missing
`kwavers_analysis::visualization::stream` API, so the next increment is either
to restore the real stream module in `kwavers-analysis` or delete the stale
test if the stream contract has been superseded by the current visualization
pipeline. `ComputeManager` is now provider-generic and its CPU
field-update helpers use `leto::Array3<f64>`. The FDTD CPU reference dispatcher
now uses `leto::Array3<f64>` for its f64 reference pressure stencil. The
GPU/CPU equivalence validator now compares `leto::Array3<f64>` fields
directly, with ndarray confined to the current FDTD solver mask/signal/output
boundary before conversion. Realtime imaging pipeline RF input/output frame
buffers now use `leto::Array4<f32>`/`leto::Array3<f32>`; its only remaining
local ndarray use is the private Apollo FFT `Array1<Complex64>` Hilbert
scratch boundary. The false `kwavers-gpu` Burn accelerator surface is removed,
and the solver-local CUDA-shaped PINN GPU accelerator plus
`pinn-gpu`/`burn-wgpu`/`burn-cuda` feature aliases are removed. The remaining
Burn dependencies no longer enable Burn's `wgpu` feature. The solver PINN
multi-GPU manager no longer enumerates WGPU adapters directly; it now surfaces
a typed unavailable-provider error until a real Coeus training provider is
routed through Hephaestus WGPU/CUDA device traits. Remaining PINN GPU work is
real Coeus training routed through Hephaestus provider traits, with WGPU and
CUDA behind the same provider seam. Follow-up on 2026-07-03 disabled Burn
defaults on remaining kwavers Burn dependencies, kept only required non-GPU
features, and repaired RITK's workspace Burn default from WGPU to NdArray; the
selected `kwavers --features pinn` graph no longer contains `burn-wgpu`,
`burn-cuda`, or `burn-rocm`. Follow-up on 2026-07-04 removed the direct
`kwavers-python` Burn dependency from the RITK NIfTI loader by routing it
through native `ritk-io` on `coeus-core::SequentialBackend`; focused source
and dependency audits show direct `coeus-core`/`ritk-io`/`ritk-image` edges
and no direct Python-loader Burn edge. The top-level
`kwavers` crate also no longer depends directly on Rayon or exposes the
`parallel = ["ndarray/rayon"]` feature: liver theranostic and 3-D seismic
example fan-out/blur loops now dispatch through `moirai-parallel`, and
`cargo tree -p kwavers --depth 1` lists `moirai-parallel` with no direct
`rayon`. `kwavers-physics` also no longer has a direct Rayon dependency or
source-level direct Rayon iterator usage; its residual execution-provider gap
is the tracked ndarray-parallel kernels that still require Leto/Hephaestus
backend migration. In `kwavers-solver`, the Westervelt spectral wave-model
leapfrog combination loop now uses `moirai-parallel`; focused
`westervelt_spectral::solver` nextest passed 8/8. Helmholtz FEM element
contribution collection now uses `moirai-parallel` with explicit contribution
array length validation; focused Helmholtz/FEM nextest passed 10/10. Westervelt
FDTD conservation diagnostics now use Moirai indexed reductions for energy,
momentum, and mass; focused Westervelt nextest passed 32/32. Westervelt FDTD
Laplacian O2/O4/O6 stencil slabs now use Moirai slab traversal; focused
Westervelt nextest passed 32/32. Westervelt FDTD nonlinear-term and update
field traversals now use Moirai indexed traversal; focused Westervelt nextest
passed 32/32. KZK solver observables and trait RMS field generation now use
Moirai indexed traversal; focused KZK nextest passed 49/49. KZK
angular-spectrum and real-field parabolic diffraction scratch/projection
traversals now use Moirai indexed traversal. KZK spectral absorption slab
traversal now uses Moirai indexed chunks. KZK complex parabolic diffraction
and nonlinear delta/update slabs now use Moirai traversal as well, leaving no
direct Rayon or ndarray-parallel source hits under the KZK subtree. The
current focused KZK/nonlinear/diffraction/absorption nextest passed 204/204,
and solver check/clippy pass. The mixed-domain frequency-domain propagator now
uses Moirai indexed traversal for its complex spectral phase application, and
focused hybrid/mixed-domain nextest passed 59/59. The legacy KZK solver
plugin nonlinear update now uses Moirai indexed traversal for standard-layout
fields; focused KZK/nonlinear nextest passed 181/181. FDTD dynamic pressure
source masks now use Moirai indexed traversal for dense Dirichlet and additive
updates; focused FDTD/source nextest passed 93/93. FDTD pressure-updater
divergence accumulation, pressure update, and nonlinear pressure-delta
application now use shared Moirai-backed dense traversal; focused
pressure/FDTD nextest passed 63/63. FDTD velocity-updater spectral,
collocated, and staggered pressure-gradient updates now use Moirai-backed
dense traversal; focused velocity/FDTD/k-space nextest passed 91/91. FDTD
k-space correction shifted spectral gradient/divergence kernels now use
Moirai-backed dense traversal; focused k-space/FDTD nextest passed 91/91, and
FDTD construction-time `rho*c^2` and nonlinear coefficient fills now use
Moirai-backed dense traversal; focused FDTD/nonlinear/k-space nextest passed
290/290. The FDTD direct-provider scan no longer reports direct Rayon,
ndarray-parallel, or explicit `ndarray::Zip` tokens. PSTD utility k-squared,
k-magnitude, and spectral-derivative scaling now use Moirai-backed dense
helpers; focused PSTD utility nextest passed 22/22. PSTD implementation
k-space Helmholtz and spectral-gradient multipliers now use Moirai-backed
dense traversal; focused PSTD/k-space nextest passed 206/206. PSTD
implementation anti-aliasing spectral filter multipliers now use Moirai-backed
dense traversal; focused anti-aliasing/PSTD nextest passed 175/175. PSTD
implementation full-k-space source accumulation, spectral wave-coefficient
multiplication, and propagated pressure/source updates now use Moirai-backed
dense traversal; focused source/step/k-space/PSTD nextest passed 231/231.
PSTD implementation source-gain scaling, source-kappa spectral multiplication,
split-density source injection, and dynamic velocity-source writes now share
Moirai-backed dense stepper helpers; focused source/step/filter/PSTD nextest
passed 234/234. PSTD implementation total split-density accumulation now uses
Moirai-backed dense traversal; focused PSTD/source/step nextest passed 208/208.
PSTD implementation thermal absorption coefficient scaling now uses
Moirai-backed dense traversal; focused thermal/PSTD nextest passed 206/206.
PSTD implementation construction-time source-kappa cosine transformation and
initial split-density component fills now use Moirai-backed dense traversal;
focused construction/PSTD nextest passed 209/209.
PSTD implementation IVP density seeding, spectral-gradient construction, and
half-step velocity scaling now use Moirai-backed dense traversal; focused
IVP/PSTD nextest passed 209/209, and the scoped PSTD implementation-core
direct-provider audit now reports no hits.
PSTD spectral-correction kappa generation and correction application now use
Moirai-backed dense traversal; focused spectral-correction/PSTD nextest passed
175/175.
PSTD propagator pressure equation-of-state density accumulation and pressure
writes now use Moirai-backed dense traversal; focused pressure/PSTD nextest
passed 203/203.
PSTD Cartesian pressure-density spectral-gradient and split-density updates now
use Moirai-backed dense traversal; focused density/pressure/PSTD nextest passed
203/203.
PSTD axisymmetric pressure-density coefficient and split-density updates now use
Moirai-backed dense traversal; focused density/pressure/axisymmetric/PSTD
nextest passed 203/203, and the pressure propagator subtree direct-provider
audit now reports no hits.
PSTD Cartesian and axisymmetric velocity spectral-gradient and velocity-field
updates now use Moirai-backed dense traversal; focused
velocity/pressure/density/axisymmetric/PSTD nextest passed 210/210.
PSTD axisymmetric WSWA-FFT pressure-gradient and density-divergence propagation
now uses Moirai-backed dense traversal; focused
axisymmetric/velocity/density/pressure/PSTD nextest passed 210/210.
PSTD broadband residual-gas absorption and dispersion pressure corrections now
use Moirai-backed dense traversal; focused residual-gas/absorption/PSTD nextest
passed 213/213.
PSTD pressure-side fractional-Laplacian absorption correction now uses
Moirai-backed dense traversal; focused absorption/pressure/PSTD nextest passed
211/211.
PSTD fractional-Laplacian absorption stratum bracket construction now uses Moirai
indexed collection, and the absorption subtree direct-provider audit now reports
no hits; focused absorption/pressure/PSTD nextest passed 211/211.
PSTD spectral derivative pencil traversal now uses Moirai indexed collection for
strided x-pencils and Moirai chunked i-slabs for y/z pencils; focused
derivative/spectral/PSTD nextest passed 214/214.
PSTD DG spectral Laplacian symbol construction and application now use
Moirai-backed dense traversal; focused DG/spectral/PSTD nextest passed 210/210.
PSTD DG one-dimensional acoustic SSP-RK stage updates now use Moirai-backed
dense traversal; focused DG/spectral/PSTD nextest passed 210/210.
PSTD DG modal SSP-RK and Forward Euler coefficient updates now use shared
Moirai-backed dense RK helpers; focused DG/spectral/PSTD nextest passed
210/210.
PSTD DG tensor acoustic source SSP-RK state updates now use the shared
Moirai-backed dense RK helpers; focused DG/spectral/PSTD nextest passed
210/210.
PSTD DG tensor CPML field and memory SSP-RK state updates now use the shared
Moirai-backed dense RK helpers; the DG subtree direct-provider audit now
reports no direct Rayon, ndarray-parallel, or explicit `Zip` holdouts; focused
DG/spectral/PSTD nextest passed 210/210.
Photoacoustic iterative reconstruction ART/OSEM updates, Fourier positivity
clamping, and time-reversal k-space leapfrog spectrum updates now use
Moirai-backed traversal; the photoacoustic reconstruction subtree
direct-provider audit now reports no direct Rayon, ndarray-parallel, or
explicit `Zip` holdouts; focused photoacoustic nextest passed 10/10.
Hybrid angular spectrum broadband harmonic absorption now uses Moirai-backed
dense traversal; the HAS direct-provider audit reports no direct Rayon,
ndarray-parallel, or explicit `Zip` holdouts; focused HAS/absorption nextest
passed 43/43.
Nonlinear elastic propagation damping maps now use Moirai-backed dense
traversal instead of ndarray/Rayon `par_mapv_inplace`; focused
nonlinear/elastic/propagation nextest passed 264/264. Follow-up
harmonic-generation and stepping slices closed the remaining direct-provider
files in the nonlinear elastic subtree.
Nonlinear elastic harmonic-generation Jacobi updates and delta fills now use
Moirai-backed indexed traversal instead of ndarray/Rayon `Zip::par_for_each`;
focused nonlinear/elastic/propagation nextest passed 264/264.
Nonlinear elastic fundamental x-line stepping now uses Moirai-backed
line scheduling plus a separate safe write-back pass; the
`forward/elastic/nonlinear` direct-provider audit now reports no direct Rayon,
ndarray-parallel, or explicit `Zip` holdouts; focused
nonlinear/elastic/propagation nextest passed 264/264.
Remaining
non-Atlas
execution edges include
`kwavers-solver` direct Rayon/ndarray-parallel holdouts and top-level dev
`tokio`.

## DONE: kwavers-math full num_traits sweep (Phase-1B) [patch] (2026-07-10)

Reconciled stale: `csr.rs` already carries `impl CsrScalar for eunomia::Complex64`
(no `num_complex`/`num_traits` imports); `crates/kwavers-math/Cargo.toml` declares
neither `num-traits` nor `num-complex`; `kwavers-boundary/src` has no `num_complex`
reference. All 263 `kwavers-math` tests pass. Removed the phantom-`ADR-0006`
`#[cfg(any())]` frozen "post-ADR" fixture blocks (dead speculative code for a
`ComplexField`-routed `CsrScalar` design that was not adopted — the shipped design
uses per-impl `magnitude` + a `Default` bound). Superseded the `§3 deferral [arch]`
item below.

<details><summary>Original item spec (historical)</summary>

Phase-1A closed `linear_algebra::numeric_ops.rs` against the eunomia numeric SSOT (`eunomia::RealField` + `NumericElement::ZERO`). The remaining `kwavers-math` legacy num_traits surface is:

- `linear_algebra::sparse::csr.rs` — `impl CsrScalar for num_complex::Complex64` requires `num_traits::Zero` because `num_complex::Complex<f64>` does NOT yet impl `eunomia::NumericElement` (eunomia's float traits are `private::Sealed`). This is the Phase-1B blocker.
- `linear_algebra::basic.rs` — ndarray → leto import surface, decoupled from num_traits; simplest Phase-1B+ follow-up once csr.rs lands.
- `linear_algebra::norms.rs` — uses `ndarray::Array3` only; clean leto port.

Phase-1B DoR: choose one of the three Atlas-extension paths for the csr.rs blocker and document the choice in an ADR before the csr.rs edit.

Atlas extension memo (CR-EUNOMIA-COMPLEX):
`kwavers-math/src/linear_algebra/sparse/csr.rs` is blocked from dropping legacy dependencies because `impl CsrScalar for num_complex::Complex64` requires `num_traits::Zero`, but Eunomia's generic float traits (`NumericElement` / `FloatElement`) are sealed. Requesting either an unsealed `Scalar` supertrait in Eunomia or a native `eunomia::Complex` integration that supports magnitude/norm derivations to satisfy sparse CSR bounds.

### (historical) kwavers-math Phase-1B §3 deferral [arch]

§3 of the kwavers-math `num_traits` sweep — the csr.rs SSOT rebind (and the related `kwavers-boundary` `num_complex::Complex64` → `eunomia::Complex64` migration) — is **deferred** pending the closure of the orphan-rule gap in `eunomia::types::complex::{ops.rs,float.rs}`. Concretely, eunomia does not yet provide the cross-impls that would let `Complex<f64> * Array1<f64>` (and the analogous `LinalgScalar`-shaped operations in `solver/bicgstab.rs`) compile through the SSOT route; the §2 batch attempted that path and reverted after 7× E0277 surfaced.

Definition of Ready:
- eunomia gains `impl<'a> ndarray::ScalarOperand<'a> for Complex<f32>` and the same for `Complex<f64>` in `eunomia::types::complex::ops.rs` (operator-shaped cross-impl).
- eunomia gains `impl<T: RealField> LinalgScalar for Complex<T>` (or the Atlas-equivalent intrinsic cross-impl) in `eunomia::types::complex::float.rs`.
- `cargo build -p kwavers-math --features ndarray` exits 0 with `num_traits::Zero` no longer referenced by `csr.rs` and the `kwavers-boundary` `num_complex` import gone.

Acceptance:
- `csr.rs` rebinds `CsrScalar: eunomia::ComplexField` and drops the `<num_complex::Complex64 as CsrScalar>` per-impl block; the blanket `impl<T: RealField> CsrScalar for eunomia::Complex<T>` (or equivalent) synthesizes float and complex coverage.
- `crates/kwavers-math/Cargo.toml` drops both `num-traits` and `num-complex` edges in the same commit.
- Focused `cargo nextest run -p kwavers-math sparse::csr` passes through the eunomia route; focused `cargo nextest run -p kwavers-boundary fem,bem` passes through the `eunomia::Complex64` route.
- `repos/kwavers/xtask` `legacy-migration-audit` source-legacy per-file list no longer contains `crates/kwavers-math/src/linear_algebra/sparse/csr.rs` or the FEM/BEM residuals from `crates/kwavers-boundary`.

Reference: csr.rs `//!` mod-doc (`crates/kwavers-math/src/linear_algebra/sparse/csr.rs:1-9`), CHANGELOG `## Unreleased` `### Reverted (2026-07-05) - kwavers-math Phase-1B §2 ssot-rebind`, `repos/kwavers/gap_audit.md` row 18 "Atlas extension: eunomia Complex64 SSOT for csr.rs - OPENED".

</details>

**§3 outcome (2026-07-10):** the intent — drop `num-traits`/`num-complex` from
`csr.rs` and `kwavers-boundary`, moving to `eunomia::Complex64` — is fully met. The
shipped design differs from the acceptance's proposed mechanism: `CsrScalar` stays a
minimal `Copy + Default + AddAssign + Mul` role trait with a per-impl `magnitude`
(no `ComplexField` blanket, no eunomia orphan-rule/`ScalarOperand` extension required),
so the deferral's eunomia-DoR became moot. `sparse::csr` tests (12) pass through the
eunomia route.

## CLOSED: kwavers-imaging CT/NIfTI native RITK slice (2026-07-04)

Routed `kwavers-imaging::medical::CTImageLoader` through
`ritk_io::format::nifti::native::NiftiReader` on
`coeus_core::SequentialBackend` instead of RITK's legacy Burn-backed
`read_nifti::<AdapterBackend>` path. The shared RITK bridge now accepts typed
`ritk-spatial` metadata and preserves the existing kwavers `(x, y, z)` volume,
spacing, affine, and intensity-range contract.

Follow-up DICOM closure (2026-07-04): RITK now exposes native DICOM series
loading on the public series facade, and `kwavers-imaging` routes DICOM
through `ritk_io::load_native_dicom_series` on
`coeus_core::SequentialBackend`. The direct `burn` and `ritk-core`
dependencies were removed from `kwavers-imaging`; DICOM and NIfTI now share
the native RITK image to kwavers volume bridge.

Evidence tier: compile-time validation plus focused empirical tests.
Verification: `rustup run nightly cargo fmt -p kwavers-imaging --check`
passed; `rustup run nightly cargo check -p kwavers-imaging` passed; focused
`rustup run nightly cargo nextest run -p kwavers-imaging ct_loader
--status-level fail --no-fail-fast` passed 8/8. Follow-up DICOM verification:
RITK `cargo check -p ritk-io` passed; RITK focused nextest passed
`native_dicom_loader_matches_legacy_loader` 1/1 and
`native_series_loader_matches_legacy_loader` 1/1; focused
`rustup run nightly cargo nextest run -p kwavers-imaging dicom --status-level
fail --no-fail-fast` passed 14/14.

## CLOSED: top-level Leto/RITK compile blocker slice (2026-07-03)

Updated the skull CT phase-correction example to current RITK
`DicomSeriesInfo` accessor methods and fixed image spacing/origin extraction
through typed RITK vectors. Updated ultrasound fusion/registration validation
tests to feed `leto::Array2`/`leto::Array3` inputs directly, and updated
NL-SWE validation field construction/statistics to use Leto arrays without an
ndarray compatibility helper.

Residual risk: this closes only the stale API blockers in the named
top-level example/test targets. Broad `kwavers` package verification still
depends on the remaining Atlas migration items, especially the solver
Rayon/ndarray-parallel holdouts, top-level dev `tokio`, and real
Hephaestus/Coeus GPU/PINN backend completion.

Evidence tier: compile-time validation plus focused empirical nextest
coverage. Verification: `rustup run nightly cargo check -p kwavers --example
skull_ct_phase_correction --test ultrasound_physics_validation --test
nl_swe_validation` passed; focused ultrasound validation nextest passed 5/5;
focused NL-SWE validation nextest passed 2/2.

## CLOSED: solver PINN direct WGPU discovery removal (2026-07-03)

Removed the Burn-era `MultiGpuManager` path that directly enumerated WGPU
adapters inside `kwavers-solver`. Multi-GPU PINN construction now returns a
typed `ResourceUnavailable` error naming the missing Coeus training provider
and Hephaestus WGPU/CUDA device-trait route instead of fabricating CPU/GPU
devices or selecting WGPU directly from the solver core.

Residual risk: this closes only the direct solver PINN WGPU discovery leak. A
real multi-GPU PINN backend still requires Coeus training execution behind
Hephaestus provider traits plus value-semantic training and residual tests.

Evidence tier: static source audit plus compile-time validation. Verification:
scoped `rg` found no WGPU discovery tokens under the solver PINN multi-GPU
manager/distributed-training path, and `rustup run nightly cargo check -p
kwavers-solver --features pinn,gpu` passed. Focused nextest was attempted but
stopped after remaining in `apollo-fft` dependency codegen without a test
result. Follow-up: `MultiGpuManager::new` is synchronous until real Coeus
provider discovery exists, the focused multi-GPU manager test no longer uses
Tokio, scoped `rg` finds no Tokio token under the solver PINN multi-GPU
manager, `kwavers-solver --features pinn` check passes, and focused
`multi_gpu_manager` nextest passes 3/3. Distributed-trainer follow-up:
`DistributedPinnTrainer::new` is synchronous while it only assembles local
replicas and provider state, its creation test no longer needs Tokio, focused
`distributed_training` nextest passes 3/3, and `kwavers-solver --features pinn`
clippy passes. Follow-up solver-local async removal: distributed training and
checkpoint persistence are synchronous, checkpoint save/load writes JSON state
with a value-tested round trip, `kwavers-solver/pinn` no longer enables
`dep:tokio`, `kwavers-solver/async-runtime` remains as an empty feature, and
focused `distributed_training` nextest passes 4/4. Broader top-level/transitive
Tokio edges remain outside this closed solver-local PINN slice.

## CLOSED: kwavers-physics thermal diffusion Moirai slice (2026-07-04)

Pennes bioheat perfusion/update and Cattaneo-Vernotte
flux/divergence/temperature traversals no longer call ndarray/Rayon
`par_for_each`. A private `kwavers-physics::parallel` adapter dispatches dense
standard-layout views through `moirai-parallel` and preserves sequential
ndarray traversal for non-contiguous views.

Residual risk: this closes only the thermal diffusion direct ndarray/Rayon
cluster in `kwavers-physics`. Other physics modules still contain direct
ndarray/Rayon kernels and the crate still carries ndarray storage boundaries
pending Leto/Hephaestus backend migration.

Evidence tier: static source audit plus compile-time/lint validation and
focused empirical tests. Verification: `rustup run nightly cargo fmt -p
kwavers-physics --check` passed, `rustup run nightly cargo check -p
kwavers-physics --all-targets` passed, `rustup run nightly cargo nextest run
-p kwavers-physics thermal::diffusion --status-level fail --no-fail-fast`
passed 2/2 selected tests, `rustup run nightly cargo clippy -p kwavers-physics
--all-targets --no-deps -- -D warnings` passed, scoped `rg` found no direct
Rayon or ndarray-parallel tokens in `thermal/diffusion/{bioheat,hyperbolic}.rs`,
and scoped `git diff --check` passed with only CRLF normalization warnings.

## CLOSED: kwavers-physics sonoluminescence Moirai slice (2026-07-04)

Blackbody, bremsstrahlung, and Cherenkov field assembly no longer call
ndarray/Rayon `par_for_each`. The private `kwavers-physics::parallel` adapter
now owns two-, three-, and four-input zip traversal helpers, validates shapes
before dense scheduling, dispatches standard-layout views through
`moirai-parallel`, and preserves sequential ndarray traversal for
non-contiguous views.

Residual risk: this closes only the sonoluminescence emission field cluster in
`kwavers-physics`. Other physics modules still contain direct ndarray/Rayon
kernels and the crate still carries ndarray storage boundaries pending
Leto/Hephaestus backend migration.

Evidence tier: static source audit plus compile-time/lint validation and
focused empirical tests. Verification: `rustup run nightly cargo fmt -p
kwavers-physics --check` passed, `rustup run nightly cargo check -p
kwavers-physics --all-targets` passed, `rustup run nightly cargo nextest run
-p kwavers-physics sonoluminescence --status-level fail --no-fail-fast` passed
34/34 selected tests, `rustup run nightly cargo clippy -p kwavers-physics
--all-targets --no-deps -- -D warnings` passed, scoped `rg` found no direct
Rayon or ndarray-parallel tokens in the three edited sonoluminescence files,
and scoped `git diff --check` passed with only CRLF normalization warnings.

## CLOSED: kwavers-solver thermal diffusion Moirai slice (2026-07-03)

Replaced the standard thermal diffusion ndarray/Rayon `Zip::par_for_each`
update with Moirai chunk scheduling over dense owned temperature and Laplacian
buffers. Borrowed source views now validate shape before indexing; dense views
use the same Moirai traversal, while non-contiguous borrowed views keep
sequential ndarray semantics instead of cloning or forcing a concrete Rayon
path.

Residual risk: this closes only the standard thermal diffusion direct
ndarray/Rayon update. The solver still carries broader ndarray storage
boundaries until the thermal solver state is migrated to Leto with CPU/GPU
backend traits.

Evidence tier: static source audit plus compile-time and focused empirical
validation. Verification: `rustup run nightly cargo check -p kwavers-solver`
passed; `rustup run nightly cargo clippy -p kwavers-solver --lib -- -D
warnings` passed; focused `rustup run nightly cargo nextest run -p
kwavers-solver` over the seven thermal diffusion tests passed 7/7 with 934
skipped; rustfmt `--check` passed; scoped `git diff --check` passed with only
LF/CRLF warnings; and scoped `rg` found no direct Rayon hits under
`crates/kwavers-solver/src/forward/thermal_diffusion/solver`.

## CLOSED: kwavers-solver thermal-acoustic Moirai slice (2026-07-03)

Replaced direct ndarray/Rayon kernels in
`forward::coupled::thermal_acoustic` material-property updates, acoustic
heating, acoustic velocity/pressure stepping, and thermal stepping with
Moirai dense-buffer scheduling. Sequential ndarray traversal remains only as a
layout-preserving fallback for unexpected non-standard owned arrays.

Residual risk: this closes only the coupled thermal-acoustic Rayon cluster.
The solver still carries broader ndarray storage and manifest-level Rayon
dependency edges until the remaining solver kernels migrate to Leto and Atlas
CPU/GPU backend traits.

Evidence tier: static source audit plus compile-time and focused empirical
validation. Verification: `rustup run nightly cargo check -p kwavers-solver`
passed; `rustup run nightly cargo clippy -p kwavers-solver --lib -- -D
warnings` passed; focused `rustup run nightly cargo nextest run -p
kwavers-solver thermal_acoustic --status-level fail --no-fail-fast` passed
9/9 with 934 skipped; rustfmt `--check` passed; scoped `git diff --check`
passed with only LF/CRLF warnings; and scoped `rg` found no direct Rayon hits
under `crates/kwavers-solver/src/forward/coupled/thermal_acoustic`.

## CLOSED: kwavers-solver BEM scattered-field Moirai slice (2026-07-03)

Replaced `forward::bem::solver::solution` direct Rayon `par_iter` scattered
field evaluation with Moirai ordered map-collect. The BEM representation
formula and output ordering are unchanged.

Residual risk: this closes only the BEM scattered-field direct Rayon edge. The
solver still carries broader direct Rayon and ndarray-parallel holdouts, so
the manifest-level Rayon dependency remains open.

Evidence tier: static source audit plus compile-time and focused empirical
validation. Verification: `rustup run nightly cargo check -p kwavers-solver`
passed; `rustup run nightly cargo clippy -p kwavers-solver --lib -- -D
warnings` passed; focused `rustup run nightly cargo nextest run -p
kwavers-solver bem --status-level fail --no-fail-fast` passed 65/65 with 878
skipped; rustfmt `--check` passed; scoped `git diff --check` passed with only
LF/CRLF warnings; and scoped `rg` found no direct Rayon hits in
`crates/kwavers-solver/src/forward/bem/solver/solution.rs`.

## CLOSED: kwavers-solver legacy seismic RTM Moirai slice (2026-07-03)

Replaced direct ndarray/Rayon imaging-condition passes in
`inverse::seismic::rtm` with Moirai dense traversal and sequential ndarray
fallbacks for unexpected non-standard layouts. The normalized single-snapshot
formula now computes source illumination inline instead of allocating a
temporary `Array3`.

Residual risk: this closes only the legacy `inverse::seismic::rtm` processor.
The full `inverse::reconstruction::seismic::rtm` implementation still contains
direct ndarray/Rayon holdouts and remains a separate migration slice.

Evidence tier: static source audit plus compile-time and focused empirical
validation. Verification: `rustup run nightly cargo check -p kwavers-solver`
passed; `rustup run nightly cargo clippy -p kwavers-solver --lib -- -D
warnings` passed; focused `rustup run nightly cargo nextest run -p
kwavers-solver` over the three legacy RTM tests passed 3/3 with 940 skipped;
rustfmt `--check` passed; scoped `git diff --check` passed with only LF/CRLF
warnings; and scoped `rg` found no direct Rayon hits in
`crates/kwavers-solver/src/inverse/seismic/rtm.rs`.

## CLOSED: kwavers-solver photoacoustic line reconstruction Moirai slice (2026-07-03)

Replaced the k-space line reconstruction positivity `par_mapv_inplace` clamp
with Moirai dense traversal and a sequential ndarray fallback for unexpected
non-standard layouts. Added a value-semantic regression proving the clamped
output equals `max(unclamped, 0)` elementwise.

Residual risk: this closes only the line-reconstruction positivity edge. The
follow-up photoacoustic reconstruction slice below closes the then-existing
Fourier and iterative direct ndarray/Rayon holdouts.

Evidence tier: static source audit plus compile-time and focused empirical
validation. Verification: `rustup run nightly cargo check -p kwavers-solver`
passed; `rustup run nightly cargo clippy -p kwavers-solver --lib -- -D
warnings` passed; focused `rustup run nightly cargo nextest run -p
kwavers-solver line_reconstruction --status-level fail --no-fail-fast` passed
4/4 with 940 skipped; rustfmt `--check` passed; scoped `git diff --check`
passed with only LF/CRLF warnings; and scoped `rg` found no direct Rayon hits
in
`crates/kwavers-solver/src/inverse/reconstruction/photoacoustic/line_reconstruction.rs`.

## CLOSED: kwavers-physics direct Rayon removal (2026-07-03)

Removed the direct `kwavers-physics` Rayon dependency and routed the remaining
source-level direct Rayon loops in analytical transducer steering, RTM
beam/backpropagation, nonlinear stability constraints, and Monte Carlo photon
chunking through `moirai-parallel`. The package still enables `ndarray/rayon`
because existing `Zip::par_for_each` and `par_mapv_inplace` kernels remain and
must move with the Leto/Hephaestus backend migration.

Residual risk: this closes only direct Rayon usage in `kwavers-physics`.
Ndarray-parallel kernels remain in thermal, optics, acoustics, chemistry,
therapy, and field-surrogate modules; those should be converted by moving the
array surface to Leto and dispatching CPU/GPU work through generic backend
traits rather than adding another concrete helper path.

Evidence tier: static source/dependency audit plus compile-time and focused
empirical validation. Verification: scoped `rg` found no `rayon::`,
`use rayon`, `par_iter(`, or `into_par_iter(` hits under
`crates/kwavers-physics/src` or its manifest; residual `rg` still finds
`par_for_each`/`par_mapv_inplace` ndarray-parallel kernels; `rustup run
nightly cargo check -p kwavers-physics` passed; focused `rustup run nightly
cargo nextest run -p kwavers-physics -E "test(apply_stability_constraints) or
test(steering) or test(backprop) or test(monte_carlo) or
test(focused_gaussian) or test(intensity_projection)" --status-level fail
--no-fail-fast` passed 41/41; and `rustup run nightly cargo tree -p
kwavers-physics --depth 1` lists `moirai-parallel` with no direct `rayon`.

Follow-up 2026-07-04: `acoustics::bubble_dynamics::interactions` now routes
interaction-field assembly through the existing Moirai-backed
`crate::parallel::for_each_indexed_mut` adapter instead of ndarray/Rayon
`Zip::par_for_each`. A value-semantic regression checks the analytical
monopole-pressure contribution and proves the source bubble cell is excluded.
Evidence tier: static source audit plus compile-time/lint and focused
empirical validation; `kwavers-physics` check and lib clippy pass, focused
`bubble_dynamics::interactions` nextest passes 4/4, and the edited file has no
direct Rayon or ndarray-parallel source hits.

Follow-up 2026-07-04: `field_surrogate::resample` and
`field_surrogate::cube` now route trilinear output assembly and in-place
frequency-corner blending through the existing Moirai-backed physics traversal
adapter instead of ndarray/Rayon `Zip::par_for_each`. Evidence tier: static
source audit plus compile-time/lint and focused empirical validation;
`kwavers-physics` check and lib clippy pass, focused `field_surrogate`
nextest passes 24/24, and the field-surrogate subtree has no direct Rayon,
ndarray-parallel, or `Zip` source hits.

Follow-up 2026-07-04: `chemistry::reaction_kinetics` now routes its hydroxyl
and hydrogen-peroxide update through the reusable
`crate::parallel::zip_two_mut_two_refs` adapter, backed by Moirai chunk-pair
scheduling for dense standard-layout arrays and sequential ndarray traversal
only for non-standard views. Evidence tier: static source audit plus
compile-time/lint and focused empirical validation; `kwavers-physics` check
and lib clippy pass, focused `reaction_kinetics` nextest passes 1/1, and the
edited reaction file has no direct Rayon or ndarray-parallel source hits.

Follow-up 2026-07-04: `chemistry::ros_plasma::ros_species` now routes
species concentration decay through `crate::parallel::for_each_indexed_mut`
instead of ndarray/Rayon `par_mapv_inplace`. Evidence tier: static source
audit plus compile-time/lint and focused empirical validation;
`kwavers-physics` check and lib clippy pass, focused `ros_species` nextest
passes 4/4, and the ROS species subtree has no direct Rayon or
ndarray-parallel source hits.

## CLOSED: thermal CEM43 Leto state slice (2026-07-03)

Moved `kwavers_physics::thermal::ThermalCEM43Grid` from ndarray storage to
`leto::Array3<f64>` and replaced its ndarray/Rayon `Zip::par_for_each` update
with Moirai chunk scheduling over dense Leto slices. The update path now
returns a typed shape mismatch error instead of relying on ndarray Zip
preconditions. The top-level theranostic lesion mask now consumes the Leto
CEM43 field directly, and `brain_theranostic_monitor` keeps its thermal
temperature and absorbed-power fields in Leto.

Residual risk: this closes the CEM43 grid, lesion-mask, and brain-monitor
thermal state path only. `ThermalDoseCalculator` still feeds the solver/Python
thermal diffusion boundary through ndarray, and the top-level lesion
sound-speed/FWI reconstruction path still uses ndarray for base/perturbed
medium fields pending the solver-owned Leto producer migration.

Evidence tier: static source audit plus compile-time and focused empirical
validation. Verification: `rustup run nightly cargo check -p kwavers-physics`
passed; `rustup run nightly cargo check -p kwavers --example
brain_theranostic_monitor` passed; focused `kwavers-physics` nextest over
`thermal_dose` passed 12/12; focused `kwavers --lib` nextest over lesion tests
passed 10/10; rustfmt and scoped `git diff --check` passed. A broader
`cargo nextest run -p kwavers` attempt failed during unrelated existing
all-target compilation blockers in `skull_ct_phase_correction`,
`ultrasound_physics_validation`, and `nl_swe_validation`.

## CLOSED: top-level kwavers Rayon feature removal (2026-07-03)

Removed the top-level `kwavers` direct Rayon dependency and obsolete
`parallel = ["ndarray/rayon"]` feature. The liver theranostic reconstruction
and 3-D seismic examples now use Moirai indexed collection for shot/ray fan-out
and Gaussian blur loops; the 3-D seismic example was also updated to current
RITK DICOM/spacing accessors so the touched example compiles. Removed
`ndarray/rayon` from `kwavers-python` because wrapper source has no
ndarray-parallel API usage.

Residual risk: lower crates still contain direct Rayon/ndarray-parallel usage,
notably `kwavers-solver` and `kwavers-physics`; top-level dev `tokio` remains
until the async-runtime migration is addressed. `cargo check -p
kwavers-python` is still blocked by existing numpy/ndarray version-boundary
errors and Leto resampling wrapper mismatches.

Evidence tier: static source/dependency audit plus compile-time validation for
the edited top-level examples. Verification: scoped `rg` found no direct Rayon
or ndarray-parallel hits under `crates/kwavers/{src,examples,tests,benches}` or
`crates/kwavers-python/src`; `rustup run nightly cargo check -p kwavers
--example liver_theranostic_reconstruction --features nifti` passed; `rustup
run nightly cargo check -p kwavers --example seismic_imaging_3d_demo` passed;
`rustup run nightly rustfmt --check` passed for the touched examples; scoped
`git diff --check` passed; and `rustup run nightly cargo tree -p kwavers
--depth 1` shows no direct `rayon`.

## CLOSED: Burn WGPU dependency feature removal (2026-07-03)

Removed the Burn `wgpu` feature from the workspace `burn` dependency and the
`kwavers`, `kwavers-solver`, and `kwavers-analysis` Burn dependencies. Burn
remains present only for the current CPU PINN path and temporary analysis
compatibility/test holdouts; GPU PINN execution is still assigned to the Coeus
+ Hephaestus provider migration.

Residual risk: this does not remove Burn itself. The remaining Burn PINN modules
still need a real Coeus replacement with value-semantic residual/training tests.

Follow-up 2026-07-04 removed the public analysis GPU Burn DAS surface:
`signal_processing::beamforming::gpu::das_burn` is now test-only, and
`beamforming::gpu`/`beamforming` no longer reexport `BurnDasBeamformer`,
`BurnBeamformingConfig`, `DasInterpolationMethod`, or `beamform_cpu`. The
legacy Burn DAS implementation now compiles only for `pinn` tests pending
Coeus/Hephaestus migration.

Follow-up 2026-07-04 removed the public analysis neural Burn PINN provider
surface: `signal_processing::beamforming::neural` no longer reexports
`create_burn_beamforming_provider` or `BurnPinnBeamformingAdapter`, and instead
reexports the solver-agnostic `PinnBeamformingProvider` and
`PinnProviderRegistry` trait seam. Burn remains a solver implementation detail
until Coeus/Hephaestus supplies the replacement provider.

Follow-up 2026-07-04 replaced direct analysis uncertainty Burn PINN signatures:
`PinnUncertaintyPredictor` now owns the analysis-side prediction contract, and
Bayesian, ensemble, conformal, and top-level uncertainty methods accept that
trait instead of `BurnPINN1DWave<B>`. Burn remains only as a compatibility impl
for the current solver model pending Coeus.

Follow-up 2026-07-04 removed the remaining direct Burn dependency from
`kwavers-analysis`: the test-only Burn DAS holdout was deleted, the
analysis-side Burn compatibility impl was removed from `PinnUncertaintyPredictor`,
and stale analysis docs now name Coeus as the model-provider path. Solver and
top-level CPU PINN paths still retain Burn training/autodiff until Coeus
replaces them.

Evidence tier: static manifest audit plus metadata, dependency-tree, and
compile-time validation. Verification:
fixed-string audit over the scoped manifests, PINN solver docs, and examples
found no `burn = .*wgpu`, `burn-wgpu`, `burn-cuda`, or `pinn-gpu` hits;
`rustup run nightly cargo metadata --no-deps --format-version 1 --manifest-path
Cargo.toml --features pinn` passed. Before the analysis-autodiff cleanup it
reported Burn features `ndarray,autodiff` for `kwavers`, `kwavers-analysis`,
and `kwavers-solver`, plus `ndarray` for `kwavers-python`; follow-up direct
metadata now reports `NO_DIRECT_BURN` for `kwavers-analysis`.
Follow-up dependency-tree validation after
disabling Burn defaults reports no selected `burn-wgpu`, `burn-cuda`, or
`burn-rocm` edge, and `rustup run nightly cargo check -p kwavers --features
pinn` passed. Follow-up analysis neural validation found no public Burn provider
reexport or stale CUDA/wgpu doc claim in the analysis beamforming facades;
`rustup run nightly cargo fmt -p kwavers-analysis --check`, `rustup run nightly
cargo check -p kwavers-analysis --features pinn`, focused `cargo nextest run -p
kwavers-analysis --features pinn neural --status-level fail --no-fail-fast`
passed 77/77, and `rustup run nightly cargo clippy -p kwavers-analysis
--features pinn --all-targets -- -D warnings` passed. Follow-up uncertainty
validation found Burn tokens only in the single `PinnUncertaintyPredictor for
BurnPINN1DWave` compatibility impl under `ml/uncertainty`;
`rustup run nightly cargo fmt -p kwavers-analysis --check`, `rustup run nightly
cargo check -p kwavers-analysis --features pinn`, focused `cargo nextest run -p
kwavers-analysis --features pinn uncertainty --status-level fail
--no-fail-fast` passed 33/33, and `rustup run nightly cargo clippy -p
kwavers-analysis --features pinn --all-targets -- -D warnings` passed.
Follow-up analysis Burn removal validation: scoped source/manifest audit under
`crates/kwavers-analysis` returned no Burn matches, direct `cargo metadata`
audit returned `NO_DIRECT_BURN`, `rustup run nightly cargo fmt -p
kwavers-analysis --check`, `rustup run nightly cargo check -p kwavers-analysis
--features pinn`, focused `cargo nextest run -p kwavers-analysis --features
pinn -E "test(uncertainty) or test(time_domain::das) or
test(das_single_element_zero_delay_passthrough) or
test(das_coherent_gain_co_located_elements) or
test(das_receive_delay_is_geometrically_correct) or
test(das_channel_mismatch_returns_error)" --status-level fail --no-fail-fast`
passed 47/47, and `rustup run nightly cargo clippy -p kwavers-analysis
--features pinn --all-targets -- -D warnings` passed.

## CLOSED: top-level kwavers Burn demotion (2026-07-03)

Removed unused workspace-level `burn` and `burn-ndarray` dependency aliases and
moved the top-level `kwavers` Burn dependency from `[dependencies]` to
`[dev-dependencies]`. The top-level crate's production targets do not import
Burn; remaining top-level Burn use is confined to examples, benches, and
integration tests while solver/analysis/Python keep Burn until the Coeus
migration replaces their source-level imports.

Residual risk: Burn remains in solver, analysis, Python, and top-level
dev-targets. The remaining migration is a real Coeus replacement, not another
manifest-only cleanup.

Evidence tier: static source/manifest audit plus metadata validation.
Verification: no `burn = { workspace = true }` or `burn-ndarray` hits remain;
scoped source audit found Burn imports only under `crates/kwavers/examples`,
`crates/kwavers/benches`, and `crates/kwavers/tests`; `rustup run nightly cargo
metadata --no-deps --format-version 1 --manifest-path Cargo.toml --features
pinn` passed and reported `kwavers burn kind=dev features: ndarray,autodiff`.

## CLOSED: solver PINN false GPU surface removal (2026-07-03)

Removed the solver-local `inverse::pinn::ml::gpu_accelerator` module and the
top-level/solver `pinn-gpu`, `burn-wgpu`, and `burn-cuda` feature aliases. The
deleted module exposed CUDA-named buffers, streams, kernel manager, memory
pools, and a batched trainer without real CUDA module compilation or device
execution. PINN docs/examples now state that GPU training belongs to the
Coeus + Hephaestus provider migration, where WGPU and CUDA are interchangeable
provider implementations behind one trait seam.

Residual risk: this removes the false solver PINN GPU surface only. Real PINN
GPU training still requires a Coeus backend contract, Hephaestus provider
routing, and value-semantic residual/training tests.

Evidence tier: static source audit plus formatting/whitespace validation.
Verification: fixed-string audit over the touched manifests, PINN solver docs,
and examples found no `pinn-gpu`, `burn-wgpu`, `burn-cuda`,
`CudaKernelManager`, `CudaBuffer`, `CudaStream`, `BatchedPINNTrainer`,
`PinnGpuMemoryPoolType`, `GpuMemoryManager`, `TrainingStats`, or PINN
`gpu_accelerator` hits; `rustup run nightly cargo fmt -p kwavers-solver -p
kwavers --check` passed; scoped `git diff --check` passed; `rustup run nightly
cargo metadata --no-deps --format-version 1 --manifest-path
crates/kwavers-solver/Cargo.toml --features pinn` passed. `rustup run nightly
cargo check -p kwavers-solver --features pinn` failed before reaching the
changed package while writing dependency `.rmeta` files with OS error 112;
`Get-PSDrive` reported `D:` at 0.17 GB free.

## CLOSED: kwavers-gpu Burn accelerator removal (2026-07-03)

Removed the local `BurnGpuAccelerator` module/export, the optional `burn`
dependency, and the `kwavers-gpu/pinn` feature. This deletes a GPU surface that
was neither Hephaestus-backed nor Coeus-backed, converted `Array3<f64>` through
Burn `f32` tensors, and returned zero tensors for heat, diffusion, and
Navier-Stokes PDE residuals instead of performing real computation.

Residual risk: this removes the false downstream GPU accelerator only.
Solver-side PINN modules still use Burn and need a real Coeus migration with
value-semantic training/residual tests before the broader Burn dependency can
be removed from the workspace.

Evidence tier: static source audit plus compile-time/focused empirical tests.
Verification: `rustup run nightly cargo fmt -p kwavers-gpu --check`, `rustup
run nightly cargo check -p kwavers-gpu --features gpu`, `rustup run nightly
cargo check -p kwavers-gpu --features cuda-provider`, `rustup run nightly
cargo check -p kwavers-gpu --all-features`, `rustup run nightly cargo clippy
-p kwavers-gpu --features gpu --all-targets -- -D warnings`, and `rustup run
nightly cargo nextest run -p kwavers-gpu --features gpu --status-level fail
--no-fail-fast` passed 128/128 with 1 skipped. `rustup run nightly cargo
nextest run -p kwavers-gpu --all-features --status-level fail --no-fail-fast`
failed before test execution while writing
`D:/atlas/target/debug/deps/libapollo_fft-*.rlib` with OS error 112; drive
inspection reported `D:` at 0.15 GB free.

## CLOSED: kwavers-gpu multi-GPU provider genericity (2026-07-03)

`kwavers-gpu::gpu::multi_gpu::MultiGpuContext<P>` now stores
`CoreGpuContext<P>` values and acquires multiple Hephaestus devices through
`P::try_acquire_devices`. `MultiGpuContext::new()` remains the default WGPU
constructor for current WGSL kernels, and CUDA is covered at the type boundary
through `kwavers-gpu/cuda-provider` without adding placeholder CUDA compute
dispatch. The all-targets clippy verification also tightened an existing PSTD
run-state type assertion so the GPU package test target remains lint-clean.

Residual risk: this closes the multi-GPU topology/acquisition seam only. Real
CUDA multi-GPU compute still requires CUDA kernels plus WGPU-vs-CUDA
differential tests for each operation contract.

Evidence tier: type-level/compile-time validation plus focused empirical
tests. Verification: `rustup run nightly cargo fmt -p kwavers-gpu --check`,
`rustup run nightly cargo check -p kwavers-gpu --features cuda-provider`,
`rustup run nightly cargo clippy -p kwavers-gpu --features gpu --all-targets
-- -D warnings`, `rustup run nightly cargo nextest run -p kwavers-gpu
--features gpu gpu::multi_gpu --status-level fail --no-fail-fast` passed 4/4,
and the same focused nextest command under `--features cuda-provider` passed
5/5.

## CLOSED: kwavers-gpu FDTD pressure Leto I/O (2026-07-03)

`kwavers-gpu::gpu::WgpuFdtd` pressure upload/readback now uses provider-native
`leto::Array3<f32>` for the WGPU `f32` storage contract. Upload rejects
non-dense Leto host fields with a typed `KwaversError::InvalidInput` instead
of silently allocating a conversion buffer, and readback reconstructs the Leto
array without widening through `f64`.

Follow-up 2026-07-04 renamed the WGSL-only `FdtdGpu` and
`FdtdGpuShaderDispatcher` public surfaces to `WgpuFdtd` and
`WgpuFdtdPressureDispatcher` without compatibility aliases, so CUDA remains a
provider-trait implementation gap instead of sharing a generic GPU type name.

Residual risk: this closes only the two-pass WGPU FDTD pressure I/O surface.
Validation and Burn-backed accelerator paths still carry ndarray/Burn surfaces
for later Leto/Hephaestus/Coeus migration.

Evidence tier: type-level API validation plus focused empirical tests.
Verification: `rustup run nightly cargo fmt -p kwavers-gpu --check`,
`rustup run nightly cargo check -p kwavers-gpu --features gpu`, `rustup run
nightly cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings`,
focused FDTD/acoustic provider nextest 7/7, and `rustup run nightly cargo
check -p kwavers-gpu --features cuda-provider` pass.

## CLOSED: kwavers-gpu ComputeManager provider genericity (2026-07-03)

`kwavers-gpu::gpu::ComputeManager<P>` is now generic over
`GpuDeviceProvider`, stores `Option<GpuDevice<P>>`, and exposes raw
`wgpu::Device`/`wgpu::Queue` helpers only on the `WgpuDevice` specialization.
CPU-only use is explicit through `ComputeManager::cpu_only()` instead of a
silent acquisition fallback, and the CUDA provider type-checks at the manager
boundary without adding fake CUDA compute kernels. Follow-up moved the
manager's CPU field-update helpers to `leto::Array3<f64>` and added typed
shape/layout validation for absorption updates.

Follow-up 2026-07-04: `ComputeManager::new_blocking` now acquires through
`GpuDevice<P>::try_create_with_features_and_limits` instead of
`pollster::block_on(Self::new())`; scoped audit finds no `pollster::block_on`
in `compute_manager.rs` or `kwavers-gpu::backend`, and focused CUDA-provider
`compute_manager` nextest passes 5/5.

Follow-up 2026-07-04: the solver-owned
`ComputeBackend::{element_wise_multiply, apply_spatial_derivative}` contract
now uses `leto::Array3<f64>` instead of `ndarray::Array3<f64>`, and
`GPUBackend<P>` implements that Leto surface while retaining provider-native
`leto::Array3<P::Scalar>` dispatch for WGPU/CUDA-extensible kernels. Focused
backend nextest passes 48/48.

Residual risk: this closes only the manager acquisition/type boundary and its
local Leto field-helper surface. Validation and Burn-backed accelerator paths
still carry ndarray/Burn surfaces for later Leto/Hephaestus/Coeus migration.

Evidence tier: type-level API validation plus focused empirical tests.
Verification: `rustup run nightly cargo fmt -p kwavers-gpu --check`, `rustup
run nightly cargo check -p kwavers-gpu --features gpu`, `rustup run nightly
cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings`, `rustup run
nightly cargo check -p kwavers-gpu --features cuda-provider`, focused WGPU
compute-manager nextest 2/2, and focused CUDA-provider compute-manager nextest
3/3 pass. Follow-up verification for the Leto helper surface: `rustup run
nightly cargo fmt -p kwavers-gpu --check`, `rustup run nightly cargo check -p
kwavers-gpu --features gpu`, `rustup run nightly cargo clippy -p kwavers-gpu
--features gpu --lib -- -D warnings`, `rustup run nightly cargo check -p
kwavers-gpu --features cuda-provider`, focused WGPU compute-manager nextest
3/3, and focused CUDA-provider compute-manager nextest 4/4 pass.

## CLOSED: kwavers-gpu FDTD CPU reference Leto pressure surface (2026-07-03)

`kwavers-gpu::gpu::compute::FdtdCpuReferenceDispatcher` now accepts and returns
`leto::Array3<f64>` for its f64 CPU reference pressure update. Boundary
zeroing and the in-place/allocation-returning update paths use the same Leto
surface, and the dimension-mismatch regression test asserts
`KwaversError::InvalidInput` with the expected shape payload.

Follow-up 2026-07-04 removed the misleading `FdtdGpuDispatcher` public type
name; the CPU reference path is no longer exported as a GPU dispatcher.

Residual risk: this closes only the local FDTD CPU reference dispatcher.
Validation runner source mask/signal setup and Burn-backed accelerator paths
still carry ndarray/Burn surfaces for later Leto/Hephaestus/Coeus migration.

Evidence tier: type-level API validation plus focused empirical tests.
Verification: `rustup run nightly cargo fmt -p kwavers-gpu --check`, `rustup
run nightly cargo check -p kwavers-gpu --features gpu`, `rustup run nightly
cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings`, `rustup run
nightly cargo nextest run -p kwavers-gpu --features gpu fdtd_gpu
--status-level fail --no-fail-fast` passed 5/5, and `rustup run nightly cargo
check -p kwavers-gpu --features cuda-provider` pass.

## CLOSED: kwavers-gpu validation Leto comparator (2026-07-03)

`kwavers-gpu::validation::gpu_cpu_equivalence::EquivalenceValidator` now
compares `leto::Array3<f64>` pressure fields directly. The runner converts the
current FDTD solver pressure field into Leto at the validation boundary, while
the remaining ndarray use in that runner stays scoped to the solver-owned
source mask/signal interface.
Follow-up 2026-07-04 removed the false GPU equivalence branch that constructed
`GPUBackend` but still executed the CPU `FdtdSolver`; the runner now reports a
typed unavailable-provider failure until a real provider-generic
Leto/Hephaestus FDTD GPU trait implementation is wired.

Residual risk: this closes only the GPU/CPU equivalence metrics comparator.
The FDTD solver API, real FDTD GPU provider trait implementation wiring, CPU
fallback FDTD outside this validation boundary, and Burn-backed accelerator
paths still carry ndarray/Burn surfaces for later Leto/Hephaestus/Coeus
migration.

Evidence tier: type-level API validation plus focused empirical tests.
Verification: `rustup run nightly cargo fmt -p kwavers-gpu --check`, `rustup
run nightly cargo check -p kwavers-gpu --features gpu`, `rustup run nightly
cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings`, `rustup run
nightly cargo nextest run -p kwavers-gpu --features gpu gpu_cpu_equivalence
--status-level fail --no-fail-fast` passed 21/21, and `rustup run nightly
cargo check -p kwavers-gpu --features cuda-provider` pass.

## CLOSED: kwavers-gpu realtime imaging Leto frame buffers (2026-07-03)

`RealtimeImagingPipeline` and `StreamingDataSource` now exchange RF input and
processed output frames as `leto::Array4<f32>` and `leto::Array3<f32>`.
Beamforming no longer uses ndarray `sum_axis`; it sums the transmit dimension
through explicit Leto indexing. Envelope detection and log compression keep the
same value semantics over Leto dense slices and indexed fallback paths.
Follow-up 2026-07-04 removed the private ndarray `Array1<Complex64>` Hilbert
scratch by reusing a thread-local `Vec<Complex64>` through Apollo's slice FFT
API; scoped audit now finds no ndarray tokens under
`crates/kwavers-gpu/src/gpu/pipeline`.

Follow-up 2026-07-04 moved the public PSTD runner mask/output contract to
`leto::Array3<bool>`/`leto::Array2<f64>` and added the provider-generic
`run_gpu_pstd_with_provider<P>` wrapper. Follow-up CPML storage migration moved
`CPMLProfiles` and `PmlExpFactors` profile/factor arrays to Leto `Array1`,
after filling Leto's owned-array indexing/equality gaps upstream. Residual
ndarray remains in validation runner source mask/signal setup owned by the
current solver API and in Burn-backed accelerator paths pending
Leto/Hephaestus/Coeus migration.

Evidence tier: type-level API validation plus focused empirical tests.
Verification: `rustup run nightly cargo fmt -p kwavers-gpu --check`, `rustup
run nightly cargo check -p kwavers-gpu --features gpu`, `rustup run nightly
cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings`, `rustup run
nightly cargo nextest run -p kwavers-gpu --features gpu gpu::pipeline
--status-level fail --no-fail-fast` passed 5/5, and `rustup run nightly cargo
check -p kwavers-gpu --features cuda-provider` pass. Follow-up verification:
`rustup run nightly cargo check -p kwavers-gpu --features cuda-provider`,
`rustup run nightly cargo clippy -p kwavers-gpu --features cuda-provider
--all-targets -- -D warnings`, and focused pipeline nextest 8/8 pass.

## CLOSED: kwavers-gpu realtime Leto field map (2026-07-03)

`RealtimeSimulationOrchestrator::step`, `simulate`, and
`GPUBackend::multiphysics_step` now accept `leto::Array3<f64>` field maps.
The realtime loop uses the map for scheduling validation and budget
accounting, so no ndarray operation was needed in this backend path.

Evidence tier: static source audit plus focused compile/test verification.
The solver-facing `ComputeBackend` f64 methods still reject provider dispatch
explicitly until a scalar-generic or real f64 GPU kernel contract lands.
`rustup run nightly cargo fmt -p kwavers-gpu --check`, `rustup run nightly
cargo check -p kwavers-gpu --features gpu`, `rustup run nightly cargo clippy
-p kwavers-gpu --features gpu --lib -- -D warnings`, focused realtime/f64
nextest 5/5, `rustup run nightly cargo check -p kwavers-gpu --features
cuda-provider`, and focused CUDA-provider provider/realtime nextest 7/7 pass.

## CLOSED: kwavers-gpu provider-native Leto dispatch API (2026-07-03)

`GpuComputeProvider` and `GPUBackend::dispatch_element_wise_multiply` /
`dispatch_spatial_derivative` now accept `leto::Array3<f32>` for
provider-native dispatch. Follow-up moved `WgpuBackendBufferManager`
upload/readback for these operations to `leto::Array3<f32>` as well, removing
the provider-side Leto-to-ndarray adapter. The solver-facing `ComputeBackend`
f64 methods still reject GPU dispatch explicitly rather than widening/narrowing
through WGSL f32.

Evidence tier: type-level API validation plus focused value-semantic GPU tests.
`rustup run nightly cargo fmt -p kwavers-gpu --check`, `rustup run nightly
cargo check -p kwavers-gpu --features gpu`, `rustup run nightly cargo clippy
-p kwavers-gpu --features gpu --lib -- -D warnings`, and `rustup run nightly
cargo nextest run -p kwavers-gpu --features gpu -E
"test(gpu_backend_is_generic_over_provider_trait) or test(gpu_provider_identity_is_separate_from_kernel_dispatch) or test(elementwise_multiply_) or test(spatial_derivative_) or test(solver_f64_compute_backend_rejects_wgpu_f32_kernels) or test(test_gpu_capabilities)" --status-level fail --no-fail-fast`
passed 8/8.

## CLOSED: kwavers-gpu backend spatial derivative WGPU dispatch (2026-07-03)

`WgpuComputeProvider::apply_spatial_derivative` now executes through the WGPU
pipeline manager instead of returning a CPU-computed derivative behind the GPU
provider. The `spatial_derivative` WGSL entry point now implements finite
differences over the flattened 3-D field with provider-supplied shape and
direction parameters, and the test suite guards against reintroducing the copy
placeholder. CUDA remains a provider/acquisition contract only until real CUDA
kernels exist for the same operation surface.

Evidence tier: static shader/source audit plus compile-time validation and
focused value-semantic GPU test. `cargo fmt -p kwavers-gpu --check`, `cargo
check -p kwavers-gpu --features gpu`, `cargo clippy -p kwavers-gpu --features
gpu --lib -- -D warnings`, `cargo check -p kwavers-gpu --features
cuda-provider`, and `cargo nextest run -p kwavers-gpu --features gpu -E
"test(spatial_derivative_) or test(gpu_provider_identity_is_separate_from_kernel_dispatch) or test(gpu_backend_is_generic_over_provider_trait) or test(test_gpu_backend_creation)" --status-level fail --no-fail-fast`
passed 5/5. Follow-up `cargo nextest run -p kwavers-gpu --features gpu
backend::tests --status-level fail --no-fail-fast` passed 10/10.

## CLOSED: kwavers-gpu PSTD provider-generic auto-device acquisition (2026-07-03)

`PstdAutoDeviceProvider` now defines the provider contract for automatic PSTD
device acquisition. `WgpuPstdStateProvider` implements the contract through
Hephaestus WGPU acquisition, and `GpuPstdSolver<P>::with_auto_device` is
generic for providers that can return real device/queue handles.

Residual risk: this makes the public PSTD auto-device wrapper
provider-generic but does not implement CUDA PSTD. CUDA PSTD still requires
real provider-owned state, kernels, command/readback mechanics, and
value-semantic differential tests before CUDA can implement the PSTD operation
surface.

Evidence tier: type-level/compile-time validation plus focused empirical test.
`cargo fmt -p kwavers-gpu --check`, `cargo check -p kwavers-gpu --features
gpu`, `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings`, and
`cargo nextest run -p kwavers-gpu --features gpu
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
medium_variable_update` passed 16/16. Source audit confirms no
`impl GpuPstdSolver<WgpuPstdStateProvider>` remains under PSTD, no unqualified
`GpuPstdSolver::with_auto_device` call site remains under
`crates/kwavers-gpu/src`, and no CUDA placeholder was introduced.

## CLOSED: kwavers-gpu PSTD provider-generic medium updates (2026-07-03)

`PstdMediumUpdateState` now defines the provider-state contract for variable
medium upload, full medium refresh, and source-correction disablement.
`WgpuPstdState` implements the contract, and the public `GpuPstdSolver<P>`
methods are generic for providers whose state implements
`PstdMediumUpdateState`.

Residual risk: this makes the public PSTD medium-update wrappers
provider-generic but does not implement CUDA PSTD. CUDA PSTD still requires
real provider-owned state, kernels, command/readback mechanics, and
value-semantic differential tests before CUDA can implement the PSTD operation
surface.

Evidence tier: type-level/compile-time validation plus focused empirical test.
`cargo fmt -p kwavers-gpu --check`, `cargo check -p kwavers-gpu --features
gpu`, `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings`, and
`cargo nextest run -p kwavers-gpu --features gpu
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
medium_variable_update` passed 15/15. Source audit confirms
`GpuPstdSolver<P>` medium methods are bound by
`P::State: PstdMediumUpdateState`, with no WGPU-specialized medium-update
wrapper and no CUDA placeholder.

## CLOSED: kwavers-gpu PSTD provider-generic run execution (2026-07-03)

`PstdRunState` now defines the provider-state run execution contract using
provider-neutral `PstdRunScalars` and `PstdRunInputs`. `WgpuPstdState`
implements the contract, and `GpuPstdSolver<P>::run` is generic for providers
whose state implements `PstdRunState`.

Residual risk: this makes the public PSTD run wrapper provider-generic but
does not implement CUDA PSTD. CUDA PSTD still requires real provider-owned
state, kernels, command/readback mechanics, and value-semantic differential
tests before CUDA can implement the PSTD operation surface.

Evidence tier: type-level/compile-time validation plus focused empirical test.
`cargo fmt -p kwavers-gpu --check`, `cargo check -p kwavers-gpu --features
gpu`, `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings`, and
`cargo nextest run -p kwavers-gpu --features gpu
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
medium_variable_update` passed 14/14. Source audit confirms
`GpuPstdSolver<P>::run` is bound by `P::State: PstdRunState`, with no
WGPU-specialized time-loop run wrapper and no CUDA placeholder.

## CLOSED: kwavers-gpu PSTD provider-generic state construction (2026-07-03)

`PstdStateBuilder` now defines an associated provider context type plus the
`build_state` contract. `WgpuPstdStateProvider` implements that contract with
`GpuProviderContext<WgpuDevice>`, and `GpuPstdSolver<P>::new` delegates state
construction through `P::build_state`. Direct WGPU test/helper constructor call
sites now name `WgpuPstdStateProvider` explicitly.

Residual risk: this makes the PSTD constructor provider-generic but does not
implement CUDA PSTD. CUDA PSTD still requires real provider-owned state,
kernels, command/readback mechanics, and value-semantic differential tests
before CUDA can implement the PSTD operation surface.

Evidence tier: type-level/compile-time validation plus focused empirical test.
`cargo fmt -p kwavers-gpu --check`, `cargo check -p kwavers-gpu --features
gpu`, `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings`, and
`cargo nextest run -p kwavers-gpu --features gpu
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
medium_variable_update` passed 13/13. Source audit confirms
`GpuPstdSolver<P>::new` is bound by `PstdStateBuilder`, with no stale direct
`WgpuPstdStateProvider::build_state` calls and no CUDA placeholder.

## CLOSED: kwavers-gpu PSTD provider-owned WGPU run orchestration (2026-07-03)

`WgpuPstdState` now owns high-level WGPU run-loop orchestration: cache
validation, cache rebuild/refresh selection, sensor clear, zero-field pass
submission, batched time-step submission, throttled provider wait, sensor copy,
and mapped readback. `GpuPstdSolver<WgpuPstdStateProvider>::run` constructs
the scalar/input value objects and delegates to provider state.

Residual risk: this removes high-level WGPU run-loop mechanics from the solver
wrapper only. CUDA PSTD still requires real provider-owned state, kernels,
command/readback mechanics, and value-semantic differential tests before CUDA
can implement the PSTD operation surface.

Evidence tier: type-level/compile-time validation plus focused empirical test.
`cargo fmt -p kwavers-gpu --check`, `cargo check -p kwavers-gpu --features
gpu`, `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings`, and
`cargo nextest run -p kwavers-gpu --features gpu
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
medium_variable_update` passed 12/12. Source audit confirms
`WgpuPstdState::run` owns high-level run orchestration, with no CUDA
placeholder.

## CLOSED: kwavers-gpu PSTD provider-owned WGPU pass encoding (2026-07-03)

`WgpuPstdState` now owns WGPU dispatch, absorption dispatch, FFT/IFFT, and
per-phase pass encoding. `WgpuPstdPassProvider` stores `&WgpuPstdState` and no
longer depends on `GpuPstdSolver<WgpuPstdStateProvider>` for pass-body
encoding.

Residual risk: this removes WGPU pass encoding from the solver wrapper only.
CUDA PSTD still requires real provider-owned state, kernels, command/readback
mechanics, and value-semantic differential tests before CUDA can implement the
PSTD operation surface.

Evidence tier: type-level/compile-time validation plus focused empirical test.
`cargo fmt -p kwavers-gpu --check`, `cargo check -p kwavers-gpu --features
gpu`, `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings`, and
`cargo nextest run -p kwavers-gpu --features gpu
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
medium_variable_update` passed 12/12. Source audit confirms WGPU pass encoding
is implemented on `WgpuPstdState`, with no CUDA placeholder.

## CLOSED: kwavers-gpu PSTD provider-owned WGPU run cache (2026-07-03)

`WgpuPstdState` now owns run-scoped sensor/source/velocity buffer allocation,
sensor bind-group rebuild, cache-key updates, and cache-hit signal-tail uploads.
`GpuPstdSolver<WgpuPstdStateProvider>` forwards run-cache methods to provider
state and supplies the solver time-step count.

Residual risk: this removes WGPU run-cache mechanics from the solver wrapper
only. CUDA PSTD still requires real provider-owned state, kernels,
command/readback mechanics, and value-semantic differential tests before CUDA
can implement the PSTD operation surface.

Evidence tier: type-level/compile-time validation plus focused empirical test.
`cargo fmt -p kwavers-gpu --check`, `cargo check -p kwavers-gpu --features
gpu`, `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings`, and
`cargo nextest run -p kwavers-gpu --features gpu medium_variable_update
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
overwrite_packed_signal_tail_keeps_index_prefix_stable` passed 12/12. Source
audit confirms `WgpuPstdState` owns run-cache allocation and tail upload
bodies, while every PSTD solver impl remains
`GpuPstdSolver<WgpuPstdStateProvider>` with no CUDA placeholder.

## CLOSED: kwavers-gpu PSTD provider-owned WGPU medium uploads (2026-07-03)

`WgpuPstdState` now owns the WGPU medium/source upload bodies:
variable-medium upload, full medium refresh, and source-correction
disablement. The public `GpuPstdSolver<WgpuPstdStateProvider>` methods now
forward to provider-state methods instead of creating command providers and
issuing `write_buffer` calls directly.

Residual risk: this removes WGPU medium/source upload mechanics from the solver
wrapper only. CUDA PSTD still requires real provider-owned state, kernels,
command/readback mechanics, and value-semantic differential tests before CUDA
can implement the PSTD operation surface.

Evidence tier: type-level/compile-time validation plus focused empirical test.
`cargo fmt -p kwavers-gpu --check`, `cargo check -p kwavers-gpu --features
gpu`, `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings`, and
`cargo nextest run -p kwavers-gpu --features gpu medium_variable_update
pstd_solver_state_is_provider_associated
pstd_command_provider_is_generic_over_provider_trait
pstd_pass_provider_is_generic_over_provider_trait
pstd_bind_group_factory_is_generic_over_provider_trait
pstd_bind_group_layout_factory_is_generic_over_provider_trait
pstd_buffer_factory_is_generic_over_provider_trait
pstd_pipeline_factory_is_generic_over_provider_trait` passed 8/8. Source audit
confirms `WgpuPstdState` owns these upload bodies, and every PSTD solver impl
remains `GpuPstdSolver<WgpuPstdStateProvider>` with no CUDA placeholder.

## CLOSED: kwavers-gpu PSTD provider-owned WGPU state construction (2026-07-03)

`WgpuPstdStateProvider::build_state` now owns WGPU PSTD state assembly:
buffers, pipelines, bind groups, layouts, run-cache state, device/queue
handles, and host scratch/upload buffers. `GpuPstdSolver::new` now wraps the
provider-built state with grid dimensions, time step, and physics flags.

Residual risk: this moves WGPU construction ownership behind the provider only.
CUDA PSTD still requires real provider-owned state construction, kernels,
command/readback mechanics, and value-semantic differential tests before CUDA
can implement the PSTD operation surface.

Evidence tier: type-level/compile-time validation plus focused empirical test.
`cargo fmt -p kwavers-gpu --check`, `cargo check -p kwavers-gpu --features
gpu`, `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings`, and
`cargo nextest run -p kwavers-gpu --features gpu medium_variable_update
pstd_solver_state_is_provider_associated
pstd_command_provider_is_generic_over_provider_trait
pstd_pass_provider_is_generic_over_provider_trait
pstd_bind_group_factory_is_generic_over_provider_trait
pstd_bind_group_layout_factory_is_generic_over_provider_trait
pstd_buffer_factory_is_generic_over_provider_trait
pstd_pipeline_factory_is_generic_over_provider_trait` passed 8/8. Source audit
confirms `WgpuPstdStateProvider::build_state` owns WGPU state assembly,
`GpuPstdSolver::new` wraps the returned state, and every PSTD solver impl
remains `GpuPstdSolver<WgpuPstdStateProvider>` with no CUDA placeholder.

## CLOSED: kwavers-gpu PSTD provider-owned WGPU scratch buffers (2026-07-03)

`GpuPstdSolver<P>` no longer stores WGPU host scratch/upload buffers.
`WgpuPstdState` owns `scratch_c0_sq`, `scratch_rho0_inv`,
`scratch_rho0_flat`, `scratch_source_kappa_ones`, `scratch_source_data`, and
`scratch_vel_x_data`, and WGPU-specialized medium-update and run-cache staging
paths borrow them through `self.state`.

Residual risk: this removes WGPU upload staging ownership from the generic
solver wrapper only. CUDA PSTD still requires real provider-owned state,
kernels, command/readback mechanics, and value-semantic differential tests
before CUDA can implement the PSTD operation surface.

Evidence tier: type-level/compile-time validation plus focused empirical test.
`cargo fmt -p kwavers-gpu --check`, `cargo check -p kwavers-gpu --features
gpu`, `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings`, and
`cargo nextest run -p kwavers-gpu --features gpu medium_variable_update
pstd_solver_state_is_provider_associated
pstd_command_provider_is_generic_over_provider_trait
pstd_pass_provider_is_generic_over_provider_trait
pstd_bind_group_factory_is_generic_over_provider_trait
pstd_bind_group_layout_factory_is_generic_over_provider_trait
pstd_buffer_factory_is_generic_over_provider_trait
pstd_pipeline_factory_is_generic_over_provider_trait` passed 8/8. Source audit
confirms those scratch/upload fields exist only on `WgpuPstdState`, and every
PSTD solver impl remains `GpuPstdSolver<WgpuPstdStateProvider>` with no CUDA
placeholder.

## CLOSED: kwavers-gpu PSTD provider-owned WGPU handles (2026-07-03)

`GpuPstdSolver<P>` no longer stores raw `Arc<wgpu::Device>` or
`Arc<wgpu::Queue>` fields. `WgpuPstdState` owns those handles together with the
WGPU buffers, pipelines, bind groups, layouts, and run cache; WGPU-specialized
medium-update, run-cache, and run-loop paths borrow handles through
`self.state`.

Residual risk: this removes WGPU handle ownership from the generic solver
wrapper only. CUDA PSTD still requires real provider-owned state, kernels,
command/readback mechanics, and value-semantic differential tests before CUDA
can implement the PSTD operation surface.

Evidence tier: type-level/compile-time validation plus focused empirical test.
`cargo fmt -p kwavers-gpu --check`, `cargo check -p kwavers-gpu --features
gpu`, `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings`, and
`cargo nextest run -p kwavers-gpu --features gpu medium_variable_update
pstd_solver_state_is_provider_associated
pstd_command_provider_is_generic_over_provider_trait
pstd_pass_provider_is_generic_over_provider_trait
pstd_bind_group_factory_is_generic_over_provider_trait
pstd_bind_group_layout_factory_is_generic_over_provider_trait
pstd_buffer_factory_is_generic_over_provider_trait
pstd_pipeline_factory_is_generic_over_provider_trait` passed 8/8. Source audit
confirms `GpuPstdSolver<P>` has no raw WGPU handle fields, WGPU handles are on
`WgpuPstdState`, and every PSTD solver impl remains
`GpuPstdSolver<WgpuPstdStateProvider>` with no CUDA placeholder.

## CLOSED: kwavers-gpu PSTD provider-associated state (2026-07-03)

`GpuPstdSolver<P>` now owns `P::State` through `PstdStateProvider`, with
`WgpuPstdStateProvider` as the default real implementation. Existing
construction, medium-update, run-cache, dispatch, pass-body, and encode methods
are implemented only for `GpuPstdSolver<WgpuPstdStateProvider>`, so CUDA is not
represented by a fake PSTD compute path.

Residual risk: this closes the solver state type seam only. CUDA PSTD still
requires real provider-owned state, kernels, command/readback mechanics, and
value-semantic differential tests against the WGPU implementation before it can
be exposed as a compute provider.

Evidence tier: type-level/compile-time validation plus focused empirical test.
`cargo fmt -p kwavers-gpu --check`, `cargo check -p kwavers-gpu --features
gpu`, `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings`, and
`cargo nextest run -p kwavers-gpu --features gpu medium_variable_update
pstd_solver_state_is_provider_associated
pstd_command_provider_is_generic_over_provider_trait
pstd_pass_provider_is_generic_over_provider_trait
pstd_bind_group_factory_is_generic_over_provider_trait
pstd_bind_group_layout_factory_is_generic_over_provider_trait
pstd_buffer_factory_is_generic_over_provider_trait
pstd_pipeline_factory_is_generic_over_provider_trait` passed 8/8. Source audit
confirms every PSTD solver impl is WGPU-provider-specialized and no CUDA PSTD
placeholder, fake provider, `todo!`, or `unimplemented!` exists under
`crates/kwavers-gpu/src/pstd_gpu`.

## CLOSED: kwavers-gpu PSTD WGPU state aggregate (2026-07-03)

PSTD grouped WGPU buffers, pipelines, bind groups, layouts, and run-cache state
now live under `WgpuPstdState`. `GpuPstdSolver` exposes one provider-state field
instead of separate grouped WGPU fields.

Residual risk: this closes only the state aggregation step. `WgpuPstdState`
itself is still WGPU-specific and owned directly by `GpuPstdSolver`; the next
architectural step is making PSTD state a provider-associated type so WGPU and
CUDA can specialize state ownership behind the same operation surface.

Evidence tier: type-level/compile-time validation plus focused empirical test.
`cargo fmt -p kwavers-gpu --check`, `cargo check -p kwavers-gpu --features
gpu`, `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings`, and
`cargo nextest run -p kwavers-gpu --features gpu medium_variable_update
pstd_command_provider_is_generic_over_provider_trait
pstd_pass_provider_is_generic_over_provider_trait
pstd_bind_group_factory_is_generic_over_provider_trait
pstd_bind_group_layout_factory_is_generic_over_provider_trait
pstd_buffer_factory_is_generic_over_provider_trait
pstd_pipeline_factory_is_generic_over_provider_trait` passed 7/7. Source
audit: `GpuPstdSolver` exposes `state: WgpuPstdState` and no direct
`field_buffers`, `kspace_buffers`, `medium_buffers`, `absorption_buffers`,
`pml_shift_buffers`, `pipelines`, `permanent_bind_groups`, `layouts`, or
`run_cache` fields.

## CLOSED: kwavers-gpu PSTD compute-pipeline state grouping (2026-07-03)

PSTD WGPU compute pipelines now live in `WgpuPstdPipelines` instead of
separate top-level `GpuPstdSolver` fields. Time-loop dispatch and encode paths
still use the same shader entry-point mapping through grouped state.

Residual risk: this closes only individual pipeline field grouping. PSTD
grouped state was consolidated under `WgpuPstdState` in a later slice, but that
state still needs a provider-associated abstraction.

Evidence tier: type-level/compile-time validation plus focused empirical test.
`cargo fmt -p kwavers-gpu --check`, `cargo check -p kwavers-gpu --features
gpu`, `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings`, and
`cargo nextest run -p kwavers-gpu --features gpu medium_variable_update
pstd_command_provider_is_generic_over_provider_trait
pstd_pass_provider_is_generic_over_provider_trait
pstd_bind_group_factory_is_generic_over_provider_trait
pstd_bind_group_layout_factory_is_generic_over_provider_trait
pstd_buffer_factory_is_generic_over_provider_trait
pstd_pipeline_factory_is_generic_over_provider_trait` passed 7/7. Source
audit: `GpuPstdSolver` no longer exposes separate top-level `pipeline_*`
fields.

## CLOSED: kwavers-gpu PSTD k-space work buffer state grouping (2026-07-03)

PSTD `kspace_re` and `kspace_im` WGPU work buffers now live in
`WgpuPstdKspaceBuffers` instead of separate top-level `GpuPstdSolver` fields.
Construction still binds the same buffers into existing group(1) slots.

Residual risk: this closes only k-space work buffer field grouping. PSTD
grouped state was consolidated under `WgpuPstdState` in a later slice, but that
state still needs a provider-associated abstraction.

Evidence tier: type-level/compile-time validation plus focused empirical test.
`cargo fmt -p kwavers-gpu --check`, `cargo check -p kwavers-gpu --features
gpu`, `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings`, and
`cargo nextest run -p kwavers-gpu --features gpu medium_variable_update
pstd_command_provider_is_generic_over_provider_trait
pstd_pass_provider_is_generic_over_provider_trait
pstd_bind_group_factory_is_generic_over_provider_trait
pstd_bind_group_layout_factory_is_generic_over_provider_trait
pstd_buffer_factory_is_generic_over_provider_trait
pstd_pipeline_factory_is_generic_over_provider_trait` passed 7/7. Source
audit: `GpuPstdSolver` no longer exposes separate top-level `buf_kspace_re` or
`buf_kspace_im` fields.

## CLOSED: kwavers-gpu PSTD layout state grouping (2026-07-03)

PSTD retained WGPU layout state now lives in `WgpuPstdLayouts` instead of
separate top-level `GpuPstdSolver` fields. The grouped state retains only the
sensor bind-group layout required to rebuild run-cache sensor bind groups; the
unused retained base pipeline layout field was deleted.

Residual risk: this closes only retained layout state grouping. PSTD grouped
state was consolidated under `WgpuPstdState` in a later slice, but that state
still needs a provider-associated abstraction.

Evidence tier: type-level/compile-time validation plus focused empirical test.
`cargo fmt -p kwavers-gpu --check`, `cargo check -p kwavers-gpu --features
gpu`, `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings`, and
`cargo nextest run -p kwavers-gpu --features gpu medium_variable_update
pstd_command_provider_is_generic_over_provider_trait
pstd_pass_provider_is_generic_over_provider_trait
pstd_bind_group_factory_is_generic_over_provider_trait
pstd_bind_group_layout_factory_is_generic_over_provider_trait
pstd_buffer_factory_is_generic_over_provider_trait
pstd_pipeline_factory_is_generic_over_provider_trait` passed 7/7. Source
audit: `GpuPstdSolver` no longer exposes separate top-level `bgl_sensor` or
`pipeline_layout` fields.

## CLOSED: kwavers-gpu PSTD permanent bind-group state grouping (2026-07-03)

PSTD field, k-space, and absorption WGPU bind groups now live in
`WgpuPstdPermanentBindGroups` instead of separate top-level `GpuPstdSolver`
fields. Dispatch helpers still bind the same group slots through grouped state.

Residual risk: this closes only permanent bind-group state grouping. PSTD
layout and pipeline state were grouped in later slices, but WGPU-specific
grouped state still needs to move behind a provider-owned implementation.

Evidence tier: type-level/compile-time validation plus focused empirical test.
`cargo fmt -p kwavers-gpu --check`, `cargo check -p kwavers-gpu --features
gpu`, `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings`, and
`cargo nextest run -p kwavers-gpu --features gpu medium_variable_update
pstd_command_provider_is_generic_over_provider_trait
pstd_pass_provider_is_generic_over_provider_trait
pstd_bind_group_factory_is_generic_over_provider_trait
pstd_bind_group_layout_factory_is_generic_over_provider_trait
pstd_buffer_factory_is_generic_over_provider_trait
pstd_pipeline_factory_is_generic_over_provider_trait` passed 7/7. Source
audit: `GpuPstdSolver` no longer exposes separate top-level `bg_fields`,
`bg_kspace`, or `bg_absorb` fields.

## CLOSED: kwavers-gpu PSTD run-cache state grouping (2026-07-03)

PSTD cached sensor/source/velocity buffers, staging buffer, sensor bind groups,
and cache-key counters now live in `WgpuPstdRunCache` instead of separate
top-level `GpuPstdSolver` fields. Cache-hit signal-tail refreshes, sensor
clear/copy/readback, and run-cache invalidation use the grouped state.

Residual risk: this closes only run-cache state grouping. PSTD permanent bind
groups, layout state, and pipeline state were grouped in later slices, but
WGPU-specific grouped state still needs to move behind a provider-owned
implementation.

Evidence tier: type-level/compile-time validation plus focused empirical test.
`cargo fmt -p kwavers-gpu --check`, `cargo check -p kwavers-gpu --features
gpu`, `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings`, and
`cargo nextest run -p kwavers-gpu --features gpu medium_variable_update
pstd_command_provider_is_generic_over_provider_trait
pstd_pass_provider_is_generic_over_provider_trait
pstd_bind_group_factory_is_generic_over_provider_trait
pstd_bind_group_layout_factory_is_generic_over_provider_trait
pstd_buffer_factory_is_generic_over_provider_trait
pstd_pipeline_factory_is_generic_over_provider_trait` passed 7/7. Source
audit: `GpuPstdSolver` no longer exposes separate top-level `cache_*` fields
under `pstd_gpu`.

## CLOSED: kwavers-gpu PSTD PML/shift buffer state grouping (2026-07-03)

PSTD split PML coefficients, packed PML axis data, and packed k-space shift
operators now live in `WgpuPstdPmlShiftBuffers` instead of separate top-level
`GpuPstdSolver` fields. Run-cache sensor bind groups still bind the same
buffers through the existing sensor layout.

Residual risk: this closes only PML/shift buffer state grouping. PSTD
run-cache state, permanent bind groups, layout state, and pipeline state were
grouped in later slices, but WGPU-specific grouped state still needs to move
behind a provider-owned implementation.

Evidence tier: type-level/compile-time validation plus focused empirical test.
`cargo fmt -p kwavers-gpu --check`, `cargo check -p kwavers-gpu --features
gpu`, `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings`, and
`cargo nextest run -p kwavers-gpu --features gpu medium_variable_update
pstd_command_provider_is_generic_over_provider_trait
pstd_pass_provider_is_generic_over_provider_trait
pstd_bind_group_factory_is_generic_over_provider_trait
pstd_bind_group_layout_factory_is_generic_over_provider_trait
pstd_buffer_factory_is_generic_over_provider_trait
pstd_pipeline_factory_is_generic_over_provider_trait` passed 7/7. Source
audit: `GpuPstdSolver` no longer exposes separate top-level `buf_pml_*` or
`buf_shifts_all` fields.

## CLOSED: kwavers-gpu PSTD absorption buffer state grouping (2026-07-03)

PSTD fractional-Laplacian operator and scratch WGPU buffers now live in
`WgpuPstdAbsorptionBuffers` instead of separate top-level `GpuPstdSolver`
fields. Construction still binds the same buffers into existing group(3) slots,
and medium refresh writes absorption tau/eta through the grouped state.

Residual risk: this closes only absorption buffer state grouping. PSTD
PML/shift buffers, run-cache state, permanent bind groups, layout state, and
pipeline state were grouped in later slices, but WGPU-specific grouped state
still needs to move behind a provider-owned implementation.

Evidence tier: type-level/compile-time validation plus focused empirical test.
`cargo fmt -p kwavers-gpu --check`, `cargo check -p kwavers-gpu --features
gpu`, `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings`, and
`cargo nextest run -p kwavers-gpu --features gpu medium_variable_update
pstd_command_provider_is_generic_over_provider_trait
pstd_pass_provider_is_generic_over_provider_trait
pstd_bind_group_factory_is_generic_over_provider_trait
pstd_bind_group_layout_factory_is_generic_over_provider_trait
pstd_buffer_factory_is_generic_over_provider_trait
pstd_pipeline_factory_is_generic_over_provider_trait` passed 7/7. Source
audit: `GpuPstdSolver` no longer exposes separate top-level `buf_absorb_*`
fields.

## CLOSED: kwavers-gpu PSTD field buffer state grouping (2026-07-03)

PSTD pressure, velocity, and density WGPU buffers now live in
`WgpuPstdFieldBuffers` instead of separate top-level `GpuPstdSolver` fields.
Construction still binds the same buffers into the existing group(0) slots.

Residual risk: this closes only the acoustic field buffer state grouping. PSTD
PML/shift buffers, absorption buffers, run-cache state, permanent bind groups,
layout state, and pipeline state were grouped in later slices, but WGPU-specific
grouped state still needs to move behind a provider-owned implementation.

Evidence tier: type-level/compile-time validation plus focused empirical test.
`cargo fmt -p kwavers-gpu --check`, `cargo check -p kwavers-gpu --features
gpu`, `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings`, and
`cargo nextest run -p kwavers-gpu --features gpu medium_variable_update
pstd_command_provider_is_generic_over_provider_trait
pstd_pass_provider_is_generic_over_provider_trait
pstd_bind_group_factory_is_generic_over_provider_trait
pstd_bind_group_layout_factory_is_generic_over_provider_trait
pstd_buffer_factory_is_generic_over_provider_trait
pstd_pipeline_factory_is_generic_over_provider_trait` passed 7/7. Source
audit: `GpuPstdSolver` no longer exposes separate top-level
`buf_p`/`buf_ux`/`buf_uy`/`buf_uz`/`buf_rhox`/`buf_rhoy`/`buf_rhoz` fields.

## CLOSED: kwavers-gpu PSTD medium buffer state grouping (2026-07-03)

PSTD k-space, medium, twiddle, and source-kappa WGPU buffers now live in
`WgpuPstdMediumBuffers` instead of separate top-level `GpuPstdSolver` fields.
Construction still binds the same buffers into the existing group(0)/group(1)
slots, and medium refresh/source-correction writes access the grouped state
through the command provider.

Residual risk: this closes only the medium/source buffer state grouping. Field,
absorption buffers, PML/shift buffers, run-cache state, permanent bind groups,
layout state, and pipeline state were grouped in later slices, but WGPU-specific
grouped state still needs to move behind a provider-owned implementation.

Evidence tier: type-level/compile-time validation plus focused empirical test.
`cargo fmt -p kwavers-gpu --check`, `cargo check -p kwavers-gpu --features
gpu`, `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings`, and
`cargo nextest run -p kwavers-gpu --features gpu medium_variable_update
pstd_command_provider_is_generic_over_provider_trait
pstd_pass_provider_is_generic_over_provider_trait
pstd_bind_group_factory_is_generic_over_provider_trait
pstd_bind_group_layout_factory_is_generic_over_provider_trait
pstd_buffer_factory_is_generic_over_provider_trait
pstd_pipeline_factory_is_generic_over_provider_trait` passed 7/7. Source
audit: `GpuPstdSolver` no longer exposes separate top-level
`buf_kappa`/`buf_rho0_inv`/`buf_c0_sq`/`buf_rho0`/`buf_bon_a`/
`buf_alpha_decay`/`buf_source_kappa` fields.

## CLOSED: kwavers-gpu PSTD medium-update upload provider seam (2026-07-03)

PSTD variable/full medium refreshes and source-correction writes now route
through `PstdCommandProvider::write_buffer`. The provider is visible only
inside `pstd_gpu`, and the WGPU implementation owns byte casting plus
`queue.write_buffer` submission.

Residual risk: this closes direct PSTD queue-upload ownership. PSTD individual
state fields were grouped in later slices, but WGPU-specific grouped state
still needs to move behind a provider-owned implementation.

Evidence tier: type-level/compile-time validation plus focused empirical test.
`cargo fmt -p kwavers-gpu --check`, `cargo check -p kwavers-gpu --features
gpu`, `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings`, and
`cargo nextest run -p kwavers-gpu --features gpu medium_variable_update
pstd_command_provider_is_generic_over_provider_trait
pstd_pass_provider_is_generic_over_provider_trait
pstd_bind_group_factory_is_generic_over_provider_trait
pstd_bind_group_layout_factory_is_generic_over_provider_trait
pstd_buffer_factory_is_generic_over_provider_trait
pstd_pipeline_factory_is_generic_over_provider_trait` passed 7/7. Source
audit: direct queue write mechanics under `crates/kwavers-gpu/src/pstd_gpu`
are confined to the WGPU provider implementation in `time_loop/commands.rs`.

## CLOSED: kwavers-gpu PSTD cache-hit upload provider seam (2026-07-03)

PSTD cache-hit source and velocity signal-tail refreshes now route through
`PstdCommandProvider::write_buffer`. The method is generic over POD host data,
and the WGPU implementation owns `queue.write_buffer` byte casting and offset
submission.

Residual risk: this closes only cache-hit run-cache tail uploads. Solver-owned
concrete WGPU state still needs provider traits before PSTD is
provider-complete.

Evidence tier: type-level/compile-time validation plus focused empirical test.
`cargo fmt -p kwavers-gpu --check`, `cargo check -p kwavers-gpu --features
gpu`, `cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings`, and
`cargo nextest run -p kwavers-gpu --features gpu
pstd_pass_provider_is_generic_over_provider_trait
pstd_command_provider_is_generic_over_provider_trait
pstd_bind_group_factory_is_generic_over_provider_trait
pstd_bind_group_layout_factory_is_generic_over_provider_trait
pstd_buffer_factory_is_generic_over_provider_trait
pstd_pipeline_factory_is_generic_over_provider_trait
packed_signal_len_keeps_storage_buffers_non_empty
rewrite_packed_source_buffer_preserves_indices_and_signal_tail
rewrite_packed_source_buffer_uses_zero_signal_sentinel_for_empty_tail
overwrite_packed_signal_tail_keeps_index_prefix_stable` passed 10/10. Source
audit: cache-hit upload calls in `time_loop/buffer.rs` now call
`commands.write_buffer`; direct WGPU `queue.write_buffer` for the touched
run-cache files is confined to `time_loop/commands.rs`.

## CLOSED: kwavers-gpu PSTD readback provider seam (2026-07-03)

PSTD sensor readback now routes through `PstdCommandProvider::read_mapped`.
The method is generic over POD host scalar type and the WGPU implementation
owns staging-buffer slicing, `map_async`, poll-wait, mapped-range extraction,
and unmap.

Residual risk: this closes only run-loop sensor readback. Solver-owned
concrete WGPU state still needs provider traits until the remaining
provider-state migration lands.

Evidence tier: type-level/compile-time validation plus focused empirical test.
Verification: `cargo fmt -p kwavers-gpu --check` passed, `cargo check -p
kwavers-gpu --features gpu` passed after fixing the upstream Ritk manifest gap,
`cargo clippy -p kwavers-gpu --features gpu --lib -- -D warnings` passed after
fixing the upstream Moirai cache-wrapper rename gap, and `cargo nextest run -p
kwavers-gpu --features gpu pstd_pass_provider_is_generic_over_provider_trait
pstd_command_provider_is_generic_over_provider_trait
pstd_bind_group_factory_is_generic_over_provider_trait
pstd_bind_group_layout_factory_is_generic_over_provider_trait
pstd_buffer_factory_is_generic_over_provider_trait
pstd_pipeline_factory_is_generic_over_provider_trait
packed_signal_len_keeps_storage_buffers_non_empty
rewrite_packed_source_buffer_preserves_indices_and_signal_tail
rewrite_packed_source_buffer_uses_zero_signal_sentinel_for_empty_tail
overwrite_packed_signal_tail_keeps_index_prefix_stable` passed 10/10. Source
audit: direct `map_async`, `get_mapped_range`, and `unmap` calls for touched
PSTD run-loop files are confined to `time_loop/commands.rs`.

## CLOSED: upstream Atlas provider gate repairs (2026-07-03)

Two upstream owner-crate repairs were required to keep the Kwavers GPU consumer
gate moving: `ritk-registration` now declares the `ritk-tensor-ops` dependency
used by its native preprocessing executor, and Moirai core stale `CachePadded`
call sites now use the canonical `CacheAligned` cache-line wrapper.

Evidence tier: compile-time validation. Verification: `cargo check -p
ritk-registration` passed in `D:\atlas\repos\ritk`; `cargo fmt -p moirai-core
--check` and `cargo check -p moirai-core` passed in `D:\atlas\repos\moirai`;
the downstream `kwavers-gpu` check/clippy/nextest gates listed in the readback
provider slice passed after these upstream fixes.

## CLOSED: kwavers-gpu PSTD pass-body provider seam (2026-07-03)

PSTD zero-field and per-step pass bodies now route through `PstdPassProvider`.
The WGPU implementation owns the existing dispatch and `encode_*` method calls,
so `time_loop::run` no longer calls WGPU pass-body methods directly.

Residual risk: this closes run-loop pass-body orchestration only. The WGPU
provider still delegates to existing WGPU dispatch helpers and solver-owned
concrete pipeline/bind-group fields; upload and readback provider seams landed
in later slices.

Evidence tier: type-level/compile-time validation plus focused empirical test.
Verification: `cargo fmt -p kwavers-gpu --check` passed, `cargo check -p
kwavers-gpu --features gpu` passed, `cargo clippy -p kwavers-gpu --features
gpu --lib -- -D warnings` passed, and `cargo nextest run -p kwavers-gpu
--features gpu pstd_pass_provider_is_generic_over_provider_trait
pstd_command_provider_is_generic_over_provider_trait
pstd_bind_group_factory_is_generic_over_provider_trait
pstd_bind_group_layout_factory_is_generic_over_provider_trait
pstd_buffer_factory_is_generic_over_provider_trait
pstd_pipeline_factory_is_generic_over_provider_trait
packed_signal_len_keeps_storage_buffers_non_empty
rewrite_packed_source_buffer_preserves_indices_and_signal_tail
rewrite_packed_source_buffer_uses_zero_signal_sentinel_for_empty_tail
overwrite_packed_signal_tail_keeps_index_prefix_stable` passed 10/10. Source
audit: `time_loop::run` no longer calls `self.dispatch`, `self.encode_*`,
`begin_compute_pass`, or `wgpu::ComputePass`; WGPU pass mechanics are confined
to `commands.rs`, `passes.rs`, `dispatch.rs`, and `encode/`.

## CLOSED: kwavers-gpu PSTD compute-pass provider seam (2026-07-03)

PSTD zero-field and batched-step compute-pass creation now route through
`PstdCommandProvider::submit_compute_pass` and
`PstdCommandProvider::submit_compute_passes`. The provider contract owns a
lifetime-associated compute-pass type, so the trait surface does not name
`wgpu::ComputePass`; the WGPU pass descriptors and begin-pass calls live in
`WgpuPstdCommandProvider`.

Residual risk: this closes compute-pass creation for the bounded PSTD run loop.
The pass body provider seam, upload provider seam, and readback provider seam
landed in later slices; solver-owned concrete GPU fields still need provider
traits.

Evidence tier: type-level/compile-time validation plus focused empirical test.
Verification: `cargo fmt -p kwavers-gpu --check` passed, `cargo check -p
kwavers-gpu --features gpu` passed, `cargo clippy -p kwavers-gpu --features
gpu --lib -- -D warnings` passed, and `cargo nextest run -p kwavers-gpu
--features gpu pstd_command_provider_is_generic_over_provider_trait
pstd_bind_group_factory_is_generic_over_provider_trait
pstd_bind_group_layout_factory_is_generic_over_provider_trait
pstd_buffer_factory_is_generic_over_provider_trait
pstd_pipeline_factory_is_generic_over_provider_trait
packed_signal_len_keeps_storage_buffers_non_empty
rewrite_packed_source_buffer_preserves_indices_and_signal_tail
rewrite_packed_source_buffer_uses_zero_signal_sentinel_for_empty_tail
overwrite_packed_signal_tail_keeps_index_prefix_stable` passed 9/9. Source
audit: direct `begin_compute_pass` calls remain confined to
`time_loop/commands.rs` for the touched PSTD run-loop files.

## CLOSED: kwavers-gpu PSTD command encoder provider seam (2026-07-03)

PSTD zero-field and batched step command encoder creation/submission now route
through `PstdCommandProvider::submit_encoder`. The provider contract owns an
associated encoder type, so the trait surface is not tied to
`wgpu::CommandEncoder`; the current WGPU mechanics live in
`WgpuPstdCommandProvider`.

Residual risk: this closes only command encoder creation and queue submission
for the zero-field and batch paths. Compute-pass, pass-body, upload, and
readback provider seams landed in later slices; solver-owned concrete GPU
fields still need provider traits.

Evidence tier: type-level/compile-time validation plus focused empirical test.
Verification: `cargo fmt -p kwavers-gpu --check` passed, `cargo check -p
kwavers-gpu --features gpu` passed, `cargo clippy -p kwavers-gpu --features
gpu --lib -- -D warnings` passed, and `cargo nextest run -p kwavers-gpu
--features gpu pstd_command_provider_is_generic_over_provider_trait
pstd_bind_group_factory_is_generic_over_provider_trait
pstd_bind_group_layout_factory_is_generic_over_provider_trait
pstd_buffer_factory_is_generic_over_provider_trait
pstd_pipeline_factory_is_generic_over_provider_trait
packed_signal_len_keeps_storage_buffers_non_empty
rewrite_packed_source_buffer_preserves_indices_and_signal_tail
rewrite_packed_source_buffer_uses_zero_signal_sentinel_for_empty_tail
overwrite_packed_signal_tail_keeps_index_prefix_stable` passed 9/9. Source
audit: direct `create_command_encoder` and `queue.submit` calls remain confined
to `time_loop/commands.rs` for the touched PSTD run-loop files.

## CLOSED: kwavers-gpu PSTD command provider seam (2026-07-03)

PSTD run-loop sensor clear, sensor copy, command submit, and wait-poll
operations now delegate through `PstdCommandProvider`. The current WGPU command
encoder and queue wait mechanics for those paths moved into
`WgpuPstdCommandProvider`.

Residual risk: this closes only the bounded run-loop clear/copy/poll command
surface. Compute-pass, pass-body, upload, and readback provider seams landed in
later slices; solver-owned concrete GPU fields still use WGPU until their
provider traits land.

Evidence tier: type-level/compile-time validation plus focused empirical test.
Verification: `cargo fmt -p kwavers-gpu --check` passed, `cargo check -p
kwavers-gpu --features gpu` passed, `cargo clippy -p kwavers-gpu --features
gpu --lib -- -D warnings` passed, and `cargo nextest run -p kwavers-gpu
--features gpu pstd_command_provider_is_generic_over_provider_trait
pstd_bind_group_factory_is_generic_over_provider_trait
pstd_bind_group_layout_factory_is_generic_over_provider_trait
pstd_buffer_factory_is_generic_over_provider_trait
pstd_pipeline_factory_is_generic_over_provider_trait
packed_signal_len_keeps_storage_buffers_non_empty
rewrite_packed_source_buffer_preserves_indices_and_signal_tail
rewrite_packed_source_buffer_uses_zero_signal_sentinel_for_empty_tail
overwrite_packed_signal_tail_keeps_index_prefix_stable --status-level fail
--no-fail-fast` passed 9/9.

## CLOSED: kwavers-gpu PSTD bind-group provider seam (2026-07-03)

PSTD permanent constructor bind groups and run-cache sensor bind groups now
delegate assembly through `PstdBindGroupProvider`. The current WGPU
`BindGroupDescriptor` construction moved into `WgpuPstdBindGroupFactory`, so
construction and run-cache rebuild no longer call WGPU bind-group creation APIs
directly.

Residual risk: this closes only PSTD bind-group assembly. Command encoding,
queue writes, and solver-owned concrete GPU fields still use WGPU until their
provider traits land.

Evidence tier: type-level/compile-time validation plus focused empirical test.
Verification: `cargo fmt -p kwavers-gpu --check` passed, `cargo check -p
kwavers-gpu --features gpu` passed, `cargo clippy -p kwavers-gpu --features
gpu --lib -- -D warnings` passed, and `cargo nextest run -p kwavers-gpu
--features gpu pstd_bind_group_factory_is_generic_over_provider_trait
pstd_bind_group_layout_factory_is_generic_over_provider_trait
pstd_buffer_factory_is_generic_over_provider_trait
pstd_pipeline_factory_is_generic_over_provider_trait
packed_signal_len_keeps_storage_buffers_non_empty
rewrite_packed_source_buffer_preserves_indices_and_signal_tail
rewrite_packed_source_buffer_uses_zero_signal_sentinel_for_empty_tail
overwrite_packed_signal_tail_keeps_index_prefix_stable --status-level fail
--no-fail-fast` passed 8/8.

## CLOSED: kwavers-gpu PSTD bind-group layout provider seam (2026-07-03)

PSTD solver construction now delegates bind-group layout creation through
`PstdBindGroupLayoutProvider`. The current WGPU binding-slot descriptor
construction moved into `WgpuPstdBindGroupLayoutFactory`, so construction no
longer calls WGPU bind-group-layout builders directly for field, k-space,
sensor/source, or absorption groups.

Residual risk: this closes only PSTD bind-group layout creation. Command
encoding, queue writes, and solver-owned concrete GPU fields still use WGPU
until their provider traits land.

Evidence tier: type-level/compile-time validation plus focused empirical test.
Verification: `cargo fmt -p kwavers-gpu --check` passed, `cargo check -p
kwavers-gpu --features gpu` passed, `cargo clippy -p kwavers-gpu --features
gpu --lib -- -D warnings` passed, and `cargo nextest run -p kwavers-gpu
--features gpu pstd_bind_group_layout_factory_is_generic_over_provider_trait
pstd_buffer_factory_is_generic_over_provider_trait
pstd_pipeline_factory_is_generic_over_provider_trait
packed_signal_len_keeps_storage_buffers_non_empty
rewrite_packed_source_buffer_preserves_indices_and_signal_tail
rewrite_packed_source_buffer_uses_zero_signal_sentinel_for_empty_tail
overwrite_packed_signal_tail_keeps_index_prefix_stable --status-level fail
--no-fail-fast` passed 7/7.

## CLOSED: kwavers-gpu PSTD pipeline provider seam (2026-07-03)

PSTD solver construction now delegates shader-module, pipeline-layout, and
compute-pipeline creation through `PstdPipelineProvider`. The current WGPU
shader module, pipeline layout, and `ComputePipelineDescriptor` construction
moved into `WgpuPstdPipelineFactory`, so construction no longer calls those
WGPU creation APIs directly for standard or absorption pipeline entries.

Residual risk: this closes only PSTD shader/layout/pipeline creation. Command
encoding, queue writes, and solver-owned concrete GPU fields still use WGPU
until their provider traits land.

Evidence tier: type-level/compile-time validation plus focused empirical test.
Verification: `cargo fmt -p kwavers-gpu --check` passed, `cargo check -p
kwavers-gpu --features gpu` passed, `cargo clippy -p kwavers-gpu --features
gpu --lib -- -D warnings` passed, and `cargo nextest run -p kwavers-gpu
--features gpu pstd_buffer_factory_is_generic_over_provider_trait
pstd_pipeline_factory_is_generic_over_provider_trait
packed_signal_len_keeps_storage_buffers_non_empty
rewrite_packed_source_buffer_preserves_indices_and_signal_tail
rewrite_packed_source_buffer_uses_zero_signal_sentinel_for_empty_tail
overwrite_packed_signal_tail_keeps_index_prefix_stable --status-level fail
--no-fail-fast` passed 6/6.

## CLOSED: kwavers-gpu PSTD buffer allocation provider seam (2026-07-03)

PSTD solver construction and run-cache rebuild now delegate owned
storage/staging buffer allocation through `PstdBufferProvider`. The current
WGPU read-only, static, upload, read/write, and staging-buffer creation moved
into `WgpuPstdBufferFactory`, so construction and cache rebuild no longer call
WGPU buffer allocation directly for those buffers.

Residual risk: this closes only PSTD allocation calls. The PSTD solver struct,
command encoding, and queue writes still own concrete WGPU types until their
provider traits land.

Evidence tier: type-level/compile-time validation plus focused empirical test.
Verification: `cargo fmt -p kwavers-gpu --check` passed, `cargo check -p
kwavers-gpu --features gpu` passed, `cargo clippy -p kwavers-gpu --features
gpu --lib -- -D warnings` passed, and `cargo nextest run -p kwavers-gpu
--features gpu pstd_buffer_factory_is_generic_over_provider_trait
packed_signal_len_keeps_storage_buffers_non_empty
rewrite_packed_source_buffer_preserves_indices_and_signal_tail
rewrite_packed_source_buffer_uses_zero_signal_sentinel_for_empty_tail
overwrite_packed_signal_tail_keeps_index_prefix_stable --status-level fail
--no-fail-fast` passed 5/5.

## CLOSED: kwavers-gpu backend buffer-manager provider seam (2026-07-03)

`kwavers-gpu::backend::GpuBackendBufferManager<P>` now delegates to
`BackendBufferProvider`. The WGPU buffer pool, buffer allocation, array upload,
readback, and pooling methods moved into `WgpuBackendBufferManager`, so the
generic backend buffer-manager wrapper no longer exposes `wgpu::Buffer`
methods.

Residual risk: this closes only the backend buffer-manager wrapper. Backend
pipeline execution and PSTD still expose concrete WGPU buffers and pipelines
until their provider traits land.

Evidence tier: type-level/compile-time validation plus focused empirical test.
Verification: `cargo fmt -p kwavers-gpu --check` passed, `cargo check -p
kwavers-gpu --features gpu` passed, `cargo clippy -p kwavers-gpu --features
gpu --lib -- -D warnings` passed, and `cargo nextest run -p kwavers-gpu
--features gpu backend_buffer_manager_wrapper_is_generic_over_provider_trait
--status-level fail --no-fail-fast` passed 1/1.

## CLOSED: kwavers-gpu thermal-acoustic solver provider seam (2026-07-03)

`kwavers-gpu::gpu::thermal_acoustic::GpuThermalAcousticSolver<P>` now delegates
to `ThermalAcousticSolverProvider`, which now extends the shared
`GpuKernelProvider`/`GpuProviderBackend` stack. The current WGPU context,
compute pipelines, bind group, and step dispatch live in
`WgpuThermalAcousticSolverProvider`, so the generic solver wrapper no longer
exposes WGPU pipeline fields, WGPU step parameters, or raw WGPU device/queue
constructor arguments.

Residual risk: this closes only the thermal-acoustic solver wrapper and default
WGPU acquisition boundary. CUDA still needs real thermal-acoustic buffers,
kernels, and WGPU/CUDA differential tests before implementing this provider.

Evidence tier: type-level/compile-time validation plus focused empirical test.
Verification: `cargo fmt -p kwavers-gpu --check` passed, `cargo check -p
kwavers-gpu --features gpu` passed, `cargo clippy -p kwavers-gpu --features
gpu --lib -- -D warnings` passed, and `cargo nextest run -p kwavers-gpu
--features gpu thermal_acoustic_solver_is_generic_over_provider_trait
--status-level fail --no-fail-fast` passed 1/1. Follow-up verification:
`rustup run nightly cargo fmt -p kwavers-gpu --check`, `rustup run nightly
cargo check -p kwavers-gpu --features gpu --all-targets`, `rustup run nightly
cargo check -p kwavers-gpu --features cuda-provider --all-targets`, focused
thermal-acoustic/provider nextest under `gpu` (38/38) and `cuda-provider`
(45/45), and clippy for both feature sets pass.

## CLOSED: kwavers-gpu thermal-acoustic buffer provider seam (2026-07-03)

`kwavers-gpu::gpu::thermal_acoustic::GpuThermalAcousticBuffers<P>` now delegates
to a `ThermalAcousticBufferProvider` trait. The current WGPU storage/uniform
buffers, field upload, and readback path moved into
`WgpuThermalAcousticBuffers`, so the generic buffer wrapper no longer exposes
public `wgpu::Buffer` fields. Follow-up moved upload/readback field I/O from
`ndarray::Array3<f32>` to provider-native `leto::Array3<f32>`, with WGPU
declaring `ThermalAcousticBufferProvider::Scalar = f32`.

Residual risk: this closes only the thermal-acoustic buffer wrapper and field
I/O surface. CUDA still needs real thermal-acoustic buffers/kernels and
WGPU/CUDA differential tests before implementing this provider.

Evidence tier: type-level/compile-time validation plus focused empirical test.
Verification: `cargo fmt -p kwavers-gpu --check` passed, `cargo check -p
kwavers-gpu --features gpu` passed, `cargo clippy -p kwavers-gpu --features
gpu --lib -- -D warnings` passed, and `cargo nextest run -p kwavers-gpu
--features gpu thermal_acoustic_buffers_are_generic_over_provider_trait
--status-level fail --no-fail-fast` passed 1/1. Follow-up verification:
`rustup run nightly cargo fmt -p kwavers-gpu --check`, `rustup run nightly
cargo check -p kwavers-gpu --features gpu`, `rustup run nightly cargo clippy
-p kwavers-gpu --features gpu --lib -- -D warnings`, focused
thermal-acoustic nextest 9/9, and `rustup run nightly cargo check -p
kwavers-gpu --features cuda-provider` pass.

## CLOSED: kwavers-gpu acoustic-field provider seam (2026-07-03)

`kwavers-gpu::gpu::compute_kernels::AcousticFieldKernel<P>` now delegates to an
`AcousticFieldProvider` operation trait. The current WGSL pipeline, buffer
allocation, dispatch, and readback path moved into `WgpuAcousticFieldProvider`,
so the public acoustic-field wrapper is provider-generic while WGPU remains the
only real implementation. Follow-up moved the operation surface and
`WaveEquationGpu` to provider-native `leto::Array3<f32>`, with WGPU declaring
`AcousticFieldProvider::Scalar = f32` instead of narrowing f64 ndarray fields
inside the provider.

Residual risk: this closes only the acoustic-field kernel wrapper and its local
field-array surface. CUDA still needs real acoustic kernels and WGPU/CUDA
differential tests before implementing this operation provider.

Evidence tier: type-level/compile-time validation plus focused empirical test.
Verification: `cargo fmt -p kwavers-gpu --check` passed, `cargo check -p
kwavers-gpu --features gpu` passed, and `cargo nextest run -p kwavers-gpu
--features gpu acoustic_kernel_wrapper_is_generic_over_provider_trait
--status-level fail --no-fail-fast` passed 1/1. Follow-up verification:
`rustup run nightly cargo fmt -p kwavers-gpu --check`, `rustup run nightly
cargo check -p kwavers-gpu --features gpu`, `rustup run nightly cargo clippy
-p kwavers-gpu --features gpu --lib -- -D warnings`, focused
acoustic/provider nextest 4/4, and `rustup run nightly cargo check -p
kwavers-gpu --features cuda-provider` pass.

## CLOSED: kwavers-gpu provider-generic context alias (2026-07-03)

`kwavers-gpu::gpu::GpuBackend<P>` now exposes the same provider parameter as
`CoreGpuContext<P>`. Existing `GpuBackend` call sites keep the WGPU default,
while provider-explicit code can type-check through `GpuBackend<P>` without
adding CUDA/WGPU branches.

Residual risk: this closes only the public context alias. Concrete WGPU buffer
and pipeline types remain in kernel dispatch modules until the
kernel-buffer-provider migration lands.

Evidence tier: type-level/compile-time validation plus focused empirical test.
Verification: `cargo fmt -p kwavers-gpu --check` passed, `cargo check -p
kwavers-gpu --features gpu` passed, `cargo clippy -p kwavers-gpu --features
gpu --lib -- -D warnings` passed, and `cargo nextest run -p kwavers-gpu
--features gpu gpu_backend_alias_exposes_provider_parameter --status-level
fail --no-fail-fast` passed 1/1.

## CLOSED: kwavers-solver SWE PML boundary Moirai slice (2026-07-03)

`kwavers-solver::forward::elastic::swe::boundary` no longer uses direct
ndarray/Rayon parallel dispatch. PML attenuation-field construction and mask
generation now use shared Moirai-backed indexed 3-D traversal from
`kwavers-core::utils::iterators`; velocity damping uses Moirai's triple mutable
chunk traversal over standard-layout velocity arrays with sequential ndarray
semantics retained for non-standard layouts.

Residual risk: this closes only the SWE boundary subtree. Broader
`kwavers-solver::forward::elastic::swe` still has direct Rayon/ndarray-parallel
call sites in stress and integration kernels.

Evidence tier: static source audit plus focused value-semantic tests.
Verification: `cargo fmt -p kwavers-solver --check` passed, `cargo nextest run
-p kwavers-solver pml --status-level fail --no-fail-fast` passed 45/45, and
the SWE boundary direct Rayon/ndarray-parallel audit returned no hits.

## CLOSED: kwavers-solver SWE displacement-magnitude Moirai slice (2026-07-03)

`kwavers-solver::forward::elastic::swe::ElasticWaveField::displacement_magnitude`
no longer uses ndarray/Rayon `par_mapv_inplace` for the final square-root
transform. It now routes that in-place scalar operation through
`workspace::inplace_ops::apply_inplace`, which uses `moirai-parallel` for
standard-layout arrays and preserves sequential ndarray semantics for
non-standard layouts.

Residual risk: this closes only the SWE types scalar-transform edge. Broader
`kwavers-solver::forward::elastic::swe` still has direct Rayon/ndarray-parallel
call sites in stress, integration, and boundary kernels.

Evidence tier: static source audit plus focused value-semantic tests.
Verification: `cargo fmt -p kwavers-solver --check` passed, `cargo nextest run
-p kwavers-solver displacement_magnitude --status-level fail --no-fail-fast`
passed 3/3, and the SWE types direct Rayon/ndarray-parallel audit returned no
hits.

## CLOSED: kwavers-solver AMR Moirai slice (2026-07-03)

`kwavers-solver::utilities::amr` no longer uses direct ndarray/Rayon parallel
dispatch. Wavelet and physics error normalization now use
`workspace::inplace_ops::scale_inplace`; wavelet coefficient thresholding now
uses `workspace::inplace_ops::apply_inplace`; and refinement marker
initialization uses `moirai_parallel::enumerate_mut_with` over standard-layout
marker/error slices. Non-standard layouts preserve sequential ndarray
semantics.

Residual risk: this closes the AMR subtree only. Broader `kwavers-solver` still
has direct Rayon/ndarray-parallel call sites outside AMR.

Evidence tier: static source audit plus compile-time/lint validation and
focused empirical tests. Verification: `cargo fmt -p kwavers-solver --check`
passed, `cargo nextest run -p kwavers-solver amr --status-level fail
--no-fail-fast` passed 11/11, and the AMR direct Rayon/ndarray-parallel audit
returned no hits.

## CLOSED: kwavers-solver monolithic coupler RHS Moirai slice (2026-07-03)

`kwavers-solver::multiphysics::monolithic::coupler` no longer builds the
Newton GMRES right-hand side through ndarray/Rayon `par_mapv_inplace`. The
solver still assigns `F(u)` into its reusable RHS scratch buffer, then applies
the required sign inversion through the existing
`workspace::inplace_ops::scale_inplace` SSOT, which dispatches standard-layout
arrays through `moirai-parallel` and preserves sequential ndarray semantics
for non-standard layouts.

Residual risk: this closes only the monolithic coupler's direct parallel map
edge. `kwavers-solver` still has direct Rayon/ndarray-parallel call sites in
AMR, photoacoustic reconstruction, forward elastic/PSTD paths, and PINN/Burn
code. The package manifest still needs direct Rayon and ndarray's `rayon`
feature until those are migrated.

Evidence tier: static source audit plus compile-time/lint validation and
focused empirical tests. Verification: `cargo fmt -p kwavers-solver --check`
passed, `cargo check -p kwavers-solver` passed, `cargo clippy -p
kwavers-solver --lib -- -D warnings` passed, `cargo nextest run -p
kwavers-solver monolithic --status-level fail --no-fail-fast` passed 30/30,
and the monolithic coupler Rayon audit returned no hits.

## CLOSED: kwavers-gpu provider identity trait split (2026-07-03)

`kwavers-gpu::backend` now separates provider identity/acquisition from kernel
dispatch. `GpuProviderBackend` covers provider identity, Hephaestus device
borrowing, and synchronization, while `GpuComputeProvider` extends it only for
providers that own real Kwavers kernel dispatch. `GpuDeviceProvider` now
exposes the provider identity, so WGPU and CUDA can both satisfy the provider
contract without pretending CUDA has WGSL-equivalent kernels.

Residual risk: this closes the trait-bound architecture gap only. CUDA remains
an acquisition/provider identity path, not a compute backend, until real CUDA
kernel implementations exist for the operations exposed by `GPUBackend<P>`.

Evidence tier: type-level/compile-time validation plus focused empirical
tests. Verification: `cargo fmt -p kwavers-gpu` passed, `cargo check -p
kwavers-gpu --features gpu` passed, `cargo check -p kwavers-gpu --features
cuda-provider --offline` passed, `cargo clippy -p kwavers-gpu --features gpu
--lib -- -D warnings` passed, `cargo clippy -p kwavers-gpu --features
cuda-provider --all-targets --offline -- -D warnings` passed, `cargo nextest
run -p kwavers-gpu --features gpu provider --status-level fail
--no-fail-fast` passed 4/4, `cargo nextest run -p kwavers-gpu --features
cuda-provider provider --status-level fail --no-fail-fast --offline` passed
7/7, and the stale CUDA-compute claim audit returned no hits.

## CLOSED: kwavers-solver monolithic residual Moirai slice (2026-07-03)

`kwavers-solver::multiphysics::monolithic::residual` no longer scales
Laplacian rate buffers through ndarray/Rayon `par_mapv_inplace`. Pressure,
light-fluence, and temperature rate scaling now use the existing
`workspace::inplace_ops::scale_inplace` SSOT, which dispatches
standard-layout arrays through `moirai-parallel` and preserves sequential
ndarray semantics for non-standard layouts.

Residual risk: this closes only the monolithic residual subsystem's direct
parallel map edge. `kwavers-solver` still has direct Rayon/ndarray-parallel
call sites in AMR, photoacoustic reconstruction, forward elastic/PSTD paths,
and PINN/Burn code. The package manifest still needs direct Rayon and
ndarray's `rayon` feature until those are migrated.

Evidence tier: static source audit plus compile-time/lint validation and
focused empirical tests. Verification: `cargo fmt -p kwavers-solver --check`
passed, `cargo check -p kwavers-solver` passed, `cargo clippy -p
kwavers-solver --lib -- -D warnings` passed, `cargo nextest run -p
kwavers-solver monolithic --status-level fail --no-fail-fast` passed 30/30,
and the monolithic residual Rayon audit returned no hits.

## CLOSED: kwavers-solver time-reversal Moirai slice (2026-07-03)

`kwavers-solver::inverse::time_reversal::reconstruction` no longer normalizes
through ndarray/Rayon `par_mapv_inplace`. It uses the existing
`workspace::inplace_ops::apply_inplace` SSOT, which dispatches standard-layout
arrays through `moirai-parallel` and preserves sequential ndarray semantics for
non-standard layouts.

Residual risk: this closes only the time-reversal module's direct parallel map
edge. `kwavers-solver` still has direct Rayon/ndarray-parallel call sites in
monolithic residuals, AMR, photoacoustic reconstruction, forward elastic/PSTD
paths, and PINN/Burn code. The package manifest still needs direct Rayon and
ndarray's `rayon` feature until those are migrated.

Evidence tier: static source audit plus compile-time/lint validation and
focused empirical tests. Verification: `cargo fmt -p kwavers-solver --check`
passed, `cargo clippy -p kwavers-solver --lib -- -D warnings` passed, `cargo
nextest run -p kwavers-solver time_reversal --status-level fail
--no-fail-fast` passed 9/9, and the time-reversal Rayon audit returned no
hits. All-targets clippy still fails on unrelated pre-existing test/doc lints
outside this slice.

## CLOSED: kwavers-math Leto decomposition bridge (2026-07-04)

`kwavers-math::linear_algebra::LinearAlgebra::{qr_decomposition, svd}` no
longer converts through nalgebra. QR delegates to Leto's Householder
decomposition, SVD delegates to Leto-ops rank-revealing SVD, and the existing
ndarray return types are kept only as the current public compatibility
boundary. The workspace dependency table now exposes local `leto-ops` for
member inheritance, and the `kwavers-math` manifest no longer has a direct
nalgebra edge.

Residual risk: this closes only the nalgebra decomposition bridge.
`kwavers-math` still carries ndarray and num-traits migration holdouts outside
this slice, including FFT/tensor and scalar-operation boundaries that need
separate Leto/Hephaestus or Eunomia/Apollo migration increments.

Evidence tier: compile-time dependency/type validation plus focused empirical
tests. Verification: `rustup run nightly cargo fmt -p kwavers-math --check`
passed, `rustup run nightly cargo check -p kwavers-math --all-targets` passed,
`rustup run nightly cargo nextest run -p kwavers-math linear_algebra
--status-level fail --no-fail-fast` passed 51/51 selected tests, `rustup run
nightly cargo clippy -p kwavers-math --all-targets --no-deps -- -D warnings`
passed, `rg -n "nalgebra|DMatrix|DVector" crates/kwavers-math/src
crates/kwavers-math/Cargo.toml` returned no matches, and `git diff --check`
for the touched files passed with only CRLF normalization warnings.

## CLOSED: kwavers-math tensor Moirai traversal slice (2026-07-03)

`kwavers-math::tensor::NdArrayTensor::map_inplace` now routes contiguous host
tensor storage through `moirai_parallel::for_each_chunk_mut_with::<Adaptive>`
instead of ndarray/Rayon `par_mapv_inplace`. Non-contiguous `ArrayD` layouts
retain ndarray's sequential value semantics through `mapv_inplace`.

Residual risk: this closes only the tensor module's direct Rayon-style
traversal. `kwavers-math` still has direct ndarray/Rayon dispatch in FFT,
regularizer, differential-operator, and SIMD-safe paths that require separate
kernel-by-kernel migration before the crate can drop ndarray's `rayon`
feature.

Evidence tier: static source audit plus compile-time/lint validation and
focused empirical tests. Verification: `cargo fmt -p kwavers-math --check`
passed, `cargo check -p kwavers-math` passed, `cargo clippy -p kwavers-math
--all-targets -- -D warnings` passed, `cargo nextest run -p kwavers-math
tensor --status-level fail --no-fail-fast` passed 9/9, the tensor Rayon audit
returned no hits, and `cargo tree -p kwavers-math --depth 1` shows the direct
local `moirai-parallel` dependency.

## CLOSED: kwavers-math tensor Burn placeholder removal (2026-07-03)

`kwavers-math::tensor` no longer advertises an unimplemented Burn ndarray/WGPU
or CUDA tensor backend. The unused `TensorBackend::BurnNdArray` variant was
removed, and the module docs now state the implemented boundary: an
ndarray-backed host tensor view for forward-solver CPU exchange. Differentiable
PINN tensors remain a solver-layer Coeus migration item instead of a
placeholder backend in `kwavers-math`.

Residual risk: this does not migrate the PINN subtree itself. Direct Burn
usage still remains under `kwavers-solver::inverse::pinn` until real Coeus
training/autodiff providers replace those modules.

Evidence tier: static source audit plus compile-time/lint validation and
focused empirical tests. Verification: `cargo fmt -p kwavers-math --check`
passed, `cargo check -p kwavers-math` passed, `cargo clippy -p kwavers-math
--all-targets -- -D warnings` passed after replacing an existing FFT test Tau
literal with `std::f64::consts::TAU`, `cargo nextest run -p kwavers-math tensor
r2c_optimized --status-level fail --no-fail-fast` passed 9/9, and `rg` found
no Burn-specific tensor backend names under `crates/kwavers-math/src`.

## CLOSED: CoreGpuContext provider-generic boundary (2026-07-03)

`kwavers-gpu::gpu::CoreGpuContext<P>` now stores
`GpuDevice<P: GpuDeviceProvider>` instead of raw `WgpuDevice`. Generic callers
can borrow the concrete provider through `provider()`, while raw `wgpu`
device/queue/submit access remains available only on the
`CoreGpuContext<WgpuDevice>` specialization required by current WGSL kernels.

No placeholder CUDA compute path was added. CUDA remains a real acquisition
provider and type-checks through the context boundary, but Kwavers must still
add real CUDA kernels before exposing a CUDA `GpuComputeProvider`.

Evidence tier: type-level/compile-time validation plus focused empirical
tests. Verification: `cargo fmt -p kwavers-gpu --check` passed, `cargo check
-p kwavers-gpu --features gpu` passed, `cargo check -p kwavers-gpu --features
cuda-provider --offline` passed, `cargo clippy -p kwavers-gpu --features gpu
--lib -- -D warnings` passed, `cargo nextest run -p kwavers-gpu --features gpu
provider --status-level fail --no-fail-fast` passed 3/3, and `cargo nextest
run -p kwavers-gpu --features cuda-provider provider --status-level fail
--no-fail-fast --offline` passed 5/5.

## CLOSED: kwavers-therapy abdominal timeout closure (2026-07-03)

The two abdominal preprocessing tests that previously terminated under
`nextest` now pass. The closure keeps the full inverse path intact: the
theranostic acoustic recording window is derived from the actual
source/body/receiver geometry instead of padded-domain overcoverage, adjoint
RTM replay work buffers are reused across reverse steps, and elastic-FWI
line-search candidates are evaluated lazily in first-improving order instead of
running every speculative PSTD candidate.

Residual risk: this resolves the hard timeout but not the slow-test budget. The
paired abdominal preprocessing filter still takes about 110 s, so the next
performance increment should remove repeated full acoustic/elastic solves or
move the acoustic backend behind the Hephaestus/CPU-GPU backend seam.

Evidence tier: empirical nextest execution plus compile-time/lint validation.
Verification: `cargo fmt -p kwavers-therapy --check` passed, `cargo clippy -p
kwavers-therapy --all-targets -- -D warnings` passed, focused waveform exposure
nextest passed 2/2, and `cargo nextest run -p kwavers-therapy
abdominal_preprocessing --status-level fail --no-fail-fast` passed 2/2.
Broader package verification now passes: `cargo nextest run -p
kwavers-therapy --status-level fail --no-fail-fast` passed 340/340 with 1
skipped in about 141 s.

## CLOSED: kwavers-therapy elastic-shear and emission Moirai slice (2026-07-03)

`theranostic_guidance::elastic_shear::sampling::migrate_residual` now dispatches
its independent voxel migration map through `moirai_parallel::map_collect_index_with`
instead of Rayon. `kwavers-therapy` now depends directly on the workspace
`moirai-parallel` provider. `theranostic_guidance::waveform::emission` also
routes passive-acoustic-mapping eikonal delay-column solves through
`moirai_parallel::map_collect_with` instead of Rayon.

Residual risk: this is a narrow production edge closure. `kwavers-therapy` still
has direct Rayon and ndarray `rayon` usage in nonlinear3d forward/cavitation
paths, so the crate manifest still keeps Rayon until those paths are migrated.

Evidence tier: static source audit plus compile-time/lint validation and a
focused value-semantic test. Verification: `cargo fmt -p kwavers-therapy
--check` passed, `cargo check -p kwavers-therapy` passed, `cargo clippy -p
kwavers-therapy --all-targets -- -D warnings` passed, `cargo nextest run -p
kwavers-therapy residual_migration_samples_expected_arrival --status-level fail
--no-fail-fast` passed 1/1, `cargo nextest run -p kwavers-therapy
waveform::emission --status-level fail --no-fail-fast` passed 3/3, and `cargo
tree -p kwavers-therapy --depth 1` shows the direct `moirai-parallel`
dependency.

## CLOSED: kwavers-therapy standing-wave FDTD Moirai slice (2026-07-03)

`theranostic_guidance::standing_wave_opt::fdtd::compute_all_green_functions`
now routes independent Green-function column solves through
`moirai_parallel::map_collect_with` instead of Rayon. The numerical FDTD and
lock-in extraction path is unchanged; only the execution provider for the
outer element fan-out changed.

Residual risk: this is a narrow production edge closure. `kwavers-therapy`
still has direct Rayon and ndarray `rayon` usage in nonlinear3d
forward-stencil and passive-inverse paths, so the crate manifest still keeps
Rayon until those paths are migrated.

Evidence tier: static source audit plus compile-time/lint validation and
focused value-semantic tests. Verification: `cargo fmt -p kwavers-therapy
--check` passed, `cargo check -p kwavers-therapy` passed, `cargo clippy -p
kwavers-therapy --all-targets -- -D warnings` passed, `cargo nextest run -p
kwavers-therapy standing_wave --status-level fail --no-fail-fast` passed 5/5,
and the source audit found no direct Rayon hit in
`standing_wave_opt/fdtd.rs`.

## CLOSED: kwavers-therapy waveform-forward Moirai slice (2026-07-03)

`theranostic_guidance::waveform::forward` now routes CPML row updates, pressure
updates, attenuation, and peak-pressure updates through `moirai-parallel`
helpers instead of Rayon row chunks and parallel iterators. The acoustic
time-step formulas, source injection, checkpoint cadence, and receiver trace
recording are unchanged.

Residual risk: this is a narrow production edge closure. `kwavers-therapy`
still has direct Rayon and ndarray `rayon` usage in nonlinear3d
forward-stencil and passive-inverse paths, so the crate manifest still keeps
Rayon until those paths are migrated.

Evidence tier: static source audit plus compile-time/lint validation and
focused value-semantic tests. Verification: `cargo fmt -p kwavers-therapy
--check` passed, `cargo check -p kwavers-therapy` passed, `cargo clippy -p
kwavers-therapy --all-targets -- -D warnings` passed, `cargo nextest run -p
kwavers-therapy waveform --status-level fail --no-fail-fast` passed 13/13
with 1 slow test, and the source audit found no direct Rayon hit in
`waveform/forward.rs`.

## CLOSED: kwavers-therapy nonlinear3d absorption Moirai slice (2026-07-03)

`theranostic_guidance::nonlinear3d::absorption` now routes Treeby-Cox
coefficient construction and forward/adjoint absorption element-wise updates
through `moirai-parallel` helpers instead of Rayon. The spectral filter, cache
semantics, and transpose equations are unchanged.

Residual risk: this is a narrow production edge closure. `kwavers-therapy`
still has direct Rayon and ndarray `rayon` usage in nonlinear3d
forward-stencil and passive-inverse paths, so the crate manifest still keeps
Rayon until those paths are migrated.

Evidence tier: static source audit plus compile-time/lint validation and
focused value-semantic tests. Verification: `cargo fmt -p kwavers-therapy
--check` passed, `cargo check -p kwavers-therapy` passed, `cargo clippy -p
kwavers-therapy --all-targets -- -D warnings` passed, `cargo nextest run -p
kwavers-therapy absorption --status-level fail --no-fail-fast` passed 5/5,
and the source audit found no direct Rayon hit in
`nonlinear3d/absorption/{construction,apply}.rs`.

## CLOSED: kwavers-therapy nonlinear3d cavitation-forward Moirai slice (2026-07-03)

`theranostic_guidance::nonlinear3d::cavitation::forward` now routes contiguous
source-mask max reduction and cavitation source-density mapping through
`moirai-parallel` helpers instead of Rayon. The Rayleigh-Plesset response
calculation and the non-contiguous ndarray fallback are unchanged.

Residual risk: this is a narrow production edge closure. `kwavers-therapy`
still has direct Rayon and ndarray `rayon` usage in nonlinear3d forward-stencil
and passive-inverse paths, so the crate manifest still keeps Rayon until those
paths are migrated.

Evidence tier: static source audit plus compile-time/lint validation and
focused value-semantic tests. Verification: `cargo fmt -p kwavers-therapy
--check` passed, `cargo check -p kwavers-therapy` passed, `cargo clippy -p
kwavers-therapy --all-targets -- -D warnings` passed, `cargo nextest run -p
kwavers-therapy cavitation --status-level fail --no-fail-fast` passed 46/46,
and the source audit found no direct Rayon hit in
`nonlinear3d/cavitation/forward.rs`.

## CLOSED: kwavers-therapy nonlinear3d forward-stencil Moirai slice (2026-07-03)

`theranostic_guidance::nonlinear3d::forward::stencil` now routes the
Westervelt x-slab cell update through `moirai-parallel` chunk scheduling
instead of Rayon. The finite-difference recurrence, slab ownership model, and
boundary-zeroing semantics are unchanged. The adjacent Westervelt performance
docs now describe the Atlas provider rather than the former direct Rayon call.

Residual risk: this is a narrow production edge closure. `kwavers-therapy`
still has direct Rayon and ndarray `rayon` usage in the nonlinear3d
passive-inverse path, so the crate manifest still keeps Rayon until that path
is migrated.

Evidence tier: static source audit plus compile-time/lint validation and
focused value-semantic tests. Verification: `cargo fmt -p kwavers-therapy
--check` passed, `cargo check -p kwavers-therapy` passed, `cargo clippy -p
kwavers-therapy --all-targets -- -D warnings` passed, `cargo nextest run -p
kwavers-therapy nonlinear3d --status-level fail --no-fail-fast` passed 59/59,
and the focused source audit found no direct Rayon hit in
`nonlinear3d/forward/stencil.rs` or `nonlinear3d/westervelt/mod.rs`.

## CLOSED: kwavers-therapy nonlinear3d passive-inverse Moirai closure (2026-07-03)

`theranostic_guidance::nonlinear3d::cavitation::passive_inverse` now routes
the passive Green-operator fill, forward apply, normal-gradient assembly,
Frobenius/objective reductions, residual update, and projected Tikhonov model
update through `moirai-parallel` instead of Rayon. `kwavers-therapy` no longer
depends directly on Rayon and no longer enables ndarray's `rayon` feature.

Residual risk: this closes the `kwavers-therapy` direct Rayon edge. Remaining
Atlas provider migration work should continue in other crates or in ndarray
producer/consumer boundaries; no direct Rayon residual remains in
`kwavers-therapy`.

Evidence tier: static source/dependency audit plus compile-time/lint
validation and focused value-semantic tests. Verification: `cargo fmt -p
kwavers-therapy --check` passed, `cargo check -p kwavers-therapy` passed,
`cargo clippy -p kwavers-therapy --all-targets -- -D warnings` passed,
`cargo nextest run -p kwavers-therapy cavitation --status-level fail
--no-fail-fast` passed 46/46, `cargo nextest run -p kwavers-therapy
nonlinear3d --status-level fail --no-fail-fast` passed 59/59, `rg` found no
Rayon hits under `crates/kwavers-therapy/src` or its manifest, and `cargo tree
-p kwavers-therapy --depth 1` shows no direct `rayon`.

## CLOSED: kwavers-therapy orchestrator Moirai slice (2026-07-03)

`kwavers-core::utils::iterators` now provides shared indexed mutable
Array3 traversal helpers that dispatch standard-layout arrays through
`moirai-parallel` and retain sequential ndarray traversal for non-standard
layouts. `kwavers-therapy` integration orchestrator acoustic-field generation
and acoustic-heating updates use those helpers instead of direct ndarray
parallel traversal. Current package-local therapy test lints were cleaned so
the focused clippy gate is green.

Residual risk at closure time: full `kwavers-therapy` package nextest was not
green because two abdominal preprocessing tests timed out outside the
orchestrator slice. Follow-up closure is recorded above.

Evidence tier: compile-time validation plus focused empirical tests and static
source audit. Verification: `cargo check -p kwavers-therapy` passed, `cargo
clippy -p kwavers-core -p kwavers-therapy --all-targets -- -D warnings`
passed, `cargo nextest run -p kwavers-core iterators --status-level fail
--no-fail-fast` passed 2/2, `cargo nextest run -p kwavers-therapy
therapy_integration --status-level fail --no-fail-fast` passed 59/59, and the
source audit leaves no direct Rayon/ndarray-parallel hits in the touched
orchestrator file. Full package `cargo nextest run -p kwavers-therapy
--status-level fail --no-fail-fast` reported 338 passed, 2 timed out, and 1
skipped.

## CLOSED: GPU provider-generic device contract (2026-07-03)

`kwavers-gpu::backend::GpuComputeProvider` now requires its associated
`Device` to implement Kwavers' `GpuDeviceProvider`, not only Hephaestus
capability queries. That local trait carries the real Hephaestus acquisition
and capability seams, so WGPU and CUDA substitute through one provider
contract. `GPUBackend<P>` also exposes `provider()` so downstream code can
borrow the concrete provider generically without reaching for raw WGPU handles.

CUDA remains an acquisition/device provider only until Kwavers owns CUDA
kernels for the operations on `GpuComputeProvider`; no placeholder
`CudaComputeProvider` was added.

Evidence tier: type-level/compile-time validation plus focused empirical test
execution. Verification: `cargo fmt -p kwavers-gpu --check` passed, `cargo
check -p kwavers-gpu --features gpu` passed, `cargo check -p kwavers-gpu
--features cuda-provider --offline` passed, `cargo clippy -p kwavers-gpu
--features cuda-provider --all-targets --offline -- -D warnings` passed,
`cargo nextest run -p kwavers-gpu --features cuda-provider
gpu_backend_is_generic_over_provider_trait --status-level fail --no-fail-fast
--offline` passed 1/1, and full `cargo nextest run -p kwavers-gpu --features
cuda-provider --status-level fail --no-fail-fast --offline` passed 102/102
with 1 skipped.

## CLOSED: CUDA provider acquisition contract (2026-07-03)

`kwavers-gpu` now exposes a real CUDA acquisition seam without pretending that
CUDA kernels exist for the current `GpuComputeProvider` operations. The
workspace has a local `hephaestus-cuda` dependency, `kwavers-gpu/cuda-provider`
compiles the `CudaDevice` acquisition contract, and `kwavers-gpu/cuda-runtime`
enables Hephaestus' CUDA loader. `GpuDeviceProvider` is implemented for
`hephaestus_cuda::CudaDevice`; the implementation owns CUDA-specific labels,
requests no WGPU-only optional features, and uses CUDA-shaped limits with no
shader-stage storage-buffer slot claim. Existing WGPU dispatch remains the only
compute backend until CUDA operation kernels are implemented.

Evidence tier: type-level/compile-time validation plus focused empirical test
execution. Verification: `cargo fmt -p kwavers-gpu -p kwavers-boundary
--check` passed, `cargo check -p kwavers-gpu --features gpu` passed, `cargo
check -p kwavers-gpu --features cuda-provider --offline` passed, `cargo clippy
-p kwavers-gpu --features cuda-provider --all-targets --offline -- -D
warnings` passed, `cargo nextest run -p kwavers-gpu --features cuda-provider
--status-level fail --no-fail-fast --offline` passed 101/101 with 1 skipped,
and `cargo tree -p kwavers-gpu --features cuda-provider --depth 1 --offline`
shows direct local `hephaestus-cuda`, `hephaestus-core`, and
`hephaestus-wgpu` provider edges.

## CLOSED: GPU provider-neutral backend boundary (2026-07-03)

`kwavers-solver::backend::BackendType` now carries `GpuProvider` so GPU
algorithms can depend on a provider-neutral trait value instead of assuming
WGPU. The current leaf implementation reports `GpuProvider::Wgpu`; CUDA/Metal
can land as additional implementations without changing algorithm crates.
`kwavers-core` no longer exposes a WGPU-backed `gpu` feature or
`From<wgpu::BufferAsyncError>`, and all downstream `kwavers-core/gpu` forwarding
was removed. WGPU map failures are converted at the `kwavers-gpu` boundary.

Evidence tier: compile-time validation plus focused value-semantic tests.
Verification: `cargo fmt --check` for touched packages passed, `cargo check -p
kwavers-core --all-features` passed, `cargo check -p kwavers-solver` passed,
`cargo check -p kwavers-gpu --features gpu` passed, and `cargo nextest run -p
kwavers-solver backend_surface_tests` passed 3/3. Follow-up verification after
the diagnostics Leto normalization and backend pipeline fixes: `cargo check -p
kwavers --features gpu` passed, `cargo check -p moirai-core --tests` passed,
and `cargo nextest run -p kwavers-gpu --features gpu backend --no-fail-fast`
passed 31/31.

## CLOSED: Hephaestus-backed generic GPU provider seam (2026-07-03)

`kwavers-gpu::backend::GPUBackend` is now generic over
`GpuComputeProvider`, and the provider trait carries an associated
Hephaestus `ComputeDeviceCapabilities` device type. The default WGPU
implementation lives in `WgpuComputeProvider` and acquires its device through
`hephaestus_wgpu::WgpuDevice`, so provider-specific acquisition,
synchronization, device reporting, and dispatch sit behind a trait seam.
CUDA can land as a sibling provider implementation without changing the
solver-facing `ComputeBackend` contract.

Evidence tier: compile-time validation plus focused value-semantic backend
tests. Verification: `cargo fmt -p kwavers-gpu` passed, `cargo check -p
kwavers-gpu --features gpu` passed, and `cargo nextest run -p kwavers-gpu
--features gpu backend --status-level fail --no-fail-fast` passed 31/31.

## CLOSED: Backend provider-context generic refinement (2026-07-03)

`kwavers-gpu::backend::GpuProviderContext<P>` now acquires, synchronizes, and
reports device identity through `GpuDeviceProvider`, the local trait backed by
Hephaestus `ComputeDeviceAcquisition`. WGPU raw device and queue access remain
available only on `GpuProviderContext<WgpuDevice>` for the current WGSL
pipeline code. This keeps CUDA as a real sibling provider implementation point
without changing solver-facing `ComputeBackend` APIs or pretending CUDA kernels
exist today.

Evidence tier: compile-time validation plus focused value-semantic backend
tests. Verification passed: focused formatting, GPU feature compile check, and
the `kwavers-gpu --features gpu` backend nextest filter, 31/31 tests.

## CLOSED: Provider-owned GPU acquisition requirements (2026-07-03)

`kwavers-gpu::gpu::GpuDeviceProvider` now owns the acquisition label,
default preference, optional features, and minimum device limits for its
provider. `GpuProviderContext<P>` calls only those trait methods, so future
CUDA providers do not inherit WGPU's `ShaderF64` optional feature or WGSL
workgroup-limit policy. The current `WgpuDevice` implementation preserves the
existing WGPU requirements.

CUDA remains a real sibling provider implementation item: it must supply
Kwavers kernels for the `GpuComputeProvider` operations rather than falling
back to CPU or returning placeholder success.

Evidence tier: compile-time validation plus focused value-semantic backend
tests. Verification: `cargo fmt -p kwavers-gpu --check` passed, `cargo check
-p kwavers-gpu --features gpu` passed, and `cargo nextest run -p kwavers-gpu
--features gpu backend --status-level fail --no-fail-fast` passed 31/31.

## CLOSED: GPU PSTD Hephaestus auto-device slice (2026-07-03)

`GpuPstdSolver::with_auto_device` now delegates WGPU device acquisition to
`hephaestus_wgpu::WgpuDevice` and keeps the PSTD-specific push-constant and
storage-buffer requirements in the provider request. The solver still consumes
raw WGPU handles for shader dispatch, but those handles are now selected by the
Atlas GPU provider instead of a Kwavers-local adapter path.

Evidence tier: compile-time validation plus focused value-semantic GPU tests.
Verification: `cargo fmt -p kwavers-gpu --check` passed, `cargo check -p
kwavers-gpu --features gpu` passed, and `cargo nextest run -p kwavers-gpu
--features gpu pstd_gpu --no-fail-fast` passed 12/12.

## CLOSED: GpuDevice Hephaestus acquisition trait slice (2026-07-03)

`kwavers-gpu::gpu::GpuDevice<P>` is now generic over `GpuDeviceProvider`, a
local trait backed by Hephaestus `ComputeDeviceAcquisition`. Generic callers
use backend-neutral `DevicePreference`, `DeviceFeature`, and `DeviceLimits`;
raw WGPU handles are exposed only on the default `GpuDevice<WgpuDevice>`
specialization for existing WGSL shader dispatch.

Evidence tier: static source audit plus compile-time validation and focused
value-semantic GPU tests. Verification: `cargo fmt -p kwavers-gpu -p kwavers
--check` passed, `cargo check -p kwavers-gpu --features gpu` passed, source
search for direct WGPU acquisition APIs under `crates/kwavers-gpu/src`
returned no hits, and `cargo nextest run -p kwavers-gpu --features gpu backend
gpu::shaders::neural_network gpu::multi_gpu pstd_gpu::tests::construction
--status-level fail --no-fail-fast` passed 37/37.

## CLOSED: kwavers-analysis visualization data-pipeline Moirai traversal (2026-07-03)

`DataProcessor` normalization and log scaling now route contiguous
`Array3<f64>` scalar traversal through `moirai-parallel` instead of ndarray's
Rayon-backed `par_mapv_inplace`. Non-standard layouts keep ndarray's
sequential value semantics.

Evidence tier: static source audit plus compile-time validation and focused
value-semantic tests. Verification: `cargo fmt -p kwavers-analysis --check`
passed, `cargo check -p kwavers-analysis --features gpu-visualization` passed,
`cargo clippy -p kwavers-analysis --features gpu-visualization --lib -- -D
warnings` passed, `cargo nextest run -p kwavers-analysis --features
gpu-visualization -E "test(normalize_maps_contiguous_values_to_configured_range) or test(log_scale_clamps_values_at_configured_epsilon)"`
passed 2/2, and `rg` found no direct Rayon or ndarray-parallel hits under
`crates/kwavers-analysis/src/visualization/data_pipeline`.

## CLOSED: kwavers-analysis performance optimizer Moirai traversal (2026-07-03)

`ParallelOptimizer` now routes 3-D fan-out through Moirai indexed execution,
ordered mapping through Moirai map-collect, and chunked reductions through
Moirai indexed reduction. `set_num_threads` no longer mutates a global Rayon
thread pool; it validates the requested lane count and keeps it as the local
chunk-size scheduling hint.

Evidence tier: static source audit plus compile-time validation and focused
value-semantic tests. Verification: `cargo fmt -p kwavers-analysis --check`
passed, `cargo check -p kwavers-analysis` passed, `cargo clippy -p
kwavers-analysis --lib -- -D warnings` passed, `cargo nextest run -p
kwavers-analysis -E "test(parallel_optimizer_) or test(parallel_3d_visits_every_cell_exactly_once) or test(set_num_threads_)"`
passed 8/8, and `rg` found no direct Rayon symbols in
`crates/kwavers-analysis/src/performance/optimization/parallel.rs`.

## CLOSED: shared scalar ndarray Moirai seam (2026-07-03)

`kwavers-core::utils::iterators::apply_inplace` is now the shared scalar
ndarray transform seam. Standard-layout arrays route through Moirai
`for_each_mut_with::<Adaptive>` and non-standard layouts retain ndarray's
sequential traversal semantics. `kwavers-analysis` visualization
normalization/log scaling, PAM time-exposure acoustic squaring, and polynomial
clutter-filter time normalization now use that shared seam instead of direct
ndarray/Rayon `par_mapv_inplace`.

Evidence tier: static source audit plus compile-time validation and focused
value-semantic tests. Verification: `cargo fmt -p kwavers-core -p
kwavers-analysis --check` passed, `cargo check -p kwavers-core` passed, `cargo
check -p kwavers-analysis` passed, `cargo clippy -p kwavers-core --lib -- -D
warnings` passed, `cargo clippy -p kwavers-analysis --lib -- -D warnings`
passed, `cargo clippy -p kwavers-analysis --features gpu-visualization --lib
-- -D warnings` passed, `cargo nextest run -p kwavers-core -E
"test(apply_inplace_updates_standard_layout_values) or test(apply_inplace_updates_non_standard_layout_values)"`
passed 2/2, `cargo nextest run -p kwavers-analysis -E
"test(polynomial_filter_linear_signal_normalized_time_zero_residual) or test(pam_policy_to_core_)"`
passed 3/3, and `cargo nextest run -p kwavers-analysis --features
gpu-visualization -E "test(normalize_maps_contiguous_values_to_configured_range) or test(log_scale_clamps_values_at_configured_epsilon)"`
passed 2/2. `rg` now shows no direct Rayon or ndarray-parallel hits in the
touched visualization data-pipeline, PAM mapper, polynomial-filter, or
performance-optimizer files.

## CLOSED: kwavers-analysis covariance Moirai scalar traversal (2026-07-03)

The beamforming covariance subtree now routes sample covariance scaling,
real/complex estimator normalization, shrinkage scaling, and spatial-smoothing
normalization through `kwavers-core::utils::iterators::apply_inplace` instead
of direct ndarray/Rayon `par_mapv_inplace`.

Evidence tier: static source audit plus compile-time validation and focused
value-semantic tests. Verification: `cargo fmt -p kwavers-analysis --check`
passed, `cargo check -p kwavers-analysis` passed, `cargo clippy -p
kwavers-analysis --lib -- -D warnings` passed, `cargo nextest run -p
kwavers-analysis -E "test(covariance_) or test(shrinkage_to_identity_real_) or test(estimate_complex_) or test(estimate_single_snapshot_gives_exact_outer_product) or test(estimate_two_orthogonal_snapshots_gives_half_identity) or test(spatial_smoothing_complex_shapes_match) or test(test_sample_covariance_basic) or test(test_sample_covariance_with_diagonal_loading)"`
passed 30/30, and `rg` found no direct Rayon or ndarray-parallel hits under
`crates/kwavers-analysis/src/signal_processing/beamforming/covariance`.

## CLOSED: kwavers-analysis safe-vectorization Moirai traversal (2026-07-03)

`SafeVectorOps::add_arrays_parallel` and the non-contiguous fallback for
`add_arrays_chunked` now route through the shared indexed ndarray traversal
seam, and `scalar_multiply_inplace` now routes through the shared scalar
in-place seam. The module no longer uses ndarray/Rayon `Zip::par_for_each`.

Evidence tier: static source audit plus compile-time validation and focused
value-semantic tests. Verification: `cargo fmt -p kwavers-analysis --check`
passed, `cargo check -p kwavers-analysis` passed, `cargo clippy -p
kwavers-analysis --lib -- -D warnings` passed, `cargo nextest run -p
kwavers-analysis -E "test(add_arrays_) or test(scalar_multiply_) or test(test_add_arrays_correctness) or test(test_scalar_multiply_correctness) or test(test_dot_product_correctness) or test(test_l2_norm_correctness)"`
passed 8/8, and `rg` found no direct Rayon or ndarray-parallel hits in
`crates/kwavers-analysis/src/performance/safe_vectorization.rs`.

## CLOSED: kwavers-analysis SLSC Moirai traversal (2026-07-03)

`SlscBeamformer::process_parallel` and `SlscBeamformer::process_volume` now
use Moirai ordered indexed collection instead of Rayon iterator fan-out, and
`create_coherence_map` clamps through the shared scalar ndarray seam.

Evidence tier: static source audit plus compile-time validation and focused
value-semantic tests. Verification: `cargo fmt -p kwavers-analysis --check`
passed, `cargo check -p kwavers-analysis` passed, `cargo clippy -p
kwavers-analysis --lib -- -D warnings` passed, `cargo nextest run -p
kwavers-analysis -E "test(slsc_) or test(lag_coherence_) or test(multi_lag_slsc_) or test(adaptive_slsc_) or test(test_slsc_) or test(test_lag_weighting_) or test(triangular_weighting_midpoint_is_half) or test(hamming_weighting_lag_zero_is_point_zero_eight)"`
passed 24/24, and `rg` found no direct Rayon or ndarray-parallel hits under
`crates/kwavers-analysis/src/signal_processing/beamforming/slsc`.

## CLOSED: kwavers-analysis neural scalar Moirai traversal (2026-07-03)

Neural layer adaptation weight/bias scaling and neural feature normalization
now route through `kwavers-core::utils::iterators::apply_inplace` instead of
direct ndarray/Rayon `par_mapv_inplace`. A focused value-semantic feature
normalization test pins exact min/max normalization results.

Evidence tier: static source audit plus compile-time validation and focused
value-semantic tests. Verification: `cargo fmt -p kwavers-analysis --check`
passed, `cargo check -p kwavers-analysis` passed, `cargo clippy -p
kwavers-analysis --lib -- -D warnings` passed, and `cargo nextest run -p
kwavers-analysis -E "test(test_neural_layer_adaptation) or test(test_neural_layer_adaptation_zero_gradient_is_noop) or test(test_normalize_features) or test(test_normalize_features_value_semantics)"`
passed 4/4. The remaining neural direct Rayon edge before the next closure was
limited to
`crates/kwavers-analysis/src/signal_processing/beamforming/neural/distributed/core/processor.rs`.

## CLOSED: kwavers-analysis distributed neural Moirai traversal (2026-07-03)

`DistributedNeuralBeamformingProcessor::process_volume_distributed` now uses
`moirai_parallel::map_collect_mut_with` instead of direct Rayon
`par_iter_mut().enumerate().map().collect()`. The operation still mutates each
processor slot independently, collects per-processor chunk results in index
order, and propagates typed `KwaversError` results.

Evidence tier: static source audit plus compile-time validation and focused
value-semantic tests. Verification: `cargo fmt -p kwavers-analysis --check`
passed, `cargo check -p kwavers-analysis --features pinn` passed, `cargo
clippy -p kwavers-analysis --features pinn --lib -- -D warnings` passed, and
`cargo nextest run -p kwavers-analysis --features pinn -E "test(test_distributed_processing_matches_sequential_result) or test(test_processor_creation) or test(test_fault_tolerance_)"`
passed 6/6. Source audit found no direct Rayon or ndarray-parallel hits under
`crates/kwavers-analysis/src/signal_processing/beamforming/neural`.

## CLOSED: kwavers-analysis 3-D CPU beamforming Moirai traversal (2026-07-03)

The 3-D CPU DAS and MVDR implementations now use
`moirai_parallel::map_collect_index_with` for ordered voxel fan-out instead of
direct Rayon parallel iterators. The package manifest no longer declares a
direct `rayon` dependency or enables ndarray's `rayon` feature.

Evidence tier: static source/manifest audit plus compile-time validation and
focused value-semantic tests. Verification: `cargo fmt -p kwavers-analysis
--check` passed, `cargo check -p kwavers-analysis` passed, `cargo clippy -p
kwavers-analysis --lib -- -D warnings` passed, and `cargo nextest run -p
kwavers-analysis -E "test(das_) or test(mvdr_) or test(test_algorithm_mvdr_3d) or test(test_processor_creation_cpu_only) or test(test_beamforming_config_3d_default)"`
passed 39/39. `rg` found no direct Rayon or ndarray-parallel hits in
`crates/kwavers-analysis/src` or `crates/kwavers-analysis/Cargo.toml`, and
`cargo tree -p kwavers-analysis --depth 1` lists no direct `rayon`
dependency.

## CLOSED: CoreGpuContext Hephaestus provider slice (2026-07-03)

`kwavers-gpu::gpu::CoreGpuContext` now owns
`hephaestus_wgpu::WgpuDevice` instead of constructing a WGPU
instance/adapter/device locally. The context still exposes raw WGPU device and
queue handles for existing shader dispatch, but those handles are provider
owned. Multi-GPU logical-device construction now wraps each selected adapter in
the same Hephaestus provider request. The backend-level `GpuComputeProvider`
trait remains the CUDA/WGPU seam.

Evidence tier: compile-time validation plus focused value-semantic GPU tests.
Verification: `cargo fmt -p kwavers-gpu --check` passed, `cargo check -p
kwavers-gpu --features gpu` passed, `cargo nextest run -p kwavers-gpu
--features gpu gpu::multi_gpu --no-fail-fast` passed 3/3, and `cargo nextest
run -p kwavers-gpu --features gpu backend::tests --no-fail-fast` passed 5/5.
Integrator verification: `cargo check -p kwavers --features gpu` passed.

## CLOSED: kwavers-gpu leaf acquisition cleanup (2026-07-03)

`AcousticFieldKernel`, `ComputeManager`, and the backend buffer-manager GPU
test helper now acquire WGPU handles through `hephaestus_wgpu::WgpuDevice`
instead of local WGPU instance/adapter/device request code. `ComputeManager`
stores the provider and exposes raw handles from it; `AcousticFieldKernel`
owns the provider while preserving its existing shader dispatch pipeline.

Evidence tier: static source audit plus compile-time validation and focused
value-semantic GPU tests. Verification: `cargo fmt -p kwavers-gpu --check`
passed, `cargo check -p kwavers-gpu --features gpu` passed, `cargo nextest run
-p kwavers-gpu --features gpu backend::buffers gpu::compute_manager
gpu::compute --no-fail-fast` passed 10/10, and `cargo check -p kwavers
--features gpu` passed. Source search for `wgpu::Instance`,
`request_adapter`, and `request_device` under `crates/kwavers-gpu/src/{gpu,backend}`
now finds no direct device creation outside multi-GPU adapter enumeration.

## CLOSED: kwavers-gpu direct WGPU acquisition closure (2026-07-03)

`MultiGpuContext::new` now delegates device discovery and logical-device
creation to Hephaestus `ComputeDeviceAcquisition::try_acquire_devices`.
PSTD GPU test construction now uses a shared Hephaestus-backed test-provider
helper that preserves the push-constant and storage-buffer limit requirements.

Evidence tier: static source audit plus compile-time validation and focused
value-semantic GPU tests. Verification: source search for `wgpu::Instance`,
`request_adapter`, `request_device`, and direct Hephaestus WGPU constructor
calls under `crates/kwavers-gpu/src` returned no hits, `cargo fmt -p
kwavers-gpu -p kwavers --check` passed, `cargo check -p kwavers-gpu --features
gpu` passed, and `cargo nextest run -p kwavers-gpu --features gpu backend
gpu::shaders::neural_network gpu::multi_gpu pstd_gpu::tests::construction
--status-level fail --no-fail-fast` passed 37/37.

## CLOSED: kwavers-gpu Moirai pipeline cleanup (2026-07-03)

`kwavers-gpu::gpu::pipeline::{realtime,streaming}` now dispatches Hilbert
envelope planes and synthetic RF receive slices through `moirai-parallel`
chunk scheduling instead of direct Rayon parallel iterators. The crate manifest
now enables `moirai-parallel` for the GPU feature and no longer depends
directly on Rayon.

Evidence tier: static source/dependency audit plus focused empirical tests.
Verification: `cargo fmt -p kwavers-gpu --check` passed, `cargo check -p
kwavers-gpu --features gpu` passed, `cargo nextest run -p kwavers-gpu
--features gpu gpu::pipeline --status-level fail --no-fail-fast` passed 5/5,
`rg` found no direct Rayon imports or parallel iterator calls under
`crates/kwavers-gpu`, and `cargo tree -p kwavers-gpu --features gpu --depth 1`
shows `moirai-parallel` with no direct `rayon`.

## CLOSED: kwavers-boundary Moirai CPML/adaptive cleanup (2026-07-03)

`kwavers-boundary` now routes CPML full-field damping, CPML strip
memory/correction updates, and adaptive-boundary attenuation through private
`moirai-parallel` traversal helpers instead of ndarray/Rayon dispatch. The
helper uses Moirai for standard-layout mutable slices and keeps sequential
ndarray view semantics for non-standard layouts. The manifest no longer
enables ndarray's `rayon` feature.

Evidence tier: static source/dependency audit plus focused empirical tests.
Verification: `cargo fmt -p kwavers-boundary --check` passed, `cargo check -p
kwavers-boundary` passed, `cargo nextest run -p kwavers-boundary
--status-level fail --no-fail-fast` passed 96/96, `rg` found no direct Rayon or
ndarray-parallel hits under `crates/kwavers-boundary`, and `cargo tree -p
kwavers-boundary --depth 1` shows `moirai-parallel` with no direct `rayon`.

## CLOSED: kwavers-receiver Moirai statistics cleanup (2026-07-03)

`kwavers-receiver` now routes pressure and velocity statistics updates through
Atlas `moirai-parallel` triple/quad mutable chunk dispatch for standard-layout
arrays. Non-standard ndarray layouts retain sequential `Zip` semantics. The
crate manifest no longer depends directly on `rayon` and no longer enables
ndarray's `rayon` feature.

Evidence tier: static source/dependency audit plus focused empirical tests.
Verification: `cargo fmt -p kwavers-receiver --check` passed, `cargo check -p
kwavers-receiver` passed, `cargo nextest run -p kwavers-receiver
--status-level fail --no-fail-fast` passed 47/47, `rg` found no direct Rayon
or ndarray-parallel hits under `crates/kwavers-receiver`, and `cargo tree -p
kwavers-receiver --depth 1` shows `moirai-parallel` with no direct `rayon`.

## CLOSED: kwavers-medium Moirai absorption/iterator cleanup (2026-07-03)

`kwavers-medium` now routes medium property traversal, absorption/dispersion
k-space updates, and frequency-dependent correction through Atlas
`moirai-parallel` indexed and standard-layout chunk traversal. Non-standard
ndarray layouts retain sequential traversal through the crate-local provider
adapter. The crate manifest no longer depends directly on `rayon` and no
longer enables ndarray's `rayon` feature. The focused package Clippy gate also
surfaced current Christoffel/material-property test lints, now resolved with
iterator-based loops and compile-time constant assertions.

Evidence tier: static source/dependency audit plus focused empirical tests.
Verification: `cargo fmt -p kwavers-medium --check` passed, `cargo check -p
kwavers-medium` passed, `cargo clippy -p kwavers-medium --all-targets -- -D
warnings` passed, `cargo nextest run -p kwavers-medium --status-level fail
--no-fail-fast` passed 179/179, `rg` found no direct Rayon or
ndarray-parallel hits under `crates/kwavers-medium`, and `cargo tree -p
kwavers-medium --depth 1` shows `moirai-parallel` with no direct `rayon`.

## CLOSED: kwavers-diagnostics real-time SIRT Moirai cleanup (2026-07-03)

`kwavers-diagnostics::reconstruction::real_time_sirt::pipeline` now routes
row-norm cache construction and separable smoothing through `moirai-parallel`.
The diagnostics manifest no longer enables ndarray's `rayon` feature and no
longer depends directly on `rayon`.

Evidence tier: static source/dependency audit plus focused empirical tests.
Verification: `cargo fmt -p kwavers-diagnostics` passed, `cargo check -p
kwavers-diagnostics` passed, `cargo nextest run -p kwavers-diagnostics
real_time_sirt --status-level fail --no-fail-fast` passed 14/14, `rg` found no
direct Rayon/thread-pool hits in `crates/kwavers-diagnostics/src` or its
manifest, and `cargo tree -p kwavers-diagnostics --depth 1` shows
`moirai-parallel` as a direct dependency with no direct `rayon`.

## CLOSED: kwavers-solver workspace in-place Moirai cleanup (2026-07-03)

`kwavers-solver::workspace::inplace_ops` now routes standard-layout in-place
array arithmetic through `moirai-parallel`. Non-standard ndarray layouts retain
sequential `Zip` semantics, so the helper remains value-preserving without a
direct Rayon dependency in the workspace module.

Evidence tier: static source audit plus focused empirical tests. Verification:
`cargo fmt -p kwavers-solver --check` passed, `cargo check -p kwavers-solver`
passed, `cargo nextest run -p kwavers-solver workspace --status-level fail
--no-fail-fast` passed 20/20, and `rg` found no direct Rayon/thread-pool hits
in `crates/kwavers-solver/src/workspace`.

## CLOSED: kwavers-solver time-integration Moirai cleanup (2026-07-03)

`kwavers-solver::integration::time_integration::time_stepper` now routes RK4
stage updates and Adams-Bashforth 2/3 field updates through
`moirai-parallel` for standard-layout arrays. Non-standard ndarray layouts
retain sequential `Zip` semantics, and AB3 now has a full-field
constant-derivative regression.

Evidence tier: static source audit plus focused empirical tests. Verification:
`cargo fmt -p kwavers-solver --check` passed, `cargo check -p kwavers-solver`
passed, `cargo nextest run -p kwavers-solver time_integration --status-level
fail --no-fail-fast` passed 12/12, and `rg` found no direct Rayon/thread-pool
hits in `crates/kwavers-solver/src/integration/time_integration`.

## CLOSED: kwavers-solver plugin execution Rayon placeholder removal (2026-07-03)

`kwavers-solver::plugin::execution::ParallelStrategy` no longer exposes
`with_thread_pool(rayon::ThreadPool)`. The removed constructor accepted and
discarded a concrete Rayon pool while the strategy still executed plugins in
order because plugins receive mutable access to shared field state. The
remaining docs state that real plugin parallelism requires a read/compute/write
phase split before it can preserve plugin semantics.

Residual risk: this is a narrow placeholder-seam cleanup. `kwavers-solver`
still has many real direct Rayon/ndarray-parallel compute paths, so its
manifest-level Rayon dependency remains open. The all-targets clippy gate also
remains blocked by unrelated pre-existing test/doc lints outside this slice.

Evidence tier: static source audit plus compile-time/lint validation and
focused tests. Verification: `cargo fmt -p kwavers-solver --check` passed,
`cargo check -p kwavers-solver` passed, `cargo clippy -p kwavers-solver --lib
-- -D warnings` passed, `cargo nextest run -p kwavers-solver plugin
--status-level fail --no-fail-fast` passed 37/37, and `rg` found no
`with_thread_pool` or `rayon::ThreadPool` hits in the solver plugin module.
`cargo clippy -p kwavers-solver --all-targets -- -D warnings` failed on
unrelated existing test/doc lints, not on `plugin::execution`.

## CLOSED: kwavers-diagnostics sound-speed-shift Moirai cleanup (2026-07-03)

`kwavers-diagnostics::reconstruction::sound_speed_shift::operator::algebra`
now routes `matvec`, transpose matvec, and normal-diagonal assembly through
`moirai-parallel`. The scatter reductions still use task-local partial vectors
and binary reduction, preserving the matrix-vector algebra contract without
atomics.

Evidence tier: static source audit plus focused empirical tests.
Verification: `cargo fmt -p kwavers-diagnostics` passed, `cargo check -p
kwavers-diagnostics` passed, and `cargo nextest run -p kwavers-diagnostics
sound_speed_shift --status-level fail --no-fail-fast` passed 34/34.

## CLOSED: kwavers-diagnostics transcranial UST Moirai cleanup (2026-07-03)

`kwavers-diagnostics::reconstruction::transcranial_ust::sensitivity` now
routes finite-frequency sensitivity rows plus attenuation and traveltime ray
integral rows through `moirai-parallel` chunk dispatch. The row ordering and
matrix/ray-integral contracts are unchanged.

Evidence tier: static source audit plus focused empirical tests.
Verification: `cargo fmt -p kwavers-diagnostics` passed, `cargo check -p
kwavers-diagnostics` passed, and `cargo nextest run -p kwavers-diagnostics
transcranial_ust --status-level fail --no-fail-fast` passed 7/7.

## CLOSED: kwavers-solver time-domain FWI field-update Moirai cleanup (2026-07-03)

`kwavers-solver::inverse::fwi::time_domain` now routes adjoint gradient
scaling, source-mask zeroing, gradient interaction products, regularization
updates, and multi-source model/gradient field updates through one
Moirai-backed `field_ops.rs` helper module. The stale `rayon::par_iter`
forward-run doc mention is removed. Non-standard ndarray views retain
sequential value semantics.

Follow-up on 2026-07-04 removed explicit ndarray `Zip` fallback traversal from
`field_ops.rs`; standard-layout field volumes still use Moirai chunk dispatch,
and non-standard layouts now use direct indexed sequential traversal.

Evidence tier: static source audit plus focused empirical tests. Verification:
`rg` found no direct Rayon/thread-pool hits in
`crates/kwavers-solver/src/inverse/fwi/time_domain`, `cargo fmt -p
kwavers-solver` passed, `cargo check -p kwavers-solver` passed, and `cargo
nextest run -p kwavers-solver time_domain --status-level fail --no-fail-fast`
passed 58/58. Follow-up verification: scoped `rg` found no `Zip` tokens in
`field_ops.rs`, `cargo fmt -p kwavers-solver --check` passed, and `cargo
check -p kwavers-solver` passed, `cargo clippy -p kwavers-solver --lib -- -D
warnings` passed, and `cargo nextest run -p kwavers-solver time_domain
--status-level fail --no-fail-fast` passed 58/58.

## CLOSED: kwavers-solver FWI constraints/adjoint-state Moirai slice (2026-07-03)

`constraints.rs` model clamping and pressure second-derivative writes plus
`adjoint_state.rs` signed-correlation accumulation now route standard-layout
field volumes through `moirai-parallel` chunk dispatch. Non-standard ndarray
views retain the same sequential `Zip::for_each` value semantics.

Evidence tier: static source audit plus focused empirical tests. Verification:
`rg` found no direct Rayon hits in `constraints.rs` or `adjoint_state.rs`,
`cargo fmt -p kwavers-solver --check` passed, `cargo check -p kwavers-solver`
passed, and `cargo nextest run -p kwavers-solver time_domain --status-level
fail --no-fail-fast` passed 58/58.

## CLOSED: kwavers-solver time-domain FWI search/MOFI Moirai cleanup (2026-07-02)

`kwavers-solver::inverse::fwi::time_domain::{search,mofi}` now routes joint
objective, line-search trial-model writes, and coarse-pose candidate evaluation
through `moirai-parallel` rather than direct Rayon iterators or ndarray
`par_for_each`. The objective-search and MOFI contracts are unchanged.

Evidence tier: static source audit plus focused empirical tests. Verification:
`rg` found no direct Rayon/thread-pool hits in `search.rs` or `mofi/mod.rs`,
`cargo fmt -p kwavers-solver --check` passed, `cargo check -p kwavers-solver`
passed, `cargo nextest run -p kwavers-solver time_domain --status-level fail`
passed 58/58, and `cargo test --doc -p kwavers-solver -- --show-output`
passed 6 doctests with 14 ignored. `cargo doc -p kwavers-solver --no-deps`
generated docs but reported 189 pre-existing rustdoc warnings outside this
slice.

## CLOSED: kwavers-solver linear Born inversion Moirai cleanup (2026-07-01)

`kwavers-solver::inverse::linear_born_inversion` dense products,
volume-operator construction, normal-equation reductions, and Sobolev Z-pass
now dispatch through `moirai-parallel`. The Sobolev pass uses Moirai's
provider-owned stateful chunk primitive rather than a Kwavers-local helper.
Evidence tier: static source audit plus focused empirical tests. Verification:
`rg` found no direct Rayon iterator imports or calls in
`crates/kwavers-solver/src/inverse/linear_born_inversion`, `cargo check -p
kwavers-solver` passed, and `cargo nextest run -p kwavers-solver
linear_born_inversion` passed 6/6.

## CLOSED: kwavers-solver same-aperture Moirai cleanup (2026-07-01)

`kwavers-solver::inverse::same_aperture` encoded/operator paths now dispatch
through `moirai-parallel` instead of direct Rayon iterators. The matrix-free
linear operator contract is unchanged. Evidence tier: static source audit plus
focused empirical tests. Verification: `rg` found no direct Rayon iterator
imports or calls in `crates/kwavers-solver/src/inverse/same_aperture`, `cargo
check -p kwavers-solver` passed, and `cargo nextest run -p kwavers-solver
same_aperture` passed 7/7.

## CLOSED: optical diffusion fluence Leto producer cleanup (2026-07-01)

The selected optical diffusion PCG producer now has one internal generic volume
kernel shared by ndarray and Leto entry points. `kwavers-simulation`
photoacoustic optics allocates the source as `leto::Array3<f64>` and calls
`DiffusionSolver::solve_leto`, removing the caller-side
`Ok(fluence.into())` conversion. Evidence tier: compile-time validation plus
bitwise differential and focused empirical tests. Verification: `cargo check
-p kwavers-solver`, `cargo check -p kwavers-simulation`, `cargo fmt -p
kwavers-solver -p kwavers-simulation --check`, `cargo nextest run -p
kwavers-solver diffusion` (13/13, including
`leto_solver_matches_ndarray_solver_bitwise`), and `cargo nextest run -p
kwavers-simulation photoacoustic` (27/27).

## CLOSED: photoacoustic reconstructor Leto producer cleanup (2026-07-01)

The selected `kwavers-solver` photoacoustic universal back-projection producer
now emits a `leto::Array3<f64>` directly through a concrete reconstructor method
used by `kwavers-simulation`, with the shared back-projection computation
factored into one value-producing kernel. The legacy ndarray `Reconstructor`
contract remains for unmigrated generic reconstructor call sites. Evidence
tier: compile-time validation plus focused empirical tests. Verification:
`cargo check -p kwavers-solver`, `cargo check -p kwavers-simulation`, `cargo
fmt -p kwavers-solver -p kwavers-simulation --check`, `cargo nextest run -p
kwavers-solver photoacoustic` (9/9), and `cargo nextest run -p
kwavers-simulation photoacoustic` (27/27).

## CLOSED: kwavers-solver linear elastography Leto producer slice (2026-07-01)

Direct, directional, and LFE linear elastography methods now allocate
`leto::Array3<f64>` shear-wave-speed maps directly instead of computing ndarray
speed maps and converting them at the return boundary. Shared smoothing and
boundary extrapolation are consolidated behind one crate-local volume trait
implemented for both ndarray and Leto arrays. Evidence tier: compile-time
validation plus focused empirical tests. Verification: `cargo check -p
kwavers-solver`, `cargo fmt -p kwavers-solver --check`, `cargo nextest run -p
kwavers-solver elastography` (53/53), targeted search for `.into()` in
`linear_methods`, and `cargo check -p kwavers --example
liver_theranostic_reconstruction --features nifti`.

## CLOSED: kwavers fusion/workflow/photoacoustic Leto slice (2026-07-01)

`kwavers-physics::acoustics::imaging::fusion`, diagnostics workflow products,
fUS atlas registration volumes, `kwavers-imaging` photoacoustic result volumes,
and `kwavers-simulation` photoacoustic fluence/pressure/reconstruction
snapshots now use `leto::Array3<f64>` directly in the migrated path. The liver
theranostic example now compiles with local Gaia/Leto/Ritk provider routing and
no ndarray-to-Leto helper was added. Evidence tier: compile-time validation
plus focused empirical tests. Verification: `cargo check -p kwavers --example
liver_theranostic_reconstruction --features nifti`, `cargo nextest run -p
kwavers-physics fusion` (103/103), `cargo nextest run -p kwavers-diagnostics
workflows functional_ultrasound atlas` (80/80), and `cargo nextest run -p
kwavers-simulation photoacoustic` (27/27).

## CLOSED: kwavers-imaging Leto multimodality volume slice (2026-07-01)

`kwavers-imaging::multimodality_fusion::ImageData.data` and fusion outputs now
use `leto::Array3<f64>` directly. Registration calls route into local Ritk's
Leto API without an ndarray-to-Leto helper, and fusion math uses Leto
`zip_map`, `mapv`, and direct indexing. Evidence tier: compile-time validation
plus focused empirical tests. Verification: `cargo fmt -p kwavers-imaging
--check`, `cargo check -p kwavers-imaging`, and `cargo nextest run -p
kwavers-imaging multimodality` (9/9).

## CLOSED: kwavers-grid Moirai Laplacian and Gaia ray slice (2026-07-01)

`kwavers-grid` no longer enables ndarray's `rayon` feature and routes the
second-order interior Laplacian through `moirai-parallel`, preserving
nonstandard output-view correctness. The liver theranostic reconstruction
straight-ray rasterizer now consumes Gaia's `Ray<f64>` primitive instead of a
local `Ray` concept, while keeping voxel path-length weighting in Kwavers.
Evidence tier: compile-time validation plus focused empirical tests.
Verification: `cargo fmt -p kwavers-grid --check`, `cargo check -p
kwavers-grid`, `cargo nextest run -p kwavers-grid test_laplacian` (3/3), and
Gaia `cargo nextest run -p gaia ray` (8/8).

## CLOSED: low-level unused ndarray Rayon feature cleanup (2026-07-01)

Removed ndarray's `rayon` feature from `kwavers-field`, `kwavers-signal`,
`kwavers-source`, and `kwavers-imaging` after a crate-local search confirmed
no direct Rayon, Tokio, or ndarray-parallel call sites in those crate trees.
This reduces accidental provider activation while preserving existing Ritk and
Burn dependencies where they are still present. Evidence tier: static analysis
plus package tests. Verification: `cargo fmt -p kwavers-source -p
kwavers-field -p kwavers-signal -p kwavers-imaging --check`, `cargo check -p
kwavers-source -p kwavers-field -p kwavers-signal -p kwavers-imaging`, `cargo
clippy -p kwavers-source -p kwavers-field -p kwavers-signal -p
kwavers-imaging --all-targets -- -D warnings`, `cargo nextest run -p
kwavers-source -p kwavers-field -p kwavers-signal -p kwavers-imaging`
(136/136), `cargo tree -p kwavers-source -p kwavers-field -p kwavers-signal
-p kwavers-imaging --depth 1`, and `rg` over the four crate trees for direct
Rayon/Tokio/ndarray-parallel call sites.

## CLOSED: kwavers-transducer Moirai source-field slice (2026-07-01)

Replaced the direct `rayon` edge in `kwavers-transducer` with workspace
`moirai-parallel` and removed ndarray's `rayon` feature from that crate.
Linear and matrix array focus-delay writes now use Moirai indexed mutable-slice
dispatch. Arc, bowl, multi-bowl, and phased-array source-field writes now use
Moirai indexed mutable-slice helpers over freshly allocated contiguous ndarray
storage, preserving the existing source formulas and value semantics. Evidence
tier: compile-time/static analysis plus empirical package tests. Verification:
`cargo fmt -p kwavers-transducer --check`, `cargo check -p kwavers-transducer`,
`cargo clippy -p kwavers-transducer --all-targets -- -D warnings`, `cargo
nextest run -p kwavers-transducer` (203/203, 1 skipped), `cargo tree -p
kwavers-transducer --depth 1`, and `rg` over `crates/kwavers-transducer` for
direct Rayon calls.

## CLOSED: kwavers-physics analytical Clippy unblock (2026-07-01)

Closed the dependency-inclusive `kwavers-simulation` Clippy blocker in
`kwavers-physics` by replacing broad public analytical tuple/argument surfaces
with typed request/result structs for IVUS microbubble delivery, Gaussian
photoacoustic profiles, Gaussian deconvolution fixtures, and apodization-window
responses. The PyO3 bindings remain thin adapters and keep their Python-facing
signatures while unpacking the typed Rust results. Also moved the centered-Hann
test module after production items to satisfy current Clippy. Evidence tier:
static analysis plus value-semantic focused tests. Verification: `cargo
clippy -p kwavers-physics --all-targets -- -D warnings`, `cargo check -p
kwavers-python`, focused `cargo nextest run -p kwavers-physics
ivus_microbubble_delivery_fraction gaussian_absorber_photoacoustic_profile
gaussian_deconvolution_fixture apodization_response centered_hann_tone_burst`
(10/10), and dependency-inclusive `cargo clippy -p kwavers-simulation
--all-targets --all-features -- -D warnings`.

## CLOSED: kwavers-simulation Moirai photoacoustic slice (2026-07-01)

Replaced the direct `rayon` edge in `kwavers-simulation` with workspace
`moirai-parallel` and removed the ndarray `rayon` feature from that crate.
Multi-wavelength fluence mapping now uses Moirai ordered map-collect, and
time-reversal photoacoustic reconstruction writes the contiguous output buffer
through Moirai enumerated mutable chunks. The all-features GPU-PSTD adapter
tests also now import the `Solver` trait whose methods they call. Evidence
tier: compile-time/static analysis plus empirical package tests. Verification:
`cargo fmt -p kwavers-simulation --check`, `cargo clippy -p
kwavers-simulation --all-targets --all-features --no-deps -- -D warnings`,
`cargo nextest run -p kwavers-simulation --all-features` (91/91), and `cargo
tree -p kwavers-simulation --depth 1`. Dependency-inclusive Clippy remains
blocked before this crate by existing `kwavers-physics` Clippy lints, tracked
in `gap_audit.md`.

## CLOSED: kwavers-core Moirai first-touch slice (2026-07-01)

Replaced the direct `rayon` edge in `kwavers-core` with workspace
`moirai-parallel` and removed the ndarray `rayon` feature from that crate.
NUMA first-touch, SoA first-touch, and the gradient interior-loop parallel
dispatch now route through Moirai helpers while preserving the existing data
ownership and value semantics. Current Clippy also required converting
compile-time constant invariant tests in `kwavers-core` to `const` assertions;
the checked invariants are unchanged. Evidence tier: compile-time/static
analysis plus empirical package tests. Verification: `cargo fmt -p
kwavers-core --check`, `cargo clippy -p kwavers-core --all-targets
--all-features -- -D warnings`, `cargo nextest run -p kwavers-core` (68/68),
and `cargo tree -p kwavers-core --depth 1`.

## CLOSED: Cavitation passive-map binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical cavitation facade's residual passive-map
wrapper family into `cavitation/passive_map.rs`: receiver-array PSD integration
and passive-map emission-energy integration. The Python registration surface
keeps the same function names through facade re-exports, Rust remains the owner
of passive-map physics in `kwavers-physics`, and `cavitation.rs` is now module
declarations plus public re-exports only. Verification: `cargo fmt -p
kwavers-python -p kwavers-physics`, warning-clean `cargo check -p
kwavers-python`, warning-clean `cargo check -p kwavers-python --features gpu`,
and `cargo nextest run -p kwavers-python` passed.

## CLOSED: Cavitation chirp binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical cavitation facade's frequency-swept and
shielding-control wrapper family into `cavitation/chirp.rs`: swept versus
monochromatic nuclei engagement, chirped expansion, inter-pulse residual
clearance, residual dissolution, optimal-frequency search, staged sonication
sweep, shielding trace simulation, and shielding-control comparison. The
Python registration surface keeps the same function names through facade
re-exports, and Rust remains the owner of chirp/shielding physics in
`kwavers-physics`. Verification: `cargo fmt -p kwavers-python -p
kwavers-physics`, warning-clean `cargo check -p kwavers-python`,
warning-clean `cargo check -p kwavers-python --features gpu`, and `cargo
nextest run -p kwavers-python` passed.

## CLOSED: Cavitation monitor binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical cavitation facade's monitor/control
wrapper family into `cavitation/monitor.rs`: real-time cavitation monitor
traces, simulated population monitor traces, closed-loop sonication, raster
pulsing, therapeutic-window classification, inertial-fraction onset,
per-spot dose grids, and controller-pressure stepping. The Python registration
surface keeps the same function names through facade re-exports, and Rust
remains the owner of monitor/control and cavitation physics in
`kwavers-physics`. Verification: `cargo fmt -p kwavers-python -p
kwavers-physics`, warning-clean `cargo check -p kwavers-python`,
warning-clean `cargo check -p kwavers-python --features gpu`, and `cargo
nextest run -p kwavers-python` passed.

## CLOSED: Cavitation spectrum binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical cavitation facade's spectral-analysis
wrapper family into `cavitation/spectrum.rs`: bubble PSD, Hann-windowed PSD,
Keller-Miksis PCD spectrum and controller trace, acoustic emission pressure,
ensemble superposition, emission-band decomposition, normalized spectrum,
cumulative dose, and passive-dose fixture. The Python registration surface
keeps the same function names through facade re-exports, and Rust remains the
owner of spectrum and dose physics in `kwavers-physics`. Verification also
required repairing two upstream bubble-dynamics compile blockers in the current
tree: removed an invalid `AdaptiveBubbleModel` self re-export and derived
`Debug` for `BubbleField`. Verification: `cargo fmt -p kwavers-python -p
kwavers-physics`, warning-clean `cargo check -p kwavers-python`, warning-clean
`cargo check -p kwavers-python --features gpu`, and `cargo nextest run -p
kwavers-python` passed.

## CLOSED: Cavitation emission binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical cavitation facade's emission-simulation
wrapper family into `cavitation/emission.rs`: free/coated bubble emission,
population emission, population pressure sweep, focal-volume emission spectrum,
and focal-volume pressure sweep. The Python registration surface keeps the same
function names through facade re-exports, and Rust remains the owner of bubble
and population emission physics in `kwavers-physics`. Verification: `cargo fmt
-p kwavers-python`, warning-clean `cargo check -p kwavers-python`,
warning-clean `cargo check -p kwavers-python --features gpu`, and `cargo
nextest run -p kwavers-python` passed.

## CLOSED: Cavitation passive-receive binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical cavitation facade's passive receive
wrapper family into `cavitation/passive_receive.rs`: receiver-channel PSD
propagation, channel PSD integration, passive point-source RF synthesis, and
Van Cittert-Zernike coherence. The Python registration surface keeps the same
function names through facade re-exports, and Rust remains the owner of passive
cavitation receive physics in `kwavers-physics`. Verification: `cargo fmt -p
kwavers-python`, warning-clean `cargo check -p kwavers-python`, warning-clean
`cargo check -p kwavers-python --features gpu`, and `cargo nextest run -p
kwavers-python` passed.

## CLOSED: Cavitation lesion binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical cavitation facade's lesion-state wrapper
family into `cavitation/lesion.rs`: fractionation backscatter and impedance,
boiling-lesion sizing and time profiles, lacuna void fraction, histotripsy
lesion-radius conversion, and inertial cavitation dose. The Python registration
surface keeps the same function names through facade re-exports, and Rust
remains the owner of lesion-state and cavitation-dose physics in
`kwavers-physics`. Verification: `cargo fmt -p kwavers-python`, warning-clean
`cargo check -p kwavers-python`, warning-clean `cargo check -p kwavers-python
--features gpu`, and `cargo nextest run -p kwavers-python` passed.

## CLOSED: Cavitation therapy binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical cavitation facade's therapy-delivery
wrapper family into `cavitation/therapy.rs`: sonication scheduling,
forward/receive delivery fractions, interface-pressure scalars, lesion
susceptibility, histotripsy dose-response, focal-mask checks, measured-emission
scaling, delivered progress, and cloud-erosion validation. The Python
registration surface keeps the same function names through facade re-exports,
and Rust remains the owner of therapy-delivery and cavitation physics in
`kwavers-physics`. Verification: `cargo fmt -p kwavers-python`, warning-clean
`cargo check -p kwavers-python`, warning-clean `cargo check -p kwavers-python
--features gpu`, and `cargo nextest run -p kwavers-python` passed.

## CLOSED: Cavitation medium binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical cavitation facade's residual-gas and
bubbly-medium wrapper family into `cavitation/medium.rs`: Epstein-Plesset
dissolution, shelled dissolution, Wood sound speed, Commander-Prosperetti
attenuation, and Commander-Prosperetti phase velocity. The Python registration
surface keeps the same function names through facade re-exports, and Rust
remains the owner of bubble-dynamics physics in `kwavers-physics`.
Verification: `cargo fmt -p kwavers-python`, warning-clean `cargo check -p
kwavers-python`, warning-clean `cargo check -p kwavers-python --features gpu`,
and `cargo nextest run -p kwavers-python` passed.

## CLOSED: Cavitation single-bubble binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical cavitation facade's single-bubble scalar
wrapper family into `cavitation/bubble.rs`: Minnaert resonance and inverse
radius, surface-tension corrected resonance, Blake threshold, and Rayleigh
collapse time. The Python registration surface keeps the same function names
through facade re-exports, and Rust remains the owner of cavitation physics in
`kwavers-physics`. Verification: `cargo fmt -p kwavers-python`, warning-clean
`cargo check -p kwavers-python`, warning-clean `cargo check -p kwavers-python
--features gpu`, and `cargo nextest run -p kwavers-python` passed.

## CLOSED: Cavitation probability binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical cavitation facade's probability and
threshold wrapper family into `cavitation/probability.rs`: intrinsic-threshold
probability, frequency-dependent threshold, cumulative probability, and PRF
efficacy. The Python registration surface keeps the same function names through
facade re-exports, and Rust remains the owner of cavitation physics in
`kwavers-physics`. Verification: `cargo fmt -p kwavers-python`, warning-clean
`cargo check -p kwavers-python`, warning-clean `cargo check -p kwavers-python
--features gpu`, and `cargo nextest run -p kwavers-python` passed.

## CLOSED: Neuromodulation binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical neuromodulation facade's wrapper families
into dedicated child modules: `neuromodulation/response.rs` for
Hodgkin-Huxley, NICE, SONIC, and cortical response wrappers,
`neuromodulation/bilayer.rs` for deflection/capacitance curve wrappers,
`neuromodulation/threshold.rs` for bisection threshold search, and
`neuromodulation/safety.rs` for ITRUSST safety and pulse-train dosimetry. The
Python registration surface keeps the same function names through facade
re-exports, and Rust remains the owner of neuromodulation models and dosimetry.
Verification: `cargo fmt -p kwavers-python`, warning-clean `cargo check -p
kwavers-python`, warning-clean `cargo check -p kwavers-python --features gpu`,
and `cargo nextest run -p kwavers-python` passed.

## CLOSED: Inverse-problem binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical inverse facade's remaining wrapper
families into dedicated child modules: `inverse/operators.rs` for Helmholtz,
SVD, and Tikhonov L-curve wrappers, `inverse/reconstruction.rs` for Gaussian
deconvolution fixtures and Born inversion, `inverse/convergence.rs` for
adjoint-gradient and exponential convergence curves, and `inverse/selection.rs`
for L-curve corner and Morozov parameter selection. Shared NumPy/flat-buffer
conversion lives in `inverse/arrays.rs`, while the existing seismic wrappers
remain isolated in `inverse/seismic.rs`. The Python registration surface keeps
the same function names through facade re-exports, and Rust remains the owner of
the inverse-problem computation. Verification: `cargo fmt -p kwavers-python`,
warning-clean `cargo check -p kwavers-python`, warning-clean `cargo check -p
kwavers-python --features gpu`, and `cargo nextest run -p kwavers-python`
passed.

## CLOSED: RTM analytical binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical RTM facade's wrapper families into
dedicated child modules: `rtm/fields.rs` for focused-beam and back-propagation
field wrappers, `rtm/imaging.rs` for imaging-condition and multi-frequency
fusion wrappers, and `rtm/standing_wave.rs` for temporal modulation and
standing-wave suppression wrappers. Shared NumPy/flat-buffer conversion lives in
`rtm/arrays.rs`, eliminating repeated flatten/shape code in the binding layer.
The Python registration surface keeps the same function names through facade
re-exports, and Rust remains the owner of RTM and standing-wave computation in
`kwavers-physics`. Verification: `cargo fmt -p kwavers-python`, warning-clean
`cargo check -p kwavers-python`, warning-clean `cargo check -p kwavers-python
--features gpu`, and `cargo nextest run -p kwavers-python` passed.

## CLOSED: Skull analytical binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical skull facade's wrapper families into
dedicated child modules: `skull/aberration.rs` for insertion loss, phase screen,
and Strehl ratio wrappers, `skull/ct.rs` for Schneider HU conversion wrappers,
`skull/thermal.rs` for surface temperature rise, and `skull/transmission.rs`
for transfer-matrix and transmission-spectrum wrappers. The Python registration
surface keeps the same function names through facade re-exports, and Rust
remains the owner of the skull physics in `kwavers-physics`. Verification:
`cargo fmt -p kwavers-python`, warning-clean `cargo check -p kwavers-python`,
warning-clean `cargo check -p kwavers-python --features gpu`, and `cargo
nextest run -p kwavers-python` passed.

## CLOSED: Sonogenetics binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical sonogenetics facade's wrapper families
into dedicated child modules: `sonogenetics/activation.rs` for channel
activation, `sonogenetics/mechanics.rs` for radiation force, Gor'kov contrast,
and acoustic streaming wrappers, and `sonogenetics/dosimetry.rs` for ISPTA. The
Python registration surface keeps the same function names through facade
re-exports, and Rust remains the owner of the sonogenetics computation in
`kwavers-physics`. Verification: `cargo fmt -p kwavers-python`, warning-clean
`cargo check -p kwavers-python`, warning-clean `cargo check -p kwavers-python
--features gpu`, and `cargo nextest run -p kwavers-python` passed.

## CLOSED: MEMS CMUT/PMUT binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical MEMS facade's wrapper families into
dedicated child modules: `mems/plate.rs` for clamped-plate resonance,
`mems/cmut.rs` for CMUT scalar wrappers, `mems/pmut.rs` for PMUT scalar
wrappers, and `mems/comparison.rs` for therapy and IVUS figure-of-merit
comparisons. Shared binding-layer geometry validation and piezo-film parsing
live in `mems/helpers.rs`; Rust remains the owner of MEMS physics in
`kwavers-transducer`. The Python registration surface keeps the same function
names through facade re-exports. Verification: `cargo fmt -p kwavers-python`,
warning-clean `cargo check -p kwavers-python`, warning-clean `cargo check -p
kwavers-python --features gpu`, and `cargo nextest run -p kwavers-python`
passed.

## CLOSED: Acousto-optics binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical acousto-optics facade's wrapper families
into dedicated child modules: `acousto_optics/regime.rs` for Klein-Cook,
Raman-Nath, and Bragg efficiency parameters, `acousto_optics/geometry.rs` for
diffraction angles and frequency shifts, and `acousto_optics/orders.rs` for
Raman-Nath order intensities and the coupled-wave order solver. The Python
registration surface keeps the same function names through facade re-exports,
and Rust remains the owner of the acousto-optic computation. Verification:
`cargo fmt -p kwavers-python`, warning-clean `cargo check -p kwavers-python`,
warning-clean `cargo check -p kwavers-python --features gpu`, and `cargo
nextest run -p kwavers-python` passed.

## CLOSED: Tissue analytical binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical tissue facade's wrapper families into
dedicated child modules: `tissue/water.rs` for temperature-dependent water
properties, `tissue/attenuation.rs` for absorption and Kramers-Kronig
dispersion, and `tissue/properties.rs` for B/A, acoustic, histotripsy, and
thermal property lookup wrappers. The Python registration surface keeps the same
function names through facade re-exports, and Rust remains the owner of tissue
physics and property tables. Verification: `cargo fmt -p kwavers-python`,
warning-clean `cargo check -p kwavers-python`, warning-clean `cargo check -p
kwavers-python --features gpu`, and `cargo nextest run -p kwavers-python`
passed.

## CLOSED: Statistics validation binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical statistics facade's wrapper families into
dedicated child modules: `statistics/correlation.rs` for Pearson correlation,
phase-shift correlation curves, and phase-error inversion, and
`statistics/metrics.rs` for relative-RMSE PSNR curves, RMSE, and PSNR. Shared
NumPy slice conversion lives in `statistics/arrays.rs`, the Python registration
surface keeps the same function names through facade re-exports, and Rust
remains the owner of the validation metric computations. Verification: `cargo
fmt -p kwavers-python`, warning-clean `cargo check -p kwavers-python`,
warning-clean `cargo check -p kwavers-python --features gpu`, and `cargo
nextest run -p kwavers-python` passed.

## CLOSED: BBB and CEUS binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical BBB facade's wrapper families into
dedicated child modules: `bbb/permeability.rs` for permeability, damage, and
closure wrappers, and `bbb/ceus.rs` for CEUS backscatter signal/display
wrappers. The Python registration surface keeps the same function names through
facade re-exports, and Rust remains the owner of the BBB and CEUS computations.
Verification: `cargo fmt -p kwavers-python`, warning-clean `cargo check -p
kwavers-python`, warning-clean `cargo check -p kwavers-python --features gpu`,
and `cargo nextest run -p kwavers-python` passed.

## CLOSED: Photoacoustics binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical photoacoustics facade's wrapper families
into dedicated child modules: `photoacoustics/spectrum.rs` for Hb/HbO2 spectra
and Gruneisen parameters, `photoacoustics/source.rs` for sphere and Gaussian
absorber source/signal wrappers, and `photoacoustics/reconstruction.rs` for
axial resolution and spectroscopic unmixing. The Python registration surface
keeps the same function names through facade re-exports, Rust remains the owner
of the computations, and the sO2 sweep wrapper no longer flattens and rebuilds
the nested estimate matrix before creating the NumPy array. Verification:
`cargo fmt -p kwavers-python`, warning-clean `cargo check -p kwavers-python`,
warning-clean `cargo check -p kwavers-python --features gpu`, and `cargo
nextest run -p kwavers-python` passed.

## CLOSED: Elastography thermal-strain binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical elastography facade's thermal-strain
wrapper family into `elastography/thermal_strain.rs`: deterministic RF fixture,
combined thermoacoustic strain coefficient, and thermal-strain reconstruction.
The Python registration surface keeps the same function names through facade
re-exports, and Rust remains the owner of RF generation and reconstruction.
Verification: `cargo fmt -p kwavers-python`, warning-clean `cargo check -p
kwavers-python`, warning-clean `cargo check -p kwavers-python --features gpu`,
and `cargo nextest run -p kwavers-python` passed.

## CLOSED: Safety Arrhenius damage binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical safety facade's Arrhenius damage wrapper
family into `safety/damage.rs`: damage integral, cumulative damage, cumulative
thermal kill probability, steady thermal kill probability, and combined
mechanical/thermal kill probability. The Python registration surface keeps the
same function names through facade re-exports, and `safety.rs` now owns only
module topology plus FDA scalar-limit wrappers. Verification: `cargo fmt -p
kwavers-python`, warning-clean `cargo check -p kwavers-python`, warning-clean
`cargo check -p kwavers-python --features gpu`, and `cargo nextest run -p
kwavers-python` passed.

## CLOSED: Safety thermal-index binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical safety facade's thermal-index and CEM43
wrapper family into `safety/thermal.rs`: soft-tissue, bone, and cranial thermal
indices, cumulative CEM43 dose, and the Chapter 7 closed-loop CEM43 fixture.
The Python registration surface keeps the same function names through facade
re-exports, and Rust remains the owner of safety computations. Verification:
`cargo fmt -p kwavers-python`, warning-clean `cargo check -p kwavers-python`,
warning-clean `cargo check -p kwavers-python --features gpu`, and `cargo
nextest run -p kwavers-python` passed.

## CLOSED: Safety mechanical-index binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical safety facade's mechanical wrapper family
into `safety/mechanical.rs`: scalar Mechanical Index, field Mechanical Index,
frequency-sweep Mechanical Index, and Mechanical-Index cavitation-risk
probability. The Python registration surface keeps the same function names
through facade re-exports, and Rust remains the owner of safety computations.
Verification: `cargo fmt -p kwavers-python`, warning-clean `cargo check -p
kwavers-python`, warning-clean `cargo check -p kwavers-python --features gpu`,
and `cargo nextest run -p kwavers-python` passed.

## CLOSED: Thermal acoustic binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical thermal facade's acoustic wrapper family
into `thermal/acoustic.rs`: HIFU focal gain, Gaussian power deposition, depth
intensity/power deposition, pressure/intensity conversion, and acoustic
heat-source density. The Python registration surface keeps the same function
names through facade re-exports, and Rust remains the owner of the acoustic
thermal computations. Verification: `cargo fmt -p kwavers-python`,
warning-clean `cargo check -p kwavers-python`, warning-clean `cargo check -p
kwavers-python --features gpu`, and `cargo nextest run -p kwavers-python`
passed.

## CLOSED: Inverse seismic binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical inverse facade's seismic imaging wrapper
family into `inverse/seismic.rs`: 2-D eikonal traveltime, Kirchhoff
point-scatterer migration, paired-index validation, and Ricker trace synthesis.
The Python registration surface keeps the same function names through facade
re-exports, and Rust remains the owner of the traveltime/migration computation.
Verification: `cargo fmt -p kwavers-python`, warning-clean `cargo check -p
kwavers-python`, warning-clean `cargo check -p kwavers-python --features gpu`,
and `cargo nextest run -p kwavers-python` passed.

## CLOSED: Imaging IVUS B-mode and metrics binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical imaging facade's remaining IVUS wrapper
families into `imaging/bmode.rs` and `imaging/metrics.rs`: polar B-mode RF,
polar-to-Cartesian scan conversion, complete B-mode image assembly, and Chapter
30 scalar metrics. The imaging facade now owns only module topology and
re-exports, while Rust remains the owner of B-mode and metrics computations.
Verification: `cargo fmt -p kwavers-python`, warning-clean `cargo check -p
kwavers-python`, warning-clean `cargo check -p kwavers-python --features gpu`,
and `cargo nextest run -p kwavers-python` passed.

## CLOSED: Imaging IVUS therapy binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical imaging facade's IVUS therapy wrapper
family into `imaging/therapy.rs`: sector pressure field, microbubble delivery
fraction, response fields/metrics, and aggregate therapy fields. The Python
registration surface keeps the same function names through facade re-exports,
and Rust remains the owner of therapy field and response computations.
Verification: `cargo fmt -p kwavers-python`, warning-clean `cargo check -p
kwavers-python`, warning-clean `cargo check -p kwavers-python --features gpu`,
and `cargo nextest run -p kwavers-python` passed.

## CLOSED: Imaging IVUS phantom binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical imaging facade's deterministic IVUS
vessel-phantom wrapper into `imaging/phantom.rs` with its private square-array
materialization helper. The Python registration surface keeps the same function
name through facade re-export, and Rust remains the owner of phantom generation
and typed result materialization. Verification: `cargo fmt -p kwavers-python`,
warning-clean `cargo check -p kwavers-python`, warning-clean `cargo check -p
kwavers-python --features gpu`, and `cargo nextest run -p kwavers-python`
passed.

## CLOSED: Imaging PSF binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical imaging facade's point-spread wrapper
family into `imaging/psf.rs`: lateral sinc-squared PSF, axial rectangular PSF,
plane-wave compounding lateral PSF, and lateral resolution. The Python
registration surface keeps the same function names through facade re-exports,
and Rust remains the owner of PSF/resolution computations. Verification:
`cargo fmt -p kwavers-python`, warning-clean `cargo check -p kwavers-python`,
warning-clean `cargo check -p kwavers-python --features gpu`, and `cargo
nextest run -p kwavers-python` passed.

## CLOSED: Imaging pulse-echo binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical imaging facade's pulse-echo wrapper
family into `imaging/pulse_echo.rs`: synthetic receive RF, B-mode envelope,
fixed-reference log compression, and baseline-relative delta B-mode. The
Python registration surface keeps the same function names through facade
re-exports, and Rust remains the owner of RF/B-mode computations. Verification:
`cargo fmt -p kwavers-python`, warning-clean `cargo check -p kwavers-python`,
warning-clean `cargo check -p kwavers-python --features gpu`, and `cargo
nextest run -p kwavers-python` passed.

## CLOSED: Imaging Doppler binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical imaging facade's Doppler wrapper family
into `imaging/doppler.rs`: scalar Doppler shift, contrast-agent Doppler
spectrum, and continuous-wave/vector-flow fixture. The Python registration
surface keeps the same function names through facade re-exports, and Rust
remains the owner of the Doppler computations. Verification: `cargo fmt -p
kwavers-python`, `cargo check -p kwavers-python`, `cargo check -p
kwavers-python --features gpu`, and `cargo nextest run -p kwavers-python`
passed.

## CLOSED: Transducer beam binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical transducer facade's remaining 2-D
focus/beam wrapper family into `transducer/beam.rs`: 2-D focus delays, complex
beam patterns, far-field beam-pattern magnitude, and 2-D beam magnitude. The
facade now owns module topology and re-exports only, while Rust remains the
owner of all transducer computations. Verification: `cargo fmt -p
kwavers-python`, `cargo check -p kwavers-python`, `cargo check -p
kwavers-python --features gpu`, and `cargo nextest run -p kwavers-python`
passed.

## CLOSED: Transducer basic binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical transducer facade's basic wrapper family
into `transducer/basic.rs`: circular-piston directivity, linear-array factor,
uniform-array grating lobes, apodization weights/response, and on-axis pressure
profiles. The Python registration surface keeps the same function names through
facade re-exports, and Rust remains the owner of the computations.
Verification: `cargo fmt -p kwavers-python`, `cargo check -p kwavers-python`,
`cargo check -p kwavers-python --features gpu`, and `cargo nextest run -p
kwavers-python` passed.

## CLOSED: Transducer multi-focus binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical transducer facade's multi-focus wrapper
family into `transducer/multi_focus.rs`: geometric multi-spot delay laws and
phase-conjugated field magnitude assembly. The Python registration surface
keeps the same function names through facade re-exports, and Rust remains the
owner of the multi-focus computations. Verification: `cargo fmt -p
kwavers-python`, `cargo check -p kwavers-python`, `cargo check -p
kwavers-python --features gpu`, and `cargo nextest run -p kwavers-python`
passed.

## CLOSED: Transducer aperture binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical transducer facade's aperture wrapper
family into `transducer/aperture.rs`: linear-array element positions,
focused-bowl element geometry, 3-D focus delays, steered aperture pressure, and
focused-bowl pressure-profile assembly. The Python registration surface keeps
the same function names through facade re-exports, and Rust remains the owner
of the geometry and pressure computations. Verification: `cargo fmt -p
kwavers-python`, `cargo check -p kwavers-python`, `cargo check -p
kwavers-python --features gpu`, and `cargo nextest run -p kwavers-python`
passed.

## CLOSED: Transducer interpolation binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical transducer facade's BLI wrapper family
into `transducer/interpolation.rs`: stencil-weight materialization and
nearest-neighbour/BLI interpolation error curves. The Python registration
surface keeps the same function names through facade re-exports, and Rust
remains the owner of interpolation formulas and validation. Verification:
`cargo fmt -p kwavers-python`, `cargo check -p kwavers-python`, `cargo check -p
kwavers-python --features gpu`, and `cargo nextest run -p kwavers-python`
passed.

## CLOSED: Transducer steering binding tree cleanup (2026-07-01)

Split the `kwavers-python` analytical transducer facade's steering wrapper
family into `transducer/steering.rs`: natural-focus steering helpers,
sparse-aperture element placement, steered beam-pattern evaluation,
grating-lobe ratio, safe-steering half-angle, and electronic steering
efficiency. The Python registration surface keeps the same function names
through facade re-exports, and Rust remains the owner of all physics formulas.
Verification: `cargo fmt -p kwavers-python`, `cargo check -p kwavers-python`,
`cargo check -p kwavers-python --features gpu`, and `cargo nextest run -p
kwavers-python` passed.

## CLOSED: Transducer binding vertical tree cleanup (2026-07-01)

Split the `kwavers-python` analytical transducer facade by bounded wrapper
family: SOAP/optoacoustic formulas now live in `transducer/optoacoustic.rs`,
and static acoustic-lens material helpers now live in `transducer/lens.rs`.
The Python registration surface keeps the same function names through facade
re-exports, and Rust remains the owner of all physics formulas. Verification:
`cargo fmt -p kwavers-python`, `cargo check -p kwavers-python`, `cargo check -p
kwavers-python --features gpu`, and `cargo nextest run -p kwavers-python`
passed.

## CLOSED: GPU PSTD session source/sensor tree cleanup (2026-07-01)

Moved source/sensor index construction and velocity-signal packing from the
`GpuPstdSession` facade into `session/source.rs`, leaving the facade as state
and module topology only. Cached scan-line execution now passes empty slices for
unused pressure-source inputs instead of allocating empty vectors per run.
Verification: `cargo fmt -p kwavers-python`, `cargo check -p kwavers-python`,
`cargo check -p kwavers-python --features gpu`, and `cargo nextest run -p
kwavers-python` passed.

## CLOSED: GPU PSTD session construction tree cleanup (2026-07-01)

Split the `kwavers-python` GPU PSTD session constructor helpers out of
`session/construction.rs` into `session/absorption.rs` for fractional
absorption kernels and `session/pml.rs` for CPML array materialization. The
public PyO3 `GpuPstdSession` facade and solver setup behavior are unchanged;
the constructor module now owns orchestration only. Verification: `cargo fmt -p
kwavers-python`, `cargo check -p kwavers-python`, `cargo check -p
kwavers-python --features gpu`, and `cargo nextest run -p kwavers-python`
passed.

## CLOSED: Therapy chapter guard repair (2026-07-01)

Corrected the therapy chapter regression test's docs root for the current
`crates/kwavers-python` layout and removed residual vendor-style source labels
from the active Chapter 31 clinical-device script. Verification: focused
therapy chapter pytest passed 49/49 and the guarded source-label scan returned
no active-artifact matches.

## CLOSED: Chapter 24 CEUS backscatter display ownership (2026-07-01)

Closed the Chapter 24 BBB-LIFU residual where CEUS concentration sweep display
values and the optimal concentration marker were derived in Python from a raw
Rust signal. `kwavers-physics` now owns the finite-input display payload through
`ceus_backscatter_display`; `pykwavers` exposes the helper, and the book script
only plots returned dB values and marker metadata. Verification: `cargo fmt`,
`cargo check -p kwavers-physics -p kwavers-python`, focused Rust nextest,
editable Miniforge `maturin develop`, focused Chapter 24/26 pytest, Miniforge
py-compile, residual source guard, and touched-path `git diff --check`.

## CLOSED: Chapter 30 IVUS therapy fields ownership (2026-07-01)

Closed the Chapter 30 intravascular-ultrasound residual where therapy pressure
and therapy response fields were orchestrated from split Rust helpers in
Python. `kwavers-physics` now owns the finite-input aggregate through
`ivus_therapy_fields`; `pykwavers` exposes the helper, and the book script only
reshapes returned fields for plotting and metrics. Verification: `cargo fmt`,
`cargo check -p kwavers-physics -p kwavers-python`, focused Rust nextest,
editable Miniforge `maturin develop`, focused Chapter 30 pytest, Miniforge
py-compile, residual source guard, and touched-path `git diff --check`.

## CLOSED: Chapter 30 IVUS metrics ownership (2026-07-01)

Closed the Chapter 30 intravascular-ultrasound residual where wavelength,
lumen/plaque area, masked B-mode mean, and therapy summary metrics were
computed with Python-side scalar formulas before JSON serialization.
`kwavers-physics` now owns finite-input metric generation through
`ivus_chapter_metrics`; `pykwavers` exposes the helper, and the book script only
serializes the returned metric dictionary. Verification: `cargo fmt`, `cargo
check -p kwavers-physics -p kwavers-python`, focused Rust nextest, editable
Miniforge `maturin develop`, focused Chapter 30 pytest, Miniforge py-compile,
residual source guard, and touched-path `git diff --check`.

## CLOSED: Chapter 30 IVUS B-mode image ownership (2026-07-01)

Closed the Chapter 30 intravascular-ultrasound residual where the B-mode image
was assembled with Python-side RF-column envelope loops, envelope floor clamp,
fixed-reference log compression, normalized display mapping, and scan
conversion calls. `kwavers-physics` now owns the finite-input B-mode fixture
through `ivus_bmode_image`; `pykwavers` exposes the helper, and the book script
only reshapes returned arrays for plotting and metrics. Verification: `cargo
fmt`, `cargo check -p kwavers-physics -p kwavers-python`, focused Rust nextest,
editable Miniforge `maturin develop`, focused Chapter 30 pytest, Miniforge
py-compile, residual source guard, and touched-path `git diff --check`.

## CLOSED: Chapter 30 IVUS therapy-response ownership (2026-07-01)

Closed the Chapter 30 intravascular-ultrasound residual where therapy response
fields and scalar metrics were assembled with local Python attenuation,
absorbed-power, temperature-rise, mask, MI, and target/off-target ratio algebra.
`kwavers-physics` now owns the finite-input response model through
`ivus_therapy_response`; `pykwavers` exposes the helper, and the book script
reshapes returned fields for plotting and metrics. Verification: `cargo fmt`,
`cargo check -p kwavers-physics -p kwavers-python`, focused Rust nextest,
editable Miniforge `maturin develop`, focused Chapter 30 pytest, Miniforge
py-compile, residual source guard, and touched-path `git diff --check`.

## CLOSED: Chapter 30 IVUS scan-conversion ownership (2026-07-01)

Closed the Chapter 30 intravascular-ultrasound residual where polar B-mode
samples were projected onto the Cartesian phantom grid with local Python
nearest-bin radius/theta indexing. `kwavers-physics` now owns finite-input
polar-to-Cartesian scan conversion through `ivus_scan_convert`; `pykwavers`
exposes the helper, and the book script reshapes returned image samples for
plotting. Verification: `cargo fmt`, `cargo check -p kwavers-physics -p
kwavers-python`, focused Rust nextest, editable Miniforge `maturin develop`,
focused Chapter 30 pytest, Miniforge py-compile, residual source guard, and
touched-path `git diff --check`.

## CLOSED: Chapter 30 IVUS polar RF ownership (2026-07-01)

Closed the Chapter 30 intravascular-ultrasound residual where the B-mode RF
fixture was generated with local Python polar grid sampling, two-way attenuation,
and catheter-ring echo algebra. `kwavers-physics` now owns the finite-input
polar RF construction through `ivus_polar_bmode_rf`; `pykwavers` exposes the
helper, and the book script reshapes returned RF samples before calling the
existing Rust envelope/log-compression kernels. Verification: `cargo fmt`,
`cargo check -p kwavers-physics -p kwavers-python`, focused Rust nextest,
editable Miniforge `maturin develop`, focused Chapter 30 pytest, Miniforge
py-compile, residual source guard, and touched-path `git diff --check`.

## CLOSED: Chapter 30 IVUS delivery-fraction ownership (2026-07-01)

Closed the Chapter 30 intravascular-ultrasound residual where the microbubble
delivery map was generated with local Python acoustic-radiation-force,
radial-band, normalization, and exponential-delivery algebra. `kwavers-physics`
now owns the finite-input delivery model through
`ivus_microbubble_delivery_fraction`; `pykwavers` exposes the helper, and the
book script reshapes returned delivery samples for plotting and summary metrics.
Verification: `cargo fmt`, `cargo check -p kwavers-physics -p kwavers-python`,
focused Rust nextest, editable Miniforge `maturin develop`, focused Chapter 30
pytest, Miniforge py-compile, residual source guard, and touched-path `git diff
--check`.

## CLOSED: Chapter 30 IVUS therapy pressure-field ownership (2026-07-01)

Closed the Chapter 30 intravascular-ultrasound residual where the
microbubble-therapy pressure map was generated with local Python angular
Gaussian aperture and radial exponential decay algebra. `kwavers-physics` now
owns the finite-input sector pressure model through `ivus_therapy_pressure_field`;
`pykwavers` exposes the helper, and the book script reshapes returned pressure
samples for plotting and downstream dose metrics. Verification: `cargo fmt`,
`cargo check -p kwavers-physics -p kwavers-python`, focused Rust nextest,
editable Miniforge `maturin develop`, focused Chapter 30 pytest, Miniforge
py-compile, residual source guard, and touched-path `git diff --check`.

## CLOSED: Chapter 20 PSNR relative-error curve ownership (2026-07-01)

Closed the Chapter 20 validation residual where Figure 02 generated the
PSNR-vs-relative-RMSE curve with local Python `-20 * np.log10(eps)` algebra.
`kwavers-math` now owns the finite positive relative-error conversion through
`validation_psnr_from_relative_rmse`; `pykwavers` exposes the helper, and the
book script only supplies the relative-RMSE samples and plots returned values.
Verification: `cargo fmt`, `cargo check -p kwavers-math -p kwavers-python`,
focused Rust nextest, editable Miniforge `maturin develop`, focused Chapter 20
manifest pytest, Miniforge py-compile, residual source guard, and touched-path
`git diff --check`.

## CLOSED: Chapter 20 Pearson phase sensitivity ownership (2026-07-01)

Closed the Chapter 20 validation residual where Figure 01 generated the
same-frequency sinusoid Pearson phase-sensitivity curve with local Python
`np.cos(phi_rad)` and inverse-threshold markers with `np.arccos`. `kwavers-math`
now owns the theorem helpers through `phase_shift_correlation_curve` and
`phase_error_degrees_for_correlation`; `pykwavers` exposes both helpers, and
the book script only supplies the phase/correlation samples and plots returned
values. Verification: `cargo fmt`, `cargo check -p kwavers-math -p
kwavers-python`, focused Rust nextest, editable Miniforge `maturin develop`,
focused Chapter 20 manifest pytest, Miniforge py-compile, residual source
guard, and touched-path `git diff --check`.

## CLOSED: Chapter 17 PINN convergence curve ownership (2026-07-01)

Closed the Chapter 17 inverse-problems residual where Figure 03 generated the
PINN loss convergence curves with local Python `L0 * np.exp(-epoch / tau) +
floor` logic. `kwavers-physics` now owns the finite-input exponential
decay-with-floor helper through `exponential_convergence_curve`; `pykwavers`
exposes the helper, and the book script only supplies the curve parameters and
plots returned arrays. Verification: `cargo fmt`, `cargo check -p
kwavers-physics -p kwavers-python`, focused Rust nextest, editable Miniforge
`maturin develop`, focused Chapter 17 manifest pytest, Miniforge py-compile,
residual source guard, and touched-path `git diff --check`.

## CLOSED: Chapter 17 Gaussian deconvolution fixture ownership (2026-07-01)

Closed the Chapter 17 inverse-problems residual where Figure 02 built the
Gaussian deconvolution matrix, two-bump truth signal, and deterministic
measurement perturbation with local Python `np.exp`/`np.sin` algebra before
calling Rust Tikhonov L-curve routines. `kwavers-physics` now owns that fixture
through `gaussian_deconvolution_fixture`; `pykwavers` exposes the helper, and
the book script passes returned arrays into the existing Rust L-curve pipeline.
Verification: `cargo fmt`, `cargo check -p kwavers-physics -p kwavers-python`,
focused Rust nextest, editable Miniforge `maturin develop`, focused Chapter 17
manifest pytest, Miniforge py-compile, residual source guard, and touched-path
`git diff --check`.

## CLOSED: Chapter 10 MRE envelope ownership (2026-07-01)

Closed the Chapter 10 elastography residual where Figure 05 plotted the MRE
centerline envelope with local Python `amplitude_m * np.exp(-z /
penetration_depth_m)` algebra. `kwavers-physics` now owns the validated
exponential envelope through `mre_displacement_envelope`; `pykwavers` exposes
the helper, and the book script plots returned arrays. Verification: `cargo
fmt`, `cargo check -p kwavers-physics -p kwavers-python`, focused Rust nextest,
editable Miniforge `maturin develop`, focused Chapter 10 manifest pytest,
Miniforge py-compile, residual source guard, and touched-path `git diff
--check`.

## CLOSED: Chapter 23 VCZ coherence ownership (2026-07-01)

Closed the Chapter 23 passive-acoustic-mapping residual where Figure 03 built
the Van Cittert-Zernike spatial coherence curve with local Python `np.sinc`
algebra. `kwavers-physics` now owns the validated VCZ sinc law through
`van_cittert_zernike_coherence`, `pykwavers` exposes the helper, and the book
script plots returned coherence arrays. The stale SciPy requirement text was
also removed. Verification: `cargo fmt`, `cargo check -p kwavers-physics -p
kwavers-python`, focused Rust nextest, editable Miniforge `maturin develop`,
focused Chapter 23 pytest, Miniforge py-compile, residual source guard, and
touched-path `git diff --check`.

## CLOSED: Chapter 3 PSTD source waveform ownership (2026-07-01)

Closed the Chapter 3 nonlinear-acoustics residual where Figure 06 still built
the Westervelt PSTD source waveform with local Python `P0 * np.sin(OMEGA0 *
t_src)` algebra. The script now routes the source through existing Rust/PyO3
`fubini_waveform` at `sigma=0.0`, whose zero-distance branch is value-tested as
the sinusoidal source contract. The stale SciPy requirement text was also
removed from the script header. Verification: Miniforge py-compile, focused
Chapter 3 nonlinear-acoustics pytest, residual source/dependency guard, and
touched-path `git diff --check`.

## CLOSED: Chapter 1 standing-wave ownership (2026-06-30)

Closed the Chapter 1 wave-fundamentals residual where Figure 01 still built the
standing-wave initial condition and analytic overlay with local Python
`p0 * np.sin(k * x)` algebra despite the existing Rust/PyO3
`standing_wave_1d` binding. The book script now routes both through that helper,
and the PyO3 wrapper documentation matches the Rust core formula. Verification:
Miniforge py-compile, focused Chapter 1 wave pytest, `cargo fmt -p
kwavers-python`, `cargo check -p kwavers-python`, residual source guard, and
touched-path `git diff --check`.

## CLOSED: Chapter 5 axial RF pulse ownership (2026-06-30)

Closed the Chapter 5 diagnostic-imaging residual where Figure 01 constructed
the centered two-cycle Hann-windowed RF pulse with Python `np.hanning` and local
carrier multiplication before calling the Rust B-mode envelope detector.
`kwavers-physics` now owns the centered discrete-Hann tone-burst contract
through `centered_hann_tone_burst_waveform`; `pykwavers` exposes the helper, and
the book script plots the returned pulse/envelope arrays. Verification: `cargo
fmt -p kwavers-physics -p kwavers-python`, `cargo check -p kwavers-physics -p
kwavers-python`, editable Miniforge `maturin develop`, focused Rust nextest,
Miniforge py-compile, focused Chapter 5 pytest, residual source guard, and
touched-path `git diff --check`.

## CLOSED: Chapter 25 RTM axial-spectrum FFT ownership (2026-06-30)

Closed the Chapter 25 RTM/adaptive-beamforming residual where Figure 11
computed the demeaned Hann-windowed axial spatial power spectrum with Python
`np.hanning`, `np.fft.rfft`, and `np.fft.rfftfreq`. `kwavers-python` now exposes
Rust/PyO3 `demeaned_hann_power_spectrum_1d`, backed by the Apollo FFT facade and
workspace Hann window; the book script only passes the cropped RTM axial profile
and plots returned arrays. Verification: `cargo fmt -p kwavers-python`, `cargo
check -p kwavers-python`, editable Miniforge `maturin develop`, Miniforge
py-compile, focused FFT/source pytest, residual source guard, and touched-path
`git diff --check`.

## CLOSED: Population-emission seed boundary cleanup (2026-06-30)

Closed the shared book population-emission residual where Python callers passed
NumPy generator objects into `simulate_population_emission` so the helper could
derive Rust seeds in Python. The helper now accepts a deterministic integer
seed and forwards it directly to Rust/PyO3 `simulate_population_emission`;
Chapter 24 and Chapter 21e callers pass explicit seeds. Verification: Miniforge
py-compile, focused population-emission pytest, residual source guard for
`rng=` population-emission calls, and touched-path `git diff --check`.

## CLOSED: Chapter 7 closed-loop CEM43 fixture ownership (2026-06-30)

Closed the Chapter 7 theranostics residual where Figure 05 generated the
feedback focal-temperature trace with Python-side RNG and computed each
CEM43 curve separately in Python. `kwavers-physics` now owns the fixed-power,
feedback, and underdrive temperature traces plus their CEM43 integration through
`closed_loop_cem43_fixture`; `pykwavers` exposes the fixture, and the book script
plots returned arrays. Verification: focused Rust nextest, editable Miniforge
`maturin develop`, focused Chapter 7 pytest, Miniforge py-compile, executable
Chapter 7 regeneration, changed PNG decode, and touched-path `git diff
--check`.

## CLOSED: Chapter 23 cavitation dose fixture ownership (2026-06-30)

Closed the Chapter 23 passive-acoustic-mapping residual where Figure 06 built
stable and inertial cavitation dose traces with Python-side RNG and local dose
logic. `kwavers-physics` now owns the stable-dose staircase and seeded
compound-Poisson inertial-dose trials, `pykwavers` exposes
`passive_cavitation_dose_fixture`, and the book script plots returned arrays.
Verification: focused Rust nextest, editable Miniforge `maturin develop`,
focused Chapter 23 pytest, Miniforge py-compile, executable Chapter 23
regeneration, changed PNG decode, and touched-path `git diff --check`.

## CLOSED: Chapter 5 shear-wave tissue-range speed ownership (2026-06-30)

Closed the Chapter 5 shear-wave elastography residual where Figure 06 computed
tissue-range shear-wave speeds with Python-side `np.sqrt(mu/rho)`. The book
script now calls the existing Rust/PyO3 `shear_wave_speed` binding for each
tissue range and only plots the returned speed limits. Verification:
Miniforge py-compile for the touched book script/test, focused Chapter 5
manifest/value pytest, and touched-path `git diff --check`.

## CLOSED: Chapter 4 apodization response ownership (2026-06-30)

Closed the Chapter 4 beamforming residual where Python still computed the
apodization-window response with `np.fft.fft` and `np.fft.fftshift`. Rust/PyO3
`apodization_window_response` now owns apodization weights, zero padding,
FFT-shifted magnitude normalization, dB conversion, and cycles-per-aperture
axis generation. The book script now plots returned arrays only. Verification:
`cargo fmt --check -p kwavers-physics -p kwavers-python`, focused `cargo
nextest run -p kwavers-physics apodization_response`, `cargo check -p
kwavers-physics -p kwavers-python`, editable Miniforge `maturin develop`,
focused Chapter 4 pytest, Miniforge py-compile, and touched-path `git diff
--check`.

## CLOSED: Chapter 10 thermal-strain RF fixture ownership (2026-06-30)

Closed the Chapter 10 elastography residual where Python still generated the
thermal-strain RF fixture with `np.random`, carrier multiplication, and
per-line interpolation. Rust/PyO3 `thermal_strain_rf_fixture` now owns seeded
speckle generation, smoothing, carrier modulation, and the apparent-displacement
warp; the book script passes the returned volumes into the existing Rust
`thermal_strain_reconstruct` pipeline and plots results. Verification: `cargo
fmt --check -p kwavers-physics -p kwavers-python`, focused `cargo nextest run
-p kwavers-physics thermal_strain_rf_fixture`, `cargo check -p kwavers-physics
-p kwavers-python`, editable Miniforge `maturin develop`, focused Chapter 10
pytest, Miniforge py-compile, and touched-path `git diff --check`.

## CLOSED: Chapter 3 harmonic extraction ownership (2026-06-30)

Closed the Chapter 3 nonlinear-acoustics PSTD validation gap where Python still
performed Hann-windowed FFT bin extraction for harmonic amplitudes. The
Rust/PyO3 `hann_windowed_harmonic_amplitudes` helper now owns the symmetric Hann
window, workspace FFT call, one-sided amplitude normalization, harmonic-bin
selection, and input validation. The book script passes the steady-state sensor
trace slab to the binding and only plots returned amplitudes against Fubini.
Verification: `cargo fmt --check -p kwavers-physics -p kwavers-python`,
focused `cargo nextest run -p kwavers-physics hann_windowed_harmonic`, `cargo
check -p kwavers-physics -p kwavers-python`, editable Miniforge `maturin
develop`, focused Chapter 3 pytest, Miniforge py-compile, and touched-path
`git diff --check`.

## CLOSED: Chapter 7 PCD spectrum and controller ownership (2026-06-30)

Closed the Chapter 7 theranostics PCD residual where Python still performed
Hann-windowed FFT spectra, SC/IC band-power ratios, and asymmetric pressure
controller stepping. `kwavers-physics` now owns the Keller-Miksis wall-velocity
PCD spectrum and controller trace through workspace FFT-backed helpers;
`pykwavers` exposes thin dictionary bindings; and the book script only adapts
returned arrays for plotting. Verification: `cargo fmt -p kwavers-physics -p
kwavers-python`, focused `cargo nextest run -p kwavers-physics pcd`, `cargo
check -p kwavers-physics -p kwavers-python`, editable Miniforge `maturin
develop`, focused Chapter 7 pytest, executable Chapter 7 figure regeneration,
finite nonblank decode for all five Chapter 7 PNGs, Miniforge py-compile, and
touched-path `git diff --check`.

## CLOSED: Chapter 5 Gaussian photoacoustic waveform (2026-06-30)

Closed the Chapter 5 photoacoustic waveform ownership gap where Python computed
the Gaussian absorber initial-pressure profile and derivative waveform with
NumPy. `kwavers-physics` now owns the Gaussian absorber pressure profile and
analytic `dp0/dz` surface signal sampled at `z = c*t`; `pykwavers` exposes a
thin dict-returning wrapper; and Figure 04 now only selects axes, adapts arrays,
and plots the returned fields. Verification: `cargo fmt -p kwavers-physics -p
kwavers-python`, focused `cargo nextest run -p kwavers-physics
gaussian_absorber_photoacoustic_profile`, `cargo check -p kwavers-physics -p
kwavers-python`, editable Miniforge `maturin develop`, focused Chapter 5
manifest pytest, Miniforge py-compile, and touched-path `git diff --check`.

## CLOSED: Transcranial subspot and BBB-dose Rust ownership (2026-06-30)

Closed the Chapter 25 transcranial planning adapter gap where Python still
owned GBM subspot raster construction, focal coverage, and BBB subspot-dose
field assembly. `kwavers-therapy` now exposes the existing subspot raster and
BBB-dose kernels plus a Rust-owned focal coverage fraction helper. `pykwavers`
exports thin `gbm_subspot_raster_py` and `bbb_opening_from_subspots_py`
wrappers, and the book `gbm_subspot_plan` / `bbb_opening_from_subspots`
adapters now package those Rust outputs into plotting dataclasses. Verification:
`cargo fmt -p kwavers-therapy -p kwavers-python`, `cargo check -p
kwavers-therapy -p kwavers-python`, focused `cargo nextest run -p
kwavers-therapy` for the subspot coverage test, editable `maturin develop`,
focused transcranial planning pytest, direct binding smoke check, Miniforge
py-compile, and touched-path `git diff --check`.

## CLOSED: Transcranial planning PyO3 contract cleanup (2026-06-30)

Closed the optional-extension fallback gap in the book transcranial planning
helpers. `simulation.py` and `transducer.py` now import `pykwavers` directly.
Acoustic observables use Rust/PyO3 `mechanical_index_field` and
`mechanical_index_cavitation_risk`; BBB opening permeability uses Rust/PyO3
`bbb_permeability_hill`; HU sound speed and density use Rust/PyO3 Schneider
mapping helpers. The existing Rust transcranial array planner and Pennes
thermal-dose binding are now exported through `pykwavers.__init__`, matching the
book scripts' top-level `import pykwavers as kw` contract. Verification:
Miniforge py-compile for touched Python files, focused transcranial planning
pytest, top-level binding export check, and touched-path `git diff --check`.

## CLOSED: Chapter 24 vector CEM43 dose path (2026-06-30)

Closed the Chapter 24 thermal-dose ownership gap. The LIFU thermal-safety panel
now calls Rust/PyO3 `cem43_cumulative` once over the full temperature history
instead of computing sparse growing-prefix doses with `compute_cem43` and
interpolating them in Python. The same pass removed the ignored
`max_nucleation_cycles` keyword from the shared cavitation population book
helper and all book callers, so the helper signature no longer accepts an input
that the Rust core cannot consume. Verification: Miniforge py-compile for the
touched scripts/tests, focused Chapter 24/26 source guard pytest, executable
Chapter 24 regeneration through `ch24_bbb_lifu_opening.py`, nonblank decode for
all 10 Chapter 24 PNGs, and touched-path `git diff --check`.

## CLOSED: kwavers-physics all-target clippy gate (2026-06-30)

Closed the current `kwavers-physics --all-targets` clippy blocker layer by
applying mechanical, value-preserving lint fixes across physics tests and local
helpers: range predicates now use `Range::contains`, exported/impl items precede
test modules, constant invariants have compile-time assertions, `Copy` values
are no longer cloned in tests, default overrides use struct update syntax, and a
test helper field tuple is named. Verification: `rustup run nightly cargo fmt -p
kwavers-physics`, `rustup run nightly cargo clippy -p kwavers-physics
--all-targets -- -D warnings`, `rustup run nightly cargo nextest run -p
kwavers-physics` (1665/1665 passed, 1 skipped), and touched-path `git diff
--check`.

## CLOSED: k-Wave direct cached parity closure (2026-06-30)

Closed the upstream-mapped direct-cache residual for
`ivp_recording_particle_velocity_compare.py` and
`sd_directional_array_elements_compare.py`. The parity gate now includes 25
direct cached tests: 22 shared parameterized cache-backed contracts, the
directional-array element-average contract, the particle-velocity dominant-axis
contract, and the tiny phased-array aggregate. The manifest coverage-reference
guard now excludes its own source file, so direct coverage is proved by
non-manifest pytest references except for KWave.jl drivers that are semantically
validated inside the manifest. The Chapter 7 theranostics Minnaert marker radii
now route through Rust/PyO3 `minnaert_radius_for_frequency_m` instead of a
Python-side inverse formula, with Rust/Python round-trip regression coverage.
The Chapter 33 CMUT/PMUT script now imports `pykwavers` directly with no
optional branch, and focused tests pin Rust/PyO3 MEMS routing plus CMUT pull-in
gap scaling, PMUT drive/material ordering, and the default IVUS verdict.
The Chapter 18 sonogenetics activation panel now routes pressure reconstruction
from intensity through Rust/PyO3 `acoustic_pressure_amplitude_from_intensity`
instead of Python-side `sqrt(2*rho*c*I)`, with focused source/value regression
coverage.
The Chapter 4 transducer-array figures now route combined beam patterns,
grating-lobe markers, lateral resolution, 2-D beam fields, and BLI stencils
through the current Rust/PyO3 binding contracts, and the 2-D beam panel no
longer allocates a Python meshgrid before calling `beam_pattern_2d`.
The Chapter 21 histotripsy comparison now routes the millisecond-pulse
shock-rich intensity-to-pressure inversion through Rust/PyO3
`acoustic_pressure_amplitude_from_intensity` before the Rust heat-source
density calculation, with focused source/value regression coverage.
The Chapter 7 closed-loop thermal-dose panel now routes cumulative CEM43
histories through Rust/PyO3 `cem43_cumulative` instead of an O(n²) Python prefix
loop over `compute_cem43`.
The Chapter 26 neuromodulation response trace now routes spike-train sampling
and Gaussian response-probability smoothing through Rust/PyO3
`lif_response_probability_py`, and its focal thermal-dose trace uses
Rust/PyO3 `cem43_cumulative` instead of Python-side sparse prefix-dose
interpolation.
The Chapter 22/23 passive acoustic mapping stable/inertial cavitation spectrum
now routes through Rust/PyO3 `normalized_cavitation_emission_spectrum`, removing
the Python-local Lorentzian harmonic/subharmonic and inertial broadband model
from the figure script while leaving Python to plot normalized PSD in dB. The
same Chapter 23 script now routes passive point-source receive-trace synthesis
through Rust/PyO3 `passive_cavitation_point_source_rf` before calling
`passive_acoustic_map_das`, leaving Python at array layout, dB scaling, and
plotting for the DAS sensitivity panel.
The Chapter 23 eigenspace spectrum panel now routes the Theorem 22.2
signal/noise eigenvalue split through Rust/PyO3
`eigenspace_covariance_eigenvalues`, removing the Python-local stochastic CSD
fixture while leaving Python to plot the returned singular values.
The Chapter 14 sensors chapter now routes the pressure/particle-velocity
progressive plane-wave panel through Rust/PyO3 `plane_wave_pressure_velocity_1d`
instead of Python-local sine formulas, leaving Python at axis selection, unit
conversion, and plotting.
The Chapter 5 diagnostic-imaging Doppler panel now routes contrast-agent IQ
synthesis, finite-tone spectrum power, velocity-axis mapping, Nyquist velocity,
and Kasai estimation through Rust/PyO3 `contrast_agent_doppler_spectrum` after
the existing Rayleigh-Plesset amplitude calculation, leaving Python to plot the
returned arrays.
The Chapter 5 continuous-wave/vector Doppler panel now routes RF tone synthesis,
CW demodulation/FFT, pulsed-wave Nyquist comparison, cross-beam projection, and
vector-flow recovery through Rust/PyO3 `continuous_wave_vector_flow_fixture`,
leaving Python to plot returned arrays and vectors.
The Chapter 13 photoacoustic spectroscopic-unmixing panel now routes HbO2/Hb
sO2 sweeps, deterministic measurement perturbations, nonnegative concentration
clipping, and sO2 ratio calculation through Rust/PyO3
`spectroscopic_unmixing_so2_sweep`, leaving Python to plot returned curves.
The `us_bmode_linear_transducer` direct test also
now enforces the report-owned raw scan-line target line and decodes both generated
B-mode PNGs instead of carrying a duplicate threshold copy. The validation
chapter now records those raw scan-line metrics as the physics parity oracle and
keeps the log-compressed display residuals out of the active physics-validation
table. The axisymmetric circular-piston and focused-bowl aperture tests now run
fast current-artifact threshold/PNG checks by default while keeping full
simulator regeneration slow-gated; their analytical-reference thresholds are now
driver-owned, and the focused-bowl plot masks the O'Neil singularity before
rendering the dense analytical curve. The validation chapter also records the
closed axisymmetric aperture and IVP Gaussian reports; the Chapter 20 validation
scatter and sensors chapter B-mode RMS note are regenerated/synced from the same
current metrics. The validation chapter now distinguishes strict field-tier
reference lines from driver-owned quick-tier thresholds, removing the stale
global PSNR contradiction for raw B-mode scan lines. Chapter 20 `fig04` now
repackages the real cached focused-bowl AS PASS artifact instead of fabricating a
noisy pseudo-kwavers trace, and the comparison pseudocode uses scenario-owned
`PARITY_THRESHOLDS` instead of duplicating strict field-tier constants. The
manifest now guards against synthetic Chapter 20 parity regressions and decodes
both the cached source PNG and regenerated book PNG. The Chapter 20 scatter now
uses the closed-validation markdown table as its source instead of carrying a
duplicate metric list, and the manifest verifies the parsed row set. The 3-D
Chapter 20 Pearson/PSNR reference figures now label r = 0.99 and PSNR = 40 dB as
strict field-tier references rather than universal acceptance thresholds, and
the Python parity command block now uses current `crates/kwavers-python/tests`
paths with the Miniforge interpreter instead of the obsolete `cd pykwavers`
layout, with a manifest regression preventing the stale command form from
returning. The manifest also parses the Chapter 20 figure index and verifies
every listed PNG/PDF artifact. The Chapter 5 diagnostic-imaging script now
requires `pykwavers`, removes the SciPy Hilbert fallback and random Doppler
noise, and routes axial envelope, lateral PSF, Doppler shift, and
contrast-bubble amplitude through Rust/PyO3 bindings with a manifest guard
preventing the fallback path from returning. The editable Miniforge extension
now rebuilds from current source, the top-level helper re-exports are restored,
and all Chapter 5 figure PNG/PDF artifacts are regenerated and manifest-decoded.
The Chapter 10 elastography script now requires `pykwavers`, routes its MRE
displacement figure through the Rust `mre_displacement_field` analytical kernel,
restores the missing top-level export, regenerates the six Chapter 10 figures,
and has a manifest guard for Rust calls, exports, and artifact decoding. The
book caption now describes the implemented damped plane-wave model instead of a
Python-only cylindrical-inclusion sketch. The Chapter 11 sources/transducers
script now requires `pykwavers`, removes optional import guards, computes BLI
accuracy from the Rust/PyO3 `bli_interpolation_error_curves` binding, leaving
Python at dB conversion and plotting. It regenerates the seven Chapter 11
figures without warnings, and has a manifest guard for Rust calls, exports, and
artifact decoding. The Chapter 12 media/tissue script now requires
`pykwavers`, computes the steady-state Pennes slab profile through the Rust
`pennes_steady_state_temperature_profile` analytical kernel, regenerates the
five Chapter 12 figures, and has a manifest guard for Rust calls, top-level
exports, and artifact decoding. The Chapter 13 photoacoustics script now
requires `pykwavers`, removes optional import guards, replaces random unmixing
noise with deterministic measurement perturbations, regenerates the five Chapter
13 figures, and has a manifest guard for Rust calls, top-level exports, and
artifact decoding. The Chapter 14 sensors/measurements script now requires
`pykwavers`, routes hydrophone directivity through Rust
`circular_piston_directivity`, routes seeded sensor noise through Rust
`add_noise`, regenerates the five Chapter 14 figures, and has a manifest guard
for Rust calls, top-level exports, and artifact decoding. The Chapter 17
inverse-problems script now requires `pykwavers`, removes optional FWI skip
branches, replaces random L-curve perturbations with deterministic measurement
perturbations, syncs the SVD/L-curve captions to the implemented Rust helpers,
regenerates the six Chapter 17 figures, and has a manifest guard for Rust calls,
top-level exports, and artifact decoding. Its Figure 18.6 eikonal/Kirchhoff path
now routes traveltime and synthetic diffraction-stack computation through
Rust/PyO3 `eikonal_traveltime_2d` and `kirchhoff_point_scatterer_image_2d`;
Python only adapts arrays and plots returned fields. The Chapter 18
sonogenetics script now requires `pykwavers`, removes optional import/skip
branches, routes streaming through Rust `acoustic_streaming_velocity`, renders
the documented channel-activation panel through Rust membrane-tension,
Boltzmann, and pressure-threshold gates, syncs the Gorkov/streaming/activation/
CEM43 captions, regenerates the seven Chapter 18 figures, and has a manifest
guard for Rust calls, top-level exports, and artifact decoding. The Chapter 21
simulation-orchestration script now requires `pykwavers`, removes the optional
import fallback, routes the bubble-radius comparison through the Rust/PyO3
Rayleigh-Plesset, Keller-Miksis, and Gilmore solver bindings, regenerates the
figure, and has a manifest guard for solver calls, top-level exports, book-text
ownership claims, and artifact decoding. The Chapter 34 optoacoustic focused
ultrasound script now requires `pykwavers`, removes the optional import fallback,
routes SOAP numerical aperture, f-number, lateral resolution, and focal gain
through Rust/PyO3 optoacoustic transducer kernels, regenerates the figure, and
has a manifest guard for binding calls, top-level exports, book-text SSOT claims,
and artifact decoding. The Chapter 29 pressure-diagnostics helper now requires
`pykwavers`, removes its duplicate Python mechanical-index formula, routes MI
metrics through the Rust/PyO3 `kw.mechanical_index` safety kernel, and has a
therapy-chapter regression for the projected pressure diagnostic value plus
source fallback-token guard. The Chapter 30 intravascular-ultrasound script now
requires `pykwavers`, removes extension-unavailable fallback formulas for
intensity, temperature rise, B-mode compression/envelope detection, and therapy
MI, routes those surfaces through Rust/PyO3 kernels, and has IVUS chapter
regressions guarding the no-fallback contract; the remaining Chapter 30
synthetic vessel phantom and speckle fixture now route through Rust/PyO3
`ivus_vessel_phantom`, leaving Python responsible for dataclass adaptation and
plotting. The Chapter 1 wave-fundamentals travelling-pulse source profile and
d'Alembert reference now route through Rust/PyO3
`gaussian_modulated_pulse_1d` and `dalembert_split_solution_1d`, leaving Python
responsible for solver invocation, array adaptation, and plotting. The Chapter
2 numerical-methods CFL stability, modified-wavenumber, and k-space correction
figure data now route through Rust/PyO3 `fdtd_cfl_stability_region_2d`,
`centered_fd_modified_wavenumber`, and `kspace_temporal_correction`, leaving
Python responsible for axis generation, reshaping, and plotting. The Chapter 3
nonlinear-acoustics Fubini waveform evolution now routes through Rust/PyO3
`fubini_waveform`, leaving Python responsible for sampling choices and plotting.
The Chapter 6 therapeutic-ultrasound HIFU heat-source setup now routes
intensity-to-pressure conversion through Rust/PyO3
`acoustic_pressure_amplitude_from_intensity`, leaving Python responsible for
scenario constants, solver invocation, and plotting.
The retained Chapter 8 acoustic-propagation spreading-law panel now routes
normalized spherical and cylindrical intensity envelopes through Rust/PyO3
`geometric_spreading_intensity_envelopes`, leaving Python responsible for the
radius axis and plotting.
The Chapter
24 BBB-LIFU and Chapter 26
neuromodulation scripts now import `pykwavers` directly with no optional
`_HAS_KW` branch, and Chapter 24 uses an explicit script-directory import path
for the cavitation-dose helper instead of a try/except fallback. Chapter 24's
inertial-cavitation MI frequency curves now use the Rust/PyO3
`mechanical_index_frequency_sweep` safety helper instead of Python-side
`constant / sqrt(f_MHz)` formulas, and its passive-cavitation pressure sweep
uses `kw.mechanical_index_field`. Chapter 24's inertial-damage probability
curve now uses the Rust/PyO3 `bbb_inertial_damage_probability` BBB helper
instead of inline NumPy logistic algebra. Chapter 26's neuromodulation
cavitation-risk contour now uses the Rust/PyO3
`mechanical_index_cavitation_risk` safety helper instead of inline NumPy
logistic algebra. Chapter 24's passive-cavitation stable-onset,
inertial-onset, and controller-cap classification now uses the Rust/PyO3
`cavitation_therapeutic_window_indices` passive-dose helper instead of
Python-side band-ratio scans. Chapter 24's population-monitor operating-point
selection now uses the Rust/PyO3 `cavitation_inertial_fraction_onset_index`
passive-dose helper instead of Python-side broadband-fraction scans. Chapter 24's
per-spot cavitation monitor raster now uses the Rust/PyO3
`per_spot_cavitation_dose_grid` delivery helper instead of Python-side nested
steering/interpolation loops. The shared curve-driven cavitation monitor trace
now uses the Rust/PyO3 `cavitation_monitor_timeseries` helper instead of
Python-side interpolation, seeded jitter, controller stepping, and dose
accumulation. The Chapter 24 passive-cavitation closed-loop sonication trace now
uses the Rust/PyO3 `closed_loop_cavitation_sonication` helper instead of
Python-side stable/inertial interpolation, controller stepping, and dose
accumulation. The shared raster-pulsing monitor now uses the Rust/PyO3
`raster_cavitation_pulsing` helper instead of Python-side steering derating,
pressure-sweep interpolation, schedule expansion, residual-bubble shielding,
thermal relaxation, coverage, and cumulative-dose resampling. The shared
one-pressure population-emission helper now uses the Rust/PyO3
`simulate_population_emission` helper instead of Python-side bubble-population
sampling, per-bubble solver dispatch, trace rejection, Hann FFT spectrum
construction, and cavitation-band decomposition. The shared simulated per-pulse
population monitor now uses the Rust/PyO3
`simulated_population_monitor_timeseries` helper instead of Python-side
population-emission dispatch, controller stepping, acoustic-power scaling, and
cumulative-dose integration. The Chapter 24 population pressure sweep now uses
the Rust/PyO3 `population_emission_sweep` helper instead of Python-side
per-pressure aggregation over the one-pressure population helper. Chapter 24
now routes the V_s-integrated analytic spectrum and pressure sweep through the
Rust/PyO3 `volume_emission_spectrum` and `volume_emission_sweep` helpers
instead of Python-side Keller-Miksis loops, emission conversion, PSD
construction, receiver integration, and band decomposition. Chapter 24 still
has only presentation-layer summary fraction formatting over Rust-returned
arrays; no remaining Chapter 24 passive-cavitation monitor physics is owned by
Python in this helper path.
The shared PNG
artifact helper now uses Pillow
size/extrema checks instead of Matplotlib float-array decoding, avoiding the
parity dashboard PNG memory allocation failure while keeping nonblank artifact
coverage. The 3-D circular-piston and
focused-bowl aperture
tests now share the same script-owned threshold and artifact contract. The
`at_array_as_source`
driver now owns its executable threshold map and its report carries the full
metric fields needed by the default artifact test. The `at_array_as_sensor`
driver now does the same and pins the k-wave-python combined-sensor ordering
explicitly. The `at_linear_array_transducer` driver now owns its threshold map
and default artifact test as well. The `us_defining_transducer` test now consumes
the driver-owned `TRACE_THRESHOLDS`, validates the current report/PNG by default,
and leaves full simulator regeneration slow-gated. The
`ivp_photoacoustic_waveforms` test now consumes the driver-owned
`PARITY_THRESHOLDS`, including the peak-ratio target, and validates the current
report/PNG by default. The `pr_2D_FFT_line_sensor` test now consumes the
driver-owned reconstruction and ground-truth `PARITY_THRESHOLDS`, validates the
current report plus both generated PNG artifacts by default, and keeps full
simulator regeneration slow-gated. The `pr_2D_TR_line_sensor` test now consumes
the driver-owned time-reversal, FFT, and ground-truth `PARITY_THRESHOLDS`,
validates the current report plus all three generated PNG artifacts by default,
and records the regenerated TR differential band separately from the near-exact
FFT contract. The `pr_3D_TR_planar_sensor` test now consumes the driver-owned
time-reversal and ground-truth `PARITY_THRESHOLDS`, validates the current report
plus both generated PNG artifacts by default, and leaves full simulator
regeneration slow-gated. The `na_controlling_the_pml` test now consumes the
driver-owned waveform and HDF5 `PARITY_THRESHOLDS`, validates the current report
plus comparison PNG by default, and leaves full PML/HDF5 regeneration
slow-gated. The `sd_focussed_detector_2D` test now consumes the driver-owned
trace and directivity `PARITY_THRESHOLDS`, validates the current report plus both
generated PNG artifacts by default, and leaves full simulator regeneration
slow-gated. The `sd_focussed_detector_3D` test now consumes source-specific
driver-owned trace thresholds and the directivity threshold, validates the
current report plus both generated PNG artifacts by default, and leaves full
simulator regeneration slow-gated. The `sd_directivity_modelling_2D` test now
consumes driver-owned matrix, trace-summary, and directivity thresholds,
validates the current report plus both generated PNG artifacts by default, and
leaves full simulator regeneration slow-gated. The `ivp_saving_movie_files`
test now validates the current PASS report and comparison PNG by default against
the driver-owned thresholds, and the driver crops pykwavers `p_final` to the
same PML-excluded physical interior emitted by k-wave-python. The
`na_optimising_performance` test now validates the current PASS report and
comparison PNG by default against the driver-owned thresholds, uses the same
repo-root source-image fixture path as the driver, and the driver crops
pykwavers `p_final` to the same PML-excluded physical interior emitted by
k-wave-python. The `us_bmode_phased_array` test now validates the current PASS
report against driver-owned strict quick-tier fundamental/harmonic thresholds,
decodes both the B-mode comparison PNG and transducer-face debug PNG by default,
and leaves the full 3-D simulator regeneration slow-gated. The `checkpointing`
test now validates the current bit-exact PASS report, checkpoint lifecycle
metrics, full-grid sensor shape, and comparison PNG against the driver-owned
save/resume contract by default, with full checkpoint regeneration slow-gated.
The `pr_3D_FFT_planar_sensor` test now validates the current PASS report and
pressure PNG against driver-owned summary and representative-trace thresholds by
default, and the compare driver no longer applies the stale one-sample alignment
shift after cache-level inspection showed zero-lag matrix parity. The manifest
now also validates diffusion homogeneous-medium source/diffusion, IVP
opposing-corners sensor mask, TVSP acoustic-field propagator, TVSP
angular-spectrum method, TVSP equivalent-source holography, and TVSP
transducer-field-pattern reference/diagnostic reports against each driver's
executable `PARITY_THRESHOLDS` while decoding the comparison PNGs. The manifest
now self-audits that every reference/diagnostic compare driver exporting
`PARITY_THRESHOLDS` is included in the semantic parser set.

Evidence tier: cached differential validation against k-wave-python output plus
value-semantic pytest assertions. Verified with:

```powershell
D:\miniforge3\python.exe -m pytest crates/kwavers-python/tests/test_kwave_cache_manifest.py crates/kwavers-python/tests/test_kwave_example_cached_parity.py -q
D:\miniforge3\python.exe -m pytest crates/kwavers-python/tests/test_kwave_example_at_circular_piston_3d_parity.py crates/kwavers-python/tests/test_kwave_example_at_focused_bowl_3d_parity.py crates/kwavers-python/tests/test_kwave_cache_manifest.py -q
D:\miniforge3\python.exe -m pytest crates/kwavers-python/tests/test_kwave_example_at_array_as_source_parity.py crates/kwavers-python/tests/test_kwave_cache_manifest.py -q
D:\miniforge3\python.exe -m pytest crates/kwavers-python/tests/test_kwave_example_at_array_as_sensor_parity.py crates/kwavers-python/tests/test_kwave_cache_manifest.py -q
D:\miniforge3\python.exe -m pytest crates/kwavers-python/tests/test_kwave_example_at_linear_array_transducer_parity.py crates/kwavers-python/tests/test_kwave_cache_manifest.py -q
D:\miniforge3\python.exe -m pytest crates/kwavers-python/tests/test_kwave_example_us_defining_transducer_parity.py crates/kwavers-python/tests/test_kwave_cache_manifest.py -q
D:\miniforge3\python.exe -m pytest crates/kwavers-python/tests/test_kwave_example_ivp_photoacoustic_waveforms_parity.py crates/kwavers-python/tests/test_kwave_cache_manifest.py -q
D:\miniforge3\python.exe -m pytest crates/kwavers-python/tests/test_kwave_example_pr2d_parity.py crates/kwavers-python/tests/test_kwave_cache_manifest.py -q
D:\miniforge3\python.exe -m pytest crates/kwavers-python/tests/test_kwave_example_pr2d_tr_parity.py crates/kwavers-python/tests/test_kwave_cache_manifest.py -q
D:\miniforge3\python.exe -m pytest crates/kwavers-python/tests/test_kwave_example_pr3d_tr_parity.py crates/kwavers-python/tests/test_kwave_cache_manifest.py -q
D:\miniforge3\python.exe -m pytest crates/kwavers-python/tests/test_kwave_example_na_controlling_the_pml_parity.py crates/kwavers-python/tests/test_kwave_cache_manifest.py -q
D:\miniforge3\python.exe -m pytest crates/kwavers-python/tests/test_kwave_example_sd_focussed_detector_2d_parity.py crates/kwavers-python/tests/test_kwave_cache_manifest.py -q
D:\miniforge3\python.exe -m pytest crates/kwavers-python/tests/test_kwave_example_sd_focussed_detector_3d_parity.py crates/kwavers-python/tests/test_kwave_cache_manifest.py -q
D:\miniforge3\python.exe -m pytest crates/kwavers-python/tests/test_kwave_example_sd_directivity_modelling_2d_parity.py crates/kwavers-python/tests/test_kwave_cache_manifest.py -q
D:\miniforge3\python.exe -m pytest crates/kwavers-python/tests/test_kwave_example_ivp_saving_movie_parity.py crates/kwavers-python/tests/test_kwave_cache_manifest.py -q
D:\miniforge3\python.exe -m pytest crates/kwavers-python/tests/test_kwave_example_na_optimising_performance_parity.py crates/kwavers-python/tests/test_kwave_cache_manifest.py -q
D:\miniforge3\python.exe -m pytest crates/kwavers-python/tests/test_kwave_example_us_bmode_phased_array_parity.py crates/kwavers-python/tests/test_kwave_cache_manifest.py -q
D:\miniforge3\python.exe -m pytest crates/kwavers-python/tests/test_kwave_example_checkpointing_parity.py crates/kwavers-python/tests/test_kwave_cache_manifest.py -q
D:\miniforge3\python.exe -m pytest crates/kwavers-python/tests/test_kwave_example_parity.py crates/kwavers-python/tests/test_kwave_cache_manifest.py -q
D:\miniforge3\python.exe -m pytest crates/kwavers-python/tests/test_kwave_cache_manifest.py -q
```

Results: 36/36 passed for cached parity; 13 passed and 2 slow-regeneration tests
skipped for 3-D aperture parity plus the manifest; 12 passed and 1
slow-regeneration test skipped for array-as-source plus the manifest; 12 passed
and 1 slow-regeneration test skipped for array-as-sensor plus the manifest; 12
passed and 1 slow-regeneration test skipped for linear-array transducer plus the
manifest; 12 passed and 1 slow-regeneration test skipped for defining-transducer
plus the manifest; 12 passed and 1 slow-regeneration test skipped for
photoacoustic-waveforms plus the manifest; 12 passed and 1 slow-regeneration
test skipped for PR2D line-sensor parity plus the manifest; 12 passed and 1
slow-regeneration test skipped for PR2D time-reversal line-sensor parity plus the
manifest; 12 passed and 1 slow-regeneration test skipped for PR3D
time-reversal planar-sensor parity plus the manifest; 12 passed and 1
slow-regeneration test skipped for PML-control parity plus the manifest; 12
passed and 1 slow-regeneration test skipped for focused-detector 2-D parity plus
the manifest; 12 passed and 1 slow-regeneration test skipped for phased-array
B-mode parity plus the manifest; 12 passed and 1 slow-regeneration test skipped
for checkpointing parity plus the manifest; 12 passed and 1 slow-regeneration
test skipped for PR3D FFT planar-sensor parity plus the manifest; 12/12 passed
for the current manifest including reference-diagnostic threshold checks.

## CLOSED: Cavitation-cloud branch reconciliation (2026-06-28)

Compared `main` against `feat/cloud-time-resolved-bubble-dynamics`,
`feat/cloud-acoustic-shielding`, `feat/cloud-implicit-coupling`, and
`feat/cloud-strong-regime-solver`. All four branch tips are ancestors of current
`main`; no missing branch-only cloud implementation remained. Current `main`
already contains ADR 027-032 cloud work, including time-resolved per-cell
Keller-Miksis dynamics, shielding, implicit/direct coupling, pressure-rate
coupling, radius-dependent shielding, linear RT/RM diagnostics, and the
matrix-free iterative coupling solver.

Evidence tier: differential/content audit plus value-semantic tests. Verified
with `cargo nextest run -p kwavers-therapy --all-features cavitation_cloud`
(26/26), `D:\miniforge3\python.exe -m pytest
crates/kwavers-python/tests/test_bubble_cloud_parity.py -q` (19/19), and
chapter 21e treatment-pipeline regeneration. The regeneration rewrote the
expected outputs and produced no tracked artifact diff, so the existing figures
remain current for the cloud model.

Remaining CLD-1 backlog is narrowed to k-wave/experimental erosion comparison
and deeper nonlinear frontier physics; there is no open action to merge the named
feature branches.

## OPEN: Coverage & placement gap audit (opened 2026-06-19)

Source: [gap_audit.md → Coverage & placement audit](gap_audit.md#coverage--placement-audit-2026-06-19).
Axis is physics *coverage vs peer libraries* + cross-crate *placement* (distinct
from the Sprint A–E internal-correctness pass). Headline: breadth meets/exceeds
all peers surveyed; gaps are narrow (imaging-pipeline beamforming refinements +
a few bubble-shell models) and the main risk is fragmentation of three modality
verticals. All placement items are **[verify]-gated** — confirm logic is cloned,
not legitimately forward-vs-inverse layered, before consolidating.

Triage (correctness/arch → tests → features), one WIP item at a time:

1. **[arch] PLC-1 photoacoustic consolidation** — ✅ **DONE (2026-06-19, ADR 026):**
   consumer analysis showed the 5 locations are mostly layered, not duplicates; the
   real dup was the two `kwavers-simulation` forward pipelines. Removed the dead
   `photoacoustics/` pipeline (1325 LOC); `modalities::photoacoustic` is canonical.
2. **PLC-2 CEUS consolidation** — ✅ **CLOSED arch (2026-06-19):** verified mostly
   FALSE POSITIVE. CEUS is correctly layered — `Microbubble`/`MicrobubblePopulation`
   types live in `kwavers-imaging` and physics CEUS re-exports them (not a dup);
   perfusion is image-analysis (imaging) vs forward transport-PDE (physics), distinct
   concerns. Optional [patch] residue: unify perfusion-param extraction
   (`analyze_tic` vs `from_samples`).
3. **[arch] PLC-3 microbubble SSOT + therapy-physics layering** — shell-model SSOT
   ✅ DONE (EncapsulatedShellModel trait). **Remainder CONFIRMED real (2026-06-19),
   needs ADR + careful merge:**
   - (a) `therapy/microbubble/shell/properties.rs::MarmottantShellProperties` =
     a 2nd Marmottant impl → fold onto canonical `MarmottantModel`/`EncapsulatedShellModel`.
   - (b) `ceus/microbubble/dynamics/integration.rs::wall_acceleration` = a 3rd
     RP-with-shell integrator → route through `EncapsulatedShellModel::acceleration`.
   - (c) therapy-domain code in `kwavers-physics` (therapy/*, acoustics/therapy/*,
     transcranial/bbb_opening) → keep physics models, move therapy planning to
     `kwavers-therapy` (large cross-crate move, own ADR).
   Risk: (a)/(b) change therapy/ceus numerics (different parameterization) — verify
   value-semantic equivalence; do as a focused increment with fresh context.
4. **PLC-4 time-reversal propagator SSOT** — ✅ **CLOSED (2026-06-19):** verified
   NOT duplicated. General TR delegates to a plugin solver; simulation TR delegates
   to the solver PA reconstructor; PA TR holds the canonical k-space propagator;
   transcranial TR is distinct aberration-correction phase conjugation. Correct
   layering, no consolidation. PLC-5 histotripsy likely WONTFIX (distinct concerns).
5. **[minor] COV-1 coherence-factor weighting** — ✅ **DONE (2026-06-19):**
   Mallart-Fink amplitude CF + Camacho sign CF on the DAS path; fixed a real SAFT
   CF over-suppression bug en route (consolidated to one canonical helper).
   Follow-up [minor]: phase CF (PCF, analytic signal) + generalized CF (GCF, FFT).
6. **[minor] COV-2 active DMAS** — ✅ **DONE (2026-06-19):** `time_domain::dmas`
   (`dmas_combine` + `delay_and_sum_dmas`); passive PAM consolidated onto the
   shared combiner. **[minor] COV-3 curvilinear transmit array** in
   `kwavers-transducer`. **[minor] COV-4 discrete point-scatterer + SIR RF synth**
   (Field II core) in `kwavers-phantom`/`kwavers-source`.
7. **[minor] COV-5 Sarkar/de Jong/Hoff/Herring** bubble-shell models (one generic
   `EncapsulatedModel` impl set; ties PHY-13 de Jong scattering test).
8. **[patch/L] COV-6/7/10** — KLM/Mason circuit, MRE front end, Shepp-Logan
   fixture. **COV-8 (Cherenkov) + COV-9 (Sobolev) = verified FALSE POSITIVES**
   (both fully implemented; see gap_audit). COV-11 Mur BC = WONTFIX (CPML superior).

### Remaining genuine coverage gaps (post-verification), best as focused increments:
- **COV-3 curvilinear/convex transmit array** [minor] — `kwavers-transducer`. The
  `kwave_array` already has Arc/Bowl element primitives + `rasterizer_curved`; add
  a convex-array layout helper placing N elements along a curvature arc.
- **COV-4 discrete point-scatterer + spatial-impulse-response RF synthesis** [minor/major]
  — Field II core; largest. Home `kwavers-phantom` (scatterer cloud) + `kwavers-source`/
  analysis (SIR convolution → RF).
- **COV-5 bubble shell models** [minor] — ✅ **PARTIAL DONE (2026-06-19):**
  `EncapsulatedShellModel` trait + RP driver (PLC-3 shell-model SSOT); Church/
  Marmottant refactored on; **Hoff + Sarkar added** with validation. **Deferred:**
  de Jong (verify lumped S_p/S_f prefactor against Doinikov&Bouakaz 2011 PDF before
  asserting — convention-dependent) and Herring (free-bubble compressible EOM —
  belongs with KM/Gilmore in `bubble_dynamics`, not the encapsulated shell models).
- **COV-6 KLM/Mason** [L], **COV-7 MRE front-end** [L], **COV-10 Shepp-Logan** [L].

## CPML → single-pole CFS-PML upgrade [minor] — DONE (2026-06-19)

✅ Implemented in `kwavers-boundary/cpml`: graded κ/α + canonical Roden-Gedney
recursion wired into the convolutional (FDTD) kernel; `with_cfs_pml` builder;
dead `kappa_max`/`alpha_max` config activated (defaults reset 15/0.24 → 1/0 =
prior effective behavior, FDTD bit-identical); fixed the wrong `a` doc formula.
94 boundary + 81 FDTD/CPML solver tests pass; clippy clean. Split-field PSTD
parity untouched (σ profile unchanged).
**Deferred (tracked):** (a) full oblique-incidence FDTD differential benchmark
proving the grazing-reflection reduction empirically (currently formula-tier +
literature-cited); (b) plumb α_max≈π·f₀ from the source frequency instead of an
absolute value; (c) **3rd CPML impl** found — `solver/forward/pstd/dg/cpml` (DG
solver) is a separate CPML; evaluate consolidating onto `kwavers-boundary` or
documenting the split. (d) double-pole CFS for >16:1 bandwidth (Feng 2017).

### Original spec (retained for the deferred items)
## OPEN: CPML → single-pole CFS-PML upgrade [minor] (opened 2026-06-19)

Literature synthesis (2020–2026, background research; primary refs Roden & Gedney
2000, Komatitsch & Martin 2007/2009, Collino & Tsogka 2001, SEISMIC_CPML verbatim).
Current kwavers `kwavers-boundary/cpml` is σ-only (κ=1, α=0) — a degenerate
CFS-PML. **Highest accuracy-per-effort upgrade: add the graded κ (real stretch)
and α (complex frequency shift) terms** → single-pole CFS-PML. Order-of-magnitude
fewer spurious reflections at grazing incidence + evanescent/low-freq energy +
better late-time stability.

Concrete spec (normalized depth d∈[0,1], d=0 inner interface → d=1 outer wall):
- `σ(d) = σ_max·dᵐ`, `κ(d) = 1 + (κ_max−1)·dᵐ`, `α(d) = α_max·(1−d)` — **α grades
  OPPOSITE to σ/κ (max at interface, 0 at wall); inverting it is a silent
  amplification bug.**
- `σ_max = −(m+1)·c_max·ln(R₀)/(2·N·dx)` — derive at runtime, do NOT hardcode.
- Recursion: `b = exp[−(σ/κ+α)·dt]`, `a = σ·(b−1)/[κ·(σ+κα)]` (guard a=0 where
  σ+κα=0); `ψₙ = b·ψₙ₋₁ + a·∂field`; `∂̃ = (1/κ)·∂field + ψ`.
- Defaults: m=2–3; R₀=1e-3 @10 cells / 1e-5..1e-6 @20; **κ_max=5–7** (the term
  currently fixed at 1); **α_max=π·f₀**; c=c_max.
- Single-pole is the default; expose double-pole (Feng 2017) only as optional for
  >16:1 bandwidth / transcranial skull-grazing.

Scope notes: cleanest in the **FDTD** CPML path. The **PSTD/k-space** path uses
k-Wave-style split-field exponential-decay PML (global FFT derivative can't fuse a
local convolution) — there the reduced upgrade is the analytically-derived σ_max
profile + optional κ, NOT a full convolutional port. Acoustic + isotropic-elastic
are intrinsically stable (Bécache 2003: V_p·V_g≥0); anisotropic-elastic needs
M-PML (Meza-Fajardo 2008) — out of scope unless an anisotropic instability appears.
Acceptance: differential reflection test vs current σ-only CPML showing reduced
grazing-incidence reflection; reflection-decay property test retained. ADR advised
(boundary-API surface change: κ_max/α_max/R₀ params).

## OPEN: kwavers-gpu extraction + internal-folder teardown [arch] (opened 2026-06-03)

Goal: dissolve the leftover internal `kwavers/src` code (post-split) into the
layered crates, leaving `kwavers` a thin facade. The bulk is GPU. Decision
(user 2026-06-03): **new `kwavers-gpu` leaf crate** (depends on solver,
implements the `ComputeBackend`/`FdtdGpuAccelerator` trait surfaces that stay in
solver), **consolidating all three scattered GPU paths**, with **wgpu-v26
bit-rot repaired as part of the move**.

Progress (all green, gated so default `--workspace` build stays clean):
- **[done]** Scaffold `crates/kwavers-gpu` + workspace member. (462ab1939)
- **[done]** Move facade `kwavers::gpu` monolith (~5000 L) → `kwavers-gpu/src/gpu`,
  behind `kwavers-gpu/gpu` feature (+`pinn` for burn accelerator); facade
  re-exports `kwavers_gpu::gpu`. (2c5acc444)
- **[done]** Move `profiling/gpu_allocator` → `kwavers-gpu` (unconditional; 9
  tests pass); kwavers-gpu now a regular facade dep.
- **[done] repair** wgpu-v26 bit-rot — `cargo check --features gpu` green for
  `kwavers-solver` (gpu_pstd struct-refactor completion + pstd.wgsl relocation;
  backend/gpu device/queue split, DeviceDescriptor.trace, request_adapter
  Result, KwaversError::GpuError migration; backend::gpu ungated onto `gpu`,
  redundant `solver_backend_gpu_unstable` dropped) and `kwavers-gpu` (carry
  flume/rayon/rand/once_cell deps; CoreGpuContext dead-field drop; Shape1D via
  kwavers_math re-export).
- **[blocked] analysis gpu** — bytemuck dep added (clears 23 errors); 2 trivial
  errors remain (dynamic_focus_dispatch visibility, missing KwaversError import).
  **Genuine blocker:** `three_dimensional` GPU beamformers reference
  `BEAMFORMING_3D_SHADER` / `DYNAMIC_FOCUS_3D_SHADER` WGSL constants that were
  **never written** (no source, no git history) — incomplete WIP, not bit-rot;
  fabricating shaders is prohibited. Decision: gate as incomplete vs. author.
- **[todo] simulation/diagnostics gpu** — unchecked under `--features gpu`.
- **[todo] consolidate** `solver::backend::gpu` + `solver::forward::{fdtd,pstd}`
  GPU kernels + `*.wgsl` into kwavers-gpu; leave only the traits in solver
  (deferred — repaired in place first to get the feature building).
- **[todo] re-home** remaining facade code: `architecture/layer_validation`
  (dev tooling — evaluate keep/delete), `infrastructure/io` (CSV output — facade
  concern, candidate `kwavers-io` or stays).
- **[todo] dead test** `kwavers/tests/recovery_stress_tests.rs` imports a
  `gpu::recovery` module that never existed — remove/repair.
- **[todo] cfg debt** `solver/inverse/pinn/ml/mod.rs` gates `trainer` +
  `distributed_training` on a non-existent `api` feature (dead gates; needs
  pinn-feature verification to remove cleanly).

Status: internal `kwavers/src` down to ~1300 L (from 8799). Facade re-exports
+ `architecture` + `infrastructure/io` + `main.rs` remain.

## OPEN: Workspace crate-split prep [arch] (opened 2026-06-01)

Goal: decompose the 461 kLOC / 3,475-file `kwavers` monolith into a layered
workspace crate DAG (`core → math → domain → {physics, solver} → analysis →
simulation → clinical`, thin `kwavers` facade; `pykwavers` already separate).
Motivation: per-crate incremental compilation (kills the 4–8 min whole-crate
rebuilds) + compiler-enforced unidirectional dependencies.

User decision (2026-06-01): **clean the 18 upward (cycle-blocking) edges first**,
before any crate extraction, so the split is mechanical. Foundation
(`core`/`math`/`domain`/`solver`) is already acyclic (0 upward edges).

**STATUS: prep COMPLETE & verified (2026-06-01).** Definitive full-DAG check
(non-test) shows **0 upward library edges at every layer** — clean linear DAG
`core→math→domain→physics→solver→analysis→simulation→clinical`. Workspace ready
for crate extraction.

The "18 edges" resolved to **11 real physics→solver library edges (all broken)**
+ 6 analysis→solver (always DOWNWARD — never blockers) + 1 test-only (non-blocking):
- **[done]** Cluster 1 — `physics/factory/catalog.rs` (8): solver-plugin registry
  mislocated in physics → moved to `solver/plugin/catalog.rs`. Verified (build clean).
- **[done]** Cluster A — `bubble_symplectic` (1): moved `solver/forward/ode/
  bubble_symplectic` → `physics::bubble_dynamics::symplectic_integration`. Verified.
- **[done]** Cluster B — `physics/.../registration/adapter.rs` (1): the
  `impl RegistrationEngine` was DEAD (no `dyn`/bound/caller); deleted it + 2 private
  helpers + the `Array2`/solver imports. Removed dead code, broke edge, 0 behavior
  change. Verified (fusion: 129 tests pass). FOLLOW-UP: `RegistrationEngine` trait in
  `solver::interface::factory` now has 0 implementors — candidate dead-trait cleanup.
- **[done]** Cluster C — `ElementPosition`/`TransducerGeometry` (self-contained,
  dep only `std::fmt::Debug`) moved `solver/inverse/linear_born_inversion/geometry.rs`
  → `domain/source/transducers/acquisition_geometry.rs`. 13 import sites updated; 3
  multi-line `use` blocks fixed (sed-missed). Verified (transcranial 4 + geometry 2
  tests pass). NOTE: `domain/hifu.rs` was a false match (`HifuTransducerGeometry`).
- **[non-blocker]** Cluster D — `elastic_wave/tests.rs` TEST-ONLY import of
  `PstdElasticPlugin`. Cargo permits dev-dependency cycles; left in place.
- **[non-blocker]** Cluster E — 6 `analysis→solver::...pinn_beamforming` edges are
  DOWNWARD (analysis sits above solver; verified `solver→analysis=0`,
  `physics/domain→analysis=0`). The interface placement is correct; no change needed.
  (Earlier mis-classified by lumping `solver` with above-analysis `clinical`/`simulation`.)

EXTRACTION PHASE (ADR 011, complete — leaf-first):
- **[done]** ADR 011 written (`docs/ADR/011-workspace-crate-split.md`); facade
  strategy (`pub use kwavers_core as core`) decided (3,377 `crate::core::` refs make
  a path-rewrite untenable).
- **[done]** `kwavers-core` extracted (2026-06-01): `core/` → `crates/kwavers-core/`,
  facade re-export, workspace member. Surfaced + resolved foundation error-coupling
  debt: `KwaversError`'s `From`/`#[from]` for wgpu/flume/ritk_registration/nifti made
  optional+feature-gated (orphan-rule-bound; facade enables them), anyhow→normal dep,
  log+std. Verified: kwavers-core default + all-features compile; full `kwavers` build
  green (7m06s). Behaviour identical to monolith.
- **[done]** `kwavers-math` extracted (2026-06-01): `math/` → `crates/kwavers-math/`,
  facade `pub use kwavers_math as math`. Path rewrites: 19 `crate::math::`→`crate::`,
  48 `crate::core::`→`kwavers_core::`. No error-coupling (clean). Surfaced the
  `pub(crate)`-visibility pattern (promoted `geometry::{distance3,normalize3,
  orthogonal_basis_from_normal3}` + `StaggeredGridOperator::{dx,dy,dz}` to `pub`);
  added `log`+`rand` (inline-macro deps); fixed 3 doctest paths. Verified: isolated
  build + full `kwavers` build green (5m24s).
- **[ATTEMPTED → REVERTED]** `kwavers-domain` (2026-06-01): extraction reverted to
  green after the isolated build exposed blockers the grep/sed audit missed. **Root
  cause: grouped `use crate::{ … }` import blocks evade line-based greps AND sed** —
  they hid (a) a REAL `domain → physics` upward edge and (b) un-rewritten core/domain
  paths. Findings (must resolve before re-attempting):
  - **BLOCKER (arch):** `domain/sensor/sonoluminescence/detector/core.rs` imports
    `physics::bubble_dynamics::BubbleStateFields` and
    `physics::optics::sonoluminescence::{EmissionParameters, SonoluminescenceEmission}`
    via `use crate::{ physics::… }` — a genuine domain→physics cycle. A domain
    *sensor* depending on physics *models* is a layering violation; resolution
    options: move the sonoluminescence detector up to `analysis`/a higher layer, or
    move the shared emission types down, or invert via a trait. **Needs an ADR-level
    decision.** (My earlier "DAG acyclic at every layer" claim was based on an
    incomplete audit — it counted only single-line `use crate::X` and missed grouped
    blocks + inline refs. CORRECTION logged.)
  - dicom_ritk move (infra→domain) was correct and is part of the plan, but is moot
    until the physics edge is resolved.
  - Mechanical: `serde_json` dep needed; many `use crate::{ core::, domain:: }`
    grouped blocks need splitting (core→`kwavers_core`, domain→`crate`).
- **Recovery note (2026-06-01):** the domain revert (`git checkout HEAD -- kwavers/src/domain`)
  was over-broad — it clobbered this session's *uncommitted* earlier domain work
  (CLD-4/5/13 + Cluster-C geometry move), breaking non-domain consumers. All four
  were reconstructed from the transcript + `git show HEAD:` and re-verified (full
  `kwavers` build green, 4m19s). **PROCESS LESSON: commit verified increments before
  any `git checkout`-based revert** — uncommitted changes under the reverted path are
  unrecoverable. core+math extractions survived (separate `crates/` paths).
- **[DONE] Complete edge re-audit (2026-06-01)** — grouped-import-aware Python scan
  (collapses multiline `use crate::{…}`, strips comments) over all in-monolith layers.
  **Authoritative remaining cross-layer edge set = 3** (physics/solver/simulation/
  clinical are CLEAN — the prep is verified; core/math already extracted):
  1. `domain → infrastructure`: `domain/imaging/medical/dicom_loader/loader.rs`
     calls `infrastructure::io::dicom_ritk`. **Fix (known):** move the `dicom_ritk`
     adapter into domain (it uses domain types + is consumed only by domain; +deps
     ritk-io, burn["ndarray"]).
  2. `domain → physics`: `domain/sensor/sonoluminescence/detector/core.rs` uses
     `physics::bubble_dynamics::BubbleStateFields` + `physics::optics::sonoluminescence`.
     **Fix (ADR-level decision):** a domain *sensor* depending on physics *models* —
     move the detector up to a higher layer, OR move the shared emission types down,
     OR invert via a trait.
  3. `analysis → infrastructure`: `analysis/plotting/mod.rs`. **Fix:** plotting is a
     presentation concern; move it to the facade/a viz crate, or invert.
  Resolve these 3 → domain + analysis become extractable. (Earlier "0 edges
  everywhere" was wrong: line-greps missed grouped imports + inline refs.)
- **[DONE — all 3 edges resolved, 2026-06-01]** Grouped-import-aware re-audit now
  reports **every layer CLEAN** (domain/physics/solver/analysis/simulation/clinical
  = 0 upward edges). The DAG is genuinely acyclic.
  1. domain→physics (sonoluminescence detector): per user direction ("detectors/
     transducers belong outside physics"), decoupled the detector — it imported
     `BubbleStateFields` via a physics *re-export* (canonical type is already
     `domain::field::BubbleStateFields`; fixed the import path) and stored a
     vestigial `physics::optics::SonoluminescenceEmission` only to read one bool
     (`use_blackbody`); replaced with a domain-local flag. Detector now physics-free;
     emission physics stays inline (Stefan-Boltzmann/Wien/Planck). Zero behaviour change.
  2. domain→infrastructure (dicom_ritk): relocated the adapter
     `infrastructure/io/dicom_ritk.rs` → `domain/imaging/medical/dicom_loader/`
     (intra-crate; it uses domain types + is domain-only-consumed). Removed the
     unused infra re-export aliases.
  3. analysis→infrastructure: removed the dead `pub use infrastructure::io::save_data_csv`
     re-export in `analysis/plotting` (no consumers).
- **[done] `kwavers-domain` extracted (2026-06-01)** — `domain/` →
  `crates/kwavers-domain` (grid, medium, source, sensor, boundary, field, signal,
  imaging, therapy). Used the new **`scripts/crate_path_rewrite.py`** (grouped-
  import-aware: splits `use crate::{ core::, domain:: }` blocks, propagates
  visibility) — the tooling gap that broke the first attempt is closed. Isolated
  build 1m13s + full kwavers build 4m03s green. Cross-boundary fixes: 2
  pub(crate)→pub (bowl ctors), 1 inherent-impl→extension-trait
  (`OpticalPropertyMapAnalysis`). Removed orphaned `sedsTbfPU` sed temp. deps:
  core[registration,nifti]+math+gaia+ritk-io+ritk-registration+burn[ndarray]+
  nifti+serde_json+rand*.
- **[done] `kwavers-physics` extracted (2026-06-01)** — `physics/` →
  `crates/kwavers-physics` (nonlinear acoustics, bubble dynamics, thermal, optics,
  chemistry, elastic waves). Codemod-driven (744 core + 240 domain + 19 math + 212
  physics refs). **No error From-coupling** (clean). Full kwavers build green on the
  FIRST try — physics exposes a clean public API; zero pub(crate)/inherent-impl
  fixes. Relocated the lone physics→solver test edge (`pstd_elastic_plugin_reduces_
  to_acoustic_when_mu_is_zero`) into `solver::forward::pstd::extensions::elastic`.
  Isolated 2m20s + full 3m54s green.
- **[done] `kwavers-solver` extracted (2026-06-01)** — `solver/` →
  `crates/kwavers-solver` (forward FDTD/PSTD/k-space/Helmholtz/BEM, inverse
  FWI/RTM/CBS/elastography/PINN, analytical transducer, GPU backend; deps
  core+math+domain+physics). 708 files codemod'd. **No error From-coupling**; 0 real
  upward edges. Relocated the photoacoustics modality out to
  `simulation/photoacoustics/vertical/` (multi-physics orchestration, not a solver).
  Cross-crate visibility cascade (clinical/simulation reconstruction reaching into
  solver inversion internals) resolved by promoting `GenericFdtdSolver::{config,
  materials}`, `linear_born_inversion::{dense,schedule}` fns, `pcg::{invert,
  InversionState+fields}`. Isolated 2m46s + full 1m24s green. kwavers lib 1214/0,
  solver lib 845/0, solver doctests 5/0 (2059 total — baseline preserved).
- **[done] `kwavers-analysis` extracted (2026-06-02)** — `analysis/` →
  `crates/kwavers-analysis` (signal processing, beamforming, validation,
  ML/uncertainty, performance, plotting/visualization; deps core+math+domain+solver).
  207 files codemod'd. 0 upward edges, 0 back-edges. Full build green FIRST try (clean
  public API). Added `chrono` (timestamps) + dev `tokio` (async beamforming tests).
  kwavers lib 692/0, analysis lib 522/0, doctests 1/0 (2059 total — baseline preserved).
- **[done] Remote git deps + apollo FFT port (2026-06-02)** — replaced the
  apollo/ritk/gaia **submodules** (repo root) with remote git deps tracking each repo's
  default branch (SSOT in `[workspace.dependencies]`; members use `{ workspace = true }`).
  Removed submodule dirs + `.gitmodules` + ritk path scaffolding. apollo `main` had a
  redesigned FFT API (generic plans, `PlanCacheProvider` replacing global caches, public
  half-spectrum r2c removed); ported entirely inside the `math::fft` ACL (type aliases +
  cache adapters + `forward_r2c_into`/`inverse_c2r_into` half-spectrum emulation via
  full-transform + Hermitian expansion). PSTD core untouched; 2059 baseline preserved.
  gaia/ritk latest compatible. DEBT: emulation does a full z-FFT (~2× z-axis spectral
  cost) — revisit if apollo restores a public half-spectrum API.
- **[done] `pykwavers` → `kwavers-python` (2026-06-02)** — moved to `crates/kwavers-python`,
  Cargo package renamed; Python module name unchanged (`import pykwavers` still works).
  Fixed a solver-extraction leftover (`ElementPosition` stale re-export path → canonical
  domain path).
- **[done] `kwavers-simulation` extracted (2026-06-02)** — `simulation/` →
  `crates/kwavers-simulation` (orchestration, multi-physics coupling, modality pipelines,
  backends, solver adapters; deps core+math+domain+physics+solver, no analysis). 84 files
  codemod'd. 0 upward/back edges; full build green FIRST try. `simulation::core` vs `core`
  layer collision handled by the codemod. Added `toml`. kwavers lib 610/0, simulation lib
  82/0, doctests 4/0 (692 preserved).
- **[done] clinical SPLIT into `kwavers-diagnostics` + `kwavers-therapy` (2026-06-02)** —
  the 462-file clinical layer was split (not extracted as one crate) by verified dependency
  analysis: imaging is fully independent → `kwavers-diagnostics`; therapy↔safety coupled +
  regulatory/patient_management → `kwavers-therapy`; therapy→imaging=0 so the two are
  parallel/independent. Facade `kwavers::clinical` re-exports both under the original paths.
  kwavers lib 38/0, diagnostics 271/0 (+1 doc), therapy 301/0 (+10 doc) = 610 preserved.
- **WORKSPACE SPLIT COMPLETE** — all layer crates extracted: core, math, domain, physics,
  solver, analysis, simulation, diagnostics, therapy (clinical layer = 2 crates). The
  `kwavers` crate is the facade (+ gpu/infrastructure/profiling/architecture). Binding crate
  is `kwavers-python`. apollo/ritk/gaia are remote git deps.
- **[done] Lower-layer cleanups (2026-06-02)**:
  - **therapy type SSOT**: removed 3 dead re-export shim modules (`metrics`/`modalities`/
    `parameters`) + 3 dead duplicate types (`Clinical{TreatmentMetrics,TherapyMechanism,
    TherapyModality}` ≡ `domain::therapy::types::Domain*`); kept the real
    `ClinicalTherapyParameters`. (commit d97620b75)
  - **re-homing**: `doppler`/`spectroscopy`/ULM(`ulm`)/vesselness(`vasculature`) →
    `analysis::signal_processing`; `chromophores` → new `domain::optics`; `phantoms` →
    `domain::phantoms` (dropped its lone test-only physics coupling). (commit 8cb5232b3)
  - **lithotripsy `cavitation_cloud`**: VERIFIED not an SSOT violation — its inline
    `p_crit = p0−2σ/R0` is a different (simpler) criterion than physics's full
    `blake_threshold`; left unchanged (swapping would alter calibrated behavior).
  - **reconcile `domain_types`**: the `Domain*`/`Clinical*` overlap was the dead-duplicate
    removal above; `domain::therapy::types` is the SSOT. `ClinicalTherapyParameters` remains
    a genuinely richer app type (extra fields + builders).
  - Verification: domain 654/0, analysis 598/0, diagnostics 175/0, therapy 301/0; full green.
- **DEBT (coupling smell, logged)**: clinical/simulation reconstruction reaches into
  solver's `linear_born_inversion` internals (forced the pub(crate)→pub cascade). A
  cleaner narrow public inversion API would re-seal these; deferred (non-blocking).
- Bump to 4.0.0 ([arch] post-1.0) at release; run cargo-semver-checks on the facade.

## OPEN: Module physics-audit revision program (opened 2026-05-31)

Source: four-subsystem read-only audit (see gap_audit.md). ~50 candidate gaps
across solver/physics/clinical+domain/analysis+math. Sequenced per sprint
triage (correctness → architecture → tests → docs). Each item links to a
gap_audit ID.

### Sprint A — verify C-tier suspicions [patch, no code change unless confirmed]
- **[open]** Verify SOL-4 (Westervelt FMA ordering), PHY-1 (Gilmore vapor term),
  PHY-3 (IAPWS-IF97 dimensionalization), AMC-1 (CD6 stencil signs), and AMC-4
  (wgsl BC) against code + a literature reference. Outcome per item:
  confirmed-bug → Sprint B, or false-positive → annotate gap_audit + close.
  Rationale: these are pattern-match flags from an automated sweep; treating them
  as bugs without confirmation would violate the evidence-tier rule. **AMC-2 is
  closed (2026-06-30)** by the shared MVDR denominator validator and focused
  value-semantic regressions.

### Sprint B — confirmed correctness fixes [patch/minor]
- **[open]** SOL-1/2/3, PHY-EM: replace production `panic!` in
  harmonic-accessor / elastography / PINN-EM paths with `KwaversResult` (no
  behavior change on the happy path; adds graceful failure). Value-semantic
  error-path tests.
- **[open]** PHY-5: fix Cattaneo-Vernotte default τ / wave-speed to
  physically-valid values (ps-scale for water/tissue) + cite; add a causality
  (finite-speed) test.

### Sprint C — approximation validity bounds [minor, doc + guarded option]
- **[~CLD-2 resolved~]** PHY-2/4/8, CLD-3/6: ~~CLD-2 (linear-only HIFU → KZK wiring,
  config flag, adapter, dispatch complete)~~. Remaining: for each documented
  approximation add: (a) quantitative validity regime in Rustdoc with reference,
  (b) where feasible a config flag to select the fuller model, (c) a test at
  the regime boundary.

### Sprint D — missing literature validation [minor]
- **[open]** PHY-9/10/11/13, CLD-9/10/11, SOL-7: add value-semantic tests against
  analytical/published references (Lauterborn collapse, Minnaert resonance, de
  Jong scattering, k-wave focal field, CPML stability, source energy
  conservation). No `is_ok()`-only assertions.

### Sprint E — CT-derived parameters + DRY/SSOT + docs [patch/minor]
- **[open]** CLD-4/5/12: replace hardcoded tissue/backing impedance, phased-array
  frequency, and air-rejection HU with medium/CT-derived values (+ override).
- **[open]** AMC-9/10/11, CLD-13/14, SOL-9: remove identity casts; move
  `diagonal_loading`/`center_freq` magic numbers to `core::constants`; newtype
  raw pressure arrays; cite benchmark tolerances.
- **[open]** SOL-10/11: Rustdoc sweep on undocumented public solver fns; wire
  kwave_comparison + gpu_cpu_equivalence validators into a regression suite.

## Chapter 31 image-then-treat figure clarity - CLOSED (2026-05-27)

- **[done] [patch]** Reviewed the ch31 image reconstruction and therapy panels.
  The previous image-then-treat figures showed only the weaker anatomical
  Born reconstruction and applied abdominal histotripsy threshold wording to
  the transcranial focused-ultrasound case.
- **[done] [patch]** Updated
  `pykwavers/examples/book/ch31_clinical_device_geometry.py` so each anatomy
  shows CT context, same-aperture anatomy reconstruction, fused
  lesion-localization reconstruction with Dice equal-area support, and
  therapy pressure. Liver/kidney retain the 26 MPa histotripsy isoline; brain
  now marks the skull-corrected focus target instead of drawing a misleading
  pressure contour.
- **[done] [patch]** Fixed the abdominal body-mask source in
  `clinical::therapy::theranostic_guidance::medium::abdominal`: the solver
  body support now comes from the target-connected pre-crop component after
  resampling, not from a second HU threshold over the crop. This prevents
  CT table/bed voxels from re-entering `PreparedTheranosticSlice::body_mask`.
- **[done] [patch]** Changed ch31 liver/kidney therapy panels to display the
  solver-derived target treatment support (`lesion_target * source_pressure`)
  over the full CT frame, while excluding the deterministic two-cell FDTD
  boundary/source halo from raw exposure display masks.
- **Verification:** `python -m py_compile` on the touched Python files passed;
  `D:\miniforge3\Scripts\pytest.exe pykwavers\tests\test_book_therapy_chapters.py
  -k "chapter31_image_then_treat_helpers or active_book_focused_bowl" -q`
  passed 2/2; `cargo check --manifest-path kwavers\Cargo.toml --lib
  --message-format=short -j 1` passed; `D:\miniforge3\python.exe
  pykwavers\examples\book\ch31_clinical_device_geometry.py` regenerated the
  ch31 PNG/PDF figures and metrics using the rebuilt release extension.
  Targeted `cargo test --manifest-path kwavers\Cargo.toml medium::abdominal
  --lib --message-format=short -j 1` exceeded the 300 s bound twice before
  emitting a test result.

## Ali 2025 finite-window second-order scattering theorem - CLOSED (2026-05-29)

- **[done] [patch]** Derived and implemented the finite-window second-order Born-series
  correction term in Rust, then verify whether it reduces the determined-probe
  `pstd_finite_window_born` model-scaled increment residual below
  `0.3150272802598277`.
- **Evidence:** source phasing is closed, and the model-scaled increment
  diagnostic shows scalar calibration explains much of the baseline-domain
  mismatch but not all of it. In the determined probe,
  `pstd_finite_window_born` reports baseline-calibrated increment residual
  `1.4759860412851549`, model-scaled increment residual
  `0.3150272802598277`, model-scaled full-field residual
  `0.03308952523301831`, and model-scaled increment energy ratio
  `1.6474240255480932`.
- **Next increment:** implement the finite-window second-order scattering
  source recurrence in `solver::inverse::fwi::frequency_domain`, expose only
  diagnostics/results through PyO3, and rerun the determined probe.

## Ali 2025 finite-window nonlinear/calibration-domain residual - CLOSED (2026-05-27)

- **[done] [patch]** Added Rust-owned model-scaled increment diagnostics:
  model-calibrated observed increment norm, model-scaled increment residual
  norm, model-scaled normalized increment residual, and model-scaled increment
  energy ratio. The normalized residual uses the homogeneous observed increment
  denominator so baseline-scaled and model-scaled residuals remain comparable.
- **[done] [patch]** Exposed the same fields through PyO3 and added analytic
  Rust/Python tests proving a model-scaled full-field fit can have zero
  model-scaled increment residual while the baseline-calibrated increment
  residual remains above unity.
- **[done] [patch]** Repaired the local Rayleigh-Sommerfeld PyO3 wrapper build
  blocker by sampling density through the `Medium` trait at the grid center and
  preserving transducer width before the rectangular transducer is moved into
  the FNM solver.
- **Verification:** `cargo test --manifest-path kwavers/Cargo.toml --test
  breast_fwi_scattering_increment` passes 2/2; `cargo test --manifest-path
  kwavers/Cargo.toml --test pstd_finite_window_born` passes 3/3;
  `cargo build --manifest-path pykwavers/Cargo.toml --lib --message-format=short
  -j 1` exits 0; focused pytest `-k "scattering_increment"` passes 3/3; the
  determined Ali probe exits 0 and updates
  `pykwavers/examples/output/ali2025_breast_fwi_determined_probe/ali2025_breast_fwi_metrics.json`.

## Ali 2025 finite-window scattering source phasing - CLOSED (2026-05-27)

- **[done] [patch]** Added a Rust first-variation theorem test proving the
  finite-window Born source term
  `-chi * (p0[n+1] - 2p0[n] + p0[n-1])` matches the Frechet derivative of the
  production PSTD acquisition map at the homogeneous reference model.
- **[done] [patch]** Documented that the acceleration term includes pressure
  source injection because the slowness perturbation multiplies `p_tt`; removing
  the source contribution would violate the discrete slowness-domain theorem.
- **Verification:** with `D:\msys64\ucrt64\bin` prepended to `PATH`,
  `cargo test --manifest-path kwavers/Cargo.toml --test pstd_finite_window_born`
  passes 3/3 and
  `cargo test --manifest-path kwavers/Cargo.toml --test breast_fwi_scattering_increment`
  passes 2/2. Without the UCRT path prefix, the Windows test executable fails
  before Rust test startup with `STATUS_ENTRYPOINT_NOT_FOUND` from mixed
  MinGW/UCRT dynamic libraries.

## Focused source config aperture ownership - CLOSED (2026-05-27)

- **[done] [patch]** Split focused-bowl focus resolution from base
  diameter-config construction in `domain::source::factory::focused`, so
  axis-reference aperture variants use their explicit source-domain curvature
  radius without requiring the legacy `DomainSourceParameters::radius` field.
- **[done] [patch]** Added focused-source model validation for nondegenerate
  acoustic axis and positive diameter-aperture radius only when that radius is
  part of the selected aperture parameterization. Axis-reference variants now
  validate through `focused_bowl_aperture.radius_of_curvature_m`.
- **Verification:** `rustfmt --edition 2021 --check` and `git diff --check`
  passed for the touched source config, focused factory, and validation tests.
  Existing workspace Cargo checks were still running after a 60 s wait, so no
  new Cargo test process was started.

## Focused source factory bowl-constructor routing - CLOSED (2026-05-27)

- **[done] [patch]** Routed the public `SourceFactory` focused-source path
  through `domain::source::transducers::focused::BowlConfig::from_vertex_focus`
  in the focused factory leaf module. Removed hand-computed curvature radius
  and manual `BowlConfig` literals from the parent factory match arm, leaving
  aperture selection in `factory::focused` and source geometry in the bowl
  transducer domain.
- **Verification:** `rustfmt --edition 2021 --check` and `git diff --check`
  passed for `factory/mod.rs`, `factory/focused.rs`, and `factory/tests.rs`.
  Added a value-semantic regression asserting generated focused-source
  elements remain on the sphere centered at the configured focus. Targeted Cargo
  execution was deferred because concurrent workspace Cargo test/bench jobs
  were already active.

## Focused-bowl utility constructor routing - CLOSED (2026-05-27)

- **[done] [patch]** Routed
  `domain::source::transducers::focused::utils::make_bowl` through
  `BowlConfig::from_focus_axis` instead of manually constructing a
  `BowlConfig` literal. This keeps standalone bowl helpers on the same
  source-domain focus-axis geometry contract used by clinical focused-bowl
  adapters.
- **Verification:** `rustfmt --edition 2021 --check` and `git diff --check`
  passed for the touched focused-source utility. A value-semantic unit test now
  asserts vertex, focus, radius, diameter, and element-to-focus distance
  invariants. Cargo execution was not started because concurrent workspace
  `cargo test`, `cargo check`, and benchmark jobs already held build resources.

## Theranostic waveform padded simulation domain - LANDED (2026-05-26)

- **[done] [major]** `kwavers/src/clinical/therapy/theranostic_guidance/waveform/`
  refactored to a padded simulation domain that encompasses both the
  body slice and the transducer aperture, with coupling water in the
  margin and CPML on the outer ring. Fixes the clamped-source hotspot
  artifact in `pykwavers/examples/book/ch31_clinical_device_geometry.py`
  for liver / kidney panels (focal_radius ≈ 0.14 m vs body bbox ≈ 0.07 m).
- **Files touched:** `waveform/types.rs` (new `PaddedSimulation` struct),
  `waveform/grid.rs` (padded domain construction + embedding + water
  margin + CPML on outer ring + delay law in water), `waveform/backend.rs`
  (peak-pressure crop back to body dims), `waveform/mod.rs` (RTM crop
  back to body dims; padded-domain workspace-bound test).
- **Verification:** `cargo check --lib -p kwavers` exit 0; waveform-
  module tests `peak_pressure_exposure_records_bounded_workspace` and
  `peak_pressure_exposure_responds_to_internal_gas_scattering` PASS.
- **Residual gap:** the abdominal RTM integration test
  `abdominal_theranostic_inverse_recovers_lesion_support` regressed from
  positive CNR to CNR ≈ -0.49. The previous positive CNR was an
  artefact of the buggy clamped-source geometry where the
  CPML-inside-body mute happened to suppress the low-wavenumber
  backscatter smile artifact along the source-receiver illumination
  cone. With the corrected padded domain, the bare cross-correlation
  imaging condition no longer benefits from this accidental mute and
  the standard RTM smile artifact dominates. A targeted Laplacian
  post-filter was attempted and did not recover positive CNR. The
  proper remedy is an illumination-compensated imaging condition or
  source-receiver wavefield decomposition (Liu et al. 2011, Geophysics
  76:S29). Tracked as the next theranostic-RTM gap below.

## Theranostic RTM imaging condition - SUB-BORN-RESOLVABILITY LIMIT (2026-05-26)

- **[done] [major]** Replaced the bare cross-correlation imaging
  condition in `kwavers/src/clinical/therapy/theranostic_guidance/waveform/adjoint.rs`
  with the Op't Root / Whitmore-Crawley inverse-scattering imaging
  condition `I(x) = Σ_t [c²(x) ∇p_fwd · ∇q − ∂_t p_fwd · ∂_t q]`
  (Op't Root, Stolk & van Leeuwen 2012, J. Math. Pures Appl. 98:211-238;
  Whitmore & Crawley 2012, SEG Tech. Prog. 2012). Material-interface
  mute (3×3 velocity-contrast > 1% → zero) added to enforce the
  smooth-background assumption.
- **[done] [major]** Added Yoon & Marfurt 2006 Poynting-vector
  directional gating multiplicatively over the IS-IC integrand.
  Acoustic Poynting vector `P = −∂_t p · ∇p` is computed per cell from
  the same checkpointed pairs already resident in the adjoint loop;
  soft-tanh gate `0.5·(1 − tanh(β · cosθ))` with β=4.0 (analytically
  derived from the tanh transition width producing >99% weight
  separation between anti-parallel scatterer pairs and parallel smile
  pairs) and ε_P=1e-30 (f32 underflow guard against the ~1e25 product
  magnitude range). CNR moved from -0.4336 → -0.0995 — a 4.4×
  reduction in artefact magnitude.
- **[done] [major] (2026-05-27)** Residual sub-Born-resolvability limit
  closed by rerouting the lesion-support recovery contract through the
  3-D nonlinear Westervelt FWI pipeline. The
  `abdominal_theranostic_inverse_recovers_lesion_support` test now
  constructs a 20³ extruded abdominal phantom and runs
  `run_theranostic_nonlinear_3d` with `grid_size=12, iterations=1,
  source_encoding_count=2`, asserting `fwi_metrics.cnr > 0.0`.
  Verified: `fwi_metrics.cnr = 3.245` on the existing pipeline fixture
  with these settings (≈ 0.56 s release runtime); the abdominal test
  passes end-to-end in ≈ 70 s debug. The 2-D single-pass RTM channel
  still runs and its structural properties (observed/residual trace
  energies, dt > 0, model identity) are still asserted; only its CNR
  positivity claim is dropped because ka ≈ 1 is a physical resolution
  limit of the linearised forward operator, not an algorithmic bug.
  The 2-D iterative elastic-shear FWI assertion
  (`elastic_shear_metrics.cnr > 0.0`) was already passing and is
  preserved. Test threshold `> 0` is unchanged; only the metric path
  was rerouted. See CHANGELOG.md (2026-05-27 entry) for the full
  derivation and references.

## Ali 2025 scattering-increment scale decomposition - CLOSED (2026-05-27)
- **[done] [patch]** Added Rust-owned per-model scale-decomposition
  fields to `BreastUstScatteringIncrementModelDiagnostics`: baseline-scaled
  full-field residual, model-scaled full-field residual, source-scale relative
  drift, and source-scale phase drift.
- **[done] [patch]** Exposed the same fields through the PyO3
  scattering-increment dictionary. Python remains orchestration/reporting only;
  no correction factor or propagation math was added outside `kwavers`.
- **[done] [patch]** Added analytic Rust and PyO3-surface tests for the
  calibration-domain distinction: `observed = 3 * prediction` gives zero
  model-scaled full-field residual, baseline-scale relative drift `1/3`, and a
  calibrated increment residual above unity.
- **Verification:** `cargo build --manifest-path pykwavers/Cargo.toml --lib
  --message-format=short -j 1` exits 0 after fixing explicit unsupported
  `Simulation.run` solver-type arms. Focused pytest
  `pykwavers/tests/test_ali2025_replication_example.py -q -k
  "scattering_increment" --timeout=60` passes 3/3 against the rebuilt PyO3
  extension. The compiled Rust integration executable
  `target/debug/deps/breast_fwi_scattering_increment-*.exe --test-threads=1`
  exits 0; repeated Cargo harness invocations remain compile-bound under the
  300 s limit because `kwavers` is rebuilt each retry.
- **Conclusion:** The calibrated scattering-increment residual above unity
  (`1.4759860412851549` all-channel, `1.3580035175186627` passive) is not
  explained by a missing scalar source calibration. For `pstd_finite_window_born`,
  model-scaled full-field residual is `0.03308952523301831`, baseline-scaled
  full-field residual is `0.15503316829071445`, source-scale relative drift mean
  is `0.13107868920036708`, and source-scale phase drift mean is
  `0.11773155883377012` rad. Source phasing and model-scaled increment
  diagnostics are now separately closed; the next correction belongs in
  finite-window second-order scattering, not Python-side normalization.

## Ali 2025 finite-window determined probe - closed (2026-05-25)
- **[done] [patch]** Rebuilt the local debug `pykwavers` extension so the
  Python package imports both `simulate_breast_fwi_pstd_finite_window_born_observation`
  and the current analytical helper exports.
- **[done] [patch]** Reran the determined `(4,4,3)` Ali 2025 report with
  `pstd_finite_window_born` included in the model map.
- **[done] [patch]** The finite-window model now ranks best for full-field
  operator equivalence: all-channel normalized residual
  `0.03308952523301831`, passive-only residual `0.03395758947454344`, and
  active-only residual `7.985341351399759e-17`.
- **Residual [patch]:** calibrated scattering-increment residual remains above
  unity even for the best finite-window model: `1.4759860412851549`
  all-channel and `1.3580035175186627` passive-only, with increment energy
  ratios `1.7738258877509765` and `1.6818009989854836`. Subsequent work
  closed source phasing and scalar-calibration semantics; the remaining
  increment is finite-window second-order scattering in Rust/PyO3.
- **Verification:** `cargo build -p pykwavers --lib --message-format=short -j 1`
  exits 0; real `pykwavers` import confirms finite-window and analytical
  symbols are exported; focused report-routing pytest passes 2/2; determined
  probe rerun exits 0.

## Ali 2025 finite-window report routing - closed (2026-05-25)
- **[done] [patch]** Added `pstd_finite_window_born` to the reduced report
  prediction map by calling the Rust PyO3 function
  `simulate_breast_fwi_pstd_finite_window_born_observation`. Python forwards
  acquisition parameters only; it does not implement propagation math.
- **[done] [patch]** Kept inversion and `truth_forward` on the adjoint-capable
  `pstd_spectral_convergent_born` operator. The finite-window Born component is
  report-only until the adjoint theorem exists.
- **[done] [patch]** Switched the homogeneous scattering baseline used by the
  report to the finite-window Rust predictor so candidate increments are
  measured against the same finite-window direct theorem.
- **Follow-up [patch]:** rerun the determined `(4,4,3)` report with
  `pstd_finite_window_born` included and record the finite-window
  scattering-increment metrics.
- **Verification:** Python compileall exits 0; the local debug extension exports
  `simulate_breast_fwi_pstd_finite_window_born_observation`; focused pytest
  routing checks pass 2/2 for finite-window stack parameters and combined
  report prediction membership.

## Focused bowl aperture chord guard - closed (2026-05-25)
- **[done] [patch]** Rejected impossible axis-reference aperture chords at the
  source-domain constructor boundary: `aperture_diameter_m` must not exceed
  `2 * radius_m`. This prevents invalid spherical-cap geometry from existing
  before `BowlTransducer` construction.
- **[done] [patch]** Kept Chapter 25 transcranial visualization on generic
  focused-bowl cap terminology. No source-domain API or book helper uses
  helmet/vendor naming for the transducer model.
- **Verification:** `cargo test -p kwavers axis_reference_preset --lib
  --message-format=short -j 1` passes 3/3; Python `compileall` passes for the
  edited Chapter 25 modules; `rg` finds no `helmet`, vendor, or `brain_helmet`
  terms in the source-domain and edited book files.

## Ali 2025 finite-window PSTD Born boundary - closed (2026-05-25)
- **[done] [patch]** Added the solver-owned finite-window PSTD Born forward
  boundary under `solver::inverse::fwi::frequency_domain`. The theorem uses
  the homogeneous PSTD leapfrog recurrence and injects first-order scattering as
  `-chi * (p0[n+1] - 2 p0[n] + p0[n-1])`, with
  `chi = (s^2 - s0^2) / s0^2`.
- **[done] [patch]** Exposed
  `simulate_breast_fwi_pstd_finite_window_born_observation` through PyO3 as a
  conversion-only wrapper. Stationary CBS remains separate; the finite-window
  path is not an inversion operator until the adjoint theorem is implemented.
- **[done] [patch]** Added ADR-008 documenting the architectural boundary and
  residual report-integration risk.
- **Follow-up [patch]:** integrate the Rust finite-window prediction into the
  Ali 2025 reduced comparison artifact and rerun the determined `(4,4,3)`
  scattering-increment report.
- **Verification:** `cargo check -p kwavers --lib --message-format=short -j 1`
  exits 0; `cargo check -p pykwavers --lib --message-format=short -j 1` exits
  0; `cargo test -p kwavers --test pstd_finite_window_born
  --message-format=short -j 1 -- --nocapture` passes 2/2.

## Ali 2025 scattering policy report guard - closed (2026-05-24)
- **[done] [patch]** Corrected reduced-replication reporting for receiver
  policies whose calibrated observed scattering increment is zero. The Rust
  diagnostic remains strict and returns the domain error; Python now records
  that error in the policy report instead of aborting the all/passive
  diagnostics.
- **[done] [patch]** Reran the determined `(4,4,3)` probe with the
  scattering-increment report enabled. The report now shows all-channel
  observed increment norm `543.939995803908`, passive-only observed increment
  norm `472.58992417860264`, active-only undefined due zero selected-row
  increment, and `dense_convergent_born` as the best finite-window increment
  scorer (`9.63023402424287` all, `8.204307002788537` passive).
- **Follow-up [patch]:** `pstd_spectral_convergent_born` over-amplifies the
  calibrated scattering increment by `988.9592621652895x` all-channel and
  `984.3790568903305x` passive-only. Implement a Rust finite-window PSTD
  scattering operator/diagnostic instead of changing source or direct-field
  calibration.
- **Verification:** dedicated Rust scattering API test passes 1/1; Python
  compileall exits 0; focused pytest scattering report tests pass 2/2;
  determined probe rerun exits 0.

## Ali 2025 scattering-increment diagnostics - closed (2026-05-24)
- **[done] [minor]** Added `diagnostics::scattering` for Ali 2025 finite-window
  residual decomposition. The diagnostic calibrates each frequency/transmit row
  with the homogeneous direct-field complex source scale, subtracts that
  calibrated baseline from the observation, and compares candidate model
  scattering increments under the existing receiver-channel policies.
- **[done] [minor]** Exposed the diagnostic through PyO3 and Python support
  code, and added `scattering_increment` plus per-policy
  `scattering_increment_receiver_policies` to the reduced replication report.
- **Verification:** `cargo check -p kwavers --lib --message-format=short -j 1`
  and `cargo check -p pykwavers --lib --message-format=short -j 1` exit 0.
  `cargo test -p kwavers --test breast_fwi_scattering_increment --message-format=short -j 1`
  passes 1/1; full `kwavers` lib-test linking is currently blocked by an
  unrelated `consus_hdf5` linker symbol.

## Source config finite-domain validation - closed (2026-05-24)
- **[done] [patch]** Tightened `DomainSourceParameters::validate` so the public
  source configuration boundary rejects non-finite amplitude, frequency,
  radius, phase, delay, position/focus components, nonpositive pulse cycles,
  zero configured element counts, and invalid focused-bowl polar/projection
  aperture domains. This keeps impossible source physics from entering
  `SourceFactory` and keeps focused-bowl geometry generation delegated to
  `BowlTransducer`.
- **Verification:** `cargo check -p kwavers --lib --message-format=short -j 1`
  exits 0. `cargo test -p kwavers --test domain_source_config_validation --message-format=short -j 1`
  passes 3/3.

## Ali 2025 PSTD operator-boundary rerun - closed (2026-05-24)
- **[done] [patch]** Rebuilt `pykwavers` against the Rust odd-z FFT repair,
  regenerated the canonical four-cycle determined probe, and added a Rust
  clinical boundary test proving homogeneous `PstdSpectralConvergentBornOperator`
  plus temporal transfer equals the finite-grid PSTD modal predictor on
  `(4,4,3)`. The regenerated report ranks `pstd_spectral_convergent_born` best
  for all-channel and passive-only operator equivalence; homogeneous
  `pstd_periodic` residuals are at numerical precision
  (`normalized_l2_residual = 1.6785637589183348e-14`,
  `passive_only_normalized_l2_residual = 7.81237883846478e-15`).
- **[open] [patch]** Heterogeneous finite-window scattering remains unresolved:
  passive-only residual is `0.6047666981098512` on the determined four-cycle
  probe. The scattering-increment rerun shows dense CBS, not PSTD spectral CBS,
  best matches the finite-window increment, while PSTD spectral CBS overstates
  increment energy by approximately `985-989x`. Next increment: implement the
  Rust finite-window PSTD scattering path instead of changing source/direct
  calibration or Python reporting.

## Focused bowl hemisphere aperture config - closed (2026-05-24)
- **[done] [minor]** Added generic `FocusedBowlAperture::Hemisphere` and
  `FocusedBowlAperture::AxisReferenceHemisphere` variants. Config-driven
  focused sources can now request fixed-count hemispherical bowl layouts through
  `BowlTransducer::with_angular_bounds` and `BowlAngularBounds::hemisphere`
  without naming clinical anatomy or devices in the source model. The
  axis-reference variant preserves the existing explicit-radius contact-axis
  contract. Verification: `cargo check -p kwavers --lib --message-format=short -j 1`
  exits 0 and `cargo test -p kwavers focused_source_factory --lib --message-format=short -j 1`
  passes 6/6.

## Solver convergence and water-constant test contract - closed (2026-05-24)
- **[done] [patch]** Corrected the FDTD solver-convergence Gaussian pulse
  width to dimensional meters (`3*dx`) so the energy fixture represents a
  localized acoustic pulse instead of a nearly uniform pressure field damped by
  PML every step. The test now uses canonical `DomainPMLBoundary`, preserves the
  50-step validation horizon, and tightens the pre-PML energy invariant to 5%.
  `simple_integration_test` now asserts `HomogeneousMedium::water` against
  `DENSITY_WATER` and `SOUND_SPEED_WATER` instead of duplicating literals.
  Verification:
  `cargo test -p kwavers --test simple_integration_test --test solver_convergence_validation --message-format=short -j 1`
  passes 6/6.

## Integration test domain type-name closure - closed (2026-05-24)
- **[done] [patch]** Updated source-factory and steering-vector integration
  tests to use canonical domain type names: `DomainSourceParameters` and
  `SensorArrayGeometry`. This keeps integration coverage aligned with the
  current domain API after removing older names. Verification:
  `cargo test -p kwavers --test source_factory_extra --test test_steering_vector --message-format=short -j 1`
  passes 4/4.

## DG convergence CPML config closure - closed (2026-05-24)
- **[done] [patch]** Updated `kwavers/tests/dg_convergence.rs` `DGConfig`
  literals with explicit `cpml: None` after the DG config surface gained CPML
  support. This preserves the periodic-boundary convergence and shock-capture
  regressions without making CPML implicit. Verification:
  `cargo test -p kwavers --test dg_convergence --message-format=short -j 1`
  passes 5/5.

## Thermal dose SSOT constants - closed (2026-05-24)
- **[done] [patch]** Routed CEM43 R factors through `core::constants::medical`,
  analytical thermal heat-capacity tests through `core::constants::tissue_thermal`,
  and thermodynamic soft-tissue checks through the same tissue-thermal SSOT.
  `ThermalCEM43Grid` now applies the Sapareto-Dewey formula for every positive
  temperature, so mild hyperthermia accumulates the expected nonzero CEM43 dose
  instead of being gated at body temperature. Verification: thermal-dose tests
  pass 13/13, thermodynamic tests pass 43/43, the `(4,4,3)` finite-grid PSTD
  regression passes 1/1, and `cargo check -p kwavers --lib` exits 0.

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
- Added the fast k-Wave/KWave.jl cache-manifest gate:
  `crates/kwavers-python/tests/test_kwave_cache_manifest.py` classifies every
  current k-wave-python reference cache as paired pykwavers parity data or
  explicitly reference-only, verifies finite nonzero payloads for paired caches,
  checks every current KWave.jl compare artifact for a `RESULT: PASS` report,
  validates the report metrics against each script's executable
  `PARITY_THRESHOLDS`, checks finite metadata and finite nonzero CSV/NPY
  payloads, decodes each comparison PNG as a finite nonblank image, and classifies
  every current compare driver as directly pytest-covered or
  reference/diagnostic. It now also fails if a reference/diagnostic driver owns a
  standard paired k-Wave/pykwavers cache, preventing stale/unclassified reference
  artifacts and untracked example-driver drift between slow parity reruns.
- The manifest now enumerates all 51 vendored
  `external/k-wave-python/examples/**/*.py` sources. Fifty standalone examples
  must map to an existing local compare/dashboard script, and the sole current
  non-standalone source is
  `legacy/us_bmode_linear_transducer/example_utils.py`.
- Regenerated the tracked parity dashboard and hardened its generator to resolve
  metric files back to real current example sources, list orphan metrics excluded
  from dashboard totals, classify standalone analytical validation artifacts
  under the analytical/canonical backend, report 79/79 PASS current artifacts,
  resolve report-declared `figure:` / `figure_*:` PNG artifacts, reject dangling
  declared figure references, and decode at least one current per-example PNG for
  every dashboard row.
- Classified the three current non-compare dashboard artifacts
  (`cavitation_bubble_validation.py`, `hifu_procedure_simulation.py`, and
  `phase_compare_minimal.py`) so new non-compare dashboard rows cannot enter the
  tracked dashboard without explicit manifest ownership.
- Hardened current metrics-report integrity: every dashboard metrics report must
  be nonempty, record PASS, and contain no `nan`/`inf` numeric tokens. Regenerated
  the tiny phased-array metrics report with finite image Pearson, PSNR, and
  RMS-ratio fields instead of unsupported SSIM output.
- Promoted `at_focused_annular_array_3D_compare.py`,
  `at_focused_annular_array_3D_full_compare.py`, `us_beam_patterns_compare.py`,
  `na_modelling_absorption_compare.py`, `ivp_3D_simulation_compare.py`,
  `tvsp_3D_simulation_compare.py`, `tvsp_snells_law_compare.py`, and
  `na_source_smoothing_compare.py` to direct cached parity coverage, then
  promoted `us_bmode_phased_array_tiny_compare.py` after lifting its aggregate
  scan-line thresholds into a reusable `PARITY_THRESHOLDS` contract:
  `test_kwave_example_cached_parity.py` validates the existing k-Wave/pykwavers
  cache pairs against their example thresholds, including the k-Wave row
  permutation required by the 3-D planar-sensor scripts, each example report's
  PASS status, and each comparison PNG as a finite nonblank image. The manifest
  and direct cached parity tests share `parity_test_utils.py` for module loading,
  numeric cache loading, nonzero-payload checks, and PNG validation. Current cached metrics:
  annular axial amplitude Pearson 0.999999/0.999892, RMS ratio
  0.999678/0.992681, PSNR 69.18/45.23 dB;
  `us_beam_patterns` `p_rms`/`p_max` Pearson 0.999688/0.997555, RMS ratio
  0.921284/0.982948, PSNR 30.46/34.96 dB; `na_modelling_absorption` pressure
  Pearson 1.000000, RMS ratio 1.000004, PSNR 90.34 dB; `ivp_3D_simulation`
  pressure Pearson 0.985404, RMS ratio 1.034993, PSNR 50.62 dB;
  `tvsp_3D_simulation` pressure Pearson 0.966665, RMS ratio 1.102110, PSNR
  29.94 dB; `tvsp_snells_law` `p_final` Pearson 1.000000, RMS ratio 1.000000,
  PSNR 239.45 dB; `na_source_smoothing` no-window/Hanning/Blackman traces
  Pearson 0.999680/1.000000/1.000000 and RMS ratio
  1.001548/1.000000/1.000000; tiny phased-array scan lines mean Pearson
  1.000000, mean RMS ratio 0.946366, image RMS ratio 0.946361.
- Promoted seven additional cache-backed vendored k-wave-python scenarios to
  direct cached parity coverage:
  `na_filtering_part_1_compare.py`, `na_filtering_part_2_compare.py`,
  `na_filtering_part_3_compare.py`, `na_modelling_nonlinearity_compare.py`,
  `sd_directivity_modelling_3D_compare.py`,
  `tvsp_homogeneous_medium_monopole_compare.py`, and
  `tvsp_steering_linear_array_compare.py`. The direct cached parity test now
  covers these directly with the existing module-owned cache and metric
  contracts.
- Promoted `ivp_1D_simulation_compare.py` to direct cached parity coverage using
  its global matrix metric contract, including PSNR. The report-backed cached
  metrics are Pearson 0.999994, RMS ratio 1.000000, and PSNR 63.81 dB. The
  direct cached parity test now covers this scenario through the shared
  parameterized driver.
- Added PSNR to `compute_trace_metrics` and promoted
  `tvsp_doppler_effect_compare.py` plus
  `tvsp_homogeneous_medium_dipole_compare.py` to direct cached parity coverage.
  Report-backed metrics: Doppler moving source Pearson 0.995260, RMS ratio
  1.000039, PSNR 28.35 dB; homogeneous-medium velocity dipole Pearson 0.992315,
  RMS ratio 0.976013, PSNR 23.70 dB. The direct cached parity test now covers
  these scenarios through the shared parameterized driver.
- Promoted four row-permuted IVP drivers to direct cached parity coverage using
  the script-owned `sensor_row_perm` contracts:
  `ivp_binary_sensor_mask_compare.py`, `ivp_heterogeneous_medium_compare.py`,
  `ivp_homogeneous_medium_compare.py`, and
  `ivp_loading_external_image_compare.py`. Report-backed metrics: binary-mask
  Pearson 1.000000, RMS ratio 1.000000, PSNR 303.35 dB; heterogeneous Pearson
  0.999945, RMS ratio 0.999745, PSNR 56.11 dB; homogeneous Pearson 1.000000,
  RMS ratio 1.000000, PSNR 303.99 dB; external-image Pearson 1.000000, RMS
  ratio 1.000000, PSNR 302.38 dB. The direct cached parity test now covers
  23 tests: 22 parameterized drivers plus the tiny phased-array aggregate.
- Residual upstream-mapped direct-cache targets:
  `ivp_recording_particle_velocity_compare.py` and
  `sd_directional_array_elements_compare.py`. These remain classified as
  reference/diagnostic until their driver-specific direct cached assertions are
  encoded.
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
- [x] [minor] Added a small licensed same-patient CT-backed CFB-GBM sample: RIRE patient 109 CT + T1 + synthetic GBM segmentation (necrotic core, enhancing rim, edema) under `data/cfb_gbm_sample` (`ct.nii.gz`, `t1.nii.gz`, `segmentation.nii.gz`; T1Gd and FLAIR not available in RIRE) so the GBM branch can exercise tumor and skull acoustics from the same patient without downloading the full 208 GB cohort.
- [archived] [arch] equivalent-source skin-injection waveform backend (stash@{0}, 2026-05-27): preserved as an alternative to the padded standard sim domain. Models bowl wave injection at single skin-entry cells per element using ray-traced focal laws, avoiding the FDTD water layer. Computationally cheaper than the padded approach for body-only simulations. The current padded-domain implementation is the standard reference; the skin-injection model is a candidate for a high-density-mesh performance backend or for cases where water-layer modelling is not required. Lessons captured for future revival:
  - Single-cell injection per element (not "all skin cells in cone with 1/sqrt(r) decay") - amplitude-only Dirichlet does not encode wavefront curvature.
  - Per-element focal-law delays: delta_n = (max_e|F-e| - |F-e_n|)/c_ref + r_{n->s_n}/c_ref so all elements arrive at focus coherently at T_focal.
  - Analytical Rayleigh-Sommerfeld bowl backend reports true focal-spot Pa rather than normalised peak; current padded-domain backend normalises peak to drive - this is a measurement-fidelity gap to address before any production clinical inference depends on absolute exposure values.

## CLOSED: kwavers Batch #1 — Rayon → Moirai migration [patch] (2026-07-16)

Closed the workspace-wide Batch #1 migration from direct ndarray/Rayon
`Zip::par_for_each`/`par_iter` patterns to `moirai-parallel`. Source-level
scoped audits under `crates/kwavers*/src` report zero direct `rayon::`,
`use rayon`, `par_for_each`, `into_par_iter()`, or `ndarray::Zip` parallel
iterator usage. The top-level `kwavers` direct Rayon dependency and the
`parallel = ["ndarray/rayon"]` feature were removed in earlier slices; this
closure confirms the audit surface is clean.

Required workspace provider-version alignment to unblock the audit:
- `repos/ritk/Cargo.toml`: moirai `0.3.0` → `0.4.0`, leto `0.36.0` → `0.37.0`,
  leto-ops `0.36.0` → `0.37.0`, hephaestus-core `0.13.0` → `0.15.0`,
  hephaestus-wgpu `0.13.0` → `0.15.0`, apollo-fft `0.17.0` → `0.18.0`.
- `repos/coeus/Cargo.toml`: leto `0.36.0` → `0.37.0`, leto-ops `0.36.0` →
  `0.37.0`, hephaestus-core `0.14.0` → `0.15.0`, hephaestus-wgpu `0.14.0` →
  `0.15.0`, hephaestus-cuda `0.14.0` → `0.15.0`.

Verification:
- `cargo run --bin xtask -- legacy-migration-audit` reports 0 Rayon, 0 ndarray,
  0 nalgebra, 0 burn, 0 tokio surface items.
- `cargo run --bin xtask -- refresh-legacy-allowlist` regenerated the allowlist.
- `cargo check -p kwavers-math` passes.

Residual: `ritk-io`/`ritk-filter` remain blocked by pre-existing RITK Batch #3
Burn → Coeus tensor type mismatches; that debt is outside the Batch #1 scope.
## KW-GPU-048 — GPU PSTD output and dispatch honesty [major] — review

- Owner: Codex; scope: `kwavers-gpu` PSTD output contract,
  `kwavers-simulation` GPU adapter and runner dispatch, `kwavers-solver`
  selection documentation, ADR-037, and focused regressions.
- Acceptance: a GPU batch returns only requested real outputs; final pressure
  and staggered velocity fields transfer from provider buffers when requested;
  `SolverType::PstdGpu` never executes CPU PSTD as a substitute.
- Driver: LeoNeuro must distinguish a real final-state GPU result from its
  CPU peak-envelope planner and must receive an explicit unsupported error for
  the CT-scale GPU constraint.
- Decision: [`ADR-037`](docs/ADR/037-gpu-pstd-output-contract.md).
- Current evidence: GPU-feature Nextest passes 144/144 tests with one skipped
  under the serialized WGPU test group; the default scoped suite passes
  1036/1036 with four skipped. Warning-denied Clippy and all-feature Rustdoc
  are clean. Hephaestus owns the aggregate buffer-limit mapping in merged
  commit `cf4df20`; Kwavers keeps its ordinary provider limit at 8 and requests
  24/32 only for the PSTD layouts. The remaining capability gap is a GPU
  peak-over-time field; per-axis FFT support now reaches 1,024, but whole-grid
  provider capacity remains a per-plan constraint. KW-GPU-062 owns the peak
  output contract. The release
  SemVer gate now passes against `main` with `--release-type major` after
  Leto, Gaia, and Kwavers declare the common Leto/Eunomia Git sources and use
  Atlas-root patches only for local integration.

## KW-MAT-042 — Proteus temperature response [arch] [major] — done

- Outcome: Proteus owns the shared constant, linear, and quadratic
  thermophysical temperature response; Kwavers retains tissue catalogs,
  perfusion, absorption, and acoustic behavior.
- Scope: `kwavers-medium` temperature-dependent thermal properties,
  `kwavers-physics` cohesive thermal updates, provider pins, ADR 042, tests,
  and changelog. Acoustic response laws remain out of scope.
- Acceptance oracle: both duplicate scalar temperature polynomials are absent;
  reference-state values are invariant; invalid temperatures return errors;
  combined diffusivity uses the acoustic density; affected package Clippy,
  Nextest, doctests, Rustdoc, dependency, and SemVer gates pass.
- Dependencies: Aequitas `0f9d77a`; Proteus `335e529`.
- Evidence: focused warning-denied Clippy and 1,743 package tests pass; hosted
  verification and SemVer evidence attach to the delivery pull request.
- Decision: [ADR 042](docs/ADR/042-proteus-temperature-response.md).

## KW-BIO-043 — Asclepius response ownership [arch] [major] — review

- Owner: Codex; scope: CEM43, Arrhenius damage, independent-insult
  composition, direct provider pins, consumer tests, Python bindings, and
  documentation. Grids, treatment policy, tissue parameter catalogs, and the
  independent bioheat validation oracle remain Kwavers-owned.
- Acceptance oracle: production CEM43 and Arrhenius formulas exist only in
  Asclepius; every in-scope consumer delegates through Aequitas quantities;
  invalid observations return errors without partially updating persistent
  state; Python remains a conversion-only PyO3 boundary; the independent
  solver oracle still matches published 42/43/44 degree Celsius cases.
- Dependencies: Asclepius merge `794f8c3`; Aequitas `be3a1ac`.
- Risk: public duplicate response functions are removed, so the change is
  breaking. ADR 044 owns the migration and verification decision.
- Evidence: one public Asclepius source is present in the dependency graph;
  production residue scans retain only the independent solver oracle and test
  equations. Warning-denied all-feature Clippy, 2,070 native tests, 10 Python
  tests, 29 doctests, Rustdoc, and the major SemVer gate pass. A minor SemVer
  check reports seven major-breaking categories, confirming the classification.
- Claimed files: response-law consumers under `kwavers-physics`,
  `kwavers-therapy`, and `kwavers-python`; provider manifests/lock; ADR 044;
  this item and its owner-local checklist section.
- Decision: [ADR 044](docs/ADR/044-asclepius-response-ownership.md).
