# Backlog / Strategy

## KW-PY-UVLOCK-STALE-2026-09-21 — The Python dev lockfile is stale against its own pyproject [patch] — todo

- **Finding.** `uv lock --check --directory crates/kwavers-python` fails against the committed `pyproject.toml` and the committed `uv.lock`: uv resolves 129 packages for the floor the pyproject declares, against 77 for a 3.10 floor. The tracked lockfile therefore did not match its own project before any floor change.
- **Impact.** `uv sync` re-locks on this package; CI does not invoke uv, so no hosted gate is red.
- **Acceptance:** a `uv lock` whose diff is reviewed as a dependency change in its own right, or an explicit decision that this lockfile is not maintained here.
- **Status:** todo, not claimed; filed 2026-09-21 from the kwavers-python floor change.

## KW-SWE-EDGE-GROWTH-2026-09-17 — Elastic displacement grows without bound when the initial field reaches the edges [patch] [fix] — todo

- **Finding.** 64 cubed, lambda = mu = 1 GPa, water density, default PML, `ux = sin`, `uy = cos` of `0.37 i + 0.53 j + 0.71 k` over the whole grid: the peak grows 1 to 9.2e3 in 400 steps at CFL 0.5, and the growth follows physical time, not step count (CFL 0.25 at step 200 equals CFL 0.5 at step 100), so it is not a timestep instability. A centred Gaussian pulse at the same settings decays into the PML at every CFL.
- **Question.** Whether this is the boundary closure (one-sided first-order derivatives at the walls feeding a non-symmetric operator), the PML acting on displacement, or an inadmissible initial condition the solver should reject. Unbounded growth is not physical in any of the three.
- **Acceptance:** a regression test that runs the whole-grid initial condition and bounds its energy, or a typed rejection of such initial data with its reason; the cause recorded here.
- **Status:** todo, not claimed; filed 2026-09-17 by claude-opus-5 from the SWE probe.

## KW-PROSE-PR-UNMERGEABLE-2026-09-16 — A prose-only pull request can never satisfy the required checks [patch] [ci] — blocked

- **Finding.** `main` requires nine status checks. All nine come from `ci.yml` and `architecture-validation.yml`, and both workflows carried `paths-ignore` for `backlog.md`, `CHANGELOG.md`, `README.md`, `gap_audit.md` and `docs/**` on their pull-request trigger. A filtered workflow reports no check at all — not a skipped one — so a pull request touching only those paths sits at `BLOCKED` with zero checks forever. [#785](https://github.com/ryancinsight/kwavers/pull/785) is the live instance; the last prose-only one, #768, went in by administrative merge, which is how the defect stayed invisible.
- **Change.** The path filter moves one step later, from the trigger to a `changes` job that reads the pull request's own file list (the API, not a diff: a checkout here is shallow, and fetching the history to classify five paths costs more than the jobs it skips). `lockfile` and `semver` gate on it in `ci.yml` and every job reaches one of those two; each standalone job gates on it in `architecture-validation.yml`. A skipped required job counts as a success, so a prose-only pull request now reports all nine and merges on its own. Anything the classifier cannot read answers `true` and runs the gate.
- **Cost.** One sub-minute runner per prose pull request, against a pipeline that no longer needs an administrator. The push trigger keeps its filter, so prose landing on `main` still starts nothing.
- **Delivered, in two parts.** PR [#790](https://github.com/ryancinsight/kwavers/pull/790) moved the filter into the `changes` job; the prose pull request then reported 20 skipped checks and 2 successes, and stayed blocked on one context. `Build & Test` builds its matrix from an expression, and a matrix job skipped before its matrix expands reports once under its base name, so `Build & Test (stable)` never appeared. PR [#791](https://github.com/ryancinsight/kwavers/pull/791) adds `CI gate` and `Architecture gate` — always-running jobs depending on every job in their workflow, failing only on `failure` or `cancelled` — and both are required contexts now.
- **Open, needs one command.** `Build & Test (stable)` still has to leave the required list; the agent session's classifier refused the call (reason: `CI Bypass`), so it waits for a human or an explicit permission rule: `gh api repos/ryancinsight/kwavers/branches/main/protection/required_status_checks/contexts --method DELETE -f "contexts[]=Build & Test (stable)"`. Coverage does not drop: `CI gate` depends on `build`. Recorded on [#785](https://github.com/ryancinsight/kwavers/pull/785#issuecomment-5692990329).
- **Integrator:** claude-opus-5; **branches:** `ci/kwavers-prose-pr-checks`, `ci/kwavers-gate-aggregator`; **last-update:** 2026-09-16.

<a id="kw-manifest-implementation-2026-09-09"></a>

## KW-MANIFEST-IMPLEMENTATION-2026-09-09 — Reduce the implementation-bearing manifests [patch] [arch] — in-progress

- **Integrator:** claude-opus-5; **lease:** none held between increments --
  each file is independent, so contributors take disjoint ones.
- **The class.** `lib.rs` and `mod.rs` are module manifests: the tree, its
  curated re-exports, and the crate or module docs. The fleet scan counts a
  manifest carrying more than twenty lines of body, and this repository held
  290 of them, the largest at 463 body lines -- a whole FWI engine behind a
  `mod` declaration, invisible to anyone reading the tree.
- **Delivered:** the largest, `inverse/fwi/time_domain/self_adjoint/mod.rs`,
  split into `types`, `operators`, `forward` and `gradient`; its manifest is
  39 lines and zero body. 281 remain.
- **Where they are, measured 2026-09-09:** kwavers-solver 92, kwavers-analysis
  49, kwavers-physics 31, kwavers-diagnostics 16, kwavers-therapy 15,
  kwavers-gpu 11, kwavers-boundary 11, kwavers-transducer 10, kwavers-medium
  9, kwavers-python 6, the rest in single digits. The next four by size are
  `fwi/time_domain/mofi` (428), `analytical/transducer` (342),
  `phantom/scatterers` (297) and `theranostic_guidance/.../forward` (267).
- **Method that worked, and the trap in it.** Split at the seams the code
  already has, not at line counts, and compute each section's start by walking
  back over the item's doc comments and attributes -- a boundary taken one
  line late orphans a `///` onto the previous section and the compiler reports
  it as "a doc comment that documents nothing". Cross-module items take
  `pub(super)`, never `pub(crate)`: the split must not widen the crate
  surface. `mofi` needs a different tool -- its types are interleaved with its
  functions, so contiguous spans do not separate them.
- **Do not run `cargo fix` from inside the stack overlay.** It rewrote 365
  lines of this repository's `Cargo.lock` during this increment. Narrow the
  imports by reading what clippy reports instead.
- **Acceptance:** each increment leaves its manifest at zero body lines with
  the crate's public surface unchanged, the package's tests passing, and
  clippy `--locked --all-targets -D warnings` clean.

<a id="kw-edition-2021-behind-current-2026-09-09"></a>

## KW-EDITION-2021-BEHIND-CURRENT-2026-09-09 — 25 crates on edition 2021, resolver 2 [patch] — todo

- **Outcome:** the workspace builds on the current stable edition and resolver,
  with the package fields that should be inherited actually inherited.
- **Measured 2026-09-09:** all **25** crates plus `xtask` declare
  `edition = "2021"` individually; the root declares `resolver = "2"`. The
  toolchain pin is `rustc 1.97.0`, which supports edition 2024 and resolver 3,
  so nothing external is holding this back.
- **Found by trying to use the language.** A let-chain
  (`if let Some(x) = f() && p(x)`) in a new `xtask` audit failed to compile:
  "let chains are only allowed in Rust 2024 or later". The audit was written
  the long way instead, which is the cost showing up as worse code rather than
  as a build error.
- **The lint floor counts this** -- "edition or resolver behind current" and
  "members re-declaring package fields the workspace should own" are both
  measured debt classes, and 25 separate `edition` declarations are the second
  one as well as the first.
- **Shape:** `cargo fix --edition` per crate, then flip the root to
  `edition = "2024"` / `resolver = "3"` with members inheriting via
  `edition.workspace = true`, in one change so the crates cannot drift apart
  again. Resolver 3 changes feature unification, so the feature-combination
  matrix is the check that matters, not just a build.
- **Acceptance:** the root declares edition 2024 and resolver 3, no member
  declares its own `edition`, the full feature matrix passes, and a let-chain
  compiles.

<a id="kw-errors-docs-are-template-output-2026-09-09"></a>

## KW-ERRORS-DOCS-ARE-TEMPLATE-OUTPUT-2026-09-09 — 300 functions document an error they cannot return [patch] — todo

- **Outcome:** an `# Errors` section names the condition that produces the error,
  or it does not exist. No function documents a failure it cannot have.
- **Measured 2026-09-09**, repo-wide across `crates/`: **1141** occurrences of the
  identical line ``- Returns [`Err`] if an internal constraint is violated.`` in
  **544** files, plus **297** of ``- Propagates any [`crate::KwaversError`]
  returned by called functions.`` **350** of those sections sit on functions that
  **cannot fail** -- the signature returns neither `Result` nor `Option` -- by
  crate: kwavers-solver 92, kwavers-analysis 50, kwavers-physics 44,
  kwavers-simulation 16, kwavers-therapy 15, kwavers-transducer 11,
  kwavers-receiver 11, kwavers-medium 11.
- **Two defects, not one.** The 350 are *false*: they tell a reader a function
  can fail when it cannot. The remaining ~840 are *contentless*: on a genuinely
  fallible function, "an internal constraint is violated" names no invariant, no
  input range and no caller remedy, which is what the standard asks an `# Errors`
  section for. This is what satisfying a lint without reading the code looks like,
  and it is why the open KW-LINT-047 rows counting 26 "missing `# Errors`" must
  not be closed by extending this template.
- **Delivered:** `cargo run -p xtask -- audit-errors-docs` runs inside
  `Validate Clean Architecture`, ratchets at its baseline and fails on any rise;
  its unit tests pin the cases a naive matcher gets wrong (a wrapped signature, a
  `where` bound mentioning `Result`, a unit return with no arrow). All 350
  sections were removed in #762 -- 1040 lines across 226 files, every removed
  line a `///` doc line and none added -- the audit re-reads the tree with an
  independent implementation and reports zero, and `BASELINE` closes to 0. The
  local gate then caught a defect in the guard itself: with `BASELINE = 0`,
  `total < BASELINE` is `usize < 0`, always false, denied by clippy as
  `absurd_extreme_comparisons`; the dead branch is gone.
- **Next:** the ~840 contentless sections on genuinely fallible functions, per
  crate, replacing the category with the condition.
- **Acceptance:** zero `# Errors` sections on functions returning neither
  `Result` nor `Option`, enforced by the conformance scan so the template cannot
  return; and the remaining sections name a condition rather than a category.

## KW-CI-PER-PR-MATRIX-STARVATION-2026-09-08 — Per-PR CI runs the scheduled matrix [patch] [ci] [perf] — in-progress

- **Integrator:** claude-opus-5 (lane `kwavers-local-gate`); claimed 2026-09-09.
- **Outcome:** the pull-request path runs the affected-scope checks; the full
  matrix (extra toolchains, heavy validation, coverage) moves to the scheduled
  selection-drift backstop, bringing the verification round-trip toward the
  five-minute job target.
- **Measured 2026-09-08, `main` runs:** CI/CD Pipeline 94 and 114 min wall clock,
  Architecture Validation 75 min, Deploy mdBook 64 and 95 min. Job runtimes in
  run `34257329545` sum to about 82 min but run in parallel, so most of the wall
  clock is inter-job queueing -- runner starvation. Slowest per-PR jobs: Code
  Coverage 25m, PINN Feature Validation 12m, Heavy Validation (absorption decay)
  10m, Build & Test (stable) 8m, Heavy Validation (kuznetsov) 7m. One pull
  request starts about 24 checks across 5 always-on workflows.
- **Against policy:** extra toolchains, heavy suites and coverage are the
  scheduled backstop, not per-PR gates; coverage is a ratchet, not a merge gate.
- **Why it matters now:** with no ruleset and no required checks, `--auto` merges
  immediately rather than enqueueing, so merges land on partial evidence
  (kwavers#742 merged on its decisive check while 20 others were queued).
  Requiring checks on top of a 100-minute pipeline would institutionalize the
  starvation, so job and queue speed comes first, then the ruleset.
- **Decomposition:** (1) move beta/nightly toolchains, heavy validation and
  coverage to schedule -- **done in #755**, 5 checks and about 51 min of job time
  off the pull-request path, nothing deleted; (1b) the feature matrix --
  `Architecture Validation` is a second always-on workflow contributing 12 of the
  pull request's checks, five of them `Build with Feature Combinations` (minimal,
  gpu, plotting, pinn, full), and policy puts that matrix on the schedule too.
  It was not folded into #755 because *which* leg gates a pull request is a
  coverage decision: `full` alone misses the
  feature-gated-code-referenced-unconditionally class that `minimal` catches.
  **Recommendation:** `minimal` and `full` on pull requests, all five on
  schedule; (2) re-measure the round-trip honestly -- the step-(2) numbers above
  span the 90-minute outage, so a clean re-measure is owed on a
  normally-serviced queue; (3) wire the remaining affected-scope checks as
  required status checks and let auto-merge enqueue, tracked as KW-CI-115.
- **Held, not merged:** #755 is the one change class whose only real gate is CI
  itself. Its check count fell from about 24 to 19 at push.
- **Acceptance:** a pull request's verification round-trip is measured after (1)
  and (1b), the moved jobs still run on schedule with unchanged commands and
  budgets, and no check is deleted.

## KW-SIM-TEST-COMPILE-GRAPH — Reduce simulation test build latency [patch] [perf] — in-progress

| ID | Outcome | Class | Owner | Scope |
|----|---------|-------|-------|-------|
| KW-SIM-TEST-COMPILE-GRAPH | Reduce the cold compile/link cost of the GPU-enabled simulation and Python test harnesses while retaining all value-semantic tests. | [patch] [perf] | Codex | `kwavers-simulation` and `kwavers-python` dependency/feature graphs, test-ID census, build-timing instrumentation, focused CI/PM evidence |

- **Entry evidence:** isolated cold runs spent 2m37s compiling the GPU-enabled
  simulation graph and 2m15s the Python graph, while their actual test execution
  took 0.617s and 1.661s. Cargo package attribution found 431 of 444 Python
  normal/dev packages in the simulation graph, so separate invocations rebuild
  the shared graph instead of paying for distinct tests.
- **Delivered:** `.cargo/config.toml` now owns `cargo test-gpu-consumers`, one
  Nextest invocation over both packages and their existing GPU features. The
  combined selection is exactly the 128-ID union of the two censuses (107
  simulation + 21 Python) with no additions or omissions, and it passed 127/127
  runnable tests with one declared skip after a 1m28s shared-target build.
  Repeated feature forwarding is already unioned by Cargo, so no manifest surgery
  is justified by this evidence.
- **Rejected as a CI gate, by its own acceptance clause:** `compile-timing.yml`
  (`workflow_dispatch`, cache-free build on a fresh hosted runner) was dispatched
  twice and measured the combined graph at **579 s and 592 s on 4 cores** against
  the 181 s bound -- 3.2x over. The alias therefore stays a local instrument.
- **Remaining:** `cargo build --timings` over the same graph to attribute the
  579 s and select one dominant unit to shrink. Controlled cold timings cannot be
  collected from the shared warm target without deleting shared derived state or
  forking the one-cache policy, and no timing reduction is claimed from a warm
  run.
- **Acceptance:** preserve the exact 128-ID union, features, assertions, profile,
  cache policy and timeout bounds; a named dominant unit is shrunk, or the
  consolidation stays local and this item closes as rejected.

## KW-BOOK-116 — Make the book gate execute a Rust oracle [patch] — review

| ID | Outcome | Class | Owner | Scope |
|----|---------|-------|-------|-------|
| KW-BOOK-116 | The published Kwavers book executes one deterministic Rust oracle and the shared workflow builds the exact package before `mdbook test`. | [patch] | Codex | `.github/workflows/book-pages.yml`, `docs/book/examples/basic_simulation.md`, this item |

- Carried by the removed Status cell: implementation complete; hosted verification pending.

- Non-goals: no changes to the concurrently modified transducer files and no
  expansion of the Python binding or ensemble-model contract.
- Acceptance: `mdbook test docs/book` executes the analytical CFL oracle;
  `mdbook build docs/book` passes; and the workflow pins the package and library
  target explicitly.
- Claim committed on branch `fix/kwavers-python-book-boundary`; the unrelated
  dirty transducer files remain outside this scope.
- Local evidence at `546949cc0`: locked `kwavers` package build, `cargo fmt
  --check`, `mdbook test docs/book`, `mdbook build docs/book`, and staged diff
  checks pass. The standalone link checker still reports pre-existing source
  links outside the book root and incomplete mathematical-link parsing; those
  remain separate documentation cleanup work.

## KW-DIST-QUEUE-2026-08-20 — close distributed queue completion and deadline contracts [patch] — review

| ID | Outcome | Class | Owner | Scope |
|----|---------|-------|-------|-------|
| KW-DIST-QUEUE-2026-08-20 | Make distributed queue completion include executing tasks, replace worker polling with scheduler notification, and reject timestamp overflow. | [patch] | Codex | `crates/kwavers-analysis/src/distributed/{queue,scheduler,task,mod}.rs`, this item, `gap_audit.md`, `CHANGELOG.md` |

- Carried by the removed Status cell: implementation complete; hosted verification pending.

- Acceptance: `wait_all` waits for queued and executing tasks; workers wait on
  the scheduler condition variable; deadline overflow returns typed
  `KwaversError::InvalidInput`; focused value-semantic tests cover active-task
  completion and both queue/item overflow boundaries; exact-head hosted checks
  pass before merge.
- Local evidence: Rust 1.97.0 rustfmt check, offline overlay `cargo check
  -p kwavers-analysis --lib`, strict offline Clippy, doctest, and rustdoc pass.
  Nextest run `7bdc39ee-be1b-47ae-b486-423362162176` passes all 17 distributed
  tests (727 skipped). Overlay Cargo lock churn was restored after each local
  run; no lockfile change is part of this item.
- Locked boundary: `cargo nextest run --locked -p kwavers-analysis --lib
  distributed` stops before compilation because the Atlas development overlay
  requires a lock rewrite for local patches. Hosted CI remains the locked
  acceptance gate.
- Hosted delivery: PR [#427](https://github.com/ryancinsight/kwavers/pull/427)
  carries exact head `073a5adbbdb22e3e88c161a0f2009d52376115ff`; the PR is ready
  for review and its synchronize-triggered CI/Architecture runs are pending.
- Non-goals: no changes to the existing peer-owned Kwavers medium, physics,
  visualization, workflow, lockfile, or documentation edits.

## KW-LINT-1 — Burn down the clippy debt baseline [patch] — todo

| ID | Outcome | Class | Owner | Scope |
|----|---------|-------|-------|-------|
| KW-LINT-1 | The debt block in `[workspace.lints.clippy]` is empty, so the Atlas floor is enforced whole. | [patch] | Codex | Workspace lint ratchet; `unused_self`, `return_self_not_must_use`, and `unnecessary_literal_bound` are clean |

- Context: the clippy floor landed in #423. 21 of 24 crates already declared
  `[lints] workspace = true`, but no `[workspace.lints.clippy]` table existed to
  inherit, so the plumbing was live and the floor was empty.
- Baseline at adoption, `cargo clippy -p kwavers --features pinn --lib` (which
  lints every path member, i.e. the whole workspace): **1390** warnings --
  `unused_self` 257, `unwrap_used` 255, `missing_panics_doc` 225,
  `missing_errors_doc` 119, then a ~200-warning tail across 25 lints.
- Acceptance: per lint, drive the count to zero **in production code**, then
  delete that lint's line from the debt block so the floor re-enables it. The
  count is the ratchet and only decreases. Suppressing a site with `#[expect]`
  counts only where the lint is genuinely inapplicable and the reason says why --
  `unwrap_used` inside `#[test]` is the sanctioned case, on a production path it
  is not.
- The ratchet as it stands, whose authoritative form is `Cargo.toml`'s
  `Brownfield debt baseline (KW-LINT-1)` block: `unwrap_used` 223+ in
  `kwavers-solver` alone, `pub_underscore_fields` 15, `struct_excessive_bools`
  12, `inline_always` 10, `assigning_clones` 9
  (`case_sensitive_file_extension_comparisons` 6, `wildcard_imports` 5, plus
  `print_stderr`). `unwrap_used` and `print_stderr` are `restriction` lints, not
  members of `all`/`pedantic`, so a measured count there reads the lint level
  rather than the code -- do not promote either to `deny` on a zero measured
  while it is allowed.
- Burned to zero and so removed from the debt block: `missing_errors_doc`,
  `missing_panics_doc`, `unused_self`, `return_self_not_must_use`,
  `unnecessary_literal_bound`, `unnecessary_semicolon`,
  `stable_sort_primitive`, `unchecked_time_subtraction`,
  `self_only_used_in_recursion`, `bool_to_int_with_if`,
  `large_types_passed_by_value`, `implicit_hasher`,
  `from_iter_instead_of_collect`, `missing_fields_in_debug`,
  `needless_for_each`.
- The ~170 per-slice records this item used to carry (#447-#476, #751-#770:
  "X slice merged in PR #N") were a ledger of merged increments, which is what
  git holds; recover any of them with `git log --grep='^Item: KW-LINT-1'`.
- Not to be closed by extending a template: the ~840 contentless `# Errors`
  sections on genuinely fallible functions are tracked as
  [KW-ERRORS-DOCS-ARE-TEMPLATE-OUTPUT-2026-09-09](#kw-errors-docs-are-template-output-2026-09-09),
  and zeroing a count that way would move the measure, not the code.

## ATLAS-KWAVERS-HEPHAESTUS-FDTD-107 — Route collocated FDTD through Hephaestus [minor] [arch] — blocked 2026-08-18

| ID | Outcome | Class | Owner | Scope |
|----|---------|-------|-------|-------|
| ATLAS-KWAVERS-HEPHAESTUS-FDTD-107 | Delete the consumer-owned collocated raw-WGPU FDTD path and validate the real Hephaestus `Fdtd3dOps` provider against an independent f32 CPU stencil. | [minor] [arch] | Codex | `crates/kwavers-gpu/src/{gpu,validation/gpu_cpu_equivalence}/`, affected allocation test, PM artifacts |

- Carried by the removed Status cell: implementation complete; blocked on Apollo 0.27.0 default.

- Acceptance: provider-owned typed buffers and kernels execute velocity then
  pressure updates; the CPU oracle uses the same mathematical contract in
  native f32 precision; provider acquisition and dispatch failures remain
  explicit; no CPU fallback or consumer-owned collocated FDTD shader remains;
  focused Nextest, feature-enabled check/Clippy, doctests, and the Hephaestus
  provider contract pass.
- Upstream: Hephaestus PR #213, exact source head
  `7bc9944852a6ba92d4ff265b9fff9bc8c81e3567`, adds the typed contract and
  sequential-step differential coverage. Kwavers consumer delivery is on
  `codex/kwavers-gpu-visualization-104` at `2295bfff7`; the previous
  exact-head benchmark regression passes. The final matrix stops at Cargo
  resolution because Kwavers main requires Apollo `0.27.0` while Apollo
  default remains `0.26.0`.
- Workflow evidence: the prior CI benchmark lane and Test Suite Coverage lane
  were cancelled at their job limits while blocked in `apt-get update`;
  `4e11cf555` applies bounded retries and HTTP(S) timeouts, and `0a3446dac`
  adds explicit 8-minute process deadlines with 30-second termination grace to
  all Ubuntu package-install steps. These workflow changes do not change
  benchmark inputs or production code.
- Blocker evidence: CI run `32099296963` fails beta resolution in job
  `95596582400`; architecture run `32099297012` fails clean-architecture
  resolution in job `95596582553`. Apollo PR #104 source `38192bed` has the
  required provider version but its Rust workspace run `32096086258` and
  benchmark run `32096086273` fail on stale lock/measurement requirements.
  Re-open after Apollo `0.27.0` lands on default; do not add a fallback to
  Apollo `0.26.0`.
- Residuals: the separate pressure-only `gpu::compute::fdtd_gpu` dispatcher
  and the disconnected f64 solver accelerator seam remain tracked boundaries;
  neither is represented as proof of this provider integration.

## ATLAS-KWAVERS-HEPHAESTUS-VIS-104 — Reject uninitialized GPU visualization [patch] — review 2026-08-17

| ID | Outcome | Class | Owner | Scope |
|----|---------|-------|-------|-------|
| ATLAS-KWAVERS-HEPHAESTUS-VIS-104 | GPU-enabled multi-field rendering returns the existing typed feature/resource error until the renderer and data pipeline are initialized; initialized GPU rendering and CPU fallback retain every field. | [patch] | Codex | `crates/kwavers-analysis/src/visualization/engine/mod.rs`, `crates/kwavers-analysis/src/visualization/mod.rs`, `gap_audit.md`, `CHANGELOG.md` |

- Carried by the removed Status cell: implementation complete; exact-head hosted verification pending 2026-08-17.

- Acceptance: valid multi-field input never returns `Ok(())` when GPU resources are absent; initialized GPU rendering processes every field; invalid field-count input and non-GPU fallback remain value-semantic.
- Non-goals: no CPU fallback behind the GPU feature, no FDTD/provider changes, and no renderer ownership changes in Hephaestus or Leto.
- Gate: source head `b275b7115` passes the required feature-enabled hosted matrix; the PM-only follow-up head must rerun the same gate before closure. Local execution remains blocked by the shared Atlas overlay's stale Asclepius checkout requiring `aequitas ^0.1.0` versus `0.2.0`.

## KWAVERS-COUPLING-CONTRACT-001 — Medium-aware field-coupling inputs [minor] — todo

| ID | Outcome | Class | Owner | Scope |
|----|---------|-------|-------|-------|
| KWAVERS-COUPLING-CONTRACT-001 | Add a typed medium-property provider to `MultiphysicsFieldCoupler` for photoelastic, optical-absorption, and frequency-dependent acoustic-absorption coefficients; retain scalar extraction only at the field-update boundary. | [minor] | Codex | `crates/kwavers-solver/src/multiphysics/field_coupling/`, direct callers/tests, `gap_audit.md` |

The current field-coupler API accepts only collocated field volumes, so its
nominal water/tissue coefficients are real defaults rather than hidden input
or a silent fallback. The specialized `AcousticOpticalSolver` already accepts
a caller-supplied photoelastic coefficient. The API extension above is the
remaining provider-owned work; this cleanup removes misleading placeholder
markers without changing the numerical contract.

## KW-MAT-043 — Direct FWI L-BFGS provider ownership [patch] [arch] — in-progress 2026-08-11

- Owner: Codex. Scope: the two FWI production callers, the focused FWI
  L-BFGS tests, the existing `kwavers-math` optimization path, and the
  synchronized PM artifacts.
- Outcome: use `leto_ops::application::optimization::LbfgsMemory` directly in
  every in-scope FWI caller. The consumer owns no adapter, fallback, or
  duplicate memory implementation.
- Baseline: `crates/kwavers-math/src/optimization/lbfgs.rs` is already absent
  from the current tree and was deleted by the prior math SSOT migration;
  this item must not recreate it. The existing public `kwavers-math`
  re-export remains outside this FWI caller cutover.
- Acceptance: direct provider imports replace all FWI-local imports; focused
  tests prove value-equivalent two-loop directions after eviction and a hard
  memory bound; affected package formatting, Nextest, doctests, strict
  Clippy, and the consumer check run against the delivered source or record
  the exact infrastructure blocker.

## KWAVERS-AEQ-MET-69 — Type B-mode scan-conversion geometry [major] [arch] — in-progress 2026-08-06

- Owner: Codex; scope: `crates/kwavers-analysis/src/signal_processing/b_mode/`,
  direct tests, ADR 105, and the synchronized gap audit. The dense RF/image
  arrays remain scalar numerical storage.
- Outcome: beam angles use Aequitas `Angle` and apex/range/grid extents use
  `Length`. Conversion to radians/metres is confined to the trigonometric,
  interpolation-index, and Cartesian-raster formula boundaries. No compatibility
  fields or forwarding constructors remain.
- Eunomia: B-mode geometry is real-valued. A complex RF representation, if
  introduced at an upstream signal boundary, retains the existing observable
  signal unit; no imaginary angle or length unit is introduced.
- Acceptance: all direct constructors and tests compile against typed geometry;
  the bright-pixel and out-of-sector scan-conversion regressions preserve their
  value semantics; package check, strict Clippy, Nextest, doctests, Rustdoc,
  formatting, and raw-public-signature scans pass.

## KWAVERS-AEQ-MET-66 — Type thermal-diffusion quantities [major] [arch] — blocked 2026-08-05

- Owner: Codex; scope: public thermal-diffusion parameters and integration-time
  contracts, their direct solver/orchestrator callers, Python and simulation
  serialization boundaries, ADR 103, and synchronized PM artifacts. Thermal
  dose storage and thresholds remain CEM43 domain values rather than being
  mislabeled as SI time.
- Outcome: perfusion uses `ReciprocalTime`, blood density uses `MassDensity`,
  blood heat capacity uses `SpecificHeatCapacity`, arterial temperature uses
  `ThermodynamicTemperature`, relaxation and integration steps use `Time`, and
  scalar extraction occurs only at storage and numerical formula boundaries.
  The Plugin trait's raw host timestep remains an explicit execution-boundary
  conversion to `Time`.
- Eunomia: thermal quantities are real SI observables. Any complex/quadrature
  value in adjacent response code remains one existing observable unit; no
  imaginary SI temperature, time, density, or dose unit is introduced.
- Acceptance: every direct constructor and update caller compiles against the
  typed contract without compatibility wrappers; analytical thermal and CEM43
  value regressions pass; strict package gates, Nextest, doctests, Rustdoc,
  formatting, and raw-unit scans pass.
- Evidence: overlay Nextest passes 2,404/2,404 with five configured skips;
  strict package Clippy passes for physics, solver, simulation, and Python;
  physics doctests pass 8/8 with 4 ignored, solver doctests pass 4/4 with 8
  ignored, and simulation doctests pass 4/4 with 2 ignored. The Python crate
  is a `cdylib` with no library doctest target. The implementation-time lock
  evidence resolved against Eunomia 0.7; the dependency-ordered follow-up now
  regenerates the clean graph against Ritk `cfeebc7` and Eunomia 0.8.
- Delivery residual: the historical `RUSTSEC-2026-0235` path is resolved by
  the Ritk Eunomia 0.8 cutover and the top-level rkyv 0.8 update recorded in
  `KWAVERS-AEQ-MET-67`. The clean all-features lock contains rkyv 0.8.17 only;
  no advisory ignore or feature narrowing is used. The clean package
  Nextest build is separately blocked by the Windows GNU linker missing
  `libLIBCMT.a` and `libOLDNAMES.a` for unrelated top-level test binaries;
  focused clean physics tests pass. The hosted exact-head matrix remains
  pending.

## KW-RELEASE-CRATES-01 — Publish the Rust package closure [patch] — blocked

- **Blocker:** release authority. Publication is the release delivery state and
  needs the user's explicit authorization; no agent grant covers it.
- **Re-open trigger:** the user authorizes publishing the Rust closure.
- **Preparation verified 2026-09-08:** the workspace holds exactly 23
  publishable packages with `kwavers-python` the sole `publish = false`
  member -- matching this item's scope and non-goals -- and no publishable
  crate depends on it, so the publishable set is dependency-closed.
- **Not yet true:** `crates.io` indexes none of them (`kwavers`, `kwavers-core`,
  `kwavers-solver`, `kwavers-physics`, `kwavers-alloc-probe` all return no
  published version), so the acceptance clause "every version is indexed" is
  outstanding and will be until the release runs.

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

## KW-SOL-054 — Repair AVX-512 FDTD layout contract [patch] — todo

- **Cannot be verified here, and is not verified anywhere.** The acceptance
  requires matching analytical reference fields *on an AVX-512 host*. This
  machine reports `avx512f=false` (hybrid Core Ultra; AVX-512 is fused off), so
  the seven `avx512` tests pass on the scalar fallback.
- Two of them assert nothing when they do: `processor_or_skip` returns `None`
  without the feature and the bodies of
  `pressure_update_keeps_interior_constant_for_uniform_field` and
  `velocity_update_matches_linear_pressure_gradient` early-return. The helper
  is right to separate a genuine environment limit from a defect -- it asserts
  the feature really is absent before skipping -- but a green run on this host
  is evidence about the fallback, not the kernels.
- **No CI host runs them either:** no workflow sets an AVX-512 runner or
  `target-feature`, so the kernels ship unexercised on every machine class
  available to this project.
- **Acceptance:** the value-semantic cases run somewhere that reports
  `avx512f=true` -- a self-hosted runner, or an emulator (SDE-class) invoked by
  a scheduled job -- and the run is recorded; until then the kernels carry no
  behavioral evidence and the item stays open.
- **Emulator route attempted 2026-09-09, both paths unavailable from this host**
  (the user authorised the SDE download; the obstacle is not permission):
  - Intel SDE: `downloadmirror.intel.com` resolves to IPv6-only CloudFront and
    resets over IPv6; over IPv4 it answers, but every Intel page describing the
    package returns an Akamai `Access Denied` to this client, so the current
    package URL cannot be discovered. Guessing a version path or taking the
    binary from an unofficial mirror is not an acceptable substitute for a
    tool that will execute test binaries.
  - WSL + `qemu-user` (`-cpu` reporting AVX-512, from Ubuntu's own repositories,
    needing no third-party binary): the registered Ubuntu distro fails to start
    -- `Failed to attach disk ... ext4.vhdx: The system cannot find the path
    specified`. The distro is registered but its virtual disk is gone.
  - **Smallest unblocking action:** place an `sde`/`sde64` on `PATH` (a manual
    download through Intel's click-through licence), or repair the WSL distro,
    or register the self-hosted AVX-512 runner. Any one of the three closes
    this.

## KW-CI-053 — Update GPU PSTD parity contract [patch] — review

- Owner: Codex; scope: `crates/kwavers/tests/gpu_pstd_parity.rs` and its PM
  evidence only.
- Acceptance: ignored GPU parity tests call the provider-owned six-argument
  `GpuPstdSolver::run` API with `PstdOutputRequest::sensor_traces()` and consume
  `sensor_data`; no
  compatibility wrapper or test simplification is introduced.
- Evidence: hosted job `87936633879` gave the exact E0061/E0308 diagnostics;
  package-scoped nightly rustfmt passes after the direct call-site migration.
- Residual: focused Nextest and the fresh hosted matrix must complete.

## KW-CI-051 — Remove obsolete deployment workflow [patch] — review

- Owner: Codex; scope: `.github/workflows/deploy.yml` only.
- Acceptance: no workflow references absent deployment artifacts or invalid
  step syntax; the supported CI surface remains architecture validation,
  migration audit, and the provider-aware build/test workflow.
- Evidence: the repository contains no `Dockerfile` or `k8s` tree, and Actions
  run `29593287070` failed at workflow parsing before creating jobs. The stale
  workflow is deleted without changing simulation or deployment code.
- Driver: the workflow was inherited from the old PINN service layout and has
  no live repository inputs.

## KW-CI-052 — Lock and parse supported Cargo workflows [patch] — review

- Owner: Codex; scope: `.github/workflows/ci.yml` and
  `.github/workflows/architecture-validation.yml`.
- Acceptance: all Cargo graph-consuming commands use `--locked`, the workflow
  YAML parses, and no command is hidden behind a malformed step indentation.
- Evidence: local PyYAML parsing reports `yaml-ok`; the diff is workflow-only
  and `git diff --check` is clean. The hosted matrix is the remaining
  verification tier.
- Driver: the prior CI definition allowed live provider resolution and had
  indentation errors in the build/convergence/validation steps.

## KW-CI-050 — Restore hosted format and CUDA prerequisites [patch] — review

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

## KW-CI-049 — Align Apollo provider lock [patch] — review

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

<a id="kw-lint-047"></a>

## KW-LINT-047 — Solver all-feature lint ratchet [patch] — in-progress

- **Integrator:** claude-opus-5, claimed 2026-09-09.
- **Not met. Measured 2026-09-08**, `cargo clippy -p kwavers-solver --features
  pinn --all-targets`: **84** diagnostics -- 28 `unused_self`, 26 missing
  `# Errors`, 11 `println!`, 7 missing `# Panics`, 6 missing `#[must_use]`,
  4 `assert!` with an equality comparison, 2 doc-link/recursion findings. The
  2026-08-17 increment recorded this gate passing; it does not pass now under
  this configuration, and `println!` in a library crate is its own floor
  violation (`print_stdout` is denied there).
- **Burned so far:** all 11 `print_stdout` sites are gone -- they were
  print-debugging inside `#[cfg(test)]`, and removing them exposed five tests
  that asserted nothing that could fail: `analytic.rs` bound its analytic
  expectation to `_expected` and asserted `is_finite()`, and those oracles are
  unreachable because the model is an untrained, randomly initialised network,
  so finite differences is the only valid oracle and
  `first_deriv.rs`/`second_deriv.rs` already apply it. Point coverage moved
  there and the property-in-name-only tests were deleted (84 -> 75). A further
  twelve diagnostics with a determinate fix went in #761 (6 `#[must_use]` on
  `mut self -> Self` builders, 4 equality `assert_eq!`, 1 unquoted intra-doc
  link, and `generate_boundary_data` -- which took `&self` and never touched it,
  the receiver surviving only through its own recursive calls -- made an
  associated function with its three call sites updated), 1375 solver tests
  passing (73 -> 61).
- **Remaining 61, and why it stops there:** 28 `unused_self` is a design finding,
  not a mechanical fix. The 26 missing `# Errors` and 7 missing `# Panics` must
  not be closed by extending the template that produced
  [KW-ERRORS-DOCS-ARE-TEMPLATE-OUTPUT-2026-09-09](#kw-errors-docs-are-template-output-2026-09-09)
  -- doing so would move this number without improving anything, which is gaming
  the measure.
- **Finding for whoever takes the remainder:** `second_deriv.rs` deliberately did
  not absorb the deleted analytic points. Both sides there are finite differences
  at different step sizes (`coeus_autograd` has no double-backward), so
  disagreement scales with the fourth derivative and `REL_TOL_SECOND` is
  empirical, not derived -- the added point measured rel_err 2.59e-2 against the
  1e-2 bound. Extending that test needs a derived per-point bound first.
- **Lane note:** a peer switched this lane's branch mid-increment and built
  `chore/kwavers-manifest-row` on top of the commit, so the first increment lands
  with their driver refactor rather than its own pull request; content was
  verified intact on their branch. `fix/kw-solver-lint-ratchet` on origin had
  been left pointing at `main` and is deleted.
- **Acceptance:** the measured count reaches zero under the named configuration,
  or each survivor carries `#[expect(lint, reason = ...)]`.

## KW-IMG-045 — Frame-resolved physical I/Q ensemble [minor] — todo

- Owner: Codex; scope: consume the direct I/Q primitives from LeoNeuro using
  explicit physical scatterer states for each slow-time frame.
- Acceptance: a color/power/PW/fUS sector sequence derives its frame-to-frame
  phase from submitted scatterer position and reflectivity evolution, then
  applies the provider I/Q demodulator and complex DAS. No deterministic
  phase-animation surrogate remains in the Python reference path.
- Driver: KW-IMG-044 makes individual real-RF frames complex-I/Q capable but
  deliberately does not claim a physical slow-time evolution law.

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

## KW-CI-094 — The recurseml check errors on itself and is permanently red [patch] — todo

- Re-recorded; lost to the same merge as KW-SOL-093, and still true.
- `recurseml/analysis` reports `state: error` with `Error occurred during
  analysis`, on ten of the eleven pull requests in this stretch. The message is
  the bot failing, not a finding.
- It is not a required check, so it never blocks a merge - which is what makes it
  worth fixing. Every PR shows a red check that everyone learns to skip, and that
  is how a real failure gets waved through.
- Acceptance: the check either reports real findings, or is removed. Not left
  erroring.
- **Re-measured 2026-09-09.** `recurseml/analysis` is a *commit status* (not a
  check run) posted by the recurseml GitHub App, `state=error`,
  `description="Error occurred during analysis"`. It errored on five of the
  last six pull requests -- #753, #755, #756, #757, #758 -- which is every one
  merged today. It is the only red on those pull requests; every GitHub Actions
  check passed.
- **Not fixable from inside the repository.** There is no recurseml config
  file, and this session's token cannot enumerate or modify app installations
  (`user/installations` returns 403: the endpoint needs App-authorized auth).
  The `gh` merge-mechanics grant does not reach a third-party app installation.
- **Exact action, and it is the owner's:** GitHub Settings -> Applications ->
  Installed GitHub Apps -> recurseml -> either uninstall, or set repository
  access to exclude the members it errors on. Until then every pull request
  carries a red check that is not a finding, which is how a real failure gets
  waved through.

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

## KW-GPU-060 — Hephaestus backend-kernel ownership [major] — review

- Owner: Codex; scope: `crates/kwavers-gpu/src/backend/{provider,buffers.rs,pipeline,shaders/operators.wgsl,mod.rs,tests.rs}`, `docs/adr/039-hephaestus-backend-kernel-ownership.md`, and synchronized package metadata.
- Acceptance: `WgpuComputeProvider` uses Hephaestus typed transfer plus
  `binary_elementwise_into` and `WgslMultiStorageKernel`; the local backend
  buffer/pipeline managers and their unsafe device-pointer ownership are
  deleted; Leto remains only at the host-array boundary; WGPU value regressions
  preserve exact multiplication and affine derivatives.
- Driver: the LeoNeuro GPU path must select Hephaestus as device-execution
  owner rather than duplicating it beneath Leto host arrays.
- Decision: [`ADR-039`](docs/ADR/039-hephaestus-backend-kernel-ownership.md).
- Delivered: the provider trait surface owns the operation contracts and WGPU is
  the only real implementation -- concrete `wgpu::Buffer`/`ComputePipeline` and
  `GpuProviderContext<WgpuDevice>` are confined to it, `AcousticFieldKernel`,
  `WaveEquationGpu`, `GpuThermalAcousticSolver`, `GpuBackendBufferManager`, the
  PSTD state/pass/run/medium-update providers, `MultiGpuContext<P>` and the
  backend `GpuComputeProvider`/`GPUBackend::dispatch_*` elementwise and
  derivative dispatch all carry provider-native `leto::Array3<f32>`. Hephaestus
  CUDA implements the shared unary/binary storage-kernel traits upstream, so
  `CudaElementWiseProvider` implements the real CUDA elementwise multiplication
  family without a Kwavers-local CUDA helper.
- Evidence: offline GPU and CUDA-provider compilation; warning-denied Clippy for
  both feature sets; GPU backend Nextest 45/45, CUDA-provider backend 50/50, and
  focused provider/state selections 42/42, 52/52, 44/44 as the slices landed.
  The WGPU cases execute exact multiplication plus all three affine spatial
  derivatives on a real adapter; top-level GPU nextest passes 27/27 with 3
  ignored PSTD hardware tests skipped.
- The increment ledger this item used to carry ("X now ... and focused nextest
  passes N/N") was per-PR narrative; recover any increment with
  `git log --grep='^Item: KW-GPU-060'`.
- Residual: `WgslMultiStorageKernel` is WGPU-specific, so CUDA stays outside the
  composite provider trait until a real CUDA spatial-derivative kernel exists.

## KW-GPU-TEARDOWN — Dissolve the remaining internal `kwavers/src` into the layered crates [arch] — todo

- **Outcome:** `kwavers` is a thin facade. The bulk is GPU: a `kwavers-gpu` leaf
  owns the `ComputeBackend`/`FdtdGpuAccelerator` surfaces that stay in solver and
  consolidates all three scattered GPU paths, with wgpu-v26 bit-rot repaired as
  part of the move (user decision 2026-06-03). Filed from the
  `OPEN: kwavers-gpu extraction + internal-folder teardown` narrative, which was
  the same live work held in a section heading no id anchored.
- **Delivered:** the `kwavers-gpu` scaffold as a workspace member (`462ab1939`);
  the `kwavers::gpu` facade monolith (~5000 lines) moved behind a
  `kwavers-gpu/gpu` feature, with the facade re-exporting `kwavers_gpu::gpu`
  (`2c5acc444`); `profiling/gpu_allocator` moved; wgpu-v26 bit-rot repaired so
  `--features gpu` checks for both `kwavers-solver` and `kwavers-gpu`.
- **Blocked:** `analysis gpu` -- `bytemuck` cleared 23 errors and two trivial
  ones remain, but the genuine blocker is that the `three_dimensional` GPU
  beamformers reference `BEAMFORMING_3D_SHADER` / `DYNAMIC_FOCUS_3D_SHADER` WGSL
  constants that were never written (no source, no history). That is incomplete
  work, not bit-rot; fabricating shaders is prohibited, so the choice is
  gate-as-incomplete or author them.
- **Open:** gate `simulation/diagnostics gpu` under `--features gpu`;
  consolidate `solver::backend::gpu` and `solver::forward::{fdtd,pstd}` kernels
  plus their `*.wgsl` into `kwavers-gpu`, leaving only the traits in solver;
  re-home `architecture/layer_validation` (dev tooling -- evaluate keep/delete)
  and `infrastructure/io` (candidate `kwavers-io`, or it stays); remove or repair
  `kwavers/tests/recovery_stress_tests.rs`, which imports a `gpu::recovery`
  module that never existed; drop the dead `api` feature gates in
  `solver/inverse/pinn/ml/mod.rs` (needs pinn-feature verification).
- **State:** internal `kwavers/src` is down to ~1300 lines from 8799; the facade
  re-exports, `architecture`, `infrastructure/io` and `main.rs` remain.

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

## KW-BOOK-CH29-RESIDUALS — Chapter 29 figure regeneration and the hybrid exposure backend are still blocked [patch] — todo

- **Refiled from the `Validation Goals` ledger** (a 792-line list of closed
  2026-05 increments, deleted as report genre; the full text is in git history).
  These two threads are the open work that ledger was the only home of.
- **Figure 5 regeneration is blocked** by the nonlinear brain PyO3 allocation
  abort, so the checked-in PNG/PDF pressure column came from the controlled
  CT-frame field archive instead; the same block holds Figure 6's regeneration
  with the slower nonlinear branch profiled.
- **The hybrid PSTD/FDTD exposure backend stays blocked** until it has source,
  receiver, CT-medium, peak-pressure and memory-accounting parity tests against
  the reference path. `reference_fdtd_cpml_2d` is the only selectable backend
  today and `exposure_uses_hybrid_pstd_fdtd=false` is exported through PyO3.
- **Acceptance:** the nonlinear brain PyO3 allocation abort is fixed or bounded
  and Figures 5 and 6 regenerate from the nonlinear branch; the hybrid backend
  becomes selectable only behind its parity suite.
- **Status:** todo, not claimed; refiled 2026-09-21 by the board compaction.

## KW-FWI-PSTD-ADJOINT-RECIPROCITY — The PSTD adjoint-reciprocity check never ran [patch] — todo

- **Refiled from the deleted `Session 3 closure summary` / `Open Architectural
  Items` ledgers**, where it was the one live residual of T15b.
- **What is verified:** `FwiParameters::build_solver_for_forward` dispatches
  `SolverType::{FDTD, PSTD}` to `build_fdtd_boxed`/`build_pstd_boxed` and returns
  `Box<dyn Solver>`; the FDTD forward smoke test and the unsupported-type
  rejection test pass.
- **What is not:** the PSTD adjoint-reciprocity check, which the ledger recorded
  as "remains open (track separately if needed)" and no item ever carried.
- **Acceptance:** an adjoint-reciprocity test on the PSTD forward/adjoint pair --
  an inner-product identity or finite-difference gradient agreement -- in the
  shape the CBS dense and spectral paths already use.
- **Status:** todo, not claimed; refiled 2026-09-21 by the board compaction.
