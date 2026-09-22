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

- **Outcome:** an `# Errors` section names the condition that produces the
  error, or it does not exist. No function documents a failure it cannot have.
- **Measured 2026-09-09**, repo-wide across `crates/`:
  - **1141** occurrences of the identical line
    ``- Returns [`Err`] if an internal constraint is violated.``, in **544**
    files. A second template supplies **297** of
    ``- Propagates any [`crate::KwaversError`] returned by called functions.``
  - **300** of those sections sit on functions that **cannot fail** -- the
    signature returns neither `Result` nor `Option`. By crate: kwavers-solver
    92, kwavers-analysis 50, kwavers-physics 44, kwavers-simulation 16,
    kwavers-therapy 15, kwavers-transducer 11, kwavers-receiver 11,
    kwavers-medium 11.
  - Specimens: `pub fn num_sensors(&self) -> usize`,
    `pub fn is_leaf(&self) -> bool`, `fn backend_type(&self) -> BackendType`,
    each carrying "Returns [`Err`] if an internal constraint is violated."
- **Two defects, not one.** The 300 are *false*: they tell a reader a function
  can fail when it cannot. The remaining ~840 are *contentless*: on a genuinely
  fallible function, "an internal constraint is violated" names no invariant,
  no input range, and no caller remedy, which is what the standard asks an
  `# Errors` section for.
- **This is what satisfying a lint without reading the code looks like.**
  `clippy::missing_errors_doc` fires on a missing section, not on a useless
  one, so a template silenced it repo-wide. The open KW-LINT-047 rows counting
  26 "missing `# Errors`" must not be closed by extending this template --
  that would be gaming the measure.
- **Order:** delete the 300 false sections first (mechanical, and they are
  wrong rather than merely empty); then replace the contentless ones on
  fallible functions, per crate, with the actual failure conditions.
- **Acceptance:** zero `# Errors` sections on functions returning neither
  `Result` nor `Option`, enforced by a check in the conformance scan so the
  template cannot return; and the remaining sections name a condition rather
  than a category.
- **Guard landed in #762:** `cargo run -p xtask -- audit-errors-docs`, running
  inside `Validate Clean Architecture`. It ratchets at **350** and fails on any
  rise. The count is 350 rather than the 300 first reported: the audit
  accumulates a signature to its arrow, so it sees the wrapped signatures a
  line-at-a-time estimate skipped. Its unit tests pin the cases that make a
  naive matcher wrong -- a wrapped signature, a `where` bound mentioning
  `Result`, a unit return with no arrow.
- **All 350 removed in the same pull request, enqueued as #762.** 1040 lines
  across 226 files; every removed line is a `///` doc line and none is added,
  so the diff carries only its declared transform. Only the template's shape
  goes -- heading, bullets, and the bare `///` separator -- and a section
  followed by prose or another heading keeps everything after the bullets. The
  946 `# Panics` headings are untouched. The audit re-reads the tree with an
  independent implementation and reports zero, so `BASELINE` closes to 0;
  reintroducing one section makes it exit 1 and name the site.
- **The local gate caught a defect in the guard itself:** with `BASELINE = 0`,
  `total < BASELINE` is `usize < 0`, always false, which clippy denies as
  `absurd_extreme_comparisons`. Refused the push; the dead branch is gone.
- **Next:** the ~840 contentless sections on genuinely fallible functions,
  where "an internal constraint is violated" names no invariant, no input
  range, and no caller remedy. Per crate, replacing the category with the
  condition.


## KW-CI-PER-PR-MATRIX-STARVATION-2026-09-08 — Per-PR CI runs the scheduled matrix [patch] [ci] [perf] — in-progress

- **Integrator:** claude-opus-5 (lane `kwavers-local-gate`, branch
  `ci/kw-per-pr-matrix-starvation`); claimed 2026-09-09.
- **Held at step (2), 2026-09-09:** #755 is written and pushed but not merged.
  The queue is stopped, not slow -- nine consecutive runs queued with zero jobs
  started, the oldest waiting 64 minutes, no self-hosted runners registered
  (atlas `ATLAS-RUNNER-STARVATION-2026-09-02`). Two consequences: the step-(2)
  round-trip cannot be measured, and #755 is the one change class the local
  gate cannot stand in for -- its only real gate is CI itself -- so it is held
  rather than admin-merged on evidence that does not exist. Check count is
  already visible though: 19 at push against about 24 before.
- **User decision 2026-09-09:** proceed on local evidence rather than wait.
  Code items are verified locally and merged on that evidence (#756, #757, #758
  landed this way while the queue was stopped). #755 alone stays held, because
  a workflow change's only real gate is CI itself.
- **Step (1) merged in #755, and step (2) measured 2026-09-09 20:55** once the
  queue resumed. Job count is the clean number: a `main` run without the change
  carries 15 job entries (14 executed), the pull-request run carrying it
  executes **9** -- `Code Coverage`, both `Heavy Validation` suites, and the
  `beta`/`nightly` matrix legs are gone from the pull-request path, exactly as
  designed, and all four still exist for the nightly schedule. Wall clock is
  *not* a usable comparison for this run: it spans the 90-minute outage
  (17:30 created, 19:25 finished), so a clean round-trip re-measure is owed on
  a normally-serviced queue.
- **Remaining:** step (3), required status checks, tracked as KW-CI-115.
- **Outcome:** the pull-request path runs the affected-scope checks; the full
  matrix (extra toolchains, heavy validation, coverage) moves to the scheduled
  selection-drift backstop, bringing verification round-trip toward the
  five-minute job target.
- **Measured 2026-09-08, `main` runs:** CI/CD Pipeline 94 and 114 min wall
  clock, Architecture Validation 75 min, Deploy mdBook 64 and 95 min. Job
  runtimes in run `34257329545` sum to about 82 min but run in parallel, so
  most of the wall clock is inter-job queueing, not work: runner starvation.
- **Slowest per-PR jobs:** Code Coverage 25m, PINN Feature Validation 12m,
  Heavy Validation (absorption decay) 10m, Build & Test (stable) 8m, Heavy
  Validation (kuznetsov) 7m, Build & Test (beta) 5m, (nightly) 4m.
- **Against policy:** extra toolchains, heavy suites, and coverage are the
  scheduled backstop, not per-PR gates; coverage is a ratchet, not a merge
  gate. One pull request currently starts about 24 checks across 5 always-on
  workflows.
- **Why it matters now:** with no ruleset and no required checks, `--auto`
  merges immediately rather than enqueueing, so there is no queue to absorb
  the latency — merges land on partial evidence (kwavers#742 merged on its
  decisive check while 20 others were queued). Requiring checks on top of a
  100-minute pipeline would institutionalize the starvation instead of curing
  it, so job and queue speed comes first, then the ruleset.
- **Decomposition:** (1) move beta/nightly toolchains, heavy validation, and
  coverage to schedule -- **done in #755**, 5 checks and about 51 min of job
  time off the pull-request path, nothing deleted; (1b) the feature matrix,
  below; (2) re-measure round-trip; (3) wire the remaining affected-scope
  checks as required status checks and let auto-merge enqueue.
- **(1b) found while delivering (1):** `Architecture Validation` is a second
  always-on workflow, measured at 75 min, contributing 12 of the pull request's
  checks -- five of them `Build with Feature Combinations` (minimal, gpu,
  plotting, pinn, full). Policy puts the feature matrix on the scheduled
  backstop with the rest of the full matrix, so the same move applies. It was
  not folded into #755 because *which* legs gate a pull request is a coverage
  decision, not a mechanical move: `full` alone misses the
  feature-gated-code-referenced-unconditionally class that `minimal` catches,
  and each single-feature leg can fail on its own `cfg` combination.
  **Recommendation:** `minimal` and `full` on pull requests -- the two
  endpoints -- all five on schedule.
- **Acceptance:** a pull request's verification round-trip is measured after
  (1), the moved jobs still run on schedule with unchanged commands and
  budgets, and no check is deleted.


## KW-SIM-TEST-COMPILE-GRAPH — Reduce simulation test build latency [patch] [perf] — in-progress

| ID | Outcome | Class | Owner | Scope |
|----|---------|-------|-------|-------|
| KW-SIM-TEST-COMPILE-GRAPH | Reduce the cold compile/link cost of the GPU-enabled simulation and Python test harnesses while retaining all value-semantic tests. | [patch] [perf] | Codex `01a0253c-6013-7552-99cc-36bbbcf77f6d` | `kwavers-simulation` and `kwavers-python` dependency/feature graphs, test-ID census, build-timing instrumentation, focused CI/PM evidence |

- **Entry evidence:** isolated cold runs spent 2m37s compiling the GPU-enabled
  simulation graph and 2m15s compiling the GPU-enabled Python graph; their
  actual test execution took 0.617s and 1.661s. Cargo package attribution found
  431 of 444 Python normal/dev packages in the simulation graph, so separate
  invocations rebuild the shared graph instead of paying for distinct tests.
- **Current exact selection:** isolated `cargo nextest list` reports 107
  simulation IDs and 21 Python IDs. The combined selection reports exactly
  their 128-ID set union with no additions or omissions. The canonical alias
  compiled the selected graph in 1m28s on the shared Windows target, then
  passed 127/127 runnable tests with one declared skip in 0.977s.
- **Implementation:** `.cargo/config.toml` now owns
  `cargo test-gpu-consumers`, one Nextest invocation over both packages
  and their existing GPU features. Repeated feature forwarding is already
  unioned by Cargo, so no manifest surgery is justified by this evidence.
- **Candidate:** source and local verification commit `5966800ba`; exact
  combined-gate output is 127 passed, one skipped after a 1m28s shared-target
  build. Standalone locked/offline metadata and the 91-source lock guard pass.
- **Acceptance:** preserve the exact 128-ID union, features, assertions,
  profile, cache policy, and timeout bounds. On two fresh equivalent four-core
  Linux runners, the combined cold compile must not exceed 181s; otherwise
  reject consolidation and use Cargo timings to select one dominant unit.
- **Remaining evidence:** controlled cold compile timings cannot be collected
  from the shared warm target without deleting shared derived state or forking
  the one-cache policy. Fresh-run confirmation and compiler critical-path/link
  attribution remain open; no timing reduction is claimed from the warm run.
- **Lease:** Codex owns fresh-run timing collection and critical-path/link
  attribution; no source region is leased while that evidence runs.
- **Takeover (2026-09-02, Claude):** the Codex claim went stale (draft PR #675
  untouched since 2026-08-31); the branch was merged with `main` (its
  draft-gating workflow hunks had already landed via #683), the alias re-run
  locally (127/127, one skip), and #675 merged on a green run. The alias is a
  local gate only — no CI job invokes it. `compile-timing.yml`
  (`workflow_dispatch`) now produces the missing evidence: a cache-free build
  of the shared graph on a fresh hosted runner, its seconds in the job
  summary against the 181 s bound. Next: dispatch it twice and record both
  numbers here; adopt the alias in CI only if both fit.
- **Hosted cold-compile evidence (2026-09-02, `compile-timing.yml` run
  33598013146, PR #688):** the combined `cargo test-gpu-consumers` graph
  cold-compiled in **579 s on a 4-core hosted runner** against the 181 s
  acceptance bound (3.2x over); the second dispatched run, 33598014974,
  measured 592s on 4 cores. Per the acceptance
  clause the consolidation is **rejected as a CI gate**: the alias stays as a
  local instrument, and the next increment is `cargo build --timings` over
  the same graph to attribute the 579 s and select one dominant unit to
  shrink (a DoR sub-item once the timings are collected).


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
| KW-LINT-1 | The debt block in `[workspace.lints.clippy]` is empty, so the Atlas floor is enforced whole. | [patch] | Codex | Workspace lint ratchet; `unused_self`, `return_self_not_must_use`, and `unnecessary_literal_bound` are clean; remaining debt lines are tracked below |

- Context: the clippy floor landed in #423. 21 of 24 crates already declared
  `[lints] workspace = true`, but no `[workspace.lints.clippy]` table existed for them to
  inherit, so the plumbing was live and the floor was empty. The floor is now the canonical
  apollo template; what kwavers trips today sits in a counted debt block beneath it.
- Baseline at adoption, measured by `cargo clippy -p kwavers --features pinn --lib` (which
  lints every path member, i.e. the whole workspace): **1390** warnings. `unused_self` 257,
  `unwrap_used` 255, `missing_panics_doc` 225, `missing_errors_doc` 119 — 62% of the total —
  then a ~200-warning tail across 25 lints.
- Acceptance: per lint, drive the count to zero **in the production code**, then delete that
  lint's line from the debt block so the floor re-enables it. The count in the comment is the
  ratchet; it only decreases. Suppressing a site with `#[expect]` counts only where the lint
  is genuinely inapplicable and the reason says why — `unwrap_used` inside `#[test]` is the
  sanctioned case, `unwrap_used` on a production path is not.
- Sequencing: `missing_panics_doc` and `missing_errors_doc` (344 combined) are documentation
  the panic/error surface should carry anyway and burn down mechanically per crate.
  `unwrap_used` (255) is the one with real correctness content and wants the panic-policy
  treatment (`?`, `ok_or_else`, or `expect("invariant: …")`), not a mass rewrite.
  `unused_self` (257) is the one to read before acting: it often marks a method that should
  be an associated function, but sometimes marks a seam deliberately taking `&self`.
- 2026-08-21 measurement: `kwavers-signal` production source is already clean when
  `missing_errors_doc` is re-enabled (`cargo clippy -p kwavers-signal --lib --no-deps
  -- -D clippy::missing_errors_doc`). The first actionable slice is therefore the three
  missing error contracts in `kwavers-solver`; acceptance is a strict package Clippy run
  with that lint re-enabled, the 901-test native inventory passing, solver doctests passing,
  and no workspace debt-block edit until the measured workspace count reaches zero.
- 2026-08-21 solver slice merged in PR #447 (`c3f480612`); the next measured slice is the
  20 `missing_errors_doc` sites in `kwavers-physics` production source. Its acceptance is
  strict package Clippy with the lint re-enabled, the package nextest gate, doctests, and
  warning-denied rustdoc; the workspace debt block remains unchanged.
- 2026-08-21 physics slice merged in PR #448 (`a6f3b08c0`); the next workspace measurement
  found 15 `missing_errors_doc` sites in `kwavers-driver` production source. Its acceptance
  is strict package Clippy with the lint re-enabled, the package nextest gate, doctests, and
  warning-denied rustdoc; the workspace debt block remains unchanged.
- 2026-08-21 driver slice merged in PR #449 (`463e10ac6`); the next workspace measurement
  found three `missing_errors_doc` sites in `kwavers-math` production source. Its acceptance
  is strict package Clippy with the lint re-enabled, the package nextest gate, doctests, and
  warning-denied rustdoc; the workspace debt block remains unchanged.
- 2026-08-21 math slice merged in PR #451 (`d8b6afa86`); the next workspace measurement
  found nine `missing_errors_doc` sites in `kwavers-transducer` production source. Its
  acceptance is strict package Clippy with the lint re-enabled, the package nextest gate,
  doctests, and warning-denied rustdoc; the workspace debt block remains unchanged.
- 2026-08-21 transducer slice merged in PR #452 (`1ab5d9b5b`); the next workspace measurement
  found one `missing_errors_doc` site in the ultrasound frequency-domain FWI API. Its
  acceptance is strict package Clippy with the lint re-enabled, the package nextest gate,
  doctests, and warning-denied rustdoc; the workspace debt block remains unchanged.
- 2026-08-21 physics FWI slice merged in PR #453 (`bfa06326f`); the next workspace measurement
  found one `missing_errors_doc` site in the solver frequency-domain FWI operator. Its
  acceptance is strict package Clippy with the lint re-enabled, the package nextest gate,
  doctests, and warning-denied rustdoc; the workspace debt block remains unchanged.
- 2026-08-21 solver FWI slice merged in PR #454 (`f9a2e3841`); the next workspace measurement
  found 11 `missing_errors_doc` sites in `kwavers-simulation` dispatch modules. Its acceptance
  is strict package Clippy with the lint re-enabled, the package nextest gate, doctests, and
  warning-denied rustdoc; the workspace debt block remains unchanged.
- 2026-08-21 simulation dispatch slice merged in PR #455 (`61f07dd92`); the next workspace
  measurement found 25 `missing_errors_doc` sites in `kwavers-diagnostics` reconstruction
  modules. Its acceptance is strict package Clippy with the lint re-enabled, the package
  nextest gate, doctests, and warning-denied rustdoc; the workspace debt block remains
  unchanged.
- 2026-08-21 diagnostics reconstruction slice merged in PR #456 (`e4bb154dc`); the next
  workspace measurement found 20 `missing_errors_doc` sites in `kwavers-therapy` therapy
  modules. Its acceptance is strict package Clippy with the lint re-enabled, the package
  nextest gate, doctests, and warning-denied rustdoc; the workspace debt block remains
  unchanged.
- 2026-08-21 therapy slice merged in PR #457 (`1d86d6436`); the next workspace measurement
  found 11 `missing_errors_doc` sites in `kwavers-python`, plus four `unused_self` and two
  `doc_markdown` warnings needed to make its strict package gate warning-clean. Its acceptance
  is strict package Clippy with the lint re-enabled and `-D warnings`, the package nextest
  gate, doctests, and warning-denied rustdoc; the workspace debt block remains unchanged.
- 2026-08-21 Python binding slice merged in PR #458 (`63fdcd37b`); the workspace
  `missing_errors_doc` measurement is now zero, so its debt-block line was removed. The next
  `missing_panics_doc` measurement found 15 sites: 11 in `kwavers-math` and four in
  `kwavers-driver`. Its acceptance is strict package Clippy with the lint re-enabled, the
  package nextest gate, doctests, and warning-denied rustdoc; the debt comment is ratcheted to
  the measured 15 until both packages are complete.
- 2026-08-21 math panic-contract slice merged in PR #459 (`30b5b457e`); the next workspace
  measurement found four `missing_panics_doc` sites in `kwavers-driver` production source.
  This driver slice is now claimed; its acceptance is strict package Clippy with the lint
  re-enabled, the package nextest gate, doctests, and warning-denied rustdoc. The debt comment
  remains at 15 until the driver slice completes.
- 2026-08-21 driver panic-contract slice merged in PR #460 (`63b3de255`); the next workspace
  measurement found three `missing_panics_doc` sites in `kwavers-signal` production source.
  This signal slice is now claimed with the same strict package Clippy, nextest, doctest, and
  warning-denied rustdoc acceptance gates.
- 2026-08-21 signal panic-contract slice merged in PR #461 (`e2c39184b`); the next workspace
  measurement found two `missing_panics_doc` sites in `kwavers-field` and 26 in
  `kwavers-medium`. The field slice is now claimed; its acceptance is the same strict package
  Clippy, nextest, doctest, and warning-denied rustdoc gate set.
- 2026-08-21 field panic-contract slice merged in PR #462 (`40b78079d`); the next workspace
  measurement leaves 26 `missing_panics_doc` sites in `kwavers-medium` production source.
  This medium slice is now claimed with the same strict package Clippy, nextest, doctest, and
  warning-denied rustdoc acceptance gates.
- 2026-08-21 medium panic-contract slice merged in PR #463 (`94214342c`); the next workspace
  measurement found one `missing_panics_doc` site in `kwavers-boundary` and six in
  `kwavers-transducer`. The boundary slice is now claimed with the same strict package Clippy,
  nextest, doctest, and warning-denied rustdoc acceptance gates.
- 2026-08-21 boundary panic-contract slice merged in PR #464 (`060721ef0`); the next workspace
  measurement leaves six `missing_panics_doc` sites in `kwavers-transducer` production source.
  This transducer slice is now claimed with the same strict package Clippy, nextest, doctest,
  and warning-denied rustdoc acceptance gates.
- 2026-08-21 transducer panic-contract slice merged in PR #465 (`5293d5181`); the refreshed
  workspace measurement found 31 `missing_panics_doc` sites in `kwavers-physics` production
  source that were absent from the prior package inventory. This physics slice is now claimed
  with the same strict package Clippy, nextest, doctest, and warning-denied rustdoc acceptance
  gates; the debt block remains unchanged until this slice completes.
- 2026-08-21 physics panic-contract slice merged in PR #466 (`b7df7597e`); the refreshed
  workspace measurement found 69 `missing_panics_doc` sites in `kwavers-solver` production
  source. This solver slice is now claimed with the same strict package Clippy, nextest, doctest,
  and warning-denied rustdoc acceptance gates; the debt block remains unchanged until this
  slice completes.
- 2026-08-21 solver panic-contract slice merged in PR #467 (`caefaf0cc`); the refreshed
  workspace measurement found seven `missing_panics_doc` sites in `kwavers-simulation` and 38
  in `kwavers-analysis`. This simulation slice is now claimed with the same strict package
  Clippy, nextest, doctest, and warning-denied rustdoc acceptance gates.
- 2026-08-21 simulation panic-contract slice merged in PR #468 (`d72208875`); the analysis
  slice remains at 38 measured `missing_panics_doc` sites and is now claimed with the same
  strict package Clippy, nextest, doctest, and warning-denied rustdoc acceptance gates.
- 2026-08-21 analysis panic-contract slice is implemented and passes strict package Clippy,
  Nextest 744/744, doctests, and warning-denied rustdoc. The increment is ready for
  integration; the workspace debt-block line remains until a refreshed workspace measurement
  confirms the lint count reaches zero.
- 2026-08-21 analysis panic-contract slice merged in PR #469 (`a94974225`). The refreshed
  workspace measurement found 11 `missing_panics_doc` sites in `kwavers-diagnostics` and 11
  in `kwavers-therapy`; the diagnostics slice is now claimed with the same strict package
  Clippy, nextest, doctest, and warning-denied rustdoc acceptance gates.
- 2026-08-21 diagnostics panic-contract slice is implemented and passes strict package Clippy,
  Nextest 191/191, doctests, and warning-denied rustdoc. The increment is ready for integration;
  the workspace debt-block line remains until a refreshed workspace measurement confirms the
  lint count reaches zero.
- 2026-08-21 diagnostics panic-contract slice merged in PR #470 (`254f79d02`). The refreshed
  workspace measurement leaves 11 `missing_panics_doc` sites in `kwavers-therapy`; that package
  is now claimed with the same strict package Clippy, nextest, doctest, and warning-denied rustdoc
  acceptance gates.
- 2026-08-21 therapy panic-contract slice is implemented and passes strict package Clippy,
  Nextest 350/350 with one intentional skip, doctests, and warning-denied rustdoc. The package
  increment is ready for integration; the workspace debt-block line remains until a refreshed
  workspace measurement confirms the lint count reaches zero.
- 2026-08-21 therapy panic-contract slice merged in PR #471 (`de5589d0f`). The refreshed
  workspace measurement found three `missing_panics_doc` sites in the `kwavers` facade; that
  slice is now claimed with the same strict package Clippy, nextest, doctest, and warning-denied
  rustdoc acceptance gates.
- 2026-08-21 facade panic-contract slice is implemented and passes strict package Clippy,
  Nextest 39/39, doctests, and warning-denied rustdoc. The increment is ready for integration;
  the workspace debt-block line remains until a refreshed workspace measurement confirms the
  lint count reaches zero.
- 2026-08-21 facade panic-contract slice merged in PR #472 (`36120168a`). The strict workspace
  `missing_panics_doc` measurement is now zero, so its debt-block line is removed. The next
  ratchet slice is `unused_self`, measured at 228 sites in the current workspace baseline.
- 2026-08-21 `unused_self` measurement distributes 228 sites across 14 packages; the largest
  slice is 107 sites in `kwavers-solver`, which is now claimed with package Clippy, nextest,
  doctest, and warning-denied rustdoc acceptance gates. The debt comment is ratcheted to 228.
- 2026-08-21 solver `unused_self` work starts with the four measured kernel methods in
  `forward/bem/burton_miller/kernels.rs`; they are converted to associated functions and all
  in-repo callers are migrated. The bounded slice passes its focused native tests, doctests,
  and warning-denied rustdoc; the remaining solver sites stay in the parent queue.
- 2026-08-21 Burton-Miller slice merged in PR #473 (`54659d09d`). The refreshed `unused_self`
  measurement is 225 sites (solver 104); the next bounded slice is the three receiver-free
  helpers in `forward/elastic/swe/gpu`, now claimed with the same focused gates.
- 2026-08-21 SWE GPU slice is implemented: the two adaptive interpolation/quality helpers and
  the transfer-time helper are associated functions with all callers migrated. Focused tests,
  doctests, and warning-denied rustdoc pass; the remaining solver sites stay queued.
- 2026-08-21 SWE GPU slice merged in PR #474 (`35943e1b9`). The refreshed `unused_self`
  measurement is 222 sites (solver 101); the next bounded slice is the four receiver-free
  helpers in `forward/helmholtz/fem/assembly.rs`, now claimed with focused gates.
- 2026-08-21 Helmholtz FEM assembly slice is implemented: its four receiver-free helpers are
  associated functions with all callers migrated. Focused tests, doctests, and warning-denied
  rustdoc pass; the remaining solver sites stay queued.
- 2026-08-21 Helmholtz assembly slice merged in PR #475 (`82c01058e`). The refreshed `unused_self`
  measurement is 218 sites (solver 97); the next bounded slice is the six receiver-free helpers
  in `forward/helmholtz/fem/basis.rs`, now claimed with focused gates.
- 2026-08-21 Helmholtz FEM basis slice is implemented: its six receiver-free basis and quadrature
  helpers are associated functions with all callers migrated. Focused tests, doctests, and
  warning-denied rustdoc pass; the remaining solver sites stay queued.
- 2026-08-21 Helmholtz basis slice merged in PR #476 (`0c202f9ef`). The refreshed `unused_self`
  measurement is 212 sites (solver 91); the next bounded slice is the single receiver-free
  interpolation helper in `forward/helmholtz/fem/solver/core/interpolation.rs`.
- 2026-08-21 FEM interpolation slice is implemented: the receiver-free shape-function helper is
  an associated function with its caller migrated. Focused tests, doctests, and warning-denied
  rustdoc pass; the remaining solver sites stay queued.
- 2026-08-21 FEM interpolation slice merged in PR #477 (`e667310bf`). The refreshed `unused_self`
  measurement is 211 sites (solver 90); the next bounded slice is the receiver-free helper in
  `forward/hybrid/adaptive_selection/selector.rs`.
- 2026-08-21 adaptive selector slice is implemented: its receiver-free region extractor is an
  associated function with its caller migrated. Focused tests, doctests, and warning-denied
  rustdoc pass; the remaining solver sites stay queued.
- 2026-08-21 adaptive selector slice merged in PR #478 (`997851405`). The refreshed `unused_self`
  measurement is 210 sites (solver 89); the next bounded slice is the receiver-free helper in
  `forward/hybrid/bem_fem_enhanced/solver.rs`.
- 2026-08-21 BEM/FEM enhanced solver slice is implemented: its receiver-free frequency validator
  is an associated function with all four callers migrated. Focused tests, doctests, and
  warning-denied rustdoc pass; the remaining solver sites stay queued.
- 2026-08-21 BEM/FEM enhanced solver slice merged in PR #479 (`935c56c06`). The refreshed
  `unused_self` measurement is 209 sites (solver 88); the next bounded slice is the three
  receiver-free interpolation helpers under `forward/hybrid/coupling/interpolation/`.
- 2026-08-21 coupling interpolation slice is implemented: the three measured receiver-free
  kernels are associated functions, and their spectral/adaptive static callers are migrated
  with the dispatch surface. Coupling tests, doctests, and warning-denied rustdoc pass; the
  remaining solver sites stay queued.
- 2026-08-21 coupling interpolation slice merged in PR #480 (`0b856c227`). The refreshed
  `unused_self` measurement is 206 sites (solver 85); the next bounded slice is the five
  receiver-free helpers in `forward/hybrid/coupling/quality.rs`.
- 2026-08-21 coupling quality slice is implemented: the five measured metric helpers and their
  now receiver-free aggregation path are associated functions, with the stateful history and
  thresholds retained on `QualityMonitor`. Coupling tests, doctests, and warning-denied rustdoc
  pass; the remaining solver sites stay queued.
- 2026-08-21 coupling quality slice merged in PR #481 (`2cc11ed3e`). The refreshed `unused_self`
  measurement is 201 sites (solver 80); the next bounded slice is the five receiver-free domain
  decomposition helpers in `forward/hybrid/domain_decomposition/`.
- 2026-08-21 domain decomposition slice is implemented: the three analyzer helpers, buffer blend
  helper, and partitioner score helper are receiver-free while stateful partition thresholds and
  overlap weights remain on their owners. Hybrid tests, doctests, and warning-denied rustdoc pass;
  the remaining solver sites stay queued.
- 2026-08-21 domain decomposition slice merged in PR #482 (`e77e575d2`). The refreshed
  `unused_self` measurement is 196 sites (solver 75); the next bounded slice is the receiver-free
  helper in `forward/hybrid/pstd_sem_coupling/coupler.rs`.
- 2026-08-21 PSTD/SEM coupling slice is implemented: the receiver-free continuity residual helper
  is an associated function with its caller migrated. Hybrid tests, doctests, and warning-denied
  rustdoc pass; the remaining solver sites stay queued.
- 2026-08-21 PSTD/SEM coupling slice merged in PR #483 (`c1cc14a67`). The refreshed `unused_self`
  measurement is 195 sites (solver 74); the next bounded slice is the three receiver-free helpers
  in `forward/hybrid/validation/suite.rs`.
- 2026-08-21 hybrid validation slice is implemented: the three measured numerical helpers and the
  now receiver-free convergence path are associated functions, with configuration-dependent
  accuracy and stability methods retaining the suite receiver. Validation tests, doctests, and
  warning-denied rustdoc pass; the remaining solver sites stay queued.
- 2026-08-21 hybrid validation slice merged in PR #484 (`b2aa69e6b`). The refreshed `unused_self`
  measurement is 192 sites (solver 71); the next bounded slice is the two receiver-free helpers
  in `forward/nonlinear/kzk/harmonic_tracking/tracker.rs`.
- 2026-08-21 KZK harmonic tracker slice is implemented: waveform analysis and harmonic amplitude
  extraction are associated functions, with configuration-dependent spectrum and shock-distance
  logic retaining the tracker receiver. KZK tests, doctests, and warning-denied rustdoc pass; the
  remaining solver sites stay queued.
- 2026-08-21 KZK harmonic tracker slice merged in PR #485 (`a0b0242c2`). The refreshed `unused_self`
  measurement is 190 sites (solver 69); the next bounded slice is the receiver-free helper in
  `forward/nonlinear/westervelt_spectral/solver/mod.rs`.
- 2026-08-21 Westervelt spectral slice is implemented: the receiver-free stability helper is an
  associated function with its wave-model and regression-test callers migrated. Westervelt tests,
  doctests, and warning-denied rustdoc pass; the remaining solver sites stay queued.
- 2026-08-21 Westervelt spectral slice merged in PR #486 (`2def919c9`). The refreshed `unused_self`
  measurement is 189 sites (solver 68); the next bounded slice is the receiver-free Burton–Miller
  assembler helper in `forward/bem/burton_miller/assembler.rs`.
- 2026-08-21 Burton–Miller assembler slice is implemented: vertex-normal accumulation is an
  associated function with both matrix assembly callers migrated. Burton–Miller tests, doctests,
  and warning-denied rustdoc pass; the remaining solver sites stay queued.
- 2026-08-21 Burton–Miller assembler slice merged in PR #487 (`e2ad3411c`). The refreshed
  `unused_self` measurement is 188 sites (solver 67); the next bounded slice is the receiver-free
  PSTD derivative operator helper in `forward/pstd/derivatives/operator.rs`.
- 2026-08-21 PSTD derivative slice is implemented: output validation is an associated function
  with all three derivative callers migrated. PSTD derivative tests, doctests, and warning-denied
  rustdoc pass; the remaining solver sites stay queued.
- 2026-08-21 PSTD derivative slice merged in PR #488 (`e9e919f87`). The refreshed `unused_self`
  measurement is 187 sites (solver 66); the next bounded slice is the receiver-free WENO7 limiter
  helper in `forward/pstd/dg/shock_capturing/limiter/weno7.rs`.
- 2026-08-21 WENO7 limiter slice is implemented: smoothness-indicator evaluation is an associated
  function with all four stencil callers migrated. WENO7 tests, doctests, and warning-denied
  rustdoc pass; the remaining solver sites stay queued.
- 2026-08-21 WENO7 limiter slice merged in PR #489 (`6738c0a69`). The refreshed `unused_self`
  measurement is 186 sites (solver 65); the next bounded slice is the receiver-free helper in
  `integration/time_integration/time_scale_separation.rs`.
- 2026-08-21 time-scale separation slice is implemented: spatial derivative evaluation is an
  associated function, with the analyzer caller migrated. Focused time-integration Nextest
  `82415a86-a9ae-431e-aea4-8216e33591b4` passes 16/16 (889 solver tests skipped); the package
  doctests pass 5/5 (8 ignored), warning-denied rustdoc passes, and the focused `unused_self`
  scan is clean in `time_scale_separation.rs`. The increment is ready for integration; the
  workspace debt-block line remains until a refreshed workspace measurement confirms the
  count.
- 2026-08-21 time-scale separation slice merged in PR #490 (`5e45e9061`). The refreshed
  `unused_self` measurement is 185 sites (solver 64); the next bounded slice is the three
  receiver-free FWI constraint helpers in `inverse/fwi/time_domain/constraints.rs`.
- 2026-08-21 FWI constraints slice is implemented: model clamping, stable-timestep calculation,
  and pressure second-derivative evaluation are associated functions, with all production and
  regression-test callers migrated. Focused Nextest `6971e012-54e7-4718-a9a7-1555a618d020`
  passes 60/60 (845 solver tests skipped); the package doctests pass 5/5 (8 ignored),
  warning-denied rustdoc passes, and the focused `unused_self` scan is clean in the touched FWI
  modules. The increment is ready for integration; the workspace debt-block line remains until
  a refreshed workspace measurement confirms the count.
- 2026-08-21 FWI constraints slice merged in PR #491 (`90a305ba4`). The refreshed `unused_self`
  measurement is 182 sites (solver 61); the next bounded slice is the three receiver-free FWI
  helpers in `inverse/fwi/time_domain/forward.rs` and `gradient.rs`.
- 2026-08-21 FWI static-helper slice is implemented: self-adjoint geometry extraction, gradient
  smoothing, and total-variation gradient calculation are associated functions, with all callers
  migrated. Focused Nextest `0907780b-c689-43a4-9338-44589dba8f18` passes 60/60 (845 solver tests
  skipped); the package doctests pass 5/5 (8 ignored), warning-denied rustdoc passes, and the
  focused `unused_self` scan is clean in the touched FWI modules. The increment is ready for
  integration; the workspace debt-block line remains until a refreshed workspace measurement
  confirms the count.
- 2026-08-21 photoacoustic filter slice is implemented: the private bandpass/FBP application
  helpers, three frequency-response constructors, and FFT application kernel are associated
  functions, with all filter-core callers migrated. Focused Nextest
  `d2c98cf4-fd72-41a5-89d2-c1d715b311e2` passes 10/10 (895 solver tests skipped); the package
  doctests pass 5/5 (8 ignored), warning-denied rustdoc passes, and the focused `unused_self`
  scan is clean in `photoacoustic/filters/core.rs`. The increment is ready for integration; the
  workspace debt-block line remains until a refreshed workspace measurement confirms the count.
- 2026-08-21 photoacoustic iterative slice is implemented: OSEM iteration and the three
  receiver-free geometric helpers (linear index conversion, Euclidean distance, and solid-angle
  factor) are associated functions, with all iterative callers migrated. The broad photoacoustic
  Nextest `b1d351de-be98-4c80-9920-7f28f705ecd9` passes 10/10 (895 solver tests skipped); the
  package doctests pass 5/5 (8 ignored), warning-denied rustdoc passes, and the focused
  `unused_self` scan is clean in `photoacoustic/iterative/`. The increment is ready for
  integration; the workspace debt-block line remains until a refreshed workspace measurement
  confirms the count.
- 2026-08-21 photoacoustic iterative slice merged in PR #494 (`c21fe669b`). The refreshed
  `unused_self` measurement is 171 sites (solver 50); the next bounded slice is the two
  receiver-free utility helpers in `inverse/reconstruction/photoacoustic/utils.rs`.
- 2026-08-21 photoacoustic utility slice is implemented: Euclidean distance, back-projection
  weighting, and forward-model construction are associated functions; all algorithm callers are
  migrated, and the now-unused `Utils` field/constructor are removed. Broad photoacoustic
  Nextest `2eb920d5-2ac5-41d4-9ef0-96db12e2f2d6` passes 10/10 (895 solver tests skipped); the
  package doctests pass 5/5 (8 ignored), warning-denied rustdoc passes, and the focused
  `unused_self` scan is clean in the photoacoustic utility/algorithm modules. The increment is
  ready for integration; the workspace debt-block line remains until a refreshed workspace
  measurement confirms the count.
- 2026-08-21 photoacoustic utility slice merged in PR #495 (`b62d402f8`). The refreshed
  `unused_self` measurement is 169 sites (solver 48); the next bounded slice is the three
  receiver-free linear-algebra helpers in `inverse/reconstruction/photoacoustic/linear_algebra/`.
- 2026-08-21 photoacoustic linear-algebra slice is implemented: truncated-SVD entry, power-method
  kernels, and the TV proximal operator are associated functions, with all callers migrated.
  Broad photoacoustic Nextest `241cbbcf-f0f6-463d-b2c6-d6fc87cb3f2e` passes 10/10 (895 solver
  tests skipped); the package doctests pass 5/5 (8 ignored), warning-denied rustdoc passes, and
  the focused `unused_self` scan is clean in the photoacoustic linear-algebra/algorithm modules.
  The increment is ready for integration; the workspace debt-block line remains until a
  refreshed workspace measurement confirms the count.
- 2026-08-21 the workspace `unused_self` measurement is 166 sites, including 45 in
  `kwavers-solver`. The next bounded slice claims the ten receiver-free seismic misfit kernels in
  `inverse/reconstruction/seismic/misfit/{envelope_phase,norm_metrics,wasserstein}.rs`; the
  dispatcher remains stateful and is out of scope. Acceptance is static conversion with all
  callers migrated, focused solver Nextest, package doctests, warning-denied rustdoc, and a
  refreshed workspace count.
- 2026-08-21 seismic misfit slice is implemented: the ten measured kernels are associated
  functions, and converting them exposed four receiver-free envelope/phase dispatch methods that
  were converted in the same cohesive module. All callers are migrated; focused Nextest
  `kwavers-solver` seismic-misfit filter passes 11/11 (894 filtered), package doctests pass 5/5
  (8 ignored), warning-denied rustdoc passes, and the refreshed workspace `unused_self` count is
  156 sites (solver 35). The increment is ready for integration.
- 2026-08-21 the next `unused_self` slice claims the seismic RTM kernels in
  `inverse/reconstruction/seismic/rtm/inherent/{laplacian,propagation}.rs` and the stateless
  imaging helpers in `inverse/seismic/rtm.rs`. Stateful configuration and propagation entry
  points remain in scope only where they consume processor state. Acceptance is complete static
  conversion with all callers migrated, focused RTM tests, package doctests, warning-denied
  rustdoc, and a refreshed workspace count.
- 2026-08-21 seismic RTM slice is implemented: Laplacian, reconstruction, and stateless imaging
  kernels are associated functions with all callers migrated; stateful propagation and settings
  remain receiver-bound. Focused Nextest `kwavers-solver` seismic-RTM filter passes 8/8 (897
  filtered), package doctests pass 5/5 (8 ignored), warning-denied rustdoc passes, and the
  refreshed workspace `unused_self` count is 151 sites (solver 30). The increment is ready for
  integration.
- 2026-08-21 the next `unused_self` slice claims the unified-SIRT receiver-free norm and
  coordinate-conversion helpers in `inverse/reconstruction/unified_sirt/reconstructor.rs`.
  Reconstruction configuration and iteration state remain receiver-bound. Acceptance is static
  conversion with all callers migrated, focused unified-SIRT tests, package doctests,
  warning-denied rustdoc, and a refreshed workspace count.
- 2026-08-21 unified-SIRT slice is implemented: norm, reshape, and linear-index helpers are
  associated functions with all callers migrated; reconstruction configuration and iteration
  state remain receiver-bound. Focused Nextest `kwavers-solver` unified-SIRT filter passes 8/8
  (897 filtered), package doctests pass 5/5 (8 ignored), warning-denied rustdoc passes, and the
  refreshed workspace `unused_self` count is 147 sites (solver 26). The increment is ready for
  integration.
- 2026-08-21 the next `unused_self` slice claims the four receiver-free time-reversal helpers in
  `inverse/time_reversal/reconstruction/mod.rs`: phase conjugation, source application, backward
  propagation, and convergence measurement. Reconstruction configuration and post-processing stay
  receiver-bound. Acceptance is static conversion with all callers migrated, focused
  time-reversal tests, package doctests, warning-denied rustdoc, and a refreshed workspace count.
- 2026-08-21 time-reversal slice is implemented: phase conjugation, source application, backward
  propagation, and convergence measurement are associated functions with all callers migrated;
  reconstruction configuration and post-processing remain receiver-bound. Focused Nextest
  `kwavers-solver` time-reversal filter passes 9/9 (896 filtered), package doctests pass 5/5 (8
  ignored), warning-denied rustdoc passes, and the refreshed workspace `unused_self` count is 143
  sites (solver 22). The increment is ready for integration.
- 2026-08-21 the next `unused_self` slice claims five receiver-free field-coupling operations in
  `multiphysics/field_coupling/coupling_ops.rs`: optical/thermal and acoustic/thermal coupling,
  relaxation, gradient calculation, and strength selection. Coupling strategy, tolerance, and
  iteration state remain receiver-bound. Acceptance is static conversion with all callers
  migrated, focused field-coupling tests, package doctests, warning-denied rustdoc, and a refreshed
  workspace count.
- 2026-08-21 field-coupling slice is implemented: optical/thermal and acoustic/thermal coupling,
  relaxation, gradient calculation, and strength selection are associated functions with all
  callers migrated; coupling strategy, tolerance, and iteration state remain receiver-bound.
  Focused Nextest `kwavers-solver` field-coupling filter passes 7/7 (898 filtered), package
  doctests pass 5/5 (8 ignored), warning-denied rustdoc passes, and the refreshed workspace
  `unused_self` count is 138 sites (solver 17). The increment is ready for integration.
- 2026-08-21 exact integrated-head validation passes: Kwavers package Nextest runs the 530-test
  suite with zero failures and eight configured skips; `cargo build --offline -p kwavers --examples`
  exits 0; `mdbook test docs/book` exits 0; and `mdbook build docs/book` exits 0, writing HTML to
  `target/book`. The run regenerated tracked PNG test figures locally; those generated deltas were
  discarded and no binary fixture changes are part of the validation increment.
- 2026-08-21 the next AMR slice claims the receiver-free error-estimation criterion and
  interpolation kernels in `utilities/amr/{criteria/methods,interpolation}.rs`. AMR smoothing,
  scheme selection, and octree state remain receiver-bound. Acceptance is static conversion of
  the measured helpers plus any exposed stateless family members, all callers migrated, focused
  AMR tests, package doctests, warning-denied rustdoc, and a refreshed workspace count.
- 2026-08-21 AMR criteria/interpolation slice is implemented: curvature estimation and the full
  receiver-free prolongation/restriction helper family are associated functions with all callers
  migrated; smoothing, scheme selection, and octree state remain receiver-bound. Focused Nextest
  `kwavers-solver` AMR filter passes 11/11 (894 filtered), package doctests pass 5/5 (8 ignored),
  warning-denied rustdoc passes, and the refreshed workspace `unused_self` count is 134 sites
  (solver 13). The increment is ready for integration.
- 2026-08-21 the next AMR slice claims the receiver-free octree node/leaf counting wrappers and
  refinement nesting helper in `utilities/amr/{octree,refinement}.rs`. Tree ownership and
  refinement configuration remain receiver-bound; wavelet transforms are a separate follow-up
  because their current inverse path requires behavioral completion, not only receiver removal.
- 2026-08-21 AMR tree slice is implemented: octree node/leaf counting wrappers and refinement
  nesting enforcement are associated functions with all callers migrated; tree ownership and
  refinement configuration remain receiver-bound. Focused Nextest `kwavers-solver` AMR filter
  passes 11/11 (894 filtered), package doctests pass 5/5 (8 ignored), warning-denied rustdoc
  passes, and the refreshed workspace `unused_self` count is 131 sites (solver 10). Wavelet
  transforms remain a separate behavioral-completion item.
- 2026-08-21 wavelet slice is implemented: CDF and Daubechies transforms are associated
  functions; Haar now performs separable 3-D forward/inverse transforms across configured levels,
  preserves odd-length tails, and has a value-semantic round-trip test. Focused Nextest
  `kwavers-solver` AMR filter passes 12/12 (894 filtered), package doctests pass 5/5 (8 ignored),
  warning-denied rustdoc passes, and the refreshed workspace `unused_self` count is 125 sites
  (solver 4). The remaining solver sites are numerical-validation helpers.
- 2026-08-21 numerical-validation slice is implemented: boundary reflection, absorption,
  spurious-reflection, and conservation-error helpers are associated functions with all callers
  migrated, including value-semantic boundary error tests. Focused Nextest numerical-accuracy
  filter passes 17/17 (889 filtered), package doctests pass 5/5 (8 ignored), warning-denied
  rustdoc passes, and the refreshed workspace `unused_self` count is 121 sites (solver 0).
- 2026-08-21 exact integrated-head validation after PR #506 (`855117c00`): full Kwavers
  Nextest passes 530/530 with eight configured skips; `cargo build --offline -p kwavers --examples`
  exits 0; `mdbook test docs/book` exits 0 across all listed chapters; and `mdbook build docs/book`
  exits 0 with HTML written to `target/book`. The test run regenerated tracked PNG outputs locally;
  those derived deltas were restored and no binary fixture changes are included.
- 2026-08-21 `kwavers-math` stateless geometry slice is implemented: `linear_geometry` is an
  associated helper with both contiguous gradient/divergence callers migrated. Package Nextest
  passes 196/196, package doctests pass 3/3 (7 ignored), warning-denied rustdoc passes, and the
  refreshed workspace `unused_self` count is 120 sites.
- 2026-08-21 latest integrated head after PR #508 (`576f70ccb`) re-smokes `cargo build --offline
  -p kwavers --examples`, `mdbook test docs/book`, and `mdbook build docs/book`; all exit 0 and
  the HTML book is written to `target/book`.
- 2026-08-21 `kwavers-signal` frequency-filter slice is implemented: the receiver-free FFT
  response helper is an associated function and all filter callers are migrated. Package Nextest
  passes 63/63, package doctests pass 4/4, warning-denied rustdoc passes, and the refreshed
  workspace `unused_self` count is 119 sites with zero in `kwavers-signal` and `kwavers-solver`.
- 2026-08-21 `kwavers-grid` geometry slice is implemented: the duplicated measure validator is
  consolidated into the vertical `geometry/validation.rs` leaf and both rectangular/spherical
  constructors use the shared receiver-free helper. Package Nextest passes 45/45, doctests pass
  1/1 (1 ignored), warning-denied rustdoc passes, and the refreshed workspace `unused_self` count
  is 117 sites with zero in `kwavers-grid`, `kwavers-signal`, and `kwavers-solver`.
- 2026-08-21 exact integrated-head validation after PR #511 (`2b157eac8`): full Kwavers Nextest
  passes 530/530 with eight configured skips (13 slow tests reported); `cargo build --offline
  -p kwavers --examples`, `mdbook test docs/book`, and `mdbook build docs/book` all exit 0, with
  HTML written to `target/book`. Test-generated PNG outputs were restored and no binary fixture
  deltas are included.
- 2026-08-21 `kwavers-mesh` tetrahedral face enumeration slice is implemented: the receiver-free
  face helper is an associated function and both mesh connectivity callers are migrated. Package
  check, `unused_self` Clippy, Nextest, doctests (1/1), and warning-denied rustdoc pass; the
  refreshed workspace `unused_self` count is 116 sites with zero in `kwavers-mesh`,
  `kwavers-grid`, `kwavers-signal`, and `kwavers-solver`.
- 2026-08-21 `kwavers-receiver` sonoluminescence slice is implemented: stateless cluster merging
  is an associated function and its detector caller is migrated. Package check, `unused_self`
  Clippy, Nextest, doctests (1/1), and warning-denied rustdoc pass; the refreshed workspace
  `unused_self` count is 115 sites with zero in `kwavers-receiver`, `kwavers-mesh`,
  `kwavers-grid`, `kwavers-signal`, and `kwavers-solver`.
- 2026-08-21 `kwavers-boundary` BEM applicator slice is implemented: Dirichlet, Neumann, Robin,
  and radiation applicators are associated functions and `apply_all` migrates all callers.
  Package check, targeted `unused_self` Clippy, Nextest, doctests (4/4; 1 ignored), and
  warning-denied rustdoc pass; package Clippy still reports the 11 pre-existing boundary sites
  in CPML/FEM/smoothing. The refreshed workspace `unused_self` count is 111 sites with zero in
  the BEM manager, `kwavers-receiver`, `kwavers-mesh`, `kwavers-grid`, `kwavers-signal`, and
  `kwavers-solver`.
- 2026-08-21 `kwavers-imaging` multimodality fusion slice is implemented: transform,
  checkerboard, difference, and multi-channel helpers are associated functions with all callers
  migrated; overlay and false-color remain receiver-bound for configured blend weights. Package
  check, targeted `unused_self` Clippy, Nextest, doctests (4/4), and warning-denied rustdoc pass;
  the refreshed workspace `unused_self` count is 107 sites with zero in `kwavers-imaging`,
  `kwavers-boundary` BEM, `kwavers-receiver`, `kwavers-mesh`, `kwavers-grid`, `kwavers-signal`,
  and `kwavers-solver`.
- 2026-08-21 `kwavers-boundary` FEM applicator slice is implemented: Dirichlet, Neumann, Robin,
  and radiation applicators are associated functions and `apply_all` migrates all callers.
  Package check, targeted `unused_self` Clippy, Nextest, doctests (4/4; 1 ignored), and
  warning-denied rustdoc pass; package Clippy still reports seven pre-existing boundary sites
  in CPML and smoothing. The refreshed workspace `unused_self` count is 103 sites with zero in
  the FEM/BEM managers, `kwavers-imaging`, `kwavers-receiver`, `kwavers-mesh`, `kwavers-grid`,
  `kwavers-signal`, and `kwavers-solver`.
- 2026-08-21 `kwavers-boundary` CPML axis-update slice is implemented: four receiver-free axis
  memory/correction helpers are associated functions and public wrapper callers are migrated.
  Package check, targeted `unused_self` Clippy, Nextest, doctests (4/4; 1 ignored), and
  warning-denied rustdoc pass; package Clippy now reports three pre-existing boundary sites in
  smoothing. The refreshed workspace `unused_self` count is 99 sites with zero in CPML, FEM/BEM
  managers, `kwavers-imaging`, `kwavers-receiver`, `kwavers-mesh`, `kwavers-grid`,
  `kwavers-signal`, and `kwavers-solver`.
- 2026-08-21 `kwavers-boundary` smoothing slice is implemented: ghost-cell neighbor collection,
  ghost-cell extrapolation, and immersed-interface correction are associated functions with all
  callers migrated. Package check, targeted `unused_self` Clippy, Nextest (97/97), doctests
  (4/4; 1 ignored), and warning-denied rustdoc pass; the refreshed workspace `unused_self` count
  is 96 sites with zero in `kwavers-boundary`, CPML, FEM/BEM managers, `kwavers-imaging`,
  `kwavers-receiver`, `kwavers-mesh`, `kwavers-grid`, `kwavers-signal`, and `kwavers-solver`.
- 2026-08-21 `kwavers-transducer` calibration slice is implemented: peak extraction, reflector
  matching, and position estimation are associated functions with all calibration callers
  migrated; quality-metric updates remain stateful. Package check, targeted `unused_self` Clippy,
  Nextest (245/245 with one configured skip), doctests (2/2; 6 ignored), and warning-denied
  rustdoc pass. The refreshed workspace `unused_self` count is 93 sites with 10 remaining in
  transducer array/k-Wave rasterizer helpers and zero in the completed calibration and boundary
  slices.
- 2026-08-21 `kwavers-transducer` flexible-array slice is implemented: normal computation and
  stress calculation are associated functions with callers migrated; strain calculation remains
  configuration-dependent. Package check, targeted `unused_self` Clippy, Nextest (245/245 with
  one configured skip), doctests (2/2; 6 ignored), and warning-denied rustdoc pass. The refreshed
  workspace `unused_self` count is 91 sites with eight remaining in transducer k-Wave rasterizer
  helpers and zero in the completed flexible calibration, flexible-array, and boundary slices.
- 2026-08-21 `kwavers-transducer` k-Wave rasterizer slice is implemented: BLI mapping, disc basis
  and sample counts, curved arc/bowl/annulus sampling and masks, planar rectangle/disc/aperture
  sampling and masks, and area-conserving mapping are associated functions with all callers
  migrated. Package check, targeted `unused_self` Clippy (zero transducer sites), Nextest
  (245/245 with one configured skip), doctests (2/2; 6 ignored), and warning-denied rustdoc pass.
  The refreshed workspace `unused_self` count is 83 sites; transducer is clean.
- 2026-08-21 `kwavers-physics` Mie-scattering slice is implemented: Rayleigh approximation,
  logarithmic derivative, coefficient construction, efficiency reductions, asymmetry, and phase
  helpers are associated functions; `MieCalculator::max_terms` remains stateful at the public
  calculation boundary. Package check, targeted `unused_self` Clippy (optics scattering clean),
  Nextest (1561/1561 with one configured skip), doctests (9/13; 4 ignored), and warning-denied
  rustdoc pass. The refreshed workspace `unused_self` count is 76 sites with 29 remaining in
  other physics domains.
- 2026-08-21 `kwavers-physics` cavitation-detection slice is implemented: broadband energy,
  subharmonic FFT spectrum, and spectral state classification are associated functions;
  baseline/history/stateful detector fields remain receiver-bound. Package check, targeted
  `unused_self` Clippy (detection paths clean), Nextest (1561/1561 with one configured skip),
  doctests (9/13; 4 ignored), and warning-denied rustdoc pass. The refreshed workspace
  `unused_self` count is 73 sites with 26 remaining in other physics domains.
- 2026-08-21 `kwavers-physics` IMEX integration slice is implemented: state/vector conversion
  and equilibrium vapor-pressure helpers are associated functions; Jacobian and thermal-rate
  helpers retain the solver/config receivers they require. Package check, targeted
  `unused_self` Clippy (IMEX paths clean), Nextest (1561/1561 with one configured skip),
  doctests (9/13; 4 ignored), and warning-denied rustdoc pass. The refreshed workspace
  `unused_self` count is 70 sites with 23 remaining in other physics domains.
- 2026-08-21 `kwavers-physics` vapor-pressure slice is implemented: Antoine, Wagner, Buck,
  IAPWS, and ice-pressure equations are associated functions; model selection and
  Clausius-Clapeyron remain receiver-bound where calculator state is required. The thermodynamics
  unit caller is migrated to associated-function syntax. Package check, targeted `unused_self`
  Clippy (vapor-pressure paths clean), Nextest (1561/1561 with one configured skip), doctests
  (9/13; 4 ignored), and warning-denied rustdoc pass. The refreshed workspace `unused_self`
  count is 66 sites with 19 remaining in other physics domains.
- 2026-08-21 `kwavers-physics` CEUS slice is implemented: nonlinear beamforming, contrast
  enhancement, and single-frequency DFT extraction are associated functions; harmonic filter
  state remains receiver-bound. Package check, targeted `unused_self` Clippy (CEUS paths clean),
  Nextest (1561/1561 with one configured skip), doctests (9/13; 4 ignored), and warning-denied
  rustdoc pass. The refreshed workspace `unused_self` count is 63 sites with 16 remaining in
  other physics domains.
- 2026-08-21 `kwavers-physics` elastography harmonic-detection slice is implemented: Hann
  windowing, FFT normalization, and SNR estimation are associated functions; detector config
  remains receiver-bound for harmonic selection. Unit callers use associated-function syntax.
  Package check, targeted `unused_self` Clippy (harmonic-detection paths clean), Nextest
  (1561/1561 with one configured skip), doctests (9/13; 4 ignored), and warning-denied rustdoc
  pass. The refreshed workspace `unused_self` count is 60 sites with 13 remaining in other
  physics domains.
- 2026-08-21 `kwavers-physics` coded-excitation slice is implemented: Barker and Golay code
  generation are associated functions; chirp generation remains receiver-bound for sampling
  frequency configuration. Package check, targeted `unused_self` Clippy (coded-excitation paths
  clean), Nextest (1561/1561 with one configured skip), doctests (9/13; 4 ignored), and
  warning-denied rustdoc pass. The refreshed workspace `unused_self` count is 58 sites with 11
  remaining in other physics domains.
- 2026-08-21 `kwavers-physics` elastic mode-conversion slice is implemented: positive-definiteness
  checking is an associated function, while stiffness-tensor fields remain receiver-bound for
  validation. Package check, targeted `unused_self` Clippy (mode-conversion path clean), Nextest
  (1561/1561 with one configured skip), doctests (9/13; 4 ignored), and warning-denied rustdoc
  pass. The refreshed workspace `unused_self` count is 57 sites with 10 remaining in other
  physics domains.
- 2026-08-21 `kwavers-physics` cortical neuromodulation slice is implemented: the M-current
  steady-state gate function is an associated function; voltage-dependent kinetics and neuron
  parameters remain receiver-bound. Package check, targeted `unused_self` Clippy (neuromodulation
  path clean), Nextest (1561/1561 with one configured skip), doctests (9/13; 4 ignored), and
  warning-denied rustdoc pass. The refreshed workspace `unused_self` count is 56 sites with 9
  remaining in other physics domains.
- 2026-08-21 `kwavers-physics` transcranial validation slice is implemented: sidelobe-level
  calculation is an associated function; grid/reference-speed-dependent focal metrics remain
  receiver-bound. Validation tests use associated-function syntax. Package check, targeted
  `unused_self` Clippy (validation path clean), Nextest (1561/1561 with one configured skip),
  doctests (9/13; 4 ignored), and warning-denied rustdoc pass. The refreshed workspace
  `unused_self` count is 55 sites with 8 remaining in other physics domains.
- 2026-08-21 `kwavers-physics` BBB-opening safety slice is implemented: maximum-safe-time,
  microbubble-dose, safety-check, and warning generation are associated functions; protocol and
  permeability state remain receiver-bound. Package check, targeted `unused_self` Clippy
  (BBB-opening paths clean), Nextest (1561/1561 with one configured skip), doctests (9/13;
  4 ignored), and warning-denied rustdoc pass. The refreshed workspace `unused_self` count is
  51 sites with 4 remaining in other physics domains.
- 2026-08-21 `kwavers-physics` treatment-planning slice is implemented: safety validation, thermal
  response, and treatment-time estimation are associated functions; transducer setup/acoustic-field
  simulation retains planner/grid state. Planner callers use associated-function syntax. Package
  check, targeted `unused_self` Clippy (treatment-planning paths clean), Nextest (1561/1561 with one
  configured skip), doctests (9/13; 4 ignored), and warning-denied rustdoc pass. The refreshed
  workspace `unused_self` count is 48 sites with 1 remaining in other physics domains.
- 2026-08-21 `kwavers-physics` plasmonics slice is implemented: the host-medium wavenumber helper is
  an associated function; particle-array geometry and interaction state remain receiver-bound.
  Package check, targeted `unused_self` Clippy (plasmonics path clean), Nextest (1561/1561 with one
  configured skip), doctests (9/13; 4 ignored), and warning-denied rustdoc pass. The refreshed
  workspace `unused_self` count is 47 sites with zero remaining in `kwavers-physics`.
- 2026-08-21 `kwavers-simulation` CEUS slice is implemented: bolus-profile generation is an
  associated function; grid, perfusion, scattering, reconstruction, and microbubble state remain
  receiver-bound. Package check, targeted `unused_self` Clippy (CEUS path clean), Nextest (87/87),
  doctests (4/6; 2 ignored), and warning-denied rustdoc pass. The refreshed workspace `unused_self`
  count is 46 sites with zero remaining in `kwavers-physics` and `kwavers-simulation`.
- 2026-08-21 `kwavers-analysis` conservation slice is implemented: conservation-law inference is
  an associated function; grid-integral, baseline, tolerance, and diagnostic state remain
  receiver-bound. Package check, targeted `unused_self` Clippy (conservation path clean), Nextest
  (744/744), doctests (1/22; 21 ignored), and warning-denied rustdoc pass. The refreshed workspace
  `unused_self` count is 45 sites with zero remaining in `kwavers-physics`, `kwavers-simulation`,
  and conservation checkers.
- 2026-08-21 `kwavers-analysis` beamforming-training slice is implemented: null-model batch data
  loss is an associated function; trainer configuration and physics-loss state remain
  receiver-bound. Package check, targeted `unused_self` Clippy (beamforming-training path clean),
  Nextest (744/744), doctests (1/22; 21 ignored), and warning-denied rustdoc pass. The refreshed
  workspace `unused_self` count is 44 sites with 25 remaining in other analysis paths.
- 2026-08-21 `kwavers-analysis` Bayesian uncertainty slice is implemented: Monte Carlo prediction
  statistics are an associated function; model configuration and predictor state remain
  receiver-bound. Calibration, decomposition, and tests use associated-function syntax. Package
  check, targeted `unused_self` Clippy (Bayesian path clean), Nextest (744/744), doctests (1/22;
  21 ignored), and warning-denied rustdoc pass. The refreshed workspace `unused_self` count is 43
  sites with 24 remaining in other analysis paths.
- 2026-08-21 `kwavers-analysis` conformal prediction slice is implemented: the receiver-free
  conformity-score kernel is an associated function, and calibration plus direct tests use static
  syntax. Package check, targeted `unused_self` Clippy (conformal path clean), Nextest (744/744),
  doctests (1/22; 21 ignored), and warning-denied rustdoc pass. The refreshed workspace
  `unused_self` count is 42 sites with 23 remaining in other analysis paths.
- 2026-08-21 `kwavers-analysis` ensemble uncertainty slice is implemented: bootstrap sampling,
  weighted statistics, and diversity kernels are associated functions; stateful model training and
  weights remain receiver-bound. Package check, targeted `unused_self` Clippy (ensemble path clean),
  Nextest (744/744), doctests (1/22; 21 ignored), and warning-denied rustdoc pass. The refreshed
  workspace `unused_self` count is 40 sites with 21 remaining in other analysis paths.
- 2026-08-21 `kwavers-analysis` uncertainty quantifier slice is implemented: beamforming CNR and
  resolution helpers plus report reliability and recommendation helpers are associated functions;
  configured quantifier state remains receiver-bound. Package check, targeted `unused_self` Clippy
  (quantifier path clean), Nextest (744/744), doctests (1/22; 21 ignored), and warning-denied rustdoc
  pass. The refreshed workspace `unused_self` count is 36 sites with 17 remaining in other analysis
  paths.
- 2026-08-21 `kwavers-analysis` neural DAS slice is implemented: receiver-free signal-quality
  assessment is an associated function; the adaptive pipeline and direct test use associated syntax.
  Package check, targeted `unused_self` Clippy (DAS path clean), Nextest (744/744), doctests (1/22;
  21 ignored), and warning-denied rustdoc pass. The refreshed workspace `unused_self` count is 35
  sites with 16 remaining in other analysis paths.
- 2026-08-21 `kwavers-analysis` neural network slice is implemented: receiver-free feature
  concatenation is an associated function; forward processing and direct tests use associated
  syntax while layer weights remain receiver-bound. Package check, targeted `unused_self` Clippy
  (network path clean), Nextest (744/744), doctests (1/22; 21 ignored), and warning-denied rustdoc
  pass. The refreshed workspace `unused_self` count is 34 sites with 15 remaining in other analysis
  paths.
- 2026-08-21 `kwavers-analysis` neural uncertainty slice is implemented: local-variance computation
  is an associated function; estimator configuration and public estimation remain receiver-bound.
  Package check, targeted `unused_self` Clippy (uncertainty path clean), Nextest (744/744), doctests
  (1/22; 21 ignored), and warning-denied rustdoc pass. The refreshed workspace `unused_self` count
  is 33 sites with 14 remaining in other analysis paths.
- 2026-08-21 `kwavers-analysis` adaptive SLSC slice is implemented: optimal-lag estimation is an
  associated function while adaptive configuration remains receiver-bound. Package check, targeted
  `unused_self` Clippy (SLSC path clean), Nextest (744/744), doctests (1/22; 21 ignored), and
  warning-denied rustdoc pass. The refreshed workspace `unused_self` count is 32 sites with 13
  remaining in other analysis paths.
- 2026-08-21 `kwavers-analysis` CPU-only 3D processor slice is implemented: the no-GPU CPU-memory
  metric is an associated function with the CPU dispatcher migrated to static syntax; GPU memory
  accounting remains provider-bound. CPU and GPU feature checks, package Clippy, Nextest (744/744),
  doctests (1/22; 21 ignored), and warning-denied rustdoc pass. The refreshed workspace `unused_self`
  count is 31 sites with 12 remaining in other analysis paths.
- 2026-08-21 `kwavers-analysis` SAFT slice is implemented: round-trip time-of-flight and Hamming
  apodization kernels are associated functions while configuration validation and reconstruction
  state remain receiver-bound. Package check, targeted `unused_self` Clippy, Nextest (744/744),
  doctests (1/22; 21 ignored), and warning-denied rustdoc pass. The refreshed workspace `unused_self`
  count is 29 sites with 10 remaining in other analysis paths.
- 2026-08-21 `kwavers-analysis` color-flow slice is implemented: spatial box averaging is an
  associated function and its value-semantic helper test uses the type-level call. Imaging, filter,
  and estimator state remain receiver-bound. Package check, targeted `unused_self` Clippy, Nextest
  (744/744), doctests (1/22; 21 ignored), and warning-denied rustdoc pass. The refreshed workspace
  `unused_self` count is 28 sites with 9 remaining in other analysis paths. The package-wide
  all-targets Clippy gate still reports two pre-existing test-only lint defects outside this slice.
- 2026-08-21 `kwavers-analysis` localization slice is implemented: multilateration geometry and
  3×3 linear-algebra kernels plus trilateration geometry/least-squares kernels are associated
  functions; sensor, sound-speed, and weighted-solver state remain receiver-bound. Package check,
  targeted `unused_self` Clippy, Nextest (744/744), doctests (1/22; 21 ignored), and warning-denied
  rustdoc pass. The refreshed workspace `unused_self` count is 23 sites with 4 remaining in other
  analysis paths. The package-wide all-targets Clippy gate still reports two pre-existing test-only
  lint defects outside this slice.
- 2026-08-21 `kwavers-analysis` PAM slice is implemented: spectrum extraction and fundamental-peak
  selection are associated functions while PAM configuration, thresholds, and harmonic output state
  remain receiver-bound. Package check, targeted `unused_self` Clippy, Nextest (744/744), doctests
  (1/22; 21 ignored), and warning-denied rustdoc pass. The refreshed workspace `unused_self` count
  is 21 sites with 2 remaining in other analysis paths. The package-wide all-targets Clippy gate
  still reports two pre-existing test-only lint defects outside this slice.
- 2026-08-21 `kwavers-analysis` polynomial clutter-filter slice is implemented: the Gauss–Jordan
  pseudo-inverse kernel is an associated function while filter configuration and polynomial-order
  state remain receiver-bound. Package check, targeted `unused_self` Clippy, Nextest (744/744),
  doctests (1/22; 21 ignored), and warning-denied rustdoc pass. The refreshed workspace `unused_self`
  count is 20 sites with 1 remaining in other analysis paths. The package-wide all-targets Clippy
  gate still reports two pre-existing test-only lint defects outside this slice.
- 2026-08-21 `kwavers-analysis` clinical-scoring slice is implemented: overall acceptability scoring
  is an associated function while validator requirements and regulatory policy remain
  receiver-bound. Package check, targeted `unused_self` Clippy, Nextest (744/744), doctests (1/22;
  21 ignored), and warning-denied rustdoc pass. The refreshed workspace `unused_self` count is 19
  sites, all remaining in `kwavers-diagnostics` and `kwavers-therapy`; `kwavers-analysis` is clean.
  The package-wide all-targets Clippy gate still reports two pre-existing test-only lint defects
  outside these slices.
- 2026-08-21 `kwavers-diagnostics` AI beamforming slice is implemented: traditional delay-and-sum
  beamforming and memory estimation are associated functions while processor configuration and
  PINN state remain receiver-bound. Package check, targeted `unused_self` Clippy, Nextest (191/191),
  doctests (1/6; 5 ignored), and warning-denied rustdoc pass. The refreshed workspace `unused_self`
  count is 17 sites: 15 in `kwavers-diagnostics` and 2 in `kwavers-therapy`. The package-wide
  all-targets Clippy gate still reports two pre-existing test-only lint defects outside this slice.
- 2026-08-21 `kwavers-diagnostics` clinical monitoring slice is implemented: quality-score
  computation is an associated function while frame history, safety logs, and monitoring metrics
  remain receiver-bound. Package check, warning-denied Clippy, Nextest (191/191), doctests (1/6;
  5 ignored), and warning-denied rustdoc pass. The refreshed workspace `unused_self` count is 16
  sites: 14 in `kwavers-diagnostics` and 2 in `kwavers-therapy`. The package-wide all-targets
  Clippy gate still reports two pre-existing test-only lint defects outside this slice.
- 2026-08-21 `kwavers-diagnostics` neural clinical analysis slice is implemented: tissue
  classification, recommendation generation, and diagnostic-confidence aggregation are associated
  functions while clinical thresholds and lesion-detection state remain receiver-bound. Package
  check, warning-denied Clippy, Nextest (191/191), doctests (1/6; 5 ignored), and warning-denied
  rustdoc pass. The refreshed workspace `unused_self` count is 13 sites: 11 in
  `kwavers-diagnostics` and 2 in `kwavers-therapy`. The package-wide all-targets Clippy gate still
  reports two pre-existing test-only lint defects outside this slice.
- 2026-08-21 `kwavers-diagnostics` neural clinical detection slice is implemented: local-statistics,
  lesion-type classification, and clinical-significance calculations are associated functions while
  threshold configuration and connected-component sizing remain receiver-bound. Package check,
  warning-denied Clippy, Nextest (191/191), doctests (1/6; 5 ignored), and warning-denied rustdoc
  pass. The refreshed workspace `unused_self` count is 10 sites: 8 in `kwavers-diagnostics` and 2
  in `kwavers-therapy`. The package-wide all-targets Clippy gate still reports two pre-existing
  test-only lint defects outside this slice.
- 2026-08-21 `kwavers-diagnostics` feature-extraction slice is implemented: gradient magnitude,
  Laplacian, local frequency, and homogeneity kernels are associated functions while window-size
  configuration remains receiver-bound for speckle variance. Package check, warning-denied Clippy,
  Nextest (191/191), doctests (1/6; 5 ignored), and warning-denied rustdoc pass. The refreshed
  workspace `unused_self` count is 6 sites: 4 in `kwavers-diagnostics` and 2 in `kwavers-therapy`.
  The package-wide all-targets Clippy gate still reports two pre-existing test-only lint defects
  outside this slice.
- 2026-08-21 `kwavers-diagnostics` neural workflow slice is implemented: median computation is an
  associated function while rolling performance history, quality metrics, and workflow lifecycle
  remain receiver-bound. Package check, warning-denied Clippy, Nextest (191/191), doctests (1/6;
  5 ignored), and warning-denied rustdoc pass. The refreshed workspace `unused_self` count is 5
  sites: 3 in `kwavers-diagnostics` and 2 in `kwavers-therapy`. The package-wide all-targets Clippy
  gate still reports two pre-existing test-only lint defects outside this slice.
- 2026-08-21 `kwavers-diagnostics` orchestrator slice is implemented: photoacoustic acquisition,
  elastography acquisition, and quality assessment are associated functions while workflow state,
  fusion, and real-time configuration remain receiver-bound. Package check, warning-denied Clippy,
  Nextest (191/191), doctests (1/6; 5 ignored), and warning-denied rustdoc pass. The refreshed
  workspace `unused_self` count is 2 sites, both in `kwavers-therapy`; diagnostics is clean. The
  package-wide all-targets Clippy gate still reports two pre-existing test-only lint defects outside
  these slices.
- 2026-08-21 `kwavers-therapy` lithotripsy slice is implemented: affine coupling coefficients and
  acoustic-intensity calculation are associated functions while cloud coupling state and grid,
  stone, and bioeffects state remain receiver-bound. Package check, warning-denied Clippy, Nextest
  (350/350 with one configured skip), doctests (8/9; 1 ignored), and warning-denied rustdoc pass.
  The exact workspace `unused_self` scan reports 0 sites; the ratchet line is removed from
  `Cargo.toml`, re-enabling the lint floor. The package-wide all-targets Clippy gate still reports
  two pre-existing test-only lint defects outside this slice.
- 2026-08-21 final Kwavers integration verification at merged revision `02fea4631`: `cargo fmt
  --all -- --check`, `cargo build --offline -p kwavers --examples`, and the focused
  `cargo nextest run --offline -p kwavers-diagnostics -p kwavers-therapy --lib` pass (541/541 with
  one configured skip). `mdbook test docs/book` passes every chapter and `mdbook build docs/book`
  completes. The exact workspace `unused_self` scan is 0, with the ratchet override deleted.
- 2026-08-21 `return_self_not_must_use` ratchet slice is complete: simulation builder fluent methods
  and therapy regulatory builders now carry `#[must_use]`. Simulation gates pass (87/87 Nextest,
  doctests 4/6 with 2 ignored, warning-denied rustdoc); therapy gates pass (350/350 Nextest with one
  configured skip, doctests 8/9 with 1 ignored, warning-denied rustdoc). The exact workspace scan is
  0 sites and the ratchet override is removed from `Cargo.toml`.
- 2026-08-21 boundary `unnecessary_literal_bound` slice is complete: six boundary-condition trait
  implementations now return `&'static str` for their literal names. Package check, warning-denied
  Clippy, Nextest (97/97), doctests (4/5; 1 ignored), and warning-denied rustdoc pass. The refreshed
  workspace `unnecessary_literal_bound` count is 13 sites, all outside `kwavers-boundary`.
- 2026-08-21 analysis localization `unnecessary_literal_bound` slice is complete: MUSIC and TDOA
  processor names now return `&'static str`. Package check, warning-denied Clippy, Nextest (744/744),
  doctests (1/22; 21 ignored), and warning-denied rustdoc pass. The refreshed workspace
  `unnecessary_literal_bound` count is 11 sites, all outside `kwavers-analysis`.
- 2026-08-21 simulation `unnecessary_literal_bound` slice is complete: CEUS and discontinuous
  Galerkin adapter names now return `&'static str`. Package check, warning-denied Clippy, Nextest
  (87/87), doctests (4/6; 2 ignored), and warning-denied rustdoc pass. The refreshed workspace
  `unnecessary_literal_bound` count is 9 sites, all outside `kwavers-simulation`.
- 2026-08-21 solver `unnecessary_literal_bound` slice is complete: six FDTD, hybrid, PSTD, and
  inverse-reconstruction solver names now return `&'static str`. Package check, warning-denied
  Clippy, Nextest (902/902 with 4 configured skips), doctests (5/13; 8 ignored), and warning-denied
  rustdoc pass. The refreshed workspace `unnecessary_literal_bound` count is 3 sites, all outside
  `kwavers-solver`.
- 2026-08-21 imaging `unnecessary_literal_bound` slice is complete: CT loader name/modality and
  DICOM loader name now return `&'static str`; receiver-dependent DICOM modality remains borrowed.
  Package check, warning-denied Clippy, Nextest (61/61), doctests (4/4), and warning-denied rustdoc
  pass. The exact workspace scan reports 0 sites and the ratchet override is removed from
  `Cargo.toml`.
- 2026-08-21 final integrated-head verification at merged revision `06ab560eb`: `cargo fmt --all --
  --check`, `cargo build --offline -p kwavers --examples`, `mdbook test docs/book`, and
  `mdbook build docs/book` pass. Workspace Nextest passes 5,781/5,781 with 7 configured skips;
  workspace doctests pass with no failures. Exact workspace scans report `unused_self=0`,
  `return_self_not_must_use=0`, and `unnecessary_literal_bound=0`.
- 2026-08-21 `unnecessary_semicolon` slice is complete: monolithic solver residual matching and
  Westervelt FWI loop control no longer carry redundant statement terminators. Combined solver and
  therapy gates pass (1,364/1,364 Nextest with 5 configured skips; doctests and warning-denied
  rustdoc pass). The exact workspace `unnecessary_semicolon` scan is 0 and its ratchet override is
  removed from `Cargo.toml`.
- 2026-08-21 small Clippy ratchet slice is complete: primitive test ordering uses `sort_unstable`,
  and exact all-targets scans confirm `stable_sort_primitive=0`, `unchecked_time_subtraction=0`,
  and `self_only_used_in_recursion=0`. The three suppression lines are removed from `Cargo.toml`.
- 2026-08-21 `bool_to_int_with_if` slice is complete: pulser switch fanout and transcranial UST
  source selection use typed `usize::from` conversions. Diagnostics and driver gates pass (685/685
  Nextest; doctests and warning-denied rustdoc pass). The exact workspace scan is 0 and its ratchet
  override is removed from `Cargo.toml`.
- 2026-08-21 `large_types_passed_by_value` slice is complete: simulated-population monitor input
  validation borrows its 264-byte input while the public trace API remains value-semantic. Physics
  gates pass (1,561/1,561 Nextest with 1 configured skip; doctests 9/13; warning-denied rustdoc).
  The exact workspace scan is 0 and its ratchet override is removed from `Cargo.toml`.
- 2026-08-21 final exact-revision validation at merged revision `192eb7575`: workspace Nextest
  passes 5,781/5,781 with 7 configured skips; workspace doctests pass with no failures. The
  previously recorded example build and mdBook test/build remain valid because the final changes
  are lint-only production edits and test-ordering cleanups. Exact workspace scans remain zero for
  `unused_self`, `return_self_not_must_use`, `unnecessary_literal_bound`, `unnecessary_semicolon`,
  `stable_sort_primitive`, `unchecked_time_subtraction`, `self_only_used_in_recursion`, and
  `bool_to_int_with_if`.
- 2026-08-21 `implicit_hasher` diagnostics slice is complete: the two public analysis workflow
  functions now generalize `HashMap` over `BuildHasher`, preserving caller-selected hashers without
  copying the tissue-property map. Warning-denied diagnostics Clippy, 191/191 Nextest, doctests
  (1 passed, 5 ignored), and warning-denied Rustdoc pass. The refreshed workspace scan reports six
  remaining production sites, all in `kwavers-driver`.
- 2026-08-21 `implicit_hasher` driver slice is complete: routing `HashSet` inputs and verification
  symbol-map `HashMap` inputs now accept caller-selected `BuildHasher` implementations. Warning-
  denied driver Clippy, 494/494 Nextest, doctests, and warning-denied Rustdoc pass. The exact
  production workspace scan (`--lib`) reports zero sites, so the ratchet override is removed from
  `Cargo.toml`; all-targets still reports test-only sites outside this production slice.
- 2026-08-21 `from_iter_instead_of_collect` physics slice is complete: sonogenetics pressure and
  tension arrays now collect directly into `Array1`, retaining the existing shape conversion and
  validation paths. Warning-denied physics Clippy, 1,561/1,561 Nextest with 1 configured skip,
  doctests (9 passed; 4 ignored), and warning-denied Rustdoc pass. The production workspace count
  decreases from 16 to 14 sites; the ratchet comment is updated.
- 2026-08-21 `from_iter_instead_of_collect` simulation slice is complete: DG recorder statistics
  collect directly into `Array1` for maximum, minimum, RMS, and final pressure values. Warning-
  denied simulation Clippy, 87/87 Nextest, doctests (4 passed; 2 ignored), and warning-denied
  Rustdoc pass. The production workspace count decreases from 14 to 10 sites; the ratchet comment
  is updated.
- 2026-08-21 `from_iter_instead_of_collect` analysis slice is complete: polynomial clutter-filter
  time coordinates now collect directly into `Array1` before optional normalization. Warning-
  denied analysis Clippy, 744/744 Nextest, doctests (1 passed; 21 ignored), and warning-denied
  Rustdoc pass. The production workspace count decreases from 10 to 9 sites; the ratchet comment
  is updated.
- 2026-08-21 `from_iter_instead_of_collect` solver slice is complete: Kuznetsov spectral vectors,
  Fourier quadrature nodes, and axisymmetric propagator grids now collect directly into `Array1`.
  Warning-denied solver Clippy, 902/902 Nextest with 4 configured skips, doctests (5 passed; 8
  ignored), and warning-denied Rustdoc pass. The production workspace count decreases from 9 to 2
  sites; the ratchet comment is updated. Nextest compilation also reports two pre-existing
  test-only unused-variable warnings in unrelated RTM tests.
- 2026-08-21 `from_iter_instead_of_collect` therapy slice is complete: the elastic-shear velocity
  source waveform now collects its sampled signal directly into `Array1`. Warning-denied therapy
  Clippy, 350/350 Nextest with 1 configured skip, doctests (8 passed; 1 ignored), and warning-
  denied Rustdoc pass. The production workspace count decreases from 2 to 1 site; the ratchet
  comment is updated.
- 2026-08-21 `from_iter_instead_of_collect` Python slice is complete: the PAM MUSIC binding now
  collects the complex steering vector directly into `Array1`, preserving the PyO3 conversion-only
  boundary. Warning-denied Python Clippy, 21/21 Nextest, and warning-denied Rustdoc pass; doctests
  are not applicable because the package intentionally has no Rust library target. The production
  workspace scan reports zero sites, so the ratchet override is removed from `Cargo.toml`.
- 2026-08-21 `missing_fields_in_debug` therapy slice is complete: `ComplianceCheck` Debug output
  now accounts for its opaque validation callable with a redacted callable marker, while retaining
  all value fields. Warning-denied therapy Clippy, 350/350 Nextest, doctests (8 passed; 1 ignored),
  and warning-denied Rustdoc pass. The production workspace count decreases from 17 to 16 sites;
  the ratchet comment is updated.
- 2026-08-21 `missing_fields_in_debug` analysis slice is complete: `PipelineCoordinator` Debug
  output now includes stage handles and synchronization/metric storage fields alongside its stage
  count. Warning-denied analysis Clippy, 744/744 Nextest, doctests (1 passed; 21 ignored), and
  warning-denied Rustdoc pass. The production workspace count decreases from 16 to 15 sites; the
  ratchet comment is updated.
- 2026-08-21 `missing_fields_in_debug` physics slice is complete: cavitation `FeedbackController`
  Debug output now includes state-estimator, safety-monitor, and adaptive-controller fields, and
  `MieTheory` includes a redacted particle-dielectric callable marker. Warning-denied physics
  Clippy, 1,561/1,561 Nextest with 1 configured skip, doctests (9 passed; 4 ignored), and
  warning-denied Rustdoc pass. The production workspace count decreases from 15 to 13 sites; the
  ratchet comment is updated.
- 2026-08-21 `missing_fields_in_debug` solver slice is complete: all thirteen remaining solver
  Debug implementations now represent every field, using values, shapes, lengths, booleans, and
  redacted markers for opaque plans/backends. Warning-denied solver Clippy, 902/902 Nextest with
  4 configured skips, doctests (5 passed; 8 ignored), warning-denied Rustdoc, and the exact
  production workspace scan pass. The `missing_fields_in_debug` ratchet override is removed.
- 2026-08-21 `pub_underscore_fields` solver slice is complete: the placeholder `_reserved` field
  is removed from `SpectralElasticConfig`, which is now an explicit unit configuration type.
  Warning-denied solver Clippy, 902/902 Nextest with 4 configured skips, doctests (5 passed; 8
  ignored), and warning-denied Rustdoc pass. The exact production workspace count decreases from
  16 to 15 sites; the root ratchet comment is updated.
- 2026-08-21 `struct_excessive_bools` receiver slice is complete: recorder channel selection now
  uses `RecordingChannels` and descriptive `RecorderChannel`/`RecordingState` enums instead of
  five independent booleans in both the configuration and runtime recorder. Warning-denied
  receiver Clippy, 48/48 Nextest, one doctest, and warning-denied Rustdoc pass. The exact
  production workspace count decreases from 14 to 12 sites; the root ratchet comment is updated.
- 2026-08-21 `needless_for_each` physics slice is complete: seven mutating normalization and
  centering passes in cavitation, inverse, RTM, and transducer analysis now use direct loops.
  Warning-denied physics Clippy, 1,561/1,561 Nextest with 1 configured skip, doctests (9 passed;
  4 ignored), warning-denied Rustdoc, and workspace warning-denied Clippy pass. The exact
  production workspace count decreases from 12 to 5 sites; the root ratchet comment is updated.
- 2026-08-21 `needless_for_each` solver slice is complete: the two CPML scratch resets now use
  allocation-free `slice.fill(0.0)` operations. Warning-denied solver Clippy, 902/902 Nextest
  with 4 configured skips, doctests (5 passed; 8 ignored), warning-denied Rustdoc, workspace
  warning-denied Clippy, and the exact production scan pass. The production workspace count
  decreases from 5 to 3 sites; the root ratchet comment is updated.
- 2026-08-21 `needless_for_each` closure slice is complete: analysis IIR coefficient normalization
  and diagnostic breast-phantom sound-speed scaling now use direct loops. Warning-denied analysis
  and diagnostics Clippy, combined Nextest (935/935), doctests (1 passed in each package; 21 and
  5 ignored), warning-denied Rustdoc for both packages, and the exact production workspace scan
  pass. The production workspace count decreases from 3 to 0 sites; the ratchet override is
  removed from `Cargo.toml`.
- 2026-08-21 `assigning_clones` grid slice is complete: gradient coefficient cache refresh now
  uses `clone_from` to reuse the existing vector allocation. Warning-denied grid Clippy, 46/46
  Nextest, doctests (1 passed; 1 ignored), warning-denied Rustdoc, and the exact production scan
  pass. The production workspace count decreases from 11 to 10 sites; the root ratchet comment is
  updated.
- 2026-08-21 `assigning_clones` imaging slice is complete: unified-loader path updates now use
  `clone_into` to reuse the configured path allocation. Warning-denied imaging Clippy, 61/61
  Nextest, 4/4 doctests, warning-denied Rustdoc, and the exact production scan pass. The production
  workspace count decreases from 10 to 9 sites; the root ratchet comment is updated.

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

## Current Aequitas integration slice — 2026-08-02

| ID | Outcome | Class | Status | Owner | Scope |
|----|---------|-------|--------|-------|-------|
| KWAVERS-AEQ-MET-69 | Type B-mode scan-conversion angles and length geometry with Aequitas; keep scalar extraction at Cartesian raster and interpolation formula boundaries and preserve Eunomia's real geometry rule. | [major] [arch] | in progress 2026-08-06 | Codex | `crates/kwavers-analysis/src/signal_processing/b_mode/`, ADR 105, `gap_audit.md` |
| KWAVERS-AEQ-MET-FLEX | Audit and type the flexible-array dynamic metrics: timestamps, focus/speed/delay contracts, calibration confidence, and deformation outputs; preserve raw dense mesh/signal storage boundaries. | [arch] [major] | blocked: peer-owned `crates/kwavers-transducer/src/flexible/array.rs` dirty; reopen on integration or scope release | Codex | `crates/kwavers-transducer/src/flexible/{array,beamforming,geometry,config}.rs`, direct callers/tests, ADR 070, PM artifacts |

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
- **First increment `93e88e24d`, 84 -> 75, riding [#759](https://github.com/ryancinsight/kwavers/pull/759).**
  All 11 `print_stdout` sites gone. They were print-debugging inside
  `#[cfg(test)]` modules, and removing them exposed why they existed: five
  tests asserted nothing that could fail and printed the property they named.
  `analytic.rs` bound its analytic expectation to `_expected` and asserted
  `is_finite()`. Those oracles are unreachable -- the model is an untrained,
  randomly initialised network, so finite differences is the only valid oracle
  and `first_deriv.rs`/`second_deriv.rs` already apply it. Point coverage moved
  there; the property-in-name-only tests were deleted.
- **Lane convergence, not a separate delivery:** a peer switched this lane's
  branch mid-increment and built `chore/kwavers-manifest-row` on top of the
  commit, so it lands with their driver refactor rather than its own pull
  request. Content verified intact on their branch (authorship, message, all
  five file changes). `fix/kw-solver-lint-ratchet` on origin had been left
  pointing at `main` and is deleted.
- **Finding for whoever takes the remainder:** `second_deriv.rs` deliberately
  did not absorb the deleted analytic points. Both sides there are finite
  differences at different step sizes (`coeus_autograd` has no double-backward),
  so disagreement scales with the fourth derivative and `REL_TOL_SECOND` is
  empirical, not derived -- the added point measured rel_err 2.59e-2 against
  the 1e-2 bound. Extending that test needs a derived per-point bound first.
- **Second increment #761, 73 -> 61.** Twelve diagnostics with a determinate
  fix: 6 `#[must_use]` on `mut self -> Self` builders, 4 `assert!(a == b)` ->
  `assert_eq!`, 1 unquoted intra-doc link, and `generate_boundary_data`, which
  took `&self` and never touched it -- the receiver survived only through its
  own recursive calls -- made an associated function with its three call sites
  updated. 1375 solver tests pass.
- **Remaining 61, and why it stops there:** 28 `unused_self` is a design
  finding, not a mechanical fix. The 26 missing `# Errors` and 7 missing
  `# Panics` must not be closed by extending the template that produced
  [KW-ERRORS-DOCS-ARE-TEMPLATE-OUTPUT-2026-09-09](#kw-errors-docs-are-template-output-2026-09-09)
  -- doing so would move this number without improving anything, which is
  gaming the measure.
- **Not met. Measured 2026-09-08**, `cargo clippy -p kwavers-solver --features
  pinn --all-targets`: 84 diagnostics -- 28 `unused_self`, 26 missing
  `# Errors`, 11 `println!`, 7 missing `# Panics`, 6 missing `#[must_use]`, 4
  `assert!` with an equality comparison, 2 doc-link/recursion findings.
- The 2026-08-17 increment recorded the gate passing; it does not pass now,
  under this configuration. `println!` in a library crate is its own floor
  violation (engineering_gates: lint floor denies `print_stdout` there), and
  `unused_self` at 28 sites is a design finding rather than a mechanical fix.
- **Acceptance:** the measured count reaches zero under the named
  configuration, or each survivor carries `#[expect(lint, reason = ...)]`.

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
