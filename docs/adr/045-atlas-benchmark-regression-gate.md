# ADR 045: Atlas benchmark regression gate

- Status: Accepted (amended 2026-09-04)
- Date: 2026-07-20
- Amended: 2026-08-31, 2026-09-04
- Change class: patch, architecture
- Closed: 2026-07-22

## Context

The former CI job ran the Kwavers Criterion suite once, saved those results as
the baseline, and immediately compared the same result tree against itself with
a local Python percentage-threshold script. That procedure could not observe a
base-to-head regression. Its fixed 15% threshold also had no family-wise error
contract across the benchmark universe.

The benchmark suite is the measurement instrument. Comparing a historical
harness against a candidate harness would confound instrument changes with
production changes.

The first exact-head workflow run, `29797805169`, exposed an independent
instrument defect before measurement: eight benchmark files were
auto-discovered as default libtest targets, while only 17 targets were
registered with Criterion's `harness = false`. Forwarding Criterion's
`--save-baseline` flag to the package library harness then failed with an
unrecognized-option error. Prior full-suite runs had silently executed those
eight auto-discovered targets as zero-test binaries.

Exact-head run `29841101698` completed all four isolated pair jobs and the
aggregate classifier reported three replicated apparent regressions. The only
production diff was a line wrap in an unrelated Kalman implementation, but
base and candidate were compiled from distinct, revision-correlated checkout
paths on every runner. Reversing execution order does not remove that path
identity confound.

Exact-head run `29867760523` then demonstrated that the complete statistical
instrument is not suitable for pull-request latency. Its four pair jobs were
still running after 157 minutes. The retained
`linear_swe_wave_propagation` scenario alone estimated 1,951 seconds for one
100-sample revision measurement. The run was cancelled before classification;
the historical 249-minute pair duration remained the expected critical path.

Exact-head run `29911114271` completed the bounded four-pair instrument but
reported replicated `grid_memory/32` and `grid_memory/128` regressions after
the candidate changed only the supply-chain source policy from the preceding
green measured head. That change cannot alter a merge-critical benchmark
executable. Statistical comparison therefore supplied weaker evidence than a
direct executable-identity check for this revision pair.

Hosted shared runners are scheduling neighbours with uncontrolled load, on
hardware this project does not own. A confidence interval measured there is
not reproducible on the next run, so its agreement across replications says
something about the runner pool on that day and nothing firm about the
revision pair. PR #681 run `33433562701` still launched four full
phase-reversed replications (jobs `99629480267`, `99629480304`, `99629480174`,
`99629480160`) after a 15m56s complete smoke: four additional 30-minute lanes
on the merge path producing evidence that is not controlled. The amendment of
2026-09-04 therefore moves the statistical instrument off the hosted merge
path entirely; the same-harness, same-path, benchmark-universe,
confidence-interval, and family-wise classification requirements transfer
unchanged to a controlled local instrument.

## Decision

PRs that change Rust production, dependency, or benchmark inputs run a
dedicated workflow. Python binding packaging and documentation changes do not
trigger a Rust performance comparison. The workflow checks out the PR base and
head and overlays the candidate `crates/kwavers/benches` tree onto the base
checkout.

The merge-critical statistical universe is the three canonical production
instruments `performance_baseline`, `critical_path_benchmarks`, and
`simd_field_ops`. These cover allocation and field baselines, the FDTD and
k-space critical paths, and production SIMD field kernels. Their existing
Criterion workloads and sample counts remain unchanged. A separate candidate
job executes every plotting-eligible benchmark once in Criterion test mode, so
every retained end-to-end scenario remains build- and execution-checked without
repeating multi-second simulations hundreds of times on every PR.

The package disables automatic benchmark discovery and explicitly registers
all 22 retained benchmark files as Criterion targets. The cleanup deletes
sleep-based PINN timing, a scalar-vs-scalar SIMD comparison, and a redundant
solver comparator whose pressure norm was labeled as acoustic energy. It also
deletes the mixed PINN/FDTD aggregate whose setup, allocation timing, and
accuracy concerns duplicate dedicated instruments without a common workload.
The retained SIMD benchmark measures the production field-operation kernels.
The package library and binary set `bench = false`, preventing Criterion
arguments from reaching their libtest harnesses. The GPU-only Hilbert pipeline
declares its required feature instead of executing an empty fallback, and
feature-gated targets have no unreachable no-op entry points. Before any
measurement, the workflow requires exact equality between the sorted
`benches/*.rs` stems and Cargo's benchmark target registry for both revisions.
It invokes `cargo bench --benches`, so Criterion arguments reach only
registered benchmark targets. Any unregistered, orphaned, default-harness, or
empty-entry-point target fails before timing.

The complete smoke job compiles the three merge-critical benchmark executables
for base and candidate from the same `kwavers-measurement` path after holding
the candidate harness and provider graph constant. It compares SHA-256 hashes
by target name. A byte-identical target cannot contain a production-code
performance difference, so the workflow records that stronger static evidence
directly. A differing hash is not a failure: it is the input to the controlled
local paired measurement below.

Pull-request CI performs no statistical timing. It schedules no Criterion
measurement job, uploads no measurement artifacts, and checks out no Atlas
classifier. Hosted wall time is never substituted for controlled paired
evidence, and no correction may weaken the Criterion instruments or move their
workloads, sample counts, or confidence contract. When merge-critical
executables differ, the reviewer reproduces the controlled local instrument
documented in the amendment below and attaches its report to the PR. The
regression decision (improvement, noise, regression per family-wise
classification) is made from that local report.

The Atlas tool derives per-comparison confidence as `1 - 0.05 / m` for `m`
benchmarks. Missing results, benchmark-universe mismatches, malformed
estimates, and insufficient confidence fail closed. There is no empirical
percentage threshold.

The workflow's jobs remain bounded by a 30-minute timeout each. The complete
smoke is additionally governed by the separate 300-second suite budget
(KW-CI-BENCH-SMOKE-BUDGET). The amendment does not alter native-test budgets.

## Controlled local instrument (amendment 2026-09-04)

The statistical requirements of the original decision are retained verbatim
and are now exercised locally, where contention is controlled and results are
reproducible. The local command is authoritative; CI never runs it.

Prerequisites: a checkout of this repository (both revisions via
`git worktree add` from the same object database so both sides compile from
the same canonical path), the pinned Atlas tool
(`ryancinsight/atlas` at `9c33b4af1ac44ba43e4d26eaf9cb215218db248e`, checkout
path `atlas-tool` below), and no competing load on the machine. Pin the
measurement to one physical core (e.g. Linux `taskset -c 2`, Windows
`/ affinity` or `start /affinity`) for every command below.

For each merge-critical target reported hash-different by CI
(`TARGETS="performance_baseline critical_path_benchmarks simd_field_ops"`
reduced to the changed subset), run the complete counterbalanced matrix:

1. Clear the Criterion tree before each pair: `rm -rf target/criterion`.
2. Pair 1 (baseline first), revision A = base, B = candidate:

   ```sh
   cargo bench --locked --package kwavers --features plotting \
     --bench "$TARGET" -- --save-baseline atlas-base          # A
   cargo run --locked --manifest-path \
     atlas-tool/tools/criterion-regression/Cargo.toml -- \
     required-confidence \
     --criterion-root target/criterion --baseline atlas-base   # confidence
   cargo bench --locked --package kwavers --features plotting \
     --bench "$TARGET" -- --baseline atlas-base \
     --confidence-level "$CONFIDENCE"                          # B
   mv target/criterion reports/first/baseline-first
   ```

3. Pair 2 (candidate first): same commands with A and B swapped, report moved
   to `reports/first/candidate-first`.
4. Repeat steps 1–3 a second time (same order balance, fresh measurement):
   `reports/second/baseline-first` and `reports/second/candidate-first`.
5. Classify:

   ```sh
   cargo run --locked --manifest-path \
     atlas-tool/tools/criterion-regression/Cargo.toml -- \
     check-replicated-counterbalanced \
     --first-baseline-first-root reports/first/baseline-first \
     --first-candidate-first-root reports/first/candidate-first \
     --second-baseline-first-root reports/second/baseline-first \
     --second-candidate-first-root reports/second/candidate-first \
     --baseline atlas-base
   ```

The four reports are the same phase-reversed, within-machine, two-replication
matrix the workflow formerly executed on hosted runners. The Atlas classifier
still requires replicated, order-balanced agreement in direction and coverage
of the same changed benchmark universe before reporting a regression. Attach
the classifier output and the four report directories to the PR as the
performance evidence for merge review.

## Rejected alternatives

- Retain the same-run save/check script: rejected because both sides contain
  the same measurement.
- Compare one base/head pair: rejected because a slowdown confined to run
  order cannot be distinguished from a production regression.
- Run all four pairs serially on one runner: rejected after exact hosted run
  `29814752294` demonstrated an approximately 18-hour schedule against the
  finite 315-minute job bound. Each base/head comparison remains co-located;
  isolated pair runners add an observed replication dimension without
  mixing machines inside a confidence interval.
- Compile the two revisions from distinct checkout paths: rejected after run
  `29841101698` reported three replicated apparent regressions without a
  semantic production delta. Path identity must not remain correlated with
  revision identity.
- Always run statistical pairs for byte-identical executables: rejected after
  run `29911114271` reported two impossible production regressions. Exact
  executable identity is stronger evidence for that case.
- Keep the complete statistical suite as a merge gate: rejected because an
  observed pair takes about 249 minutes and one long-horizon SWE measurement
  alone requests about 32 minutes per revision.
- Reduce sample counts across the retained statistical targets: rejected
  because weakening the instrument is not a latency lever; the confidence
  contract exists precisely so the evidence is decisive.
- Drop non-critical benchmark scenarios: rejected because one-pass candidate
  execution retains build and runtime coverage for every registered target.
- Keep hosted statistical timing on the merge path (amendment-era
  alternative): rejected because hosted shared runners cannot provide
  controlled conditions; the 2026-08-31 bounded form still consumed four
  30-minute lanes producing results the next run could not reproduce, and
  run `29911114271` demonstrated hosted statistical classification reporting
  impossible regressions that executable identity already excluded.
- Substitute hosted wall time or a percentage threshold for the paired
  instrument: rejected; wall-time deltas on contended runners are not
  evidence, and the family-wise confidence contract is the decision
  procedure.

## Consequences

Benchmark-relevant PRs consume one complete smoke and executable-identity job
and nothing else. Identical merge-critical executables terminate with the
identity proof. Differing executables require the reviewer to run the
documented local counterbalanced instrument and attach its report; the PR
schedules no statistical timing job, uploads no measurement artifacts, and
checks out no Atlas classifier. The merge path loses four bounded 30-minute
lanes per measured PR and gains a requirement that performance evidence be
reproducible on the reviewer's machine. Python packaging-only and
documentation-only PRs do not run the Rust instrument. Atlas remains the
single source of truth for statistical classification through the pinned
classifier invoked by the local command. Report artifacts do not encode
source-path provenance, so review establishes the same-path precondition that
the classifier cannot verify. Long-horizon scenarios remain functional
benchmark programs, but they are never repeated statistically on the
merge-critical path, locally or in CI.

## Closure evidence

The superseded workflow run `29875283986` at Tyche candidate head
`cc382dbc2243678fef55101aa106e9f8d7ad7bbf` completed all four pairs before
classifying 190 cases and reporting 37 replicated regressions. None belongs to
`performance_baseline`, `critical_path_benchmarks`, or `simd_field_ops`. The
failure therefore exercises the complete statistical universe rejected above,
not the bounded decision.

Replacement head `a85aa58e5ad350f5a72483fd541337b95ed0f8de` passes the complete
candidate smoke, all four bounded AB/BA pair jobs in 21–23 minutes, and the
aggregate classifier in run `29884797777`. Ordinary CI `29884797767`,
architecture validation `29884797709`, and legacy audit `29884797739` also
pass. PR #306 merged the checked workflow as `00d06f00e`.

Exact head `04bced11bfd92cefcf38ccbadd76f1bd203c550a` validates the
executable-identity branch in run `29913169741`. The complete benchmark smoke
and same-path base/head builds finish in 11m37s, all three merge-critical
executables are byte-identical, the pair matrix is skipped, and the explicit
regression check passes. The complete workflow finishes in 12m12s. Exact-head
CI `29913169738`, architecture validation `29913169852`, and legacy audit
`29913169756` also pass.

PR #681 run `33439956718` exposed the target-granularity correction. Smoke job
`99645450380` produced identical `performance_baseline` and `simd_field_ops`
executables and a differing `critical_path_benchmarks` executable, but the pair
matrix still measured all three. The 2026-08-31 amendment passes only
hash-different target names to the existing four-pair instrument.

The 2026-09-04 amendment (this document) removes the hosted pair and
classifier jobs entirely; the retained workflow is the complete smoke,
registry validation, and executable-identity check, and the statistical
instrument is the controlled local procedure above. Delivery evidence for the
amendment is recorded against KW-CI-LOCAL-CRITERION-EVIDENCE in `backlog.md`.

## Superseded sections (2026-08-31 form, retired 2026-09-04)

The paragraphs below describe the four-pair hosted instrument replaced by the
amendment. They are retained so the amendment's diffs against the recorded
decision stay auditable; the Decision section above is the operative form.

<details>
<summary>Retired hosted instrument (2026-08-31 – 2026-09-04)</summary>

Before each statistical measurement the workflow moved the selected clean
checkout into one `kwavers-measurement` path, ran the target set, and restored
the checkout, so both revisions compiled from the same canonical path inside
every pair.

Four isolated runners each executed one complete base/head pair over the
hash-different subset of the three merge-critical targets. Two used order
`A B` and two used `B A`, where `A` is the base revision and `B` is the
candidate. Each comparison remained within one machine, while the
phase-reversed matrix balanced revision order and samples separated
hosted-runner variation. A regression was reported only when all four
confidence intervals agreed in direction and covered the same changed
benchmark universe. Pair results were uploaded as artifacts, downloaded by a
final job, and classified by the Atlas classifier
(`check-replicated-counterbalanced`).

</details>
