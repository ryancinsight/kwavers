# ADR 045: Atlas benchmark regression gate

- Status: accepted
- Date: 2026-07-20
- Amended: 2026-07-21
- Change class: patch, architecture

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

## Decision

PRs that change Rust production, dependency, or benchmark inputs run a
dedicated workflow. Python binding packaging and documentation changes do not
trigger a Rust performance comparison. The workflow checks out the PR base and
head and overlays the candidate `crates/kwavers/benches` tree onto the base
checkout. Before each statistical measurement it moves the selected clean
checkout into one `kwavers-measurement` path, runs the target set, and restores
the checkout. Both revisions therefore compile from the same canonical path
inside every pair.

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
feature-gated targets have no unreachable no-op entry points. Before
measurement, the workflow requires exact equality between the sorted
`benches/*.rs` stems and Cargo's benchmark target registry for both revisions.
It invokes `cargo bench --benches`, so Criterion arguments reach only
registered benchmark targets.
Any unregistered, orphaned, default-harness, or empty-entry-point target fails
before timing.

The workflow consumes the Atlas-owned regression classifier and provider graph
pinned at `614914cf469f69ebd193e17c8e2c0db7dcb4a23f`. Four isolated runners each
execute one complete base/head pair. Two use order `A B` and two use `B A`,
where `A` is the base revision and `B` is the candidate. Each comparison
therefore remains within one machine, while the phase-reversed matrix balances
revision order and samples separate hosted-runner variation. A regression
is reported only when all four confidence intervals agree in direction and
cover the same benchmark universe.

The Atlas tool derives per-comparison confidence as `1 - 0.05 / m` for `m`
benchmarks. Missing results, benchmark-universe mismatches, malformed
estimates, and insufficient confidence fail closed. There is no empirical
percentage threshold.

Every smoke, pair, and classifier job has a 30-minute bound. Four matrix jobs
retain two base-first and two candidate-first comparisons, so the Atlas
classifier still requires replicated, order-balanced agreement. The bounded
statistical universe makes the PR critical path a finite engineering gate
rather than a multi-hour batch experiment. It does not alter native-test
budgets.

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
- Keep the complete statistical suite as a merge gate: rejected because an
  observed pair takes about 249 minutes and one long-horizon SWE measurement
  alone requests about 32 minutes per revision.
- Reduce sample counts across the retained statistical targets: rejected
  because the bounded critical-target universe already meets the runtime goal
  without weakening those instruments.
- Drop non-critical benchmark scenarios: rejected because one-pass candidate
  execution retains build and runtime coverage for every registered target.

## Consequences

Benchmark-relevant PRs consume one complete smoke job, four bounded pair jobs,
and one short classification job. Python packaging-only and documentation-only
PRs do not run the Rust instrument. Atlas remains the single source of truth
for statistical classification. Report artifacts do not encode source-path
provenance, so workflow review establishes the same-path precondition that the
classifier cannot verify. Long-horizon scenarios remain functional benchmark
programs, but they are not repeated statistically on the merge-critical path.
