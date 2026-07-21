# ADR 045: Atlas benchmark regression gate

- Status: accepted
- Date: 2026-07-20
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

## Decision

PRs that change Rust production, dependency, or benchmark inputs run a
dedicated workflow. It checks out the PR base and head, overlays the candidate
`crates/kwavers/benches` tree onto the base checkout, and runs the complete
plotting-enabled `kwavers` Criterion suite for both revisions.

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
pinned at `71cdc54c509d54e10daac1032d328d0b006a2ce5`. Four isolated runners each
execute one complete base/head pair. Two use order `A B` and two use `B A`,
where `A` is the base revision and `B` is the candidate. Each comparison
therefore remains within one machine, while the phase-reversed matrix balances
revision order and samples independent hosted-runner variation. A regression
is reported only when all four confidence intervals agree in direction and
cover the same benchmark universe.

The Atlas tool derives per-comparison confidence as `1 - 0.05 / m` for `m`
benchmarks. Missing results, benchmark-universe mismatches, malformed
estimates, and insufficient confidence fail closed. There is no empirical
percentage threshold.

The 315-minute pair-job budget is specific to this instrumented suite. Exact
hosted run `29814752294` disproved the earlier target-count model: serialized
execution reached the job bound during the third of eight full-suite
measurements. The first complete base/head pair took about 249 minutes,
including builds, all 19 plotting-eligible Criterion targets, and the full
sample counts. One matrix member now owns each pair, so the critical path is
one observed pair instead of four serialized pairs. This preserves all eight
measurements, targets, samples, and assertions without increasing the bound.
It does not alter native-test budgets.

## Rejected alternatives

- Retain the same-run save/check script: rejected because both sides contain
  the same measurement.
- Compare one base/head pair: rejected because a slowdown confined to run
  order cannot be distinguished from a production regression.
- Run all four pairs serially on one runner: rejected after exact hosted run
  `29814752294` demonstrated an approximately 18-hour schedule against the
  finite 315-minute job bound. Each base/head comparison remains co-located;
  independent pair runners add an observed replication dimension without
  mixing machines inside a confidence interval.
- Reduce benchmark targets or samples to retain the old timeout: rejected
  because that changes the measurement instrument.

## Consequences

Benchmark-relevant PRs consume four long, bounded pair jobs followed by one
short classification job. Documentation-only PRs do not run the instrument.
The Python gate and its duplicate threshold policy are deleted; Atlas remains
the single source of truth for statistical classification.
