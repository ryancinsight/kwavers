# 124. Run the integration tests CI already compiles, against a failure baseline

Status: Accepted
Date: 2026-08-24

## Context

`crates/kwavers/tests/` holds 92 integration test files. The strict clippy step
compiles every one of them -- it passes `--all-targets` -- so they stay
syntactically alive and appear maintained. The test-coverage job runs
`cargo nextest run --workspace --lib`, which excludes integration tests
entirely, plus four named integration binaries.

Running the suite locally: **686 integration tests, 669 passing, 17 failing.**
Those 17 had been failing with nobody informed. Two of them, in `solver_test`,
drove the FDTD plugin with a timestep 1.71x above the solver's own CFL limit;
the plugin rejected it and the tests panicked, run after run, invisibly. They
surfaced only because an unrelated `Box`-to-`Arc` migration broke their
*compilation*, which CI does check.

A test that is compiled but never run is worse than no test: it reports
coverage that does not exist, and its assertions decay against a moving
codebase with no signal.

## Decision

Run the whole integration suite in CI, enforcing a committed baseline of the
known failures (`.config/integration-test-baseline.txt`, checked by
`scripts/integration_tests.py`).

The alternative orderings both lose. Fixing all 17 before running any of them
leaves the other 669 unprotected for however long that takes -- and they are the
tests actually guarding the physics. Running the suite and accepting red makes
the signal worthless immediately.

The baseline is a **set**, not a count. A count says "17 failures" and cannot
distinguish a fixed test from a newly broken one. The set fails two ways:

- a test fails that is not listed -- a regression, named in the failure output;
- a test in the list passes -- the entry is stale and must be deleted, so a
  fixed test cannot silently leave room for a new failure to hide in.

It may shrink, never grow. Each entry is a defect owed a fix, tracked under
`KW-INTEGRATION-TESTS-UNRUN`, not an accepted behaviour.

## Consequences

- 669 integration tests gain CI protection immediately, rather than after a
  17-defect burn-down of unknown length.
- The four-name list in the test-coverage job is deleted; it was a subset of
  what now runs, and a hand-maintained list of what to run is the mechanism
  that produced this gap.
- The 17 failures are named in one committed file rather than dispersed as
  `#[ignore]` attributes, so their count is a number the board can burn down
  and CI reports it every run.
- The script refuses to report success when the suite produces no summary. A
  check that passes because it never executed is the failure mode this ADR
  exists to close, and it would be an unusually direct way to reintroduce it.
- Local runs need `--unlocked`: a tree under the Atlas development overlay
  resolves first-party crates to local paths, which `--locked` rejects before
  the suite starts.

## Alternatives rejected

- **`#[ignore]` on each failing test.** Rejected: it disperses the inventory
  across seven files, and an ignored test reads as a decision about that test
  rather than as debt owed. The `--run-ignored` escape also blurs it with the
  28 pre-existing ignores, which are a different thing.
- **A failure-count threshold.** Rejected: it cannot tell a fixed test from a
  newly broken one, so a regression hides behind any fix landing beside it.
- **Fix all 17 first.** Rejected as an ordering, not as work: it is the same
  work either way, but it leaves the passing majority unguarded meanwhile.
