<a id="kw-ci-per-pr-matrix-starvation-2026-09-08"></a>

## KW-CI-PER-PR-MATRIX-STARVATION-2026-09-08 — Per-PR CI runs the scheduled matrix [patch] [ci] [perf] — in-progress

priority: tightening; needs: none; scope: `.github/workflows/ci.yml`, `main` required checks

- **Outcome:** the pull-request path runs the affected-scope checks; the full matrix (extra toolchains, heavy validation, coverage, feature matrix) moves to the scheduled selection-drift backstop, bringing verification round-trip toward the five-minute job target.
- **Measured 2026-09-08, `main` runs:** CI/CD Pipeline 94 and 114 min wall clock, Architecture Validation 75 min, Deploy mdBook 64 and 95 min; job runtimes sum to about 82 min but run in parallel, so most wall clock is inter-job queueing (runner starvation). One pull request started about 24 checks across 5 always-on workflows.
- **Step (1) delivered in #755:** beta/nightly toolchains, heavy validation and coverage moved to schedule; job count per pull request 15 -> 9, nothing deleted. Wall clock was not comparable (the run spanned a 90-minute outage).
- **Step (1b) open, a coverage decision (re-validate: the `Architecture Validation` workflow no longer exists at origin/main):** it added five `Build with Feature Combinations` legs (minimal, gpu, plotting, pinn, full) to every pull request. Recommendation: `minimal` and `full` on pull requests (the two endpoints, since `full` alone misses feature-gated code referenced unconditionally), all five on schedule.
- **Step (2) open:** re-measure the verification round-trip on a normally serviced queue.
- **Step (3) open:** wire the remaining affected-scope checks as required status checks and let auto-merge enqueue (KW-CI-115).
- **Acceptance:** a pull request's round-trip is measured after (1), the moved jobs still run on schedule with unchanged commands and budgets, and no check is deleted.
