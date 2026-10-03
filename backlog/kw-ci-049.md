<a id="kw-ci-049"></a>

## KW-CI-049 — Align Apollo provider lock [patch] — review

priority: verification; needs: none; scope: `Cargo.lock`, provider-graph synchronization

- **Acceptance:** the lock records Apollo `0.24.0`, the version supplied by the merged Apollo PR #45, and the focused locked suite (`cargo nextest run --locked -p kwavers-gpu -p kwavers-simulation -p kwavers-solver`, 1,036/1,036 with four skipped) stays value-semantic green.
- **Re-validation finding (2026-10-03):** at origin/main `Cargo.lock` records `apollo-fft` 0.27.0, so the acceptance clause naming 0.24.0 is obsolete and cannot be met as written.
- **Residual:** three solver tests were slow and one exceeded 30 s in that run; none was named, so a profile-guided item needs the test names first (KW-TEST-SEPARATION-METHODS-BUDGET is one such case).
- **Next step:** close on confirmation that the lock-alignment intent is carried by later provider advances.
