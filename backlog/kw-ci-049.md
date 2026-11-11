<a id="kw-ci-049"></a>

## KW-CI-049 — Align Apollo provider lock [patch] — todo

priority: verification; needs: KW-SWE-CLAMP-TEST-RESIDUALS; basis: 4d6724c0ebd02de4d22abee6cdd6d41dae236354; scope: `Cargo.lock`, provider-graph synchronization

- **Outcome:** the lock follows the merged Apollo provider, and the three packages the acceptance names stay green under `--locked`.
- **Acceptance:** the lock records Apollo `0.24.0`, the version supplied by the merged Apollo PR #45, and the focused locked suite (`cargo nextest run --locked -p kwavers-gpu -p kwavers-simulation -p kwavers-solver`, 1,036/1,036 with four skipped) stays value-semantic green.
- **Witnessed.** 2fb8661b2333809f137145e4b2bdd6877ad1ad86 on `main` moved `apollo-fft` in `Cargo.lock` from 0.23.0 to 0.24.0. The lock has advanced since: at the basis it records `apollo-fft` 0.29.4 (`Cargo.lock` line 212), so the version clause was met at that commit and does not describe `main` now.
- **Open clause.** "the focused locked suite ... stays value-semantic green". `kwavers-gpu` has no default feature and its two test targets (`tests/pstd_parity.rs`, `tests/pstd_velocity_parity.rs`) are `#![cfg(feature = "gpu")]`, and `kwavers-simulation` has no `tests/` directory, so the command as written selects the library tests of the three packages and the test binary `crates/kwavers-solver/tests/allocation_contracts.rs` (11 tests, none ignored). The pipeline runs the library tests under `--locked`: `ci.yml:555` covers `kwavers-simulation` and `kwavers-solver`, and `ci.yml:1105` runs `kwavers-gpu` with `--features gpu`. Both were green in run 43013212986 (the three shard jobs, and job 130498535670, whose log line 1865 reads "175 tests run: 175 passed, 1 skipped"). No `nextest` or `cargo test` command in `.github/workflows/*.yml` selects `allocation_contracts` (the Atlas workflows called from here were not read); `KW-SWE-CLAMP-TEST-RESIDUALS` holds that gap in its acceptance. The count 1,036 with four skipped was not reproduced.
- **Next step:** when `KW-SWE-CLAMP-TEST-RESIDUALS` adds the pipeline step for `allocation_contracts`, read that job log and delete this item in the same pull request.
