<a id="kw-integration-test-binaries"></a>

## KW-INTEGRATION-TEST-BINARIES — 89 integration-test binaries make a cold push gate expire [patch] [perf] — todo

priority: tightening; needs: none; basis: 2824e0ecf45fd62c381bf935fd032326171b1fdd; scope: `crates/kwavers/tests/`, `crates/kwavers/Cargo.toml` (`[[test]]` tables), `scripts/integration_tests.py` and its baseline

- **Finding.** `crates/kwavers/tests/` holds 89 top-level `.rs` files, each linked as its own test binary; the five subdirectories hold modules of those files and none has a `main.rs`. On the push of #1024 the gate (600 s deadline) expired on its first run: clippy ended at 112 s and the test stage had reached test 69 of 1279 when the deadline fired. The second run reused the built artifacts and passed in 534 s (clippy 23 s, tests 197 s, rustdoc 103 s, verification 211 s). A push that must build these binaries therefore costs one expired run first.
- **Outcome:** the integration tests link as a few harness binaries (`tests/<area>/main.rs` declaring the files as modules), with every test kept.
- **Acceptance:** the test names listed by `cargo nextest list -p kwavers` before and after differ only in the binary prefix (a name diff, not a count); the `scripts/integration_tests.py` baseline names follow the move; a push touching `kwavers` with no test artifacts built passes the gate on its first run inside 600 s.
- **Next step:** group the 89 files by area (solver, imaging, reference parity, validation), move one area into a harness, and measure that area's link time before and after with `cargo build --timings`.
