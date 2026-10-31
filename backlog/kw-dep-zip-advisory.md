<a id="kw-dep-zip-advisory"></a>

## KW-DEP-ZIP-ADVISORY — The Security Audit fails on zip < 9 through gaia-mesh and consus-npy [patch] — todo

priority: correctness; needs: none; scope: Cargo.toml, Cargo.lock (gaia-mesh, moirai, consus-npy); upstream consus workspace zip requirement

- **Finding.** `cargo deny check` reports RUSTSEC-2026-0367, 0368 and 0369 (zip < 9.0.0). The failure is in CI run 39291584134, job 118194106642, which fails the aggregate CI gate on `main` and on every pull request.
- **Sources.**
  - zip 2.4.2 comes through gaia-mesh 0.5.2 (locked at a81f558e4).
  - zip 6.0.0 comes through consus-npy 0.2.0, because consus' workspace requires `zip = "6.0"`.
  - gaia 0.6.9 (ea7411d) is on zip 9.0.4 but requires Moirai 0.10, while kwavers pins moirai-parallel 0.8.2.
  - metis#505 records the same provider-cohort conflict.
- **Next step.** Bump consus to zip ≥ 9. Then run the co-evolution sweep in layer order, advancing kwavers to gaia 0.6.9+ and Moirai 0.10 and migrating callers natively. Do not add an advisory ignore, an older pin or a gate change.
- **Acceptance.** `cargo deny --config deny.toml --manifest-path crates/kwavers/Cargo.toml check` passes, and the CI gate is green on `main`.
