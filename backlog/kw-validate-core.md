<a id="kw-validate-core"></a>

## KW-VALIDATE-CORE — Consolidate positive/finite validators in kwavers-core [patch] — todo

priority: cleanup; needs: none; scope: 16 `validate_positive*` helpers across 8 crates, new `kwavers_core::validation` module

- **Finding.** 16 near-identical private helpers (`validate_positive_finite`, `validate_positive`, `validate_nonnegative_finite`, `validate_point`) across kwavers-analysis, diagnostics, physics, solver, source, therapy, transducer. All take `(parameter: &str, value: f64)` and return `KwaversResult<()>` via per-site `invalid(...)` constructors.
- **Home decision.** `kwavers-core` (new `kwavers_core::validation`), not eunomia: every helper returns `KwaversResult<()>`, which the foundation crate cannot name.
- **Blocked note.** Do not start while `crates/kwavers-transducer/src/transducers/physics/*` holds unrestored peer work (see `gap_audit.md`); CFDrs carries a parallel validator family worth one coordinated shape.
- **Outcome:** one documented constructor per shape in `kwavers_core::validation` with `# Errors` sections; all 16 sites delegate with no behavior change.
- **Acceptance:** `cargo test -p kwavers-core` plus affected crate suites green; clippy clean; no duplicate validator definitions remain under `crates/`.
