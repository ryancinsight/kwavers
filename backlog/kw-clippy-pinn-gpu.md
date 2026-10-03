<a id="kw-clippy-pinn-gpu"></a>

## KW-CLIPPY-PINN-GPU — `cargo clippy --features pinn,gpu` is not warning-clean outside the solver [patch] — todo

priority: verification; needs: none; scope: `crates/kwavers-diagnostics/src/workflows/orchestrator/workflow/acquisition.rs`, `crates/kwavers/examples/comprehensive_clinical_workflow/{clinical,metrics}.rs`, `crates/kwavers/examples/electromagnetic_simulation.rs`

- **Finding.** At origin/main (2026-10-03, toolchain 1.97.0), `cargo clippy --locked -p kwavers --features pinn,gpu --all-targets` exits 0 with 71 distinct warnings; under `-- -D warnings`, the gate form, `kwavers-solver` fails with 61 errors and `kwavers` is never reached, so the example diagnostics below surface only once the solver is clean.
- **Evidence, by target and lint:** `kwavers-solver` lib 61 (27 `clippy::unused_self`, 26 `clippy::missing_errors_doc`, 7 `clippy::missing_panics_doc`, 1 `clippy::missing_fields_in_debug`; tracked as KW-LINT-047); `kwavers-diagnostics` lib 1 `clippy::unused_self` (`acquisition.rs:32`); example `comprehensive_clinical_workflow` 4 `clippy::unused_self` (`clinical.rs:10`, `clinical.rs:46`, `metrics.rs:17`, `metrics.rs:48`); example `electromagnetic_simulation` 5 `clippy::missing_errors_doc` (`electromagnetic_simulation.rs:61`, `:120`, `:177`, `:229`, `:284`).
- **Outcome:** the ten diagnostics outside the solver reach zero under `pinn,gpu --all-targets -D warnings`, each fixed at its cause (an `unused_self` receiver becomes an associated function or a real use of `self`; an `# Errors` section names the failing condition, not the template KW-ERRORS-DOCS-ARE-TEMPLATE-OUTPUT-2026-09-09 removes).
- **Acceptance:** `cargo clippy --locked -p kwavers --features pinn,gpu --all-targets -- -D warnings` reports no diagnostic in `kwavers-diagnostics` or any example once KW-LINT-047 clears the solver; reproduced on a `git archive` export outside the stack overlay with the shared target dir.
