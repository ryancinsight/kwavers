<a id="kw-unc-morris-saltelli"></a>

## KW-UNC-MORRIS-SALTELLI — Sensitivity analysis offers correlation screening only, while the provider now has the Morris and Saltelli estimators [minor] — todo

priority: feature; needs: none; basis: 38eaf2f90eccd74d11ce3d0e836cbad7d4606ae0; scope: `crates/kwavers-analysis/src/ml/uncertainty/sensitivity_analysis/`, `crates/kwavers-analysis/src/ml/uncertainty/quantifier.rs`, `docs/adr/043-tyche-uncertainty-provider.md`

- **Read at the basis.** `SensitivityAnalyzer::analyze` (`analyzer.rs:41-73`) runs Latin-hypercube sampling into `CorrelationScreening` and returns squared correlations; its only non-test caller is `quantifier.rs:255`. ADR 043 (`:44-46`) says genuine Morris and Saltelli methods "remain Tyche provider capability gaps"; `gap_audit.md:263-265` says Tyche does not yet provide them.
- **Provider state.** The locked Tyche (456b4c489, `Cargo.lock`) defines `MorrisScreening` (`statistics/sensitivity/elementary.rs:172`) and Saltelli `SobolIndices` (`sobol.rs:23`), both with an `OUTPUTS` const-generic; Tyche's `backlog.md:149-150` at 1172fd967 records them delivered.
- **Outcome:** Kwavers exposes Morris screening and Saltelli first- and total-order indices through the Tyche types over `ParameterSpace` with a seeded design, restoring the capability ADR 043 removed without a local estimator.
- **Acceptance:** tests assert the estimators' values on a function with closed-form indices whose source is cited, within bounds derived from the sample count; the squared-correlation report keeps its own name; ADR 043 carries a dated revision note and the stale gap record is gone.
- **Next step:** read the Tyche estimator APIs at the locked revision and add `morris` and `sobol` entry points beside `analyze`.
