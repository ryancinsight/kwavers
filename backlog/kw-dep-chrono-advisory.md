<a id="kw-dep-chrono-advisory"></a>

## KW-DEP-CHRONO-ADVISORY — Clear the chrono advisory from the audited graph [patch] — todo

priority: correctness; needs: none; scope: `deny.toml`, the audited dependency graph, the upstream `dicom-core` release and the `plotly`/`serde_with` feature edge; milestone: M-solvers-core

- **Finding.** `RUSTSEC-2026-0543` marks `chrono` unmaintained, every version affected, so the Security Audit gate fails on the resolved graph and the default branch is red. Three of the four direct edges now name `std::time::SystemTime` and `time::OffsetDateTime`; the fourth is the `kwavers` package's integration-test dev-dependency.
- **Residual.** The audited graph still resolves `chrono` through two third-party paths: `dicom-core` (via `ritk-io`) keeps it for DICOM date values, and `serde_with` (via `plotly` under the `all-features` graph) keeps it as an optional feature dependency. Neither path is Kwavers code.
- **Hold.** `deny.toml` ignores `RUSTSEC-2026-0543` for exactly that residual, naming the two paths and the removal trigger.
- **Acceptance.** `cargo deny check advisories` reports no `chrono` advisory with the entry present; the entry is deleted when the audited graph drops `chrono` — the audited `dicom-core` release no longer depends on it (or Kwavers stops reaching it) and `serde_with`'s `chrono` feature leaves the audited graph — witnessed by a `cargo tree -e normal` naming no `chrono` package and a `cargo deny check advisories` run with the entry removed.
- **State.** Three of the four direct edges are gone and the advisory hold is recorded (2026-11-18); the `kwavers` package's dev-dependency and the residual third-party paths remain.
- **Next step.** Watch the `dicom-core` release line for a `chrono`-free release and re-run the audit graph; drop `serde_with`'s `chrono` feature from the audited feature set when the graph's feature selection stops enabling it.
