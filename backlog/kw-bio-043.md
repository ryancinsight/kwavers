<a id="kw-bio-043"></a>

## KW-BIO-043 — Asclepius response ownership [arch] [major] — review

priority: architecture; needs: none; scope: CEM43, Arrhenius damage and independent-insult composition in `kwavers-physics`, `kwavers-therapy`, `kwavers-python`; provider manifests

- **Outcome:** production CEM43 and Arrhenius formulas exist only in Asclepius; every in-scope consumer delegates through Aequitas quantities; invalid observations return errors without partially updating persistent state; Python stays a conversion-only PyO3 boundary. Grids, treatment policy, tissue parameter catalogs and the independent bioheat validation oracle stay Kwavers-owned.
- **Decision:** [ADR 044](docs/adr/044-asclepius-response-ownership.md); the change is breaking (public duplicate response functions removed).
- **Acceptance:** the independent solver oracle still matches published 42/43/44 degree Celsius cases; one public Asclepius source is in the dependency graph; production residue scans retain only the independent oracle and test equations.
- **Evidence:** warning-denied all-feature Clippy, 2,070 native tests, 10 Python tests, 29 doctests, Rustdoc, and the major SemVer gate pass (a minor-class check reports seven major-breaking categories).
- **Next step:** re-validate at the default branch (`thermal/response/cem43.rs`); close if the residue scan holds.
