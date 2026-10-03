<a id="kw-arch-036"></a>

## KW-ARCH-036 — Clinical-imaging dependency boundary [major] — review

priority: architecture; needs: none; scope: `kwavers-physics`, `kwavers-solver`, direct clinical consumers

- **Driver:** a downstream consumer's forward PSTD package reached `ritk-filter` through unconditional clinical image I/O and registration dependencies.
- **Acceptance:** that consumer no longer reaches `ritk-filter`; PSTD builds and its finite-aperture boundary regression runs through the native Kwavers path; every in-workspace user of gated clinical APIs opts in explicitly.
- **Design:** [ADR-036](docs/adr/036-clinical-imaging-feature-boundary.md); `kwavers-physics` gates its registration dependency behind `clinical-imaging`.
- **Evidence:** locked offline Physics Nextest 1,554/1,554 without the feature and 1,710/1,710 with it; consumer Nextest 29/29; reverse dependency resolution reports no `ritk-filter`.
- **Next step:** re-validate at the default branch; close if the gate holds.
