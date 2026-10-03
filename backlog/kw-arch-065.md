<a id="kw-arch-065"></a>

## KW-ARCH-065 — Consolidate optical transport in Hyperion [major] [arch] — in-progress

priority: architecture; needs: none; scope: Hyperion integration in `kwavers-medium`, `kwavers-physics`, `kwavers-solver`; provider-graph pins; ADR 046

- **Outcome:** Hyperion is the only owner of reduced scattering, coefficient validation, albedo, diffusion, effective attenuation, penetration depth, optical depth and transmission; every superseded Kwavers owner is deleted. Non-goals: general electromagnetic solvers, Monte Carlo ownership, photoacoustic source policy, chromophore spectra, release.
- **Decision:** [ADR 046](docs/adr/046-hyperion-optical-transport-ownership.md).
- **Acceptance:** direct consumers pass value-semantic, invalid-input, Nextest, Clippy, doctest, Rustdoc and SemVer gates against the locked published graph.
- **State:** implementation and lock reconciliation complete; affected Clippy, six-package doctests, focused Nextest, the provider-source uniqueness scan and the integrated workspace Nextest (6,168/6,168, 15 skipped) pass. Warning-denied Rustdoc passes for `kwavers-medium`, `kwavers-imaging` and `kwavers-phantom`; `kwavers-physics` keeps its tracked KW-DOC-038 baseline (557 warnings).
- **Open:** the major SemVer gate could not resolve `origin/main` because the pinned Aequitas required Eunomia `^0.6.0` while the canonical source published `0.7.0`, and the aggregate check showed distinct path and Git Leto identities at the RITK registration boundary. No compatibility adapter is introduced. Re-run the SemVer gate on the current graph.
