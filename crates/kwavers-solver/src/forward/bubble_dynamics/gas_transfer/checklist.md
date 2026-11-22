# Published 5.2.0 [minor]

- [x] [minor] Implement separately selectable compressible radial motion and
  gas heat transfer through the existing conservative bubble solver.
- [x] [minor] Pass the 220 and 230 kHz acceptance and refinement gates,
  focused legacy diffusion regressions, all-feature Physics tests and
  default-feature SemVer checks against 5.1.0.
- [x] [minor] Merge the approved candidate after the required CI and Architecture gates pass.
- [x] [minor] Publish Physics before Solver and verify each registry archive.
- [x] [minor] Resolve both published versions in a fresh consumer and run the radial and pulse-history checks.

## 5.2.0 registry verification

Both packages trace to merge commit `9daa1fab5028ace90a075a635e56f482a8f3172d` from
[PR #1142](https://github.com/ryancinsight/kwavers/pull/1142). The
[Physics workflow](https://github.com/ryancinsight/kwavers/actions/runs/45065305939)
publishes the upstream crate. Its archive and radial kernel pass verification
before the [Solver workflow](https://github.com/ryancinsight/kwavers/actions/runs/45067369191)
publishes the dependent crate. Archive hashes match the registry index and
fresh Cargo lockfile. Changed source files match the tagged revision.

| Crate | Registry version | Registry archive SHA-256 | Source tag |
| --- | --- | --- | --- |
| `kwavers-physics` | [5.2.0](https://crates.io/crates/kwavers-physics/5.2.0) | `f0e27fc8aed08dfc24d81eff1428cbdf3c77359f23777d0871969736763b70bb` | [`crate-kwavers-physics-v5.2.0`](https://github.com/ryancinsight/kwavers/releases/tag/crate-kwavers-physics-v5.2.0) |
| `kwavers-solver` | [5.2.0](https://crates.io/crates/kwavers-solver/5.2.0) | `f0fdfa8351e18d844895fdbb4b665dd5465fd89a92cd44fcec32a08719508b18` | [`crate-kwavers-solver-v5.2.0`](https://github.com/ryancinsight/kwavers/releases/tag/crate-kwavers-solver-v5.2.0) |

The fresh arm64 macOS consumer uses Rust 1.97.0 and a separate Cargo home.
It resolves both crates from crates.io at exactly 5.2.0 and resolves Solver's
Physics edge to that verified package. The Physics equilibrium and driven
Keller–Miksis checks pass. All 15 Solver acceptance tests pass, including the
220 and 230 kHz histories and independent time and radial-grid refinements.
The consumer runs `cargo metadata --locked`, `cargo nextest run --locked
--offline --test compressible_thermal_bubble`, and `cargo run --locked
--offline`. The test and CSV fixture match the Solver archive.

## Published 5.1.0 [minor]

KW-PHYS-004 is complete. The [design](DESIGN.md) retains the model and
acceptance evidence.

- [x] [minor] Expose the opt-in spherical Rayleigh–Plesset and Henry-transfer
  solver with a conservative gas budget, transactional steps, and a
  closure-consistent equilibrium radius.
- [x] [minor] Check prescribed transport and both independent 20 kPa
  polytropic histories. Check the isothermal time-step, grid, and reservoir
  sensitivities against the phase error budgets.
- [x] [minor] Record the verified operating range in the [design](DESIGN.md)
  and the new surface in the 5.1.0 changelog.
- [x] [minor] Publish Physics and then Solver at 5.1.0 with release approval.
  Verify each registry archive, checksum, and source revision.
- [x] [minor] Resolve both published versions in a fresh consumer and run the
  driven and quiet gas-transfer smoke test.

## Registry verification

Both crates published on November 22, 2026. The archive SHA-256 checksums
match the crates.io sparse index and the consumer's Cargo lockfile. Both
archive VCS records and GitHub release tags point to
`148ca1d26ebfe299d3e002a49662c17f93f18e78`.
[Merge PR](https://github.com/ryancinsight/kwavers/pull/1138) ·
[Successful publish job](https://github.com/ryancinsight/kwavers/actions/runs/44989877205/job/137112506597)

| Crate | Published version | Registry archive SHA-256 | Source tag |
| --- | --- | --- | --- |
| `kwavers-physics` | [5.1.0](https://crates.io/crates/kwavers-physics/5.1.0) | `0a674484ae7ebbf0ec93a76aee0ff2d7d9d6500627311e6ea87e964d1a2ab085` | [kwavers-physics-v5.1.0](https://github.com/ryancinsight/kwavers/releases/tag/kwavers-physics-v5.1.0) |
| `kwavers-solver` | [5.1.0](https://crates.io/crates/kwavers-solver/5.1.0) | `6a9a5f2cba24cd714ea120da33809e6cade2c86b3bb5be540761e8744fde4a69` | [kwavers-solver-v5.1.0](https://github.com/ryancinsight/kwavers/releases/tag/kwavers-solver-v5.1.0) |

## Fresh consumer

A standalone consumer on Rust 1.97.0 (`aarch64-apple-darwin`) started with
a new Cargo home and generated its lockfile from crates.io. It pinned both
crates to `=5.1.0` and resolved Solver's dependency to the same Physics
5.1.0 instance. Cargo metadata confirmed registry-only dependencies.
`cargo run --locked` built and ran the published API, passing the driven
and quiet coupling steps, gas conservation, and equilibrium-radius
assertions. Net bubble gas transfer was `2.073411e-21 mol` and the gas
balance residual was `1.880791e-36 mol`.
