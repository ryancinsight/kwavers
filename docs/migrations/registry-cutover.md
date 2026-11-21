# Kwavers registry cutover checklist

## Target 4.0.0 Rust registry cohort

- [x] [major] Verify one registry provider set and pass 15 Metal capacity,
  readback, and parity cases.
- [x] [major] Verify the required GitHub checks and accept the version
  increments against the previous published baselines with
  `cargo-semver-checks`.
- [x] [patch] Compile and publish all 22 packages in dependency order and
  match each crates.io archive's SHA-256 and source revision.
- [x] [patch] Verify the cohort from empty standalone lockfiles and
  build with the published provider graph.

The cohort uses merge commit `2bd53ba9af6b586c2962f608aa82b1e6e7a0819b`
from [#1124](https://github.com/ryancinsight/kwavers/pull/1124).
[Driver 0.4.0](https://crates.io/crates/kwavers-driver/0.4.0) uses merge
commit `d5853a77a32ecf31e4cd92fd6ddd2fd79b9d183c` from
[#1125](https://github.com/ryancinsight/kwavers/pull/1125). Its 7.0 MiB archive
excludes the datasheets and renders and preserves all 381 files in the
CAD model and archive directories byte for byte. Driver passes 495 cases.
Two component checks return early because their generated board fixtures
are absent from the release source.

A fresh consumer depending only on the application resolves its 21-package
Kwavers dependency graph. A second consumer resolves all 22 packages, with
Driver's Transducer feature enabled. Both resolve one Aequitas 0.5.0,
Eunomia 0.12.1, and Leto 0.52.0 provider set with zero Chrono packages.
The consumer passes the numerical smoke test with default features and
type-checks with the CUDA provider feature enabled. Each package's
`crate-<package>-v<version>` tag identifies its published source commit.

| Package | Published version |
| --- | --- |
| `kwavers` | [4.0.0](https://crates.io/crates/kwavers/4.0.0) |
| `kwavers-analysis` | [5.0.0](https://crates.io/crates/kwavers-analysis/5.0.0) |
| `kwavers-boundary` | [4.0.0](https://crates.io/crates/kwavers-boundary/4.0.0) |
| `kwavers-core` | [4.0.0](https://crates.io/crates/kwavers-core/4.0.0) |
| `kwavers-diagnostics` | [5.0.0](https://crates.io/crates/kwavers-diagnostics/5.0.0) |
| `kwavers-driver` | [0.4.0](https://crates.io/crates/kwavers-driver/0.4.0) |
| `kwavers-field` | [4.0.0](https://crates.io/crates/kwavers-field/4.0.0) |
| `kwavers-gpu` | [6.0.0](https://crates.io/crates/kwavers-gpu/6.0.0) |
| `kwavers-grid` | [4.0.0](https://crates.io/crates/kwavers-grid/4.0.0) |
| `kwavers-imaging` | [4.0.0](https://crates.io/crates/kwavers-imaging/4.0.0) |
| `kwavers-math` | [4.0.0](https://crates.io/crates/kwavers-math/4.0.0) |
| `kwavers-medium` | [5.0.0](https://crates.io/crates/kwavers-medium/5.0.0) |
| `kwavers-mesh` | [4.0.0](https://crates.io/crates/kwavers-mesh/4.0.0) |
| `kwavers-phantom` | [4.0.0](https://crates.io/crates/kwavers-phantom/4.0.0) |
| `kwavers-physics` | [4.0.0](https://crates.io/crates/kwavers-physics/4.0.0) |
| `kwavers-receiver` | [4.0.0](https://crates.io/crates/kwavers-receiver/4.0.0) |
| `kwavers-signal` | [4.0.0](https://crates.io/crates/kwavers-signal/4.0.0) |
| `kwavers-simulation` | [4.0.0](https://crates.io/crates/kwavers-simulation/4.0.0) |
| `kwavers-solver` | [4.0.0](https://crates.io/crates/kwavers-solver/4.0.0) |
| `kwavers-source` | [4.0.0](https://crates.io/crates/kwavers-source/4.0.0) |
| `kwavers-therapy` | [4.0.0](https://crates.io/crates/kwavers-therapy/4.0.0) |
| `kwavers-transducer` | [5.0.0](https://crates.io/crates/kwavers-transducer/5.0.0) |

## Target 3.0.0 Rust registry cohort

- [x] [major] Verify the native provider and clinical API cutover defined in
      [ADR 139](../adr/139-native-provider-registry-cutover.md).
- [x] [patch] Verify the exact release source, package all 22 crates, and
      confirm the archives on crates.io.
- [x] [patch] Resolve the published cohort and Kwavers main from empty
      lockfiles. Check all features, targets, and dependency kinds for Chrono.

## Aequitas 0.4 registry cutover

Kwavers source `4f82907ce181d073dda22dd731acd0e0f85f6f96` uses the Aequitas 0.4 registry cohort.
All 22 crates.io archives match their remote source tags. The driver package adds a
packaging-only change at `aa4ba0d06c25f67b7b1196fd1e2021676d7e957b`. The archive
preserves all 381 original CAD files. The driver suite has 495 passes and two
pre-existing generated-board fixture checks that return early. The release
workspace resolves 565 package versions from an empty lockfile.
The registry-only consumer resolves 506 package versions.
Kwavers main `d5853a77a32ecf31e4cd92fd6ddd2fd79b9d183c` independently resolves
565 package versions with its Aequitas 0.5 configuration. The
root-only consumer resolves 454 default and 505 all-feature package versions
without pinning transitive Kwavers crates. The all-target graphs cover normal, build,
and development edges and contain zero Chrono.

The legacy audit reports zero references, and all 39 Python binding tests pass.
Fifteen WGPU tests cover Metal absorption, bitwise coefficient readback, packed-field
limits, Bluestein workspace refusal, and pressure and velocity parity.
The cutover suite passes 2,827 cases. The triplet-pulse parity case remains
unverified because the prototype’s external RF reference files were unavailable.

## Kwavers published physics provider update

The registry lock uses one Aequitas 0.5, Leto 0.52, and Eunomia 0.12.1
provider set with the RITK and Hephaestus releases. The previous
31-binding absorption path exceeded Metal's 29-binding limit. Kwavers
#1123 reduces it to 29 with paired buffers and verifies value-preserving
GPU readback.

## 2026-11-19 — Provider and numerical module candidates

See [KW-PROVIDER-CUTOVERS](kw-provider-cutovers.md) for the
pinned providers, value-level results, and outstanding source and
integration gates. Kwavers owns the Python ndarray conversion at its
consuming boundary in accordance with Leto ADR 0017.
The native provider and Chrono graph gates pass with the cutover lock.
The [native cutover](https://github.com/ryancinsight/kwavers/pull/1118) and
[registry verification](#aequitas-04-registry-cutover) close the Aequitas 0.4 cutover. The lock guard recognizes workspace-owned packages by
name and version, and the supply-chain policy rejects Chrono in the audited graph.
