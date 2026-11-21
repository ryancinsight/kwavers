# Kwavers registry cutover checklist

## Aequitas 0.5 source cohort

- [x] [major] Verify one registry provider set and pass 15 Metal capacity,
  readback, and parity cases.
- [ ] [major] Verify the final registry lock and required checks.

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
[Registry verification](../../backlog/kw-release-crates-01.md) closes the
Aequitas 0.4 cutover. The lock guard recognizes workspace-owned packages by
name and version, and the supply-chain policy rejects Chrono in the audited graph.
