# Kwavers registry cutover checklist

## Aequitas 0.5 source cohort

- [x] [major] Verify one registry provider set and pass 15 Metal capacity,
  readback, and parity cases.
- [ ] [major] Verify the final registry lock and required checks.

## Target 3.0.0 Rust registry cohort

- [ ] [major] Verify the native provider and clinical API cutover defined in
      [ADR 139](../adr/139-native-provider-registry-cutover.md).
- [ ] [patch] Verify the exact merged revision, package all 22 crates, and
      confirm the archives on crates.io.
- [ ] [patch] Resolve the published cohort and Kwavers main from empty
      lockfiles. Check all features, targets, and dependency kinds for Chrono.

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
[Registry verification](../../backlog/kw-release-crates-01.md) remains open for
the Aequitas 0.4 candidate. The manifest uses the registry cohort. The
lock guard recognizes workspace-owned packages by name and version, and
the supply-chain policy rejects Chrono in the audited graph.
