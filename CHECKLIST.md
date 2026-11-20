# Kwavers release checklist

## Aequitas 0.5 workspace migration

- [ ] [patch] Verify a clean build and installed Python wheel with one
      crates.io Aequitas 0.5.0 and Eunomia 0.12.1 and no Chrono.
- [ ] [patch] Adopt compatible RITK registry packages and verify
      the normalized package graph before the next registry release.

## Target 3.0.0 Rust registry cohort

- [ ] [major] Verify the native provider and clinical API cutover defined in
      [ADR 139](docs/adr/139-native-provider-registry-cutover.md).
- [ ] [patch] Verify the exact merged revision, package all 22 crates, and
      confirm the archives on crates.io.
- [ ] [patch] Resolve the published cohort and Kwavers main from empty
      lockfiles. Check all features, targets, and dependency kinds for Chrono.
