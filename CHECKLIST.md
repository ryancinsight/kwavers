# Kwavers release checklist

## Target 3.0.0 Rust registry cohort

- [x] [major] Verify the native provider and clinical API cutover defined in
      [ADR 139](docs/adr/139-native-provider-registry-cutover.md).
- [x] [patch] Verify the exact release source, package all 22 crates, and
      confirm the archives on crates.io.
- [x] [patch] Resolve the published cohort and Kwavers main from empty
      lockfiles. Check all features, targets, and dependency kinds for Chrono.
