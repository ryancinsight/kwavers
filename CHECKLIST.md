# Kwavers release checklist

## Aequitas 0.5 source cohort

- [x] [major] Verify the provider choices and run 13 Metal capacity and parity cases.
- [ ] [major] Verify the final registry lock and required checks.

## Target 3.0.0 Rust registry cohort

- [ ] [major] Verify the native provider and clinical API cutover defined in
      [ADR 139](docs/adr/139-native-provider-registry-cutover.md).
- [ ] [patch] Verify the exact merged revision, package all 22 crates, and
      confirm the archives on crates.io.
- [ ] [patch] Resolve the published cohort and Kwavers main from empty
      lockfiles. Check all features, targets, and dependency kinds for Chrono.
