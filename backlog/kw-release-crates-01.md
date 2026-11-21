<a id="kw-release-crates-01"></a>

## KW-RELEASE-CRATES-01 — Publish the Rust package closure [patch] — done

priority: feature; needs: none; scope: root `Cargo.toml` workspace members; `.github/workflows/rust-release.yml`

- The approved registry closure contains 22 Rust packages. The former
  allocation probe is supplied by Mnemosyne. The Python binding and
  workspace tooling are private Cargo packages.
- [ADR 139](../docs/adr/139-native-provider-registry-cutover.md) defines the
  package boundaries and [the checklist](../CHECKLIST.md) records verification.
- All 22 crates.io archives match their remote source tags. The release uses `4f82907ce181d073dda22dd731acd0e0f85f6f96`
  and the driver packaging commit `aa4ba0d06c25f67b7b1196fd1e2021676d7e957b`.
  Fresh registry-only consumers resolve the published packages with zero Chrono.
  See the [native cutover](https://github.com/ryancinsight/kwavers/pull/1118),
  [release branch](https://github.com/ryancinsight/kwavers/tree/release/aequitas-0.4),
  [legacy audit repair](https://github.com/ryancinsight/kwavers/pull/1122),
  and [Metal absorption repair](https://github.com/ryancinsight/kwavers/pull/1123).
- **Version class to carry.** #1059 adds a defaulted method to the public trait `PstdStateBuilder` of `kwavers-gpu` under the commit type `fix`; by contract it is [minor], so a release tool that reads commit types alone assigns too small a bump (its first verdict, comment 6814685775, finding 3).
