<a id="kw-release-crates-01"></a>

## KW-RELEASE-CRATES-01 — Publish the Rust package closure [patch] — todo

priority: feature; needs: none; scope: root `Cargo.toml` workspace members; `.github/workflows/rust-release.yml`

- The approved registry closure contains 22 Rust packages. The former
  allocation probe is supplied by Mnemosyne. The Python binding and
  workspace tooling are private Cargo packages.
- [ADR 139](../docs/adr/139-native-provider-registry-cutover.md) defines the
  package boundaries and [the checklist](../docs/migrations/registry-cutover.md) records verification.
- Close when every archive is indexed with the merged revision, and a
  consumer with no lockfile resolves the published graph.
- **Version class to carry.** #1059 adds a defaulted method to the public trait `PstdStateBuilder` of `kwavers-gpu` under the commit type `fix`; by contract it is [minor], so a release tool that reads commit types alone assigns too small a bump (its first verdict, comment 6814685775, finding 3).
