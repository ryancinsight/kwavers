<a id="kw-release-crates-01"></a>

## KW-RELEASE-CRATES-01 — Publish the Rust package closure [patch] — blocked

priority: feature; needs: none; scope: root `Cargo.toml` workspace members; `.github/workflows/rust-release.yml`

- **Blocker:** release authority. Publication is the release delivery state and needs the user's explicit authorization; no agent grant covers it.
- **Re-open trigger:** the user authorizes publishing the Rust closure.
- **Preparation verified 2026-09-08:** the workspace holds exactly 23 publishable packages with `kwavers-python` the sole `publish = false` member, and no publishable crate depends on it, so the publishable set is dependency-closed.
- **Not yet true:** crates.io indexes none of them (`kwavers`, `kwavers-core`, `kwavers-solver`, `kwavers-physics`, `kwavers-alloc-probe` return no published version), so "every version is indexed" stays outstanding until the release runs.
