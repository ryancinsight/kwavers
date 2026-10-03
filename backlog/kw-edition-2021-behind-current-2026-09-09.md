<a id="kw-edition-2021-behind-current-2026-09-09"></a>

## KW-EDITION-2021-BEHIND-CURRENT-2026-09-09 — 25 crates on edition 2021, resolver 2 [patch] — todo

priority: tightening; needs: none; scope: root `Cargo.toml`, every `crates/*/Cargo.toml`, `xtask/Cargo.toml`

- **Outcome:** the workspace builds on the current stable edition and resolver, with package fields that should be inherited actually inherited.
- **Measured 2026-09-09:** all 25 crates plus `xtask` declare `edition = "2021"` individually and the root declares `resolver = "2"`; the pinned toolchain (`rustc 1.97.0`) supports edition 2024 and resolver 3, so nothing external holds this back. A let-chain in a new `xtask` audit failed to compile for this reason.
- **The lint floor counts both classes:** edition or resolver behind current, and members re-declaring package fields the workspace should own.
- **Shape:** `cargo fix --edition` per crate, then flip the root to `edition = "2024"` / `resolver = "3"` with members inheriting via `edition.workspace = true`, in one change so the crates cannot drift again. Resolver 3 changes feature unification, so the feature-combination matrix is the check that matters.
- **Acceptance:** the root declares edition 2024 and resolver 3, no member declares its own `edition`, the full feature matrix passes, and a let-chain compiles.
