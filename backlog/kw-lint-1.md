<a id="kw-lint-1"></a>

## KW-LINT-1 — Burn down the clippy debt baseline [patch] — todo

priority: tightening; needs: none; scope: `[workspace.lints.clippy]` in the root `Cargo.toml`; every `crates/*` member

- **Outcome:** the debt block in `[workspace.lints.clippy]` is empty, so the Atlas floor is enforced whole. `unused_self`, `return_self_not_must_use` and `unnecessary_literal_bound` are already clean.
- **Baseline at adoption** (`cargo clippy -p kwavers --features pinn --lib`, which lints every path member): 1390 warnings. `unused_self` 257, `unwrap_used` 255, `missing_panics_doc` 225, `missing_errors_doc` 119, then a ~200-warning tail across 25 lints.
- **State:** the `allow` lines still listed in `[workspace.lints.clippy]` are the debt and the count; a lint leaves the block when its production count reaches zero. This entry does not duplicate the count.
- **Acceptance:** per lint, drive the production count to zero, then delete its line so the floor re-enables it. `#[expect]` counts only where the lint is genuinely inapplicable and the reason says why (`unwrap_used` inside `#[test]` is the sanctioned case).
- **Sequencing:** `missing_panics_doc`/`missing_errors_doc` burn down mechanically per crate (not by extending the KW-ERRORS-DOCS template); `unwrap_used` gets the panic-policy treatment (`?`, `ok_or_else`, `expect("invariant: ...")`), not a mass rewrite; read `unused_self` before acting, since some sites are deliberate seams.
