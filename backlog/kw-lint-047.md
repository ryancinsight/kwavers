<a id="kw-lint-047"></a>

## KW-LINT-047 — Solver all-feature lint ratchet [patch] — in-progress

priority: tightening; needs: none; scope: `crates/kwavers-solver` under `cargo clippy -p kwavers-solver --features pinn --all-targets`

- **Acceptance:** the measured diagnostic count reaches zero under that configuration, or each survivor carries `#[expect(lint, reason = ...)]`. `println!` in a library crate violates the floor (`print_stdout`), and `unused_self` at 28 sites is a design finding, not a mechanical fix.
- **Measured 2026-09-08:** 84 diagnostics (28 `unused_self`, 26 missing `# Errors`, 11 `println!`, 7 missing `# Panics`, 6 missing `#[must_use]`, 4 `assert!` with an equality comparison, 2 doc-link/recursion). Two increments (#759, #761) took it to **61**: all 11 `print_stdout` sites gone (they were print-debugging in `#[cfg(test)]` modules over tests that asserted nothing that could fail), 6 `#[must_use]` builders, 4 `assert_eq!`, 1 intra-doc link, and `generate_boundary_data` made an associated function.
- **Remaining 61:** 28 `unused_self` (read each before acting), 26 missing `# Errors` and 7 missing `# Panics` (must not be closed by extending the template KW-ERRORS-DOCS-ARE-TEMPLATE-OUTPUT-2026-09-09 removes; that would game the measure), and the tail.
- **Finding for whoever extends `second_deriv.rs`:** both sides are finite differences at different step sizes (`coeus_autograd` has no double-backward), so disagreement scales with the fourth derivative and `REL_TOL_SECOND` is empirical, not derived (an added point measured rel_err 2.59e-2 against the 1e-2 bound). Derive a per-point bound before extending it.
