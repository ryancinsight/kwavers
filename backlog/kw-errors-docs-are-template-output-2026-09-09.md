<a id="kw-errors-docs-are-template-output-2026-09-09"></a>

## KW-ERRORS-DOCS-ARE-TEMPLATE-OUTPUT-2026-09-09 — 300 functions document an error they cannot return [patch] — todo

priority: verification; needs: none; scope: `# Errors` sections across `crates/`; `xtask/src/errors_doc_audit.rs`

- **Outcome:** an `# Errors` section names the condition that produces the error, or it does not exist.
- **Measured 2026-09-09:** 1141 copies of ``- Returns [`Err`] if an internal constraint is violated.`` in 544 files and 297 of a second template (``Propagates any [`crate::KwaversError`] ...``). 300 of those sit on functions that cannot fail (the signature returns neither `Result` nor `Option`); the remaining ~840 sit on fallible functions but name no invariant, input range or remedy.
- **Delivered (#762):** `cargo run -p xtask -- audit-errors-docs` runs inside `Validate Clean Architecture`; all 350 false sections it found are removed (1040 lines, 226 files, only `///` lines deleted), so its `BASELINE` is 0 and reintroducing one fails the audit and names the site.
- **Why it matters:** `clippy::missing_errors_doc` fires on a missing section, not a useless one, so a template silenced it repo-wide. KW-LINT-047's 26 missing-`# Errors` rows must not be closed by extending the template; that would be gaming the measure.
- **Remaining:** replace the ~840 contentless sections on genuinely fallible functions, per crate, with the actual failure conditions.
- **Acceptance:** no `# Errors` section on a function returning neither `Result` nor `Option` (enforced by the audit, done), and the remaining sections name a condition rather than a category.
