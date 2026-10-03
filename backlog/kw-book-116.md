<a id="kw-book-116"></a>

## KW-BOOK-116 — Make the book gate execute a Rust oracle [patch] — review

priority: verification; needs: none; scope: `.github/workflows/book-pages.yml`, `docs/book/examples/basic_simulation.md`

- **Outcome:** the published Kwavers book executes one deterministic Rust oracle and the shared workflow builds the exact package before `mdbook test`.
- **State:** implementation complete locally (locked `kwavers` build, `cargo fmt --check`, `mdbook test docs/book`, `mdbook build docs/book` pass); hosted verification pending. The standalone link checker still reports pre-existing source links outside the book root and incomplete mathematical-link parsing, tracked as separate documentation cleanup.
- **Acceptance:** `mdbook test docs/book` executes the analytical CFL oracle, `mdbook build docs/book` passes, and the workflow pins the package and library target explicitly.
- **Next step:** re-validate against `book-pages.yml` at the default branch; close if the oracle and pinned target are present, else re-claim.
