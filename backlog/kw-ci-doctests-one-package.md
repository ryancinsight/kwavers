<a id="kw-ci-doctests-one-package"></a>

## KW-CI-DOCTESTS-ONE-PACKAGE — No hosted job or push gate runs the doctests of any package but `kwavers` [patch] [ci] — todo

priority: verification; needs: none; basis: 38eaf2f90eccd74d11ce3d0e836cbad7d4606ae0; scope: `.github/workflows/ci.yml`

- **Finding.** `git grep -n -e '--doc' origin/main -- .github scripts .githooks` returns three lines, `ci.yml:452`, `:585` and `:953`, each `cargo test ... --doc -p kwavers`. No other workflow, script or hook runs a doctest, and `cargo test --workspace` does not appear.
- **Push gate.** The pre-push hook runs `cargo nextest run` and `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` (`.githooks/pre-push:2359`, `:2367`); neither executes a doc example.
- **Exposure.** Doc examples in the other 22 crates under `crates/` are compiled and run by no hosted job and no push gate. Files under `src/` holding a `///` or `//!` code-fence line (three backticks), by `git grep -l -E` at `origin/main`: `kwavers-solver` 212, `kwavers-physics` 173, `kwavers-analysis` 75. These are upper bounds (a fence also closes a block and may open a `text` block); files whose fence opens with an explicit `rust`, `no_run`, `should_panic` or `compile_fail` tag number 22, 9 and 9. The count rustdoc executes is not measured.
- **Source.** kwavers#1067 review: "Separate gap: no CI job runs `--doc` for any package but `kwavers`", owned by no item; a search of this board for `doctest` and `--doc` finds only mentions in other items' evidence and acceptance lines.
- **Acceptance:** a hosted job or the push gate runs `cargo test --locked --doc` for every workspace package with a doc example, within the job time target; a pull request adding a non-compiling doc example to `kwavers-solver` fails the required aggregate check (run id recorded), and the passing run records each package's doctest count.
- **Next step:** measure one cold `cargo test --doc -p kwavers-solver` on the CI image, then place the packages in a job or shard under the five-minute target.
