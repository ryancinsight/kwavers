<a id="kw-ci-052"></a>

## KW-CI-052 — Lock and parse supported Cargo workflows [patch] — review

priority: verification; needs: none; scope: `.github/workflows/ci.yml` and the other workflows that run cargo

- **Outcome:** every Cargo graph-consuming command in the workflows uses `--locked`, the workflow YAML parses, and no command hides behind malformed step indentation.
- **Finding at origin/main:** `architecture-validation.yml` no longer exists, but `ci.yml` still runs `cargo run -p xtask -- legacy-migration-audit` and `cargo run -p xtask -- burn-migration-audit` without `--locked`, and `benchmark-regression.yml` runs `cargo metadata` without it.
- **Acceptance:** `grep` of the workflows finds no graph-consuming `cargo` invocation lacking `--locked`, and `actionlint` parses every workflow.
- **Next step:** add `--locked` to the unlocked invocations and run `actionlint`.
