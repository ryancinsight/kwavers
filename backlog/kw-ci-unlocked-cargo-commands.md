<a id="kw-ci-unlocked-cargo-commands"></a>

## KW-CI-UNLOCKED-CARGO-COMMANDS — Three kinds of cargo command in the workflows run without `--locked` [patch] — todo

priority: verification; needs: none; basis: b3b2c3a372c8ceaf6f330637383e4d2cd11c43c4; scope: `.github/workflows/ci.yml`

- **Origin.** Split from `KW-CI-052`, which #1054 closes. Its acceptance asked that no graph-consuming `cargo` invocation lack `--locked`.
- **Read at the basis.** `git grep -n -E "cargo " -- .github/workflows`, less the lines that hold `--locked` and the comment lines, leaves eleven lines: three step names, the first lines of four commands of `benchmark-regression.yml` that carry the flag on a later line (`cargo metadata` and `cargo bench`, twice each), and four commands of `ci.yml`: `cargo fmt -- --check` in two steps named `Check formatting`, `cargo audit` in `Check for security vulnerabilities` and `cargo +nightly miri setup` in `Setup miri`. #1045 held that none of the three needs the flag, from recall and not from a lookup, and its judge agreed on the same basis (comment 6794943701 there: "my agreement is also from recall, not a lookup"). No lookup and no run has been made since.
- **Question:** does any of the three resolve the dependency graph, so that it could act on a lock other than the committed one?
- **Method and budget:** one sitting. For each command read the tool's documentation of the flag, then run the command in an export of the default branch from which `Cargo.lock` is removed, and see whether a lock is written.
- **Deliverable:** `--locked` on each command that resolves the graph, or a comment beside the command that says what was read and run; this item is deleted by that change.
