<a id="kw-ci-052"></a>

## KW-CI-052 — Lock and parse supported Cargo workflows [patch] — blocked

priority: verification; needs: none; basis: 4d6724c0ebd02de4d22abee6cdd6d41dae236354; scope: `.github/workflows/ci.yml` and the other workflows that run cargo

- **Outcome:** every Cargo graph-consuming command in the workflows uses `--locked`, the workflow YAML parses, and no command hides behind malformed step indentation.
- **Acceptance:** `grep` of the workflows finds no graph-consuming `cargo` invocation lacking `--locked`, and `actionlint` parses every workflow.
- **Witnessed.** #1045 added `--locked` to the two `cargo run -p xtask` audit steps and to `cargo deny` (`ci.yml` lines 1144, 1179 and 919). At the basis the `cargo` commands in the nine workflow files that carry no `--locked` are `cargo fmt -- --check` (`ci.yml` lines 298 and 749), `cargo audit` (327) and `cargo +nightly miri setup` (935); the judge of #1045 accepted, from recall, that none of them resolves the workspace graph. Run 43014706076 on `main` at the basis ran both audit jobs (130503569822, 130503569960) and `Security Audit` (130503569739) with the flag, each with conclusion success.
- **Open clause.** "`actionlint` parses every workflow". No job runs it. #1045 first called the shared Atlas guard, which failed before `actionlint` started (run 43010771547, job 130489568492): it rejects the local action that `benchmark-regression.yml` (lines 89 and 262) and `python-wheel-smoke.yml` (lines 133 and 159) reach through a checkout `path:`. Whether `actionlint` passes on the nine files is not known.
- **Blocker:** the Atlas guard cannot analyse those two files; ryancinsight/atlas#674 (open, not merged) changes it to resolve such references. **Re-open trigger:** that pull request merges.
- **Next step:** add a `workflow-lint` job that calls the guard at the commit carrying the fix and make `CI gate` need it; read what `actionlint` reports on the nine files and fix those findings.
