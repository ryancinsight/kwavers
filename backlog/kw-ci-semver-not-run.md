<a id="kw-ci-semver-not-run"></a>

## KW-CI-SEMVER-NOT-RUN — No hosted run compares a public surface when a change lands [patch] — todo

priority: verification; needs: none; basis: 9b2b75c590efb6503f92d01aee2fae79ef73690e; scope: `.github/workflows/ci.yml`

- **Origin.** #1059 adds a public trait method to `kwavers-gpu`; no hosted job compared its API with a baseline, and the local tool did not complete.
- **Read at the basis.** `ci.yml` calls the Atlas `semver-gate.yml` at 50802ef4c975bdb1e668b94a700d8310f0348e9c only when the event is not a pull request (line 137). In that workflow, `semver-pr` runs only on a pull request (its line 141) and `semver-release` only on the `release-gate` input, a release event or a tag push (its lines 243 to 246). So a push to `main` runs neither. The comment at `ci.yml` lines 126 to 130 says the blocking comparison runs on pushes to main. `rust-release.yml` runs the release comparison on `workflow_dispatch` and for a release tag that starts with `crate-` (lines 36 to 41).
- **Observed.** In the `main` push runs 43139311708 and 43109577233 both SemVer jobs concluded `skipped`.
- **Run by the integrator of #1059**, from its lane `worktrees/kw-swefix2` on its development host: `cargo semver-checks -p kwavers-gpu --baseline-rev origin/main --only-explicit-features --features gpu` (0.51.0) stops in `cargo update`: `eunomia = "^0.11.0"`, "candidate versions found which didn't match: 0.12.0" (`KW-DEP-EUNOMIA-REQUIREMENT-LAG`).
- **Acceptance:** a change to the source of a publishable package (`publish = true`; none is on crates.io yet, `KW-RELEASE-CRATES-01`) is compared with its baseline in a hosted run that executes (not `skipped`) before or when it lands, or the comment in `ci.yml` states where the comparison runs and an ADR records why not on landing.
- **Next step:** read the last run of the release comparison in `rust-release.yml`; then choose between a `main`-push comparison against the previous `main` commit and the corrected comment.
