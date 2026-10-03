<a id="kw-ci-094"></a>

## KW-CI-094 — The recurseml check errors on itself and is permanently red [patch] — todo

priority: verification; needs: none; scope: GitHub Settings -> Applications -> Installed GitHub Apps -> recurseml

- **Finding.** `recurseml/analysis` is a commit status posted by the recurseml GitHub App with `state=error` and `description="Error occurred during analysis"`. It errored on five of the last six pull requests measured on 2026-09-09 and is the only red on them; every GitHub Actions check passed.
- **Impact.** It is not a required check, so it never blocks a merge, but every pull request shows a red check that everyone learns to skip, which is how a real failure gets waved through.
- **Not fixable from the repository:** there is no recurseml config file, and the agent token cannot enumerate or modify app installations (`user/installations` returns 403).
- **Exact action, the owner's:** uninstall recurseml, or set its repository access to exclude the members it errors on.
- **Acceptance:** the check reports real findings or is removed; it is not left erroring.
