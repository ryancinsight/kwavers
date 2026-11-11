<a id="kw-ci-051"></a>

## KW-CI-051 — Remove obsolete deployment workflow [patch] — todo

priority: verification; needs: none; basis: a3a72689be0fbfed3f64231c52186eac79dedc15; scope: `.github/workflows/deploy.yml` (deleted)

- **Outcome:** no workflow refers to deployment artifacts the repository does not hold, and every workflow file is checked for syntax by the gate.
- **Acceptance:** no workflow references absent deployment artifacts or invalid step syntax; the supported CI surface stays architecture validation, migration audit and the provider-aware build/test workflow.
- **Witnessed.** `.github/workflows/` holds no `deploy.yml`, and `git grep -n -i -E "dockerfile|k8s|kubernetes|docker" -- .github` prints nothing.
- **Open clauses.** "invalid step syntax": no workflow lint runs; `git grep -n -i -E "actionlint|zizmor" -- .github scripts` prints nothing, so the only check is that the hosting service instantiates runs. "the supported CI surface stays architecture validation, migration audit and the provider-aware build/test workflow": `.github/workflows/` holds nine files (`adr-index.yml`, `benchmark-regression.yml`, `book-pages.yml`, `ci.yml`, `compile-timing.yml`, `gpu-parity.yml`, `python-release.yml`, `python-wheel-smoke.yml`, `rust-release.yml`) and no `architecture-validation.yml`; whether the clause is still wanted in that form is undecided.
- **Next step:** add a workflow lint job that the `CI gate` job needs; trace each of the nine files to the change that added it and settle the surface clause against those records; then close.
