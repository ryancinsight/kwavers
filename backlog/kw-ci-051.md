<a id="kw-ci-051"></a>

## KW-CI-051 — Remove obsolete deployment workflow [patch] — todo

priority: verification; needs: KW-CI-052; basis: a3a72689be0fbfed3f64231c52186eac79dedc15; scope: `.github/workflows/deploy.yml` (deleted)

- **Outcome:** no workflow refers to deployment artifacts the repository does not hold or carries invalid step syntax, and the CI surface is the one the acceptance names.
- **Acceptance:** no workflow references absent deployment artifacts or invalid step syntax; the supported CI surface stays architecture validation, migration audit and the provider-aware build/test workflow.
- **Witnessed.** `.github/workflows/` holds no `deploy.yml`, and `git grep -n -i -E "dockerfile|k8s|kubernetes|docker" -- .github` prints nothing.
- **Open clauses.** "invalid step syntax": no workflow lint runs; `git grep -n -i -E "actionlint|zizmor" -- .github scripts` prints nothing, so the only check is that the hosting service instantiates runs. Reported by the judge of #1044 and not re-read: `python-release.yml` differs from the file at its last completed run, so no run has parsed the current file. "the supported CI surface stays architecture validation, migration audit and the provider-aware build/test workflow": `.github/workflows/` holds nine files (`adr-index.yml`, `benchmark-regression.yml`, `book-pages.yml`, `ci.yml`, `compile-timing.yml`, `gpu-parity.yml`, `python-release.yml`, `python-wheel-smoke.yml`, `rust-release.yml`) and no `architecture-validation.yml`. `ci.yml` records the architecture validation and the migration audit as folded into it (comments at lines 57 to 58, 170 and 1111; jobs `architecture-checks` at line 200 and `legacy-migration-audit` at line 1118), so the clause holds if it means the surface includes those three and fails if it means the surface is only those three; which is meant is undecided.
- **Next step:** the syntax clause closes with `KW-CI-052`, whose acceptance asks that `actionlint` parse every workflow (ryancinsight/atlas holds a shared `workflow-lint.yml`); trace each of the eight files beside `ci.yml` to the change that added it and settle the reading of the surface clause against those records; then close.
