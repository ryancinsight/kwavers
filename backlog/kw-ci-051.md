<a id="kw-ci-051"></a>

## KW-CI-051 — Remove obsolete deployment workflow [patch] — todo

priority: verification; needs: none; basis: a3a72689be0fbfed3f64231c52186eac79dedc15; scope: `.github/workflows/deploy.yml` (deleted)

- **Outcome:** no workflow refers to deployment artifacts the repository does not hold or carries invalid step syntax, and the CI surface is the one the acceptance names.
- **Acceptance:** no workflow references absent deployment artifacts or invalid step syntax; the supported CI surface stays architecture validation, migration audit and the provider-aware build/test workflow.
- **Witnessed.** `.github/workflows/` holds no `deploy.yml`, and `git grep -n -i -E "dockerfile|k8s|kubernetes|docker" -- .github` prints nothing.
- **Witnessed since, by #1054.** "invalid step syntax": job `workflow-lint` of `ci.yml` calls the shared Atlas guard, which by its script runs `actionlint` once for each `.yml` or `.yaml` file under `.github/workflows` (read; the job prints no list); in run 43136209989 (dispatched on the branch of #1054) its job 130912097293 concluded success.
- **Open clause.** "the supported CI surface stays architecture validation, migration audit and the provider-aware build/test workflow": `.github/workflows/` holds nine files (`adr-index.yml`, `benchmark-regression.yml`, `book-pages.yml`, `ci.yml`, `compile-timing.yml`, `gpu-parity.yml`, `python-release.yml`, `python-wheel-smoke.yml`, `rust-release.yml`) and no `architecture-validation.yml`. `ci.yml` records the architecture validation and the migration audit as folded into it (comments at lines 57 to 58, 170 and 1111; jobs `architecture-checks` at line 200 and `legacy-migration-audit` at line 1118), so the clause holds if it means the surface includes those three and fails if it means the surface is only those three; which is meant is undecided.
- **Next step:** trace each of the eight files beside `ci.yml` to the change that added it and settle the reading of the surface clause against those records; then close.
