<a id="kw-ci-051"></a>

## KW-CI-051 — Remove obsolete deployment workflow [patch] — review

priority: verification; needs: none; scope: `.github/workflows/deploy.yml` (deleted)

- **Acceptance:** no workflow references absent deployment artifacts or invalid step syntax; the supported CI surface stays architecture validation, migration audit and the provider-aware build/test workflow.
- **Re-validation finding (2026-10-03):** `deploy.yml` was deleted by 91d784ca001, and no workflow at origin/main mentions `Dockerfile`, `k8s` or `docker`.
- **Next step:** close on confirmation.
