<a id="kw-ci-gate-passes-drafts"></a>

## KW-CI-GATE-PASSES-DRAFTS — The CI gate and Architecture gate pass a draft pull request whose jobs were all skipped [patch] [ci] — todo

priority: verification; needs: none; basis: 38eaf2f90eccd74d11ce3d0e836cbad7d4606ae0; scope: `.github/workflows/ci.yml` (`ci-gate`, `architecture-gate`)

- **Read at the basis.** Both aggregate jobs run `if: always()` and fail only when a needed job's result is `failure` or `cancelled` (`ci.yml:1415-1418`, `:1446-1449`); a `skipped` result passes. The heavy jobs skip a draft (`ci.yml:154`, `:173`, `:199`, ...).
- **Observed.** Run 43466655820 (event `pull_request`, created 2026-11-13T16:49:53Z) belongs to kwavers#1066, which was marked ready at 16:59:06Z. Of its 25 jobs 22 were skipped; `Changed paths`, `CI gate` and `Architecture gate` concluded `success`. A draft therefore reports both aggregate checks green with no test run.
- **Outcome:** both aggregates fail on a draft pull request, as the stack policy requires, and still pass a ready prose-only pull request whose code jobs skip by classification. `ready_for_review` already re-triggers the workflow (`ci.yml:21`).
- **Acceptance:** each gate step's script, extracted from the workflow and run under `bash --noprofile --norc -eo pipefail`, exits 1 for a draft input with every result `skipped`, exits 0 for a non-draft input with the same results, and still exits 1 on any `failure` or `cancelled`.
- **Next step:** pass `github.event.pull_request.draft` into both steps as an environment variable and fail on `true`.
