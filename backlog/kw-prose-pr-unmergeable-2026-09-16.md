<a id="kw-prose-pr-unmergeable-2026-09-16"></a>

## KW-PROSE-PR-UNMERGEABLE-2026-09-16 — A prose-only pull request can never satisfy the required checks [patch] [ci] — blocked

priority: verification; needs: none; scope: `.github/workflows/ci.yml`; `main` branch protection required contexts

- **Finding.** `main` required nine status checks, all from workflows whose `paths-ignore` covered `backlog.md`, `CHANGELOG.md`, `README.md`, `gap_audit.md` and `docs/**`. A filtered workflow reports no check at all, so a prose-only pull request stayed `BLOCKED` with zero checks.
- **Delivered:** the path filter moved into a `changes` job reading the pull request's file list through the API (#790), and always-running `CI gate` and `Architecture gate` aggregate jobs became required contexts (#791). A skipped required job counts as success, so a prose-only pull request now reports every required context.
- **Blocker:** `Build & Test (stable)` must leave the required list, a matrix job skipped before expansion never reports under that name. The agent session's classifier refused the call (`CI Bypass`); it needs a human or an explicit permission rule.
- **Re-open trigger:** the user runs `gh api repos/ryancinsight/kwavers/branches/main/protection/required_status_checks/contexts --method DELETE -f "contexts[]=Build & Test (stable)"` or grants that permission. Coverage does not drop: `CI gate` depends on `build`.
- **Acceptance:** a prose-only pull request reaches mergeable on its own, without an administrative merge.
