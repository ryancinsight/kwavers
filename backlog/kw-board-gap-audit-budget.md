<a id="kw-board-gap-audit-budget"></a>

## KW-BOARD-GAP-AUDIT-BUDGET — gap_audit.md holds 4,391 lines against a 1,000-line board budget [patch] — todo

priority: verification; needs: none; basis: 34fa3f47ae07e28d433b10a22c746f2c982e042e; scope: `gap_audit.md`; milestone: M-solvers-core

- **Outcome:** `gap_audit.md` holds only risks nobody is fixing and is at most 1,000 lines; every removed entry is closed (cited by its merged PR or commit), delivery narrative, or moved to a DoR backlog item.
- **Finding at basis:** `wc -l < gap_audit.md` is 4391; `## Review residuals (2026-07)` is lines 84-4314 (4231 lines); `## KW-PHYS-005 [minor]` (:4368-4388) is item-shaped and links its completed record `docs/verification/kw-phys-005.md`; `## Driver 0.5.1 external qualification` (:4389) links `docs/verification/driver-0.5.md`; `:26` cites `KW-SOL-072`, an id no board file holds. The `## Live Aequitas closure` entry (:50-83) belongs to `KW-VERIF-RESCUED-REMAINDER`. `atlas-artifact-budget.py --base origin/main --rev origin/main check` prints `WARN gap_audit.md: 4391 lines (3391 over budget)`; `grep -c "^- " gap_audit.md` is 305 over the whole file.
- **Acceptance:** (a) `wc -l < gap_audit.md` prints at most 1000 and `atlas-artifact-budget.py --base origin/main --rev HEAD check` prints no `gap_audit.md` WARN line (basis: both fail); (b) every heading or bullet the diff removes is one row of the delivering pull requests' tables with its proof, a merged PR or commit shown by `git merge-base --is-ancestor <sha> origin/main` for a closed risk, or the id of a `backlog.md` item for an open one, and `git diff -U0 origin/main..HEAD -- gap_audit.md | grep -c -E "^-(#|- )"` equals the row count; (c) `git grep -n "KW-SOL-072" -- gap_audit.md` prints nothing (basis: `:26`).
- **Wrong fixes against it:** truncating to 1000 lines or deleting open risks passes (a) and fails (b); the removal of a risk with no proof row fails (b).
- **Next step:** classify the bullet lines and paragraphs of `## Review residuals (2026-07)` (lines 84-4314) as closed, narrative or open by reading each against the default tip; deliver in slices of at most 400 changed lines, each with its own table.
