<a id="kw-board-offset-statuses"></a>

## KW-BOARD-OFFSET-STATUSES — 28 board entries carry a status the board does not hold [patch] — todo

priority: verification; needs: none; basis: db4c667d9d3bcc1d7502b3799b55c52eb3256d29; scope: `backlog.md`, `backlog/*.md`

- **Finding.** The index holds 19 entries marked `review`, 7 marked `in-progress` and 2 marked `done`; the board states are `todo` and `blocked` only, and claimed, in-review and done work is shown by pull requests. `KWAVERS-AEQ-MET-FLEX` is `done` in the index and `blocked` in its own heading.
- **Evidence.** A read-only audit by a subagent at a023ab77522cd6e03226f90602ffdcec343b8982 compared each item with `main` and with pipeline run 42970466152. Its table is in the description of the pull request that added this file. Of the 28 it classed one delivered with a caveat (`KW-CI-051`: no workflow-lint job witnesses the syntax clause), one not checkable without a private consumer (`KW-DEP-039`) and 26 partly delivered. Those classifications are leads, not verified facts: of the audit verdicts only the closure of `KW-DEP-ZIP-ADVISORY`, made in the same pull request as this item, was re-read at its sources.
- **Outcome:** every index line and heading reads `todo` or `blocked`, and each item that stays states the acceptance clause still open.
- **Acceptance:** `backlog.md` has no status outside the two; for each of the 28, either the item is deleted in a pull request that witnesses every acceptance clause, or its text names the open clause with a witness (file and line, or a run and job) and its status is `todo`, or `blocked` with a blocker and a re-open trigger; index and heading statuses agree for every item.
- **Next step:** take the eight items the audit parts classed delivered and its coordinator downgraded (`KW-APERTURE-001`, `-002`, `-003`, `KW-BIO-043`, `KW-MEDIUM-CT-001`, `KW-DOP-045`, `KW-FFT-050`, `KW-CI-050`): settle the clause the audit table names for each, then close or rewrite the item.
