<a id="kw-hook-nextest-config-ungated"></a>

## KW-HOOK-NEXTEST-CONFIG-UNGATED — A push changing only `.config/nextest.toml` passes the pre-push hook with no package gated [patch] [ci] — todo

priority: verification; needs: ATLAS-HOOK-NEXTEST-CONFIG-OWNER, ATLAS-HOOK-FLEET-BLOB-DRIFT; basis: 38eaf2f90eccd74d11ce3d0e836cbad7d4606ae0; scope: `.githooks/pre-push` (deployed copy; the cure edits the Atlas owner `scripts/git-hooks/pre-push`)

- **Finding.** `.githooks/pre-push:1463-1481` maps `Cargo.toml` or `Cargo.lock` (`:1464`) to every workspace member and drops any other path outside a package directory (`:1479`); `:1601-1603` then returns 0 with "no package owns the changed files in $workspace_manifest; local gate not needed". A `.config/nextest.toml` change sets every test's slot, group and termination time, yet gates no test. kwavers#1066 records it (body line 27, "Pre-push gating").
- **Atlas main.** Line 3 declares a "single owned copy" that `scripts/atlas-lock-form.py sync-hooks` deploys. Atlas `origin/main` f19af27ce maps at `scripts/git-hooks/pre-push:1463-1465` a root `Cargo.toml` only, under a `[workspace]` guard; `Cargo.lock` and `.config/nextest.toml` are unmapped, and `:1602` prints the same message.
- **Drift.** The kwavers copy is byte-identical to `scripts/git-hooks/pre-push` at `refs/pull/647/head` of Atlas #647 (closed unmerged, released), and differs from Atlas main's file at `:1463-1465` and in the cast-base code. It is not Atlas main's mapping.
- **Owners, filed in open draft Atlas #693 (branch `docs/atlas-board-compact-filing-20261113`).** `ATLAS-HOOK-NEXTEST-CONFIG-OWNER` covers the gap: a push changing only `.config/nextest.toml` selects the packages its `package(...)` filters name (all when a profile-level key changes). `ATLAS-HOOK-FLEET-BLOB-DRIFT` covers the copy: member copies equal Atlas main's blob, and a hook fix reaches no member before that. Neither item is merged; #693 is a draft.
- **Not covered by either.** That the kwavers copy carries the cure and a `.config/nextest.toml`-only range gates packages; this item is that check.
- **Ready when** both needs have merged on Atlas `origin/main` and the sync has deployed them here.
- **Acceptance:** `git diff` between `.githooks/pre-push` and Atlas `origin/main:scripts/git-hooks/pre-push` is empty, and the copy's mapping on a range changing only `.config/nextest.toml` selects the packages the config schedules and prints no "no package owns" line (command and revisions recorded).
- **Next step:** after both needs merge, run that diff and range check against the deployed copy, and close the item with the run.
