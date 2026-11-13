<a id="kw-hook-nextest-config-ungated"></a>

## KW-HOOK-NEXTEST-CONFIG-UNGATED — A push changing only `.config/nextest.toml` passes the pre-push hook with no package gated [patch] [ci] — blocked

priority: verification; needs: none; basis: 38eaf2f90eccd74d11ce3d0e836cbad7d4606ae0; scope: `.githooks/pre-push` (deployed copy; cure edits the Atlas owner `scripts/git-hooks/pre-push`)

- **Finding.** `.githooks/pre-push:1463-1481` maps only `Cargo.toml` and `Cargo.lock` (`:1464`) to every workspace member; any other path outside a package directory is dropped (`:1479`), and `:1601-1603` then returns 0 with "no package owns the changed files in $workspace_manifest; local gate not needed". A `.config/nextest.toml` change sets every test's slot, group and termination time, yet gates no test.
- **Observed.** kwavers#1066 records it for its `.config/nextest.toml`-only push (body line 27, "Pre-push gating").
- **Owner.** Line 3 declares a "single owned copy" that `scripts/atlas-lock-form.py sync-hooks` deploys; Atlas `origin/main` 12708acf3 holds the same mapping (`scripts/git-hooks/pre-push:1463`, `:1602`). A member copy edited here is drift.
- **Unowned.** At 12708acf3, `git grep -i -e 'nextest\.toml' -e global_inputs -e 'no package owns' -e resolved-changes -- backlog.md backlog gap_audit.md` names no hook item (its three hits concern other findings), and no open Atlas pull request (#587, #683) names one.
- **Blocker:** the cure belongs to the Atlas hook. **Re-open trigger:** an Atlas board item holds the mapping, or the synced copy here changes `:1463-1481`.
- **Acceptance:** the mapping logic of the synced copy, run on a range changing only `.config/nextest.toml`, selects the packages whose tests the config schedules (every member, or those the owner item derives) and prints no "no package owns" line.
- **Next step:** file the owning item on the Atlas board naming this ID, then cite it here.
