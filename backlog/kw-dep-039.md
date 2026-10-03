<a id="kw-dep-039"></a>

## KW-DEP-039 — Make Gaia an Atlas-local dependency [patch] — review

priority: architecture; needs: none; scope: root `Cargo.toml` Gaia dependency; a downstream consumer's SemVer integration

- **Driver:** Cargo ignores Kwavers' root `[patch]` tables when a private downstream consumer's SemVer checker packages it, so that consumer's transitive Gaia Git source resolved a historical revision lacking the Eunomia dependency.
- **Acceptance as written:** Kwavers declares the live Atlas Gaia checkout directly, deletes the redundant Gaia source patch, and the consumer's historical SemVer comparison resolves through the local Gaia-to-Eunomia graph.
- **Re-validation finding (2026-10-03):** the root `Cargo.toml` declares `gaia = { package = "gaia-mesh", version = "0.5.0", git = ... }`, a git+version source; a local-path declaration on a member mainline is quarantine the pin-discipline rule forbids, so the acceptance's local-checkout clause is obsolete. The residual the original evidence named (a Moirai-to-Themis Git edge, `themis ^0.10` against 0.9.17) belongs to Moirai portability, not Kwavers.
- **Next step:** close on confirmation that the consumer's SemVer comparison resolves through the git source.
