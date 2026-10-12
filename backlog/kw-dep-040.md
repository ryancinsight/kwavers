<a id="kw-dep-040"></a>

## KW-DEP-040 — Migrate first-party floors past Apollo 0.28 / Consus 0.2 [minor] — todo

priority: tightening; needs: none; scope: `Cargo.toml`, `Cargo.lock`

- **Outcome:** fresh (non-`--locked`) resolution succeeds against current stack mains again.
- **Acceptance:** `lockfile.py --check` green; `cargo update` over the first-party set completes; focused suites stay green; the committed lock keeps standalone form (git sources intact, no `[[patch.unused]]`).
- **State:** discovered during ATLAS-KWAVERS-HEPHAESTUS-FDTD-107 closure. Kwavers requires `apollo-fft ^0.27.0` while Apollo main serves 0.28.0, and `consus-core ^0.1.0` while Consus main serves only 0.2.0. Locked builds work (pins intact); only fresh resolution and `cargo update` break.
- **Prerequisite edge (cross-repo):** Coeus workspace requires `apollo-fft ^0.27.0` (`coeus/Cargo.toml`), so a Kwavers-only bump to ^0.28 fails standalone resolution via `coeus-autograd`. Land the Coeus `^0.28` bump first, then advance the Kwavers pins (apollo-fft 0.28.0 plus the apollo-repo subtree rev) by hand in standalone form — full `--regenerate` is blocked on the Consus 0.2 migration, whose blast radius is unknown; assess it before folding it in.
- **Related:** KW-CI-049 (lock-alignment intent; close together once its confirmation lands).
