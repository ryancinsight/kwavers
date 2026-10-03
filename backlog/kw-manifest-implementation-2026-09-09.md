<a id="kw-manifest-implementation-2026-09-09"></a>

## KW-MANIFEST-IMPLEMENTATION-2026-09-09 — Reduce the implementation-bearing manifests [patch] [arch] — in-progress

priority: tightening; needs: none; scope: every `lib.rs`/`mod.rs` carrying more than twenty body lines under `crates/`

- **The class.** `lib.rs` and `mod.rs` are module manifests: the tree, curated re-exports, and crate or module docs. The fleet scan counts a manifest with more than twenty body lines; this repository held 290, the largest at 463 body lines.
- **Delivered:** `inverse/fwi/time_domain/self_adjoint/mod.rs` split into `types`, `operators`, `forward` and `gradient` (manifest 39 lines, zero body); the MOFI aligner `mod.rs` followed. About 280 remain.
- **Where, measured 2026-09-09:** kwavers-solver 92, kwavers-analysis 49, kwavers-physics 31, kwavers-diagnostics 16, kwavers-therapy 15, kwavers-gpu 11, kwavers-boundary 11, kwavers-transducer 10, kwavers-medium 9, kwavers-python 6, the rest in single digits. Largest next: `fwi/time_domain/mofi` (428), `analytical/transducer` (342), `phantom/scatterers` (297), `theranostic_guidance/.../forward` (267).
- **Method:** split at the seams the code already has, not at line counts; compute each section's start by walking back over its doc comments and attributes (a boundary one line late orphans a `///`). Cross-module items take `pub(super)`, never `pub(crate)`. `mofi` interleaves types with functions, so contiguous spans do not separate them.
- **Do not run `cargo fix` from inside the stack overlay:** it rewrote 365 lines of this repository's `Cargo.lock`. Narrow imports by reading what clippy reports.
- **Acceptance:** each increment leaves its manifest at zero body lines with the crate's public surface unchanged, the package's tests passing, and clippy `--locked --all-targets -D warnings` clean.
