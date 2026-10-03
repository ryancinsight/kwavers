<a id="kw-mat-043"></a>

## KW-MAT-043 — Direct FWI L-BFGS provider ownership [patch] [arch] — in-progress 2026-08-11

priority: architecture; needs: none; scope: the two FWI production callers, the focused FWI L-BFGS tests, `crates/kwavers-math/src/optimization/`

- **Outcome:** use `leto_ops::application::optimization::LbfgsMemory` directly in every in-scope FWI caller; the consumer owns no adapter, fallback or duplicate memory implementation.
- **Baseline:** `crates/kwavers-math/src/optimization/lbfgs.rs` is already absent (deleted by the math SSOT migration) and must not be recreated. The public `kwavers-math` re-export stays outside this FWI caller cutover.
- **Acceptance:** direct provider imports replace all FWI-local imports; focused tests prove value-equivalent two-loop directions after eviction and a hard memory bound; affected package formatting, Nextest, doctests, strict Clippy and the consumer check pass against the delivered source, or the exact infrastructure blocker is recorded.
