<a id="kw-cbr-selection-direction"></a>

## KW-CBR-SELECTION-DIRECTION — CBR-based clutter rank selection compares in a direction its estimate cannot satisfy [patch] [fix] — todo

priority: correctness; needs: none; scope: `crates/kwavers-analysis/src/signal_processing/clutter_filter/adaptive_filter/filter.rs` (`select_clutter_rank`, `estimate_cbr`) and `.../adaptive_filter/tests.rs`

- **Finding (at origin/main a23f2f86cb0).** `estimate_cbr(eigenvalues, r)` is clutter power over residual power, so it increases with `r`: moving eigenvalues from the denominator to the numerator only grows it. `SubspaceSeparationMethod::CbrBased` returns the first `r` in `1..n` with `CBR(r) <= 10^(target_cbr_db/10)` (`filter.rs:235-243`), so it returns rank 1 whenever `CBR(1) <= target` and otherwise falls to the `(n/2).max(1)` fallback; the rank never tracks the target.
- **Mutant evidence.** Reading the dB target as an amplitude ratio (`/20` for `/10`) survives the kwavers#926 tests, so no test pins the dB-to-power convention or the comparison direction.
- **Outcome:** the intended direction is derived from the method's reference (the eigen-based clutter filter literature the type docs cite, Yu and Lovstakken 2010), recorded in the Rustdoc, and the selection is fixed to match; the dB convention is pinned by a test on eigenvalues with a known CBR curve.
- **Acceptance:** a value-semantic test on a spectrum with an analytic `CBR(r)` asserts the selected rank for at least three targets spanning the curve, not only the rank-1 and `n/2` outcomes; `cargo mutants` over `select_clutter_rank` reports the `/20` mutant and the comparison-flip mutant caught.
- **Next step:** read the cited reference for whether the target is a ceiling on residual clutter (select the smallest `r` whose residual-clutter ratio falls below it) and write the derivation into the Rustdoc.
