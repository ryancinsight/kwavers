<a id="kw-cbr-facade-consistency"></a>

## KW-CBR-FACADE-CONSISTENCY — Facade CBR integration test contradicts the corrected rank-selection contract [patch] [fix] — todo

priority: correctness; needs: none; scope: `crates/kwavers/tests/clutter_filter_integration/separation_methods.rs`, `crates/kwavers-analysis/src/signal_processing/clutter_filter/adaptive_filter/filter.rs`

- **Finding.** #1020 corrected CBR rank selection to the smallest rank with `CBR(rank) >= target` (previously rank 1 whenever `CBR(1)` sat below target). The facade test still asserts the old contract: rank-1 output equality within `N^2 eps` and a `CBR(1)`-valued history. On current code the selector walks into the tail (measured pixel 0: recorded CBR 873 against target 316.2, output matching neither rank 1 nor rank 2), so `test_all_separation_methods` fails deterministically at `separation_methods.rs:253` under both default and `full` features. PR CI never runs this suite (push-only by design) and the main-push runs that should have caught it short-circuit at earlier red gates, so the break is invisible there; the pre-push hook runs it and refuses every manifest-touching push.
- **Acceptance:** the facade CBR section asserts the corrected contract — recorded CBR at or above target and finite, selected rank at least 2 (derived: `CBR(1) <= ~1.3 < 316` on this spectrum), and filtered output bit-identical to the `FixedRank` run at the recorded rank — via a `rank_history` accessor mirroring `cbr_history`; the pre-push hook's full test stage passes.
- **Next step:** add `rank_history`, rewrite the CBR assertions and derivation, verify the focused suite plus the analysis module tests overlay-free, push through the hook.
