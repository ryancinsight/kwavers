<a id="kw-cbr-facade-test-residuals"></a>

## KW-CBR-FACADE-TEST-RESIDUALS — CBR facade test doc and rank-history test residuals from #1024 [patch] — todo

priority: verification; needs: none; basis: 2824e0ecf45fd62c381bf935fd032326171b1fdd; scope: `crates/kwavers/tests/clutter_filter_integration/separation_methods.rs` (doc comment of the CbrBased section), `crates/kwavers-analysis/src/signal_processing/clutter_filter/adaptive_filter/tests.rs` (`test_rank_history_records_applied_rank`)

- **Finding (judge of #1024, none gating).** (1) The doc comment states the residual is "at least 1.90"; the computed minimum is 1.898 (pixel 0), so the stated bound is false by 0.002 while the margin it supports holds. (2) The doc comment cites `filter.rs` by line number, which drifts when that file changes. (3) `test_rank_history_records_applied_rank` pins only `FixedRank { 3 }`: a recorder that always records 3 passes it (by inspection, not run), and only the facade test catches that.
- **Outcome:** the doc comment states a bound the computation satisfies and names the items it cites; the module test alone fails for any constant recorder.
- **Acceptance:** the bound in the doc comment is at or below 1.898 with its source; no `filter.rs:<line>` citation remains in the file; the module test asserts `rank_history()` for two distinct ranks, and a mutant recording the constant 3 fails it without the facade test.
- **Next step:** edit the doc comment, add a second rank to the module test, and run the constant-3 mutant against `-p kwavers-analysis --lib adaptive_filter`.
