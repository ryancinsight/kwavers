<a id="kw-bubble-conduction-sign"></a>
## KW-BUBBLE-CONDUCTION-SIGN — restore the composed first-law sign [patch] — review
- Reproduced on main `99a6185d`: a stationary 325.15 K bubble heats to 327.15 K instead of cooling to 323.15 K in the analytic tau/16 step.
- Implemented in physics 3.0.1: subtract outward-positive conductive power in the shared balance; five regressions cover both callers, hot/cold/equilibrium, pressure work and disabling.
- Native physics suite: 1,584 pass, zero skipped; package-only strict all-target/all-feature Clippy passes. Unchanged core native lint diagnostics and hosted gates remain separate checks.
- Merge held: main triggers package-publication attempts and book deployment; release/deployment decisions are pending.
