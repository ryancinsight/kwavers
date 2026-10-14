<a id="kw-bubble-conduction-sign"></a>
## KW-BUBBLE-CONDUCTION-SIGN — restore the composed first-law sign [patch] — review
- Reproduced on main `99a6185d`: a stationary 325.15 K bubble heats to 327.15 K instead of cooling to 323.15 K in the analytic tau/16 step.
- Implemented in physics 3.0.1: subtract outward-positive conductive power in the shared balance; five regressions cover both callers, hot/cold/equilibrium, pressure work and disabling.
- Native physics suite: 1,584 pass, zero skipped; package-only strict all-target/all-feature Clippy passes. Unchanged core native lint diagnostics and hosted gates remain separate checks.
- Merge held: main triggers package-publication attempts and book deployment; release/deployment decisions are pending.
- Validation repair: reuse same-run locked, checksum-verified crate sources; preserve all compile bounds and selected tests/benchmarks.
- Miri selection/error repair: current simulation configuration tests and the complete facade harness; no ignored errors or successful empty harnesses. Hosted final validation remains pending.
- Bounded compilation repair prepares real native TLS and feature-matched foundation artifacts; Miri inventories and independently gates the complete facade inventory without omissions.
- [patch] Facade/solver 3.0.1 retain checked numerical borrows; analytic Gaussian/PAM layout coverage and all 58 focused FD regressions pass. A channel sweep preserves per-time addition order; full-size Miri timing remains under investigation.
- Corrected manylinux prerequisite routing: the pinned Maturin action selects Docker only for build/publish. Preparations now use build help after fail-fast real compilation; installed-wheel contracts remain mandatory.
- Complete benchmark compilation and execution are independent mandatory bounded jobs. No selected program is omitted.
