# KW-TIME-TYPED-STEP — Type the `kwavers-core::time` boundary with Aequitas

- **Finding.** `crates/kwavers-core/src/time/mod.rs` carries the clock in raw
  `f64` seconds while `kwavers-gpu` already threads Aequitas quantities (#989).
  First slice (`takeover/kwavers-time-cfl-dedup`) centralizes the triplicated
  k-space CFL bound in `cfl_max_dt` with regression tests; this item tracks the
  typed-constructor remainder.
- **Oracle.** `dt [s] = dx [m] / (c [m/s])`, diagonal factor `sqrt(2)`, 0.9
  safety; `t_max = dt * (n - 1)`; adjustment preserves span.
- **Scope.** `time/mod.rs`, `units.rs`. Hot loop keeps `f64`; only the
  construction boundary becomes typed. **Needs:** KW-VALIDATE-CORE.
- **Acceptance.** Typed `from_grid_and_duration` successor; one shared CFL
  bound; `cargo test -p kwavers-core time::` green; solver sign-off.
- **Priority:** patch. **Milestone:** v1. **Basis:** `fe728d4ad`.
