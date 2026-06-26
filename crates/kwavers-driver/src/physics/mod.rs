//! Physics vertical-slice tree (Phase 0 placeholder).
//!
//! # Phase 1+ plan
//!
//! Phase 1 will migrate the flat physics modules (`src/ampacity.rs`, `src/dielectric.rs`,
//! `src/thermal.rs`, `src/emi.rs`, `src/pdn.rs`, `src/si.rs`, `src/acoustic.rs`) into the
//! sub-submodules under this tree:
//!
//! ```text
//! src/physics/
//! ├── mod.rs
//! ├── ampacity/    ← was src/ampacity.rs
//! ├── dielectric/  ← was src/dielectric.rs
//! ├── thermal/     ← was src/thermal.rs
//! ├── emi/         ← was src/emi.rs
//! ├── pdn/         ← was src/pdn.rs
//! ├── si/          ← was src/si.rs
//! └── acoustic/    ← was src/acoustic.rs (delegates to kwavers-transducer)
//! ```
//!
//! Each physics subtree will own its own types + impls + tests, with zero cross-physics
//! coupling (DIP: each leaf depends only on `crate::geometry::newtype::*`,
//! `crate::board::*`, and Twiggy primitives).
//!
//! # SSOT for the slice
//!
//! * `pub mod ampacity` — IPC-2221 width + Black electromigration + skin depth
//! * `pub mod dielectric` — Paschen + IPC-2221 voltage + CAF TTF
//! * `pub mod thermal` — 2-D heat-conduction + electro-thermal + thermal vias
//! * `pub mod emi` — commutation-loop inductance + switching/gate/recovery loss
//! * `pub mod pdn` — IR drop + target impedance + hold-up C
//! * `pub mod si` — microstrip impedance + propagation delay + skew
//! * `pub mod acoustic` — wavelength + grating-lobe + BVD resonance + f-number
//!
//! # Cut-over status
//!
//! Phase 1 has begun: this tree is no longer a placeholder. Four slices have migrated into
//! the sub-module layout:
//!
//! * `pub mod ampacity` (Phase 3a) — owns the IPC-2221 width + Black electromigration + skin
//!   depth kernel; the flat `src/ampacity.rs` was retired.
//! * `pub mod thermal` (Phase 3b) — owns 2-D heat-conduction + electro-thermal coupling +
//!   thermal vias. IR-drop (`IrDrop` + `ir_drop`) was promoted out of `src/pdn.rs` into this
//!   slice so the electro-thermal chain (`ir_drop` → `joule_source` → `solve_electrothermal`)
//!   sits in one crate plane.
//! * `pub mod dielectric` (Phase 3c) — owns Paschen air-breakdown (`paschen_breakdown_v`,
//!   `paschen_min_air`, `air_breakdown_possible`), IPC-2221B Table 6-1 B1 external uncoated
//!   conductor spacing (`ipc2221_min_spacing_mm`), and Rudra/IPC-TR-476 relative CAF
//!   time-to-failure (`caf_ttf_relative`); the flat `src/dielectric.rs` was retired.
//!   Re-exported at the crate root under `pub use physics::dielectric::{...}` so the
//!   `crate::dielectric::ipc2221_min_spacing_mm`-style API surface is preserved verbatim.
//! * `pub mod emi` (Phase 3d; MIGRATION.md formal numbering) — owns the commutation-loop
//!   inductance / EMI kernel split into 8 sub-modules: `scene.rs` holds [`CommutationLoop`](crate::physics::emi::CommutationLoop)
//!   + the `commutation_loops` scene walker + private `pad_on_net` helper;
//!   `losses.rs` / `overshoot.rs` / `radiated.rs` / `trace_partial.rs` hold the seven
//!   kernel fns; `loop.rs` (`pub mod r#loop;` in the slice facade — `loop` is a Rust
//!   keyword, raw-identifier escape on the mod decl, filename remains `loop.rs`) holds the
//!   slice-private `polygon_area_mm2` (shoelace helper, `pub(super)`) and the public
//!   `loop_inductance_nh` (μ₀·√area). The flat `src/emi.rs` was retired; 8 lifted tests
//!   consolidated into `tests.rs`. Re-exported at the crate root under
//!   `pub use physics::emi::{...}` so the `crate::loop_inductance_nh`-style API surface
//!   is preserved verbatim.
//!
//! Phase 3e `pdn` is DONE (5 files: mod.rs + target_impedance.rs + impedance.rs + cavity.rs
//! + tests.rs; 7 fns carved across three per-concern submodules — flat `src/pdn.rs` retired,
//! callers re-routed to `crate::physics::pdn::*`). Phase 3f `si` is also DONE (5 files:
//! mod.rs + impedance.rs + propagation.rs + crosstalk.rs + tests.rs; 8 existing fns carried
//! across + 3 new APIs — `impedance_target` (signal-line branching-match target),
//! `return_loss_db` (single-call RL for caller-loop iteration over freq bands), and
//! `channel_operating_margin_db` (IEEE amplitude-ratio COM) — added to fill out the
//! frequency-band-aware impedance-budget surface; flat `src/si.rs` retired, crate-root
//! re-export at `crate::physics::si::{...}`). Phase 3g `acoustic` is also DONE — 8 files (mod.rs
//! + wavelength.rs + grating.rs + focus.rs + element.rs + safety.rs + nonlinear.rs + tests.rs);
//! 18 prior pub fns carried across + 3 NEW APIs (`bvd_anti_resonance_hz` for the parallel-
//! branch BVD equivalent-circuit resonance, `isppa_w_per_m2` for FDA Track-3 spatial-peak
//! pulse-average intensity, `round_trip_attenuation_db` for the pulse-echo two-way loss). Flat
//! `src/acoustic.rs` retired; crate-root re-export at `crate::physics::acoustic::{...}`.
//! All six sub-slice physics carves (Phase 3a–3g) now sit in the carved tree — Phase 3 is COMPLETE.

pub mod acoustic;
pub mod ampacity;
pub mod dielectric;
pub mod emi;
pub mod pdn;
pub mod si;
pub mod thermal;
