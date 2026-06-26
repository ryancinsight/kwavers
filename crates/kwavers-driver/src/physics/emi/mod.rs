//! Commutation-loop inductance / EMI — `physics::emi` vertical slice (Phase 3d).
//!
//! # SSOT for the slice
//!
//! * [`scene`] — [`CommutationLoop`] struct carrying `area_mm2` + `inductance_nh` + `at`, plus
//!   the scene-walker algorithm [`commutation_loops`] that walks every `Role::Decoupling` cap
//!   tied to a device and constructs its quadrilateral VPP/GND loop for hotspot feedback.
//!   Sole owner of the placement-aware `pad_on_net` helper (private to `scene.rs`).
//! * `r#loop` (raw-identifier escape for the Rust keyword `loop`; file name remains
//!   `loop.rs`) — `polygon_area_mm2` (slice-internal shoelace helper, callable by
//!   [`scene::commutation_loops`]) plus the first-order partial-inductance estimate
//!   [`loop_inductance_nh`] (`L ≈ μ₀·√area`, mm-scale ⇒ few-nH scale).
//! * [`trace_partial`] — [`trace_partial_inductance_nh`] (Grover/IPC, ~6–10 nH/cm) — the
//!   `L·dI/dt` overshoot source on the HV switching node.
//! * [`losses`] — [`switching_loss_w`] (`f·C·V²` class-D dominant), [`gate_drive_power_w`]
//!   (`Q_g·V_drive·f`), [`reverse_recovery_loss_w`] (`Q_rr·V·f` clamp/body loss).
//! * [`overshoot`] — [`capacitive_drive_current_a`] (`C·dV/dt` for the BVD transducer
//!   load drives the HV7355's ~1.5 A peak rating) and [`inductive_overshoot_v`]
//!   (`L·dI/dt`).
//! * [`radiated`] — [`radiated_emi_dbuv_m`] small-loop CISPR-22 first-order estimate.
//!
//! # Cross-slice dependency
//!
//! [`scene::commutation_loops`] reads [`Component`](crate::place::component::Component) + [`FootprintDef`](crate::place::footprint::FootprintDef) from
//! [`crate::place`] for the scene walk; that Tier-2 dependency has been in place since
//! `place` closed at Phase 2c and is preserved verbatim. The `pad_on_net` helper relies
//! on [`crate::place::component::Component::placed_pads`] which assumes the slice-internal
//! `Role::Decoupling` filter on the cap's footprint.

/// Vacuum permeability (H/m). `pub(super)` so the constant is addressable for
/// `crate::physics::*` siblings via the super-self path: **`crate::physics::MU0`** resolves
/// here for any sibling module under `crate::physics`, while the canonical path inside the
/// slice is `crate::physics::emi::MU0` (referenced by [`super::r#loop::loop_inductance_nh`]
/// and [`super::trace_partial::trace_partial_inductance_nh`]). NOT re-exported from the
/// slice facade because no part of the SSOT API surface treats µ₀ as a permitted knob
/// (it's a unit-of-nature constant).
/// [`trace_partial::trace_partial_inductance_nh`]). Revealed to siblings via [`super::MU0`]
/// inside `crate::physics::emi::*` (and `crate::physics::*`); NOT re-exported because no part
/// of the SSOT API surface treats µ₀ as a permitted knob (it's a unit-of-nature constant).
pub(super) const MU0: f64 = 1.256_637_062e-6;

// `loop` is a Rust reserved keyword (it's the start of the `loop {}` construct). The file
// on disk is therefore named `loop.rs` but the module declaration MUST use the raw
// identifier escape `r#loop` so the parser does not choke on the keyword. Imports and
// re-exports inside the slice follow the same convention (`super::r#loop::{...}`,
// `pub use r#loop::{...}`).
pub mod losses;
pub mod overshoot;
pub mod radiated;
pub mod r#loop;
pub mod scene;
pub mod trace_partial;

#[cfg(test)]
mod tests;

// Explicit named re-exports (NOT glob `pub use scene::*;` like the ampacity slice does).
// The slice exposes exactly 10 public items and we want to lock the API surface so a future
// contributor can't accidentally promote a `pub(super)` helper (e.g. `polygon_area_mm2`,
// `MU0`) to `pub` by "harmonising" the export pattern back to glob. This matches the
// 3c-dielectric discipline (named re-exports so internal `const`s stay slice-private).
pub use losses::{gate_drive_power_w, reverse_recovery_loss_w, switching_loss_w};
pub use overshoot::{capacitive_drive_current_a, inductive_overshoot_v};
pub use radiated::radiated_emi_dbuv_m;
pub use r#loop::loop_inductance_nh;
pub use scene::{commutation_loops, CommutationLoop};
pub use trace_partial::trace_partial_inductance_nh;
