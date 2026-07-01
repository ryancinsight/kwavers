//! High-voltage dielectric withstand — Paschen air-breakdown, IPC-2221 voltage spacing,
//! and CAF time-to-failure — `physics::dielectric` vertical slice (Phase 3c).
//!
//! # SSOT for the slice
//!
//! * [`paschen`] — `V_b(pd) = B·pd / ( ln(A·pd) − ln(ln(1 + 1/γ)) )` breakdown voltage across
//!   an air gap; air-constants `A=15 (cm·Torr)⁻¹`, `B=365 V/(cm·Torr)`, secondary-emission
//!   `γ≈0.01`. The Paschen minimum for air is ~327 V, below which air cannot break down at any
//!   gap — so a 150 V HV rail is creepage-limited (not air-breakdown-limited).
//! * [`ipc2221_spacing`] — IPC-2221B Table 6-1 B1 minimum conductor spacing (mm) between
//!   conductors; the 0.60 mm floor at 150 V external uncoated is the SSOT that the
//!   [`crate::rules::CreepageRule::holohv`] HV-clearance matches.
//! * [`caf`] — Relative conductive-anodic-filament (CAF) time-to-failure (Rudra/IPC-TR-476 form:
//!   `TTF ∝ spacing² / voltage`) so a contributor can compare a candidate drill margin against a
//!   reference design (>1 is safer).
//!
//! Cross-slice dependency: the `ipc2221_min_spacing_mm(150.0) == 0.60 mm` invariant is asserted
//! against `CreepageRule::holohv()` in `tests`, locking the routing creepage rule to the IPC
//! B1 external uncoated value at the 150 V rail. Creepage ([`crate::rules`]) governs *surface*
//! tracking; this slice covers the complementary *air* breakdown across a gap.

/// Relative conductive-anodic-filament (CAF) time-to-failure between two conductors at
/// `spacing_mm` and `voltage_v` versus a reference: `TTF/TTF_ref = (s/s_ref)² · (v_ref/v)`.
pub mod caf;
/// IPC-2221B Table 6-1 B1 external uncoated minimum conductor spacing (mm) for `voltage_v`
/// between conductors, below 3050 m. Piecewise in `voltage_v`; above 500 V it grows at
/// `0.005 mm/V`.
pub mod ipc2221_spacing;
/// Paschen air-breakdown kinetics (`V_b(pd)`, minimum voltage, "can air possibly break down"
/// predicate) plus the air-constant table (A, B, γ) and the closed-form `ln(ln(1+1/γ))` term.
pub mod paschen;

#[cfg(test)]
mod tests;

// NOTE: `//` line comment, NOT `//!` module doc — this rationale is internal to the slice
// authors (not the published API). Keeps a future contributor from "harmonising" the export
// pattern back to glob `pub use paschen::*;` and re-leaking `A_AIR`/`B_AIR`/`GAMMA` from the
// paschen sub-slice into the `crate::physics::dielectric` API surface.
//
// Explicit named re-exports (not glob `pub use paschen::*;` like the ampacity slice does)
// so the slice-private `A_AIR` / `B_AIR` / `GAMMA` constants in [`paschen`] do not leak into
// the `crate::physics::dielectric` API surface.
pub use caf::caf_ttf_relative;
pub use ipc2221_spacing::ipc2221_min_spacing_mm;
pub use paschen::{air_breakdown_possible, paschen_breakdown_v, paschen_min_air};
