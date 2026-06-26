//! The physics-guided cost seam.
//!
//! [`RoutingCost`] is the extension point that makes this router *physics-guided*: the
//! negotiated-congestion search multiplies a node's congestion penalty by its **base cost**, and
//! the base cost is where design intent lives. [`PhysicsCost`] folds high-voltage creepage,
//! layer affinity, high-speed edge clearance, and reference-plane quality into that base cost so
//! the constraints *shape* the route instead of rejecting it after the fact.
//!
//! # Proximity hazard model
//!
//! The `lv_field` and `hv_field` arrays are computed using a **sum-weighted proximity** kernel:
//! each nearby pad contributes a linear ramp decaying from 1 at the pad centre to 0 at the
//! creepage clearance distance. Summing over all pads in a domain accumulates hazard when a
//! cluster of power pads surrounds a candidate cell (e.g. the dozens of VCCINT balls on a dense
//! BGA) — even if no single pad is individually dominant. The sum is clamped to `[0, 1]` so the
//! cost multiplier `1 + creepage_weight × hazard` stays in the same range as the nearest-pad
//! model. Evidence tier: property/differential tests in `mod tests` and the FPGA-tile routing
//! example.
//!
//! # Module layout
//!
//! The cost seam is split across several files (Phase 2a carve-out from the previously-flat
//! `src/cost.rs`). All references below are plain markdown backticks rather than `[`X`](path)`
//! intra-doc links — the kernels + penalty constants are `pub(super)` (crate-internal), so
//! rustdoc's `private_intra_doc_links` lint rejects clickable links to them from the public
//! `crate::cost` doc surface:
//!
//! * `routing_cost` — the dependency-inversion trait `RoutingCost`. The router depends on this
//!   trait, never on a concrete cost.
//! * `physics` — the `PhysicsCost` struct + its `PhysicsCost::new` precomputed-field builder +
//!   the per-class layer-affinity helpers; owns the per-class penalty constants
//!   (`REFERENCE_PLANE_PENALTY`, `REFERENCE_MARGIN_PENALTY`, etc.).
//! * `geometry_modulated` — the geometry-derived penalty kernels (`proximity`,
//!   `adjacent_reference_margin`, `high_speed_track_proximity`,
//!   `high_speed_adjacent_layer_track_proximity`) that `PhysicsCost::new` aggregates into
//!   precomputed fields.
//! * `adapter` — the `impl RoutingCost for PhysicsCost` bridge. Reads precomputed fields and
//!   applies the per-class penalty weights at query time.
//! * `tests` (gated `#[cfg(test)]`) — the property/differential test surface.

pub mod adapter;
pub mod geometry_modulated;
pub mod physics;
pub mod routing_cost;

#[cfg(test)]
mod tests;

pub use physics::PhysicsCost;
pub use routing_cost::RoutingCost;
