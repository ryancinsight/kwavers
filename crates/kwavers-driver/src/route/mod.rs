//! Negotiated-congestion routing: grid resource model, per-net search, and the PathFinder loop.
//!
//! # Module layout
//!
//! The route slice is split across several sibling files per the spec's
//! `route/{mod.rs, grid.rs, search.rs, pathfinder.rs, tree.rs, emission.rs, tests.rs}` layout
//! — fully migrated at Phase 2b (round-1 carved `tests.rs`, round-2 carved `tree.rs` +
//! `emission.rs`):
//!
//! * `grid` — the 3D grid resource model (per-node capacity, occupancy, history).
//! * `search` — the A*-style per-net grow-loop (state machine + cost-driven expansion).
//! * `search_guards` — diagonal-routing geometry guards (foreign-edge crossing, via-column
//!   clearance) extracted from `search` as a dedicated leaf module (SoC, ≤500 lines).
//! * `pathfinder` — the iterative PathFinder *negotiation* loop (rip-up → re-route →
//!   history accumulate → break on convergence). Hosts `Router`, `RouteOutcome`, plus
//!   the per-net `forbidden` / `via_forbidden` set construction + the schedule-driven
//!   outer iteration. The single-net `route_one` method and the `apply_to_board` emission
//!   are *not* here — they live in `tree.rs` and `emission.rs` respectively as cross-file
//!   `impl Router` blocks (Rust resolves the methods through the type regardless of which
//!   file the impl was written in).
//! * `tree` — the Prim-style (power/ground) + chain-tip (signal/HV) `route_one` method carrier.
//! * `emission` — the `apply_to_board` track/via emission + the `via_nodes` / `via_shadow_nodes`
//!   helpers used by `pathfinder` for rip-up + claim accounting during the negotiation loop.
//! * `tests` (gated `#[cfg(test)]`) — the 9 PathFinder property/empirical tests, moved out of
//!   the previously-inline `mod.rs::tests` block at Phase 2b round-1.
//!
//! # Phase 2b done (no further forward-tracking items)
//!
//! All sub-`route/*` carve-outs per spec have landed in Phase 2b (round-1 carved `tests.rs`;
//! round-2 carved `tree.rs` + `emission.rs`). Forward-looking cosmetic items that surface
//! from these carves would queue in the `## Phase 2b follow-ups — route sub-slice migration
//! follow-ups` placeholder section of `docs/MIGRATION.md` (mirrors the Phase 1d-follow-ups
//! + Phase 1e pattern).

pub mod emission;
pub mod grid;
pub mod pathfinder;
pub mod search;
pub(crate) mod search_guards;
pub mod tree;

pub use grid::{Grid, NodeId};
pub use pathfinder::{NetRoute, NetTerminals, PadObstacle, PathFinderParams, RouteOutcome, Router};

#[cfg(test)]
mod tests;
