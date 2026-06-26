//! Design-for-manufacturing post-processing of routed copper.
//!
//! The router emits one track segment per grid edge — a straight ten-cell run becomes ten abutting
//! segments meeting at nine interior vertices. Each vertex is a fabrication concern that carries no
//! electrical meaning. These passes consolidate the geometry **without moving copper**, so DRC and
//! clearance are provably unchanged while the photoplot, the vertex count, and the acid-trap surface
//! all shrink. Carved by role (Phase 4l): `tracks`, `vias`, `copper`, `diagonal`.

mod copper;
mod diagonal;
mod miter;
mod tracks;
mod vias;

#[cfg(test)]
mod tests;

pub use copper::{ground_pour, quietest_layer, widen_for_ampacity};
pub(crate) use diagonal::chamfer_diagonal_traps;
pub use diagonal::resolve_diagonal_via_clearance;
pub use miter::miter_right_angle_corners;
pub use tracks::{merge_collinear, trim_dangling_stubs};
pub(crate) use tracks::{pad_entry_stubs, remove_orphan_copper, split_track_body_junctions};
pub use vias::{dedup_vias, plane_distribute_net, teardrops};
