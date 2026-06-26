//! High-speed signal integrity audit tests.
//!
//! Split by sub-domain:
//! - `emi`            — EMI hotspots, proximity, parallel spacing, adjacent-layer parallelism
//! - `reference_plane`— reference-plane intrusion, split-domain crossing, fragmentation
//! - `stitching`      — stitching caps, transition ground vias, via-in-pad, terminal return

pub(super) mod emi;
pub(super) mod reference_plane;
pub(super) mod stitching;
