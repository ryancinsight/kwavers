//! Tests for the `dfm` slice (Phase 4l carve-out). Moved verbatim from the flat `src/dfm.rs`
//! `mod tests` block; `super::*` resolves the slice facade.

use super::*;
use crate::board::{Board, LayerId, NetId, Track};
use crate::geom::{GridSpec, Nm, Point};

mod geometry;
mod power_plane;
mod track_cleanup;

pub(super) fn board() -> Board {
    let spec = GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 2).unwrap();
    Board::new(spec)
}

pub(super) fn seg(b: &mut Board, net: NetId, x0: f64, y0: f64, x1: f64, y1: f64) {
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(x0), Nm::from_mm(y0)),
        end: Point::new(Nm::from_mm(x1), Nm::from_mm(y1)),
        width: Nm::from_mm(0.25),
        layer: LayerId(0),
        net,
    });
}

/// Total copper length on a layer — the invariant `merge_collinear` must preserve exactly.
pub(super) fn total_len(b: &Board) -> f64 {
    b.tracks.iter().map(|t| t.start.euclid(t.end)).sum()
}
