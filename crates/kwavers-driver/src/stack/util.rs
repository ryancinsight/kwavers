//! Slice-private helpers shared by the [`super::manifest`] and [`super::compatibility`] sub-files:
//! board-extent geometry, stack-net canonicalisation, and the tolerance comparison used by the
//! connector-mating check. Kept `pub(super)` so they stay off the published `crate::stack` surface.

use crate::board::Board;
use crate::geom::Nm;

/// Canonicalise a net name for inter-board stack-connector comparison (e.g. a controller's
/// `VCCO_3V3` rail and a driver's `P3V3` rail are the same stack net).
pub(super) fn canonical_stack_net(name: &str) -> &str {
    match name {
        "VCCO_3V3" => "P3V3",
        other => other,
    }
}

/// Board copper-extent width in millimetres, from the routing grid pitch and column count.
pub(super) fn board_width_mm(board: &Board) -> f64 {
    Nm(board.spec.pitch.0 * (board.spec.nx as i64 - 1)).to_mm()
}

/// Board copper-extent height in millimetres, from the routing grid pitch and row count.
pub(super) fn board_height_mm(board: &Board) -> f64 {
    Nm(board.spec.pitch.0 * (board.spec.ny as i64 - 1)).to_mm()
}

/// Push a labelled mismatch when two millimetre values differ by more than the 1 µm mating tolerance.
pub(super) fn check_close(label: &str, a: f64, b: f64, mismatches: &mut Vec<String>) {
    if (a - b).abs() > 1.0e-6 {
        mismatches.push(format!("{label} differs: {a:.6} != {b:.6}"));
    }
}
