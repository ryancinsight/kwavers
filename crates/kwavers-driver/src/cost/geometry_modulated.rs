//! Geometry-derived penalty kernels — the building blocks behind `PhysicsCost`'s
//! precomputed fields.
//!
//! (Plain backticks rather than `[`X`](path)` intra-doc links throughout this module: rustdoc's
//! `redundant_explicit_links` lint rejects clickable `[`X`](...X...)` forms when the resolution
//! is unambiguous from the link text alone — and the four kernels are `pub(super)` so
//! `private_intra_doc_links` rejects the form anyway.)
//!
//! These functions are pure geometry: each takes a `Board` (read-only),
//! a point, a layer, and the relevant `DesignRules`, and returns a
//! scalar penalty in `[0, 1]` (or `Option<f64>` for "no margin found").
//! `PhysicsCost::new` aggregates these kernels per-cell to
//! populate the precomputed penalty fields used by
//! `impl RoutingCost for PhysicsCost` at query time.
//!
//! # Functions
//!
//! (Plain backticks rather than `[`X`](path)` intra-doc links: the four kernels are
//! `pub(super)` and rustdoc's `private_intra_doc_links` lint rejects clickable links to them
//! from the public module docstring.)
//!
//! * `proximity` — sum-weighted HV/LV hazard kernel (the namesake of the proximity-hazard model
//!   documented at the parent `cost` module's docstring). The clamp-to-`[0, 1]` tail keeps
//!   `node_base` in the same `[1.0, 1.0 + weight]` range regardless of how many pads surround
//!   a cell.
//! * `adjacent_reference_margin` — minimum distance from a point to the boundary of any
//!   adjacent-layer reference zone (ground or power). Drives the high-speed reference-margin
//!   penalty in `PhysicsCost`.
//! * `high_speed_track_proximity` — proximity penalty vs existing high-speed copper on the
//!   same layer. Drives the same-layer spacing penalty.
//! * `high_speed_adjacent_layer_track_proximity` — proximity penalty vs existing high-speed
//!   copper on adjacent layers (broadside separation). Drives the broadside penalty that
//!   discourages parallel overlap across dielectric.
//!
//! All four functions are `pub(super)` — visible to the `PhysicsCost` field builder but not
//! exported as part of the crate's public API surface.

use crate::board::{Board, LayerId, NetClassKind};
use crate::geom::{dist_point_seg, distance_to_polygon_boundary, point_in_polygon, Point};
use crate::rules::DesignRules;

/// Minimum distance from a board point to the boundary of an adjacent-layer reference zone
/// (ground or power). Returns `None` if the point is not inside any adjacent-layer reference
/// zone.
///
/// Used by the `PhysicsCost` reference-margin penalty: a smaller margin triggers a larger cost.
pub(super) fn adjacent_reference_margin(board: &Board, p: Point, layer: usize) -> Option<f64> {
    board
        .zones
        .iter()
        .filter(|zone| {
            zone.layer.0.abs_diff(layer as u16) == 1
                && matches!(
                    board.class_of(zone.net),
                    NetClassKind::Ground | NetClassKind::Power
                )
                && point_in_polygon(p, &zone.polygon)
        })
        .filter_map(|zone| distance_to_polygon_boundary(p, &zone.polygon))
        .min_by(f64::total_cmp)
}

/// Same-layer high-speed copper proximity penalty in `[0, 1]`. Returns the worst (max) penalty
/// across all same-layer high-speed tracks within range.
pub(super) fn high_speed_track_proximity(
    p: Point,
    layer: LayerId,
    board: &Board,
    rules: &DesignRules,
) -> f32 {
    let mut worst: f64 = 0.0;
    for track in &board.tracks {
        if track.layer != layer
            || !matches!(
                board.class_of(track.net),
                NetClassKind::Hv | NetClassKind::Signal
            )
        {
            continue;
        }
        let preferred_gap =
            rules.high_speed_preferred_parallel_spacing_widths * track.width.0 as f64;
        if preferred_gap <= 0.0 {
            continue;
        }
        let preferred_centerline = preferred_gap + track.width.0 as f64 / 2.0;
        let centerline_distance = dist_point_seg(p, track.start, track.end);
        worst = worst.max(
            ((preferred_centerline - centerline_distance) / preferred_centerline).clamp(0.0, 1.0),
        );
    }
    worst as f32
}

/// Adjacent-layer high-speed copper proximity penalty in `[0, 1]`. Iterates tracks on the
/// `layer ± 1` planes (broadside separation); `.abs_diff(layer.0) != 1` skips non-adjacent
/// layers. Penalty value is `((preferred_centerline - centerline_distance) /
/// preferred_centerline).clamp(0, 1)`.
pub(super) fn high_speed_adjacent_layer_track_proximity(
    p: Point,
    layer: LayerId,
    board: &Board,
    rules: &DesignRules,
) -> f32 {
    let mut worst: f64 = 0.0;
    for track in &board.tracks {
        if track.layer.0.abs_diff(layer.0) != 1
            || !matches!(
                board.class_of(track.net),
                NetClassKind::Hv | NetClassKind::Signal
            )
        {
            continue;
        }
        let preferred_gap =
            rules.high_speed_preferred_parallel_spacing_widths * track.width.0 as f64;
        if preferred_gap <= 0.0 {
            continue;
        }
        let preferred_centerline = preferred_gap + track.width.0 as f64 / 2.0;
        let centerline_distance = dist_point_seg(p, track.start, track.end);
        worst = worst.max(
            ((preferred_centerline - centerline_distance) / preferred_centerline).clamp(0.0, 1.0),
        );
    }
    worst as f32
}

/// Sum-weighted proximity hazard: each pad contributes a linear ramp that decays from `1.0`
/// at the pad centre to `0.0` at `creep` nanometres. The total is clamped to `[0, 1]`.
///
/// **Why sum instead of nearest-only**: the nearest-pad model collapses to 0 once the closest
/// pad is beyond `creep`, even when a cell is surrounded by many pads each individually just
/// inside the radius. The sum model correctly raises the hazard for dense pad clusters (e.g.
/// the dozens of VCCINT/GND balls on a 484-ball BGA), guiding the router to avoid the whole
/// cluster rather than threading through gaps between individually-distant pads.
///
/// Derivation of the 0→1 bound: a single pad at distance `d` contributes `(creep-d)/creep`.
/// Summing over `N` pads can exceed 1.0 (e.g. two pads at distance 0 each contribute 1.0 for a
/// sum of 2.0). The `min(sum, 1.0)` clamp keeps `node_base` in the same `[1.0, 1.0 + weight]`
/// range as the nearest-pad model, so the existing `creepage_weight` calibration remains valid.
pub(super) fn proximity(p: Point, pads: &[Point], creep: f64) -> f32 {
    if pads.is_empty() || creep <= 0.0 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for &q in pads {
        let d = p.euclid(q);
        sum += ((creep - d) / creep).max(0.0);
    }
    sum.min(1.0) as f32
}
