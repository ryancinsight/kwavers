//! Diagonal-routing geometry guards for the A* search.
//!
//! These predicates detect physically invalid diagonal moves and via placements that the
//! occupancy bit-field cannot catch — specifically, cases where a diagonal segment's interior
//! geometry would clip a foreign net's track corner or via column, even though neither node
//! of the diagonal is occupied by the foreign net.
//!
//! All functions are `pub(super)` (accessible within `crate::route`). The budget constant is
//! exposed at the same visibility so the test module in `search.rs` can pin its calibrated value.

use crate::geom::dist_point_seg;
use crate::route::grid::{Grid, NodeId, NO_OWNER, NO_VIA};

/// Centerline-distance budget (in nanometres, as `i64`) used by the converse diagonal-track/
/// via-placement guard in [`via_clips_foreign_diagonal_track`].
///
/// Set to 0.435 mm = 435_000 nm. Derivation from the empirical DONE↔PROG DRC residue on
/// `output/full_driver/fpga_controller_tile.kicad_drc.rpt`:
/// - centerline perpendicular distance (diagonal midpoint → via center) = 0.354 mm = √2 · 0.25;
/// - DRV: `min_clearance = 0.13 mm`, `hv_track = 0.25 mm` ⇒ half-track + via-pad = 0.305 mm;
/// - DRC satisfied iff `centerline_dist ≥ 0.305 (half-track + via-pad) + 0.13 = 0.435 mm`.
///
/// The guard enforces this bound at the point a foreign net is about to commit a via. The
/// centerline-distance budget is conservative — it does not consider track orientation; that
/// bias is acceptable because the guard is only charged when a routed diagonal track is already
/// in `grid.owner`, and the router must simply push the new via off that geometry.
pub(super) const DIAGONAL_VIA_CLEARANCE_BUDGET_NM: i64 = 435_000;

/// Guard: a diagonal move from `u` to `v` would cross the interior of a foreign net's
/// axis-aligned track edge.
///
/// A diagonal step `(ux,uy) → (vx,vy)` passes through the two intermediate corner cells
/// `(ux,vy)` and `(vx,uy)`. If both corners are owned by the same foreign net, the diagonal
/// slices through an axis-aligned edge of that net — a clearance violation not captured by
/// the occupancy check on either `u` or `v`.
pub(super) fn diagonal_crosses_foreign_edge(
    grid: &Grid,
    u: (usize, usize, usize),
    v: (usize, usize, usize),
    net_id: i32,
) -> bool {
    if u.2 != v.2 || u.0 == v.0 || u.1 == v.1 {
        return false;
    }
    let spec = grid.spec();
    let a = NodeId(spec.node_index(u.0, v.1, u.2));
    let b = NodeId(spec.node_index(v.0, u.1, u.2));
    let owner_a = grid.owner(a);
    owner_a != NO_OWNER && owner_a != net_id && owner_a == grid.owner(b)
}

/// Guard: a diagonal move from `u` to `v` passes beside a foreign via column at one of the
/// two intermediate corner cells.
///
/// A via's annular ring occupies the whole layer column, so a diagonal that brushes one of
/// the two corner cells adjacent to a via column would violate the via-clearance rule. This
/// guard catches that case before the step is committed.
pub(super) fn diagonal_passes_foreign_via_corner(
    grid: &Grid,
    u: (usize, usize, usize),
    v: (usize, usize, usize),
    net_id: i32,
) -> bool {
    if u.2 != v.2 || u.0 == v.0 || u.1 == v.1 {
        return false;
    }
    let spec = grid.spec();
    let corners = [
        NodeId(spec.node_index(u.0, v.1, u.2)),
        NodeId(spec.node_index(v.0, u.1, u.2)),
    ];
    corners.iter().any(|&corner| {
        let owner = grid.via_owner(corner);
        owner != NO_VIA && owner != net_id
    })
}

/// Converse guard for the foreign-diagonal ↔ via-placement case the corner-cell check cannot
/// gate. When the router considers dropping a via at column `v`, scan `v`'s **axial** neighbours
/// (the 4 cardinal cells N, S, E, W of `v`); an opposing pair of axial neighbours that are owned
/// by the **same foreign net** implies a foreign diagonal track whose endpoints sit one cell
/// away from the proposed via, arcing around `v`'s own cell so that no node of the segment is
/// `v` itself — PathFinder's `occupancy`-driven overuse bitset never sees the geometry as
/// contended.
///
/// The 4 candidate neighbour-pairs cover the four diagonal orientations (each pair connects
/// two axial neighbours that differ by `±1` in x and `±1` in y, forming a 45° segment that
/// "wraps around" `v`):
///
///  - `W ↔ N` `(cx-1, cy) → (cx, cy+1)`
///  - `N ↔ E` `(cx, cy+1) → (cx+1, cy)`
///  - `E ↔ S` `(cx+1, cy) → (cx, cy-1)`
///  - `S ↔ W` `(cx, cy-1) → (cx-1, cy)`
///
/// For each in-bounds pair owned by the same foreign net, the perpendicular distance from
/// `v`'s column centre to the diagonal segment is checked against `budget_nm`
/// ([`DIAGONAL_VIA_CLEARANCE_BUDGET_NM`] = 0.435 mm, derived from the DONE↔PROG DRC residue).
///
/// Why only axial-neighbour pairs: a diagonal A* step cannot jump 2 cells without routing
/// through an intermediate cell; if it routed through `v`, PathFinder's overuse bitset would
/// already catch the conflict.
pub(super) fn via_clips_foreign_diagonal_track(
    grid: &Grid,
    v: (usize, usize, usize),
    net_id: i32,
    budget_nm: i64,
) -> bool {
    let (cx, cy, cl) = v;
    let spec = grid.spec();
    if cl >= spec.nlayers {
        return false;
    }
    let nx = spec.nx as i64;
    let ny = spec.ny as i64;
    let cx_i = cx as i64;
    let cy_i = cy as i64;
    // (ax, ay, bx, by): pairs of axial neighbours of `v` whose offset (1, ±1) implies a
    // diagonally adjacent segment wrapping around `v`.
    let pairs = [
        (cx_i - 1, cy_i, cx_i, cy_i + 1), // W ↔ N
        (cx_i, cy_i + 1, cx_i + 1, cy_i), // N ↔ E
        (cx_i + 1, cy_i, cx_i, cy_i - 1), // E ↔ S
        (cx_i, cy_i - 1, cx_i - 1, cy_i), // S ↔ W
    ];
    for (ax, ay, bx, by) in pairs {
        if ax < 0 || bx < 0 || ay < 0 || by < 0 || ax >= nx || bx >= nx || ay >= ny || by >= ny {
            continue;
        }
        let n_a = NodeId(spec.node_index(ax as usize, ay as usize, cl));
        let n_b = NodeId(spec.node_index(bx as usize, by as usize, cl));
        let owner_a = grid.owner(n_a);
        let owner_b = grid.owner(n_b);
        if owner_a == NO_OWNER || owner_b == NO_OWNER || owner_a != owner_b || owner_a == net_id {
            continue;
        }
        let p_a = spec.point_of(ax as usize, ay as usize);
        let p_b = spec.point_of(bx as usize, by as usize);
        let p_col = spec.point_of(cx, cy);
        let d_nm = dist_point_seg(p_col, p_a, p_b) as i64;
        if d_nm < budget_nm {
            return true;
        }
    }
    false
}

/// Guard: a diagonal move from `u` to `v` passes beside a foreign track at one of the two
/// intermediate corner cells.
///
/// Complementary to [`diagonal_passes_foreign_via_corner`] but for owned track segments:
/// if a corner cell of the diagonal is occupied by a foreign net's track, the diagonal would
/// produce an acid-trap geometry adjacent to that copper.
pub(super) fn diagonal_passes_foreign_track_corner(
    grid: &Grid,
    u: (usize, usize, usize),
    v: (usize, usize, usize),
    net_id: i32,
) -> bool {
    if u.2 != v.2 || u.0 == v.0 || u.1 == v.1 {
        return false;
    }
    let spec = grid.spec();
    let corners = [
        NodeId(spec.node_index(u.0, v.1, u.2)),
        NodeId(spec.node_index(v.0, u.1, u.2)),
    ];
    corners.iter().any(|&corner| {
        let owner = grid.owner(corner);
        owner != NO_OWNER && owner != net_id
    })
}
