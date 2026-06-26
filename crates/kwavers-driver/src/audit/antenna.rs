use std::collections::HashMap;

use crate::board::Board;
use crate::geom::{
    Nm, Point,
};
///
/// Track-track junctions coincide exactly (grid-cell centres), matched exactly by endpoint count. A
/// track end meets a pad/via *within a tolerance* (the router snaps terminals to the nearest cell
/// centre, ≤ half a pitch from the true pad centre), so that anchoring uses proximity. KiCad's
/// `track_dangling` warning is stricter for track-track copper: an endpoint that merely touches the
/// body of another segment must be split into an explicit endpoint junction. The audit mirrors that
/// manufacturing rule so unsplit T-junctions are visible before export.
pub(crate) fn dangling_ends(board: &Board) -> (usize, Vec<Point>) {
    let tol = board.spec.pitch.0 as f64; // within one cell of a pad/via ⇒ connected
    let key = |p: Point| (p.x.0, p.y.0);
    let mut ends: HashMap<(i64, i64), u32> = HashMap::new();
    for t in &board.tracks {
        *ends.entry(key(t.start)).or_default() += 1;
        *ends.entry(key(t.end)).or_default() += 1;
    }
    let anchors: Vec<Point> = board
        .pads
        .iter()
        .map(|p| p.pos)
        .chain(board.vias.iter().map(|v| v.pos))
        .collect();

    let mut count = 0;
    let mut pts = Vec::new();
    let mut seen: std::collections::HashSet<(i64, i64)> = std::collections::HashSet::new();
    for t in &board.tracks {
        for end in [t.start, t.end] {
            let k = key(end);
            if ends[&k] >= 2 || !seen.insert(k) {
                continue; // a junction, or already judged
            }
            let anchored = anchors.iter().any(|a| a.euclid(end) <= tol);
            if !anchored {
                count += 1;
                pts.push(end);
            }
        }
    }
    (count, pts)
}

/// Copper area (mm²) per layer — routed tracks plus the planar area of any fill [`crate::board::Zone`]
/// on that layer (a plane is the dominant copper on its layer, so it must be counted for a meaningful
/// balance figure). Zone area is the shoelace area of its outline (the filler trims it slightly for
/// clearance; this is the ≤1 % conservative upper bound).
#[must_use]
pub fn copper_area_per_layer(board: &Board) -> Vec<f64> {
    let mut a = vec![0.0f64; board.spec.nlayers];
    for t in &board.tracks {
        let len_mm = t.start.euclid(t.end) * 1.0e-6;
        let l = t.layer.0 as usize;
        if l < a.len() {
            a[l] += len_mm * t.width.to_mm();
        }
    }
    for z in &board.zones {
        let l = z.layer.0 as usize;
        if l < a.len() {
            a[l] += polygon_area_mm2(&z.polygon);
        }
    }
    a
}

/// Absolute shoelace area (mm²) of a closed polygon given in nanometre coordinates.
pub(crate) fn polygon_area_mm2(poly: &[Point]) -> f64 {
    if poly.len() < 3 {
        return 0.0;
    }
    let mut s2 = 0.0f64;
    for i in 0..poly.len() {
        let a = poly[i];
        let b = poly[(i + 1) % poly.len()];
        s2 += a.x.0 as f64 * b.y.0 as f64 - b.x.0 as f64 * a.y.0 as f64;
    }
    (s2.abs() / 2.0) * 1.0e-12 // nm² → mm²
}

pub(crate) fn polygon_vertex_mean(poly: &[Point]) -> Option<Point> {
    if poly.is_empty() {
        return None;
    }
    let mut x = 0_i64;
    let mut y = 0_i64;
    for p in poly {
        x += p.x.0;
        y += p.y.0;
    }
    Some(Point::new(
        Nm(x / poly.len() as i64),
        Nm(y / poly.len() as i64),
    ))
}

/// Copper imbalance relevant to **reflow warpage**: the worst relative copper-area difference between
/// **symmetric layer pairs** (layer `i` ↔ layer `n−1−i` about the stack mid-plane), in `[0, 1]`.
/// Warpage is driven by copper asymmetry about the neutral plane, so a stack with matched symmetric
/// pairs (e.g. plane↔plane, signal↔signal) reads near 0 even when planes carry far more copper than
/// signal layers; one-sided copper reads near 1. A single layer maps to 0.
#[must_use]
pub fn copper_imbalance(board: &Board) -> f64 {
    let a = copper_area_per_layer(board);
    let n = a.len();
    let mut worst = 0.0f64;
    for i in 0..n / 2 {
        let (hi, lo) = (a[i].max(a[n - 1 - i]), a[i].min(a[n - 1 - i]));
        if hi > 0.0 {
            worst = worst.max((hi - lo) / hi);
        }
    }
    worst
}

/// Different-net via pairs closer than `min_gap` (centre distance) — annular rings would overlap.
pub(crate) fn via_adjacency(board: &Board, min_gap: Nm) -> (usize, Vec<Point>) {
    let g = min_gap.0 as f64;
    let v = &board.vias;
    let mut count = 0;
    let mut pts = Vec::new();
    for i in 0..v.len() {
        for vj in v.iter().skip(i + 1) {
            if v[i].net == vj.net {
                continue;
            }
            if v[i].pos.euclid(vj.pos) < g {
                count += 1;
                pts.push(v[i].pos);
            }
        }
    }
    (count, pts)
}

/// Parallel-adjacent different-net track pairs on the same layer (crosstalk-prone).
pub(crate) fn crosstalk(board: &Board, coupling: Nm) -> usize {
    let c = coupling.0;
    let mut count = 0;
    let t = &board.tracks;
    for i in 0..t.len() {
        let (a, b) = (t[i].start, t[i].end);
        let horiz_i = a.y.0 == b.y.0;
        let vert_i = a.x.0 == b.x.0;
        if !horiz_i && !vert_i {
            continue;
        }
        for tj in t.iter().skip(i + 1) {
            if tj.net == t[i].net || tj.layer != t[i].layer {
                continue;
            }
            let (c1, c2) = (tj.start, tj.end);
            if horiz_i && c1.y.0 == c2.y.0 {
                // both horizontal: adjacent rows, overlapping x-extent
                if (a.y.0 - c1.y.0).abs() <= c
                    && a.x.0.min(b.x.0) < c1.x.0.max(c2.x.0)
                    && c1.x.0.min(c2.x.0) < a.x.0.max(b.x.0)
                {
                    count += 1;
                }
            } else if vert_i
                && c1.x.0 == c2.x.0
                && (a.x.0 - c1.x.0).abs() <= c
                && a.y.0.min(b.y.0) < c1.y.0.max(c2.y.0)
                && c1.y.0.min(c2.y.0) < a.y.0.max(b.y.0)
            {
                count += 1;
            }
        }
    }
    count
}
