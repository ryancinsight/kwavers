
use crate::board::{
    Board, NetId,
};
use crate::geom::{
    segments_cross, Nm, Point,
};
use crate::audit::fault_report::is_hv;
/// HV↔LV pad pairs closer than `coupling` — EMI aggressor↔victim zones the next placement should pull apart.
/// Unlike `near_shorts` (any different-net copper), this specifically targets the HV-class-vs-non-HV-class
/// relationship at the pad level, so it drives the placer to *separate the switching node from sensitive control*
/// (a physics-guided placement signal, independent of routed copper).
///
/// Midpoints of each qualifying pair are returned for hotspot feedback. Pairs where both pads are non-HV are
/// ignored — they would be just LV↔LV signal adjacency, already covered by `near_shorts`.
#[must_use]
pub fn emi_hotspots(board: &Board, coupling: Nm) -> Vec<Point> {
    let m = coupling.0 as f64;
    let pads: Vec<(Point, bool)> = board
        .pads
        .iter()
        .filter_map(|p| p.net.map(|n| (p.pos, is_hv(board, n))))
        .collect();
    let mut pts = Vec::new();
    for i in 0..pads.len() {
        let (pi, hi) = pads[i];
        for &(pj, hj) in pads.iter().skip(i + 1) {
            if hi == hj {
                continue; // need exactly one HV and one non-HV
            }
            if pi.euclid(pj) < m {
                pts.push(Point::new(
                    Nm((pi.x.0 + pj.x.0) / 2),
                    Nm((pi.y.0 + pj.y.0) / 2),
                ));
            }
        }
    }
    pts
}

/// Minimum spanning tree (Prim) over a net's pads, returned as flight-line segments.
pub(crate) fn net_flight_lines(points: &[Point]) -> Vec<(Point, Point)> {
    if points.len() < 2 {
        return Vec::new();
    }
    let n = points.len();
    let mut in_tree = vec![false; n];
    in_tree[0] = true;
    let mut edges = Vec::with_capacity(n - 1);
    for _ in 1..n {
        let mut best: Option<(f64, usize, usize)> = None;
        for (i, &pi) in points.iter().enumerate() {
            if !in_tree[i] {
                continue;
            }
            for (j, &pj) in points.iter().enumerate() {
                if in_tree[j] {
                    continue;
                }
                let d = pi.euclid(pj);
                if best.map(|(bd, _, _)| d < bd).unwrap_or(true) {
                    best = Some((d, i, j));
                }
            }
        }
        let (_, i, j) = best.expect("invariant: a non-tree node exists while edges remain");
        in_tree[j] = true;
        edges.push((points[i], points[j]));
    }
    edges
}

/// Count proper crossings between flight lines of *different* nets, returning crossing midpoints.
pub(crate) fn flight_crossings(board: &Board) -> (usize, Vec<Point>) {
    let mut segs: Vec<(Point, Point, NetId, bool)> = Vec::new();
    for net in &board.nets {
        let pts: Vec<Point> = board.pads_of(net.id).map(|p| p.pos).collect();
        let hv = is_hv(board, net.id);
        for (a, b) in net_flight_lines(&pts) {
            segs.push((a, b, net.id, hv));
        }
    }
    let mut count = 0;
    let mut pts = Vec::new();
    for i in 0..segs.len() {
        for j in (i + 1)..segs.len() {
            if segs[i].2 == segs[j].2 {
                continue;
            }
            if segments_cross(segs[i].0, segs[i].1, segs[j].0, segs[j].1) {
                count += 1;
                let m = Point::new(
                    Nm((segs[i].0.x.0 + segs[i].1.x.0 + segs[j].0.x.0 + segs[j].1.x.0) / 4),
                    Nm((segs[i].0.y.0 + segs[i].1.y.0 + segs[j].0.y.0 + segs[j].1.y.0) / 4),
                );
                pts.push(m);
            }
        }
    }
    (count, pts)
}

/// Point copper "features" — via and pad centres (each is layer-spanning copper), as
/// `(point, net, is_hv)`. Track copper is handled by edge-to-edge segment proximity, not points, so
/// the metric is independent of how finely the router segmented a run (post-`merge_collinear`, a
/// straight run has no interior endpoints to spuriously count).
pub(crate) fn point_features(board: &Board) -> Vec<(Point, NetId, bool, f64)> {
    let mut f = Vec::new();
    for v in &board.vias {
        // A via is copper of radius `diameter/2` (the annular pad), not a zero-area point — the
        // clearance check must subtract that radius or it over-estimates the gap and misses the
        // violation kicad-cli flags against the via's copper edge.
        f.push((v.pos, v.net, is_hv(board, v.net), v.diameter.0 as f64 / 2.0));
    }
    for p in &board.pads {
        if let Some(n) = p.net {
            // `board.pads` carries no size, so the pad's copper extent is unknown here; treat it as a
            // point (radius 0). This is a conservative under-estimate for pad clearance — a documented
            // residual until board pads carry their footprint copper extent.
            f.push((p.pos, n, is_hv(board, n), 0.0));
        }
    }
    f
}
