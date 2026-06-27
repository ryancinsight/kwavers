
use crate::board::Board;
use crate::geom::{
    Nm, Point,
};
use crate::rules::DesignRules;
use crate::audit::fault_report::is_hv;
use crate::audit::crosstalk::point_features;
/// Different-net copper whose **edge-to-edge** gap is below `risk_margin` (a graded warning band
/// beyond binary DRC); HV-involved pairs weighted 2×, `score` ramps linearly as the gap closes.
///
/// Three proximity classes: same-layer track↔track (centre-line distance minus both half-widths),
/// track↔point (point feature to a same-or-shared-layer track edge), and point↔point (vias/pads).
/// Cheap axis-bbox rejects keep it near-linear on realistic boards.
pub(crate) fn near_shorts(board: &Board, risk_margin: Nm) -> (usize, f64, Vec<Point>) {
    let m = risk_margin.0 as f64;
    let mut count = 0usize;
    let mut score = 0.0f64;
    let mut pts = Vec::new();
    let mut hit = |gap: f64, hv: bool, at: Point, count: &mut usize, score: &mut f64| {
        if gap < m {
            *count += 1;
            let w = if hv { 2.0 } else { 1.0 };
            *score += w * (m - gap.max(0.0)) / m;
            pts.push(at);
        }
    };

    // (a) same-layer track ↔ track, edge-to-edge.
    let tr = &board.tracks;
    for i in 0..tr.len() {
        let ti = &tr[i];
        let half_i = ti.width.0 as f64 / 2.0;
        for tj in tr.iter().skip(i + 1) {
            if ti.net == tj.net || ti.layer != tj.layer {
                continue;
            }
            // Cheap bbox reject (expand by margin + both half-widths).
            let pad = m + half_i + tj.width.0 as f64 / 2.0;
            let (ix0, ix1) = (ti.start.x.0.min(ti.end.x.0), ti.start.x.0.max(ti.end.x.0));
            let (jx0, jx1) = (tj.start.x.0.min(tj.end.x.0), tj.start.x.0.max(tj.end.x.0));
            if (ix0 - jx1) as f64 > pad || (jx0 - ix1) as f64 > pad {
                continue;
            }
            let (iy0, iy1) = (ti.start.y.0.min(ti.end.y.0), ti.start.y.0.max(ti.end.y.0));
            let (jy0, jy1) = (tj.start.y.0.min(tj.end.y.0), tj.start.y.0.max(tj.end.y.0));
            if (iy0 - jy1) as f64 > pad || (jy0 - iy1) as f64 > pad {
                continue;
            }
            let center = crate::geom::dist_seg_seg(ti.start, ti.end, tj.start, tj.end);
            let gap = center - half_i - tj.width.0 as f64 / 2.0;
            let hv = is_hv(board, ti.net) || is_hv(board, tj.net);
            let at = Point::new(
                Nm((ti.start.x.0 + tj.start.x.0) / 2),
                Nm((ti.start.y.0 + tj.start.y.0) / 2),
            );
            hit(gap, hv, at, &mut count, &mut score);
        }
    }

    // (b) point feature ↔ track edge (a via/pad near a foreign track) **on a layer the feature's
    // copper actually occupies**. The feature's own copper radius is subtracted so the gap is
    // copper-edge-to-copper-edge. The layer gate is what stops a 0→1 HDI micro-via from being
    // counted against a track on layer 2/3 it never touches (the dominant false-positive source).
    let feats = point_features(board);
    for ft in &feats {
        for t in tr {
            if t.net == ft.net || !ft.on_layer(t.layer.0) {
                continue;
            }
            let gap = crate::geom::dist_point_seg(ft.pos, t.start, t.end)
                - t.width.0 as f64 / 2.0
                - ft.radius;
            let hv = ft.hv || is_hv(board, t.net);
            hit(gap, hv, ft.pos, &mut count, &mut score);
        }
    }

    // (c) point ↔ point (via/pad copper of different nets): edge-to-edge gap is centre distance
    // minus both copper radii, but only when their **layer spans overlap** (two discs on disjoint
    // layers cannot clash). The bbox reject widens by the radii so a near pair is not culled.
    for i in 0..feats.len() {
        let fi = &feats[i];
        for fj in feats.iter().skip(i + 1) {
            if fi.net == fj.net || !fi.spans_overlap(fj) {
                continue;
            }
            let reach = m + fi.radius + fj.radius;
            if (fi.pos.x.0 - fj.pos.x.0).abs() as f64 > reach
                || (fi.pos.y.0 - fj.pos.y.0).abs() as f64 > reach
            {
                continue;
            }
            let at = Point::new(
                Nm((fi.pos.x.0 + fj.pos.x.0) / 2),
                Nm((fi.pos.y.0 + fj.pos.y.0) / 2),
            );
            let gap = fi.pos.euclid(fj.pos) - fi.radius - fj.radius;
            hit(gap, fi.hv || fj.hv, at, &mut count, &mut score);
        }
    }

    (count, score, pts)
}

/// Drilled-hole-to-copper clearance: a via barrel's hole edge must hold [`DesignRules::hole_clearance`]
/// from every **foreign-net** track on a layer the barrel passes through. This is the internal mirror
/// of kicad-cli's `hole_clearance` class. The barrel spans `[from, to]`, so only copper on those
/// layers is at risk; same-net copper (the via's own connection) is exempt. Pad copper is not modelled
/// here (the board pad carries no size), so this covers the dominant via-hole↔track case.
pub(crate) fn detect_hole_clearance_violations(board: &Board, rules: &DesignRules) -> (usize, Vec<Point>) {
    let hc = rules.hole_clearance().0 as f64;
    let mut count = 0usize;
    let mut pts = Vec::new();
    for v in &board.vias {
        let r = v.drill.0 as f64 / 2.0;
        let (lo, hi) = (v.from.0.min(v.to.0), v.from.0.max(v.to.0));
        for t in &board.tracks {
            if t.net == v.net || t.layer.0 < lo || t.layer.0 > hi {
                continue;
            }
            let gap =
                crate::geom::dist_point_seg(v.pos, t.start, t.end) - t.width.0 as f64 / 2.0 - r;
            if gap < hc {
                count += 1;
                pts.push(v.pos);
            }
        }
    }
    (count, pts)
}

/// Acute-angle junctions: two connected same-net, same-layer segments meeting at a shared endpoint
/// with an interior angle `< 90°` trap etchant (an "acid trap") and are a standard DFM reject. A
/// Manhattan or 45°-chamfered router produces only `≥ 90°` turns, so a clean board reads 0.
pub(crate) fn acid_traps(board: &Board) -> usize {
    let tr = &board.tracks;
    let key = |p: Point| (p.x.0, p.y.0);
    let mut count = 0;
    for i in 0..tr.len() {
        for tj in tr.iter().skip(i + 1) {
            if tr[i].net != tj.net || tr[i].layer != tj.layer {
                continue;
            }
            // Find a shared endpoint; the other two endpoints are the arms.
            let (a, b) = (tr[i].start, tr[i].end);
            let (c, d) = (tj.start, tj.end);
            let (apex, u, v) = if key(a) == key(c) {
                (a, b, d)
            } else if key(a) == key(d) {
                (a, b, c)
            } else if key(b) == key(c) {
                (b, a, d)
            } else if key(b) == key(d) {
                (b, a, c)
            } else {
                continue;
            };
            // Interior angle < 90° ⇔ dot(apex→u, apex→v) > 0.
            let (ux, uy) = ((u.x.0 - apex.x.0) as f64, (u.y.0 - apex.y.0) as f64);
            let (vx, vy) = ((v.x.0 - apex.x.0) as f64, (v.y.0 - apex.y.0) as f64);
            if ux * vx + uy * vy > 0.0 {
                count += 1;
            }
        }
    }
    count
}
