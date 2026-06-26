use std::collections::HashMap;

use crate::board::Board;
use crate::geom::{
    Nm, Point,
};
use crate::rules::DesignRules;

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

/// Detects antenna-connected traces whose characteristic impedance deviates from the 50 Ω
/// standard by more than `rules.antenna_impedance_tolerance_ohm`.
///
/// Antenna traces must be controlled-impedance 50 Ω transmission lines; a mismatch causes
/// standing waves, reduced range, and conducted/radiated EMI (article Mistake 9). The
/// Hammerstad microstrip formula is used: `Z = f(w, h, εr)` with `rules.dielectric_er` and
/// `rules.dielectric_height_mm`. Nets whose name (case-insensitive) starts with `"ANT"` are
/// treated as antenna nets. Each violating track segment contributes 1 violation.
///
/// # Vacuous condition
///
/// Returns `(0, [])` when `rules.antenna_impedance_ohm <= 0` or
/// `rules.dielectric_height_mm <= 0`.
pub(crate) fn detect_antenna_impedance_mismatch(
    board: &Board,
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    let target = rules.antenna_impedance_ohm;
    let tol = rules.antenna_impedance_tolerance_ohm;
    if target <= 0.0 || rules.dielectric_height_mm <= 0.0 {
        return (0, Vec::new());
    }

    let mut count = 0;
    let mut pts = Vec::new();

    for t in &board.tracks {
        let net_idx = t.net.0 as usize;
        if net_idx >= board.nets.len() {
            continue;
        }
        let name = &board.nets[net_idx].name;
        if !name.to_ascii_uppercase().starts_with("ANT") {
            continue;
        }
        let w_mm = t.width.to_mm();
        let h_mm = rules.dielectric_height_mm;
        let er = rules.dielectric_er.max(1.0);
        let z = crate::physics::si::impedance::microstrip_impedance(w_mm, h_mm, er);
        if (z - target).abs() > tol {
            count += 1;
            pts.push(Point::new(
                Nm((t.start.x.0 + t.end.x.0) / 2),
                Nm((t.start.y.0 + t.end.y.0) / 2),
            ));
        }
    }
    (count, pts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{LayerId, NetClassKind, Track};
    use crate::geom::GridSpec;

    fn make_ant_board(track_width_mm: f64) -> Board {
        let spec = GridSpec::cover(
            Nm::from_mm(50.0),
            Nm::from_mm(50.0),
            Nm::from_mm(0.5),
            2,
        )
        .unwrap();
        let mut b = Board::new(spec);
        let net = b.add_net("ANT_TX", NetClassKind::Signal);
        b.tracks.push(Track {
            start: Point::new(Nm(0), Nm(0)),
            end: Point::new(Nm::from_mm(10.0), Nm(0)),
            width: Nm::from_mm(track_width_mm),
            net,
            layer: LayerId(0),
        });
        b
    }

    /// A 50-Ω microstrip on FR4 (εr=4.5, h=0.2 mm) has w≈0.374 mm.
    /// Tracks far outside this width must be flagged; a width very close to 0.374 mm must pass.
    #[test]
    fn flags_wide_antenna_trace_impedance_mismatch() {
        // 2.0 mm wide → very low Z (well under 10 Ω) → mismatch
        let b = make_ant_board(2.0);
        let rules = DesignRules::holohv();
        let (n, _) = detect_antenna_impedance_mismatch(&b, &rules);
        assert_eq!(n, 1, "2 mm wide ANT trace must fail 50-Ω check");
    }

    #[test]
    fn passes_matched_antenna_trace() {
        // 0.374 mm ≈ 50 Ω on standard 4-layer prepreg (εr=4.5, h=0.2 mm); tolerance ±10 Ω
        let b = make_ant_board(0.374);
        let rules = DesignRules::holohv();
        let (n, _) = detect_antenna_impedance_mismatch(&b, &rules);
        assert_eq!(n, 0, "0.374 mm wide ANT trace should be near 50 Ω");
    }

    #[test]
    fn ignores_non_ant_nets() {
        let spec = GridSpec::cover(
            Nm::from_mm(50.0),
            Nm::from_mm(50.0),
            Nm::from_mm(0.5),
            2,
        )
        .unwrap();
        let mut b = Board::new(spec);
        let net = b.add_net("SPI_CLK", NetClassKind::Signal);
        b.tracks.push(Track {
            start: Point::new(Nm(0), Nm(0)),
            end: Point::new(Nm::from_mm(10.0), Nm(0)),
            width: Nm::from_mm(2.0),
            net,
            layer: LayerId(0),
        });
        let rules = DesignRules::holohv();
        let (n, _) = detect_antenna_impedance_mismatch(&b, &rules);
        assert_eq!(n, 0, "non-ANT nets must not be impedance-checked");
    }

    #[test]
    fn vacuous_when_target_impedance_is_zero() {
        let b = make_ant_board(2.0);
        let mut rules = DesignRules::holohv();
        rules.antenna_impedance_ohm = 0.0;
        let (n, _) = detect_antenna_impedance_mismatch(&b, &rules);
        assert_eq!(n, 0, "zero target impedance makes check vacuous");
    }
}
