//! Track- and via-geometry violation detectors.
//!
//! Covers sharp bends, track crossings, serpentine spacing, via spacing, plane
//! hotspot via clustering, and parallel-bus length mismatch.

use std::collections::{BTreeMap, HashMap};

use crate::audit::net_util::parallel_bus_group_name;
use crate::board::{Board, LayerId, NetClassKind, NetId, Track};
use crate::geom::{segments_cross, Nm, Point};
use crate::rules::DesignRules;

pub(crate) fn detect_sharp_bends(board: &Board) -> (usize, Vec<Point>) {
    let tr = &board.tracks;
    let key = |p: Point| (p.x.0, p.y.0);
    let mut count = 0;
    let mut pts = Vec::new();
    for i in 0..tr.len() {
        for tj in tr.iter().skip(i + 1) {
            if tr[i].net != tj.net || tr[i].layer != tj.layer {
                continue;
            }
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
            let (ux, uy) = ((u.x.0 - apex.x.0) as f64, (u.y.0 - apex.y.0) as f64);
            let (vx, vy) = ((v.x.0 - apex.x.0) as f64, (v.y.0 - apex.y.0) as f64);
            let u_len = ux.hypot(uy);
            let v_len = vx.hypot(vy);
            if u_len <= f64::EPSILON || v_len <= f64::EPSILON {
                continue;
            }
            let dot = ux * vx + uy * vy;
            if dot >= -1e-5 * u_len * v_len {
                count += 1;
                pts.push(apex);
            }
        }
    }
    (count, pts)
}

pub(crate) fn detect_track_crossing_violations(board: &Board) -> (usize, Vec<Point>) {
    let mut count = 0;
    let mut pts = Vec::new();
    for (i, a) in board.tracks.iter().enumerate() {
        for b in board.tracks.iter().skip(i + 1) {
            if a.layer != b.layer || a.net == b.net {
                continue;
            }
            if segments_cross(a.start, a.end, b.start, b.end) {
                count += 1;
                pts.push(Point::new(
                    Nm((a.start.x.0 + a.end.x.0 + b.start.x.0 + b.end.x.0) / 4),
                    Nm((a.start.y.0 + a.end.y.0 + b.start.y.0 + b.end.y.0) / 4),
                ));
            }
        }
    }
    (count, pts)
}

pub(crate) fn detect_serpentine_violations(
    board: &Board,
    rules: &DesignRules,
) -> (usize, usize, usize, Vec<Point>) {
    let mut spacing_violations = 0;
    let mut length_violations = 0;
    let mut compensation_distance_violations = 0;
    let mut pts = Vec::new();
    let tr = &board.tracks;
    let key = |p: Point| (p.x.0, p.y.0);

    let mut ends: HashMap<(i64, i64), u32> = HashMap::new();
    for t in tr {
        *ends.entry(key(t.start)).or_default() += 1;
        *ends.entry(key(t.end)).or_default() += 1;
    }

    for t in tr {
        let is_routing_seg = ends.get(&key(t.start)).copied().unwrap_or(0) >= 2
            && ends.get(&key(t.end)).copied().unwrap_or(0) >= 2;
        if is_routing_seg {
            let len = t.start.euclid(t.end);
            let limit = 1.5 * t.width.0 as f64;
            if len < limit {
                length_violations += 1;
                pts.push(Point::new(
                    Nm((t.start.x.0 + t.end.x.0) / 2),
                    Nm((t.start.y.0 + t.end.y.0) / 2),
                ));
            }
        }
    }

    let mut groups: HashMap<(NetId, LayerId), Vec<&Track>> = HashMap::new();
    for t in tr {
        groups.entry((t.net, t.layer)).or_default().push(t);
    }

    for ((_net, _layer), net_tracks) in groups {
        let mut bends = Vec::new();
        for i in 0..net_tracks.len() {
            for other in net_tracks.iter().skip(i + 1) {
                let t = net_tracks[i];
                let (a, b) = (t.start, t.end);
                let (c, d) = (other.start, other.end);
                let Some(apex) = (if key(a) == key(c) || key(a) == key(d) {
                    Some(a)
                } else if key(b) == key(c) || key(b) == key(d) {
                    Some(b)
                } else {
                    None
                }) else {
                    continue;
                };
                let h0 = t.start.y.0 == t.end.y.0;
                let h1 = other.start.y.0 == other.end.y.0;
                let v0 = t.start.x.0 == t.end.x.0;
                let v1 = other.start.x.0 == other.end.x.0;
                if (h0 && v1) || (v0 && h1) {
                    bends.push(apex);
                }
            }
        }

        let mut horiz = Vec::new();
        let mut vert = Vec::new();
        for t in net_tracks {
            let h = t.start.y.0 == t.end.y.0;
            let v = t.start.x.0 == t.end.x.0;
            if h {
                horiz.push(t);
            } else if v {
                vert.push(t);
            }
        }

        horiz.sort_by_key(|t| t.start.y.0);
        for i in 0..horiz.len() {
            let ti = horiz[i];
            let xi_min = ti.start.x.0.min(ti.end.x.0);
            let xi_max = ti.start.x.0.max(ti.end.x.0);
            for &tj in horiz.iter().skip(i + 1) {
                let y_diff = (ti.start.y.0 - tj.start.y.0).abs();
                let required = 4 * ti.width.0.max(tj.width.0);
                let edge_gap = y_diff - (ti.width.0 + tj.width.0) / 2;
                let xj_min = tj.start.x.0.min(tj.end.x.0);
                let xj_max = tj.start.x.0.max(tj.end.x.0);
                let overlap_min = xi_min.max(xj_min);
                let overlap_max = xi_max.min(xj_max);
                if edge_gap >= required && overlap_min >= overlap_max {
                    break;
                }
                if overlap_min < overlap_max {
                    let midpoint = Point::new(
                        Nm((overlap_min + overlap_max) / 2),
                        Nm((ti.start.y.0 + tj.start.y.0) / 2),
                    );
                    let overlap_len = overlap_max - overlap_min;
                    if edge_gap < required && overlap_len >= 10 * ti.width.0 {
                        spacing_violations += 1;
                        pts.push(midpoint);
                    }
                    if edge_gap <= rules.serpentine_compensation_bend_distance.0
                        && nearest_distance(midpoint, &bends)
                            > rules.serpentine_compensation_bend_distance.0 as f64
                    {
                        compensation_distance_violations += 1;
                        pts.push(midpoint);
                    }
                }
            }
        }

        vert.sort_by_key(|t| t.start.x.0);
        for i in 0..vert.len() {
            let ti = vert[i];
            let yi_min = ti.start.y.0.min(ti.end.y.0);
            let yi_max = ti.start.y.0.max(ti.end.y.0);
            for &tj in vert.iter().skip(i + 1) {
                let x_diff = (ti.start.x.0 - tj.start.x.0).abs();
                let required = 4 * ti.width.0.max(tj.width.0);
                let edge_gap = x_diff - (ti.width.0 + tj.width.0) / 2;
                let yj_min = tj.start.y.0.min(tj.end.y.0);
                let yj_max = tj.start.y.0.max(tj.end.y.0);
                let overlap_min = yi_min.max(yj_min);
                let overlap_max = yi_max.min(yj_max);
                if edge_gap >= required && overlap_min >= overlap_max {
                    break;
                }
                if overlap_min < overlap_max {
                    let midpoint = Point::new(
                        Nm((ti.start.x.0 + tj.start.x.0) / 2),
                        Nm((overlap_min + overlap_max) / 2),
                    );
                    let overlap_len = overlap_max - overlap_min;
                    if edge_gap < required && overlap_len >= 10 * ti.width.0 {
                        spacing_violations += 1;
                        pts.push(midpoint);
                    }
                    if edge_gap <= rules.serpentine_compensation_bend_distance.0
                        && nearest_distance(midpoint, &bends)
                            > rules.serpentine_compensation_bend_distance.0 as f64
                    {
                        compensation_distance_violations += 1;
                        pts.push(midpoint);
                    }
                }
            }
        }
    }

    (
        spacing_violations,
        length_violations,
        compensation_distance_violations,
        pts,
    )
}

pub(crate) fn nearest_distance(p: Point, points: &[Point]) -> f64 {
    points
        .iter()
        .map(|&q| p.euclid(q))
        .min_by(f64::total_cmp)
        .unwrap_or(f64::INFINITY)
}

pub(crate) fn detect_via_spacing_violations(
    board: &Board,
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    let mut count = 0;
    let mut pts = Vec::new();
    let vias = &board.vias;
    let limit = rules.min_via_to_via_spacing.0 as f64;
    for i in 0..vias.len() {
        let vi = &vias[i];
        for vj in vias.iter().skip(i + 1) {
            if vi.net == vj.net {
                continue;
            }
            let dist = vi.pos.euclid(vj.pos);
            let gap = dist - (vi.diameter.0 as f64 + vj.diameter.0 as f64) / 2.0;
            if gap < limit {
                count += 1;
                pts.push(vi.pos);
            }
        }
    }
    (count, pts)
}

pub(crate) fn detect_plane_hotspot_via_spacing_violations(
    board: &Board,
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    let mut count = 0;
    let mut pts = Vec::new();
    let vias = &board.vias;
    let limit = rules.min_via_to_via_spacing.0 as f64;
    for i in 0..vias.len() {
        let vi = &vias[i];
        if board.class_of(vi.net) == NetClassKind::Ground {
            continue;
        }
        for vj in vias.iter().skip(i + 1) {
            if vi.net != vj.net {
                continue;
            }
            let dist = vi.pos.euclid(vj.pos);
            let gap = dist - (vi.diameter.0 as f64 + vj.diameter.0 as f64) / 2.0;
            if gap < limit {
                count += 1;
                pts.push(Point::new(
                    Nm((vi.pos.x.0 + vj.pos.x.0) / 2),
                    Nm((vi.pos.y.0 + vj.pos.y.0) / 2),
                ));
            }
        }
    }
    (count, pts)
}

pub(crate) fn detect_parallel_bus_length_mismatch_violations(
    board: &Board,
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    let mut groups: BTreeMap<String, Vec<NetId>> = BTreeMap::new();
    for net in &board.nets {
        if board.class_of(net.id) != NetClassKind::Signal {
            continue;
        }
        if let Some(group) = parallel_bus_group_name(&net.name) {
            groups.entry(group).or_default().push(net.id);
        }
    }

    let mut count = 0;
    let mut pts = Vec::new();
    let tolerance_mm = rules.parallel_bus_length_tolerance.0 as f64 * 1.0e-6;
    for nets in groups.values() {
        if nets.len() < 2 {
            continue;
        }
        let mut lengths = Vec::new();
        for &net in nets {
            let mut length_mm = 0.0;
            let mut midpoint = None;
            for track in board.tracks.iter().filter(|track| track.net == net) {
                length_mm += track.start.euclid(track.end) * 1.0e-6;
                midpoint = Some(Point::new(
                    Nm((track.start.x.0 + track.end.x.0) / 2),
                    Nm((track.start.y.0 + track.end.y.0) / 2),
                ));
            }
            if length_mm > 0.0 {
                lengths.push((length_mm, midpoint));
            }
        }
        if lengths.len() < 2 {
            continue;
        }
        let min = lengths
            .iter()
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .expect("lengths has at least two routed bus nets");
        let max = lengths
            .iter()
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .expect("lengths has at least two routed bus nets");
        if max.0 - min.0 > tolerance_mm {
            count += 1;
            if let (Some(a), Some(b)) = (min.1, max.1) {
                pts.push(Point::new(Nm((a.x.0 + b.x.0) / 2), Nm((a.y.0 + b.y.0) / 2)));
            }
        }
    }

    (count, pts)
}
