use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::board::{
    split_domain_from_name, Board, LayerId, NetClassKind, NetId, SplitDomain, Track, ZoneFill,
};
use crate::geom::{
    dist_point_seg, dist_seg_seg, distance_to_polygon_boundary, point_in_polygon, segments_cross,
    GridSpec, Nm, Point,
};
use crate::place::component::is_surge_suppressor_refdes;
use crate::place::{Component, CongestionField, FootprintDef, Role};
use crate::rules::DesignRules;
use crate::verify::{parasitic_ac_coupling_check, schematic_isolation_bfs};
use crate::audit::fault_report::is_hv;
use crate::audit::antenna::{polygon_vertex_mean, polygon_area_mm2};
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

pub(crate) fn detect_via_spacing_violations(board: &Board, rules: &DesignRules) -> (usize, Vec<Point>) {
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

pub(crate) fn track_midpoint(track: &Track) -> Point {
    Point::new(
        Nm((track.start.x.0 + track.end.x.0) / 2),
        Nm((track.start.y.0 + track.end.y.0) / 2),
    )
}

pub(crate) fn diff_pair_layer_segment_lengths(tracks: &[&Track]) -> BTreeMap<LayerId, (f64, Option<Point>)> {
    let mut lengths = BTreeMap::new();
    for track in tracks {
        let entry = lengths
            .entry(track.layer)
            .or_insert_with(|| (0.0, Some(track_midpoint(track))));
        entry.0 += track.start.euclid(track.end);
    }
    lengths
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PairAxis {
    X,
    Y,
}

pub(crate) fn diff_pair_axis(board: &Board, p_net: NetId, n_net: NetId) -> Option<PairAxis> {
    let p_tracks: Vec<&Track> = board.tracks.iter().filter(|t| t.net == p_net).collect();
    let n_tracks: Vec<&Track> = board.tracks.iter().filter(|t| t.net == n_net).collect();
    for p in &p_tracks {
        let p_horizontal = p.start.y.0 == p.end.y.0;
        let p_vertical = p.start.x.0 == p.end.x.0;
        if !p_horizontal && !p_vertical {
            continue;
        }
        for n in &n_tracks {
            if p.layer != n.layer {
                continue;
            }
            if p_horizontal && n.start.y.0 == n.end.y.0 {
                return Some(PairAxis::X);
            }
            if p_vertical && n.start.x.0 == n.end.x.0 {
                return Some(PairAxis::Y);
            }
        }
    }
    None
}

pub(crate) fn detect_diff_pair_coupling_cap_symmetry_violations(
    board: &Board,
    comps: &[Component],
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    let pairs = diff_pair_members(board);
    let mut count = 0;
    let mut pts = Vec::new();
    let tolerance = rules.diff_pair_coupling_cap_symmetry_tolerance.0 as f64;

    for &(p_net, n_net) in &pairs {
        let Some(axis) = diff_pair_axis(board, p_net, n_net) else {
            continue;
        };
        let p_caps: Vec<&Component> = diff_pair_coupling_caps(comps, p_net).collect();
        let n_caps: Vec<&Component> = diff_pair_coupling_caps(comps, n_net).collect();
        if p_caps.is_empty() && n_caps.is_empty() {
            continue;
        }
        if p_caps.is_empty() || n_caps.is_empty() {
            count += 1;
            if let Some(c) = p_caps.first().or_else(|| n_caps.first()) {
                pts.push(c.placement.pos);
            }
            continue;
        }

        let best = p_caps
            .iter()
            .flat_map(|p| {
                n_caps.iter().map(move |n| match axis {
                    PairAxis::X => (p.placement.pos.x.0 - n.placement.pos.x.0).abs() as f64,
                    PairAxis::Y => (p.placement.pos.y.0 - n.placement.pos.y.0).abs() as f64,
                })
            })
            .fold(f64::INFINITY, f64::min);
        if best > tolerance {
            count += 1;
            pts.push(p_caps[0].placement.pos);
        }
    }

    (count, pts)
}

pub(crate) fn diff_pair_coupling_caps(comps: &[Component], net: NetId) -> impl Iterator<Item = &Component> {
    comps
        .iter()
        .filter(move |c| c.refdes.starts_with('C') && c.nets.contains(&Some(net)))
}

pub(crate) fn detect_diff_pair_coupling_cap_package_violations(
    board: &Board,
    comps: &[Component],
    lib: &[FootprintDef],
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    let mut count = 0;
    let mut pts = Vec::new();
    let max_courtyard = rules.diff_pair_coupling_cap_max_courtyard;

    for (p_net, n_net) in diff_pair_members(board) {
        for cap in
            diff_pair_coupling_caps(comps, p_net).chain(diff_pair_coupling_caps(comps, n_net))
        {
            if cap.fp >= lib.len() {
                continue;
            }
            let fp = &lib[cap.fp];
            if fp.courtyard.0 > max_courtyard || fp.courtyard.1 > max_courtyard {
                count += 1;
                pts.push(cap.placement.pos);
            }
        }
    }

    (count, pts)
}

pub(crate) fn diff_pair_pad_entry_distances(board: &Board, net: NetId) -> Vec<(f64, Point)> {
    let endpoints: Vec<Point> = board
        .tracks
        .iter()
        .filter(|track| track.net == net)
        .flat_map(|track| [track.start, track.end])
        .collect();
    if endpoints.is_empty() {
        return Vec::new();
    }

    let mut distances: Vec<(f64, Point)> = board
        .pads_of(net)
        .map(|pad| {
            let nearest = endpoints
                .iter()
                .map(|&endpoint| pad.pos.euclid(endpoint))
                .fold(f64::INFINITY, f64::min);
            (nearest, pad.pos)
        })
        .collect();
    distances.sort_by(|a, b| a.0.total_cmp(&b.0));
    distances
}

pub(crate) fn detect_diff_pair_pad_entry_mismatch_violations(
    board: &Board,
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    let mut count = 0;
    let mut pts = Vec::new();
    let tolerance = rules.diff_pair_pad_entry_tolerance.0 as f64;

    for (p_net, n_net) in diff_pair_members(board) {
        let p_entries = diff_pair_pad_entry_distances(board, p_net);
        let n_entries = diff_pair_pad_entry_distances(board, n_net);
        if p_entries.is_empty() && n_entries.is_empty() {
            continue;
        }
        if p_entries.len() != n_entries.len() {
            count += 1;
            if let Some((_, point)) = p_entries.first().or_else(|| n_entries.first()) {
                pts.push(*point);
            }
            continue;
        }
        let worst = p_entries
            .iter()
            .zip(&n_entries)
            .map(|(p, n)| (p.0 - n.0).abs())
            .fold(0.0, f64::max);
        if worst > tolerance {
            count += 1;
            if let Some((_, point)) = p_entries.first() {
                pts.push(*point);
            }
        }
    }

    (count, pts)
}

pub(crate) fn detect_diff_pair_pad_entry_length_violations(
    board: &Board,
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    let mut count = 0;
    let mut pts = Vec::new();
    let max_length = rules.diff_pair_pad_entry_max_length.0 as f64;

    for (p_net, n_net) in diff_pair_members(board) {
        let worst_entry = diff_pair_pad_entry_distances(board, p_net)
            .into_iter()
            .chain(diff_pair_pad_entry_distances(board, n_net))
            .max_by(|a, b| a.0.total_cmp(&b.0));
        let Some((worst, point)) = worst_entry else {
            continue;
        };
        if worst > max_length {
            count += 1;
            pts.push(point);
        }
    }

    (count, pts)
}

pub(crate) fn parallel_bus_group_name(name: &str) -> Option<String> {
    let upper = name.to_ascii_uppercase();
    let trimmed = upper.trim_end_matches(|c: char| c.is_ascii_digit());
    if trimmed.len() == upper.len() || !trimmed.contains("BUS") {
        return None;
    }
    let group = trimmed.trim_end_matches(['_', '-', '[']);
    (!group.is_empty()).then(|| group.to_string())
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

pub(crate) fn detect_diff_pair_violations(
    board: &Board,
    comps: &[Component],
    lib: &[FootprintDef],
    rules: &DesignRules,
) -> (
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    f64,
    Vec<Point>,
) {
    let mut count = 0;
    let mut layer_mismatch = 0;
    let mut via_count_mismatch = 0;
    let mut length_mismatch = 0;
    let mut segment_length_mismatch = 0;
    let mut spacing_variation = 0;
    let mut via_symmetry = 0;
    let mut total_mismatch_mm = 0.0;
    let mut pts = Vec::new();

    let pairs = diff_pair_members(board);

    for &(p_net, n_net) in &pairs {
        let p_tracks: Vec<&Track> = board.tracks.iter().filter(|t| t.net == p_net).collect();
        let n_tracks: Vec<&Track> = board.tracks.iter().filter(|t| t.net == n_net).collect();
        let p_layers: BTreeSet<LayerId> = p_tracks.iter().map(|t| t.layer).collect();
        let n_layers: BTreeSet<LayerId> = n_tracks.iter().map(|t| t.layer).collect();
        if p_layers != n_layers {
            layer_mismatch += 1;
            if let Some(t) = p_tracks.first().or_else(|| n_tracks.first()) {
                pts.push(Point::new(
                    Nm((t.start.x.0 + t.end.x.0) / 2),
                    Nm((t.start.y.0 + t.end.y.0) / 2),
                ));
            }
        }
        let p_vias = board.vias.iter().filter(|v| v.net == p_net).count();
        let n_vias = board.vias.iter().filter(|v| v.net == n_net).count();
        if p_vias != n_vias {
            via_count_mismatch += 1;
            if let Some(v) = board.vias.iter().find(|v| v.net == p_net || v.net == n_net) {
                pts.push(v.pos);
            } else if let Some(t) = p_tracks.first().or_else(|| n_tracks.first()) {
                pts.push(Point::new(
                    Nm((t.start.x.0 + t.end.x.0) / 2),
                    Nm((t.start.y.0 + t.end.y.0) / 2),
                ));
            }
        }
        if p_vias == n_vias && p_vias > 0 {
            if let Some(axis) = diff_pair_axis(board, p_net, n_net) {
                let mut p_stations: Vec<i64> = board
                    .vias
                    .iter()
                    .filter(|v| v.net == p_net)
                    .map(|v| match axis {
                        PairAxis::X => v.pos.x.0,
                        PairAxis::Y => v.pos.y.0,
                    })
                    .collect();
                let mut n_stations: Vec<i64> = board
                    .vias
                    .iter()
                    .filter(|v| v.net == n_net)
                    .map(|v| match axis {
                        PairAxis::X => v.pos.x.0,
                        PairAxis::Y => v.pos.y.0,
                    })
                    .collect();
                p_stations.sort_unstable();
                n_stations.sort_unstable();
                let worst = p_stations
                    .iter()
                    .zip(&n_stations)
                    .map(|(&p, &n)| (p - n).abs())
                    .max()
                    .unwrap_or(0);
                if worst > rules.diff_pair_via_symmetry_tolerance.0 {
                    via_symmetry += 1;
                    if let Some(v) = board.vias.iter().find(|v| v.net == p_net || v.net == n_net) {
                        pts.push(v.pos);
                    }
                }
            }
        }
        let p_len: f64 = p_tracks.iter().map(|t| t.start.euclid(t.end)).sum();
        let n_len: f64 = n_tracks.iter().map(|t| t.start.euclid(t.end)).sum();
        let delta_nm = (p_len - n_len).abs();
        let tol_nm = rules.diff_pair_length_tolerance.0 as f64;
        if delta_nm > tol_nm {
            length_mismatch += 1;
            // Per-mm mismatch fee (the LengthTolerance::DiffSignal penalty the diff-pair judge
            // folds into `risk_score` so the co-optimiser prefers lower-overshoot layouts).
            total_mismatch_mm += delta_nm / 1.0e6;
            if let Some(t) = p_tracks.first().or_else(|| n_tracks.first()) {
                pts.push(Point::new(
                    Nm((t.start.x.0 + t.end.x.0) / 2),
                    Nm((t.start.y.0 + t.end.y.0) / 2),
                ));
            }
        }

        let p_segment_lengths = diff_pair_layer_segment_lengths(&p_tracks);
        let n_segment_lengths = diff_pair_layer_segment_lengths(&n_tracks);
        let mut segment_hotspot = None;
        for (layer, &(p_segment_len, p_midpoint)) in &p_segment_lengths {
            if let Some(&(n_segment_len, n_midpoint)) = n_segment_lengths.get(layer) {
                let mismatch = (p_segment_len - n_segment_len).abs();
                if mismatch > rules.diff_pair_segment_length_tolerance.0 as f64 {
                    segment_hotspot = p_midpoint.or(n_midpoint);
                    break;
                }
            }
        }
        if let Some(point) = segment_hotspot {
            segment_length_mismatch += 1;
            pts.push(point);
        }

        let mut pair_spacings = Vec::new();
        let mut spacing_hotspot = None;
        for tp in &p_tracks {
            let hp = tp.start.y.0 == tp.end.y.0;
            let vp = tp.start.x.0 == tp.end.x.0;
            if !hp && !vp {
                continue;
            }

            for tn in &n_tracks {
                if tp.layer != tn.layer {
                    continue;
                }
                let hn = tn.start.y.0 == tn.end.y.0;
                let vn = tn.start.x.0 == tn.end.x.0;
                // Skip L-shape conversion stub legs — corridors from very short segments are
                // measurement artifacts from the DFM diagonal-to-orthogonal pass, not intentional
                // routed pair runs.
                let seg_len = |t: &Track| {
                    let dx = t.end.x.0 - t.start.x.0;
                    let dy = t.end.y.0 - t.start.y.0;
                    dx.abs().max(dy.abs())
                };
                if seg_len(tp) < 1_000_000 || seg_len(tn) < 1_000_000 {
                    // < 1 mm: below any intentional diff-pair run, above 0.5 mm L-shape stub
                    continue;
                }

                if hp && hn {
                    let xp_min = tp.start.x.0.min(tp.end.x.0);
                    let xp_max = tp.start.x.0.max(tp.end.x.0);
                    let xn_min = tn.start.x.0.min(tn.end.x.0);
                    let xn_max = tn.start.x.0.max(tn.end.x.0);
                    let overlap_min = xp_min.max(xn_min);
                    let overlap_max = xp_max.min(xn_max);

                    if overlap_min < overlap_max {
                        pair_spacings.push((tp.start.y.0 - tn.start.y.0).abs() as f64);
                        spacing_hotspot = Some(Point::new(
                            Nm((overlap_min + overlap_max) / 2),
                            Nm((tp.start.y.0 + tn.start.y.0) / 2),
                        ));
                        let y_min = tp.start.y.0.min(tn.start.y.0);
                        let y_max = tp.start.y.0.max(tn.start.y.0);
                        if (y_max - y_min) as f64 <= 1.5e6 {
                            for pad in &board.pads {
                                if pad.net != Some(p_net)
                                    && pad.net != Some(n_net)
                                    && pad.pos.x.0 >= overlap_min
                                    && pad.pos.x.0 <= overlap_max
                                    && pad.pos.y.0 > y_min
                                    && pad.pos.y.0 < y_max
                                {
                                    count += 1;
                                    pts.push(pad.pos);
                                }
                            }
                            for via in &board.vias {
                                if via.net != p_net
                                    && via.net != n_net
                                    && via.pos.x.0 >= overlap_min
                                    && via.pos.x.0 <= overlap_max
                                    && via.pos.y.0 > y_min
                                    && via.pos.y.0 < y_max
                                {
                                    count += 1;
                                    pts.push(via.pos);
                                }
                            }
                            let corridor = crate::place::Rect {
                                min: Point::new(Nm(overlap_min), Nm(y_min)),
                                max: Point::new(Nm(overlap_max), Nm(y_max)),
                            };
                            for comp in comps {
                                if comp.fp >= lib.len()
                                    || comp.nets.contains(&Some(p_net))
                                    || comp.nets.contains(&Some(n_net))
                                {
                                    continue;
                                }
                                if comp.courtyard(lib).overlap_area(corridor) > 0.0 {
                                    count += 1;
                                    pts.push(comp.placement.pos);
                                }
                            }
                        }
                    }
                } else if vp && vn {
                    let yp_min = tp.start.y.0.min(tp.end.y.0);
                    let yp_max = tp.start.y.0.max(tp.end.y.0);
                    let yn_min = tn.start.y.0.min(tn.end.y.0);
                    let yn_max = tn.start.y.0.max(tn.end.y.0);
                    let overlap_min = yp_min.max(yn_min);
                    let overlap_max = yp_max.min(yn_max);

                    if overlap_min < overlap_max {
                        pair_spacings.push((tp.start.x.0 - tn.start.x.0).abs() as f64);
                        spacing_hotspot = Some(Point::new(
                            Nm((tp.start.x.0 + tn.start.x.0) / 2),
                            Nm((overlap_min + overlap_max) / 2),
                        ));
                        let x_min = tp.start.x.0.min(tn.start.x.0);
                        let x_max = tp.start.x.0.max(tn.start.x.0);
                        if (x_max - x_min) as f64 <= 1.5e6 {
                            for pad in &board.pads {
                                if pad.net != Some(p_net)
                                    && pad.net != Some(n_net)
                                    && pad.pos.y.0 >= overlap_min
                                    && pad.pos.y.0 <= overlap_max
                                    && pad.pos.x.0 > x_min
                                    && pad.pos.x.0 < x_max
                                {
                                    count += 1;
                                    pts.push(pad.pos);
                                }
                            }
                            for via in &board.vias {
                                if via.net != p_net
                                    && via.net != n_net
                                    && via.pos.y.0 >= overlap_min
                                    && via.pos.y.0 <= overlap_max
                                    && via.pos.x.0 > x_min
                                    && via.pos.x.0 < x_max
                                {
                                    count += 1;
                                    pts.push(via.pos);
                                }
                            }
                            let corridor = crate::place::Rect {
                                min: Point::new(Nm(x_min), Nm(overlap_min)),
                                max: Point::new(Nm(x_max), Nm(overlap_max)),
                            };
                            for comp in comps {
                                if comp.fp >= lib.len()
                                    || comp.nets.contains(&Some(p_net))
                                    || comp.nets.contains(&Some(n_net))
                                {
                                    continue;
                                }
                                if comp.courtyard(lib).overlap_area(corridor) > 0.0 {
                                    count += 1;
                                    pts.push(comp.placement.pos);
                                }
                            }
                        }
                    }
                }
            }
        }
        if pair_spacings.len() >= 2 {
            let min_spacing = pair_spacings.iter().copied().fold(f64::INFINITY, f64::min);
            let max_spacing = pair_spacings
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            if max_spacing - min_spacing > rules.diff_pair_spacing_tolerance.0 as f64 {
                spacing_variation += 1;
                if let Some(point) = spacing_hotspot {
                    pts.push(point);
                }
            }
        }
    }

    let is_diff_net = |n: Option<NetId>| {
        if let Some(id) = n {
            pairs.iter().any(|&(p, n)| id == p || id == n)
        } else {
            false
        }
    };

    for c in comps {
        if c.refdes.starts_with('C') || c.refdes.starts_with("C_") {
            let has_diff_net = c.nets.iter().any(|&n| is_diff_net(n));
            if has_diff_net && c.fp < lib.len() {
                let name = lib[c.fp].name.to_uppercase();
                if name.contains("0805")
                    || name.contains("1206")
                    || name.contains("1210")
                    || name.contains("1812")
                    || name.contains("2010")
                    || name.contains("2512")
                {
                    count += 1;
                    pts.push(c.placement.pos);
                }
            }
        }
    }

    (
        count,
        layer_mismatch,
        via_count_mismatch,
        length_mismatch,
        segment_length_mismatch,
        spacing_variation,
        via_symmetry,
        total_mismatch_mm,
        pts,
    )
}

pub(crate) fn detect_high_speed_edge_violations(board: &Board, rules: &DesignRules) -> (usize, Vec<Point>) {
    let is_high_speed = |net: NetId| {
        let name = &board.nets[net.0 as usize].name;
        board.class_of(net) == NetClassKind::Hv
            || name.starts_with("TRIG")
            || name.starts_with("OUT")
            || name.starts_with("TX")
    };
    let mut count = 0;
    let mut pts = Vec::new();
    let margin = rules.high_speed_edge_clearance.0;
    let limit_x = board.spec.origin.x.0 + board.spec.pitch.0 * (board.spec.nx as i64 - 1) - margin;
    let limit_y = board.spec.origin.y.0 + board.spec.pitch.0 * (board.spec.ny as i64 - 1) - margin;

    for t in &board.tracks {
        if is_high_speed(t.net) {
            let mut bad = false;
            for pt in [t.start, t.end] {
                if pt.x.0 < margin || pt.x.0 > limit_x || pt.y.0 < margin || pt.y.0 > limit_y {
                    bad = true;
                }
            }
            if bad {
                count += 1;
                pts.push(Point::new(
                    Nm((t.start.x.0 + t.end.x.0) / 2),
                    Nm((t.start.y.0 + t.end.y.0) / 2),
                ));
            }
        }
    }
    (count, pts)
}

pub(crate) fn detect_high_speed_component_edge_violations(
    board: &Board,
    comps: &[Component],
    lib: &[FootprintDef],
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    let spec = board.spec;
    let min_x = spec.origin.x.0 + rules.high_speed_component_edge_clearance.0;
    let min_y = spec.origin.y.0 + rules.high_speed_component_edge_clearance.0;
    let max_x = spec.origin.x.0 + (spec.nx as i64 - 1) * spec.pitch.0
        - rules.high_speed_component_edge_clearance.0;
    let max_y = spec.origin.y.0 + (spec.ny as i64 - 1) * spec.pitch.0
        - rules.high_speed_component_edge_clearance.0;
    let mut count = 0;
    let mut pts = Vec::new();

    for comp in comps {
        if comp.fp >= lib.len() || !matches!(lib[comp.fp].role, Role::ActiveIc) {
            continue;
        }
        if !comp
            .nets
            .iter()
            .flatten()
            .any(|&n| is_high_speed_net(board, n))
        {
            continue;
        }

        let courtyard = comp.courtyard(lib);
        let shortfall = [
            min_x - courtyard.min.x.0,
            min_y - courtyard.min.y.0,
            courtyard.max.x.0 - max_x,
            courtyard.max.y.0 - max_y,
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        if shortfall > 0 {
            count += 1;
            pts.push(comp.placement.pos);
        }
    }

    (count, pts)
}

pub(crate) fn detect_high_speed_termination_placement_violations(
    board: &Board,
    comps: &[Component],
    lib: &[FootprintDef],
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    let mut count = 0;
    let mut pts = Vec::new();
    let max_dist = rules.high_speed_termination_distance.0 as f64;

    for terminator in comps {
        if terminator.fp >= lib.len()
            || !matches!(lib[terminator.fp].role, Role::Passive)
            || !terminator.refdes.starts_with('R')
        {
            continue;
        }

        let high_speed_pads: Vec<(Point, NetId)> = terminator
            .placed_pads(lib)
            .filter_map(|(pos, _layers, net)| net.map(|n| (pos, n)))
            .filter(|&(_pos, net)| is_high_speed_net(board, net))
            .collect();
        if high_speed_pads.is_empty() {
            continue;
        }

        let mut has_near_active_pad = false;
        for &(term_pos, term_net) in &high_speed_pads {
            for active in comps {
                if active.fp >= lib.len() || !matches!(lib[active.fp].role, Role::ActiveIc) {
                    continue;
                }
                for (active_pos, _layers, active_net) in active.placed_pads(lib) {
                    if active_net == Some(term_net) && term_pos.euclid(active_pos) <= max_dist {
                        has_near_active_pad = true;
                        break;
                    }
                }
                if has_near_active_pad {
                    break;
                }
            }
            if has_near_active_pad {
                break;
            }
        }

        if !has_near_active_pad {
            count += 1;
            pts.push(terminator.placement.pos);
        }
    }

    (count, pts)
}

pub(crate) fn is_high_speed_net(board: &Board, net: NetId) -> bool {
    let name = &board.nets[net.0 as usize].name;
    board.class_of(net) == NetClassKind::Hv
        || name.starts_with("TRIG")
        || name.starts_with("OUT")
        || name.starts_with("TX")
}

pub(crate) fn is_clock_like_net(board: &Board, net: NetId) -> bool {
    let upper = board.nets[net.0 as usize].name.to_ascii_uppercase();
    upper.starts_with("CLK") || upper.contains("_CLK") || upper.contains("CLOCK")
}

pub(crate) fn diff_pair_prefix(name: &str) -> Option<&str> {
    name.strip_suffix("_P")
        .or_else(|| name.strip_suffix("_N"))
        .or_else(|| name.strip_suffix('+'))
        .or_else(|| name.strip_suffix('-'))
        .or_else(|| name.strip_suffix("_pos"))
        .or_else(|| name.strip_suffix("_neg"))
}

pub(crate) fn are_diff_pair_mates(board: &Board, a: NetId, b: NetId) -> bool {
    let an = &board.nets[a.0 as usize].name;
    let bn = &board.nets[b.0 as usize].name;
    let Some(ap) = diff_pair_prefix(an) else {
        return false;
    };
    let Some(bp) = diff_pair_prefix(bn) else {
        return false;
    };
    ap == bp && an != bn
}

pub(crate) fn diff_pair_members(board: &Board) -> Vec<(NetId, NetId)> {
    let mut p_nets = Vec::new();
    let mut n_nets = Vec::new();
    for net in &board.nets {
        let name = &net.name;
        if name.ends_with("_P") || name.ends_with('+') || name.ends_with("_pos") {
            p_nets.push(net);
        } else if name.ends_with("_N") || name.ends_with('-') || name.ends_with("_neg") {
            n_nets.push(net);
        }
    }

    let mut pairs = Vec::new();
    for p in &p_nets {
        let Some(prefix) = diff_pair_prefix(&p.name) else {
            continue;
        };
        for n in &n_nets {
            if diff_pair_prefix(&n.name) == Some(prefix) {
                pairs.push((p.id, n.id));
                break;
            }
        }
    }
    pairs
}

pub(crate) fn diff_pair_interface_group(prefix: &str) -> Option<String> {
    let upper = prefix.to_ascii_uppercase();
    let trimmed = upper.trim_end_matches(|c: char| c.is_ascii_digit());
    if trimmed.len() == upper.len() {
        return None;
    }
    let group = trimmed.trim_end_matches(['_', '-', '[']);
    (!group.is_empty()).then(|| group.to_string())
}

pub(crate) fn detect_diff_pair_interface_mismatch_violations(board: &Board) -> (usize, usize, Vec<Point>) {
    let mut groups: BTreeMap<String, Vec<(BTreeSet<LayerId>, usize)>> = BTreeMap::new();
    let mut group_points: BTreeMap<String, Vec<Point>> = BTreeMap::new();
    for (p_net, n_net) in diff_pair_members(board) {
        let name = &board.nets[p_net.0 as usize].name;
        let Some(prefix) = diff_pair_prefix(name) else {
            continue;
        };
        let Some(group) = diff_pair_interface_group(prefix) else {
            continue;
        };
        let layers: BTreeSet<LayerId> = board
            .tracks
            .iter()
            .filter(|track| track.net == p_net || track.net == n_net)
            .map(|track| track.layer)
            .collect();
        if layers.is_empty() {
            continue;
        }
        let via_count = board
            .vias
            .iter()
            .filter(|via| via.net == p_net || via.net == n_net)
            .count();
        let point = board
            .tracks
            .iter()
            .find(|track| track.net == p_net || track.net == n_net)
            .map(track_midpoint);
        groups
            .entry(group.clone())
            .or_default()
            .push((layers, via_count));
        if let Some(point) = point {
            group_points.entry(group).or_default().push(point);
        }
    }

    let mut layer_count = 0;
    let mut via_count = 0;
    let mut pts = Vec::new();
    for (group, entries) in groups {
        if entries.len() < 2 {
            continue;
        }
        let Some((reference_layers, reference_vias)) = entries.first() else {
            continue;
        };
        let layer_mismatch = entries.iter().any(|(layers, _)| layers != reference_layers);
        let via_mismatch = entries.iter().any(|(_, vias)| vias != reference_vias);
        if layer_mismatch {
            layer_count += 1;
        }
        if via_mismatch {
            via_count += 1;
        }
        if layer_mismatch || via_mismatch {
            if let Some(points) = group_points.get(&group) {
                if let Some(&point) = points.first() {
                    pts.push(point);
                }
            }
        }
    }

    (layer_count, via_count, pts)
}

pub(crate) fn diff_pair_member_to_pair(pairs: &[(NetId, NetId)]) -> HashMap<NetId, usize> {
    let mut out = HashMap::new();
    for (idx, &(p, n)) in pairs.iter().enumerate() {
        out.insert(p, idx);
        out.insert(n, idx);
    }
    out
}

pub(crate) fn reference_zones(board: &Board) -> Vec<&crate::board::Zone> {
    board
        .zones
        .iter()
        .filter(|z| {
            let class = board.class_of(z.net);
            matches!(class, NetClassKind::Ground | NetClassKind::Power)
        })
        .collect()
}

pub(crate) fn detect_high_speed_parallel_spacing_violations(
    board: &Board,
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    let mut count = 0;
    let mut pts = Vec::new();
    for i in 0..board.tracks.len() {
        let a = &board.tracks[i];
        if !is_high_speed_net(board, a.net) {
            continue;
        }
        let a_h = a.start.y.0 == a.end.y.0;
        let a_v = a.start.x.0 == a.end.x.0;
        if !a_h && !a_v {
            continue;
        }
        for b in board.tracks.iter().skip(i + 1) {
            if a.net == b.net
                || a.layer != b.layer
                || !is_high_speed_net(board, b.net)
                || are_diff_pair_mates(board, a.net, b.net)
            {
                continue;
            }
            let b_h = b.start.y.0 == b.end.y.0;
            let b_v = b.start.x.0 == b.end.x.0;
            let (center_gap, overlap, mid) = if a_h && b_h {
                let ax0 = a.start.x.0.min(a.end.x.0);
                let ax1 = a.start.x.0.max(a.end.x.0);
                let bx0 = b.start.x.0.min(b.end.x.0);
                let bx1 = b.start.x.0.max(b.end.x.0);
                let ov0 = ax0.max(bx0);
                let ov1 = ax1.min(bx1);
                if ov1 <= ov0 {
                    continue;
                }
                (
                    (a.start.y.0 - b.start.y.0).abs() as f64,
                    (ov1 - ov0) as f64,
                    Point::new(Nm((ov0 + ov1) / 2), Nm((a.start.y.0 + b.start.y.0) / 2)),
                )
            } else if a_v && b_v {
                let ay0 = a.start.y.0.min(a.end.y.0);
                let ay1 = a.start.y.0.max(a.end.y.0);
                let by0 = b.start.y.0.min(b.end.y.0);
                let by1 = b.start.y.0.max(b.end.y.0);
                let ov0 = ay0.max(by0);
                let ov1 = ay1.min(by1);
                if ov1 <= ov0 {
                    continue;
                }
                (
                    (a.start.x.0 - b.start.x.0).abs() as f64,
                    (ov1 - ov0) as f64,
                    Point::new(Nm((a.start.x.0 + b.start.x.0) / 2), Nm((ov0 + ov1) / 2)),
                )
            } else {
                continue;
            };
            let edge_gap = center_gap - (a.width.0 as f64 + b.width.0 as f64) / 2.0;
            let width_required =
                rules.high_speed_parallel_spacing_widths * a.width.0.max(b.width.0) as f64;
            let required = if is_clock_like_net(board, a.net) || is_clock_like_net(board, b.net) {
                width_required.max(rules.high_speed_clock_parallel_keepout.0 as f64)
            } else {
                width_required
            };
            if overlap >= rules.high_speed_parallel_coupling_length.0 as f64 && edge_gap < required
            {
                count += 1;
                pts.push(mid);
            }
        }
    }
    (count, pts)
}

pub(crate) fn detect_high_speed_adjacent_layer_parallel_violations(
    board: &Board,
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    let mut count = 0;
    let mut pts = Vec::new();
    for i in 0..board.tracks.len() {
        let a = &board.tracks[i];
        if !is_high_speed_net(board, a.net) {
            continue;
        }
        let a_h = a.start.y.0 == a.end.y.0;
        let a_v = a.start.x.0 == a.end.x.0;
        if !a_h && !a_v {
            continue;
        }
        for b in board.tracks.iter().skip(i + 1) {
            if a.net == b.net
                || a.layer.0.abs_diff(b.layer.0) != 1
                || !is_high_speed_net(board, b.net)
                || are_diff_pair_mates(board, a.net, b.net)
            {
                continue;
            }
            let b_h = b.start.y.0 == b.end.y.0;
            let b_v = b.start.x.0 == b.end.x.0;
            let (center_gap, overlap, mid) = if a_h && b_h {
                let ax0 = a.start.x.0.min(a.end.x.0);
                let ax1 = a.start.x.0.max(a.end.x.0);
                let bx0 = b.start.x.0.min(b.end.x.0);
                let bx1 = b.start.x.0.max(b.end.x.0);
                let ov0 = ax0.max(bx0);
                let ov1 = ax1.min(bx1);
                if ov1 <= ov0 {
                    continue;
                }
                (
                    (a.start.y.0 - b.start.y.0).abs() as f64,
                    (ov1 - ov0) as f64,
                    Point::new(Nm((ov0 + ov1) / 2), Nm((a.start.y.0 + b.start.y.0) / 2)),
                )
            } else if a_v && b_v {
                let ay0 = a.start.y.0.min(a.end.y.0);
                let ay1 = a.start.y.0.max(a.end.y.0);
                let by0 = b.start.y.0.min(b.end.y.0);
                let by1 = b.start.y.0.max(b.end.y.0);
                let ov0 = ay0.max(by0);
                let ov1 = ay1.min(by1);
                if ov1 <= ov0 {
                    continue;
                }
                (
                    (a.start.x.0 - b.start.x.0).abs() as f64,
                    (ov1 - ov0) as f64,
                    Point::new(Nm((a.start.x.0 + b.start.x.0) / 2), Nm((ov0 + ov1) / 2)),
                )
            } else {
                continue;
            };
            let required =
                rules.high_speed_parallel_spacing_widths * a.width.0.max(b.width.0) as f64;
            let edge_gap = center_gap - (a.width.0 as f64 + b.width.0 as f64) / 2.0;
            if overlap >= rules.high_speed_parallel_coupling_length.0 as f64 && edge_gap < required
            {
                count += 1;
                pts.push(mid);
            }
        }
    }
    (count, pts)
}

pub(crate) fn detect_diff_pair_keepout_violations(board: &Board, rules: &DesignRules) -> (usize, Vec<Point>) {
    let pairs = diff_pair_members(board);
    let member_pair = diff_pair_member_to_pair(&pairs);
    let mut violations = BTreeSet::new();
    let mut pts = Vec::new();

    for (i, a) in board.tracks.iter().enumerate() {
        let Some(&pair_idx) = member_pair.get(&a.net) else {
            continue;
        };
        let pair_name = diff_pair_prefix(&board.nets[a.net.0 as usize].name)
            .unwrap_or(&board.nets[a.net.0 as usize].name)
            .to_ascii_uppercase();
        let is_clock_pair = pair_name.contains("CLK") || pair_name.contains("CLOCK");

        for (j, b) in board.tracks.iter().enumerate() {
            if i == j || a.layer != b.layer {
                continue;
            }
            if member_pair.get(&b.net).copied() == Some(pair_idx) {
                continue;
            }

            let other_pair = member_pair.get(&b.net).copied();
            let (key, required) = if let Some(other_idx) = other_pair {
                let lo = pair_idx.min(other_idx);
                let hi = pair_idx.max(other_idx);
                (
                    (lo, hi),
                    rules.diff_pair_pair_spacing_widths * a.width.0.max(b.width.0) as f64,
                )
            } else {
                (
                    (pair_idx, pairs.len() + b.net.0 as usize),
                    if is_clock_pair {
                        rules.diff_pair_clock_keepout.0 as f64
                    } else {
                        rules.diff_pair_signal_keepout.0 as f64
                    },
                )
            };

            if violations.contains(&key) {
                continue;
            }
            let edge_gap = crate::geom::dist_seg_seg(a.start, a.end, b.start, b.end)
                - (a.width.0 as f64 + b.width.0 as f64) / 2.0;
            if edge_gap < required {
                violations.insert(key);
                pts.push(Point::new(
                    Nm((a.start.x.0 + a.end.x.0 + b.start.x.0 + b.end.x.0) / 4),
                    Nm((a.start.y.0 + a.end.y.0 + b.start.y.0 + b.end.y.0) / 4),
                ));
            }
        }
    }

    (violations.len(), pts)
}

pub(crate) fn detect_reference_plane_margin_violations(
    board: &Board,
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    let zones = reference_zones(board);
    let mut count = 0;
    let mut pts = Vec::new();

    for t in &board.tracks {
        if !is_high_speed_net(board, t.net) {
            continue;
        }
        let mid = Point::new(
            Nm((t.start.x.0 + t.end.x.0) / 2),
            Nm((t.start.y.0 + t.end.y.0) / 2),
        );
        let required = rules.high_speed_reference_plane_margin_widths * t.width.0 as f64;
        let mut margin = None;
        for zone in zones.iter().filter(|z| z.layer == t.layer) {
            if point_in_polygon(mid, &zone.polygon) {
                margin = distance_to_polygon_boundary(mid, &zone.polygon);
                break;
            }
        }
        if margin.is_some_and(|m| m < required) {
            count += 1;
            pts.push(mid);
        }
    }
    (count, pts)
}

pub(crate) fn detect_reference_plane_absence_violations(board: &Board) -> (usize, Vec<Point>) {
    let zones = reference_zones(board);
    let mut count = 0;
    let mut pts = Vec::new();

    for track in &board.tracks {
        if !is_high_speed_net(board, track.net) {
            continue;
        }
        let mid = Point::new(
            Nm((track.start.x.0 + track.end.x.0) / 2),
            Nm((track.start.y.0 + track.end.y.0) / 2),
        );
        let samples = [track.start, mid, track.end];
        let has_adjacent_reference = zones.iter().any(|zone| {
            let dz = zone.layer.0.abs_diff(track.layer.0);
            dz == 1 && samples.iter().all(|&p| point_in_polygon(p, &zone.polygon))
        });
        if !has_adjacent_reference {
            count += 1;
            pts.push(mid);
        }
    }

    (count, pts)
}

pub(crate) fn detect_inner_layer_dual_ground_reference_violations(board: &Board) -> (usize, Vec<Point>) {
    let zones = reference_zones(board);
    let mut count = 0;
    let mut pts = Vec::new();

    for track in &board.tracks {
        if !is_high_speed_net(board, track.net) {
            continue;
        }
        let layer = track.layer.0 as usize;
        if layer == 0 || layer + 1 >= board.spec.nlayers {
            continue;
        }
        let mid = track_midpoint(track);
        let samples = [track.start, mid, track.end];
        let has_ground_on = |adjacent: LayerId| {
            zones.iter().any(|zone| {
                zone.layer == adjacent
                    && board.class_of(zone.net) == NetClassKind::Ground
                    && samples.iter().all(|&p| point_in_polygon(p, &zone.polygon))
            })
        };
        let lower = LayerId(track.layer.0 - 1);
        let upper = LayerId(track.layer.0 + 1);
        if !has_ground_on(lower) || !has_ground_on(upper) {
            count += 1;
            pts.push(mid);
        }
    }

    (count, pts)
}

pub(crate) fn detect_power_reference_stitching_cap_violations(
    board: &Board,
    comps: &[Component],
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    let zones = reference_zones(board);
    let max_dist = rules.power_reference_stitching_cap_distance.0 as f64;
    let mut count = 0;
    let mut pts = Vec::new();

    for track in &board.tracks {
        if !is_high_speed_net(board, track.net) {
            continue;
        }
        let mid = Point::new(
            Nm((track.start.x.0 + track.end.x.0) / 2),
            Nm((track.start.y.0 + track.end.y.0) / 2),
        );
        let samples = [track.start, mid, track.end];
        let has_ground_reference = zones.iter().any(|zone| {
            board.class_of(zone.net) == NetClassKind::Ground
                && zone.layer.0.abs_diff(track.layer.0) == 1
                && samples.iter().all(|&p| point_in_polygon(p, &zone.polygon))
        });
        if has_ground_reference {
            continue;
        }
        let power_reference = zones.iter().find(|zone| {
            board.class_of(zone.net) == NetClassKind::Power
                && zone.layer.0.abs_diff(track.layer.0) == 1
                && samples.iter().all(|&p| point_in_polygon(p, &zone.polygon))
        });
        let Some(power_zone) = power_reference else {
            continue;
        };
        let has_stitching_at = |point: Point| {
            comps.iter().any(|c| {
                c.refdes.starts_with('C')
                    && c.placement.pos.euclid(point) <= max_dist
                    && c.nets.contains(&Some(power_zone.net))
                    && c.nets
                        .iter()
                        .flatten()
                        .any(|&net| board.class_of(net) == NetClassKind::Ground)
            })
        };
        if !has_stitching_at(track.start) || !has_stitching_at(track.end) {
            count += 1;
            pts.push(mid);
        }
    }

    (count, pts)
}

pub(crate) fn power_reference_zone_for_track<'a>(
    board: &Board,
    zones: &'a [&'a crate::board::Zone],
    track: &Track,
) -> Option<&'a crate::board::Zone> {
    let mid = track_midpoint(track);
    let samples = [track.start, mid, track.end];
    let has_ground_reference = zones.iter().any(|zone| {
        board.class_of(zone.net) == NetClassKind::Ground
            && zone.layer.0.abs_diff(track.layer.0) == 1
            && samples.iter().all(|&p| point_in_polygon(p, &zone.polygon))
    });
    if has_ground_reference {
        return None;
    }
    zones.iter().copied().find(|zone| {
        board.class_of(zone.net) == NetClassKind::Power
            && zone.layer.0.abs_diff(track.layer.0) == 1
            && samples.iter().all(|&p| point_in_polygon(p, &zone.polygon))
    })
}

pub(crate) fn detect_diff_pair_stitching_cap_symmetry_violations(
    board: &Board,
    comps: &[Component],
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    let zones = reference_zones(board);
    let max_dist = rules.power_reference_stitching_cap_distance.0 as f64;
    let tolerance = rules.diff_pair_coupling_cap_symmetry_tolerance.0;
    let mut count = 0;
    let mut pts = Vec::new();

    for (p_net, n_net) in diff_pair_members(board) {
        let Some(axis) = diff_pair_axis(board, p_net, n_net) else {
            continue;
        };
        let stitching_stations = |net| {
            let mut stations = Vec::new();
            for track in board.tracks.iter().filter(|t| t.net == net) {
                let Some(power_zone) = power_reference_zone_for_track(board, &zones, track) else {
                    continue;
                };
                for point in [track.start, track.end] {
                    if let Some(cap) = comps
                        .iter()
                        .filter(|c| {
                            c.refdes.starts_with('C')
                                && c.placement.pos.euclid(point) <= max_dist
                                && c.nets.contains(&Some(power_zone.net))
                                && c.nets
                                    .iter()
                                    .flatten()
                                    .any(|&net| board.class_of(net) == NetClassKind::Ground)
                        })
                        .min_by(|a, b| {
                            a.placement
                                .pos
                                .euclid(point)
                                .total_cmp(&b.placement.pos.euclid(point))
                        })
                    {
                        stations.push(match axis {
                            PairAxis::X => cap.placement.pos.x.0,
                            PairAxis::Y => cap.placement.pos.y.0,
                        });
                    }
                }
            }
            stations.sort_unstable();
            stations.dedup();
            stations
        };

        let p_stations = stitching_stations(p_net);
        let n_stations = stitching_stations(n_net);
        if p_stations.is_empty() && n_stations.is_empty() {
            continue;
        }
        if p_stations.len() != n_stations.len() {
            count += 1;
            if let Some(track) = board
                .tracks
                .iter()
                .find(|track| track.net == p_net || track.net == n_net)
            {
                pts.push(track_midpoint(track));
            }
            continue;
        }
        let worst = p_stations
            .iter()
            .zip(&n_stations)
            .map(|(&p, &n)| (p - n).abs())
            .max()
            .unwrap_or(0);
        if worst > tolerance {
            count += 1;
            if let Some(track) = board
                .tracks
                .iter()
                .find(|track| track.net == p_net || track.net == n_net)
            {
                pts.push(track_midpoint(track));
            }
        }
    }

    (count, pts)
}

pub(crate) fn detect_reference_plane_intrusion_violations(board: &Board) -> (usize, Vec<Point>) {
    let zones = reference_zones(board);
    let mut count = 0;
    let mut pts = Vec::new();

    for track in &board.tracks {
        if matches!(
            board.class_of(track.net),
            NetClassKind::Ground | NetClassKind::Power
        ) {
            continue;
        }
        let mid = track_midpoint(track);
        if zones
            .iter()
            .filter(|zone| zone.layer == track.layer)
            .any(|zone| point_in_polygon(mid, &zone.polygon))
        {
            count += 1;
            pts.push(mid);
        }
    }

    (count, pts)
}

pub(crate) fn detect_ground_plane_fragmentation_violations(board: &Board) -> (usize, Vec<Point>) {
    let mut by_layer_net: BTreeMap<(LayerId, NetId), Vec<&crate::board::Zone>> = BTreeMap::new();
    for zone in &board.zones {
        if zone.fill == ZoneFill::ThermalRelief && board.class_of(zone.net) == NetClassKind::Ground
        {
            by_layer_net
                .entry((zone.layer, zone.net))
                .or_default()
                .push(zone);
        }
    }

    let mut count = 0;
    let mut pts = Vec::new();
    for ((_layer, _net), zones) in by_layer_net {
        if zones.len() <= 1 {
            continue;
        }
        count += 1;
        pts.extend(
            zones
                .iter()
                .skip(1)
                .filter_map(|zone| polygon_vertex_mean(&zone.polygon)),
        );
    }

    (count, pts)
}

pub(crate) fn detect_split_domain_reference_violations(board: &Board) -> (usize, Vec<Point>) {
    let zones = reference_zones(board);
    let mut count = 0;
    let mut pts = Vec::new();

    for track in &board.tracks {
        if matches!(
            board.class_of(track.net),
            NetClassKind::Ground | NetClassKind::Power
        ) {
            continue;
        }
        let signal_name = &board.nets[track.net.0 as usize].name;
        let Some(signal_domain) = split_domain_from_name(signal_name) else {
            continue;
        };
        let mid = track_midpoint(track);
        for zone in zones.iter().filter(|zone| zone.layer == track.layer) {
            let reference_name = &board.nets[zone.net.0 as usize].name;
            let Some(reference_domain) = split_domain_from_name(reference_name) else {
                continue;
            };
            if signal_domain != reference_domain && point_in_polygon(mid, &zone.polygon) {
                count += 1;
                pts.push(mid);
                break;
            }
        }
    }

    (count, pts)
}

pub(crate) fn adjacent_ground_reference_zone_indices(board: &Board, track: &Track) -> Vec<usize> {
    let zones = reference_zones(board);
    let mid = track_midpoint(track);
    let samples = [track.start, mid, track.end];
    zones
        .iter()
        .enumerate()
        .filter_map(|(idx, zone)| {
            (board.class_of(zone.net) == NetClassKind::Ground
                && zone.layer.0.abs_diff(track.layer.0) == 1
                && samples.iter().all(|&p| point_in_polygon(p, &zone.polygon)))
            .then_some(idx)
        })
        .collect()
}

pub(crate) fn detect_mixed_domain_shared_reference_violations(
    board: &Board,
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    let mut domain_tracks = Vec::new();
    for track in &board.tracks {
        if board.class_of(track.net) != NetClassKind::Signal {
            continue;
        }
        let signal_name = &board.nets[track.net.0 as usize].name;
        let Some(domain) = split_domain_from_name(signal_name) else {
            continue;
        };
        let references = adjacent_ground_reference_zone_indices(board, track);
        if !references.is_empty() {
            domain_tracks.push((track, domain, references));
        }
    }

    let mut count = 0;
    let mut pts = Vec::new();
    let max_gap = rules.diff_pair_clock_keepout.0 as f64;
    for i in 0..domain_tracks.len() {
        let (a, a_domain, a_refs) = &domain_tracks[i];
        for (b, b_domain, b_refs) in domain_tracks.iter().skip(i + 1) {
            if a_domain == b_domain || !a_refs.iter().any(|r| b_refs.contains(r)) {
                continue;
            }
            let copper_gap =
                dist_seg_seg(a.start, a.end, b.start, b.end) - (a.width.0 + b.width.0) as f64 / 2.0;
            if copper_gap <= max_gap {
                let am = track_midpoint(a);
                let bm = track_midpoint(b);
                count += 1;
                pts.push(Point::new(
                    Nm((am.x.0 + bm.x.0) / 2),
                    Nm((am.y.0 + bm.y.0) / 2),
                ));
            }
        }
    }

    (count, pts)
}

pub(crate) fn detect_virtual_split_crossing_violations(board: &Board) -> (usize, Vec<Point>) {
    let mut analog_sum = (0_i64, 0_i64, 0_i64);
    let mut digital_sum = (0_i64, 0_i64, 0_i64);
    for pad in &board.pads {
        let Some(net) = pad.net else { continue };
        match split_domain_from_name(&board.nets[net.0 as usize].name) {
            Some(SplitDomain::Analog) => {
                analog_sum.0 += pad.pos.x.0;
                analog_sum.1 += pad.pos.y.0;
                analog_sum.2 += 1;
            }
            Some(SplitDomain::Digital) => {
                digital_sum.0 += pad.pos.x.0;
                digital_sum.1 += pad.pos.y.0;
                digital_sum.2 += 1;
            }
            None => {}
        }
    }
    if analog_sum.2 == 0 || digital_sum.2 == 0 {
        return (0, Vec::new());
    }

    let analog = Point::new(
        Nm(analog_sum.0 / analog_sum.2),
        Nm(analog_sum.1 / analog_sum.2),
    );
    let digital = Point::new(
        Nm(digital_sum.0 / digital_sum.2),
        Nm(digital_sum.1 / digital_sum.2),
    );
    let use_x_axis = (analog.x.0 - digital.x.0).abs() >= (analog.y.0 - digital.y.0).abs();
    let split = if use_x_axis {
        (analog.x.0 + digital.x.0) / 2
    } else {
        (analog.y.0 + digital.y.0) / 2
    };
    let analog_less = if use_x_axis {
        analog.x.0 < split
    } else {
        analog.y.0 < split
    };

    let mut count = 0;
    let mut pts = Vec::new();
    for track in &board.tracks {
        if board.class_of(track.net) != NetClassKind::Signal {
            continue;
        }
        let Some(domain) = split_domain_from_name(&board.nets[track.net.0 as usize].name) else {
            continue;
        };
        let expected_less = match domain {
            SplitDomain::Analog => analog_less,
            SplitDomain::Digital => !analog_less,
        };
        let a = if use_x_axis {
            track.start.x.0
        } else {
            track.start.y.0
        };
        let b = if use_x_axis {
            track.end.x.0
        } else {
            track.end.y.0
        };
        let start_ok = if expected_less { a < split } else { a > split };
        let end_ok = if expected_less { b < split } else { b > split };
        if !(start_ok && end_ok) {
            count += 1;
            pts.push(track_midpoint(track));
        }
    }

    (count, pts)
}

pub(crate) fn detect_high_speed_stub_violations(board: &Board) -> (usize, Vec<Point>) {
    type TrackNodeKey = (NetId, LayerId, i64, i64);
    type TrackDirection = (i64, i64);

    let mut by_node: HashMap<TrackNodeKey, Vec<TrackDirection>> = HashMap::new();
    for t in &board.tracks {
        if !is_high_speed_net(board, t.net) {
            continue;
        }
        let endpoints = [(t.start, t.end), (t.end, t.start)];
        for (node, other) in endpoints {
            by_node
                .entry((t.net, t.layer, node.x.0, node.y.0))
                .or_default()
                .push((other.x.0 - node.x.0, other.y.0 - node.y.0));
        }
    }

    let mut count = 0;
    let mut pts = Vec::new();
    for ((_net, _layer, x, y), dirs) in by_node {
        if dirs.len() < 3 {
            continue;
        }
        count += 1;
        pts.push(Point::new(Nm(x), Nm(y)));
    }
    (count, pts)
}

pub(crate) fn detect_high_speed_transition_ground_via_violations(
    board: &Board,
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    let mut count = 0;
    let mut pts = Vec::new();
    let ground_vias: Vec<_> = board
        .vias
        .iter()
        .filter(|v| board.class_of(v.net) == NetClassKind::Ground)
        .collect();
    let max_dist = rules.high_speed_transition_ground_via_distance.0 as f64;

    for via in &board.vias {
        if via.from == via.to || !is_high_speed_net(board, via.net) {
            continue;
        }
        let has_ground_transition = ground_vias
            .iter()
            .any(|g| g.from != g.to && via.pos.euclid(g.pos) <= max_dist);
        if !has_ground_transition {
            count += 1;
            pts.push(via.pos);
        }
    }
    (count, pts)
}

pub(crate) fn detect_diff_pair_transition_ground_via_symmetry_violations(
    board: &Board,
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    let ground_vias: Vec<_> = board
        .vias
        .iter()
        .filter(|v| board.class_of(v.net) == NetClassKind::Ground && v.from != v.to)
        .collect();
    let max_dist = rules.high_speed_transition_ground_via_distance.0 as f64;
    let tolerance = rules.diff_pair_via_symmetry_tolerance.0;
    let mut count = 0;
    let mut pts = Vec::new();

    for (p_net, n_net) in diff_pair_members(board) {
        let Some(axis) = diff_pair_axis(board, p_net, n_net) else {
            continue;
        };
        let transition_ground_stations = |net| {
            let mut stations = Vec::new();
            for via in board.vias.iter().filter(|v| v.net == net && v.from != v.to) {
                if let Some(ground) = ground_vias
                    .iter()
                    .min_by(|a, b| via.pos.euclid(a.pos).total_cmp(&via.pos.euclid(b.pos)))
                {
                    if via.pos.euclid(ground.pos) <= max_dist {
                        stations.push(match axis {
                            PairAxis::X => ground.pos.x.0,
                            PairAxis::Y => ground.pos.y.0,
                        });
                    }
                }
            }
            stations.sort_unstable();
            stations
        };

        let p_stations = transition_ground_stations(p_net);
        let n_stations = transition_ground_stations(n_net);
        if p_stations.is_empty() && n_stations.is_empty() {
            continue;
        }
        if p_stations.len() != n_stations.len() {
            count += 1;
            if let Some(via) = board.vias.iter().find(|v| v.net == p_net || v.net == n_net) {
                pts.push(via.pos);
            }
            continue;
        }
        let worst = p_stations
            .iter()
            .zip(&n_stations)
            .map(|(&p, &n)| (p - n).abs())
            .max()
            .unwrap_or(0);
        if worst > tolerance {
            count += 1;
            if let Some(via) = board.vias.iter().find(|v| v.net == p_net || v.net == n_net) {
                pts.push(via.pos);
            }
        }
    }

    (count, pts)
}

pub(crate) fn detect_high_speed_terminal_ground_via_violations(
    board: &Board,
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    let mut count = 0;
    let mut pts = Vec::new();
    let max_dist = rules.high_speed_terminal_ground_via_distance.0 as f64;
    let ground_features: Vec<Point> = board
        .vias
        .iter()
        .filter(|v| board.class_of(v.net) == NetClassKind::Ground)
        .map(|v| v.pos)
        .chain(
            board
                .pads
                .iter()
                .filter(|p| {
                    p.net
                        .is_some_and(|n| board.class_of(n) == NetClassKind::Ground)
                })
                .map(|p| p.pos),
        )
        .collect();

    for pad in &board.pads {
        let Some(net) = pad.net else { continue };
        if !is_high_speed_net(board, net) {
            continue;
        }
        let has_local_return = ground_features
            .iter()
            .any(|&g| pad.pos.euclid(g) <= max_dist);
        if !has_local_return {
            count += 1;
            pts.push(pad.pos);
        }
    }

    (count, pts)
}

pub(crate) fn detect_high_speed_via_pad_proximity_violations(
    board: &Board,
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    let max_dist = rules.high_speed_via_pad_distance.0 as f64;
    let mut count = 0;
    let mut pts = Vec::new();

    for via in &board.vias {
        if !is_high_speed_net(board, via.net) {
            continue;
        }
        let near_same_net_pad = board
            .pads_of(via.net)
            .any(|pad| via.pos.euclid(pad.pos) <= max_dist);
        if !near_same_net_pad {
            count += 1;
            pts.push(via.pos);
        }
    }

    (count, pts)
}

pub(crate) fn detect_high_speed_via_diameter_violations(
    board: &Board,
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    let max_diameter = rules.via_diameter();
    let mut count = 0;
    let mut pts = Vec::new();

    for via in &board.vias {
        if is_high_speed_net(board, via.net) && via.diameter > max_diameter {
            count += 1;
            pts.push(via.pos);
        }
    }

    (count, pts)
}

pub(crate) fn detect_blind_buried_via_drill_violations(
    board: &Board,
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    let mut count = 0;
    let mut pts = Vec::new();

    for via in &board.vias {
        if matches!(
            via.kind,
            crate::board::ViaKind::Blind | crate::board::ViaKind::Buried
        ) && via.drill > rules.max_blind_buried_via_drill
        {
            count += 1;
            pts.push(via.pos);
        }
    }

    (count, pts)
}
