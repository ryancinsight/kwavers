use std::collections::BTreeSet;

use crate::audit::net_util::{
    diff_pair_layer_segment_lengths, diff_pair_member_to_pair, diff_pair_members,
    diff_pair_pad_entry_distances, diff_pair_prefix,
};
use crate::board::{Board, LayerId, NetId, Track};
use crate::geom::{dist_seg_seg, Nm, Point};
use crate::place::{Component, FootprintDef};
use crate::rules::DesignRules;

use super::PairAxis;

/// Infers the dominant axis of a diff pair from its routed segments.
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

pub(crate) fn detect_diff_pair_keepout_violations(
    board: &Board,
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
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
            let edge_gap = dist_seg_seg(a.start, a.end, b.start, b.end)
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
