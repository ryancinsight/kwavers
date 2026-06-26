//! High-speed routing violation detectors.
//!
//! Covers board-edge clearance, component edge clearance, termination placement,
//! parallel spacing (same and adjacent layer), reference-plane margin/absence/
//! intrusion/stitching, split-domain crossings, via stub/transition/terminal,
//! via diameter, and blind/buried via drill size.

use std::collections::{BTreeMap, HashMap};

use crate::board::{split_domain_from_name, Board, LayerId, NetClassKind, NetId, SplitDomain, ZoneFill};
use crate::geom::{dist_seg_seg, distance_to_polygon_boundary, point_in_polygon, Nm, Point};
use crate::place::{Component, FootprintDef, Role};
use crate::rules::DesignRules;
use crate::audit::antenna::polygon_vertex_mean;
use crate::audit::net_util::{
    adjacent_ground_reference_zone_indices, are_diff_pair_mates, is_clock_like_net,
    is_high_speed_net, reference_zones, track_midpoint,
};

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
