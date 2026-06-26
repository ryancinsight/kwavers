//! Board-edge clearance, component-edge clearance, and parallel-spacing
//! violation detectors for high-speed signals.

use crate::board::{Board, NetClassKind, NetId};
use crate::geom::{Nm, Point};
use crate::place::{Component, FootprintDef, Role};
use crate::rules::DesignRules;
use crate::audit::net_util::{are_diff_pair_mates, is_clock_like_net, is_high_speed_net};

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
