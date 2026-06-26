//! Via stub, transition ground via, terminal ground via, via-pad proximity,
//! via diameter, blind/buried drill, and termination-placement violation
//! detectors for high-speed signals.

use std::collections::HashMap;

use crate::board::{Board, NetClassKind, NetId, LayerId};
use crate::geom::{Nm, Point};
use crate::place::{Component, FootprintDef, Role};
use crate::rules::DesignRules;
use crate::audit::net_util::is_high_speed_net;

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
