use std::collections::BTreeMap;

use crate::audit::net_util::{
    diff_pair_interface_group, diff_pair_members, diff_pair_prefix, track_midpoint,
};
use crate::board::{Board, LayerId, NetClassKind};
use crate::geom::Point;
use crate::rules::DesignRules;

use super::{diff_pair_axis, PairAxis};

pub(crate) fn detect_diff_pair_interface_mismatch_violations(
    board: &Board,
) -> (usize, usize, Vec<Point>) {
    let mut groups: BTreeMap<String, Vec<(std::collections::BTreeSet<LayerId>, usize)>> =
        BTreeMap::new();
    let mut group_points: BTreeMap<String, Vec<Point>> = BTreeMap::new();
    for (p_net, n_net) in diff_pair_members(board) {
        let name = &board.nets[p_net.0 as usize].name;
        let Some(prefix) = diff_pair_prefix(name) else {
            continue;
        };
        let Some(group) = diff_pair_interface_group(prefix) else {
            continue;
        };
        let layers: std::collections::BTreeSet<LayerId> = board
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
