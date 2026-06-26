use crate::board::{Board, NetClassKind};
use crate::geom::Point;
use crate::place::{Component, FootprintDef};
use crate::rules::DesignRules;
use crate::audit::net_util::{
    diff_pair_coupling_caps, diff_pair_members, power_reference_zone_for_track,
    reference_zones, track_midpoint,
};

use super::{diff_pair_axis, PairAxis};

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
