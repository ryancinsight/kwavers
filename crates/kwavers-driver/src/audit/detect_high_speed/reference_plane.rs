//! Reference-plane margin, absence, dual-ground, power stitching-cap,
//! intrusion, and ground-plane fragmentation violation detectors.

use std::collections::BTreeMap;

use crate::board::{Board, LayerId, NetClassKind, NetId, ZoneFill};
use crate::geom::{distance_to_polygon_boundary, point_in_polygon, Nm, Point};
use crate::place::Component;
use crate::rules::DesignRules;
use crate::audit::antenna::polygon_vertex_mean;
use crate::audit::net_util::{is_high_speed_net, reference_zones, track_midpoint};

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
