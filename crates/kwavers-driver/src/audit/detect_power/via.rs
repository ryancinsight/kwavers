//! Via and microvia violation detectors.

use std::collections::BTreeSet;

use crate::audit::net_util::is_high_speed_net;
use crate::board::{Board, NetClassKind};
use crate::geom::{dist_point_seg, Point};
use crate::place::component::is_surge_suppressor_refdes;
use crate::place::{Component, FootprintDef, Role};
use crate::rules::DesignRules;

use super::point_projects_inside_segment;

/// [`crate::board::ViaKind::Micro`] whose laser drill ÷ build-up
/// dielectric thickness exceeds
/// [`crate::rules::DesignRules::max_microvia_ar`]. Wraps the per-board
/// aggregate boolean in [`crate::validate::microvia_aspect_check`] into a
/// per-via count and positions so the audit can fold the microvia-AR
/// severity into [`crate::audit::fault_report::FaultReport::risk_score`]
/// weighted by violation count.
/// Because the build-up dielectric is a per-board constant (the stack's
/// prepreg is symmetric), the extra `1e-9` slack from the validator's
/// epsilon equalisation is preserved verbatim — a value at exactly
/// the limit still reads as compliant rather than tripping the count.
pub(crate) fn detect_microvia_aspect_violations(
    board: &Board,
    rules: &DesignRules,
    build_up_mm: f64,
) -> (usize, Vec<Point>) {
    let drill_mm = rules.microvia_drill.to_mm();
    if drill_mm <= 0.0 {
        return (0, Vec::new());
    }
    let mut ar = build_up_mm / drill_mm;
    let limit = rules.max_microvia_ar;
    if ar > limit && ar <= limit + 1e-9 {
        ar = limit;
    }
    if ar <= limit {
        return (0, Vec::new());
    }
    let mut count = 0;
    let mut pts = Vec::new();
    for via in &board.vias {
        if matches!(via.kind, crate::board::ViaKind::Micro) {
            count += 1;
            pts.push(via.pos);
        }
    }
    (count, pts)
}

pub(crate) fn detect_decoupling_ground_via_violations(
    board: &Board,
    comps: &[Component],
    lib: &[FootprintDef],
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    let ground_vias: Vec<Point> = board
        .vias
        .iter()
        .filter(|v| board.class_of(v.net) == NetClassKind::Ground)
        .map(|v| v.pos)
        .collect();
    let max_dist = rules.decoupling_ground_via_distance.0 as f64;
    let mut count = 0;
    let mut pts = Vec::new();

    for cap in comps {
        if !matches!(lib[cap.fp].role, Role::Decoupling) {
            continue;
        }
        for (pad_pos, pad_layers, pad_net) in cap.placed_pads(lib) {
            let Some(pad_net) = pad_net else { continue };
            if board.class_of(pad_net) != NetClassKind::Ground {
                continue;
            }
            if pad_layers.len() > 1 {
                continue;
            }
            let has_local_ground_via = ground_vias
                .iter()
                .any(|&via_pos| pad_pos.euclid(via_pos) <= max_dist);
            if !has_local_ground_via {
                count += 1;
                pts.push(pad_pos);
            }
        }
    }

    (count, pts)
}

pub(crate) fn detect_surge_suppressor_via_violations(
    board: &Board,
    comps: &[Component],
    lib: &[FootprintDef],
) -> (usize, Vec<Point>) {
    let mut violating_vias = BTreeSet::new();
    let mut pts = Vec::new();

    for suppressor in comps {
        if suppressor.fp >= lib.len()
            || !matches!(lib[suppressor.fp].role, Role::Passive)
            || !is_surge_suppressor_refdes(&suppressor.refdes)
        {
            continue;
        }
        for (supp_pos, _supp_layers, supp_net) in suppressor.placed_pads(lib) {
            let Some(net) = supp_net else { continue };
            for connector in comps {
                if connector.fp >= lib.len() || !matches!(lib[connector.fp].role, Role::Connector) {
                    continue;
                }
                for (conn_pos, _conn_layers, conn_net) in connector.placed_pads(lib) {
                    if conn_net != Some(net) {
                        continue;
                    }
                    for (via_idx, via) in board.vias.iter().enumerate() {
                        if via.net != net
                            || !point_projects_inside_segment(via.pos, conn_pos, supp_pos)
                        {
                            continue;
                        }
                        let path_clearance = dist_point_seg(via.pos, conn_pos, supp_pos);
                        if path_clearance <= via.diameter.0 as f64 / 2.0
                            && violating_vias.insert(via_idx)
                        {
                            pts.push(via.pos);
                        }
                    }
                }
            }
        }
    }

    (violating_vias.len(), pts)
}

pub(crate) fn detect_high_speed_via_stub_violations(
    board: &Board,
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    let mut count = 0;
    let mut pts = Vec::new();

    for via in &board.vias {
        if via.from == via.to || !is_high_speed_net(board, via.net) {
            continue;
        }

        let mut used_layers = BTreeSet::new();
        for track in &board.tracks {
            if track.net == via.net && (track.start == via.pos || track.end == via.pos) {
                used_layers.insert(track.layer.0);
            }
        }
        if used_layers.len() < 2 {
            continue;
        }
        let Some(used_lo) = used_layers.first().copied() else {
            continue;
        };
        let Some(used_hi) = used_layers.last().copied() else {
            continue;
        };
        let physical_lo = via.from.0.min(via.to.0);
        let physical_hi = via.from.0.max(via.to.0);
        let stub_layers = used_lo.saturating_sub(physical_lo) + physical_hi.saturating_sub(used_hi);
        if stub_layers > rules.high_speed_max_via_stub_layers {
            count += 1;
            pts.push(via.pos);
        }
    }

    (count, pts)
}

pub(crate) fn detect_unfilled_via_in_pad_violations(board: &Board) -> (usize, Vec<Point>) {
    let mut count = 0;
    let mut pts = Vec::new();

    for via in &board.vias {
        if via.filled || board.class_of(via.net) == NetClassKind::Ground {
            continue;
        }
        let in_smd_pad = board.pads.iter().any(|pad| {
            pad.net == Some(via.net)
                && pad.layers.len() == 1
                && pad.pos.x == via.pos.x
                && pad.pos.y == via.pos.y
        });
        if in_smd_pad {
            count += 1;
            pts.push(via.pos);
        }
    }

    (count, pts)
}
