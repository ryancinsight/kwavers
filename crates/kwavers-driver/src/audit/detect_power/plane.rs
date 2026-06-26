//! Power plane, split-plane crossing, charge-recycling, and pulse-skip violation detectors.

use crate::board::{Board, NetClassKind, NetId};
use crate::geom::{Nm, Point};
use crate::place::{Component, FootprintDef, Role};
use crate::rules::DesignRules;
use crate::audit::net_util::reference_zones;
use crate::geom::point_in_polygon;

pub(crate) fn detect_active_ic_power_plane_violations(
    board: &Board,
    comps: &[Component],
    lib: &[FootprintDef],
) -> (usize, Vec<Point>) {
    let last_layer = board.spec.nlayers.saturating_sub(1) as u16;
    let mut count = 0;
    let mut pts = Vec::new();

    for comp in comps {
        if comp.fp >= lib.len() || !matches!(lib[comp.fp].role, Role::ActiveIc) {
            continue;
        }
        let fp = &lib[comp.fp];
        for (pad_idx, pad_def) in fp.pads.iter().enumerate() {
            if !pad_def.power_pin {
                continue;
            }
            let Some(net) = comp.nets.get(pad_idx).copied().flatten() else {
                continue;
            };
            if !matches!(
                board.class_of(net),
                NetClassKind::Power | NetClassKind::Ground
            ) {
                continue;
            }
            let pad_pos = comp.pad_pos(lib, pad_idx);
            let has_internal_plane = board.zones.iter().any(|zone| {
                zone.net == net
                    && zone.layer.0 > 0
                    && zone.layer.0 < last_layer
                    && point_in_polygon(pad_pos, &zone.polygon)
            });
            if !has_internal_plane {
                count += 1;
                pts.push(pad_pos);
            }
        }
    }

    (count, pts)
}

pub(crate) fn point_projects_inside_segment(p: Point, a: Point, b: Point) -> bool {
    let ax = a.x.0 as f64;
    let ay = a.y.0 as f64;
    let bx = b.x.0 as f64;
    let by = b.y.0 as f64;
    let px = p.x.0 as f64;
    let py = p.y.0 as f64;
    let dx = bx - ax;
    let dy = by - ay;
    let len2 = dx * dx + dy * dy;
    if len2 == 0.0 {
        return false;
    }
    let t = ((px - ax) * dx + (py - ay) * dy) / len2;
    t > 0.0 && t < 1.0
}

pub(crate) fn detect_split_plane_crossings(
    board: &Board,
    comps: &[Component],
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    let is_high_speed = |net: NetId| {
        let name = &board.nets[net.0 as usize].name;
        board.class_of(net) == NetClassKind::Hv
            || name.starts_with("TRIG")
            || name.starts_with("OUT")
            || name.starts_with("TX")
    };

    let ref_zones = reference_zones(board);

    let mut count = 0;
    let mut pts = Vec::new();
    let max_dist = rules.split_plane_stitching_cap_distance.0 as f64;

    let stitching_caps: Vec<&Component> = comps
        .iter()
        .filter(|c| {
            c.refdes.starts_with('C')
                && c.nets.iter().flatten().any(|&n| {
                    matches!(
                        board.class_of(n),
                        NetClassKind::Ground | NetClassKind::Power
                    )
                })
        })
        .collect();

    for t in &board.tracks {
        if is_high_speed(t.net) {
            for zone in &ref_zones {
                if t.layer == zone.layer {
                    let start_in = point_in_polygon(t.start, &zone.polygon);
                    let end_in = point_in_polygon(t.end, &zone.polygon);
                    if start_in != end_in {
                        let mid = Point::new(
                            Nm((t.start.x.0 + t.end.x.0) / 2),
                            Nm((t.start.y.0 + t.end.y.0) / 2),
                        );
                        let has_stitching = stitching_caps.iter().any(|c| {
                            c.placement.pos.euclid(mid) <= max_dist
                                && c.nets.contains(&Some(zone.net))
                                && c.nets.iter().flatten().any(|&n| {
                                    n != zone.net
                                        && matches!(
                                            board.class_of(n),
                                            NetClassKind::Ground | NetClassKind::Power
                                        )
                                })
                        });
                        if !has_stitching {
                            count += 1;
                            pts.push(mid);
                            break;
                        }
                    }
                }
            }
        }
    }

    (count, pts)
}

/// Detect boards that carry N-level-capable pulser ICs but lack a charge-recycling bus net.
///
/// N-level ICs (STHVUP32 / MAX14815 / MD1715) have a shared CR pin that must be tied to a
/// dedicated charge-recycling bus net (`CHR*`, `CR_*`, `CHREC*`). If the bus is absent the
/// energy stored in the load capacitance on each transition is dissipated rather than
/// recovered, wasting up to 50 % of driver dynamic loss. One violation is emitted per
/// N-level IC whose board net list contains no CR bus net; hotspot at the IC placement centroid.
pub(crate) fn detect_charge_recycling_violations_board(
    board: &Board,
    comps: &[Component],
    lib: &[FootprintDef],
) -> (usize, Vec<Point>) {
    let has_cr_bus = board.nets.iter().any(|n| {
        let up = n.name.to_ascii_uppercase();
        up.starts_with("CHR")
            || up.starts_with("CR_")
            || up.starts_with("CHREC")
            || up.contains("CHARGE_RECYCLE")
            || up.contains("CHARGERECYCLE")
    });
    // If a CR bus net exists, assume every N-level IC on the board is connected to it.
    if has_cr_bus {
        return (0, Vec::new());
    }
    let mut count = 0;
    let mut pts = Vec::new();
    for comp in comps {
        if comp.fp >= lib.len() {
            continue;
        }
        let name = lib[comp.fp].name.to_ascii_uppercase();
        if name.contains("STHVUP32") || name.contains("MAX14815") || name.contains("MD1715") {
            count += 1;
            pts.push(comp.placement.pos);
        }
    }
    (count, pts)
}

/// Detect boards where the configured pulse-skip fraction would exceed the 5 % RMS pressure
/// error tolerance.
///
/// Vacuous when [`DesignRules::max_skip_fraction`] is `0.0` (the default), so existing tests
/// and boards that have not configured a skip operating point are unaffected.
/// The TX channel count is derived directly from the board net list (nets whose name matches
/// `TX_*`) so no additional operating-point parameters are required beyond the design rules.
pub(crate) fn detect_pulse_skip_violations(board: &Board, rules: &DesignRules) -> (usize, Vec<Point>) {
    if rules.max_skip_fraction <= 0.0 {
        return (0, Vec::new());
    }
    let n_channels = board
        .nets
        .iter()
        .filter(|n| n.name.to_ascii_uppercase().starts_with("TX_"))
        .count();
    if n_channels == 0 {
        return (0, Vec::new());
    }
    let err = crate::pulse_skip::rms_pressure_error_fraction(n_channels, rules.max_skip_fraction);
    if err > rules.pressure_error_tol {
        (1, Vec::new())
    } else {
        (0, Vec::new())
    }
}
