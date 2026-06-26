//! Power-integrity, decoupling, and via-quality auditors.
//!
//! Each `detect_*` function identifies one family of defects and returns
//! `(count, Vec<Point>)` where the points are board-space hotspot positions
//! for the congestion-weighted placement feedback loop.
//!
//! All items are `pub(crate)` — the only caller is [`crate::audit::critic::audit`].

use std::collections::BTreeSet;

use crate::board::{Board, NetClassKind, NetId};
use crate::geom::{dist_point_seg, point_in_polygon, Nm, Point};
use crate::place::component::is_surge_suppressor_refdes;
use crate::place::{Component, FootprintDef, Role};
use crate::rules::DesignRules;
use crate::audit::net_util::{is_high_speed_net, reference_zones};

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

pub(crate) fn detect_decoupling_power_layer_violations(
    board: &Board,
    comps: &[Component],
    lib: &[FootprintDef],
) -> (usize, Vec<Point>) {
    let mut count = 0;
    let mut pts = Vec::new();

    for cap in comps {
        if cap.fp >= lib.len() || !matches!(lib[cap.fp].role, Role::Decoupling) {
            continue;
        }
        let Some(ic_idx) = cap.assoc_ic else {
            continue;
        };
        let Some(ic) = comps.get(ic_idx) else {
            continue;
        };
        if ic.fp >= lib.len() {
            continue;
        }
        let ic_fp = &lib[ic.fp];

        for (cap_pos, cap_layers, cap_net) in cap.placed_pads(lib) {
            let Some(cap_net) = cap_net else { continue };
            if !matches!(
                board.class_of(cap_net),
                NetClassKind::Power | NetClassKind::Hv
            ) {
                continue;
            }
            let has_same_layer_ic_power_pin =
                ic_fp.pads.iter().enumerate().any(|(pad_idx, ic_pad)| {
                    ic_pad.power_pin
                        && ic.nets.get(pad_idx).copied().flatten() == Some(cap_net)
                        && ic_pad
                            .layers
                            .iter()
                            .any(|layer| cap_layers.iter().any(|cap_layer| cap_layer == layer))
                });
            if !has_same_layer_ic_power_pin {
                count += 1;
                pts.push(cap_pos);
            }
        }
    }

    (count, pts)
}

pub(crate) fn detect_decoupling_loop_area_violations(
    comps: &[Component],
    lib: &[FootprintDef],
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    let mut count = 0;
    let mut pts = Vec::new();

    for loop_geom in crate::physics::emi::commutation_loops(comps, lib) {
        if loop_geom.area_mm2 > rules.max_decoupling_loop_area_mm2 {
            count += 1;
            pts.push(loop_geom.at);
        }
    }

    (count, pts)
}

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

/// **Charge-reservoir sufficiency**: for every active IC with a non-zero datasheet
/// `I_dd` (`lib[ic.fp].i_dd_a > 0`), sum the incremental drive current every
/// associated decoupling capacitor can supply over the IC's switching window:
/// `I_k = C_k · dV/dt` (`crate::physics::emi::capacitive_drive_current_a`, with the
/// board-uniform `dV` from `rules.ic_switching_dv_v` and the board-uniform `dt`
/// from `rules.ic_switching_risetime_s`). Each IC whose summed supply falls below
/// its `I_dd` rating counts as 1 violation, and the IC's first VPP pad position is
/// pushed onto the hotspot list (the bottleneck, matching
/// `detect_active_ic_power_plane_violations`'s pad-level convention rather than the
/// IC centroid). Vacuous ICs (`i_dd_a <= 0`, or `dv <= 0`, or `risetime_s <= 0`) are
/// silently skipped — matching `crate::validate::microvia_aspect_check`'s
/// pass-vacuous pattern — so boards that haven't yet wired a part-level `i_dd_a`
/// line into the library don't trigger a false positive.
pub(crate) fn detect_charge_reservoir_violations(
    comps: &[Component],
    lib: &[FootprintDef],
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    let dv = rules.ic_switching_dv_v;
    let dt = rules.ic_switching_risetime_s;
    let have_transient = dv > 0.0 && dt > 0.0;
    let mut count = 0;
    let mut pts = Vec::new();
    for (ic_idx, comp) in comps.iter().enumerate() {
        if comp.fp >= lib.len() || !matches!(lib[comp.fp].role, Role::ActiveIc | Role::Power) {
            continue;
        }
        let i_dd = lib[comp.fp].i_dd_a;
        if i_dd <= 0.0 || !have_transient {
            continue;
        }
        // Sum every cap that points to this IC and has a non-zero capacitance.
        // Caps without an `assoc_ic` (e.g. power-rail bulk) are intentionally
        // ignored here — they're not the IC's high-frequency reservoir. We also
        // skip caps whose `capacitance_f == 0.0` (library row not yet wired), so
        // the missing-data signal stays loud: a board with `i_dd > 0` but every
        // cap at `c == 0` reports `i_supply = 0`, which trips the violation (the
        // IC is, by hook or by crook, under-provisioned).
        let mut i_supply = 0.0;
        for cap in comps {
            if cap.fp >= lib.len() || cap.assoc_ic != Some(ic_idx) {
                continue;
            }
            // Mirror `emi::commutation_loops`: only `Role::Decoupling` caps form
            // the IC's high-frequency bypass reservoir. A stray `capacitance_f` on
            // a passive/resistor footprint would otherwise silently rescue an
            // under-provisioned IC.
            if !matches!(lib[cap.fp].role, Role::Decoupling) {
                continue;
            }
            let c_f = lib[cap.fp].capacitance_f;
            if c_f <= 0.0 {
                continue;
            }
            i_supply += crate::physics::emi::capacitive_drive_current_a(c_f, dv, dt);
        }
        // 1e-12 epsilon for FP-floor precision: `i_supply + 1e-12` either equals
        // or slightly exceeds `i_supply` so an exactly-balanced comparison
        // `i_supply + 1e-12 < i_dd` becomes false (no violation) at the boundary.
        // Tighter than `validate::microvia_aspect_check`'s 1e-9 because this is
        // already a per-cap sum under reasonable regime.
        if i_supply + 1.0e-12 < i_dd {
            count += 1;
            // Push the IC's first VPP pad as the hotspot — same convention as
            // `detect_active_ic_power_plane_violations`. Falls back to placement
            // centroid if the IC has no power pads (e.g. signal-only IC, but a
            // signal-only IC's `i_dd_a` is 0 vacuously here anyway).
            let hotspot = lib[comp.fp]
                .pads
                .iter()
                .position(|p| p.power_pin)
                .map(|k| comp.pad_pos(lib, k))
                .unwrap_or(comp.placement.pos);
            pts.push(hotspot);
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
