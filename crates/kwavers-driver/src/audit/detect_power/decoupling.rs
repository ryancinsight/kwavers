//! Decoupling capacitor and charge-reservoir violation detectors.

use crate::board::{Board, NetClassKind};
use crate::geom::Point;
use crate::place::{Component, FootprintDef, Role};
use crate::rules::DesignRules;

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
