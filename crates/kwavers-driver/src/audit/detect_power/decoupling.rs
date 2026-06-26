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

/// Decoupling capacitors placed farther than `rules.max_decoupling_cap_distance` from their
/// associated IC power pin.
///
/// The article ("most common PCB mistakes") requires decoupling caps to be placed "as close
/// as possible" to the component power pins. This check enforces a hard ceiling on the
/// Euclidean distance from the cap centroid to the nearest power pin of its associated IC,
/// irrespective of layer. Vacuous when no cap has an `assoc_ic` link.
pub(crate) fn detect_decoupling_cap_distance_violations(
    comps: &[Component],
    lib: &[FootprintDef],
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    let max_dist = rules.max_decoupling_cap_distance.0 as f64;
    let mut count = 0;
    let mut pts = Vec::new();

    for cap in comps {
        if cap.fp >= lib.len() || !matches!(lib[cap.fp].role, Role::Decoupling) {
            continue;
        }
        let Some(ic_idx) = cap.assoc_ic else { continue };
        let Some(ic) = comps.get(ic_idx) else { continue };
        if ic.fp >= lib.len() {
            continue;
        }
        let ic_fp = &lib[ic.fp];

        // Measure distance to the nearest IC power-pin pad.
        let mut nearest: f64 = f64::INFINITY;
        for (pad_idx, ic_pad) in ic_fp.pads.iter().enumerate() {
            if !ic_pad.power_pin {
                continue;
            }
            let d = cap.placement.pos.euclid(ic.pad_pos(lib, pad_idx));
            if d < nearest {
                nearest = d;
            }
        }
        if nearest.is_finite() && nearest > max_dist {
            count += 1;
            pts.push(cap.placement.pos);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::Nm;
    use crate::place::{PadDef, Placement};

    fn ic_fp() -> FootprintDef {
        FootprintDef::new(
            "U",
            (Nm::from_mm(4.0), Nm::from_mm(4.0)),
            Role::ActiveIc,
            vec![PadDef {
                offset: crate::geom::Point::new(Nm(0), Nm(0)),
                size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                layers: vec![crate::board::LayerId(0)],
                power_pin: true,
            }],
        )
    }

    fn cap_fp() -> FootprintDef {
        FootprintDef::new(
            "C",
            (Nm::from_mm(1.0), Nm::from_mm(0.5)),
            Role::Decoupling,
            vec![],
        )
    }

    fn make_comp(fp: usize, x: f64, y: f64, assoc: Option<usize>) -> Component {
        Component {
            fp,
            refdes: String::new(),
            nets: vec![],
            placement: Placement {
                pos: crate::geom::Point::new(Nm::from_mm(x), Nm::from_mm(y)),
                ..Placement::default()
            },
            assoc_ic: assoc,
            ..Component::default()
        }
    }

    /// IC at (10,10); cap at (10,10) is 0 mm from the power pin — passes the 3 mm budget.
    /// Cap at (20,10) is 10 mm away — exceeds the 3 mm budget.
    #[test]
    fn flags_cap_too_far_from_ic_power_pin() {
        let lib = vec![ic_fp(), cap_fp()];
        let comps = vec![
            make_comp(0, 10.0, 10.0, None),    // IC (index 0)
            make_comp(1, 10.0, 10.0, Some(0)), // NEAR cap: 0 mm from pin
            make_comp(1, 20.0, 10.0, Some(0)), // FAR cap: 10 mm from pin
        ];
        let mut rules = DesignRules::holohv();
        rules.max_decoupling_cap_distance = Nm::from_mm(3.0);
        let (n, pts) = detect_decoupling_cap_distance_violations(&comps, &lib, &rules);
        assert_eq!(n, 1, "only the far cap must be flagged");
        assert!((pts[0].x.to_mm() - 20.0).abs() < 1e-3);
    }

    #[test]
    fn passes_when_all_caps_within_budget() {
        let lib = vec![ic_fp(), cap_fp()];
        let comps = vec![
            make_comp(0, 10.0, 10.0, None),    // IC
            make_comp(1, 11.0, 10.0, Some(0)), // 1 mm away — passes 3 mm budget
        ];
        let rules = DesignRules::holohv();
        let (n, _) = detect_decoupling_cap_distance_violations(&comps, &lib, &rules);
        assert_eq!(n, 0);
    }
}
