//! Connector EMI clearance, decoupling-cap distance, termination-resistor proximity,
//! surge/ESD-suppressor proximity, and crystal/resonator proximity energy terms.
//!
//! Extracted from `compute::energy()` — lines 160–281 of the original `compute.rs`.
//! All arithmetic is bit-for-bit identical to the original; no logic was changed.

use super::config::{EnergyTerms, PlaceConfig};
use super::geom::{carries_connected_signal, rect_gap_mm};
use crate::geom::Nm;
use crate::place::component::{is_crystal_refdes, is_surge_suppressor_refdes, Component};
use crate::place::footprint::{FootprintDef, Role};

/// Accumulate connector-EMI, decoupling, termination, surge-suppressor, and crystal proximity
/// penalty terms.
///
/// # Arguments
/// * `t` — energy accumulator; `regional`, `decoupling`, and `termination` are updated.
/// * `comps` — full component list.
/// * `lib` — footprint library.
/// * `cfg` — placement configuration.
/// * `half_clear` — half the courtyard clearance (unused here, retained for API symmetry with
///   other `accumulate_*` functions so callers stay uniform).
pub(super) fn accumulate_proximity(
    t: &mut EnergyTerms,
    comps: &[Component],
    lib: &[FootprintDef],
    cfg: &PlaceConfig,
    _half_clear: Nm,
) {
    // Regional EMI floorplanning: connectors radiate and couple external noise, so sensitive
    // high-speed ICs with connected signal pads keep a clearance halo from connector courtyards.
    // The hard edge keepout still handles board-edge impedance; this term handles connector
    // proximity inside the floorplan.
    let connector_emi_clearance = (cfg.courtyard_clearance.to_mm() * 2.0).max(1.0);
    for active_ic in comps {
        let fp = &lib[active_ic.fp];
        if !matches!(fp.role, Role::ActiveIc) || !carries_connected_signal(active_ic, fp) {
            continue;
        }
        let active_rect = active_ic.courtyard(lib);
        for connector in comps {
            if !matches!(lib[connector.fp].role, Role::Connector) {
                continue;
            }
            let gap = rect_gap_mm(active_rect, connector.courtyard(lib));
            t.regional += (connector_emi_clearance - gap).max(0.0);
        }
    }

    // Decoupling: each bypass cap to the nearest power pin of its associated IC.
    for c in comps {
        if !matches!(lib[c.fp].role, Role::Decoupling) {
            continue;
        }
        let Some(ic) = c.assoc_ic else { continue };
        let cap = c.placement.pos;
        let ic_c = &comps[ic];
        let mut nearest = f64::INFINITY;
        for (k, pad) in lib[ic_c.fp].pads.iter().enumerate() {
            if pad.power_pin {
                let d = cap.euclid(ic_c.pad_pos(lib, k)) * 1.0e-6;
                nearest = nearest.min(d);
            }
        }
        if nearest.is_finite() {
            t.decoupling += nearest;
        }
    }

    // Termination: resistor-like passives on a net should sit at the active pad they terminate,
    // not merely somewhere on the net's HPWL box. The final audit applies the high-speed-specific
    // 2 mm budget; this term gives placement a continuous objective before that hard gate.
    for terminator in comps {
        if terminator.fp >= lib.len()
            || !matches!(lib[terminator.fp].role, Role::Passive)
            || !terminator.refdes.starts_with('R')
        {
            continue;
        }
        let mut nearest = f64::INFINITY;
        for (term_pos, _term_layers, term_net) in terminator.placed_pads(lib) {
            let Some(net) = term_net else { continue };
            for active in comps {
                if active.fp >= lib.len() || !matches!(lib[active.fp].role, Role::ActiveIc) {
                    continue;
                }
                for (active_pos, _active_layers, active_net) in active.placed_pads(lib) {
                    if active_net == Some(net) {
                        nearest = nearest.min(term_pos.euclid(active_pos) * 1.0e-6);
                    }
                }
            }
        }
        if nearest.is_finite() {
            t.termination += nearest;
        }
    }

    // Surge/ESD suppressors: diode-like passives on incoming connector nets should sit at the
    // connector before the protected trace enters the board. This keeps the clamp path short and
    // reduces the chance that the connector-to-clamp segment needs a parasitic via.
    for suppressor in comps {
        if suppressor.fp >= lib.len()
            || !matches!(lib[suppressor.fp].role, Role::Passive)
            || !is_surge_suppressor_refdes(&suppressor.refdes)
        {
            continue;
        }
        let mut nearest = f64::INFINITY;
        for (supp_pos, _supp_layers, supp_net) in suppressor.placed_pads(lib) {
            let Some(net) = supp_net else { continue };
            for connector in comps {
                if connector.fp >= lib.len() || !matches!(lib[connector.fp].role, Role::Connector) {
                    continue;
                }
                for (conn_pos, _conn_layers, conn_net) in connector.placed_pads(lib) {
                    if conn_net == Some(net) {
                        nearest = nearest.min(supp_pos.euclid(conn_pos) * 1.0e-6);
                    }
                }
            }
        }
        if nearest.is_finite() {
            t.regional += nearest;
        }
    }

    // Crystal/resonator support: clock-source components associated with a main IC must sit close
    // to the shared clock pins so those critical routes are short before controlled-impedance
    // routing consumes the remaining channels.
    for oscillator in comps {
        if oscillator.fp >= lib.len() || !is_crystal_refdes(&oscillator.refdes) {
            continue;
        }
        let Some(ic_idx) = oscillator.assoc_ic.filter(|ic| *ic < comps.len()) else {
            continue;
        };
        let ic = &comps[ic_idx];
        let mut nearest = f64::INFINITY;
        for (osc_pos, _osc_layers, osc_net) in oscillator.placed_pads(lib) {
            let Some(net) = osc_net else { continue };
            for (ic_pos, _ic_layers, ic_net) in ic.placed_pads(lib) {
                if ic_net == Some(net) {
                    nearest = nearest.min(osc_pos.euclid(ic_pos) * 1.0e-6);
                }
            }
        }
        if nearest.is_finite() {
            t.regional += nearest;
        }
    }
}
