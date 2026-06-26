//! ERC — electrical rule check over the *netlist*.
use crate::board::{Board, NetClassKind};
use crate::place::component::Component;
use crate::place::footprint::{FootprintDef, Role};
use crate::place::symbol_import::PinMap;
use std::collections::HashMap;

/// Electrical-rule-check findings over the schematic/netlist (before/independent of routing).
#[derive(Debug, Clone, Default)]
pub struct ErcReport {
    /// Component pads carrying no net (unconnected inputs/outputs) — `(refdes, pad index)`.
    pub unconnected_pads: Vec<(String, usize)>,
    /// Net names with exactly one terminal — a floating/open net that drives or sinks nothing.
    pub floating_nets: Vec<String>,
    /// Power/ground pins landed on a non-power net (`refdes`, pad index, net) — a likely mis-wire.
    pub power_pin_on_signal: Vec<(String, usize, String)>,
    /// Whether every ERC check passed.
    pub pass: bool,
}

/// Run the netlist ERC: flag unconnected pads, floating (single-terminal) nets, and power pins on a
/// non-power net. Pin **direction** conflicts (two drivers on a net) need an electrical pin-type
/// model the footprints do not yet carry, so they are out of scope here (noted, not silently passed).
#[must_use]
pub fn erc(
    board: &Board,
    comps: &[Component],
    lib: &[FootprintDef],
    symbol_pin_maps: &HashMap<String, PinMap>,
) -> ErcReport {
    let mut r = ErcReport::default();
    // Terminal count per net (from the placed component pads).
    let mut term_count = vec![0usize; board.nets.len()];
    for c in comps {
        let fp = &lib[c.fp];
        let pin_map = symbol_pin_maps.get(&c.refdes);
        for (k, pad) in fp.pads.iter().enumerate() {
            // A mechanical pad (empty designator — a non-plated board-lock/mounting hole) is not an
            // electrical pin: it has no net by construction and is not an unconnected-pin fault. It
            // stays a PCB keepout, but ERC (like kicad-cli, which sees no schematic pin for it) skips it.
            if fp.pad_names.get(k).is_some_and(|n| n.is_empty()) {
                continue;
            }
            match c.nets.get(k).copied().flatten() {
                None => {
                    let is_nc = if let Some(pm) = pin_map {
                        if let Some(pad_name) = fp.pad_names.get(k) {
                            if let Some(name) = pm.name_of(pad_name) {
                                name == "NC" || name == "no_connect" || name.starts_with("NC_")
                            } else {
                                true // not defined in the symbol
                            }
                        } else {
                            true // no pad name
                        }
                    } else {
                        false
                    };
                    let is_allowed_role = matches!(
                        lib[c.fp].role,
                        Role::ActiveIc | Role::Connector | Role::Power
                    );
                    if !is_nc && !is_allowed_role {
                        r.unconnected_pads.push((c.refdes.clone(), k));
                    }
                }
                Some(n) => {
                    term_count[n.0 as usize] += 1;
                    if pad.power_pin
                        && !matches!(
                            board.class_of(n),
                            NetClassKind::Power | NetClassKind::Ground | NetClassKind::Hv
                        )
                    {
                        r.power_pin_on_signal.push((
                            c.refdes.clone(),
                            k,
                            board.nets[n.0 as usize].name.clone(),
                        ));
                    }
                }
            }
        }
    }
    for (i, &count) in term_count.iter().enumerate() {
        if count == 1 {
            r.floating_nets.push(board.nets[i].name.clone());
        }
    }
    r.pass = r.unconnected_pads.is_empty()
        && r.floating_nets.is_empty()
        && r.power_pin_on_signal.is_empty();
    r
}
