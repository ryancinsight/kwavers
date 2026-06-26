//! Unified verification suite aggregator.
use crate::board::Board;
use crate::place::component::Component;
use crate::place::footprint::FootprintDef;
use crate::place::symbol_import::PinMap;
use crate::rules::DesignRules;
use std::collections::HashMap;

use super::ac_coupling::parasitic_ac_coupling_check;
use super::assembly::assembly;
use super::bom::bom;
use super::erc::erc;
use super::isolation::schematic_isolation_bfs;
use super::keepin::keepin;
use super::lvs::lvs;
// Bare-name imports of the per-axis report types so the struct definition below reads cleanly.
// Mirrors the [`super`] facade's [`pub use`] block and keeps the verbose fully-qualified
// `super::<sub>::<Type>` form out of the body.
use super::ac_coupling::AcCouplingReport;
use super::assembly::AssemblyReport;
use super::bom::BomReport;
use super::erc::ErcReport;
use super::isolation::IsolationReport;
use super::keepin::KeepinReport;
use super::lvs::LvsReport;

/// The full verification result across every sign-off axis. `all_pass` is the merge gate.
#[derive(Debug, Clone)]
pub struct Verification {
    /// Netlist electrical-rule check.
    pub erc: ErcReport,
    /// Layout-versus-schematic connectivity check.
    pub lvs: LvsReport,
    /// Physical/manufacturing design-rule check (routed-copper audit).
    pub drc: crate::audit::FaultReport,
    /// Package/courtyard assembly spacing validation.
    pub assembly: AssemblyReport,
    /// Board-edge keep-in validation (no component crosses the edge keep-out).
    pub keepin: KeepinReport,
    /// Whole-design physics validation (thermal / ampacity / creepage / SI-PI margins / HDI).
    pub physics: crate::validate::PhysicsReport,
    /// Bill-of-materials validation.
    pub bom: BomReport,
    /// Schematic galvanic isolation validation.
    pub isolation: IsolationReport,
    /// Parasitic high-frequency AC coupling check.
    pub ac_coupling: AcCouplingReport,
    /// True iff every axis passes (ERC, LVS, DRC, assembly, keep-in, physics, BOM, isolation, ac_coupling).
    pub all_pass: bool,
}

impl Verification {
    /// True iff the routed-copper DRC found no hard manufacturing fault (foreign-net copper inside
    /// clearance, via-adjacency, acid traps, dangling ends all zero). Flight-line crossings and
    /// near-shorts remain adversarial risk signals, not binary DRC failures.
    #[must_use]
    pub fn drc_clean(&self) -> bool {
        self.drc.hard_drc_clean()
    }
}

/// Run every verification axis on a routed board and its source netlist, returning the unified
/// [`Verification`]. `physics` is computed by the caller (it needs the design-specific thermal /
/// current models) and folded in so the single `all_pass` gate covers ERC + LVS + DRC + physics + BOM.
///
/// Local bindings (`erc`, `lvs`, ...) intentionally shadow the function names of the same
/// spelling within the function body — canonical Rust shorthand that lets the
/// struct-initializer `Verification { erc, lvs, … }` form name every field positionally.
#[must_use]
pub fn verify_all(
    board: &Board,
    comps: &[Component],
    lib: &[FootprintDef],
    rules: &DesignRules,
    physics: crate::validate::PhysicsReport,
    symbol_pin_maps: &HashMap<String, PinMap>,
) -> Verification {
    let erc = erc(board, comps, lib, symbol_pin_maps);
    let lvs = lvs(board);
    let drc = crate::audit::audit(board, comps, lib, rules);
    let assembly = assembly(comps, lib, rules.assembly_clearance);
    let keepin = keepin(board, comps, lib, rules.edge_clearance);
    let bom = bom(comps, lib);
    let isolation = schematic_isolation_bfs(board, comps, lib);
    let ac_coupling = parasitic_ac_coupling_check(board);
    let drc_clean = drc.hard_drc_clean();
    Verification {
        // `all_pass` is computed FIRST so it can read `erc.pass` / `lvs.pass` / … before the
        // struct-init shorthand `erc, lvs, …` moves the report values into the struct. The
        // original flat `verify.rs` used this same ordering — keeping it preserves the
        // canonical pattern across the codebase.
        all_pass: erc.pass
            && lvs.pass
            && assembly.pass
            && keepin.pass
            && bom.pass
            && physics.all_pass
            && drc_clean
            && isolation.pass
            && ac_coupling.pass,
        erc,
        lvs,
        drc,
        assembly,
        keepin,
        physics,
        bom,
        isolation,
        ac_coupling,
    }
}
