//! KiCad project emission — saves the full `<stem>.kicad_pcb` + `<stem>.kicad_dru` + `<stem>.kicad_pro`
//! triple into the directory of `pcb_path`.
//!
//! Drives [`crate::io::pcb_emit::save_kicad_pcb`] (or its flagged-failure sibling
//! [`crate::io::pcb_emit::save_kicad_pcb_flagged`]) plus the design-rule and project sidecars, so
//! the engine's committed rule set reaches `kicad-cli pcb drc` without the consumer having to
//! remember to write the sidecars separately.

use std::path::Path;

use crate::board::Board;
use crate::place::component::Component;
use crate::place::footprint::FootprintDef;
use crate::rules::{CreepageRule, DesignRules};

use super::dru_emit::write_kicad_dru;
use super::pcb_emit::{save_kicad_pcb, save_kicad_pcb_flagged};
use super::pro_emit::write_kicad_pro;

/// Write a complete KiCad project triple — `<stem>.kicad_pcb`, `<stem>.kicad_dru`, and
/// `<stem>.kicad_pro` — into the directory of `pcb_path`.
///
/// [`crate::io::pcb_emit::save_kicad_pcb`] writes only the board file. This function adds the companion
/// design-rule file (`.kicad_dru`) and minimal project descriptor (`.kicad_pro`) so that
/// `kicad-cli pcb drc` picks up the board's committed rule set (holohv: 0.13 mm clearance, 0.15 mm
/// track, …) instead of KiCad's conservative 0.20 mm default — making the external CLI DRC verdict
/// match the internal routing oracle.
///
/// The stem is derived from `pcb_path`'s filename (e.g. `output/foo.kicad_pcb` → stem `foo`);
/// the three output files all share that stem in the same directory.
pub fn save_kicad_project(
    pcb_path: &Path,
    board: &Board,
    comps: &[Component],
    lib: &[FootprintDef],
    rules: &DesignRules,
    creepage: &CreepageRule,
) -> std::io::Result<()> {
    save_kicad_pcb(pcb_path, board, comps, lib, rules)?;
    let stem = pcb_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("non-UTF-8 PCB path stem: {}", pcb_path.display()),
            )
        })?;
    let dir = pcb_path.parent().unwrap_or(Path::new("."));
    std::fs::write(
        dir.join(format!("{stem}.kicad_dru")),
        write_kicad_dru(rules, creepage),
    )?;
    std::fs::write(
        dir.join(format!("{stem}.kicad_pro")),
        write_kicad_pro(stem, rules),
    )?;
    Ok(())
}

/// Like [`save_kicad_project`] but for a board that **failed** the manufacturing gate: the
/// `.kicad_pcb` is stamped with a [`save_kicad_pcb_flagged`] `DRC FAIL` banner, while the
/// `.kicad_dru` and `.kicad_pro` sidecars are still written so the flagged board DRC-checks
/// against the engine's own rule set. This is the labelled, inspectable alternative to refusing
/// to write — a failing board you can render and DRC is a failing board you can drive a fix from.
#[allow(clippy::too_many_arguments)]
pub fn save_kicad_project_flagged(
    pcb_path: &Path,
    board: &Board,
    comps: &[Component],
    lib: &[FootprintDef],
    rules: &DesignRules,
    creepage: &CreepageRule,
    label: &str,
) -> std::io::Result<()> {
    save_kicad_pcb_flagged(pcb_path, board, comps, lib, rules, label)?;
    let stem = pcb_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("non-UTF-8 PCB path stem: {}", pcb_path.display()),
            )
        })?;
    let dir = pcb_path.parent().unwrap_or(Path::new("."));
    std::fs::write(
        dir.join(format!("{stem}.kicad_dru")),
        write_kicad_dru(rules, creepage),
    )?;
    std::fs::write(
        dir.join(format!("{stem}.kicad_pro")),
        write_kicad_pro(stem, rules),
    )?;
    Ok(())
}
