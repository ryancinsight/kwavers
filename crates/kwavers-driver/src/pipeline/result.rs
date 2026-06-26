//! The [`CoOptResult`] returned by the co-optimization loop.

use crate::place::component::Component;
use crate::place::footprint::FootprintDef;

use super::config::CoOpt;
use super::cooptimize::component_clearance_clean;

/// Outcome of [`crate::pipeline::cooptimize`].
pub struct CoOptResult {
    /// Best routed board.
    pub board: crate::board::Board,
    /// The placement that produced it.
    pub comps: Vec<Component>,
    /// The adversarial critic's report on the best board.
    pub report: crate::audit::FaultReport,
    /// Whether the best routing is legal (no over-capacity).
    pub legal: bool,
    /// Whether the best routing connected every net.
    pub complete: bool,
    /// Rounds actually run.
    pub rounds_run: usize,
    /// Copper layers available in the stack this routed on.
    pub layer_count: usize,
    /// Distinct copper layers actually carrying routed copper (≤ `layer_count`).
    pub layers_used: usize,
}

impl CoOptResult {
    /// True only when the routed result is safe to persist as an authoritative KiCad artifact.
    ///
    /// This is the same hard gate used by layer/area selection plus an LVS check over the emitted
    /// copper graph. Examples call this before writing `.kicad_pcb` files so a failed optimization
    /// cannot leave a stale DRC-failing board in the workspace.
    #[must_use]
    pub fn manufacturing_clean(&self, lib: &[FootprintDef], cfg: &CoOpt) -> bool {
        self.complete
            && self.legal
            && self.report.hard_drc_clean()
            && component_clearance_clean(self, lib, cfg)
            && crate::verify::lvs(&self.board).pass
    }

    /// Human-readable reasons why [`Self::manufacturing_clean`] would reject this result.
    #[must_use]
    pub fn manufacturing_blockers(&self, lib: &[FootprintDef], cfg: &CoOpt) -> Vec<&'static str> {
        let mut blockers = Vec::new();
        if !self.complete {
            blockers.push("incomplete LVS connectivity");
        }
        if !self.legal {
            blockers.push("illegal routed capacity");
        }
        if !self.report.hard_drc_clean() {
            blockers.push("hard internal DRC violations");
        }
        if !component_clearance_clean(self, lib, cfg) {
            blockers.push("component courtyard clearance violations");
        }
        if !crate::verify::lvs(&self.board).pass {
            blockers.push("LVS opens or shorts");
        }
        blockers
    }
}
