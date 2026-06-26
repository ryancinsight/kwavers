//! `.kicad_dru` emission — KiCad custom design rules.
//!
//! Declares every design-rule the engine routes to so `kicad-cli pcb drc` judges the board
//! against the holohv fab floor (0.13 mm clearance, 0.15 mm track, ...) rather than KiCad's
//! stricter built-in defaults. Without the `.kicad_dru` next to the board, every track /
//! drill / via is falsely flagged by kicad-cli's conservative fallback thresholds.
//!
//! Mirrors every applicable [`crate::rules::DesignRules`] field: track width, clearance, hole
//! clearance, microvia diameter / drill / annular (HDI-only — falls back to thru-class
//! thresholds otherwise), and the HV creepage rule.

use std::fmt::Write as _; // for `writeln!`/`write!` inside the `w!`/`wln!` macro expansions

use crate::rules::{CreepageRule, DesignRules};

// `w!` / `wln!` macros are declared at the slice root (`src/io/mod.rs`) and inherited
// into this submodule via the `#[macro_use] pub mod dru_emit;` declaration at the bottom
// of `mod.rs`. No `use super::wln;` import is needed.

/// Emit a KiCad custom design-rules (`.kicad_dru`) file matching [`DesignRules`], plus the
/// high-voltage creepage rule.
pub fn write_kicad_dru(rules: &DesignRules, creepage: &CreepageRule) -> String {
    let mut s = String::new();
    s.push_str("(version 1)\n");
    wln!(
        s,
        "(rule \"track_min\" (constraint track_width (min {:.3}mm)))",
        rules.min_track.to_mm()
    );
    wln!(
        s,
        "(rule \"clearance_min\" (constraint clearance (min {:.3}mm)))",
        rules.min_clearance.to_mm()
    );
    // Hole-to-copper clearance: declare the engine's own value so kicad-cli enforces it instead of
    // falling back to its conservative 0.25 mm default (which split the internal/external verdict by
    // flagging vias the engine — and `detect_hole_clearance_violations` — consider legal).
    wln!(
        s,
        "(rule \"hole_clearance_min\" (constraint hole_clearance (min {:.3}mm)))",
        rules.hole_clearance().to_mm()
    );
    if rules.via_policy == crate::rules::ViaPolicy::Hdi {
        wln!(
            s,
            "(rule \"hole_min\" (condition \"(A.Type == 'via' && A.Via_Type != 'Micro') || A.Type == 'pad'\") (constraint hole_size (min {:.3}mm)))",
            rules.min_via_drill.to_mm()
        );
        wln!(
            s,
            "(rule \"microvia_hole_min\" (condition \"A.Type == 'via' && A.Via_Type == 'Micro'\") (constraint hole_size (min {:.3}mm)))",
            rules.microvia_drill.to_mm()
        );
        wln!(
            s,
            "(rule \"annular_min\" (condition \"(A.Type == 'via' && A.Via_Type != 'Micro') || A.Type == 'pad'\") (constraint annular_width (min {:.3}mm)))",
            rules.min_annular.to_mm()
        );
        wln!(
            s,
            "(rule \"microvia_annular_min\" (condition \"A.Type == 'via' && A.Via_Type == 'Micro'\") (constraint annular_width (min {:.3}mm)))",
            rules.microvia_annular.to_mm()
        );
        wln!(
            s,
            "(rule \"via_min\" (condition \"A.Type == 'via' && A.Via_Type != 'Micro'\") (constraint via_diameter (min {:.3}mm)))",
            rules.via_diameter().to_mm()
        );
        wln!(
            s,
            "(rule \"microvia_min\" (condition \"A.Type == 'via' && A.Via_Type == 'Micro'\") (constraint via_diameter (min {:.3}mm)))",
            rules.microvia_diameter().to_mm()
        );
    } else {
        wln!(
            s,
            "(rule \"via_min\" (constraint via_diameter (min {:.3}mm)))",
            rules.via_diameter().to_mm()
        );
        wln!(
            s,
            "(rule \"hole_min\" (constraint hole_size (min {:.3}mm)))",
            rules.min_via_drill.to_mm()
        );
        wln!(
            s,
            "(rule \"annular_min\" (constraint annular_width (min {:.3}mm)))",
            rules.min_annular.to_mm()
        );
    }
    // HV creepage: HV-class copper keeps the creepage distance from any non-HV copper. Matches the
    // physics cost the router already routed to; here it becomes a checkable DRC rule.
    wln!(
        s,
        "(rule \"HV_creepage\" (condition \"A.NetClass == 'HV' && B.NetClass != 'HV'\") (constraint clearance (min {:.3}mm)))",
        creepage.hv_clearance.to_mm()
    );
    s
}

// write_kicad_dru is pure rules-string emission — no Board or schematic dependency. The DRU
// format declares the design-rule minima kicad-cli enforces against, not the board's geometry;
// siblings `pcb_emit` and `sch_emit` consume a Board because they emit the board/schematic
// geometry, but a DRU is closer to a `.kicad_pro` than to either.
