//! `.kicad_pro` emission — minimal KiCad project descriptor.
//!
//! Declares the **library-sync** checks as ignored (the engine emits footprints and symbols
//! inline-with-board, so there is no external library to drift from) and pushes the engine's
//! actual design-rule minimums into the project's `design_settings.rules`. The latter is
//! critical: without it, kicad-cli falls back to its conservative defaults (0.2 mm track,
//! 0.3 mm hole, 0.5 mm via) — stricter than the holohv process the engine routes to.
//!
//! Every *real* DRC/ERC rule (clearance, shorts, unconnected, hole spacing, …) stays at its
//! default severity; only the `lib_*` checks are silenced and the design minima are declared
//! SSOT.

use crate::rules::DesignRules;

/// Emit a minimal KiCad project (`.kicad_pro`) for `basename` (the file stem) that declares the
/// **library-sync** checks as ignored: `lib_footprint_issues` / `lib_footprint_mismatch` (DRC) and
/// `lib_symbol_issues` (ERC). These compare a board against a separately-maintained component
/// library; for engine-generated boards the footprints and symbols are authoritative-inline and
/// there is no external library to drift from, so the checks are not applicable. Every *real* DRC/ERC
/// rule (clearance, shorts, unconnected, hole spacing, …) stays at its default severity. KiCad fills
/// all other project settings with defaults.
#[must_use]
pub fn write_kicad_pro(basename: &str, rules: &DesignRules) -> String {
    // Emit the engine's *actual* design-rule minimums into the project's `design_settings.rules`,
    // where KiCad natively stores them. Without this the board carries no constraints and kicad-cli
    // falls back to its conservative defaults (0.2 mm track, 0.3 mm hole, 0.5 mm via) — stricter than
    // the holoHV process the engine routes to (0.15 / 0.2 / 0.46 mm) — so every track, drill and via
    // is falsely flagged `track_width` / `drill_out_of_range` / `via_diameter`. Declaring the real
    // process is SSOT, not masking: the board states the fab capability it was designed for.
    //
    // Each `{{` / `}}` is **one** literal `{` / `}` in the resulting JSON. The format-string is
    // sliced into one-`{{`/one-`}}` per concat-!() line so any future brace-pair imbalance is
    // visually obvious without further human counting.
    let r = rules;
    let body = format!(
        concat!(
            "{{\n",
            "  \"board\": {{\n",
            "    \"design_settings\": {{\n",
            "      \"rule_severities\": {{\n",
            "        \"lib_footprint_issues\": \"ignore\",\n",
            "        \"lib_footprint_mismatch\": \"ignore\"\n",
            "      }},\n",
            "      \"rules\": {{\n",
            "        \"min_clearance\": {:.3},\n",
            "        \"min_copper_edge_clearance\": {:.3},\n",
            "        \"min_hole_clearance\": {:.3},\n",
            "        \"min_hole_to_hole\": {:.3},\n",
            "        \"min_microvia_diameter\": {:.3},\n",
            "        \"min_microvia_drill\": {:.3},\n",
            "        \"min_through_hole_diameter\": {:.3},\n",
            "        \"min_track_width\": {:.3},\n",
            "        \"min_via_annular_width\": {:.3},\n",
            "        \"min_via_diameter\": {:.3}\n",
            "      }}\n",
            "    }}\n",
            "  }},\n",
            "  \"erc\": {{\n",
            "    \"rule_severities\": {{\n",
            "      \"lib_symbol_issues\": \"ignore\"\n",
            "    }}\n",
            "  }},\n",
            "  \"meta\": {{\n",
            "    \"filename\": \"{basename}.kicad_pro\",\n",
            "    \"version\": 3\n",
            "  }}\n",
            "}}\n",
        ),
        r.min_clearance.to_mm(),
        r.edge_clearance.to_mm(),
        r.hole_clearance().to_mm(),
        r.min_via_drill.to_mm(), // hole-to-hole floored at the mechanical drill diameter
        r.microvia_diameter().to_mm(),
        r.microvia_drill.to_mm(),
        // TODO(Phase 4a follow-up): the slot labelled `min_through_hole_diameter` is currently
        // sourced from `r.min_via_drill.to_mm()`. Through-hole DIAMETER is a different physical
        // value from via DRILL; replace with the appropriate `DesignRules` accessor (or a
        // computation such as `min_via_drill * 1.05`) once the rules facade exposes it. Slot #4
        // (`min_hole_to_hole`) is intentional — hole-to-hole spacing is floored at the
        // mechanical drill diameter per the comment above.
        r.min_via_drill.to_mm(),
        r.min_track.to_mm(),
        r.min_annular.to_mm(),
        r.via_diameter().to_mm(),
        basename = basename,
    );
    body
}
