//! `.kicad_sch` emission — KiCad schematic with global-label connectivity.
//!
//! Each component becomes a boxed symbol with one pin per pad; connectivity is carried by
//! **global labels** at every pin (same net name ⇒ same net), so no wire auto-routing is
//! needed and the netlist matches the board exactly. ERC-checkable.
//!
//! Components are placed on the 1.27 mm (50-mil) connection grid so every pin lands on-grid
//! without ERC `lib_sym_off_grid` warnings.

use std::fmt::Write as _; // for `writeln!`/`write!` inside the `w!`/`wln!` macro expansions

use crate::board::Board;
use crate::place::component::Component;
use crate::place::footprint::FootprintDef;
// `KICAD_SCH_FORMAT_VERSION` + `KICAD_GENERATOR_NAME` are referenced by the format-string
// literals below via `{KICAD_SCH_FORMAT_VERSION}` and must be in scope. The slice-level
// `#[macro_use]` propagation only carries *macros*, not `use` statements.
use crate::ssot::*;

// `w!` / `wln!` macros are declared at the slice root (`src/io/mod.rs`) and inherited
// into this submodule via the `#[macro_use] pub mod sch_emit;` declaration at the bottom
// of `mod.rs`. The `Uuid` type is a `pub(super)` item, so it IS a regular `use` candidate.
use super::Uuid;

/// Emit a KiCad schematic (`.kicad_sch`). Each component becomes a boxed symbol with one pin per
/// pad; connectivity is carried by **global labels** at every pin (same net name ⇒ same net), so no
/// wire auto-routing is needed and the netlist matches the board exactly. ERC-checkable.
pub fn write_kicad_sch(board: &Board, comps: &[Component], lib: &[FootprintDef]) -> String {
    let mut uuid = Uuid(0);
    let mut s = String::with_capacity(64 * 1024);
    s.push_str(&format!(
        "(kicad_sch (version {KICAD_SCH_FORMAT_VERSION}) (generator \"{KICAD_GENERATOR_NAME}\") (generator_version \"8.0\")\n"
    ));
    wln!(s, "  (uuid \"{}\")", uuid.next());
    s.push_str("  (paper \"A1\")\n");

    // Pin pitch / box geometry (mm). Documented in crate::ssot's `pub const` header.
    let pitch = 2.54;
    let hw = 7.62; // box half-width
    let pin_x = -(hw + 2.54); // pin connection-point x in symbol space (pin points right into box)
    let top = |n: usize| (n as f64 - 1.0) * pitch / 2.0; // first pin y (symbol space, +Y up)
    // A mechanical pad (empty designator — a non-plated board-lock/mounting hole) is not an electrical
    // pin: it gets no schematic pin, so it cannot be a `pin_not_connected` ERC fault. It remains on the
    // PCB as a drill/clearance keepout. Abstraction footprints carry no `pad_names`, so every pad of
    // theirs stays electrical (the check only fires on an explicitly-empty imported designator).
    let is_mech = |fp: &FootprintDef, k: usize| fp.pad_names.get(k).is_some_and(|nm| nm.is_empty());

    // --- lib_symbols ------------------------------------------------------------------------------
    s.push_str("  (lib_symbols\n");
    for fp in lib {
        let n = fp.pads.len().max(1);
        let hh = top(n) + pitch;
        wln!(
            s,
            "    (symbol \"kicad-routing:{}\" (exclude_from_sim no) (in_bom yes) (on_board yes)",
            fp.name
        );
        wln!(
            s,
            "      (property \"Reference\" \"U\" (at 0 {:.2} 0) (effects (font (size 1.27 1.27))))",
            hh + 1.27
        );
        wln!(
            s,
            "      (property \"Value\" \"{}\" (at 0 {:.2} 0) (effects (font (size 1.27 1.27))))",
            fp.name,
            -(hh + 1.27)
        );
        wln!(s, "      (symbol \"{}_0_1\" (rectangle (start {:.2} {:.2}) (end {:.2} {:.2}) (stroke (width 0.254) (type default)) (fill (type background))))", fp.name, -hw, hh, hw, -hh);
        wln!(s, "      (symbol \"{}_1_1\"", fp.name);
        for k in 0..n {
            if is_mech(fp, k) {
                continue; // mechanical hole — no electrical pin
            }
            let py = top(n) - k as f64 * pitch;
            wln!(
                s,
                "        (pin passive line (at {pin_x:.2} {py:.2} 0) (length 2.54) (name \"P{k}\" (effects (font (size 1.0 1.0)))) (number \"{}\" (effects (font (size 1.0 1.0)))))",
                k + 1
            );
        }
        s.push_str("      )\n    )\n");
    }
    s.push_str("  )\n");

    // --- symbol instances + global labels ---------------------------------------------------------
    // Lay components out on the 1.27 mm (50-mil) connection grid; labels carry the netlist. All
    // origins/offsets are 1.27 mm multiples so every pin lands on-grid (no ERC off-grid warnings).
    let cols = 6usize;
    let col_w = 38.1; // 30 × 1.27
    let row_h = 63.5; // 50 × 1.27
    for (i, c) in comps.iter().enumerate() {
        let fp = &lib[c.fp];
        let n = fp.pads.len().max(1);
        let sx = 12.7 + (i % cols) as f64 * col_w;
        let sy = 12.7 + (i / cols) as f64 * row_h;
        wln!(
            s,
            "  (symbol (lib_id \"kicad-routing:{}\") (at {sx:.2} {sy:.2} 0) (unit 1) (exclude_from_sim no) (in_bom yes) (on_board yes) (uuid \"{}\")",
            fp.name, uuid.next()
        );
        wln!(s, "    (property \"Reference\" \"{}\" (at {sx:.2} {:.2} 0) (effects (font (size 1.27 1.27))))", c.refdes, sy - top(n) - 3.0);
        wln!(s, "    (property \"Value\" \"{}\" (at {sx:.2} {:.2} 0) (effects (font (size 1.27 1.27))))", fp.name, sy + top(n) + 3.0);
        for k in 0..n {
            if is_mech(fp, k) {
                continue; // mechanical hole — no electrical pin
            }
            wln!(s, "    (pin \"{}\" (uuid \"{}\"))", k + 1, uuid.next());
        }
        s.push_str("    (instances (project \"tile\" (path \"/\" (reference \"");
        w!(s, "{}", c.refdes);
        s.push_str("\") (unit 1))))\n  )\n");

        // Global labels at each pin's sheet connection point (symbol +Y up ⇒ sheet y = sy - ly).
        for k in 0..n {
            let ly = top(n) - k as f64 * pitch;
            let lx_sheet = sx + pin_x;
            let ly_sheet = sy - ly;
            let net = c.nets[k].map(|nid| board.nets[nid.0 as usize].name.clone());
            if let Some(name) = net {
                wln!(
                    s,
                    "  (global_label \"{name}\" (shape input) (at {lx_sheet:.2} {ly_sheet:.2} 180) (effects (font (size 1.27 1.27)) (justify right)) (uuid \"{}\"))",
                    uuid.next()
                );
            }
        }
    }

    s.push_str("  (sheet_instances (path \"/\" (page \"1\")))\n)\n");
    s
}

/// Emit the schematic to a file.
pub fn save_kicad_sch(
    path: &std::path::Path,
    board: &Board,
    comps: &[Component],
    lib: &[FootprintDef],
) -> std::io::Result<()> {
    std::fs::write(path, write_kicad_sch(board, comps, lib))
}
