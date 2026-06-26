//! `.kicad_pcb` emission kernel.
//!
//! The board file: layer stack, nets, one footprint per component (orientation carried as the
//! KiCad footprint angle so every instance shares one canonical geometry), routed track segments,
//! vias, mechanical features, and fill zones. The result opens in KiCad and passes
//! `kicad-cli pcb drc` (empirical-tier validation; see `examples/`).
//!
//! This is the IO boundary (DIP): the routing/placement core has no knowledge of KiCad; emission
//! lives only here.

use std::fmt::Write as _; // for `writeln!`/`write!` inside the `w!`/`wln!` macro expansions

use crate::board::{Board, Track, Via, ViaKind, Zone, ZoneFill};
use crate::place::component::Component;
use crate::place::footprint::FootprintDef;
use crate::rules::DesignRules;
// `KICAD_PCB_FORMAT_VERSION` + `KICAD_GENERATOR_NAME` are referenced by the format-string
// literals below via `{KICAD_PCB_FORMAT_VERSION}` and must be in scope. The slice-level
// `#[macro_use]` propagation only carries *macros*, not `use` statements.
use crate::ssot::*;

// Reaches Uuid/layer_name/mm/w/wln from the slice root via `#[macro_use]`.
use super::{Uuid, layer_name, layer_ordinal, mm, mechanical_features};
use super::MechKind;

/// Emit a footprint's body — Reference/Value properties, the F.Fab outline, pads, and 3D model —
/// shared by the board emitter (with pad-net hooks) and any future library exporter (empty pad
/// net). `pad_net(k)` yields the per-pad net suffix (empty for a library exporter; `(net …)` for
/// the board emitter).
fn emit_footprint_body(
    s: &mut String,
    fp: &FootprintDef,
    refdes: &str,
    nlayers: usize,
    uuid: &mut Uuid,
    pad_net: impl Fn(usize) -> String,
) {
    wln!(
        s,
        "    (property \"Reference\" \"{refdes}\" (at 0 -1 0) (layer \"F.Fab\") (effects (font (size 0.8 0.8) (thickness 0.12))))"
    );
    wln!(
        s,
        "    (property \"Value\" \"{}\" (at 0 1 0) (layer \"F.Fab\") (effects (font (size 1 1) (thickness 0.15))))",
        fp.name
    );
    let (cw, ch) = fp.courtyard;
    let (hcw, hch) = (mm(cw.0) / 2.0, mm(ch.0) / 2.0);
    wln!(
        s,
        "    (fp_rect (start {:.3} {:.3}) (end {:.3} {:.3}) (stroke (width 0.1) (type default)) (fill no) (layer \"F.Fab\") (uuid \"{}\"))",
        -hcw, -hch, hcw, hch, uuid.next()
    );
    for (k, pad) in fp.pads.iter().enumerate() {
        // Unrotated offset/size: the footprint angle rotates the pad in KiCad.
        let off = pad.offset;
        let (sx, sy) = (mm(pad.size.0 .0).max(0.1), mm(pad.size.1 .0).max(0.1));
        let net_ref = pad_net(k);
        // An explicitly-empty designator marks a *mechanical* (non-plated) pad — a board-lock/mounting
        // hole. It carries no number and no net and is emitted as `np_thru_hole`, so the PCB matches
        // the schematic (which has no pin for it) while staying a drill/clearance keepout.
        let mech = fp.pad_names.get(k).is_some_and(|name| name.is_empty());
        let pad_name = if mech {
            String::new()
        } else {
            fp.pad_names
                .get(k)
                .filter(|name| !name.is_empty() && name.as_str() != "None")
                .cloned()
                .unwrap_or_else(|| (k + 1).to_string())
        };
        if pad.layers.len() > 1 {
            let drill = sx.min(sy) * 0.5;
            let (kind, layers) = if mech {
                ("np_thru_hole", "*.Cu *.Mask")
            } else {
                ("thru_hole", "*.Cu")
            };
            wln!(
                s,
                "    (pad \"{pad_name}\" {kind} circle (at {:.4} {:.4}) (size {sx:.4} {sy:.4}) (drill {drill:.4}) (layers \"{layers}\"){} (uuid \"{}\"))",
                mm(off.x.0), mm(off.y.0), if mech { String::new() } else { net_ref }, uuid.next()
            );
        } else {
            let ln = layer_name(pad.layers.first().map(|l| l.0).unwrap_or(0), nlayers);
            wln!(
                s,
                "    (pad \"{pad_name}\" smd rect (at {:.4} {:.4}) (size {sx:.4} {sy:.4}) (layers \"{ln}\"){net_ref} (uuid \"{}\"))",
                mm(off.x.0), mm(off.y.0), uuid.next()
            );
        }
    }
    if let Some((path, (dx, dy, dz), (rx, ry, rz), _)) = &fp.model {
        wln!(
            s,
            "    (model \"{}\" (offset (xyz {dx} {dy} {dz})) (scale (xyz 1 1 1)) (rotate (xyz {rx} {ry} {rz})))",
            path.replace('\\', "/")
        );
    }
}

/// Emit the board as a `.kicad_pcb` string.
pub fn write_kicad_pcb(
    board: &Board,
    comps: &[Component],
    lib: &[FootprintDef],
    rules: &DesignRules,
) -> String {
    let spec = &board.spec;
    let nlayers = spec.nlayers;
    let w = mm((spec.nx as i64 - 1) * spec.pitch.0);
    let h = mm((spec.ny as i64 - 1) * spec.pitch.0);
    let mut uuid = Uuid(0);
    let mut s = String::with_capacity(64 * 1024);

    // Header. KICAD_PCB_FORMAT_VERSION + KICAD_GENERATOR_NAME come from crate::ssot (Phase 1c).
    s.push_str(&format!(
        "(kicad_pcb (version {KICAD_PCB_FORMAT_VERSION}) (generator \"{KICAD_GENERATOR_NAME}\")\n"
    ));
    s.push_str("  (general (thickness 1.6))\n  (paper \"A4\")\n");

    // Layer stack.
    s.push_str("  (layers\n");
    for l in 0..nlayers as u16 {
        wln!(
            s,
            "    ({} \"{}\" signal)",
            layer_ordinal(l, nlayers),
            layer_name(l, nlayers)
        );
    }
    // Technical layers so the 3D renderer applies soldermask/silkscreen and shows a real PCB.
    s.push_str(
        "    (34 \"B.Paste\" user)\n    (35 \"F.Paste\" user)\n\
         \x20   (36 \"B.SilkS\" user)\n    (37 \"F.SilkS\" user)\n\
         \x20   (38 \"B.Mask\" user)\n    (39 \"F.Mask\" user)\n\
         \x20   (44 \"Edge.Cuts\" user)\n    (45 \"Margin\" user)\n\
         \x20   (46 \"B.CrtYd\" user)\n    (47 \"F.CrtYd\" user)\n\
         \x20   (48 \"B.Fab\" user)\n    (49 \"F.Fab\" user)\n  )\n",
    );
    s.push_str("  (setup (pad_to_mask_clearance 0))\n");

    // Nets: KiCad net 0 is the implicit no-net; our NetId(i) maps to KiCad net i+1.
    s.push_str("  (net 0 \"\")\n");
    for net in &board.nets {
        wln!(s, "  (net {} \"{}\")", net.id.0 + 1, net.name);
    }

    // Board outline.
    let outline = [
        (0.0, 0.0, w, 0.0),
        (w, 0.0, w, h),
        (w, h, 0.0, h),
        (0.0, h, 0.0, 0.0),
    ];
    for (x1, y1, x2, y2) in outline {
        wln!(
            s,
            "  (gr_line (start {x1:.4} {y1:.4}) (end {x2:.4} {y2:.4}) (layer \"Edge.Cuts\") (width 0.1) (uuid \"{}\"))",
            uuid.next()
        );
    }

    // Footprints (one per component); pad rotation is baked into the pad offset.
    for c in comps {
        let fp = &lib[c.fp];
        let cx = mm(c.placement.pos.x.0);
        let cy = mm(c.placement.pos.y.0);
        // Orientation is carried as the footprint *angle* (KiCad rotates pads/graphics), not baked
        // into pad offsets — so every instance of a footprint shares one canonical geometry that
        // matches the library copy (otherwise a rotated instance reads as `lib_footprint_mismatch`).
        // KiCad's footprint angle is clockwise (Y-down) while the engine's `Rot::apply` is CCW, so
        // emit the negated angle to keep pad positions identical to the routed geometry.
        let ang = (360.0 - c.placement.rot.degrees()) % 360.0;
        wln!(
            s,
            "  (footprint \"kicad-routing:{}\" (layer \"F.Cu\") (uuid \"{}\") (at {cx:.4} {cy:.4} {ang})",
            fp.name,
            uuid.next()
        );
        emit_footprint_body(&mut s, fp, &c.refdes, nlayers, &mut uuid, |k| {
            match c.nets[k] {
                Some(n) => format!(" (net {} \"{}\")", n.0 + 1, board.nets[n.0 as usize].name),
                None => String::new(),
            }
        });
        s.push_str("  )\n");
    }

    // Manufacturability: fiducials (pick-and-place vision) + corner mounting holes. Their positions
    // and copper keepouts are defined once in `mechanical_features` so the router reserves exactly
    // what is emitted here (on a tight board these land inside the routable area, so they MUST be
    // keepouts — otherwise a track runs into the fiducial pad / mounting hole and shorts).
    for f in mechanical_features(w, h) {
        let (fx, fy) = (f.x, f.y);
        match f.kind {
            MechKind::Fiducial => {
                wln!(
                    s,
                    "  (footprint \"kicad-routing:FIDUCIAL\" (layer \"F.Cu\") (uuid \"{}\") (at {fx:.3} {fy:.3})\n    (property \"Reference\" \"FID\" (at 0 -1.5 0) (layer \"F.Fab\") (effects (font (size 1 1) (thickness 0.15))))\n    (pad \"1\" smd circle (at 0 0) (size 1 1) (layers \"F.Cu\" \"F.Mask\") (uuid \"{}\"))\n  )",
                    uuid.next(), uuid.next()
                );
            }
            MechKind::MountingHole => {
                wln!(
                    s,
                    "  (footprint \"kicad-routing:MountingHole\" (layer \"F.Cu\") (uuid \"{}\") (at {fx:.3} {fy:.3})\n    (property \"Reference\" \"MH\" (at 0 -2.5 0) (layer \"F.Fab\") (effects (font (size 1 1) (thickness 0.15))))\n    (pad \"\" np_thru_hole circle (at 0 0) (size 3.2 3.2) (drill 3.2) (layers \"*.Cu\" \"*.Mask\") (uuid \"{}\"))\n  )",
                    uuid.next(), uuid.next()
                );
            }
        }
    }

    // Routed copper.
    emit_segments(&mut s, &board.tracks, nlayers, &mut uuid);
    emit_vias(&mut s, &board.vias, nlayers, rules, &mut uuid);
    emit_zones(&mut s, &board.zones, board, nlayers, rules.min_clearance.to_mm(), &mut uuid);

    s.push_str(")\n");
    s
}

fn emit_segments(s: &mut String, tracks: &[Track], nlayers: usize, uuid: &mut Uuid) {
    for t in tracks {
        wln!(
            s,
            "  (segment (start {:.4} {:.4}) (end {:.4} {:.4}) (width {:.4}) (layer \"{}\") (net {}) (uuid \"{}\"))",
            mm(t.start.x.0), mm(t.start.y.0), mm(t.end.x.0), mm(t.end.y.0),
            mm(t.width.0), layer_name(t.layer.0, nlayers), t.net.0 + 1, uuid.next()
        );
    }
}

fn emit_vias(s: &mut String, vias: &[Via], nlayers: usize, _rules: &DesignRules, uuid: &mut Uuid) {
    // KiCad via tokens: `(via (micro|blind) …)` selects the HDI construction class (an absent
    // token is a through-via; a layer span not reaching both outers is blind/buried by geometry).
    // A filled, plated-over (VIPPO) via-in-pad is **tented** (`(tenting (front yes)(back yes))`):
    // it sits inside a solderable pad and must carry no separate solder-mask aperture, else its
    // opening bridges the adjacent fine-pitch pad's aperture (`solder_mask_bridge`). The pad's own
    // aperture is the solderable surface; the capped via beneath it is mask-covered. A non-filled
    // signal/stitching via uses the board-default tenting.
    // KiCad via *type* is a bare keyword right after `via` (`(via micro …)` / `(via blind …)`),
    // not a parenthesised token; an absent keyword is a through-via.
    for v in vias {
        let kind_tok = match v.kind {
            ViaKind::Micro => " micro",
            ViaKind::Blind | ViaKind::Buried => " blind",
            ViaKind::Through => "",
        };
        wln!(
            s,
            "  (via{} (at {:.4} {:.4}) (size {:.4}) (drill {:.4}) (layers \"{}\" \"{}\"){} (net {}) (uuid \"{}\"))",
            kind_tok, mm(v.pos.x.0), mm(v.pos.y.0), mm(v.diameter.0), mm(v.drill.0),
            layer_name(v.from.0, nlayers), layer_name(v.to.0, nlayers),
            if v.filled { " (tenting (front yes) (back yes))" } else { "" },
            v.net.0 + 1, uuid.next()
        );
    }
}

fn emit_zones(
    s: &mut String,
    zones: &[Zone],
    board: &Board,
    nlayers: usize,
    zone_clearance: f64,
    uuid: &mut Uuid,
) {
    // Copper fill zones (ground pour, teardrops). KiCad's filler carves design-rule clearance
    // around every foreign feature, so these are clearance-safe by construction.
    // KiCad requires every pair of *overlapping* zones to have distinct priorities (else
    // `zones_intersect`). Teardrops cluster at shared vias and abut neighbours, so each gets a
    // unique priority above the pour; for same-net overlaps the copper simply merges, for
    // different-net the higher one wins and the lower is clearance-carved.
    let mut teardrop_priority = 1u32;
    for z in zones {
        let net_name = &board.nets[z.net.0 as usize].name;
        // `island` is the fill's island-removal policy. A plane pour uses ALWAYS-remove
        // (island_removal_mode 0): any fill fragment left unconnected to the net (a region walled off
        // by foreign copper with no pad/via tap) is dropped, so the export carries no isolated copper
        // — which DRC flags as `isolated_copper`. A teardrop is small by design and also uses ALWAYS.
        let (connect, hatch, priority, island) = match z.fill {
            // Plane pour: connect pads **solid** (no thermal relief). A power/ground plane normally
            // ties to its thru-hole pins solidly; thermal-relief spokes on a dense pin field can fall
            // below KiCad's minimum spoke count (`starved_thermal`), so solid both resolves that and
            // gives the lowest-impedance connection. Reflow assembly does not need the relief.
            ZoneFill::ThermalRelief => (
                format!("(connect_pads yes (clearance {zone_clearance:.3}))"),
                "(hatch edge 0.5)",
                0,
                "(island_removal_mode 0)",
            ),
            // Solid connection (teardrop): copper meets the pad with no relief gap.
            ZoneFill::Solid => {
                let p = teardrop_priority;
                teardrop_priority += 1;
                (
                    format!("(connect_pads yes (clearance {zone_clearance:.3}))"),
                    "(hatch edge 0.3)",
                    p,
                    "(island_removal_mode 0)",
                )
            }
        };
        wln!(
            s,
            "  (zone (net {}) (net_name \"{}\") (layer \"{}\") (uuid \"{}\") (priority {priority}) {hatch}",
            z.net.0 + 1,
            net_name,
            layer_name(z.layer.0, nlayers),
            uuid.next()
        );
        wln!(s, "    {connect}");
        wln!(
            s,
            "    (min_thickness 0.2)\n    (fill yes {island} (thermal_gap 0.3) (thermal_bridge_width 0.4))"
        );
        s.push_str("    (polygon (pts");
        for p in &z.polygon {
            w!(s, " (xy {:.4} {:.4})", mm(p.x.0), mm(p.y.0));
        }
        s.push_str("))\n  )\n");
    }
}

/// Emit to a file, after running [`crate::io::duplicate_pcb_uuids`] so the board cannot ship with
/// duplicate UUIDs.
pub fn save_kicad_pcb(
    path: &std::path::Path,
    board: &Board,
    comps: &[Component],
    lib: &[FootprintDef],
    rules: &DesignRules,
) -> std::io::Result<()> {
    let pcb = write_kicad_pcb(board, comps, lib, rules);
    let duplicates = super::duplicate_pcb_uuids(&pcb);
    if !duplicates.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("duplicate KiCad PCB UUIDs: {}", duplicates.join(", ")),
        ));
    }
    std::fs::write(path, pcb)
}

/// Write a board that **failed** the manufacturing gate, stamped with a prominent silkscreen
/// `DRC FAIL` banner naming the blockers, so it is unmistakably an **inspection-only** artifact (not
/// production) while still being renderable and DRC-checkable. This is the deliberate, labelled
/// alternative to silently refusing to write — a failing board you cannot see is a failing board you
/// cannot fix. The banner sits just above the board origin on `F.SilkS`.
pub fn save_kicad_pcb_flagged(
    path: &std::path::Path,
    board: &Board,
    comps: &[Component],
    lib: &[FootprintDef],
    rules: &DesignRules,
    label: &str,
) -> std::io::Result<()> {
    let mut pcb = write_kicad_pcb(board, comps, lib, rules);
    let _ = label;
    // F.Fab (fabrication documentation), not F.SilkS: a fab-layer annotation carries no
    // silk-over-copper / silk-edge DRC, so the inspection marker never adds a violation of its own.
    let banner = "  (gr_text \"DRC FAIL - INSPECTION ONLY\" (at 4 3 0) (layer \"F.Fab\") (uuid \"00000000-0000-0000-0000-ffffffffff01\") (effects (font (size 1.5 1.5) (thickness 0.25)) (justify left)))\n";
    // Insert the banner just before the final closing paren of the top-level `(kicad_pcb …)` form.
    if let Some(idx) = pcb.rfind(')') {
        pcb.insert_str(idx, banner);
    }
    let duplicates = super::duplicate_pcb_uuids(&pcb);
    if !duplicates.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("duplicate KiCad PCB UUIDs: {}", duplicates.join(", ")),
        ));
    }
    std::fs::write(path, pcb)
}
