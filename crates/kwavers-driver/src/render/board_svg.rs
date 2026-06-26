//! Native SVG renderer for routed PCB boards.
//!
//! Produces a self-contained vector image from the in-memory board model without any external
//! tool dependency. The output opens in any web browser, Inkscape, or vector graphics editor.
//!
//! # Layer colour scheme
//!
//! | Layer | Colour |
//! |-------|--------|
//! | F.Cu (0) | `#CC3333` red |
//! | In1.Cu | `#CC8833` orange |
//! | In2.Cu | `#33BB33` green |
//! | In3.Cu | `#5555CC` blue |
//! | In4.Cu | `#AA33AA` purple |
//! | B.Cu (last) | `#3388CC` cyan-blue |
//! | Inner > In4 | cycle the inner palette |
//!
//! Evidence tier: visual/empirical — the geometry is derived from the same board model used by
//! the LVS and DRC, so the render faithfully represents the routed copper.

use std::fmt::Write as _;

use crate::board::Board;
use crate::place::component::Component;
use crate::place::footprint::FootprintDef;
use crate::place::rotation::Rot;

/// Convert nanometres to millimetres for SVG coordinates.
#[inline]
fn nm(v: i64) -> f64 {
    v as f64 * 1.0e-6
}

/// Copper-layer stroke/fill colour.
fn layer_color(l: u16, nlayers: usize) -> &'static str {
    const INNER: &[&str] = &[
        "#CC8833", // In1 orange
        "#33BB33", // In2 green
        "#5555CC", // In3 blue
        "#AA33AA", // In4 purple
        "#33AAAA", // In5 teal
        "#AAAA33", // In6 olive
    ];
    if l == 0 {
        "#CC3333" // F.Cu — red
    } else if l as usize == nlayers - 1 {
        "#3388CC" // B.Cu — cyan-blue
    } else {
        INNER[(l as usize - 1).min(INNER.len() - 1)]
    }
}

/// KiCad copper-layer human name.
fn layer_name(l: u16, nlayers: usize) -> String {
    if l == 0 {
        "F.Cu".into()
    } else if l as usize == nlayers - 1 {
        "B.Cu".into()
    } else {
        format!("In{l}.Cu")
    }
}

macro_rules! wln {
    ($dst:expr, $($arg:tt)*) => {
        writeln!($dst, $($arg)*).expect("String write_fmt is infallible")
    };
}
macro_rules! w {
    ($dst:expr, $($arg:tt)*) => {
        write!($dst, $($arg)*).expect("String write_fmt is infallible")
    };
}

fn svg_escape_text(text: &str) -> String {
    text.chars()
        .flat_map(|ch| match ch {
            '&' => "&amp;".chars().collect::<Vec<_>>(),
            '<' => "&lt;".chars().collect(),
            '>' => "&gt;".chars().collect(),
            '"' => "&quot;".chars().collect(),
            '\'' => "&apos;".chars().collect(),
            _ => vec![ch],
        })
        .collect()
}

/// Render the routed board as an SVG string.
///
/// Layers are painted bottom-to-top so `F.Cu` copper is the topmost visible layer. Via drill
/// holes are white circles. Component courtyard rectangles and reference designators overlay last.
///
/// Write the returned string to a `.svg` file. It opens in any browser — no external tool needed.
#[must_use]
pub fn render_board_svg(board: &Board, comps: &[Component], lib: &[FootprintDef]) -> String {
    let spec = board.spec;
    let nlayers = spec.nlayers;

    // Board bounding box in mm. The grid spans cells 0…(nx-1), so the physical width
    // is (nx-1) × pitch (matching the convention used by block_mechanical and GridSpec::point_of).
    let bw = nm((spec.nx as i64 - 1) * spec.pitch.0);
    let bh = nm((spec.ny as i64 - 1) * spec.pitch.0);
    let pad = 6.0_f64;
    let vx0 = -pad;
    let vy0 = -pad;
    let vw = bw + 2.0 * pad;
    let vh = bh + 2.0 * pad;
    // 4 SVG units per mm at 96 dpi ≈ 384 ppi — a sharp on-screen default.
    let display_w = vw * 4.0;
    let display_h = vh * 4.0;

    let mut s = String::with_capacity(512 * 1024);

    // ── Header ───────────────────────────────────────────────────────────────────────────────────
    wln!(s, r##"<?xml version="1.0" encoding="UTF-8"?>"##);
    wln!(
        s,
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vx:.3} {vy:.3} {vw:.3} {vh:.3}" width="{dw:.0}mm" height="{dh:.0}mm">"##,
        vx = vx0,
        vy = vy0,
        vw = vw,
        vh = vh,
        dw = display_w,
        dh = display_h
    );
    wln!(
        s,
        "  <title>PCB Board Render ({} tracks, {} vias, {} nets)</title>",
        board.tracks.len(),
        board.vias.len(),
        board.nets.len()
    );
    wln!(
        s,
        "  <desc>kicad-routing render_board_svg — coordinates in mm</desc>"
    );

    // ── Background ───────────────────────────────────────────────────────────────────────────────
    wln!(
        s,
        "  <rect x=\"{:.3}\" y=\"{:.3}\" width=\"{:.3}\" height=\"{:.3}\" fill=\"#111111\"/>",
        vx0,
        vy0,
        vw,
        vh
    );

    // ── Board outline (Edge.Cuts) ────────────────────────────────────────────────────────────────
    wln!(s,
        "  <rect x=\"0\" y=\"0\" width=\"{:.3}\" height=\"{:.3}\" fill=\"none\" stroke=\"#FFFF00\" stroke-width=\"0.12\" stroke-dasharray=\"1,0.4\"/>",
        bw, bh);

    // ── Zone fills (semi-transparent copper pours) ───────────────────────────────────────────────
    if !board.zones.is_empty() {
        wln!(s, "  <!-- Zone fills -->");
        wln!(s, "  <g id=\"zones\" opacity=\"0.20\">");
        for z in &board.zones {
            if z.polygon.len() < 3 {
                continue;
            }
            let color = layer_color(z.layer.0, nlayers);
            w!(s, "    <polygon fill=\"{color}\" points=\"");
            for p in &z.polygon {
                w!(s, "{:.4},{:.4} ", nm(p.x.0), nm(p.y.0));
            }
            wln!(s, "\"/>");
        }
        wln!(s, "  </g>");
    }

    // ── Copper tracks (bottom layer first → F.Cu topmost) ───────────────────────────────────────
    wln!(s, "  <!-- Copper tracks -->");
    for layer_rev in 0..nlayers {
        let l = (nlayers - 1 - layer_rev) as u16;
        if !board.tracks.iter().any(|t| t.layer.0 == l) {
            continue;
        }
        let color = layer_color(l, nlayers);
        wln!(s,
            "  <g id=\"l{l}\" stroke=\"{color}\" fill=\"none\" stroke-linecap=\"round\" stroke-linejoin=\"round\">");
        for t in &board.tracks {
            if t.layer.0 != l {
                continue;
            }
            let w_mm = nm(t.width.0);
            wln!(s,
                "    <line x1=\"{:.4}\" y1=\"{:.4}\" x2=\"{:.4}\" y2=\"{:.4}\" stroke-width=\"{:.4}\"/>",
                nm(t.start.x.0), nm(t.start.y.0),
                nm(t.end.x.0),   nm(t.end.y.0),
                w_mm);
        }
        wln!(s, "  </g>");
    }

    // ── Vias ─────────────────────────────────────────────────────────────────────────────────────
    if !board.vias.is_empty() {
        wln!(s, "  <!-- Vias -->");
        wln!(s, "  <g id=\"vias\">");
        for v in &board.vias {
            let cx = nm(v.pos.x.0);
            let cy = nm(v.pos.y.0);
            let r_out = nm(v.diameter.0) * 0.5;
            let r_drill = nm(v.drill.0) * 0.5;
            // Annular copper ring.
            wln!(s,
                "    <circle cx=\"{:.4}\" cy=\"{:.4}\" r=\"{:.4}\" fill=\"#CC9933\" stroke=\"none\"/>",
                cx, cy, r_out);
            // Drill hole.
            wln!(s,
                "    <circle cx=\"{:.4}\" cy=\"{:.4}\" r=\"{:.4}\" fill=\"#222222\" stroke=\"none\"/>",
                cx, cy, r_drill.max(r_out * 0.4));
        }
        wln!(s, "  </g>");
    }

    // ── Component pads ───────────────────────────────────────────────────────────────────────────
    // Draw from the placed footprint definitions, not from `board.pads`, because the board terminal
    // list intentionally carries only routing geometry (position/layer/net) and drops pad size.
    // The SVG must show the real footprint copper so connectors do not look like coordinate marks.
    wln!(s, "  <!-- Component pads -->");
    wln!(
        s,
        "  <g id=\"pads\" stroke=\"#111111\" stroke-width=\"0.03\">"
    );
    for comp in comps {
        let fp = &lib[comp.fp];
        for (k, pad) in fp.pads.iter().enumerate() {
            let pos = comp.pad_pos(lib, k);
            let (sw, sh) = comp.placement.rot.apply_size(pad.size);
            let cx = nm(pos.x.0);
            let cy = nm(pos.y.0);
            let w_mm = nm(sw.0);
            let h_mm = nm(sh.0);
            let layer = if pad.layers.len() > 1 {
                0u16
            } else {
                pad.layers.first().map_or(0, |l| l.0)
            };
            let net = comp.nets.get(k).copied().flatten();
            let fill = if net.is_some() {
                layer_color(layer, nlayers)
            } else {
                "#555555"
            };
            let title = svg_escape_text(&format!(
                "{} pad {} at {:.3},{:.3} mm",
                comp.refdes,
                fp.pad_names.get(k).map_or("?", String::as_str),
                cx,
                cy
            ));
            let rx = (w_mm.min(h_mm) * 0.12).min(0.18);
            wln!(
                s,
                "    <rect x=\"{:.4}\" y=\"{:.4}\" width=\"{:.4}\" height=\"{:.4}\" fill=\"{fill}\" rx=\"{:.4}\"><title>{title}</title></rect>",
                cx - w_mm * 0.5,
                cy - h_mm * 0.5,
                w_mm,
                h_mm,
                rx
            );
        }
    }
    wln!(s, "  </g>");

    // ── Component courtyards ──────────────────────────────────────────────────────────────────────
    wln!(s, "  <!-- Component courtyards -->");
    wln!(s,
        "  <g id=\"courtyards\" fill=\"none\" stroke=\"#888888\" stroke-width=\"0.05\" stroke-dasharray=\"0.25,0.12\">");
    for comp in comps {
        let fp = &lib[comp.fp];
        let pos = comp.placement.pos;
        let (cw_nm, ch_nm) = match comp.placement.rot {
            Rot::R0 | Rot::R180 => fp.courtyard,
            Rot::R90 | Rot::R270 => (fp.courtyard.1, fp.courtyard.0),
        };
        let cw = nm(cw_nm.0);
        let ch = nm(ch_nm.0);
        let x = nm(pos.x.0) - cw * 0.5;
        let y = nm(pos.y.0) - ch * 0.5;
        wln!(
            s,
            "    <rect x=\"{:.4}\" y=\"{:.4}\" width=\"{:.4}\" height=\"{:.4}\"/>",
            x,
            y,
            cw,
            ch
        );
    }
    wln!(s, "  </g>");

    // ── Reference designator labels ───────────────────────────────────────────────────────────────
    wln!(s, "  <!-- Reference designators -->");
    wln!(s,
        "  <g id=\"refdes\" font-family=\"monospace\" font-size=\"0.7\" fill=\"#EEEEEE\" text-anchor=\"middle\" dominant-baseline=\"middle\">");
    for comp in comps {
        let fp = &lib[comp.fp];
        let min_side = nm(fp.courtyard.0 .0.min(fp.courtyard.1 .0));
        if min_side < 1.5 {
            continue;
        } // skip tiny parts whose label would overlap pads
        let pos = comp.placement.pos;
        wln!(
            s,
            "    <text x=\"{:.3}\" y=\"{:.3}\">{}</text>",
            nm(pos.x.0),
            nm(pos.y.0),
            &comp.refdes
        );
    }
    wln!(s, "  </g>");

    // ── Layer legend (right of the board) ────────────────────────────────────────────────────────
    {
        let leg_x = bw + 1.5;
        let leg_y = 0.0_f64;
        let row_h = 2.8_f64;
        wln!(s, "  <!-- Layer legend -->");
        wln!(s,
            "  <g id=\"legend\" font-family=\"monospace\" font-size=\"1.4\" fill=\"#DDDDDD\" dominant-baseline=\"middle\">");
        wln!(s, "    <text x=\"{:.1}\" y=\"{:.1}\" font-weight=\"bold\" font-size=\"1.6\">Layers</text>",
            leg_x, leg_y + 1.4);
        for l in 0..nlayers {
            let color = layer_color(l as u16, nlayers);
            let name = layer_name(l as u16, nlayers);
            let ly = leg_y + (l as f64 + 1.5) * row_h;
            wln!(
                s,
                "    <rect x=\"{:.1}\" y=\"{:.1}\" width=\"1.8\" height=\"1.4\" fill=\"{color}\"/>",
                leg_x,
                ly - 0.7
            );
            wln!(
                s,
                "    <text x=\"{:.1}\" y=\"{:.1}\">{name}</text>",
                leg_x + 2.2,
                ly
            );
        }
        wln!(s, "  </g>");
    }

    // ── Statistics footer ─────────────────────────────────────────────────────────────────────────
    wln!(s,
        "  <text x=\"0\" y=\"{:.1}\" font-family=\"monospace\" font-size=\"1.1\" fill=\"#666666\">{} tracks | {} vias | {} nets | {:.1}\u{00d7}{:.1} mm</text>",
        bh + 2.0,
        board.tracks.len(), board.vias.len(), board.nets.len(), bw, bh);

    wln!(s, "</svg>");
    s
}

/// Write [`render_board_svg`] output to `path`.
///
/// Creates parent directories if needed. Returns an error string on I/O failure.
pub fn save_board_svg(
    path: &std::path::Path,
    board: &Board,
    comps: &[Component],
    lib: &[FootprintDef],
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }
    let svg = render_board_svg(board, comps, lib);
    std::fs::write(path, svg.as_bytes())
        .map_err(|e| format!("cannot write {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{LayerId, NetClassKind};
    use crate::geom::{GridSpec, Nm, Point};
    use crate::place::component::{Component, Placement};
    use crate::place::footprint::{PadDef, Role};

    #[test]
    fn render_uses_real_rotated_pad_size() {
        let spec =
            GridSpec::cover(Nm::from_mm(12.0), Nm::from_mm(12.0), Nm::from_mm(1.0), 2).unwrap();
        let mut board = Board::new(spec);
        let net = board.add_net("PWR", NetClassKind::Power);
        let footprint = FootprintDef::new(
            "J",
            (Nm::from_mm(4.0), Nm::from_mm(4.0)),
            Role::Connector,
            vec![PadDef {
                offset: Point::new(Nm(0), Nm(0)),
                size: (Nm::from_mm(3.0), Nm::from_mm(1.0)),
                layers: vec![LayerId(0)],
                power_pin: false,
            }],
        );
        let component = Component {
            fp: 0,
            nets: vec![Some(net)],
            refdes: "J1".into(),
            placement: Placement {
                pos: Point::new(Nm::from_mm(5.0), Nm::from_mm(5.0)),
                rot: Rot::R90,
            },
            assoc_ic: None,
            locked: true,
            ..Default::default()
        };

        let svg = render_board_svg(&board, &[component], &[footprint]);

        assert!(
            svg.contains("width=\"1.0000\" height=\"3.0000\""),
            "R90 pad render must swap footprint pad width/height instead of using fixed pseudo-pad size"
        );
        assert!(
            svg.contains("<title>J1 pad ? at 5.000,5.000 mm</title>"),
            "pad tooltip must identify the real refdes/pad coordinate"
        );
    }
}
