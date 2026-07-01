//! KiCad `.kicad_pcb` → [`Board`] parser.
//!
//! Reads a KiCad PCB file (v6/v7/v8/v9 s-expression format) and reconstructs the [`Board`]
//! model: copper layer stack, net declarations, component pad positions, routed track segments,
//! and vias. The parsed board feeds `audit`, `verify`, and `validate` without any re-routing —
//! ideal for auditing existing designs (e.g. the downloaded KiCad demo boards) against
//! kwavers-driver's physics / DFM checks.
//!
//! # What is parsed
//!
//! | KiCad element | Action |
//! |---|---|
//! | `(layers …)` | Copper-layer count + name→[`LayerId`] map |
//! | `(net id "name")` | Net registry → [`crate::board::NetId`] map keyed by KiCad id |
//! | `(footprint … (at …) (pad …))` | Pad positions (rotated+translated to board space) |
//! | `(segment …)` | [`Track`] segments |
//! | `(via …)` | [`Via`]s |
//! | `(gr_line … (layer "Edge.Cuts"))` | Board outline bounding box → [`GridSpec`] |
//!
//! # What is not parsed
//!
//! Silkscreen, courtyard, fab layer graphics, zone fills, 3D models, DRC markers, and
//! text/reference annotations are ignored — they carry no electrical topology.
//!
//! # Coordinate conventions
//!
//! KiCad stores coordinates in millimetres (`f64`). The kwavers engine stores coordinates in
//! nanometres (`i64`). Conversion: `nm = (mm * 1_000_000.0).round() as i64`. Pad positions are
//! the absolute board-space position after applying the footprint's `(at fp_x fp_y fp_rot)`
//! transform to the pad's relative `(at pad_dx pad_dy)`:
//!
//! ```text
//! theta = fp_rot * PI / 180.0   (degrees → radians)
//! abs_x = fp_x + pad_dx * cos(theta) - pad_dy * sin(theta)
//! abs_y = fp_y + pad_dx * sin(theta) + pad_dy * cos(theta)
//! ```
//!
//! # Grid pitch
//!
//! The routing grid is set to 0.25 mm (250 000 nm) — the kwavers-driver default pitch. All parsed
//! pad/track/via positions are stored at exact nm resolution; the grid is used only for the
//! `GridSpec` cell decomposition.
//!
//! # Evidence tier
//!
//! Value-semantic unit tests (in `crate::io::tests`) parse the KiCad demo boards and assert
//! exact net counts, pad counts, and track counts extracted by hand from the source files.

use std::collections::BTreeMap;

use crate::board::{Board, LayerId, NetClassKind, Pad, Track, Via, ViaKind};
use crate::geom::{GridSpec, Nm, Point};
use crate::place::sexpr::{child, num, parse_sexpr, Sexpr};

// ────────────────────────────────────────────────────────────────────────────
// Grid pitch constant — 0.25 mm matches the kwavers-driver autorouter default.
const PITCH_NM: i64 = 250_000;

// ────────────────────────────────────────────────────────────────────────────
// Internal helpers

/// KiCad mm coordinate → engine nm.
#[inline]
fn nm(mm: f64) -> Nm {
    Nm((mm * 1_000_000.0).round() as i64)
}

/// Parse a `(start x y)` or `(end x y)` child of a node → `(mm_x, mm_y)`.
fn xy(node: &Sexpr, key: &str) -> Option<(f64, f64)> {
    let c = child(node, key)?;
    let list = c.as_list()?;
    Some((num(list, 1)?, num(list, 2)?))
}

// ────────────────────────────────────────────────────────────────────────────
// Copper-layer extraction

/// Parse the `(layers …)` section and return copper layers sorted by ascending KiCad ordinal.
/// Each entry is `(kicad_ordinal, layer_name)`.
fn copper_layers(items: &[Sexpr]) -> Vec<(u32, String)> {
    let Some(layers_node) = items.iter().find(|s| s.head() == Some("layers")) else {
        return vec![];
    };
    let list = match layers_node.as_list() {
        Some(l) => l,
        None => return vec![],
    };
    let mut cu: Vec<(u32, String)> = list
        .iter()
        .skip(1) // skip the "layers" head atom
        .filter_map(|entry| {
            let el = entry.as_list()?;
            let ordinal: u32 = el.first()?.as_atom()?.parse().ok()?;
            let name = el.get(1)?.as_atom()?;
            // Only keep copper layers: name ends with ".Cu" or equals "F.Cu" / "B.Cu"
            if name.ends_with(".Cu") {
                Some((ordinal, name.to_owned()))
            } else {
                None
            }
        })
        .collect();
    cu.sort_by_key(|(ord, _)| *ord);
    cu
}

// ────────────────────────────────────────────────────────────────────────────
// Board-outline bounding box from Edge.Cuts

/// Find the bounding box of all `(gr_line …)` / `(gr_arc …)` elements on the `Edge.Cuts` layer.
/// Returns `(min_x, min_y, max_x, max_y)` in mm, or `None` if no outline was found.
fn edge_cuts_bbox(items: &[Sexpr]) -> Option<(f64, f64, f64, f64)> {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    let mut found = false;
    for item in items {
        let Some(list) = item.as_list() else {
            continue;
        };
        let head = list.first().and_then(|s| s.as_atom()).unwrap_or("");
        if head != "gr_line" && head != "gr_arc" && head != "gr_rect" && head != "gr_circle" {
            continue;
        }
        let s_node = Sexpr::List(list.to_vec());
        // Check the layer is Edge.Cuts
        let layer_str = child(&s_node, "layer")
            .and_then(|l| l.as_list())
            .and_then(|l| l.get(1))
            .and_then(|a| a.as_atom())
            .unwrap_or("");
        if layer_str != "Edge.Cuts" {
            continue;
        }
        found = true;
        // Collect all coordinate atoms: (start x y) and (end x y)
        for coord_key in ["start", "end", "mid"] {
            if let Some((x, y)) = xy(&s_node, coord_key) {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }
        // Handle (at x y) for circles/arcs
        if let Some((x, y)) = xy(&s_node, "at") {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    if found {
        Some((min_x, min_y, max_x, max_y))
    } else {
        None
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Footprint → pad extraction

/// Apply a 2D rotation (degrees, KiCad CW convention) to a relative pad offset.
#[inline]
fn rotate_pad(dx: f64, dy: f64, rot_deg: f64) -> (f64, f64) {
    // KiCad uses clockwise-positive rotation, and standard trig is CCW, so we negate the angle.
    let theta = -rot_deg * std::f64::consts::PI / 180.0;
    let (sin, cos) = theta.sin_cos();
    (dx * cos - dy * sin, dx * sin + dy * cos)
}

/// Parse the copper layer list from a pad's `(layers …)` child.
/// Handles both explicit names (`"F.Cu"`, `"B.Cu"`, `"In1.Cu"`) and wildcards (`"*.Cu"`).
fn pad_copper_layers(pad_node: &Sexpr, layer_map: &BTreeMap<String, LayerId>) -> Vec<LayerId> {
    let Some(layers_node) = child(pad_node, "layers") else {
        return vec![];
    };
    let Some(list) = layers_node.as_list() else {
        return vec![];
    };
    let mut ids: Vec<LayerId> = Vec::new();
    for atom in list.iter().skip(1) {
        let Some(name) = atom.as_atom() else {
            continue;
        };
        if name == "*.Cu" || name == "*.cu" {
            // All copper layers
            ids.extend(layer_map.values().copied());
            break;
        } else {
            if let Some(&id) = layer_map.get(name) {
                ids.push(id);
            }
        }
    }
    ids.sort_by_key(|l| l.0);
    ids.dedup();
    ids
}

/// Extract [`Pad`]s from a `(footprint …)` node.
fn extract_pads(
    fp_node: &Sexpr,
    layer_map: &BTreeMap<String, LayerId>,
    net_map: &BTreeMap<u32, crate::board::NetId>,
    out: &mut Vec<Pad>,
) {
    let Some(fp_list) = fp_node.as_list() else {
        return;
    };
    // Footprint absolute position and rotation.
    let (fp_x, fp_y, fp_rot) = {
        let at = Sexpr::List(fp_list.to_vec());
        let at_node = child(&at, "at");
        match at_node.and_then(|a| a.as_list()) {
            Some(l) => {
                let x = num(l, 1).unwrap_or(0.0);
                let y = num(l, 2).unwrap_or(0.0);
                let rot = num(l, 3).unwrap_or(0.0);
                (x, y, rot)
            }
            None => (0.0, 0.0, 0.0),
        }
    };
    let fp_as_node = Sexpr::List(fp_list.to_vec());
    for item in fp_list.iter().skip(1) {
        if item.head() != Some("pad") {
            continue;
        }
        let Some(pad_list) = item.as_list() else {
            continue;
        };
        // Skip non-plated thru-hole (mechanical mounting holes) — no electrical net.
        let pad_type = pad_list.get(2).and_then(|s| s.as_atom()).unwrap_or("");
        if pad_type == "np_thru_hole" {
            continue;
        }
        // Pad relative position.
        let (pad_dx, pad_dy) = {
            let at = child(item, "at");
            match at.and_then(|a| a.as_list()) {
                Some(l) => (num(l, 1).unwrap_or(0.0), num(l, 2).unwrap_or(0.0)),
                None => (0.0, 0.0),
            }
        };
        // Absolute position via footprint transform.
        let (rot_dx, rot_dy) = rotate_pad(pad_dx, pad_dy, fp_rot);
        let abs_x = fp_x + rot_dx;
        let abs_y = fp_y + rot_dy;
        let pos = Point::new(nm(abs_x), nm(abs_y));
        // Net assignment — `(net id "name")` child on the pad.
        let net = child(item, "net")
            .and_then(|n| n.as_list())
            .and_then(|l| l.get(1))
            .and_then(|a| a.as_atom())
            .and_then(|s| s.parse::<u32>().ok())
            .and_then(|id| net_map.get(&id).copied());
        // Copper layers.
        let layers = pad_copper_layers(item, layer_map);
        if !layers.is_empty() || net.is_some() {
            out.push(Pad { pos, layers, net });
        }
        let _ = fp_as_node; // suppress warning — fp_as_node is available for future extension
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Public API

/// Parse a KiCad `.kicad_pcb` file text into a [`Board`].
///
/// The returned board contains the copper stack, net table, pad positions, and all routed
/// segments and vias present in the file. The routing grid uses a 0.25 mm pitch over the
/// board outline bounding box derived from the `Edge.Cuts` layer.
///
/// When the board has no `Edge.Cuts` elements, a fallback 100 mm × 100 mm outline is used.
///
/// # Errors
///
/// Returns [`crate::Error::Manifest`] on S-expression parse failure, or
/// [`crate::Error::Geometry`] if the computed grid would be empty.
pub fn parse_kicad_pcb(text: &str) -> crate::Result<Board> {
    let root = parse_sexpr(text)?;
    let items = root
        .as_list()
        .ok_or_else(|| crate::error::manifest::parse_msg("kicad_pcb root must be a list"))?;

    // 1. Copper layers → name map.
    let cu = copper_layers(items);
    let nlayers = cu.len().max(2);
    // Map KiCad layer name → kwavers LayerId (0 = F.Cu, nlayers-1 = B.Cu).
    let layer_map: BTreeMap<String, LayerId> = cu
        .iter()
        .enumerate()
        .map(|(i, (_, name))| (name.clone(), LayerId(i as u16)))
        .collect();

    // 2. Board outline → GridSpec.
    let (min_x_mm, min_y_mm, max_x_mm, max_y_mm) =
        edge_cuts_bbox(items).unwrap_or((0.0, 0.0, 100.0, 100.0));
    let width = nm(max_x_mm - min_x_mm);
    let height = nm(max_y_mm - min_y_mm);
    let pitch = Nm(PITCH_NM);
    let mut spec = GridSpec::cover(width, height, pitch, nlayers)?;
    // Shift origin so (0,0) in engine space maps to (min_x_mm, min_y_mm) in KiCad space.
    spec.origin = Point::new(nm(min_x_mm), nm(min_y_mm));

    // 3. Net declarations → NetId map.
    let mut board = Board::new(spec);
    let mut net_map: BTreeMap<u32, crate::board::NetId> = BTreeMap::new();
    for item in items {
        let Some(list) = item.as_list() else {
            continue;
        };
        if list.first().and_then(|s| s.as_atom()) != Some("net") {
            continue;
        }
        let Some(id_str) = list.get(1).and_then(|s| s.as_atom()) else {
            continue;
        };
        let Ok(kicad_id) = id_str.parse::<u32>() else {
            continue;
        };
        if kicad_id == 0 {
            continue; // net 0 is the unconnected sentinel
        }
        let name = list.get(2).and_then(|s| s.as_atom()).unwrap_or("");
        if name.is_empty() {
            continue;
        }
        let nid = board.add_net(name, NetClassKind::Signal);
        net_map.insert(kicad_id, nid);
    }

    // 4. Footprint pads.
    for item in items {
        if item.head() == Some("footprint") {
            extract_pads(item, &layer_map, &net_map, &mut board.pads);
        }
    }

    // 5. Track segments.
    for item in items {
        let Some(list) = item.as_list() else {
            continue;
        };
        if list.first().and_then(|s| s.as_atom()) != Some("segment") {
            continue;
        }
        let s = Sexpr::List(list.to_vec());
        let Some((sx, sy)) = xy(&s, "start") else {
            continue;
        };
        let Some((ex, ey)) = xy(&s, "end") else {
            continue;
        };
        let width_mm = child(&s, "width")
            .and_then(|w| w.as_list())
            .and_then(|l| num(l, 1))
            .unwrap_or(0.25);
        let layer_name = child(&s, "layer")
            .and_then(|l| l.as_list())
            .and_then(|l| l.get(1))
            .and_then(|a| a.as_atom())
            .unwrap_or("");
        let Some(&layer) = layer_map.get(layer_name) else {
            continue; // non-copper or unknown layer
        };
        // Net is `(net id)` in KiCad 9; `(net "name")` in older files — handle both.
        let net_kicad_id: Option<u32> = child(&s, "net")
            .and_then(|n| n.as_list())
            .and_then(|l| l.get(1))
            .and_then(|a| a.as_atom())
            .and_then(|s| s.parse::<u32>().ok());
        let Some(net) = net_kicad_id.and_then(|id| net_map.get(&id).copied()) else {
            continue; // unconnected or unknown net
        };
        board.tracks.push(Track {
            start: Point::new(nm(sx), nm(sy)),
            end: Point::new(nm(ex), nm(ey)),
            width: nm(width_mm),
            layer,
            net,
        });
    }

    // 6. Vias.
    for item in items {
        let Some(list) = item.as_list() else {
            continue;
        };
        if list.first().and_then(|s| s.as_atom()) != Some("via") {
            continue;
        }
        let s = Sexpr::List(list.to_vec());
        let Some((vx, vy)) = xy(&s, "at") else {
            continue;
        };
        let size_mm = child(&s, "size")
            .and_then(|n| n.as_list())
            .and_then(|l| num(l, 1))
            .unwrap_or(0.8);
        let drill_mm = child(&s, "drill")
            .and_then(|n| n.as_list())
            .and_then(|l| num(l, 1))
            .unwrap_or(0.4);
        // `(layers "F.Cu" "B.Cu")` — extract the two span layers.
        let via_layers: Vec<LayerId> = child(&s, "layers")
            .and_then(|l| l.as_list())
            .map(|l| {
                l.iter()
                    .skip(1)
                    .filter_map(|a| a.as_atom())
                    .filter_map(|name| layer_map.get(name).copied())
                    .collect()
            })
            .unwrap_or_default();
        let (from, to) = match via_layers.as_slice() {
            [a, b, ..] => (*a.min(b), *a.max(b)),
            [a] => (*a, *a),
            [] => (LayerId(0), LayerId((nlayers - 1) as u16)),
        };
        // Net: `(net id)` style.
        let net_kicad_id: Option<u32> = child(&s, "net")
            .and_then(|n| n.as_list())
            .and_then(|l| l.get(1))
            .and_then(|a| a.as_atom())
            .and_then(|v| v.parse::<u32>().ok());
        let Some(net) = net_kicad_id.and_then(|id| net_map.get(&id).copied()) else {
            continue;
        };
        let kind = Via::classify(from, to, nlayers);
        board.vias.push(Via {
            pos: Point::new(nm(vx), nm(vy)),
            drill: nm(drill_mm),
            diameter: nm(size_mm),
            net,
            from,
            to,
            kind,
            filled: matches!(kind, ViaKind::Micro),
        });
    }

    Ok(board)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal 2-layer PCB: one net, one pad, one segment.
    const MINIMAL_PCB: &str = r#"(kicad_pcb
  (version 20241229)
  (layers
    (0 "F.Cu" signal)
    (31 "B.Cu" signal)
    (25 "Edge.Cuts" user)
  )
  (net 0 "")
  (net 1 "GND")
  (footprint "Test:R"
    (layer "F.Cu")
    (at 10 10 0)
    (pad "1" smd rect (at 0 0) (size 1 1) (layers "F.Cu") (net 1 "GND"))
    (pad "2" smd rect (at 2 0) (size 1 1) (layers "F.Cu") (net 1 "GND"))
  )
  (segment (start 10 10) (end 12 10) (width 0.25) (layer "F.Cu") (net 1))
  (gr_line (start 0 0) (end 20 0) (layer "Edge.Cuts"))
  (gr_line (start 20 0) (end 20 20) (layer "Edge.Cuts"))
  (gr_line (start 20 20) (end 0 20) (layer "Edge.Cuts"))
  (gr_line (start 0 20) (end 0 0) (layer "Edge.Cuts"))
)"#;

    #[test]
    fn parse_minimal_pcb_net_count() {
        let board = parse_kicad_pcb(MINIMAL_PCB).expect("minimal PCB must parse");
        assert_eq!(board.nets.len(), 1, "one net (GND)");
        assert_eq!(board.nets[0].name, "GND");
    }

    #[test]
    fn parse_minimal_pcb_pad_count() {
        let board = parse_kicad_pcb(MINIMAL_PCB).expect("minimal PCB must parse");
        assert_eq!(board.pads.len(), 2, "two pads on the resistor");
        for pad in &board.pads {
            assert!(pad.net.is_some(), "pad must be assigned to GND");
        }
    }

    #[test]
    fn parse_minimal_pcb_track_count() {
        let board = parse_kicad_pcb(MINIMAL_PCB).expect("minimal PCB must parse");
        assert_eq!(board.tracks.len(), 1, "one track segment");
        let t = &board.tracks[0];
        assert_eq!(t.width, nm(0.25), "track width 0.25 mm");
        assert_eq!(t.layer, LayerId(0), "track on F.Cu");
    }

    #[test]
    fn parse_minimal_pcb_board_dimensions() {
        let board = parse_kicad_pcb(MINIMAL_PCB).expect("minimal PCB must parse");
        // Edge.Cuts spans (0,0)→(20,20) mm → 20_000_000 nm in each axis.
        let width = board.spec.nx as i64 * board.spec.pitch.0;
        let height = board.spec.ny as i64 * board.spec.pitch.0;
        assert!(
            width >= 20_000_000,
            "board width must cover 20 mm; got {} nm",
            width
        );
        assert!(
            height >= 20_000_000,
            "board height must cover 20 mm; got {} nm",
            height
        );
    }

    #[test]
    fn parse_minimal_pcb_layer_count() {
        let board = parse_kicad_pcb(MINIMAL_PCB).expect("minimal PCB must parse");
        assert_eq!(board.spec.nlayers, 2, "2-layer board");
    }

    #[test]
    fn parse_pcb_with_thru_hole_via() {
        let src = r#"(kicad_pcb
  (version 20241229)
  (layers
    (0 "F.Cu" signal)
    (31 "B.Cu" signal)
  )
  (net 1 "VCC")
  (via (at 5 5) (size 0.8) (drill 0.4) (layers "F.Cu" "B.Cu") (net 1))
)"#;
        let board = parse_kicad_pcb(src).expect("via PCB must parse");
        assert_eq!(board.vias.len(), 1, "one via");
        let v = &board.vias[0];
        assert_eq!(v.kind, ViaKind::Through, "F.Cu → B.Cu is Through");
        assert_eq!(v.diameter, nm(0.8));
        assert_eq!(v.drill, nm(0.4));
    }

    #[test]
    fn parse_pcb_np_thru_hole_not_included() {
        let src = r#"(kicad_pcb
  (version 20241229)
  (layers (0 "F.Cu" signal) (31 "B.Cu" signal))
  (footprint "Test:MH"
    (layer "F.Cu")
    (at 5 5 0)
    (pad "1" np_thru_hole circle (at 0 0) (size 3 3) (drill 3) (layers "*.Cu"))
  )
)"#;
        let board = parse_kicad_pcb(src).expect("must parse");
        assert_eq!(board.pads.len(), 0, "np_thru_hole pads must be excluded");
    }

    #[test]
    fn parse_pcb_wildcard_layers_assigns_all_copper() {
        let src = r#"(kicad_pcb
  (version 20241229)
  (layers (0 "F.Cu" signal) (31 "B.Cu" signal))
  (net 1 "GND")
  (footprint "Test:CONN"
    (layer "F.Cu")
    (at 0 0 0)
    (pad "1" thru_hole circle (at 0 0) (size 1.6 1.6) (drill 0.8) (layers "*.Cu") (net 1 "GND"))
  )
)"#;
        let board = parse_kicad_pcb(src).expect("must parse");
        assert_eq!(board.pads.len(), 1);
        // A wildcard "*.Cu" pad on a 2-layer board must have 2 layer entries.
        assert_eq!(
            board.pads[0].layers.len(),
            2,
            "*.Cu wildcard must expand to all copper layers"
        );
    }

    #[test]
    fn parse_pcb_footprint_rotation_applied() {
        // Footprint at (10, 10) rotated 90°; pad at relative (2, 0).
        // After 90° KiCad-CW rotation: (dx, dy) → (dy, -dx) for CW convention.
        // Using the KiCad convention: CW positive → we negate theta in rotate_pad,
        // so theta = -90° → cos(-90°) = 0, sin(-90°) = -1.
        // abs_x = 10 + 2 * 0   - 0 * (-1) = 10
        // abs_y = 10 + 2 * (-1) + 0 * 0   =  8
        let src = r#"(kicad_pcb
  (version 20241229)
  (layers (0 "F.Cu" signal) (31 "B.Cu" signal))
  (net 1 "SIG")
  (footprint "Test:R"
    (layer "F.Cu")
    (at 10 10 90)
    (pad "1" smd rect (at 2 0) (size 1 1) (layers "F.Cu") (net 1 "SIG"))
  )
)"#;
        let board = parse_kicad_pcb(src).expect("must parse");
        assert_eq!(board.pads.len(), 1);
        let p = &board.pads[0];
        // Allow ±1 nm rounding tolerance.
        let expected_x = nm(10.0);
        let expected_y = nm(8.0);
        let dx = (p.pos.x.0 - expected_x.0).abs();
        let dy = (p.pos.y.0 - expected_y.0).abs();
        assert!(
            dx <= 1,
            "pad X after 90° rotation: expected {}, got {}",
            expected_x.0,
            p.pos.x.0
        );
        assert!(
            dy <= 1,
            "pad Y after 90° rotation: expected {}, got {}",
            expected_y.0,
            p.pos.y.0
        );
    }

    #[test]
    fn parse_pcb_malformed_input_returns_error() {
        let err = parse_kicad_pcb("(kicad_pcb (broken").expect_err("unclosed must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("manifest"),
            "error must be a Manifest parse error: {msg}"
        );
    }
}
