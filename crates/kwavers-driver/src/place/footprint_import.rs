//! Import **real** manufacturer footprints from KiCad `.kicad_mod` files.
//!
//! The placer/router operate on [`FootprintDef`]s. Synthesizing those as generic pad rings
//! (`perimeter_pads`) is enough to exercise routing, but it is **not fabrication-real**: the pad
//! geometry and pin map do not match the actual part, so the generated gerbers would not assemble the
//! component. This module parses the exact KiCad footprint shipped in the vendor CAD package
//! (`docs/cad_models/<part>/…/footprints.pretty/*.kicad_mod`) into a [`FootprintDef`] with the real
//! pad positions, sizes, layer sets, and **pin identifiers** — so the optimiser places and routes the
//! genuine footprint and net assignments are made by pin name, not by positional abstraction.
//!
//! The parser is a minimal s-expression reader (no external dependency) that extracts the `pad`,
//! `F.CrtYd` (courtyard), and `model` tokens; the rest of the footprint art (silk, fab, paste) is not
//! needed for place-and-route and is ignored.
//!
//! The S-expression kernel (`Sexpr`, `parse_sexpr`, `child`, `num`, `xyz_child`) lives in
//! `place::sexpr` (`pub(crate)`) so `io::pcb_parse` (a second consumer) shares the same SSOT
//! instead of duplicating the parser.
//!
//! Evidence tier: value-semantic unit tests parse real vendor files committed under `docs/` and assert
//! the exact pad count and known pin positions (differential against the source file).

use super::rotation::RotationPolicy;
use super::sexpr::{child, num, parse_sexpr, xyz_child};
use crate::board::LayerId;
use crate::geom::{Nm, Point};
use crate::place::footprint::{FootprintDef, Model3D, PadDef, Role};

/// Import a real footprint from a `.kicad_mod` file into a [`FootprintDef`] with exact pads.
///
/// `role` tags the part for placement; `power_pins` lists the pin **names** (e.g. `"VPP"`, `"GND"`,
/// or pad numbers) that are power/ground (the geometry file does not carry that intent). The courtyard
/// is taken from the `F.CrtYd` extents (falling back to the pad bounding box plus 0.25 mm). SMD pads
/// are reachable on the top layer; thru-hole/NPTH pads on all layers.
pub fn import_kicad_mod(
    path: impl AsRef<std::path::Path>,
    role: Role,
    power_pins: &[&str],
) -> Result<FootprintDef, crate::Error> {
    let path_buf = path.as_ref().to_path_buf();
    let text = std::fs::read_to_string(&path_buf)
        .map_err(|source| crate::error::manifest::io_at(path_buf.clone(), source))?;
    let root = parse_sexpr(&text)?;
    let items = root.as_list().ok_or_else(|| {
        crate::error::manifest::parse_msg("root s-expression is an atom, expected a list")
    })?;
    let name = items
        .get(1)
        .and_then(|s| s.as_atom())
        .unwrap_or("imported")
        .to_string();

    let mut pads = Vec::new();
    let mut pad_names = Vec::new();
    let mut crtyd_min = (f64::INFINITY, f64::INFINITY);
    let mut crtyd_max = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    let mut pad_min = (f64::INFINITY, f64::INFINITY);
    let mut pad_max = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    let mut model: Option<Model3D> = None;

    for item in items {
        match item.head() {
            Some("pad") => {
                let list = item
                    .as_list()
                    .expect("invariant: head() returned Some implies item is a list");
                // (pad "<name>" <type> <shape> (at x y [rot]) (size w h) (layers ...) ...)
                let pad_type = list.get(2).and_then(|s| s.as_atom()).unwrap_or("smd");
                // A non-plated thru-hole (`np_thru_hole`) is a *mechanical* feature — a board-lock or
                // mounting hole, not an electrical pad (KiCad gives these an empty/`""` designator;
                // this Molex part labels them `"None"`). It carries no net and must not become a
                // schematic pin or an ERC subject, but it *is* physical copper-clearance + drill
                // keepout on the PCB. Mark it mechanical with an empty pad name (the convention the
                // schematic/ERC use to skip it); the PCB-footprint and obstacle paths keep it.
                let pad_name = if pad_type == "np_thru_hole" {
                    String::new()
                } else {
                    list.get(1)
                        .and_then(|s| s.as_atom())
                        .unwrap_or("")
                        .to_string()
                };
                // `(at x y [rot])` — the optional third value is the pad's own rotation in degrees
                // (about its centre). IC side-pads are defined with their long axis vertical and a 90°
                // rotation that turns it horizontal (pointing away from the package); dropping the
                // rotation leaves the long axis along the pin-pitch direction, so adjacent pads overlap
                // and short. `PadDef.size` is an axis-aligned box, so a 90°/270° rotation swaps (w, h).
                let at = child(item, "at").and_then(|a| {
                    let l = a.as_list()?;
                    Some((num(l, 1)?, num(l, 2)?, num(l, 3).unwrap_or(0.0)))
                });
                let size = child(item, "size").and_then(|a| {
                    let l = a.as_list()?;
                    Some((num(l, 1)?, num(l, 2)?))
                });
                let (Some((x, y, rot)), Some((w0, h0))) = (at, size) else {
                    continue;
                };
                // Normalise rotation to [0,180); a quarter-turn swaps the axis-aligned extents.
                let quarter_turned = {
                    let r = rot.rem_euclid(180.0);
                    (45.0..135.0).contains(&r)
                };
                let (w, h) = if quarter_turned { (h0, w0) } else { (w0, h0) };
                // Thru-hole / NPTH pads reach every copper layer; SMD pads only the top.
                let layers = if pad_type.contains("thru") || pad_type == "np_thru_hole" {
                    vec![LayerId(0), LayerId(1)]
                } else {
                    vec![LayerId(0)]
                };
                pad_min = (pad_min.0.min(x - w / 2.0), pad_min.1.min(y - h / 2.0));
                pad_max = (pad_max.0.max(x + w / 2.0), pad_max.1.max(y + h / 2.0));
                pads.push(PadDef {
                    offset: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
                    size: (Nm::from_mm(w), Nm::from_mm(h)),
                    layers,
                    power_pin: power_pins.contains(&pad_name.as_str()),
                });
                pad_names.push(pad_name);
            }
            // Accumulate courtyard extents from F.CrtYd graphics only.
            Some("fp_line") | Some("fp_poly") | Some("fp_rect")
                if child(item, "layer")
                    .and_then(|l| l.as_list()?.get(1)?.as_atom())
                    .map(|s| s.contains("CrtYd"))
                    .unwrap_or(false) =>
            {
                for (key, idx) in [("start", 1), ("end", 1)] {
                    if let Some(p) = child(item, key) {
                        if let (Some(px), Some(py)) = (
                            num(
                                p.as_list()
                                    .expect("invariant: KiCad (start/end x y) nodes are lists"),
                                idx,
                            ),
                            num(
                                p.as_list()
                                    .expect("invariant: KiCad (start/end x y) nodes are lists"),
                                idx + 1,
                            ),
                        ) {
                            crtyd_min = (crtyd_min.0.min(px), crtyd_min.1.min(py));
                            crtyd_max = (crtyd_max.0.max(px), crtyd_max.1.max(py));
                        }
                    }
                }
            }
            Some("model") => {
                let p = item
                    .as_list()
                    .expect("invariant: head() returned Some implies item is a list")
                    .get(1)
                    .and_then(|s| s.as_atom());
                if let Some(p) = p {
                    let offset = xyz_child(item, "offset").unwrap_or((0.0, 0.0, 0.0));
                    let rotate = xyz_child(item, "rotate").unwrap_or((0.0, 0.0, 0.0));
                    model = Some((p.to_string(), offset, rotate, None));
                }
            }
            _ => {}
        }
    }

    if pads.is_empty() {
        // Move `path_buf` rather than re-fetching through the generic `path`:
        // `path_buf` is still owned here (the IO error envelope earlier used a
        // clone) and the function returns on this branch, so no later use.
        // Routes through the cross-file SSOT helper at `crate::error::manifest::no_pads`
        // so the construction shape stays identical to every other module.
        return Err(crate::error::manifest::no_pads(path_buf));
    }

    // Courtyard: prefer the F.CrtYd extents; else the pad bounding box + a 0.25 mm ring. KiCad
    // footprints may choose any local origin; `FootprintDef` requires pad offsets relative to the
    // courtyard centre, so imported pad coordinates are translated to that canonical origin. The 3D
    // model offset is expressed in that same local footprint frame, so it must receive the identical
    // origin translation or the rendered STEP body drifts away from the pads even while DRC passes.
    let (w, h, cx, cy) = if crtyd_min.0.is_finite() && crtyd_max.0 > crtyd_min.0 {
        (
            crtyd_max.0 - crtyd_min.0,
            crtyd_max.1 - crtyd_min.1,
            (crtyd_min.0 + crtyd_max.0) / 2.0,
            (crtyd_min.1 + crtyd_max.1) / 2.0,
        )
    } else {
        (
            pad_max.0 - pad_min.0 + 0.5,
            pad_max.1 - pad_min.1 + 0.5,
            (pad_min.0 + pad_max.0) / 2.0,
            (pad_min.1 + pad_max.1) / 2.0,
        )
    };
    for pad in &mut pads {
        pad.offset = Point::new(
            Nm::from_mm(pad.offset.x.to_mm() - cx),
            Nm::from_mm(pad.offset.y.to_mm() - cy),
        );
    }
    if let Some((_, offset, _, _)) = &mut model {
        offset.0 -= cx;
        offset.1 -= cy;
    }

    Ok(FootprintDef {
        name,
        courtyard: (Nm::from_mm(w), Nm::from_mm(h)),
        role,
        rotation_policy: RotationPolicy::for_role(role),
        pads,
        pad_names,
        model,
        ball_pitch: None,
        i_dd_a: 0.0,
        capacitance_f: 0.0,
        dielectric_grade: crate::place::footprint::DielectricGrade::Unknown,
        package_form_factor: crate::place::footprint::PackageFormFactor::Unknown,
    })
}
