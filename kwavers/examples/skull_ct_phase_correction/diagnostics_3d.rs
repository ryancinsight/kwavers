use std::collections::BTreeSet;
use std::f64::consts::{PI, TAU};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::Result;

use super::{CtVolume, ElementProjection, HU_BONE_LOWER};

const MAX_SKULL_POINTS: usize = 50_000;
const MAX_AIR_CAVITY_POINTS: usize = 12_000;
const SVG_WIDTH: f64 = 1200.0;
const SVG_HEIGHT: f64 = 900.0;
const TRANSDUCER_POSTERIOR_TILT_DEG: f64 = 18.0;
const AIR_CAVITY_HU_UPPER: f64 = -450.0;

#[derive(Debug, Clone, Copy)]
struct Point3 {
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Debug, Clone, Copy)]
struct Point2 {
    x: f64,
    y: f64,
    depth: f64,
}

#[derive(Debug, Clone)]
struct SkullSample {
    points: Vec<Point3>,
    min: Point3,
    max: Point3,
}

#[derive(Debug, Clone)]
struct AirCavitySample {
    points: Vec<Point3>,
    min: Point3,
    max: Point3,
}

#[derive(Debug, Clone)]
struct ElementPoint {
    point: Point3,
    phase: f64,
    disabled: bool,
}

#[derive(Debug, Clone)]
struct ArrayDiagnostic {
    elements: Vec<ElementPoint>,
    natural_focus: Point3,
}

pub(crate) fn companion_path(path: &Path, suffix: &str, extension: &str) -> PathBuf {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("skull_ct_phase_correction");
    path.with_file_name(format!("{stem}{suffix}.{extension}"))
}

pub(crate) fn write_three_dimensional_diagnostics(
    output_path: &Path,
    ct: &CtVolume,
    elements: &[ElementProjection],
) -> Result<()> {
    let skull = sample_skull_boundary(ct);
    let svg_path = companion_path(output_path, "_3d", "svg");
    let obj_path = companion_path(output_path, "_3d", "obj");
    write_svg(&svg_path, ct, &skull, elements)?;
    write_obj(&obj_path, ct, &skull, elements)?;
    Ok(())
}

fn sample_skull_boundary(ct: &CtVolume) -> SkullSample {
    let (nx, ny, nz) = ct.hu.dim();
    let center_x = 0.5 * (nx.saturating_sub(1) as f64);
    let center_y = 0.5 * (ny.saturating_sub(1) as f64);
    let center_z = 0.5 * (nz.saturating_sub(1) as f64);
    let sx = ct.spacing_m[0] * 1e3;
    let sy = ct.spacing_m[1] * 1e3;
    let sz = ct.spacing_m[2] * 1e3;
    let mut points = Vec::with_capacity(MAX_SKULL_POINTS);
    let mut min = Point3 {
        x: f64::INFINITY,
        y: f64::INFINITY,
        z: f64::INFINITY,
    };
    let mut max = Point3 {
        x: f64::NEG_INFINITY,
        y: f64::NEG_INFINITY,
        z: f64::NEG_INFINITY,
    };

    for z in 1..nz.saturating_sub(1) {
        for y in (1..ny.saturating_sub(1)).step_by(2) {
            for x in (1..nx.saturating_sub(1)).step_by(2) {
                if ct.hu[[x, y, z]] < HU_BONE_LOWER || !is_boundary_bone(ct, x, y, z) {
                    continue;
                }
                let hash = x.wrapping_mul(73_856_093)
                    ^ y.wrapping_mul(19_349_663)
                    ^ z.wrapping_mul(83_492_791);
                if hash % 2 != 0 {
                    continue;
                }
                let point = Point3 {
                    x: (x as f64 - center_x) * sx,
                    // Display AP axis is flipped so the sagittal view places
                    // anterior/orbital anatomy to screen-right.
                    y: -(y as f64 - center_y) * sy,
                    // This diagnostic frame preserves the RITK slice ordering
                    // that places the cranial vault superior to the skull base
                    // for the selected CT series.
                    z: (z as f64 - center_z) * sz,
                };
                include_point(&mut min, &mut max, point);
                points.push(point);
                if points.len() >= MAX_SKULL_POINTS {
                    return SkullSample { points, min, max };
                }
            }
        }
    }

    SkullSample { points, min, max }
}

fn is_boundary_bone(ct: &CtVolume, x: usize, y: usize, z: usize) -> bool {
    ct.hu[[x - 1, y, z]] < HU_BONE_LOWER
        || ct.hu[[x + 1, y, z]] < HU_BONE_LOWER
        || ct.hu[[x, y - 1, z]] < HU_BONE_LOWER
        || ct.hu[[x, y + 1, z]] < HU_BONE_LOWER
        || ct.hu[[x, y, z - 1]] < HU_BONE_LOWER
        || ct.hu[[x, y, z + 1]] < HU_BONE_LOWER
}

fn include_point(min: &mut Point3, max: &mut Point3, point: Point3) {
    min.x = min.x.min(point.x);
    min.y = min.y.min(point.y);
    min.z = min.z.min(point.z);
    max.x = max.x.max(point.x);
    max.y = max.y.max(point.y);
    max.z = max.z.max(point.z);
}

fn write_svg(
    path: &Path,
    ct: &CtVolume,
    skull: &SkullSample,
    elements: &[ElementProjection],
) -> Result<()> {
    let air_cavities = sample_anterior_air_cavities(ct, skull);
    let avoidance = orbital_avoidance_zone(skull, &air_cavities);
    let array = array_diagnostic_mm(skull, &avoidance, elements);
    let mut projected = Vec::with_capacity(skull.points.len() + array.elements.len() + 4);
    for point in &skull.points {
        projected.push(project(*point));
    }
    for point in &air_cavities.points {
        projected.push(project(*point));
    }
    for element in &array.elements {
        projected.push(project(element.point));
    }
    for point in ac_pc_plane_points(skull) {
        projected.push(project(point));
    }
    projected.push(project(array.natural_focus));

    let bounds = projection_bounds(&projected);
    let scale = 0.82
        * (SVG_WIDTH / (bounds.1.x - bounds.0.x).max(1.0))
            .min(SVG_HEIGHT / (bounds.1.y - bounds.0.y).max(1.0));
    let origin = Point2 {
        x: 0.5 * SVG_WIDTH - 0.5 * scale * (bounds.0.x + bounds.1.x),
        y: 0.5 * SVG_HEIGHT - 0.5 * scale * (bounds.0.y + bounds.1.y),
        depth: 0.0,
    };

    let mut out = BufWriter::new(File::create(path)?);
    writeln!(
        out,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">"#
    )?;
    writeln!(
        out,
        r##"<rect width="100%" height="100%" fill="#f8fafc"/>"##
    )?;
    writeln!(
        out,
        r##"<text x="40" y="44" font-family="Arial" font-size="24" fill="#0f172a">RITK skull CT inside 1024-element 650 kHz hemispherical array</text>"##
    )?;
    writeln!(
        out,
        r##"<text x="40" y="74" font-family="Arial" font-size="14" fill="#475569">Display pose: skull stays in CT pose; transducer is tilted posteriorly so rays avoid anterior orbital/nasal region.</text>"##
    )?;

    write_plane(&mut out, skull, scale, origin)?;
    write_skull_points(&mut out, skull, scale, origin)?;
    write_orbital_avoidance_zone(&mut out, &avoidance, scale, origin)?;
    write_focus_rays(
        &mut out,
        array.natural_focus,
        &array.elements,
        scale,
        origin,
    )?;
    write_element_points(&mut out, &array.elements, scale, origin)?;
    write_orientation_axes(&mut out)?;
    write_legend(
        &mut out,
        ct,
        skull,
        &air_cavities,
        elements,
        &array.elements,
    )?;
    writeln!(out, "</svg>")?;
    Ok(())
}

fn sample_anterior_air_cavities(ct: &CtVolume, skull: &SkullSample) -> AirCavitySample {
    let (nx, ny, nz) = ct.hu.dim();
    let center_x = 0.5 * (nx.saturating_sub(1) as f64);
    let center_y = 0.5 * (ny.saturating_sub(1) as f64);
    let center_z = 0.5 * (nz.saturating_sub(1) as f64);
    let sx = ct.spacing_m[0] * 1e3;
    let sy = ct.spacing_m[1] * 1e3;
    let sz = ct.spacing_m[2] * 1e3;
    let ap = skull.max.y - skull.min.y;
    let si = skull.max.z - skull.min.z;
    let lr = skull.max.x - skull.min.x;
    let anterior_min_y = skull.min.y + 0.58 * ap;
    let anterior_max_y = skull.max.y - 0.04 * ap;
    let medial_min_x = skull.min.x + 0.18 * lr;
    let medial_max_x = skull.max.x - 0.18 * lr;
    let inferior_z = skull.min.z + 0.32 * si;
    let superior_z = skull.min.z + 0.66 * si;
    let mut points = Vec::with_capacity(MAX_AIR_CAVITY_POINTS);
    let mut min = Point3 {
        x: f64::INFINITY,
        y: f64::INFINITY,
        z: f64::INFINITY,
    };
    let mut max = Point3 {
        x: f64::NEG_INFINITY,
        y: f64::NEG_INFINITY,
        z: f64::NEG_INFINITY,
    };

    for z in 3..nz.saturating_sub(3) {
        for y in (3..ny.saturating_sub(3)).step_by(2) {
            for x in (3..nx.saturating_sub(3)).step_by(2) {
                if ct.hu[[x, y, z]] > AIR_CAVITY_HU_UPPER || !is_air_adjacent_to_bone(ct, x, y, z) {
                    continue;
                }
                let point = Point3 {
                    x: (x as f64 - center_x) * sx,
                    y: -(y as f64 - center_y) * sy,
                    z: (z as f64 - center_z) * sz,
                };
                let Some((posterior_surface_y, anterior_surface_y)) =
                    ap_bone_span_display_mm(ct, x, z, center_y, sy)
                else {
                    continue;
                };
                let local_ap = anterior_surface_y - posterior_surface_y;
                if point.x < medial_min_x
                    || point.x > medial_max_x
                    || point.y < anterior_min_y
                    || point.y > anterior_max_y
                    || point.z < inferior_z
                    || point.z > superior_z
                    || local_ap < 0.18 * ap
                    || point.y <= posterior_surface_y + 0.55 * local_ap
                    || point.y >= anterior_surface_y - 2.0
                {
                    continue;
                }
                let hash = x.wrapping_mul(73_856_093)
                    ^ y.wrapping_mul(19_349_663)
                    ^ z.wrapping_mul(83_492_791);
                if hash % 3 != 0 {
                    continue;
                }
                include_point(&mut min, &mut max, point);
                points.push(point);
                if points.len() >= MAX_AIR_CAVITY_POINTS {
                    return AirCavitySample { points, min, max };
                }
            }
        }
    }

    AirCavitySample { points, min, max }
}

fn ap_bone_span_display_mm(
    ct: &CtVolume,
    x: usize,
    z: usize,
    center_y: f64,
    sy: f64,
) -> Option<(f64, f64)> {
    let (_, ny, _) = ct.hu.dim();
    let mut posterior = f64::INFINITY;
    let mut anterior = f64::NEG_INFINITY;
    for y in 0..ny {
        if ct.hu[[x, y, z]] < HU_BONE_LOWER {
            continue;
        }
        let display_y = -(y as f64 - center_y) * sy;
        posterior = posterior.min(display_y);
        anterior = anterior.max(display_y);
    }
    posterior.is_finite().then_some((posterior, anterior))
}

fn is_air_adjacent_to_bone(ct: &CtVolume, x: usize, y: usize, z: usize) -> bool {
    for offset in 1..=3 {
        if ct.hu[[x - offset, y, z]] >= HU_BONE_LOWER
            || ct.hu[[x + offset, y, z]] >= HU_BONE_LOWER
            || ct.hu[[x, y - offset, z]] >= HU_BONE_LOWER
            || ct.hu[[x, y + offset, z]] >= HU_BONE_LOWER
            || ct.hu[[x, y, z - offset]] >= HU_BONE_LOWER
            || ct.hu[[x, y, z + offset]] >= HU_BONE_LOWER
        {
            return true;
        }
    }
    false
}

fn write_plane<W: Write>(
    out: &mut W,
    skull: &SkullSample,
    scale: f64,
    origin: Point2,
) -> Result<()> {
    let plane = ac_pc_plane_points(skull);
    write!(out, r#"<polygon points=""#)?;
    for point in plane {
        let p = transform(project(point), scale, origin);
        write!(out, "{:.2},{:.2} ", p.x, p.y)?;
    }
    writeln!(
        out,
        r##"" fill="#f59e0b" fill-opacity="0.09" stroke="#f59e0b" stroke-width="1.2" stroke-dasharray="6 5"/>"##
    )?;
    Ok(())
}

fn write_skull_points<W: Write>(
    out: &mut W,
    skull: &SkullSample,
    scale: f64,
    origin: Point2,
) -> Result<()> {
    let mut points = skull.points.clone();
    points.sort_by(|a, b| {
        project(*a)
            .depth
            .partial_cmp(&project(*b).depth)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut bins = BTreeSet::new();
    for point in &points {
        let p = transform(project(*point), scale, origin);
        bins.insert(((p.x / 3.0).round() as i32, (p.y / 3.0).round() as i32));
    }

    writeln!(
        out,
        r##"<g id="skull-sagittal-density" fill="#334155" fill-opacity="0.16">"##
    )?;
    for (x, y) in bins {
        writeln!(
            out,
            r#"<rect x="{:.2}" y="{:.2}" width="3.2" height="5.2"/>"#,
            x as f64 * 3.0 - 1.6,
            y as f64 * 3.0 - 2.6
        )?;
    }
    writeln!(out, "</g>")?;

    writeln!(
        out,
        r##"<g id="skull-boundary" fill="#0f172a" fill-opacity="0.40">"##
    )?;
    for point in points {
        let p = transform(project(point), scale, origin);
        writeln!(out, r#"<circle cx="{:.2}" cy="{:.2}" r="0.72"/>"#, p.x, p.y)?;
    }
    writeln!(out, "</g>")?;
    Ok(())
}

fn write_element_points<W: Write>(
    out: &mut W,
    element_points: &[ElementPoint],
    scale: f64,
    origin: Point2,
) -> Result<()> {
    let mut points = element_points.to_vec();
    points.sort_by(|a, b| {
        project(a.point)
            .depth
            .partial_cmp(&project(b.point).depth)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    writeln!(
        out,
        r##"<g id="hemispherical-array" stroke="#0f172a" stroke-opacity="0.45" stroke-width="0.35">"##
    )?;
    for element in points {
        let p = transform(project(element.point), scale, origin);
        if element.disabled {
            writeln!(
                out,
                r##"<g stroke="#64748b" stroke-width="1.1" stroke-opacity="0.85"><circle cx="{:.2}" cy="{:.2}" r="2.9" fill="#cbd5e1"/><line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}"/><line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}"/></g>"##,
                p.x,
                p.y,
                p.x - 3.1,
                p.y - 3.1,
                p.x + 3.1,
                p.y + 3.1,
                p.x - 3.1,
                p.y + 3.1,
                p.x + 3.1,
                p.y - 3.1
            )?;
            continue;
        }
        let color = phase_color_hex(element.phase);
        writeln!(
            out,
            r#"<circle cx="{:.2}" cy="{:.2}" r="2.6" fill="{color}"/>"#,
            p.x, p.y
        )?;
    }
    writeln!(out, "</g>")?;
    Ok(())
}

fn write_focus_rays<W: Write>(
    out: &mut W,
    focus: Point3,
    element_points: &[ElementPoint],
    scale: f64,
    origin: Point2,
) -> Result<()> {
    let projected_focus = transform(project(focus), scale, origin);
    writeln!(
        out,
        r##"<g id="downward-acoustic-rays" stroke="#2563eb" stroke-width="0.65" stroke-opacity="0.38" fill="none">"##
    )?;
    for (idx, element) in element_points.iter().enumerate() {
        if idx % 24 != 0 {
            continue;
        }
        if element.disabled {
            continue;
        }
        let source = transform(project(element.point), scale, origin);
        writeln!(
            out,
            r#"<line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}"/>"#,
            source.x, source.y, projected_focus.x, projected_focus.y
        )?;
    }
    writeln!(out, "</g>")?;
    writeln!(
        out,
        r##"<g font-family="Arial" font-size="14" fill="#1d4ed8">"##
    )?;
    writeln!(
        out,
        r##"<circle cx="{:.2}" cy="{:.2}" r="5.5" fill="#2563eb"/>"##,
        projected_focus.x, projected_focus.y
    )?;
    writeln!(
        out,
        r#"<text x="{:.2}" y="{:.2}">natural geometric focus</text>"#,
        projected_focus.x + 10.0,
        projected_focus.y + 4.0
    )?;
    writeln!(out, "</g>")?;
    Ok(())
}

fn write_orbital_avoidance_zone<W: Write>(
    out: &mut W,
    zone: &AvoidanceZone,
    scale: f64,
    origin: Point2,
) -> Result<()> {
    if !zone.air_points.is_empty() {
        let mut bins = BTreeSet::new();
        for point in &zone.air_points {
            let p = transform(project(*point), scale, origin);
            bins.insert(((p.x / 4.0).round() as i32, (p.y / 4.0).round() as i32));
        }
        writeln!(
            out,
            r##"<g id="ct-air-cavity-samples" fill="#ea580c" fill-opacity="0.34">"##
        )?;
        for (x, y) in bins {
            writeln!(
                out,
                r#"<rect x="{:.2}" y="{:.2}" width="4.4" height="4.4"/>"#,
                x as f64 * 4.0 - 2.2,
                y as f64 * 4.0 - 2.2
            )?;
        }
        writeln!(out, "</g>")?;
        return Ok(());
    }
    let center = transform(project(zone.center), scale, origin);
    writeln!(
        out,
        r##"<ellipse cx="{:.2}" cy="{:.2}" rx="{:.2}" ry="{:.2}" fill="#dc2626" fill-opacity="0.10" stroke="#dc2626" stroke-width="1.4" stroke-dasharray="5 4"/>"##,
        center.x,
        center.y,
        zone.radius_y * scale,
        zone.radius_z * scale
    )?;
    writeln!(
        out,
        r##"<text x="{:.2}" y="{:.2}" font-family="Arial" font-size="14" fill="#b91c1c">fallback orbital/sinus no-pass mask</text>"##,
        center.x + zone.radius_y * scale + 8.0,
        center.y
    )?;
    Ok(())
}

fn write_orientation_axes<W: Write>(out: &mut W) -> Result<()> {
    writeln!(
        out,
        r##"<g font-family="Arial" font-size="15" fill="#0f172a" stroke="#0f172a" stroke-width="1.5">"##
    )?;
    writeln!(out, r#"<line x1="1090" y1="730" x2="1090" y2="590"/>"#)?;
    writeln!(
        out,
        r##"<path d="M1090 590 L1083 604 L1097 604 Z" fill="#0f172a"/>"##
    )?;
    writeln!(out, r#"<text x="1110" y="598">superior</text>"#)?;
    writeln!(out, r#"<text x="1110" y="730">inferior / neck</text>"#)?;
    writeln!(out, r#"<line x1="1025" y1="730" x2="1155" y2="730"/>"#)?;
    writeln!(out, r#"<text x="1005" y="756">posterior</text>"#)?;
    writeln!(out, r#"<text x="1125" y="756">anterior</text>"#)?;
    writeln!(out, "</g>")?;
    Ok(())
}

fn write_legend<W: Write>(
    out: &mut W,
    ct: &CtVolume,
    skull: &SkullSample,
    air_cavities: &AirCavitySample,
    elements: &[ElementProjection],
    element_points: &[ElementPoint],
) -> Result<()> {
    let nonzero = elements
        .iter()
        .filter(|element| element.correction_rad.abs() > 1e-12)
        .count();
    let disabled = element_points
        .iter()
        .filter(|element| element.disabled)
        .count();
    writeln!(
        out,
        r##"<g font-family="Arial" font-size="14" fill="#0f172a">"##
    )?;
    writeln!(
        out,
        r#"<text x="40" y="815">dark blue-gray: HU >= 300 skull silhouette and boundary from RITK-loaded CT</text>"#
    )?;
    writeln!(
        out,
        r#"<text x="40" y="838">orange: HU &lt;= -450 anterior bone-adjacent CT air; gray X: disabled no-pass elements</text>"#
    )?;
    writeln!(
        out,
        r#"<text x="40" y="861">disabled elements: {disabled}/{}; nonzero corrections: {nonzero}/{}; skull boundary points: {}; air-cavity points: {}</text>"#,
        element_points.len(),
        elements.len(),
        skull.points.len(),
        air_cavities.points.len()
    )?;
    writeln!(
        out,
        r#"<text x="650" y="815">RITK origin [mm]: [{:.2}, {:.2}, {:.2}]</text>"#,
        ct.origin_mm[0], ct.origin_mm[1], ct.origin_mm[2]
    )?;
    writeln!(
        out,
        r#"<text x="650" y="838">RITK direction rows: [{:.2}, {:.2}, {:.2}] / [{:.2}, {:.2}, {:.2}] / [{:.2}, {:.2}, {:.2}]</text>"#,
        ct.direction[0][0],
        ct.direction[0][1],
        ct.direction[0][2],
        ct.direction[1][0],
        ct.direction[1][1],
        ct.direction[1][2],
        ct.direction[2][0],
        ct.direction[2][1],
        ct.direction[2][2]
    )?;
    writeln!(out, "</g>")?;
    Ok(())
}

fn write_obj(
    path: &Path,
    ct: &CtVolume,
    skull: &SkullSample,
    elements: &[ElementProjection],
) -> Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(
        out,
        "# RITK-derived skull CT and hemispherical array point geometry"
    )?;
    writeln!(
        out,
        "# Approximate AC-PC plane is local z=0 through volume center; array local +z is superior and concavity faces inferiorly."
    )?;
    writeln!(
        out,
        "# RITK origin mm {:.9} {:.9} {:.9}",
        ct.origin_mm[0], ct.origin_mm[1], ct.origin_mm[2]
    )?;
    writeln!(out, "g skull_boundary_hu_ge_300")?;
    for point in &skull.points {
        writeln!(out, "v {:.9} {:.9} {:.9}", point.x, point.y, point.z)?;
    }
    write_point_indices(&mut out, 1, skull.points.len())?;

    let air_cavities = sample_anterior_air_cavities(ct, skull);
    let avoidance = orbital_avoidance_zone(skull, &air_cavities);
    let array = array_diagnostic_mm(skull, &avoidance, elements);
    let first_element = skull.points.len() + 1;
    writeln!(out, "g array_elements_phase_colored_by_comment_order")?;
    writeln!(
        out,
        "# natural_focus_mm {:.9} {:.9} {:.9}",
        array.natural_focus.x, array.natural_focus.y, array.natural_focus.z
    )?;
    for element in array.elements {
        writeln!(
            out,
            "# phase_correction_rad {:.12} disabled_by_orbital_nasal_avoidance {}",
            element.phase, element.disabled
        )?;
        writeln!(
            out,
            "v {:.9} {:.9} {:.9}",
            element.point.x, element.point.y, element.point.z
        )?;
    }
    write_point_indices(&mut out, first_element, elements.len())?;
    Ok(())
}

fn write_point_indices<W: Write>(out: &mut W, start: usize, count: usize) -> Result<()> {
    for chunk_start in (start..start + count).step_by(16) {
        write!(out, "p")?;
        for idx in chunk_start..(chunk_start + 16).min(start + count) {
            write!(out, " {idx}")?;
        }
        writeln!(out)?;
    }
    Ok(())
}

fn array_diagnostic_mm(
    skull: &SkullSample,
    avoidance: &AvoidanceZone,
    elements: &[ElementProjection],
) -> ArrayDiagnostic {
    let (min_x, max_x, min_y, max_y) = elements.iter().fold(
        (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ),
        |(min_x, max_x, min_y, max_y), element| {
            (
                min_x.min(element.x_m),
                max_x.max(element.x_m),
                min_y.min(element.y_m),
                max_y.max(element.y_m),
            )
        },
    );
    let aperture_center_x = 0.5 * (min_x + max_x) * 1e3;
    let aperture_center_y = 0.5 * (min_y + max_y) * 1e3;
    let skull_center_x = 0.5 * (skull.min.x + skull.max.x);
    let skull_center_y = 0.5 * (skull.min.y + skull.max.y);
    let rim_z = skull.min.z + 0.18 * (skull.max.z - skull.min.z);
    let natural_focus_untilted = Point3 {
        x: skull_center_x,
        y: skull_center_y,
        z: rim_z,
    };
    let mut points: Vec<ElementPoint> = elements
        .iter()
        .map(|element| {
            let untilted = Point3 {
                x: skull_center_x + (element.x_m * 1e3 - aperture_center_x),
                y: skull_center_y - (element.y_m * 1e3 - aperture_center_y),
                z: rim_z + element.bowl_z_m * 1e3,
            };
            let point = transducer_pose(untilted, natural_focus_untilted);
            ElementPoint {
                point,
                phase: element.correction_rad,
                disabled: false,
            }
        })
        .collect();
    let array_center = points.iter().fold(
        Point3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        |sum, element| Point3 {
            x: sum.x + element.point.x,
            y: sum.y + element.point.y,
            z: sum.z + element.point.z,
        },
    );
    let inv_n = 1.0 / points.len().max(1) as f64;
    let shift_x = skull_center_x - array_center.x * inv_n;
    let shift_y = skull_center_y - array_center.y * inv_n;
    let natural_focus = Point3 {
        x: natural_focus_untilted.x + shift_x,
        y: natural_focus_untilted.y + shift_y,
        z: natural_focus_untilted.z,
    };
    for element in &mut points {
        element.point.x += shift_x;
        element.point.y += shift_y;
        element.disabled = segment_intersects_ellipse(element.point, natural_focus, avoidance);
    }
    ArrayDiagnostic {
        elements: points,
        natural_focus,
    }
}

#[derive(Debug, Clone)]
struct AvoidanceZone {
    center: Point3,
    radius_y: f64,
    radius_z: f64,
    clearance_y: f64,
    clearance_z: f64,
    air_points: Vec<Point3>,
}

fn orbital_avoidance_zone(skull: &SkullSample, air: &AirCavitySample) -> AvoidanceZone {
    let ap = skull.max.y - skull.min.y;
    let si = skull.max.z - skull.min.z;
    if !air.points.is_empty() {
        let (min_y, max_y) = percentile_bounds(air.points.iter().map(|point| point.y), 0.05, 0.95);
        let (min_z, max_z) = percentile_bounds(air.points.iter().map(|point| point.z), 0.05, 0.95);
        let center = Point3 {
            x: 0.5 * (air.min.x + air.max.x),
            y: 0.5 * (min_y + max_y),
            z: 0.5 * (min_z + max_z),
        };
        return AvoidanceZone {
            center,
            radius_y: (0.58 * (max_y - min_y)).max(0.055 * ap),
            radius_z: (0.62 * (max_z - min_z)).max(0.070 * si),
            clearance_y: 0.022 * ap,
            clearance_z: 0.020 * si,
            air_points: air.points.clone(),
        };
    }
    AvoidanceZone {
        center: Point3 {
            x: 0.0,
            y: skull.max.y - 0.18 * ap,
            z: skull.max.z - 0.34 * si,
        },
        radius_y: 0.12 * ap,
        radius_z: 0.14 * si,
        clearance_y: 0.12 * ap,
        clearance_z: 0.14 * si,
        air_points: Vec::new(),
    }
}

fn percentile_bounds<I>(values: I, lower: f64, upper: f64) -> (f64, f64)
where
    I: Iterator<Item = f64>,
{
    let mut sorted: Vec<f64> = values.filter(|value| value.is_finite()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let last = sorted.len().saturating_sub(1);
    let lower_idx = ((last as f64) * lower).round() as usize;
    let upper_idx = ((last as f64) * upper).round() as usize;
    (sorted[lower_idx.min(last)], sorted[upper_idx.min(last)])
}

fn segment_intersects_ellipse(source: Point3, target: Point3, zone: &AvoidanceZone) -> bool {
    if !zone.air_points.is_empty() {
        return zone.air_points.iter().any(|point| {
            segment_intersects_point_neighborhood(
                source,
                target,
                *point,
                zone.clearance_y,
                zone.clearance_z,
            )
        });
    }
    let dy = target.y - source.y;
    let dz = target.z - source.z;
    let sy = source.y - zone.center.y;
    let sz = source.z - zone.center.z;
    let a = (dy / zone.radius_y).powi(2) + (dz / zone.radius_z).powi(2);
    let b = 2.0 * (sy * dy / zone.radius_y.powi(2) + sz * dz / zone.radius_z.powi(2));
    let c = (sy / zone.radius_y).powi(2) + (sz / zone.radius_z).powi(2) - 1.0;
    if a <= f64::EPSILON {
        return c <= 0.0;
    }
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return false;
    }
    let root = discriminant.sqrt();
    let t0 = (-b - root) / (2.0 * a);
    let t1 = (-b + root) / (2.0 * a);
    (0.0..=1.0).contains(&t0) || (0.0..=1.0).contains(&t1)
}

fn segment_intersects_point_neighborhood(
    source: Point3,
    target: Point3,
    point: Point3,
    clearance_y: f64,
    clearance_z: f64,
) -> bool {
    let dy = (target.y - source.y) / clearance_y;
    let dz = (target.z - source.z) / clearance_z;
    let py = (point.y - source.y) / clearance_y;
    let pz = (point.z - source.z) / clearance_z;
    let length_sq = dy * dy + dz * dz;
    if length_sq <= f64::EPSILON {
        let normalized_sq = py * py + pz * pz;
        return normalized_sq <= 1.0;
    }
    let t = ((py * dy + pz * dz) / length_sq).clamp(0.0, 1.0);
    let nearest_y = t * dy;
    let nearest_z = t * dz;
    let distance_sq = (py - nearest_y).powi(2) + (pz - nearest_z).powi(2);
    distance_sq <= 1.0
}

fn transducer_pose(point: Point3, pivot: Point3) -> Point3 {
    let pitch = TRANSDUCER_POSTERIOR_TILT_DEG.to_radians();
    let dy = point.y - pivot.y;
    let dz = point.z - pivot.z;
    let y = pivot.y + dy * pitch.cos() - dz * pitch.sin();
    let z = pivot.z + dy * pitch.sin() + dz * pitch.cos();
    Point3 { x: point.x, y, z }
}

fn ac_pc_plane_points(skull: &SkullSample) -> [Point3; 4] {
    let margin = 18.0;
    [
        Point3 {
            x: skull.min.x - margin,
            y: skull.min.y - margin,
            z: 0.0,
        },
        Point3 {
            x: skull.max.x + margin,
            y: skull.min.y - margin,
            z: 0.0,
        },
        Point3 {
            x: skull.max.x + margin,
            y: skull.max.y + margin,
            z: 0.0,
        },
        Point3 {
            x: skull.min.x - margin,
            y: skull.max.y + margin,
            z: 0.0,
        },
    ]
}

fn projection_bounds(points: &[Point2]) -> (Point2, Point2) {
    let mut min = Point2 {
        x: f64::INFINITY,
        y: f64::INFINITY,
        depth: 0.0,
    };
    let mut max = Point2 {
        x: f64::NEG_INFINITY,
        y: f64::NEG_INFINITY,
        depth: 0.0,
    };
    for point in points {
        min.x = min.x.min(point.x);
        min.y = min.y.min(point.y);
        max.x = max.x.max(point.x);
        max.y = max.y.max(point.y);
    }
    (min, max)
}

fn transform(point: Point2, scale: f64, origin: Point2) -> Point2 {
    Point2 {
        x: origin.x + scale * point.x,
        y: origin.y + scale * point.y,
        depth: point.depth,
    }
}

fn project(point: Point3) -> Point2 {
    // Sagittal diagnostic projection: screen x is AP-like local y; screen y
    // is negative anatomical z so superior is visually up. Depth keeps left-
    // right information only for painter's-order sorting.
    Point2 {
        x: point.y,
        y: -point.z,
        depth: point.x,
    }
}

fn phase_color_hex(phase: f64) -> String {
    let wrapped = (phase + PI).rem_euclid(TAU) - PI;
    let t = (wrapped + PI) / TAU;
    let r = (255.0 * (1.0 - (2.0 * (t - 0.0).abs()).min(1.0))) as u8;
    let g = (255.0 * (1.0 - (2.0 * (t - 0.5).abs()).min(1.0))) as u8;
    let b = (255.0 * (1.0 - (2.0 * (t - 1.0).abs()).min(1.0))) as u8;
    format!("#{r:02x}{g:02x}{b:02x}")
}
