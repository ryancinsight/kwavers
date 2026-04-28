use std::f64::consts::{PI, TAU};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::Result;

use super::{CtVolume, ElementProjection, HU_BONE_LOWER};

const MAX_SKULL_POINTS: usize = 50_000;
const SVG_WIDTH: f64 = 1200.0;
const SVG_HEIGHT: f64 = 900.0;
const DISPLAY_HEAD_PITCH_DEG: f64 = -15.0;

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
                let point = display_head_pose(Point3 {
                    x: (x as f64 - center_x) * sx,
                    y: (y as f64 - center_y) * sy,
                    // This diagnostic frame preserves the RITK slice ordering
                    // that places the cranial vault superior to the skull base
                    // for the selected CT series.
                    z: (z as f64 - center_z) * sz,
                });
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
    let element_points = element_points_mm(ct, skull, elements);
    let mut projected = Vec::with_capacity(skull.points.len() + element_points.len() + 4);
    for point in &skull.points {
        projected.push(project(*point));
    }
    for (point, _) in &element_points {
        projected.push(project(*point));
    }
    for point in ac_pc_plane_points(skull) {
        projected.push(project(point));
    }
    projected.push(project(focus_point(skull)));

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
    writeln!(out, r##"<rect width="100%" height="100%" fill="#f8fafc"/>"##)?;
    writeln!(
        out,
        r##"<text x="40" y="44" font-family="Arial" font-size="24" fill="#0f172a">RITK skull CT inside 1024-element 650 kHz hemispherical array</text>"##
    )?;
    writeln!(
        out,
        r##"<text x="40" y="74" font-family="Arial" font-size="14" fill="#475569">Display pose: head pitched anterior-down inside the helmet so the ray fan avoids the anterior/orbital side.</text>"##
    )?;

    write_plane(&mut out, skull, scale, origin)?;
    write_skull_points(&mut out, skull, scale, origin)?;
    write_orbital_avoidance_zone(&mut out, skull, scale, origin)?;
    write_focus_rays(&mut out, skull, &element_points, scale, origin)?;
    write_element_points(&mut out, &element_points, scale, origin)?;
    write_orientation_axes(&mut out)?;
    write_legend(&mut out, ct, skull, elements)?;
    writeln!(out, "</svg>")?;
    Ok(())
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
    writeln!(out, r##"<g id="skull-boundary" fill="#64748b" fill-opacity="0.22">"##)?;
    for point in points {
        let p = transform(project(point), scale, origin);
        writeln!(out, r#"<circle cx="{:.2}" cy="{:.2}" r="0.58"/>"#, p.x, p.y)?;
    }
    writeln!(out, "</g>")?;
    Ok(())
}

fn write_element_points<W: Write>(
    out: &mut W,
    element_points: &[(Point3, f64)],
    scale: f64,
    origin: Point2,
) -> Result<()> {
    let mut points = element_points.to_vec();
    points.sort_by(|a, b| {
        project(a.0)
            .depth
            .partial_cmp(&project(b.0).depth)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    writeln!(out, r##"<g id="hemispherical-array" stroke="#0f172a" stroke-opacity="0.45" stroke-width="0.35">"##)?;
    for (point, phase) in points {
        let p = transform(project(point), scale, origin);
        let color = phase_color_hex(phase);
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
    skull: &SkullSample,
    element_points: &[(Point3, f64)],
    scale: f64,
    origin: Point2,
) -> Result<()> {
    let focus = focus_point(skull);
    let avoidance = orbital_avoidance_zone(skull);
    let projected_focus = transform(project(focus), scale, origin);
    writeln!(
        out,
        r##"<g id="downward-acoustic-rays" stroke="#2563eb" stroke-width="0.65" stroke-opacity="0.38" fill="none">"##
    )?;
    for (idx, (point, _)) in element_points.iter().enumerate() {
        if idx % 24 != 0 {
            continue;
        }
        if segment_intersects_ellipse(*point, focus, avoidance) {
            continue;
        }
        let source = transform(project(*point), scale, origin);
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
        r#"<text x="{:.2}" y="{:.2}">inferior focus / down-facing normals</text>"#,
        projected_focus.x + 10.0,
        projected_focus.y + 4.0
    )?;
    writeln!(out, "</g>")?;
    Ok(())
}

fn write_orbital_avoidance_zone<W: Write>(
    out: &mut W,
    skull: &SkullSample,
    scale: f64,
    origin: Point2,
) -> Result<()> {
    let zone = orbital_avoidance_zone(skull);
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
        r##"<text x="{:.2}" y="{:.2}" font-family="Arial" font-size="14" fill="#b91c1c">schematic orbital avoidance zone</text>"##,
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
    writeln!(out, r##"<path d="M1090 590 L1083 604 L1097 604 Z" fill="#0f172a"/>"##)?;
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
    elements: &[ElementProjection],
) -> Result<()> {
    let nonzero = elements
        .iter()
        .filter(|element| element.correction_rad.abs() > 1e-12)
        .count();
    writeln!(
        out,
        r##"<g font-family="Arial" font-size="14" fill="#0f172a">"##
    )?;
    writeln!(out, r#"<text x="40" y="815">gray: HU >= 300 skull boundary sampled from RITK-loaded CT</text>"#)?;
    writeln!(out, r#"<text x="40" y="838">colored points: 1024 array elements, color = wrapped correction phase [-pi, pi]</text>"#)?;
    writeln!(
        out,
        r#"<text x="40" y="861">nonzero corrections: {nonzero}/{}; skull boundary points: {}</text>"#,
        elements.len(),
        skull.points.len()
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
    writeln!(out, "# RITK-derived skull CT and hemispherical array point geometry")?;
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

    let first_element = skull.points.len() + 1;
    writeln!(out, "g array_elements_phase_colored_by_comment_order")?;
    for (point, phase) in element_points_mm(ct, skull, elements) {
        writeln!(
            out,
            "# phase_correction_rad {:.12}",
            phase
        )?;
        writeln!(out, "v {:.9} {:.9} {:.9}", point.x, point.y, point.z)?;
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

fn element_points_mm(
    ct: &CtVolume,
    skull: &SkullSample,
    elements: &[ElementProjection],
) -> Vec<(Point3, f64)> {
    let (nx, ny, _) = ct.hu.dim();
    let center_x = 0.5 * (nx.saturating_sub(1) as f64) * ct.spacing_m[0] * 1e3;
    let center_y = 0.5 * (ny.saturating_sub(1) as f64) * ct.spacing_m[1] * 1e3;
    let rim_z = skull.min.z + 0.18 * (skull.max.z - skull.min.z);
    elements
        .iter()
        .map(|element| {
            (
                Point3 {
                    x: element.x_m * 1e3 - center_x,
                    y: element.y_m * 1e3 - center_y,
                    z: rim_z + element.bowl_z_m * 1e3,
                },
                element.correction_rad,
            )
        })
        .collect()
}

fn focus_point(skull: &SkullSample) -> Point3 {
    Point3 {
        x: 0.0,
        y: -0.12 * (skull.max.y - skull.min.y),
        z: 0.70 * skull.max.z + 0.30 * skull.min.z,
    }
}

#[derive(Debug, Clone, Copy)]
struct AvoidanceZone {
    center: Point3,
    radius_y: f64,
    radius_z: f64,
}

fn orbital_avoidance_zone(skull: &SkullSample) -> AvoidanceZone {
    let ap = skull.max.y - skull.min.y;
    let si = skull.max.z - skull.min.z;
    AvoidanceZone {
        center: Point3 {
            x: 0.0,
            y: skull.max.y - 0.18 * ap,
            z: skull.max.z - 0.34 * si,
        },
        radius_y: 0.12 * ap,
        radius_z: 0.14 * si,
    }
}

fn segment_intersects_ellipse(source: Point3, target: Point3, zone: AvoidanceZone) -> bool {
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

fn display_head_pose(point: Point3) -> Point3 {
    let pitch = DISPLAY_HEAD_PITCH_DEG.to_radians();
    let y = point.y * pitch.cos() - point.z * pitch.sin();
    let z = point.y * pitch.sin() + point.z * pitch.cos();
    Point3 {
        x: point.x,
        y,
        z,
    }
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
