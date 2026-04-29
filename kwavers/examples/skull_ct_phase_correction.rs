//! Skull CT DICOM phase-correction example for an Insightec-style array.
//!
//! Usage:
//!
//! ```text
//! cargo run -p kwavers --features ritk --example skull_ct_phase_correction -- \
//!   <dicom_dir> <output.ppm> [series_instance_uid]
//! ```
//!
//! The example loads a skull CT DICOM series through RITK, converts HU to a
//! speed-of-sound skull map, computes thin-phase-screen correction at 650 kHz,
//! samples corrections for a 1024-element hemispherical array, and writes:
//!
//! - `<output.ppm>`: axial, coronal, and sagittal phase-correction panels.
//! - `<output>.csv`: per-element coordinates, correction phase, skull thickness,
//!   and cortical/trabecular density metrics.
//! - `<output>_element_maps.ppm` and `.svg`: 2D transducer element maps.

use std::env;
use std::f64::consts::{PI, TAU};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use burn::backend::NdArray;
use kwavers::domain::grid::Grid;
use kwavers::physics::acoustics::skull::{AberrationCorrection, HeterogeneousSkull};
use kwavers::physics::skull::heterogeneous::SkullLayer;
use ndarray::{Array1, Array2, Array3};
use ritk_io::{load_dicom_series, scan_dicom_directory};

#[path = "skull_ct_phase_correction/diagnostics_3d.rs"]
mod diagnostics_3d;

type Backend = NdArray<f32>;

const FREQUENCY_HZ: f64 = 650_000.0;
const ELEMENT_COUNT: usize = 1024;
const EXABLATE_HEMISPHERE_RADIUS_M: f64 = 0.150;
const C_WATER_M_PER_S: f64 = 1482.0;
const C_CORTICAL_BONE_M_PER_S: f64 = 2800.0;
const RHO_WATER_KG_PER_M3: f64 = 1000.0;
const RHO_CORTICAL_BONE_KG_PER_M3: f64 = 1900.0;
const HU_BONE_LOWER: f64 = 300.0;
const PANEL: usize = 384;
const MAP_PANEL: usize = 384;
const RAY_STEP_FRACTION: f64 = 0.5;

#[derive(Debug, Clone)]
struct CtVolume {
    hu: Array3<f64>,
    spacing_m: [f64; 3],
    origin_mm: [f64; 3],
    direction: [[f64; 3]; 3],
}

#[derive(Debug, Clone, Copy)]
struct Point3Meters {
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct ElementProjection {
    element: usize,
    pub(crate) x_m: f64,
    pub(crate) y_m: f64,
    pub(crate) bowl_z_m: f64,
    pub(crate) correction_rad: f64,
    skull_thickness_m: f64,
    cortical_thickness_m: f64,
    trabecular_thickness_m: f64,
    mean_density_kg_m3: f64,
    mean_sound_speed_m_s: f64,
    cortical_mean_density_kg_m3: Option<f64>,
    trabecular_mean_density_kg_m3: Option<f64>,
    trabecular_to_cortical_sdr: Option<f64>,
    cortical_to_trabecular_density_ratio: Option<f64>,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 || args.len() > 4 {
        bail!(
            "usage: {} <dicom_dir> <output.ppm> [series_instance_uid]",
            args.first().map(String::as_str).unwrap_or("example")
        );
    }

    let dicom_dir = PathBuf::from(&args[1]);
    let output_path = PathBuf::from(&args[2]);
    let series_uid = args.get(3).map(String::as_str);

    let ct = load_ct_with_ritk(&dicom_dir, series_uid)?;
    let skull = skull_from_hu(&ct);
    let (nx, ny, nz) = skull.sound_speed.dim();
    let grid = Grid::new(
        nx,
        ny,
        nz,
        ct.spacing_m[0],
        ct.spacing_m[1],
        ct.spacing_m[2],
    )
    .context("failed to create kwavers grid from CT spacing")?;

    let correction = AberrationCorrection::new(&grid, &skull).with_water_speed(C_WATER_M_PER_S);
    let correction_volume = correction
        .compute_correction_phases(FREQUENCY_HZ)
        .context("failed to compute volumetric phase correction")?;
    let aperture_phase = correction
        .aperture_phase_map(FREQUENCY_HZ)
        .context("failed to compute aperture phase map")?;

    let (element_x, element_y, element_z) = hemispherical_projected_elements(&grid);
    let focus = phase_target_point(&grid);
    let element_corrections = compute_focused_element_corrections(
        FREQUENCY_HZ,
        &element_x,
        &element_y,
        &element_z,
        focus,
        &grid,
        &skull,
    );
    let projections = element_projection_records(
        &element_x,
        &element_y,
        &element_z,
        &element_corrections,
        &grid,
        &ct,
        &skull,
        focus,
    );

    write_three_plane_ppm(
        &output_path,
        &ct.hu,
        &correction_volume,
        &aperture_phase,
        &projections,
        &grid,
    )?;
    write_element_csv(&output_path.with_extension("csv"), &projections)?;
    write_element_map_ppm(
        &diagnostics_3d::companion_path(&output_path, "_element_maps", "ppm"),
        &projections,
        &grid,
    )?;
    write_element_map_svg(
        &diagnostics_3d::companion_path(&output_path, "_element_maps", "svg"),
        &projections,
        &grid,
    )?;
    write_pressure_field_ppm(
        &diagnostics_3d::companion_path(&output_path, "_pressure_field", "ppm"),
        &ct.hu,
        &projections,
        focus,
        &grid,
        FREQUENCY_HZ,
    )?;
    diagnostics_3d::write_three_dimensional_diagnostics(&output_path, &ct, &projections)
        .context("failed to write 3D skull-array diagnostics")?;

    println!(
        "Loaded CT {}x{}x{} voxels, spacing [{:.3}, {:.3}, {:.3}] mm",
        nx,
        ny,
        nz,
        1e3 * ct.spacing_m[0],
        1e3 * ct.spacing_m[1],
        1e3 * ct.spacing_m[2]
    );
    println!(
        "Computed {} element corrections at {:.0} kHz; wrote {}, {}, {}, {}, and {}",
        projections.len(),
        FREQUENCY_HZ / 1e3,
        output_path.display(),
        output_path.with_extension("csv").display(),
        diagnostics_3d::companion_path(&output_path, "_element_maps", "ppm").display(),
        diagnostics_3d::companion_path(&output_path, "_element_maps", "svg").display(),
        diagnostics_3d::companion_path(&output_path, "_pressure_field", "ppm").display()
    );
    println!(
        "Wrote 3D diagnostics {} and {}",
        diagnostics_3d::companion_path(&output_path, "_3d", "svg").display(),
        diagnostics_3d::companion_path(&output_path, "_3d", "obj").display()
    );
    Ok(())
}

fn load_ct_with_ritk(path: &Path, selected_uid: Option<&str>) -> Result<CtVolume> {
    let series = scan_dicom_directory(path)
        .with_context(|| format!("RITK failed to scan DICOM directory '{}'", path.display()))?;
    if series.is_empty() {
        bail!("no DICOM series found in '{}'", path.display());
    }

    let selected = match selected_uid {
        Some(uid) => series
            .iter()
            .find(|candidate| candidate.series_instance_uid == uid)
            .with_context(|| format!("series UID '{uid}' was not found"))?,
        None if series.len() == 1 => &series[0],
        None => {
            eprintln!("Multiple DICOM series found:");
            for item in &series {
                eprintln!(
                    "  UID={} modality={} description={} files={}",
                    item.series_instance_uid,
                    item.modality,
                    item.series_description,
                    item.file_paths.len()
                );
            }
            bail!("select one series by passing its SeriesInstanceUID");
        }
    };

    let device = Default::default();
    let image = load_dicom_series::<Backend>(selected, &device).with_context(|| {
        format!(
            "RITK failed to load series '{}'",
            selected.series_instance_uid
        )
    })?;
    let [depth, rows, cols] = image.shape();
    let spacing = image.spacing().to_vec();
    if spacing.len() != 3 {
        bail!("RITK returned invalid spacing rank {}", spacing.len());
    }
    let origin_vec = image.origin().to_vec();
    if origin_vec.len() != 3 {
        bail!("RITK returned invalid origin rank {}", origin_vec.len());
    }
    let direction_matrix = image.direction().0;

    let tensor_data = image.data().clone().into_data();
    let values = tensor_data
        .as_slice::<f32>()
        .map_err(|err| anyhow::anyhow!("RITK tensor data is not f32 contiguous: {err:?}"))?;
    if values.len() != depth * rows * cols {
        bail!(
            "RITK data length mismatch: got {}, expected {}",
            values.len(),
            depth * rows * cols
        );
    }

    let mut hu = Array3::<f64>::zeros((cols, rows, depth));
    for z in 0..depth {
        for y in 0..rows {
            for x in 0..cols {
                let src = z * rows * cols + y * cols + x;
                hu[[x, y, z]] = f64::from(values[src]);
            }
        }
    }

    Ok(CtVolume {
        hu,
        spacing_m: [spacing[0] * 1e-3, spacing[1] * 1e-3, spacing[2] * 1e-3],
        origin_mm: [origin_vec[0], origin_vec[1], origin_vec[2]],
        direction: [
            [
                direction_matrix[(0, 0)],
                direction_matrix[(0, 1)],
                direction_matrix[(0, 2)],
            ],
            [
                direction_matrix[(1, 0)],
                direction_matrix[(1, 1)],
                direction_matrix[(1, 2)],
            ],
            [
                direction_matrix[(2, 0)],
                direction_matrix[(2, 1)],
                direction_matrix[(2, 2)],
            ],
        ],
    })
}

fn skull_from_hu(ct: &CtVolume) -> HeterogeneousSkull {
    HeterogeneousSkull::from_ct_hill(
        &ct.hu,
        C_CORTICAL_BONE_M_PER_S,
        RHO_CORTICAL_BONE_KG_PER_M3,
        20.0,
    )
    .expect("Hill BVF skull model should accept finite RITK CT HU data")
}

fn hemispherical_projected_elements(grid: &Grid) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let center_x = 0.5 * (grid.nx.saturating_sub(1) as f64) * grid.dx;
    let center_y = 0.5 * (grid.ny.saturating_sub(1) as f64) * grid.dy;
    let grid_radius = 0.48 * (grid.nx as f64 * grid.dx).min(grid.ny as f64 * grid.dy);
    let projection_scale = (grid_radius / EXABLATE_HEMISPHERE_RADIUS_M).min(1.0);

    let golden_angle = PI * (3.0 - 5.0_f64.sqrt());
    let mut x = Vec::with_capacity(ELEMENT_COUNT);
    let mut y = Vec::with_capacity(ELEMENT_COUNT);
    let mut z = Vec::with_capacity(ELEMENT_COUNT);

    for n in 0..ELEMENT_COUNT {
        let u = (n as f64 + 0.5) / ELEMENT_COUNT as f64;
        let polar = u.acos();
        let azimuth = n as f64 * golden_angle;
        let r_xy = EXABLATE_HEMISPHERE_RADIUS_M * polar.sin();
        let bowl_x = r_xy * azimuth.cos();
        let bowl_y = r_xy * azimuth.sin();
        // Local +z is superior in the diagnostic coordinate frame: the
        // hemispherical cap sits above the skull with its concavity directed
        // inferiorly toward the neck.
        let bowl_z = EXABLATE_HEMISPHERE_RADIUS_M * polar.cos();
        x.push(center_x + projection_scale * bowl_x);
        y.push(center_y + projection_scale * bowl_y);
        z.push(bowl_z);
    }

    (x, y, z)
}

fn phase_target_point(grid: &Grid) -> Point3Meters {
    Point3Meters {
        x: 0.5 * (grid.nx.saturating_sub(1) as f64) * grid.dx,
        y: 0.5 * (grid.ny.saturating_sub(1) as f64) * grid.dy,
        z: 0.5 * (grid.nz.saturating_sub(1) as f64) * grid.dz,
    }
}

fn element_source_point(x_m: f64, y_m: f64, bowl_z_m: f64, grid: &Grid) -> Point3Meters {
    Point3Meters {
        x: x_m,
        y: y_m,
        z: (grid.nz.saturating_sub(1) as f64) * grid.dz + bowl_z_m,
    }
}

fn nearest_voxel(point: Point3Meters, grid: &Grid) -> Option<(usize, usize, usize)> {
    if point.x < 0.0 || point.y < 0.0 || point.z < 0.0 {
        return None;
    }
    let i = (point.x / grid.dx).round() as isize;
    let j = (point.y / grid.dy).round() as isize;
    let k = (point.z / grid.dz).round() as isize;
    if i < 0
        || j < 0
        || k < 0
        || i >= grid.nx as isize
        || j >= grid.ny as isize
        || k >= grid.nz as isize
    {
        return None;
    }
    Some((i as usize, j as usize, k as usize))
}

fn ray_step_m(grid: &Grid) -> f64 {
    RAY_STEP_FRACTION * grid.dx.min(grid.dy).min(grid.dz)
}

fn compute_focused_element_corrections(
    frequency: f64,
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
    focus: Point3Meters,
    grid: &Grid,
    skull: &HeterogeneousSkull,
) -> Array1<f64> {
    let k_water = 2.0 * PI * frequency / C_WATER_M_PER_S;
    let step = ray_step_m(grid);
    let mut corrections = Array1::zeros(xs.len().min(ys.len()).min(zs.len()));

    for (idx, ((&x_m, &y_m), &bowl_z_m)) in xs.iter().zip(ys).zip(zs).enumerate() {
        let source = element_source_point(x_m, y_m, bowl_z_m, grid);
        let dx = focus.x - source.x;
        let dy = focus.y - source.y;
        let dz = focus.z - source.z;
        let length = (dx * dx + dy * dy + dz * dz).sqrt();
        if length <= f64::EPSILON {
            continue;
        }
        let samples = (length / step).ceil().max(1.0) as usize;
        let ds = length / samples as f64;
        let mut phase = 0.0_f64;
        for s in 0..=samples {
            let u = s as f64 / samples as f64;
            let point = Point3Meters {
                x: source.x + u * dx,
                y: source.y + u * dy,
                z: source.z + u * dz,
            };
            let Some((i, j, k)) = nearest_voxel(point, grid) else {
                continue;
            };
            let c_local = skull.sound_speed[[i, j, k]];
            if c_local <= 0.0 {
                continue;
            }
            let k_local = 2.0 * PI * frequency / c_local;
            phase += (k_local - k_water) * ds;
        }
        corrections[idx] = -phase;
    }

    corrections
}

fn element_projection_records(
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
    corrections: &Array1<f64>,
    grid: &Grid,
    ct: &CtVolume,
    skull: &HeterogeneousSkull,
    focus: Point3Meters,
) -> Vec<ElementProjection> {
    xs.iter()
        .zip(ys)
        .zip(zs)
        .zip(corrections)
        .enumerate()
        .map(|(element, (((&x_m, &y_m), &bowl_z_m), &correction_rad))| {
            let metrics = element_path_metrics(x_m, y_m, bowl_z_m, focus, grid, ct, skull);
            ElementProjection {
                element,
                x_m,
                y_m,
                bowl_z_m,
                correction_rad: wrap_phase(correction_rad),
                skull_thickness_m: metrics.skull_thickness_m,
                cortical_thickness_m: metrics.cortical_thickness_m,
                trabecular_thickness_m: metrics.trabecular_thickness_m,
                mean_density_kg_m3: metrics.mean_density_kg_m3,
                mean_sound_speed_m_s: metrics.mean_sound_speed_m_s,
                cortical_mean_density_kg_m3: metrics.cortical_mean_density_kg_m3,
                trabecular_mean_density_kg_m3: metrics.trabecular_mean_density_kg_m3,
                trabecular_to_cortical_sdr: metrics.trabecular_to_cortical_sdr,
                cortical_to_trabecular_density_ratio: metrics.cortical_to_trabecular_density_ratio,
            }
        })
        .collect()
}

#[derive(Debug, Clone, Copy)]
struct ElementPathMetrics {
    skull_thickness_m: f64,
    cortical_thickness_m: f64,
    trabecular_thickness_m: f64,
    mean_density_kg_m3: f64,
    mean_sound_speed_m_s: f64,
    cortical_mean_density_kg_m3: Option<f64>,
    trabecular_mean_density_kg_m3: Option<f64>,
    trabecular_to_cortical_sdr: Option<f64>,
    cortical_to_trabecular_density_ratio: Option<f64>,
}

fn element_path_metrics(
    x_m: f64,
    y_m: f64,
    bowl_z_m: f64,
    focus: Point3Meters,
    grid: &Grid,
    ct: &CtVolume,
    skull: &HeterogeneousSkull,
) -> ElementPathMetrics {
    let source = element_source_point(x_m, y_m, bowl_z_m, grid);
    let dx = focus.x - source.x;
    let dy = focus.y - source.y;
    let dz = focus.z - source.z;
    let length = (dx * dx + dy * dy + dz * dz).sqrt();
    let step = ray_step_m(grid);
    let samples = (length / step).ceil().max(1.0) as usize;
    let ds = length / samples as f64;
    let mut skull_count = 0_usize;
    let mut cortical_count = 0_usize;
    let mut trabecular_count = 0_usize;
    let mut density_sum = 0.0_f64;
    let mut sound_speed_sum = 0.0_f64;
    let mut cortical_density_sum = 0.0_f64;
    let mut trabecular_density_sum = 0.0_f64;

    for s in 0..=samples {
        let u = s as f64 / samples as f64;
        let point = Point3Meters {
            x: source.x + u * dx,
            y: source.y + u * dy,
            z: source.z + u * dz,
        };
        let Some((i, j, k)) = nearest_voxel(point, grid) else {
            continue;
        };
        let hu = ct.hu[[i, j, k]];
        if hu < HU_BONE_LOWER {
            continue;
        }
        skull_count += 1;
        density_sum += skull.density[[i, j, k]];
        sound_speed_sum += skull.sound_speed[[i, j, k]];
        match HeterogeneousSkull::classify_layer(hu) {
            SkullLayer::Cortical => {
                cortical_count += 1;
                cortical_density_sum += skull.density[[i, j, k]];
            }
            SkullLayer::Diploe => {
                trabecular_count += 1;
                trabecular_density_sum += skull.density[[i, j, k]];
            }
            SkullLayer::SoftTissue => {}
        }
    }

    let mean_density = if skull_count > 0 {
        density_sum / skull_count as f64
    } else {
        RHO_WATER_KG_PER_M3
    };
    let mean_sound_speed = if skull_count > 0 {
        sound_speed_sum / skull_count as f64
    } else {
        C_WATER_M_PER_S
    };
    let cortical_mean_density =
        (cortical_count > 0).then_some(cortical_density_sum / cortical_count as f64);
    let trabecular_mean_density =
        (trabecular_count > 0).then_some(trabecular_density_sum / trabecular_count as f64);
    let trabecular_to_cortical_sdr = trabecular_mean_density
        .zip(cortical_mean_density)
        .map(|(trabecular, cortical)| trabecular / cortical);
    let cortical_to_trabecular_density_ratio = cortical_mean_density
        .zip(trabecular_mean_density)
        .map(|(cortical, trabecular)| cortical / trabecular);

    ElementPathMetrics {
        skull_thickness_m: skull_count as f64 * ds,
        cortical_thickness_m: cortical_count as f64 * ds,
        trabecular_thickness_m: trabecular_count as f64 * ds,
        mean_density_kg_m3: mean_density,
        mean_sound_speed_m_s: mean_sound_speed,
        cortical_mean_density_kg_m3: cortical_mean_density,
        trabecular_mean_density_kg_m3: trabecular_mean_density,
        trabecular_to_cortical_sdr,
        cortical_to_trabecular_density_ratio,
    }
}

fn compute_pressure_slice(
    elements: &[ElementProjection],
    focus: Point3Meters,
    grid: &Grid,
    frequency_hz: f64,
) -> Array2<f64> {
    let mut pressure = Array2::<f64>::zeros((grid.ny, grid.nz));
    let mid_x = 0.5 * (grid.nx.saturating_sub(1) as f64) * grid.dx;
    let k_water = 2.0 * PI * frequency_hz / C_WATER_M_PER_S;

    for j in 0..grid.ny {
        for k in 0..grid.nz {
            let target = Point3Meters {
                x: mid_x,
                y: j as f64 * grid.dy,
                z: k as f64 * grid.dz,
            };
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            let mut incoherent = 0.0_f64;

            for element in elements {
                let source = element_source_point(element.x_m, element.y_m, element.bowl_z_m, grid);
                let dx = target.x - source.x;
                let dy = target.y - source.y;
                let dz = target.z - source.z;
                let distance = (dx * dx + dy * dy + dz * dz).sqrt().max(grid.dx);
                let fdx = focus.x - source.x;
                let fdy = focus.y - source.y;
                let fdz = focus.z - source.z;
                let focus_distance = (fdx * fdx + fdy * fdy + fdz * fdz).sqrt().max(grid.dx);
                let attenuation_np =
                    20.0 * (frequency_hz * 1e-6) * element.skull_thickness_m * 100.0 / 8.686;
                let phase = k_water * (distance - focus_distance) + element.correction_rad;
                let amplitude =
                    element_transmission_gain(element) * (-attenuation_np).exp() / distance;
                re += amplitude * phase.cos();
                im += amplitude * phase.sin();
                incoherent += amplitude;
            }

            pressure[[j, k]] = if incoherent > f64::EPSILON {
                (re * re + im * im).sqrt() / incoherent
            } else {
                0.0
            };
        }
    }

    pressure
}

fn element_transmission_gain(element: &ElementProjection) -> f64 {
    if element.skull_thickness_m <= 0.0 {
        return 1.0;
    }
    let z_water = RHO_WATER_KG_PER_M3 * C_WATER_M_PER_S;
    let z_skull = element.mean_density_kg_m3 * element.mean_sound_speed_m_s;
    let reflection = ((z_skull - z_water) / (z_skull + z_water)).abs();
    (1.0 - reflection * reflection).clamp(0.0, 1.0)
}

fn write_three_plane_ppm(
    path: &Path,
    hu: &Array3<f64>,
    correction: &Array3<f64>,
    aperture: &Array2<f64>,
    elements: &[ElementProjection],
    grid: &Grid,
) -> Result<()> {
    let width = PANEL * 3;
    let height = PANEL;
    let mut rgb = vec![0_u8; width * height * 3];
    let (nx, ny, nz) = hu.dim();
    let mid_x = nx / 2;
    let mid_y = ny / 2;
    let mid_z = nz / 2;

    let axial = PlaneSpec::new(0, "axial", nx, ny, |x, y| (x, y, mid_z));
    let coronal = PlaneSpec::new(1, "coronal", nx, nz, |x, z| (x, mid_y, z));
    let sagittal = PlaneSpec::new(2, "sagittal", ny, nz, |y, z| (mid_x, y, z));

    draw_plane(&mut rgb, width, height, &axial, hu, correction);
    draw_plane(&mut rgb, width, height, &coronal, hu, correction);
    draw_plane(&mut rgb, width, height, &sagittal, hu, correction);
    draw_elements_on_axial(&mut rgb, width, height, elements, grid);
    draw_aperture_phase_strip(&mut rgb, width, height, aperture);

    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "P6\n{} {}\n255", width, height)?;
    out.write_all(&rgb)?;
    Ok(())
}

fn write_pressure_field_ppm(
    path: &Path,
    hu: &Array3<f64>,
    elements: &[ElementProjection],
    focus: Point3Meters,
    grid: &Grid,
    frequency_hz: f64,
) -> Result<()> {
    let width = PANEL;
    let height = PANEL;
    let mut rgb = vec![8_u8; width * height * 3];
    let pressure = compute_pressure_slice(elements, focus, grid, frequency_hz);
    let max_pressure = finite_max(pressure.iter().copied()).max(f64::EPSILON);
    let mid_x = grid.nx / 2;

    for py in 0..height {
        for px in 0..width {
            let j = ((width - 1 - px) * grid.ny / width).min(grid.ny - 1);
            let k = ((height - 1 - py) * grid.nz / height).min(grid.nz - 1);
            let db = 20.0 * (pressure[[j, k]] / max_pressure).max(1.0e-4).log10();
            let pressure_rgb = pressure_color(db);
            let gray = ct_window(hu[[mid_x, j, k]]);
            let mut color = blend_gray_color(gray, pressure_rgb, 0.72);
            if hu[[mid_x, j, k]] >= HU_BONE_LOWER {
                color = blend_rgb(color, pressure_rgb, 0.70);
                color = blend_rgb(color, [245, 245, 220], 0.22);
            }
            put_pixel(&mut rgb, width, height, px, py, color);
        }
    }

    draw_pressure_transducer_overlay(&mut rgb, width, height, elements, focus, grid);

    let focus_px = (width as isize - 1)
        - ((focus.y / (grid.ny as f64 * grid.dy)) * width as f64).round() as isize;
    let focus_py = (height as isize - 1)
        - ((focus.z / (grid.nz as f64 * grid.dz)) * height as f64).round() as isize;
    draw_cross(
        &mut rgb,
        width,
        height,
        focus_px,
        focus_py,
        6,
        [255, 255, 255],
    );

    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "P6\n{} {}\n255", width, height)?;
    out.write_all(&rgb)?;
    Ok(())
}

fn pressure_color(db: f64) -> [u8; 3] {
    let t = ((db + 24.0) / 24.0).clamp(0.0, 1.0);
    let r = (255.0 * t.powf(0.65)) as u8;
    let g = (220.0 * (1.0 - (2.0 * t - 1.0).abs()).max(0.0)) as u8;
    let b = (255.0 * (1.0 - t).powf(0.8)) as u8;
    [r, g, b]
}

fn draw_pressure_transducer_overlay(
    rgb: &mut [u8],
    width: usize,
    height: usize,
    elements: &[ElementProjection],
    focus: Point3Meters,
    grid: &Grid,
) {
    let focus_px = (width as isize - 1)
        - ((focus.y / (grid.ny as f64 * grid.dy)) * width as f64).round() as isize;
    let focus_py = (height as isize - 1)
        - ((focus.z / (grid.nz as f64 * grid.dz)) * height as f64).round() as isize;

    for (idx, element) in elements.iter().enumerate() {
        let (px, py) = meridional_transducer_pixel(element, width, height, grid);
        if idx % 32 == 0 && px >= 0 && (px as usize) < width {
            draw_line_blend(
                rgb,
                width,
                height,
                px,
                py,
                focus_px,
                focus_py,
                [125, 211, 252],
                0.20,
            );
        }
    }

    for element in elements {
        let (px, py) = meridional_transducer_pixel(element, width, height, grid);
        if px >= 0 && (px as usize) < width {
            draw_disc(
                rgb,
                width,
                height,
                px,
                py,
                2,
                phase_color(element.correction_rad),
            );
        }
    }
}

fn meridional_transducer_pixel(
    element: &ElementProjection,
    width: usize,
    height: usize,
    grid: &Grid,
) -> (isize, isize) {
    let px = (width as isize - 1)
        - ((element.y_m / (grid.ny as f64 * grid.dy)) * width as f64).round() as isize;
    let z_norm = (element.bowl_z_m / EXABLATE_HEMISPHERE_RADIUS_M).clamp(0.0, 1.0);
    let py = ((0.04 + 0.23 * (1.0 - z_norm)) * height as f64).round() as isize;
    (px, py)
}

struct PlaneSpec<F>
where
    F: Fn(usize, usize) -> (usize, usize, usize),
{
    panel: usize,
    _name: &'static str,
    src_w: usize,
    src_h: usize,
    index: F,
}

impl<F> PlaneSpec<F>
where
    F: Fn(usize, usize) -> (usize, usize, usize),
{
    fn new(panel: usize, name: &'static str, src_w: usize, src_h: usize, index: F) -> Self {
        Self {
            panel,
            _name: name,
            src_w,
            src_h,
            index,
        }
    }
}

fn draw_plane<F>(
    rgb: &mut [u8],
    width: usize,
    height: usize,
    plane: &PlaneSpec<F>,
    hu: &Array3<f64>,
    correction: &Array3<f64>,
) where
    F: Fn(usize, usize) -> (usize, usize, usize),
{
    for py in 0..PANEL {
        for px in 0..PANEL {
            let sx = px * plane.src_w / PANEL;
            let sy = (PANEL - 1 - py) * plane.src_h / PANEL;
            let (i, j, k) = (plane.index)(sx.min(plane.src_w - 1), sy.min(plane.src_h - 1));
            let gray = ct_window(hu[[i, j, k]]);
            let phase_rgb = phase_color(correction[[i, j, k]]);
            let blended = blend_gray_color(gray, phase_rgb, 0.62);
            put_pixel(rgb, width, height, plane.panel * PANEL + px, py, blended);
        }
    }
}

fn draw_elements_on_axial(
    rgb: &mut [u8],
    width: usize,
    height: usize,
    elements: &[ElementProjection],
    grid: &Grid,
) {
    for element in elements {
        let px = ((element.x_m / (grid.nx as f64 * grid.dx)) * PANEL as f64).round() as isize;
        let py = (PANEL as isize - 1)
            - ((element.y_m / (grid.ny as f64 * grid.dy)) * PANEL as f64).round() as isize;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let x = px + dx;
                let y = py + dy;
                if x >= 0 && y >= 0 {
                    put_pixel(rgb, width, height, x as usize, y as usize, [255, 255, 255]);
                }
            }
        }
    }
}

fn draw_aperture_phase_strip(rgb: &mut [u8], width: usize, height: usize, aperture: &Array2<f64>) {
    let (nx, ny) = aperture.dim();
    let strip_h = 20;
    for px in 0..PANEL {
        let ix = px * nx / PANEL;
        for py in 0..strip_h {
            let iy = py * ny / strip_h;
            let color = phase_color(-aperture[[ix.min(nx - 1), iy.min(ny - 1)]]);
            put_pixel(rgb, width, height, px, height - strip_h + py, color);
        }
    }
}

fn write_element_map_ppm(path: &Path, elements: &[ElementProjection], grid: &Grid) -> Result<()> {
    let width = 2 * MAP_PANEL;
    let height = 2 * MAP_PANEL;
    let mut rgb = vec![248_u8; width * height * 3];
    draw_element_scalar_panel(
        &mut rgb,
        width,
        height,
        (0, 0),
        elements,
        grid,
        |element| Some(element.correction_rad),
        ScalarColor::Phase { min: -PI, max: PI },
    );
    draw_element_scalar_panel(
        &mut rgb,
        width,
        height,
        (1, 0),
        elements,
        grid,
        |element| Some(1e3 * element.skull_thickness_m),
        ScalarColor::Sequential {
            min: 0.0,
            max: finite_max(
                elements
                    .iter()
                    .map(|element| 1e3 * element.skull_thickness_m),
            )
            .max(1.0),
        },
    );
    draw_element_scalar_panel(
        &mut rgb,
        width,
        height,
        (0, 1),
        elements,
        grid,
        |element| element.trabecular_to_cortical_sdr,
        ScalarColor::Sequential { min: 0.0, max: 1.0 },
    );
    draw_element_scalar_panel(
        &mut rgb,
        width,
        height,
        (1, 1),
        elements,
        grid,
        |element| element.cortical_to_trabecular_density_ratio,
        ScalarColor::Sequential {
            min: 1.0,
            max: finite_max(
                elements
                    .iter()
                    .filter_map(|element| element.cortical_to_trabecular_density_ratio),
            )
            .max(1.0),
        },
    );

    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "P6\n{} {}\n255", width, height)?;
    out.write_all(&rgb)?;
    Ok(())
}

fn write_element_map_svg(path: &Path, elements: &[ElementProjection], grid: &Grid) -> Result<()> {
    let panel = 380.0_f64;
    let gutter = 54.0_f64;
    let margin = 56.0_f64;
    let width = 2.0 * panel + gutter + 2.0 * margin;
    let height = 2.0 * panel + gutter + 2.0 * margin + 46.0;
    let thickness_max = finite_max(
        elements
            .iter()
            .map(|element| 1e3 * element.skull_thickness_m),
    )
    .max(1.0);
    let ratio_max = finite_max(
        elements
            .iter()
            .filter_map(|element| element.cortical_to_trabecular_density_ratio),
    )
    .max(1.0);

    let mut out = BufWriter::new(File::create(path)?);
    writeln!(
        out,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">"#
    )?;
    writeln!(
        out,
        r##"<rect width="100%" height="100%" fill="#f8fafc"/>"##
    )?;
    writeln!(
        out,
        r##"<text x="40" y="32" font-family="Arial" font-size="22" fill="#0f172a">2D transducer element maps from RITK skull CT</text>"##
    )?;
    write_element_svg_panel(
        &mut out,
        (margin, margin + 28.0),
        panel,
        "phase correction [rad]",
        elements,
        grid,
        |element| Some(element.correction_rad),
        ScalarColor::Phase { min: -PI, max: PI },
    )?;
    write_element_svg_panel(
        &mut out,
        (margin + panel + gutter, margin + 28.0),
        panel,
        "skull thickness [mm]",
        elements,
        grid,
        |element| Some(1e3 * element.skull_thickness_m),
        ScalarColor::Sequential {
            min: 0.0,
            max: thickness_max,
        },
    )?;
    write_element_svg_panel(
        &mut out,
        (margin, margin + panel + gutter + 28.0),
        panel,
        "trabecular/cortical SDR",
        elements,
        grid,
        |element| element.trabecular_to_cortical_sdr,
        ScalarColor::Sequential { min: 0.0, max: 1.0 },
    )?;
    write_element_svg_panel(
        &mut out,
        (margin + panel + gutter, margin + panel + gutter + 28.0),
        panel,
        "cortical/trabecular density ratio",
        elements,
        grid,
        |element| element.cortical_to_trabecular_density_ratio,
        ScalarColor::Sequential {
            min: 1.0,
            max: ratio_max,
        },
    )?;
    writeln!(
        out,
        r##"<text x="40" y="{:.2}" font-family="Arial" font-size="13" fill="#475569">Gray elements have no valid cortical/trabecular pair along the element-to-focus ray; metrics are exported in the CSV.</text>"##,
        height - 18.0
    )?;
    writeln!(out, "</svg>")?;
    Ok(())
}

fn write_element_svg_panel<W, F>(
    out: &mut W,
    origin: (f64, f64),
    panel: f64,
    title: &str,
    elements: &[ElementProjection],
    grid: &Grid,
    value: F,
    color: ScalarColor,
) -> Result<()>
where
    W: Write,
    F: Fn(&ElementProjection) -> Option<f64>,
{
    writeln!(
        out,
        r##"<text x="{:.2}" y="{:.2}" font-family="Arial" font-size="15" fill="#0f172a">{title}</text>"##,
        origin.0,
        origin.1 - 10.0
    )?;
    writeln!(
        out,
        r##"<rect x="{:.2}" y="{:.2}" width="{panel:.2}" height="{panel:.2}" fill="#ffffff" stroke="#0f172a" stroke-width="1"/>"##,
        origin.0, origin.1
    )?;
    for element in elements {
        let x = origin.0 + (element.x_m / (grid.nx as f64 * grid.dx)) * panel;
        let y = origin.1 + panel - (element.y_m / (grid.ny as f64 * grid.dy)) * panel;
        let pixel = value(element)
            .filter(|value| value.is_finite())
            .map(|value| scalar_color(value, color))
            .unwrap_or([148, 163, 184]);
        let fill = rgb_hex(pixel);
        writeln!(
            out,
            r#"<circle cx="{x:.2}" cy="{y:.2}" r="2.2" fill="{fill}"/>"#
        )?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum ScalarColor {
    Phase { min: f64, max: f64 },
    Sequential { min: f64, max: f64 },
}

fn draw_element_scalar_panel<F>(
    rgb: &mut [u8],
    width: usize,
    height: usize,
    panel: (usize, usize),
    elements: &[ElementProjection],
    grid: &Grid,
    value: F,
    color: ScalarColor,
) where
    F: Fn(&ElementProjection) -> Option<f64>,
{
    let x0 = panel.0 * MAP_PANEL;
    let y0 = panel.1 * MAP_PANEL;
    draw_panel_border(rgb, width, height, x0, y0);
    for element in elements {
        let px = x0 as isize
            + ((element.x_m / (grid.nx as f64 * grid.dx)) * MAP_PANEL as f64).round() as isize;
        let py = y0 as isize + (MAP_PANEL as isize - 1)
            - ((element.y_m / (grid.ny as f64 * grid.dy)) * MAP_PANEL as f64).round() as isize;
        let pixel = value(element)
            .filter(|value| value.is_finite())
            .map(|value| scalar_color(value, color))
            .unwrap_or([148, 163, 184]);
        draw_disc(rgb, width, height, px, py, 2, pixel);
    }
}

fn draw_panel_border(rgb: &mut [u8], width: usize, height: usize, x0: usize, y0: usize) {
    let border = [15, 23, 42];
    for x in x0..(x0 + MAP_PANEL).min(width) {
        put_pixel(rgb, width, height, x, y0, border);
        put_pixel(
            rgb,
            width,
            height,
            x,
            (y0 + MAP_PANEL - 1).min(height - 1),
            border,
        );
    }
    for y in y0..(y0 + MAP_PANEL).min(height) {
        put_pixel(rgb, width, height, x0, y, border);
        put_pixel(
            rgb,
            width,
            height,
            (x0 + MAP_PANEL - 1).min(width - 1),
            y,
            border,
        );
    }
}

fn draw_disc(
    rgb: &mut [u8],
    width: usize,
    height: usize,
    cx: isize,
    cy: isize,
    radius: isize,
    color: [u8; 3],
) {
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy > radius * radius {
                continue;
            }
            let x = cx + dx;
            let y = cy + dy;
            if x >= 0 && y >= 0 {
                put_pixel(rgb, width, height, x as usize, y as usize, color);
            }
        }
    }
}

fn draw_cross(
    rgb: &mut [u8],
    width: usize,
    height: usize,
    cx: isize,
    cy: isize,
    radius: isize,
    color: [u8; 3],
) {
    for d in -radius..=radius {
        let x = cx + d;
        if x >= 0 && cy >= 0 {
            put_pixel(rgb, width, height, x as usize, cy as usize, color);
        }
        let y = cy + d;
        if cx >= 0 && y >= 0 {
            put_pixel(rgb, width, height, cx as usize, y as usize, color);
        }
    }
}

fn draw_line_blend(
    rgb: &mut [u8],
    width: usize,
    height: usize,
    x0: isize,
    y0: isize,
    x1: isize,
    y1: isize,
    color: [u8; 3],
    alpha: f64,
) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let (mut x, mut y) = (x0, y0);

    loop {
        if x >= 0 && y >= 0 && (x as usize) < width && (y as usize) < height {
            let idx = 3 * (y as usize * width + x as usize);
            let base = [rgb[idx], rgb[idx + 1], rgb[idx + 2]];
            let blended = blend_rgb(base, color, alpha);
            rgb[idx..idx + 3].copy_from_slice(&blended);
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

fn blend_rgb(base: [u8; 3], overlay: [u8; 3], alpha: f64) -> [u8; 3] {
    let a = alpha.clamp(0.0, 1.0);
    [
        ((1.0 - a) * f64::from(base[0]) + a * f64::from(overlay[0])) as u8,
        ((1.0 - a) * f64::from(base[1]) + a * f64::from(overlay[1])) as u8,
        ((1.0 - a) * f64::from(base[2]) + a * f64::from(overlay[2])) as u8,
    ]
}

fn scalar_color(value: f64, color: ScalarColor) -> [u8; 3] {
    match color {
        ScalarColor::Phase { min, max } => {
            let normalized = ((value - min) / (max - min).max(f64::EPSILON)).clamp(0.0, 1.0);
            let phase = min + normalized * (max - min);
            phase_color(phase)
        }
        ScalarColor::Sequential { min, max } => {
            let t = ((value - min) / (max - min).max(f64::EPSILON)).clamp(0.0, 1.0);
            sequential_color(t)
        }
    }
}

fn sequential_color(t: f64) -> [u8; 3] {
    let r = (255.0 * t) as u8;
    let g = (255.0 * (1.0 - (2.0 * t - 1.0).abs())) as u8;
    let b = (255.0 * (1.0 - t)) as u8;
    [r, g, b]
}

fn finite_max<I>(values: I) -> f64
where
    I: Iterator<Item = f64>,
{
    values
        .filter(|value| value.is_finite())
        .fold(0.0_f64, f64::max)
}

fn rgb_hex(rgb: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", rgb[0], rgb[1], rgb[2])
}

fn put_pixel(rgb: &mut [u8], width: usize, height: usize, x: usize, y: usize, color: [u8; 3]) {
    if x >= width || y >= height {
        return;
    }
    let idx = 3 * (y * width + x);
    rgb[idx..idx + 3].copy_from_slice(&color);
}

fn ct_window(hu: f64) -> u8 {
    let normalized = ((hu + 1000.0) / 3000.0).clamp(0.0, 1.0);
    (255.0 * normalized) as u8
}

fn phase_color(phase: f64) -> [u8; 3] {
    let wrapped = wrap_phase(phase);
    let t = (wrapped + PI) / TAU;
    let r = (255.0 * (1.0 - (2.0 * (t - 0.0).abs()).min(1.0))) as u8;
    let g = (255.0 * (1.0 - (2.0 * (t - 0.5).abs()).min(1.0))) as u8;
    let b = (255.0 * (1.0 - (2.0 * (t - 1.0).abs()).min(1.0))) as u8;
    [r, g, b]
}

fn blend_gray_color(gray: u8, color: [u8; 3], alpha: f64) -> [u8; 3] {
    let base = f64::from(gray);
    [
        ((1.0 - alpha) * base + alpha * f64::from(color[0])) as u8,
        ((1.0 - alpha) * base + alpha * f64::from(color[1])) as u8,
        ((1.0 - alpha) * base + alpha * f64::from(color[2])) as u8,
    ]
}

fn wrap_phase(phase: f64) -> f64 {
    (phase + PI).rem_euclid(TAU) - PI
}

fn write_element_csv(path: &Path, elements: &[ElementProjection]) -> Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(
        out,
        "element,x_m,y_m,bowl_z_m,phase_correction_rad,skull_thickness_mm,cortical_thickness_mm,trabecular_thickness_mm,mean_density_kg_m3,mean_sound_speed_m_s,cortical_mean_density_kg_m3,trabecular_mean_density_kg_m3,trabecular_to_cortical_sdr,cortical_to_trabecular_density_ratio"
    )?;
    for element in elements {
        writeln!(
            out,
            "{},{:.9},{:.9},{:.9},{:.12},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{}",
            element.element,
            element.x_m,
            element.y_m,
            element.bowl_z_m,
            element.correction_rad,
            1e3 * element.skull_thickness_m,
            1e3 * element.cortical_thickness_m,
            1e3 * element.trabecular_thickness_m,
            element.mean_density_kg_m3,
            element.mean_sound_speed_m_s,
            format_optional(element.cortical_mean_density_kg_m3),
            format_optional(element.trabecular_mean_density_kg_m3),
            format_optional(element.trabecular_to_cortical_sdr),
            format_optional(element.cortical_to_trabecular_density_ratio)
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_grid() -> Grid {
        Grid::new(11, 11, 11, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid test grid")
    }

    fn test_ct_with_oblique_bone() -> (CtVolume, HeterogeneousSkull) {
        let mut hu = Array3::<f64>::zeros((11, 11, 11));
        hu[[7, 5, 8]] = 800.0;
        let skull = HeterogeneousSkull::from_ct_hill(
            &hu,
            C_CORTICAL_BONE_M_PER_S,
            RHO_CORTICAL_BONE_KG_PER_M3,
            20.0,
        )
        .expect("finite test CT");
        (
            CtVolume {
                hu,
                spacing_m: [1.0e-3; 3],
                origin_mm: [0.0; 3],
                direction: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            },
            skull,
        )
    }

    #[test]
    fn focused_ray_metrics_detect_oblique_peripheral_bone() {
        let grid = small_grid();
        let (ct, skull) = test_ct_with_oblique_bone();
        let focus = Point3Meters {
            x: 5.0e-3,
            y: 5.0e-3,
            z: 5.0e-3,
        };

        let metrics = element_path_metrics(10.0e-3, 5.0e-3, 2.0e-3, focus, &grid, &ct, &skull);

        assert!(
            metrics.skull_thickness_m > 0.0,
            "oblique element-to-focus ray must count intersected skull"
        );
        assert!(
            metrics.mean_sound_speed_m_s > C_WATER_M_PER_S,
            "bone intersection must increase mean sound speed"
        );
    }

    #[test]
    fn focused_corrections_are_computed_for_every_element() {
        let grid = small_grid();
        let (_ct, skull) = test_ct_with_oblique_bone();
        let focus = Point3Meters {
            x: 5.0e-3,
            y: 5.0e-3,
            z: 5.0e-3,
        };
        let xs = [10.0e-3, 5.0e-3];
        let ys = [5.0e-3, 10.0e-3];
        let zs = [2.0e-3, 2.0e-3];

        let corrections =
            compute_focused_element_corrections(FREQUENCY_HZ, &xs, &ys, &zs, focus, &grid, &skull);

        assert_eq!(corrections.len(), 2);
        assert!(corrections.iter().all(|value| value.is_finite()));
        assert!(
            corrections[0].abs() > corrections[1].abs(),
            "ray through synthetic bone must have larger aberration than water-only ray"
        );
    }

    #[test]
    fn pressure_slice_constructs_peak_at_focus_for_symmetric_sources() {
        let grid = small_grid();
        let focus = Point3Meters {
            x: 5.0e-3,
            y: 5.0e-3,
            z: 5.0e-3,
        };
        let elements = vec![
            ElementProjection {
                element: 0,
                x_m: 5.0e-3,
                y_m: 2.0e-3,
                bowl_z_m: 2.0e-3,
                correction_rad: 0.0,
                skull_thickness_m: 0.0,
                cortical_thickness_m: 0.0,
                trabecular_thickness_m: 0.0,
                mean_density_kg_m3: RHO_WATER_KG_PER_M3,
                mean_sound_speed_m_s: C_WATER_M_PER_S,
                cortical_mean_density_kg_m3: None,
                trabecular_mean_density_kg_m3: None,
                trabecular_to_cortical_sdr: None,
                cortical_to_trabecular_density_ratio: None,
            },
            ElementProjection {
                element: 1,
                x_m: 5.0e-3,
                y_m: 8.0e-3,
                bowl_z_m: 2.0e-3,
                correction_rad: 0.0,
                skull_thickness_m: 0.0,
                cortical_thickness_m: 0.0,
                trabecular_thickness_m: 0.0,
                mean_density_kg_m3: RHO_WATER_KG_PER_M3,
                mean_sound_speed_m_s: C_WATER_M_PER_S,
                cortical_mean_density_kg_m3: None,
                trabecular_mean_density_kg_m3: None,
                trabecular_to_cortical_sdr: None,
                cortical_to_trabecular_density_ratio: None,
            },
        ];

        let pressure = compute_pressure_slice(&elements, focus, &grid, 500_000.0);
        let focus_pressure = pressure[[5, 5]];

        assert!(
            focus_pressure > 0.999,
            "focus-referenced phases must construct coherent gain at the target"
        );
    }
}

fn format_optional(value: Option<f64>) -> String {
    value
        .filter(|value| value.is_finite())
        .map(|value| format!("{value:.6}"))
        .unwrap_or_default()
}
