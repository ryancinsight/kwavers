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
//! - `<output>.csv`: per-element projected coordinates and correction phase.

use std::env;
use std::f64::consts::{PI, TAU};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use burn::backend::NdArray;
use kwavers::domain::grid::Grid;
use kwavers::physics::acoustics::skull::{AberrationCorrection, HeterogeneousSkull};
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
const HU_BONE_UPPER: f64 = 2000.0;
const PANEL: usize = 384;

#[derive(Debug, Clone)]
struct CtVolume {
    hu: Array3<f64>,
    spacing_m: [f64; 3],
    origin_mm: [f64; 3],
    direction: [[f64; 3]; 3],
}

#[derive(Debug, Clone)]
pub(crate) struct ElementProjection {
    element: usize,
    pub(crate) x_m: f64,
    pub(crate) y_m: f64,
    pub(crate) bowl_z_m: f64,
    pub(crate) correction_rad: f64,
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
    let element_corrections = correction
        .compute_element_corrections(FREQUENCY_HZ, &element_x, &element_y)
        .context("failed to sample element phase corrections")?;
    let projections =
        element_projection_records(&element_x, &element_y, &element_z, &element_corrections);

    write_three_plane_ppm(
        &output_path,
        &ct.hu,
        &correction_volume,
        &aperture_phase,
        &projections,
        &grid,
    )?;
    write_element_csv(&output_path.with_extension("csv"), &projections)?;
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
        "Computed {} element corrections at {:.0} kHz; wrote {} and {}",
        projections.len(),
        FREQUENCY_HZ / 1e3,
        output_path.display(),
        output_path.with_extension("csv").display()
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
    let dims = ct.hu.dim();
    let mut sound_speed = Array3::<f64>::from_elem(dims, C_WATER_M_PER_S);
    let mut density = Array3::<f64>::from_elem(dims, RHO_WATER_KG_PER_M3);
    let mut attenuation = Array3::<f64>::zeros(dims);

    for ((x, y, z), hu) in ct.hu.indexed_iter() {
        let bone_fraction =
            ((*hu - HU_BONE_LOWER) / (HU_BONE_UPPER - HU_BONE_LOWER)).clamp(0.0, 1.0);
        sound_speed[[x, y, z]] =
            C_WATER_M_PER_S + bone_fraction * (C_CORTICAL_BONE_M_PER_S - C_WATER_M_PER_S);
        density[[x, y, z]] = RHO_WATER_KG_PER_M3
            + bone_fraction * (RHO_CORTICAL_BONE_KG_PER_M3 - RHO_WATER_KG_PER_M3);
        attenuation[[x, y, z]] = 20.0 * bone_fraction;
    }

    HeterogeneousSkull {
        sound_speed,
        density,
        attenuation,
    }
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

fn element_projection_records(
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
    corrections: &Array1<f64>,
) -> Vec<ElementProjection> {
    xs.iter()
        .zip(ys)
        .zip(zs)
        .zip(corrections)
        .enumerate()
        .map(
            |(element, (((&x_m, &y_m), &bowl_z_m), &correction_rad))| ElementProjection {
                element,
                x_m,
                y_m,
                bowl_z_m,
                correction_rad: wrap_phase(correction_rad),
            },
        )
        .collect()
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
    writeln!(out, "element,x_m,y_m,bowl_z_m,phase_correction_rad")?;
    for element in elements {
        writeln!(
            out,
            "{},{:.9},{:.9},{:.9},{:.12}",
            element.element, element.x_m, element.y_m, element.bowl_z_m, element.correction_rad
        )?;
    }
    Ok(())
}
