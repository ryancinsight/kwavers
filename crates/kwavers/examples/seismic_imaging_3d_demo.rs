// seismic_imaging_3d_demo.rs — True 3D transcranial ultrasound FWI demo.
//
// Extends seismic_imaging_demo.rs (2D quasi-3D: NX=64, NY=2, NZ=64) to full 3D
// (NX=64, NY=48, NZ=64) with a Fibonacci-sphere acquisition geometry, trilinear
// CT resampling, and explicit co-registered T1 MRI input selection.
//
// Compile: cargo check --release --example seismic_imaging_3d_demo
//
// # References
//
// - Aubry 2003: JASA 113(1) — skull bone-volume-fraction acoustic model.
// - Marsac 2017: J. Ther. Ultrasound — transcranial FWI protocol.
// - Guasch 2020: npj Digital Medicine — 3D brain FWI pipeline.
// - Duck 1990: "Physical Properties of Tissue" — soft-tissue velocities.
// - Treeby & Cox 2010: JASA — fractional-Laplacian absorption.
// - MNI ICBM 2009c: https://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/

mod seismic_imaging;
#[path = "seismic_imaging/metrics.rs"]
mod seismic_metrics;
#[path = "seismic_imaging/volume_artifacts.rs"]
mod seismic_volume_artifacts;
#[path = "seismic_imaging/volume_phantom.rs"]
mod seismic_volume_phantom;
use seismic_metrics::{print_quality_pairs, print_quality_report};

use aequitas::systems::si::quantities::{Frequency, Pressure, Time};
use anyhow::Context as _;
use coeus_core::MoiraiBackend;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use kwavers_signal::DomainRickerWavelet;
use kwavers_solver::inverse::fwi::time_domain::{FwiGeometry, FwiProcessor};
use kwavers_solver::inverse::seismic::parameters::{FwiParameters, RegularizationParameters};
use kwavers_source::{GridSource, SourceMode};
use leto::{Array2, Array3};
use moirai_parallel::{Adaptive, map_collect_index_with};
use ritk_io::ImageReader;
use ritk_io::format::nifti::native::NiftiReader as NativeNiftiReader;
use seismic_imaging::medium::SkullModel;
use seismic_imaging::render::{put_pixel, velocity_color, write_png};
use std::f64::consts::PI;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[path = "support/brain_prior.rs"]
mod brain_prior;
#[path = "support/seismic_input.rs"]
mod seismic_input;
use brain_prior::BrainPriorMode;
use seismic_input::SeismicInputMode;

// ─────────────────────────────────────────────────────────────────────────────
// Grid constants — TRUE 3D (NY = 48, not 2)
// ─────────────────────────────────────────────────────────────────────────────

/// Grid spacing `m`.  3 mm gives λ/3.3 resolution at 150 kHz in water.
///
/// Reference: Marsac 2017 — mean skull thickness ≈ 7 mm → at least 2 voxels
/// through bone at 3 mm spacing.
const DX: f64 = 3.0e-3;

/// Grid dimensions — full 3-D volume.
const NX: usize = 64; // lateral     192 mm
const NY: usize = 48; // elevation   144 mm  (true 3-D, was 2)
const NZ: usize = 64; // depth       192 mm

// ─────────────────────────────────────────────────────────────────────────────
// Skull phantom geometry — radii in voxels from grid centre
// ─────────────────────────────────────────────────────────────────────────────
//
// Same geometry as seismic_imaging_demo.rs — the 3D phantom is a sphere.
// CPML thickness = 10 cells; physical domain: ix ∈ [10,53], iy ∈ [4,43], iz ∈ [10,53].

const R_HEAD: f64 = 18.0; // 54 mm — outer scalp surface
const R_SKULL_OUT: f64 = 16.0; // 48 mm — outer cortical / scalp boundary
const R_DIPLOE: f64 = 14.0; // 42 mm — outer diploe boundary
const R_SKULL_IN: f64 = 12.0; // 36 mm — inner cortical / brain boundary
const R_BRAIN: f64 = 11.0; // 33 mm — brain surface (CSF buffer ≈ 3 mm)

// ─────────────────────────────────────────────────────────────────────────────
// Hounsfield-unit phantom labels
// ─────────────────────────────────────────────────────────────────────────────

const HU_WATER: f64 = 0.0;
const HU_SCALP: f64 = 40.0;
const HU_CORTICAL_OUT: f64 = 720.0;
const HU_DIPLOE: f64 = 380.0;
const HU_CORTICAL_IN: f64 = 660.0;
const HU_BRAIN: f64 = 35.0;

// ─────────────────────────────────────────────────────────────────────────────
// Stage-2 brain tissue FWI constants (Duck 1990; Guasch 2020)
// ─────────────────────────────────────────────────────────────────────────────

const C_GRAY: f64 = 1541.0; // gray matter [m/s]
const C_WHITE: f64 = 1520.0; // white matter [m/s]
const C_CSF: f64 = 1505.0; // cerebrospinal fluid [m/s]

const BRAIN_C_MIN: f64 = 1480.0; // m/s
const BRAIN_C_MAX: f64 = 1560.0; // m/s
const BONE_VELOCITY_THRESHOLD: f64 = 1714.0; // m/s

const F0_HZ: f64 = 150_000.0; // Hz — default Ricker centre frequency
const P0_PA: f64 = 1.0e5; // Pa — peak source pressure
const STEP_SIZE: f64 = 50.0; // m/s per normalised gradient step

const MNI_INNER_SKULL_RADIUS_MM: f64 = 82.0; // mm

// ─────────────────────────────────────────────────────────────────────────────
// 3D acquisition geometry constants
// ─────────────────────────────────────────────────────────────────────────────

const N_SPHERE_ELEMENTS: usize = 24; // elements on Fibonacci sphere
const N_SHOTS_3D: usize = 12; // every other element transmits
const N_RECEIVERS_3D: usize = N_SPHERE_ELEMENTS - 1;
const R_ARRAY_3D: f64 = 21.0; // voxels from grid centre

// ─────────────────────────────────────────────────────────────────────────────
// Transcranial focused-bowl reference constant (same as 2D demo)
// ─────────────────────────────────────────────────────────────────────────────

const TRANSCRANIAL_FOCUSED_BOWL_ELEMENT_COUNT: usize = 1024;

// ─────────────────────────────────────────────────────────────────────────────
// Visualisation layout constants
// ─────────────────────────────────────────────────────────────────────────────

const PANEL: usize = 320; // pixels per square panel
const COLORBAR_H: usize = 20; // colorbar height below each panel

// ─────────────────────────────────────────────────────────────────────────────
// Dataset paths (compile-time constants, derived from CARGO_MANIFEST_DIR)
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// Structs
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
// MNI brain velocity model (3D)
// ─────────────────────────────────────────────────────────────────────────────

/// Load MNI ICBM 2009c tissue probability maps and build a 3D brain velocity
/// model by sampling the full (x, y, z) MNI coordinate for each FWI voxel.
///
/// # Tissue velocity mapping (Duck 1990)
///
/// ```text
/// c(x) = p_gm(x) × C_GRAY + p_wm(x) × C_WHITE + p_csf(x) × C_CSF
///       + (1 − p_gm − p_wm − p_csf) × c_water
/// ```
///
/// # Spatial mapping
///
/// Each FWI voxel offset (dx_fwi, dy_fwi, dz_fwi) from grid centre is scaled to MNI
/// voxel offsets:
/// ```text
/// fwi_to_mni = MNI_INNER_SKULL_RADIUS_MM / (R_SKULL_IN × DX × 1e3)
/// mni_coord  = mni_centre + fwi_offset_mm × fwi_to_mni
/// ```
fn build_brain_velocity_3d(
    skull_phantom: &SkullModel,
    mni_dir: &Path,
) -> anyhow::Result<Array3<f64>> {
    let backend = MoiraiBackend;

    let load = |name: &str| -> anyhow::Result<Array3<f64>> {
        let path = mni_dir.join(name);
        anyhow::ensure!(
            path.exists(),
            "MNI tissue map not found: '{}' — download from {}",
            path.display(),
            "https://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_sym_09c_nifti.zip"
        );
        let img = ImageReader::read(&NativeNiftiReader::new(backend), &path)
            .with_context(|| format!("NIfTI load failed: '{}'", path.display()))?;
        let [depth, rows, cols] = img.shape();
        let vals = img
            .data_slice()
            .map_err(|e| anyhow::anyhow!("NIfTI data not f32: {e:?}"))?;
        let mut vol = Array3::<f64>::zeros((cols, rows, depth));
        for z in 0..depth {
            for y in 0..rows {
                for x in 0..cols {
                    vol[[x, y, z]] =
                        f64::from(vals[z * rows * cols + y * cols + x]).clamp(0.0, 1.0);
                }
            }
        }
        Ok(vol)
    };

    let gm = load("mni_icbm152_gm_tal_nlin_sym_09c.nii")?;
    let wm = load("mni_icbm152_wm_tal_nlin_sym_09c.nii")?;
    let csf = load("mni_icbm152_csf_tal_nlin_sym_09c.nii")?;

    let [mni_nx, mni_ny, mni_nz] = gm.shape();
    let cx_mni = mni_nx / 2;
    let cy_mni = mni_ny / 2;
    let cz_mni = mni_nz / 2;

    let fwi_inner_mm = R_SKULL_IN * DX * 1e3;
    let fwi_to_mni = MNI_INNER_SKULL_RADIUS_MM / fwi_inner_mm;

    let mut brain_model = skull_phantom.acoustic().sound_speed.clone();

    for iz in 0..NZ {
        for iy in 0..NY {
            for ix in 0..NX {
                let dx_fwi = ix as f64 - (NX / 2) as f64;
                let dy_fwi = iy as f64 - (NY / 2) as f64;
                let dz_fwi = iz as f64 - (NZ / 2) as f64;
                let r3 = (dx_fwi * dx_fwi + dy_fwi * dy_fwi + dz_fwi * dz_fwi).sqrt();
                if r3 >= R_SKULL_IN {
                    continue;
                }

                let mni_x = (cx_mni as f64 + dx_fwi * DX * 1e3 * fwi_to_mni).round() as isize;
                let mni_y = (cy_mni as f64 + dy_fwi * DX * 1e3 * fwi_to_mni).round() as isize;
                let mni_z = (cz_mni as f64 + dz_fwi * DX * 1e3 * fwi_to_mni).round() as isize;

                if mni_x < 0
                    || mni_x >= mni_nx as isize
                    || mni_y < 0
                    || mni_y >= mni_ny as isize
                    || mni_z < 0
                    || mni_z >= mni_nz as isize
                {
                    continue;
                }
                let mx = mni_x as usize;
                let my = mni_y as usize;
                let mz = mni_z as usize;

                let p_gm = gm[[mx, my, mz]];
                let p_wm = wm[[mx, my, mz]];
                let p_csf = csf[[mx, my, mz]];
                let p_rest = (1.0 - p_gm - p_wm - p_csf).clamp(0.0, 1.0);
                let c_tissue =
                    p_gm * C_GRAY + p_wm * C_WHITE + p_csf * C_CSF + p_rest * SOUND_SPEED_WATER_SIM;

                brain_model[[ix, iy, iz]] = c_tissue;
            }
        }
    }

    Ok(brain_model)
}

// ─────────────────────────────────────────────────────────────────────────────
// T1 MRI loading and tissue velocity mapping
// ─────────────────────────────────────────────────────────────────────────────

/// Load a T1 MRI NIfTI file.
///
/// Transposes [Z,Y,X] tensor to [X,Y,Z] and normalises by the 99th percentile
/// of non-zero voxels.  Returns (vol_normalized, spacing_mm).
fn load_t1_mri(path: &Path) -> anyhow::Result<(Array3<f64>, [f64; 3])> {
    let backend = MoiraiBackend;

    println!("  T1 NIfTI file   : {}", path.display());
    let img = ImageReader::read(&NativeNiftiReader::new(backend), path)
        .with_context(|| format!("T1 NIfTI read failed for '{}'", path.display()))?;

    let [depth, rows, cols] = img.shape();
    let spacing = img.spacing().into_vector().to_array();
    let values = img
        .data_slice()
        .map_err(|e| anyhow::anyhow!("T1 tensor data is not f32: {e:?}"))?;
    anyhow::ensure!(
        values.len() == depth * rows * cols,
        "T1 data length mismatch: got {}, expected {}",
        values.len(),
        depth * rows * cols
    );

    // Transpose [Z,Y,X] → [X,Y,Z].
    let mut vol = Array3::<f64>::zeros((cols, rows, depth));
    for z in 0..depth {
        for y in 0..rows {
            for x in 0..cols {
                vol[[x, y, z]] = f64::from(values[z * rows * cols + y * cols + x]).max(0.0);
            }
        }
    }

    // Compute 99th percentile of non-zero voxels for normalisation.
    let mut nonzero: Vec<f64> = vol.iter().copied().filter(|&v| v > 0.0).collect();
    let p99 = if nonzero.is_empty() {
        1.0
    } else {
        nonzero.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((nonzero.len() as f64) * 0.99).floor() as usize;
        nonzero[idx.min(nonzero.len() - 1)].max(1.0)
    };

    for v in vol.iter_mut() {
        *v = (*v / p99).clamp(0.0, 1.0);
    }

    Ok((vol, [spacing[0], spacing[1], spacing[2]]))
}

/// Duck (1990) tissue velocity from T1 normalised intensity.
///
/// | Intensity range | Tissue | c [m/s] |
/// |-----------------|--------|---------|
/// | > 0.70          | WM     | 1520    |
/// | 0.35–0.70       | GM     | 1541    |
/// | 0.05–0.35       | CSF    | 1505    |
/// | ≤ 0.05          | water  | 1500    |
#[inline]
fn t1_to_velocity(t1_norm: f64) -> f64 {
    if t1_norm > 0.70 {
        C_WHITE
    } else if t1_norm > 0.35 {
        C_GRAY
    } else if t1_norm > 0.05 {
        C_CSF
    } else {
        SOUND_SPEED_WATER_SIM
    }
}

/// Map T1 MRI to 3D brain tissue velocity model.
///
/// Voxels geometrically outside r_3d ≥ R_SKULL_IN retain their CT-derived
/// skull velocities.  Inside the skull the T1 intensity is sampled via the
/// same scale factor used for CT resampling (atlas-space approximation).
fn build_brain_velocity_from_t1(
    skull_phantom: &SkullModel,
    t1: &Array3<f64>,
    t1_spacing: [f64; 3],
) -> Array3<f64> {
    let [t1_nx, t1_ny, t1_nz] = t1.shape();
    let cx_t1 = t1_nx as f64 / 2.0;
    let cy_t1 = t1_ny as f64 / 2.0;
    let cz_t1 = t1_nz as f64 / 2.0;

    // Same skull-fitting scale as CT: project FWI offset to T1 voxel offset.
    // Assumes T1 is roughly co-registered with the CT (atlas-space approximation).
    let fwi_inner_mm = R_SKULL_IN * DX * 1e3;
    let t1_inner_skull_mm = MNI_INNER_SKULL_RADIUS_MM; // 82 mm
    let fwi_to_t1 = t1_inner_skull_mm / (fwi_inner_mm * t1_spacing[0]);

    let mut model = skull_phantom.acoustic().sound_speed.clone();

    for ix in 0..NX {
        for iy in 0..NY {
            for iz in 0..NZ {
                let dx_fwi = ix as f64 - (NX / 2) as f64;
                let dy_fwi = iy as f64 - (NY / 2) as f64;
                let dz_fwi = iz as f64 - (NZ / 2) as f64;
                let r_3d = (dx_fwi * dx_fwi + dy_fwi * dy_fwi + dz_fwi * dz_fwi).sqrt();
                if r_3d >= R_SKULL_IN {
                    continue;
                }

                let tx = (cx_t1 + dx_fwi * fwi_to_t1).clamp(0.0, t1_nx as f64 - 1.001);
                let ty = (cy_t1 + dy_fwi * fwi_to_t1).clamp(0.0, t1_ny as f64 - 1.001);
                let tz = (cz_t1 + dz_fwi * fwi_to_t1).clamp(0.0, t1_nz as f64 - 1.001);

                let t1_val = seismic_volume_phantom::trilinear_hu(t1, tx, ty, tz);
                model[[ix, iy, iz]] = t1_to_velocity(t1_val);
            }
        }
    }
    model
}

/// Build a deterministic homogeneous brain prior without external datasets.
fn build_uniform_brain_velocity_3d(skull_phantom: &SkullModel) -> Array3<f64> {
    let mut model = skull_phantom.acoustic().sound_speed.clone();
    let cx = NX as f64 / 2.0;
    let cy = NY as f64 / 2.0;
    let cz = NZ as f64 / 2.0;
    for ix in 0..NX {
        for iy in 0..NY {
            for iz in 0..NZ {
                let dx = ix as f64 - cx;
                let dy = iy as f64 - cy;
                let dz = iz as f64 - cz;
                if (dx * dx + dy * dy + dz * dz).sqrt() < R_SKULL_IN {
                    model[[ix, iy, iz]] = SOUND_SPEED_WATER_SIM;
                }
            }
        }
    }
    model
}

fn build_brain_prior_3d(
    skull_phantom: &SkullModel,
    prior: &BrainPriorMode,
) -> anyhow::Result<Array3<f64>> {
    match prior {
        BrainPriorMode::Uniform => Ok(build_uniform_brain_velocity_3d(skull_phantom)),
        BrainPriorMode::Mni(path) | BrainPriorMode::MniT1 { mni: path, .. } => {
            build_brain_velocity_3d(skull_phantom, path)
        }
        BrainPriorMode::T1(path) => {
            let (t1, spacing) = load_t1_mri(path).with_context(|| {
                format!("explicit T1 prior could not be loaded: {}", path.display())
            })?;
            Ok(build_brain_velocity_from_t1(skull_phantom, &t1, spacing))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fibonacci-sphere acquisition geometry
// ─────────────────────────────────────────────────────────────────────────────

/// Generate N positions on a sphere of radius r using the Fibonacci lattice.
///
/// # Algorithm
///
/// For i ∈ [0, n):
/// ```text
/// y_norm = (2(i + 0.5)/n) - 1              ∈ (−1, 1)
/// r_xz   = √(1 − y_norm²)
/// φ      = 2π × i / φ_gold                 (φ_gold = (1 + √5)/2)
/// x_off  = r × r_xz × cos(φ)
/// y_off  = r × y_norm
/// z_off  = r × r_xz × sin(φ)
/// ```
///
/// Positions are clamped to [6, NX/NY/NZ − 7] to stay within the physical domain.
///
/// Reference: González 2010 — Fibonacci lattice for uniform sphere sampling.
fn fibonacci_sphere_elements(n: usize, r: f64, cx: f64, cy: f64, cz: f64) -> Vec<[usize; 3]> {
    let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
    (0..n)
        .map(|i| {
            let y_norm = (2.0 * (i as f64 + 0.5) / n as f64) - 1.0;
            let r_xz = (1.0 - y_norm * y_norm).sqrt();
            let phi = 2.0 * PI * i as f64 / golden_ratio;
            let x_off = r * r_xz * phi.cos();
            let y_off = r * y_norm;
            let z_off = r * r_xz * phi.sin();
            let ix = ((cx + x_off).round() as isize).clamp(6, NX as isize - 7) as usize;
            let iy = ((cy + y_off).round() as isize).clamp(6, NY as isize - 7) as usize;
            let iz = ((cz + z_off).round() as isize).clamp(6, NZ as isize - 7) as usize;
            [ix, iy, iz]
        })
        .collect()
}

/// Build a receiver mask for all Fibonacci-sphere elements except the source.
fn build_receiver_mask_3d(all_elements: &[[usize; 3]], source_idx: usize) -> Array3<bool> {
    let mut mask = Array3::<bool>::from_elem((NX, NY, NZ), false);
    for (idx, &[ix, iy, iz]) in all_elements.iter().enumerate() {
        if idx != source_idx {
            mask[[ix, iy, iz]] = true;
        }
    }
    mask
}

/// Build FwiGeometry for one Fibonacci-sphere shot.
///
/// Source at `source_pos`; receivers at all other Fibonacci-sphere elements.
fn build_shot_3d(
    source_pos: [usize; 3],
    all_elements: &[[usize; 3]],
    shot_idx: usize,
    f0_hz: f64,
    nt: usize,
    dt: f64,
) -> KwaversResult<FwiGeometry> {
    let [ix, iy, iz] = source_pos;
    let mut source_mask = Array3::<f64>::zeros((NX, NY, NZ));
    source_mask[[ix, iy, iz]] = 1.0;

    let wavelet =
        DomainRickerWavelet::causal(Frequency::from_base(f0_hz), Pressure::from_base(P0_PA))?;
    let mut p_signal = Array2::<f64>::zeros((1, nt));
    for (t, pressure) in wavelet.samples(Time::from_base(dt), nt)?.enumerate() {
        p_signal[[0, t]] = pressure;
    }

    let mut source = GridSource::new_empty();
    source.p_mask = Some(source_mask);
    source.p_signal = Some(p_signal);
    source.p_mode = SourceMode::Dirichlet;

    Ok(FwiGeometry::new(
        source,
        build_receiver_mask_3d(all_elements, shot_idx),
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// Gaussian blur (3D separable)
// ─────────────────────────────────────────────────────────────────────────────

/// Separable 3D Gaussian blur applied sequentially in x → y → z.
///
/// Each pass maps one flat output index through Moirai with clamped boundary
/// handling.
///
/// Boundary voxels use reflect-padding (clamp-at-edge).
///
/// Reference: Guasch (2020) npj Digital Medicine — §Methods, CT initial model.
fn gaussian_blur_3d(model: &Array3<f64>, sigma: f64) -> Array3<f64> {
    let radius = (3.0 * sigma).ceil() as usize;
    let kernel_size = 2 * radius + 1;

    // 1-D normalised Gaussian kernel.
    let raw: Vec<f64> = (0..kernel_size)
        .map(|i| {
            let x = i as f64 - radius as f64;
            (-x * x / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let ksum: f64 = raw.iter().sum();
    let kernel: Vec<f64> = raw.iter().map(|&k| k / ksum).collect();

    let cell_count = NX * NY * NZ;

    // Pass 1: convolve along x → tmp_x. Parallel over all output voxels.
    let tmp_x_values = map_collect_index_with::<Adaptive, _, _>(cell_count, |idx| {
        let ix = idx / (NY * NZ);
        let rem = idx % (NY * NZ);
        let iy = rem / NZ;
        let iz = rem % NZ;
        let mut acc = 0.0_f64;
        for (ki, &kw) in kernel.iter().enumerate() {
            let si =
                (ix as isize + ki as isize - radius as isize).clamp(0, NX as isize - 1) as usize;
            acc += kw * model[[si, iy, iz]];
        }
        acc
    });
    let tmp_x = Array3::<f64>::from_shape_vec((NX, NY, NZ), tmp_x_values)
        .expect("invariant: flat Moirai x-pass preserves model shape length");

    // Pass 2: convolve along y → tmp_y. Parallel over all output voxels.
    let tmp_y_values = map_collect_index_with::<Adaptive, _, _>(cell_count, |idx| {
        let ix = idx / (NY * NZ);
        let rem = idx % (NY * NZ);
        let iy = rem / NZ;
        let iz = rem % NZ;
        let mut acc = 0.0_f64;
        for (ki, &kw) in kernel.iter().enumerate() {
            let sj =
                (iy as isize + ki as isize - radius as isize).clamp(0, NY as isize - 1) as usize;
            acc += kw * tmp_x[[ix, sj, iz]];
        }
        acc
    });
    let tmp_y = Array3::<f64>::from_shape_vec((NX, NY, NZ), tmp_y_values)
        .expect("invariant: flat Moirai y-pass preserves model shape length");

    // Pass 3: convolve along z.
    let out_values = map_collect_index_with::<Adaptive, _, _>(cell_count, |idx| {
        let ix = idx / (NY * NZ);
        let rem = idx % (NY * NZ);
        let iy = rem / NZ;
        let iz = rem % NZ;
        let mut acc = 0.0_f64;
        for (ki, &kw) in kernel.iter().enumerate() {
            let sk =
                (iz as isize + ki as isize - radius as isize).clamp(0, NZ as isize - 1) as usize;
            acc += kw * tmp_y[[ix, iy, sk]];
        }
        acc
    });

    Array3::<f64>::from_shape_vec((NX, NY, NZ), out_values)
        .expect("invariant: flat Moirai z-pass preserves model shape length")
}

// ─────────────────────────────────────────────────────────────────────────────
// Reconstruction quality metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Print RMSE, Pearson r, max |error|, ±10 m/s fraction for brain voxels only.
///
/// Brain voxels defined geometrically: r_3d < R_SKULL_IN from grid centre.
fn print_quality_report_brain(true_model: &Array3<f64>, reconstructed: &Array3<f64>) {
    let cx = (NX / 2) as f64;
    let cy = (NY / 2) as f64;
    let cz = (NZ / 2) as f64;
    let free_pairs: Vec<(f64, f64)> = true_model
        .indexed_iter()
        .filter(|([ix, iy, iz], _)| {
            let dx = *ix as f64 - cx;
            let dy = *iy as f64 - cy;
            let dz = *iz as f64 - cz;
            (dx * dx + dy * dy + dz * dz).sqrt() < R_SKULL_IN
        })
        .map(|([ix, iy, iz], &t)| (t, reconstructed[[ix, iy, iz]]))
        .collect();
    print_quality_pairs(&free_pairs);
}

// ─────────────────────────────────────────────────────────────────────────────
// Image output
// ─────────────────────────────────────────────────────────────────────────────

/// Blue ← white → red diverging colormap.
#[allow(dead_code)]
fn diverging_color(value: f64, max_abs: f64) -> [u8; 3] {
    if max_abs < f64::EPSILON {
        return [200, 200, 200];
    }
    let t = (value / max_abs).clamp(-1.0, 1.0);
    if t >= 0.0 {
        let gb = (255.0 * (1.0 - t)) as u8;
        [255, gb, gb]
    } else {
        let rg = (255.0 * (1.0 + t)) as u8;
        [rg, rg, 255]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> KwaversResult<()> {
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("warn"));

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  3D Transcranial Ultrasound FWI — Brain Reconstruction   ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // ── [ 1 / 7 ]  3D skull phantom ──────────────────────────────────────
    let input_mode = SeismicInputMode::from_env("KWAVERS_SEISMIC_INPUT_MODE")
        .map_err(KwaversError::InvalidInput)?;
    println!("[ 1 / 7 ]  Building 3D skull phantom ({input_mode:?}) …");
    let (phantom, _ct_vol) = seismic_volume_phantom::build_phantom_3d(&input_mode)
        .map_err(|error| KwaversError::InvalidInput(error.to_string()))?;

    let _c_min = phantom
        .acoustic()
        .sound_speed
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let _c_max = phantom
        .acoustic()
        .sound_speed
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let hu_min = phantom.hu().iter().copied().fold(f64::INFINITY, f64::min);
    let hu_max = phantom
        .hu()
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    println!("  HU range        : [{hu_min:.0}, {hu_max:.0}]");

    // ── [ 2 / 7 ]  T1 MRI ────────────────────────────────────────────────
    println!("\n[ 2 / 7 ]  Loading explicit T1 MRI input …");

    let t1_result = match &input_mode {
        SeismicInputMode::CtMri { mri, .. } => {
            let result = load_t1_mri(mri).with_context(|| {
                format!(
                    "explicit T1 MRI input could not be loaded: {}",
                    mri.display()
                )
            })?;
            let [t1_nx, t1_ny, t1_nz] = result.0.shape();
            println!(
                "  T1 loaded       : {t1_nx}×{t1_ny}×{t1_nz} voxels @ [{:.2},{:.2},{:.2}] mm",
                result.1[0], result.1[1], result.1[2]
            );
            Some(result)
        }
        SeismicInputMode::Synthetic | SeismicInputMode::Ct(_) => {
            println!("  T1 mode         : disabled for this explicit input selection");
            None
        }
    };

    // ── [ 3 / 7 ]  Computational grid ────────────────────────────────────
    println!("\n[ 3 / 7 ]  Constructing 3D computational grid …");
    let grid = Grid::new(NX, NY, NZ, DX, DX, DX)?;
    println!("  Grid OK  ({NX}×{NY}×{NZ} @ {:.0} mm)", DX * 1e3);

    // ── [ 4 / 7 ]  Multi-scale FWI parameters ────────────────────────────
    println!("\n[ 4 / 7 ]  Configuring multi-scale FWI …");

    // CFL-stable timestep for 3D PSTD: dt ≤ 0.3 × dx / (c_max × √3).
    // Use actual phantom c_max (not the 2D-demo hardcoded 2621 m/s) with 10 % safety margin.
    let c_max_phantom = phantom
        .acoustic()
        .sound_speed
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let dt = 0.9 * 0.3 * DX / (c_max_phantom * 3.0_f64.sqrt());
    let t_transit = 3.0_f64.sqrt() * NX as f64 * DX / SOUND_SPEED_WATER_SIM;

    // Multi-scale frequency schedule: 40 kHz (5 iter) → 80 kHz (7 iter) → 150 kHz (10 iter).
    let scales: &[(f64, usize)] = &[(40_000.0, 5), (80_000.0, 7), (150_000.0, 10)];

    println!("  dt              : {:.1} ns", dt * 1e9);
    println!(
        "  Scales          : {} → {} → {} kHz  ({}-{}-{} iterations)",
        scales[0].0 * 1e-3,
        scales[1].0 * 1e-3,
        scales[2].0 * 1e-3,
        scales[0].1,
        scales[1].1,
        scales[2].1,
    );

    // ── [ 5 / 7 ]  Fibonacci-sphere acquisition geometry ─────────────────
    println!("\n[ 5 / 7 ]  Building Fibonacci-sphere acquisition geometry …");
    println!("  Array aperture   : {N_SPHERE_ELEMENTS} elements at R={R_ARRAY_3D} voxels");
    println!(
        "  Bowl reference   : {TRANSCRANIAL_FOCUSED_BOWL_ELEMENT_COUNT} elements (full hemispherical array)"
    );
    println!("  Transmits        : {N_SHOTS_3D} shots; receivers/shot = {N_RECEIVERS_3D}");

    let cx_grid = (NX / 2) as f64;
    let cy_grid = (NY / 2) as f64;
    let cz_grid = (NZ / 2) as f64;

    let all_elements =
        fibonacci_sphere_elements(N_SPHERE_ELEMENTS, R_ARRAY_3D, cx_grid, cy_grid, cz_grid);

    // Transmit every other element (even indices → 12 shots).
    let transmit_indices: Vec<usize> = (0..N_SPHERE_ELEMENTS).step_by(2).collect();
    assert_eq!(transmit_indices.len(), N_SHOTS_3D);

    for (shot_num, &elem_idx) in transmit_indices.iter().enumerate() {
        let [ix, iy, iz] = all_elements[elem_idx];
        println!(
            "  Shot {:2}: (ix={:2}, iy={:2}, iz={:2}) = ({:.1} mm, {:.1} mm, {:.1} mm)",
            shot_num,
            ix,
            iy,
            iz,
            ix as f64 * DX * 1e3,
            iy as f64 * DX * 1e3,
            iz as f64 * DX * 1e3
        );
    }

    // ── [ 6 / 7 ]  Multi-scale 3D skull FWI ─────────────────────────────
    println!("\n[ 6 / 7 ]  Running multi-scale 3D transcranial FWI …");

    let true_model = phantom.acoustic().sound_speed.clone();

    // Initial model: Gaussian-blurred CT prior (σ = 3 voxels ≈ 9 mm).
    let initial_model = gaussian_blur_3d(&true_model, 3.0);
    let mut current_model = initial_model.clone();

    // Pre-compute observed gathers at the finest scale (150 kHz) for J₀.
    let nt_fine = ((t_transit * 1.2 + 3.0 / F0_HZ) / dt).ceil() as usize;
    let mut shots_fine: Vec<(FwiGeometry, Array2<f64>)> = Vec::with_capacity(N_SHOTS_3D);
    {
        let tmp_fwi = FwiProcessor::new(FwiParameters {
            max_iterations: 1,
            frequency: F0_HZ,
            nt: nt_fine,
            dt,
            n_trace: N_RECEIVERS_3D,
            n_depth: 1,
            step_size: STEP_SIZE,
            tolerance: 1e-12,
            regularization: RegularizationParameters {
                tikhonov_weight: 0.0,
                tv_weight: 0.0,
                directional_tv_weight: 0.0,
                directional_tv_adaptive: false,
                smoothness_weight: 0.0,
            },
            source_mute_radius: 4,
            ..FwiParameters::default()
        });
        let t0 = Instant::now();
        for &elem_idx in &transmit_indices {
            let geom = build_shot_3d(
                all_elements[elem_idx],
                &all_elements,
                elem_idx,
                F0_HZ,
                nt_fine,
                dt,
            )?;
            let obs = tmp_fwi.generate_synthetic_data(&true_model, &geom, &grid)?;
            shots_fine.push((geom, obs));
        }
        println!(
            "  {} observed gathers at {} kHz ({:.1} s)",
            N_SHOTS_3D,
            F0_HZ * 1e-3,
            t0.elapsed().as_secs_f32()
        );
    }

    let j_initial = {
        let fwi_tmp = FwiProcessor::new(FwiParameters {
            max_iterations: 1,
            frequency: F0_HZ,
            nt: nt_fine,
            dt,
            n_trace: N_RECEIVERS_3D,
            n_depth: 1,
            step_size: STEP_SIZE,
            tolerance: 1e-12,
            regularization: RegularizationParameters {
                tikhonov_weight: 0.0,
                tv_weight: 0.0,
                directional_tv_weight: 0.0,
                directional_tv_adaptive: false,
                smoothness_weight: 0.0,
            },
            source_mute_radius: 4,
            ..FwiParameters::default()
        });
        let mut j = 0.0_f64;
        for (geom, obs) in &shots_fine {
            let d_syn = fwi_tmp.generate_synthetic_data(&initial_model, geom, &grid)?;
            j += d_syn
                .iter()
                .zip(obs.iter())
                .map(|(&s, &o)| (s - o).powi(2))
                .sum::<f64>()
                * 0.5
                * dt;
        }
        j
    };

    println!("\n  Quality before inversion (all voxels):");
    print_quality_report(&true_model, &initial_model);
    println!("  J₀ (150 kHz)    : {j_initial:.6e} Pa²·s  ({N_SHOTS_3D} shots)");

    let t_inv = Instant::now();

    // Multi-scale inversion loop.
    for (scale_idx, &(f0, n_iter)) in scales.iter().enumerate() {
        let nt_scale = ((t_transit * 1.2 + 3.0 / f0) / dt).ceil() as usize;
        let mute_r = ((SOUND_SPEED_WATER_SIM / (2.0 * f0)) / DX).floor() as usize;
        let mute_r = mute_r.clamp(2, 12);

        let mut scale_shots: Vec<(FwiGeometry, Array2<f64>)> = Vec::with_capacity(N_SHOTS_3D);
        let fwi_scale = FwiProcessor::new(FwiParameters {
            max_iterations: n_iter,
            frequency: f0,
            nt: nt_scale,
            dt,
            n_trace: N_RECEIVERS_3D,
            n_depth: 1,
            step_size: STEP_SIZE,
            tolerance: 1e-12,
            regularization: RegularizationParameters {
                tikhonov_weight: 0.0,
                tv_weight: 0.0,
                directional_tv_weight: 0.0,
                directional_tv_adaptive: false,
                smoothness_weight: 0.0,
            },
            source_mute_radius: mute_r,
            ..FwiParameters::default()
        });

        let t_scale = Instant::now();
        for &elem_idx in &transmit_indices {
            let geom = build_shot_3d(
                all_elements[elem_idx],
                &all_elements,
                elem_idx,
                f0,
                nt_scale,
                dt,
            )?;
            let obs = fwi_scale.generate_synthetic_data(&true_model, &geom, &grid)?;
            scale_shots.push((geom, obs));
        }

        println!(
            "\n  ── Scale {} / {} : f₀ = {:.0} kHz, {} iter, nt = {}, mute_r = {} ──",
            scale_idx + 1,
            scales.len(),
            f0 * 1e-3,
            n_iter,
            nt_scale,
            mute_r
        );

        current_model = fwi_scale.invert_multi_source(&scale_shots, &current_model, &grid)?;
        current_model = current_model.mapv(|c| c.max(SOUND_SPEED_WATER_SIM));

        let c_now_max = current_model
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let c_now_min = current_model.iter().copied().fold(f64::INFINITY, f64::min);
        println!(
            "    Scale {} done ({:.1} s): c ∈ [{:.0}, {:.0}] m/s",
            scale_idx + 1,
            t_scale.elapsed().as_secs_f32(),
            c_now_min,
            c_now_max
        );
    }

    let reconstructed = current_model;
    println!(
        "\n  FWI completed in {:.1} s",
        t_inv.elapsed().as_secs_f32()
    );

    // Final J at 150 kHz.
    let j_final = {
        let fwi_tmp = FwiProcessor::new(FwiParameters {
            max_iterations: 1,
            frequency: F0_HZ,
            nt: nt_fine,
            dt,
            n_trace: N_RECEIVERS_3D,
            n_depth: 1,
            step_size: STEP_SIZE,
            tolerance: 1e-12,
            regularization: RegularizationParameters {
                tikhonov_weight: 0.0,
                tv_weight: 0.0,
                directional_tv_weight: 0.0,
                directional_tv_adaptive: false,
                smoothness_weight: 0.0,
            },
            source_mute_radius: 4,
            ..FwiParameters::default()
        });
        let mut j = 0.0_f64;
        for (geom, obs) in &shots_fine {
            let d_syn = fwi_tmp.generate_synthetic_data(&reconstructed, geom, &grid)?;
            j += d_syn
                .iter()
                .zip(obs.iter())
                .map(|(&s, &o)| (s - o).powi(2))
                .sum::<f64>()
                * 0.5
                * dt;
        }
        j
    };

    println!("\n  Quality after inversion (all voxels):");
    print_quality_report(&true_model, &reconstructed);
    println!("  J₀              : {j_initial:.6e} Pa²·s");
    println!(
        "  J_final         : {j_final:.6e} Pa²·s  (reduction: {:.1}×)",
        if j_final > 0.0 {
            j_initial / j_final
        } else {
            f64::INFINITY
        }
    );

    // ── [ 7 / 7 ]  Stage-2 brain tissue FWI ─────────────────────────────
    let brain_prior =
        BrainPriorMode::from_env("KWAVERS_BRAIN_PRIOR").map_err(KwaversError::InvalidInput)?;
    println!("\n[ 7 / 7 ]  Stage-2 3D brain tissue FWI ({brain_prior:?}) …");

    let (_brain_true_model, brain_reconstructed, t1_brain_model) = match build_brain_prior_3d(
        &phantom,
        &brain_prior,
    ) {
        Err(e) => {
            return Err(KwaversError::InvalidInput(format!(
                "selected brain prior failed: {e:#}"
            )));
        }
        Ok(brain_true) => {
            // Skull mask: bone voxels frozen at CT-derived velocity.
            let skull_mask: Array3<bool> = phantom
                .acoustic()
                .sound_speed
                .mapv(|c| c > BONE_VELOCITY_THRESHOLD);
            let n_frozen = skull_mask.iter().filter(|&&b| b).count();
            let n_free = skull_mask.len() - n_frozen;
            println!(
                "  Skull mask        : {n_frozen} frozen bone voxels, {n_free} free brain voxels"
            );

            let (bt_min, bt_max) = skull_mask
                .indexed_iter()
                .filter(|(_, &frozen)| !frozen)
                .map(|([ix, iy, iz], _)| brain_true[[ix, iy, iz]])
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), c| {
                    (mn.min(c), mx.max(c))
                });
            println!("  True brain c      : [{bt_min:.1}, {bt_max:.1}] m/s");

            // Build T1-derived initial brain model when T1 was loaded.
            let t1_brain = match (&brain_prior, t1_result.as_ref()) {
                (_, Some((t1_vol, t1_sp))) => {
                    Some(build_brain_velocity_from_t1(&phantom, t1_vol, *t1_sp))
                }
                (BrainPriorMode::T1(path) | BrainPriorMode::MniT1 { t1: path, .. }, None) => {
                    let (t1_vol, t1_sp) = load_t1_mri(path).with_context(|| {
                        format!("explicit T1 prior could not be loaded: {}", path.display())
                    })?;
                    Some(build_brain_velocity_from_t1(&phantom, &t1_vol, t1_sp))
                }
                _ => None,
            };

            // Brain FWI initial model: T1-derived if available, otherwise uniform water.
            let mut brain_initial = match &t1_brain {
                Some(t1_model) => t1_model.clone(),
                None => skull_mask.mapv(|frozen| {
                    if frozen {
                        0.0_f64
                    } else {
                        SOUND_SPEED_WATER_SIM
                    }
                }),
            };
            // Fill frozen voxels with CT skull velocity.
            let [bi_nx, bi_ny, bi_nz] = brain_initial.shape();
            for i in 0..bi_nx {
                for j in 0..bi_ny {
                    for k in 0..bi_nz {
                        if skull_mask[[i, j, k]] {
                            brain_initial[[i, j, k]] = phantom.acoustic().sound_speed[[i, j, k]];
                        }
                    }
                }
            }

            // Stage-2 FWI at 400 kHz, 15 iterations.
            let f0_brain = 400_000.0_f64;
            let n_brain_iter: usize = 15;
            let step_brain = 30.0_f64;
            let nt_brain = {
                let domain_transit = (NX as f64 * DX) / SOUND_SPEED_WATER_SIM;
                let source_dur = 3.0 / f0_brain;
                ((domain_transit + source_dur) / dt).ceil() as usize
            };

            let fwi_brain = FwiProcessor::new(FwiParameters {
                max_iterations: n_brain_iter,
                frequency: f0_brain,
                nt: nt_brain,
                dt,
                n_trace: N_RECEIVERS_3D,
                n_depth: 1,
                step_size: step_brain,
                tolerance: 1e-14,
                regularization: RegularizationParameters {
                    tikhonov_weight: 0.0,
                    tv_weight: 0.0,
                    directional_tv_weight: 0.0,
                    directional_tv_adaptive: false,
                    smoothness_weight: 0.0,
                },
                source_mute_radius: 2,
                ..FwiParameters::default()
            });

            let mut brain_shots: Vec<(FwiGeometry, Array2<f64>)> = Vec::with_capacity(N_SHOTS_3D);
            let t_brain_obs = Instant::now();
            for &elem_idx in &transmit_indices {
                let geom = build_shot_3d(
                    all_elements[elem_idx],
                    &all_elements,
                    elem_idx,
                    f0_brain,
                    nt_brain,
                    dt,
                )?;
                match fwi_brain.generate_synthetic_data(&brain_true, &geom, &grid) {
                    Ok(obs) => brain_shots.push((geom, obs)),
                    Err(e) => {
                        eprintln!("  Brain gather failed for element {elem_idx}: {e:#}");
                    }
                }
            }
            println!(
                "  {N_SHOTS_3D} brain gathers at {:.0} kHz ({:.1} s)",
                f0_brain * 1e-3,
                t_brain_obs.elapsed().as_secs_f32()
            );

            if brain_shots.is_empty() {
                return Err(KwaversError::InvalidInput(format!(
                    "brain FWI produced no successful gathers from {N_SHOTS_3D} shots"
                )));
            }

            println!(
                "  Running {n_brain_iter} iterations at {:.0} kHz (nt={nt_brain}) …",
                f0_brain * 1e-3
            );
            let t_brain_inv = Instant::now();
            let brain_recon = fwi_brain
                .invert_multi_source_masked(
                    &brain_shots,
                    &brain_initial,
                    &phantom.acoustic().sound_speed,
                    &skull_mask,
                    BRAIN_C_MIN,
                    BRAIN_C_MAX,
                    &grid,
                )
                .map_err(|error| {
                    KwaversError::InvalidInput(format!("brain FWI inversion failed: {error:#}"))
                })?;
            println!(
                "  Brain FWI done ({:.1} s)",
                t_brain_inv.elapsed().as_secs_f32()
            );
            println!("  Quality (brain voxels only, r_3d < R_SKULL_IN):");
            print_quality_report_brain(&brain_true, &brain_recon);
            (Some(brain_true), Some(brain_recon), t1_brain)
        }
    };

    // ── Image output ──────────────────────────────────────────────────────
    let output_dir: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/seismic_imaging_3d_demo"));

    std::fs::create_dir_all(&output_dir)
        .map_err(|e| KwaversError::InvalidInput(format!("cannot create output dir: {e}")))?;

    let abs_dir = std::fs::canonicalize(&output_dir).map_err(|error| {
        KwaversError::InvalidInput(format!("cannot canonicalize output dir: {error}"))
    })?;

    // Skull velocity colourmap bounds.
    let c_lo = SOUND_SPEED_WATER_SIM;
    let c_hi = kwavers_core::constants::acoustic_parameters::SOUND_SPEED_SKULL_CORTICAL;

    let axial_path = abs_dir.join("brain3d_fwi_axial.png");
    let coronal_path = abs_dir.join("brain3d_fwi_coronal.png");
    let sagittal_path = abs_dir.join("brain3d_fwi_sagittal.png");
    let t1_tissue_path = abs_dir.join("brain3d_t1_tissue.png");
    let brain_tissue_path = abs_dir.join("brain3d_brain_tissue.png");

    // Three-plane skull FWI images (write from reconstructed model).
    seismic_volume_artifacts::write_orthogonal_slices_png(&axial_path, &reconstructed, c_lo, c_hi)
        .map_err(|e| KwaversError::InvalidInput(format!("axial PNG write failed: {e}")))?;

    seismic_volume_artifacts::write_orthogonal_slices_png(
        &coronal_path,
        &reconstructed,
        c_lo,
        c_hi,
    )
    .map_err(|e| KwaversError::InvalidInput(format!("coronal PNG write failed: {e}")))?;

    seismic_volume_artifacts::write_orthogonal_slices_png(
        &sagittal_path,
        &reconstructed,
        c_lo,
        c_hi,
    )
    .map_err(|e| KwaversError::InvalidInput(format!("sagittal PNG write failed: {e}")))?;

    // T1-derived tissue velocity image.
    if let Some(t1_brain) = &t1_brain_model {
        seismic_volume_artifacts::write_orthogonal_slices_png(
            &t1_tissue_path,
            t1_brain,
            BRAIN_C_MIN,
            BRAIN_C_MAX,
        )
        .map_err(|e| KwaversError::InvalidInput(format!("T1 tissue PNG write failed: {e}")))?;
        println!("  T1 tissue image  : {}", t1_tissue_path.display());
    }

    // Brain tissue FWI result image.
    if let Some(bt_recon) = &brain_reconstructed {
        seismic_volume_artifacts::write_orthogonal_slices_png(
            &brain_tissue_path,
            bt_recon,
            BRAIN_C_MIN,
            BRAIN_C_MAX,
        )
        .map_err(|e| KwaversError::InvalidInput(format!("brain tissue PNG write failed: {e}")))?;
        println!("  Brain tissue image: {}", brain_tissue_path.display());
    }

    println!("\n  Output directory  : {}", abs_dir.display());
    println!("\n  Wrote images:");
    println!(
        "    {}  (axial+coronal+sagittal skull velocity, reconstructed)",
        axial_path.display()
    );
    println!("    {}", coronal_path.display());
    println!("    {}", sagittal_path.display());

    // ── Summary footer ────────────────────────────────────────────────────
    println!("\n  Physics references:");
    println!("    Aubry (2003) — HU bone-volume-fraction acoustic model");
    println!("    Marsac (2017) — transcranial FWI protocol (150 kHz–650 kHz)");
    println!("    Guasch (2020) — full-waveform inversion of the human brain");
    println!("    Treeby & Cox (2010) — fractional-Laplacian absorption model");
    println!("    Virieux & Operto (2009) — review of FWI in geophysics");
    println!("    Duck (1990) — tissue acoustic properties");
    println!("    chris T1/T2 MRI (niivue-images, Rorden 2024) — individual subject MRI");
    println!("    CT_Philips NIfTI (niivue-images) — CT input");
    println!("    MNI ICBM 2009c (Fonov 2009) — atlas tissue probability maps");

    Ok(())
}
