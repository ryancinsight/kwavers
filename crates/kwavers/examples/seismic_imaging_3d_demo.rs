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
#[path = "seismic_imaging/volume_acquisition.rs"]
mod seismic_volume_acquisition;
#[path = "seismic_imaging/volume_artifacts.rs"]
mod seismic_volume_artifacts;
#[path = "seismic_imaging/volume_brain_inversion.rs"]
mod seismic_volume_brain_inversion;
#[path = "seismic_imaging/volume_brain_model.rs"]
mod seismic_volume_brain_model;
#[path = "seismic_imaging/volume_initial_model.rs"]
mod seismic_volume_initial_model;
#[path = "seismic_imaging/volume_phantom.rs"]
mod seismic_volume_phantom;
#[path = "seismic_imaging/volume_reporting.rs"]
mod seismic_volume_reporting;
#[path = "seismic_imaging/volume_skull_inversion.rs"]
mod seismic_volume_skull_inversion;

use anyhow::Context as _;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use kwavers_solver::inverse::fwi::time_domain::{FwiGeometry, FwiProcessor};
use kwavers_solver::inverse::seismic::parameters::{FwiParameters, RegularizationParameters};
use leto::{Array2, Array3};

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

// ─────────────────────────────────────────────────────────────────────────────
// T1 MRI loading and tissue velocity mapping
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// Fibonacci-sphere acquisition geometry
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// Gaussian blur (3D separable)
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// Image output
// ─────────────────────────────────────────────────────────────────────────────

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
            let result = seismic_volume_brain_model::load_t1_mri(mri).with_context(|| {
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

    let all_elements = seismic_volume_acquisition::fibonacci_sphere_elements(
        N_SPHERE_ELEMENTS,
        R_ARRAY_3D,
        cx_grid,
        cy_grid,
        cz_grid,
    );

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
    let reconstructed = seismic_volume_skull_inversion::run_skull_inversion(
        &phantom,
        &grid,
        &all_elements,
        &transmit_indices,
        dt,
        t_transit,
        scales,
    )?;

    // ── [ 7 / 7 ]  Stage-2 brain tissue FWI ─────────────────────────────
    let brain_prior =
        BrainPriorMode::from_env("KWAVERS_BRAIN_PRIOR").map_err(KwaversError::InvalidInput)?;
    println!("\n[ 7 / 7 ]  Stage-2 3D brain tissue FWI ({brain_prior:?}) …");

    let brain_result = seismic_volume_brain_inversion::run_brain_inversion(
        &phantom,
        &brain_prior,
        &grid,
        dt,
        &all_elements,
        &transmit_indices,
        t1_result.as_ref(),
    )?;
    let brain_reconstructed = brain_result.reconstructed;
    let t1_brain_model = brain_result.t1_model;

    let output_dir = std::env::args()
        .nth(1)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("target/seismic_imaging_3d_demo"));
    seismic_volume_reporting::write_outputs(
        output_dir,
        &reconstructed,
        t1_brain_model.as_ref(),
        brain_reconstructed.as_ref(),
    )?;

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
