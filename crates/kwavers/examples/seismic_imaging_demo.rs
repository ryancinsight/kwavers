//! Transcranial Ultrasound FWI — Brain Reconstruction Demo.
//!
//! # Physical pipeline
//!
//! ```text
//! Skull CT phantom  →  c(x), ρ(x)  →  FDTD forward  →  synthetic traces
//!                                                               │
//!                              ← adjoint source ←  L2 residual
//!                              │
//!                              FDTD adjoint (time-reversed, back-propagated)
//!                              │
//!                              gradient ∂J/∂c  →  model update  →  brain image
//! ```
//!
//! # Skull phantom
//!
//! Coronal cross-section (x–z plane) of a human head modelled as concentric
//! shells centred at (NX/2, NZ/2) = (32, 32):
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │               water coupling bath                   │
//! │         ┌─────────────────────────┐                 │
//! │         │   scalp  (HU ≈  40)    │                  │
//! │         │  ┌─────────────────┐   │                  │
//! │         │  │  outer cortical │   │                  │
//! │         │  │  bone (HU≈720) │   │  ← z (depth)     │
//! │         │  │  ┌───────────┐  │   │                  │
//! │         │  │  │  diploe   │  │   │                  │
//! │         │  │  │ (HU≈380) │  │   │                  │
//! │         │  │  │ ┌───────┐ │  │   │                  │
//! │         │  │  │ │ inner │ │  │   │                  │
//! │  SRC    │  │  │ │ cort. │ │  │  RECV               │
//! │  (left  │  │  │ │┌─────┐│ │  │  (right             │
//! │  arc)   │  │  │ ││brain││ │  │   arc)              │
//! │         │  │  │ │└─────┘│ │  │                     │
//! └─────────┴──┴──┴─┴───────┴─┴──┴─────────────────────┘
//!             ↑ x (lateral, left→right)
//! ```
//!
//! # Full-ring acquisition geometry
//!
//! The 2-D FWI acquisition uses sixteen active element locations uniformly
//! distributed around the full ring at R_ARRAY = 20 voxels from the grid
//! centre.  Eight of them transmit in sequence (every other element) while the
//! remaining fifteen act as receivers.  Full-ring coverage provides illumination
//! from all azimuths, eliminating the shadow zone that limits superior-hemisphere
//! geometries and improving convergence for inferior skull structures.
//!
//! # Initial model — CT-derived smooth prior
//!
//! Starting FWI from a homogeneous 1500 m/s background causes cycle-skipping:
//! at f₀ = 150 kHz, T = 6.7 μs, but skull transmission delay ≈ 5 μs > T/2.
//! The wave never converges from such a large initial model error.
//!
//! Instead, the initial model is a Gaussian-blurred version of the true skull
//! model (σ = 3 voxels ≈ 9 mm).  This is the standard clinical approach: a
//! low-resolution CT scan is always available and provides a smooth but
//! geometrically correct bone map.  Starting from this smooth map ensures travel
//! times are within λ/2 of the truth, so the FWI refines boundaries rather
//! than fighting cycle-skipping.
//!
//! Reference: Guasch (2020) — CT-based initial model for brain FWI, §Methods.
//!
//! # FWI objective and gradient
//!
//! Acoustic L2 misfit (Tarantola 1984; Virieux & Operto 2009):
//!
//! ```text
//! J(c) = (dt / 2) Σ_{r,t} [d_syn(r,t; c) − d_obs(r,t)]²
//!
//! ∂J/∂m(x) = −∫₀ᵀ λ(x, T−t) ∂²p(x,t)/∂t² dt,   m = c⁻²
//! ∂J/∂c(x) = −2 c(x)⁻³ ∂J/∂m(x)
//! ```
//!
//! # References
//!
//! - Aubry, J.-F. et al. (2003). Experimental demonstration of noninvasive
//!   transskull adaptive focusing. *JASA*, 113(1), 84–93.
//! - Marsac, L. et al. (2017). Ex vivo optimisation of a heterogeneous speed of
//!   sound model of the human skull. *Int. J. Hyperthermia*, 33(6), 635–645.
//! - Guasch, L. et al. (2020). Full-waveform inversion imaging of the human
//!   brain. *npj Digital Medicine*, 3, 28.
//! - Tarantola, A. (1984). Inversion of seismic reflection data in the acoustic
//!   approximation. *Geophysics*, 49(8), 1259–1266.
//! - Virieux, J. & Operto, S. (2009). An overview of full-waveform inversion in
//!   exploration geophysics. *Geophysics*, 74(6), WCC1–WCC26.

#[path = "seismic_imaging/acquisition.rs"]
mod seismic_acquisition;
#[path = "seismic_imaging/brain_inversion.rs"]
mod seismic_brain_inversion;
#[path = "seismic_imaging/brain_model.rs"]
mod seismic_brain_model;
mod seismic_imaging;
#[path = "seismic_imaging/initial_model.rs"]
mod seismic_initial_model;
#[path = "seismic_imaging/metrics.rs"]
mod seismic_metrics;
#[path = "seismic_imaging/phantom.rs"]
mod seismic_phantom;
#[path = "seismic_imaging/planar_artifacts.rs"]
mod seismic_planar_artifacts;
#[path = "seismic_imaging/planar_auxiliary.rs"]
mod seismic_planar_auxiliary;
#[path = "seismic_imaging/planar_inversion.rs"]
mod seismic_planar_inversion;
#[path = "seismic_imaging/planar_reporting.rs"]
mod seismic_planar_reporting;
#[path = "seismic_imaging/planar_schedule.rs"]
mod seismic_planar_schedule;
#[path = "seismic_imaging/rtm.rs"]
mod seismic_rtm;
use seismic_metrics::print_quality_pairs;

use kwavers_core::constants::{
    acoustic_parameters::SOUND_SPEED_SKULL_CORTICAL, fundamental::SOUND_SPEED_WATER_SIM,
};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use kwavers_solver::inverse::fwi::time_domain::{FwiGeometry, FwiProcessor};
use kwavers_solver::inverse::seismic::parameters::{FwiParameters, RegularizationParameters};
use leto::{Array2, Array3};
use std::path::PathBuf;

#[path = "support/brain_prior.rs"]
mod brain_prior;
#[path = "support/seismic_input.rs"]
mod seismic_input;
use brain_prior::BrainPriorMode;
use seismic_input::SeismicInputMode;

use seismic_imaging::render::{put_pixel, velocity_color, write_png};

// ─────────────────────────────────────────────────────────────────────────────
// Grid constants
// ─────────────────────────────────────────────────────────────────────────────

/// Grid spacing `m`.  3 mm gives λ/3.3 resolution at 150 kHz in water.
///
/// Reference: Marsac 2017 — mean skull thickness ≈ 7 mm → at least 2 voxels
/// through bone at 3 mm spacing.
const DX: f64 = 3.0e-3;

/// Grid dimensions.  NY = 2 satisfies the FDTD staggered-stencil minimum while
/// keeping the second y-plane acoustically transparent (identical medium).
const NX: usize = 64; // lateral  192 mm
const NY: usize = 2; // quasi-2-D embedding
const NZ: usize = 64; // depth    192 mm

// ─────────────────────────────────────────────────────────────────────────────
// Skull phantom geometry — radii in voxels from centre (32, 32)
// ─────────────────────────────────────────────────────────────────────────────
//
// # CPML geometry constraint
//
// FDTD solver uses CPML thickness = 10 cells on all active boundaries.
// Physical domain: ix ∈ [10, 53], iz ∈ [10, 53] (44×44 voxels = 132 mm).
//
// The skull must fit entirely inside the physical domain:
//   max(R_HEAD + cx, cz) ≤ NX − CPML = 54  →  R_HEAD ≤ 22
//
// We use R_HEAD = 18 (54 mm radius) to provide a 6-voxel water bath margin
// between the outer scalp edge (ix = 32±18 = 14..50) and the CPML boundary
// (ix = 10 and ix = 54).
//
// Reference: Marsac 2017 — head radius ≈ 80 mm; skull thickness ≈ 7 mm.
// Scaled proportionally: 18/26 × original = 0.69 scaling factor.

const R_HEAD: f64 = 18.0; // 54 mm — outer scalp surface
const R_SKULL_OUT: f64 = 16.0; // 48 mm — outer cortical / scalp boundary
const R_DIPLOE: f64 = 14.0; // 42 mm — outer diploe boundary
const R_SKULL_IN: f64 = 12.0; // 36 mm — inner cortical / brain boundary
const R_BRAIN: f64 = 11.0; // 33 mm — brain surface (CSF buffer ≈ 3 mm)

// ─────────────────────────────────────────────────────────────────────────────
// Hounsfield-unit phantom labels
// ─────────────────────────────────────────────────────────────────────────────

/// Typical HU values per skull layer (Aubry 2003 Table I; Marsac 2017 Table 1).
const HU_WATER: f64 = 0.0; // water coupling bath
const HU_SCALP: f64 = 40.0; // soft tissue / scalp / dura
const HU_CORTICAL_OUT: f64 = 720.0; // outer cortical bone
const HU_DIPLOE: f64 = 380.0; // trabecular / diploe
const HU_CORTICAL_IN: f64 = 660.0; // inner cortical bone
const HU_BRAIN: f64 = 35.0; // grey/white matter average

/// Phase-correction example Medimodel series UID.  The directory contains
/// additional derived CT series with inconsistent orientation; this UID is the
/// same 67-slice skull CT series used by the companion example.
/// Full hemispherical aperture size used by the companion phase-correction demo.
const TRANSCRANIAL_FOCUSED_BOWL_ELEMENT_COUNT: usize = 1024;

/// Colormap bounds [m/s] for skull image panels.
const C_LO: f64 = SOUND_SPEED_WATER_SIM; // 1500 m/s → blue
const C_HI: f64 = SOUND_SPEED_SKULL_CORTICAL; // 3100 m/s → red

// ─────────────────────────────────────────────────────────────────────────────
// Stage-2 brain tissue FWI constants (Guasch 2020 §Methods)
// ─────────────────────────────────────────────────────────────────────────────

/// Brain tissue sound speeds from Duck (1990) "Physical Properties of Tissue".
const C_GRAY: f64 = 1541.0; // gray matter [m/s]
const C_WHITE: f64 = 1520.0; // white matter [m/s]
const C_CSF: f64 = 1505.0; // cerebrospinal fluid [m/s]

/// Velocity bounds for brain tissue FWI (excludes bone which is frozen).
const BRAIN_C_MIN: f64 = 1480.0; // m/s
const BRAIN_C_MAX: f64 = 1560.0; // m/s

/// Velocity threshold for classifying a voxel as bone (frozen in Stage 2).
/// Corresponds to BVF > 0.143, approximately HU ≈ 143 → cortical bone onset.
const BONE_VELOCITY_THRESHOLD: f64 = 1714.0; // m/s

/// Peak frequency for brain tissue FWI.
/// At 400 kHz: λ_brain ≈ 3.8 mm; tissue velocity errors < 3% → no cycle-skipping.
const F0_BRAIN_HZ: f64 = 400_000.0; // Hz

/// Number of FWI iterations for brain tissue Stage 2.
const N_BRAIN_ITER: usize = 20;

/// Step size for brain tissue FWI (smaller than skull FWI — brain Δc is tiny).
const STEP_SIZE_BRAIN: f64 = 30.0; // m/s per normalised gradient step

/// MNI ICBM 2009c inner-skull radius at the coronal mid-plane `mm`.
/// The inner cortical surface is ≈ 82 mm from the brain centroid in this atlas.
const MNI_INNER_SKULL_RADIUS_MM: f64 = 82.0;

// ─────────────────────────────────────────────────────────────────────────────
// Gaussian blur for CT-derived initial model
// ─────────────────────────────────────────────────────────────────────────────

/// FWI gradient descent step size [m/s].
const STEP_SIZE: f64 = 50.0;

/// Pixel size per model panel [px].
const PANEL: usize = 320;

/// Colorbar height below each panel [px].
const COLORBAR_H: usize = 20;

// ─────────────────────────────────────────────────────────────────────────────
// Reconstruction quality metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Print RMSE, Pearson r, max |error|, ±10 m/s fraction for brain voxels only
/// (geometric: r < R_SKULL_IN from grid center, independent of FWI frozen mask).
///
/// Uses the same geometric boundary as `write_brain_tissue_png` so the quality
/// metrics and visualization are consistent.
fn print_quality_report_brain(true_model: &Array3<f64>, reconstructed: &Array3<f64>) {
    let cx = (NX / 2) as f64;
    let cz = (NZ / 2) as f64;
    let free_pairs: Vec<(f64, f64)> = true_model
        .indexed_iter()
        .filter(|([ix, _iy, iz], _)| {
            let r = (((*ix as f64) - cx).powi(2) + ((*iz as f64) - cz).powi(2)).sqrt();
            r < R_SKULL_IN
        })
        .map(|([ix, _iy, iz], &t)| (t, reconstructed[[ix, _iy, iz]]))
        .collect();
    print_quality_pairs(&free_pairs);
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> KwaversResult<()> {
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("warn"));

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   Transcranial Ultrasound FWI — Brain Reconstruction     ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // ── 1. Skull phantom ──────────────────────────────────────────────────
    println!("[ 1 / 6 ]  Building skull phantom …");
    let input_mode = SeismicInputMode::from_env("KWAVERS_SEISMIC_INPUT_MODE")
        .map_err(KwaversError::InvalidInput)?;
    println!("  Input mode       : {input_mode:?}");
    let (phantom, ct_vol) = seismic_phantom::build_phantom_for_demo(&input_mode)
        .map_err(|error| KwaversError::InvalidInput(error.to_string()))?;

    let c_min = phantom
        .acoustic()
        .sound_speed
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let c_max = phantom
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

    println!(
        "  Grid            : {NX}×{NY}×{NZ} voxels @ {:.0} mm",
        DX * 1e3
    );
    println!(
        "  Domain          : {:.0}×{:.0} mm",
        NX as f64 * DX * 1e3,
        NZ as f64 * DX * 1e3
    );
    println!("  HU range        : [{hu_min:.0}, {hu_max:.0}]");
    println!("  Sound-speed     : [{c_min:.0}, {c_max:.0}] m/s");
    println!(
        "  Head radius     : {:.0} mm  (R_HEAD = {R_HEAD} voxels)",
        R_HEAD * DX * 1e3
    );
    println!(
        "  Skull thickness : ~{:.0} mm  (outer cortical → inner cortical)",
        (R_SKULL_OUT - R_SKULL_IN) * DX * 1e3
    );
    println!("  Brain radius    : {:.0} mm", R_BRAIN * DX * 1e3);
    println!("  Layers          : water coupling / scalp / cortical bone / diploe / brain");

    // ── 2. Grid ───────────────────────────────────────────────────────────
    println!("\n[ 2 / 6 ]  Constructing computational grid …");
    let grid = Grid::new(NX, NY, NZ, DX, DX, DX)?;
    println!("  Grid OK");

    // ── 3. Multi-scale FWI parameters ────────────────────────────────────
    println!("\n[ 3 / 6 ]  Configuring multi-scale FWI …");

    // CFL-stable timestep: dt ≤ 0.3 × dx / (c_max × √3).
    // Fixed across all scales — determined by maximum velocity, not frequency.
    let dt = 0.3 * DX / (c_max * 3.0_f64.sqrt());

    // Full domain transit time at water speed (same for all scales).
    let t_transit = (NX as f64 * DX) / SOUND_SPEED_WATER_SIM;

    let scales = seismic_planar_schedule::configure(dt, t_transit);

    // ── 4. Full-ring acquisition geometry ─────────────────────────────────
    println!("\n[ 4 / 6 ]  Building full-ring acquisition geometry …");
    println!(
        "  Full aperture    : {TRANSCRANIAL_FOCUSED_BOWL_ELEMENT_COUNT} elements, 650 kHz design authority"
    );
    println!(
        "  FWI section      : {} active full-ring samples",
        seismic_acquisition::FWI_ACTIVE_ELEMENTS
    );
    println!(
        "  Transmits        : {} shots; receivers/shot = {} on same full ring",
        seismic_acquisition::N_SHOTS,
        seismic_acquisition::N_RECEIVERS
    );
    for (s, &element_index) in seismic_acquisition::TRANSMIT_ELEMENT_INDICES
        .iter()
        .enumerate()
    {
        let (ix, iz) = seismic_acquisition::ACTIVE_TRANSDUCER_POSITIONS[element_index];
        println!(
            "  Shot {:1}: (x={:2}, y=0, z={:2}) = ({:.1} mm, {:.1} mm)",
            s,
            ix,
            iz,
            ix as f64 * DX * 1e3,
            iz as f64 * DX * 1e3
        );
    }

    // ── 5. Multi-scale FWI ────────────────────────────────────────────────
    println!("\n[ 5 / 6 ]  Running multi-scale transcranial FWI …");

    let inversion =
        seismic_planar_inversion::run_skull_inversion(&phantom, &grid, dt, t_transit, scales)?;
    let true_model = inversion.true_model;
    let initial_model = inversion.initial_model;
    let reconstructed = inversion.reconstructed;
    let shots_fine = inversion.shots_fine;

    // ── 6. Stage-2 brain tissue FWI (Guasch 2020 style) ─────────────────
    let brain_prior =
        BrainPriorMode::from_env("KWAVERS_BRAIN_PRIOR").map_err(KwaversError::InvalidInput)?;
    println!("\n[ 6 / 7 ]  Stage-2 brain tissue FWI ({brain_prior:?}) …");

    let brain_result = seismic_brain_inversion::run_brain_fwi(&phantom, &brain_prior, &grid, dt)?;
    println!("  Quality (brain voxels only, r < R_SKULL_IN):");
    print_quality_report_brain(&brain_result.true_model, &brain_result.reconstructed);
    let brain_true_model = Some(brain_result.true_model);
    let brain_reconstructed = Some(brain_result.reconstructed);

    // ── 7. RTM — zero-lag cross-correlation imaging ───────────────────────
    println!("\n[ 7 / 7 ]  Reverse Time Migration (reflectivity image) …");

    let rtm_image = seismic_rtm::run_rtm(&shots_fine, &grid)?;

    // ── Image output ──────────────────────────────────────────────────────
    let output_dir: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/seismic_imaging_demo"));

    seismic_planar_reporting::write_outputs(seismic_planar_reporting::PlanarOutput {
        output_dir,
        phantom: &phantom,
        ct_vol: ct_vol.as_ref(),
        true_model: &true_model,
        initial_model: &initial_model,
        reconstructed: &reconstructed,
        brain_true: brain_true_model.as_ref(),
        brain_reconstructed: brain_reconstructed.as_ref(),
        rtm_image: &rtm_image,
    })?;

    // ── Summary ───────────────────────────────────────────────────────────
    println!("\n═══════════════════════════════════════════════════════════");
    println!(
        "  Reconstructed velocity range: [{:.0}, {:.0}] m/s",
        reconstructed.iter().copied().fold(f64::INFINITY, f64::min),
        reconstructed
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
    );
    println!("  True velocity range         : [{c_min:.0}, {c_max:.0}] m/s");
    println!();
    println!("  Physics verified against:");
    println!("    Aubry (2003)              — HU → c, ρ bone-volume-fraction model");
    println!("    Marsac (2017)             — skull acoustic properties + geometry");
    println!("    Guasch (2020)             — transcranial FWI methodology");
    println!("    Ricker (1953)             — source wavelet");
    println!("    Tarantola (1984)          — adjoint-state FWI gradient");
    println!("    Virieux & Operto (2009)   — FWI objective and chain rule");
    println!();
    println!("  To use an explicit CT input:");
    println!("    set KWAVERS_SEISMIC_INPUT_MODE=ct:path\\to\\ct_dicom_or_nifti");
    println!("    cargo run --example seismic_imaging_demo");
    println!("═══════════════════════════════════════════════════════════");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brain_support_fills_non_bone_region_between_skull_edges() {
        let mut hu = Array3::<f64>::from_elem((NX, NY, NZ), HU_WATER);
        let row = NZ / 2;
        hu[[20, 0, row]] = 500.0;
        hu[[44, 0, row]] = 500.0;
        hu[[32, 0, row]] = -1000.0;
        hu[[20, 1, row]] = 500.0;
        hu[[44, 1, row]] = 500.0;

        let mask = seismic_brain_model::brain_support_from_hu(&hu);

        assert!(mask[[32, row]]);
        assert!(!mask[[20, row]]);
        assert!(!mask[[44, row]]);
        assert!(!mask[[10, row]]);
        assert_eq!((21..44).filter(|&ix| mask[[ix, row]]).count(), 23);
    }
}
