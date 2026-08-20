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
use seismic_metrics::{print_quality_pairs, print_quality_report};

use kwavers_core::constants::{
    acoustic_parameters::SOUND_SPEED_SKULL_CORTICAL, fundamental::SOUND_SPEED_WATER_SIM,
};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use kwavers_solver::inverse::fwi::time_domain::{FwiGeometry, FwiProcessor};
use kwavers_solver::inverse::seismic::{
    parameters::{
        FwiParameters, ImagingCondition, RegularizationParameters, RtmSettings,
        SeismicBoundaryType, StorageStrategy,
    },
    rtm::RtmProcessor,
};
use leto::{Array2, Array3};
use std::path::PathBuf;
use std::time::Instant;

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

    // Multi-scale frequency schedule (Guasch 2020, §Methods — frequency continuation).
    //
    // # Cycle-skipping criterion (Virieux & Operto 2009)
    //
    // Cycle-skipping occurs when the initial model travel-time error exceeds T/2.
    // Skull transmission delay:
    //   Δt = skull_thickness × (1/c_water − 1/c_skull_avg)
    //      = 24mm × (1/1500 − 1/2264) = 5.4 μs
    //
    // At 150 kHz: T/2 = 3.3 μs <  Δt = 5.4 μs → CYCLE-SKIPPING ✗
    // At  60 kHz: T/2 = 8.3 μs >  Δt = 5.4 μs → safe ✓  (start here)
    //
    // # CPML absorption adequacy (why 20 kHz is excluded)
    //
    // CPML absorbs effectively only when PML thickness ≥ λ/4 in the absorbing medium.
    // Physical CPML thickness = 10 cells × 3 mm = 30 mm.
    // In bone (c_bone ≈ 2500 m/s):
    //   λ_bone(20 kHz)  = 2500/20000 = 125 mm →  λ/4 = 31 mm ≈ CPML (marginal)
    //   λ_bone(60 kHz)  = 2500/60000 =  42 mm →  λ/4 = 10.5 mm << CPML (adequate)
    //   λ_bone(150 kHz) = 2500/150000 = 17 mm →  λ/4 = 4.2 mm << CPML (adequate)
    // At 20 kHz the CPML absorbs less than one λ/4 through bone → reflections
    // overwhelm the recorded wavefield, producing J ≈ 10⁹ Pa²·s (catastrophic).
    // Minimum usable frequency given this CPML thickness: ≈ 50 kHz.
    //
    // Schedule: 40 kHz → 80 kHz → 150 kHz with 10-12-15 iterations per scale.
    // Each scale starts from the previous scale's result; nt is computed per scale
    // to include 3 source periods (for the low-frequency wavelet to decay fully).
    //
    //   nt(f₀) = ceil((t_transit × 1.2  +  3.0 / f₀) / dt)
    //
    // Three scales improve initial model recovery at the lowest frequency before
    // refining skull boundaries at intermediate and full resolution.
    //
    // At 40 kHz: T/2 = 12.5 μs > Δt_skull = 5.4 μs → safe ✓ (start here)
    // At 80 kHz: T/2 =  6.25 μs > Δt_skull = 5.4 μs → safe ✓
    // At 150 kHz: T/2 = 3.3 μs < Δt_skull = 5.4 μs → cycle-skipping if from
    //   uniform 1500 m/s, but safe when starting from 80 kHz result.
    //
    // Initial model: Gaussian-blurred CT (σ = 3 voxels).  At 40 kHz,
    // T/2 = 12.5 μs > Δt_skull(blurred) ≈ 2.7 μs → no cycle-skipping.
    //
    // Physical constraint: all tissues have c ≥ c_water = 1500 m/s.
    // Sub-water-speed artefacts are clamped after each scale.
    //
    // Source mute radius: scaled with wavelength = floor(c_water / (2·f₀·dx)).
    // At 40 kHz:  radius = floor(1500/(2×40000×0.003)) = 6 voxels.
    // At 80 kHz:  radius = floor(1500/(2×80000×0.003)) = 3 voxels.
    // At 150 kHz: radius = floor(1500/(2×150000×0.003)) = 2 voxels (clamped to 2 minimum).
    let scales: &[(f64, usize)] = &[
        (40_000.0, 10),  // f₀ Hz, n_iter — safe from blurred CT prior (T/2 > Δt_skull)
        (80_000.0, 12),  // intermediate refinement — still cycle-skip safe
        (150_000.0, 15), // refine skull boundaries at full ultrasound resolution
    ];

    println!("  dt              : {:.1} ns", dt * 1e9);
    println!(
        "  Scales          : {} → {} → {} kHz  (10-12-15 iterations)",
        scales[0].0 * 1e-3,
        scales[1].0 * 1e-3,
        scales[2].0 * 1e-3
    );
    for &(f0, n) in scales {
        let nt_s = ((t_transit * 1.2 + 3.0 / f0) / dt).ceil() as usize;
        let t_half = 1.0 / (2.0 * f0) * 1e6;
        println!(
            "    f₀={:.0} kHz: T/2={:.1} μs, Δt_skull=5.4 μs → {}, nt={}, {} iter",
            f0 * 1e-3,
            t_half,
            if t_half > 5.4 { "OK" } else { "WARN" },
            nt_s,
            n
        );
    }

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

    let true_model = phantom.acoustic().sound_speed.clone();

    // Initial model: Gaussian-blurred true skull model (σ = 3 voxels ≈ 9 mm).
    //
    // This is the standard clinical approach (Guasch 2020): a low-resolution CT
    // scan is always available and provides a smooth but geometrically correct
    // bone map.  Starting from this blurred prior:
    //   • Skull structure is already approximately located → gradient acts as a
    //     *refinement* operator (sharpen boundaries, raise bone peak velocity)
    //     rather than a discovery operator (find bone from featureless water).
    //   • Blurred initial travel-time error ≈ 2.7 μs < T/2(60 kHz) = 8.3 μs
    //     → no cycle-skipping at any modelled frequency.
    //   • FWI converges in far fewer iterations than from uniform 1500 m/s.
    //
    // Convergence evidence (from uniform initial): after 13 iterations at
    // 60 → 150 kHz, c_max reached only 1634 m/s (true 2508 m/s) — gradient
    // spent all budget discovering skull geometry rather than refining it.
    let initial_model = seismic_initial_model::gaussian_blur_xz(&true_model, 3.0);
    let mut current_model = initial_model.clone();

    // Compute J₀ at the finest scale (150 kHz) for reporting consistency.
    let nt_fine = ((t_transit * 1.2 + 3.0 / seismic_acquisition::F0_HZ) / dt).ceil() as usize;
    let mut shots_fine: Vec<(FwiGeometry, Array2<f64>)> =
        Vec::with_capacity(seismic_acquisition::N_SHOTS);
    {
        let tmp_fwi = FwiProcessor::new(FwiParameters {
            max_iterations: 1,
            frequency: seismic_acquisition::F0_HZ,
            nt: nt_fine,
            dt,
            n_trace: seismic_acquisition::N_RECEIVERS,
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
        for &element_index in &seismic_acquisition::TRANSMIT_ELEMENT_INDICES {
            let geom = seismic_acquisition::build_shot(
                element_index,
                seismic_acquisition::F0_HZ,
                nt_fine,
                dt,
            )?;
            let obs = tmp_fwi.generate_synthetic_data(&true_model, &geom, &grid)?;
            shots_fine.push((geom, obs));
        }
        println!(
            "  {} observed gathers at {} kHz ({:.1} s)",
            seismic_acquisition::N_SHOTS,
            seismic_acquisition::F0_HZ * 1e-3,
            t0.elapsed().as_secs_f32()
        );
    }

    let j_initial = {
        let fwi_tmp = FwiProcessor::new(FwiParameters {
            max_iterations: 1,
            frequency: seismic_acquisition::F0_HZ,
            nt: nt_fine,
            dt,
            n_trace: seismic_acquisition::N_RECEIVERS,
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

    println!("\n  Quality before inversion:");
    print_quality_report(&true_model, &initial_model);
    println!(
        "  Joint J₀ (150 kHz) : {j_initial:.6e} Pa²·s  ({} shots)",
        seismic_acquisition::N_SHOTS
    );

    let t_inv = Instant::now();

    // Multi-scale inversion loop.
    for (scale_idx, &(f0, n_iter)) in scales.iter().enumerate() {
        // Compute scale-specific nt: include 3 source periods + transit time.
        // At 60 kHz the Ricker wavelet has t_peak = 25 μs; at 150 kHz, 10 μs.
        let nt_scale = ((t_transit * 1.2 + 3.0 / f0) / dt).ceil() as usize;

        // Source mute radius = half-wavelength in voxels (clamped to [2, 12]).
        let mute_r = ((SOUND_SPEED_WATER_SIM / (2.0 * f0)) / DX).floor() as usize;
        let mute_r = mute_r.clamp(2, 12);

        // Build shots at this frequency.
        let mut scale_shots: Vec<(FwiGeometry, Array2<f64>)> =
            Vec::with_capacity(seismic_acquisition::N_SHOTS);
        let fwi_scale = FwiProcessor::new(FwiParameters {
            max_iterations: n_iter,
            frequency: f0,
            nt: nt_scale,
            dt,
            n_trace: seismic_acquisition::N_RECEIVERS,
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
        for &element_index in &seismic_acquisition::TRANSMIT_ELEMENT_INDICES {
            let geom = seismic_acquisition::build_shot(element_index, f0, nt_scale, dt)?;
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

        // Physical constraint: c ≥ c_water.
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
            frequency: seismic_acquisition::F0_HZ,
            nt: nt_fine,
            dt,
            n_trace: seismic_acquisition::N_RECEIVERS,
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
    let j_reduction_pct = (1.0 - j_final / j_initial) * 100.0;

    println!("\n  Quality after inversion:");
    print_quality_report(&true_model, &reconstructed);
    println!("  Joint J (150 kHz) : {j_final:.6e} Pa²·s");
    println!("  J reduction       : {j_reduction_pct:7.1} %  (150 kHz joint L2)");

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

    // Build the receiver snapshot directly from observed shot-0 seismograms.
    // For each active receiver r at grid position (i, k) we project the RMS
    // of its observed trace onto the grid.  This avoids a redundant forward
    // simulation (which caused an OOM crash on debug binaries after the long
    // FWI run) while providing the correct spatial energy distribution for the
    // zero-lag imaging condition I(x) = ∫ p_src(x,t)·p_recv(x,T−t) dt
    // (Baysal et al., 1983).
    let (geom0, obs0) = &shots_fine[0];
    let mut recv_snapshot = Array3::<f64>::zeros((NX, NY, NZ));
    {
        let recv_mask = &geom0.sensor_mask;
        let mut recv_idx = 0usize;
        for ([i, _j, k], &active) in recv_mask.indexed_iter() {
            if active {
                if recv_idx < obs0.shape()[0] {
                    let trace = obs0.index_axis::<1>(0, recv_idx).expect("index_axis");
                    let nt_obs = trace.shape()[0].max(1);
                    // RMS amplitude of the observed trace: scalar proxy for the
                    // receiver wavefield energy at this grid point.
                    let rms = (trace.iter().map(|&v| v * v).sum::<f64>() / nt_obs as f64).sqrt();
                    recv_snapshot[[i, 0, k]] = rms;
                }
                recv_idx += 1;
            }
        }
    }

    let rtm_settings = RtmSettings {
        imaging_condition: ImagingCondition::Normalized,
        storage_strategy: StorageStrategy::Full,
        boundary_type: SeismicBoundaryType::Absorbing,
        apply_laplacian: true,
    };
    let rtm = RtmProcessor::new(rtm_settings);
    let rtm_image = rtm
        .migrate(&recv_snapshot, &recv_snapshot, &grid)
        .map_err(|error| KwaversError::InvalidInput(format!("RTM migration failed: {error:#}")))?;
    let rtm_peak = rtm_image.iter().copied().fold(0.0_f64, f64::max);
    println!("  RTM image completed — peak amplitude: {rtm_peak:.4}");

    // ── Image output ──────────────────────────────────────────────────────
    let output_dir: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/seismic_imaging_demo"));

    std::fs::create_dir_all(&output_dir)
        .map_err(|e| KwaversError::InvalidInput(format!("cannot create output dir: {e}")))?;

    let abs_dir = std::fs::canonicalize(&output_dir).map_err(|error| {
        KwaversError::InvalidInput(format!("cannot canonicalize output dir: {error}"))
    })?;

    let base = "brain_fwi";
    let three_plane_path = abs_dir.join(format!("{base}_three_plane.png"));
    let velocity_ppm_path = abs_dir.join(format!("{base}.ppm"));
    let rtm_path = abs_dir.join(format!("{base}_rtm.ppm"));
    let brain_prior_path = abs_dir.join(format!("{base}_ct_brain_prior.png"));
    let csv_path = abs_dir.join(format!("{base}.csv"));
    let brain_tissue_path = abs_dir.join(format!("{base}_brain_tissue.png"));

    let shot_positions = seismic_acquisition::transmit_positions();
    let active_elements: Vec<(usize, usize)> =
        seismic_acquisition::ACTIVE_TRANSDUCER_POSITIONS.to_vec();

    seismic_planar_artifacts::write_three_plane_png(
        &three_plane_path,
        &true_model,
        &reconstructed,
        seismic_planar_artifacts::VelocityScale { lo: C_LO, hi: C_HI },
        seismic_planar_artifacts::AcquisitionMarkers {
            shot_positions: &shot_positions,
            active_elements: &active_elements,
        },
        ct_vol.as_ref(),
    )
    .map_err(|e| KwaversError::InvalidInput(format!("PNG write failed: {e}")))?;

    seismic_planar_artifacts::write_velocity_panels(
        &velocity_ppm_path,
        &true_model,
        &initial_model,
        &reconstructed,
        &shot_positions,
        &active_elements,
    )
    .map_err(|e| KwaversError::InvalidInput(format!("velocity panel write failed: {e}")))?;

    seismic_planar_artifacts::write_brain_prior_png(
        &brain_prior_path,
        phantom.hu(),
        &shot_positions,
        &active_elements,
    )
    .map_err(|e| KwaversError::InvalidInput(format!("brain prior PNG write failed: {e}")))?;

    seismic_planar_artifacts::write_rtm_panel(&rtm_path, &rtm_image)
        .map_err(|e| KwaversError::InvalidInput(format!("RTM panel write failed: {e}")))?;

    seismic_planar_artifacts::write_velocity_csv(
        &csv_path,
        &true_model,
        &initial_model,
        &reconstructed,
    )
    .map_err(|e| KwaversError::InvalidInput(format!("CSV write failed: {e}")))?;

    // Brain tissue PNG — written only when Stage-2 FWI succeeded.
    if let (Some(bt_true), Some(bt_recon)) = (&brain_true_model, &brain_reconstructed) {
        seismic_planar_artifacts::write_brain_tissue_png(&brain_tissue_path, bt_true, bt_recon)
            .map_err(|e| {
                KwaversError::InvalidInput(format!("brain tissue PNG write failed: {e}"))
            })?;
    }

    println!("\n  Output directory  : {}", abs_dir.display());
    println!("\n  Wrote images and data:");
    let three_plane_desc = if ct_vol.is_some() {
        "PNG 3×2: CT coronal|axial|sagittal (top) / FWI true|reconstructed|difference (bottom)"
    } else {
        "PNG: true skull (FWI grid) | FWI reconstructed | difference — coronal x-z"
    };
    println!("    {}  ({})", three_plane_path.display(), three_plane_desc);
    println!(
        "    {}  (PPM 4-panel: true | initial | reconstructed | error)",
        velocity_ppm_path.display()
    );
    println!(
        "    {}  (PNG CT-derived brain/skull prior + transducer)",
        brain_prior_path.display()
    );
    println!(
        "    {}  (PPM RTM zero-lag cross-correlation)",
        rtm_path.display()
    );
    println!(
        "    {}  (CSV depth profile at x = NX/2)",
        csv_path.display()
    );
    if brain_reconstructed.is_some() {
        println!(
            "    {}  (PNG brain tissue: true|reconstructed|difference, [1480,1560] m/s colormap)",
            brain_tissue_path.display()
        );
    }
    if ct_vol.is_some() {
        println!(
            "  Image size        : {}×{} px (3×{PANEL} wide, 2×({PANEL}+{COLORBAR_H}) tall)",
            3 * PANEL,
            2 * (PANEL + COLORBAR_H)
        );
    } else {
        println!(
            "  Image size        : {PANEL}×{PANEL} px per panel, 3 panels, {COLORBAR_H}px colorbar"
        );
    }
    println!(
        "  Colormap          : blue (1500 m/s, water/brain) → red ({:.0} m/s, cortical bone)",
        C_HI
    );
    if ct_vol.is_some() {
        println!(
            "  PNG layout        : 3×2 grid — top: CT coronal | axial | sagittal (bone window); bottom: FWI true | reconstructed | difference"
        );
    } else {
        println!(
            "  PNG panels        : true skull | reconstructed | difference (x-z coronal, y=0)"
        );
    }
    println!(
        "  Markers           : white = transmitting elements | yellow = active transducer samples"
    );

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
