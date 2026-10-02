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

#![expect(
    clippy::print_stderr,
    reason = "The example's console report is its user-visible output contract"
)]

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

use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
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
use std::io::Write;

#[path = "seismic_imaging/parameters.rs"]
mod seismic_parameters;
use seismic_parameters::{
    BONE_VELOCITY_THRESHOLD, BRAIN_C_MAX, BRAIN_C_MIN, COLORBAR_H, C_CSF, C_GRAY, C_HI, C_LO,
    C_WHITE, DX, F0_BRAIN_HZ, HU_BRAIN, HU_CORTICAL_IN, HU_CORTICAL_OUT, HU_DIPLOE, HU_SCALP,
    HU_WATER, MNI_INNER_SKULL_RADIUS_MM, NX, NY, NZ, N_BRAIN_ITER, PANEL, R_BRAIN, R_DIPLOE,
    R_HEAD, R_SKULL_IN, R_SKULL_OUT, STEP_SIZE, STEP_SIZE_BRAIN,
    TRANSCRANIAL_FOCUSED_BOWL_ELEMENT_COUNT,
};

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

    let _ = writeln!(
        std::io::stdout().lock(),
        "╔══════════════════════════════════════════════════════════╗"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "║   Transcranial Ultrasound FWI — Brain Reconstruction     ║"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "╚══════════════════════════════════════════════════════════╝\n"
    );

    // ── 1. Skull phantom ──────────────────────────────────────────────────
    let _ = writeln!(
        std::io::stdout().lock(),
        "[ 1 / 7 ]  Building skull phantom …"
    );
    let input_mode = SeismicInputMode::from_env("KWAVERS_SEISMIC_INPUT_MODE")
        .map_err(KwaversError::InvalidInput)?;
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Input mode       : {input_mode:?}"
    );
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

    let _ = writeln!(
        std::io::stdout().lock(),
        "  Grid            : {NX}×{NY}×{NZ} voxels @ {:.0} mm",
        DX * 1e3
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Domain          : {:.0}×{:.0} mm",
        NX as f64 * DX * 1e3,
        NZ as f64 * DX * 1e3
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  HU range        : [{hu_min:.0}, {hu_max:.0}]"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Sound-speed     : [{c_min:.0}, {c_max:.0}] m/s"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Head radius     : {:.0} mm  (R_HEAD = {R_HEAD} voxels)",
        R_HEAD * DX * 1e3
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Skull thickness : ~{:.0} mm  (outer cortical → inner cortical)",
        (R_SKULL_OUT - R_SKULL_IN) * DX * 1e3
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Brain radius    : {:.0} mm",
        R_BRAIN * DX * 1e3
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Layers          : water coupling / scalp / cortical bone / diploe / brain"
    );

    // ── 2. Grid ───────────────────────────────────────────────────────────
    let _ = writeln!(
        std::io::stdout().lock(),
        "\n[ 2 / 7 ]  Constructing computational grid …"
    );
    let grid = Grid::new(NX, NY, NZ, DX, DX, DX)?;
    let _ = writeln!(std::io::stdout().lock(), "  Grid OK");

    // ── 3. Multi-scale FWI parameters ────────────────────────────────────
    let _ = writeln!(
        std::io::stdout().lock(),
        "\n[ 3 / 7 ]  Configuring multi-scale FWI …"
    );

    // CFL-stable timestep: dt ≤ 0.3 × dx / (c_max × √3).
    // Fixed across all scales — determined by maximum velocity, not frequency.
    let dt = 0.3 * DX / (c_max * 3.0_f64.sqrt());

    // Full domain transit time at water speed (same for all scales).
    let t_transit = (NX as f64 * DX) / SOUND_SPEED_WATER_SIM;

    let scales = seismic_planar_schedule::configure(dt, t_transit);

    // ── 4. Full-ring acquisition geometry ─────────────────────────────────
    let _ = writeln!(
        std::io::stdout().lock(),
        "\n[ 4 / 7 ]  Building full-ring acquisition geometry …"
    );
    let _ = writeln!(std::io::stdout().lock(),
        "  Full aperture    : {TRANSCRANIAL_FOCUSED_BOWL_ELEMENT_COUNT} elements, 650 kHz design authority"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  FWI section      : {} active full-ring samples",
        seismic_acquisition::FWI_ACTIVE_ELEMENTS
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Transmits        : {} shots; receivers/shot = {} on same full ring",
        seismic_acquisition::N_SHOTS,
        seismic_acquisition::N_RECEIVERS
    );
    for (s, &element_index) in seismic_acquisition::TRANSMIT_ELEMENT_INDICES
        .iter()
        .enumerate()
    {
        let (ix, iz) = seismic_acquisition::ACTIVE_TRANSDUCER_POSITIONS[element_index];
        let _ = writeln!(
            std::io::stdout().lock(),
            "  Shot {:1}: (x={:2}, y=0, z={:2}) = ({:.1} mm, {:.1} mm)",
            s,
            ix,
            iz,
            ix as f64 * DX * 1e3,
            iz as f64 * DX * 1e3
        );
    }

    // ── 5. Multi-scale FWI ────────────────────────────────────────────────
    let _ = writeln!(
        std::io::stdout().lock(),
        "\n[ 5 / 7 ]  Running multi-scale transcranial FWI …"
    );

    let inversion =
        seismic_planar_inversion::run_skull_inversion(&phantom, &grid, dt, t_transit, scales)?;
    let true_model = inversion.true_model;
    let initial_model = inversion.initial_model;
    let reconstructed = inversion.reconstructed;
    let shots_fine = inversion.shots_fine;

    // ── 6. Stage-2 brain tissue FWI (Guasch 2020 style) ─────────────────
    let brain_prior =
        BrainPriorMode::from_env("KWAVERS_BRAIN_PRIOR").map_err(KwaversError::InvalidInput)?;
    let _ = writeln!(
        std::io::stdout().lock(),
        "\n[ 6 / 7 ]  Stage-2 brain tissue FWI ({brain_prior:?}) …"
    );

    let brain_result = seismic_brain_inversion::run_brain_fwi(&phantom, &brain_prior, &grid, dt)?;
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Quality (brain voxels only, r < R_SKULL_IN):"
    );
    print_quality_report_brain(&brain_result.true_model, &brain_result.reconstructed);
    let brain_true_model = Some(brain_result.true_model);
    let brain_reconstructed = Some(brain_result.reconstructed);

    // ── 7. RTM — zero-lag cross-correlation imaging ───────────────────────
    let _ = writeln!(
        std::io::stdout().lock(),
        "\n[ 7 / 7 ]  Reverse Time Migration (reflectivity image) …"
    );

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
    let _ = writeln!(
        std::io::stdout().lock(),
        "\n═══════════════════════════════════════════════════════════"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Reconstructed velocity range: [{:.0}, {:.0}] m/s",
        reconstructed.iter().copied().fold(f64::INFINITY, f64::min),
        reconstructed
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  True velocity range         : [{c_min:.0}, {c_max:.0}] m/s"
    );
    let _ = writeln!(std::io::stdout().lock());
    let _ = writeln!(std::io::stdout().lock(), "  Physics verified against:");
    let _ = writeln!(
        std::io::stdout().lock(),
        "    Aubry (2003)              — HU → c, ρ bone-volume-fraction model"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "    Marsac (2017)             — skull acoustic properties + geometry"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "    Guasch (2020)             — transcranial FWI methodology"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "    Ricker (1953)             — source wavelet"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "    Tarantola (1984)          — adjoint-state FWI gradient"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "    Virieux & Operto (2009)   — FWI objective and chain rule"
    );
    let _ = writeln!(std::io::stdout().lock());
    let _ = writeln!(std::io::stdout().lock(), "  To use an explicit CT input:");
    let _ = writeln!(
        std::io::stdout().lock(),
        "    set KWAVERS_SEISMIC_INPUT_MODE=ct:path\\to\\ct_dicom_or_nifti"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "    cargo run --example seismic_imaging_demo"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "═══════════════════════════════════════════════════════════"
    );

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
