//! Separated nonlinear 3-D theranostic simulation and inversion.
//!
//! This module is intentionally separate from the reduced Born/linear RTM
//! workflow. It solves a nonlinear Westervelt forward problem on a CT-derived
//! 3-D volume, performs discrete-adjoint FWI for sound-speed perturbation, and
//! reconstructs a Rayleigh-Plesset cavitation source from passive same-aperture
//! receiver data.

mod absorption;
mod adjoint;
mod aperture;
mod aperture_bowl;
mod cavitation;
mod checkpoint;
mod encoding;
mod forward;
mod grid;
mod metrics;
mod optimization;
mod regularization;
mod steering;
mod stencil;
mod types;
pub(crate) mod volume;
mod westervelt;

use ndarray::Array3;

use crate::core::error::KwaversResult;

pub use types::{
    Nonlinear3dConfig, Nonlinear3dResult, VolumeReconstructionMetrics,
    THERANOSTIC_CAVITATION_INVERSE_MODEL, THERANOSTIC_NONLINEAR_3D_MODEL,
    THERANOSTIC_NONLINEAR_3D_PROPAGATOR,
};

use super::AnatomyKind;
use aperture::build_aperture;
use cavitation::run_cavitation_inverse;
use metrics::{fused_score, metrics_from_score};
use steering::calibrate_electronic_steering;
use volume::prepare_volume;
use westervelt::run_fwi;

/// The full-resolution `ct_hu` and `label_volume` arrays are consumed
/// (moved in) so the caller cannot inadvertently hold hundreds of MB of
/// CT data alive across the memory-intensive Westervelt FWI loop.
/// Both arrays are dropped immediately after `prepare_volume` produces
/// the resampled grid-size³ volume.
pub fn run_theranostic_nonlinear_3d(
    anatomy: AnatomyKind,
    ct_hu: Array3<f64>,
    label_volume: Option<Array3<i16>>,
    spacing_mm: [f64; 3],
    config: &Nonlinear3dConfig,
    target_fraction_xyz: Option<[f64; 3]>,
) -> KwaversResult<Nonlinear3dResult> {
    config.validate()?;
    let volume = prepare_volume(
        anatomy,
        &ct_hu,
        label_volume.as_ref(),
        spacing_mm,
        config,
        target_fraction_xyz,
    )?;
    // Release the full-resolution source images before the FWI loop.
    // On real CT volumes (e.g. 512×512×300 brain scan ≈ 600 MB at f64)
    // holding these alive across all forward/adjoint passes exhausts RAM.
    drop(ct_hu);
    drop(label_volume);
    let aperture = build_aperture(&volume, config)?;
    let steering = calibrate_electronic_steering(&volume, &aperture, config);
    let fwi = run_fwi(&volume, &steering.aperture, config);
    let cavitation =
        run_cavitation_inverse(&volume, &steering.aperture, &fwi.peak_pressure_pa, config);
    let wavelength_min_m = minimum_wavelength_m(&volume.true_sound_speed_m_s, config.frequency_hz);
    let points_per_wavelength_min = wavelength_min_m / volume.spacing_m;
    let resolution_meets_min_ppw = points_per_wavelength_min >= config.min_points_per_wavelength;
    let fusion_score = fused_score(
        &fwi.multiparameter_fwi_score,
        &cavitation.reconstructed_density,
        &volume.body_mask,
    );
    let fusion_score_vec = fusion_score.iter().copied().collect::<Vec<_>>();
    let target_vec = volume.target_mask.iter().copied().collect::<Vec<_>>();
    let body_vec = volume.body_mask.iter().copied().collect::<Vec<_>>();
    let fusion_metrics = metrics_from_score(&fusion_score_vec, &target_vec, &body_vec);
    let active_voxels = volume.body_mask.iter().filter(|active| **active).count();
    Ok(Nonlinear3dResult {
        ct_hu: volume.ct_hu,
        label: volume.label,
        body_mask: volume.body_mask,
        target_mask: volume.target_mask,
        inversion_mask: volume.inversion_mask,
        background_sound_speed_m_s: volume.background_sound_speed_m_s,
        true_sound_speed_m_s: volume.true_sound_speed_m_s,
        reconstructed_sound_speed_m_s: fwi.reconstructed_sound_speed_m_s,
        reconstructed_delta_c_m_s: fwi.reconstructed_delta_c_m_s,
        background_beta: volume.background_beta,
        true_beta: volume.true_beta,
        reconstructed_beta: fwi.reconstructed_beta,
        reconstructed_delta_beta: fwi.reconstructed_delta_beta,
        multiparameter_fwi_score: fwi.multiparameter_fwi_score,
        nonlinear_fusion_score: fusion_score,
        westervelt_peak_pressure_pa: fwi.peak_pressure_pa,
        cavitation_source_density: cavitation.source_density,
        reconstructed_cavitation_density: cavitation.reconstructed_density,
        fwi_objective_history: fwi.objective_history,
        fwi_iteration_diagnostics: fwi.iteration_diagnostics,
        cavitation_objective_history: cavitation.objective_history,
        therapy_points_m: steering.aperture.therapy_points_m,
        receiver_points_m: steering.aperture.receiver_points_m,
        spacing_m: volume.spacing_m,
        source_dimensions: volume.source_dimensions,
        source_spacing_m: volume.source_spacing_m,
        crop_bounds_index: volume.crop_bounds_index,
        treatment_window_radius_m: config.treatment_window_radius_m,
        wavelength_min_m,
        points_per_wavelength_min,
        resolution_meets_min_ppw,
        dt_s: fwi.dt_s,
        time_steps: fwi.time_steps,
        source_scale: fwi.source_scale,
        source_plan_metrics: fwi.source_plan_metrics,
        electronic_steering_metrics: steering.metrics,
        active_voxels,
        fwi_metrics: fwi.metrics,
        cavitation_metrics: cavitation.metrics,
        fusion_metrics,
        aperture_model: steering.aperture.model_name,
        model_family: THERANOSTIC_NONLINEAR_3D_MODEL,
        propagator_model: THERANOSTIC_NONLINEAR_3D_PROPAGATOR,
        cavitation_inverse_model: THERANOSTIC_CAVITATION_INVERSE_MODEL,
        is_full_wave_inversion: true,
        uses_nonlinear_wave_propagation: true,
        uses_rayleigh_plesset: true,
    })
}

fn minimum_wavelength_m(sound_speed_m_s: &Array3<f64>, frequency_hz: f64) -> f64 {
    let min_speed = sound_speed_m_s
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .fold(f64::INFINITY, f64::min);
    if min_speed.is_finite() && frequency_hz > 0.0 {
        min_speed / frequency_hz
    } else {
        0.0
    }
}

#[cfg(test)]
mod checkpoint_tests;
#[cfg(test)]
mod tests;
