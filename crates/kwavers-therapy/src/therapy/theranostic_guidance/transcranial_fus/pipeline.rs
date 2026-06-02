use ndarray::{Array1, Array2, Array3};

use kwavers_core::error::{KwaversError, KwaversResult};

use super::bbb::bbb_opening_dose;
use super::geometry::focused_cap_positions;
use super::observables::acoustic_fus_observables;
use super::pressure::rayleigh_pressure_field;
use super::skull_ray::{skull_path_phase_correction, SkullPathPhaseCorrectionInput};
use super::subspot::gbm_subspot_raster;
use super::types::{TranscranialFusPlan, TranscranialFusPlanConfig};

/// Execute the complete transcranial FUS therapy planning pipeline.
///
/// Inputs are CT volume, tissue masks, voxel spacing, focus voxel index, and
/// planning configuration. Returns [`TranscranialFusPlan`] containing all
/// computed fields and planning outputs.
pub fn run_transcranial_fus_planning(
    ct_hu: &Array3<f64>,
    skull_mask: &Array3<bool>,
    _brain_mask: &Array3<bool>,
    tumor_mask: &Array3<bool>,
    spacing_m: [f64; 3],
    target_index: [usize; 3],
    config: &TranscranialFusPlanConfig,
) -> KwaversResult<TranscranialFusPlan> {
    if config.element_count < 1 {
        return Err(KwaversError::InvalidInput(
            "element_count must be at least 1".to_owned(),
        ));
    }
    if spacing_m.iter().any(|&s| !s.is_finite() || s <= 0.0) {
        return Err(KwaversError::InvalidInput(
            "all spacing_m values must be positive and finite".to_owned(),
        ));
    }
    if !config.duty_cycle.is_finite() || config.duty_cycle <= 0.0 || config.duty_cycle > 1.0 {
        return Err(KwaversError::InvalidInput(
            "duty_cycle must be in (0, 1]".to_owned(),
        ));
    }

    let element_positions = focused_cap_positions(
        config.element_count,
        config.radius_m,
        config.cap_min_polar_rad,
        config.cap_max_polar_rad,
    )?;

    let (phases_rad, delays_s, skull_lengths_m, amplitude_weights) =
        skull_path_phase_correction(SkullPathPhaseCorrectionInput {
            ct_hu,
            spacing_m,
            target_index_xyz: target_index,
            element_positions: &element_positions,
            frequency_hz: config.frequency_hz,
            brain_c: config.brain_c,
            skull_c: config.skull_c,
            skull_mask,
            samples_per_ray: config.samples_per_ray,
        })?;

    let active = Array1::from_elem(config.element_count, true);

    let shape = ct_hu.dim();
    let pressure_pa = rayleigh_pressure_field(
        &element_positions,
        &phases_rad,
        &amplitude_weights,
        &active,
        [shape.0, shape.1, shape.2],
        spacing_m,
        target_index,
        config.frequency_hz,
        config.brain_c,
        config.target_peak_pa,
        config.chunk_size,
    )?;

    let (intensity_w_m2, mechanical_index, cavitation_probability) = acoustic_fus_observables(
        &pressure_pa,
        config.frequency_hz,
        config.rho_brain,
        config.brain_c,
        config.inertial_mi_threshold,
    );

    let subspot_indices = if tumor_mask.iter().any(|&v| v) {
        gbm_subspot_raster(tumor_mask, spacing_m, config.pitch_m)?
    } else {
        Array2::from_shape_fn((1, 3), |(_, col)| target_index[col])
    };

    let (bbb_dose, bbb_permeability, bbb_stable_cavitation, bbb_inertial_risk) = bbb_opening_dose(
        tumor_mask,
        &subspot_indices,
        spacing_m,
        config.mechanical_index_bbb,
        config.sonication_s,
        config.duty_cycle,
        config.focal_radius_m,
        config.d50,
        config.hill_n,
    );

    Ok(TranscranialFusPlan {
        pressure_pa,
        intensity_w_m2,
        mechanical_index,
        cavitation_probability,
        phases_rad,
        delays_s,
        skull_lengths_m,
        amplitude_weights,
        element_positions_m: element_positions,
        subspot_indices,
        bbb_dose,
        bbb_permeability,
        bbb_stable_cavitation,
        bbb_inertial_risk,
        focus_index: target_index,
        element_count: config.element_count,
        frequency_hz: config.frequency_hz,
    })
}
