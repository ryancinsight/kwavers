use leto::{
    Array1,
    Array3,
};

use kwavers_core::error::{KwaversError, KwaversResult};

use super::super::geometry::focused_cap_positions;
use super::super::pressure::{
    pressure_peak_and_scale, rayleigh_pressure_field_unscaled, scale_pressure_field,
};
use super::super::skull_ray::{skull_path_phase_correction, SkullPathPhaseCorrectionInput};
use super::metrics::evaluate_pressure_field;
use super::placement::select_skull_aware_placement;
use super::types::{SkullAdaptiveBenchmarkConfig, SkullAdaptiveBenchmarkResult};

pub fn run_skull_adaptive_transcranial_benchmark(
    ct_hu: &Array3<f64>,
    skull_mask: &Array3<bool>,
    brain_mask: &Array3<bool>,
    spacing_m: [f64; 3],
    target_index: [usize; 3],
    config: &SkullAdaptiveBenchmarkConfig,
) -> KwaversResult<SkullAdaptiveBenchmarkResult> {
    validate_inputs(
        ct_hu,
        skull_mask,
        brain_mask,
        spacing_m,
        target_index,
        config,
    )?;

    let element_positions_m = focused_cap_positions(
        config.fus.element_count,
        config.fus.radius_m,
        config.fus.cap_min_polar_rad,
        config.fus.cap_max_polar_rad,
    )?;
    let (phases_rad, delays_s, skull_lengths_m, amplitude_weights) =
        skull_path_phase_correction(SkullPathPhaseCorrectionInput {
            ct_hu,
            spacing_m,
            target_index_xyz: target_index,
            element_positions: &element_positions_m,
            frequency_hz: config.fus.frequency_hz,
            brain_c: config.fus.brain_c,
            skull_c: config.fus.skull_c,
            skull_mask,
            samples_per_ray: config.fus.samples_per_ray,
        })?;
    let placement = select_skull_aware_placement(
        element_positions_m,
        &skull_lengths_m,
        &amplitude_weights,
        config,
    )?;

    let shape = ct_hu.dim();
    let dims = [shape.0, shape.1, shape.2];
    let reference_unscaled = rayleigh_pressure_field_unscaled(
        &placement.element_positions_m,
        &phases_rad,
        &amplitude_weights,
        &placement.active_elements,
        dims,
        spacing_m,
        target_index,
        config.fus.frequency_hz,
        config.fus.brain_c,
        config.fus.chunk_size,
    )?;
    let (_, reference_scale) =
        pressure_peak_and_scale(&reference_unscaled, config.fus.target_peak_pa)?;
    let reference_pressure_pa = scale_pressure_field(reference_unscaled, reference_scale);

    let zero_phases = Array1::<f64>::zeros(config.fus.element_count);
    let baseline_unscaled = rayleigh_pressure_field_unscaled(
        &placement.element_positions_m,
        &zero_phases,
        &amplitude_weights,
        &placement.active_elements,
        dims,
        spacing_m,
        target_index,
        config.fus.frequency_hz,
        config.fus.brain_c,
        config.fus.chunk_size,
    )?;
    let baseline_pressure_pa = scale_pressure_field(baseline_unscaled, reference_scale);
    let metrics =
        evaluate_pressure_field(&reference_pressure_pa, &baseline_pressure_pa, spacing_m)?;

    Ok(SkullAdaptiveBenchmarkResult {
        reference_pressure_pa,
        baseline_pressure_pa,
        metrics,
        placement,
        phases_rad,
        delays_s,
        skull_lengths_m,
        amplitude_weights,
        focus_index: target_index,
        spacing_m,
        frequency_hz: config.fus.frequency_hz,
        target_peak_pa: config.fus.target_peak_pa,
    })
}

fn validate_inputs(
    ct_hu: &Array3<f64>,
    skull_mask: &Array3<bool>,
    brain_mask: &Array3<bool>,
    spacing_m: [f64; 3],
    target_index: [usize; 3],
    config: &SkullAdaptiveBenchmarkConfig,
) -> KwaversResult<()> {
    let shape = ct_hu.dim();
    if shape != skull_mask.dim() || shape != brain_mask.dim() {
        return Err(KwaversError::InvalidInput(
            "ct_hu, skull_mask, and brain_mask must have identical shapes".to_owned(),
        ));
    }
    if shape.0 < 8 || shape.1 < 8 || shape.2 < 8 {
        return Err(KwaversError::InvalidInput(
            "benchmark CT volume must be at least 8 x 8 x 8".to_owned(),
        ));
    }
    if target_index[0] >= shape.0 || target_index[1] >= shape.1 || target_index[2] >= shape.2 {
        return Err(KwaversError::InvalidInput(
            "target_index is outside the CT volume".to_owned(),
        ));
    }
    if !brain_mask.iter().any(|value| *value) {
        return Err(KwaversError::InvalidInput(
            "brain_mask must contain at least one voxel".to_owned(),
        ));
    }
    if !skull_mask.iter().any(|value| *value) {
        return Err(KwaversError::InvalidInput(
            "skull_mask must contain at least one voxel".to_owned(),
        ));
    }
    if spacing_m
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(KwaversError::InvalidInput(
            "spacing_m values must be positive and finite".to_owned(),
        ));
    }
    if config.fus.element_count < 16 {
        return Err(KwaversError::InvalidInput(
            "benchmark element_count must be at least 16".to_owned(),
        ));
    }
    if config.fus.samples_per_ray < 2 {
        return Err(KwaversError::InvalidInput(
            "samples_per_ray must be at least 2".to_owned(),
        ));
    }
    Ok(())
}
