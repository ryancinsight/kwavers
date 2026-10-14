//! Matrix-free frequency-domain forward model.

use super::acquisition::TransmissionAcquisition;
use super::born_geometry::BornGeometry;
use super::born_kernel::fields::FactoredProducts;
use super::born_kernel::{cell_volume, integrate, VOXELS_PER_BLOCK};
use super::cbs::{
    real_scattering_potential, real_scattering_potential_for_operator, sample_array_for_operator,
    solve_volume_field_with_operator, source_density_for_operator, CbsConfig, GreenOperatorKind,
    GridSpec,
};
use super::types::{Config, FREQUENCY_DOMAIN_FWI_SOLVER_MODEL};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::fft::Complex64;
use kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::sound_speed_to_slowness;
use leto::{Array2, Array3};

/// Simulate complex receiver pressure for all cylindrical-wave transmits.
///
/// # Errors
/// Returns an error when volume, geometry, frequency, or config values violate
/// the frequency-domain FWI contract.
pub fn simulate_frequency_observation(
    sound_speed_m_s: &Array3<f64>,
    acquisition: &dyn TransmissionAcquisition,
    frequency_hz: f64,
    config: &Config,
) -> KwaversResult<Array2<Complex64>> {
    let slowness = sound_speed_to_slowness(sound_speed_m_s)?;
    predict_frequency_rows(
        &slowness,
        acquisition,
        frequency_hz,
        config,
        acquisition.transmission_count(),
    )
}

pub(super) fn predict_frequency_rows(
    slowness_s_per_m: &Array3<f64>,
    acquisition: &dyn TransmissionAcquisition,
    frequency_hz: f64,
    config: &Config,
    transmissions: usize,
) -> KwaversResult<Array2<Complex64>> {
    validate_forward_inputs(
        slowness_s_per_m,
        acquisition,
        frequency_hz,
        config,
        transmissions,
    )?;
    config.forward_operator.predict_receiver_rows(
        slowness_s_per_m,
        acquisition,
        frequency_hz,
        config,
        transmissions,
    )
}

pub(super) fn predict_born_rows(
    slowness_s_per_m: &Array3<f64>,
    acquisition: &dyn TransmissionAcquisition,
    frequency_hz: f64,
    config: &Config,
    transmissions: usize,
) -> KwaversResult<Array2<Complex64>> {
    let shape = slowness_s_per_m.shape();
    let cell_volume = cell_volume(config.spacing_m);
    let omega = TWO_PI * frequency_hz;
    let reference_slowness = 1.0 / config.reference_sound_speed_m_s;
    let potential_volume = real_scattering_potential(omega, slowness_s_per_m, reference_slowness)?
        .into_iter()
        .map(|value| value * cell_volume)
        .collect::<Vec<_>>();
    let geometry = BornGeometry::new(shape, frequency_hz, acquisition, config, transmissions);

    let mut output = Array2::zeros([transmissions, acquisition.receiver_count()]);
    for (transmit, row) in geometry.transmits.iter().enumerate() {
        for (receiver_index, &field) in row.receivers.iter().enumerate() {
            let scattered = integrate::<_, _, VOXELS_PER_BLOCK>(
                &FactoredProducts::new(&row.incident, &geometry.fields[field]),
                &potential_volume,
            );
            output[[transmit, receiver_index]] = row.direct[receiver_index] + scattered;
        }
    }

    Ok(output)
}

pub(super) fn predict_cbs_rows(
    slowness_s_per_m: &Array3<f64>,
    acquisition: &dyn TransmissionAcquisition,
    frequency_hz: f64,
    config: &Config,
    transmissions: usize,
    cbs_config: CbsConfig,
    operator: GreenOperatorKind,
) -> KwaversResult<Array2<Complex64>> {
    let omega = TWO_PI * frequency_hz;
    let reference_slowness = 1.0 / config.reference_sound_speed_m_s;
    let reference_wavenumber = omega * reference_slowness;
    let shape = slowness_s_per_m.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
    let grid = GridSpec::new((nx, ny, nz), config.spacing_m)?;
    let potential = real_scattering_potential_for_operator(
        omega,
        slowness_s_per_m,
        reference_slowness,
        operator,
    )?;
    let mut output = Array2::zeros([transmissions, acquisition.receiver_count()]);
    for transmit in 0..transmissions {
        let source_density = source_density_for_operator(
            grid,
            acquisition.sources(transmit),
            reference_wavenumber,
            operator,
        )?;
        let solution = solve_volume_field_with_operator(
            grid,
            reference_wavenumber,
            &potential,
            &source_density,
            cbs_config,
            operator,
        )?;
        for (receiver_index, pressure) in sample_array_for_operator(
            grid,
            &solution.field,
            acquisition.receivers(transmit),
            operator,
        )?
        .into_iter()
        .enumerate()
        {
            output[[transmit, receiver_index]] = pressure;
        }
    }
    Ok(output)
}

pub(super) fn validate_forward_inputs(
    slowness_s_per_m: &Array3<f64>,
    acquisition: &dyn TransmissionAcquisition,
    frequency_hz: f64,
    config: &Config,
    transmissions: usize,
) -> KwaversResult<()> {
    if slowness_s_per_m.shape().contains(&0) {
        return Err(KwaversError::InvalidInput(
            "FWI slowness volume must be nonempty".to_owned(),
        ));
    }
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "FWI frequency must be positive and finite, got {frequency_hz}"
        )));
    }
    if transmissions == 0 || transmissions > acquisition.transmission_count() {
        return Err(KwaversError::InvalidInput(format!(
            "transmissions must be in 1..={}, got {transmissions}",
            acquisition.transmission_count()
        )));
    }
    validate_config(config)?;
    config.forward_operator.validate_for_grid({
        let shape = slowness_s_per_m.shape();
        let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
        GridSpec::new((nx, ny, nz), config.spacing_m)?
    })?;
    for &slowness in slowness_s_per_m.iter() {
        if !slowness.is_finite() || slowness <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "slowness must be positive and finite for {FREQUENCY_DOMAIN_FWI_SOLVER_MODEL}, got {slowness}"
            )));
        }
    }
    Ok(())
}

pub(super) fn validate_config(config: &Config) -> KwaversResult<()> {
    if !config.reference_sound_speed_m_s.is_finite() || config.reference_sound_speed_m_s <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "reference sound speed must be positive and finite, got {}",
            config.reference_sound_speed_m_s
        )));
    }
    if !config.spacing_m.is_finite() || config.spacing_m <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "voxel spacing must be positive and finite, got {}",
            config.spacing_m
        )));
    }
    if !config.initial_step_s_per_m.is_finite() || config.initial_step_s_per_m <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "initial slowness step must be positive and finite, got {}",
            config.initial_step_s_per_m
        )));
    }
    if config.min_sound_speed_m_s <= 0.0
        || config.max_sound_speed_m_s <= config.min_sound_speed_m_s
        || !config.min_sound_speed_m_s.is_finite()
        || !config.max_sound_speed_m_s.is_finite()
    {
        return Err(KwaversError::InvalidInput(format!(
            "invalid sound-speed bounds [{}, {}]",
            config.min_sound_speed_m_s, config.max_sound_speed_m_s
        )));
    }
    if !config.tikhonov_weight.is_finite() || config.tikhonov_weight < 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "Tikhonov weight must be finite and nonnegative, got {}",
            config.tikhonov_weight
        )));
    }
    config.forward_operator.validate()?;
    Ok(())
}
