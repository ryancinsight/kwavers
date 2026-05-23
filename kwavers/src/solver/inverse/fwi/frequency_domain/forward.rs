//! Matrix-free frequency-domain forward model.

use super::cbs::{
    real_scattering_potential, sample_array_with_bli, solve_volume_field_with_operator,
    source_density_for_operator, CbsConfig, GreenOperatorKind, GridSpec,
};
use super::types::{Config, FREQUENCY_DOMAIN_FWI_SOLVER_MODEL};
use crate::core::error::{KwaversError, KwaversResult};
use crate::physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::{
    sound_speed_to_slowness, MultiRowRingArray,
};
use crate::solver::inverse::linear_born_inversion::ElementPosition;
use ndarray::{Array2, Array3};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Simulate complex receiver pressure for all cylindrical-wave transmits.
///
/// # Errors
/// Returns an error when volume, geometry, frequency, or config values violate
/// the frequency-domain FWI contract.
pub fn simulate_frequency_observation(
    sound_speed_m_s: &Array3<f64>,
    array: &MultiRowRingArray,
    frequency_hz: f64,
    config: &Config,
) -> KwaversResult<Array2<Complex64>> {
    let slowness = sound_speed_to_slowness(sound_speed_m_s)?;
    predict_frequency_rows(
        &slowness,
        array,
        frequency_hz,
        config,
        array.circumferential_elements(),
    )
}

pub(super) fn predict_frequency_rows(
    slowness_s_per_m: &Array3<f64>,
    array: &MultiRowRingArray,
    frequency_hz: f64,
    config: &Config,
    transmissions: usize,
) -> KwaversResult<Array2<Complex64>> {
    validate_forward_inputs(slowness_s_per_m, array, frequency_hz, config, transmissions)?;
    config.forward_operator.predict_receiver_rows(
        slowness_s_per_m,
        array,
        frequency_hz,
        config,
        transmissions,
    )
}

pub(super) fn predict_born_rows(
    slowness_s_per_m: &Array3<f64>,
    array: &MultiRowRingArray,
    frequency_hz: f64,
    config: &Config,
    transmissions: usize,
) -> KwaversResult<Array2<Complex64>> {
    let (nx, ny, nz) = slowness_s_per_m.dim();
    let cell_volume = config.spacing_m.powi(3);
    let omega = 2.0 * PI * frequency_hz;
    let reference_slowness = 1.0 / config.reference_sound_speed_m_s;
    let reference_wavenumber = omega * reference_slowness;
    let min_distance = 0.5 * config.spacing_m;
    let centers = voxel_centers((nx, ny, nz), config.spacing_m);
    let potential = real_scattering_potential(omega, slowness_s_per_m, reference_slowness)?;

    let mut output = Array2::zeros((transmissions, array.element_count()));
    for transmit in 0..transmissions {
        let sources = array.cylindrical_source(transmit);
        let incident = incident_field(&sources, &centers, reference_wavenumber, min_distance);
        for (receiver_index, &receiver) in array.elements().iter().enumerate() {
            let direct = sources
                .iter()
                .map(|&source| outgoing_green(source, receiver, reference_wavenumber, min_distance))
                .sum::<Complex64>();
            let mut scattered = Complex64::new(0.0, 0.0);
            for ((_, point), (&potential_value, &incident_value)) in
                centers.iter().zip(potential.iter().zip(incident.iter()))
            {
                let receiver_green =
                    outgoing_green(*point, receiver, reference_wavenumber, min_distance);
                scattered += incident_value * receiver_green * (potential_value * cell_volume);
            }
            output[[transmit, receiver_index]] = direct + scattered;
        }
    }

    Ok(output)
}

pub(super) fn predict_cbs_rows(
    slowness_s_per_m: &Array3<f64>,
    array: &MultiRowRingArray,
    frequency_hz: f64,
    config: &Config,
    transmissions: usize,
    cbs_config: CbsConfig,
    operator: GreenOperatorKind,
) -> KwaversResult<Array2<Complex64>> {
    let omega = 2.0 * PI * frequency_hz;
    let reference_slowness = 1.0 / config.reference_sound_speed_m_s;
    let reference_wavenumber = omega * reference_slowness;
    let grid = GridSpec::new(slowness_s_per_m.dim(), config.spacing_m)?;
    let potential = real_scattering_potential(omega, slowness_s_per_m, reference_slowness)?;
    let mut output = Array2::zeros((transmissions, array.element_count()));
    for transmit in 0..transmissions {
        let source_density =
            source_density_for_operator(grid, &array.cylindrical_source(transmit), operator)?;
        let solution = solve_volume_field_with_operator(
            grid,
            reference_wavenumber,
            &potential,
            &source_density,
            cbs_config,
            operator,
        )?;
        for (receiver_index, pressure) in sample_array_with_bli(grid, &solution.field, array)?
            .into_iter()
            .enumerate()
        {
            output[[transmit, receiver_index]] = pressure;
        }
    }
    Ok(output)
}

pub(super) fn incident_field(
    sources: &[ElementPosition],
    centers: &[(usize, ElementPosition)],
    wavenumber: f64,
    min_distance: f64,
) -> Vec<Complex64> {
    centers
        .iter()
        .map(|(_, point)| {
            sources
                .iter()
                .map(|&source| outgoing_green(source, *point, wavenumber, min_distance))
                .sum()
        })
        .collect()
}

pub(super) fn outgoing_green(
    source: ElementPosition,
    receiver: ElementPosition,
    wavenumber: f64,
    min_distance: f64,
) -> Complex64 {
    let dx = source.x_m - receiver.x_m;
    let dy = source.y_m - receiver.y_m;
    let dz = source.z_m - receiver.z_m;
    let distance = (dx * dx + dy * dy + dz * dz).sqrt().max(min_distance);
    Complex64::from_polar(1.0 / (4.0 * PI * distance), wavenumber * distance)
}

pub(super) fn voxel_centers(
    dimensions: (usize, usize, usize),
    spacing_m: f64,
) -> Vec<(usize, ElementPosition)> {
    let (nx, ny, nz) = dimensions;
    let cx = 0.5 * nx as f64;
    let cy = 0.5 * ny as f64;
    let cz = 0.5 * nz as f64;
    let mut centers = Vec::with_capacity(nx * ny * nz);
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                centers.push((
                    ix * ny * nz + iy * nz + iz,
                    ElementPosition {
                        x_m: (ix as f64 + 0.5 - cx) * spacing_m,
                        y_m: (iy as f64 + 0.5 - cy) * spacing_m,
                        z_m: (iz as f64 + 0.5 - cz) * spacing_m,
                    },
                ));
            }
        }
    }
    centers
}

pub(super) fn validate_forward_inputs(
    slowness_s_per_m: &Array3<f64>,
    array: &MultiRowRingArray,
    frequency_hz: f64,
    config: &Config,
    transmissions: usize,
) -> KwaversResult<()> {
    if slowness_s_per_m.is_empty() {
        return Err(KwaversError::InvalidInput(
            "FWI slowness volume must be nonempty".to_owned(),
        ));
    }
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "FWI frequency must be positive and finite, got {frequency_hz}"
        )));
    }
    if transmissions == 0 || transmissions > array.circumferential_elements() {
        return Err(KwaversError::InvalidInput(format!(
            "transmissions must be in 1..={}, got {transmissions}",
            array.circumferential_elements()
        )));
    }
    validate_config(config)?;
    config
        .forward_operator
        .validate_for_grid(GridSpec::new(slowness_s_per_m.dim(), config.spacing_m)?)?;
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
