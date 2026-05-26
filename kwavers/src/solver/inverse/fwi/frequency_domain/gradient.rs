//! Discrete adjoint gradient for frequency-domain FWI.

use super::cbs::{
    apply_shifted_green_adjoint_operator, real_scattering_potential,
    real_scattering_potential_for_operator, receiver_adjoint_for_operator,
    sample_array_for_operator, scattering_slowness_derivative_factor_for_operator,
    solve_adjoint_volume_field_with_operator, solve_volume_field_with_operator,
    source_density_for_operator, GridSpec,
};
use super::forward::{incident_field, outgoing_green, validate_forward_inputs, voxel_centers};
use super::types::{Config, FrequencyObservation};
use crate::core::error::{KwaversError, KwaversResult};
use crate::physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::{
    complex_l2_objective, complex_source_scale, helmholtz_slowness_derivative, MultiRowRingArray,
};
use crate::solver::inverse::linear_born_inversion::ElementPosition;
use ndarray::Array3;
use num_complex::Complex64;
use crate::core::constants::numerical::{TWO_PI};

pub(super) fn objective_and_gradient(
    slowness_s_per_m: &Array3<f64>,
    observations: &[FrequencyObservation],
    array: &MultiRowRingArray,
    config: &Config,
) -> KwaversResult<(f64, Array3<f64>)> {
    if observations.is_empty() {
        return Err(KwaversError::InvalidInput(
            "FWI requires at least one frequency observation".to_owned(),
        ));
    }

    let mut objective = 0.0;
    let mut gradient = Array3::zeros(slowness_s_per_m.dim());
    for observation in observations {
        let rows = observation.observed_pressure.nrows();
        if observation.observed_pressure.ncols() != array.element_count() {
            return Err(KwaversError::DimensionMismatch(format!(
                "receiver count mismatch: observed {}, geometry {}",
                observation.observed_pressure.ncols(),
                array.element_count()
            )));
        }
        validate_forward_inputs(
            slowness_s_per_m,
            array,
            observation.frequency_hz,
            config,
            rows,
        )?;
        if config.forward_operator.uses_volume_field_adjoint() {
            accumulate_dense_cbs_frequency_gradient(
                slowness_s_per_m,
                observation,
                array,
                config,
                &mut objective,
                &mut gradient,
            )?;
        } else {
            accumulate_frequency_gradient(
                slowness_s_per_m,
                observation,
                array,
                config,
                &mut objective,
                &mut gradient,
            )?;
        }
    }

    apply_tikhonov(slowness_s_per_m, config, &mut objective, &mut gradient);
    Ok((objective, gradient))
}

fn accumulate_dense_cbs_frequency_gradient(
    slowness_s_per_m: &Array3<f64>,
    observation: &FrequencyObservation,
    array: &MultiRowRingArray,
    config: &Config,
    objective: &mut f64,
    gradient: &mut Array3<f64>,
) -> KwaversResult<()> {
    let (cbs_config, operator) = config
        .forward_operator
        .cbs_descriptor(config, observation.frequency_hz)?
        .ok_or_else(|| {
            KwaversError::InvalidInput(
                "CBS gradient requires a convergent Born forward operator".to_owned(),
            )
        })?;

    let rows = observation.observed_pressure.nrows();
    let omega = TWO_PI * observation.frequency_hz;
    let reference_slowness = 1.0 / config.reference_sound_speed_m_s;
    let reference_wavenumber = omega * reference_slowness;
    let grid = GridSpec::new(slowness_s_per_m.dim(), config.spacing_m)?;
    let potential = real_scattering_potential_for_operator(
        omega,
        slowness_s_per_m,
        reference_slowness,
        operator,
    )?;
    let slowness = slowness_s_per_m.iter().copied().collect::<Vec<_>>();
    let slowness_derivative_factor =
        scattering_slowness_derivative_factor_for_operator(omega, operator)?;

    for transmit in 0..rows {
        let source_density = source_density_for_operator(
            grid,
            &array.cylindrical_source(transmit),
            reference_wavenumber,
            operator,
        )?;
        let forward_solution = solve_volume_field_with_operator(
            grid,
            reference_wavenumber,
            &potential,
            &source_density,
            cbs_config,
            operator,
        )?;
        let predicted = sample_array_for_operator(grid, &forward_solution.field, array, operator)?;
        let observed = observation.observed_pressure.row(transmit).to_vec();
        let source_scale = if config.estimate_source_scaling {
            complex_source_scale(&predicted, &observed)?
        } else {
            Complex64::new(1.0, 0.0)
        };
        let scaled_prediction = predicted
            .iter()
            .map(|&value| source_scale * value)
            .collect::<Vec<_>>();
        *objective += complex_l2_objective(&scaled_prediction, &observed)
            .expect("validated equal row lengths");

        let receiver_residual = scaled_prediction
            .iter()
            .zip(observed.iter())
            .map(|(&predicted_value, &observed_value)| {
                source_scale.conj() * (predicted_value - observed_value)
            })
            .collect::<Vec<_>>();
        let adjoint_rhs = receiver_adjoint_for_operator(grid, array, &receiver_residual, operator)?;
        let adjoint_solution = solve_adjoint_volume_field_with_operator(
            grid,
            reference_wavenumber,
            &potential,
            &adjoint_rhs,
            forward_solution.epsilon,
            cbs_config,
            operator,
        )?;
        let green_adjoint = apply_shifted_green_adjoint_operator(
            grid,
            reference_wavenumber,
            forward_solution.epsilon,
            &adjoint_solution.field,
            operator,
        );
        accumulate_dense_cbs_slowness_gradient(
            gradient,
            &forward_solution.field,
            &green_adjoint,
            &slowness,
            slowness_derivative_factor,
        );
    }
    Ok(())
}

fn accumulate_dense_cbs_slowness_gradient(
    gradient: &mut Array3<f64>,
    forward_field: &[Complex64],
    green_adjoint: &[Complex64],
    slowness: &[f64],
    slowness_derivative_factor: f64,
) {
    let (_, ny, nz) = gradient.dim();
    for (linear_index, ((&forward_value, &adjoint_value), &slowness_value)) in forward_field
        .iter()
        .zip(green_adjoint.iter())
        .zip(slowness.iter())
        .enumerate()
    {
        let ix = linear_index / (ny * nz);
        let rem = linear_index % (ny * nz);
        let iy = rem / nz;
        let iz = rem % nz;
        gradient[[ix, iy, iz]] -= (adjoint_value.conj() * forward_value).re
            * (slowness_derivative_factor * slowness_value);
    }
}

fn accumulate_frequency_gradient(
    slowness_s_per_m: &Array3<f64>,
    observation: &FrequencyObservation,
    array: &MultiRowRingArray,
    config: &Config,
    objective: &mut f64,
    gradient: &mut Array3<f64>,
) -> KwaversResult<()> {
    let rows = observation.observed_pressure.nrows();
    let omega = TWO_PI * observation.frequency_hz;
    let reference_slowness = 1.0 / config.reference_sound_speed_m_s;
    let reference_wavenumber = omega * reference_slowness;
    let min_distance = 0.5 * config.spacing_m;
    let cell_volume = config.spacing_m.powi(3);
    let centers = voxel_centers(slowness_s_per_m.dim(), config.spacing_m);
    let potential = real_scattering_potential(omega, slowness_s_per_m, reference_slowness)?;
    let slowness = slowness_s_per_m.iter().copied().collect::<Vec<_>>();

    for transmit in 0..rows {
        let sources = array.cylindrical_source(transmit);
        let incident = incident_field(&sources, &centers, reference_wavenumber, min_distance);
        let predicted = predicted_row(
            &sources,
            &centers,
            &incident,
            &potential,
            array,
            reference_wavenumber,
            min_distance,
            cell_volume,
        );
        let observed = observation.observed_pressure.row(transmit).to_vec();
        let source_scale = if config.estimate_source_scaling {
            complex_source_scale(&predicted, &observed)?
        } else {
            Complex64::new(1.0, 0.0)
        };
        let scaled_prediction = predicted
            .iter()
            .map(|&value| source_scale * value)
            .collect::<Vec<_>>();
        *objective += complex_l2_objective(&scaled_prediction, &observed)
            .expect("validated equal row lengths");
        accumulate_row_adjoint(
            gradient,
            &centers,
            &incident,
            &slowness,
            array,
            omega,
            reference_wavenumber,
            min_distance,
            cell_volume,
            source_scale,
            &scaled_prediction,
            &observed,
        );
    }
    Ok(())
}

fn predicted_row(
    sources: &[ElementPosition],
    centers: &[(usize, ElementPosition)],
    incident: &[Complex64],
    potential: &[f64],
    array: &MultiRowRingArray,
    reference_wavenumber: f64,
    min_distance: f64,
    cell_volume: f64,
) -> Vec<Complex64> {
    array
        .elements()
        .iter()
        .map(|&receiver| {
            let direct = sources
                .iter()
                .map(|&source| outgoing_green(source, receiver, reference_wavenumber, min_distance))
                .sum::<Complex64>();
            let scattered = centers
                .iter()
                .zip(incident.iter().zip(potential.iter()))
                .map(|((_, point), (&incident_value, &potential_value))| {
                    incident_value
                        * outgoing_green(*point, receiver, reference_wavenumber, min_distance)
                        * (potential_value * cell_volume)
                })
                .sum::<Complex64>();
            direct + scattered
        })
        .collect()
}

fn accumulate_row_adjoint(
    gradient: &mut Array3<f64>,
    centers: &[(usize, ElementPosition)],
    incident: &[Complex64],
    slowness: &[f64],
    array: &MultiRowRingArray,
    omega: f64,
    reference_wavenumber: f64,
    min_distance: f64,
    cell_volume: f64,
    source_scale: Complex64,
    predicted: &[Complex64],
    observed: &[Complex64],
) {
    let (_, ny, nz) = gradient.dim();
    for (receiver_index, &receiver) in array.elements().iter().enumerate() {
        let residual = predicted[receiver_index] - observed[receiver_index];
        for ((linear_index, point), (&incident_value, &slowness_value)) in
            centers.iter().zip(incident.iter().zip(slowness.iter()))
        {
            let derivative = source_scale
                * incident_value
                * outgoing_green(*point, receiver, reference_wavenumber, min_distance)
                * (helmholtz_slowness_derivative(omega, slowness_value) * cell_volume);
            let ix = linear_index / (ny * nz);
            let rem = linear_index % (ny * nz);
            let iy = rem / nz;
            let iz = rem % nz;
            gradient[[ix, iy, iz]] += (derivative.conj() * residual).re;
        }
    }
}

fn apply_tikhonov(
    slowness_s_per_m: &Array3<f64>,
    config: &Config,
    objective: &mut f64,
    gradient: &mut Array3<f64>,
) {
    if config.tikhonov_weight <= 0.0 {
        return;
    }
    let reference = 1.0 / config.reference_sound_speed_m_s;
    for (grad, &slowness) in gradient.iter_mut().zip(slowness_s_per_m.iter()) {
        let diff = slowness - reference;
        *objective += 0.5 * config.tikhonov_weight * diff * diff;
        *grad += config.tikhonov_weight * diff;
    }
}

#[must_use]
pub(super) fn dot(a: &Array3<f64>, b: &Array3<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[must_use]
pub(super) fn max_abs(a: &Array3<f64>) -> f64 {
    a.iter().map(|value| value.abs()).fold(0.0, f64::max)
}
