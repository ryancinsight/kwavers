//! Discrete adjoint gradient for frequency-domain FWI.

use super::acquisition::TransmissionAcquisition;
use super::born_geometry::BornGeometry;
use super::born_kernel::fields::{FactoredProducts, StoredProducts};
use super::born_kernel::{cell_volume, VOXELS_PER_BLOCK};
use super::cbs::{
    apply_shifted_green_adjoint_operator, real_scattering_potential,
    real_scattering_potential_for_operator, receiver_adjoint_for_operator,
    sample_array_for_operator, scattering_slowness_derivative_factor_for_operator,
    solve_adjoint_volume_field_with_operator, solve_volume_field_with_operator,
    source_density_for_operator, GridSpec,
};
use super::types::{Config, FrequencyObservation};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::fft::Complex64;
use kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::{
    complex_l2_objective, complex_source_scale,
};
use leto::Array3;

mod born_receiver;
mod born_row;
pub(super) mod hessian;
pub(super) mod objective;
#[cfg(test)]
pub(super) mod test_support;
pub(super) mod vector;
use born_receiver::BornReceiverTerms;
use kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::helmholtz_slowness_derivative;

fn accumulate_finite_window_frequency_gradient(
    slowness_s_per_m: &Array3<f64>,
    observation: &FrequencyObservation,
    acquisition: &dyn TransmissionAcquisition,
    config: &Config,
    objective: &mut f64,
    gradient: &mut Array3<f64>,
) -> KwaversResult<()> {
    let fw_config = config
        .forward_operator
        .finite_window_adjoint_config()
        .ok_or_else(|| {
            KwaversError::InvalidInput(
                "forward operator reports uses_finite_window_adjoint but \
                 finite_window_adjoint_config returned None"
                    .to_owned(),
            )
        })
        .map(|mut c| {
            c.reference_sound_speed_m_s = config.reference_sound_speed_m_s;
            c.spacing_m = config.spacing_m;
            c
        })?;
    let rows = observation.observed_pressure.shape()[0];
    super::finite_window::finite_window_pstd_born_gradient(
        &kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::slowness_to_sound_speed(slowness_s_per_m)?,
        acquisition,
        observation.frequency_hz,
        &observation.observed_pressure,
        fw_config,
        rows,
        config.estimate_source_scaling,
        objective,
        gradient,
    )
}

fn accumulate_dense_cbs_frequency_gradient(
    slowness_s_per_m: &Array3<f64>,
    observation: &FrequencyObservation,
    acquisition: &dyn TransmissionAcquisition,
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

    let rows = observation.observed_pressure.shape()[0];
    let omega = TWO_PI * observation.frequency_hz;
    let reference_slowness = 1.0 / config.reference_sound_speed_m_s;
    let reference_wavenumber = omega * reference_slowness;
    let shape = slowness_s_per_m.shape();
    let grid = GridSpec::new((shape[0], shape[1], shape[2]), config.spacing_m)?;
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
            acquisition.sources(transmit),
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
        let predicted = sample_array_for_operator(
            grid,
            &forward_solution.field,
            acquisition.receivers(transmit),
            operator,
        )?;
        let observed = observation
            .observed_pressure
            .index_axis::<1>(0, transmit)
            .expect("transmit index in bounds")
            .iter()
            .copied()
            .collect::<Vec<_>>();
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
        let adjoint_rhs = receiver_adjoint_for_operator(
            grid,
            acquisition.receivers(transmit),
            &receiver_residual,
            operator,
        )?;
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
    let [_nx, ny, nz] = gradient.shape();
    // Retain one checked strided borrow rather than reborrowing the entire
    // backing allocation for each voxel update.
    let mut gradient_view = gradient.view_mut();
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
        gradient_view[[ix, iy, iz]] -= (adjoint_value.conj() * forward_value).re
            * (slowness_derivative_factor * slowness_value);
    }
}

fn accumulate_frequency_gradient(
    slowness_s_per_m: &Array3<f64>,
    observation: &FrequencyObservation,
    geometry: &BornGeometry,
    config: &Config,
    objective: &mut f64,
    gradient: &mut Array3<f64>,
) -> KwaversResult<()> {
    let omega = TWO_PI * observation.frequency_hz;
    let reference_slowness = 1.0 / config.reference_sound_speed_m_s;
    let cell_volume = cell_volume(config.spacing_m);
    // These exact products depend on the model/frequency, not the receiver.
    // Hoist them without regrouping any multiplication in prediction/adjoint.
    let potential_volume = real_scattering_potential(omega, slowness_s_per_m, reference_slowness)?
        .into_iter()
        .map(|value| value * cell_volume)
        .collect::<Vec<_>>();
    let derivative_volume = slowness_s_per_m
        .iter()
        .map(|&value| helmholtz_slowness_derivative(omega, value) * cell_volume)
        .collect::<Vec<_>>();
    let gradient = gradient
        .as_slice_mut()
        .expect("freshly allocated gradient has dense C-order storage");

    for (transmit, row) in geometry.transmits.iter().enumerate() {
        let terms = BornReceiverTerms {
            potential_volume: &potential_volume,
            derivative_volume: &derivative_volume,
        };
        let observed = observation
            .observed_pressure
            .index_axis::<1>(0, transmit)
            .expect("transmit index in bounds")
            .iter()
            .copied()
            .collect::<Vec<_>>();
        if let Some(products) = &geometry.products {
            assert!(
                !config.estimate_source_scaling,
                "prepared unit-source products require fixed source scaling"
            );
            let products = &products[transmit];
            born_row::accumulate(
                &observed,
                false,
                |receiver| {
                    let (fields, _) = products.fields(receiver);
                    terms.predict(row.direct[receiver], &StoredProducts::new(fields))
                },
                |scale, residuals, gradient| {
                    assert_eq!(scale.re.to_bits(), 1.0_f64.to_bits());
                    assert_eq!(scale.im.to_bits(), 0.0_f64.to_bits());
                    let receivers = residuals
                        .iter()
                        .enumerate()
                        .map(|(receiver, &residual)| {
                            let (_, fields) = products.fields(receiver);
                            (StoredProducts::new(fields), residual)
                        })
                        .collect::<Vec<_>>();
                    terms.accumulate_adjoint::<VOXELS_PER_BLOCK, _>(gradient, &receivers);
                },
                objective,
                gradient,
            )?;
        } else {
            born_row::accumulate(
                &observed,
                config.estimate_source_scaling,
                |receiver| {
                    terms.predict(
                        row.direct[receiver],
                        &FactoredProducts::new(
                            &row.incident,
                            &geometry.fields[row.receivers[receiver]],
                        ),
                    )
                },
                |scale, residuals, gradient| {
                    // Preserve the original left-associated scale * incident * G.
                    let scaled = row
                        .incident
                        .iter()
                        .map(|&value| scale * value)
                        .collect::<Vec<_>>();
                    let receivers = row
                        .receivers
                        .iter()
                        .zip(residuals)
                        .map(|(&field, &residual)| {
                            (
                                FactoredProducts::new(&scaled, &geometry.fields[field]),
                                residual,
                            )
                        })
                        .collect::<Vec<_>>();
                    terms.accumulate_adjoint::<VOXELS_PER_BLOCK, _>(gradient, &receivers);
                },
                objective,
                gradient,
            )?;
        }
    }
    Ok(())
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
