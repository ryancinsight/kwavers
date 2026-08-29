//! Allocation-free validation and temporal planning for SWE propagation.

use super::super::super::super::integration::integrator::calculate_stable_timestep;
use super::super::super::super::types::{ElasticWaveConfig, ElasticWaveField};
use kwavers_core::error::{KwaversResult, ValidationError};
use kwavers_grid::Grid;
use leto::Array3;

#[derive(Clone, Copy, Debug)]
pub(super) struct PropagationPlan {
    pub(super) dt: f64,
    pub(super) steps: usize,
    pub(super) save_every: usize,
}

impl PropagationPlan {
    pub(super) fn for_field(
        grid: &Grid,
        lambda: &Array3<f64>,
        mu: &Array3<f64>,
        density: &Array3<f64>,
        config: &ElasticWaveConfig,
        field: &ElasticWaveField,
        duration_s: f64,
    ) -> KwaversResult<Self> {
        validate_field_shapes(grid, field)?;
        Self::for_initial_time(grid, lambda, mu, density, config, field.time, duration_s)
    }

    pub(super) fn for_initial_time(
        grid: &Grid,
        lambda: &Array3<f64>,
        mu: &Array3<f64>,
        density: &Array3<f64>,
        config: &ElasticWaveConfig,
        initial_time: f64,
        duration_s: f64,
    ) -> KwaversResult<Self> {
        validate_finite_positive(
            "duration_s",
            duration_s,
            "must be finite and greater than zero",
        )?;
        if !config.time_step.is_finite() || config.time_step < 0.0 {
            return Err(ValidationError::InvalidValue {
                parameter: "ElasticWaveConfig.time_step".to_owned(),
                value: config.time_step,
                reason: "must be finite and non-negative; zero selects automatic CFL".to_owned(),
            }
            .into());
        }
        if !initial_time.is_finite() {
            return Err(ValidationError::InvalidValue {
                parameter: "ElasticWaveField.time".to_owned(),
                value: initial_time,
                reason: "must be finite".to_owned(),
            }
            .into());
        }

        let dt = if config.time_step == 0.0 {
            calculate_stable_timestep(grid, lambda, mu, density, config.cfl_factor)
        } else {
            config.time_step
        };
        validate_finite_positive(
            "effective_time_step",
            dt,
            "must be finite and greater than zero",
        )?;

        let steps_f64 = (duration_s / dt).ceil();
        if !steps_f64.is_finite() || steps_f64 < 1.0 || steps_f64 >= usize::MAX as f64 {
            return Err(ValidationError::InvalidValue {
                parameter: "simulation_step_count".to_owned(),
                value: steps_f64,
                reason: "must be finite, positive, and representable as usize".to_owned(),
            }
            .into());
        }
        let end_time = steps_f64.mul_add(dt, initial_time);
        if !end_time.is_finite() {
            return Err(ValidationError::InvalidValue {
                parameter: "simulation_end_time".to_owned(),
                value: end_time,
                reason: "must be finite".to_owned(),
            }
            .into());
        }

        Ok(Self {
            dt,
            steps: steps_f64 as usize,
            save_every: config.save_every.max(1),
        })
    }
}

fn validate_field_shapes(grid: &Grid, field: &ElasticWaveField) -> KwaversResult<()> {
    let expected = [grid.nx, grid.ny, grid.nz];
    for (component, actual) in [
        ("ux", field.ux.shape()),
        ("uy", field.uy.shape()),
        ("uz", field.uz.shape()),
        ("vx", field.vx.shape()),
        ("vy", field.vy.shape()),
        ("vz", field.vz.shape()),
    ] {
        if actual != expected {
            return Err(ValidationError::DimensionMismatch {
                expected: format!("ElasticWaveField.{component} shape {expected:?}"),
                actual: format!("{actual:?}"),
            }
            .into());
        }
    }
    Ok(())
}

fn validate_finite_positive(parameter: &str, value: f64, reason: &str) -> KwaversResult<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(ValidationError::InvalidValue {
            parameter: parameter.to_owned(),
            value,
            reason: reason.to_owned(),
        }
        .into());
    }
    Ok(())
}
