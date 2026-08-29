//! Allocation-free validation and temporal planning for SWE propagation.

use super::super::super::super::integration::integrator::calculate_stable_timestep;
use super::super::super::super::types::{ElasticWaveConfig, ElasticWaveField};
use core::alloc::Layout;
use kwavers_core::error::{KwaversResult, NumericalError, ValidationError};
use kwavers_grid::Grid;
use leto::Array3;

#[derive(Clone, Copy, Debug)]
pub(super) struct PropagationPlan {
    pub(super) dt: f64,
    pub(super) steps: usize,
    pub(super) save_every: usize,
}

impl PropagationPlan {
    pub(super) fn history_layout(self) -> KwaversResult<(usize, Layout)> {
        derive_history_layout(self.steps, self.save_every)
    }

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

        let steps_f64 = if duration_s <= dt {
            1.0
        } else {
            (duration_s / dt).ceil()
        };
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

fn derive_history_layout(steps: usize, save_every: usize) -> KwaversResult<(usize, Layout)> {
    let save_every = save_every.max(1);
    let capacity = steps.div_ceil(save_every).checked_add(1).ok_or_else(|| {
        NumericalError::InvalidOperation(
            "Elastic wave history snapshot count exceeds usize".to_owned(),
        )
    })?;
    let layout = Layout::array::<ElasticWaveField>(capacity).map_err(|_| {
        NumericalError::InvalidOperation(
            "Elastic wave history header layout exceeds addressable memory".to_owned(),
        )
    })?;
    Ok((capacity, layout))
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

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::error::KwaversError;

    #[test]
    fn history_layout_covers_complete_snapshot_schedule() {
        for (steps, save_every, expected_capacity) in [
            (0, 1, 1),
            (1, 1, 2),
            (4, 2, 3),
            (5, 2, 4),
            (3, 4, 2),
            (3, 0, 4),
        ] {
            let (capacity, layout) =
                derive_history_layout(steps, save_every).expect("addressable schedule");
            assert_eq!(capacity, expected_capacity);
            assert_eq!(layout.size(), capacity * size_of::<ElasticWaveField>());
        }
    }

    #[test]
    fn history_layout_rejects_count_and_address_space_overflow() {
        let count_error = derive_history_layout(usize::MAX, 1)
            .expect_err("snapshot count addition must be checked");
        assert_invalid_operation(
            count_error,
            "Elastic wave history snapshot count exceeds usize",
        );

        let first_unaddressable_capacity =
            (isize::MAX as usize / size_of::<ElasticWaveField>()) + 1;
        let layout_error = derive_history_layout(first_unaddressable_capacity - 1, 1)
            .expect_err("header layout must fit the address space");
        assert_invalid_operation(
            layout_error,
            "Elastic wave history header layout exceeds addressable memory",
        );
    }

    fn assert_invalid_operation(error: KwaversError, expected: &str) {
        match error {
            KwaversError::Numerical(NumericalError::InvalidOperation(actual)) => {
                assert_eq!(actual, expected);
            }
            other => panic!("expected numerical invalid operation, got {other}"),
        }
    }
}
