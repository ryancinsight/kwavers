//! Validation Module for Time-Reversal
//!
//! Validates input data and sensor configurations.

use crate::{
    core::error::{KwaversError, KwaversResult, ValidationError},
    domain::grid::Grid,
};
use ndarray::Array2;

/// Validator for time-reversal inputs
#[derive(Debug)]
pub struct InputValidator;

impl InputValidator {
    /// Validate sensor data for time-reversal
    pub fn validate_sensor_data(
        pressure_data: &Array2<f64>,
        sensor_indices: &[(usize, usize, usize)],
        grid: &Grid,
    ) -> KwaversResult<()> {
        let num_sensors = sensor_indices.len();
        if num_sensors == 0 {
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "sensor_indices".to_string(),
                value: "empty".to_string(),
                constraint: "must contain at least one sensor".to_string(),
            }));
        }

        if pressure_data.nrows() != num_sensors {
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "pressure_data".to_string(),
                value: format!("rows={}", pressure_data.nrows()),
                constraint: format!("must match number of sensors ({num_sensors})"),
            }));
        }
        if pressure_data.ncols() == 0 {
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "pressure_data".to_string(),
                value: "0 time steps".to_string(),
                constraint: "must contain time series data".to_string(),
            }));
        }

        for (idx, &(i, j, k)) in sensor_indices.iter().enumerate() {
            if i >= grid.nx || j >= grid.ny || k >= grid.nz {
                return Err(KwaversError::Validation(ValidationError::FieldValidation {
                    field: format!("sensor_indices[{idx}]"),
                    value: format!("({i}, {j}, {k})"),
                    constraint: format!(
                        "must be in-bounds for grid ({}, {}, {})",
                        grid.nx, grid.ny, grid.nz
                    ),
                }));
            }
        }

        Ok(())
    }

    /// Validate signal length consistency
    pub fn validate_signal_lengths(
        signals: &[Vec<f64>],
        expected_length: Option<usize>,
    ) -> KwaversResult<()> {
        if signals.is_empty() {
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "signals".to_string(),
                value: "empty".to_string(),
                constraint: "must contain at least one signal".to_string(),
            }));
        }

        let first_length = signals[0].len();

        // Check all signals have the same length
        for (i, signal) in signals.iter().enumerate() {
            if signal.len() != first_length {
                return Err(KwaversError::Validation(ValidationError::FieldValidation {
                    field: format!("signal[{i}]"),
                    value: format!("length={}", signal.len()),
                    constraint: format!("must match first signal length={first_length}"),
                }));
            }
        }

        // Check against expected length if provided
        if let Some(expected) = expected_length {
            if first_length != expected {
                return Err(KwaversError::Validation(ValidationError::FieldValidation {
                    field: "signal_length".to_string(),
                    value: first_length.to_string(),
                    constraint: format!("must be {expected}"),
                }));
            }
        }

        Ok(())
    }

    /// Validate grid dimensions
    pub fn validate_grid_dimensions(grid: &Grid) -> KwaversResult<()> {
        if grid.nx == 0 || grid.ny == 0 || grid.nz == 0 {
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "grid_dimensions".to_string(),
                value: format!("({}, {}, {})", grid.nx, grid.ny, grid.nz),
                constraint: "all dimensions must be non-zero".to_string(),
            }));
        }

        if grid.dx <= 0.0 || grid.dy <= 0.0 || grid.dz <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "grid_spacing".to_string(),
                value: format!("({}, {}, {})", grid.dx, grid.dy, grid.dz),
                constraint: "all spacings must be positive".to_string(),
            }));
        }

        Ok(())
    }
}
