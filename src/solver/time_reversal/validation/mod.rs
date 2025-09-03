//! Validation Module for Time-Reversal
//!
//! Validates input data and sensor configurations.

use crate::{
    error::{KwaversError, KwaversResult, ValidationError},
    grid::Grid,
    sensor::SensorData,
};

/// Validator for time-reversal inputs
pub struct InputValidator;

impl InputValidator {
    /// Validate sensor data for time-reversal
    pub fn validate_sensor_data(sensor_data: &SensorData, grid: &Grid) -> KwaversResult<()> {
        if sensor_data.is_empty() {
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "sensor_data".to_string(),
                value: "empty".to_string(),
                constraint: "must contain at least one sensor".to_string(),
            }));
        }

        // Check sensor positions are within grid
        for sensor in sensor_data.sensors() {
            let pos = sensor.position();
            if pos[0] >= grid.nx || pos[1] >= grid.ny || pos[2] >= grid.nz {
                return Err(KwaversError::Validation(ValidationError::FieldValidation {
                    field: "sensor_position".to_string(),
                    value: format!("{pos:?}"),
                    constraint: format!(
                        "must be within grid bounds ({}, {}, {})",
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
