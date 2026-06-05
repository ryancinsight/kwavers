//! Shared validation for acoustic field-analysis routines.

use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_grid::Grid;
use ndarray::ArrayView3;

pub(super) fn validate_pressure_field_domain(
    pressure_field: ArrayView3<f64>,
    grid: &Grid,
) -> KwaversResult<()> {
    validate_pressure_field_shape(pressure_field, grid)?;
    validate_finite_pressure_field(pressure_field)
}

pub(super) fn validate_pressure_field_shape(
    pressure_field: ArrayView3<f64>,
    grid: &Grid,
) -> KwaversResult<()> {
    let expected = (grid.nx, grid.ny, grid.nz);
    let actual = pressure_field.dim();
    if actual != expected {
        return Err(KwaversError::Validation(
            ValidationError::DimensionMismatch {
                expected: format!("{expected:?}"),
                actual: format!("{actual:?}"),
            },
        ));
    }

    Ok(())
}

pub(super) fn validate_finite_pressure_field(pressure_field: ArrayView3<f64>) -> KwaversResult<()> {
    for ((ix, iy, iz), &pressure) in pressure_field.indexed_iter() {
        if !pressure.is_finite() {
            return Err(validation_error(format!(
                "Pressure field contains nonfinite value {pressure} at [{ix}, {iy}, {iz}]"
            )));
        }
    }

    Ok(())
}

pub(super) fn invalid_parameter(parameter: &str, value: f64, reason: &str) -> KwaversError {
    KwaversError::Validation(ValidationError::InvalidValue {
        parameter: parameter.to_owned(),
        value,
        reason: reason.to_owned(),
    })
}

pub(super) fn validation_error(message: impl Into<String>) -> KwaversError {
    KwaversError::Validation(ValidationError::ConstraintViolation {
        message: message.into(),
    })
}
