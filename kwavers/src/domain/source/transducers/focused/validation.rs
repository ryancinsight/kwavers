//! Shared validation helpers for focused source geometries.

use crate::core::error::{KwaversError, KwaversResult, ValidationError};

#[inline]
pub(super) fn positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

pub(super) fn validate_positive_finite_field(field: &'static str, value: f64) -> KwaversResult<()> {
    if positive_finite(value) {
        Ok(())
    } else {
        Err(field_validation_error(
            field,
            value.to_string(),
            "must be positive and finite",
        ))
    }
}

pub(super) fn validate_finite_field(field: &'static str, value: f64) -> KwaversResult<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(field_validation_error(
            field,
            value.to_string(),
            "must be finite",
        ))
    }
}

pub(super) fn validate_finite_vector<const N: usize>(
    field: &'static str,
    value: [f64; N],
) -> KwaversResult<()> {
    if value.iter().all(|component| component.is_finite()) {
        Ok(())
    } else {
        Err(field_validation_error(
            field,
            format!("{value:?}"),
            "must contain only finite coordinates",
        ))
    }
}

pub(super) fn validate_element_count(element_count: usize) -> KwaversResult<()> {
    if element_count > 0 {
        Ok(())
    } else {
        Err(field_validation_error(
            "element_count",
            element_count.to_string(),
            "must be at least one",
        ))
    }
}

pub(super) fn field_validation_error(
    field: &'static str,
    value: String,
    constraint: &'static str,
) -> KwaversError {
    KwaversError::Validation(ValidationError::FieldValidation {
        field: field.to_owned(),
        value,
        constraint: constraint.to_owned(),
    })
}
