//! Checks on caller-supplied geometry and coefficients, reported as configuration errors.

use kwavers_core::error::{ConfigError, KwaversError, KwaversResult};

pub(super) fn validate_positive(parameter: &str, value: f64) -> KwaversResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(invalid(parameter, value.to_string(), "finite and > 0"))
    }
}

pub(super) fn validate_point(parameter: &str, point: [f64; 3]) -> KwaversResult<()> {
    if point.iter().all(|coordinate| coordinate.is_finite()) {
        Ok(())
    } else {
        Err(invalid(
            parameter,
            format!("{point:?}"),
            "all coordinates finite",
        ))
    }
}

pub(super) fn invalid(parameter: &str, value: String, constraint: &str) -> KwaversError {
    KwaversError::Config(ConfigError::InvalidValue {
        parameter: parameter.to_owned(),
        value,
        constraint: constraint.to_owned(),
    })
}
