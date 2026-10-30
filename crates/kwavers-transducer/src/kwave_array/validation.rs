//! Input validation for [`KWaveArray`](super::KWaveArray) element constructors.
//!
//! Every rejection is a `KwaversError::Config` naming the offending
//! parameter, so invalid inputs never panic (not even in Python, where they
//! surface as `ValueError`).

use kwavers_core::error::{ConfigError, KwaversError, KwaversResult};

/// Build the `Config::InvalidValue` error the constructors return for
/// rejected inputs.
pub(super) fn invalid_value(parameter: &str, value: String, constraint: &str) -> KwaversError {
    KwaversError::Config(ConfigError::InvalidValue {
        parameter: parameter.to_owned(),
        value,
        constraint: constraint.to_owned(),
    })
}

/// Reject a NaN or infinite element position with a typed error.
pub(super) fn check_finite_position(
    parameter: &str,
    position: (f64, f64, f64),
) -> KwaversResult<()> {
    if position.0.is_finite() && position.1.is_finite() && position.2.is_finite() {
        Ok(())
    } else {
        Err(invalid_value(
            parameter,
            format!("({}, {}, {})", position.0, position.1, position.2),
            "all coordinates must be finite",
        ))
    }
}

/// Reject a NaN, infinite, or non-positive dimension with a typed error.
pub(super) fn check_positive_finite(parameter: &str, value: f64) -> KwaversResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(invalid_value(
            parameter,
            value.to_string(),
            "must be finite and positive",
        ))
    }
}
