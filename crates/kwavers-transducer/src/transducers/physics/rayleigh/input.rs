//! Checks on caller-supplied geometry and coefficients, reported as configuration errors.

use kwavers_core::error::{ConfigError, KwaversError};

pub(super) fn invalid(parameter: &str, value: String, constraint: &str) -> KwaversError {
    KwaversError::Config(ConfigError::InvalidValue {
        parameter: parameter.to_owned(),
        value,
        constraint: constraint.to_owned(),
    })
}
