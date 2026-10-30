//! Consolidated positive/finite parameter validators (KW-VALIDATE-CORE).
//!
//! One documented constructor per shape; every crate delegates here instead
//! of defining a private copy. The accept/reject sets are identical to the
//! helpers this replaces: finite and (for the positive shapes) strictly
//! greater than zero, with the caller's parameter name, value, and reason
//! carried verbatim into the error.

use crate::error::{ConfigError, KwaversError, KwaversResult, ValidationError};

/// Require `value` to be finite and strictly positive.
///
/// # Errors
///
/// - Returns [`KwaversError::Validation`] carrying `parameter`, `value`,
///   and the caller's `reason` when `value` is nonfinite or `<= 0.0`.
#[inline]
pub fn validate_positive_finite(parameter: &str, value: f64, reason: &str) -> KwaversResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(KwaversError::Validation(ValidationError::InvalidValue {
            parameter: parameter.to_owned(),
            value,
            reason: reason.to_owned(),
        }))
    }
}

/// Require `value` to be finite and nonnegative.
///
/// # Errors
///
/// - Returns [`KwaversError::Validation`] carrying `parameter`, `value`,
///   and the caller's `reason` when `value` is nonfinite or `< 0.0`.
#[inline]
pub fn validate_nonnegative_finite(parameter: &str, value: f64, reason: &str) -> KwaversResult<()> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(KwaversError::Validation(ValidationError::InvalidValue {
            parameter: parameter.to_owned(),
            value,
            reason: reason.to_owned(),
        }))
    }
}

/// Require every coordinate of a 3-point to be finite.
///
/// The error keeps the [`ConfigError::InvalidValue`] shape (string value,
/// `constraint` naming) that the transducer callers already produce, so
/// delegation changes no observable error.
///
/// # Errors
///
/// - Returns [`KwaversError::Config`] carrying `parameter`, the debug
///   rendering of `point`, and the caller's `reason` as `constraint` when
///   any coordinate is nonfinite.
#[inline]
pub fn validate_finite_point(parameter: &str, point: [f64; 3], reason: &str) -> KwaversResult<()> {
    if point.iter().all(|coordinate| coordinate.is_finite()) {
        Ok(())
    } else {
        Err(KwaversError::Config(ConfigError::InvalidValue {
            parameter: parameter.to_owned(),
            value: format!("{point:?}"),
            constraint: reason.to_owned(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn invalid_value(error: KwaversError) -> (String, f64, String) {
        match error {
            KwaversError::Validation(ValidationError::InvalidValue {
                parameter,
                value,
                reason,
            }) => (parameter, value, reason),
            other => panic!("expected invalid-value error, got {other:?}"),
        }
    }

    #[test]
    fn positive_finite_accepts_strictly_positive() {
        for value in [f64::MIN_POSITIVE, 1.0e-6, 0.5, 1.0e12] {
            validate_positive_finite("p", value, "r").expect("finite positive must pass");
        }
    }

    #[test]
    fn positive_finite_rejects_nonfinite_and_nonpositive() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.0, -1.5] {
            let error = validate_positive_finite("gain", value, "must be finite and positive")
                .expect_err("nonfinite or non-positive must fail");
            let (parameter, rejected, reason) = invalid_value(error);
            assert_eq!(parameter, "gain");
            assert_eq!(reason, "must be finite and positive");
            if value.is_nan() {
                assert!(rejected.is_nan());
            } else {
                assert_eq!(rejected, value);
            }
        }
    }

    #[test]
    fn nonnegative_finite_accepts_zero() {
        for value in [0.0, 1.0e-9, 3.0] {
            validate_nonnegative_finite("p", value, "r").expect("finite nonnegative must pass");
        }
        for value in [f64::NAN, f64::INFINITY, -1.0e-12] {
            validate_nonnegative_finite("p", value, "r")
                .expect_err("nonfinite or negative must fail");
        }
    }

    #[test]
    fn finite_point_checks_every_coordinate() {
        validate_finite_point("origin", [0.0, 0.0, 0.0], "all coordinates finite")
            .expect("finite point must pass");
        let error = validate_finite_point("probe", [1.0, f64::NAN, 3.0], "all coordinates finite")
            .expect_err("nonfinite coordinate must fail");
        match error {
            KwaversError::Config(ConfigError::InvalidValue {
                parameter,
                value,
                constraint,
            }) => {
                assert_eq!(parameter, "probe");
                assert_eq!(constraint, "all coordinates finite");
                assert!(
                    value.contains("NaN"),
                    "value carries the point, got {value}"
                );
            }
            other => panic!("expected config invalid-value error, got {other:?}"),
        }
    }
}
