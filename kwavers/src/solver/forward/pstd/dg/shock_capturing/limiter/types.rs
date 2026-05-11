//! WENOLimiter struct and public interface.

use crate::core::constants::numerical::{NUMERICAL_SHOCK_DETECTION_THRESHOLD, WENO_EPSILON};
use crate::core::error::{ConfigError, KwaversError, KwaversResult, ValidationError};
use ndarray::Array3;

/// WENO-based shock limiter
#[derive(Debug, Clone)]
pub struct WENOLimiter {
    /// WENO order (3, 5, or 7)
    pub(super) order: usize,
    /// Small parameter to avoid division by zero
    pub(super) epsilon: f64,
    /// Threshold for shock detection
    pub(super) shock_threshold: f64,
}

impl WENOLimiter {
    /// New.
    /// # Errors
    /// - Returns [`KwaversError::Config`] if the precondition for a Config-class constraint is violated.
    ///
    pub fn new(order: usize) -> KwaversResult<Self> {
        if order != 3 && order != 5 && order != 7 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "weno_order".to_owned(),
                value: order.to_string(),
                constraint: "WENO order must be 3, 5, or 7".to_owned(),
            }));
        }

        Ok(Self {
            order,
            epsilon: WENO_EPSILON,
            shock_threshold: NUMERICAL_SHOCK_DETECTION_THRESHOLD,
        })
    }

    /// Apply WENO limiting, writing the result into a caller-provided output buffer.
    ///
    /// ## Performance
    /// Zero allocations per call: `output` is pre-allocated by the caller (typically
    /// a time-stepper scratch field). Reads always come from the immutable `field`
    /// argument; shocked cells are overwritten in `output` while unshocked cells
    /// are set to `field` values via an initial `output.assign(field)`.
    ///
    /// ## Precondition
    /// `output` must have the same shape as `field`.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    pub fn limit_field_into(
        &self,
        field: &Array3<f64>,
        shock_indicator: &Array3<f64>,
        output: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        debug_assert_eq!(field.dim(), output.dim(), "output shape must match field");
        output.assign(field); // initialize: unshocked cells keep original values
        match self.order {
            3 => self.weno3_limit_into(field, shock_indicator, output)?,
            5 => self.weno5_limit(output, shock_indicator)?, // already reads stencil in-place
            7 => self.weno7_limit_into(field, shock_indicator, output)?,
            _ => {
                return Err(KwaversError::Validation(ValidationError::FieldValidation {
                    field: "weno_order".to_owned(),
                    value: self.order.to_string(),
                    constraint: "must be 3, 5, or 7".to_owned(),
                }));
            }
        }
        Ok(())
    }

    /// Convenience wrapper — allocates and returns the limited field.
    /// Prefer [`Self::limit_field_into`] in time-step loops to avoid per-step allocation.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn limit_field(
        &self,
        field: &Array3<f64>,
        shock_indicator: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let mut output = field.clone();
        self.limit_field_into(field, shock_indicator, &mut output)?;
        Ok(output)
    }
}
