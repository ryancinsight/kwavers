//! Parameter validation for the control system

use super::parameter::{ParameterType, ParameterValue};
use crate::domain::core::error::KwaversResult;

/// Result of parameter validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub message: Option<String>,
    pub corrected_value: Option<ParameterValue>,
}

impl ValidationResult {
    /// Create a valid result
    pub fn valid() -> Self {
        Self {
            is_valid: true,
            message: None,
            corrected_value: None,
        }
    }

    /// Create an invalid result with message
    pub fn invalid(message: impl Into<String>) -> Self {
        Self {
            is_valid: false,
            message: Some(message.into()),
            corrected_value: None,
        }
    }

    /// Create a result with corrected value
    pub fn corrected(value: ParameterValue, message: impl Into<String>) -> Self {
        Self {
            is_valid: true,
            message: Some(message.into()),
            corrected_value: Some(value),
        }
    }
}

/// Parameter validator
#[derive(Debug)]
pub struct ParameterValidator;

impl ParameterValidator {
    /// Validate a parameter value against its type definition
    pub fn validate(value: &ParameterValue, param_type: &ParameterType) -> ValidationResult {
        match (value, param_type) {
            (ParameterValue::Float(v), ParameterType::Float { min, max, .. }) => {
                if *v < *min {
                    ValidationResult::corrected(
                        ParameterValue::Float(*min),
                        format!("Value clamped to minimum: {}", min),
                    )
                } else if *v > *max {
                    ValidationResult::corrected(
                        ParameterValue::Float(*max),
                        format!("Value clamped to maximum: {}", max),
                    )
                } else {
                    ValidationResult::valid()
                }
            }
            (ParameterValue::Integer(v), ParameterType::Integer { min, max, .. }) => {
                if *v < *min {
                    ValidationResult::corrected(
                        ParameterValue::Integer(*min),
                        format!("Value clamped to minimum: {}", min),
                    )
                } else if *v > *max {
                    ValidationResult::corrected(
                        ParameterValue::Integer(*max),
                        format!("Value clamped to maximum: {}", max),
                    )
                } else {
                    ValidationResult::valid()
                }
            }
            (ParameterValue::Boolean(_), ParameterType::Boolean) => ValidationResult::valid(),
            (ParameterValue::Enum(v), ParameterType::Enum { options }) => {
                if options.contains(v) {
                    ValidationResult::valid()
                } else {
                    ValidationResult::invalid(format!("Invalid enum value: {}", v))
                }
            }
            (ParameterValue::Vector3(v), ParameterType::Vector3 { min, max, .. }) => {
                let mut validated = *v;
                let mut was_validated = false;

                for i in 0..3 {
                    if v[i] < *min {
                        validated[i] = *min;
                        was_validated = true;
                    } else if v[i] > *max {
                        validated[i] = *max;
                        was_validated = true;
                    }
                }

                if was_validated {
                    ValidationResult::corrected(
                        ParameterValue::Vector3(validated),
                        "Vector components clamped to range",
                    )
                } else {
                    ValidationResult::valid()
                }
            }
            (ParameterValue::Color(v), ParameterType::Color) => {
                let mut validated = *v;
                let mut was_validated = false;

                for i in 0..3 {
                    if v[i] < 0.0 {
                        validated[i] = 0.0;
                        was_validated = true;
                    } else if v[i] > 1.0 {
                        validated[i] = 1.0;
                        was_validated = true;
                    }
                }

                if was_validated {
                    ValidationResult::corrected(
                        ParameterValue::Color(validated),
                        "Color components clamped to [0, 1]",
                    )
                } else {
                    ValidationResult::valid()
                }
            }
            _ => ValidationResult::invalid("Type mismatch"),
        }
    }

    /// Validate and apply a parameter update
    pub fn validate_and_apply(
        value: ParameterValue,
        param_type: &ParameterType,
    ) -> KwaversResult<ParameterValue> {
        let result = Self::validate(&value, param_type);

        if result.is_valid {
            Ok(result.corrected_value.unwrap_or(value))
        } else {
            Err(crate::domain::core::error::KwaversError::InvalidInput(
                result
                    .message
                    .unwrap_or_else(|| "Validation failed".to_string()),
            ))
        }
    }
}
