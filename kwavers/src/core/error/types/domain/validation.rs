//! Validation-specific error types
//!
//! Specialized error handling for input validation

use thiserror::Error;

/// Validation error types with detailed context
#[derive(Error, Debug, Clone)]
pub enum ValidationErrorType {
    #[error("Input validation failed: {field} with value {value} failed constraint {constraint}")]
    InputValidation {
        field: String,
        value: String,
        constraint: String,
    },
    
    #[error("Range validation failed: {field} = {value} not in range [{min}, {max}]")]
    RangeValidation {
        field: String,
        value: f64,
        min: f64,
        max: f64,
    },
    
    #[error("Type validation failed: expected {expected_type}, got {actual_type}")]
    TypeValidation {
        expected_type: String,
        actual_type: String,
    },
    
    #[error("Format validation failed: {field} format invalid, expected {expected_format}")]
    FormatValidation {
        field: String,
        expected_format: String,
    },
    
    #[error("Consistency validation failed: {reason}")]
    ConsistencyValidation { reason: String },
    
    #[error("Business rule violation: {rule}")]
    BusinessRuleViolation { rule: String },
}
