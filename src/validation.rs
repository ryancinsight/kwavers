//! Validation system for kwavers
//!
//! This module provides a validation framework using
//! struct methods and the validator crate for complex validations.

use crate::error::ValidationError;

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<ValidationError>,
}

impl ValidationResult {
    /// Create a successful validation result
    pub fn success() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
        }
    }

    /// Create a failed validation result with errors
    pub fn failure(errors: Vec<ValidationError>) -> Self {
        Self {
            is_valid: false,
            errors,
        }
    }

    /// Add an error to the validation result
    pub fn add_error(&mut self, error: ValidationError) {
        self.errors.push(error);
        self.is_valid = false;
    }

    /// Combine two validation results
    pub fn combine(mut self, other: ValidationResult) -> Self {
        self.errors.extend(other.errors);
        self.is_valid = self.is_valid && other.is_valid;
        self
    }

    /// Create from a vector of errors
    pub fn from_errors(errors: Vec<ValidationError>) -> Self {
        Self {
            is_valid: errors.is_empty(),
            errors,
        }
    }
}

/// Trait for validatable configuration structs
pub trait Validatable {
    /// Validate the configuration
    fn validate(&self) -> ValidationResult;
}

/// Helper functions for common validation patterns
pub mod validators {
    use super::*;

    /// Validate that a value is within a range
    pub fn validate_range<T: PartialOrd + Copy + std::fmt::Display>(
        value: T,
        min: T,
        max: T,
        field_name: &str,
    ) -> ValidationResult {
        if value < min || value > max {
            ValidationResult::failure(vec![ValidationError::RangeValidation {
                field: field_name.to_string(),
                value: value.to_string(),
                min: min.to_string(),
                max: max.to_string(),
            }])
        } else {
            ValidationResult::success()
        }
    }

    /// Validate that a value is positive
    pub fn validate_positive<T: PartialOrd + Default + Copy + std::fmt::Display>(
        value: T,
        field_name: &str,
    ) -> ValidationResult {
        if value <= T::default() {
            ValidationResult::failure(vec![ValidationError::RangeValidation {
                field: field_name.to_string(),
                value: value.to_string(),
                min: "0".to_string(),
                max: "∞".to_string(),
            }])
        } else {
            ValidationResult::success()
        }
    }

    /// Validate string length
    pub fn validate_string_length(
        value: &str,
        min_len: usize,
        max_len: usize,
        field_name: &str,
    ) -> ValidationResult {
        let len = value.len();
        if len < min_len || len > max_len {
            ValidationResult::failure(vec![ValidationError::FieldValidation {
                field: field_name.to_string(),
                value: format!("length {}", len),
                constraint: format!("length between {} and {}", min_len, max_len),
            }])
        } else {
            ValidationResult::success()
        }
    }

    /// Validate that a collection is not empty
    pub fn validate_not_empty<T>(
        collection: &[T],
        field_name: &str,
    ) -> ValidationResult {
        if collection.is_empty() {
            ValidationResult::failure(vec![ValidationError::FieldValidation {
                field: field_name.to_string(),
                value: "empty".to_string(),
                constraint: "non-empty".to_string(),
            }])
        } else {
            ValidationResult::success()
        }
    }
}