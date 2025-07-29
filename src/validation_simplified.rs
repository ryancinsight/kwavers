//! Simplified validation system for kwavers
//!
//! This module provides a streamlined validation framework following
//! KISS and DRY principles while maintaining comprehensive validation capabilities.

use crate::error::KwaversResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Validation result with essential information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// Create a new valid result
    pub fn valid() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }
    
    /// Create a new invalid result with an error
    pub fn invalid(error: impl Into<String>) -> Self {
        Self {
            is_valid: false,
            errors: vec![error.into()],
            warnings: Vec::new(),
        }
    }
    
    /// Add an error
    pub fn add_error(&mut self, error: impl Into<String>) {
        self.is_valid = false;
        self.errors.push(error.into());
    }
    
    /// Add a warning
    pub fn add_warning(&mut self, warning: impl Into<String>) {
        self.warnings.push(warning.into());
    }
    
    /// Merge with another result
    pub fn merge(&mut self, other: ValidationResult) {
        self.is_valid = self.is_valid && other.is_valid;
        self.errors.extend(other.errors);
        self.warnings.extend(other.warnings);
    }
}

/// Trait for types that can be validated
pub trait Validatable {
    fn validate(&self) -> ValidationResult;
}

/// Validation rules for numeric values
pub struct NumericValidator {
    min: Option<f64>,
    max: Option<f64>,
    must_be_positive: bool,
    must_be_finite: bool,
}

impl NumericValidator {
    pub fn new() -> Self {
        Self {
            min: None,
            max: None,
            must_be_positive: false,
            must_be_finite: true,
        }
    }
    
    pub fn min(mut self, min: f64) -> Self {
        self.min = Some(min);
        self
    }
    
    pub fn max(mut self, max: f64) -> Self {
        self.max = Some(max);
        self
    }
    
    pub fn positive(mut self) -> Self {
        self.must_be_positive = true;
        self
    }
    
    pub fn validate(&self, value: f64, field_name: &str) -> ValidationResult {
        let mut result = ValidationResult::valid();
        
        if self.must_be_finite && !value.is_finite() {
            result.add_error(format!("{} must be finite, got {}", field_name, value));
        }
        
        if self.must_be_positive && value <= 0.0 {
            result.add_error(format!("{} must be positive, got {}", field_name, value));
        }
        
        if let Some(min) = self.min {
            if value < min {
                result.add_error(format!("{} must be >= {}, got {}", field_name, min, value));
            }
        }
        
        if let Some(max) = self.max {
            if value > max {
                result.add_error(format!("{} must be <= {}, got {}", field_name, max, value));
            }
        }
        
        result
    }
}

/// Common validation functions
pub mod validators {
    use super::*;
    
    /// Validate grid dimensions
    pub fn validate_grid_dimensions(nx: usize, ny: usize, nz: usize) -> ValidationResult {
        let mut result = ValidationResult::valid();
        
        if nx == 0 || ny == 0 || nz == 0 {
            result.add_error(format!("Grid dimensions must be positive: ({}, {}, {})", nx, ny, nz));
        }
        
        let total_points = nx * ny * nz;
        if total_points > 100_000_000 {
            result.add_error(format!("Grid too large: {} points (max 100M)", total_points));
        } else if total_points > 10_000_000 {
            result.add_warning("Large grid size may impact performance");
        }
        
        result
    }
    
    /// Validate grid spacing
    pub fn validate_grid_spacing(dx: f64, dy: f64, dz: f64) -> ValidationResult {
        let mut result = ValidationResult::valid();
        
        let dx_val = NumericValidator::new().positive().validate(dx, "dx");
        let dy_val = NumericValidator::new().positive().validate(dy, "dy");
        let dz_val = NumericValidator::new().positive().validate(dz, "dz");
        
        result.merge(dx_val);
        result.merge(dy_val);
        result.merge(dz_val);
        
        // Check aspect ratio
        let max_spacing = dx.max(dy).max(dz);
        let min_spacing = dx.min(dy).min(dz);
        if max_spacing / min_spacing > 10.0 {
            result.add_warning("Large aspect ratio in grid spacing may affect accuracy");
        }
        
        result
    }
    
    /// Validate time parameters
    pub fn validate_time_params(dt: f64, t_end: f64, cfl: f64) -> ValidationResult {
        let mut result = ValidationResult::valid();
        
        result.merge(NumericValidator::new().positive().validate(dt, "dt"));
        result.merge(NumericValidator::new().positive().validate(t_end, "t_end"));
        result.merge(NumericValidator::new().positive().max(1.0).validate(cfl, "CFL"));
        
        if dt > t_end {
            result.add_error("Time step dt cannot be larger than t_end");
        }
        
        result
    }
    
    /// Validate medium properties
    pub fn validate_medium_properties(props: &HashMap<String, f64>) -> ValidationResult {
        let mut result = ValidationResult::valid();
        
        // Required properties
        if let Some(&density) = props.get("density") {
            result.merge(NumericValidator::new().positive().validate(density, "density"));
        } else {
            result.add_error("Missing required property: density");
        }
        
        if let Some(&sound_speed) = props.get("sound_speed") {
            result.merge(NumericValidator::new().positive().validate(sound_speed, "sound_speed"));
        } else {
            result.add_error("Missing required property: sound_speed");
        }
        
        // Optional properties
        if let Some(&attenuation) = props.get("attenuation") {
            result.merge(NumericValidator::new().min(0.0).validate(attenuation, "attenuation"));
        }
        
        if let Some(&nonlinearity) = props.get("nonlinearity") {
            result.merge(NumericValidator::new().min(0.0).validate(nonlinearity, "nonlinearity"));
        }
        
        result
    }
    
    /// Validate simulation stability
    pub fn validate_stability(dt: f64, dx: f64, sound_speed: f64, cfl_number: f64) -> ValidationResult {
        let mut result = ValidationResult::valid();
        
        let max_dt = cfl_number * dx / sound_speed;
        if dt > max_dt {
            result.add_error(format!(
                "Time step too large for stability: dt={:.3e} > max_dt={:.3e} (CFL={:.2})",
                dt, max_dt, cfl_number
            ));
        } else if dt > 0.9 * max_dt {
            result.add_warning("Time step close to stability limit");
        }
        
        result
    }
}

/// Validation builder for complex validation scenarios
pub struct ValidationBuilder {
    results: Vec<ValidationResult>,
}

impl ValidationBuilder {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }
    
    pub fn validate<F>(&mut self, validation_fn: F) -> &mut Self
    where
        F: FnOnce() -> ValidationResult,
    {
        self.results.push(validation_fn());
        self
    }
    
    pub fn validate_if<F>(&mut self, condition: bool, validation_fn: F) -> &mut Self
    where
        F: FnOnce() -> ValidationResult,
    {
        if condition {
            self.results.push(validation_fn());
        }
        self
    }
    
    pub fn build(self) -> ValidationResult {
        let mut final_result = ValidationResult::valid();
        for result in self.results {
            final_result.merge(result);
        }
        final_result
    }
}

/// Extension trait for easy validation
pub trait ValidationExt {
    fn validate_range(&self, min: f64, max: f64, field: &str) -> ValidationResult;
    fn validate_positive(&self, field: &str) -> ValidationResult;
}

impl ValidationExt for f64 {
    fn validate_range(&self, min: f64, max: f64, field: &str) -> ValidationResult {
        NumericValidator::new().min(min).max(max).validate(*self, field)
    }
    
    fn validate_positive(&self, field: &str) -> ValidationResult {
        NumericValidator::new().positive().validate(*self, field)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_numeric_validator() {
        let validator = NumericValidator::new().min(0.0).max(10.0);
        
        assert!(validator.validate(5.0, "test").is_valid);
        assert!(!validator.validate(-1.0, "test").is_valid);
        assert!(!validator.validate(11.0, "test").is_valid);
    }
    
    #[test]
    fn test_validation_builder() {
        let result = ValidationBuilder::new()
            .validate(|| ValidationResult::valid())
            .validate(|| ValidationResult::invalid("Test error"))
            .build();
            
        assert!(!result.is_valid);
        assert_eq!(result.errors.len(), 1);
    }
    
    #[test]
    fn test_grid_validation() {
        let result = validators::validate_grid_dimensions(100, 100, 100);
        assert!(result.is_valid);
        
        let result = validators::validate_grid_dimensions(0, 100, 100);
        assert!(!result.is_valid);
    }
}