// src/validation.rs
//! Advanced validation system for kwavers
//!
//! This module provides a sophisticated validation framework following
//! multiple design principles:
//!
//! Design Principles Implemented:
//! - SOLID: Single responsibility for each validator, open/closed for extensibility
//! - CUPID: Composable validation components, predictable validation behavior
//! - GRASP: Information expert for validation logic, controller for validation flow
//! - ACID: Atomic validation operations, consistent validation states
//! - DRY: Shared validation patterns and utilities
//! - KISS: Simple, clear validation interfaces
//! - YAGNI: Only implement necessary validation features
//! - SSOT: Single source of truth for validation rules
//! - CCP: Common closure for related validation handling
//! - CRP: Common reuse of validation utilities
//! - ADP: Acyclic dependency in validation hierarchy

use crate::error::{KwaversResult, ValidationError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Validation result with detailed information
/// 
/// Implements SSOT principle as the single source of truth for validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
    pub context: ValidationContext,
    pub metadata: ValidationMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    pub field: String,
    pub message: String,
    pub severity: WarningSeverity,
    pub suggestion: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum WarningSeverity {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationContext {
    pub validator_name: String,
    pub field_path: Vec<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub additional_info: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetadata {
    pub validation_time_ms: u64,
    pub rules_applied: Vec<String>,
    pub performance_metrics: HashMap<String, f64>,
}

impl ValidationResult {
    /// Create a new valid validation result
    pub fn valid(validator_name: String) -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            context: ValidationContext {
                validator_name,
                field_path: Vec::new(),
                timestamp: chrono::Utc::now(),
                additional_info: HashMap::new(),
            },
            metadata: ValidationMetadata {
                validation_time_ms: 0,
                rules_applied: Vec::new(),
                performance_metrics: HashMap::new(),
            },
        }
    }
    
    /// Create a new invalid validation result
    pub fn invalid(validator_name: String, errors: Vec<ValidationError>) -> Self {
        Self {
            is_valid: false,
            errors,
            warnings: Vec::new(),
            context: ValidationContext {
                validator_name,
                field_path: Vec::new(),
                timestamp: chrono::Utc::now(),
                additional_info: HashMap::new(),
            },
            metadata: ValidationMetadata {
                validation_time_ms: 0,
                rules_applied: Vec::new(),
                performance_metrics: HashMap::new(),
            },
        }
    }
    
    /// Add an error to the validation result
    pub fn add_error(&mut self, error: ValidationError) {
        self.is_valid = false;
        self.errors.push(error);
    }
    
    /// Add a warning to the validation result
    pub fn add_warning(&mut self, warning: ValidationWarning) {
        self.warnings.push(warning);
    }
    
    /// Merge with another validation result
    pub fn merge(&mut self, other: ValidationResult) {
        self.is_valid = self.is_valid && other.is_valid;
        self.errors.extend(other.errors);
        self.warnings.extend(other.warnings);
        self.metadata.rules_applied.extend(other.metadata.rules_applied);
        
        // Merge performance metrics
        for (key, value) in other.metadata.performance_metrics {
            *self.metadata.performance_metrics.entry(key).or_insert(0.0) += value;
        }
    }
    
    /// Get error count
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }
    
    /// Get warning count
    pub fn warning_count(&self) -> usize {
        self.warnings.len()
    }
    
    /// Check if result has errors
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
    
    /// Check if result has warnings
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
    
    /// Get summary of validation result
    pub fn summary(&self) -> String {
        if self.is_valid {
            format!("Valid ({} warnings)", self.warning_count())
        } else {
            format!("Invalid ({} errors, {} warnings)", self.error_count(), self.warning_count())
        }
    }
}

/// Validation rule trait
/// 
/// Implements Single Responsibility Principle - each rule has one validation purpose
pub trait ValidationRule: Send + Sync {
    /// Get rule name
    fn name(&self) -> &str;
    
    /// Get rule description
    fn description(&self) -> &str;
    
    /// Check if rule is applicable to a value type
    fn is_applicable(&self, _value: &ValidationValue) -> bool {
        true // Default: applicable to all values
    }
    
    /// Validate a value
    fn validate(&self, value: &ValidationValue, _context: &ValidationContext) -> ValidationResult;
    
    /// Clone the rule into a new boxed instance
    /// This enables proper cloning of trait objects
    fn clone_box(&self) -> Box<dyn ValidationRule>;
    
    /// Get rule priority (lower numbers have higher priority)
    fn priority(&self) -> u32 {
        0
    }
}

/// Validation value types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ValidationValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Array(Vec<ValidationValue>),
    Object(HashMap<String, ValidationValue>),
    Null,
}

impl ValidationValue {
    /// Get value as string
    pub fn as_string(&self) -> Option<&str> {
        match self {
            ValidationValue::String(s) => Some(s),
            _ => None,
        }
    }
    
    /// Get value as integer
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            ValidationValue::Integer(i) => Some(*i),
            ValidationValue::Float(f) => Some(*f as i64),
            _ => None,
        }
    }
    
    /// Get value as float
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ValidationValue::Float(f) => Some(*f),
            ValidationValue::Integer(i) => Some(*i as f64),
            _ => None,
        }
    }
    
    /// Get value as boolean
    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            ValidationValue::Boolean(b) => Some(*b),
            _ => None,
        }
    }
    
    /// Get value as array
    pub fn as_array(&self) -> Option<&Vec<ValidationValue>> {
        match self {
            ValidationValue::Array(arr) => Some(arr),
            _ => None,
        }
    }
    
    /// Get value as object
    pub fn as_object(&self) -> Option<&HashMap<String, ValidationValue>> {
        match self {
            ValidationValue::Object(obj) => Some(obj),
            _ => None,
        }
    }
    
    /// Check if value is null
    pub fn is_null(&self) -> bool {
        matches!(self, ValidationValue::Null)
    }
}

/// Range validation rule
/// 
/// Validates that numeric values fall within specified bounds
#[derive(Debug, Clone)]
pub struct RangeValidationRule {
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub field_name: String,
}

impl RangeValidationRule {
    pub fn new(field_name: String, min: Option<f64>, max: Option<f64>) -> Self {
        Self {
            min,
            max,
            field_name,
        }
    }
}

impl ValidationRule for RangeValidationRule {
    fn name(&self) -> &str {
        "range_validation"
    }
    
    fn description(&self) -> &str {
        "Validates that a numeric value is within a specified range"
    }
    
    fn validate(&self, value: &ValidationValue, context: &ValidationContext) -> ValidationResult {
        let start_time = std::time::Instant::now();
        
        if let Some(val) = value.as_float() {
            let mut errors = Vec::new();
            
            if let Some(min) = self.min {
                if val < min {
                    errors.push(ValidationError::RangeValidation {
                        field: self.field_name.clone(),
                        value: val,
                        min,
                        max: f64::INFINITY,
                    });
                }
            }
            
            if let Some(max) = self.max {
                if val > max {
                    errors.push(ValidationError::RangeValidation {
                        field: self.field_name.clone(),
                        value: val,
                        min: f64::NEG_INFINITY,
                        max,
                    });
                }
            }
            
            let validation_time = start_time.elapsed().as_millis() as u64;
            
            if errors.is_empty() {
                let mut result = ValidationResult::valid(self.name().to_string());
                result.metadata.validation_time_ms = validation_time;
                result.metadata.rules_applied.push(self.name().to_string());
                result
            } else {
                let mut result = ValidationResult::invalid(self.name().to_string(), errors);
                result.metadata.validation_time_ms = validation_time;
                result.metadata.rules_applied.push(self.name().to_string());
                result
            }
        } else {
            ValidationResult::valid(self.name().to_string())
        }
    }
    
    fn is_applicable(&self, value: &ValidationValue) -> bool {
        value.as_float().is_some() || value.as_integer().is_some()
    }

    fn clone_box(&self) -> Box<dyn ValidationRule> {
        Box::new((*self).clone())
    }
}

/// String length validation rule
#[derive(Debug, Clone)]
pub struct StringLengthValidationRule {
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
    pub field_name: String,
}

impl StringLengthValidationRule {
    pub fn new(field_name: String, min_length: Option<usize>, max_length: Option<usize>) -> Self {
        Self {
            min_length,
            max_length,
            field_name,
        }
    }
}

impl ValidationRule for StringLengthValidationRule {
    fn name(&self) -> &str {
        "string_length_validation"
    }
    
    fn description(&self) -> &str {
        "Validates that a string value has a length within specified bounds"
    }
    
    fn validate(&self, value: &ValidationValue, context: &ValidationContext) -> ValidationResult {
        let start_time = std::time::Instant::now();
        
        if let Some(s) = value.as_string() {
            let mut errors = Vec::new();
            
            if let Some(min_len) = self.min_length {
                if s.len() < min_len {
                    errors.push(ValidationError::RangeValidation {
                        field: self.field_name.clone(),
                        value: s.len() as f64,
                        min: min_len as f64,
                        max: f64::INFINITY,
                    });
                }
            }
            
            if let Some(max_len) = self.max_length {
                if s.len() > max_len {
                    errors.push(ValidationError::RangeValidation {
                        field: self.field_name.clone(),
                        value: s.len() as f64,
                        min: 0.0,
                        max: max_len as f64,
                    });
                }
            }
            
            let validation_time = start_time.elapsed().as_millis() as u64;
            
            if errors.is_empty() {
                let mut result = ValidationResult::valid(self.name().to_string());
                result.metadata.validation_time_ms = validation_time;
                result.metadata.rules_applied.push(self.name().to_string());
                result
            } else {
                let mut result = ValidationResult::invalid(self.name().to_string(), errors);
                result.metadata.validation_time_ms = validation_time;
                result.metadata.rules_applied.push(self.name().to_string());
                result
            }
        } else {
            ValidationResult::valid(self.name().to_string())
        }
    }
    
    fn is_applicable(&self, value: &ValidationValue) -> bool {
        value.as_string().is_some()
    }

    fn clone_box(&self) -> Box<dyn ValidationRule> {
        Box::new((*self).clone())
    }
}

/// Pattern validation rule
#[derive(Debug, Clone)]
pub struct PatternValidationRule {
    pub pattern: String,
    pub field_name: String,
    pub description: String,
}

impl PatternValidationRule {
    pub fn new(field_name: String, pattern: String, description: String) -> Self {
        Self {
            pattern,
            field_name,
            description,
        }
    }
}

impl ValidationRule for PatternValidationRule {
    fn name(&self) -> &str {
        "pattern_validation"
    }
    
    fn description(&self) -> &str {
        &self.description
    }
    
    fn validate(&self, value: &ValidationValue, context: &ValidationContext) -> ValidationResult {
        let start_time = std::time::Instant::now();
        
        if let Some(s) = value.as_string() {
            // Simple pattern matching (could be enhanced with regex)
            let is_valid = s.contains(&self.pattern);
            
            let validation_time = start_time.elapsed().as_millis() as u64;
            
            if is_valid {
                let mut result = ValidationResult::valid(self.name().to_string());
                result.metadata.validation_time_ms = validation_time;
                result.metadata.rules_applied.push(self.name().to_string());
                result
            } else {
                let error = ValidationError::FieldValidation {
                    field: self.field_name.clone(),
                    value: s.to_string(),
                    constraint: format!("pattern: {}", self.pattern),
                };
                let mut result = ValidationResult::invalid(self.name().to_string(), vec![error]);
                result.metadata.validation_time_ms = validation_time;
                result.metadata.rules_applied.push(self.name().to_string());
                result
            }
        } else {
            ValidationResult::valid(self.name().to_string())
        }
    }
    
    fn is_applicable(&self, value: &ValidationValue) -> bool {
        value.as_string().is_some()
    }

    fn clone_box(&self) -> Box<dyn ValidationRule> {
        Box::new((*self).clone())
    }
}

/// Required field validation rule
#[derive(Debug, Clone)]
pub struct RequiredFieldValidationRule {
    pub field_name: String,
}

impl RequiredFieldValidationRule {
    pub fn new(field_name: String) -> Self {
        Self { field_name }
    }
}

impl ValidationRule for RequiredFieldValidationRule {
    fn name(&self) -> &str {
        "required_field_validation"
    }
    
    fn description(&self) -> &str {
        "Validates that a required field is present and not null"
    }
    
    fn validate(&self, value: &ValidationValue, _context: &ValidationContext) -> ValidationResult {
        let start_time = std::time::Instant::now();
        
        if value.is_null() {
            let error = ValidationError::FieldValidation {
                field: self.field_name.clone(),
                value: "null".to_string(),
                constraint: "required".to_string(),
            };
            let mut result = ValidationResult::invalid(self.name().to_string(), vec![error]);
            result.metadata.validation_time_ms = start_time.elapsed().as_millis() as u64;
            result.metadata.rules_applied.push(self.name().to_string());
            result
        } else {
            let mut result = ValidationResult::valid(self.name().to_string());
            result.metadata.validation_time_ms = start_time.elapsed().as_millis() as u64;
            result.metadata.rules_applied.push(self.name().to_string());
            result
        }
    }

    fn clone_box(&self) -> Box<dyn ValidationRule> {
        Box::new((*self).clone())
    }
}

/// Validation pipeline for composable validation
/// 
/// Implements Controller pattern from GRASP principles
pub struct ValidationPipeline {
    rules: Vec<Box<dyn ValidationRule>>,
    context: ValidationContext,
}

impl Clone for ValidationPipeline {
    fn clone(&self) -> Self {
        Self {
            rules: self.rules.iter().map(|rule| rule.clone_box()).collect(),
            context: self.context.clone(),
        }
    }
}

impl ValidationPipeline {
    /// Create a new validation pipeline
    pub fn new(validator_name: String) -> Self {
        Self {
            rules: Vec::new(),
            context: ValidationContext {
                validator_name,
                field_path: Vec::new(),
                timestamp: chrono::Utc::now(),
                additional_info: HashMap::new(),
            },
        }
    }
    
    /// Add a validation rule to the pipeline
    pub fn add_rule(&mut self, rule: Box<dyn ValidationRule>) {
        self.rules.push(rule);
    }
    
    /// Add multiple validation rules
    pub fn add_rules(&mut self, rules: Vec<Box<dyn ValidationRule>>) {
        self.rules.extend(rules);
    }
    
    /// Set field path for context
    pub fn with_field_path(mut self, field_path: Vec<String>) -> Self {
        self.context.field_path = field_path;
        self
    }
    
    /// Add additional context information
    pub fn with_info(mut self, key: String, value: String) -> Self {
        self.context.additional_info.insert(key, value);
        self
    }
    
    /// Validate a value using all rules in the pipeline
    pub fn validate(&self, value: &ValidationValue) -> ValidationResult {
        let start_time = std::time::Instant::now();
        let mut final_result = ValidationResult::valid(self.context.validator_name.clone());
        
        // Sort rules by priority
        let mut sorted_rules: Vec<&Box<dyn ValidationRule>> = self.rules.iter().collect();
        sorted_rules.sort_by_key(|rule| rule.priority());
        
        for rule in sorted_rules {
            if rule.is_applicable(value) {
                let rule_result = rule.validate(value, &self.context);
                final_result.merge(rule_result);
                
                // Early termination if critical error found
                if final_result.has_errors() && rule.priority() == 0 {
                    break;
                }
            }
        }
        
        final_result.metadata.validation_time_ms = start_time.elapsed().as_millis() as u64;
        final_result
    }
    
    /// Get number of rules in pipeline
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }
    
    /// Clear all rules
    pub fn clear_rules(&mut self) {
        self.rules.clear();
    }
}

/// Validation manager for global validation handling
/// 
/// Implements Information Expert principle by managing validation state
pub struct ValidationManager {
    pipelines: Arc<RwLock<HashMap<String, ValidationPipeline>>>,
    rule_registry: Arc<RwLock<HashMap<String, Box<dyn ValidationRule>>>>,
    validation_cache: Arc<RwLock<HashMap<String, ValidationResult>>>,
}

impl ValidationManager {
    /// Create a new validation manager
    pub fn new() -> Self {
        Self {
            pipelines: Arc::new(RwLock::new(HashMap::new())),
            rule_registry: Arc::new(RwLock::new(HashMap::new())),
            validation_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Register a validation rule
    pub fn register_rule(&self, name: String, rule: Box<dyn ValidationRule>) {
        let mut registry = self.rule_registry.write().unwrap();
        registry.insert(name, rule);
    }
    
    /// Get a validation rule by name with proper cloning support
    pub fn get_rule(&self, name: &str) -> Option<Box<dyn ValidationRule>> {
        let registry = self.rule_registry.read().unwrap();
        registry.get(name).map(|rule| rule.clone_box())
    }
    
    /// Create a validation pipeline with basic structure
    pub fn create_pipeline(&self, name: String) -> ValidationPipeline {
        log::debug!("Creating validation pipeline: {}", name);
        ValidationPipeline::new(name)
    }
    
    /// Register a validation pipeline
    pub fn register_pipeline(&self, name: String, pipeline: ValidationPipeline) {
        let mut pipelines = self.pipelines.write().unwrap();
        pipelines.insert(name, pipeline);
    }
    
    /// Get a registered pipeline
    pub fn get_pipeline(&self, name: &str) -> Option<ValidationPipeline> {
        let pipelines = self.pipelines.read().unwrap();
        pipelines.get(name).cloned()
    }
    
    /// Validate using a registered pipeline
    pub fn validate_with_pipeline(&self, pipeline_name: &str, value: &ValidationValue) -> KwaversResult<ValidationResult> {
        let pipeline = self.get_pipeline(pipeline_name).ok_or_else(|| {
            ValidationError::FieldValidation {
                field: "pipeline".to_string(),
                value: pipeline_name.to_string(),
                constraint: "must be registered".to_string(),
            }
        })?;
        
        Ok(pipeline.validate(value))
    }
    
    /// Cache validation result
    pub fn cache_result(&self, key: String, result: ValidationResult) {
        let mut cache = self.validation_cache.write().unwrap();
        cache.insert(key, result);
    }
    
    /// Get cached validation result
    pub fn get_cached_result(&self, key: &str) -> Option<ValidationResult> {
        let cache = self.validation_cache.read().unwrap();
        cache.get(key).cloned()
    }
    
    /// Clear validation cache
    pub fn clear_cache(&self) {
        let mut cache = self.validation_cache.write().unwrap();
        cache.clear();
    }
    
    /// Get all registered pipeline names
    pub fn get_pipeline_names(&self) -> Vec<String> {
        let pipelines = self.pipelines.read().unwrap();
        pipelines.keys().cloned().collect()
    }
    
    /// Get all registered rule names
    pub fn get_rule_names(&self) -> Vec<String> {
        let registry = self.rule_registry.read().unwrap();
        registry.keys().cloned().collect()
    }
}

impl Default for ValidationManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Validation builder for fluent validation construction
/// 
/// Implements Builder pattern for complex validation creation
pub struct ValidationBuilder {
    pipeline: ValidationPipeline,
}

impl ValidationBuilder {
    /// Create a new validation builder
    pub fn new(validator_name: String) -> Self {
        Self {
            pipeline: ValidationPipeline::new(validator_name),
        }
    }
    
    /// Add range validation rule
    pub fn with_range(mut self, field_name: String, min: Option<f64>, max: Option<f64>) -> Self {
        let rule = Box::new(RangeValidationRule::new(field_name, min, max));
        self.pipeline.add_rule(rule);
        self
    }
    
    /// Add string length validation rule
    pub fn with_string_length(mut self, field_name: String, min_length: Option<usize>, max_length: Option<usize>) -> Self {
        let rule = Box::new(StringLengthValidationRule::new(field_name, min_length, max_length));
        self.pipeline.add_rule(rule);
        self
    }
    
    /// Add pattern validation rule
    pub fn with_pattern(mut self, field_name: String, pattern: String, description: String) -> Self {
        let rule = Box::new(PatternValidationRule::new(field_name, pattern, description));
        self.pipeline.add_rule(rule);
        self
    }
    
    /// Add required field validation rule
    pub fn with_required(mut self, field_name: String) -> Self {
        let rule = Box::new(RequiredFieldValidationRule::new(field_name));
        self.pipeline.add_rule(rule);
        self
    }
    
    /// Add custom validation rule
    pub fn with_custom(mut self, rule: Box<dyn ValidationRule>) -> Self {
        self.pipeline.add_rule(rule);
        self
    }
    
    /// Set field path
    pub fn with_field_path(mut self, field_path: Vec<String>) -> Self {
        self.pipeline = self.pipeline.with_field_path(field_path);
        self
    }
    
    /// Add context information
    pub fn with_info(mut self, key: String, value: String) -> Self {
        self.pipeline = self.pipeline.with_info(key, value);
        self
    }
    
    /// Build the validation pipeline
    pub fn build(self) -> ValidationPipeline {
        self.pipeline
    }
}

/// Validation utilities for common operations
/// 
/// Implements DRY principle by providing reusable validation utilities
pub mod utils {
    use super::*;
    
    /// Create a basic validation pipeline for common scenarios
    pub fn create_basic_validation_pipeline(field_name: String) -> ValidationPipeline {
        ValidationBuilder::new("basic_validation".to_string())
            .with_required(field_name.clone())
            .with_string_length(field_name.clone(), Some(1), Some(100))
            .build()
    }
    
    /// Create a numeric validation pipeline
    pub fn create_numeric_validation_pipeline(field_name: String, min: f64, max: f64) -> ValidationPipeline {
        ValidationBuilder::new("numeric_validation".to_string())
            .with_required(field_name.clone())
            .with_range(field_name, Some(min), Some(max))
            .build()
    }
    
    /// Create an email validation pipeline
    pub fn create_email_validation_pipeline(field_name: String) -> ValidationPipeline {
        ValidationBuilder::new("email_validation".to_string())
            .with_required(field_name.clone())
            .with_string_length(field_name.clone(), Some(5), Some(254))
            .with_pattern(field_name, "@".to_string(), "must contain @ symbol".to_string())
            .build()
    }
    
    /// Validate multiple values with the same pipeline
    pub fn validate_multiple(
        pipeline: &ValidationPipeline,
        values: &[(&str, ValidationValue)]
    ) -> HashMap<String, ValidationResult> {
        let mut results = HashMap::new();
        
        for (name, value) in values {
            let result = pipeline.validate(value);
            results.insert(name.to_string(), result);
        }
        
        results
    }
    
    /// Check if all validation results are valid
    pub fn all_valid(results: &[ValidationResult]) -> bool {
        results.iter().all(|result| result.is_valid)
    }
    
    /// Get all errors from multiple validation results
    pub fn collect_errors(results: &[ValidationResult]) -> Vec<ValidationError> {
        results.iter()
            .flat_map(|result| result.errors.clone())
            .collect()
    }
    
    /// Get all warnings from multiple validation results
    pub fn collect_warnings(results: &[ValidationResult]) -> Vec<ValidationWarning> {
        results.iter()
            .flat_map(|result| result.warnings.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_result_creation() {
        let result = ValidationResult::valid("test_validator".to_string());
        assert!(result.is_valid);
        assert_eq!(result.error_count(), 0);
        
        let error = ValidationError::FieldValidation {
            field: "test_field".to_string(),
            value: "test_value".to_string(),
            constraint: "test_constraint".to_string(),
        };
        let result = ValidationResult::invalid("test_validator".to_string(), vec![error]);
        assert!(!result.is_valid);
        assert_eq!(result.error_count(), 1);
    }
    
    #[test]
    fn test_range_validation_rule() {
        let rule = RangeValidationRule::new("test_field".to_string(), Some(0.0), Some(100.0));
        
        let valid_value = ValidationValue::Float(50.0);
        let result = rule.validate(&valid_value, &ValidationContext {
            validator_name: "test".to_string(),
            field_path: vec![],
            timestamp: chrono::Utc::now(),
            additional_info: HashMap::new(),
        });
        assert!(result.is_valid);
        
        let invalid_value = ValidationValue::Float(150.0);
        let result = rule.validate(&invalid_value, &ValidationContext {
            validator_name: "test".to_string(),
            field_path: vec![],
            timestamp: chrono::Utc::now(),
            additional_info: HashMap::new(),
        });
        assert!(!result.is_valid);
    }
    
    #[test]
    fn test_validation_pipeline() {
        let mut pipeline = ValidationPipeline::new("test_pipeline".to_string());
        
        let range_rule = Box::new(RangeValidationRule::new("test_field".to_string(), Some(0.0), Some(100.0)));
        let required_rule = Box::new(RequiredFieldValidationRule::new("test_field".to_string()));
        
        pipeline.add_rule(range_rule);
        pipeline.add_rule(required_rule);
        
        let value = ValidationValue::Float(50.0);
        let result = pipeline.validate(&value);
        assert!(result.is_valid);
    }
    
    #[test]
    fn test_validation_builder() {
        let pipeline = ValidationBuilder::new("test_builder".to_string())
            .with_required("test_field".to_string())
            .with_range("test_field".to_string(), Some(0.0), Some(100.0))
            .build();
        
        assert_eq!(pipeline.rule_count(), 2);
    }
    
    #[test]
    fn test_validation_manager() {
        let manager = ValidationManager::new();
        
        let rule = Box::new(RangeValidationRule::new("test_field".to_string(), Some(0.0), Some(100.0)));
        manager.register_rule("range_rule".to_string(), rule);
        
        // Test that the pipeline can be validated directly
        let value = ValidationValue::Float(50.0);
        let pipeline = ValidationBuilder::new("test_pipeline".to_string())
            .with_range("test_field".to_string(), Some(0.0), Some(100.0))
            .build();
        
        let result = pipeline.validate(&value);
        assert!(result.is_valid);
        
        // Register the pipeline after testing it
        manager.register_pipeline("test_pipeline".to_string(), pipeline);
    }
    
    #[test]
    fn test_validation_utilities() {
        let pipeline = utils::create_basic_validation_pipeline("test_field".to_string());
        let value = ValidationValue::String("test".to_string());
        let result = pipeline.validate(&value);
        assert!(result.is_valid);
        
        let results = vec![result];
        assert!(utils::all_valid(&results));
    }
}