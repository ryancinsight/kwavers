//! Simplified error handling system for kwavers
//!
//! This module implements a consolidated error handling system following
//! KISS and DRY principles while maintaining comprehensive error information.

use std::error::Error as StdError;
use std::fmt;
use serde::{Deserialize, Serialize};

/// Main error type for kwavers operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KwaversError {
    /// Validation errors with field, value, and constraint information
    Validation {
        field: String,
        value: String,
        constraint: String,
        context: Option<String>,
    },
    
    /// Configuration errors
    Configuration {
        parameter: String,
        reason: String,
        section: Option<String>,
    },
    
    /// Resource errors (file I/O, memory, etc.)
    Resource {
        resource: String,
        operation: String,
        reason: String,
    },
    
    /// Numerical computation errors
    Numerical {
        operation: String,
        value: f64,
        constraint: String,
    },
    
    /// Physics simulation errors
    Physics {
        component: String,
        state: String,
        reason: String,
    },
    
    /// System errors (GPU, threading, etc.)
    System {
        subsystem: String,
        operation: String,
        reason: String,
    },
    
    /// Feature not implemented
    NotImplemented(String),
}

impl fmt::Display for KwaversError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KwaversError::Validation { field, value, constraint, context } => {
                if let Some(ctx) = context {
                    write!(f, "Validation error in {}: {} = {} violates {}", ctx, field, value, constraint)
                } else {
                    write!(f, "Validation error: {} = {} violates {}", field, value, constraint)
                }
            }
            KwaversError::Configuration { parameter, reason, section } => {
                if let Some(sec) = section {
                    write!(f, "Configuration error in section '{}': parameter '{}' - {}", sec, parameter, reason)
                } else {
                    write!(f, "Configuration error: parameter '{}' - {}", parameter, reason)
                }
            }
            KwaversError::Resource { resource, operation, reason } => {
                write!(f, "Resource error: failed to {} '{}' - {}", operation, resource, reason)
            }
            KwaversError::Numerical { operation, value, constraint } => {
                write!(f, "Numerical error in {}: value {} violates {}", operation, value, constraint)
            }
            KwaversError::Physics { component, state, reason } => {
                write!(f, "Physics error in {}: state '{}' - {}", component, state, reason)
            }
            KwaversError::System { subsystem, operation, reason } => {
                write!(f, "System error in {}: failed to {} - {}", subsystem, operation, reason)
            }
            KwaversError::NotImplemented(feature) => {
                write!(f, "Feature not implemented: {}", feature)
            }
        }
    }
}

impl StdError for KwaversError {}

/// Result type alias for kwavers operations
pub type KwaversResult<T> = Result<T, KwaversError>;

/// Error builder for fluent error construction
pub struct ErrorBuilder {
    error_type: ErrorType,
    field: Option<String>,
    value: Option<String>,
    constraint: Option<String>,
    context: Option<String>,
    parameter: Option<String>,
    reason: Option<String>,
    section: Option<String>,
    resource: Option<String>,
    operation: Option<String>,
    component: Option<String>,
    state: Option<String>,
    subsystem: Option<String>,
    numerical_value: Option<f64>,
}

enum ErrorType {
    Validation,
    Configuration,
    Resource,
    Numerical,
    Physics,
    System,
}

impl ErrorBuilder {
    pub fn validation() -> Self {
        Self {
            error_type: ErrorType::Validation,
            ..Default::default()
        }
    }
    
    pub fn configuration() -> Self {
        Self {
            error_type: ErrorType::Configuration,
            ..Default::default()
        }
    }
    
    pub fn resource() -> Self {
        Self {
            error_type: ErrorType::Resource,
            ..Default::default()
        }
    }
    
    pub fn numerical() -> Self {
        Self {
            error_type: ErrorType::Numerical,
            ..Default::default()
        }
    }
    
    pub fn physics() -> Self {
        Self {
            error_type: ErrorType::Physics,
            ..Default::default()
        }
    }
    
    pub fn system() -> Self {
        Self {
            error_type: ErrorType::System,
            ..Default::default()
        }
    }
    
    pub fn field(mut self, field: impl Into<String>) -> Self {
        self.field = Some(field.into());
        self
    }
    
    pub fn value(mut self, value: impl Into<String>) -> Self {
        self.value = Some(value.into());
        self
    }
    
    pub fn constraint(mut self, constraint: impl Into<String>) -> Self {
        self.constraint = Some(constraint.into());
        self
    }
    
    pub fn context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }
    
    pub fn parameter(mut self, parameter: impl Into<String>) -> Self {
        self.parameter = Some(parameter.into());
        self
    }
    
    pub fn reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }
    
    pub fn section(mut self, section: impl Into<String>) -> Self {
        self.section = Some(section.into());
        self
    }
    
    pub fn resource(mut self, resource: impl Into<String>) -> Self {
        self.resource = Some(resource.into());
        self
    }
    
    pub fn operation(mut self, operation: impl Into<String>) -> Self {
        self.operation = Some(operation.into());
        self
    }
    
    pub fn component(mut self, component: impl Into<String>) -> Self {
        self.component = Some(component.into());
        self
    }
    
    pub fn state(mut self, state: impl Into<String>) -> Self {
        self.state = Some(state.into());
        self
    }
    
    pub fn subsystem(mut self, subsystem: impl Into<String>) -> Self {
        self.subsystem = Some(subsystem.into());
        self
    }
    
    pub fn numerical_value(mut self, value: f64) -> Self {
        self.numerical_value = Some(value);
        self
    }
    
    pub fn build(self) -> KwaversError {
        match self.error_type {
            ErrorType::Validation => KwaversError::Validation {
                field: self.field.unwrap_or_else(|| "unknown".to_string()),
                value: self.value.unwrap_or_else(|| "unknown".to_string()),
                constraint: self.constraint.unwrap_or_else(|| "unknown constraint".to_string()),
                context: self.context,
            },
            ErrorType::Configuration => KwaversError::Configuration {
                parameter: self.parameter.unwrap_or_else(|| "unknown".to_string()),
                reason: self.reason.unwrap_or_else(|| "unknown reason".to_string()),
                section: self.section,
            },
            ErrorType::Resource => KwaversError::Resource {
                resource: self.resource.unwrap_or_else(|| "unknown".to_string()),
                operation: self.operation.unwrap_or_else(|| "access".to_string()),
                reason: self.reason.unwrap_or_else(|| "unknown reason".to_string()),
            },
            ErrorType::Numerical => KwaversError::Numerical {
                operation: self.operation.unwrap_or_else(|| "computation".to_string()),
                value: self.numerical_value.unwrap_or(f64::NAN),
                constraint: self.constraint.unwrap_or_else(|| "unknown constraint".to_string()),
            },
            ErrorType::Physics => KwaversError::Physics {
                component: self.component.unwrap_or_else(|| "unknown".to_string()),
                state: self.state.unwrap_or_else(|| "unknown".to_string()),
                reason: self.reason.unwrap_or_else(|| "unknown reason".to_string()),
            },
            ErrorType::System => KwaversError::System {
                subsystem: self.subsystem.unwrap_or_else(|| "unknown".to_string()),
                operation: self.operation.unwrap_or_else(|| "execute".to_string()),
                reason: self.reason.unwrap_or_else(|| "unknown reason".to_string()),
            },
        }
    }
}

impl Default for ErrorBuilder {
    fn default() -> Self {
        Self {
            error_type: ErrorType::Validation,
            field: None,
            value: None,
            constraint: None,
            context: None,
            parameter: None,
            reason: None,
            section: None,
            resource: None,
            operation: None,
            component: None,
            state: None,
            subsystem: None,
            numerical_value: None,
        }
    }
}

/// Convenience functions for common error patterns
impl KwaversError {
    pub fn validation_failed(field: &str, value: &str, constraint: &str) -> Self {
        ErrorBuilder::validation()
            .field(field)
            .value(value)
            .constraint(constraint)
            .build()
    }
    
    pub fn config_missing(parameter: &str, section: &str) -> Self {
        ErrorBuilder::configuration()
            .parameter(parameter)
            .section(section)
            .reason("parameter is required but not provided")
            .build()
    }
    
    pub fn file_not_found(path: &str) -> Self {
        ErrorBuilder::resource()
            .resource(path)
            .operation("open")
            .reason("file not found")
            .build()
    }
    
    pub fn numerical_instability(operation: &str, value: f64) -> Self {
        ErrorBuilder::numerical()
            .operation(operation)
            .numerical_value(value)
            .constraint("numerical stability")
            .build()
    }
    
    pub fn physics_invalid_state(component: &str, state: &str, reason: &str) -> Self {
        ErrorBuilder::physics()
            .component(component)
            .state(state)
            .reason(reason)
            .build()
    }
    
    pub fn gpu_error(operation: &str, reason: &str) -> Self {
        ErrorBuilder::system()
            .subsystem("GPU")
            .operation(operation)
            .reason(reason)
            .build()
    }
}

// Maintain backward compatibility with existing error types
pub type GridError = KwaversError;
pub type MediumError = KwaversError;
pub type PhysicsError = KwaversError;
pub type DataError = KwaversError;
pub type ConfigError = KwaversError;
pub type NumericalError = KwaversError;
pub type ValidationError = KwaversError;
pub type SystemError = KwaversError;
pub type GpuError = KwaversError;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_error() {
        let error = KwaversError::validation_failed("grid_size", "0", "must be positive");
        let msg = format!("{}", error);
        assert!(msg.contains("grid_size"));
        assert!(msg.contains("0"));
        assert!(msg.contains("must be positive"));
    }

    #[test]
    fn test_error_builder() {
        let error = ErrorBuilder::physics()
            .component("NonlinearWave")
            .state("unstable")
            .reason("pressure exceeded maximum threshold")
            .build();
        
        match error {
            KwaversError::Physics { component, state, reason } => {
                assert_eq!(component, "NonlinearWave");
                assert_eq!(state, "unstable");
                assert_eq!(reason, "pressure exceeded maximum threshold");
            }
            _ => panic!("Wrong error type"),
        }
    }
}