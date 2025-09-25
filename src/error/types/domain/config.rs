//! Configuration-specific error types
//!
//! Specialized error handling for configuration validation

use thiserror::Error;

/// Configuration error types with detailed context
#[derive(Error, Debug, Clone)]
pub enum ConfigErrorType {
    #[error("Invalid configuration value: {parameter} = {value}, constraint: {constraint}")]
    InvalidValue {
        parameter: String,
        value: String,
        constraint: String,
    },
    
    #[error("Missing required configuration: {parameter} is required but not provided")]
    MissingRequired { parameter: String },
    
    #[error("Configuration conflict: {parameter1} = {value1} conflicts with {parameter2} = {value2}")]
    ParameterConflict {
        parameter1: String,
        value1: String,
        parameter2: String,
        value2: String,
    },
    
    #[error("Configuration parsing failed: {source}")]
    ParseError { source: String },
    
    #[error("Configuration file not found: {path}")]
    FileNotFound { path: String },
    
    #[error("Configuration validation failed: {reason}")]
    ValidationFailed { reason: String },
}