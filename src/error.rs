// src/error.rs
//! Error handling for the kwavers simulation framework
//! 
//! This module provides a comprehensive error handling system following SOLID principles:
//! - Single Responsibility: Each error type has a single, clear purpose
//! - Open/Closed: Easy to extend with new error types without modifying existing code
//! - Liskov Substitution: All error types can be used interchangeably where Result<T, KwaversError> is expected
//! - Interface Segregation: Specific error types for different domains
//! - Dependency Inversion: High-level modules depend on error abstractions, not concrete implementations

use std::fmt;

/// Main error type for the kwavers simulation framework
#[derive(Debug, Clone)]
pub enum KwaversError {
    /// Grid-related errors
    Grid(GridError),
    /// Medium property errors
    Medium(MediumError),
    /// Physics simulation errors
    Physics(PhysicsError),
    /// I/O and data errors
    Data(DataError),
    /// Configuration errors
    Config(ConfigError),
    /// Numerical computation errors
    Numerical(NumericalError),
}

/// Grid configuration and validation errors
#[derive(Debug, Clone)]
pub enum GridError {
    InvalidDimensions { nx: usize, ny: usize, nz: usize },
    InvalidSpacing { dx: f64, dy: f64, dz: f64 },
    OutOfBounds { x: f64, y: f64, z: f64 },
    SizeMismatch { expected: (usize, usize, usize), actual: (usize, usize, usize) },
}

/// Medium property validation errors
#[derive(Debug, Clone)]
pub enum MediumError {
    InvalidProperty { property: String, value: f64, expected_range: (f64, f64) },
    MissingProperty { property: String },
    InconsistentProperties { description: String },
    TissueTypeNotFound { tissue_type: String },
}

/// Physics simulation errors
#[derive(Debug, Clone)]
pub enum PhysicsError {
    InstabilityDetected { time: f64, max_pressure: f64 },
    ConvergenceFailure { iterations: usize, residual: f64 },
    InvalidTimeStep { dt: f64, recommended_max: f64 },
    ModelNotInitialized { model_name: String },
    IncompatibleModels { model1: String, model2: String, reason: String },
}

/// Data I/O and format errors
#[derive(Debug, Clone)]
pub enum DataError {
    FileNotFound { path: String },
    InvalidFormat { expected: String, found: String },
    SerializationFailed { reason: String },
    DeserializationFailed { reason: String },
    InsufficientData { required: usize, available: usize },
}

/// Configuration validation errors
#[derive(Debug, Clone)]
pub enum ConfigError {
    MissingParameter { parameter: String },
    InvalidValue { parameter: String, value: String, reason: String },
    ConflictingSettings { setting1: String, setting2: String, reason: String },
    UnsupportedFeature { feature: String },
}

/// Numerical computation errors
#[derive(Debug, Clone)]
pub enum NumericalError {
    DivisionByZero { context: String },
    NumericalOverflow { value: f64, context: String },
    NumericalUnderflow { value: f64, context: String },
    MatrixSingular { matrix_name: String },
    FFTError { size: (usize, usize, usize), reason: String },
    IterationLimitExceeded { max_iterations: usize, operation: String },
}

impl fmt::Display for KwaversError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KwaversError::Grid(e) => write!(f, "Grid error: {}", e),
            KwaversError::Medium(e) => write!(f, "Medium error: {}", e),
            KwaversError::Physics(e) => write!(f, "Physics error: {}", e),
            KwaversError::Data(e) => write!(f, "Data error: {}", e),
            KwaversError::Config(e) => write!(f, "Configuration error: {}", e),
            KwaversError::Numerical(e) => write!(f, "Numerical error: {}", e),
        }
    }
}

impl fmt::Display for GridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GridError::InvalidDimensions { nx, ny, nz } => {
                write!(f, "Invalid grid dimensions: {}x{}x{} (all must be > 0)", nx, ny, nz)
            }
            GridError::InvalidSpacing { dx, dy, dz } => {
                write!(f, "Invalid grid spacing: dx={}, dy={}, dz={} (all must be > 0)", dx, dy, dz)
            }
            GridError::OutOfBounds { x, y, z } => {
                write!(f, "Coordinates ({}, {}, {}) are outside grid bounds", x, y, z)
            }
            GridError::SizeMismatch { expected, actual } => {
                write!(f, "Array size mismatch: expected {:?}, got {:?}", expected, actual)
            }
        }
    }
}

impl fmt::Display for MediumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MediumError::InvalidProperty { property, value, expected_range } => {
                write!(f, "Invalid {}: {} (expected range: {:?})", property, value, expected_range)
            }
            MediumError::MissingProperty { property } => {
                write!(f, "Missing required property: {}", property)
            }
            MediumError::InconsistentProperties { description } => {
                write!(f, "Inconsistent medium properties: {}", description)
            }
            MediumError::TissueTypeNotFound { tissue_type } => {
                write!(f, "Unknown tissue type: {}", tissue_type)
            }
        }
    }
}

impl fmt::Display for PhysicsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PhysicsError::InstabilityDetected { time, max_pressure } => {
                write!(f, "Numerical instability at t={}: max pressure = {}", time, max_pressure)
            }
            PhysicsError::ConvergenceFailure { iterations, residual } => {
                write!(f, "Failed to converge after {} iterations (residual: {})", iterations, residual)
            }
            PhysicsError::InvalidTimeStep { dt, recommended_max } => {
                write!(f, "Time step {} too large (recommended max: {})", dt, recommended_max)
            }
            PhysicsError::ModelNotInitialized { model_name } => {
                write!(f, "Model '{}' not properly initialized", model_name)
            }
            PhysicsError::IncompatibleModels { model1, model2, reason } => {
                write!(f, "Models '{}' and '{}' are incompatible: {}", model1, model2, reason)
            }
        }
    }
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataError::FileNotFound { path } => write!(f, "File not found: {}", path),
            DataError::InvalidFormat { expected, found } => {
                write!(f, "Invalid format: expected {}, found {}", expected, found)
            }
            DataError::SerializationFailed { reason } => {
                write!(f, "Serialization failed: {}", reason)
            }
            DataError::DeserializationFailed { reason } => {
                write!(f, "Deserialization failed: {}", reason)
            }
            DataError::InsufficientData { required, available } => {
                write!(f, "Insufficient data: need {}, have {}", required, available)
            }
        }
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::MissingParameter { parameter } => {
                write!(f, "Missing required parameter: {}", parameter)
            }
            ConfigError::InvalidValue { parameter, value, reason } => {
                write!(f, "Invalid value for {}: '{}' ({})", parameter, value, reason)
            }
            ConfigError::ConflictingSettings { setting1, setting2, reason } => {
                write!(f, "Conflicting settings '{}' and '{}': {}", setting1, setting2, reason)
            }
            ConfigError::UnsupportedFeature { feature } => {
                write!(f, "Unsupported feature: {}", feature)
            }
        }
    }
}

impl fmt::Display for NumericalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NumericalError::DivisionByZero { context } => {
                write!(f, "Division by zero in {}", context)
            }
            NumericalError::NumericalOverflow { value, context } => {
                write!(f, "Numerical overflow: {} in {}", value, context)
            }
            NumericalError::NumericalUnderflow { value, context } => {
                write!(f, "Numerical underflow: {} in {}", value, context)
            }
            NumericalError::MatrixSingular { matrix_name } => {
                write!(f, "Singular matrix: {}", matrix_name)
            }
            NumericalError::FFTError { size, reason } => {
                write!(f, "FFT error for size {:?}: {}", size, reason)
            }
            NumericalError::IterationLimitExceeded { max_iterations, operation } => {
                write!(f, "Iteration limit ({}) exceeded for {}", max_iterations, operation)
            }
        }
    }
}

impl std::error::Error for KwaversError {}
impl std::error::Error for GridError {}
impl std::error::Error for MediumError {}
impl std::error::Error for PhysicsError {}
impl std::error::Error for DataError {}
impl std::error::Error for ConfigError {}
impl std::error::Error for NumericalError {}

// From implementations for automatic error conversion
impl From<GridError> for KwaversError {
    fn from(err: GridError) -> Self {
        KwaversError::Grid(err)
    }
}

impl From<MediumError> for KwaversError {
    fn from(err: MediumError) -> Self {
        KwaversError::Medium(err)
    }
}

impl From<PhysicsError> for KwaversError {
    fn from(err: PhysicsError) -> Self {
        KwaversError::Physics(err)
    }
}

impl From<DataError> for KwaversError {
    fn from(err: DataError) -> Self {
        KwaversError::Data(err)
    }
}

impl From<ConfigError> for KwaversError {
    fn from(err: ConfigError) -> Self {
        KwaversError::Config(err)
    }
}

impl From<NumericalError> for KwaversError {
    fn from(err: NumericalError) -> Self {
        KwaversError::Numerical(err)
    }
}

/// Convenience type alias for Results in kwavers
pub type KwaversResult<T> = Result<T, KwaversError>;

/// Trait for validating simulation parameters
/// Follows Interface Segregation Principle - specific validation interfaces
pub trait Validate {
    type Error;
    
    /// Validate the object's current state
    fn validate(&self) -> Result<(), Self::Error>;
    
    /// Validate and provide suggestions for fixing issues
    fn validate_with_suggestions(&self) -> Result<(), (Self::Error, Vec<String>)> {
        self.validate().map_err(|e| (e, vec![]))
    }
}

/// Trait for objects that can be reset to a safe state
/// Follows Single Responsibility Principle
pub trait Resettable {
    /// Reset to a safe, default state
    fn reset(&mut self);
    
    /// Reset with specific parameters
    fn reset_with_params(&mut self, params: &std::collections::HashMap<String, f64>);
}

/// Trait for performance monitoring
/// Follows Interface Segregation Principle
pub trait PerformanceMonitor {
    /// Get performance metrics
    fn get_metrics(&self) -> std::collections::HashMap<String, f64>;
    
    /// Reset performance counters
    fn reset_metrics(&mut self);
    
    /// Check if performance is within acceptable bounds
    fn check_performance(&self) -> Result<(), PhysicsError>;
}