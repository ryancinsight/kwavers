// src/error.rs
//! Comprehensive error handling system for kwavers
//!
//! This module implements a sophisticated error handling system following
//! multiple design principles:
//!
//! Design Principles Implemented:
//! - SOLID: Single responsibility for each error type, open/closed for extensibility
//! - CUPID: Composable error handling, predictable error behavior
//! - GRASP: Information expert for error context, controller for error flow
//! - ACID: Atomic error operations, consistent error states
//! - DRY: Shared error patterns and utilities
//! - KISS: Simple, clear error messages
//! - YAGNI: Only implement necessary error types
//! - SSOT: Single source of truth for error definitions
//! - CCP: Common closure for related error handling
//! - CRP: Common reuse of error utilities
//! - ADP: Acyclic dependency in error hierarchy

use std::error::Error as StdError;
use std::fmt;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Main error type for kwavers operations
/// 
/// Implements SSOT principle as the single source of truth for all errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KwaversError {
    /// Grid-related errors
    Grid(GridError),
    /// Medium-related errors
    Medium(MediumError),
    /// Physics simulation errors
    Physics(PhysicsError),
    /// Data I/O and format errors
    Data(DataError),
    /// Configuration errors
    Config(ConfigError),
    /// Numerical computation errors
    Numerical(NumericalError),
    /// Validation errors
    Validation(ValidationError),
    /// System errors (memory, threading, etc.)
    System(SystemError),
    /// GPU acceleration errors
    Gpu(GpuError),
    /// Composite error with multiple underlying errors
    Composite(CompositeError),
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
            KwaversError::Validation(e) => write!(f, "Validation error: {}", e),
            KwaversError::System(e) => write!(f, "System error: {}", e),
            KwaversError::Gpu(e) => write!(f, "GPU error: {}", e),
            KwaversError::Composite(e) => write!(f, "Composite error: {}", e),
        }
    }
}

impl StdError for KwaversError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            KwaversError::Grid(e) => Some(e),
            KwaversError::Medium(e) => Some(e),
            KwaversError::Physics(e) => Some(e),
            KwaversError::Data(e) => Some(e),
            KwaversError::Config(e) => Some(e),
            KwaversError::Numerical(e) => Some(e),
            KwaversError::Validation(e) => Some(e),
            KwaversError::System(e) => Some(e),
            KwaversError::Gpu(e) => Some(e),
            KwaversError::Composite(e) => Some(e),
        }
    }
}

/// Grid-related errors
/// 
/// Implements Single Responsibility Principle - only handles grid-specific errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GridError {
    /// Invalid grid dimensions
    InvalidDimensions {
        nx: usize,
        ny: usize,
        nz: usize,
        reason: String,
    },
    /// Invalid grid spacing
    InvalidSpacing {
        dx: f64,
        dy: f64,
        dz: f64,
        reason: String,
    },
    /// Grid size too large for available memory
    OutOfMemory {
        required_bytes: usize,
        available_bytes: usize,
    },
    /// Invalid grid index access
    IndexOutOfBounds {
        index: (usize, usize, usize),
        dimensions: (usize, usize, usize),
    },
    /// Grid not properly initialized
    NotInitialized,
    /// Grid validation failed
    ValidationFailed {
        field: String,
        value: String,
        constraint: String,
    },
}

impl fmt::Display for GridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GridError::InvalidDimensions { nx, ny, nz, reason } => {
                write!(f, "Invalid grid dimensions ({}, {}, {}): {}", nx, ny, nz, reason)
            }
            GridError::InvalidSpacing { dx, dy, dz, reason } => {
                write!(f, "Invalid grid spacing ({}, {}, {}): {}", dx, dy, dz, reason)
            }
            GridError::OutOfMemory { required_bytes, available_bytes } => {
                write!(f, "Grid requires {} bytes but only {} available", required_bytes, available_bytes)
            }
            GridError::IndexOutOfBounds { index, dimensions } => {
                write!(f, "Index {:?} out of bounds for dimensions {:?}", index, dimensions)
            }
            GridError::NotInitialized => {
                write!(f, "Grid not properly initialized")
            }
            GridError::ValidationFailed { field, value, constraint } => {
                write!(f, "Grid validation failed: {} = {} violates {}", field, value, constraint)
            }
        }
    }
}

impl StdError for GridError {}

/// Medium-related errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MediumError {
    /// Invalid medium properties
    InvalidProperties {
        property: String,
        value: f64,
        constraint: String,
    },
    /// Medium not found
    NotFound {
        medium_name: String,
    },
    /// Medium initialization failed
    InitializationFailed {
        reason: String,
    },
    /// Medium validation failed
    ValidationFailed {
        field: String,
        value: String,
        constraint: String,
    },
    /// Incompatible medium types
    IncompatibleTypes {
        type1: String,
        type2: String,
        reason: String,
    },
}

impl fmt::Display for MediumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MediumError::InvalidProperties { property, value, constraint } => {
                write!(f, "Invalid medium property {} = {}: {}", property, value, constraint)
            }
            MediumError::NotFound { medium_name } => {
                write!(f, "Medium '{}' not found", medium_name)
            }
            MediumError::InitializationFailed { reason } => {
                write!(f, "Medium initialization failed: {}", reason)
            }
            MediumError::ValidationFailed { field, value, constraint } => {
                write!(f, "Medium validation failed: {} = {} violates {}", field, value, constraint)
            }
            MediumError::IncompatibleTypes { type1, type2, reason } => {
                write!(f, "Incompatible medium types {} and {}: {}", type1, type2, reason)
            }
        }
    }
}

impl StdError for MediumError {}

/// Physics simulation errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PhysicsError {
    /// Model not initialized
    ModelNotInitialized {
        model_name: String,
    },
    /// Incompatible models
    IncompatibleModels {
        model1: String,
        model2: String,
        reason: String,
    },
    /// Invalid configuration
    InvalidConfiguration {
        component: String,
        reason: String,
    },
    /// Simulation instability
    Instability {
        field: String,
        location: (usize, usize, usize),
        value: f64,
        threshold: f64,
    },
    /// Convergence failure
    ConvergenceFailure {
        iteration: usize,
        max_iterations: usize,
        residual: f64,
        tolerance: f64,
    },
    /// Invalid state
    InvalidState {
        expected: String,
        actual: String,
    },
    /// Time step too large
    TimeStepTooLarge {
        dt: f64,
        max_dt: f64,
        reason: String,
    },
    /// General simulation error
    SimulationError {
        message: String,
    },
}

impl fmt::Display for PhysicsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PhysicsError::ModelNotInitialized { model_name } => {
                write!(f, "Model '{}' not initialized", model_name)
            }
            PhysicsError::IncompatibleModels { model1, model2, reason } => {
                write!(f, "Incompatible models '{}' and '{}': {}", model1, model2, reason)
            }
            PhysicsError::InvalidConfiguration { component, reason } => {
                write!(f, "Invalid configuration for component '{}': {}", component, reason)
            }
            PhysicsError::Instability { field, location, value, threshold } => {
                write!(f, "Instability in field '{}' at {:?}: {} > {}", field, location, value, threshold)
            }
            PhysicsError::ConvergenceFailure { iteration, max_iterations, residual, tolerance } => {
                write!(f, "Convergence failed after {} iterations (max {}): residual {} > tolerance {}", 
                       iteration, max_iterations, residual, tolerance)
            }
            PhysicsError::InvalidState { expected, actual } => {
                write!(f, "Invalid state: expected {}, got {}", expected, actual)
            }
            PhysicsError::TimeStepTooLarge { dt, max_dt, reason } => {
                write!(f, "Time step {} too large (max {}): {}", dt, max_dt, reason)
            }
            PhysicsError::SimulationError { message } => {
                write!(f, "Simulation error: {}", message)
            }
        }
    }
}

impl StdError for PhysicsError {}

/// Data I/O and format errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataError {
    /// File not found
    FileNotFound {
        path: String,
    },
    /// File read error
    ReadError {
        path: String,
        reason: String,
    },
    /// File write error
    WriteError {
        path: String,
        reason: String,
    },
    /// Invalid data format
    InvalidFormat {
        format: String,
        reason: String,
    },
    /// Data corruption
    Corruption {
        location: String,
        reason: String,
    },
    /// Insufficient data
    InsufficientData {
        required: usize,
        available: usize,
    },
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataError::FileNotFound { path } => {
                write!(f, "File not found: {}", path)
            }
            DataError::ReadError { path, reason } => {
                write!(f, "Failed to read file '{}': {}", path, reason)
            }
            DataError::WriteError { path, reason } => {
                write!(f, "Failed to write file '{}': {}", path, reason)
            }
            DataError::InvalidFormat { format, reason } => {
                write!(f, "Invalid format '{}': {}", format, reason)
            }
            DataError::Corruption { location, reason } => {
                write!(f, "Data corruption at {}: {}", location, reason)
            }
            DataError::InsufficientData { required, available } => {
                write!(f, "Insufficient data: required {}, available {}", required, available)
            }
        }
    }
}

impl StdError for DataError {}

/// Configuration errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigError {
    /// Missing required parameter
    MissingParameter {
        parameter: String,
        section: String,
    },
    /// Invalid parameter value
    InvalidValue {
        parameter: String,
        value: String,
        constraint: String,
    },
    /// Configuration file not found
    FileNotFound {
        path: String,
    },
    /// Configuration parse error
    ParseError {
        line: usize,
        column: usize,
        reason: String,
    },
    /// Configuration validation failed
    ValidationFailed {
        section: String,
        reason: String,
    },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::MissingParameter { parameter, section } => {
                write!(f, "Missing required parameter '{}' in section '{}'", parameter, section)
            }
            ConfigError::InvalidValue { parameter, value, constraint } => {
                write!(f, "Invalid value for parameter '{}': {} violates {}", parameter, value, constraint)
            }
            ConfigError::FileNotFound { path } => {
                write!(f, "Configuration file not found: {}", path)
            }
            ConfigError::ParseError { line, column, reason } => {
                write!(f, "Configuration parse error at line {}, column {}: {}", line, column, reason)
            }
            ConfigError::ValidationFailed { section, reason } => {
                write!(f, "Configuration validation failed in section '{}': {}", section, reason)
            }
        }
    }
}

impl StdError for ConfigError {}

/// Numerical computation errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NumericalError {
    /// Division by zero
    DivisionByZero {
        operation: String,
        location: String,
    },
    /// Overflow
    Overflow {
        operation: String,
        value: f64,
        limit: f64,
    },
    /// Underflow
    Underflow {
        operation: String,
        value: f64,
        limit: f64,
    },
    /// NaN result
    NaN {
        operation: String,
        inputs: Vec<f64>,
    },
    /// Infinite result
    Infinite {
        operation: String,
        inputs: Vec<f64>,
    },
    /// Numerical instability
    Instability {
        operation: String,
        condition: String,
    },
}

impl fmt::Display for NumericalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NumericalError::DivisionByZero { operation, location } => {
                write!(f, "Division by zero in {} at {}", operation, location)
            }
            NumericalError::Overflow { operation, value, limit } => {
                write!(f, "Overflow in {}: {} > {}", operation, value, limit)
            }
            NumericalError::Underflow { operation, value, limit } => {
                write!(f, "Underflow in {}: {} < {}", operation, value, limit)
            }
            NumericalError::NaN { operation, inputs } => {
                write!(f, "NaN result in {} with inputs {:?}", operation, inputs)
            }
            NumericalError::Infinite { operation, inputs } => {
                write!(f, "Infinite result in {} with inputs {:?}", operation, inputs)
            }
            NumericalError::Instability { operation, condition } => {
                write!(f, "Numerical instability in {}: {}", operation, condition)
            }
        }
    }
}

impl StdError for NumericalError {}

/// Validation errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationError {
    /// Field validation failed
    FieldValidation {
        field: String,
        value: String,
        constraint: String,
    },
    /// Range validation failed
    RangeValidation {
        field: String,
        value: f64,
        min: f64,
        max: f64,
    },
    /// Type validation failed
    TypeValidation {
        field: String,
        expected_type: String,
        actual_type: String,
    },
    /// Dependency validation failed
    DependencyValidation {
        component: String,
        missing_dependency: String,
    },
    /// State validation failed
    StateValidation {
        component: String,
        expected_state: String,
        actual_state: String,
    },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::FieldValidation { field, value, constraint } => {
                write!(f, "Field validation failed: {} = {} violates {}", field, value, constraint)
            }
            ValidationError::RangeValidation { field, value, min, max } => {
                write!(f, "Range validation failed: {} = {} not in [{}, {}]", field, value, min, max)
            }
            ValidationError::TypeValidation { field, expected_type, actual_type } => {
                write!(f, "Type validation failed: {} expected {}, got {}", field, expected_type, actual_type)
            }
            ValidationError::DependencyValidation { component, missing_dependency } => {
                write!(f, "Dependency validation failed: component '{}' missing dependency '{}'", 
                       component, missing_dependency)
            }
            ValidationError::StateValidation { component, expected_state, actual_state } => {
                write!(f, "State validation failed: component '{}' expected {}, got {}", 
                       component, expected_state, actual_state)
            }
        }
    }
}

impl StdError for ValidationError {}

/// System errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemError {
    /// Memory allocation failed
    MemoryAllocation {
        requested_bytes: usize,
        reason: String,
    },
    /// Thread creation failed
    ThreadCreation {
        reason: String,
    },
    /// Thread synchronization failed
    ThreadSync {
        operation: String,
        reason: String,
    },
    /// System resource exhausted
    ResourceExhausted {
        resource: String,
        reason: String,
    },
    /// System call failed
    SystemCall {
        call: String,
        reason: String,
    },
}

impl fmt::Display for SystemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SystemError::MemoryAllocation { requested_bytes, reason } => {
                write!(f, "Memory allocation failed for {} bytes: {}", requested_bytes, reason)
            }
            SystemError::ThreadCreation { reason } => {
                write!(f, "Thread creation failed: {}", reason)
            }
            SystemError::ThreadSync { operation, reason } => {
                write!(f, "Thread synchronization failed in {}: {}", operation, reason)
            }
            SystemError::ResourceExhausted { resource, reason } => {
                write!(f, "System resource '{}' exhausted: {}", resource, reason)
            }
            SystemError::SystemCall { call, reason } => {
                write!(f, "System call '{}' failed: {}", call, reason)
            }
        }
    }
}

impl StdError for SystemError {}

/// GPU acceleration errors
/// 
/// Implements SOLID principles with specific error types for GPU operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuError {
    /// No GPU devices found
    NoDevicesFound,
    /// GPU device initialization failed
    DeviceInitialization {
        device_id: u32,
        reason: String,
    },
    /// GPU memory allocation failed
    MemoryAllocation {
        requested_bytes: usize,
        available_bytes: usize,
        reason: String,
    },
    /// GPU memory transfer failed
    MemoryTransfer {
        direction: String,
        size_bytes: usize,
        reason: String,
    },
    /// GPU kernel compilation failed
    KernelCompilation {
        kernel_name: String,
        reason: String,
    },
    /// GPU kernel execution failed
    KernelExecution {
        kernel_name: String,
        reason: String,
    },
    /// GPU backend not available
    BackendNotAvailable {
        backend: String,
        reason: String,
    },
    /// GPU performance below threshold
    PerformanceThreshold {
        actual_performance: f64,
        required_performance: f64,
        metric: String,
    },
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::NoDevicesFound => {
                write!(f, "No GPU devices found")
            }
            GpuError::DeviceInitialization { device_id, reason } => {
                write!(f, "GPU device {} initialization failed: {}", device_id, reason)
            }
            GpuError::MemoryAllocation { requested_bytes, available_bytes, reason } => {
                write!(f, "GPU memory allocation failed: requested {} bytes, available {} bytes: {}", 
                       requested_bytes, available_bytes, reason)
            }
            GpuError::MemoryTransfer { direction, size_bytes, reason } => {
                write!(f, "GPU memory transfer ({}) failed for {} bytes: {}", 
                       direction, size_bytes, reason)
            }
            GpuError::KernelCompilation { kernel_name, reason } => {
                write!(f, "GPU kernel '{}' compilation failed: {}", kernel_name, reason)
            }
            GpuError::KernelExecution { kernel_name, reason } => {
                write!(f, "GPU kernel '{}' execution failed: {}", kernel_name, reason)
            }
            GpuError::BackendNotAvailable { backend, reason } => {
                write!(f, "GPU backend '{}' not available: {}", backend, reason)
            }
            GpuError::PerformanceThreshold { actual_performance, required_performance, metric } => {
                write!(f, "GPU performance below threshold: {} = {:.2}, required {:.2}", 
                       metric, actual_performance, required_performance)
            }
        }
    }
}

impl StdError for GpuError {}

/// Composite error for multiple underlying errors
/// 
/// Implements CCP (Common Closure Principle) by grouping related errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeError {
    pub errors: Vec<KwaversError>,
    pub context: String,
}

impl fmt::Display for CompositeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Composite error in {}: ", self.context)?;
        for (i, error) in self.errors.iter().enumerate() {
            if i > 0 {
                write!(f, "; ")?;
            }
            write!(f, "{}", error)?;
        }
        Ok(())
    }
}

impl StdError for CompositeError {}

/// Error context for better debugging
/// 
/// Implements Information Expert principle by providing detailed error context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub location: String,
    pub stack_trace: Vec<String>,
    pub additional_info: HashMap<String, String>,
}

impl ErrorContext {
    pub fn new(location: String) -> Self {
        Self {
            timestamp: chrono::Utc::now(),
            location,
            stack_trace: Vec::new(),
            additional_info: HashMap::new(),
        }
    }
    
    pub fn with_info(mut self, key: String, value: String) -> Self {
        self.additional_info.insert(key, value);
        self
    }
    
    pub fn add_stack_frame(mut self, frame: String) -> Self {
        self.stack_trace.push(frame);
        self
    }
}

/// Error recovery strategies
/// 
/// Implements CRP (Common Reuse Principle) by providing reusable recovery strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    /// Retry the operation
    Retry {
        max_attempts: usize,
        delay_ms: u64,
    },
    /// Use fallback value
    Fallback {
        value: String,
        reason: String,
    },
    /// Skip the operation
    Skip {
        reason: String,
    },
    /// Abort the operation
    Abort {
        reason: String,
    },
    /// Degrade gracefully
    Degrade {
        mode: String,
        reason: String,
    },
}

/// Enhanced error with context and recovery information
/// 
/// Implements ACID principles for error handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedError {
    pub error: KwaversError,
    pub context: ErrorContext,
    pub recovery_strategy: Option<RecoveryStrategy>,
    pub severity: ErrorSeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Debug,
    Info,
    Warning,
    Error,
    Critical,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorSeverity::Debug => write!(f, "DEBUG"),
            ErrorSeverity::Info => write!(f, "INFO"),
            ErrorSeverity::Warning => write!(f, "WARNING"),
            ErrorSeverity::Error => write!(f, "ERROR"),
            ErrorSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

impl fmt::Display for EnhancedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {} at {}: {}", 
               self.severity, self.error, self.context.location, self.context.timestamp)
    }
}

impl StdError for EnhancedError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(&self.error)
    }
}

/// Error builder for fluent error construction
/// 
/// Implements Builder pattern for complex error creation
pub struct ErrorBuilder {
    error: Option<KwaversError>,
    context: ErrorContext,
    recovery_strategy: Option<RecoveryStrategy>,
    severity: ErrorSeverity,
}

impl ErrorBuilder {
    pub fn new(location: String) -> Self {
        Self {
            error: None,
            context: ErrorContext::new(location),
            recovery_strategy: None,
            severity: ErrorSeverity::Error,
        }
    }
    
    pub fn with_error(mut self, error: KwaversError) -> Self {
        self.error = Some(error);
        self
    }
    
    pub fn with_info(mut self, key: String, value: String) -> Self {
        self.context = self.context.with_info(key, value);
        self
    }
    
    pub fn with_recovery(mut self, strategy: RecoveryStrategy) -> Self {
        self.recovery_strategy = Some(strategy);
        self
    }
    
    pub fn with_severity(mut self, severity: ErrorSeverity) -> Self {
        self.severity = severity;
        self
    }
    
    pub fn build(self) -> Result<EnhancedError, String> {
        let error = self.error.ok_or("No error specified")?;
        Ok(EnhancedError {
            error,
            context: self.context,
            recovery_strategy: self.recovery_strategy,
            severity: self.severity,
        })
    }
}

/// Result type alias for kwavers operations
pub type KwaversResult<T> = Result<T, KwaversError>;

/// Enhanced result type with context
pub type EnhancedResult<T> = Result<T, EnhancedError>;

// Automatic conversions for easier error handling
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

impl From<ValidationError> for KwaversError {
    fn from(err: ValidationError) -> Self {
        KwaversError::Validation(err)
    }
}

impl From<SystemError> for KwaversError {
    fn from(err: SystemError) -> Self {
        KwaversError::System(err)
    }
}

impl From<CompositeError> for KwaversError {
    fn from(err: CompositeError) -> Self {
        KwaversError::Composite(err)
    }
}

/// Error utilities for common operations
/// 
/// Implements DRY principle by providing reusable error utilities
pub mod utils {
    use super::*;
    
    /// Create a grid error with context
    pub fn grid_error(location: &str, error: GridError) -> EnhancedError {
        ErrorBuilder::new(location.to_string())
            .with_error(KwaversError::Grid(error))
            .build()
            .unwrap_or_else(|_| panic!("Failed to build error"))
    }
    
    /// Create a physics error with context
    pub fn physics_error(location: &str, error: PhysicsError) -> EnhancedError {
        ErrorBuilder::new(location.to_string())
            .with_error(KwaversError::Physics(error))
            .build()
            .unwrap_or_else(|_| panic!("Failed to build error"))
    }
    
    /// Create a validation error with context
    pub fn validation_error(location: &str, error: ValidationError) -> EnhancedError {
        ErrorBuilder::new(location.to_string())
            .with_error(KwaversError::Validation(error))
            .build()
            .unwrap_or_else(|_| panic!("Failed to build error"))
    }
    
    /// Check if an error is recoverable
    pub fn is_recoverable(error: &KwaversError) -> bool {
        matches!(error,
            KwaversError::Validation(_) |
            KwaversError::Config(_) |
            KwaversError::Data(_)
        )
    }
    
    /// Get error severity
    pub fn get_severity(error: &KwaversError) -> ErrorSeverity {
        match error {
            KwaversError::System(_) => ErrorSeverity::Critical,
            KwaversError::Numerical(_) => ErrorSeverity::Error,
            KwaversError::Physics(_) => ErrorSeverity::Error,
            KwaversError::Grid(_) => ErrorSeverity::Error,
            KwaversError::Medium(_) => ErrorSeverity::Error,
            KwaversError::Data(_) => ErrorSeverity::Warning,
            KwaversError::Config(_) => ErrorSeverity::Warning,
            KwaversError::Validation(_) => ErrorSeverity::Info,
            KwaversError::Gpu(_) => ErrorSeverity::Error,
            KwaversError::Composite(_) => ErrorSeverity::Error,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let grid_error = GridError::InvalidDimensions {
            nx: 0,
            ny: 0,
            nz: 0,
            reason: "All dimensions must be positive".to_string(),
        };
        
        let kwavers_error = KwaversError::Grid(grid_error);
        assert!(kwavers_error.to_string().contains("Invalid grid dimensions"));
    }
    
    #[test]
    fn test_error_conversion() {
        let grid_error = GridError::NotInitialized;
        let kwavers_error: KwaversError = grid_error.into();
        
        match kwavers_error {
            KwaversError::Grid(_) => (),
            _ => panic!("Expected Grid error"),
        }
    }
    
    #[test]
    fn test_error_builder() {
        let enhanced_error = ErrorBuilder::new("test_location".to_string())
            .with_error(KwaversError::Grid(GridError::NotInitialized))
            .with_info("test_key".to_string(), "test_value".to_string())
            .with_severity(ErrorSeverity::Warning)
            .build()
            .unwrap();
        
        assert_eq!(enhanced_error.severity, ErrorSeverity::Warning);
        assert_eq!(enhanced_error.context.additional_info.get("test_key"), Some(&"test_value".to_string()));
    }
    
    #[test]
    fn test_error_utilities() {
        let grid_error = GridError::NotInitialized;
        let enhanced_error = utils::grid_error("test", grid_error);
        
        assert_eq!(enhanced_error.context.location, "test");
        assert!(matches!(enhanced_error.error, KwaversError::Grid(_)));
    }
    
    #[test]
    fn test_error_severity() {
        let system_error = KwaversError::System(SystemError::MemoryAllocation {
            requested_bytes: 1024,
            reason: "test".to_string(),
        });
        
        assert_eq!(utils::get_severity(&system_error), ErrorSeverity::Critical);
    }
    
    #[test]
    fn test_recoverable_errors() {
        let validation_error = KwaversError::Validation(ValidationError::FieldValidation {
            field: "test".to_string(),
            value: "test".to_string(),
            constraint: "test".to_string(),
        });
        
        assert!(utils::is_recoverable(&validation_error));
    }
}