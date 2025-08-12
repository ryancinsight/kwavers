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
    /// Visualization and rendering errors
    Visualization(String),
    /// Feature not yet implemented
    NotImplemented(String),
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
            KwaversError::Visualization(e) => write!(f, "Visualization error: {}", e),
            KwaversError::Composite(e) => write!(f, "Composite error: {}", e),
            KwaversError::NotImplemented(e) => write!(f, "Feature not yet implemented: {}", e),
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
            KwaversError::Visualization(_) => None,
            KwaversError::Composite(e) => Some(e),
            KwaversError::NotImplemented(_) => None,
        }
    }
}

impl KwaversError {
    /// Helper to create a field validation error
    pub fn field_validation(field: &str, value: impl ToString, constraint: &str) -> Self {
        KwaversError::Validation(ValidationError::FieldValidation {
            field: field.to_string(),
            value: value.to_string(),
            constraint: constraint.to_string(),
        })
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
        model: String,
        reason: String,
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
    /// Invalid state
    InvalidState {
        field: String,
        value: String,
        reason: String,
    },
    /// Simulation instability
    Instability {
        field: String,
        location: (usize, usize, usize),
        value: f64,
    },
    /// Conservation violation
    ConservationViolation {
        quantity: String,
        error: f64,
        tolerance: f64,
    },
    /// Convergence failure
    ConvergenceFailure {
        solver: String,
        iterations: usize,
        residual: f64,
    },
    /// Invalid field index
    InvalidFieldIndex(usize),
    /// State error
    StateError(String),
    /// Dimension mismatch
    DimensionMismatch,
    /// Unauthorized field access
    UnauthorizedFieldAccess {
        field: String,
        operation: String,
    },
}

impl fmt::Display for PhysicsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PhysicsError::ModelNotInitialized { model, reason } => {
                write!(f, "Model '{}' not initialized: {}", model, reason)
            }
            PhysicsError::IncompatibleModels { model1, model2, reason } => {
                write!(f, "Incompatible models '{}' and '{}': {}", model1, model2, reason)
            }
            PhysicsError::InvalidConfiguration { component, reason } => {
                write!(f, "Invalid configuration for component '{}': {}", component, reason)
            }
            PhysicsError::InvalidState { field, value, reason } => {
                write!(f, "Invalid state for field '{}': value '{}' violates {}", field, value, reason)
            }
            PhysicsError::Instability { field, location, value } => {
                write!(f, "Instability in field '{}' at {:?}: {}", field, location, value)
            }
            PhysicsError::ConservationViolation { quantity, error, tolerance } => {
                write!(f, "Conservation violation for '{}': error {} > tolerance {}", quantity, error, tolerance)
            }
            PhysicsError::ConvergenceFailure { solver, iterations, residual } => {
                write!(f, "Convergence failed for solver '{}' after {} iterations: residual {}", solver, iterations, residual)
            }
            PhysicsError::InvalidFieldIndex(index) => {
                write!(f, "Invalid field index: {}", index)
            }
            PhysicsError::StateError(reason) => {
                write!(f, "State management error: {}", reason)
            }
            PhysicsError::DimensionMismatch => {
                write!(f, "Dimension mismatch in physics simulation")
            }
            PhysicsError::UnauthorizedFieldAccess { field, operation } => {
                write!(f, "Unauthorized access to field '{}' during '{}' operation", field, operation)
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
    /// Thread pool creation failed
    ThreadPoolCreation {
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
        error_code: i32,
        reason: String,
    },
    /// IO operation failed
    Io {
        operation: String,
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
            SystemError::ThreadPoolCreation { reason } => {
                write!(f, "Thread pool creation failed: {}", reason)
            }
            SystemError::ThreadSync { operation, reason } => {
                write!(f, "Thread synchronization failed in {}: {}", operation, reason)
            }
            SystemError::ResourceExhausted { resource, reason } => {
                write!(f, "System resource '{}' exhausted: {}", resource, reason)
            }
            SystemError::SystemCall { call, error_code, reason } => {
                write!(f, "System call '{}' failed with error code {}: {}", call, error_code, reason)
            }
            SystemError::Io { operation, reason } => {
                write!(f, "IO operation '{}' failed: {}", operation, reason)
            }
        }
    }
}

impl StdError for SystemError {}

impl From<std::io::Error> for KwaversError {
    fn from(err: std::io::Error) -> Self {
        KwaversError::System(SystemError::Io {
            operation: "file".to_string(),
            reason: err.to_string(),
        })
    }
}

/// Memory transfer direction for GPU operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryTransferDirection {
    /// Host to device transfer
    HostToDevice,
    /// Device to host transfer
    DeviceToHost,
    /// Device to device transfer
    DeviceToDevice,
}

impl fmt::Display for MemoryTransferDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryTransferDirection::HostToDevice => write!(f, "HostToDevice"),
            MemoryTransferDirection::DeviceToHost => write!(f, "DeviceToHost"),
            MemoryTransferDirection::DeviceToDevice => write!(f, "DeviceToDevice"),
        }
    }
}

/// GPU-related errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuError {
    /// Device initialization failed
    DeviceInitialization {
        device_id: u32,
        reason: String,
    },

    /// Device detection failed
    DeviceDetection {
        reason: String,
    },

    /// Memory allocation failed
    MemoryAllocation {
        requested_bytes: usize,
        available_bytes: usize,
        reason: String,
    },

    /// Memory transfer failed
    MemoryTransfer {
        direction: MemoryTransferDirection,
        size_bytes: usize,
        reason: String,
    },

    /// Kernel compilation failed
    KernelCompilation {
        kernel_name: String,
        reason: String,
    },

    /// Kernel execution failed
    KernelExecution {
        kernel_name: String,
        reason: String,
    },

    /// Backend not available
    BackendNotAvailable {
        backend: String,
        reason: String,
    },

    /// No GPU devices found
    NoDevicesFound,
    
    /// Invalid operation
    InvalidOperation {
        operation: String,
        reason: String,
    },
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::DeviceInitialization { device_id, reason } => {
                write!(f, "GPU device initialization failed (device {}): {}", device_id, reason)
            }
            GpuError::DeviceDetection { reason } => {
                write!(f, "GPU device detection failed: {}", reason)
            }
            GpuError::MemoryAllocation { requested_bytes, available_bytes, reason } => {
                write!(f, "GPU memory allocation failed (requested {} bytes, available {} bytes): {}", 
                       requested_bytes, available_bytes, reason)
            }
            GpuError::MemoryTransfer { direction, size_bytes, reason } => {
                write!(f, "GPU memory transfer ({:?}) of {} bytes failed: {}", direction, size_bytes, reason)
            }
            GpuError::KernelCompilation { kernel_name, reason } => {
                write!(f, "GPU kernel compilation failed ({}): {}", kernel_name, reason)
            }
            GpuError::KernelExecution { kernel_name, reason } => {
                write!(f, "GPU kernel execution failed ({}): {}", kernel_name, reason)
            }
            GpuError::BackendNotAvailable { backend, reason } => {
                write!(f, "GPU backend not available ({}): {}", backend, reason)
            }
            GpuError::NoDevicesFound => {
                write!(f, "No GPU devices found")
            }
            GpuError::InvalidOperation { operation, reason } => {
                write!(f, "Invalid GPU operation '{}': {}", operation, reason)
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

/// Validation success type for explicit success reporting
#[derive(Debug, Clone)]
pub struct ValidationSuccess {
    pub message: String,
    pub details: Option<HashMap<String, String>>,
}

impl ValidationSuccess {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            details: None,
        }
    }
    
    pub fn with_details(mut self, details: HashMap<String, String>) -> Self {
        self.details = Some(details);
        self
    }
}

/// Validation result type
pub type ValidationResult = Result<ValidationSuccess, ValidationError>;

// Note: EnhancedError and related types removed as they were unused
// and violated YAGNI principle. The standard KwaversError provides
// sufficient error handling capabilities.

// ErrorBuilder and related utilities removed as they depended on
// the removed EnhancedError type. Use KwaversError directly instead.

/// Result type alias for kwavers operations
pub type KwaversResult<T> = Result<T, KwaversError>;

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
    
    // Note: Functions that referenced EnhancedError and ErrorSeverity have been removed
    // as those types were removed for violating YAGNI principle.
    // Use KwaversError directly for error handling.
    
    /// Check if an error is recoverable
    pub fn is_recoverable(error: &KwaversError) -> bool {
        matches!(error, 
            KwaversError::Config(_) |
            KwaversError::Validation(_) |
            KwaversError::Data(_)
        )
    }
    
    /// Check if an error is critical
    pub fn is_critical(error: &KwaversError) -> bool {
        matches!(error,
            KwaversError::System(_) |
            KwaversError::Gpu(GpuError::DeviceDetection { .. }) |
            KwaversError::Gpu(GpuError::MemoryAllocation { .. })
        )
    }
    
    /// Get a human-readable error category
    pub fn get_category(error: &KwaversError) -> &'static str {
        match error {
            KwaversError::System(_) => "System",
            KwaversError::Numerical(_) => "Numerical",
            KwaversError::Physics(_) => "Physics",
            KwaversError::Grid(_) => "Grid",
            KwaversError::Medium(_) => "Medium",
            KwaversError::Data(_) => "Data",
            KwaversError::Config(_) => "Configuration",
            KwaversError::Validation(_) => "Validation",
            KwaversError::Gpu(_) => "GPU",
            KwaversError::Visualization(_) => "Visualization",
            KwaversError::Composite(_) => "Composite",
            KwaversError::NotImplemented(_) => "Not Implemented",
        }
    }
}

/// Advanced error handling utilities using iterator combinators
pub mod advanced {
    
    use std::fmt::Debug;
    
    /// Chain multiple fallible operations with early return on error
    pub struct ErrorChain<T, E> {
        operations: Vec<Box<dyn FnOnce(T) -> Result<T, E>>>,
    }
    
    impl<T, E> ErrorChain<T, E> {
        pub fn new() -> Self {
            Self {
                operations: Vec::new(),
            }
        }
        
        /// Add an operation to the chain
        pub fn then<F>(mut self, f: F) -> Self
        where
            F: FnOnce(T) -> Result<T, E> + 'static,
        {
            self.operations.push(Box::new(f));
            self
        }
        
        /// Execute all operations in sequence
        pub fn execute(self, initial: T) -> Result<T, E> {
            self.operations.into_iter()
                .try_fold(initial, |acc, op| op(acc))
        }
    }
    
    /// Collect multiple Results into a single Result
    pub trait ResultCollector<T, E> {
        /// Collect all Ok values or return the first error
        fn collect_results(self) -> Result<Vec<T>, E>;
        
        /// Collect all values, separating successes and failures
        fn partition_results(self) -> (Vec<T>, Vec<E>);
    }
    
    impl<I, T, E> ResultCollector<T, E> for I
    where
        I: Iterator<Item = Result<T, E>>,
    {
        fn collect_results(self) -> Result<Vec<T>, E> {
            self.collect()
        }
        
        fn partition_results(self) -> (Vec<T>, Vec<E>) {
            let (oks, errs): (Vec<_>, Vec<_>) = self.partition(Result::is_ok);
            
            let values = oks.into_iter()
                .filter_map(|r| r.ok())
                .collect();
                
            let errors = errs.into_iter()
                .filter_map(|r| r.err())
                .collect();
                
            (values, errors)
        }
    }
    
    /// Error accumulator for collecting multiple errors
    #[derive(Debug, Clone)]
    pub struct ErrorAccumulator<E> {
        errors: Vec<E>,
        context: Vec<String>,
    }
    
    impl<E> ErrorAccumulator<E> {
        pub fn new() -> Self {
            Self {
                errors: Vec::new(),
                context: Vec::new(),
            }
        }
        
        /// Add an error to the accumulator
        pub fn add_error(&mut self, error: E) {
            self.errors.push(error);
        }
        
        /// Add context information
        pub fn with_context<S: Into<String>>(mut self, context: S) -> Self {
            self.context.push(context.into());
            self
        }
        
        /// Check if any errors were accumulated
        pub fn has_errors(&self) -> bool {
            !self.errors.is_empty()
        }
        
        /// Convert to a Result
        pub fn into_result<T, F>(self, ok_value: F) -> Result<T, Self>
        where
            F: FnOnce() -> T,
        {
            if self.has_errors() {
                Err(self)
            } else {
                Ok(ok_value())
            }
        }
        
        /// Get all accumulated errors
        pub fn errors(&self) -> &[E] {
            &self.errors
        }
    }
    
    /// Retry logic with exponential backoff
    pub struct RetryStrategy<E> {
        max_attempts: usize,
        backoff_ms: u64,
        backoff_factor: f64,
        error_handler: Option<Box<dyn Fn(&E, usize)>>,
    }
    
    impl<E> RetryStrategy<E> {
        pub fn new() -> Self {
            Self {
                max_attempts: 3,
                backoff_ms: 100,
                backoff_factor: 2.0,
                error_handler: None,
            }
        }
        
        pub fn with_max_attempts(mut self, attempts: usize) -> Self {
            self.max_attempts = attempts;
            self
        }
        
        pub fn with_backoff(mut self, initial_ms: u64, factor: f64) -> Self {
            self.backoff_ms = initial_ms;
            self.backoff_factor = factor;
            self
        }
        
        pub fn with_error_handler<F>(mut self, handler: F) -> Self
        where
            F: Fn(&E, usize) + 'static,
        {
            self.error_handler = Some(Box::new(handler));
            self
        }
        
        /// Execute an operation with retry logic
        pub fn execute<T, F>(&self, mut operation: F) -> Result<T, E>
        where
            F: FnMut() -> Result<T, E>,
            E: Debug,
        {
            let mut last_error = None;
            let mut backoff = self.backoff_ms;
            
            for attempt in 1..=self.max_attempts {
                match operation() {
                    Ok(value) => return Ok(value),
                    Err(e) => {
                        if let Some(ref handler) = self.error_handler {
                            handler(&e, attempt);
                        }
                        
                        last_error = Some(e);
                        
                        if attempt < self.max_attempts {
                            std::thread::sleep(std::time::Duration::from_millis(backoff));
                            backoff = (backoff as f64 * self.backoff_factor) as u64;
                        }
                    }
                }
            }
            
            Err(last_error.unwrap())
        }
    }
    
    /// Transform errors using a pipeline
    pub trait ErrorTransform<T, E> {
        /// Map the error to a different type
        fn map_error<F, E2>(self, f: F) -> Result<T, E2>
        where
            F: FnOnce(E) -> E2;
        
        /// Add context to the error
        fn with_error_context<F, S>(self, f: F) -> Result<T, String>
        where
            F: FnOnce(&E) -> S,
            S: Into<String>;
    }
    
    impl<T, E> ErrorTransform<T, E> for Result<T, E> {
        fn map_error<F, E2>(self, f: F) -> Result<T, E2>
        where
            F: FnOnce(E) -> E2,
        {
            self.map_err(f)
        }
        
        fn with_error_context<F, S>(self, f: F) -> Result<T, String>
        where
            F: FnOnce(&E) -> S,
            S: Into<String>,
        {
            self.map_err(|e| f(&e).into())
        }
    }
    
    /// Validate multiple conditions and collect all failures
    pub struct ValidationChain<T> {
        value: T,
        validators: Vec<Box<dyn Fn(&T) -> Result<(), String>>>,
    }
    
    impl<T> ValidationChain<T> {
        pub fn new(value: T) -> Self {
            Self {
                value,
                validators: Vec::new(),
            }
        }
        
        /// Add a validation rule
        pub fn validate<F>(mut self, validator: F) -> Self
        where
            F: Fn(&T) -> Result<(), String> + 'static,
        {
            self.validators.push(Box::new(validator));
            self
        }
        
        /// Run all validations and collect errors
        pub fn run(self) -> Result<T, Vec<String>> {
            let errors: Vec<String> = self.validators.iter()
                .filter_map(|validator| {
                    validator(&self.value).err()
                })
                .collect();
            
            if errors.is_empty() {
                Ok(self.value)
            } else {
                Err(errors)
            }
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
            _ => panic!("Unexpected error type"),
        }
    }
    
    // Tests for EnhancedError and ErrorBuilder removed as those types
    // were removed for violating YAGNI principle
    
    #[test]
    fn test_error_utilities() {
        let validation_error = KwaversError::Validation(ValidationError::FieldValidation {
            field: "test".to_string(),
            value: "invalid".to_string(),
            constraint: "must be positive".to_string(),
        });
        
        assert!(utils::is_recoverable(&validation_error));
        assert!(!utils::is_critical(&validation_error));
        assert_eq!(utils::get_category(&validation_error), "Validation");
    }
}