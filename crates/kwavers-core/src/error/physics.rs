//! Physics simulation error types

use serde::{Deserialize, Serialize};
use std::error::Error as StdError;
use std::fmt;

/// Physics-related errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PhysicsError {
    /// Invalid physical parameter
    InvalidParameter {
        parameter: String,
        value: f64,
        reason: String,
    },
    /// Numerical instability detected with CFL violation
    NumericalInstability { timestep: f64, cfl_limit: f64 },
    /// General numerical instability with custom message
    NumericalInstabilityGeneral { message: String },
    /// Conservation law violation
    ConservationViolation {
        quantity: String,
        initial: f64,
        current: f64,
        tolerance: f64,
    },
    /// Divergence in iterative solver
    SolverDivergence { iterations: usize, residual: f64 },
    /// Invalid boundary condition
    InvalidBoundaryCondition { boundary: String, reason: String },
    /// Shock formation detected
    ShockFormation {
        location: (usize, usize, usize),
        pressure: f64,
    },
    /// Cavitation detected
    CavitationDetected {
        location: (usize, usize, usize),
        pressure: f64,
    },
    /// Dimension mismatch
    DimensionMismatch,
    /// Invalid field dimensions
    InvalidFieldDimensions { expected: String, actual: String },
    /// Invalid state
    InvalidState {
        field: String,
        value: String,
        reason: String,
    },
    /// Convergence failure
    ConvergenceFailure {
        solver: String,
        iterations: usize,
        residual: f64,
    },
    /// Invalid configuration
    InvalidConfiguration { parameter: String, reason: String },
    /// Model not initialized
    ModelNotInitialized { model: String, reason: String },
    /// Unauthorized field access
    UnauthorizedFieldAccess { field: String, operation: String },
    /// Invalid field index
    InvalidFieldIndex(usize),
    /// State error
    StateError(String),
    /// Instability
    Instability {
        field: String,
        value: f64,
        threshold: f64,
    },
}

impl fmt::Display for PhysicsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter {
                parameter,
                value,
                reason,
            } => {
                write!(f, "Invalid parameter {parameter}: {value} ({reason})")
            }
            Self::NumericalInstability {
                timestep,
                cfl_limit,
            } => {
                write!(
                    f,
                    "Numerical instability: timestep {timestep} exceeds CFL limit {cfl_limit}"
                )
            }
            Self::NumericalInstabilityGeneral { message } => {
                write!(f, "Numerical instability: {message}")
            }
            Self::ConservationViolation {
                quantity,
                initial,
                current,
                tolerance,
            } => {
                write!(
                    f,
                    "Conservation violation for {quantity}: initial={initial}, current={current}, tolerance={tolerance}"
                )
            }
            Self::SolverDivergence {
                iterations,
                residual,
            } => {
                write!(
                    f,
                    "Solver diverged after {iterations} iterations, residual: {residual}"
                )
            }
            Self::InvalidBoundaryCondition { boundary, reason } => {
                write!(f, "Invalid boundary condition at {boundary}: {reason}")
            }
            Self::ShockFormation { location, pressure } => {
                write!(f, "Shock formation at {location:?}, pressure: {pressure}")
            }
            Self::CavitationDetected { location, pressure } => {
                write!(
                    f,
                    "Cavitation detected at {location:?}, pressure: {pressure}"
                )
            }
            Self::DimensionMismatch => {
                write!(f, "Dimension mismatch in physics calculation")
            }
            Self::InvalidFieldDimensions { expected, actual } => {
                write!(
                    f,
                    "Invalid field dimensions: expected {expected}, got {actual}"
                )
            }
            Self::InvalidState {
                field,
                value,
                reason,
            } => {
                write!(f, "Invalid state for {field}: {value} ({reason})")
            }
            Self::ConvergenceFailure {
                solver,
                iterations,
                residual,
            } => {
                write!(
                    f,
                    "Convergence failure in {solver}: {iterations} iterations, residual {residual}"
                )
            }
            Self::InvalidConfiguration { parameter, reason } => {
                write!(f, "Invalid configuration for {parameter}: {reason}")
            }
            Self::ModelNotInitialized { model, reason } => {
                write!(f, "Model '{model}' not initialized: {reason}")
            }
            Self::UnauthorizedFieldAccess { field, operation } => {
                write!(
                    f,
                    "Unauthorized access to field '{field}' during '{operation}' operation"
                )
            }
            Self::InvalidFieldIndex(index) => {
                write!(f, "Invalid field index: {index}")
            }
            Self::StateError(reason) => {
                write!(f, "State management error: {reason}")
            }
            Self::Instability {
                field,
                value,
                threshold,
            } => {
                write!(
                    f,
                    "Physics instability in {field}: value {value} exceeds threshold {threshold}"
                )
            }
        }
    }
}

impl StdError for PhysicsError {}
