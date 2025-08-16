//! Physics simulation error types

use std::error::Error as StdError;
use std::fmt;
use serde::{Deserialize, Serialize};

/// Physics-related errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PhysicsError {
    /// Invalid physical parameter
    InvalidParameter {
        parameter: String,
        value: f64,
        reason: String,
    },
    /// Numerical instability detected
    NumericalInstability {
        timestep: f64,
        cfl_limit: f64,
    },
    /// Conservation law violation
    ConservationViolation {
        quantity: String,
        initial: f64,
        current: f64,
        tolerance: f64,
    },
    /// Divergence in iterative solver
    SolverDivergence {
        iterations: usize,
        residual: f64,
    },
    /// Invalid boundary condition
    InvalidBoundaryCondition {
        boundary: String,
        reason: String,
    },
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
}

impl fmt::Display for PhysicsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter { parameter, value, reason } => {
                write!(f, "Invalid parameter {}: {} ({})", parameter, value, reason)
            }
            Self::NumericalInstability { timestep, cfl_limit } => {
                write!(f, "Numerical instability: timestep {} exceeds CFL limit {}", timestep, cfl_limit)
            }
            Self::ConservationViolation { quantity, initial, current, tolerance } => {
                write!(f, "Conservation violation for {}: initial={}, current={}, tolerance={}", 
                       quantity, initial, current, tolerance)
            }
            Self::SolverDivergence { iterations, residual } => {
                write!(f, "Solver diverged after {} iterations, residual: {}", iterations, residual)
            }
            Self::InvalidBoundaryCondition { boundary, reason } => {
                write!(f, "Invalid boundary condition at {}: {}", boundary, reason)
            }
            Self::ShockFormation { location, pressure } => {
                write!(f, "Shock formation at {:?}, pressure: {}", location, pressure)
            }
            Self::CavitationDetected { location, pressure } => {
                write!(f, "Cavitation detected at {:?}, pressure: {}", location, pressure)
            }
        }
    }
}

impl StdError for PhysicsError {}