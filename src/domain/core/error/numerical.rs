//! Numerical computation error types

use serde::{Deserialize, Serialize};
use std::error::Error as StdError;
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NumericalError {
    Overflow,
    Underflow,
    DivisionByZero {
        operation: String,
        location: String,
    },
    InvalidOperation(String),
    Instability {
        operation: String,
        condition: f64,
    },
    NaN {
        operation: String,
        inputs: String,
    },
    MatrixDimension {
        operation: String,
        expected: String,
        actual: String,
    },
    SingularMatrix {
        operation: String,
        condition_number: f64,
    },
    UnsupportedOperation {
        operation: String,
        reason: String,
    },
    SolverFailed {
        method: String,
        reason: String,
    },
    ConvergenceFailed {
        method: String,
        iterations: usize,
        error: f64,
    },
    NotImplemented {
        feature: String,
    },
    InvalidGridSpacing {
        dx: f64,
        dy: f64,
        dz: f64,
    },
    InsufficientGridPoints {
        required: usize,
        actual: usize,
        direction: String,
    },
    InterpolationOutOfBounds {
        point: f64,
        min: f64,
        max: f64,
    },
}

impl fmt::Display for NumericalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overflow => write!(f, "Numerical overflow"),
            Self::Underflow => write!(f, "Numerical underflow"),
            Self::DivisionByZero {
                operation,
                location,
            } => {
                write!(f, "Division by zero in {operation} at {location}")
            }
            Self::InvalidOperation(op) => write!(f, "Invalid operation: {op}"),
            Self::Instability {
                operation,
                condition,
            } => {
                write!(
                    f,
                    "Numerical instability in {operation}: condition number {condition}"
                )
            }
            Self::NaN { operation, inputs } => {
                write!(f, "NaN value in {operation}: inputs {inputs}")
            }
            Self::MatrixDimension {
                operation,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Matrix dimension error in {operation}: expected {expected}, got {actual}"
                )
            }
            Self::SingularMatrix {
                operation,
                condition_number,
            } => {
                write!(
                    f,
                    "Singular matrix in {operation}: condition number {condition_number}"
                )
            }
            Self::UnsupportedOperation { operation, reason } => {
                write!(f, "Unsupported operation {operation}: {reason}")
            }
            Self::SolverFailed { method, reason } => {
                write!(f, "Solver '{method}' failed: {reason}")
            }
            Self::ConvergenceFailed {
                method,
                iterations,
                error,
            } => {
                write!(
                    f,
                    "Method '{method}' failed to converge after {iterations} iterations (error: {error:.2e})"
                )
            }
            Self::NotImplemented { feature } => {
                write!(f, "Feature not implemented: {feature}")
            }
            Self::InvalidGridSpacing { dx, dy, dz } => {
                write!(
                    f,
                    "Invalid grid spacing: dx={dx}, dy={dy}, dz={dz} (all must be positive)"
                )
            }
            Self::InsufficientGridPoints {
                required,
                actual,
                direction,
            } => {
                write!(
                    f,
                    "Insufficient grid points in {direction} direction: need {required}, got {actual}"
                )
            }
            Self::InterpolationOutOfBounds { point, min, max } => {
                write!(
                    f,
                    "Interpolation point {point} outside domain [{min}, {max}]"
                )
            }
        }
    }
}

impl StdError for NumericalError {}
