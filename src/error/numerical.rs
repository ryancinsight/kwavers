//! Numerical computation error types

use serde::{Deserialize, Serialize};
use std::error::Error as StdError;
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize))]
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
                write!(f, "Division by zero in {} at {}", operation, location)
            }
            Self::InvalidOperation(op) => write!(f, "Invalid operation: {}", op),
            Self::Instability {
                operation,
                condition,
            } => {
                write!(
                    f,
                    "Numerical instability in {}: condition number {}",
                    operation, condition
                )
            }
            Self::NaN { operation, inputs } => {
                write!(f, "NaN value in {}: inputs {}", operation, inputs)
            }
            Self::MatrixDimension {
                operation,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Matrix dimension error in {}: expected {}, got {}",
                    operation, expected, actual
                )
            }
            Self::SingularMatrix {
                operation,
                condition_number,
            } => {
                write!(
                    f,
                    "Singular matrix in {}: condition number {}",
                    operation, condition_number
                )
            }
            Self::UnsupportedOperation { operation, reason } => {
                write!(f, "Unsupported operation {}: {}", operation, reason)
            }
            Self::SolverFailed { method, reason } => {
                write!(f, "Solver '{}' failed: {}", method, reason)
            }
            Self::ConvergenceFailed {
                method,
                iterations,
                error,
            } => {
                write!(
                    f,
                    "Method '{}' failed to converge after {} iterations (error: {:.2e})",
                    method, iterations, error
                )
            }
            Self::NotImplemented { feature } => {
                write!(f, "Feature not implemented: {}", feature)
            }
        }
    }
}

impl StdError for NumericalError {}
