//! Numerical computation error types

use std::error::Error as StdError;
use std::fmt;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NumericalError {
    Overflow,
    Underflow,
    DivisionByZero,
    InvalidOperation(String),
    Instability {
        operation: String,
        condition: f64,
    },
    NaN {
        operation: String,
        inputs: String,
    },
    MatrixDimension,
    SingularMatrix,
    UnsupportedOperation,
}

impl fmt::Display for NumericalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overflow => write!(f, "Numerical overflow"),
            Self::Underflow => write!(f, "Numerical underflow"),
            Self::DivisionByZero => write!(f, "Division by zero"),
            Self::InvalidOperation(op) => write!(f, "Invalid operation: {}", op),
            Self::Instability { operation, condition } => {
                write!(f, "Numerical instability in {}: condition number {}", operation, condition)
            }
            Self::NaN { operation, inputs } => {
                write!(f, "NaN value in {}: inputs {}", operation, inputs)
            }
            Self::MatrixDimension => write!(f, "Matrix dimension error"),
            Self::SingularMatrix => write!(f, "Singular matrix"),
            Self::UnsupportedOperation => write!(f, "Unsupported numerical operation"),
        }
    }
}

impl StdError for NumericalError {}
