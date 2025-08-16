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
}

impl fmt::Display for NumericalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overflow => write!(f, "Numerical overflow"),
            Self::Underflow => write!(f, "Numerical underflow"),
            Self::DivisionByZero => write!(f, "Division by zero"),
            Self::InvalidOperation(op) => write!(f, "Invalid operation: {}", op),
        }
    }
}

impl StdError for NumericalError {}
