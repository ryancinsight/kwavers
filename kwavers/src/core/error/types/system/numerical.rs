//! Numerical computation error types
//!
//! Mathematical and computational error handling

use thiserror::Error;

/// Numerical computation error types
#[derive(Error, Debug, Clone)]
pub enum NumericalErrorType {
    #[error("Division by zero: operation {operation} resulted in division by zero")]
    DivisionByZero { operation: String },
    
    #[error("Overflow detected: {operation} resulted in overflow")]
    Overflow { operation: String },
    
    #[error("Underflow detected: {operation} resulted in underflow")]
    Underflow { operation: String },
    
    #[error("NaN encountered: {operation} produced NaN value")]
    NanEncountered { operation: String },
    
    #[error("Infinity encountered: {operation} produced infinite value")]
    InfinityEncountered { operation: String },
    
    #[error("Precision loss: {operation} lost significant precision")]
    PrecisionLoss { operation: String },
    
    #[error("Matrix singular: matrix is not invertible")]
    SingularMatrix,
    
    #[error("FFT error: {transform_type} failed with {reason}")]
    FftError {
        transform_type: String,
        reason: String,
    },
    
    #[error("Integration error: {method} failed to converge")]
    IntegrationError { method: String },
}
