//! I/O-specific error types
//!
//! File, network, and data format error handling

use thiserror::Error;

/// I/O operation error types
#[derive(Error, Debug)]
pub enum IoErrorType {
    #[error("File operation failed: {operation} on {path}")]
    FileOperation { operation: String, path: String },
    
    #[error("File format error: {format} not supported for {file_type}")]
    UnsupportedFormat { format: String, file_type: String },
    
    #[error("Data corruption detected: {checksum_expected} != {checksum_actual}")]
    DataCorruption {
        checksum_expected: String,
        checksum_actual: String,
    },
    
    #[error("Network operation failed: {operation} to {endpoint}")]
    NetworkOperation { operation: String, endpoint: String },
    
    #[error("Permission denied: insufficient permissions for {operation}")]
    PermissionDenied { operation: String },
    
    #[error("Resource unavailable: {resource} is currently unavailable")]
    ResourceUnavailable { resource: String },
    
    #[error("Timeout occurred: {operation} timed out after {duration_ms}ms")]
    Timeout { operation: String, duration_ms: u64 },
}