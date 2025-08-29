//! Data pipeline for efficient GPU data transfer and processing
//!
//! This module provides a high-performance pipeline for transferring simulation
//! data to GPU for visualization, with support for various processing operations.

mod operations;
mod statistics;
mod transfer;
mod processing;

pub use operations::ProcessingOperation;
pub use statistics::TransferStatistics;
pub use transfer::DataPipeline;
pub use processing::{ProcessingConfig, ProcessingStage};

// Re-export core types
pub use transfer::{TransferMode, TransferOptions};