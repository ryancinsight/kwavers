//! Data pipeline for efficient GPU data transfer and processing
//!
//! This module provides a high-performance pipeline for transferring simulation
//! data to GPU for visualization, with support for various processing operations.

mod operations;
mod processing;
mod statistics;
mod transfer;

pub use operations::ProcessingOperation;
pub use processing::{ProcessingConfig, ProcessingStage};
pub use statistics::TransferStatistics;
pub use transfer::DataPipeline;

// Re-export core types
pub use transfer::{TransferMode, TransferOptions};
