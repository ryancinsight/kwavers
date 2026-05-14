//! GPU Compute Pipeline Management
//!
//! Manages compute shader compilation, pipeline creation, and execution.

pub mod manager;
#[cfg(test)]
mod tests;
pub mod types;

pub use manager::PipelineManager;
pub use types::PipelineType;
