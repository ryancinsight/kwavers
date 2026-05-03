//! GPU Compute Pipeline Management
//!
//! Manages compute shader compilation, pipeline creation, and execution.

pub mod manager;
pub mod types;
#[cfg(test)]
mod tests;

pub use manager::PipelineManager;
pub use types::PipelineType;
