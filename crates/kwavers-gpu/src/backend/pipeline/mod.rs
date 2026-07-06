//! WGPU compute pipeline management
//!
//! Manages WGSL compute shader compilation, pipeline creation, and execution.

pub mod manager;
#[cfg(test)]
mod tests;
pub mod types;

pub use manager::WgpuPipelineManager;
pub use types::PipelineType;
