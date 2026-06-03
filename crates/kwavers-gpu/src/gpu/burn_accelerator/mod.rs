//! Burn-based GPU Acceleration Framework

mod accelerator;
mod types;

pub use accelerator::BurnGpuAccelerator;
pub use types::{
    EquationType, GpuConfig, GpuOperation, GpuPhysicsParameters, MemoryStrategy, Precision,
};
