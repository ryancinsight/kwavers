//! Burn-based GPU Acceleration Framework

mod accelerator;
mod types;

pub use accelerator::BurnGpuAccelerator;
pub use types::{
    EquationType, GpuConfig, GpuOperation, MemoryStrategy, PhysicsParameters, Precision,
};
