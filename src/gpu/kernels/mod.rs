//! GPU compute kernels for acoustic simulation

pub mod fdtd;
pub mod pml;
pub mod pressure;

/// Workgroup size for compute shaders
pub const WORKGROUP_SIZE: u32 = 8;

/// Calculate workgroup count for dimension
pub fn workgroup_count(size: u32) -> u32 {
    (size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
}
