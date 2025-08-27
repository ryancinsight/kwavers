//! GPU Kernel configuration

use super::KernelType;

/// GPU kernel optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// Level 1 optimization (memory coalescing)
    Level1,
    /// Level 2 optimization (shared memory, loop unrolling)
    Level2,
    /// Level 3 optimization (register blocking, texture memory)
    Level3,
}

/// GPU kernel configuration
#[derive(Debug, Clone)]
pub struct KernelConfig {
    pub kernel_type: KernelType,
    pub optimization_level: OptimizationLevel,
    pub block_size: (u32, u32, u32),
    pub grid_size: (u32, u32, u32),
    pub shared_memory_size: u32,
    pub registers_per_thread: u32,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            kernel_type: KernelType::AcousticWave,
            optimization_level: OptimizationLevel::Level2,
            block_size: (16, 16, 4),
            grid_size: (1, 1, 1),
            shared_memory_size: 0,
            registers_per_thread: 32,
        }
    }
}
