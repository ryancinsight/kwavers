//! GPU Kernel type definitions

use super::config::KernelConfig;
use crate::gpu::GpuPerformanceMetrics;

/// GPU kernel types for different physics operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelType {
    /// Acoustic wave propagation kernel
    AcousticWave,
    /// Thermal diffusion kernel
    ThermalDiffusion,
    /// Forward FFT kernel
    FFTForward,
    /// Inverse FFT kernel
    FFTInverse,
    /// Memory copy kernel
    MemoryCopy,
    /// Boundary condition kernel
    BoundaryCondition,
}

/// Compiled kernel representation
pub struct CompiledKernel {
    pub kernel_type: KernelType,
    pub source_code: String,
    pub binary_code: Option<Vec<u8>>,
    pub config: KernelConfig,
    pub performance_metrics: Option<GpuPerformanceMetrics>,
}
