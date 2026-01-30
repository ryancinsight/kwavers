//! Backend trait definitions and types
//!
//! Defines the abstraction layer for different computational backends (CPU, GPU, etc.)

use crate::core::error::KwaversResult;
use ndarray::Array3;

/// Type of compute backend
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// CPU-based (parallel with rayon)
    CPU,
    /// GPU-based (WGPU for cross-platform)
    GPU,
}

/// Compute device information
#[derive(Debug, Clone)]
pub struct ComputeDevice {
    /// Device ID
    pub id: usize,

    /// Device name
    pub name: String,

    /// Backend type
    pub backend_type: BackendType,

    /// Total available memory (bytes)
    pub total_memory: usize,

    /// Currently available memory (bytes)
    pub available_memory: usize,

    /// Number of compute units
    pub compute_units: usize,

    /// Peak performance (FLOPS)
    pub peak_performance: f64,
}

/// Backend capabilities
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    /// Supports FFT operations
    pub supports_fft: bool,

    /// Supports 64-bit floating point
    pub supports_f64: bool,

    /// Supports 32-bit floating point
    pub supports_f32: bool,

    /// Supports asynchronous operations
    pub supports_async: bool,

    /// Maximum parallelism level
    pub max_parallelism: usize,

    /// Supports unified memory (GPU-CPU)
    pub supports_unified_memory: bool,
}

/// Backend abstraction trait
pub trait Backend {
    /// Get backend type
    fn backend_type(&self) -> BackendType;

    /// Get backend capabilities
    fn capabilities(&self) -> BackendCapabilities;

    /// Check if backend is available
    fn is_available(&self) -> bool;

    /// Synchronize all operations
    fn synchronize(&self) -> KwaversResult<()>;

    /// Get available compute devices
    fn devices(&self) -> Vec<ComputeDevice>;

    /// Select a specific compute device
    fn select_device(&mut self, device_id: usize) -> KwaversResult<()>;

    /// Execute 3D FFT
    fn fft_3d(&self, data: &mut Array3<f64>) -> KwaversResult<()>;

    /// Execute 3D inverse FFT
    fn ifft_3d(&self, data: &mut Array3<f64>) -> KwaversResult<()>;

    /// Element-wise multiplication
    fn element_wise_multiply(
        &self,
        a: &Array3<f64>,
        b: &Array3<f64>,
        out: &mut Array3<f64>,
    ) -> KwaversResult<()>;

    /// Apply spatial derivative
    fn apply_spatial_derivative(
        &self,
        field: &Array3<f64>,
        direction: usize,
        out: &mut Array3<f64>,
    ) -> KwaversResult<()>;

    /// Estimate performance for a given problem size
    fn estimate_performance(&self, problem_size: (usize, usize, usize)) -> f64;
}
