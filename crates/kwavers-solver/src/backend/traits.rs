//! Backend trait definitions and types
//!
//! Defines the abstraction layer for different computational backends (CPU, GPU, etc.)

use kwavers_core::error::KwaversResult;
use leto::Array3;

/// Type of compute backend
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// CPU-based provider.
    CPU,
    /// GPU-based provider selected at the integration boundary.
    GPU(GpuProvider),
}

impl BackendType {
    /// Returns the GPU provider for GPU backends.
    #[must_use]
    pub const fn gpu_provider(self) -> Option<GpuProvider> {
        match self {
            Self::CPU => None,
            Self::GPU(provider) => Some(provider),
        }
    }
}

/// Concrete GPU provider behind a compute backend implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuProvider {
    /// WGPU/WebGPU provider.
    Wgpu,
    /// CUDA provider.
    Cuda,
    /// Metal provider.
    Metal,
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
    /// Supports FFT operations exposed through this backend trait.
    ///
    /// GPU FFT is owned by Apollo through `kwavers_math::fft::gpu_fft`, so a
    /// `ComputeBackend` implementation reports `false` here unless it exposes
    /// FFT methods through this trait.
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

/// Backend abstraction trait.
///
/// The scalar type is part of the backend contract so GPU providers can expose
/// their native precision through one trait surface instead of a parallel
/// WGPU/CUDA API.
pub trait ComputeBackend {
    /// Scalar type accepted by this backend's operation kernels.
    type Scalar: Copy + Send + Sync + 'static;

    /// Get backend type
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn backend_type(&self) -> BackendType;

    /// Get the concrete GPU provider, if this backend is GPU-backed.
    #[must_use]
    fn gpu_provider(&self) -> Option<GpuProvider> {
        self.backend_type().gpu_provider()
    }

    /// Get backend capabilities
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn capabilities(&self) -> BackendCapabilities;

    /// Check if backend is available
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn is_available(&self) -> bool;

    /// Synchronize all operations
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn synchronize(&self) -> KwaversResult<()>;

    /// Get available compute devices
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn devices(&self) -> Vec<ComputeDevice>;

    /// Select a specific compute device
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn select_device(&mut self, device_id: usize) -> KwaversResult<()>;

    // NOTE: 3D FFT is intentionally NOT part of this trait. GPU FFT is owned by
    // Apollo (`kwavers_math::fft::gpu_fft`, via Apollo's `FftBackend` trait) —
    // the single source of truth — so the backend does not reimplement it.

    /// Element-wise multiplication
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn element_wise_multiply(
        &self,
        a: &Array3<Self::Scalar>,
        b: &Array3<Self::Scalar>,
        out: &mut Array3<Self::Scalar>,
    ) -> KwaversResult<()>;

    /// Apply spatial derivative
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn apply_spatial_derivative(
        &self,
        field: &Array3<Self::Scalar>,
        direction: usize,
        out: &mut Array3<Self::Scalar>,
    ) -> KwaversResult<()>;

    /// Estimate performance for a given problem size
    fn estimate_performance(&self, problem_size: (usize, usize, usize)) -> f64;
}
