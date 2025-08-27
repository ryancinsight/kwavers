//! GPU FFT module providing high-performance FFT implementations
//!
//! This module is organized by backend to maintain separation of concerns:
//! - `plan`: FFT planning and workspace management
//! - `cuda`: CUDA-specific implementation
//! - `opencl`: OpenCL-specific implementation  
//! - `webgpu`: WebGPU-specific implementation
//! - `kernels`: Shared kernel algorithms
//! - `transpose`: Matrix transpose operations

pub mod kernels;
pub mod plan;
pub mod transpose;

#[cfg(feature = "cuda")]
pub mod cuda;

// OpenCL and WebGPU FFT implementations are not yet available
// These will be implemented when the respective backends are fully developed

// Re-export main types
pub use kernels::FftKernel;
pub use plan::{FftDirection, GpuFftPlan};
pub use transpose::TransposeOperation;

use crate::error::KwaversResult;
use ndarray::Array3;
use num_complex::Complex;

/// Trait for GPU FFT implementations
pub trait GpuFftBackend: Send + Sync {
    /// Create a new FFT plan for the given dimensions
    fn create_plan(
        &self,
        nx: usize,
        ny: usize,
        nz: usize,
        forward: bool,
    ) -> KwaversResult<GpuFftPlan>;

    /// Execute FFT on complex data
    fn execute(&mut self, plan: &GpuFftPlan, data: &mut Array3<Complex<f64>>) -> KwaversResult<()>;

    /// Get backend name for debugging
    fn backend_name(&self) -> &str;
}

/// Factory function to create appropriate backend
pub fn create_fft_backend() -> KwaversResult<Box<dyn GpuFftBackend>> {
    #[cfg(feature = "cuda")]
    {
        return Ok(Box::new(cuda::CudaFftBackend::new()?));
    }

    // OpenCL and WebGPU backends not yet implemented
    #[cfg(feature = "opencl")]
    {
        use crate::error::KwaversError;
        return Err(KwaversError::NotImplemented(
            "OpenCL FFT backend not yet implemented".to_string(),
        ));
    }

    #[cfg(feature = "webgpu")]
    {
        use crate::error::KwaversError;
        return Err(KwaversError::NotImplemented(
            "WebGPU FFT backend not yet implemented".to_string(),
        ));
    }

    #[cfg(not(feature = "cuda"))]
    {
        use crate::error::KwaversError;
        Err(KwaversError::Config(
            crate::error::ConfigError::MissingParameter {
                parameter: "GPU backend".to_string(),
                section: "features".to_string(),
            },
        ))
    }
}
