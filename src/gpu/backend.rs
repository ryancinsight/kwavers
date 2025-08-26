//! GPU backend abstraction
//!
//! This module defines the GPU backend types and selection logic,
//! following SOLID principles with clear separation of concerns.

use crate::error::{GpuError, KwaversResult};

/// GPU backend type following SSOT principle
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuBackend {
    /// NVIDIA CUDA backend
    Cuda,
    /// OpenCL backend (cross-platform)
    OpenCL,
    /// WebGPU backend (future-proof)
    WebGpu,
}

impl GpuBackend {
    /// Select the best available backend
    pub fn auto_select() -> KwaversResult<Self> {
        // Try CUDA first (typically fastest)
        if Self::cuda_available() {
            return Ok(Self::Cuda);
        }

        // Fall back to OpenCL
        if Self::opencl_available() {
            return Ok(Self::OpenCL);
        }

        // WebGPU as last resort
        if Self::webgpu_available() {
            return Ok(Self::WebGpu);
        }

        Err(GpuError::NoDeviceFound.into())
    }

    /// Check if CUDA is available
    fn cuda_available() -> bool {
        // This would check for CUDA runtime
        cfg!(feature = "cuda") && std::env::var("CUDA_PATH").is_ok()
    }

    /// Check if OpenCL is available
    fn opencl_available() -> bool {
        cfg!(feature = "opencl")
    }

    /// Check if WebGPU is available
    fn webgpu_available() -> bool {
        cfg!(feature = "webgpu")
    }

    /// Get backend name for display
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cuda => "CUDA",
            Self::OpenCL => "OpenCL",
            Self::WebGpu => "WebGPU",
        }
    }
}

/// Get the GPU float type string for kernel generation
pub fn gpu_float_type_str() -> &'static str {
    if cfg!(feature = "gpu-f64") {
        "double"
    } else {
        "float"
    }
}

/// Get the GPU float type size
pub fn gpu_float_size() -> usize {
    if cfg!(feature = "gpu-f64") {
        8
    } else {
        4
    }
}
