//! GPU device management
//!
//! This module handles GPU device enumeration and selection,
//! following GRASP principles with Information Expert pattern.

use crate::error::{GpuError, KwaversResult};
use crate::gpu::backend::GpuBackend;
use log::{debug, info};

/// GPU device information following SSOT principle
#[derive(Debug, Clone)]
pub struct GpuDevice {
    pub id: usize,
    pub name: String,
    pub memory_bytes: usize,
    pub compute_capability: (u32, u32),
    pub max_threads_per_block: usize,
    pub max_grid_dimensions: [usize; 3],
    pub backend: GpuBackend,
}

impl GpuDevice {
    /// Create a new GPU device descriptor
    pub fn new(id: usize, backend: GpuBackend) -> Self {
        Self {
            id,
            name: format!("GPU Device {}", id),
            memory_bytes: 4 * 1024 * 1024 * 1024, // Default 4GB
            compute_capability: (7, 5),           // Default compute capability
            max_threads_per_block: 1024,
            max_grid_dimensions: [65535, 65535, 65535],
            backend,
        }
    }

    /// Query device properties (backend-specific)
    pub fn query_properties(&mut self) -> KwaversResult<()> {
        match self.backend {
            GpuBackend::Cuda => self.query_cuda_properties(),
            GpuBackend::OpenCL => self.query_opencl_properties(),
            GpuBackend::WebGpu => self.query_webgpu_properties(),
        }
    }

    /// Query CUDA device properties
    #[cfg(feature = "cuda")]
    fn query_cuda_properties(&mut self) -> KwaversResult<()> {
        // Would use CUDA API to get actual properties
        debug!("Querying CUDA device {} properties", self.id);
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    fn query_cuda_properties(&mut self) -> KwaversResult<()> {
        Err(GpuError::BackendNotAvailable("CUDA".to_string()).into())
    }

    /// Query OpenCL device properties
    #[cfg(feature = "opencl")]
    fn query_opencl_properties(&mut self) -> KwaversResult<()> {
        debug!("Querying OpenCL device {} properties", self.id);
        Ok(())
    }

    #[cfg(not(feature = "opencl"))]
    fn query_opencl_properties(&mut self) -> KwaversResult<()> {
        Err(GpuError::BackendNotAvailable("OpenCL".to_string()).into())
    }

    /// Query WebGPU device properties
    #[cfg(feature = "webgpu")]
    fn query_webgpu_properties(&mut self) -> KwaversResult<()> {
        debug!("Querying WebGPU device {} properties", self.id);
        Ok(())
    }

    #[cfg(not(feature = "webgpu"))]
    fn query_webgpu_properties(&mut self) -> KwaversResult<()> {
        Err(GpuError::BackendNotAvailable("WebGPU".to_string()).into())
    }

    /// Check if device has sufficient memory
    pub fn has_sufficient_memory(&self, required_bytes: usize) -> bool {
        self.memory_bytes >= required_bytes
    }

    /// Get available memory (approximate)
    pub fn available_memory(&self) -> usize {
        // In practice, would query runtime for actual available memory
        self.memory_bytes * 8 / 10 // Assume 80% available
    }
}

/// Device selector following Strategy pattern
pub struct DeviceSelector {
    min_memory: Option<usize>,
    preferred_backend: Option<GpuBackend>,
    min_compute_capability: Option<(u32, u32)>,
}

impl DeviceSelector {
    /// Create a new device selector
    pub fn new() -> Self {
        Self {
            min_memory: None,
            preferred_backend: None,
            min_compute_capability: None,
        }
    }

    /// Set minimum memory requirement
    pub fn with_min_memory(mut self, bytes: usize) -> Self {
        self.min_memory = Some(bytes);
        self
    }

    /// Set preferred backend
    pub fn with_backend(mut self, backend: GpuBackend) -> Self {
        self.preferred_backend = Some(backend);
        self
    }

    /// Set minimum compute capability
    pub fn with_compute_capability(mut self, major: u32, minor: u32) -> Self {
        self.min_compute_capability = Some((major, minor));
        self
    }

    /// Select best device from available devices
    pub fn select(&self, devices: &[GpuDevice]) -> Option<&GpuDevice> {
        devices
            .iter()
            .filter(|d| {
                // Check memory requirement
                if let Some(min_mem) = self.min_memory {
                    if d.memory_bytes < min_mem {
                        return false;
                    }
                }

                // Check backend preference
                if let Some(backend) = self.preferred_backend {
                    if d.backend != backend {
                        return false;
                    }
                }

                // Check compute capability
                if let Some((major, minor)) = self.min_compute_capability {
                    if d.compute_capability.0 < major
                        || (d.compute_capability.0 == major && d.compute_capability.1 < minor)
                    {
                        return false;
                    }
                }

                true
            })
            .max_by_key(|d| d.memory_bytes) // Select device with most memory
    }
}

impl Default for DeviceSelector {
    fn default() -> Self {
        Self::new()
    }
}

/// Enumerate available GPU devices
pub fn enumerate_devices() -> KwaversResult<Vec<GpuDevice>> {
    let mut devices = Vec::new();

    // Try each backend
    for backend in [GpuBackend::Cuda, GpuBackend::OpenCL, GpuBackend::WebGpu] {
        if let Ok(backend_devices) = enumerate_backend_devices(backend) {
            devices.extend(backend_devices);
        }
    }

    if devices.is_empty() {
        return Err(GpuError::NoDeviceFound.into());
    }

    info!("Found {} GPU device(s)", devices.len());
    Ok(devices)
}

/// Enumerate devices for a specific backend
fn enumerate_backend_devices(backend: GpuBackend) -> KwaversResult<Vec<GpuDevice>> {
    match backend {
        GpuBackend::Cuda if cfg!(feature = "cuda") => {
            // Would use CUDA API to enumerate devices
            Ok(vec![GpuDevice::new(0, backend)])
        }
        GpuBackend::OpenCL if cfg!(feature = "opencl") => {
            // Would use OpenCL API to enumerate devices
            Ok(vec![GpuDevice::new(0, backend)])
        }
        GpuBackend::WebGpu if cfg!(feature = "webgpu") => {
            // Would use WebGPU API to enumerate devices
            Ok(vec![GpuDevice::new(0, backend)])
        }
        _ => Ok(vec![]),
    }
}
