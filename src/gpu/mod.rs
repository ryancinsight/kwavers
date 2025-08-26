//! GPU acceleration module
//!
//! This module provides GPU acceleration for acoustic wave simulations,
//! following SOLID/CUPID principles with proper domain separation.

pub mod backend;
pub mod benchmarks;
pub mod context;
pub mod cuda;
pub mod device;
pub mod fft;
pub mod kernels;
pub mod memory;
pub mod memory_manager;
pub mod opencl;
pub mod traits;
pub mod wgpu_backend;

// Re-export key types for convenience
pub use backend::{gpu_float_type_str, GpuBackend};
pub use context::GpuContext;
pub use device::{enumerate_devices, DeviceSelector, GpuDevice};
pub use memory_manager::GpuMemoryManager;
pub use traits::GpuFieldOps;

use crate::error::{GpuError, KwaversResult};
use crate::grid::Grid;
use log::{debug, info, warn};
use std::sync::Arc;

/// GPU-accelerated solver configuration
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Selected backend
    pub backend: GpuBackend,
    /// Device ID to use
    pub device_id: usize,
    /// Enable async operations
    pub async_operations: bool,
    /// Memory pool size in bytes
    pub memory_pool_size: usize,
    /// Kernel cache size
    pub kernel_cache_size: usize,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            backend: GpuBackend::Cuda,
            device_id: 0,
            async_operations: true,
            memory_pool_size: 1024 * 1024 * 1024, // 1GB
            kernel_cache_size: 100,
        }
    }
}

/// GPU-accelerated field operations manager
pub struct GpuFieldManager {
    context: Arc<GpuContext>,
    memory_manager: GpuMemoryManager,
    config: GpuConfig,
}

impl GpuFieldManager {
    /// Create a new GPU field manager
    pub fn new(config: GpuConfig) -> KwaversResult<Self> {
        // Enumerate and select device
        let devices = enumerate_devices()?;
        let selector = DeviceSelector::new()
            .with_backend(config.backend)
            .with_min_memory(config.memory_pool_size);

        let device = selector
            .select(&devices)
            .ok_or(GpuError::NoDeviceFound)?
            .clone();

        info!(
            "Selected GPU device: {} ({})",
            device.name,
            device.backend.name()
        );

        // Create context
        let context = Arc::new(GpuContext::new(Arc::new(device))?);

        // Create memory manager
        let memory_manager = GpuMemoryManager::new(context.clone(), config.memory_pool_size)?;

        Ok(Self {
            context,
            memory_manager,
            config,
        })
    }

    /// Get the GPU context
    pub fn context(&self) -> &Arc<GpuContext> {
        &self.context
    }

    /// Get the memory manager
    pub fn memory_manager(&self) -> &GpuMemoryManager {
        &self.memory_manager
    }

    /// Check if GPU acceleration is available
    pub fn is_available() -> bool {
        enumerate_devices().map(|d| !d.is_empty()).unwrap_or(false)
    }

    /// Get available GPU memory
    pub fn available_memory(&self) -> usize {
        self.context.device().available_memory()
    }

    /// Synchronize GPU operations
    pub fn synchronize(&self) -> KwaversResult<()> {
        self.context.synchronize()
    }
}

/// Initialize GPU subsystem
pub fn initialize() -> KwaversResult<()> {
    info!("Initializing GPU subsystem");

    // Check for available backends
    let backend = GpuBackend::auto_select()?;
    info!("Selected GPU backend: {}", backend.name());

    // Enumerate devices
    let devices = enumerate_devices()?;
    info!("Found {} GPU device(s)", devices.len());

    for device in devices {
        info!(
            "  Device {}: {} ({} MB memory)",
            device.id,
            device.name,
            device.memory_bytes / (1024 * 1024)
        );
    }

    Ok(())
}

/// Check if GPU is available and properly configured
pub fn check_gpu_availability() -> bool {
    GpuFieldManager::is_available()
}

/// Get GPU information string
pub fn gpu_info() -> String {
    if let Ok(devices) = enumerate_devices() {
        if !devices.is_empty() {
            let device = &devices[0];
            format!(
                "{} - {} ({}MB memory)",
                device.backend.name(),
                device.name,
                device.memory_bytes / (1024 * 1024)
            )
        } else {
            "No GPU devices found".to_string()
        }
    } else {
        "GPU not available".to_string()
    }
}
