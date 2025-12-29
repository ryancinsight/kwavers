//! GPU device management
//!
//! This module provides high-level GPU device initialization and management.
//! It handles device discovery, capability querying, and resource management.
//!
//! # Examples
//!
//! ```no_run
//! # use kwavers::gpu::device::GpuDevice;
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a GPU device with high performance preference
//! let device = GpuDevice::create(wgpu::PowerPreference::HighPerformance).await?;
//!
//! // Query device information
//! let info = device.info();
//! println!("Using GPU: {}", info.name);
//! println!("Backend: {}", info.backend);
//!
//! // Check device limits
//! let limits = device.limits();
//! println!("Max buffer size: {} bytes", limits.max_buffer_size);
//! # Ok(())
//! # }
//! ```

use crate::{KwaversError, KwaversResult};
use std::sync::Arc;

/// Information about a GPU device
///
/// Contains metadata about the GPU including name, vendor, device type, and backend.
/// This information is useful for debugging, logging, and device selection.
///
/// # Fields
///
/// * `name` - Human-readable device name (e.g., "NVIDIA GeForce RTX 3080")
/// * `vendor` - PCI vendor ID (e.g., 0x10DE for NVIDIA)
/// * `device_type` - Device type as string (e.g., "DiscreteGpu", "IntegratedGpu")
/// * `backend` - Graphics backend being used (e.g., "Vulkan", "Metal", "Dx12")
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device name
    pub name: String,
    /// PCI vendor ID
    pub vendor: u32,
    /// Device type (DiscreteGpu, IntegratedGpu, VirtualGpu, Cpu, Other)
    pub device_type: String,
    /// Graphics backend (Vulkan, Metal, Dx12, Dx11, Gl, BrowserWebGpu)
    pub backend: String,
}

/// GPU device wrapper providing high-level device management
///
/// `GpuDevice` encapsulates a wgpu device and queue, providing a convenient
/// interface for GPU resource management. It handles device initialization,
/// capability querying, and provides access to device limits and features.
///
/// The device and queue are reference-counted (Arc) to allow safe sharing
/// across threads and components.
///
/// # Examples
///
/// ```no_run
/// # use kwavers::gpu::device::GpuDevice;
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Create device with power preference
/// let device = GpuDevice::create(wgpu::PowerPreference::LowPower).await?;
///
/// // Use device for buffer operations
/// let buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
///     label: Some("my_buffer"),
///     size: 1024,
///     usage: wgpu::BufferUsages::STORAGE,
///     mapped_at_creation: false,
/// });
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct GpuDevice {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    info: DeviceInfo,
    limits: wgpu::Limits,
}

impl GpuDevice {
    /// Create a new GPU device with specified power preference
    ///
    /// Initializes a GPU device by requesting an adapter and device from wgpu.
    /// This is an async operation that may take some time depending on the system.
    ///
    /// # Arguments
    ///
    /// * `power_preference` - Hint for device selection:
    ///   - `HighPerformance` - Prefer discrete/powerful GPUs
    ///   - `LowPower` - Prefer integrated/power-efficient GPUs
    ///
    /// # Returns
    ///
    /// Returns `Ok(GpuDevice)` on success, or an error if no suitable GPU is found.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No GPU adapter matching the criteria is found
    /// - Device creation fails (e.g., driver issues, insufficient permissions)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kwavers::gpu::device::GpuDevice;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// // Create high-performance device for compute
    /// let device = GpuDevice::create(wgpu::PowerPreference::HighPerformance).await?;
    ///
    /// // Create low-power device for mobile
    /// let mobile_device = GpuDevice::create(wgpu::PowerPreference::LowPower).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn create(power_preference: wgpu::PowerPreference) -> KwaversResult<Self> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or_else(|| {
                KwaversError::Config(crate::ConfigError::InvalidValue {
                    parameter: "gpu".to_string(),
                    value: "none".to_string(),
                    constraint: "GPU adapter not found".to_string(),
                })
            })?;

        let adapter_info = adapter.get_info();
        let limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("kwavers_gpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .map_err(|e| {
                KwaversError::Config(crate::ConfigError::InvalidValue {
                    parameter: "gpu_device".to_string(),
                    value: format!("{:?}", e),
                    constraint: "Failed to create GPU device".to_string(),
                })
            })?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            info: DeviceInfo {
                name: adapter_info.name,
                vendor: adapter_info.vendor,
                device_type: format!("{:?}", adapter_info.device_type),
                backend: format!("{:?}", adapter_info.backend),
            },
            limits,
        })
    }

    /// Get reference to underlying wgpu device
    ///
    /// Provides access to the wgpu device for buffer creation, shader compilation,
    /// and other GPU operations.
    ///
    /// # Returns
    ///
    /// Reference to the underlying `wgpu::Device`.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get reference to device queue
    ///
    /// The queue is used for submitting command buffers, writing to buffers,
    /// and other GPU submission operations.
    ///
    /// # Returns
    ///
    /// Reference to the underlying `wgpu::Queue`.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Get device information
    ///
    /// Returns metadata about the GPU including name, vendor, and backend.
    /// Useful for logging, debugging, and device-specific optimizations.
    ///
    /// # Returns
    ///
    /// Reference to `DeviceInfo` containing device metadata.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kwavers::gpu::device::GpuDevice;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let device = GpuDevice::create(wgpu::PowerPreference::HighPerformance).await?;
    /// let info = device.info();
    /// println!("GPU: {} (Vendor: 0x{:X})", info.name, info.vendor);
    /// println!("Backend: {}", info.backend);
    /// # Ok(())
    /// # }
    /// ```
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// Get device limits
    ///
    /// Returns the device's resource limits including maximum buffer sizes,
    /// workgroup dimensions, and other constraints. These limits determine
    /// what operations are possible on the device.
    ///
    /// # Returns
    ///
    /// Reference to `wgpu::Limits` containing device constraints.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kwavers::gpu::device::GpuDevice;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let device = GpuDevice::create(wgpu::PowerPreference::HighPerformance).await?;
    /// let limits = device.limits();
    /// println!("Max buffer size: {} bytes", limits.max_buffer_size);
    /// println!("Max workgroup size: {:?}", (
    ///     limits.max_compute_workgroup_size_x,
    ///     limits.max_compute_workgroup_size_y,
    ///     limits.max_compute_workgroup_size_z
    /// ));
    /// # Ok(())
    /// # }
    /// ```
    pub fn limits(&self) -> &wgpu::Limits {
        &self.limits
    }

    /// Check if device supports a specific feature
    ///
    /// Tests whether the GPU supports an optional wgpu feature.
    /// This is useful for enabling advanced functionality when available.
    ///
    /// # Arguments
    ///
    /// * `feature` - The feature to check (e.g., `Features::TIMESTAMP_QUERY`)
    ///
    /// # Returns
    ///
    /// `true` if the feature is supported, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kwavers::gpu::device::GpuDevice;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let device = GpuDevice::create(wgpu::PowerPreference::HighPerformance).await?;
    ///
    /// if device.supports_feature(wgpu::Features::TIMESTAMP_QUERY) {
    ///     println!("GPU supports timestamp queries for profiling");
    /// }
    ///
    /// if device.supports_feature(wgpu::Features::SHADER_F64) {
    ///     println!("GPU supports 64-bit floating point in shaders");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn supports_feature(&self, feature: wgpu::Features) -> bool {
        self.device.features().contains(feature)
    }
}
