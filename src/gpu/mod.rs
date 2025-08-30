//! GPU acceleration module for acoustic simulations
//!
//! This module provides GPU-accelerated implementations of core algorithms
//! using wgpu-rs for cross-platform GPU compute.

pub mod buffers;
pub mod compute;
pub mod fdtd;
pub mod kernels;
pub mod kspace;
pub mod pipeline;

pub use buffers::{BufferManager, GpuBuffer};
pub use compute::GpuCompute;
pub use fdtd::FdtdGpu;
pub use kspace::KSpaceGpu;
pub use pipeline::{ComputePipeline, PipelineLayout};

use crate::error::{KwaversError, KwaversResult};

/// GPU device capabilities
#[derive(Debug, Clone))]
pub struct GpuCapabilities {
    /// Maximum buffer size in bytes
    pub max_buffer_size: u64,
    /// Maximum workgroup size
    pub max_workgroup_size: [u32; 3],
    /// Maximum compute invocations per workgroup
    pub max_compute_invocations: u32,
    /// Supports 64-bit floats
    pub supports_f64: bool,
    /// Supports atomic operations
    pub supports_atomics: bool,
}

/// Main GPU context for acoustic simulations
#[derive(Debug))]
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    capabilities: GpuCapabilities,
    compute: GpuCompute,
    buffer_manager: BufferManager,
}

impl GpuContext {
    /// Create a new GPU context
    pub async fn new() -> KwaversResult<Self> {
        // Create instance with all backends
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });

        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| {
                KwaversError::System(crate::error::SystemError::ResourceUnavailable {
                    resource: "GPU adapter".to_string(),
                })
            })?;

        // Get adapter info and limits
        let info = adapter.get_info();
        let limits = adapter.limits();

        log::info!(
            "GPU: {} ({:?}) - Driver: {}",
            info.name,
            info.backend,
            info.driver
        );

        // Request device with required features
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Kwavers GPU Device"),
                    required_features: wgpu::Features::MAPPABLE_PRIMARY_BUFFERS
                        | wgpu::Features::PUSH_CONSTANTS,
                    required_limits: wgpu::Limits {
                        max_buffer_size: limits.max_buffer_size,
                        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
                        max_compute_workgroup_storage_size: 16384,
                        max_compute_invocations_per_workgroup: 256,
                        max_compute_workgroup_size_x: 256,
                        max_compute_workgroup_size_y: 256,
                        max_compute_workgroup_size_z: 64,
                        max_push_constant_size: 128,
                        ..Default::default()
                    },
                },
                None,
            )
            .await
            .map_err(|e| {
                KwaversError::System(crate::error::SystemError::ResourceUnavailable {
                    resource: format!("GPU device: {}", e),
                })
            })?;

        let capabilities = GpuCapabilities {
            max_buffer_size: limits.max_buffer_size as u64,
            max_workgroup_size: [
                limits.max_compute_workgroup_size_x,
                limits.max_compute_workgroup_size_y,
                limits.max_compute_workgroup_size_z,
            ],
            max_compute_invocations: limits.max_compute_invocations_per_workgroup,
            supports_f64: adapter.features().contains(wgpu::Features::SHADER_F64),
            supports_atomics: true, // Most modern GPUs support this
        };

        let compute = GpuCompute::new(&device);
        let buffer_manager = BufferManager::new(&device);

        Ok(Self {
            device,
            queue,
            capabilities,
            compute,
            buffer_manager,
        })
    }

    /// Get device reference
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get queue reference
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Get capabilities
    pub fn capabilities(&self) -> &GpuCapabilities {
        &self.capabilities
    }

    /// Submit command buffer
    pub fn submit(&self, commands: wgpu::CommandBuffer) {
        self.queue.submit(std::iter::once(commands));
    }
}
