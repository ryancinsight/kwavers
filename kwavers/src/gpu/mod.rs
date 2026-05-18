//! GPU acceleration module for acoustic simulations
//!
//! This module provides GPU-accelerated implementations of core algorithms
//! using wgpu-rs for cross-platform GPU compute.
//! ## Not yet implemented
//!
//! - **Bubble dynamics kernels**: CUDA/OpenCL/wgsl compute shaders for Keller-Miksis
//!   equation integration on-device.
//! - **GPU-accelerated PINN training**: Automatic differentiation for physics-informed
//!   networks without CPU round-trips.
//! - **Unified multi-GPU memory**: Shared memory management for large-scale cavitation
//!   simulations across multiple devices.
//! - **Memory coalescing**: Layout optimizations for 3D grid accesses on GPU.

pub mod backend;
pub mod buffer;
pub mod buffers;
#[cfg(feature = "pinn")]
pub mod burn_accelerator;
pub mod compute;
pub mod compute_kernels;
pub mod compute_manager;
pub mod device;
pub mod fdtd;
pub mod kernels;
pub mod kspace;
pub mod memory;
pub mod multi_gpu;
pub mod pipeline;
pub mod shaders;
pub mod thermal_acoustic;

pub use backend::GpuBackend;
// Single canonical GpuBuffer — `gpu/buffer.rs` is the SSOT.
// `gpu/buffers.rs` exposes only GpuBufferManager (the named-pool layer).
pub use buffer::{BufferUsage, GpuBuffer};
pub use buffers::GpuBufferManager;
#[cfg(feature = "pinn")]
pub use burn_accelerator::BurnGpuAccelerator;
pub use compute::{FdtdGpuDispatcher, GpuCompute};
pub use compute_kernels::{AcousticFieldKernel, WaveEquationGpu};
pub use device::{DeviceInfo, GpuDevice};
pub use fdtd::FdtdGpu;
pub use kspace::{KSpaceGpu, KspaceShiftGpu};
pub use memory::{GpuMemoryPoolType, UnifiedMemoryManager};
pub use multi_gpu::{GpuAffinity, MultiGpuContext};
pub use shaders::neural_network::NeuralNetworkShader;
pub use thermal_acoustic::{
    GpuThermalAcousticBuffers, GpuThermalAcousticConfig, GpuThermalAcousticSolver,
};

use crate::core::error::{KwaversError, KwaversResult};

/// GPU device capabilities
#[derive(Debug, Clone)]
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
#[derive(Debug)]
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    capabilities: GpuCapabilities,
}

impl GpuContext {
    /// Create a new GPU context
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub async fn new() -> KwaversResult<Self> {
        // Create instance with all backends
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|_| {
                KwaversError::System(crate::core::error::SystemError::ResourceUnavailable {
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
            .request_device(&wgpu::DeviceDescriptor {
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
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| {
                KwaversError::System(crate::core::error::SystemError::ResourceUnavailable {
                    resource: format!("GPU device: {}", e),
                })
            })?;

        let capabilities = GpuCapabilities {
            max_buffer_size: limits.max_buffer_size,
            max_workgroup_size: [
                limits.max_compute_workgroup_size_x,
                limits.max_compute_workgroup_size_y,
                limits.max_compute_workgroup_size_z,
            ],
            max_compute_invocations: limits.max_compute_invocations_per_workgroup,
            supports_f64: adapter.features().contains(wgpu::Features::SHADER_F64),
            supports_atomics: true, // Most modern GPUs support this
        };

        Ok(Self {
            device,
            queue,
            capabilities,
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
