//! GPU acceleration module for acoustic simulations
//!
//! This module provides GPU-accelerated implementations of core algorithms
//! through provider-generic Hephaestus device seams. The current production
//! kernels use WGSL over WGPU; CUDA and future providers enter through the
//! provider traits before they implement real kernel dispatch.
//! ## Not yet implemented
//!
//! - **Bubble dynamics kernels**: CUDA/OpenCL/WGSL compute shaders for Keller-Miksis
//!   equation integration on-device.
//! - **GPU-accelerated PINN training**: Automatic differentiation for physics-informed
//!   networks without CPU round-trips.
//! - **Unified multi-GPU memory**: Shared memory management for large-scale cavitation
//!   simulations across multiple devices.
//! - **Memory coalescing**: Layout optimizations for 3D grid accesses on GPU.

pub mod backend;
pub mod buffer;
pub mod buffers;
pub mod compute;
pub mod compute_kernels;
pub mod compute_manager;
pub mod device;
pub mod fdtd;
pub mod memory;
pub mod multi_gpu;
pub mod pipeline;
pub mod shaders;
pub mod thermal_acoustic;

pub use backend::GpuBackend;
// Single canonical GpuBufferData — `gpu/buffer.rs` is the SSOT.
// `gpu/buffers.rs` exposes only GpuBufferManager (the named-pool layer).
pub use buffer::{BufferUsage, GpuBufferData};
pub use buffers::GpuBufferManager;
pub use compute::{FdtdCpuReferenceDispatcher, WgpuComputeCommands};
pub use compute_kernels::{AcousticFieldKernel, WaveEquationGpu};
pub use device::{GpuDevice, GpuDeviceInfo, GpuDeviceProvider};
pub use fdtd::{FdtdGpuProvider, WgpuFdtd};
pub use memory::{GpuMemoryPoolType, UnifiedMemoryManager};
pub use multi_gpu::{GpuAffinity, MultiGpuContext};
pub use shaders::neural_network::NeuralNetworkShader;
pub use thermal_acoustic::{
    GpuThermalAcousticBuffers, GpuThermalAcousticConfig, GpuThermalAcousticSolver,
    ThermalAcousticBufferProvider, ThermalAcousticSolverProvider, WgpuThermalAcousticBuffers,
    WgpuThermalAcousticSolverProvider,
};

use hephaestus_core::{DeviceFeature, DeviceLimits, DevicePreference};
use hephaestus_wgpu::WgpuDevice;
use kwavers_core::error::{KwaversError, KwaversResult};

pub(crate) fn map_buffer_async_error(
    context: &'static str,
    err: wgpu::BufferAsyncError,
) -> KwaversError {
    KwaversError::GpuError(format!("{context}: {err}"))
}

/// GPU device capabilities
#[derive(Debug, Clone)]
pub struct CoreGpuCapabilities {
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
pub struct CoreGpuContext<P = WgpuDevice>
where
    P: GpuDeviceProvider,
{
    device: GpuDevice<P>,
    capabilities: CoreGpuCapabilities,
}

impl<P> CoreGpuContext<P>
where
    P: GpuDeviceProvider,
{
    /// Acquire a provider-backed GPU context with explicit requirements.
    ///
    /// # Errors
    ///
    /// Returns a system resource error when the provider cannot satisfy the
    /// requested device preference, optional features, or limits.
    pub fn acquire_with_requirements(
        label: &str,
        device_preference: DevicePreference,
        optional_features: &[DeviceFeature],
        required_limits: DeviceLimits,
    ) -> KwaversResult<Self> {
        let provider =
            P::try_acquire_device(label, device_preference, optional_features, required_limits)
                .map_err(|e| {
                    KwaversError::System(kwavers_core::error::SystemError::ResourceUnavailable {
                        resource: format!("GPU device: {e}"),
                    })
                })?;

        Ok(Self::from_provider(provider))
    }

    /// Build a context from an already-acquired Hephaestus provider.
    #[must_use]
    pub fn from_provider(provider: P) -> Self {
        let device = GpuDevice::from_provider(provider);
        let info = device.info();
        log::info!("GPU: {} ({})", info.name, info.backend);

        let limits = device.limits();
        let capabilities = CoreGpuCapabilities {
            max_buffer_size: limits.max_buffer_size,
            max_workgroup_size: [
                limits.max_compute_workgroup_size_x,
                limits.max_compute_workgroup_size_y,
                limits.max_compute_workgroup_size_z,
            ],
            max_compute_invocations: limits.max_compute_invocations_per_workgroup,
            supports_f64: device.supports_feature(DeviceFeature::ShaderF64),
            supports_atomics: P::supports_core_atomics(),
        };

        Self {
            device,
            capabilities,
        }
    }

    /// Borrow the concrete provider device.
    #[must_use]
    pub fn provider(&self) -> &P {
        self.device.provider()
    }

    /// Get capabilities
    pub fn capabilities(&self) -> &CoreGpuCapabilities {
        &self.capabilities
    }
}

impl CoreGpuContext<WgpuDevice> {
    /// Create a new WGPU context for the current WGSL kernels.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub async fn new() -> KwaversResult<Self> {
        Self::try_new()
    }

    /// Create a new WGPU context synchronously for the current WGSL kernels.
    ///
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    pub fn try_new() -> KwaversResult<Self> {
        Self::acquire_with_requirements(
            "Kwavers GPU Device",
            DevicePreference::HighPerformance,
            &[
                DeviceFeature::MappablePrimaryBuffers,
                DeviceFeature::PushConstants,
            ],
            Self::required_limits(),
        )
    }

    pub(crate) fn required_limits() -> DeviceLimits {
        let baseline = device::minimal_compute_limits();
        DeviceLimits {
            max_buffer_size: baseline.max_buffer_size,
            max_storage_buffers_per_shader_stage: baseline.max_storage_buffers_per_shader_stage,
            max_compute_workgroup_storage_size: 16384,
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroup_size_x: 256,
            max_compute_workgroup_size_y: 256,
            max_compute_workgroup_size_z: 64,
            max_push_constant_size: 128,
        }
    }

    /// Get device reference
    pub fn device(&self) -> &wgpu::Device {
        self.device.wgpu_device()
    }

    /// Get queue reference
    pub fn queue(&self) -> &wgpu::Queue {
        self.device.wgpu_queue()
    }

    /// Submit command buffer
    pub fn submit(&self, commands: wgpu::CommandBuffer) {
        self.queue().submit(std::iter::once(commands));
    }
}
