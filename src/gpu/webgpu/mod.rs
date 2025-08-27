//! WebGPU backend implementation
//!
//! This module provides GPU acceleration using the WebGPU standard,
//! supporting Vulkan, Metal, DirectX 12, and browser WebGPU backends.

pub mod context;
pub mod kernels;
pub mod memory;
pub mod shaders;

pub use context::WebGpuContext;
pub use memory::{allocate_memory, device_to_host, host_to_device};

use crate::error::KwaversResult;
use crate::gpu::GpuDevice;

/// Detect available WebGPU devices synchronously
pub fn detect_devices() -> KwaversResult<Vec<GpuDevice>> {
    #[cfg(feature = "wgpu")]
    {
        use crate::gpu::GpuBackend;

        // Use async runtime to detect devices
        let runtime = tokio::runtime::Runtime::new()?;
        runtime.block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });

            let adapters = instance.enumerate_adapters(wgpu::Backends::all());
            let mut devices = Vec::new();

            for (id, adapter) in adapters.enumerate() {
                let info = adapter.get_info();
                devices.push(GpuDevice {
                    id,
                    name: info.name.clone(),
                    backend: match info.backend {
                        wgpu::Backend::Vulkan => GpuBackend::WebGpu,
                        wgpu::Backend::Metal => GpuBackend::WebGpu,
                        wgpu::Backend::Dx12 => GpuBackend::WebGpu,
                        wgpu::Backend::BrowserWebGpu => GpuBackend::WebGpu,
                        _ => GpuBackend::WebGpu,
                    },
                    compute_units: 0, // WebGPU doesn't expose this directly
                    memory_size: 0,   // Would need to query limits
                    max_workgroup_size: adapter.limits().max_compute_workgroup_size_x as usize,
                });
            }

            Ok(devices)
        })
    }

    #[cfg(not(feature = "wgpu"))]
    {
        Ok(Vec::new())
    }
}
