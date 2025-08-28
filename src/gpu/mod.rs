//! GPU acceleration support using wgpu-rs
//!
//! Provides unified GPU compute for integrated and discrete GPUs
//! through WebGPU API standard.

pub mod backend;
pub mod buffer;
pub mod compute;
pub mod device;
pub mod kernels;
pub mod memory;

pub use backend::GpuBackend;
pub use buffer::{BufferUsage, GpuBuffer};
pub use compute::ComputePipeline;
pub use device::{DeviceInfo, GpuDevice};

use crate::KwaversResult;

/// GPU capability detection
pub fn is_gpu_available() -> bool {
    pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await;
        adapter.is_some()
    })
}

/// Get available GPU devices
pub fn enumerate_devices() -> Vec<DeviceInfo> {
    pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapters = instance.enumerate_adapters(wgpu::Backends::all());

        adapters
            .map(|adapter| {
                let info = adapter.get_info();
                DeviceInfo {
                    name: info.name,
                    vendor: info.vendor,
                    device_type: match info.device_type {
                        wgpu::DeviceType::DiscreteGpu => "Discrete GPU",
                        wgpu::DeviceType::IntegratedGpu => "Integrated GPU",
                        wgpu::DeviceType::VirtualGpu => "Virtual GPU",
                        wgpu::DeviceType::Cpu => "CPU",
                        wgpu::DeviceType::Other => "Other",
                    }
                    .to_string(),
                    backend: format!("{:?}", info.backend),
                }
            })
            .collect()
    })
}
