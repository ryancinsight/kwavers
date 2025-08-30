//! GPU device management

use crate::{KwaversError, KwaversResult};
use std::sync::Arc;

/// Information about a GPU device
#[derive(Debug, Clone))]
#[derive(Debug))]
pub struct DeviceInfo {
    pub name: String,
    pub vendor: u32,
    pub device_type: String,
    pub backend: String,
}

/// GPU device wrapper
#[derive(Debug))]
pub struct GpuDevice {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    info: DeviceInfo,
    limits: wgpu::Limits,
}

impl GpuDevice {
    /// Create GPU device
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

    /// Get device reference
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get queue reference
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Get device info
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// Get device limits
    pub fn limits(&self) -> &wgpu::Limits {
        self.limits.clone()
    }

    /// Check if device supports feature
    pub fn supports_feature(&self, feature: wgpu::Features) -> bool {
        self.device.features().contains(feature)
    }
}
