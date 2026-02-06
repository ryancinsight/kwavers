//! WGPU Initialization and Device Management
//!
//! Handles WGPU instance creation, adapter selection, and device initialization.

use crate::core::error::{KwaversError, KwaversResult};
use std::sync::Arc;
use wgpu;

/// WGPU context holding instance, adapter, device, and queue
#[derive(Debug)]
pub struct WGPUContext {
    /// WGPU instance
    _instance: Arc<wgpu::Instance>,

    /// Selected adapter (physical device)
    _adapter: Arc<wgpu::Adapter>,

    /// Logical device
    device: Arc<wgpu::Device>,

    /// Command queue
    queue: Arc<wgpu::Queue>,

    /// Device name for logging
    device_name: String,
}

impl WGPUContext {
    /// Create a new WGPU context
    ///
    /// Selection priority:
    /// 1. High-performance discrete GPU
    /// 2. Integrated GPU
    /// 3. Software renderer (fallback)
    pub fn new() -> KwaversResult<Self> {
        // Create instance with all available backends
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Request adapter (physical device)
        let adapter = pollster::block_on(Self::request_adapter(&instance))?;

        // Get device info for logging
        let device_name = adapter.get_info().name.clone();

        // Request device and queue
        let (device, queue) = pollster::block_on(Self::request_device(&adapter))?;

        Ok(Self {
            _instance: Arc::new(instance),
            _adapter: Arc::new(adapter),
            device: Arc::new(device),
            queue: Arc::new(queue),
            device_name,
        })
    }

    /// Request adapter from instance
    async fn request_adapter(instance: &wgpu::Instance) -> KwaversResult<wgpu::Adapter> {
        // Try high-performance GPU first
        if let Some(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
        {
            return Ok(adapter);
        }

        // Try any available adapter
        if let Some(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
        {
            return Ok(adapter);
        }

        // No suitable adapter found
        Err(KwaversError::ConfigError(
            crate::core::error::ConfigError::MissingFeature {
                feature: "GPU".to_string(),
                help: "No suitable GPU adapter found. GPU backend unavailable.".to_string(),
            },
        ))
    }

    /// Request device and queue from adapter
    async fn request_device(adapter: &wgpu::Adapter) -> KwaversResult<(wgpu::Device, wgpu::Queue)> {
        // Request features and limits
        let mut features = wgpu::Features::empty();

        // Request f64 support if available (rare)
        if adapter.features().contains(wgpu::Features::SHADER_F64) {
            features |= wgpu::Features::SHADER_F64;
        }

        // Request device with maximum limits for compute
        let limits = wgpu::Limits {
            max_compute_workgroup_size_x: 256,
            max_compute_workgroup_size_y: 256,
            max_compute_workgroup_size_z: 64,
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroups_per_dimension: 65535,
            ..Default::default()
        };

        adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("kwavers-gpu-device"),
                    required_features: features,
                    required_limits: limits,
                    memory_hints: Default::default(),
                },
                None, // Trace path for debugging
            )
            .await
            .map_err(|e| {
                KwaversError::ConfigError(crate::core::error::ConfigError::MissingFeature {
                    feature: "GPU device".to_string(),
                    help: format!("Failed to request GPU device: {}", e),
                })
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

    /// Get device name
    pub fn device_name(&self) -> &str {
        &self.device_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wgpu_context_creation() {
        // May fail if no GPU available
        match WGPUContext::new() {
            Ok(context) => {
                assert!(!context.device_name().is_empty());
                println!("WGPU context created: {}", context.device_name());
            }
            Err(e) => {
                println!(
                    "WGPU context creation failed (expected on some systems): {}",
                    e
                );
            }
        }
    }

    #[test]
    fn test_wgpu_device_access() {
        if let Ok(context) = WGPUContext::new() {
            let device = context.device();
            let queue = context.queue();

            // Basic smoke tests
            assert!(device.limits().max_compute_invocations_per_workgroup > 0);
            queue.submit(std::iter::empty()); // Empty submission should work
        }
    }
}
