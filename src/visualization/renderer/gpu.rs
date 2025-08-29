//! GPU context and resource management

use crate::error::{KwaversError, KwaversResult};
use crate::visualization::VisualizationConfig;

/// GPU context for accelerated rendering
pub struct GpuContext {
    #[cfg(feature = "gpu-visualization")]
    device: std::sync::Arc<wgpu::Device>,
    #[cfg(feature = "gpu-visualization")]
    queue: std::sync::Arc<wgpu::Queue>,
    memory_usage: usize,
}

impl GpuContext {
    /// Create a new GPU context
    pub fn new(_config: &VisualizationConfig) -> KwaversResult<Self> {
        #[cfg(feature = "gpu-visualization")]
        {
            // Initialize GPU
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                dx12_shader_compiler: Default::default(),
            });

            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }))
                .ok_or_else(|| {
                    KwaversError::System(crate::error::SystemError::ResourceUnavailable {
                        resource: "GPU adapter".to_string(),
                    })
                })?;

            let (device, queue) = pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Kwavers GPU Device"),
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None,
            ))
            .map_err(|e| {
                KwaversError::System(crate::error::SystemError::ResourceUnavailable {
                    resource: format!("GPU device: {}", e),
                })
            })?;

            Ok(Self {
                device: std::sync::Arc::new(device),
                queue: std::sync::Arc::new(queue),
                memory_usage: 0,
            })
        }

        #[cfg(not(feature = "gpu-visualization"))]
        {
            Ok(Self { memory_usage: 0 })
        }
    }

    /// Upload data to GPU
    pub fn upload_volume(&mut self, data: &[f32]) -> KwaversResult<()> {
        #[cfg(feature = "gpu-visualization")]
        {
            let buffer_size = data.len() * std::mem::size_of::<f32>();
            self.memory_usage += buffer_size;

            // Create buffer and upload data
            let _buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Volume Buffer"),
                    contents: bytemuck::cast_slice(data),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
        }

        Ok(())
    }

    /// Get memory usage
    pub fn memory_usage(&self) -> usize {
        self.memory_usage
    }
}
