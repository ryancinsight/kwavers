use crate::core::error::{KwaversError, KwaversResult};

#[derive(Debug)]
struct ComputeManager {
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
}

impl ComputeManager {
    /// New.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub async fn new() -> KwaversResult<Self> {
        match Self::init_gpu().await {
            Ok((device, queue)) => Ok(Self {
                device: Some(device),
                queue: Some(queue),
            }),
            Err(_) => Ok(Self {
                device: None,
                queue: None,
            }),
        }
    }
    /// New blocking.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new_blocking() -> KwaversResult<Self> {
        pollster::block_on(Self::new())
    }
    /// Has gpu.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn has_gpu(&self) -> bool {
        self.device.is_some() && self.queue.is_some()
    }
    /// Device.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn device(&self) -> KwaversResult<&wgpu::Device> {
        self.device.as_ref().ok_or_else(|| {
            KwaversError::System(crate::core::error::SystemError::ResourceUnavailable {
                resource: "GPU device".to_string(),
            })
        })
    }
    /// Queue.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn queue(&self) -> KwaversResult<&wgpu::Queue> {
        self.queue.as_ref().ok_or_else(|| {
            KwaversError::System(crate::core::error::SystemError::ResourceUnavailable {
                resource: "GPU queue".to_string(),
            })
        })
    }
    /// Create buffer.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn create_buffer(
        &self,
        size_bytes: usize,
        usage: wgpu::BufferUsages,
    ) -> KwaversResult<wgpu::Buffer> {
        let device = self.device()?;
        Ok(device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size_bytes as u64,
            usage,
            mapped_at_creation: false,
        }))
    }
    /// Write buffer.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn write_buffer<T: bytemuck::Pod>(
        &self,
        buffer: &wgpu::Buffer,
        data: &[T],
    ) -> KwaversResult<()> {
        let queue = self.queue()?;
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));
        Ok(())
    }

    async fn init_gpu() -> KwaversResult<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|_| KwaversError::GpuError("No GPU adapter found".into()))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Kwavers EM Compute Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| KwaversError::GpuError(format!("Failed to create device: {e}")))?;

        Ok((device, queue))
    }
}
