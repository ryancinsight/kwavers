//! GPU backend abstraction

use super::{ComputePipeline, GpuBuffer, GpuDevice};
use crate::KwaversResult;
use ndarray::Array3;
use std::sync::Arc;

/// GPU backend for acoustic simulations
pub struct GpuBackend {
    device: Arc<GpuDevice>,
    pipelines: Vec<ComputePipeline>,
}

impl GpuBackend {
    /// Create GPU backend
    pub async fn create() -> KwaversResult<Self> {
        let device = Arc::new(GpuDevice::create(wgpu::PowerPreference::HighPerformance).await?);

        Ok(Self {
            device,
            pipelines: Vec::new(),
        })
    }

    /// Create with specific power preference
    pub async fn create_with_preference(preference: wgpu::PowerPreference) -> KwaversResult<Self> {
        let device = Arc::new(GpuDevice::create(preference).await?);

        Ok(Self {
            device,
            pipelines: Vec::new(),
        })
    }

    /// Transfer field to GPU
    pub fn upload_field(&self, field: &Array3<f64>) -> KwaversResult<GpuBuffer> {
        let data: Vec<f32> = field.iter().map(|&x| x as f32).collect();

        GpuBuffer::create_with_data(
            self.device.device(),
            &data,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )
    }

    /// Download field from GPU
    pub async fn download_field(
        &self,
        buffer: &GpuBuffer,
        shape: (usize, usize, usize),
    ) -> KwaversResult<Array3<f64>> {
        let data = buffer
            .read_to_vec::<f32>(self.device.device(), self.device.queue())
            .await?;

        let array = Array3::from_shape_vec(shape, data.iter().map(|&x| x as f64).collect())
            .map_err(|e| {
                crate::KwaversError::Dimension(crate::error::DimensionError::InvalidShape {
                    expected: format!("{:?}", shape),
                    actual: format!("{} elements", data.len()),
                    context: format!("GPU download: {}", e),
                })
            })?;

        Ok(array)
    }

    /// Compile compute shader
    pub fn compile_shader(
        &mut self,
        name: &str,
        source: &str,
        entry_point: &str,
    ) -> KwaversResult<usize> {
        let pipeline = ComputePipeline::create(self.device.device(), name, source, entry_point)?;

        self.pipelines.push(pipeline);
        Ok(self.pipelines.len() - 1)
    }

    /// Execute compute pipeline
    pub fn dispatch(
        &self,
        pipeline_index: usize,
        workgroups: (u32, u32, u32),
        buffers: &[&GpuBuffer],
    ) -> KwaversResult<()> {
        if pipeline_index >= self.pipelines.len() {
            return Err(crate::KwaversError::Config(
                crate::ConfigError::InvalidValue {
                    field: "pipeline_index".to_string(),
                    value: pipeline_index.to_string(),
                    expected: format!("0..{}", self.pipelines.len()),
                },
            ));
        }

        let pipeline = &self.pipelines[pipeline_index];
        pipeline.dispatch(
            self.device.device(),
            self.device.queue(),
            workgroups,
            buffers,
        )
    }

    /// Get device info
    pub fn device_info(&self) -> &super::DeviceInfo {
        self.device.info()
    }

    /// Check if operation fits in memory
    pub fn can_fit(&self, bytes: usize) -> bool {
        // Conservative estimate - use 80% of max buffer size
        let max_size = self.device.limits().max_buffer_size as usize;
        bytes < (max_size * 4) / 5
    }
}
