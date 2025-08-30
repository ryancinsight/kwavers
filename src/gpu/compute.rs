//! GPU-accelerated compute kernels for wave propagation
//!
//! Uses wgpu for cross-platform GPU acceleration of critical 3D operations.

use crate::error::KwaversResult;
use ndarray::Array3;
use wgpu::util::DeviceExt;

/// GPU compute engine for accelerated wave propagation
pub struct GpuCompute {
    device: wgpu::Device,
    queue: wgpu::Queue,
    fdtd_pipeline: wgpu::ComputePipeline,
    laplacian_pipeline: wgpu::ComputePipeline,
}

impl GpuCompute {
    /// Create a new GPU compute engine
    pub async fn new() -> KwaversResult<Self> {
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
            .ok_or_else(|| {
                crate::error::KwaversError::GpuError("No suitable GPU adapter found".to_string())
            })?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Kwavers GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| crate::error::KwaversError::GpuError(e.to_string()))?;

        // Create compute pipelines
        let fdtd_pipeline = Self::create_fdtd_pipeline(&device);
        let laplacian_pipeline = Self::create_laplacian_pipeline(&device);

        Ok(Self {
            device,
            queue,
            fdtd_pipeline,
            laplacian_pipeline,
        })
    }

    /// Create FDTD compute pipeline
    fn create_fdtd_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FDTD Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fdtd.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FDTD Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("FDTD Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FDTD Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "fdtd_update",
        })
    }

    /// Create Laplacian compute pipeline
    fn create_laplacian_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Laplacian Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/laplacian.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Laplacian Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Laplacian Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Laplacian Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "laplacian_3d",
        })
    }

    /// Compute 3D Laplacian on GPU
    pub async fn laplacian_3d(
        &self,
        input: &Array3<f32>,
        dx: f32,
        dy: f32,
        dz: f32,
    ) -> KwaversResult<Array3<f32>> {
        let (nx, ny, nz) = input.dim();
        let size = (nx * ny * nz) as u64 * std::mem::size_of::<f32>() as u64;

        // Create GPU buffers
        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Input Buffer"),
                contents: bytemuck::cast_slice(input.as_slice().unwrap()),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create uniform buffer for parameters
        let params = [dx, dy, dz, nx as f32, ny as f32, nz as f32];
        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Params Buffer"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group
        let bind_group_layout = self.laplacian_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Laplacian Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute compute pass
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Laplacian Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Laplacian Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.laplacian_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch with 8x8x8 workgroups
            let workgroups_x = (nx + 7) / 8;
            let workgroups_y = (ny + 7) / 8;
            let workgroups_z = (nz + 7) / 8;
            compute_pass.dispatch_workgroups(
                workgroups_x as u32,
                workgroups_y as u32,
                workgroups_z as u32,
            );
        }

        // Copy result back to CPU
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, size);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back results
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);
        rx.await
            .unwrap()
            .map_err(|e| crate::error::KwaversError::GpuError(e.to_string()))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(Array3::from_shape_vec((nx, ny, nz), result)
            .map_err(|e| crate::error::KwaversError::ShapeError(e.to_string()))?)
    }

    /// GPU-accelerated FDTD update
    pub async fn fdtd_update(
        &self,
        pressure: &Array3<f32>,
        velocity: &Array3<f32>,
        density: &Array3<f32>,
        sound_speed: &Array3<f32>,
        dt: f32,
    ) -> KwaversResult<Array3<f32>> {
        let (nx, ny, nz) = pressure.dim();

        // Similar GPU implementation for FDTD
        // This would use the fdtd_pipeline with appropriate buffers

        // For now, return a placeholder
        Ok(pressure.clone())
    }
}
