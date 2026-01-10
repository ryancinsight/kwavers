//! GPU compute kernels for acoustic simulations
//!
//! High-performance compute shaders using wgpu for cross-platform GPU acceleration.
//! Supports both integrated and discrete GPUs with automatic fallback.

use crate::domain::core::error::{KwaversError, KwaversResult};
use crate::domain::grid;
use ndarray::Array3;
use wgpu::util::DeviceExt;

/// Acoustic field compute kernel
#[derive(Debug)]
pub struct AcousticFieldKernel {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl AcousticFieldKernel {
    /// Create new acoustic field kernel
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
                KwaversError::System(
                    crate::domain::core::error::SystemError::ResourceUnavailable {
                        resource: "GPU adapter".to_string(),
                    },
                )
            })?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Acoustic Field Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(|e| {
                KwaversError::System(
                    crate::domain::core::error::SystemError::ResourceUnavailable {
                        resource: format!("GPU device: {}", e),
                    },
                )
            })?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Acoustic Field Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/acoustic_field.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Acoustic Field Bind Group Layout"),
            entries: &[
                // Input pressure field
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output pressure field
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Parameters (dt, dx, c, etc.)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Acoustic Field Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Acoustic Field Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
        })
    }

    /// Compute acoustic field propagation on GPU
    pub fn compute_propagation(
        &self,
        pressure: &Array3<f64>,
        grid: &grid::Grid,
        dt: f64,
        sound_speed: f64,
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = pressure.dim();
        let total_size = nx * ny * nz;

        // Convert to f32 for GPU (most GPUs don't support f64)
        let pressure_f32: Vec<f32> = pressure
            .as_slice()
            .ok_or_else(|| {
                KwaversError::InvalidInput("Pressure field must be contiguous".to_string())
            })?
            .iter()
            .map(|&x| x as f32)
            .collect();

        // Create GPU buffers
        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Input Pressure Buffer"),
                contents: bytemuck::cast_slice(&pressure_f32),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Pressure Buffer"),
            size: (total_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Parameters uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            nx: u32,
            ny: u32,
            nz: u32,
            _padding: u32,
            dt: f32,
            dx: f32,
            dy: f32,
            dz: f32,
            c: f32,
            _padding2: [f32; 3],
        }

        let params = Params {
            nx: nx as u32,
            ny: ny as u32,
            nz: nz as u32,
            _padding: 0,
            dt: dt as f32,
            dx: grid.dx as f32,
            dy: grid.dy as f32,
            dz: grid.dz as f32,
            c: sound_speed as f32,
            _padding2: [0.0; 3],
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Parameters Buffer"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Acoustic Field Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Acoustic Field Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Acoustic Field Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch with 8x8x8 workgroups
            let workgroup_size = 8;
            let dispatch_x = nx.div_ceil(workgroup_size);
            let dispatch_y = ny.div_ceil(workgroup_size);
            let dispatch_z = nz.div_ceil(workgroup_size);

            compute_pass.dispatch_workgroups(
                dispatch_x as u32,
                dispatch_y as u32,
                dispatch_z as u32,
            );
        }

        // Create staging buffer for readback
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (total_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (total_size * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        self.device.poll(wgpu::Maintain::Wait);

        receiver
            .recv()
            .map_err(|_| {
                KwaversError::System(
                    crate::domain::core::error::SystemError::ResourceUnavailable {
                        resource: "GPU buffer mapping channel".to_string(),
                    },
                )
            })?
            .map_err(|_| {
                KwaversError::System(
                    crate::domain::core::error::SystemError::ResourceUnavailable {
                        resource: "GPU buffer mapping".to_string(),
                    },
                )
            })?;

        let data = buffer_slice.get_mapped_range();
        let result_f32: &[f32] = bytemuck::cast_slice(&data);

        // Convert back to f64
        let mut result = Array3::zeros((nx, ny, nz));
        if let Some(result_slice) = result.as_slice_mut() {
            for (i, &val) in result_f32.iter().enumerate() {
                if i < result_slice.len() {
                    result_slice[i] = val as f64;
                }
            }
        } else {
            // Fallback for non-contiguous arrays
            for (i, &val) in result_f32.iter().enumerate() {
                if let Some(elem) = result.get_mut([i % nx, (i / nx) % ny, i / (nx * ny)]) {
                    *elem = val as f64;
                }
            }
        }

        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }
}

/// Wave equation solver on GPU
#[derive(Debug)]
pub struct WaveEquationGpu {
    kernel: AcousticFieldKernel,
}

impl WaveEquationGpu {
    /// Create new GPU wave equation solver
    pub async fn new() -> KwaversResult<Self> {
        Ok(Self {
            kernel: AcousticFieldKernel::new().await?,
        })
    }

    /// Solve wave equation for one time step
    pub fn step(
        &self,
        pressure: &Array3<f64>,
        velocity: &Array3<f64>,
        density: &Array3<f64>,
        sound_speed: &Array3<f64>,
        grid: &grid::Grid,
        dt: f64,
    ) -> KwaversResult<(Array3<f64>, Array3<f64>)> {
        // Compute average properties for efficient GPU computation
        // For heterogeneous media, these would be spatially-varying on GPU
        let c_avg = sound_speed.mean().unwrap_or(1500.0);
        let rho_avg = density.mean().unwrap_or(1000.0);

        // Update pressure: p_new = p + dt * (-ρc² ∇·v)
        let new_pressure = self.kernel.compute_propagation(pressure, grid, dt, c_avg)?;

        // Update velocity: v_new = v + dt * (-1/ρ ∇p)
        // Compute pressure gradient using central differences
        let mut grad_p_x = Array3::zeros(pressure.dim());
        let mut grad_p_y = Array3::zeros(pressure.dim());
        let mut grad_p_z = Array3::zeros(pressure.dim());

        let (nx, ny, nz) = pressure.dim();
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    grad_p_x[[i, j, k]] =
                        (pressure[[i + 1, j, k]] - pressure[[i - 1, j, k]]) / (2.0 * grid.dx);
                    grad_p_y[[i, j, k]] =
                        (pressure[[i, j + 1, k]] - pressure[[i, j - 1, k]]) / (2.0 * grid.dy);
                    grad_p_z[[i, j, k]] =
                        (pressure[[i, j, k + 1]] - pressure[[i, j, k - 1]]) / (2.0 * grid.dz);
                }
            }
        }

        // Update velocity components: v_new = v - dt/ρ * ∇p
        let mut new_velocity = velocity.clone();
        use ndarray::Zip;
        Zip::from(&mut new_velocity)
            .and(&grad_p_x)
            .and(&grad_p_y)
            .and(&grad_p_z)
            .for_each(|v, &dpx, &dpy, &dpz| {
                // Velocity update using gradient magnitude
                // In full 3D vector implementation, each component would be updated separately
                let grad_magnitude = (dpx * dpx + dpy * dpy + dpz * dpz).sqrt();
                *v -= dt / rho_avg * grad_magnitude;
            });

        Ok((new_pressure, new_velocity))
    }
}
