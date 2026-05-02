//! Constructor and GPU resource initialization for `GPUEMSolver`.
//!
//! SRP: changes when the GPU pipeline layout, shader path, or CFL validation changes.

use super::compute::ComputeManager;
use super::config::EMConfig;
use super::fields::EMFieldData;
use super::solver::GPUEMSolver;
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{Array4, Array5};
use std::collections::HashMap;

impl GPUEMSolver {
    /// Create a new GPU-accelerated electromagnetic solver.
    pub fn new(config: EMConfig) -> KwaversResult<Self> {
        Self::validate_config(&config)?;
        let compute_manager = ComputeManager::new_blocking()?;
        Ok(Self {
            config,
            compute_manager,
            field_data: None,
            gpu_buffers: HashMap::new(),
            compute_pipeline: None,
            bind_group_layout: None,
            bind_group: None,
        })
    }

    /// Validate solver configuration against CFL stability and positivity constraints.
    fn validate_config(config: &EMConfig) -> KwaversResult<()> {
        if config.grid_size.contains(&0) {
            return Err(KwaversError::Validation(
                crate::core::error::ValidationError::ConstraintViolation {
                    message: "grid_size must be positive in all dimensions".to_string(),
                },
            ));
        }
        if config.time_steps == 0 {
            return Err(KwaversError::Validation(
                crate::core::error::ValidationError::ConstraintViolation {
                    message: "time_steps must be positive".to_string(),
                },
            ));
        }
        // CFL stability: Δt ≤ Δx_min / (c√3) × courant_factor
        let c = 1.0 / (config.permittivity * config.permeability).sqrt();
        let dx_min = config
            .spatial_steps
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));
        let cfl_limit = dx_min / (c * 3.0_f64.sqrt());
        if config.time_step > cfl_limit * config.courant_factor {
            return Err(KwaversError::Validation(
                crate::core::error::ValidationError::ConstraintViolation {
                    message: format!(
                        "time_step {:.2e}s exceeds CFL stability limit {:.2e}s",
                        config.time_step,
                        cfl_limit * config.courant_factor
                    ),
                },
            ));
        }
        Ok(())
    }

    /// Initialize electromagnetic fields from optional initial conditions.
    pub fn initialize_fields(
        &mut self,
        initial_conditions: Option<&EMFieldData>,
    ) -> KwaversResult<()> {
        let [nx, ny, nz] = self.config.grid_size;
        let time_points: Vec<f64> = (0..self.config.time_steps)
            .map(|i| i as f64 * self.config.time_step)
            .collect();
        let coordinates = [
            (0..nx)
                .map(|i| i as f64 * self.config.spatial_steps[0])
                .collect(),
            (0..ny)
                .map(|j| j as f64 * self.config.spatial_steps[1])
                .collect(),
            (0..nz)
                .map(|k| k as f64 * self.config.spatial_steps[2])
                .collect(),
        ];
        let field_data = if let Some(init) = initial_conditions {
            init.clone()
        } else {
            EMFieldData {
                electric_field: Array5::zeros((self.config.time_steps, nx, ny, nz, 3)),
                magnetic_field: Array5::zeros((self.config.time_steps, nx, ny, nz, 3)),
                current_density: Array5::zeros((self.config.time_steps, nx, ny, nz, 3)),
                charge_density: Array4::zeros((self.config.time_steps, nx, ny, nz)),
                time_points,
                coordinates,
            }
        };
        self.create_gpu_buffers(&field_data)?;
        self.field_data = Some(field_data);
        Ok(())
    }

    /// Allocate GPU storage buffers and upload initial field data.
    fn create_gpu_buffers(&mut self, field_data: &EMFieldData) -> KwaversResult<()> {
        if !self.compute_manager.has_gpu() {
            return Ok(());
        }
        let electric_field = field_data.electric_field.as_slice().ok_or_else(|| {
            KwaversError::GpuError("Electric field storage not contiguous".into())
        })?;
        let magnetic_field = field_data.magnetic_field.as_slice().ok_or_else(|| {
            KwaversError::GpuError("Magnetic field storage not contiguous".into())
        })?;
        let current_density = field_data.current_density.as_slice().ok_or_else(|| {
            KwaversError::GpuError("Current density storage not contiguous".into())
        })?;

        let electric_buffer = self.compute_manager.create_buffer(
            field_data.electric_field.len() * std::mem::size_of::<f64>(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        )?;
        let magnetic_buffer = self.compute_manager.create_buffer(
            field_data.magnetic_field.len() * std::mem::size_of::<f64>(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        )?;
        let current_density_buffer = self.compute_manager.create_buffer(
            field_data.current_density.len() * std::mem::size_of::<f64>(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        )?;

        self.compute_manager
            .write_buffer(&electric_buffer, electric_field)?;
        self.compute_manager
            .write_buffer(&magnetic_buffer, magnetic_field)?;
        self.compute_manager
            .write_buffer(&current_density_buffer, current_density)?;

        self.gpu_buffers
            .insert("electric".to_string(), electric_buffer);
        self.gpu_buffers
            .insert("magnetic".to_string(), magnetic_buffer);
        self.gpu_buffers
            .insert("current_density".to_string(), current_density_buffer);

        self.ensure_pipeline_resources()?;
        self.create_bind_group()?;
        Ok(())
    }

    /// Build bind group layout and compute pipeline (idempotent).
    fn ensure_pipeline_resources(&mut self) -> KwaversResult<()> {
        if self.compute_pipeline.is_some() && self.bind_group_layout.is_some() {
            return Ok(());
        }
        let device = self.compute_manager.device()?;

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("EM Time Step Layout"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("EM Time Step Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../../../../gpu/shaders/electromagnetic.wgsl"
            ))),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("EM Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("EM Time Step Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        self.bind_group_layout = Some(bind_group_layout);
        self.compute_pipeline = Some(compute_pipeline);
        Ok(())
    }

    /// Create the wgpu bind group from current GPU buffers and layout.
    fn create_bind_group(&mut self) -> KwaversResult<()> {
        let electric_buffer = self
            .gpu_buffers
            .get("electric")
            .ok_or_else(|| KwaversError::GpuError("Missing electric GPU buffer".into()))?;
        let magnetic_buffer = self
            .gpu_buffers
            .get("magnetic")
            .ok_or_else(|| KwaversError::GpuError("Missing magnetic GPU buffer".into()))?;
        let current_density_buffer = self
            .gpu_buffers
            .get("current_density")
            .ok_or_else(|| KwaversError::GpuError("Missing current_density GPU buffer".into()))?;
        let device = self.compute_manager.device()?;
        let bind_group_layout = self
            .bind_group_layout
            .as_ref()
            .ok_or_else(|| KwaversError::GpuError("Bind group layout not initialized".into()))?;

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("EM Time Step Bind Group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: electric_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: magnetic_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: current_density_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: electric_buffer.as_entire_binding(),
                },
            ],
        });
        self.bind_group = Some(bind_group);
        Ok(())
    }
}
