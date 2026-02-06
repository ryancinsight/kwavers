//! GPU Compute Pipeline Management
//!
//! Manages compute shader compilation, pipeline creation, and execution.

use super::buffers::BufferManager;
use super::init::WGPUContext;
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;
use std::collections::HashMap;
use wgpu;

/// Pipeline manager for compute shader execution
#[derive(Debug)]
pub struct PipelineManager {
    /// Compiled compute pipelines
    pipelines: HashMap<PipelineType, wgpu::ComputePipeline>,

    /// Bind group layouts
    layouts: HashMap<PipelineType, wgpu::BindGroupLayout>,
}

/// Types of compute pipelines
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineType {
    /// FFT 3D (forward)
    FFT3D,
    /// Inverse FFT 3D
    IFFT3D,
    /// Element-wise multiply
    ElementWiseMultiply,
    /// Spatial derivative (k-space operator)
    SpatialDerivative,
}

impl PipelineManager {
    /// Create a new pipeline manager and compile all shaders
    pub fn new(device: &wgpu::Device) -> KwaversResult<Self> {
        let mut pipelines = HashMap::new();
        let mut layouts = HashMap::new();

        // Compile all pipelines
        Self::compile_fft_pipeline(device, &mut pipelines, &mut layouts)?;
        Self::compile_elementwise_pipeline(device, &mut pipelines, &mut layouts)?;
        Self::compile_derivative_pipeline(device, &mut pipelines, &mut layouts)?;

        Ok(Self { pipelines, layouts })
    }

    /// Compile FFT pipeline
    fn compile_fft_pipeline(
        device: &wgpu::Device,
        pipelines: &mut HashMap<PipelineType, wgpu::ComputePipeline>,
        layouts: &mut HashMap<PipelineType, wgpu::BindGroupLayout>,
    ) -> KwaversResult<()> {
        // For now, use a simple pass-through shader as placeholder
        // Full FFT implementation will be added in shader files
        let shader_source = include_str!("shaders/fft.wgsl");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fft-shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create bind group layout
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fft-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fft-pipeline-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fft-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "fft_main",
            compilation_options: Default::default(),
            cache: None,
        });

        layouts.insert(PipelineType::FFT3D, layout);
        layouts.insert(PipelineType::IFFT3D, layout.clone()); // Same layout for IFFT
        pipelines.insert(PipelineType::FFT3D, pipeline);

        Ok(())
    }

    /// Compile element-wise operation pipeline
    fn compile_elementwise_pipeline(
        device: &wgpu::Device,
        pipelines: &mut HashMap<PipelineType, wgpu::ComputePipeline>,
        layouts: &mut HashMap<PipelineType, wgpu::BindGroupLayout>,
    ) -> KwaversResult<()> {
        let shader_source = include_str!("shaders/operators.wgsl");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("operators-shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Layout: 3 buffers (input A, input B, output)
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("elementwise-layout"),
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("elementwise-pipeline-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("elementwise-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "elementwise_multiply",
            compilation_options: Default::default(),
            cache: None,
        });

        layouts.insert(PipelineType::ElementWiseMultiply, layout);
        pipelines.insert(PipelineType::ElementWiseMultiply, pipeline);

        Ok(())
    }

    /// Compile spatial derivative pipeline
    fn compile_derivative_pipeline(
        device: &wgpu::Device,
        pipelines: &mut HashMap<PipelineType, wgpu::ComputePipeline>,
        layouts: &mut HashMap<PipelineType, wgpu::BindGroupLayout>,
    ) -> KwaversResult<()> {
        let shader_source = include_str!("shaders/operators.wgsl");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("derivative-shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Layout: 2 buffers (input, output)
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("derivative-layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("derivative-pipeline-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("derivative-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "spatial_derivative",
            compilation_options: Default::default(),
            cache: None,
        });

        layouts.insert(PipelineType::SpatialDerivative, layout);
        pipelines.insert(PipelineType::SpatialDerivative, pipeline);

        Ok(())
    }

    /// Execute FFT 3D on GPU
    pub fn execute_fft_3d(
        &self,
        data: &mut Array3<f64>,
        context: &WGPUContext,
        buffer_manager: &BufferManager,
    ) -> KwaversResult<()> {
        // Get pipeline
        let pipeline = self.pipelines.get(&PipelineType::FFT3D).ok_or_else(|| {
            KwaversError::ConfigError(crate::core::error::ConfigError::MissingFeature {
                feature: "FFT3D pipeline".to_string(),
                help: "Pipeline not compiled".to_string(),
            })
        })?;

        // Create buffer from data
        let shape = data.shape();
        let n_elements = shape[0] * shape[1] * shape[2];
        let buffer_size = n_elements * std::mem::size_of::<f32>();

        let buffer = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft-buffer"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Write data to buffer
        buffer_manager.write_array_to_buffer(context.queue(), &buffer, data)?;

        // Create bind group
        let layout = self.layouts.get(&PipelineType::FFT3D).unwrap();
        let bind_group = context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fft-bind-group"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            });

        // Execute compute shader
        let mut encoder =
            context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fft-encoder"),
                });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fft-pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups((n_elements as u32 + 255) / 256, 1, 1);
        }

        context.queue().submit(std::iter::once(encoder.finish()));

        // Read back result
        *data = buffer_manager.read_buffer_to_array_sync(
            context.device(),
            &buffer,
            (shape[0], shape[1], shape[2]),
        )?;

        Ok(())
    }

    /// Execute inverse FFT 3D.
    ///
    /// When the GPU FFT shader is fully implemented this should apply
    /// conjugate → FFT → conjugate → scale(1/N). Currently delegates to
    /// the forward FFT which is itself a pass-through placeholder.
    pub fn execute_ifft_3d(
        &self,
        data: &mut Array3<f64>,
        context: &WGPUContext,
        buffer_manager: &BufferManager,
    ) -> KwaversResult<()> {
        let n = data.len() as f64;
        // Conjugate (negate imaginary parts — real data is invariant)
        self.execute_fft_3d(data, context, buffer_manager)?;
        // Scale by 1/N for inverse normalisation
        if n > 0.0 {
            data.mapv_inplace(|v| v / n);
        }
        Ok(())
    }

    /// Execute element-wise multiply
    pub fn execute_element_wise_multiply(
        &self,
        a: &Array3<f64>,
        b: &Array3<f64>,
        out: &mut Array3<f64>,
        context: &WGPUContext,
        buffer_manager: &BufferManager,
    ) -> KwaversResult<()> {
        // Get pipeline
        let pipeline = self
            .pipelines
            .get(&PipelineType::ElementWiseMultiply)
            .ok_or_else(|| {
                KwaversError::ConfigError(crate::core::error::ConfigError::MissingFeature {
                    feature: "ElementWiseMultiply pipeline".to_string(),
                    help: "Pipeline not compiled".to_string(),
                })
            })?;

        // Create buffers
        let shape = a.shape();
        let n_elements = shape[0] * shape[1] * shape[2];
        let buffer_size = n_elements * std::mem::size_of::<f32>();

        let buffer_a = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("buffer-a"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let buffer_b = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("buffer-b"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let buffer_out = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("buffer-out"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Write input data
        buffer_manager.write_array_to_buffer(context.queue(), &buffer_a, a)?;
        buffer_manager.write_array_to_buffer(context.queue(), &buffer_b, b)?;

        // Create bind group
        let layout = self
            .layouts
            .get(&PipelineType::ElementWiseMultiply)
            .unwrap();
        let bind_group = context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("multiply-bind-group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer_a.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffer_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffer_out.as_entire_binding(),
                    },
                ],
            });

        // Execute
        let mut encoder =
            context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("multiply-encoder"),
                });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("multiply-pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups((n_elements as u32 + 255) / 256, 1, 1);
        }

        context.queue().submit(std::iter::once(encoder.finish()));

        // Read back
        *out = buffer_manager.read_buffer_to_array_sync(
            context.device(),
            &buffer_out,
            (shape[0], shape[1], shape[2]),
        )?;

        Ok(())
    }

    /// Compute spatial derivative using central finite differences.
    ///
    /// `direction`: 0 = x (axis-0), 1 = y (axis-1), 2 = z (axis-2).
    /// Uses second-order central differences in the interior and first-order
    /// one-sided differences at the boundaries.  When the GPU k-space
    /// derivative shader is ready this CPU path should be replaced.
    pub fn execute_spatial_derivative(
        &self,
        field: &Array3<f64>,
        direction: usize,
        out: &mut Array3<f64>,
        _context: &WGPUContext,
        _buffer_manager: &BufferManager,
    ) -> KwaversResult<()> {
        let _pipeline = self
            .pipelines
            .get(&PipelineType::SpatialDerivative)
            .ok_or_else(|| {
                KwaversError::ConfigError(crate::core::error::ConfigError::MissingFeature {
                    feature: "SpatialDerivative pipeline".to_string(),
                    help: "Pipeline not compiled".to_string(),
                })
            })?;

        let (nx, ny, nz) = field.dim();
        let mut result = Array3::<f64>::zeros((nx, ny, nz));

        // Assume unit grid spacing; caller should rescale by 1/dx
        match direction {
            0 => {
                // ∂f/∂x along axis-0
                for j in 0..ny {
                    for k in 0..nz {
                        if nx >= 2 {
                            result[[0, j, k]] = field[[1, j, k]] - field[[0, j, k]];
                        }
                        for i in 1..nx.saturating_sub(1) {
                            result[[i, j, k]] =
                                (field[[i + 1, j, k]] - field[[i - 1, j, k]]) * 0.5;
                        }
                        if nx >= 2 {
                            result[[nx - 1, j, k]] =
                                field[[nx - 1, j, k]] - field[[nx - 2, j, k]];
                        }
                    }
                }
            }
            1 => {
                // ∂f/∂y along axis-1
                for i in 0..nx {
                    for k in 0..nz {
                        if ny >= 2 {
                            result[[i, 0, k]] = field[[i, 1, k]] - field[[i, 0, k]];
                        }
                        for j in 1..ny.saturating_sub(1) {
                            result[[i, j, k]] =
                                (field[[i, j + 1, k]] - field[[i, j - 1, k]]) * 0.5;
                        }
                        if ny >= 2 {
                            result[[i, ny - 1, k]] =
                                field[[i, ny - 1, k]] - field[[i, ny - 2, k]];
                        }
                    }
                }
            }
            2 => {
                // ∂f/∂z along axis-2
                for i in 0..nx {
                    for j in 0..ny {
                        if nz >= 2 {
                            result[[i, j, 0]] = field[[i, j, 1]] - field[[i, j, 0]];
                        }
                        for k in 1..nz.saturating_sub(1) {
                            result[[i, j, k]] =
                                (field[[i, j, k + 1]] - field[[i, j, k - 1]]) * 0.5;
                        }
                        if nz >= 2 {
                            result[[i, j, nz - 1]] =
                                field[[i, j, nz - 1]] - field[[i, j, nz - 2]];
                        }
                    }
                }
            }
            _ => {
                return Err(KwaversError::InvalidInput(format!(
                    "spatial derivative direction must be 0, 1, or 2; got {}",
                    direction
                )));
            }
        }

        *out = result;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_type_enum() {
        assert_ne!(PipelineType::FFT3D, PipelineType::IFFT3D);
        assert_ne!(PipelineType::FFT3D, PipelineType::ElementWiseMultiply);
    }

    // Full pipeline tests require GPU device
    // Tested in integration tests
}
