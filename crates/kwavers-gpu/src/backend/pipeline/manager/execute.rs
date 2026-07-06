//! WGPU pipeline execution methods for WgpuPipelineManager.

use super::super::super::buffers::WgpuBackendBufferManager;
use super::super::super::init::GpuProviderContext;
use super::super::types::PipelineType;
use hephaestus_wgpu::WgpuDevice;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3 as LetoArray3;
use wgpu;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DerivativeParams {
    nx: u32,
    ny: u32,
    nz: u32,
    direction: u32,
}

fn validate_non_empty_shape(operation: &str, shape: [usize; 3]) -> KwaversResult<()> {
    if shape.contains(&0) {
        return Err(KwaversError::InvalidInput(format!(
            "{operation}: all dimensions must be non-zero; got {shape:?}"
        )));
    }

    Ok(())
}

fn validate_matching_shape(
    operation: &str,
    name: &str,
    expected: [usize; 3],
    actual: [usize; 3],
) -> KwaversResult<()> {
    if actual != expected {
        return Err(KwaversError::InvalidInput(format!(
            "{operation}: {name} shape {actual:?} must match input shape {expected:?}"
        )));
    }

    Ok(())
}

impl super::WgpuPipelineManager {
    /// Execute element-wise multiply.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn execute_element_wise_multiply(
        &self,
        a: &LetoArray3<f32>,
        b: &LetoArray3<f32>,
        out: &mut LetoArray3<f32>,
        context: &GpuProviderContext<WgpuDevice>,
        buffer_manager: &WgpuBackendBufferManager,
    ) -> KwaversResult<()> {
        let pipeline = self
            .pipelines
            .get(&PipelineType::ElementWiseMultiply)
            .ok_or_else(|| {
                KwaversError::GpuError(
                    "ElementWiseMultiply pipeline: Pipeline not compiled".to_string(),
                )
            })?;

        let shape = a.shape();
        validate_non_empty_shape("element_wise_multiply", shape)?;
        validate_matching_shape("element_wise_multiply", "rhs", shape, b.shape())?;
        validate_matching_shape("element_wise_multiply", "output", shape, out.shape())?;

        let n_elements = shape.iter().product::<usize>();
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

        buffer_manager.write_provider_array_to_buffer(context.queue(), &buffer_a, a)?;
        buffer_manager.write_provider_array_to_buffer(context.queue(), &buffer_b, b)?;

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
            compute_pass.dispatch_workgroups((n_elements as u32).div_ceil(256), 1, 1);
        }

        context.queue().submit(std::iter::once(encoder.finish()));

        *out = buffer_manager.read_buffer_to_provider_array_sync(
            context.device(),
            context.queue(),
            &buffer_out,
            shape,
        )?;

        Ok(())
    }

    /// Compute spatial derivative using central finite differences.
    ///
    /// `direction`: 0 = x (axis-0), 1 = y (axis-1), 2 = z (axis-2).
    /// Second-order central differences in the interior; first-order one-sided at boundaries.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn execute_spatial_derivative(
        &self,
        field: &LetoArray3<f32>,
        direction: usize,
        out: &mut LetoArray3<f32>,
        context: &GpuProviderContext<WgpuDevice>,
        buffer_manager: &WgpuBackendBufferManager,
    ) -> KwaversResult<()> {
        if direction > 2 {
            return Err(KwaversError::InvalidInput(format!(
                "spatial derivative direction must be 0, 1, or 2; got {}",
                direction
            )));
        }

        let pipeline = self
            .pipelines
            .get(&PipelineType::SpatialDerivative)
            .ok_or_else(|| {
                KwaversError::GpuError(
                    "SpatialDerivative pipeline: Pipeline not compiled".to_string(),
                )
            })?;

        let [nx, ny, nz] = field.shape();
        validate_non_empty_shape("spatial_derivative", [nx, ny, nz])?;
        validate_matching_shape("spatial_derivative", "output", [nx, ny, nz], out.shape())?;

        let n_elements = nx * ny * nz;
        let buffer_size = n_elements * std::mem::size_of::<f32>();

        let buffer_field = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("spatial-derivative-field"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let buffer_out = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("spatial-derivative-output"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = DerivativeParams {
            nx: nx as u32,
            ny: ny as u32,
            nz: nz as u32,
            direction: direction as u32,
        };
        let params_buffer = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("spatial-derivative-params"),
            size: std::mem::size_of::<DerivativeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        buffer_manager.write_provider_array_to_buffer(context.queue(), &buffer_field, field)?;
        context
            .queue()
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        let layout = self
            .layouts
            .get(&PipelineType::SpatialDerivative)
            .ok_or_else(|| {
                KwaversError::GpuError(
                    "SpatialDerivative layout: Bind group layout not compiled".to_string(),
                )
            })?;
        let bind_group = context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("spatial-derivative-bind-group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer_field.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffer_out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffer_out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder =
            context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("spatial-derivative-encoder"),
                });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("spatial-derivative-pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups((n_elements as u32).div_ceil(256), 1, 1);
        }

        context.queue().submit(std::iter::once(encoder.finish()));

        *out = buffer_manager.read_buffer_to_provider_array_sync(
            context.device(),
            context.queue(),
            &buffer_out,
            [nx, ny, nz],
        )?;

        Ok(())
    }
}
