//! Pipeline execution methods for PipelineManager.

use super::super::super::buffers::GpuBackendBufferManager;
use super::super::super::init::WGPUContext;
use super::super::types::PipelineType;
use kwavers_core::error::{KwaversError, KwaversResult};
use ndarray::Array3;
use wgpu;

impl super::PipelineManager {
    /// Execute element-wise multiply.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn execute_element_wise_multiply(
        &self,
        a: &Array3<f64>,
        b: &Array3<f64>,
        out: &mut Array3<f64>,
        context: &WGPUContext,
        buffer_manager: &GpuBackendBufferManager,
    ) -> KwaversResult<()> {
        let pipeline = self
            .pipelines
            .get(&PipelineType::ElementWiseMultiply)
            .ok_or_else(|| {
                KwaversError::GpuError(format!("{}: {}", "ElementWiseMultiply pipeline".to_string(), "Pipeline not compiled".to_string()))
            })?;

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

        buffer_manager.write_array_to_buffer(context.queue(), &buffer_a, a)?;
        buffer_manager.write_array_to_buffer(context.queue(), &buffer_b, b)?;

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
            compute_pass.dispatch_workgroups((n_elements as u32 + 255) / 256, 1, 1);
        }

        context.queue().submit(std::iter::once(encoder.finish()));

        *out = buffer_manager.read_buffer_to_array_sync(
            context.device(),
            context.queue(),
            &buffer_out,
            (shape[0], shape[1], shape[2]),
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
        field: &Array3<f64>,
        direction: usize,
        out: &mut Array3<f64>,
        _context: &WGPUContext,
        _buffer_manager: &GpuBackendBufferManager,
    ) -> KwaversResult<()> {
        let _pipeline = self
            .pipelines
            .get(&PipelineType::SpatialDerivative)
            .ok_or_else(|| {
                KwaversError::GpuError(format!("{}: {}", "SpatialDerivative pipeline".to_string(), "Pipeline not compiled".to_string()))
            })?;

        let (nx, ny, nz) = field.dim();
        let mut result = Array3::<f64>::zeros((nx, ny, nz));

        match direction {
            0 => {
                for j in 0..ny {
                    for k in 0..nz {
                        if nx >= 2 {
                            result[[0, j, k]] = field[[1, j, k]] - field[[0, j, k]];
                        }
                        for i in 1..nx.saturating_sub(1) {
                            result[[i, j, k]] = (field[[i + 1, j, k]] - field[[i - 1, j, k]]) * 0.5;
                        }
                        if nx >= 2 {
                            result[[nx - 1, j, k]] = field[[nx - 1, j, k]] - field[[nx - 2, j, k]];
                        }
                    }
                }
            }
            1 => {
                for i in 0..nx {
                    for k in 0..nz {
                        if ny >= 2 {
                            result[[i, 0, k]] = field[[i, 1, k]] - field[[i, 0, k]];
                        }
                        for j in 1..ny.saturating_sub(1) {
                            result[[i, j, k]] = (field[[i, j + 1, k]] - field[[i, j - 1, k]]) * 0.5;
                        }
                        if ny >= 2 {
                            result[[i, ny - 1, k]] = field[[i, ny - 1, k]] - field[[i, ny - 2, k]];
                        }
                    }
                }
            }
            2 => {
                for i in 0..nx {
                    for j in 0..ny {
                        if nz >= 2 {
                            result[[i, j, 0]] = field[[i, j, 1]] - field[[i, j, 0]];
                        }
                        for k in 1..nz.saturating_sub(1) {
                            result[[i, j, k]] = (field[[i, j, k + 1]] - field[[i, j, k - 1]]) * 0.5;
                        }
                        if nz >= 2 {
                            result[[i, j, nz - 1]] = field[[i, j, nz - 1]] - field[[i, j, nz - 2]];
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
