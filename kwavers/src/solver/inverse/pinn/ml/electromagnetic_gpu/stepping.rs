//! Time-stepping and source injection for `GPUEMSolver`.
//!
//! SRP: changes when the FDTD update equations, GPU dispatch strategy, or
//! staging-buffer download protocol changes.

use super::fields::EMFieldData;
use super::solver::GPUEMSolver;
use crate::core::error::{KwaversError, KwaversResult};

impl GPUEMSolver {
    /// Add a current source at `position` with a per-time-step `time_profile`.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn add_current_source(
        &mut self,
        position: [usize; 3],
        current: [f64; 3],
        time_profile: &[f64],
    ) -> KwaversResult<()> {
        let field_data = self.field_data.as_mut().ok_or_else(|| {
            KwaversError::System(crate::core::error::SystemError::InvalidOperation {
                operation: "add_current_source".to_string(),
                reason: "Fields not initialized".to_string(),
            })
        })?;

        let [i, j, k] = position;
        for (t, &amplitude) in time_profile.iter().enumerate() {
            if t < field_data.current_density.shape()[0] {
                field_data.current_density[[t, i, j, k, 0]] += current[0] * amplitude;
                field_data.current_density[[t, i, j, k, 1]] += current[1] * amplitude;
                field_data.current_density[[t, i, j, k, 2]] += current[2] * amplitude;
            }
        }

        // Re-upload modified current_density to GPU if a buffer exists.
        if let Some(current_density_buffer) = self.gpu_buffers.get("current_density") {
            let field_data = self.field_data.as_ref().ok_or_else(|| {
                KwaversError::System(crate::core::error::SystemError::InvalidOperation {
                    operation: "add_current_source".to_string(),
                    reason: "Fields not initialized".to_string(),
                })
            })?;
            let current_density = field_data.current_density.as_slice().ok_or_else(|| {
                KwaversError::GpuError("Current density storage not contiguous".into())
            })?;
            self.compute_manager
                .write_buffer(current_density_buffer, current_density)?;
        }
        Ok(())
    }

    /// Run the full electromagnetic simulation and return field data.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn solve(&mut self) -> KwaversResult<&EMFieldData> {
        if self.field_data.is_none() {
            self.initialize_fields(None)?;
        }
        for step in 1..self.config.time_steps {
            self.time_step(step)?;
        }
        self.download_results()?;
        self.field_data.as_ref().ok_or_else(|| {
            KwaversError::System(crate::core::error::SystemError::InvalidOperation {
                operation: "solve".to_string(),
                reason: "Fields not initialized".to_string(),
            })
        })
    }

    fn time_step(&mut self, step: usize) -> KwaversResult<()> {
        if self.compute_manager.has_gpu() && self.gpu_buffers.contains_key("electric") {
            self.time_step_gpu(step)
        } else {
            self.time_step_cpu(step)
        }
    }

    /// GPU FDTD time step — dispatches one compute pass per step.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn time_step_gpu(&mut self, _step: usize) -> KwaversResult<()> {
        let device = self.compute_manager.device()?;
        let queue = self.compute_manager.queue()?;
        let compute_pipeline = self
            .compute_pipeline
            .as_ref()
            .ok_or_else(|| KwaversError::GpuError("Compute pipeline not initialized".into()))?;
        let bind_group = self
            .bind_group
            .as_ref()
            .ok_or_else(|| KwaversError::GpuError("Bind group not initialized".into()))?;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("EM Time Step Encoder"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("EM Time Step Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(compute_pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            let [nx, ny, nz] = self.config.grid_size;
            compute_pass.dispatch_workgroups(
                (nx / 8).max(1) as u32,
                (ny / 8).max(1) as u32,
                (nz / 8).max(1) as u32,
            );
        }
        queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// CPU FDTD time step (development fallback — no-op pending full FDTD implementation).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn time_step_cpu(&mut self, _step: usize) -> KwaversResult<()> {
        Ok(())
    }

    /// Download electric and magnetic field results from GPU to CPU memory.
    /// # Errors
    /// - Returns [`KwaversError::GpuError`] if the precondition for a GpuError-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn download_results(&mut self) -> KwaversResult<()> {
        if let Some(field_data) = self.field_data.as_mut() {
            if let (Some(electric_buffer), Some(magnetic_buffer)) = (
                self.gpu_buffers.get("electric"),
                self.gpu_buffers.get("magnetic"),
            ) {
                let device = self.compute_manager.device()?;
                let queue = self.compute_manager.queue()?;

                let electric_staging = self.compute_manager.create_buffer(
                    field_data.electric_field.len() * std::mem::size_of::<f64>(),
                    wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                )?;
                let magnetic_staging = self.compute_manager.create_buffer(
                    field_data.magnetic_field.len() * std::mem::size_of::<f64>(),
                    wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                )?;

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("EM Download Encoder"),
                });
                encoder.copy_buffer_to_buffer(
                    electric_buffer,
                    0,
                    &electric_staging,
                    0,
                    (field_data.electric_field.len() * std::mem::size_of::<f64>()) as u64,
                );
                encoder.copy_buffer_to_buffer(
                    magnetic_buffer,
                    0,
                    &magnetic_staging,
                    0,
                    (field_data.magnetic_field.len() * std::mem::size_of::<f64>()) as u64,
                );
                queue.submit(Some(encoder.finish()));

                let electric_slice = electric_staging.slice(..);
                let magnetic_slice = magnetic_staging.slice(..);

                let (electric_tx, electric_rx) =
                    std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
                electric_slice.map_async(wgpu::MapMode::Read, move |res| {
                    let _ = electric_tx.send(res);
                });
                let (magnetic_tx, magnetic_rx) =
                    std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
                magnetic_slice.map_async(wgpu::MapMode::Read, move |res| {
                    let _ = magnetic_tx.send(res);
                });

                device.poll(wgpu::PollType::Wait);

                electric_rx
                    .recv()
                    .map_err(|e| KwaversError::GpuError(format!("Buffer mapping canceled: {e}")))?
                    .map_err(|e| {
                        KwaversError::GpuError(format!("Electric buffer mapping failed: {e:?}"))
                    })?;
                magnetic_rx
                    .recv()
                    .map_err(|e| KwaversError::GpuError(format!("Buffer mapping canceled: {e}")))?
                    .map_err(|e| {
                        KwaversError::GpuError(format!("Magnetic buffer mapping failed: {e:?}"))
                    })?;

                {
                    let electric_bytes = electric_slice.get_mapped_range();
                    let electric_values: &[f64] = bytemuck::cast_slice(&electric_bytes);
                    let dst = field_data.electric_field.as_slice_mut().ok_or_else(|| {
                        KwaversError::GpuError("Electric field storage not contiguous".into())
                    })?;
                    if dst.len() != electric_values.len() {
                        return Err(KwaversError::GpuError(
                            "Electric field download size mismatch".into(),
                        ));
                    }
                    dst.copy_from_slice(electric_values);
                }
                {
                    let magnetic_bytes = magnetic_slice.get_mapped_range();
                    let magnetic_values: &[f64] = bytemuck::cast_slice(&magnetic_bytes);
                    let dst = field_data.magnetic_field.as_slice_mut().ok_or_else(|| {
                        KwaversError::GpuError("Magnetic field storage not contiguous".into())
                    })?;
                    if dst.len() != magnetic_values.len() {
                        return Err(KwaversError::GpuError(
                            "Magnetic field download size mismatch".into(),
                        ));
                    }
                    dst.copy_from_slice(magnetic_values);
                }

                electric_staging.unmap();
                magnetic_staging.unmap();
            }
        }
        Ok(())
    }
}
