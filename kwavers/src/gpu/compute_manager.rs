//! GPU compute manager with automatic CPU fallback
//!
//! Provides high-level interface for GPU compute operations

use crate::core::constants::numerical;
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;

/// GPU compute manager with automatic CPU fallback dispatch.
///
/// Holds optional `device`/`queue` handles acquired at construction time.
/// GPU dispatch kernels (FDTD, k-space, absorption, nonlinear) are deferred to
/// a future sprint; all current paths use the CPU implementations.
#[derive(Debug)]
pub struct ComputeManager {
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
}

impl ComputeManager {
    /// Create new compute manager with automatic GPU detection.
    ///
    /// Falls back to CPU-only mode when no GPU adapter is available.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
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

    /// Create new compute manager (blocking)
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new_blocking() -> KwaversResult<Self> {
        pollster::block_on(Self::new())
    }

    /// Get device reference (error if GPU unavailable)
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

    /// Get queue reference (error if GPU unavailable)
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

    /// Create a GPU buffer (error if GPU unavailable)
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

    /// Write typed data into a GPU buffer (error if GPU unavailable)
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

    /// Initialize GPU if available
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    async fn init_gpu() -> KwaversResult<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
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
                label: Some("Kwavers Compute Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| KwaversError::GpuError(format!("Failed to create device: {}", e)))?;

        Ok((device, queue))
    }

    /// Check if GPU is available
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn has_gpu(&self) -> bool {
        self.device.is_some()
    }

    /// Update FDTD pressure field
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn fdtd_update(
        &self,
        pressure: &mut Array3<f64>,
        velocity_x: &Array3<f64>,
        velocity_y: &Array3<f64>,
        velocity_z: &Array3<f64>,
        dx: f64,
        dy: f64,
        dz: f64,
        dt: f64,
        c0: f64,
        rho0: f64,
    ) -> KwaversResult<()> {
        // Validate CFL condition
        let cfl = dt * c0 * ((1.0 / dx).powi(2) + (1.0 / dy).powi(2) + (1.0 / dz).powi(2)).sqrt();
        if cfl > numerical::CFL_MAX {
            return Err(KwaversError::InvalidInput(format!(
                "CFL number {} exceeds maximum {}",
                cfl,
                numerical::CFL_MAX
            )));
        }

        self.fdtd_cpu(
            pressure, velocity_x, velocity_y, velocity_z, dx, dy, dz, dt, c0, rho0,
        )
    }

    /// CPU implementation of FDTD using SIMD
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn fdtd_cpu(
        &self,
        pressure: &mut Array3<f64>,
        velocity_x: &Array3<f64>,
        velocity_y: &Array3<f64>,
        velocity_z: &Array3<f64>,
        dx: f64,
        dy: f64,
        dz: f64,
        dt: f64,
        c0: f64,
        rho0: f64,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = pressure.dim();
        let bulk_modulus = rho0 * c0 * c0;
        let pressure_prev = pressure.clone();

        // Use SIMD for inner loop where possible
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    // Compute velocity divergence
                    let dvx_dx = (velocity_x[[i + 1, j, k]] - velocity_x[[i, j, k]]) / dx;
                    let dvy_dy = (velocity_y[[i, j + 1, k]] - velocity_y[[i, j, k]]) / dy;
                    let dvz_dz = (velocity_z[[i, j, k + 1]] - velocity_z[[i, j, k]]) / dz;

                    let divergence = dvx_dx + dvy_dy + dvz_dz;

                    // Update pressure
                    pressure[[i, j, k]] = pressure_prev[[i, j, k]] - bulk_modulus * dt * divergence;
                }
            }
        }

        Ok(())
    }

    /// Apply absorption to pressure field
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn apply_absorption(
        &self,
        pressure: &mut Array3<f64>,
        absorption: &Array3<f64>,
        dt: f64,
    ) -> KwaversResult<()> {
        self.absorption_cpu(pressure, absorption, dt)
    }

    /// CPU implementation of absorption
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn absorption_cpu(
        &self,
        pressure: &mut Array3<f64>,
        absorption: &Array3<f64>,
        dt: f64,
    ) -> KwaversResult<()> {
        // Use SIMD for element-wise operations
        let decay = absorption.mapv(|a| (-a * dt).exp());
        // Apply decay using element-wise multiplication instead of scale_inplace
        pressure.zip_mut_with(&decay, |p, &d| *p *= d);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_absorption_cpu_decay_is_applied() {
        let manager =
            ComputeManager::new_blocking().expect("ComputeManager::new_blocking should succeed");
        let dt: f64 = 1e-3;

        let mut pressure = Array3::from_elem((2, 2, 2), 1.0);
        let absorption = Array3::from_elem((2, 2, 2), 2.0);
        let expected = (-2.0_f64 * dt).exp();

        manager
            .absorption_cpu(&mut pressure, &absorption, dt)
            .expect("absorption_cpu should succeed");

        for &p in pressure.iter() {
            assert!((p - expected).abs() <= 1e-12);
        }
    }
}
