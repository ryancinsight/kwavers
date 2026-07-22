//! Provider-generic GPU compute manager.
//!
//! Provides high-level access to an optional Hephaestus GPU provider while
//! keeping raw WGPU handles confined to the WGPU specialization.

use crate::gpu::device::{GpuDevice, GpuDeviceProvider};
use hephaestus_wgpu::WgpuDevice;
use kwavers_core::constants::numerical;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3 as LetoArray3;

/// GPU compute manager for a concrete Hephaestus provider.
///
/// Holds an optional acquired provider. GPU dispatch kernels (FDTD, k-space,
/// absorption, nonlinear) are still represented by provider-specific kernel
/// contracts elsewhere; current field-update helpers below are CPU routines.
#[derive(Debug)]
pub struct ComputeManager<P = WgpuDevice>
where
    P: GpuDeviceProvider,
{
    provider: Option<GpuDevice<P>>,
}

impl<P> ComputeManager<P>
where
    P: GpuDeviceProvider,
{
    /// Acquire a compute manager for the selected GPU provider.
    ///
    /// # Errors
    ///
    /// Returns an error when the selected provider cannot be acquired with its
    /// declared Kwavers requirements.
    ///
    pub async fn new() -> KwaversResult<Self> {
        let provider = GpuDevice::<P>::create_with_features_and_limits(
            P::acquisition_preference(),
            P::optional_features(),
            P::required_limits(),
        )
        .await?;

        Ok(Self {
            provider: Some(provider),
        })
    }

    /// Create a CPU-only compute manager without acquiring a GPU provider.
    #[must_use]
    pub const fn cpu_only() -> Self {
        Self { provider: None }
    }

    /// Create a provider-backed compute manager (blocking).
    ///
    /// # Errors
    ///
    /// Returns an error when the selected provider cannot be acquired with its
    /// declared Kwavers requirements.
    ///
    pub fn new_blocking() -> KwaversResult<Self> {
        let provider = GpuDevice::<P>::try_create_with_features_and_limits(
            P::acquisition_preference(),
            P::optional_features(),
            P::required_limits(),
        )?;

        Ok(Self {
            provider: Some(provider),
        })
    }

    /// Borrow the acquired provider device wrapper.
    #[must_use]
    pub const fn provider(&self) -> Option<&GpuDevice<P>> {
        self.provider.as_ref()
    }

    /// Check if the selected GPU provider is available.
    #[must_use]
    pub fn has_gpu(&self) -> bool {
        self.provider.is_some()
    }

    /// Update FDTD pressure field
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
    ///
    // Args are independent field arrays and scalar grid/medium/step parameters with no cohesive grouping.
    #[allow(clippy::too_many_arguments)]
    pub fn fdtd_update(
        &self,
        pressure: &mut LetoArray3<f64>,
        velocity_x: &LetoArray3<f64>,
        velocity_y: &LetoArray3<f64>,
        velocity_z: &LetoArray3<f64>,
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
    // Args are independent field arrays and scalar grid/medium/step parameters with no cohesive grouping.
    #[allow(clippy::too_many_arguments)]
    fn fdtd_cpu(
        &self,
        pressure: &mut LetoArray3<f64>,
        velocity_x: &LetoArray3<f64>,
        velocity_y: &LetoArray3<f64>,
        velocity_z: &LetoArray3<f64>,
        dx: f64,
        dy: f64,
        dz: f64,
        dt: f64,
        c0: f64,
        rho0: f64,
    ) -> KwaversResult<()> {
        let [nx, ny, nz] = pressure.shape();
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
        pressure: &mut LetoArray3<f64>,
        absorption: &LetoArray3<f64>,
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
        pressure: &mut LetoArray3<f64>,
        absorption: &LetoArray3<f64>,
        dt: f64,
    ) -> KwaversResult<()> {
        if pressure.shape() != absorption.shape() {
            return Err(KwaversError::InvalidInput(format!(
                "Pressure shape {:?} must match absorption shape {:?}",
                pressure.shape(),
                absorption.shape()
            )));
        }

        let pressure_values = pressure.as_slice_mut().ok_or_else(|| {
            KwaversError::InvalidInput(
                "ComputeManager pressure field must be dense row-major Leto Array3".to_string(),
            )
        })?;
        let absorption_values = absorption.as_slice().ok_or_else(|| {
            KwaversError::InvalidInput(
                "ComputeManager absorption field must be dense row-major Leto Array3".to_string(),
            )
        })?;

        for (pressure_value, absorption_value) in
            pressure_values.iter_mut().zip(absorption_values.iter())
        {
            *pressure_value *= (-*absorption_value * dt).exp();
        }

        Ok(())
    }
}

impl ComputeManager<WgpuDevice> {
    /// Get device reference (error if GPU unavailable)
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn device(&self) -> KwaversResult<&wgpu::Device> {
        self.provider
            .as_ref()
            .map(GpuDevice::wgpu_device)
            .ok_or_else(|| {
                KwaversError::System(kwavers_core::error::SystemError::ResourceUnavailable {
                    resource: "GPU device".to_string(),
                })
            })
    }

    /// Get queue reference (error if GPU unavailable)
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn queue(&self) -> KwaversResult<&wgpu::Queue> {
        self.provider
            .as_ref()
            .map(GpuDevice::wgpu_queue)
            .ok_or_else(|| {
                KwaversError::System(kwavers_core::error::SystemError::ResourceUnavailable {
                    resource: "GPU queue".to_string(),
                })
            })
    }

    /// Create a GPU buffer (error if GPU unavailable)
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
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
    /// - Propagates any `KwaversError` returned by called functions.
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_absorption_cpu_decay_is_applied() {
        let manager = ComputeManager::<WgpuDevice>::cpu_only();
        let dt: f64 = 1e-3;

        let mut pressure = LetoArray3::from_elem([2, 2, 2], 1.0);
        let absorption = LetoArray3::from_elem([2, 2, 2], 2.0);
        let expected = (-2.0_f64 * dt).exp();

        manager
            .absorption_cpu(&mut pressure, &absorption, dt)
            .expect("absorption_cpu should succeed");

        for &p in pressure.iter() {
            assert!((p - expected).abs() <= 1e-12);
        }
    }

    #[test]
    fn absorption_cpu_rejects_shape_mismatch() {
        let manager = ComputeManager::<WgpuDevice>::cpu_only();
        let mut pressure = LetoArray3::from_elem([2, 2, 2], 1.0);
        let absorption = LetoArray3::from_elem([2, 2, 1], 2.0);

        let error = manager
            .absorption_cpu(&mut pressure, &absorption, 1e-3)
            .expect_err("shape mismatch must be rejected");

        match error {
            KwaversError::InvalidInput(message) => {
                assert!(message.contains("[2, 2, 2]"));
                assert!(message.contains("[2, 2, 1]"));
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn compute_manager_cpu_only_has_no_provider() {
        let manager = ComputeManager::<WgpuDevice>::cpu_only();
        assert!(!manager.has_gpu());
        assert!(manager.provider().is_none());
    }

    #[cfg(feature = "cuda-provider")]
    #[test]
    fn compute_manager_accepts_non_default_provider() {
        fn assert_manager<P: GpuDeviceProvider>() {
            assert!(std::mem::size_of::<ComputeManager<P>>() > 0);
        }

        assert_manager::<hephaestus_cuda::CudaDevice>();
    }

    #[cfg(feature = "cuda-provider")]
    #[test]
    fn compute_manager_blocking_constructor_is_provider_generic() {
        fn assert_constructor<P: GpuDeviceProvider>() {
            let _constructor: fn() -> KwaversResult<ComputeManager<P>> =
                ComputeManager::<P>::new_blocking;
        }

        assert_constructor::<WgpuDevice>();
        assert_constructor::<hephaestus_cuda::CudaDevice>();
    }
}
