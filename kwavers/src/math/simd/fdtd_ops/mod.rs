//! SIMD-accelerated FDTD pressure / velocity update kernels.

use super::config::{MathSimdLevel, SimdConfig};

mod pressure;
mod velocity;

/// SIMD-accelerated FDTD operations
#[derive(Debug)]
pub struct FdtdSimdOps {
    pub(super) config: SimdConfig,
}

impl FdtdSimdOps {
    /// Create new FDTD SIMD operations
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: SimdConfig::detect(),
        }
    }

    /// SIMD-accelerated pressure update (3D FDTD)
    ///
    /// Updates pressure field using: p^{n+1} = 2p^n - p^{n-1} + c²Δt²∇²p
    #[allow(clippy::too_many_arguments)]
    pub fn update_pressure_3d(
        &self,
        pressure: &mut [f32],
        pressure_prev: &[f32],
        laplacian: &[f32],
        c_squared_dt_squared: f32,
        nx: usize,
        ny: usize,
        nz: usize,
    ) {
        match self.config.level {
            #[cfg(target_arch = "x86_64")]
            #[allow(unsafe_code)]
            MathSimdLevel::Avx2 => unsafe {
                self.update_pressure_avx2(
                    pressure,
                    pressure_prev,
                    laplacian,
                    c_squared_dt_squared,
                    nx,
                    ny,
                    nz,
                );
            },
            #[cfg(target_arch = "x86_64")]
            #[allow(unsafe_code)]
            MathSimdLevel::Avx512 => unsafe {
                self.update_pressure_avx512(
                    pressure,
                    pressure_prev,
                    laplacian,
                    c_squared_dt_squared,
                    nx,
                    ny,
                    nz,
                );
            },
            _ => self.update_pressure_scalar(
                pressure,
                pressure_prev,
                laplacian,
                c_squared_dt_squared,
                nx,
                ny,
                nz,
            ),
        }
    }

    /// SIMD-accelerated velocity update (3D FDTD)
    #[allow(clippy::too_many_arguments)]
    pub fn update_velocity_3d(
        &self,
        velocity: &mut [f32],
        velocity_prev: &[f32],
        pressure_gradient: &[f32],
        dt_over_rho: f32,
        nx: usize,
        ny: usize,
        nz: usize,
    ) {
        match self.config.level {
            #[cfg(target_arch = "x86_64")]
            MathSimdLevel::Avx2 => {
                #[allow(unsafe_code)]
                unsafe {
                    self.update_velocity_avx2(
                        velocity,
                        velocity_prev,
                        pressure_gradient,
                        dt_over_rho,
                        nx,
                        ny,
                        nz,
                    );
                }
            }
            _ => self.update_velocity_scalar(
                velocity,
                velocity_prev,
                pressure_gradient,
                dt_over_rho,
                nx,
                ny,
                nz,
            ),
        }
    }
}

impl Default for FdtdSimdOps {
    fn default() -> Self {
        Self::new()
    }
}
