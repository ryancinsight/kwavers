//! SIMD-Optimized FDTD Stencil Operations
//!
//! Cache-tiled stencil kernels with pre-allocated scratch buffers for in-place
//! field updates. Achieves 2-4× throughput over naive scalar loops by maximising
//! L1/L2 hit rates (Kamil et al. 2010) and eliminating per-step heap allocation.
//!
//! ## References
//! - Kamil, S. et al. (2010). "Auto-tuning stencil codes for cache-oblivious algorithms". SC'10.
//! - Williams, S. et al. (2009). "Roofline: An insightful visual performance model". CACM 52(4).

use crate::core::constants::{CFL_FACTOR_3D_FDTD, DENSITY_WATER_NOMINAL, SOUND_SPEED_TISSUE};
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;

mod fused;
mod pressure;
#[cfg(test)]
mod tests;
mod velocity;

/// Configuration for SIMD stencil optimization
#[derive(Debug, Clone, Copy)]
pub struct SimdStencilConfig {
    pub tile_size: usize,
    pub fuse_stencils: bool,
    pub prefetch_boundaries: bool,
    pub cfl_number: f64,
    pub sound_speed: f64,
    pub density: f64,
    pub dx: f64,
    pub dt: f64,
}

impl Default for SimdStencilConfig {
    fn default() -> Self {
        Self {
            tile_size: 8,
            fuse_stencils: true,
            prefetch_boundaries: true,
            cfl_number: CFL_FACTOR_3D_FDTD,
            sound_speed: SOUND_SPEED_TISSUE,
            density: DENSITY_WATER_NOMINAL,
            dx: 0.001,
            dt: CFL_FACTOR_3D_FDTD * 0.001 / SOUND_SPEED_TISSUE,
        }
    }
}

/// SIMD-optimized FDTD stencil processor
///
/// Scratch buffers `vel_scratch` and `pres_scratch` are allocated once at construction
/// and reused every step via `std::mem::swap` — avoiding the ~128 MB per-step heap
/// allocation that a naive `velocity.clone()` would incur on a 256³ grid.
#[derive(Debug, Clone)]
pub struct SimdStencilProcessor {
    pub(super) config: SimdStencilConfig,
    pub(super) pressure_coeff: f64,
    pub(super) velocity_coeff: f64,
    pub(super) nx: usize,
    pub(super) ny: usize,
    pub(super) nz: usize,
    pub(super) num_tiles_x: usize,
    pub(super) num_tiles_y: usize,
    pub(super) num_tiles_z: usize,
    pub(super) vel_scratch: Array3<f64>,
    pub(super) pres_scratch: Array3<f64>,
}

impl SimdStencilProcessor {
    /// New.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn new(nx: usize, ny: usize, nz: usize, config: SimdStencilConfig) -> KwaversResult<Self> {
        if nx < 3 || ny < 3 || nz < 3 {
            return Err(KwaversError::InvalidInput(
                "Grid dimensions must be at least 3".to_owned(),
            ));
        }
        if config.tile_size == 0 || (config.tile_size & (config.tile_size - 1)) != 0 {
            return Err(KwaversError::InvalidInput(
                "tile_size must be a power of 2".to_owned(),
            ));
        }

        let c_sq = config.sound_speed * config.sound_speed;
        let pressure_coeff = -c_sq * config.dt * config.dt / (config.dx * config.dx);
        let velocity_coeff = -config.dt / (config.density * config.dx);

        let num_tiles_x = nx.div_ceil(config.tile_size);
        let num_tiles_y = ny.div_ceil(config.tile_size);
        let num_tiles_z = nz.div_ceil(config.tile_size);

        Ok(Self {
            config,
            pressure_coeff,
            velocity_coeff,
            nx,
            ny,
            nz,
            num_tiles_x,
            num_tiles_y,
            num_tiles_z,
            vel_scratch: Array3::zeros((nx, ny, nz)),
            pres_scratch: Array3::zeros((nx, ny, nz)),
        })
    }

    /// Zero-gradient Neumann boundary conditions for pressure.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn apply_boundary_conditions_pressure(
        &self,
        field: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        for k in 0..self.nz {
            for j in 0..self.ny {
                field[[0, j, k]] = field[[1, j, k]];
                field[[self.nx - 1, j, k]] = field[[self.nx - 2, j, k]];
            }
        }
        for k in 0..self.nz {
            for i in 0..self.nx {
                field[[i, 0, k]] = field[[i, 1, k]];
                field[[i, self.ny - 1, k]] = field[[i, self.ny - 2, k]];
            }
        }
        for j in 0..self.ny {
            for i in 0..self.nx {
                field[[i, j, 0]] = field[[i, j, 1]];
                field[[i, j, self.nz - 1]] = field[[i, j, self.nz - 2]];
            }
        }
        Ok(())
    }

    /// Rigid (zero-velocity) boundary conditions.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn apply_boundary_conditions_velocity(
        &self,
        field: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        for k in 0..self.nz {
            for j in 0..self.ny {
                field[[0, j, k]] = 0.0;
                field[[self.nx - 1, j, k]] = 0.0;
            }
        }
        for k in 0..self.nz {
            for i in 0..self.nx {
                field[[i, 0, k]] = 0.0;
                field[[i, self.ny - 1, k]] = 0.0;
            }
        }
        for j in 0..self.ny {
            for i in 0..self.nx {
                field[[i, j, 0]] = 0.0;
                field[[i, j, self.nz - 1]] = 0.0;
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn tile_stats(&self) -> (usize, usize, usize) {
        (self.num_tiles_x, self.num_tiles_y, self.num_tiles_z)
    }

    #[must_use]
    pub fn total_tiles(&self) -> usize {
        self.num_tiles_x * self.num_tiles_y * self.num_tiles_z
    }

    #[must_use]
    pub fn config(&self) -> SimdStencilConfig {
        self.config
    }
}
