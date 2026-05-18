//! AVX-512 Optimized FDTD Stencil Operations
//!
//! Advanced SIMD-accelerated stencil operations targeting AVX-512 instruction set,
//! achieving 8x vectorization for 64-bit floating point operations.
//!
//! ## Architecture
//!
//! ```text
//! math/simd.rs (SIMD capability detection)
//!    ↓
//! solver/forward/fdtd/
//!    ├── simd_stencil.rs (Generic SIMD stencil processor)
//!    ├── avx512_stencil/  (AVX-512 specialized implementation)
//!    └── dispatch.rs (Runtime dispatch to optimal implementation)
//! ```
//!
//! Sub-module responsibilities (SRP boundaries):
//! - `construction` — `new` constructor, coefficient precomputation
//! - `pressure`     — `update_pressure_avx512` + unsafe AVX-512 kernel
//! - `velocity`     — `update_velocity_avx512` + unsafe AVX-512 kernel

use crate::core::constants::{CFL_FACTOR_3D_FDTD, DENSITY_WATER_NOMINAL, SOUND_SPEED_TISSUE};
use crate::math::simd::{MathSimdLevel, SimdConfig};
use std::marker::PhantomData;

mod construction;
mod pressure;
#[cfg(test)]
mod tests;
mod velocity;

/// AVX-512 stencil processor configuration
#[derive(Debug, Clone, Copy)]
pub struct FdtdAvx512Config {
    pub tile_size: usize,
    pub use_fma: bool,
    pub prefetch_boundaries: bool,
    pub sound_speed: f64,
    pub density: f64,
    pub dx: f64,
    pub dt: f64,
}

impl Default for FdtdAvx512Config {
    fn default() -> Self {
        Self {
            tile_size: 8,
            use_fma: true,
            prefetch_boundaries: true,
            sound_speed: SOUND_SPEED_TISSUE,
            density: DENSITY_WATER_NOMINAL,
            dx: 0.001,
            dt: CFL_FACTOR_3D_FDTD * 0.001 / SOUND_SPEED_TISSUE,
        }
    }
}

/// AVX-512 optimized FDTD stencil processor.
///
/// Implements high-performance stencil operations using AVX-512 instructions.
/// Operates on 3D grids with 8-wide vectorization for f64 elements.
#[derive(Debug)]
pub struct FdtdAvx512StencilProcessor {
    pub(super) config: FdtdAvx512Config,
    pub(super) nx: usize,
    pub(super) ny: usize,
    pub(super) nz: usize,
    /// Precomputed pressure coefficient: -c²Δt²/Δx²
    pub(super) pressure_coeff: f64,
    /// Precomputed velocity coefficient: -Δt/(ρΔx)
    pub(super) velocity_coeff: f64,
    /// Central coefficient for pressure Laplacian: (2 - 6*pressure_coeff)
    pub(super) pressure_central_coeff: f64,
    pub(super) simd_config: SimdConfig,
    pub(super) _phantom: PhantomData<()>,
}

impl FdtdAvx512StencilProcessor {
    /// Get performance metrics from last update.
    #[must_use]
    pub fn get_metrics(&self) -> FdtdAvx512Metrics {
        FdtdAvx512Metrics {
            grid_size: (self.nx, self.ny, self.nz),
            simd_level: self.simd_config.level,
            vector_width: 8,
            alignment: 64,
        }
    }
}

/// Performance metrics for AVX-512 stencil processing.
#[derive(Debug, Clone)]
pub struct FdtdAvx512Metrics {
    pub grid_size: (usize, usize, usize),
    pub simd_level: MathSimdLevel,
    pub vector_width: usize,
    pub alignment: usize,
}
