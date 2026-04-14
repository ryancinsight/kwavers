//! SIMD-Optimized FDTD Stencil Operations
//!
//! This module implements high-performance stencil operations for FDTD solvers
//! using SIMD vectorization to achieve 2-4x performance improvement over scalar code.
//!
//! ## Key Optimizations
//!
//! ### 1. Vector Tile Processing
//! - Process 4×4×4 or 8×8×8 spatial tiles in vectorized fashion
//! - Maximize cache locality and reduce memory bandwidth
//! - Reduce instruction overhead from loop control
//!
//! ### 2. Stencil Fusion
//! - Combine multiple stencil operations (e.g., pressure and velocity updates)
//! - Reduce memory loads/stores through data reuse
//! - Improve arithmetic intensity (FLOPs per memory access)
//!
//! ### 3. Vectorized Arithmetic
//! - AVX2: 4 double-precision floating point operations in parallel
//! - AVX-512: 8 double-precision operations in parallel
//! - Automatic fallback to scalar for portability
//!
//! ### 4. Memory Layout Optimization
//! - Padding for alignment to SIMD vector boundaries
//! - Structure-of-arrays (SoA) layout for better vectorization
//! - Prefetch hints for non-temporal access patterns
//!
//! ## Physics Model
//!
//! **3D FDTD Pressure Update**:
//! ```text
//! p[i,j,k]^(n+1) = (2p[i,j,k]^n - p[i,j,k]^(n-1))
//!                   - c² Δt² ( ∂u/∂x + ∂v/∂y + ∂w/∂z )
//! ```
//!
//! **3D FDTD Velocity Updates**:
//! ```text
//! u[i,j,k]^(n+1) = u[i,j,k]^n - (Δt/ρ) ∂p/∂x
//! v[i,j,k]^(n+1) = v[i,j,k]^n - (Δt/ρ) ∂p/∂y
//! w[i,j,k]^(n+1) = w[i,j,k]^n - (Δt/ρ) ∂p/∂z
//! ```
//!
//! ## Performance Model
//!
//! **Scalar Baseline**: ~5 GFLOPS on single core (estimate for 3D FDTD)
//! **SIMD AVX2**: ~20 GFLOPS (4× speedup with fusion and tile processing)
//! **SIMD AVX-512**: ~40 GFLOPS (8× speedup on capable hardware)
//!
//! Actual speedup depends on:
//! - Hardware cache hierarchy (L1/L2/L3 hit rates)
//! - Memory bandwidth utilization
//! - Compiler vectorization quality
//! - Stencil pattern (3-point, 5-point, 7-point)
//!
//! ## References
//!
//! - Williams et al. (2009): "Roofline: An insightful visual performance model"
//! - Kamil et al. (2010): "Auto-tuning stencil codes for cache-oblivious algorithms"
//! - Gorelick & Gerber (2013): "The Software Optimization Cookbook"

use crate::core::constants::{CFL_FACTOR_3D_FDTD, DENSITY_WATER_NOMINAL, SOUND_SPEED_TISSUE};
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;
pub mod pressure;
pub mod velocity;

/// Configuration for SIMD stencil optimization
#[derive(Debug, Clone, Copy)]
pub struct SimdStencilConfig {
    /// Tile size for 3D processing (4, 8, or 16)
    pub tile_size: usize,

    /// Enable stencil fusion (combine multiple operations)
    pub fuse_stencils: bool,

    /// Enable boundary condition prefetching
    pub prefetch_boundaries: bool,

    /// CFL stability number (typical: 0.3 for FDTD)
    pub cfl_number: f64,

    /// Sound speed (m/s)
    pub sound_speed: f64,

    /// Density (kg/m³)
    pub density: f64,

    /// Grid spacing (m)
    pub dx: f64,

    /// Time step (s)
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
            // dt = CFL_FACTOR_3D_FDTD * dx / c = 0.3 * 0.001 / 1540 ≈ 1.95e-7 s
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
    /// Configuration
    pub(super) config: SimdStencilConfig,

    /// Precomputed coefficient for pressure update
    pub(super) pressure_coeff: f64,

    /// Precomputed coefficient for velocity update
    pub(super) velocity_coeff: f64,

    /// Grid dimensions
    pub(super) nx: usize,
    pub(super) ny: usize,
    pub(super) nz: usize,

    /// Number of tiles in each dimension
    pub(super) num_tiles_x: usize,
    pub(super) num_tiles_y: usize,
    pub(super) num_tiles_z: usize,

    /// Pre-allocated swap buffer for in-place velocity update (avoids per-step clone)
    pub(super) vel_scratch: Array3<f64>,

    /// Pre-allocated swap buffer for in-place pressure update
    pub(super) pres_scratch: Array3<f64>,
}

impl SimdStencilProcessor {
    /// Create new SIMD stencil processor
    pub fn new(nx: usize, ny: usize, nz: usize, config: SimdStencilConfig) -> KwaversResult<Self> {
        if nx < 3 || ny < 3 || nz < 3 {
            return Err(KwaversError::InvalidInput(
                "Grid dimensions must be at least 3".to_string(),
            ));
        }

        if config.tile_size == 0 || (config.tile_size & (config.tile_size - 1)) != 0 {
            return Err(KwaversError::InvalidInput(
                "tile_size must be a power of 2".to_string(),
            ));
        }

        // Precompute coefficients
        let c_sq = config.sound_speed * config.sound_speed;
        let pressure_coeff = -c_sq * config.dt * config.dt / (config.dx * config.dx);
        let velocity_coeff = -config.dt / (config.density * config.dx);

        let num_tiles_x = nx.div_ceil(config.tile_size);
        let num_tiles_y = ny.div_ceil(config.tile_size);
        let num_tiles_z = nz.div_ceil(config.tile_size);

        let vel_scratch = Array3::zeros((nx, ny, nz));
        let pres_scratch = Array3::zeros((nx, ny, nz));

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
            vel_scratch,
            pres_scratch,
        })
    }

    /// Get tile statistics for profiling
    pub fn tile_stats(&self) -> (usize, usize, usize) {
        (self.num_tiles_x, self.num_tiles_y, self.num_tiles_z)
    }

    /// Get total number of tiles
    pub fn total_tiles(&self) -> usize {
        self.num_tiles_x * self.num_tiles_y * self.num_tiles_z
    }

    /// Get configuration
    pub fn config(&self) -> SimdStencilConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stencil_creation() {
        let config = SimdStencilConfig::default();
        let result = SimdStencilProcessor::new(64, 64, 64, config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dimension_validation() {
        let config = SimdStencilConfig::default();
        let result = SimdStencilProcessor::new(2, 64, 64, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_tile_statistics() {
        let config = SimdStencilConfig::default();
        let processor = SimdStencilProcessor::new(64, 64, 64, config).unwrap();
        let (tx, ty, tz) = processor.tile_stats();
        assert!(tx > 0 && ty > 0 && tz > 0);
        assert_eq!(processor.total_tiles(), tx * ty * tz);
    }

    #[test]
    fn test_stability_check() {
        let config = SimdStencilConfig::default();
        // CFL constraint: c·dt/dx ≤ CFL_FACTOR_3D_FDTD (equality allowed — at stability limit)
        let cfl = config.sound_speed * config.dt / config.dx;
        assert!(
            cfl <= config.cfl_number + f64::EPSILON * 10.0,
            "CFL {cfl:.6} exceeds cfl_number {:.6}",
            config.cfl_number
        );
    }
}
