//! Constructor for `FdtdAvx512StencilProcessor`.
//!
//! SRP: changes when precomputed coefficient layout or SIMD detection policy changes.

use super::{FdtdAvx512Config, FdtdAvx512StencilProcessor};
use crate::core::error::{KwaversError, KwaversResult};
use crate::math::simd::SimdConfig;
use std::marker::PhantomData;

impl FdtdAvx512StencilProcessor {
    /// Create a new AVX-512 stencil processor.
    ///
    /// # Arguments
    /// * `nx`, `ny`, `nz` — grid dimensions (must be ≥ 4)
    /// * `config` — processor configuration
    ///
    /// # Returns
    /// `Err` if dimensions are < 4 or `tile_size` is not a power of two.
    /// # Errors
    /// - Returns [`KwaversError::FeatureNotAvailable`] if the precondition for a FeatureNotAvailable-class constraint is violated.
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn new(nx: usize, ny: usize, nz: usize, config: FdtdAvx512Config) -> KwaversResult<Self> {
        if nx < 4 || ny < 4 || nz < 4 {
            return Err(KwaversError::InvalidInput(
                "Grid dimensions must be >= 4 for AVX-512 stencil".to_owned(),
            ));
        }
        if config.tile_size == 0 || (config.tile_size & (config.tile_size - 1)) != 0 {
            return Err(KwaversError::InvalidInput(
                "Tile size must be power of 2".to_owned(),
            ));
        }

        let c_sq = config.sound_speed * config.sound_speed;
        let pressure_coeff = -c_sq * config.dt * config.dt / (config.dx * config.dx);
        let pressure_central_coeff = 6.0f64.mul_add(-pressure_coeff, 2.0);
        let velocity_coeff = -config.dt / (config.density * config.dx);

        let simd_config = SimdConfig::detect();

        #[cfg(target_arch = "x86_64")]
        if simd_config.level < crate::math::simd::MathSimdLevel::Avx512 {
            return Err(KwaversError::FeatureNotAvailable(
                "AVX-512 not available on this CPU".to_owned(),
            ));
        }

        Ok(Self {
            config,
            nx,
            ny,
            nz,
            pressure_coeff,
            velocity_coeff,
            pressure_central_coeff,
            simd_config,
            _phantom: PhantomData,
        })
    }
}
