//! Constructor for `Avx512StencilProcessor`.
//!
//! SRP: changes when precomputed coefficient layout or SIMD detection policy changes.

use super::{Avx512Config, Avx512StencilProcessor};
use crate::core::error::{KwaversError, KwaversResult};
use crate::math::simd::SimdConfig;
use std::marker::PhantomData;

impl Avx512StencilProcessor {
    /// Create a new AVX-512 stencil processor.
    ///
    /// # Arguments
    /// * `nx`, `ny`, `nz` — grid dimensions (must be ≥ 4)
    /// * `config` — processor configuration
    ///
    /// # Returns
    /// `Err` if dimensions are < 4 or `tile_size` is not a power of two.
    pub fn new(nx: usize, ny: usize, nz: usize, config: Avx512Config) -> KwaversResult<Self> {
        if nx < 4 || ny < 4 || nz < 4 {
            return Err(KwaversError::InvalidInput(
                "Grid dimensions must be >= 4 for AVX-512 stencil".to_string(),
            ));
        }
        if config.tile_size == 0 || (config.tile_size & (config.tile_size - 1)) != 0 {
            return Err(KwaversError::InvalidInput(
                "Tile size must be power of 2".to_string(),
            ));
        }

        let c_sq = config.sound_speed * config.sound_speed;
        let pressure_coeff = -c_sq * config.dt * config.dt / (config.dx * config.dx);
        let pressure_central_coeff = 2.0 - 6.0 * pressure_coeff;
        let velocity_coeff = -config.dt / (config.density * config.dx);

        let simd_config = SimdConfig::detect();

        #[cfg(target_arch = "x86_64")]
        if simd_config.level < crate::math::simd::SimdLevel::Avx512 {
            return Err(KwaversError::FeatureNotAvailable(
                "AVX-512 not available on this CPU".to_string(),
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
