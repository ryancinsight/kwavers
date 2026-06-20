//! Constructor for `FdtdAvx512StencilProcessor`.
//!
//! SRP: changes when precomputed coefficient layout or SIMD detection policy changes.

use super::{FdtdAvx512Config, FdtdAvx512StencilProcessor};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::simd::SimdConfig;
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
        // Leapfrog stencil for the acoustic wave equation `∂²p/∂t² = c²∇²p`:
        //   p^(n+1) = (2 − 6 r)·p^n − p^(n-1) + r·Σ_neighbors
        // with r = c²·Δt²/Δx². This matches the docstring in `pressure.rs`.
        // The previous code had `pressure_coeff = −r` and assembled the update
        // with double-negated signs, which evaluated to `p^(n+1) = 2p − p_prev
        // − c²Δt²·∇²p` — the *wrong-sign* wave equation, producing anti-
        // propagating / amplifying numerics. Fixed to the canonical leapfrog.
        let pressure_coeff = c_sq * config.dt * config.dt / (config.dx * config.dx);
        let pressure_central_coeff = 6.0f64.mul_add(-pressure_coeff, 2.0);
        // The leapfrog coefficients drive every stencil tap; a non-finite value
        // (from a zero/NaN spacing or overflow in c²Δt²) would silently poison the
        // whole field. Catch it at construction in debug/test builds.
        debug_assert!(
            pressure_coeff.is_finite() && pressure_central_coeff.is_finite(),
            "non-finite leapfrog coeffs: r={pressure_coeff}, central={pressure_central_coeff}"
        );
        // Centered-difference momentum equation
        //   u^(n+1) = u^n − (Δt / (2 ρ Δx))·(p[+1] − p[−1])
        // The factor 1/2 from the centered difference was previously omitted,
        // making the velocity update twice as aggressive.
        let velocity_coeff = -config.dt / (2.0 * config.density * config.dx);
        debug_assert!(
            velocity_coeff.is_finite(),
            "non-finite velocity coeff: {velocity_coeff}"
        );

        let simd_config = SimdConfig::detect();

        #[cfg(target_arch = "x86_64")]
        if simd_config.level < kwavers_math::simd::MathSimdLevel::Avx512 {
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
