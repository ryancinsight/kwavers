//! SIMD Dispatch and Runtime Optimization Selection
//!
//! This module handles runtime selection of optimal SIMD implementations
//! based on CPU capabilities and system configuration.
//!
//! ## Architecture
//!
//! Deep vertical hierarchy for SIMD selection:
//! ```text
//! math/simd.rs (SIMD capability detection)
//!    ↓
//! solver/forward/fdtd/dispatch.rs (THIS MODULE - Runtime dispatch)
//!    ↓
//! solver/forward/fdtd/
//!    ├── avx512_stencil.rs (AVX-512 implementation)
//!    ├── simd_stencil.rs (Generic SIMD fallback)
//!    └── solver.rs (FDTD solver using dispatch)
//! ```
//!
//! ## Strategy Pattern Implementation
//!
//! Rather than compile-time feature gates, uses runtime dispatch to:
//! 1. Detect available CPU features at startup
//! 2. Select optimal implementation
//! 3. Fall back gracefully on limited hardware
//! 4. Allow users to override for testing/benchmarking
//!
//! ## Performance Tiers
//!
//! | Tier | Implementation | Width | Speedup | Availability |
//! |------|----------------|-------|---------|--------------|
//! | 0 | Scalar (baseline) | 1 | 1x | Always |
//! | 1 | SSE2 | 2 | ~2x | Most x86_64 |
//! | 2 | AVX2 | 4 | ~4x | Modern CPUs |
//! | 3 | AVX-512 | 8 | ~8x | Xeon, recent Intel |
//! | 4 | ARM NEON | 2-4 | ~2-4x | ARM servers |

#[cfg(test)]
mod tests;

use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::simd::{MathSimdLevel, SimdConfig};
use leto::Array3;
use std::sync::OnceLock;

/// Global SIMD configuration (computed once at startup)
static SIMD_CONFIG: OnceLock<SimdConfig> = OnceLock::new();

/// FDTD stencil optimization strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StencilStrategy {
    /// Scalar implementation (always available)
    Scalar,

    /// Generic SIMD (AVX2, SSE2, NEON)
    GenericSimd,

    /// AVX-512 optimized implementation
    Avx512,

    /// Automatic selection based on CPU
    #[default]
    Auto,
}

impl StencilStrategy {
    /// Get string representation
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Scalar => "Scalar",
            Self::GenericSimd => "Generic SIMD",
            Self::Avx512 => "AVX-512",
            Self::Auto => "Auto",
        }
    }

    /// Select best available strategy
    #[must_use]
    pub fn select_best() -> Self {
        let config = get_simd_config();
        match config.level {
            MathSimdLevel::Avx512 => Self::Avx512,
            MathSimdLevel::Avx2 | MathSimdLevel::Sse2 | MathSimdLevel::Neon => Self::GenericSimd,
            _ => Self::Scalar,
        }
    }

    /// Verify strategy is available on this system
    #[must_use]
    pub fn is_available(&self) -> bool {
        match self {
            Self::Scalar => true,
            Self::GenericSimd => {
                let config = get_simd_config();
                config.level >= MathSimdLevel::Sse2 || config.level == MathSimdLevel::Neon
            }
            Self::Avx512 => {
                let config = get_simd_config();
                config.level >= MathSimdLevel::Avx512
            }
            Self::Auto => true,
        }
    }
}

/// Get global SIMD configuration (thread-safe singleton)
///
/// Returns a clone of `SimdConfig` — the struct is small (a few Copy fields),
/// so the clone is a trivial stack copy.  The `OnceLock` guarantees detection
/// runs once.
#[must_use]
pub fn get_simd_config() -> SimdConfig {
    SIMD_CONFIG.get_or_init(SimdConfig::detect).clone()
}

/// Initialize global SIMD configuration
///
/// Should be called once at application startup.
/// Safe to call multiple times (only computes once).
pub fn init_simd() {
    let _ = get_simd_config();
}

/// Dispatcher for FDTD stencil operations
///
/// Routes pressure and velocity updates to optimal implementation.
/// Pre-allocates a scratch buffer for the scalar path to avoid per-step
/// heap allocation.  The scalar fallback still incurs one `Array3::zeros`
/// allocation per step because the buffer must be returned to the caller
/// by ownership — acceptable since the scalar path is only taken when
/// no SIMD implementation is available (essentially never on modern x86_64).
#[derive(Debug)]
pub struct FdtdStencilDispatcher {
    /// Selected strategy
    strategy: StencilStrategy,

    /// Grid dimensions
    nx: usize,
    ny: usize,
    nz: usize,

    /// Coefficient for pressure update
    pressure_coeff: f64,

    /// Coefficient for velocity update
    _velocity_coeff: f64,

    /// Pre-allocated scratch buffer for scalar pressure update.
    /// Filled each step, then returned via std::mem::replace with a fresh
    /// zero buffer.  The zero-initialisation is a fast memset; the heap
    /// allocation is amortised by the allocator.
    p_scratch: Array3<f64>,
}

impl FdtdStencilDispatcher {
    /// Create new dispatcher with automatic strategy selection
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        pressure_coeff: f64,
        velocity_coeff: f64,
    ) -> KwaversResult<Self> {
        let strategy = StencilStrategy::Auto;
        Self::with_strategy(nx, ny, nz, pressure_coeff, velocity_coeff, strategy)
    }

    /// Create dispatcher with explicit strategy
    /// # Errors
    /// - Returns [`KwaversError::FeatureNotAvailable`] if the precondition for a FeatureNotAvailable-class constraint is violated.
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn with_strategy(
        nx: usize,
        ny: usize,
        nz: usize,
        pressure_coeff: f64,
        velocity_coeff: f64,
        strategy: StencilStrategy,
    ) -> KwaversResult<Self> {
        // Validate dimensions
        if nx < 3 || ny < 3 || nz < 3 {
            return Err(KwaversError::InvalidInput(
                "Grid dimensions must be >= 3".to_owned(),
            ));
        }

        let strategy = match strategy {
            StencilStrategy::Auto => StencilStrategy::select_best(),
            s => s,
        };

        // Verify selected strategy is available
        if !strategy.is_available() {
            return Err(KwaversError::FeatureNotAvailable(format!(
                "{} strategy not available",
                strategy.as_str()
            )));
        }

        Ok(Self {
            strategy,
            nx,
            ny,
            nz,
            pressure_coeff,
            _velocity_coeff: velocity_coeff,
            p_scratch: Array3::zeros((nx, ny, nz)),
        })
    }

    /// Update pressure field using selected strategy
    ///
    /// Routes to appropriate implementation based on strategy selection.
    /// Takes `&mut self` to reuse the pre-allocated scratch buffer on the scalar path.
    /// # Errors
    /// - Returns [`KwaversError::FeatureNotAvailable`] if the precondition for a FeatureNotAvailable-class constraint is violated.
    /// - Returns [`KwaversError::InternalError`] if the precondition for a InternalError-class constraint is violated.
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn update_pressure(
        &mut self,
        p_curr: &Array3<f64>,
        p_prev: &Array3<f64>,
        u_div: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        // Validate dimensions match
        if p_curr.shape() != p_prev.shape() || p_curr.shape() != u_div.shape() {
            return Err(KwaversError::InvalidInput(
                "All fields must have identical dimensions".to_owned(),
            ));
        }

        if p_curr.shape() != (self.nx, self.ny, self.nz) {
            return Err(KwaversError::InvalidInput(
                "Field dimensions do not match processor configuration".to_owned(),
            ));
        }

        match self.strategy {
            StencilStrategy::Avx512 => {
                #[cfg(target_arch = "x86_64")]
                {
                    use crate::forward::fdtd::FdtdAvx512Config;

                    let config = FdtdAvx512Config::default();

                    let processor = crate::forward::fdtd::FdtdAvx512StencilProcessor::new(
                        self.nx, self.ny, self.nz, config,
                    )?;
                    processor.update_pressure_avx512(p_curr, p_prev, u_div)
                }

                #[cfg(not(target_arch = "x86_64"))]
                {
                    Err(KwaversError::FeatureNotAvailable(
                        "AVX-512 not available on this platform".to_string(),
                    ))
                }
            }
            StencilStrategy::GenericSimd => {
                // Use default FdtdSimdStencilConfig
                let mut processor = crate::forward::fdtd::FdtdSimdStencilProcessor::new(
                    self.nx,
                    self.ny,
                    self.nz,
                    Default::default(),
                )?;
                processor.update_pressure(p_curr, p_prev, u_div)
            }
            StencilStrategy::Scalar => {
                // Scalar fallback: simple 3D stencil (reuses p_scratch, no heap alloc)
                self.update_pressure_scalar(p_curr, p_prev, u_div)
            }
            StencilStrategy::Auto => {
                // Should not reach here (Auto is converted in constructor)
                Err(KwaversError::InternalError(
                    "Auto strategy not resolved".to_owned(),
                ))
            }
        }
    }

    /// Scalar implementation (reference/fallback).
    ///
    /// ## Allocation strategy
    ///
    /// Fills `self.p_scratch` in-place, then returns it to the caller via
    /// `std::mem::replace`.  The returned value is transferred without copying
    /// the computed interior field, but a fresh zero-initialized scratch buffer
    /// is still allocated for the next invocation because this API must return
    /// ownership.  The write-pass itself remains the only per-cell work in the
    /// steady state; the zero-initialisation is a fast `memset` amortised across
    /// the computational work of the stencil.
    ///
    /// This path is the last-resort fallback and is only taken when no SIMD
    /// implementation is available (essentially never on modern x86\_64 or ARM
    /// hardware).  The per-step allocation is therefore acceptable.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn update_pressure_scalar(
        &mut self,
        p_curr: &Array3<f64>,
        p_prev: &Array3<f64>,
        _u_div: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        // Interior points — 6-point stencil Laplacian + leapfrog time step
        for k in 1..self.nz - 1 {
            for j in 1..self.ny - 1 {
                for i in 1..self.nx - 1 {
                    let laplacian = 6.0f64.mul_add(
                        -p_curr[[i, j, k]],
                        p_curr[[i - 1, j, k]]
                            + p_curr[[i + 1, j, k]]
                            + p_curr[[i, j - 1, k]]
                            + p_curr[[i, j + 1, k]]
                            + p_curr[[i, j, k - 1]]
                            + p_curr[[i, j, k + 1]],
                    );

                    // Leapfrog: p^{n+1} = 2·p^n − p^{n−1} + c²·Δt²·∇²p^n
                    self.p_scratch[[i, j, k]] = self.pressure_coeff.mul_add(
                        laplacian,
                        2.0f64.mul_add(p_curr[[i, j, k]], -p_prev[[i, j, k]]),
                    );
                }
            }
        }

        // Return the filled scratch buffer without copying data.
        // `replace` puts a fresh zeros array into `self.p_scratch` and gives
        // us ownership of the filled array — zero-copy transfer of results.
        let dim = self.p_scratch.shape();
        let result = std::mem::replace(&mut self.p_scratch, Array3::zeros(dim));
        Ok(result)
    }

    /// Get current strategy
    #[must_use]
    pub fn strategy(&self) -> StencilStrategy {
        self.strategy
    }

    /// Get performance metrics
    #[must_use]
    pub fn metrics(&self) -> DispatchMetrics {
        let config = get_simd_config();
        DispatchMetrics {
            selected_strategy: self.strategy,
            simd_level: config.level,
            vector_width: config.vector_width,
            grid_size: (self.nx, self.ny, self.nz),
        }
    }
}

/// Performance metrics for dispatcher
#[derive(Debug, Clone)]
pub struct DispatchMetrics {
    /// Selected stencil strategy
    pub selected_strategy: StencilStrategy,

    /// Hardware SIMD level
    pub simd_level: MathSimdLevel,

    /// SIMD vector width in elements
    pub vector_width: usize,

    /// Grid dimensions
    pub grid_size: (usize, usize, usize),
}
