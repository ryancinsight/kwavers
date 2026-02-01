//! SIMD Dispatch and Runtime Optimization Selection
//!
//! This module handles runtime selection of optimal SIMD implementations
//! based on CPU capabilities and system configuration.
//!
//! ## Architecture
//!
//! Deep vertical hierarchy for SIMD selection:
//! ```
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

use crate::core::error::{KwaversError, KwaversResult};
use crate::math::simd::{SimdConfig, SimdLevel};
use ndarray::Array3;
use std::sync::OnceLock;

/// Global SIMD configuration (computed once at startup)
static SIMD_CONFIG: OnceLock<SimdConfig> = OnceLock::new();

/// FDTD stencil optimization strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
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
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Scalar => "Scalar",
            Self::GenericSimd => "Generic SIMD",
            Self::Avx512 => "AVX-512",
            Self::Auto => "Auto",
        }
    }

    /// Select best available strategy
    pub fn select_best() -> Self {
        let config = get_simd_config();
        match config.level {
            SimdLevel::Avx512 => Self::Avx512,
            SimdLevel::Avx2 | SimdLevel::Sse2 | SimdLevel::Neon => Self::GenericSimd,
            _ => Self::Scalar,
        }
    }

    /// Verify strategy is available on this system
    pub fn is_available(&self) -> bool {
        match self {
            Self::Scalar => true,
            Self::GenericSimd => {
                let config = get_simd_config();
                config.level >= SimdLevel::Sse2 || config.level == SimdLevel::Neon
            }
            Self::Avx512 => {
                let config = get_simd_config();
                config.level >= SimdLevel::Avx512
            }
            Self::Auto => true,
        }
    }
}

/// Get global SIMD configuration (thread-safe singleton)
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
}

impl FdtdStencilDispatcher {
    /// Create new dispatcher with automatic strategy selection
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
                "Grid dimensions must be >= 3".to_string(),
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
        })
    }

    /// Update pressure field using selected strategy
    ///
    /// Routes to appropriate implementation based on strategy selection.
    pub fn update_pressure(
        &self,
        p_curr: &Array3<f64>,
        p_prev: &Array3<f64>,
        u_div: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        // Validate dimensions match
        if p_curr.shape() != p_prev.shape() || p_curr.shape() != u_div.shape() {
            return Err(KwaversError::InvalidInput(
                "All fields must have identical dimensions".to_string(),
            ));
        }

        if p_curr.dim() != (self.nx, self.ny, self.nz) {
            return Err(KwaversError::InvalidInput(
                "Field dimensions do not match processor configuration".to_string(),
            ));
        }

        match self.strategy {
            StencilStrategy::Avx512 => {
                #[cfg(target_arch = "x86_64")]
                {
                    use crate::solver::forward::fdtd::Avx512Config;

                    let config = Avx512Config {
                        tile_size: 8,
                        use_fma: true,
                        prefetch_boundaries: true,
                        sound_speed: 1540.0,
                        density: 1000.0,
                        dx: 0.001,
                        dt: 1.62e-7,
                    };

                    let processor = crate::solver::forward::fdtd::Avx512StencilProcessor::new(
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
                // Use default SimdStencilConfig
                let processor = crate::solver::forward::fdtd::SimdStencilProcessor::new(
                    self.nx,
                    self.ny,
                    self.nz,
                    Default::default(),
                )?;
                processor.update_pressure(p_curr, p_prev, u_div)
            }
            StencilStrategy::Scalar => {
                // Scalar fallback: simple 3D stencil
                self.update_pressure_scalar(p_curr, p_prev, u_div)
            }
            StencilStrategy::Auto => {
                // Should not reach here (Auto is converted in constructor)
                Err(KwaversError::InternalError(
                    "Auto strategy not resolved".to_string(),
                ))
            }
        }
    }

    /// Scalar implementation (reference/fallback)
    fn update_pressure_scalar(
        &self,
        p_curr: &Array3<f64>,
        p_prev: &Array3<f64>,
        _u_div: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let mut p_new = Array3::zeros((self.nx, self.ny, self.nz));

        // Interior points
        for k in 1..self.nz - 1 {
            for j in 1..self.ny - 1 {
                for i in 1..self.nx - 1 {
                    // 6-point stencil laplacian
                    let laplacian = p_curr[[i - 1, j, k]]
                        + p_curr[[i + 1, j, k]]
                        + p_curr[[i, j - 1, k]]
                        + p_curr[[i, j + 1, k]]
                        + p_curr[[i, j, k - 1]]
                        + p_curr[[i, j, k + 1]]
                        - 6.0 * p_curr[[i, j, k]];

                    // Pressure update: p_new = 2*p_curr - p_prev + coeff*laplacian
                    p_new[[i, j, k]] = 2.0 * p_curr[[i, j, k]] - p_prev[[i, j, k]]
                        + self.pressure_coeff * laplacian;
                }
            }
        }

        Ok(p_new)
    }

    /// Get current strategy
    pub fn strategy(&self) -> StencilStrategy {
        self.strategy
    }

    /// Get performance metrics
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
    pub simd_level: SimdLevel,

    /// SIMD vector width in elements
    pub vector_width: usize,

    /// Grid dimensions
    pub grid_size: (usize, usize, usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_config_detection() {
        init_simd();
        let config = get_simd_config();
        println!("Detected SIMD level: {:?}", config.level);
        assert!(config.enabled || config.level == SimdLevel::Scalar);
    }

    #[test]
    fn test_strategy_selection() {
        let best = StencilStrategy::select_best();
        println!("Best strategy: {}", best.as_str());
        assert!(best.is_available());
    }

    #[test]
    fn test_dispatcher_creation() {
        let result = FdtdStencilDispatcher::new(32, 32, 32, -1.0, -1.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dispatcher_invalid_dimensions() {
        let result = FdtdStencilDispatcher::new(2, 32, 32, -1.0, -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_dispatcher_strategy_override() {
        let result =
            FdtdStencilDispatcher::with_strategy(32, 32, 32, -1.0, -1.0, StencilStrategy::Scalar);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().strategy(), StencilStrategy::Scalar);
    }

    #[test]
    fn test_pressure_update_scalar() {
        let dispatcher =
            FdtdStencilDispatcher::with_strategy(16, 16, 16, -1.0, -1.0, StencilStrategy::Scalar)
                .unwrap();

        let p_curr = Array3::zeros((16, 16, 16));
        let p_prev = Array3::zeros((16, 16, 16));
        let u_div = Array3::zeros((16, 16, 16));

        let result = dispatcher.update_pressure(&p_curr, &p_prev, &u_div);
        assert!(result.is_ok());
    }

    #[test]
    fn test_metrics() {
        let dispatcher = FdtdStencilDispatcher::new(32, 32, 32, -1.0, -1.0).unwrap();
        let metrics = dispatcher.metrics();
        println!(
            "Strategy: {}, SIMD Level: {:?}, Vector Width: {}",
            metrics.selected_strategy.as_str(),
            metrics.simd_level,
            metrics.vector_width
        );
    }
}
