//! Performance optimization module - properly modularized

pub mod cache;
pub mod config;
pub mod gpu;
pub mod memory;
pub mod parallel;
pub mod simd;

pub use cache::{AccessPattern, CacheOptimizer};
pub use config::{HardwareOptimizationConfig, PerfOptSimdLevel};
pub use gpu::GpuOptimizer;
pub use memory::{BandwidthOptimizer, MemoryOptimizer, PrefetchStrategy};
pub use parallel::ParallelOptimizer;
pub use simd::SimdOptimizer;

use kwavers_core::error::KwaversResult;

/// Stencil kernel for finite difference computations
#[derive(Debug, Clone)]
pub struct StencilKernel {
    /// Stencil coefficients
    pub coefficients: Vec<f64>,
    /// Stencil radius
    pub radius: usize,
    /// GridDimension (1D, 2D, or 3D)
    pub dimension: usize,
}

impl StencilKernel {
    /// Create a new stencil kernel
    #[must_use]
    pub fn new(coefficients: Vec<f64>, radius: usize, dimension: usize) -> Self {
        Self {
            coefficients,
            radius,
            dimension,
        }
    }

    /// Apply stencil to a point
    #[must_use]
    pub fn apply(&self, data: &[f64], index: usize, stride: usize) -> f64 {
        let mut result = 0.0;

        for (coeff_idx, offset) in (-(self.radius as isize)..=(self.radius as isize)).enumerate() {
            let data_idx = (index as isize + offset * stride as isize) as usize;
            if data_idx < data.len() {
                result += data[data_idx] * self.coefficients[coeff_idx];
            }
        }

        result
    }
}

/// Main optimization orchestrator.
///
/// `ParallelOptimizer` is re-exported from the `parallel` sub-module and can be
/// used independently; it is not stored as a field here until the parallel
/// dispatch strategy is wired into `apply_optimizations`.
#[derive(Debug)]
pub struct PerformanceOptimizer {
    config: HardwareOptimizationConfig,
    simd: SimdOptimizer,
    cache: CacheOptimizer,
    memory: MemoryOptimizer,
    gpu: Option<GpuOptimizer>,
}

impl PerformanceOptimizer {
    /// Create a new optimizer with the given configuration
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(config: HardwareOptimizationConfig) -> KwaversResult<Self> {
        let simd = SimdOptimizer::new(config.simd_level);
        let cache = CacheOptimizer::new(config.cache_block_size);
        let memory = MemoryOptimizer::new(config.prefetch_distance);

        let gpu = if config.multi_gpu || config.kernel_fusion {
            Some(GpuOptimizer::new(config.gpu_streams)?)
        } else {
            None
        };

        Ok(Self {
            config,
            simd,
            cache,
            memory,
            gpu,
        })
    }

    /// Apply all optimizations based on configuration
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn optimize(&self) -> KwaversResult<()> {
        if self.config.enable_simd {
            self.simd.apply_optimizations()?;
        }

        if self.config.cache_blocking {
            self.cache.optimize_blocking()?;
        }

        // Note: Parallel optimization is applied automatically through rayon
        // when using parallel iterators - no explicit optimization needed

        if self.config.prefetching {
            self.memory.enable_prefetching()?;
        }

        if let Some(ref gpu) = self.gpu {
            gpu.optimize_kernels()?;
        }

        Ok(())
    }
}
