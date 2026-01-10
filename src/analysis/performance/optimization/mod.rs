//! Performance optimization module - properly modularized

pub mod cache;
pub mod config;
pub mod gpu;
pub mod memory;
pub mod parallel;
pub mod simd;

pub use cache::{AccessPattern, CacheOptimizer};
pub use config::{OptimizationConfig, SimdLevel};
pub use gpu::GpuOptimizer;
pub use memory::{BandwidthOptimizer, MemoryOptimizer, PrefetchStrategy};
pub use parallel::ParallelOptimizer;
pub use simd::SimdOptimizer;

use crate::core::error::KwaversResult;

/// Stencil kernel for finite difference computations
#[derive(Debug, Clone)]
pub struct StencilKernel {
    /// Stencil coefficients
    pub coefficients: Vec<f64>,
    /// Stencil radius
    pub radius: usize,
    /// Dimension (1D, 2D, or 3D)
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

/// Main optimization orchestrator
#[derive(Debug)]
pub struct PerformanceOptimizer {
    config: OptimizationConfig,
    simd: SimdOptimizer,
    cache: CacheOptimizer,
    #[allow(dead_code)] // Used for parallel optimization strategies
    parallel: ParallelOptimizer,
    memory: MemoryOptimizer,
    gpu: Option<GpuOptimizer>,
}

impl PerformanceOptimizer {
    /// Create a new optimizer with the given configuration
    pub fn new(config: OptimizationConfig) -> KwaversResult<Self> {
        let simd = SimdOptimizer::new(config.simd_level);
        let cache = CacheOptimizer::new(config.cache_block_size);
        let parallel = ParallelOptimizer::new();
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
            parallel,
            memory,
            gpu,
        })
    }

    /// Apply all optimizations based on configuration
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
