//! Performance optimization module - properly modularized

pub mod cache;
pub mod config;
pub mod memory;
pub mod parallel;
pub mod simd;

pub use cache::{AccessPattern, CacheOptimizer};
pub use config::PerfOptSimdLevel;
pub use memory::{BandwidthOptimizer, MemoryOptimizer, PrefetchStrategy};
pub use parallel::ParallelOptimizer;
pub use simd::SimdOptimizer;

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
