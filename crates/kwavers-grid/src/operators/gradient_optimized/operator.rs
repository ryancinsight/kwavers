//! Gradient operator builder and operator types.

use super::super::coefficients::FdAccuracyOrder;
use super::cache::GradientCache;
use super::functions::{gradient_optimized, gradient_with_strategy};
use crate::compat::leto::{Array3, ArrayView3};
use crate::Grid;
use kwavers_core::error::KwaversResult;
use leto::Array3 as LetoArray3;
use eunomia::FloatElement;

/// Boundary handling strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryStrategy {
    /// Zero padding at boundaries
    ZeroPadding,
    /// Mirror boundaries
    Mirror,
    /// Periodic boundaries
    Periodic,
    /// Extrapolate from interior
    Extrapolate,
}

/// Gradient operator builder for optimized configurations
#[derive(Debug, Clone)]
pub struct GradientOperatorBuilder {
    /// Use parallel computation
    pub parallel: bool,
    /// Use caching
    pub caching: bool,
    /// Chunk size for parallel processing
    pub chunk_size: usize,
    /// Boundary handling strategy
    pub boundary_strategy: BoundaryStrategy,
}

impl Default for GradientOperatorBuilder {
    fn default() -> Self {
        Self {
            parallel: true,
            caching: true,
            chunk_size: 16,
            boundary_strategy: BoundaryStrategy::ZeroPadding,
        }
    }
}

impl GradientOperatorBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set parallel computation
    #[must_use]
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Set caching
    #[must_use]
    pub fn with_caching(mut self, caching: bool) -> Self {
        self.caching = caching;
        self
    }

    /// Set chunk size
    #[must_use]
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Set boundary strategy
    #[must_use]
    pub fn with_boundary_strategy(mut self, strategy: BoundaryStrategy) -> Self {
        self.boundary_strategy = strategy;
        self
    }

    /// Build the gradient operator
    #[must_use]
    pub fn build(&self) -> GradientOperator {
        GradientOperator {
            parallel: self.parallel,
            caching: self.caching,
            chunk_size: self.chunk_size,
            boundary_strategy: self.boundary_strategy,
        }
    }
}

/// Optimized gradient operator
#[derive(Debug, Clone)]
pub struct GradientOperator {
    /// Use parallel computation
    parallel: bool,
    /// Use caching
    caching: bool,
    /// Chunk size for parallel processing
    chunk_size: usize,
    /// Boundary handling strategy
    boundary_strategy: BoundaryStrategy,
}

impl GradientOperator {
    /// Compute gradient using optimized configuration
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn compute<T>(
        &self,
        field: &ArrayView3<T>,
        grid: &Grid,
        order: FdAccuracyOrder,
        cache: Option<&GradientCache<T>>,
    ) -> KwaversResult<(Array3<T>, Array3<T>, Array3<T>)>
    where
        T: FloatElement + Clone + Send + Sync + Default,
    {
        let _chunk_size = self.chunk_size.max(1);
        if self.parallel {
            if self.caching && cache.is_some() {
                gradient_optimized(field, grid, order, cache)
            } else {
                gradient_with_strategy(field, grid, order, self.boundary_strategy)
            }
        } else {
            gradient_with_strategy(field, grid, order, self.boundary_strategy)
        }
    }

    /// Compute gradient for leto arrays using the same operator configuration.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn compute_leto<T>(
        &self,
        field: &LetoArray3<T>,
        grid: &Grid,
        order: FdAccuracyOrder,
        cache: Option<&GradientCache<T>>,
    ) -> KwaversResult<(LetoArray3<T>, LetoArray3<T>, LetoArray3<T>)>
    where
        T: FloatElement + Clone + Send + Sync + Default,
    {
        let field_view = field.view();
        self.compute(&field_view, grid, order, cache)
    }
}
