//! Gradient operator builder and operator types.

use super::super::coefficients::FdAccuracyOrder;
use super::cache::GradientCache;
use super::functions::{compute_gradient, fill_gradient};
use crate::operators::stencil::validate_field_shape;
use crate::Grid;
use eunomia::FloatElement;
use kwavers_core::error::{GridError, KwaversError, KwaversResult};
use leto::{Array3, ArrayView3};

/// Boundary handling strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryStrategy {
    /// Zero padding at boundaries
    ZeroPadding,
    /// Reflect samples about the two endpoint nodes.
    Mirror,
    /// Periodic boundaries
    Periodic,
    /// Extend each endpoint value as a constant.
    Extrapolate,
}

/// Gradient operator builder for optimized configurations
#[derive(Debug, Clone)]
pub struct GradientOperatorBuilder {
    /// Use parallel computation
    pub parallel: bool,
    /// Use caching
    pub caching: bool,
    /// Number of z-rows per work chunk (zero selects one row).
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

    /// Set the number of z-rows per work chunk.
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
    /// Number of z-rows per work chunk (zero selects one row).
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
        compute_gradient(
            field,
            grid,
            order,
            cache.filter(|_| self.caching),
            Some(self.boundary_strategy),
            self.parallel.then_some(self.chunk_size),
        )
    }
    /// Write x, y and z derivatives directly into caller-owned row-major buffers.
    /// Each buffer must have one element per grid node, with z varying fastest.
    ///
    /// # Errors
    /// Field and output size mismatches are reported before any output is changed.
    pub fn compute_into<T: FloatElement>(
        &self,
        field: &ArrayView3<T>,
        grid: &Grid,
        order: FdAccuracyOrder,
        cache: Option<&GradientCache<T>>,
        outputs: [&mut [T]; 3],
    ) -> KwaversResult<()> {
        validate_field_shape(field, grid)?;
        let n = field.size();
        for (axis, output) in ["x", "y", "z"].into_iter().zip(&outputs) {
            if output.len() != n {
                return Err(KwaversError::Grid(GridError::DimensionMismatch {
                    expected: format!("{n} elements for gradient {axis}"),
                    actual: output.len().to_string(),
                }));
            }
        }
        fill_gradient(
            field,
            grid,
            order,
            cache.filter(|_| self.caching),
            Some(self.boundary_strategy),
            self.parallel.then_some(self.chunk_size),
            outputs,
        );
        Ok(())
    }
}
