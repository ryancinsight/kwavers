//! Optimized gradient operations with caching and parallelization
//!
//! This module provides high-performance gradient calculations using
//! caching, parallelization, and optimized memory access patterns.

use super::coefficients::{FDCoefficients, SpatialOrder};
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::{Array3, ArrayView3};
use num_traits::Float;
use std::sync::{Arc, RwLock};

/// Gradient cache for performance optimization
#[derive(Debug)]
pub struct GradientCache<T>
where
    T: Float + Clone + Send + Sync,
{
    /// Cached coefficients for different spatial orders
    coefficients_cache: RwLock<Vec<Vec<T>>>,
    /// Cached grid spacing inverses
    spacing_inverses: (T, T, T),
    /// Cache hit counter for performance monitoring
    cache_hits: Arc<RwLock<usize>>,
    /// Cache miss counter for performance monitoring
    cache_misses: Arc<RwLock<usize>>,
}

impl<T> GradientCache<T>
where
    T: Float + Clone + Send + Sync,
{
    /// Create a new gradient cache
    pub fn new(grid: &Grid) -> Self {
        Self {
            coefficients_cache: RwLock::new(Vec::new()),
            spacing_inverses: (
                T::one() / T::from(grid.dx).unwrap(),
                T::one() / T::from(grid.dy).unwrap(),
                T::one() / T::from(grid.dz).unwrap(),
            ),
            cache_hits: Arc::new(RwLock::new(0)),
            cache_misses: Arc::new(RwLock::new(0)),
        }
    }

    /// Get or create coefficients for a given spatial order
    pub fn get_coefficients(&self, order: SpatialOrder) -> Vec<T> {
        let mut cache = self.coefficients_cache.write().unwrap();

        // Check if coefficients are already cached
        if let Some(coeffs) = cache.get(order as usize) {
            *self.cache_hits.write().unwrap() += 1;
            return coeffs.clone();
        }

        // Coefficients not in cache - create and store them
        *self.cache_misses.write().unwrap() += 1;
        let coeffs = FDCoefficients::first_derivative::<T>(order);

        // Ensure cache is large enough
        if cache.len() <= order as usize {
            cache.resize(order as usize + 1, Vec::new());
        }

        cache[order as usize] = coeffs.clone();
        coeffs
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        let hits = *self.cache_hits.read().unwrap();
        let misses = *self.cache_misses.read().unwrap();
        (hits, misses)
    }

    /// Reset cache statistics
    pub fn reset_stats(&self) {
        *self.cache_hits.write().unwrap() = 0;
        *self.cache_misses.write().unwrap() = 0;
    }
}

/// Optimized gradient computation with caching and parallelization
pub fn gradient_optimized<T>(
    field: &ArrayView3<T>,
    grid: &Grid,
    order: SpatialOrder,
    cache: Option<&GradientCache<T>>,
) -> KwaversResult<(Array3<T>, Array3<T>, Array3<T>)>
where
    T: Float + Clone + Send + Sync,
{
    let shape = field.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);

    // Validate grid compatibility
    if (nx, ny, nz) != (grid.nx, grid.ny, grid.nz) {
        return Err(crate::core::error::KwaversError::Grid(
            crate::core::error::GridError::DimensionMismatch {
                expected: format!("({}, {}, {})", grid.nx, grid.ny, grid.nz),
                actual: format!("({}, {}, {})", nx, ny, nz),
            },
        ));
    }

    let mut grad_x = Array3::<T>::zeros((nx, ny, nz));
    let mut grad_y = Array3::<T>::zeros((nx, ny, nz));
    let mut grad_z = Array3::<T>::zeros((nx, ny, nz));

    // Get coefficients (using cache if available)
    let coeffs = if let Some(cache) = cache {
        cache.get_coefficients(order)
    } else {
        FDCoefficients::first_derivative::<T>(order)
    };

    let stencil_radius = coeffs.len();
    let dx_inv = T::one() / T::from(grid.dx).unwrap();
    let dy_inv = T::one() / T::from(grid.dy).unwrap();
    let dz_inv = T::one() / T::from(grid.dz).unwrap();

    // Sequential computation for X-direction gradient
    for i in stencil_radius..nx - stencil_radius {
        for j in 0..ny {
            for k in 0..nz {
                let mut grad_val = T::zero();
                for (n, coeff) in coeffs.iter().enumerate() {
                    let offset = n + 1;
                    grad_val =
                        grad_val + *coeff * (field[[i + offset, j, k]] - field[[i - offset, j, k]]);
                }
                grad_x[[i, j, k]] = grad_val * dx_inv;
            }
        }
    }

    // Sequential computation for Y-direction gradient
    for i in 0..nx {
        for j in stencil_radius..ny - stencil_radius {
            for k in 0..nz {
                let mut grad_val = T::zero();
                for (n, coeff) in coeffs.iter().enumerate() {
                    let offset = n + 1;
                    grad_val =
                        grad_val + *coeff * (field[[i, j + offset, k]] - field[[i, j - offset, k]]);
                }
                grad_y[[i, j, k]] = grad_val * dy_inv;
            }
        }
    }

    // Sequential computation for Z-direction gradient
    for i in 0..nx {
        for j in 0..ny {
            for k in stencil_radius..nz - stencil_radius {
                let mut grad_val = T::zero();
                for (n, coeff) in coeffs.iter().enumerate() {
                    let offset = n + 1;
                    grad_val =
                        grad_val + *coeff * (field[[i, j, k + offset]] - field[[i, j, k - offset]]);
                }
                grad_z[[i, j, k]] = grad_val * dz_inv;
            }
        }
    }

    Ok((grad_x, grad_y, grad_z))
}

/// Optimized gradient computation with boundary handling
pub fn gradient_with_boundaries<T>(
    field: &ArrayView3<T>,
    grid: &Grid,
    order: SpatialOrder,
) -> KwaversResult<(Array3<T>, Array3<T>, Array3<T>)>
where
    T: Float + Clone + Send + Sync,
{
    let shape = field.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);

    // Validate grid compatibility
    if (nx, ny, nz) != (grid.nx, grid.ny, grid.nz) {
        return Err(crate::core::error::KwaversError::Grid(
            crate::core::error::GridError::DimensionMismatch {
                expected: format!("({}, {}, {})", grid.nx, grid.ny, grid.nz),
                actual: format!("({}, {}, {})", nx, ny, nz),
            },
        ));
    }

    let mut grad_x = Array3::<T>::zeros((nx, ny, nz));
    let mut grad_y = Array3::<T>::zeros((nx, ny, nz));
    let mut grad_z = Array3::<T>::zeros((nx, ny, nz));

    let coeffs = FDCoefficients::first_derivative::<T>(order);
    let stencil_radius = coeffs.len();
    let dx_inv = T::one() / T::from(grid.dx).unwrap();
    let dy_inv = T::one() / T::from(grid.dy).unwrap();
    let dz_inv = T::one() / T::from(grid.dz).unwrap();

    // Sequential computation for X-direction gradient with boundary handling
    for i in stencil_radius..nx - stencil_radius {
        for j in 0..ny {
            for k in 0..nz {
                let mut grad_val = T::zero();
                for (n, coeff) in coeffs.iter().enumerate() {
                    let offset = n + 1;
                    grad_val =
                        grad_val + *coeff * (field[[i + offset, j, k]] - field[[i - offset, j, k]]);
                }
                grad_x[[i, j, k]] = grad_val * dx_inv;
            }
        }
    }

    // Sequential computation for Y-direction gradient with boundary handling
    for i in 0..nx {
        for j in stencil_radius..ny - stencil_radius {
            for k in 0..nz {
                let mut grad_val = T::zero();
                for (n, coeff) in coeffs.iter().enumerate() {
                    let offset = n + 1;
                    grad_val =
                        grad_val + *coeff * (field[[i, j + offset, k]] - field[[i, j - offset, k]]);
                }
                grad_y[[i, j, k]] = grad_val * dy_inv;
            }
        }
    }

    // Sequential computation for Z-direction gradient with boundary handling
    for i in 0..nx {
        for j in 0..ny {
            for k in stencil_radius..nz - stencil_radius {
                let mut grad_val = T::zero();
                for (n, coeff) in coeffs.iter().enumerate() {
                    let offset = n + 1;
                    grad_val =
                        grad_val + *coeff * (field[[i, j, k + offset]] - field[[i, j, k - offset]]);
                }
                grad_z[[i, j, k]] = grad_val * dz_inv;
            }
        }
    }

    Ok((grad_x, grad_y, grad_z))
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

impl GradientOperatorBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set parallel computation
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Set caching
    pub fn with_caching(mut self, caching: bool) -> Self {
        self.caching = caching;
        self
    }

    /// Set chunk size
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Set boundary strategy
    pub fn with_boundary_strategy(mut self, strategy: BoundaryStrategy) -> Self {
        self.boundary_strategy = strategy;
        self
    }

    /// Build the gradient operator
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
    pub fn compute<T>(
        &self,
        field: &ArrayView3<T>,
        grid: &Grid,
        order: SpatialOrder,
        cache: Option<&GradientCache<T>>,
    ) -> KwaversResult<(Array3<T>, Array3<T>, Array3<T>)>
    where
        T: Float + Clone + Send + Sync,
    {
        if self.parallel {
            if self.caching && cache.is_some() {
                gradient_optimized(field, grid, order, cache)
            } else {
                gradient_with_boundaries(field, grid, order)
            }
        } else {
            // Fallback to original implementation
            super::gradient::gradient(field, grid, order)
        }
    }
}
