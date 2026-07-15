//! Gradient cache for performance optimization.

use super::super::coefficients::{FDCoefficients, FdAccuracyOrder};
use crate::Grid;
use eunomia::FloatElement;
use std::sync::{Arc, RwLock};

/// Gradient cache for performance optimization
#[derive(Debug)]
pub struct GradientCache<T>
where
    T: FloatElement + Clone + Send + Sync,
{
    /// Cached coefficients for different spatial orders
    pub(super) coefficients_cache: RwLock<Vec<Vec<T>>>,
    /// Cached grid spacing inverses
    pub(super) spacing_inverses: (T, T, T),
    /// Cache hit counter for performance monitoring
    cache_hits: Arc<RwLock<usize>>,
    /// Cache miss counter for performance monitoring
    cache_misses: Arc<RwLock<usize>>,
}

impl<T> GradientCache<T>
where
    T: FloatElement + Clone + Send + Sync,
{
    /// Create a new gradient cache
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn new(grid: &Grid) -> Self {
        Self {
            coefficients_cache: RwLock::new(Vec::new()),
            spacing_inverses: (
                T::from_f64(1.0) / T::from_f64(grid.dx),
                T::from_f64(1.0) / T::from_f64(grid.dy),
                T::from_f64(1.0) / T::from_f64(grid.dz),
            ),
            cache_hits: Arc::new(RwLock::new(0)),
            cache_misses: Arc::new(RwLock::new(0)),
        }
    }

    /// Get or create coefficients for a given spatial order
    pub fn get_coefficients(&self, order: FdAccuracyOrder) -> Vec<T> {
        let mut cache = self
            .coefficients_cache
            .write()
            .unwrap_or_else(|e| e.into_inner());

        if let Some(coeffs) = cache.get(order as usize) {
            *self.cache_hits.write().unwrap_or_else(|e| e.into_inner()) += 1;
            return coeffs.clone();
        }

        *self.cache_misses.write().unwrap_or_else(|e| e.into_inner()) += 1;
        let coeffs = FDCoefficients::first_derivative::<T>(order);

        if cache.len() <= order as usize {
            cache.resize(order as usize + 1, Vec::new());
        }

        cache[order as usize] = coeffs.clone();
        coeffs
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        let hits = *self.cache_hits.read().unwrap_or_else(|e| e.into_inner());
        let misses = *self.cache_misses.read().unwrap_or_else(|e| e.into_inner());
        (hits, misses)
    }

    /// Reset cache statistics
    pub fn reset_stats(&self) {
        *self.cache_hits.write().unwrap_or_else(|e| e.into_inner()) = 0;
        *self.cache_misses.write().unwrap_or_else(|e| e.into_inner()) = 0;
    }
}
