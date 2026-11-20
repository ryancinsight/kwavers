//! Gradient cache for performance optimization.

use super::super::coefficients::{FDCoefficients, FdAccuracyOrder};
use crate::Grid;
use eunomia::FloatElement;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    OnceLock,
};

/// Gradient cache for performance optimization.
///
/// Coefficients are keyed by the [`FdAccuracyOrder`] discriminant, a zero-sized
/// compile-time value, so the table has one slot per order and no hash or
/// allocation is needed to locate a slot. Each slot is initialized once; a hit
/// returns a borrow into that block instead of cloning a `Vec`.
#[derive(Debug)]
pub struct GradientCache<T>
where
    T: FloatElement + Clone + Send + Sync,
{
    /// One immutable coefficient block per accuracy order.
    coefficients: [OnceLock<Box<[T]>>; FdAccuracyOrder::COUNT],
    /// Cached grid spacing inverses
    pub(super) spacing_inverses: (T, T, T),
    /// Cache hit counter for performance monitoring
    cache_hits: AtomicUsize,
    /// Cache miss counter for performance monitoring
    cache_misses: AtomicUsize,
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
            coefficients: std::array::from_fn(|_| OnceLock::new()),
            spacing_inverses: (
                T::from_f64(1.0) / T::from_f64(grid.dx),
                T::from_f64(1.0) / T::from_f64(grid.dy),
                T::from_f64(1.0) / T::from_f64(grid.dz),
            ),
            cache_hits: AtomicUsize::new(0),
            cache_misses: AtomicUsize::new(0),
        }
    }

    /// Borrow the coefficients for a spatial order, computing them on first use.
    ///
    /// The returned slice lives as long as the cache and is the same allocation
    /// on every call for a given order. Concurrent first callers publish one
    /// block through `OnceLock`; losers borrow it without recomputing.
    pub fn get_coefficients(&self, order: FdAccuracyOrder) -> &[T] {
        let slot = &self.coefficients[order.index()];
        if let Some(coeffs) = slot.get() {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
            return coeffs.as_ref();
        }
        slot.get_or_init(|| {
            self.cache_misses.fetch_add(1, Ordering::Relaxed);
            FDCoefficients::first_derivative::<T>(order).into_boxed_slice()
        })
        .as_ref()
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (
            self.cache_hits.load(Ordering::Relaxed),
            self.cache_misses.load(Ordering::Relaxed),
        )
    }

    /// Reset cache statistics
    pub fn reset_stats(&self) {
        self.cache_hits.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    const ALL_ORDERS: [FdAccuracyOrder; FdAccuracyOrder::COUNT] = [
        FdAccuracyOrder::Second,
        FdAccuracyOrder::Fourth,
        FdAccuracyOrder::Sixth,
        FdAccuracyOrder::Eighth,
    ];

    #[test]
    fn the_order_index_is_dense_and_count_follows_the_last_variant() {
        for (index, order) in ALL_ORDERS.iter().enumerate() {
            assert_eq!(order.index(), index);
        }
        assert_eq!(FdAccuracyOrder::COUNT, ALL_ORDERS.len());
    }

    #[test]
    fn borrowed_coefficients_match_the_source_for_every_order() {
        let cache = GradientCache::<f64>::new(&Grid::default());
        for order in ALL_ORDERS {
            let expected = FDCoefficients::first_derivative::<f64>(order);
            let got = cache.get_coefficients(order);
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn a_hit_borrows_the_same_allocation() {
        let cache = GradientCache::<f64>::new(&Grid::default());
        let first = cache.get_coefficients(FdAccuracyOrder::Sixth);
        let ptr = first.as_ptr();
        let second = cache.get_coefficients(FdAccuracyOrder::Sixth);
        assert!(std::ptr::eq(ptr, second.as_ptr()));
    }

    #[test]
    fn counters_count_one_miss_per_order_and_later_calls_as_hits() {
        let cache = GradientCache::<f64>::new(&Grid::default());
        assert_eq!(cache.cache_stats(), (0, 0));
        for order in ALL_ORDERS {
            let _ = cache.get_coefficients(order);
        }
        assert_eq!(cache.cache_stats(), (0, FdAccuracyOrder::COUNT));
        let _ = cache.get_coefficients(FdAccuracyOrder::Second);
        assert_eq!(cache.cache_stats(), (1, FdAccuracyOrder::COUNT));
        cache.reset_stats();
        assert_eq!(cache.cache_stats(), (0, 0));
    }

    #[test]
    fn concurrent_first_calls_publish_one_block_per_order() {
        let cache = Arc::new(GradientCache::<f64>::new(&Grid::default()));
        let expected = ALL_ORDERS.map(FDCoefficients::first_derivative::<f64>);
        let threads: Vec<_> = (0..4)
            .map(|_| {
                let cache = Arc::clone(&cache);
                std::thread::spawn(move || {
                    ALL_ORDERS.map(|order| cache.get_coefficients(order).to_vec())
                })
            })
            .collect();
        for thread in threads {
            let got = thread.join().expect("worker thread must not panic");
            for (got, expected) in got.iter().zip(expected.iter()) {
                assert_eq!(got, expected);
            }
        }
        assert_eq!(cache.cache_stats().1, FdAccuracyOrder::COUNT);
    }
}
