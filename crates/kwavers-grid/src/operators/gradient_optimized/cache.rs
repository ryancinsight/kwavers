//! Gradient cache for performance optimization.

use super::super::coefficients::FdAccuracyOrder;
use crate::Grid;
use eunomia::FloatElement;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    OnceLock,
};

/// Lazily converts dimensionless weights to the field precision once per order.
///
/// Four inline slots hold up to four weights each. A shared cache can serve
/// grids with different spacings.
#[derive(Debug)]
pub struct GradientCache<T>
where
    T: FloatElement + Clone + Send + Sync,
{
    /// One immutable coefficient block per accuracy order.
    coefficients: [OnceLock<[T; 4]>; FdAccuracyOrder::COUNT],
    /// Cache hit counter for performance monitoring
    cache_hits: AtomicUsize,
    /// Cache miss counter for performance monitoring
    cache_misses: AtomicUsize,
}

impl<T> GradientCache<T>
where
    T: FloatElement + Clone + Send + Sync,
{
    /// Create an empty coefficient cache. Spacing is read from each computation's grid.
    pub fn new(_grid: &Grid) -> Self {
        Self {
            coefficients: std::array::from_fn(|_| OnceLock::new()),
            cache_hits: AtomicUsize::new(0),
            cache_misses: AtomicUsize::new(0),
        }
    }

    /// Borrow the coefficients for a spatial order, computing them on first use.
    ///
    /// The returned slice borrows inline storage, including on first use.
    pub fn get_coefficients(&self, order: FdAccuracyOrder) -> &[T] {
        let mut initialized = false;
        let weights = self.coefficients[order.index()].get_or_init(|| {
            initialized = true;
            order.first_weights()
        });
        if initialized {
            self.cache_misses.fetch_add(1, Ordering::Relaxed);
        } else {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
        }
        &weights[..order.radius()]
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
    use crate::operators::FDCoefficients;
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
    fn a_hit_borrows_the_same_storage() {
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
        assert_eq!(
            cache.cache_stats(),
            (3 * FdAccuracyOrder::COUNT, FdAccuracyOrder::COUNT)
        );
    }
}
