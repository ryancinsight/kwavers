//! Allocation oracle for the gradient coefficient cache.
//!
//! A cache hit borrows an initialized block and must not allocate; the previous
//! `Vec<Vec<T>>` cache cloned a `Vec<T>` on every call.

use kwavers_grid::operators::{FdAccuracyOrder, GradientCache};
use kwavers_grid::Grid;
use mnemosyne::counting::{measure, CountingAllocator};
use mnemosyne::Mnemosyne;

#[global_allocator]
static GLOBAL: CountingAllocator<Mnemosyne> = CountingAllocator::new(Mnemosyne);

#[test]
fn first_use_allocates_one_block_and_every_hit_allocates_nothing() {
    let cache = GradientCache::<f64>::new(&Grid::default());

    let (initial, init_change) = measure(|| cache.get_coefficients(FdAccuracyOrder::Fourth));
    assert_eq!(init_change.allocations, 1);
    assert_eq!(init_change.reallocations, 0);

    for _ in 0..8 {
        let (borrowed, hit_change) = measure(|| cache.get_coefficients(FdAccuracyOrder::Fourth));
        assert_eq!(hit_change.allocations, 0);
        assert_eq!(hit_change.reallocations, 0);
        std::hint::black_box(borrowed);
    }

    std::hint::black_box(initial);
}
