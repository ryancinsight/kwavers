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
fn first_use_and_every_hit_borrow_inline_weights_without_allocating() {
    let cache = GradientCache::<f64>::new(&Grid::default());

    let (initial, init_change) = measure(|| cache.get_coefficients(FdAccuracyOrder::Fourth));
    assert_eq!(init_change.allocations, 0);
    assert_eq!(init_change.reallocations, 0);

    for _ in 0..8 {
        let (borrowed, hit_change) = measure(|| cache.get_coefficients(FdAccuracyOrder::Fourth));
        assert_eq!(hit_change.allocations, 0);
        assert_eq!(hit_change.reallocations, 0);
        std::hint::black_box(borrowed);
    }

    std::hint::black_box(initial);
}

#[test]
fn reusable_serial_gradient_allocates_nothing_on_cold_and_warm_calls() {
    use kwavers_grid::operators::GradientOperatorBuilder;
    use leto::Array3;
    let grid = Grid::new(11, 9, 10, 0.25, 0.5, 2.0).unwrap();
    let input = Array3::<f64>::ones([11, 9, 10]);
    let cache = GradientCache::new(&grid);
    let op = GradientOperatorBuilder::new().with_parallel(false).build();
    let order = FdAccuracyOrder::Eighth;
    let [mut x, mut y, mut z] = std::array::from_fn(|_| vec![f64::NAN; input.len()]);
    for selected in [None, Some(&cache), Some(&cache)] {
        let (result, counts) = measure(|| {
            op.compute_into(
                &input.view(),
                &grid,
                order,
                selected,
                [&mut x, &mut y, &mut z],
            )
        });
        result.unwrap();
        assert_eq!((counts.allocations, counts.reallocations), (0, 0));
    }
    let (ex, ey, ez) = op.compute(&input.view(), &grid, order, None).unwrap();
    for (expected, actual) in [(ex, x), (ey, y), (ez, z)] {
        assert!(expected.iter().eq(actual.iter()));
    }
}
