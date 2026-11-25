//! Repeated higher-order applications borrow both volumes and the stencil weights.
use kwavers_grid::{
    operators::{
        laplacian::{LaplacianBoundary, LaplacianConfig, LaplacianOperator},
        FdAccuracyOrder,
    },
    Grid,
};
use leto::Array3;
use mnemosyne::counting::{measure, CountingAllocator};
use mnemosyne::Mnemosyne;

#[global_allocator]
static GLOBAL: CountingAllocator<Mnemosyne> = CountingAllocator::new(Mnemosyne);

#[test]
fn higher_order_apply_mut_has_no_heap_traffic() {
    let grid = Grid::new(12, 13, 14, 0.1, 0.25, 0.5).unwrap();
    let input = Array3::from_elem([12, 13, 14], 1.0);
    let mut output = Array3::zeros([12, 13, 14]);
    for boundary in [
        LaplacianBoundary::Dirichlet,
        LaplacianBoundary::Neumann,
        LaplacianBoundary::Periodic,
    ] {
        for order in [
            FdAccuracyOrder::Fourth,
            FdAccuracyOrder::Sixth,
            FdAccuracyOrder::Eighth,
        ] {
            let op = LaplacianOperator::new(&grid, LaplacianConfig { order, boundary });
            let (result, counts) = measure(|| op.apply_mut(input.view(), output.view_mut()));
            result.unwrap();
            assert_eq!(
                (counts.allocations, counts.reallocations),
                (0, 0),
                "{order:?} {boundary:?}"
            );
            std::hint::black_box(&output);
        }
    }
}
