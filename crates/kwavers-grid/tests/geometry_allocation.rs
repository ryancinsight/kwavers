use std::alloc::System;

use kwavers_grid::geometry::{GeometricDomain, RectangularDomain, SphericalDomain};
use stats_alloc::{Region, StatsAlloc, INSTRUMENTED_SYSTEM};
use tyche_core::Seed;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

#[test]
fn fixed_domains_allocate_only_their_output_matrix() {
    let construction = Region::new(GLOBAL);
    let rectangle =
        RectangularDomain::new_3d(-1.0, 2.0, 3.0, 5.0, -4.0, -2.0).expect("valid cuboid");
    let ball = SphericalDomain::new_3d(1.0, -2.0, 3.0, 4.0).expect("valid ball");
    let construction_stats = construction.change();
    assert_eq!(construction_stats.allocations, 0);
    assert_eq!(construction_stats.reallocations, 0);

    assert_one_output_allocation(|| {
        rectangle
            .sample_interior(257, Seed::new(17))
            .expect("addressable rectangle output")
    });
    assert_one_output_allocation(|| {
        rectangle
            .sample_boundary(257, Seed::new(17))
            .expect("addressable rectangle boundary output")
    });
    assert_one_output_allocation(|| {
        ball.sample_interior(257, Seed::new(17))
            .expect("addressable ball output")
    });
    assert_one_output_allocation(|| {
        ball.sample_boundary(257, Seed::new(17))
            .expect("addressable ball boundary output")
    });
}

fn assert_one_output_allocation<T>(operation: impl FnOnce() -> T) {
    let region = Region::new(GLOBAL);
    let output = operation();
    let stats = region.change();
    assert_eq!(stats.allocations, 1);
    assert_eq!(stats.reallocations, 0);
    std::hint::black_box(&output);
}
