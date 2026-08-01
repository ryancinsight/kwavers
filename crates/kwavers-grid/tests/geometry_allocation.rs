use kwavers_alloc_probe::{ThreadScopedAllocator, Window};
use kwavers_grid::geometry::{GeometricDomain, RectangularDomain, SphericalDomain};
use tyche_core::Seed;

#[global_allocator]
static GLOBAL: ThreadScopedAllocator = ThreadScopedAllocator;

#[test]
fn fixed_domains_allocate_only_their_output_matrix() {
    let construction = Window::open();
    let rectangle =
        RectangularDomain::new_3d(-1.0, 2.0, 3.0, 5.0, -4.0, -2.0).expect("valid cuboid");
    let ball = SphericalDomain::new_3d(1.0, -2.0, 3.0, 4.0).expect("valid ball");
    let construction_change = construction.change();
    assert_eq!(construction_change.allocations, 0);
    assert_eq!(construction_change.reallocations, 0);
    drop(construction);

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
    let window = Window::open();
    let output = operation();
    let change = window.change();
    assert_eq!(change.allocations, 1);
    assert_eq!(change.reallocations, 0);
    std::hint::black_box(&output);
}
