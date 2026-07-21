use std::alloc::System;

use kwavers_grid::geometry::RectangularDomain;
use kwavers_solver::inverse::pinn::{CollocationSampler, CollocationSamplingStrategy};
use stats_alloc::{Region, StatsAlloc, INSTRUMENTED_SYSTEM};
use tyche_core::Seed;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

#[test]
fn tyche_designs_allocate_only_their_output_matrix() {
    let domain = RectangularDomain::new_3d(-1.0, 2.0, 3.0, 5.0, -4.0, -2.0).expect("valid cuboid");
    for strategy in [
        CollocationSamplingStrategy::LatinHypercube,
        CollocationSamplingStrategy::Sobol,
    ] {
        let sampler = CollocationSampler::new(domain, strategy, Seed::new(17));
        let region = Region::new(GLOBAL);
        let points = sampler
            .sample_interior(257)
            .expect("addressable collocation output");
        let stats = region.change();
        assert_eq!(stats.allocations, 1);
        assert_eq!(stats.reallocations, 0);
        std::hint::black_box(points);
    }
}
