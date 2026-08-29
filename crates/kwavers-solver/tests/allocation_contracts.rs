use kwavers_alloc_probe::{ThreadScopedAllocator, Window};
#[cfg(feature = "pinn")]
use kwavers_grid::geometry::RectangularDomain;
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_solver::forward::elastic::swe::{ElasticWaveConfig, ElasticWaveSolver};
#[cfg(feature = "pinn")]
use kwavers_solver::inverse::pinn::{CollocationSampler, CollocationSamplingStrategy};
use leto::Array3;
#[cfg(feature = "pinn")]
use tyche_core::Seed;

#[global_allocator]
static GLOBAL: ThreadScopedAllocator = ThreadScopedAllocator;

#[cfg(feature = "pinn")]
#[test]
fn tyche_designs_allocate_only_their_output_matrix() {
    let domain = RectangularDomain::new_3d(-1.0, 2.0, 3.0, 5.0, -4.0, -2.0).expect("valid cuboid");
    for strategy in [
        CollocationSamplingStrategy::LatinHypercube,
        CollocationSamplingStrategy::Sobol,
    ] {
        let sampler = CollocationSampler::new(domain, strategy, Seed::new(17));
        let window = Window::open();
        let points = sampler
            .sample_interior(257)
            .expect("addressable collocation output");
        let change = window.change();
        assert_eq!(change.allocations, 1);
        assert_eq!(change.reallocations, 0);
        std::hint::black_box(points);
    }
}

#[test]
fn swe_history_reserves_snapshot_headers_once() {
    const STEPS: usize = 17;
    const EXPECTED_SNAPSHOTS: usize = 18;
    let grid = Grid::new(2, 2, 2, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid grid");
    let medium = HomogeneousMedium::new(1_000.0, 1_500.0, 0.5, 1.0, &grid);
    let config = ElasticWaveConfig {
        time_step: 1.0e-6,
        simulation_time: STEPS as f64 * 1.0e-6,
        save_every: 1,
        pml_thickness: 1,
        ..ElasticWaveConfig::default()
    };
    let solver = ElasticWaveSolver::new(&grid, &medium, config).expect("valid solver");
    let displacement = Array3::zeros((2, 2, 2));

    let window = Window::open();
    let history = solver
        .propagate_waves(&displacement)
        .expect("valid propagation");
    let change = window.change();

    assert_eq!(history.len(), EXPECTED_SNAPSHOTS);
    assert!(history.capacity() >= EXPECTED_SNAPSHOTS);
    assert_eq!(change.reallocations, 0);
    std::hint::black_box(history);
}
