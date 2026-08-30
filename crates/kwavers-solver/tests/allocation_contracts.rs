use kwavers_alloc_probe::{Change, ThreadScopedAllocator, Window};
#[cfg(feature = "pinn")]
use kwavers_grid::geometry::RectangularDomain;
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_solver::forward::elastic::swe::{
    ElasticDisplacementSnapshot, ElasticWaveConfig, ElasticWaveField, ElasticWaveSolver,
};
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

fn swe_solver(steps: usize) -> ElasticWaveSolver {
    let grid = Grid::new(2, 2, 2, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid grid");
    let medium = HomogeneousMedium::new(1_000.0, 1_500.0, 0.5, 1.0, &grid);
    let config = ElasticWaveConfig {
        time_step: 1.0e-6,
        simulation_time: steps as f64 * 1.0e-6,
        save_every: 1,
        pml_thickness: 1,
        ..ElasticWaveConfig::default()
    };
    ElasticWaveSolver::new(&grid, &medium, config).expect("valid solver")
}

fn measure_full_history(
    solver: &ElasticWaveSolver,
    displacement: &Array3<f64>,
) -> (Vec<ElasticWaveField>, Change) {
    let window = Window::open();
    let history = solver
        .propagate_waves(displacement)
        .expect("valid full-field propagation");
    let change = window.change();
    drop(window);
    (history, change)
}

fn measure_displacement_history(
    solver: &ElasticWaveSolver,
) -> (Vec<ElasticDisplacementSnapshot>, Change) {
    let window = Window::open();
    let history = solver
        .propagate_displacement_history_with_body_force_only_override(None)
        .expect("valid displacement propagation");
    let change = window.change();
    drop(window);
    (history, change)
}

#[test]
fn swe_displacement_history_avoids_velocity_snapshot_allocations() {
    const STEPS: usize = 17;
    const EXPECTED_SNAPSHOTS: usize = STEPS + 1;
    let solver = swe_solver(STEPS);
    let displacement = Array3::zeros((2, 2, 2));

    drop(
        solver
            .propagate_waves(&displacement)
            .expect("warm full-field propagation"),
    );
    drop(
        solver
            .propagate_displacement_history_with_body_force_only_override(None)
            .expect("warm displacement propagation"),
    );

    let (full_history, full_change) = measure_full_history(&solver, &displacement);
    let (displacement_history, displacement_change) = measure_displacement_history(&solver);

    assert_eq!(full_history.len(), EXPECTED_SNAPSHOTS);
    assert_eq!(displacement_history.len(), EXPECTED_SNAPSHOTS);
    assert!(full_history.capacity() >= EXPECTED_SNAPSHOTS);
    assert!(displacement_history.capacity() >= EXPECTED_SNAPSHOTS);
    assert_eq!(full_change.reallocations, 0);
    assert_eq!(displacement_change.reallocations, 0);
    assert_eq!(
        full_change
            .allocations
            .checked_sub(displacement_change.allocations),
        Some((EXPECTED_SNAPSHOTS * 3) as u64),
        "the projected history must omit exactly three array allocations per snapshot"
    );
    std::hint::black_box((full_history, displacement_history));
}
