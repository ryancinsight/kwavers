use kwavers_alloc_probe::{Change, ThreadScopedAllocator, Window};
#[cfg(feature = "pinn")]
use kwavers_grid::geometry::RectangularDomain;
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_solver::forward::elastic::swe::{
    ArrivalDetection, ElasticBodyForceConfig, ElasticDisplacementSnapshot, ElasticWaveConfig,
    ElasticWaveField, ElasticWaveSolver, VolumetricWaveConfig, WaveFrontTracker,
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

fn volumetric_tracker_solver() -> ElasticWaveSolver {
    let grid = Grid::new(6, 6, 6, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid tracker grid");
    let medium = HomogeneousMedium::new(1_000.0, 1_500.0, 0.5, 1.0, &grid);
    let config = ElasticWaveConfig {
        time_step: 1.0e-6,
        pml_thickness: 1,
        ..ElasticWaveConfig::default()
    };
    let mut solver = ElasticWaveSolver::new(&grid, &medium, config).expect("valid tracker solver");
    solver.set_volumetric_config(VolumetricWaveConfig {
        arrival_detection: ArrivalDetection::EnergyThreshold { threshold: 0.0 },
        duration_s: 4.0e-6,
        max_snapshots: 5,
        ..VolumetricWaveConfig::default()
    });
    solver
}

fn tracker_body_force() -> ElasticBodyForceConfig {
    ElasticBodyForceConfig::GaussianImpulse {
        center_m: [2.5e-3; 3],
        sigma_m: [1.0e-3; 3],
        direction: [1.0, -0.5, 0.25],
        t0_s: 0.0,
        sigma_t_s: 1.0e-6,
        impulse_n_per_m3_s: 1.0e6,
    }
}

fn measure_full_tracker(
    solver: &ElasticWaveSolver,
    body_force: &ElasticBodyForceConfig,
) -> (WaveFrontTracker, Change) {
    let window = Window::open();
    let (_, tracker) = solver
        .propagate_volumetric_waves_with_body_forces(std::slice::from_ref(body_force), &[0.0], &[])
        .expect("valid full-history tracker propagation");
    let change = window.change();
    drop(window);
    (tracker, change)
}

fn measure_compact_tracker(
    solver: &ElasticWaveSolver,
    body_force: &ElasticBodyForceConfig,
) -> (WaveFrontTracker, Change) {
    let window = Window::open();
    let tracker = solver
        .track_volumetric_waves_with_body_forces(std::slice::from_ref(body_force), &[0.0])
        .expect("valid compact tracker propagation");
    let change = window.change();
    drop(window);
    (tracker, change)
}

fn assert_tracker_bits_eq(actual: &WaveFrontTracker, expected: &WaveFrontTracker) {
    assert_eq!(actual.arrival_times.shape(), expected.arrival_times.shape());
    assert_eq!(actual.amplitudes.shape(), expected.amplitudes.shape());
    assert_eq!(
        actual.tracking_quality.shape(),
        expected.tracking_quality.shape()
    );
    assert!(actual
        .arrival_times
        .iter()
        .zip(&expected.arrival_times)
        .all(|(&actual, &expected)| actual.to_bits() == expected.to_bits()));
    assert!(actual
        .amplitudes
        .iter()
        .zip(&expected.amplitudes)
        .all(|(&actual, &expected)| actual.to_bits() == expected.to_bits()));
    assert!(actual
        .tracking_quality
        .iter()
        .zip(&expected.tracking_quality)
        .all(|(&actual, &expected)| actual.to_bits() == expected.to_bits()));
}

#[test]
fn volumetric_tracker_omits_full_field_snapshot_allocations() {
    const SNAPSHOTS: u64 = 5;
    const FULL_FIELD_ARRAYS: u64 = 6;
    const COMPACT_HISTORY_ARRAYS: u64 = 2;

    let solver = volumetric_tracker_solver();
    let body_force = tracker_body_force();

    drop(measure_full_tracker(&solver, &body_force).0);
    drop(measure_compact_tracker(&solver, &body_force).0);

    let (full_tracker, full_change) = measure_full_tracker(&solver, &body_force);
    let (compact_tracker, compact_change) = measure_compact_tracker(&solver, &body_force);

    assert_eq!(full_change.reallocations, 0);
    assert_eq!(compact_change.reallocations, 0);
    assert_eq!(
        full_change
            .allocations
            .checked_sub(compact_change.allocations),
        Some(SNAPSHOTS * FULL_FIELD_ARRAYS + 1 - COMPACT_HISTORY_ARRAYS),
        "the compact recorder must replace six arrays per snapshot and one header with two retained arrays"
    );
    assert_tracker_bits_eq(&compact_tracker, &full_tracker);
    std::hint::black_box((full_tracker, compact_tracker));
}
