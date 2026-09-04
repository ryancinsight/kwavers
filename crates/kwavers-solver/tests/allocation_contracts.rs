use kwavers_alloc_probe::{Change, ThreadScopedAllocator, Window};
use kwavers_core::error::{KwaversError, SystemError};
#[cfg(feature = "pinn")]
use kwavers_grid::geometry::RectangularDomain;
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_solver::forward::elastic::swe::{
    ArrivalDetection, ElasticBodyForceConfig, ElasticDisplacementSnapshot, ElasticWaveConfig,
    ElasticWaveField, ElasticWaveSolver, VolumetricWaveConfig, WaveFrontTracker,
};
use kwavers_solver::forward::viscoacoustic::ViscoacousticMemorySolver;
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

fn viscoacoustic_sensor_solver() -> (ViscoacousticMemorySolver, [usize; 2], Array3<f64>) {
    let initial_pressure =
        Array3::from_shape_fn((8, 1, 1), |[i, _, _]| if i == 1 { 1.0 } else { 0.0 });
    let mut solver = ViscoacousticMemorySolver::new_1d(8, 1.0e-4, 1.0e-8, 1_000.0, 2.25e9, &[])
        .expect("valid viscoacoustic solver");
    solver.step();
    solver
        .set_pressure(&initial_pressure)
        .expect("matching pressure shape");
    let near = solver
        .add_pressure_sensor((2, 0, 0))
        .expect("valid near sensor index");
    let far = solver
        .add_pressure_sensor((5, 0, 0))
        .expect("valid far sensor index");
    (solver, [near, far], initial_pressure)
}

fn step_viscoacoustic(solver: &mut ViscoacousticMemorySolver, samples: usize) {
    for _ in 0..samples {
        solver.step();
    }
}

fn assert_sensor_bits_eq(
    actual: &ViscoacousticMemorySolver,
    actual_sensors: [usize; 2],
    expected: &ViscoacousticMemorySolver,
    expected_sensors: [usize; 2],
) {
    for (actual_sensor, expected_sensor) in actual_sensors.into_iter().zip(expected_sensors) {
        let actual_trace = actual.sensor_trace(actual_sensor);
        let expected_trace = expected.sensor_trace(expected_sensor);
        assert_eq!(actual_trace.len(), expected_trace.len());
        assert!(actual_trace
            .iter()
            .zip(expected_trace)
            .all(|(&actual, &expected)| actual.to_bits() == expected.to_bits()));
    }
}

#[test]
fn reserved_viscoacoustic_sensor_traces_do_not_allocate_during_stepping() {
    const SAMPLES: usize = 65;
    let (mut reference, reference_sensors, _) = viscoacoustic_sensor_solver();
    let (mut reserved, reserved_sensors, initial_pressure) = viscoacoustic_sensor_solver();
    reserved
        .reserve_sensor_samples(SAMPLES)
        .expect("addressable sensor history");

    let window = Window::open();
    step_viscoacoustic(&mut reserved, SAMPLES);
    let first_change = window.change();
    drop(window);

    let window = Window::open();
    step_viscoacoustic(&mut reference, SAMPLES);
    let unreserved_change = window.change();
    drop(window);

    assert_eq!(first_change.allocations, 0);
    assert_eq!(first_change.reallocations, 0);
    assert_eq!(unreserved_change.allocations, 2);
    assert_eq!(unreserved_change.reallocations, 10);
    assert_sensor_bits_eq(&reserved, reserved_sensors, &reference, reference_sensors);
    let first_traces = reserved_sensors.map(|sensor| reserved.sensor_trace(sensor).to_vec());

    reserved
        .set_pressure(&initial_pressure)
        .expect("matching pressure shape");
    let window = Window::open();
    step_viscoacoustic(&mut reserved, SAMPLES);
    let repeated_change = window.change();
    drop(window);

    assert_eq!(repeated_change.allocations, 0);
    assert_eq!(repeated_change.reallocations, 0);
    for (sensor, expected) in reserved_sensors.into_iter().zip(first_traces) {
        let actual = reserved.sensor_trace(sensor);
        assert_eq!(actual.len(), expected.len());
        assert!(actual
            .iter()
            .zip(expected)
            .all(|(&actual, expected)| actual.to_bits() == expected.to_bits()));
    }
}

/// Warm construction must measure exactly the KW-VISCOACOUSTIC-DIMENSIONAL-STATE
/// acceptance oracle: 10/12/15 allocations for the standard 1-D/2-D/3-D
/// constructor and the retained-byte reductions of the inactive-axis storage
/// layout, with warm stepping allocation-free.
///
/// The byte figures are absolute (the probe counts requested bytes minus
/// released bytes over the construction window, and a warm window sees no
/// intermediate releases). They decode exactly: the shape-independent part is
/// 7 state/scratch arrays + cbuf + 3 modulus fields = 393,216 B at 4,096
/// cells, plus per-shape wavenumber vectors (1-D: 32,784 B, 2-D: 1,032 B,
/// 3-D: 384 B) and one cached-FFT event (warm ⇒ none). The old layout
/// measured 426,000/394,248/393,600 B; the candidate removes the inactive
/// arrays and wavenumber vectors — 98,320 B (vy+vz+gz+ky+kz) in 1-D and
/// 65,544 B (vz+gz+kz) in 2-D — leaving exactly 327,680/328,704 B, while the
/// full 3-D control retains its 15 events and 393,600 B.
#[test]
fn warm_construction_matches_inactive_axis_storage_oracle() {
    /// (shape, expected warm allocations, expected retained bytes).
    const CASES: [([usize; 3], u64, u64); 3] = [
        ([4096, 1, 1], 10, 327_680),
        ([64, 64, 1], 12, 328_704),
        ([16, 16, 16], 15, 393_600),
    ];
    const DT: f64 = 1.0e-8;

    for &(shape, allocations, retained) in &CASES {
        let [nx, ny, nz] = shape;
        let build = || {
            ViscoacousticMemorySolver::new(
                nx,
                ny,
                nz,
                1.0e-4,
                1.0e-4,
                1.0e-4,
                DT,
                1_000.0,
                2.25e9,
                &[],
            )
            .expect("valid oracle solver parameters")
        };

        // Warm every shape-keyed cache (FFT plan, wavenumber-independent
        // allocation paths) so the window isolates the constructor itself.
        drop(build());

        let window = Window::open();
        let mut solver = build();
        let change = window.change();
        drop(window);

        assert_eq!(
            change.allocations, allocations,
            "warm construction allocation count at {shape:?}"
        );
        assert_eq!(change.reallocations, 0, "warm construction at {shape:?}");
        assert_eq!(
            change.bytes_retained(),
            retained,
            "warm construction retained bytes at {shape:?}"
        );

        let initial = Array3::<f64>::zeros(shape);
        solver.set_pressure(&initial).expect("matching shape");

        // Warm the step path itself: the per-size FFT work buffers allocate
        // lazily on first transform, so the measurement window must open on
        // an already-exercised step ("warm stepping" in the oracle).
        for _ in 0..4 {
            solver.step();
        }

        let window = Window::open();
        for _ in 0..64 {
            solver.step();
        }
        let stepping = window.change();
        drop(window);

        assert_eq!(
            stepping.allocations, 0,
            "warm stepping must stay allocation-free at {shape:?}"
        );
        assert_eq!(
            stepping.reallocations, 0,
            "warm stepping must stay reallocation-free at {shape:?}"
        );
        std::hint::black_box(solver.pressure());
    }
}

#[test]
fn viscoacoustic_sensor_reservation_rejects_unrepresentable_history() {
    let (mut solver, sensors, _) = viscoacoustic_sensor_solver();
    step_viscoacoustic(&mut solver, 3);
    let traces_before = sensors.map(|sensor| solver.sensor_trace(sensor).to_vec());
    let error = solver
        .reserve_sensor_samples(usize::MAX)
        .expect_err("unrepresentable sensor history must fail before stepping");

    match error {
        KwaversError::System(SystemError::MemoryAllocation {
            requested_bytes,
            reason,
        }) => {
            assert_eq!(requested_bytes, usize::MAX);
            assert!(reason.starts_with("sensor 0 trace reservation failed:"));
        }
        other => panic!("unexpected reservation error: {other:?}"),
    }
    for (sensor, expected) in sensors.into_iter().zip(traces_before) {
        let actual = solver.sensor_trace(sensor);
        assert_eq!(actual.len(), expected.len());
        assert!(actual
            .iter()
            .zip(expected)
            .all(|(&actual, expected)| actual.to_bits() == expected.to_bits()));
    }
}
