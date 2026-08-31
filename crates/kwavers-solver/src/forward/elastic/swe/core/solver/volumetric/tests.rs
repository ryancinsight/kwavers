use super::tracking::{compact_history_bytes, matched_filter_window, WindowMetric};
use super::{snapshot_schedule, SnapshotSchedule};
use crate::forward::elastic::swe::{
    ArrivalDetection, ElasticBodyForceConfig, ElasticWaveConfig, ElasticWaveSolver,
    VolumetricWaveConfig, WaveFrontTracker,
};
use kwavers_core::error::{KwaversError, NumericalError};
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;

#[test]
fn snapshot_schedule_respects_the_configured_bound() {
    assert_eq!(
        snapshot_schedule(464, 256).expect("valid snapshot bound"),
        SnapshotSchedule {
            stride: 2,
            capacity: 233,
        }
    );
    assert_eq!(
        snapshot_schedule(100, 256).expect("valid snapshot bound"),
        SnapshotSchedule {
            stride: 1,
            capacity: 101,
        }
    );
    assert_eq!(
        snapshot_schedule(100, 2).expect("valid snapshot bound"),
        SnapshotSchedule {
            stride: 100,
            capacity: 2,
        }
    );
}

#[test]
fn snapshot_schedule_capacity_matches_retained_states() {
    for steps in 0..=1_024 {
        for configured_max in 2..=512 {
            let schedule = snapshot_schedule(steps, configured_max)
                .expect("bounds at or above two must produce a schedule");
            let retained = 1 + steps / schedule.stride + usize::from(steps % schedule.stride != 0);

            assert_eq!(schedule.capacity, retained);
            assert!(retained <= configured_max);
        }
    }
}

#[test]
fn snapshot_schedule_rejects_fewer_than_two_states() {
    for configured_max in [0, 1] {
        let error = snapshot_schedule(100, configured_max)
            .expect_err("initial and final states require a capacity of at least two");
        match error {
            KwaversError::Numerical(NumericalError::InvalidOperation(message)) => {
                assert_eq!(message, "Volumetric max_snapshots must be at least two");
            }
            other => panic!("expected InvalidOperation, got {other}"),
        }
    }
}

#[test]
fn matched_filter_preserves_the_original_window_index() {
    let series = [0.0, 0.0, 2.0, 3.0];
    let derivative = [0.0, 0.0, 2.0, 1.0];
    let mut metrics = [WindowMetric::default(); 4];

    let (start, metric) = matched_filter_window(&series, &derivative, &[1.0], 0.5, &mut metrics)
        .expect("two windows exceed the correlation floor");

    assert_eq!(start, 2);
    assert_eq!(metric.amplitude, 2.0);
    assert_eq!(metric.quality, 1.0);
}

#[test]
fn compact_history_meets_the_coverage_memory_bound() {
    const ELIGIBLE_VOXELS: usize = 40 * 40 * 20;
    const MAX_SNAPSHOTS: usize = 256;
    const MEMORY_BOUND: usize = 80 * 1024 * 1024;

    let bytes = compact_history_bytes(ELIGIBLE_VOXELS, MAX_SNAPSHOTS)
        .expect("the coverage history layout must be addressable");

    assert_eq!(bytes, 65_538_048);
    assert!(bytes < MEMORY_BOUND);
    assert_eq!(compact_history_bytes(usize::MAX, 2), None);
}

#[test]
fn tracker_only_propagation_matches_positive_threshold_crossing() {
    const THRESHOLD: f64 = 1.0e-30;
    let tracker = assert_tracker_paths_match(ArrivalDetection::EnergyThreshold {
        threshold: THRESHOLD,
    });
    let mut detected = tracker
        .amplitudes
        .iter()
        .zip(&tracker.tracking_quality)
        .filter(|&(&amplitude, _)| amplitude > 0.0);

    assert!(detected.any(|(&amplitude, &quality)| {
        amplitude >= THRESHOLD && quality.to_bits() == 1.0_f64.to_bits()
    }));
}

#[test]
fn tracker_only_propagation_matches_threshold_fallback() {
    let tracker = assert_tracker_paths_match(ArrivalDetection::EnergyThreshold {
        threshold: f64::MAX,
    });
    let mut detected = tracker
        .amplitudes
        .iter()
        .zip(&tracker.tracking_quality)
        .filter(|&(&amplitude, _)| amplitude > 0.0);

    assert!(detected.any(|(&amplitude, &quality)| amplitude < f64::MAX && quality < 1.0));
}

#[test]
fn tracker_only_propagation_matches_full_history_matched_filter() {
    let tracker = assert_tracker_paths_match(ArrivalDetection::MatchedFilter {
        template: vec![0.0, 1.0, 0.0],
        min_corr: 1.0e-20,
    });
    assert!(tracker.arrival_times.iter().any(|value| value.is_finite()));
}

#[test]
fn tracker_only_propagation_rejects_mismatched_force_times() {
    let solver = tracker_test_solver(ArrivalDetection::EnergyThreshold { threshold: 0.0 });
    let compact_error = solver
        .track_volumetric_waves_with_body_forces(&[], &[0.0])
        .expect_err("force timing must be one-to-one");
    let full_error = solver
        .propagate_volumetric_waves_with_body_forces(&[], &[0.0], &[])
        .expect_err("full-history force timing must be one-to-one");

    assert_force_time_mismatch(compact_error);
    assert_force_time_mismatch(full_error);
}

fn assert_force_time_mismatch(error: KwaversError) {
    match error {
        KwaversError::Numerical(NumericalError::InvalidOperation(message)) => {
            assert_eq!(
                message,
                "body_forces and push_times must have the same length"
            );
        }
        other => panic!("expected InvalidOperation, got {other}"),
    }
}

fn assert_tracker_paths_match(arrival_detection: ArrivalDetection) -> WaveFrontTracker {
    let full_solver = tracker_test_solver(arrival_detection.clone());
    let compact_solver = tracker_test_solver(arrival_detection);
    let body_force = ElasticBodyForceConfig::GaussianImpulse {
        center_m: [3.5e-3, 3.5e-3, 3.5e-3],
        sigma_m: [1.0e-3; 3],
        direction: [1.0, -0.5, 0.25],
        t0_s: 0.0,
        sigma_t_s: 1.0e-6,
        impulse_n_per_m3_s: 1.0e6,
    };
    let body_forces = [body_force];
    let push_times = [1.0e-6];

    let (_, full_tracker) = full_solver
        .propagate_volumetric_waves_with_body_forces(&body_forces, &push_times, &[])
        .expect("full-history volumetric propagation");
    let compact_tracker = compact_solver
        .track_volumetric_waves_with_body_forces(&body_forces, &push_times)
        .expect("tracker-only volumetric propagation");

    assert_tracker_bits_eq(&compact_tracker, &full_tracker);
    compact_tracker
}

fn tracker_test_solver(arrival_detection: ArrivalDetection) -> ElasticWaveSolver {
    let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid tracker grid");
    let medium = HomogeneousMedium::new(1_000.0, 1_500.0, 0.5, 1.0, &grid);
    let config = ElasticWaveConfig {
        time_step: 1.0e-6,
        pml_thickness: 1,
        ..ElasticWaveConfig::default()
    };
    let mut solver = ElasticWaveSolver::new(&grid, &medium, config).expect("valid tracker solver");
    solver.set_volumetric_config(VolumetricWaveConfig {
        arrival_detection,
        tracking_decimation: [1, 2, 1],
        duration_s: 4.0e-6,
        max_snapshots: 5,
        ..VolumetricWaveConfig::default()
    });
    solver
}

fn assert_tracker_bits_eq(actual: &WaveFrontTracker, expected: &WaveFrontTracker) {
    assert_eq!(actual.arrival_times.shape(), expected.arrival_times.shape());
    assert_eq!(actual.amplitudes.shape(), expected.amplitudes.shape());
    assert_eq!(
        actual.tracking_quality.shape(),
        expected.tracking_quality.shape()
    );
    for (actual, expected) in actual.arrival_times.iter().zip(&expected.arrival_times) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    for (actual, expected) in actual.amplitudes.iter().zip(&expected.amplitudes) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    for (actual, expected) in actual
        .tracking_quality
        .iter()
        .zip(&expected.tracking_quality)
    {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}
