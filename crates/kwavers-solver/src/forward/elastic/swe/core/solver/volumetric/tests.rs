use super::{matched_filter_window, snapshot_schedule, SnapshotSchedule, WindowMetric};
use kwavers_core::error::{KwaversError, NumericalError};

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
