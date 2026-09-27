use super::Recorder;
use crate::recorder::traits::RecorderTrait;
use crate::recorder::{RecorderConfig, RecordingState};
use crate::{GridPoint, GridSensorSet};
use kwavers_core::time::Time;
use kwavers_field::indices::{LIGHT_IDX, PRESSURE_IDX};
use kwavers_grid::Grid;
use leto::Array4;

/// Field value encoding its channel, position, and step, so a transposed or
/// shifted trace cannot match.
fn field_value(field: usize, [i, j, k]: [usize; 3], step: usize) -> f64 {
    (field * 1000 + i * 100 + j * 10 + k) as f64 + 0.25 * step as f64
}

fn recorder_with_two_sensors() -> (Recorder, Grid) {
    let grid = Grid::new(3, 3, 3, 1e-3, 1e-3, 1e-3).unwrap();
    let mut sensor = GridSensorSet::new();
    sensor.push(GridPoint::new(1, 0, 0));
    sensor.push(GridPoint::new(0, 2, 1));
    let config = RecorderConfig::create("unused")
        .with_temperature_recording(RecordingState::Enabled)
        .with_snapshot_interval(1000);
    let recorder = Recorder::from_config(config, sensor, &Time::new(1e-7, 4), &grid);
    (recorder, grid)
}

#[test]
fn channel_traces_are_sensor_major_after_recording() {
    let (mut recorder, grid) = recorder_with_two_sensors();
    recorder.initialize(&grid).unwrap();
    assert!(recorder.pressure_data().is_none());
    assert!(recorder.light_data().is_none());

    let points = [[1, 0, 0], [0, 2, 1]];
    for step in 0..3 {
        let fields =
            Array4::from_shape_fn((3, 3, 3, 3), |[f, i, j, k]| field_value(f, [i, j, k], step));
        recorder.record(&fields, step).unwrap();
    }

    for (channel, data) in [
        (PRESSURE_IDX, recorder.pressure_data().unwrap()),
        (LIGHT_IDX, recorder.light_data().unwrap()),
    ] {
        assert_eq!(data.shape(), [2, 3]);
        for (s, &point) in points.iter().enumerate() {
            for step in 0..3 {
                assert_eq!(data[[s, step]], field_value(channel, point, step));
            }
        }
    }
}
