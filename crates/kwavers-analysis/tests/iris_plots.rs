//! Value checks for the public HTML plotting functions.

#![cfg(feature = "plotting")]

use kwavers_analysis::plotting::{
    plot_field_comparison, plot_positions, plot_pressure_field_2d, plot_pressure_field_3d,
    plot_recorder_data, plot_temporal_evolution,
};
use kwavers_core::time::Time;
use kwavers_grid::Grid;
use kwavers_receiver::{
    grid_sampling::{GridPoint, GridSensorSet},
    recorder::{Recorder, RecorderConfig, RecorderTrait, RecordingState},
};
use leto::{Array3, Array4};
use serde_json::{json, Value};
use std::path::Path;
use tracing_subscriber::fmt::format::FmtSpan;

fn trace() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    )
}

fn payload(path: &Path) -> Value {
    let html = std::fs::read_to_string(path).expect("plot writes HTML");
    assert!(
        html.contains("getContext(\"2d\")"),
        "Canvas renderer is embedded"
    );
    let json = html
        .split_once("id=\"data\">")
        .expect("data script starts")
        .1
        .split_once("</script>")
        .expect("data script ends")
        .0;
    serde_json::from_str(json).expect("plot has valid JSON")
}

#[test]
fn field_plots_keep_coordinates_pressure_and_difference_order() {
    let _trace = trace();
    let (grid, a, b, output) = tracing::info_span!("arrange").in_scope(|| {
        let grid = Grid::new(2, 3, 2, 0.25, 0.5, 0.75).expect("valid grid");
        let a = Array3::from_shape_fn([2, 3, 2], |[x, y, z]| {
            16.0 * [0., 1.][x] + 4.0 * [0., 1., 2.][y] + [0., 1.][z]
        });
        let b = Array3::from_shape_fn([2, 3, 2], |[x, y, z]| {
            [1., 3.][x] + [0., 1., 2.][y] + [2., 0.][z]
        });
        (
            grid,
            a,
            b,
            tempfile::tempdir().expect("plot test directory"),
        )
    });
    let heat = output.path().join("pressure.html");
    let diff = output.path().join("difference.html");
    let cloud = output.path().join("cloud.html");
    let sensors = output.path().join("sensors.html");
    tracing::info_span!("act").in_scope(|| {
        plot_pressure_field_2d(&a, &grid, 1, "Pressure", heat.to_str().expect("UTF-8 path"));
        plot_field_comparison(
            &a,
            &b,
            &grid,
            1,
            "Difference",
            diff.to_str().expect("UTF-8 path"),
        );
        plot_pressure_field_3d(&a, &grid, "Cloud", cloud.to_str().expect("UTF-8 path"));
        plot_positions(
            &[(0., 0.5, 0.75), (0.25, 0., 0.)],
            "Sensors",
            sensors.to_str().expect("UTF-8 path"),
        );
    });

    let _assert = tracing::info_span!("assert").entered();
    let heatmap = payload(&heat);
    assert_eq!(heatmap["x"], json!([0., 0.25]));
    assert_eq!(heatmap["y"], json!([0., 0.5, 1.0]));
    assert_eq!(heatmap["values"], json!([1., 17., 5., 21., 9., 25.]));
    assert_eq!(payload(&diff)["values"], json!([0., 14., 3., 17., 6., 20.]));
    let scatter = payload(&cloud);
    assert_eq!(scatter["points"][11], json!([0.25, 1.0, 0.75]));
    assert_eq!(scatter["markers"]["values"][11], json!(25.));
    assert_eq!(
        payload(&sensors)["points"],
        json!([[0., 0.5, 0.75], [0.25, 0., 0.]])
    );
}

#[test]
fn recorder_plot_uses_recorded_times_and_sensor_order() {
    let _trace = trace();
    let (recorder, output) = tracing::info_span!("arrange").in_scope(|| {
        let grid = Grid::new(2, 2, 2, 0.25, 0.5, 0.75).expect("valid grid");
        let sensors =
            GridSensorSet::from_points(vec![GridPoint::new(1, 0, 0), GridPoint::new(0, 0, 0)]);
        let config = RecorderConfig::create("test").with_light_recording(RecordingState::Disabled);
        let mut recorder = Recorder::from_config(config, sensors, &Time::new(0.125, 5), &grid);
        let mut fields = Array4::zeros([1, 2, 2, 2]);
        fields[[0, 1, 0, 0]] = 7.0;
        fields[[0, 0, 0, 0]] = -2.0;
        recorder.record(&fields, 1).expect("first pressure samples");
        fields[[0, 1, 0, 0]] = 11.0;
        fields[[0, 0, 0, 0]] = 3.0;
        recorder
            .record(&fields, 4)
            .expect("second pressure samples");
        (recorder, tempfile::tempdir().expect("plot test directory"))
    });
    let recorded = output.path().join("recorded.html");
    let evolution = output.path().join("evolution.html");
    tracing::info_span!("act").in_scope(|| {
        plot_recorder_data(&recorder, recorded.to_str().expect("UTF-8 path"));
        plot_temporal_evolution(
            &[1., -1.],
            &[0.125, 0.5],
            "Evolution",
            "Pressure (Pa)",
            evolution.to_str().expect("UTF-8 path"),
        );
    });

    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(
        payload(&recorded)["series"],
        json!([
            {"name": "Sensor 0", "x": [0.125, 0.5], "y": [7.0, 11.0]},
            {"name": "Sensor 1", "x": [0.125, 0.5], "y": [-2.0, 3.0]},
        ])
    );
    assert_eq!(
        payload(&evolution)["series"],
        json!([
            {"name": "Evolution", "x": [0.125, 0.5], "y": [1.0, -1.0]}
        ])
    );
}
