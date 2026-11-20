use iris::view::SeriesView;
use iris_plot_html::Plot;

use super::{output::save, Recorder};

/// Export one time series using the supplied times in seconds.
pub fn plot_temporal_evolution(
    data: &[f64],
    time_points: &[f64],
    title: &str,
    y_label: &str,
    filename: &str,
) {
    match SeriesView::new("Evolution", time_points, data) {
        Ok(series) => save(
            filename,
            Ok(Plot::lines(title, ["Time (s)", y_label], &[series])),
        ),
        Err(error) => save(filename, Err(error.into())),
    }
}

/// Export every recorded pressure trace with its original sample times.
pub fn plot_recorder_data(recorder: &Recorder, filename: &str) {
    let traces = &recorder.pressure_sensor_data;
    if traces.is_empty() || traces.n_sensors() == 0 {
        log::warn!("No pressure data recorded for {filename}");
        return;
    }
    if traces.n_steps() != recorder.recorded_steps.len() {
        log::error!("Pressure samples and recorded times differ in length for {filename}");
        return;
    }

    let samples = traces.to_sensor_major();
    let storage = samples.as_slice().expect("sensor-major copy is contiguous");
    let names: Vec<_> = (0..traces.n_sensors())
        .map(|i| format!("Sensor {i}"))
        .collect();
    let series = storage
        .chunks_exact(traces.n_steps())
        .zip(&names)
        .map(|(row, name)| SeriesView::new(name.as_str(), &recorder.recorded_steps, row))
        .collect::<Result<Vec<_>, _>>();

    match series {
        Ok(series) => save(
            filename,
            Ok(Plot::lines(
                "Recorded Pressure Evolution",
                ["Time (s)", "Pressure (Pa)"],
                &series,
            )),
        ),
        Err(error) => save(filename, Err(error.into())),
    }
}
