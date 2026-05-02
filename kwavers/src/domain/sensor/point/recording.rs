use super::PointSensor;
use crate::domain::grid::Grid;
use ndarray::ArrayView3;

impl PointSensor {
    /// Record field values at all sensor locations for current timestep.
    pub fn record(&mut self, field: ArrayView3<f64>, _grid: &Grid, _time_step: usize) {
        for (sensor_idx, interp) in self.interp_data.iter().enumerate() {
            let value = interp.interpolate(field);
            self.time_history[sensor_idx].push(value);
        }
        self.n_timesteps += 1;
    }

    /// Get maximum absolute pressure at specific sensor.
    pub fn max_pressure(&self, sensor_idx: usize) -> Option<f64> {
        self.time_history.get(sensor_idx).and_then(|history| {
            history
                .iter()
                .map(|v| v.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
        })
    }

    /// Get RMS pressure at specific sensor.
    ///
    /// ```text
    /// p_rms = √(1/N Σᵢ p[i]²)
    /// ```
    pub fn rms_pressure(&self, sensor_idx: usize) -> Option<f64> {
        self.time_history.get(sensor_idx).map(|history| {
            if history.is_empty() {
                return 0.0;
            }
            let sum_squares: f64 = history.iter().map(|v| v * v).sum();
            (sum_squares / (history.len() as f64)).sqrt()
        })
    }

    /// Export time history to CSV format.
    ///
    /// Header: `time, sensor_0, sensor_1, ..., sensor_N`
    pub fn to_csv(&self, dt: f64) -> String {
        let mut csv = String::new();

        csv.push_str("time");
        for i in 0..self.n_sensors() {
            csv.push_str(&format!(",sensor_{}", i));
        }
        csv.push('\n');

        for t in 0..self.n_timesteps {
            csv.push_str(&format!("{:.6e}", (t as f64) * dt));
            for sensor_idx in 0..self.n_sensors() {
                if let Some(history) = self.time_history.get(sensor_idx) {
                    if let Some(&value) = history.get(t) {
                        csv.push_str(&format!(",{:.6e}", value));
                    } else {
                        csv.push_str(",0.0");
                    }
                }
            }
            csv.push('\n');
        }

        csv
    }
}
