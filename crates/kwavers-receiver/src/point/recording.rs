use super::PointSensor;
use kwavers_grid::Grid;
use leto::ArrayView3;

impl PointSensor {
    /// Record field values at all sensor locations for current timestep.
    pub fn record(&mut self, field: ArrayView3<f64>, _grid: &Grid, _time_step: usize) {
        self.time_history.push_step(
            self.interp_data
                .iter()
                .map(|interp| interp.interpolate(field)),
        );
    }

    /// Get maximum absolute pressure at specific sensor.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[must_use]
    pub fn max_pressure(&self, sensor_idx: usize) -> Option<f64> {
        if sensor_idx >= self.time_history.n_sensors() {
            return None;
        }
        self.time_history
            .sensor(sensor_idx)
            .map(f64::abs)
            .max_by(f64::total_cmp)
    }

    /// Get RMS pressure at specific sensor.
    ///
    /// ```text
    /// p_rms = √(1/N Σᵢ p`i`²)
    /// ```
    #[must_use]
    pub fn rms_pressure(&self, sensor_idx: usize) -> Option<f64> {
        if sensor_idx >= self.time_history.n_sensors() {
            return None;
        }
        let n_steps = self.time_history.n_steps();
        if n_steps == 0 {
            return Some(0.0);
        }
        let sum_squares: f64 = self.time_history.sensor(sensor_idx).map(|v| v * v).sum();
        Some((sum_squares / (n_steps as f64)).sqrt())
    }

    /// Export time history to CSV format.
    ///
    /// Header: `time, sensor_0, sensor_1, ..., sensor_N`
    #[must_use]
    pub fn to_csv(&self, dt: f64) -> String {
        let mut csv = String::new();

        csv.push_str("time");
        for i in 0..self.n_sensors() {
            csv.push_str(&format!(",sensor_{}", i));
        }
        csv.push('\n');

        for (t, row) in self.time_history.steps().enumerate() {
            csv.push_str(&format!("{:.6e}", (t as f64) * dt));
            for value in row {
                csv.push_str(&format!(",{:.6e}", value));
            }
            csv.push('\n');
        }

        csv
    }
}
