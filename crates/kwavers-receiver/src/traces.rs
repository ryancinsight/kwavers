//! Contiguous per-step sensor traces.
//!
//! A recorder samples every sensor once per time step and appends the row;
//! storage is therefore time-major: step `t` occupies
//! `samples[t * n_sensors..(t + 1) * n_sensors]`. Appending a step is one
//! `extend` into a single buffer, and a per-sensor trace is a stride-`n_sensors`
//! walk over the same buffer.

use leto::Array2;

/// Time-major sample buffer for a fixed set of sensors.
///
/// # Examples
///
/// ```
/// use kwavers_receiver::SensorTraces;
///
/// let mut traces = SensorTraces::new(2);
/// traces.push_step([1.0, 2.0]);
/// traces.push_step([3.0, 4.0]);
///
/// assert_eq!(traces.n_steps(), 2);
/// assert_eq!(traces.step(1), Some(&[3.0, 4.0][..]));
/// assert_eq!(traces.sensor(1).collect::<Vec<_>>(), vec![2.0, 4.0]);
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SensorTraces {
    n_sensors: usize,
    n_steps: usize,
    samples: Vec<f64>,
}

impl SensorTraces {
    /// Empty traces for `n_sensors` sensors.
    #[must_use]
    pub fn new(n_sensors: usize) -> Self {
        Self {
            n_sensors,
            n_steps: 0,
            samples: Vec::new(),
        }
    }

    /// Reserve room for `additional_steps` further steps.
    pub fn reserve_steps(&mut self, additional_steps: usize) {
        self.samples
            .reserve(additional_steps.saturating_mul(self.n_sensors));
    }

    /// Append one step holding one sample per sensor, in sensor order.
    ///
    /// # Panics
    ///
    /// Panics if `values` does not yield exactly [`Self::n_sensors`] samples;
    /// a short or long row would shift every later step.
    #[track_caller]
    pub fn push_step(&mut self, values: impl IntoIterator<Item = f64>) {
        let start = self.samples.len();
        self.samples.extend(values);
        assert_eq!(
            self.samples.len() - start,
            self.n_sensors,
            "invariant: a step holds one sample per sensor"
        );
        self.n_steps += 1;
    }

    /// Append one step from fallible samples, leaving the traces unchanged
    /// when any sample fails.
    ///
    /// # Errors
    ///
    /// Returns the first error `values` yields.
    ///
    /// # Panics
    ///
    /// Panics if `values` yields a sample count other than [`Self::n_sensors`].
    #[track_caller]
    pub fn try_push_step<E>(
        &mut self,
        values: impl IntoIterator<Item = Result<f64, E>>,
    ) -> Result<(), E> {
        let start = self.samples.len();
        for value in values {
            match value {
                Ok(sample) => self.samples.push(sample),
                Err(error) => {
                    self.samples.truncate(start);
                    return Err(error);
                }
            }
        }
        assert_eq!(
            self.samples.len() - start,
            self.n_sensors,
            "invariant: a step holds one sample per sensor"
        );
        self.n_steps += 1;
        Ok(())
    }

    /// Number of sensors per step.
    #[must_use]
    pub fn n_sensors(&self) -> usize {
        self.n_sensors
    }

    /// Number of recorded steps.
    #[must_use]
    pub fn n_steps(&self) -> usize {
        self.n_steps
    }

    /// Whether no step has been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n_steps == 0
    }

    /// Samples of step `t` in sensor order, or `None` past the last step.
    #[must_use]
    pub fn step(&self, t: usize) -> Option<&[f64]> {
        let start = t.checked_mul(self.n_sensors)?;
        (t < self.n_steps).then(|| &self.samples[start..start + self.n_sensors])
    }

    /// Steps in time order, each a slice in sensor order.
    pub fn steps(&self) -> impl ExactSizeIterator<Item = &[f64]> {
        (0..self.n_steps).map(|t| &self.samples[t * self.n_sensors..(t + 1) * self.n_sensors])
    }

    /// Trace of sensor `s` in time order; empty when `s` is out of range.
    pub fn sensor(&self, s: usize) -> impl ExactSizeIterator<Item = f64> + '_ {
        let len = if s < self.n_sensors { self.n_steps } else { 0 };
        self.samples
            .iter()
            .skip(s)
            .step_by(self.n_sensors.max(1))
            .take(len)
            .copied()
    }

    /// Discard every step, keeping the allocation.
    pub fn clear(&mut self) {
        self.samples.clear();
        self.n_steps = 0;
    }

    /// Sensor-major copy, shape `[n_sensors, n_steps]`.
    #[must_use]
    pub fn to_sensor_major(&self) -> Array2<f64> {
        let mut out = Array2::zeros([self.n_sensors, self.n_steps]);
        for (t, row) in self.steps().enumerate() {
            for (s, &value) in row.iter().enumerate() {
                out[[s, t]] = value;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::SensorTraces;

    #[test]
    fn sensor_walk_and_sensor_major_copy_transpose_the_rows() {
        let mut traces = SensorTraces::new(3);
        traces.push_step([1.0, 2.0, 3.0]);
        traces.push_step([4.0, 5.0, 6.0]);

        assert_eq!(traces.sensor(2).collect::<Vec<_>>(), vec![3.0, 6.0]);
        assert_eq!(traces.sensor(3).len(), 0);
        assert_eq!(traces.step(2), None);
        let major = traces.to_sensor_major();
        assert_eq!(major.shape(), [3, 2]);
        assert_eq!(major[[1, 0]], 2.0);
        assert_eq!(major[[2, 1]], 6.0);
    }

    #[test]
    fn failed_step_leaves_earlier_steps_untouched() {
        let mut traces = SensorTraces::new(2);
        traces.push_step([1.0, 2.0]);
        let failed = traces.try_push_step([Ok(7.0), Err("sensor 1")]);

        assert_eq!(failed, Err("sensor 1"));
        assert_eq!(traces.n_steps(), 1);
        assert_eq!(traces.steps().collect::<Vec<_>>(), vec![&[1.0, 2.0][..]]);
    }

    #[test]
    fn zero_sensor_steps_are_counted() {
        let mut traces = SensorTraces::new(0);
        traces.push_step([]);
        traces.push_step([]);

        assert_eq!(traces.n_steps(), 2);
        assert_eq!(traces.step(1), Some(&[][..]));
        assert_eq!(traces.sensor(0).len(), 0);
        assert_eq!(traces.to_sensor_major().shape(), [0, 2]);
    }

    #[test]
    #[should_panic(expected = "one sample per sensor")]
    fn short_row_is_rejected() {
        SensorTraces::new(2).push_step([1.0]);
    }
}
