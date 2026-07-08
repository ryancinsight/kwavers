//! Time-step recording for SensorRecorder.

use kwavers_core::error::KwaversResult;

use super::SensorRecorder;

#[doc(hidden)]
pub trait PressureSamples3 {
    fn value_at(&self, i: usize, j: usize, k: usize) -> f64;
}

impl PressureSamples3 for ndarray::Array3<f64> {
    fn value_at(&self, i: usize, j: usize, k: usize) -> f64 {
        self[[i, j, k]]
    }
}

impl PressureSamples3 for leto::Array3<f64> {
    fn value_at(&self, i: usize, j: usize, k: usize) -> f64 {
        self[[i, j, k]]
    }
}

impl SensorRecorder {
    /// Record one pressure time step at all sensor positions.
    ///
    /// Also updates pressure spatial statistics (max/min/rms/final) when
    /// the recorder was created with [`with_modes`](Self::with_modes) or
    /// [`with_spec`](Self::with_spec) and the relevant modes were requested.
    ///
    /// Silently returns `Ok(())` after `expected_steps` have been recorded
    /// (matching k-Wave's behaviour of stopping after the sensor buffer is full).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn record_step<P>(&mut self, pressure_field: &P) -> KwaversResult<()>
    where
        P: PressureSamples3 + crate::recorder::pressure_statistics::PressureArray3Access,
    {
        if let Some(ref mut stats) = self.stats {
            stats.update(pressure_field);
        }

        if self.sensor_indices.is_empty() {
            self.next_step = self.next_step.saturating_add(1);
            return Ok(());
        }

        if self.next_step >= self.expected_steps {
            return Ok(());
        }

        if let Some(ref mut pressure) = self.pressure {
            for (row, &(i, j, k)) in self.sensor_indices.iter().enumerate() {
                pressure[[row, self.next_step]] = pressure_field.value_at(i, j, k);
            }
        }

        self.next_step += 1;
        Ok(())
    }
}
