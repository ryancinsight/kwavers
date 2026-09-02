use super::ViscoacousticMemorySolver;
use kwavers_core::error::{KwaversError, KwaversResult, SystemError};
use std::alloc::Layout;

impl ViscoacousticMemorySolver {
    /// Register a soft additive pressure source at `index`.
    ///
    /// The solver applies `signal[step]` while the current step is within the
    /// supplied signal.
    ///
    /// # Errors
    ///
    /// Returns an error when `index` is outside the solver grid.
    pub fn add_pressure_source(
        &mut self,
        index: (usize, usize, usize),
        signal: Vec<f64>,
    ) -> KwaversResult<()> {
        self.check_index(index)?;
        self.pressure_sources.push((index, signal));
        Ok(())
    }

    /// Register a pressure sensor at `index` and return its trace identifier.
    ///
    /// Each call to [`Self::step`] appends the pressure at `index` to the
    /// sensor's trace. Call [`Self::reserve_sensor_samples`] after registering
    /// all sensors when the remaining step count is known.
    ///
    /// # Errors
    ///
    /// Returns an error when `index` is outside the solver grid.
    pub fn add_pressure_sensor(&mut self, index: (usize, usize, usize)) -> KwaversResult<usize> {
        self.check_index(index)?;
        self.pressure_sensors.push(index);
        self.sensor_record.push(Vec::new());
        Ok(self.pressure_sensors.len() - 1)
    }

    /// Reserve `additional_samples` trace entries for every registered sensor.
    ///
    /// Reserving the known remaining step count before propagation moves trace
    /// allocation out of [`Self::step`]. A pressure reset clears trace lengths
    /// but retains this capacity for a repeated run.
    ///
    /// # Errors
    ///
    /// Returns [`SystemError::MemoryAllocation`] when the requested trace size
    /// is not representable or the allocator rejects a reservation. Existing
    /// sample values and lengths remain unchanged; a failed reservation may
    /// leave capacity already acquired for an earlier sensor available.
    pub fn reserve_sensor_samples(&mut self, additional_samples: usize) -> KwaversResult<()> {
        for (sensor, trace) in self.sensor_record.iter_mut().enumerate() {
            let required_samples =
                trace.len().checked_add(additional_samples).ok_or_else(|| {
                    sensor_reservation_error(sensor, usize::MAX, "sample count overflow")
                })?;
            let requested_bytes = Layout::array::<f64>(required_samples)
                .map_err(|error| sensor_reservation_error(sensor, usize::MAX, &error.to_string()))?
                .size();
            trace
                .try_reserve_exact(additional_samples)
                .map_err(|error| {
                    sensor_reservation_error(sensor, requested_bytes, &error.to_string())
                })?;
        }
        Ok(())
    }

    /// Return the recorded pressure trace for `id`.
    ///
    /// # Panics
    ///
    /// Panics when `id` is not a sensor identifier returned by
    /// [`Self::add_pressure_sensor`].
    #[must_use]
    pub fn sensor_trace(&self, id: usize) -> &[f64] {
        &self.sensor_record[id]
    }

    fn check_index(&self, (i, j, k): (usize, usize, usize)) -> KwaversResult<()> {
        if i < self.nx && j < self.ny && k < self.nz {
            Ok(())
        } else {
            Err(KwaversError::InvalidInput(
                "source/sensor index out of grid bounds".to_owned(),
            ))
        }
    }
}

fn sensor_reservation_error(sensor: usize, requested_bytes: usize, reason: &str) -> KwaversError {
    SystemError::MemoryAllocation {
        requested_bytes,
        reason: format!("sensor {sensor} trace reservation failed: {reason}"),
    }
    .into()
}
