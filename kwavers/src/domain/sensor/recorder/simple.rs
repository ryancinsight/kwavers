use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use crate::domain::sensor::recorder::config::RecordingMode;
use crate::domain::sensor::recorder::pressure_statistics::{PressureFieldStatistics, SampledStatistics};
use ndarray::{Array1, Array2, Array3};

#[derive(Debug, Clone)]
pub struct SensorRecorder {
    sensor_indices: Vec<(usize, usize, usize)>,
    pressure: Option<Array2<f64>>,
    expected_steps: usize,
    next_step: usize,
    /// Optional full-grid statistics, enabled via `with_modes`.
    stats: Option<PressureFieldStatistics>,
}

impl SensorRecorder {
    pub fn new(
        sensor_mask: Option<&Array3<bool>>,
        shape: (usize, usize, usize),
        expected_steps: usize,
    ) -> KwaversResult<Self> {
        Self::with_modes(sensor_mask, shape, expected_steps, &[])
    }

    /// Create a recorder that also tracks spatial statistics (p_max, p_min, p_rms, p_final).
    ///
    /// Pass `modes` containing any of [`RecordingMode::MaxPressure`], [`RecordingMode::MinPressure`],
    /// [`RecordingMode::RmsPressure`], [`RecordingMode::FinalPressure`], or
    /// [`RecordingMode::AllStatistics`] to enable statistics tracking. Passing an empty slice
    /// disables statistics (equivalent to `new`).
    ///
    /// Statistics are accumulated over the full pressure grid at each step and sampled at
    /// sensor positions on extraction. This matches k-Wave's `sensor.record = {'p_max', 'p_rms'}`
    /// behaviour.
    pub fn with_modes(
        sensor_mask: Option<&Array3<bool>>,
        shape: (usize, usize, usize),
        expected_steps: usize,
        modes: &[RecordingMode],
    ) -> KwaversResult<Self> {
        let sensor_indices = match sensor_mask {
            Some(mask) => {
                let mask_dim = mask.dim();
                if mask_dim != shape {
                    if mask_dim == (1, 1, 1) && !mask[[0, 0, 0]] {
                        Vec::new()
                    } else {
                        return Err(KwaversError::Validation(
                            ValidationError::ConstraintViolation {
                                message: format!(
                                    "Sensor mask shape mismatch: expected {shape:?}, got {mask_dim:?}"
                                ),
                            },
                        ));
                    }
                } else {
                    // Iterate in Fortran-order (column-major) for k-wave compatibility:
                    // x-index changes fastest, matching MATLAB's find(sensor_mask) ordering.
                    let (nx, ny, nz) = shape;
                    let mut indices = Vec::new();
                    for k in 0..nz {
                        for j in 0..ny {
                            for i in 0..nx {
                                if mask[[i, j, k]] {
                                    indices.push((i, j, k));
                                }
                            }
                        }
                    }
                    indices
                }
            }
            None => Vec::new(),
        };

        let pressure = if sensor_indices.is_empty() {
            None
        } else {
            Some(Array2::zeros((sensor_indices.len(), expected_steps)))
        };

        let needs_stats = modes.iter().any(|m| {
            matches!(
                m,
                RecordingMode::MaxPressure
                    | RecordingMode::MinPressure
                    | RecordingMode::RmsPressure
                    | RecordingMode::FinalPressure
                    | RecordingMode::AllStatistics
                    | RecordingMode::MaxMinPressure
                    | RecordingMode::MaxPressureAll
                    | RecordingMode::MinPressureAll
            )
        });

        let stats = if needs_stats {
            Some(PressureFieldStatistics::new(shape.0, shape.1, shape.2))
        } else {
            None
        };

        Ok(Self {
            sensor_indices,
            pressure,
            expected_steps,
            next_step: 0,
            stats,
        })
    }

    /// Create a `SensorRecorder` from a pre-ordered list of grid indices.
    ///
    /// Use this for transducer sensors where nodes must be emitted in element order
    /// (element 0 nodes first, element 1 nodes next, …) matching k-Wave's layout.
    pub fn from_ordered_indices(
        indices: Vec<(usize, usize, usize)>,
        expected_steps: usize,
    ) -> KwaversResult<Self> {
        let pressure = if indices.is_empty() {
            None
        } else {
            Some(Array2::zeros((indices.len(), expected_steps)))
        };
        Ok(Self {
            sensor_indices: indices,
            pressure,
            expected_steps,
            next_step: 0,
            stats: None,
        })
    }

    pub fn sensor_indices(&self) -> &[(usize, usize, usize)] {
        &self.sensor_indices
    }

    pub fn extract_pressure_data(&self) -> Option<Array2<f64>> {
        self.pressure.clone()
    }

    /// Extract p_max sampled at sensor positions. Returns `None` if statistics were
    /// not enabled via `with_modes`.
    #[must_use]
    pub fn extract_p_max(&self) -> Option<Array1<f64>> {
        let stats = self.stats.as_ref()?;
        let sampled = stats.sample_at_positions(&self.sensor_indices);
        Some(sampled.p_max)
    }

    /// Extract p_min sampled at sensor positions.
    #[must_use]
    pub fn extract_p_min(&self) -> Option<Array1<f64>> {
        let stats = self.stats.as_ref()?;
        let sampled = stats.sample_at_positions(&self.sensor_indices);
        Some(sampled.p_min)
    }

    /// Extract p_rms (root-mean-square) sampled at sensor positions.
    #[must_use]
    pub fn extract_p_rms(&self) -> Option<Array1<f64>> {
        let stats = self.stats.as_ref()?;
        let sampled = stats.sample_at_positions(&self.sensor_indices);
        Some(sampled.p_rms)
    }

    /// Extract the final pressure field at sensor positions.
    #[must_use]
    pub fn extract_p_final(&self) -> Option<Array1<f64>> {
        let stats = self.stats.as_ref()?;
        let sampled = stats.sample_at_positions(&self.sensor_indices);
        Some(sampled.p_final)
    }

    /// Extract all statistics at once (avoids repeated sampling).
    #[must_use]
    pub fn extract_all_stats(&self) -> Option<SampledStatistics> {
        let stats = self.stats.as_ref()?;
        Some(stats.sample_at_positions(&self.sensor_indices))
    }

    pub fn record_step(&mut self, pressure_field: &Array3<f64>) -> KwaversResult<()> {
        // Update spatial statistics (full grid) when enabled
        if let Some(ref mut stats) = self.stats {
            stats.update(pressure_field);
        }

        if self.sensor_indices.is_empty() {
            self.next_step = self.next_step.saturating_add(1);
            return Ok(());
        }

        let Some(ref mut pressure) = self.pressure else {
            return Ok(());
        };

        if self.next_step >= self.expected_steps {
            return Ok(());
        }

        for (row, &(i, j, k)) in self.sensor_indices.iter().enumerate() {
            pressure[[row, self.next_step]] = pressure_field[[i, j, k]];
        }

        self.next_step += 1;
        Ok(())
    }
}
