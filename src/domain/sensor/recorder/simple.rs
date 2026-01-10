use crate::domain::core::error::{KwaversError, KwaversResult, ValidationError};
use ndarray::{Array2, Array3};

#[derive(Debug, Clone)]
pub struct SensorRecorder {
    sensor_indices: Vec<(usize, usize, usize)>,
    pressure: Option<Array2<f64>>,
    expected_steps: usize,
    next_step: usize,
}

impl SensorRecorder {
    pub fn new(
        sensor_mask: Option<&Array3<bool>>,
        shape: (usize, usize, usize),
        expected_steps: usize,
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
                    mask.indexed_iter()
                        .filter_map(
                            |((i, j, k), &is_sensor)| {
                                if is_sensor {
                                    Some((i, j, k))
                                } else {
                                    None
                                }
                            },
                        )
                        .collect()
                }
            }
            None => Vec::new(),
        };

        let pressure = if sensor_indices.is_empty() {
            None
        } else {
            Some(Array2::zeros((sensor_indices.len(), expected_steps)))
        };

        Ok(Self {
            sensor_indices,
            pressure,
            expected_steps,
            next_step: 0,
        })
    }

    pub fn sensor_indices(&self) -> &[(usize, usize, usize)] {
        &self.sensor_indices
    }

    pub fn extract_pressure_data(&self) -> Option<Array2<f64>> {
        self.pressure.clone()
    }

    pub fn record_step(&mut self, pressure_field: &Array3<f64>) -> KwaversResult<()> {
        if self.sensor_indices.is_empty() {
            self.next_step = self.next_step.saturating_add(1);
            return Ok(());
        }

        let Some(ref mut pressure) = self.pressure else {
            // Should be unreachable if indices is not empty
            return Ok(());
        };

        if self.next_step >= self.expected_steps {
            // Silently ignore overflow or log warning?
            // For now, let's just return to avoid panic
            return Ok(());
        }

        for (row, &(i, j, k)) in self.sensor_indices.iter().enumerate() {
            pressure[[row, self.next_step]] = pressure_field[[i, j, k]];
        }

        self.next_step += 1;
        Ok(())
    }
}
