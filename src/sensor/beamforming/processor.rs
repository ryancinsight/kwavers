//! Beamforming processor

use super::config::BeamformingConfig;
use crate::error::KwaversResult;
use crate::utils::linear_algebra::LinearAlgebra;
use ndarray::Array2;

/// Beamforming processor for array algorithms
#[derive(Debug, Debug))]
pub struct BeamformingProcessor {
    pub config: BeamformingConfig,
    sensor_positions: Vec<[f64; 3]>,
    num_sensors: usize,
}

impl BeamformingProcessor {
    /// Create new beamforming processor
    pub fn new(config: BeamformingConfig, sensor_positions: Vec<[f64; 3]>) -> Self {
        let num_sensors = sensor_positions.len();
        Self {
            config,
            sensor_positions,
            num_sensors,
        }
    }

    /// Get number of sensors
    pub fn num_sensors(&self) -> usize {
        self.num_sensors
    }

    /// Get sensor positions
    pub fn sensor_positions(&self) -> &[[f64; 3] {
        &self.sensor_positions
    }

    /// Compute eigendecomposition of a matrix
    pub fn eigendecomposition(
        &self,
        matrix: &Array2<f64>,
    ) -> KwaversResult<(ndarray::Array1<f64>, Array2<f64>)> {
        LinearAlgebra::eigendecomposition(matrix)
    }

    /// Compute matrix inverse
    pub fn matrix_inverse(&self, matrix: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        LinearAlgebra::matrix_inverse(matrix)
    }
}
