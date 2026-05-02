use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{Array1, Array2};
use num_complex::Complex;

use super::MUSICConfig;

/// MUSIC processor for direction-of-arrival estimation
#[derive(Debug)]
pub struct MUSICProcessor {
    pub(super) config: MUSICConfig,
}

impl MUSICProcessor {
    /// Create new MUSIC processor
    pub fn new(config: &MUSICConfig) -> KwaversResult<Self> {
        config.config.validate()?;

        let num_sensors = config.config.sensor_positions.len();

        if num_sensors < 2 {
            return Err(KwaversError::InvalidInput(
                "MUSIC requires at least 2 sensors".to_string(),
            ));
        }

        if let Some(k) = config.num_sources {
            if k == 0 {
                return Err(KwaversError::InvalidInput(
                    "Number of sources must be > 0".to_string(),
                ));
            }

            if k >= num_sensors {
                return Err(KwaversError::InvalidInput(format!(
                    "Number of sources ({}) must be < number of sensors ({})",
                    k, num_sensors
                )));
            }
        }

        if config.num_snapshots < num_sensors {
            return Err(KwaversError::InvalidInput(format!(
                "Number of snapshots ({}) must be ≥ number of sensors ({})",
                config.num_snapshots, num_sensors
            )));
        }

        if config.grid_resolution == 0 {
            return Err(KwaversError::InvalidInput(
                "Grid resolution must be > 0".to_string(),
            ));
        }

        Ok(Self {
            config: config.clone(),
        })
    }

    /// Estimate spatial covariance matrix from complex sensor snapshots.
    ///
    /// Returns R = (1/N) ∑ₙ x(n) x(n)^H ∈ ℂ^(M×M) with optional diagonal loading.
    pub fn estimate_covariance(
        &self,
        snapshots: &Array2<Complex<f64>>,
    ) -> KwaversResult<Array2<Complex<f64>>> {
        let (num_sensors, num_snapshots) = snapshots.dim();

        if num_snapshots == 0 {
            return Err(KwaversError::InvalidInput(
                "Cannot compute covariance from zero snapshots".to_string(),
            ));
        }

        let mut covariance = Array2::zeros((num_sensors, num_sensors));

        for i in 0..num_sensors {
            for j in 0..num_sensors {
                let mut sum = Complex::new(0.0, 0.0);
                for k in 0..num_snapshots {
                    sum += snapshots[[i, k]] * snapshots[[j, k]].conj();
                }
                covariance[[i, j]] = sum / num_snapshots as f64;
            }
        }

        if self.config.diagonal_loading > 0.0 {
            let trace: Complex<f64> = (0..num_sensors).map(|i| covariance[[i, i]]).sum();
            let loading = self.config.diagonal_loading * (trace / num_sensors as f64).re;

            for i in 0..num_sensors {
                covariance[[i, i]] += Complex::new(loading, 0.0);
            }
        }

        Ok(covariance)
    }

    /// Compute steering vector a(θ) = exp(-j k ||θ − r_m||) for each sensor.
    pub(super) fn steering_vector(
        source_position: [f64; 3],
        sensor_positions: &[[f64; 3]],
        frequency: f64,
        speed_of_sound: f64,
    ) -> Array1<Complex<f64>> {
        let num_sensors = sensor_positions.len();
        let mut steering = Array1::zeros(num_sensors);

        let k = 2.0 * std::f64::consts::PI * frequency / speed_of_sound;

        for (m, sensor_pos) in sensor_positions.iter().enumerate() {
            let dx = source_position[0] - sensor_pos[0];
            let dy = source_position[1] - sensor_pos[1];
            let dz = source_position[2] - sensor_pos[2];
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();

            let phase = -k * distance;
            steering[m] = Complex::new(phase.cos(), phase.sin());
        }

        steering
    }
}
