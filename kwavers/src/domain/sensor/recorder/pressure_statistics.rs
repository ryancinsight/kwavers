//! Pressure field statistics for k-Wave parity recording modes
//!
//! Implements spatial statistics: p_max, p_min, p_rms, p_final

use ndarray::{Array1, Array3};

/// Spatial statistics for pressure field over time
#[derive(Debug, Clone)]
pub struct PressureFieldStatistics {
    /// Maximum pressure at each spatial point (p_max)
    pub p_max: Array3<f64>,
    /// Minimum pressure at each spatial point (p_min)
    pub p_min: Array3<f64>,
    /// Sum of squared pressures for RMS calculation (p_rms)
    pub p_squared_sum: Array3<f64>,
    /// Final pressure field (p_final)
    pub p_final: Array3<f64>,
    /// Number of time steps recorded
    pub time_step_count: usize,
    /// Shape of the grid
    shape: (usize, usize, usize),
}

impl PressureFieldStatistics {
    /// Create new statistics tracker for given grid dimensions
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            p_max: Array3::from_elem((nx, ny, nz), f64::NEG_INFINITY),
            p_min: Array3::from_elem((nx, ny, nz), f64::INFINITY),
            p_squared_sum: Array3::zeros((nx, ny, nz)),
            p_final: Array3::zeros((nx, ny, nz)),
            time_step_count: 0,
            shape: (nx, ny, nz),
        }
    }

    /// Update statistics with current pressure field
    pub fn update(&mut self, pressure: &Array3<f64>) {
        assert_eq!(
            pressure.shape(),
            &[self.shape.0, self.shape.1, self.shape.2]
        );

        // Update max/min
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                for k in 0..self.shape.2 {
                    let p = pressure[[i, j, k]];
                    self.p_max[[i, j, k]] = self.p_max[[i, j, k]].max(p);
                    self.p_min[[i, j, k]] = self.p_min[[i, j, k]].min(p);
                    self.p_squared_sum[[i, j, k]] += p * p;
                }
            }
        }

        // Store final pressure
        self.p_final = pressure.clone();
        self.time_step_count += 1;
    }

    /// Calculate RMS pressure field
    #[must_use]
    pub fn p_rms(&self) -> Array3<f64> {
        if self.time_step_count == 0 {
            return Array3::zeros(self.shape);
        }

        self.p_squared_sum
            .mapv(|v| (v / self.time_step_count as f64).sqrt())
    }

    /// Get p_max as array
    #[must_use]
    pub fn get_p_max(&self) -> &Array3<f64> {
        &self.p_max
    }

    /// Get p_min as array
    #[must_use]
    pub fn get_p_min(&self) -> &Array3<f64> {
        &self.p_min
    }

    /// Get p_final as array
    #[must_use]
    pub fn get_p_final(&self) -> &Array3<f64> {
        &self.p_final
    }

    /// Get global maximum pressure value
    #[must_use]
    pub fn global_p_max(&self) -> f64 {
        self.p_max.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    }

    /// Get global minimum pressure value
    #[must_use]
    pub fn global_p_min(&self) -> f64 {
        self.p_min.iter().fold(f64::INFINITY, |a, &b| a.min(b))
    }

    /// Get global RMS pressure value
    #[must_use]
    pub fn global_p_rms(&self) -> f64 {
        if self.time_step_count == 0 {
            return 0.0;
        }
        let sum_squared: f64 = self.p_squared_sum.iter().sum();
        let n = (self.shape.0 * self.shape.1 * self.shape.2) as f64;
        (sum_squared / (self.time_step_count as f64 * n)).sqrt()
    }

    /// Sample statistics at sensor positions
    pub fn sample_at_positions(&self, positions: &[(usize, usize, usize)]) -> SampledStatistics {
        let mut p_max = Vec::with_capacity(positions.len());
        let mut p_min = Vec::with_capacity(positions.len());
        let mut p_rms = Vec::with_capacity(positions.len());
        let mut p_final = Vec::with_capacity(positions.len());

        for &(i, j, k) in positions {
            p_max.push(self.p_max[[i, j, k]]);
            p_min.push(self.p_min[[i, j, k]]);
            p_rms.push((self.p_squared_sum[[i, j, k]] / self.time_step_count as f64).sqrt());
            p_final.push(self.p_final[[i, j, k]]);
        }

        SampledStatistics {
            p_max: Array1::from(p_max),
            p_min: Array1::from(p_min),
            p_rms: Array1::from(p_rms),
            p_final: Array1::from(p_final),
        }
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        self.p_max.fill(f64::NEG_INFINITY);
        self.p_min.fill(f64::INFINITY);
        self.p_squared_sum.fill(0.0);
        self.p_final.fill(0.0);
        self.time_step_count = 0;
    }
}

/// Statistics sampled at specific sensor positions
#[derive(Debug, Clone)]
pub struct SampledStatistics {
    pub p_max: Array1<f64>,
    pub p_min: Array1<f64>,
    pub p_rms: Array1<f64>,
    pub p_final: Array1<f64>,
}

impl SampledStatistics {
    /// Get number of sensors
    #[must_use]
    pub fn num_sensors(&self) -> usize {
        self.p_max.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pressure_statistics_basic() {
        let mut stats = PressureFieldStatistics::new(4, 4, 4);

        // Simulate 3 time steps
        for t in 0..3 {
            let pressure = Array3::from_elem((4, 4, 4), (t as f64 + 1.0) * 1000.0);
            stats.update(&pressure);
        }

        assert_eq!(stats.time_step_count, 3);
        assert_eq!(stats.global_p_max(), 3000.0);
        assert_eq!(stats.global_p_min(), 1000.0);

        // RMS should be sqrt((1000² + 2000² + 3000²) / 3) = sqrt(14000000/3)
        let expected_rms =
            ((1000.0f64.powi(2) + 2000.0f64.powi(2) + 3000.0f64.powi(2)) / 3.0).sqrt();
        assert!((stats.global_p_rms() - expected_rms).abs() < 1e-10);
    }

    #[test]
    fn test_sample_at_positions() {
        let mut stats = PressureFieldStatistics::new(10, 10, 10);

        let pressure = Array3::from_elem((10, 10, 10), 5000.0);
        stats.update(&pressure);

        let positions = vec![(0, 0, 0), (5, 5, 5), (9, 9, 9)];
        let sampled = stats.sample_at_positions(&positions);

        assert_eq!(sampled.num_sensors(), 3);
        assert!(sampled.p_max.iter().all(|&v| v == 5000.0));
    }
}
