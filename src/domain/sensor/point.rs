//! Point Sensors for Arbitrary Position Sampling
//!
//! This module implements point sensors that sample acoustic fields at arbitrary
//! (x, y, z) locations using trilinear interpolation from the surrounding grid points.
//!
//! # Mathematical Specification
//!
//! Point sensors evaluate the field at locations that may not align with the
//! computational grid using trilinear interpolation:
//!
//! ```text
//! p(x, y, z) = Σᵢⱼₖ wᵢⱼₖ · p[i,j,k]                    (1)
//!
//! where wᵢⱼₖ are trilinear interpolation weights:
//!
//! wᵢⱼₖ = (1 - ξ)(1 - η)(1 - ζ)  for (i,j,k)          (2a)
//!      +    ξ   (1 - η)(1 - ζ)  for (i+1,j,k)        (2b)
//!      + (1 - ξ)   η   (1 - ζ)  for (i,j+1,k)        (2c)
//!      +    ξ      η   (1 - ζ)  for (i+1,j+1,k)      (2d)
//!      + (1 - ξ)(1 - η)   ζ     for (i,j,k+1)        (2e)
//!      +    ξ   (1 - η)   ζ     for (i+1,j,k+1)      (2f)
//!      + (1 - ξ)   η      ζ     for (i,j+1,k+1)      (2g)
//!      +    ξ      η      ζ     for (i+1,j+1,k+1)    (2h)
//!
//! and (ξ, η, ζ) are local coordinates in [0,1]:
//!
//! ξ = (x - xᵢ) / dx                                   (3a)
//! η = (y - yⱼ) / dy                                   (3b)
//! ζ = (z - zₖ) / dz                                   (3c)
//! ```
//!
//! # Properties
//!
//! - **Interpolation Order**: O(h²) accuracy where h = max(dx, dy, dz)
//! - **Smoothness**: C⁰ continuous (continuous but not differentiable at grid points)
//! - **Monotonicity**: Preserves monotonicity of underlying field
//! - **Partition of Unity**: Σ wᵢⱼₖ = 1 (exact at grid points)
//!
//! # Applications
//!
//! Point sensors are essential for:
//! - Comparison with experimental hydrophone measurements
//! - Validation against k-Wave simulations (k-Wave uses point sensors extensively)
//! - Beam profile characterization at specific locations
//! - Focal spot measurement
//!
//! # Usage Example
//!
//! ```rust,no_run
//! use kwavers::domain::sensor::point::{PointSensor, PointSensorConfig};
//! use kwavers::domain::grid::Grid;
//!
//! // Create grid
//! let grid = Grid::new(128, 128, 128, 0.5e-3, 0.5e-3, 0.5e-3).unwrap();
//!
//! // Define sensor locations (e.g., hydrophone positions)
//! let locations = vec![
//!     [0.032, 0.032, 0.064], // Focal point
//!     [0.032, 0.040, 0.064], // Off-axis
//! ];
//!
//! let config = PointSensorConfig { locations };
//! let mut sensor = PointSensor::new(config, &grid).unwrap();
//!
//! // During simulation: sample field at each time step
//! // sensor.record(&pressure_field, &grid, time_step);
//!
//! // After simulation: retrieve time histories
//! // let focal_history = sensor.time_history(0);
//! ```
//!
//! # References
//!
//! 1. Treeby & Cox (2010). "k-Wave: MATLAB toolbox for the simulation and
//!    reconstruction of photoacoustic wave fields." *J. Biomed. Opt.*, 15(2), 021314.
//! 2. Press et al. (2007). *Numerical Recipes* (3rd ed.), Ch. 3: Interpolation.
//! 3. Burden & Faires (2010). *Numerical Analysis* (9th ed.), Ch. 3: Interpolation.
//!
//! # Author
//!
//! Ryan Clanton (@ryancinsight)
//! Sprint 217 Session 9 - k-Wave Gap Analysis

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::grid::Grid;
use ndarray::{Array1, Array2, ArrayView3};
use serde::{Deserialize, Serialize};

/// Point sensor configuration
///
/// # Validation
///
/// - All sensor locations must be within the grid domain
/// - Locations must be finite (no NaN, Inf)
/// - At least one sensor location required
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointSensorConfig {
    /// Sensor locations in physical coordinates [m]
    ///
    /// Each location is [x, y, z] in meters.
    pub locations: Vec<[f64; 3]>,
}

impl PointSensorConfig {
    /// Create point sensor configuration
    pub fn new(locations: Vec<[f64; 3]>) -> Self {
        Self { locations }
    }

    /// Validate configuration against grid
    fn validate(&self, grid: &Grid) -> KwaversResult<()> {
        if self.locations.is_empty() {
            return Err(KwaversError::Validation(
                crate::core::error::validation::ValidationError::ConstraintViolation {
                    message: "At least one sensor location required".to_string(),
                },
            ));
        }

        let (nx, ny, nz) = grid.dimensions();
        let (dx, dy, dz) = grid.spacing();
        let domain_size = [nx as f64 * dx, ny as f64 * dy, nz as f64 * dz];

        for (idx, &[x, y, z]) in self.locations.iter().enumerate() {
            // Check finite
            if !x.is_finite() || !y.is_finite() || !z.is_finite() {
                return Err(KwaversError::Validation(
                    crate::core::error::validation::ValidationError::ConstraintViolation {
                        message: format!(
                            "Sensor {} location must be finite: [{}, {}, {}]",
                            idx, x, y, z
                        ),
                    },
                ));
            }

            // Check within domain
            if x < 0.0
                || x > domain_size[0]
                || y < 0.0
                || y > domain_size[1]
                || z < 0.0
                || z > domain_size[2]
            {
                return Err(KwaversError::Validation(
                    crate::core::error::validation::ValidationError::ConstraintViolation {
                        message: format!(
                            "Sensor {} location [{}, {}, {}] outside domain [0, 0, 0] to [{}, {}, {}]",
                            idx, x, y, z, domain_size[0], domain_size[1], domain_size[2]
                        ),
                    },
                ));
            }
        }

        Ok(())
    }
}

/// Grid indices and interpolation weights for a point sensor
///
/// # Mathematical Representation
///
/// For a sensor at (x, y, z), we store:
/// - (i, j, k): Lower corner grid indices
/// - (ξ, η, ζ): Local coordinates in [0, 1]
///
/// The 8 surrounding grid points are:
/// ```text
/// (i,j,k), (i+1,j,k), (i,j+1,k), (i+1,j+1,k),
/// (i,j,k+1), (i+1,j,k+1), (i,j+1,k+1), (i+1,j+1,k+1)
/// ```
#[derive(Debug, Clone)]
struct InterpolationData {
    /// Lower corner grid indices (i, j, k)
    indices: [usize; 3],
    /// Local coordinates (ξ, η, ζ) in [0, 1]
    local_coords: [f64; 3],
}

impl InterpolationData {
    /// Compute trilinear interpolation weights
    ///
    /// Returns 8 weights corresponding to the 8 surrounding grid points.
    ///
    /// # Mathematical Specification
    ///
    /// Given local coordinates (ξ, η, ζ), the 8 weights are:
    /// ```text
    /// w[0] = (1-ξ)(1-η)(1-ζ)   [i,   j,   k  ]
    /// w[1] =    ξ (1-η)(1-ζ)   [i+1, j,   k  ]
    /// w[2] = (1-ξ)   η (1-ζ)   [i,   j+1, k  ]
    /// w[3] =    ξ    η (1-ζ)   [i+1, j+1, k  ]
    /// w[4] = (1-ξ)(1-η)   ζ    [i,   j,   k+1]
    /// w[5] =    ξ (1-η)   ζ    [i+1, j,   k+1]
    /// w[6] = (1-ξ)   η    ζ    [i,   j+1, k+1]
    /// w[7] =    ξ    η    ζ    [i+1, j+1, k+1]
    /// ```
    fn weights(&self) -> [f64; 8] {
        let [xi, eta, zeta] = self.local_coords;
        let one_minus_xi = 1.0 - xi;
        let one_minus_eta = 1.0 - eta;
        let one_minus_zeta = 1.0 - zeta;

        [
            one_minus_xi * one_minus_eta * one_minus_zeta, // [i,   j,   k  ]
            xi * one_minus_eta * one_minus_zeta,           // [i+1, j,   k  ]
            one_minus_xi * eta * one_minus_zeta,           // [i,   j+1, k  ]
            xi * eta * one_minus_zeta,                     // [i+1, j+1, k  ]
            one_minus_xi * one_minus_eta * zeta,           // [i,   j,   k+1]
            xi * one_minus_eta * zeta,                     // [i+1, j,   k+1]
            one_minus_xi * eta * zeta,                     // [i,   j+1, k+1]
            xi * eta * zeta,                               // [i+1, j+1, k+1]
        ]
    }

    /// Interpolate field value at sensor location
    ///
    /// # Mathematical Operation
    ///
    /// ```text
    /// p_sensor = Σₖ wₖ · p[indices[k]]                   (4)
    /// ```
    fn interpolate(&self, field: ArrayView3<f64>) -> f64 {
        let [i, j, k] = self.indices;
        let weights = self.weights();
        let shape = field.shape();

        // Clamp indices to valid range (handles boundary cases)
        let i1 = (i + 1).min(shape[0] - 1);
        let j1 = (j + 1).min(shape[1] - 1);
        let k1 = (k + 1).min(shape[2] - 1);

        // Trilinear interpolation with 8 grid points
        weights[0] * field[[i, j, k]]
            + weights[1] * field[[i1, j, k]]
            + weights[2] * field[[i, j1, k]]
            + weights[3] * field[[i1, j1, k]]
            + weights[4] * field[[i, j, k1]]
            + weights[5] * field[[i1, j, k1]]
            + weights[6] * field[[i, j1, k1]]
            + weights[7] * field[[i1, j1, k1]]
    }
}

/// Point sensor for sampling acoustic fields at arbitrary locations
///
/// # Data Storage
///
/// - **Locations**: Physical coordinates (x, y, z) [m] for each sensor
/// - **Interpolation Data**: Precomputed grid indices and local coordinates
/// - **Time History**: 2D array [n_sensors × n_timesteps] of recorded values
///
/// # Performance
///
/// Interpolation data is precomputed during initialization, so recording
/// at each time step requires only:
/// - 8 memory reads (grid values)
/// - 8 multiplications + 7 additions (weight computation)
/// - O(1) time complexity per sensor
///
/// For N sensors and T timesteps: O(NT) total cost.
#[derive(Debug, Clone)]
pub struct PointSensor {
    /// Sensor locations [m]
    locations: Vec<[f64; 3]>,
    /// Precomputed interpolation data for each sensor
    interp_data: Vec<InterpolationData>,
    /// Time history [n_sensors × n_timesteps]
    time_history: Vec<Vec<f64>>,
    /// Number of recorded timesteps
    n_timesteps: usize,
}

impl PointSensor {
    /// Create new point sensor
    ///
    /// # Arguments
    ///
    /// * `config` - Sensor configuration with locations
    /// * `grid` - Computational grid for coordinate mapping
    ///
    /// # Returns
    ///
    /// Point sensor instance with precomputed interpolation data
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - No sensor locations provided
    /// - Any location is outside grid domain
    /// - Any location has non-finite coordinates
    pub fn new(config: PointSensorConfig, grid: &Grid) -> KwaversResult<Self> {
        config.validate(grid)?;

        let n_sensors = config.locations.len();
        let (dx, dy, dz) = grid.spacing();

        // Precompute interpolation data for all sensors
        let interp_data: Vec<InterpolationData> = config
            .locations
            .iter()
            .map(|&[x, y, z]| {
                // Convert physical coordinates to grid indices (fractional)
                let fi = x / dx;
                let fj = y / dy;
                let fk = z / dz;

                // Lower corner indices
                let i = fi.floor() as usize;
                let j = fj.floor() as usize;
                let k = fk.floor() as usize;

                // Local coordinates in [0, 1]
                let xi = fi - (i as f64);
                let eta = fj - (j as f64);
                let zeta = fk - (k as f64);

                InterpolationData {
                    indices: [i, j, k],
                    local_coords: [xi, eta, zeta],
                }
            })
            .collect();

        // Initialize time history storage
        let time_history = vec![Vec::new(); n_sensors];

        Ok(Self {
            locations: config.locations,
            interp_data,
            time_history,
            n_timesteps: 0,
        })
    }

    /// Record field values at all sensor locations for current timestep
    ///
    /// # Mathematical Operation
    ///
    /// For each sensor s:
    /// ```text
    /// history[s][t] = interpolate(field, location[s])   (5)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `field` - Current pressure (or other scalar) field
    /// * `_grid` - Grid reference (currently unused, kept for API consistency)
    /// * `_time_step` - Current time step index (currently unused)
    pub fn record(&mut self, field: ArrayView3<f64>, _grid: &Grid, _time_step: usize) {
        for (sensor_idx, interp) in self.interp_data.iter().enumerate() {
            let value = interp.interpolate(field);
            self.time_history[sensor_idx].push(value);
        }
        self.n_timesteps += 1;
    }

    /// Get time history for specific sensor
    ///
    /// # Arguments
    ///
    /// * `sensor_idx` - Sensor index (0-based)
    ///
    /// # Returns
    ///
    /// Time history as 1D array, or `None` if sensor index invalid
    pub fn time_history(&self, sensor_idx: usize) -> Option<Array1<f64>> {
        self.time_history
            .get(sensor_idx)
            .map(|history| Array1::from_vec(history.clone()))
    }

    /// Get all time histories as 2D array [n_sensors × n_timesteps]
    pub fn all_time_histories(&self) -> Array2<f64> {
        let n_sensors = self.locations.len();
        let n_timesteps = self.n_timesteps;

        let mut histories = Array2::<f64>::zeros((n_sensors, n_timesteps));
        for (i, history) in self.time_history.iter().enumerate() {
            for (j, &value) in history.iter().enumerate() {
                histories[[i, j]] = value;
            }
        }
        histories
    }

    /// Get sensor locations [m]
    pub fn locations(&self) -> &[[f64; 3]] {
        &self.locations
    }

    /// Get number of sensors
    pub fn n_sensors(&self) -> usize {
        self.locations.len()
    }

    /// Get number of recorded timesteps
    pub fn n_timesteps(&self) -> usize {
        self.n_timesteps
    }

    /// Clear recorded time history (reset for new simulation)
    pub fn clear(&mut self) {
        for history in &mut self.time_history {
            history.clear();
        }
        self.n_timesteps = 0;
    }

    /// Get maximum recorded pressure at specific sensor
    ///
    /// # Returns
    ///
    /// Maximum absolute pressure value, or `None` if no data or invalid index
    pub fn max_pressure(&self, sensor_idx: usize) -> Option<f64> {
        self.time_history.get(sensor_idx).and_then(|history| {
            history
                .iter()
                .map(|v| v.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
        })
    }

    /// Get RMS pressure at specific sensor
    ///
    /// # Mathematical Formula
    ///
    /// ```text
    /// p_rms = √(1/N Σᵢ p[i]²)                            (6)
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

    /// Export time history to CSV format
    ///
    /// # Format
    ///
    /// ```text
    /// time, sensor_0, sensor_1, ..., sensor_N
    /// 0.0,  p[0,0],   p[1,0],   ..., p[N,0]
    /// dt,   p[0,1],   p[1,1],   ..., p[N,1]
    /// ...
    /// ```
    pub fn to_csv(&self, dt: f64) -> String {
        let mut csv = String::new();

        // Header
        csv.push_str("time");
        for i in 0..self.n_sensors() {
            csv.push_str(&format!(",sensor_{}", i));
        }
        csv.push('\n');

        // Data rows
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array3;

    fn create_test_grid() -> Grid {
        Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap() // 32×32×32, 1mm spacing
    }

    #[test]
    fn test_point_sensor_creation() {
        let grid = create_test_grid();
        let locations = vec![[0.016, 0.016, 0.016]]; // Center of domain

        let config = PointSensorConfig::new(locations);
        let sensor = PointSensor::new(config, &grid).unwrap();

        assert_eq!(sensor.n_sensors(), 1);
        assert_eq!(sensor.n_timesteps(), 0);
    }

    #[test]
    fn test_point_sensor_validation() {
        let grid = create_test_grid();

        // Location outside domain should fail
        let locations = vec![[0.100, 0.016, 0.016]]; // x > domain size
        let config = PointSensorConfig::new(locations);
        assert!(PointSensor::new(config, &grid).is_err());

        // Non-finite location should fail
        let locations = vec![[f64::NAN, 0.016, 0.016]];
        let config = PointSensorConfig::new(locations);
        assert!(PointSensor::new(config, &grid).is_err());

        // Empty locations should fail
        let config = PointSensorConfig::new(vec![]);
        assert!(PointSensor::new(config, &grid).is_err());
    }

    #[test]
    fn test_trilinear_interpolation_at_grid_point() {
        let grid = create_test_grid();
        let field = Array3::<f64>::from_shape_fn((32, 32, 32), |(i, j, k)| (i + j + k) as f64);

        // Sensor exactly at grid point (16, 16, 16)
        let locations = vec![[0.016, 0.016, 0.016]]; // Corresponds to grid index (16,16,16)
        let config = PointSensorConfig::new(locations);
        let mut sensor = PointSensor::new(config, &grid).unwrap();

        sensor.record(field.view(), &grid, 0);

        let history = sensor.time_history(0).unwrap();
        let expected = field[[16, 16, 16]];
        assert_abs_diff_eq!(history[0], expected, epsilon = 1e-12);
    }

    #[test]
    fn test_trilinear_interpolation_midpoint() {
        let grid = create_test_grid();

        // Create field where p[i,j,k] = i + 10*j + 100*k
        let field = Array3::<f64>::from_shape_fn((32, 32, 32), |(i, j, k)| {
            (i as f64) + 10.0 * (j as f64) + 100.0 * (k as f64)
        });

        // Sensor at exact midpoint between grid points (10, 10, 10) and (11, 11, 11)
        // Physical location: (10.5 * dx, 10.5 * dy, 10.5 * dz)
        let locations = vec![[0.0105, 0.0105, 0.0105]];
        let config = PointSensorConfig::new(locations);
        let mut sensor = PointSensor::new(config, &grid).unwrap();

        sensor.record(field.view(), &grid, 0);

        // At midpoint with ξ=η=ζ=0.5, all 8 corners have equal weight (1/8)
        // Expected value is average of 8 corners
        let corners = [
            field[[10, 10, 10]],
            field[[11, 10, 10]],
            field[[10, 11, 10]],
            field[[11, 11, 10]],
            field[[10, 10, 11]],
            field[[11, 10, 11]],
            field[[10, 11, 11]],
            field[[11, 11, 11]],
        ];
        let expected: f64 = corners.iter().sum::<f64>() / 8.0;

        let history = sensor.time_history(0).unwrap();
        assert_abs_diff_eq!(history[0], expected, epsilon = 1e-9);
    }

    #[test]
    fn test_multiple_sensors() {
        let grid = create_test_grid();
        let locations = vec![
            [0.016, 0.016, 0.016], // Sensor 0
            [0.020, 0.016, 0.016], // Sensor 1
            [0.016, 0.020, 0.016], // Sensor 2
        ];

        let config = PointSensorConfig::new(locations);
        let mut sensor = PointSensor::new(config, &grid).unwrap();

        assert_eq!(sensor.n_sensors(), 3);

        // Record multiple timesteps
        for t in 0..10 {
            let field = Array3::<f64>::from_elem((32, 32, 32), (t as f64) * 10.0);
            sensor.record(field.view(), &grid, t);
        }

        assert_eq!(sensor.n_timesteps(), 10);

        // Check all sensors recorded
        for i in 0..3 {
            let history = sensor.time_history(i).unwrap();
            assert_eq!(history.len(), 10);
        }
    }

    #[test]
    fn test_time_history_recording() {
        let grid = create_test_grid();
        let locations = vec![[0.016, 0.016, 0.016]];
        let config = PointSensorConfig::new(locations);
        let mut sensor = PointSensor::new(config, &grid).unwrap();

        // Record sinusoidal time signal
        let omega = 2.0 * std::f64::consts::PI * 1e6; // 1 MHz
        let dt = 1e-7; // 0.1 μs

        for t in 0..100 {
            let time = (t as f64) * dt;
            let pressure = (omega * time).sin();
            let field = Array3::<f64>::from_elem((32, 32, 32), pressure);
            sensor.record(field.view(), &grid, t);
        }

        let history = sensor.time_history(0).unwrap();
        assert_eq!(history.len(), 100);

        // Check first few values
        for t in 0..5 {
            let time = (t as f64) * dt;
            let expected = (omega * time).sin();
            assert_abs_diff_eq!(history[t], expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_max_and_rms_pressure() {
        let grid = create_test_grid();
        let locations = vec![[0.016, 0.016, 0.016]];
        let config = PointSensorConfig::new(locations);
        let mut sensor = PointSensor::new(config, &grid).unwrap();

        // Record known values
        let values = vec![1.0, -3.0, 2.0, -1.0, 0.0];
        for &val in &values {
            let field = Array3::<f64>::from_elem((32, 32, 32), val);
            sensor.record(field.view(), &grid, 0);
        }

        // Max absolute value should be 3.0
        let max_p = sensor.max_pressure(0).unwrap();
        assert_abs_diff_eq!(max_p, 3.0, epsilon = 1e-12);

        // RMS = sqrt((1 + 9 + 4 + 1 + 0)/5) = sqrt(15/5) = sqrt(3)
        let rms_p = sensor.rms_pressure(0).unwrap();
        let expected_rms = (15.0_f64 / 5.0).sqrt();
        assert_abs_diff_eq!(rms_p, expected_rms, epsilon = 1e-12);
    }

    #[test]
    fn test_clear_history() {
        let grid = create_test_grid();
        let locations = vec![[0.016, 0.016, 0.016]];
        let config = PointSensorConfig::new(locations);
        let mut sensor = PointSensor::new(config, &grid).unwrap();

        // Record some data
        for t in 0..10 {
            let field = Array3::<f64>::from_elem((32, 32, 32), t as f64);
            sensor.record(field.view(), &grid, t);
        }

        assert_eq!(sensor.n_timesteps(), 10);

        // Clear
        sensor.clear();

        assert_eq!(sensor.n_timesteps(), 0);
        assert_eq!(sensor.time_history(0).unwrap().len(), 0);
    }

    #[test]
    fn test_csv_export() {
        let grid = create_test_grid();
        let locations = vec![[0.016, 0.016, 0.016], [0.020, 0.016, 0.016]];
        let config = PointSensorConfig::new(locations);
        let mut sensor = PointSensor::new(config, &grid).unwrap();

        // Record 3 timesteps
        for t in 0..3 {
            let field = Array3::<f64>::from_elem((32, 32, 32), (t as f64) * 10.0);
            sensor.record(field.view(), &grid, t);
        }

        let dt = 1e-7;
        let csv = sensor.to_csv(dt);

        // Check header
        assert!(csv.contains("time,sensor_0,sensor_1"));

        // Check first data row
        assert!(csv.contains("0.000000e0,0.000000e0,0.000000e0"));
    }

    #[test]
    fn test_all_time_histories() {
        let grid = create_test_grid();
        let locations = vec![[0.016, 0.016, 0.016], [0.020, 0.016, 0.016]];
        let config = PointSensorConfig::new(locations);
        let mut sensor = PointSensor::new(config, &grid).unwrap();

        // Record 5 timesteps
        for t in 0..5 {
            let field = Array3::<f64>::from_elem((32, 32, 32), (t as f64) * 10.0);
            sensor.record(field.view(), &grid, t);
        }

        let histories = sensor.all_time_histories();
        assert_eq!(histories.shape(), &[2, 5]); // 2 sensors, 5 timesteps

        // Check values
        for t in 0..5 {
            let expected = (t as f64) * 10.0;
            assert_abs_diff_eq!(histories[[0, t]], expected, epsilon = 1e-12);
            assert_abs_diff_eq!(histories[[1, t]], expected, epsilon = 1e-12);
        }
    }
}
