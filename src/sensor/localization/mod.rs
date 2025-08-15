//! Multi-lateration localization system for cavitation detection
//!
//! Implements trilateration and multilateration algorithms for localizing
//! cavitation events using time-of-arrival (TOA) or time-difference-of-arrival (TDOA)
//! measurements from multiple sensors.
//!
//! # Literature References
//!
//! 1. **Gyöngy, M., & Coussios, C. C. (2010)**. "Passive cavitation mapping for 
//!    localization and tracking of bubble activity during high intensity focused 
//!    ultrasound exposure." *Journal of the Acoustical Society of America*, 128(4), 
//!    EL175-EL180. DOI: 10.1121/1.3467491
//!
//! 2. **Foy, W. H. (1976)**. "Position-location solutions by Taylor-series estimation."
//!    *IEEE Transactions on Aerospace and Electronic Systems*, AES-12(2), 187-194.
//!    DOI: 10.1109/TAES.1976.308294
//!
//! 3. **Chan, Y. T., & Ho, K. C. (1994)**. "A simple and efficient estimator for 
//!    hyperbolic location." *IEEE Transactions on Signal Processing*, 42(8), 1905-1915.
//!    DOI: 10.1109/78.301830
//!
//! 4. **Caffery, J. J. (2000)**. "Wireless Location in CDMA Cellular Radio Systems."
//!    Kluwer Academic Publishers. ISBN: 978-0792379690

use crate::error::{KwaversResult, KwaversError, PhysicsError};
use crate::constants::physics::SOUND_SPEED_WATER;
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

pub mod trilateration;
pub mod tdoa;
pub mod calibration;
pub mod phantom;

pub use trilateration::{TrilaterationSolver, TrilaterationResult};
pub use tdoa::TDOASolver;
pub use calibration::{SensorCalibration, CalibrationPhantom};
pub use phantom::{CentroidPhantom, PhantomTarget};

/// Time Difference of Arrival measurement
#[derive(Debug, Clone)]
pub struct TDOAMeasurement {
    /// Sensor pair (reference sensor, measurement sensor)
    pub sensor_pair: (usize, usize),
    /// Time difference (t_measurement - t_reference) in seconds
    pub time_difference: f64,
    /// Measurement uncertainty in seconds
    pub uncertainty: f64,
}

/// Sensor configuration for localization
#[derive(Debug, Clone)]
pub struct Sensor {
    /// Unique sensor identifier
    pub id: usize,
    /// Sensor position [x, y, z] in meters
    pub position: [f64; 3],
    /// Sensor calibration data
    pub calibration: SensorCalibration,
    /// Sensor sensitivity [V/Pa]
    pub sensitivity: f64,
    /// Sensor bandwidth [Hz]
    pub bandwidth: (f64, f64),
}

impl Sensor {
    /// Create a new sensor
    pub fn new(id: usize, position: [f64; 3]) -> Self {
        Self {
            id,
            position,
            calibration: SensorCalibration::default(),
            sensitivity: 1.0,
            bandwidth: (20e3, 10e6), // 20 kHz to 10 MHz typical for cavitation
        }
    }
    
    /// Distance to a point
    pub fn distance_to(&self, point: &[f64; 3]) -> f64 {
        let dx = self.position[0] - point[0];
        let dy = self.position[1] - point[1];
        let dz = self.position[2] - point[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
    
    /// Time of flight from a point
    pub fn time_of_flight(&self, point: &[f64; 3], sound_speed: f64) -> f64 {
        self.distance_to(point) / sound_speed
    }
}

/// Sensor array for localization
#[derive(Debug, Clone)]
pub struct SensorArray {
    /// Collection of sensors
    sensors: Vec<Sensor>,
    /// Sound speed in medium [m/s]
    sound_speed: f64,
    /// Array geometry type
    geometry: ArrayGeometry,
}

/// Array geometry types
#[derive(Debug, Clone, PartialEq)]
pub enum ArrayGeometry {
    /// Linear array
    Linear,
    /// Planar array
    Planar,
    /// Volumetric array
    Volumetric,
    /// Spherical array
    Spherical,
    /// Arbitrary positions
    Arbitrary,
}

impl SensorArray {
    /// Create a new sensor array
    pub fn new(sensors: Vec<Sensor>, sound_speed: f64) -> Self {
        let geometry = Self::detect_geometry(&sensors);
        Self {
            sensors,
            sound_speed,
            geometry,
        }
    }
    
    /// Detect array geometry from sensor positions
    fn detect_geometry(sensors: &[Sensor]) -> ArrayGeometry {
        if sensors.len() < 2 {
            return ArrayGeometry::Arbitrary;
        }
        
        // Check if all sensors are on a line (linear array)
        let linear_tolerance = 1e-3;
        let first = &sensors[0].position;
        let second = &sensors[1].position;
        
        // Direction vector
        let dir = [
            second[0] - first[0],
            second[1] - first[1],
            second[2] - first[2],
        ];
        let dir_norm = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        
        if dir_norm > 0.0 {
            let dir_unit = [dir[0] / dir_norm, dir[1] / dir_norm, dir[2] / dir_norm];
            
            let all_linear = sensors.iter().skip(2).all(|s| {
                let to_sensor = [
                    s.position[0] - first[0],
                    s.position[1] - first[1],
                    s.position[2] - first[2],
                ];
                
                // Project onto direction
                let projection = to_sensor[0] * dir_unit[0] 
                    + to_sensor[1] * dir_unit[1] 
                    + to_sensor[2] * dir_unit[2];
                
                // Check perpendicular distance
                let perp_x = to_sensor[0] - projection * dir_unit[0];
                let perp_y = to_sensor[1] - projection * dir_unit[1];
                let perp_z = to_sensor[2] - projection * dir_unit[2];
                let perp_dist = (perp_x * perp_x + perp_y * perp_y + perp_z * perp_z).sqrt();
                
                perp_dist < linear_tolerance
            });
            
            if all_linear {
                return ArrayGeometry::Linear;
            }
        }
        
        // Check if all sensors are on a plane (planar array)
        if sensors.len() >= 3 {
            // Use first three non-collinear points to define plane
            // This is a simplified check
            let planar_tolerance = 1e-3;
            let all_planar = sensors.iter().all(|s| {
                // Simplified: check if all z-coordinates are similar
                (s.position[2] - first[2]).abs() < planar_tolerance
            });
            
            if all_planar {
                return ArrayGeometry::Planar;
            }
        }
        
        // Check for spherical arrangement
        let center = Self::calculate_centroid(sensors);
        let first_radius = Self::distance(&sensors[0].position, &center);
        let spherical_tolerance = 1e-2;
        
        let all_spherical = sensors.iter().all(|s| {
            let radius = Self::distance(&s.position, &center);
            (radius - first_radius).abs() < spherical_tolerance
        });
        
        if all_spherical {
            return ArrayGeometry::Spherical;
        }
        
        // Otherwise volumetric or arbitrary
        ArrayGeometry::Volumetric
    }
    
    /// Calculate centroid of sensor positions
    fn calculate_centroid(sensors: &[Sensor]) -> [f64; 3] {
        let n = sensors.len() as f64;
        let sum = sensors.iter().fold([0.0, 0.0, 0.0], |acc, s| {
            [
                acc[0] + s.position[0],
                acc[1] + s.position[1],
                acc[2] + s.position[2],
            ]
        });
        [sum[0] / n, sum[1] / n, sum[2] / n]
    }
    
    /// Calculate distance between two points
    fn distance(p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
        let dx = p1[0] - p2[0];
        let dy = p1[1] - p2[1];
        let dz = p1[2] - p2[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Multi-lateration solver
pub struct MultiLaterationSolver {
    /// Sensor array
    array: SensorArray,
    /// Solver configuration
    config: SolverConfig,
    /// Calibration data
    calibration: Option<CalibrationData>,
}

/// Solver configuration
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Maximum iterations for iterative solvers
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Use weighted least squares
    pub use_weighting: bool,
    /// Regularization parameter
    pub regularization: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            use_weighting: true,
            regularization: 1e-10,
        }
    }
}

/// Calibration data for the system
#[derive(Debug, Clone)]
pub struct CalibrationData {
    /// Sensor position corrections
    pub position_corrections: HashMap<usize, [f64; 3]>,
    /// Time delay corrections
    pub time_delays: HashMap<usize, f64>,
    /// Sound speed correction factor
    pub sound_speed_factor: f64,
}

impl MultiLaterationSolver {
    /// Create a new multi-lateration solver
    pub fn new(array: SensorArray) -> Self {
        Self {
            array,
            config: SolverConfig::default(),
            calibration: None,
        }
    }
    
    /// Set solver configuration
    pub fn with_config(mut self, config: SolverConfig) -> Self {
        self.config = config;
        self
    }
    
    /// Calibrate using phantom targets
    pub fn calibrate_with_phantom(
        &mut self,
        phantom: &CentroidPhantom,
        measurements: &[PhantomMeasurement],
    ) -> KwaversResult<CalibrationData> {
        let calibrator = calibration::Calibrator::new(&self.array);
        let calib_data = calibrator.calibrate(phantom, measurements)?;
        self.calibration = Some(calib_data.clone());
        Ok(calib_data)
    }
    
    /// Localize using time-of-arrival measurements
    pub fn localize_toa(
        &self,
        arrival_times: &HashMap<usize, f64>,
    ) -> KwaversResult<LocalizationResult> {
        // Need at least 4 sensors for 3D localization
        if arrival_times.len() < 4 {
            return Err(KwaversError::Physics(PhysicsError::InvalidConfiguration {
                component: "TOA localization".to_string(),
                reason: "Need at least 4 sensors for 3D localization".to_string(),
            }));
        }
        
        // Apply calibration if available
        let corrected_times = self.apply_time_calibration(arrival_times);
        
        // Use iterative least squares solver
        let initial_guess = self.compute_initial_guess(&corrected_times)?;
        let solution = self.iterative_least_squares_toa(initial_guess, &corrected_times)?;
        
        Ok(LocalizationResult {
            position: solution,
            uncertainty: self.estimate_uncertainty(&solution, &corrected_times)?,
            residual: self.compute_residual(&solution, &corrected_times),
            num_sensors: arrival_times.len(),
        })
    }
    
    /// Localize using time-difference-of-arrival measurements
    pub fn localize_tdoa(
        &self,
        time_differences: &[TDOAMeasurement],
    ) -> KwaversResult<LocalizationResult> {
        // Need at least 3 TDOA measurements for 3D localization
        if time_differences.len() < 3 {
            return Err(KwaversError::Physics(PhysicsError::InvalidConfiguration {
                component: "TDOA localization".to_string(),
                reason: "Need at least 3 measurements for 3D localization".to_string(),
            }));
        }
        
        let solver = tdoa::TDOASolver::new(&self.array);
        let solution = solver.solve(time_differences)?;
        
        Ok(LocalizationResult {
            position: solution.position,
            uncertainty: solution.uncertainty,
            residual: solution.residual,
            num_sensors: time_differences.len() + 1,
        })
    }
    
    /// Apply time calibration corrections
    fn apply_time_calibration(&self, times: &HashMap<usize, f64>) -> HashMap<usize, f64> {
        if let Some(calib) = &self.calibration {
            times.iter().map(|(id, t)| {
                let correction = calib.time_delays.get(id).unwrap_or(&0.0);
                (*id, t - correction)
            }).collect()
        } else {
            times.clone()
        }
    }
    
    /// Compute initial guess for position
    fn compute_initial_guess(&self, times: &HashMap<usize, f64>) -> KwaversResult<[f64; 3]> {
        // Use centroid of sensors as initial guess
        let sensors: Vec<Sensor> = times.keys()
            .filter_map(|id| self.array.sensors.iter().find(|s| s.id == *id).cloned())
            .collect();
        
        if sensors.is_empty() {
            return Err(KwaversError::Physics(PhysicsError::InvalidConfiguration {
                component: "localization".to_string(),
                reason: "No valid sensors found".to_string(),
            }));
        }
        
        Ok(SensorArray::calculate_centroid(&sensors))
    }
    
    /// Iterative least squares solver for TOA
    fn iterative_least_squares_toa(
        &self,
        mut position: [f64; 3],
        times: &HashMap<usize, f64>,
    ) -> KwaversResult<[f64; 3]> {
        for _ in 0..self.config.max_iterations {
            let (jacobian, residuals) = self.build_toa_system(&position, times)?;
            
            // Solve: J^T * J * delta = -J^T * r
            let jt = jacobian.transpose();
            let jtj = &jt * &jacobian;
            
            // Add regularization
            let mut jtj_reg = jtj.clone();
            for i in 0..3 {
                jtj_reg[(i, i)] += self.config.regularization;
            }
            
            let jtr = &jt * &residuals;
            
            // Solve for position update
            if let Some(delta) = jtj_reg.lu().solve(&(-jtr)) {
                position[0] += delta[0];
                position[1] += delta[1];
                position[2] += delta[2];
                
                // Check convergence
                let delta_norm = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
                if delta_norm < self.config.tolerance {
                    break;
                }
            } else {
                return Err(KwaversError::Physics(PhysicsError::InvalidState {
                    field: "linear_system".to_string(),
                    value: "singular".to_string(),
                    reason: "Failed to solve linear system in localization".to_string(),
                }));
            }
        }
        
        Ok(position)
    }
    
    /// Build Jacobian and residual for TOA system
    fn build_toa_system(
        &self,
        position: &[f64; 3],
        times: &HashMap<usize, f64>,
    ) -> KwaversResult<(DMatrix<f64>, DVector<f64>)> {
        let n = times.len();
        let mut jacobian = DMatrix::zeros(n, 3);
        let mut residuals = DVector::zeros(n);
        
        for (i, (sensor_id, measured_time)) in times.iter().enumerate() {
            let sensor = self.array.sensors.iter()
                .find(|s| s.id == *sensor_id)
                .ok_or_else(|| KwaversError::Physics(PhysicsError::InvalidConfiguration {
                    component: "sensor_array".to_string(),
                    reason: format!("Sensor {} not found", sensor_id),
                }))?;
            
            let distance = sensor.distance_to(position);
            let predicted_time = distance / self.array.sound_speed;
            
            // Residual
            residuals[i] = measured_time - predicted_time;
            
            // Jacobian entries: partial derivatives of time with respect to position
            if distance > 1e-10 {
                jacobian[(i, 0)] = -(position[0] - sensor.position[0]) / (distance * self.array.sound_speed);
                jacobian[(i, 1)] = -(position[1] - sensor.position[1]) / (distance * self.array.sound_speed);
                jacobian[(i, 2)] = -(position[2] - sensor.position[2]) / (distance * self.array.sound_speed);
            }
        }
        
        Ok((jacobian, residuals))
    }
    
    /// Estimate position uncertainty
    fn estimate_uncertainty(
        &self,
        position: &[f64; 3],
        times: &HashMap<usize, f64>,
    ) -> KwaversResult<[f64; 3]> {
        let (jacobian, residuals) = self.build_toa_system(position, times)?;
        
        // Estimate measurement noise variance
        let n = residuals.len();
        let sigma_squared = if n > 3 {
            residuals.dot(&residuals) / (n - 3) as f64
        } else {
            1e-12 // Default small variance
        };
        
        // Covariance matrix: (J^T * J)^(-1) * sigma^2
        let jt = jacobian.transpose();
        let jtj = &jt * &jacobian;
        
        if let Some(cov) = jtj.lu().try_inverse() {
            let cov_scaled = cov * sigma_squared;
            Ok([
                cov_scaled[(0, 0)].sqrt(),
                cov_scaled[(1, 1)].sqrt(),
                cov_scaled[(2, 2)].sqrt(),
            ])
        } else {
            Ok([1e-3, 1e-3, 1e-3]) // Default uncertainty
        }
    }
    
    /// Compute residual error
    fn compute_residual(&self, position: &[f64; 3], times: &HashMap<usize, f64>) -> f64 {
        times.iter().map(|(sensor_id, measured_time)| {
            if let Some(sensor) = self.array.sensors.iter().find(|s| s.id == *sensor_id) {
                let predicted_time = sensor.distance_to(position) / self.array.sound_speed;
                (measured_time - predicted_time).powi(2)
            } else {
                0.0
            }
        }).sum::<f64>().sqrt()
    }
}

/// Localization result
#[derive(Debug, Clone)]
pub struct LocalizationResult {
    /// Estimated position [x, y, z] in meters
    pub position: [f64; 3],
    /// Position uncertainty (standard deviation) [σx, σy, σz] in meters
    pub uncertainty: [f64; 3],
    /// Residual error
    pub residual: f64,
    /// Number of sensors used
    pub num_sensors: usize,
}

/// Phantom measurement for calibration
#[derive(Debug, Clone)]
pub struct PhantomMeasurement {
    /// Known phantom position
    pub phantom_position: [f64; 3],
    /// Measured arrival times at each sensor
    pub arrival_times: HashMap<usize, f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sensor_array_geometry_detection() {
        // Linear array
        let linear_sensors = vec![
            Sensor::new(0, [0.0, 0.0, 0.0]),
            Sensor::new(1, [0.1, 0.0, 0.0]),
            Sensor::new(2, [0.2, 0.0, 0.0]),
            Sensor::new(3, [0.3, 0.0, 0.0]),
        ];
        let linear_array = SensorArray::new(linear_sensors, SOUND_SPEED_WATER);
        assert_eq!(linear_array.geometry, ArrayGeometry::Linear);
        
        // Planar array
        let planar_sensors = vec![
            Sensor::new(0, [0.0, 0.0, 0.0]),
            Sensor::new(1, [0.1, 0.0, 0.0]),
            Sensor::new(2, [0.0, 0.1, 0.0]),
            Sensor::new(3, [0.1, 0.1, 0.0]),
        ];
        let planar_array = SensorArray::new(planar_sensors, SOUND_SPEED_WATER);
        assert_eq!(planar_array.geometry, ArrayGeometry::Planar);
        
        // Volumetric array
        let volume_sensors = vec![
            Sensor::new(0, [0.0, 0.0, 0.0]),
            Sensor::new(1, [0.1, 0.0, 0.0]),
            Sensor::new(2, [0.0, 0.1, 0.0]),
            Sensor::new(3, [0.0, 0.0, 0.1]),
        ];
        let volume_array = SensorArray::new(volume_sensors, SOUND_SPEED_WATER);
        assert_eq!(volume_array.geometry, ArrayGeometry::Volumetric);
    }
    
    #[test]
    fn test_toa_localization() {
        // Create a simple sensor array
        let sensors = vec![
            Sensor::new(0, [0.0, 0.0, 0.0]),
            Sensor::new(1, [0.1, 0.0, 0.0]),
            Sensor::new(2, [0.0, 0.1, 0.0]),
            Sensor::new(3, [0.0, 0.0, 0.1]),
        ];
        let array = SensorArray::new(sensors, SOUND_SPEED_WATER);
        
        // Create solver
        let solver = MultiLaterationSolver::new(array);
        
        // Simulate measurements from a known source
        let true_position = [0.05, 0.05, 0.05];
        let mut arrival_times = HashMap::new();
        
        for sensor in &solver.array.sensors {
            let tof = sensor.time_of_flight(&true_position, SOUND_SPEED_WATER);
            arrival_times.insert(sensor.id, tof);
        }
        
        // Localize
        let result = solver.localize_toa(&arrival_times).unwrap();
        
        // Check accuracy (should be very close to true position)
        assert!((result.position[0] - true_position[0]).abs() < 1e-6);
        assert!((result.position[1] - true_position[1]).abs() < 1e-6);
        assert!((result.position[2] - true_position[2]).abs() < 1e-6);
        assert!(result.residual < 1e-10);
    }
}