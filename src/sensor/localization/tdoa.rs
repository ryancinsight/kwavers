//! Time-difference-of-arrival (TDOA) solver

use crate::sensor::localization::{SensorArray, TDOAMeasurement, LocalizationResult};
use crate::error::KwaversResult;
use nalgebra::{DMatrix, DVector};

/// Time Difference of Arrival (TDOA) solver using Chan-Ho algorithm
/// 
/// Reference: Y.T. Chan and K.C. Ho, "A simple and efficient estimator for 
/// hyperbolic location," IEEE Trans. Signal Process., vol. 42, no. 8, 
/// pp. 1905-1915, Aug. 1994.
pub struct TDOASolver<'a> {
    /// Sensor array
    array: &'a SensorArray,
    /// Speed of sound
    sound_speed: f64,
}

impl<'a> TDOASolver<'a> {
    /// Create new TDOA solver
    pub fn new(array: &'a SensorArray) -> Self {
        Self {
            array,
            sound_speed: array.sound_speed(),
        }
    }
    
    /// Set sound speed
    pub fn with_sound_speed(mut self, speed: f64) -> Self {
        self.sound_speed = speed;
        self
    }
    
    /// Solve for position using TDOA measurements
    /// 
    /// Implements the Chan-Ho algorithm for hyperbolic positioning
    pub fn solve(&self, measurements: &[TDOAMeasurement]) -> KwaversResult<LocalizationResult> {
        let num_sensors = measurements.len() + 1; // +1 for reference sensor
        
        if num_sensors < 4 {
            return Err(crate::error::KwaversError::Physics(
                crate::error::PhysicsError::InvalidParameter {
                    component: "TDOA".to_string(),
                    reason: "TDOA requires at least 4 sensors for 3D localization".to_string()
                }
            ));
        }
        
        // Get sensor positions
        let positions = self.array.get_sensor_positions();
        
        // Reference sensor is at index 0
        let ref_pos = &positions[0];
        
        // Convert TDOA to range differences
        let mut range_diffs = Vec::new();
        for meas in measurements {
            let range_diff = meas.time_difference * self.sound_speed;
            range_diffs.push(range_diff);
        }
        
        // Build matrices for Chan-Ho algorithm
        // A matrix: linearized system
        let mut a_matrix = DMatrix::zeros(measurements.len(), 3);
        let mut b_vector = DVector::zeros(measurements.len());
        
        for (i, meas) in measurements.iter().enumerate() {
            let sensor_idx = meas.sensor_pair.1;
            let sensor_pos = &positions[sensor_idx];
            
            // Direction from reference to sensor
            let dx = sensor_pos[0] - ref_pos[0];
            let dy = sensor_pos[1] - ref_pos[1];
            let dz = sensor_pos[2] - ref_pos[2];
            
            // Distance from reference to sensor
            let d_ref_sensor = (dx * dx + dy * dy + dz * dz).sqrt();
            
            // Fill A matrix (partial derivatives)
            a_matrix[(i, 0)] = dx / d_ref_sensor;
            a_matrix[(i, 1)] = dy / d_ref_sensor;
            a_matrix[(i, 2)] = dz / d_ref_sensor;
            
            // Fill b vector (range differences)
            b_vector[i] = range_diffs[i];
        }
        
        // First stage: Linear least squares
        let ata = &a_matrix.transpose() * &a_matrix;
        let atb = &a_matrix.transpose() * &b_vector;
        
        // Solve using Cholesky decomposition
        let solution = match ata.cholesky() {
            Some(chol) => chol.solve(&atb),
            None => {
                // Fall back to SVD if matrix is not positive definite
                let svd = a_matrix.clone().svd(true, true);
                match svd.solve(&b_vector, 1e-10) {
                    Ok(sol) => sol,
                    Err(_) => {
                        return Err(crate::error::KwaversError::Physics(
                            crate::error::PhysicsError::ConvergenceFailure {
                                solver: "SVD".to_string(),
                                iterations: 0,
                                residual: f64::INFINITY,
                            }
                        ));
                    }
                }
            }
        };
        
        // Extract position estimate (relative to reference)
        let mut position = [0.0; 3];
        position[0] = ref_pos[0] + solution[0];
        position[1] = ref_pos[1] + solution[1];
        position[2] = ref_pos[2] + solution[2];
        
        // Calculate residuals for uncertainty estimation
        let residuals = &a_matrix * &solution - &b_vector;
        let residual_norm = residuals.norm();
        
        // Estimate uncertainty (simplified - could use CRLB for better estimate)
        let dof = measurements.len() as f64 - 3.0; // degrees of freedom
        let variance = if dof > 0.0 {
            (residual_norm * residual_norm) / dof
        } else {
            1e-6
        };
        
        let uncertainty = [
            variance.sqrt(),
            variance.sqrt(),
            variance.sqrt(),
        ];
        
        Ok(LocalizationResult {
            position,
            uncertainty,
            residual: residual_norm,
            num_sensors,
        })
    }
}