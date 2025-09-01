// localization/tdoa.rs - Time Difference of Arrival processing

use super::{Position, SensorArray};
use crate::error::KwaversResult;
use serde::{Deserialize, Serialize};

/// TDOA measurement between sensor pair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TDOAMeasurement {
    /// Sensor pair (reference sensor, measurement sensor)
    pub sensor_pair: (usize, usize),
    /// Time difference (t_measurement - t_reference) in seconds
    pub time_difference: f64,
    /// Measurement uncertainty in seconds
    pub uncertainty: f64,
}

impl TDOAMeasurement {
    /// Create new TDOA measurement
    pub fn new(sensor1: usize, sensor2: usize, time_diff: f64) -> Self {
        Self {
            sensor_pair: (sensor1, sensor2),
            time_difference: time_diff,
            uncertainty: 1e-9, // Default 1 ns
        }
    }

    /// Convert to distance difference
    pub fn to_distance_difference(&self, sound_speed: f64) -> f64 {
        self.time_difference * sound_speed
    }
}

/// TDOA processor for hyperbolic localization
#[derive(Debug)]
pub struct TDOAProcessor {
    measurements: Vec<TDOAMeasurement>,
    sound_speed: f64,
}

impl TDOAProcessor {
    /// Create new TDOA processor
    pub fn new(sound_speed: f64) -> Self {
        Self {
            measurements: Vec::new(),
            sound_speed,
        }
    }

    /// Add measurement
    pub fn add_measurement(&mut self, measurement: TDOAMeasurement) {
        self.measurements.push(measurement);
    }

    /// Process measurements to find source location
    pub fn process(&self, array: &SensorArray) -> KwaversResult<Position> {
        // Simplified implementation
        // Full implementation would solve hyperbolic equations

        if self.measurements.len() < 3 {
            return Err(crate::error::KwaversError::InvalidInput(
                "Need at least 3 TDOA measurements for 3D localization".to_string(),
            ));
        }

        // For now, return centroid as placeholder
        Ok(array.centroid())
    }

    /// Calculate residuals for given position
    pub fn calculate_residuals(&self, position: &Position, array: &SensorArray) -> Vec<f64> {
        let mut residuals = Vec::new();

        for measurement in &self.measurements {
            let (ref_id, meas_id) = measurement.sensor_pair;

            if let (Some(ref_sensor), Some(meas_sensor)) =
                (array.get_sensor(ref_id), array.get_sensor(meas_id))
            {
                let ref_distance = ref_sensor.distance_to(position);
                let meas_distance = meas_sensor.distance_to(position);
                let predicted_diff = (meas_distance - ref_distance) / self.sound_speed;

                residuals.push(predicted_diff - measurement.time_difference);
            }
        }

        residuals
    }

    /// Get number of measurements
    pub fn num_measurements(&self) -> usize {
        self.measurements.len()
    }
}
