// localization/beamforming.rs - Beamforming-based localization

use super::{Position, SensorArray};
use crate::error::KwaversResult;
use ndarray::Array2;

/// Beamformer for source localization
#[derive(Debug)]
pub struct Beamformer {
    #[allow(dead_code)]
    steering_vectors: Array2<f64>,
    #[allow(dead_code)]
    frequency: f64,
    #[allow(dead_code)]
    sound_speed: f64,
}

impl Beamformer {
    /// Create new beamformer with computed steering vectors
    ///
    /// **Implementation**: Computes plane wave steering vectors for 360° azimuthal coverage
    /// Steering vector for direction θ: a(θ) = exp(-j*2π*f*τ_i(θ)) where τ_i is delay to sensor i
    ///
    /// **References**:
    /// - Van Trees (2002) "Optimum Array Processing" Chapter 2
    /// - Johnson & Dudgeon (1993) "Array Signal Processing" §2.3
    #[must_use]
    pub fn new(array: &SensorArray, frequency: f64, sound_speed: f64) -> Self {
        let num_sensors = array.num_sensors();
        let mut steering_vectors = Array2::zeros((num_sensors, 360));

        // Compute steering vectors for each azimuthal angle (0-359 degrees)
        let wavelength = sound_speed / frequency;
        let k = 2.0 * std::f64::consts::PI / wavelength; // Wave number

        for angle_deg in 0..360 {
            let angle_rad = (angle_deg as f64) * std::f64::consts::PI / 180.0;
            let direction = [angle_rad.cos(), angle_rad.sin(), 0.0];

            // Compute steering vector for this direction
            let positions = array.get_sensor_positions();
            for (sensor_idx, position) in positions.iter().enumerate() {
                // Time delay from reference point to sensor
                let pos_array = position.to_array();
                let delay =
                    (pos_array[0] * direction[0] + pos_array[1] * direction[1]) / sound_speed;
                // Phase of complex exponential (using magnitude only for simplicity)
                let phase = k * sound_speed * delay;
                steering_vectors[[sensor_idx, angle_deg]] = phase.cos();
            }
        }

        Self {
            steering_vectors,
            frequency,
            sound_speed,
        }
    }

    /// Scan for source direction using delay-and-sum beamforming
    ///
    /// **Implementation**: Basic delay-and-sum beamformer searching over azimuthal angles
    /// Finds direction with maximum beam power per Van Trees (2002) Chapter 2
    pub fn scan(&self, data: &Array2<f64>) -> KwaversResult<Position> {
        let mut max_power = 0.0;
        let mut best_angle = 0;

        // Search over all steering directions
        for angle in 0..360 {
            let power = self.beam_power(data, angle as f64);
            if power > max_power {
                max_power = power;
                best_angle = angle;
            }
        }

        // Convert best angle to position (unit distance)
        let angle_rad = (best_angle as f64) * std::f64::consts::PI / 180.0;
        Ok(Position::new(angle_rad.cos(), angle_rad.sin(), 0.0))
    }

    /// Calculate beam power at direction
    ///
    /// **Implementation**: Delay-and-sum beamforming power calculation
    /// Power = |w^H * x|² where w is steering vector and x is sensor data
    #[must_use]
    pub fn beam_power(&self, data: &Array2<f64>, direction: f64) -> f64 {
        let angle_idx = (direction as usize) % 360;

        // Extract steering vector for this direction
        let steering = self.steering_vectors.column(angle_idx);

        // Compute beamformed output: sum of weighted sensor data
        let mut power = 0.0;
        if let Some(data_col) = data.columns().into_iter().next() {
            for (i, &s) in steering.iter().enumerate() {
                if i < data_col.len() {
                    power += s * data_col[i];
                }
            }
        }

        power.abs()
    }
}
