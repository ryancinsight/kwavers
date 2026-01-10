// localization/beamforming.rs - Beamforming-based localization (DEPRECATED)
//
// Architectural note (SSOT / deep vertical tree):
// - Beamforming algorithms and numerics are owned by `crate::domain::sensor::beamforming`.
// - Localization owns only orchestration and search policy, implemented in
//   `crate::domain::sensor::localization::beamforming_search`.
//
// This file intentionally provides only the legacy `Beamformer` type for API stability.
// New code must use `beamforming_search::BeamformSearch`.

use super::{Position, SensorArray};
use crate::core::error::KwaversResult;
use ndarray::Array2;
use num_complex::Complex64;

/// Beamformer for source localization (legacy, direction-of-arrival scan).
///
/// Prefer `crate::domain::sensor::localization::beamforming_search::BeamformSearch`
/// for SSOT-compliant beamforming-based localization.
#[derive(Debug)]
pub struct Beamformer {
    #[allow(dead_code)]
    steering_vectors: Array2<Complex64>,
    #[allow(dead_code)]
    frequency: f64,
    #[allow(dead_code)]
    sound_speed: f64,
}

impl Beamformer {
    /// Create new beamformer with computed steering vectors.
    ///
    /// **Implementation**: Computes plane wave steering vectors for 360° azimuthal coverage.
    /// Steering vector for direction θ: a(θ) = exp(-j*2π*f*τ_i(θ)) where τ_i is delay to sensor i.
    ///
    /// **References**:
    /// - Van Trees (2002) "Optimum Array Processing" Chapter 2
    /// - Johnson & Dudgeon (1993) "Array Signal Processing" §2.3
    #[must_use]
    pub fn new(array: &SensorArray, frequency: f64, sound_speed: f64) -> Self {
        let num_sensors = array.num_sensors();
        let mut steering_vectors = Array2::<Complex64>::zeros((num_sensors, 360));
        let positions: Vec<[f64; 3]> = array
            .get_sensor_positions()
            .iter()
            .map(|p| p.to_array())
            .collect();

        for angle_deg in 0..360 {
            let angle_rad = (angle_deg as f64) * std::f64::consts::PI / 180.0;
            let direction = [angle_rad.cos(), angle_rad.sin(), 0.0];
            let steering = crate::domain::sensor::beamforming::SteeringVector::compute_plane_wave(
                direction,
                frequency,
                &positions,
                sound_speed,
            );

            for (sensor_idx, &s) in steering.iter().enumerate() {
                steering_vectors[[sensor_idx, angle_deg]] = s;
            }
        }

        Self {
            steering_vectors,
            frequency,
            sound_speed,
        }
    }

    /// Scan for source direction using delay-and-sum beamforming.
    ///
    /// **Implementation**: Basic delay-and-sum beamformer searching over azimuthal angles.
    /// Finds direction with maximum beam power per Van Trees (2002) Chapter 2.
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

    /// Calculate beam power at direction.
    ///
    /// **Implementation**: Delay-and-sum beamforming power calculation.
    /// Power = |w^H * x|² where w is steering vector and x is sensor data.
    #[must_use]
    pub fn beam_power(&self, data: &Array2<f64>, direction: f64) -> f64 {
        let angle_idx = (direction as usize) % 360;

        // Extract steering vector for this direction
        let steering = self.steering_vectors.column(angle_idx);

        // Compute beamformed output with complex arithmetic
        let mut beamformed = Complex64::new(0.0, 0.0);
        if let Some(data_col) = data.columns().into_iter().next() {
            for (i, &s) in steering.iter().enumerate() {
                if i < data_col.len() {
                    let data_complex = Complex64::new(data_col[i], 0.0);
                    beamformed += s * data_complex;
                }
            }
        }

        beamformed.norm_sqr()
    }
}
