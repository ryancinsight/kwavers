//! Dynamic focusing functionality
//!
//! Implements dynamic focusing and multi-focus patterns.
//!
//! References:
//! - Ebbini & Cain (1989): "Multiple-focus ultrasound phased-array pattern synthesis"

use crate::core::error::KwaversResult;
use ndarray::{Array1, Array2};

use crate::core::constants::SOUND_SPEED_WATER;
use crate::physics::phase_modulation::phase_shifting::core::{
    calculate_wavelength, wrap_phase, MAX_FOCAL_POINTS, MIN_FOCAL_DISTANCE,
};
use crate::core::constants::numerical::{TWO_PI};

/// Dynamic focusing controller
#[derive(Debug)]
pub struct DynamicFocusing {
    /// Array element positions
    element_positions: Array2<f64>,
    /// Operating frequency
    frequency: f64,
    /// Current focal points
    focal_points: Vec<[f64; 3]>,
    /// Phase distribution
    phase_distribution: Array1<f64>,
    /// Amplitude weights for apodization
    amplitude_weights: Array1<f64>,
}

impl DynamicFocusing {
    /// Create a new dynamic focusing controller
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(element_positions: Array2<f64>, frequency: f64) -> Self {
        let num_elements = element_positions.nrows();
        Self {
            element_positions,
            frequency,
            focal_points: Vec::new(),
            phase_distribution: Array1::zeros(num_elements),
            amplitude_weights: Array1::ones(num_elements),
        }
    }

    /// Set single focal point
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn set_focal_point(&mut self, x: f64, y: f64, z: f64) -> KwaversResult<()> {
        let focal_distance = z.mul_add(z, x.mul_add(x, y * y)).sqrt();

        if focal_distance < MIN_FOCAL_DISTANCE {
            return Err(crate::core::error::KwaversError::InvalidInput(format!(
                "Focal distance below minimum of {} mm",
                MIN_FOCAL_DISTANCE * 1000.0
            )));
        }

        self.focal_points = vec![[x, y, z]];
        self.calculate_phase_distribution()?;
        Ok(())
    }

    /// Set multiple focal points
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn set_multiple_focal_points(&mut self, points: Vec<[f64; 3]>) -> KwaversResult<()> {
        if points.len() > MAX_FOCAL_POINTS {
            return Err(crate::core::error::KwaversError::InvalidInput(format!(
                "Number of focal points exceeds maximum of {MAX_FOCAL_POINTS}"
            )));
        }

        for point in &points {
            let focal_distance = point[2]
                .mul_add(point[2], point[1].mul_add(point[1], point[0].powi(2)))
                .sqrt();
            if focal_distance < MIN_FOCAL_DISTANCE {
                return Err(crate::core::error::KwaversError::InvalidInput(format!(
                    "Focal distance below minimum of {} mm",
                    MIN_FOCAL_DISTANCE * 1000.0
                )));
            }
        }

        self.focal_points = points;
        self.calculate_phase_distribution()?;
        Ok(())
    }

    /// Calculate phase distribution for current focal points
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn calculate_phase_distribution(&mut self) -> KwaversResult<()> {
        if self.focal_points.is_empty() {
            self.phase_distribution.fill(0.0);
            return Ok(());
        }

        let wavelength = calculate_wavelength(self.frequency, SOUND_SPEED_WATER);
        let k = TWO_PI / wavelength;

        self.phase_distribution.fill(0.0);

        for focal_point in &self.focal_points {
            let reference_distance = focal_point[2]
                .mul_add(
                    focal_point[2],
                    focal_point[1].mul_add(focal_point[1], focal_point[0].powi(2)),
                )
                .sqrt();

            for i in 0..self.element_positions.nrows() {
                let pos = self.element_positions.row(i);
                let dx = focal_point[0] - pos[0];
                let dy = focal_point[1] - pos[1];
                let dz = focal_point[2] - pos[2];
                let distance = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();

                // Superposition of phases
                self.phase_distribution[i] +=
                    -k * (distance - reference_distance) / self.focal_points.len() as f64;
            }
        }

        // Wrap phases
        for phase in &mut self.phase_distribution {
            *phase = wrap_phase(*phase);
        }

        Ok(())
    }

    /// Apply apodization for sidelobe reduction
    pub fn apply_apodization(&mut self, window_type: ApodizationType) {
        let weights = window_type.weights(self.amplitude_weights.len());
        for (w, v) in self.amplitude_weights.iter_mut().zip(weights.iter()) {
            *w = *v;
        }
    }

    /// Get phase distribution
    #[must_use]
    pub fn get_phase_distribution(&self) -> &Array1<f64> {
        &self.phase_distribution
    }

    /// Get amplitude weights
    #[must_use]
    pub fn get_amplitude_weights(&self) -> &Array1<f64> {
        &self.amplitude_weights
    }

    /// Calculate intensity at a point
    #[must_use]
    pub fn calculate_intensity(&self, x: f64, y: f64, z: f64) -> f64 {
        let wavelength = calculate_wavelength(self.frequency, SOUND_SPEED_WATER);
        let k = TWO_PI / wavelength;

        let mut sum_real = 0.0;
        let mut sum_imag = 0.0;

        for i in 0..self.element_positions.nrows() {
            let pos = self.element_positions.row(i);
            let dx = x - pos[0];
            let dy = y - pos[1];
            let dz = z - pos[2];
            let distance = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();

            let phase = self.phase_distribution[i] + k * distance;
            let amplitude = self.amplitude_weights[i];

            sum_real += amplitude * phase.cos();
            sum_imag += amplitude * phase.sin();
        }

        sum_imag.mul_add(sum_imag, sum_real.powi(2)) / self.element_positions.nrows() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use crate::core::constants::numerical::MHZ_TO_HZ;
    use ndarray::arr2;

    fn linear_array() -> Array2<f64> {
        arr2(&[[-0.001, 0.0, 0.0], [0.0, 0.0, 0.0], [0.001, 0.0, 0.0]])
    }

    #[test]
    fn focusing_accepts_documented_one_millimeter_bound() {
        let mut focusing = DynamicFocusing::new(linear_array(), MHZ_TO_HZ);

        focusing.set_focal_point(0.0, 0.0, 0.001).unwrap();

        let phases = focusing.get_phase_distribution();
        assert_relative_eq!(phases[1], 0.0, epsilon = 1.0e-12);
        assert_relative_eq!(phases[0], phases[2], epsilon = 1.0e-12);
    }

    #[test]
    fn focusing_rejects_sub_millimeter_single_and_multiple_points() {
        let mut focusing = DynamicFocusing::new(linear_array(), MHZ_TO_HZ);

        let single_error = focusing.set_focal_point(0.0, 0.0, 0.0005).unwrap_err();
        assert!(format!("{single_error}").contains("1"));

        let multi_error = focusing
            .set_multiple_focal_points(vec![[0.0, 0.0, 0.001], [0.0, 0.0, 0.0005]])
            .unwrap_err();
        assert!(format!("{multi_error}").contains("1"));
    }
}

pub use crate::math::signal::ApodizationType;
