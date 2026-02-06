//! Beam steering functionality
//!
//! Implements electronic beam steering for phased arrays.
//!
//! References:
//! - Wooh & Shi (1999): "A simulation study of the beam steering characteristics for linear phased arrays"

use crate::core::error::KwaversResult;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

use crate::physics::acoustics::analytical::patterns::phase_shifting::core::{
    calculate_wavelength, wrap_phase, MAX_STEERING_ANGLE, SPEED_OF_SOUND,
};

/// Beam steering controller
#[derive(Debug)]
pub struct BeamSteering {
    /// Array element positions
    element_positions: Array2<f64>,
    /// Operating frequency
    frequency: f64,
    /// Current steering angles (azimuth, elevation)
    steering_angles: (f64, f64),
    /// Phase distribution
    phase_distribution: Array1<f64>,
    /// Grating lobe threshold
    grating_lobe_threshold: f64,
}

impl BeamSteering {
    /// Create a new beam steering controller
    #[must_use]
    pub fn new(element_positions: Array2<f64>, frequency: f64) -> Self {
        let num_elements = element_positions.nrows();
        Self {
            element_positions,
            frequency,
            steering_angles: (0.0, 0.0),
            phase_distribution: Array1::zeros(num_elements),
            grating_lobe_threshold: 0.1,
        }
    }

    /// Set steering angles (azimuth and elevation in degrees)
    pub fn set_steering_angles(&mut self, azimuth: f64, elevation: f64) -> KwaversResult<()> {
        if azimuth.abs() > MAX_STEERING_ANGLE || elevation.abs() > MAX_STEERING_ANGLE {
            return Err(crate::core::error::KwaversError::InvalidInput(format!(
                "Steering angles exceed maximum of {MAX_STEERING_ANGLE} degrees"
            )));
        }

        self.steering_angles = (azimuth, elevation);
        self.calculate_phase_distribution()?;
        Ok(())
    }

    /// Calculate phase distribution for current steering angles
    fn calculate_phase_distribution(&mut self) -> KwaversResult<()> {
        let wavelength = calculate_wavelength(self.frequency, SPEED_OF_SOUND);
        let k = 2.0 * PI / wavelength;

        let az_rad = self.steering_angles.0.to_radians();
        let el_rad = self.steering_angles.1.to_radians();

        // Direction cosines
        let kx = k * az_rad.sin() * el_rad.cos();
        let ky = k * el_rad.sin();

        for (i, phase) in self.phase_distribution.iter_mut().enumerate() {
            let pos = self.element_positions.row(i);
            *phase = -(kx * pos[0] + ky * pos[1]);
            *phase = wrap_phase(*phase);
        }

        Ok(())
    }

    /// Check for grating lobes
    #[must_use]
    pub fn check_grating_lobes(&self) -> bool {
        let wavelength = calculate_wavelength(self.frequency, SPEED_OF_SOUND);

        // Find element spacing
        let mut min_spacing = f64::INFINITY;
        for i in 0..self.element_positions.nrows() - 1 {
            let pos1 = self.element_positions.row(i);
            let pos2 = self.element_positions.row(i + 1);
            let spacing = ((pos2[0] - pos1[0]).powi(2) + (pos2[1] - pos1[1]).powi(2)).sqrt();
            if spacing > 0.0 && spacing < min_spacing {
                min_spacing = spacing;
            }
        }

        // Grating lobe condition: d * sin(θ) < λ
        let max_sin_theta = wavelength / min_spacing;
        let current_sin_theta = self
            .steering_angles
            .0
            .to_radians()
            .sin()
            .abs()
            .max(self.steering_angles.1.to_radians().sin().abs());

        current_sin_theta >= max_sin_theta * (1.0 - self.grating_lobe_threshold)
    }

    /// Get phase distribution
    #[must_use]
    pub fn get_phase_distribution(&self) -> &Array1<f64> {
        &self.phase_distribution
    }

    /// Calculate beam pattern at given angles
    #[must_use]
    pub fn calculate_beam_pattern(&self, theta: f64, phi: f64) -> f64 {
        let wavelength = calculate_wavelength(self.frequency, SPEED_OF_SOUND);
        let k = 2.0 * PI / wavelength;

        let theta_rad = theta.to_radians();
        let phi_rad = phi.to_radians();

        // Direction vector
        let kx = k * theta_rad.sin() * phi_rad.cos();
        let ky = k * theta_rad.sin() * phi_rad.sin();

        let mut sum_real = 0.0;
        let mut sum_imag = 0.0;

        for i in 0..self.element_positions.nrows() {
            let pos = self.element_positions.row(i);
            let phase = self.phase_distribution[i] + kx * pos[0] + ky * pos[1];
            sum_real += phase.cos();
            sum_imag += phase.sin();
        }

        (sum_real.powi(2) + sum_imag.powi(2)).sqrt() / self.element_positions.nrows() as f64
    }
}
