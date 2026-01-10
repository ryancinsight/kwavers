//! Dynamic focusing functionality
//!
//! Implements dynamic focusing and multi-focus patterns.
//!
//! References:
//! - Ebbini & Cain (1989): "Multiple-focus ultrasound phased-array pattern synthesis"

use crate::domain::core::error::KwaversResult;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

use crate::physics::phase_modulation::phase_shifting::core::{
    calculate_wavelength, wrap_phase, MAX_FOCAL_POINTS, MIN_FOCAL_DISTANCE, SPEED_OF_SOUND,
};

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
    pub fn set_focal_point(&mut self, x: f64, y: f64, z: f64) -> KwaversResult<()> {
        let focal_distance = (x * x + y * y + z * z).sqrt();

        if focal_distance < MIN_FOCAL_DISTANCE / 1000.0 {
            return Err(crate::domain::core::error::KwaversError::InvalidInput(
                format!("Focal distance below minimum of {MIN_FOCAL_DISTANCE} mm"),
            ));
        }

        self.focal_points = vec![[x, y, z]];
        self.calculate_phase_distribution()?;
        Ok(())
    }

    /// Set multiple focal points
    pub fn set_multiple_focal_points(&mut self, points: Vec<[f64; 3]>) -> KwaversResult<()> {
        if points.len() > MAX_FOCAL_POINTS {
            return Err(crate::domain::core::error::KwaversError::InvalidInput(
                format!("Number of focal points exceeds maximum of {MAX_FOCAL_POINTS}"),
            ));
        }

        for point in &points {
            let focal_distance = (point[0].powi(2) + point[1].powi(2) + point[2].powi(2)).sqrt();
            if focal_distance < MIN_FOCAL_DISTANCE / 1000.0 {
                return Err(crate::domain::core::error::KwaversError::InvalidInput(
                    format!("Focal distance below minimum of {MIN_FOCAL_DISTANCE} mm"),
                ));
            }
        }

        self.focal_points = points;
        self.calculate_phase_distribution()?;
        Ok(())
    }

    /// Calculate phase distribution for current focal points
    fn calculate_phase_distribution(&mut self) -> KwaversResult<()> {
        if self.focal_points.is_empty() {
            self.phase_distribution.fill(0.0);
            return Ok(());
        }

        let wavelength = calculate_wavelength(self.frequency, SPEED_OF_SOUND);
        let k = 2.0 * PI / wavelength;

        self.phase_distribution.fill(0.0);

        for focal_point in &self.focal_points {
            let reference_distance =
                (focal_point[0].powi(2) + focal_point[1].powi(2) + focal_point[2].powi(2)).sqrt();

            for i in 0..self.element_positions.nrows() {
                let pos = self.element_positions.row(i);
                let dx = focal_point[0] - pos[0];
                let dy = focal_point[1] - pos[1];
                let dz = focal_point[2] - pos[2];
                let distance = (dx * dx + dy * dy + dz * dz).sqrt();

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
        let n = self.amplitude_weights.len();

        match window_type {
            ApodizationType::Uniform => {
                self.amplitude_weights.fill(1.0);
            }
            ApodizationType::Hamming => {
                for i in 0..n {
                    let x = i as f64 / (n - 1) as f64;
                    self.amplitude_weights[i] = 0.54 - 0.46 * (2.0 * PI * x).cos();
                }
            }
            ApodizationType::Hanning => {
                for i in 0..n {
                    let x = i as f64 / (n - 1) as f64;
                    self.amplitude_weights[i] = 0.5 * (1.0 - (2.0 * PI * x).cos());
                }
            }
            ApodizationType::Gaussian => {
                let sigma = n as f64 / 6.0;
                let center = (n - 1) as f64 / 2.0;
                for i in 0..n {
                    let x = (i as f64 - center) / sigma;
                    self.amplitude_weights[i] = (-0.5 * x * x).exp();
                }
            }
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
        let wavelength = calculate_wavelength(self.frequency, SPEED_OF_SOUND);
        let k = 2.0 * PI / wavelength;

        let mut sum_real = 0.0;
        let mut sum_imag = 0.0;

        for i in 0..self.element_positions.nrows() {
            let pos = self.element_positions.row(i);
            let dx = x - pos[0];
            let dy = y - pos[1];
            let dz = z - pos[2];
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();

            let phase = self.phase_distribution[i] + k * distance;
            let amplitude = self.amplitude_weights[i];

            sum_real += amplitude * phase.cos();
            sum_imag += amplitude * phase.sin();
        }

        (sum_real.powi(2) + sum_imag.powi(2)) / self.element_positions.nrows() as f64
    }
}

/// Apodization window types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ApodizationType {
    /// Uniform weighting
    Uniform,
    /// Hamming window
    Hamming,
    /// Hanning window
    Hanning,
    /// Gaussian window
    Gaussian,
}
