//! Phase shifter implementation
//!
//! Core phase shifting functionality for beam control.

use super::core::{
    calculate_wavelength, quantize_phase, ShiftingStrategy, MAX_FOCAL_POINTS, MAX_STEERING_ANGLE,
    MIN_FOCAL_DISTANCE, SPEED_OF_SOUND,
};

/// Default quantization levels for phase control
const DEFAULT_QUANTIZATION_LEVELS: u32 = 256;
use crate::core::error::KwaversResult;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// Phase shifter for beam control
#[derive(Debug)]
pub struct PhaseShifter {
    strategy: ShiftingStrategy,
    element_positions: Array2<f64>,
    #[allow(dead_code)] // Operating frequency for phase calculations
    operating_frequency: f64,
    wavelength: f64,
    phase_offsets: Array1<f64>,
    quantization_enabled: bool,
}

impl PhaseShifter {
    /// Create a new phase shifter
    #[must_use]
    pub fn new(element_positions: Array2<f64>, operating_frequency: f64) -> Self {
        let wavelength = calculate_wavelength(operating_frequency, SPEED_OF_SOUND);
        let num_elements = element_positions.nrows();
        let phase_offsets = Array1::zeros(num_elements);

        Self {
            strategy: ShiftingStrategy::Linear,
            element_positions,
            operating_frequency,
            wavelength,
            phase_offsets,
            quantization_enabled: false,
        }
    }

    /// Set shifting strategy
    pub fn set_strategy(&mut self, strategy: ShiftingStrategy) {
        self.strategy = strategy;
    }

    /// Enable phase quantization
    pub fn enable_quantization(&mut self, enable: bool) {
        self.quantization_enabled = enable;
    }

    /// Calculate phase shifts for linear steering
    pub fn calculate_linear_phases(&mut self, steering_angle: f64) -> KwaversResult<Array1<f64>> {
        let angle_rad = steering_angle.to_radians();

        if angle_rad.abs() > MAX_STEERING_ANGLE.to_radians() {
            return Err(crate::core::error::KwaversError::InvalidInput(format!(
                "Steering angle exceeds maximum of {MAX_STEERING_ANGLE} degrees"
            )));
        }

        let k = 2.0 * PI / self.wavelength;
        let mut phases = Array1::zeros(self.element_positions.nrows());

        for (i, phase) in phases.iter_mut().enumerate() {
            let position = self.element_positions.row(i);
            *phase = -k * position[0] * angle_rad.sin();

            if self.quantization_enabled {
                *phase = quantize_phase(*phase, DEFAULT_QUANTIZATION_LEVELS);
            }
        }

        self.phase_offsets = phases.clone();
        Ok(phases)
    }

    /// Calculate phase shifts for spherical focusing
    pub fn calculate_spherical_phases(
        &mut self,
        focal_point: &[f64; 3],
    ) -> KwaversResult<Array1<f64>> {
        let focal_distance =
            (focal_point[0].powi(2) + focal_point[1].powi(2) + focal_point[2].powi(2)).sqrt();

        if focal_distance < MIN_FOCAL_DISTANCE / 1000.0 {
            return Err(crate::core::error::KwaversError::InvalidInput(format!(
                "Focal distance below minimum of {MIN_FOCAL_DISTANCE} mm"
            )));
        }

        let k = 2.0 * PI / self.wavelength;
        let mut phases = Array1::zeros(self.element_positions.nrows());

        for (i, phase) in phases.iter_mut().enumerate() {
            let position = self.element_positions.row(i);
            let dx = focal_point[0] - position[0];
            let dy = focal_point[1] - position[1];
            let dz = focal_point[2] - position[2];
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();

            *phase = -k * (distance - focal_distance);

            if self.quantization_enabled {
                *phase = quantize_phase(*phase, DEFAULT_QUANTIZATION_LEVELS);
            }
        }

        self.phase_offsets = phases.clone();
        Ok(phases)
    }

    /// Calculate phase shifts for multiple focal points
    pub fn calculate_multipoint_phases(
        &mut self,
        focal_points: &[Vec<f64>],
    ) -> KwaversResult<Array1<f64>> {
        if focal_points.len() > MAX_FOCAL_POINTS {
            return Err(crate::core::error::KwaversError::InvalidInput(format!(
                "Number of focal points exceeds maximum of {MAX_FOCAL_POINTS}"
            )));
        }

        let k = 2.0 * PI / self.wavelength;
        let mut phases = Array1::zeros(self.element_positions.nrows());
        let num_points = focal_points.len() as f64;

        for focal_point in focal_points {
            if focal_point.len() != 3 {
                return Err(crate::core::error::KwaversError::InvalidInput(
                    "Focal points must be 3D coordinates".to_string(),
                ));
            }

            let focal_distance =
                (focal_point[0].powi(2) + focal_point[1].powi(2) + focal_point[2].powi(2)).sqrt();

            for (i, phase) in phases.iter_mut().enumerate() {
                let position = self.element_positions.row(i);
                let dx = focal_point[0] - position[0];
                let dy = focal_point[1] - position[1];
                let dz = focal_point[2] - position[2];
                let distance = (dx * dx + dy * dy + dz * dz).sqrt();

                // Superposition with equal weighting
                *phase += (-k * (distance - focal_distance)) / num_points;
            }
        }

        if self.quantization_enabled {
            for phase in &mut phases {
                *phase = quantize_phase(*phase, DEFAULT_QUANTIZATION_LEVELS);
            }
        }

        self.phase_offsets = phases.clone();
        Ok(phases)
    }

    /// Get current phase offsets
    #[must_use]
    pub fn get_phase_offsets(&self) -> &Array1<f64> {
        &self.phase_offsets
    }

    /// Apply phase shifts based on current strategy
    pub fn apply_phases(&mut self, target: &[f64]) -> KwaversResult<Array1<f64>> {
        match self.strategy {
            ShiftingStrategy::Linear => {
                if target.len() != 1 {
                    return Err(crate::core::error::KwaversError::InvalidInput(
                        "Linear steering requires single angle".to_string(),
                    ));
                }
                self.calculate_linear_phases(target[0])
            }
            ShiftingStrategy::Custom => {
                if target.len() != 3 {
                    return Err(crate::core::error::KwaversError::InvalidInput(
                        "Spherical focusing requires 3D point".to_string(),
                    ));
                }
                self.calculate_spherical_phases(&[target[0], target[1], target[2]])
            }
            _ => Err(crate::core::error::KwaversError::NotImplemented(format!(
                "Strategy {:?} not yet implemented",
                self.strategy
            ))),
        }
    }
}
