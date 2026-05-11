//! Phase shifter implementation
//!
//! Core phase shifting functionality for beam control.

use super::core::{
    calculate_wavelength, quantize_phase, wrap_phase, ShiftingStrategy, MAX_FOCAL_POINTS,
    MAX_STEERING_ANGLE, MIN_FOCAL_DISTANCE, SPEED_OF_SOUND,
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn enable_quantization(&mut self, enable: bool) {
        self.quantization_enabled = enable;
    }

    /// Calculate phase shifts for linear steering.
    ///
    /// # Algorithm
    ///
    /// For a target steering angle `theta` in the x-z plane, the element phase
    /// follows the narrowband phased-array steering law
    ///
    /// ```text
    /// phi_i = -k x_i sin(theta),    k = 2 pi / lambda.
    /// ```
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn calculate_linear_phases(&mut self, steering_angle: f64) -> KwaversResult<Array1<f64>> {
        let angle_rad = steering_angle.to_radians();

        if angle_rad.abs() > MAX_STEERING_ANGLE {
            return Err(crate::core::error::KwaversError::InvalidInput(format!(
                "Steering angle exceeds maximum of {} degrees",
                MAX_STEERING_ANGLE.to_degrees()
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn calculate_spherical_phases(
        &mut self,
        focal_point: &[f64; 3],
    ) -> KwaversResult<Array1<f64>> {
        let focal_distance =
            focal_point[2].mul_add(focal_point[2], focal_point[1].mul_add(focal_point[1], focal_point[0].powi(2))).sqrt();

        if focal_distance < MIN_FOCAL_DISTANCE {
            return Err(crate::core::error::KwaversError::InvalidInput(format!(
                "Focal distance below minimum of {} mm",
                MIN_FOCAL_DISTANCE * 1000.0
            )));
        }

        let k = 2.0 * PI / self.wavelength;
        let mut phases = Array1::zeros(self.element_positions.nrows());

        for (i, phase) in phases.iter_mut().enumerate() {
            let position = self.element_positions.row(i);
            let dx = focal_point[0] - position[0];
            let dy = focal_point[1] - position[1];
            let dz = focal_point[2] - position[2];
            let distance = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();

            *phase = -k * (distance - focal_distance);

            if self.quantization_enabled {
                *phase = quantize_phase(*phase, DEFAULT_QUANTIZATION_LEVELS);
            }
        }

        self.phase_offsets = phases.clone();
        Ok(phases)
    }

    /// Calculate phase shifts for multiple focal points
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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
                    "Focal points must be 3D coordinates".to_owned(),
                ));
            }

            let focal_distance =
                focal_point[2].mul_add(focal_point[2], focal_point[1].mul_add(focal_point[1], focal_point[0].powi(2))).sqrt();

            if focal_distance < MIN_FOCAL_DISTANCE {
                return Err(crate::core::error::KwaversError::InvalidInput(format!(
                    "Focal distance below minimum of {} mm",
                    MIN_FOCAL_DISTANCE * 1000.0
                )));
            }

            for (i, phase) in phases.iter_mut().enumerate() {
                let position = self.element_positions.row(i);
                let dx = focal_point[0] - position[0];
                let dy = focal_point[1] - position[1];
                let dz = focal_point[2] - position[2];
                let distance = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();

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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn get_phase_offsets(&self) -> &Array1<f64> {
        &self.phase_offsets
    }

    /// Apply phase shifts based on current strategy.
    ///
    /// # Dispatch contract
    ///
    /// - [`ShiftingStrategy::Linear`]: `target = [angle_degrees]`.
    /// - [`ShiftingStrategy::Focused`]: `target = [x, y, z]` in metres.
    /// - [`ShiftingStrategy::MultiFocus`]: packed `[x, y, z]` triples in metres.
    /// - [`ShiftingStrategy::Custom`]: one phase per array element.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn apply_phases(&mut self, target: &[f64]) -> KwaversResult<Array1<f64>> {
        match self.strategy {
            ShiftingStrategy::Linear => {
                if target.len() != 1 {
                    return Err(crate::core::error::KwaversError::InvalidInput(
                        "Linear steering requires single angle".to_owned(),
                    ));
                }
                self.calculate_linear_phases(target[0])
            }
            ShiftingStrategy::Focused => {
                if target.len() != 3 {
                    return Err(crate::core::error::KwaversError::InvalidInput(
                        "Spherical focusing requires 3D point".to_owned(),
                    ));
                }
                self.calculate_spherical_phases(&[target[0], target[1], target[2]])
            }
            ShiftingStrategy::MultiFocus => {
                if target.is_empty() || !target.len().is_multiple_of(3) {
                    return Err(crate::core::error::KwaversError::InvalidInput(
                        "Multi-focus steering requires one or more 3D focal points".to_owned(),
                    ));
                }

                let focal_points: Vec<Vec<f64>> =
                    target.chunks_exact(3).map(|chunk| chunk.to_vec()).collect();
                self.calculate_multipoint_phases(&focal_points)
            }
            ShiftingStrategy::Custom => {
                if target.len() != self.element_positions.nrows() {
                    return Err(crate::core::error::KwaversError::InvalidInput(format!(
                        "Custom phase pattern requires {} phases, got {}",
                        self.element_positions.nrows(),
                        target.len()
                    )));
                }

                let mut phases = Array1::zeros(self.element_positions.nrows());
                for (phase, &target_phase) in phases.iter_mut().zip(target) {
                    *phase = wrap_phase(target_phase);
                    if self.quantization_enabled {
                        *phase = quantize_phase(*phase, DEFAULT_QUANTIZATION_LEVELS);
                    }
                }

                self.phase_offsets = phases.clone();
                Ok(phases)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::arr2;

    fn linear_array() -> Array2<f64> {
        arr2(&[[-0.001, 0.0, 0.0], [0.0, 0.0, 0.0], [0.001, 0.0, 0.0]])
    }

    #[test]
    fn focused_strategy_applies_spherical_phase_law() {
        let mut shifter = PhaseShifter::new(linear_array(), 1.0e6);
        shifter.set_strategy(ShiftingStrategy::Focused);

        let phases = shifter.apply_phases(&[0.0, 0.0, 0.02]).unwrap();
        let k = 2.0 * PI / shifter.wavelength;
        let outer_distance = (0.001_f64.powi(2) + 0.02_f64.powi(2)).sqrt();
        let expected_outer = -k * (outer_distance - 0.02);

        assert_relative_eq!(phases[0], expected_outer, epsilon = 1e-12);
        assert_relative_eq!(phases[1], 0.0, epsilon = 1e-12);
        assert_relative_eq!(phases[2], expected_outer, epsilon = 1e-12);
        assert_eq!(shifter.get_phase_offsets(), &phases);
    }

    #[test]
    fn multifocus_strategy_accepts_flat_focal_point_chunks() {
        let mut shifter = PhaseShifter::new(linear_array(), 1.0e6);
        shifter.set_strategy(ShiftingStrategy::MultiFocus);

        let phases = shifter
            .apply_phases(&[0.0, 0.0, 0.02, 0.001, 0.0, 0.02])
            .unwrap();
        let k = 2.0 * PI / shifter.wavelength;
        let left_to_center_focus = (0.001_f64.powi(2) + 0.02_f64.powi(2)).sqrt();
        let left_to_right_focus = (0.002_f64.powi(2) + 0.02_f64.powi(2)).sqrt();
        let center_to_right_focus = (0.001_f64.powi(2) + 0.02_f64.powi(2)).sqrt();
        let reference = (0.001_f64.powi(2) + 0.02_f64.powi(2)).sqrt();

        let expected_left =
            (-k * (left_to_center_focus - 0.02) - k * (left_to_right_focus - reference)) / 2.0;
        let expected_center = (-k * (0.02 - 0.02) - k * (center_to_right_focus - reference)) / 2.0;
        let expected_right = (-k * (left_to_center_focus - 0.02) - k * (0.02 - reference)) / 2.0;

        assert_relative_eq!(phases[0], expected_left, epsilon = 1e-12);
        assert_relative_eq!(phases[1], expected_center, epsilon = 1e-12);
        assert_relative_eq!(phases[2], expected_right, epsilon = 1e-12);
    }

    #[test]
    fn custom_strategy_sets_wrapped_phase_pattern() {
        let mut shifter = PhaseShifter::new(linear_array(), 1.0e6);
        shifter.set_strategy(ShiftingStrategy::Custom);

        let phases = shifter.apply_phases(&[0.0, PI, 3.0 * PI]).unwrap();

        assert_relative_eq!(phases[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(phases[1], PI, epsilon = 1e-12);
        assert_relative_eq!(phases[2], PI, epsilon = 1e-12);
        assert_eq!(shifter.get_phase_offsets().len(), 3);
    }

    #[test]
    fn linear_strategy_accepts_documented_sixty_degree_bound() {
        let mut shifter = PhaseShifter::new(linear_array(), 1.0e6);

        shifter.set_strategy(ShiftingStrategy::Linear);
        let phases = shifter.apply_phases(&[60.0]).unwrap();
        let k = 2.0 * PI / shifter.wavelength;
        let expected_left = -k * -0.001 * 60.0_f64.to_radians().sin();

        assert_relative_eq!(phases[0], expected_left, epsilon = 1e-12);
        assert_relative_eq!(phases[1], 0.0, epsilon = 1e-12);
        assert_relative_eq!(phases[2], -expected_left, epsilon = 1e-12);
    }

    #[test]
    fn strategy_input_shapes_and_domains_are_rejected_before_computation() {
        let mut shifter = PhaseShifter::new(linear_array(), 1.0e6);

        shifter.set_strategy(ShiftingStrategy::Linear);
        let steering_error = shifter.apply_phases(&[61.0]).unwrap_err();
        assert!(format!("{steering_error}").contains("Steering angle exceeds maximum"));

        shifter.set_strategy(ShiftingStrategy::Focused);
        let focus_shape_error = shifter.apply_phases(&[0.0, 0.0]).unwrap_err();
        assert!(format!("{focus_shape_error}").contains("3D point"));
        let focus_distance_error = shifter.apply_phases(&[0.0, 0.0, 0.0005]).unwrap_err();
        assert!(format!("{focus_distance_error}").contains("1"));

        shifter.set_strategy(ShiftingStrategy::MultiFocus);
        let multifocus_error = shifter.apply_phases(&[0.0, 0.0]).unwrap_err();
        assert!(format!("{multifocus_error}").contains("3D focal points"));
        let multifocus_distance_error = shifter.apply_phases(&[0.0, 0.0, 0.0005]).unwrap_err();
        assert!(format!("{multifocus_distance_error}").contains("1"));

        shifter.set_strategy(ShiftingStrategy::Custom);
        let custom_error = shifter.apply_phases(&[0.0, 1.0]).unwrap_err();
        assert!(format!("{custom_error}").contains("requires 3 phases"));
    }
}
