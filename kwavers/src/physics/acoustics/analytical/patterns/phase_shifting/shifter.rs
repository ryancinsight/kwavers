//! Phase shifter implementation
//!
//! Core phase shifting functionality for beam control.

use super::core::{
    calculate_wavelength, quantize_phase, wrap_phase, ShiftingStrategy, MAX_FOCAL_POINTS,
    MAX_STEERING_ANGLE, MIN_FOCAL_DISTANCE,
};
use crate::core::constants::numerical::TWO_PI;
use crate::core::constants::SOUND_SPEED_WATER;

/// Default quantization levels for phase control
const DEFAULT_QUANTIZATION_LEVELS: u32 = 256;
use crate::core::error::KwaversResult;
use ndarray::{Array1, Array2};

/// Phase shifter for beam control
#[derive(Debug)]
pub struct PhaseShifter {
    strategy: ShiftingStrategy,
    element_positions: Array2<f64>,
    pub(crate) wavelength: f64,
    phase_offsets: Array1<f64>,
    quantization_enabled: bool,
}

impl PhaseShifter {
    /// Create a new phase shifter
    #[must_use]
    pub fn new(element_positions: Array2<f64>, operating_frequency: f64) -> Self {
        let wavelength = calculate_wavelength(operating_frequency, SOUND_SPEED_WATER);
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

    #[inline]
    fn apply_phase_quantization(enabled: bool, phase: f64) -> f64 {
        if enabled {
            quantize_phase(phase, DEFAULT_QUANTIZATION_LEVELS)
        } else {
            phase
        }
    }

    fn focal_distance(point: &[f64]) -> KwaversResult<f64> {
        if point.len() != 3 {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "Focal points must be 3D coordinates".to_owned(),
            ));
        }

        let focal_distance = point[2]
            .mul_add(point[2], point[1].mul_add(point[1], point[0].powi(2)))
            .sqrt();

        if focal_distance < MIN_FOCAL_DISTANCE {
            return Err(crate::core::error::KwaversError::InvalidInput(format!(
                "Focal distance below minimum of {} mm",
                MIN_FOCAL_DISTANCE * 1000.0
            )));
        }

        Ok(focal_distance)
    }

    /// Fill the cached phase buffer from focal-point slices.
    ///
    /// # Contract
    ///
    /// The iterator is traversed once for validation before the cached phase
    /// buffer is mutated. This preserves the rejection invariant: invalid
    /// focal input leaves the last valid phase pattern intact. The same
    /// kernel serves public vector-backed focal points and flat packed
    /// `[x, y, z]` dispatch, so multi-focus phase synthesis has one
    /// authoritative implementation.
    fn calculate_multipoint_phases_from_slices<'b, I>(
        &mut self,
        focal_points: I,
        num_points: usize,
    ) -> KwaversResult<Array1<f64>>
    where
        I: Clone + Iterator<Item = &'b [f64]>,
    {
        if num_points > MAX_FOCAL_POINTS {
            return Err(crate::core::error::KwaversError::InvalidInput(format!(
                "Number of focal points exceeds maximum of {MAX_FOCAL_POINTS}"
            )));
        }

        for focal_point in focal_points.clone() {
            Self::focal_distance(focal_point)?;
        }

        let k = TWO_PI / self.wavelength;
        let num_points_f64 = num_points as f64;
        self.phase_offsets.fill(0.0);

        for focal_point in focal_points {
            let focal_distance = Self::focal_distance(focal_point)?;

            for (i, phase) in self.phase_offsets.iter_mut().enumerate() {
                let position = self.element_positions.row(i);
                let dx = focal_point[0] - position[0];
                let dy = focal_point[1] - position[1];
                let dz = focal_point[2] - position[2];
                let distance = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();

                *phase += (-k * (distance - focal_distance)) / num_points_f64;
            }
        }

        if self.quantization_enabled {
            for phase in &mut self.phase_offsets {
                *phase = quantize_phase(*phase, DEFAULT_QUANTIZATION_LEVELS);
            }
        }

        Ok(self.phase_offsets.clone())
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

        let k = TWO_PI / self.wavelength;
        let quantization_enabled = self.quantization_enabled;
        self.phase_offsets.fill(0.0);

        for (i, phase) in self.phase_offsets.iter_mut().enumerate() {
            let position = self.element_positions.row(i);
            *phase = Self::apply_phase_quantization(
                quantization_enabled,
                -k * position[0] * angle_rad.sin(),
            );
        }

        Ok(self.phase_offsets.clone())
    }

    /// Calculate phase shifts for spherical focusing
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn calculate_spherical_phases(
        &mut self,
        focal_point: &[f64; 3],
    ) -> KwaversResult<Array1<f64>> {
        let focal_distance = Self::focal_distance(focal_point)?;

        let k = TWO_PI / self.wavelength;
        let quantization_enabled = self.quantization_enabled;
        self.phase_offsets.fill(0.0);

        for (i, phase) in self.phase_offsets.iter_mut().enumerate() {
            let position = self.element_positions.row(i);
            let dx = focal_point[0] - position[0];
            let dy = focal_point[1] - position[1];
            let dz = focal_point[2] - position[2];
            let distance = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();

            *phase = Self::apply_phase_quantization(
                quantization_enabled,
                -k * (distance - focal_distance),
            );
        }

        Ok(self.phase_offsets.clone())
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

        self.calculate_multipoint_phases_from_slices(
            focal_points.iter().map(Vec::as_slice),
            focal_points.len(),
        )
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
    ///
    /// The internal phase buffer is reused across calls. The returned
    /// [`Array1`] is the owned snapshot required by the public API; callers
    /// that only need a borrowed view can read [`Self::get_phase_offsets`]
    /// after this method returns.
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

                self.calculate_multipoint_phases_from_slices(
                    target.chunks_exact(3),
                    target.len() / 3,
                )
            }
            ShiftingStrategy::Custom => {
                if target.len() != self.element_positions.nrows() {
                    return Err(crate::core::error::KwaversError::InvalidInput(format!(
                        "Custom phase pattern requires {} phases, got {}",
                        self.element_positions.nrows(),
                        target.len()
                    )));
                }

                self.phase_offsets.fill(0.0);
                let quantization_enabled = self.quantization_enabled;
                for (phase, &target_phase) in self.phase_offsets.iter_mut().zip(target) {
                    *phase = Self::apply_phase_quantization(
                        quantization_enabled,
                        wrap_phase(target_phase),
                    );
                }

                Ok(self.phase_offsets.clone())
            }
        }
    }
}
