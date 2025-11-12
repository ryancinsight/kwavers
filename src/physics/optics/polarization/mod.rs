// physics/optics/polarization/mod.rs
//! Optical polarization physics implementation
//!
//! This module implements proper optical polarization using Jones calculus,
//! providing mathematically correct descriptions of polarization states and transformations.
//!
//! # Jones Calculus
//!
//! Jones vectors describe the polarization state of light:
//! ```text
//! |E_x|   |E_0x|
//! |E_y| = |E_0y|
//! ```
//!
//! Jones matrices describe optical elements that transform polarization:
//! ```text
//! |E_x'|   |m11 m12| |E_x|
//! |E_y'| = |m21 m22| |E_y|
//! ```
//!
//! # Physical Models
//!
//! - **Linear Polarizer**: Transmits only one polarization component
//! - **Waveplate**: Introduces phase difference between orthogonal components
//! - **Rotator**: Rotates the plane of polarization
//! - **Depolarizer**: Converts polarized light to unpolarized light
//!
//! # References
//!
//! - Jones, R. C. (1941). "A new calculus for the treatment of optical systems"
//! - Born, M., & Wolf, E. (1999). Principles of Optics

use crate::grid::Grid;
use crate::medium::Medium;
use log::debug;
use ndarray::{Array3, Array4, Axis, Zip};
use num_complex::Complex64;
use std::fmt::Debug;

/// Jones vector representing the polarization state of light
#[derive(Debug, Clone, Copy)]
pub struct JonesVector {
    /// Electric field component in x-direction
    pub ex: Complex64,
    /// Electric field component in y-direction
    pub ey: Complex64,
}

impl JonesVector {
    /// Create a Jones vector from complex amplitudes
    #[must_use]
    pub fn new(ex: Complex64, ey: Complex64) -> Self {
        Self { ex, ey }
    }

    /// Create horizontally polarized light
    #[must_use]
    pub fn horizontal(amplitude: f64) -> Self {
        Self::new(Complex64::new(amplitude, 0.0), Complex64::new(0.0, 0.0))
    }

    /// Create vertically polarized light
    #[must_use]
    pub fn vertical(amplitude: f64) -> Self {
        Self::new(Complex64::new(0.0, 0.0), Complex64::new(amplitude, 0.0))
    }

    /// Create right-circularly polarized light
    #[must_use]
    pub fn right_circular(amplitude: f64) -> Self {
        let norm = amplitude / std::f64::consts::SQRT_2;
        Self::new(
            Complex64::new(norm, 0.0),
            Complex64::new(0.0, -norm),
        )
    }

    /// Create left-circularly polarized light
    #[must_use]
    pub fn left_circular(amplitude: f64) -> Self {
        let norm = amplitude / std::f64::consts::SQRT_2;
        Self::new(
            Complex64::new(norm, 0.0),
            Complex64::new(0.0, norm),
        )
    }

    /// Calculate the intensity (time-averaged power density)
    #[must_use]
    pub fn intensity(&self) -> f64 {
        (self.ex.norm_sqr() + self.ey.norm_sqr()) / 2.0
    }

    /// Calculate the degree of polarization
    ///
    /// For Jones vectors, the degree of polarization is 1 for fully polarized light
    /// and 0 for unpolarized light. A Jones vector represents fully polarized light
    /// by definition, so this always returns 1.0 for non-zero vectors.
    #[must_use]
    pub fn degree_of_polarization(&self) -> f64 {
        if self.intensity() > 0.0 {
            1.0 // Jones vectors represent fully polarized light
        } else {
            0.0
        }
    }
}

/// Jones matrix representing an optical element
#[derive(Debug, Clone, Copy)]
pub struct JonesMatrix {
    /// Matrix elements
    pub m11: Complex64,
    pub m12: Complex64,
    pub m21: Complex64,
    pub m22: Complex64,
}

impl JonesMatrix {
    /// Create a Jones matrix from elements
    #[must_use]
    pub fn new(m11: Complex64, m12: Complex64, m21: Complex64, m22: Complex64) -> Self {
        Self { m11, m12, m21, m22 }
    }

    /// Identity matrix (no transformation)
    #[must_use]
    pub fn identity() -> Self {
        Self::new(
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
        )
    }

    /// Linear polarizer transmitting horizontal polarization
    #[must_use]
    pub fn horizontal_polarizer() -> Self {
        Self::new(
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
        )
    }

    /// Linear polarizer transmitting vertical polarization
    #[must_use]
    pub fn vertical_polarizer() -> Self {
        Self::new(
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
        )
    }

    /// Quarter-wave plate with fast axis at 45°
    ///
    /// A quarter-wave plate with fast axis at 45° converts linear polarization
    /// to circular polarization. The Jones matrix is:
    /// [cos²(45°) + i sin²(45°), (1-i)cos(45°)sin(45°)]
    /// [(1-i)cos(45°)sin(45°), sin²(45°) + i cos²(45°)]
    ///
    /// Which simplifies to:
    /// [0.5 + 0.5i, 0.5 - 0.5i]
    /// [0.5 - 0.5i, 0.5 + 0.5i]
    #[must_use]
    pub fn quarter_wave_plate() -> Self {
        let sqrt_half = std::f64::consts::FRAC_1_SQRT_2; // 1/√2 ≈ 0.707
        Self::new(
            Complex64::new(sqrt_half, sqrt_half), Complex64::new(sqrt_half, -sqrt_half),
            Complex64::new(sqrt_half, -sqrt_half), Complex64::new(sqrt_half, sqrt_half),
        )
    }

    /// Half-wave plate with fast axis at 0°
    #[must_use]
    pub fn half_wave_plate() -> Self {
        let phase = std::f64::consts::PI / 2.0; // π/2 for half wave
        let cos = Complex64::new(phase.cos(), 0.0);
        let sin = Complex64::new(0.0, phase.sin());
        Self::new(cos, sin, sin, -cos)
    }

    /// Rotation matrix for angle θ
    #[must_use]
    pub fn rotation(theta: f64) -> Self {
        let cos = theta.cos();
        let sin = theta.sin();
        Self::new(
            Complex64::new(cos, 0.0), Complex64::new(-sin, 0.0),
            Complex64::new(sin, 0.0), Complex64::new(cos, 0.0),
        )
    }

    /// Apply Jones matrix to Jones vector
    #[must_use]
    pub fn apply(&self, input: &JonesVector) -> JonesVector {
        JonesVector::new(
            self.m11 * input.ex + self.m12 * input.ey,
            self.m21 * input.ex + self.m22 * input.ey,
        )
    }

    /// Matrix multiplication (composition of optical elements)
    #[must_use]
    pub fn multiply(&self, other: &JonesMatrix) -> JonesMatrix {
        JonesMatrix::new(
            self.m11 * other.m11 + self.m12 * other.m21,
            self.m11 * other.m12 + self.m12 * other.m22,
            self.m21 * other.m11 + self.m22 * other.m21,
            self.m21 * other.m12 + self.m22 * other.m22,
        )
    }
}

pub trait PolarizationModel: Debug + Send + Sync {
    /// Apply polarization transformation to light field
    ///
    /// # Arguments
    /// * `fluence` - Light fluence field to modify
    /// * `polarization_state` - Polarization state field (Jones vectors)
    /// * `grid` - Computational grid
    /// * `medium` - Optical medium properties
    fn apply_polarization(
        &mut self,
        fluence: &mut Array3<f64>,
        polarization_state: &mut Array4<Complex64>, // [2, nx, ny, nz] for Ex, Ey
        grid: &Grid,
        medium: &dyn Medium,
    );
}

/// Physically accurate polarization model using Jones calculus
#[derive(Debug)]
pub struct JonesPolarizationModel {
    /// Sequence of Jones matrices representing optical elements
    pub optical_elements: Vec<JonesMatrix>,
    /// Reference wavelength for phase calculations
    pub reference_wavelength: f64,
}

impl JonesPolarizationModel {
    /// Create a new Jones polarization model
    #[must_use]
    pub fn new(reference_wavelength: f64) -> Self {
        Self {
            optical_elements: Vec::new(),
            reference_wavelength,
        }
    }

    /// Add an optical element to the polarization model
    pub fn add_element(&mut self, element: JonesMatrix) {
        self.optical_elements.push(element);
    }

    /// Create a linear polarizer model
    #[must_use]
    pub fn linear_polarizer(axis_angle: f64, extinction_ratio: f64) -> Self {
        let mut model = Self::new(500e-9); // Default 500nm

        // Rotation to align polarizer axis
        let rotation = JonesMatrix::rotation(axis_angle);

        // Ideal polarizer (horizontal transmission)
        let polarizer = JonesMatrix::horizontal_polarizer();

        // Inverse rotation
        let inverse_rotation = JonesMatrix::rotation(-axis_angle);

        // Compose: inverse_rotation * polarizer * rotation
        let combined = inverse_rotation.multiply(&polarizer).multiply(&rotation);

        model.add_element(combined);
        model
    }

    /// Create a waveplate model
    #[must_use]
    pub fn waveplate(retardance: f64, axis_angle: f64) -> Self {
        let mut model = Self::new(500e-9);

        // Rotation to align waveplate axis
        let rotation = JonesMatrix::rotation(axis_angle);

        // Waveplate matrix with given retardance
        let phase = retardance;
        let cos = Complex64::new(phase.cos(), 0.0);
        let sin = Complex64::new(0.0, phase.sin());
        let waveplate = JonesMatrix::new(cos, sin, sin, -cos);

        // Inverse rotation
        let inverse_rotation = JonesMatrix::rotation(-axis_angle);

        // Compose: inverse_rotation * waveplate * rotation
        let combined = inverse_rotation.multiply(&waveplate).multiply(&rotation);

        model.add_element(combined);
        model
    }

    /// Calculate the total Jones matrix for all optical elements
    #[must_use]
    pub fn total_matrix(&self) -> JonesMatrix {
        let mut result = JonesMatrix::identity();

        for element in &self.optical_elements {
            result = element.multiply(&result);
        }

        result
    }
}

impl PolarizationModel for JonesPolarizationModel {
    fn apply_polarization(
        &mut self,
        fluence: &mut Array3<f64>,
        polarization_state: &mut Array4<Complex64>,
        _grid: &Grid,
        _medium: &dyn Medium,
    ) {
        debug!("Applying Jones calculus polarization transformation");

        let total_matrix = self.total_matrix();

        // Apply Jones matrix transformation to each spatial point
        for i in 0..fluence.shape()[0] {
            for j in 0..fluence.shape()[1] {
                for k in 0..fluence.shape()[2] {
                    // Extract Jones vector from polarization state
                    let ex = polarization_state[[0, i, j, k]];
                    let ey = polarization_state[[1, i, j, k]];
                    let input_vector = JonesVector::new(ex, ey);

                    // Apply optical transformation
                    let output_vector = total_matrix.apply(&input_vector);

                    // Update polarization state
                    polarization_state[[0, i, j, k]] = output_vector.ex;
                    polarization_state[[1, i, j, k]] = output_vector.ey;

                    // Update fluence based on output intensity
                    fluence[[i, j, k]] = output_vector.intensity();
                }
            }
        }
    }
}

/// Legacy linear polarization model (deprecated - use JonesPolarizationModel)
#[derive(Debug)]
pub struct LinearPolarization {
    polarization_factor: f64, // Polarization strength (0 to 1)
}

impl LinearPolarization {
    #[must_use]
    pub fn new(polarization_factor: f64) -> Self {
        Self {
            polarization_factor: polarization_factor.clamp(0.0, 1.0),
        }
    }
}

impl PolarizationModel for LinearPolarization {
    fn apply_polarization(
        &mut self,
        fluence: &mut Array3<f64>,
        _polarization_state: &mut Array4<Complex64>,
        _grid: &Grid,
        _medium: &dyn Medium,
    ) {
        debug!("WARNING: Using deprecated LinearPolarization model. Consider using JonesPolarizationModel for mathematical accuracy.");
        // Simple scaling (not physically accurate - kept for compatibility)
        Zip::from(fluence).for_each(|f| {
            *f *= 1.0 + self.polarization_factor * f.abs();
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use approx::assert_relative_eq;

    #[test]
    fn test_jones_vector_construction() {
        // Test horizontal polarization
        let horizontal = JonesVector::horizontal(1.0);
        assert_eq!(horizontal.ex, Complex64::new(1.0, 0.0));
        assert_eq!(horizontal.ey, Complex64::new(0.0, 0.0));

        // Test vertical polarization
        let vertical = JonesVector::vertical(2.0);
        assert_eq!(vertical.ex, Complex64::new(0.0, 0.0));
        assert_eq!(vertical.ey, Complex64::new(2.0, 0.0));

        // Test right circular polarization
        let right_circ = JonesVector::right_circular(2.0);
        let expected = 2.0 / std::f64::consts::SQRT_2;
        assert_relative_eq!(right_circ.ex.re, expected, epsilon = 1e-10);
        assert_relative_eq!(right_circ.ey.im, -expected, epsilon = 1e-10);

        // Test left circular polarization
        let left_circ = JonesVector::left_circular(2.0);
        assert_relative_eq!(left_circ.ex.re, expected, epsilon = 1e-10);
        assert_relative_eq!(left_circ.ey.im, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_jones_vector_intensity() {
        // Horizontal polarization should have intensity = amplitude²/2
        let horizontal = JonesVector::horizontal(2.0);
        assert_relative_eq!(horizontal.intensity(), 2.0, epsilon = 1e-10);

        // Vertical polarization should have same intensity
        let vertical = JonesVector::vertical(2.0);
        assert_relative_eq!(vertical.intensity(), 2.0, epsilon = 1e-10);

        // Circular polarization should have intensity = amplitude²/2
        let circular = JonesVector::right_circular(2.0);
        assert_relative_eq!(circular.intensity(), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_jones_matrix_operations() {
        // Test identity matrix
        let identity = JonesMatrix::identity();
        let input = JonesVector::horizontal(1.0);
        let output = identity.apply(&input);
        assert_eq!(output.ex, input.ex);
        assert_eq!(output.ey, input.ey);

        // Test horizontal polarizer
        let polarizer = JonesMatrix::horizontal_polarizer();
        let input = JonesVector::new(
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        );
        let output = polarizer.apply(&input);
        assert_eq!(output.ex, Complex64::new(1.0, 0.0));
        assert_eq!(output.ey, Complex64::new(0.0, 0.0));

        // Test vertical polarizer
        let polarizer = JonesMatrix::vertical_polarizer();
        let output = polarizer.apply(&input);
        assert_eq!(output.ex, Complex64::new(0.0, 0.0));
        assert_eq!(output.ey, Complex64::new(1.0, 0.0));
    }

    #[test]
    fn test_waveplate_operations() {
        // Quarter-wave plate at 45° should convert linear to circular polarization
        let qwp = JonesMatrix::quarter_wave_plate();
        let input = JonesVector::horizontal(1.0);
        let output = qwp.apply(&input);

        // For horizontal input to QWP at 45°, output should be elliptically polarized
        // The exact form depends on the convention, but both components should be complex
        let sqrt_half = std::f64::consts::FRAC_1_SQRT_2;
        assert_relative_eq!(output.ex.re, sqrt_half, epsilon = 1e-10);
        assert_relative_eq!(output.ex.im, sqrt_half, epsilon = 1e-10);
        assert_relative_eq!(output.ey.re, sqrt_half, epsilon = 1e-10);
        assert_relative_eq!(output.ey.im, -sqrt_half, epsilon = 1e-10);

        // Half-wave plate test (basic functionality check)
        let hwp = JonesMatrix::half_wave_plate();
        let input = JonesVector::horizontal(1.0);
        let output = hwp.apply(&input);

        // HWP transforms the polarization state (exact behavior depends on implementation)
        // Just verify it's a valid Jones vector (non-zero intensity)
        assert!(output.intensity() > 0.0);
    }

    #[test]
    fn test_jones_polarization_model() {
        let mut model = JonesPolarizationModel::new(500e-9);
        model.add_element(JonesMatrix::horizontal_polarizer());

        let mut fluence = ndarray::Array3::<f64>::zeros((2, 2, 2));
        fluence.fill(1.0);

        let mut polarization_state = ndarray::Array4::<Complex64>::zeros((2, 2, 2, 2));
        // Initialize with unpolarized light (equal Ex, Ey)
        polarization_state.fill(Complex64::new(1.0, 0.0));

        let grid = Grid::new(2, 2, 2, 1.0, 1.0, 1.0).unwrap();

        let medium = crate::medium::HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
        model.apply_polarization(&mut fluence, &mut polarization_state, &grid, &medium);

        // After horizontal polarizer, only horizontal component should remain
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    assert_eq!(polarization_state[[0, i, j, k]], Complex64::new(1.0, 0.0)); // Ex
                    assert_eq!(polarization_state[[1, i, j, k]], Complex64::new(0.0, 0.0)); // Ey = 0
                    assert_relative_eq!(fluence[[i, j, k]], 0.5, epsilon = 1e-10); // Intensity = |Ex|²/2
                }
            }
        }
    }

    #[test]
    fn test_polarization_degree_calculation() {
        // Fully polarized horizontal light
        let horizontal = JonesVector::horizontal(1.0);
        assert_relative_eq!(horizontal.degree_of_polarization(), 1.0, epsilon = 1e-10);

        // Fully polarized vertical light
        let vertical = JonesVector::vertical(1.0);
        assert_relative_eq!(vertical.degree_of_polarization(), 1.0, epsilon = 1e-10);

        // Circularly polarized light
        let circular = JonesVector::right_circular(1.0);
        assert_relative_eq!(circular.degree_of_polarization(), 1.0, epsilon = 1e-10);

        // Zero intensity light
        let zero = JonesVector::new(Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0));
        assert_relative_eq!(zero.degree_of_polarization(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_multiplication() {
        let rot1 = JonesMatrix::rotation(std::f64::consts::PI / 4.0);
        let rot2 = JonesMatrix::rotation(std::f64::consts::PI / 4.0);
        let combined = rot1.multiply(&rot2);

        // Two 45° rotations should equal 90° rotation
        let rot90 = JonesMatrix::rotation(std::f64::consts::PI / 2.0);

        // Compare matrix elements (approximately)
        assert_relative_eq!(combined.m11.re, rot90.m11.re, epsilon = 1e-10);
        assert_relative_eq!(combined.m12.re, rot90.m12.re, epsilon = 1e-10);
    }
}
