//! Jones matrix representing optical elements
//!
//! ## Mathematical Foundation
//!
//! Jones matrices transform polarization states:
//! ```text
//! |E'⟩ = M |E⟩
//! ```
//!
//! Composition: M_total = M_n · ... · M_2 · M_1
//!
//! ## References
//!
//! - Jones, R. C. (1941). "A new calculus for the treatment of optical systems"
//! - Born, M., & Wolf, E. (1999). Principles of Optics

use super::jones_vector::JonesVector;
use num_complex::Complex64;

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
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        )
    }

    /// Linear polarizer transmitting horizontal polarization
    #[must_use]
    pub fn horizontal_polarizer() -> Self {
        Self::new(
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        )
    }

    /// Linear polarizer transmitting vertical polarization
    #[must_use]
    pub fn vertical_polarizer() -> Self {
        Self::new(
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        )
    }

    /// Quarter-wave plate with fast axis at 45°
    ///
    /// Converts linear polarization to circular polarization.
    /// Matrix (with 1/√2 normalization):
    /// ```text
    /// [1/√2 + i/√2,  1/√2 - i/√2]
    /// [1/√2 - i/√2,  1/√2 + i/√2]
    /// ```
    #[must_use]
    pub fn quarter_wave_plate() -> Self {
        let sqrt_half = std::f64::consts::FRAC_1_SQRT_2;
        Self::new(
            Complex64::new(sqrt_half, sqrt_half),
            Complex64::new(sqrt_half, -sqrt_half),
            Complex64::new(sqrt_half, -sqrt_half),
            Complex64::new(sqrt_half, sqrt_half),
        )
    }

    /// Half-wave plate with fast axis at 0°
    #[must_use]
    pub fn half_wave_plate() -> Self {
        let phase = std::f64::consts::PI / 2.0;
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
            Complex64::new(cos, 0.0),
            Complex64::new(-sin, 0.0),
            Complex64::new(sin, 0.0),
            Complex64::new(cos, 0.0),
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
