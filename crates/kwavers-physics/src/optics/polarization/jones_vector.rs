//! Jones vector representing the polarization state of light
//!
//! ## Mathematical Foundation
//!
//! Jones vectors describe the electric field polarization state:
//! ```text
//! |E⟩ = |E_x|
//!       |E_y|
//! ```
//!
//! Intensity: I = (|E_x|² + |E_y|²) / 2
//!
//! ## References
//!
//! - Jones, R. C. (1941). "A new calculus for the treatment of optical systems"

use num_complex::Complex64;

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
        Self::new(Complex64::new(norm, 0.0), Complex64::new(0.0, -norm))
    }

    /// Create left-circularly polarized light
    #[must_use]
    pub fn left_circular(amplitude: f64) -> Self {
        let norm = amplitude / std::f64::consts::SQRT_2;
        Self::new(Complex64::new(norm, 0.0), Complex64::new(0.0, norm))
    }

    /// Calculate the intensity (time-averaged power density)
    #[must_use]
    pub fn intensity(&self) -> f64 {
        (self.ex.norm_sqr() + self.ey.norm_sqr()) / 2.0
    }

    /// Calculate the degree of polarization
    ///
    /// Jones vectors represent fully polarized light by definition,
    /// so this returns 1.0 for non-zero vectors and 0.0 for zero vectors.
    #[must_use]
    pub fn degree_of_polarization(&self) -> f64 {
        if self.intensity() > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}
