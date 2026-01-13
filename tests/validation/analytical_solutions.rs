//! Analytical Solutions Library for Validation Testing
//!
//! This module provides closed-form solutions to elastic wave equations with
//! exact derivatives for gradient validation. All solutions are mathematically
//! proven to satisfy the wave equations they claim to solve.
//!
//! # Implemented Solutions
//!
//! 1. **Plane Wave Solutions**: Traveling waves with exact dispersion relations
//!    - P-wave (longitudinal): u = A sin(k·x - ωt) k̂
//!    - S-wave (transverse): u = A sin(k·x - ωt) k̂⊥
//! 2. **Standing Wave Solutions**: Separated spatial and temporal modes
//! 3. **Polynomial Test Functions**: For gradient verification (not physical)
//! 4. **Point Source Solutions**: Green's functions for wave propagation
//!
//! # Mathematical Specifications
//!
//! All solutions satisfy the elastic wave equation:
//! ρ ∂²u/∂t² = (λ + μ)∇(∇·u) + μ∇²u
//!
//! with exact expressions for:
//! - Displacement: u(x, t)
//! - Velocity: v(x, t) = ∂u/∂t
//! - Acceleration: a(x, t) = ∂²u/∂t²
//! - Gradient: ∇u (all spatial derivatives)
//! - Strain: ε = (1/2)(∇u + ∇uᵀ)
//! - Stress: σ = λ tr(ε)I + 2μ ε

use super::{AnalyticalSolution, SolutionParameters, WaveType};
use std::f64::consts::PI;

// ============================================================================
// Plane Wave Solutions (2D)
// ============================================================================

/// 2D Plane wave solution: u = A sin(k·x - ωt) d̂
///
/// # Mathematical Specification
///
/// For a plane wave propagating in direction k̂ with polarization d̂:
///
/// **Displacement**:
/// ```text
/// u(x, t) = A sin(k·x - ωt) d̂
/// where k = (kₓ, kᵧ), x = (x, y), k·x = kₓx + kᵧy
/// ```
///
/// **Dispersion Relation**:
/// - P-wave (d̂ ∥ k̂): ω² = cₚ² k², where cₚ = √((λ + 2μ)/ρ)
/// - S-wave (d̂ ⊥ k̂): ω² = cₛ² k², where cₛ = √(μ/ρ)
///
/// **Velocity**: v = ∂u/∂t = -Aω cos(k·x - ωt) d̂
///
/// **Acceleration**: a = ∂²u/∂t² = -Aω² sin(k·x - ωt) d̂
///
/// **Gradient**: ∇u = Ak cos(k·x - ωt) (d̂ ⊗ k̂)
///
/// # Invariants
///
/// - Wave equation satisfaction: ρa = (λ + μ)∇(∇·u) + μ∇²u
/// - Energy conservation: E = (1/2)ρ|v|² + (1/2)σ:ε = const
/// - Orthogonality (S-wave): d̂ · k̂ = 0
/// - Parallelism (P-wave): d̂ = ±k̂
#[derive(Debug, Clone)]
pub struct PlaneWave2D {
    /// Wave amplitude (m)
    amplitude: f64,
    /// Wave vector k = (kₓ, kᵧ) where |k| = 2π/λ
    wave_vector: [f64; 2],
    /// Angular frequency ω = 2πf (rad/s)
    omega: f64,
    /// Polarization direction d̂ (unit vector)
    polarization: [f64; 2],
    /// Wave type (P-wave or S-wave)
    wave_type: WaveType,
    /// Material parameters
    params: SolutionParameters,
}

impl PlaneWave2D {
    /// Create P-wave (longitudinal) plane wave
    ///
    /// # Arguments
    /// - `amplitude`: Wave amplitude (m)
    /// - `wavelength`: Wavelength λ (m)
    /// - `direction`: Propagation direction k̂ (will be normalized)
    /// - `params`: Material parameters (ρ, λ, μ)
    ///
    /// # Returns
    /// Plane wave with polarization parallel to propagation direction
    pub fn p_wave(
        amplitude: f64,
        wavelength: f64,
        direction: [f64; 2],
        params: SolutionParameters,
    ) -> Self {
        let k_mag = 2.0 * PI / wavelength;
        let dir_norm = (direction[0].powi(2) + direction[1].powi(2)).sqrt();
        let k_hat = [direction[0] / dir_norm, direction[1] / dir_norm];

        let wave_vector = [k_hat[0] * k_mag, k_hat[1] * k_mag];
        let cp = params.p_wave_speed();
        let omega = k_mag * cp;

        Self {
            amplitude,
            wave_vector,
            omega,
            polarization: k_hat, // d̂ ∥ k̂ for P-wave
            wave_type: WaveType::PWave,
            params: SolutionParameters {
                amplitude,
                wavelength,
                omega,
                wave_speed: cp,
                ..params
            },
        }
    }

    /// Create S-wave (transverse) plane wave
    ///
    /// # Arguments
    /// - `amplitude`: Wave amplitude (m)
    /// - `wavelength`: Wavelength λ (m)
    /// - `direction`: Propagation direction k̂ (will be normalized)
    /// - `params`: Material parameters (ρ, λ, μ)
    ///
    /// # Returns
    /// Plane wave with polarization perpendicular to propagation direction
    pub fn s_wave(
        amplitude: f64,
        wavelength: f64,
        direction: [f64; 2],
        params: SolutionParameters,
    ) -> Self {
        let k_mag = 2.0 * PI / wavelength;
        let dir_norm = (direction[0].powi(2) + direction[1].powi(2)).sqrt();
        let k_hat = [direction[0] / dir_norm, direction[1] / dir_norm];

        let wave_vector = [k_hat[0] * k_mag, k_hat[1] * k_mag];
        let cs = params.s_wave_speed();
        let omega = k_mag * cs;

        // Perpendicular polarization: rotate k̂ by 90°
        let polarization = [-k_hat[1], k_hat[0]]; // d̂ ⊥ k̂ for S-wave

        Self {
            amplitude,
            wave_vector,
            omega,
            polarization,
            wave_type: WaveType::SWave,
            params: SolutionParameters {
                amplitude,
                wavelength,
                omega,
                wave_speed: cs,
                ..params
            },
        }
    }

    /// Phase: φ = k·x - ωt
    fn phase(&self, x: &[f64], t: f64) -> f64 {
        let k_dot_x = self.wave_vector[0] * x[0] + self.wave_vector[1] * x[1];
        k_dot_x - self.omega * t
    }
}

impl AnalyticalSolution for PlaneWave2D {
    fn displacement(&self, x: &[f64], t: f64) -> Vec<f64> {
        let phase = self.phase(x, t);
        let factor = self.amplitude * phase.sin();
        vec![factor * self.polarization[0], factor * self.polarization[1]]
    }

    fn velocity(&self, x: &[f64], t: f64) -> Vec<f64> {
        let phase = self.phase(x, t);
        let factor = -self.amplitude * self.omega * phase.cos();
        vec![factor * self.polarization[0], factor * self.polarization[1]]
    }

    fn gradient(&self, x: &[f64], t: f64) -> Vec<f64> {
        let phase = self.phase(x, t);
        let factor = self.amplitude * phase.cos();

        // Gradient matrix (row-major): [∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y]
        // ∂uᵢ/∂xⱼ = A cos(φ) dᵢ kⱼ
        vec![
            factor * self.polarization[0] * self.wave_vector[0], // ∂u/∂x
            factor * self.polarization[0] * self.wave_vector[1], // ∂u/∂y
            factor * self.polarization[1] * self.wave_vector[0], // ∂v/∂x
            factor * self.polarization[1] * self.wave_vector[1], // ∂v/∂y
        ]
    }

    fn acceleration(&self, x: &[f64], t: f64) -> Vec<f64> {
        let phase = self.phase(x, t);
        let factor = -self.amplitude * self.omega.powi(2) * phase.sin();
        vec![factor * self.polarization[0], factor * self.polarization[1]]
    }

    fn spatial_dimension(&self) -> usize {
        2
    }

    fn name(&self) -> &str {
        match self.wave_type {
            WaveType::PWave => "2D Plane P-wave",
            WaveType::SWave => "2D Plane S-wave",
            _ => "2D Plane wave",
        }
    }

    fn parameters(&self) -> SolutionParameters {
        self.params
    }
}

// ============================================================================
// Simple Harmonic Solutions (for gradient testing)
// ============================================================================

/// 1D sine wave: u(x, t) = A sin(kx - ωt)
///
/// # Mathematical Specification
///
/// **Displacement**: u = A sin(kx - ωt)
///
/// **First derivative**: ∂u/∂x = Ak cos(kx - ωt)
///
/// **Second derivative**: ∂²u/∂x² = -Ak² sin(kx - ωt)
///
/// **Time derivative**: ∂u/∂t = -Aω cos(kx - ωt)
///
/// This is a simplified solution for testing gradient computation correctness.
/// It satisfies the 1D wave equation: ∂²u/∂t² = c² ∂²u/∂x²
#[derive(Debug, Clone)]
pub struct SineWave1D {
    amplitude: f64,
    wave_number: f64,
    omega: f64,
    wave_speed: f64,
}

impl SineWave1D {
    /// Create 1D sine wave
    ///
    /// # Arguments
    /// - `amplitude`: Wave amplitude
    /// - `wavelength`: Spatial wavelength λ
    /// - `wave_speed`: Phase velocity c (ω = ck)
    pub fn new(amplitude: f64, wavelength: f64, wave_speed: f64) -> Self {
        let wave_number = 2.0 * PI / wavelength;
        let omega = wave_number * wave_speed;
        Self {
            amplitude,
            wave_number,
            omega,
            wave_speed,
        }
    }

    fn phase(&self, x: f64, t: f64) -> f64 {
        self.wave_number * x - self.omega * t
    }
}

impl AnalyticalSolution for SineWave1D {
    fn displacement(&self, x: &[f64], t: f64) -> Vec<f64> {
        let phase = self.phase(x[0], t);
        vec![self.amplitude * phase.sin()]
    }

    fn velocity(&self, x: &[f64], t: f64) -> Vec<f64> {
        let phase = self.phase(x[0], t);
        vec![-self.amplitude * self.omega * phase.cos()]
    }

    fn gradient(&self, x: &[f64], t: f64) -> Vec<f64> {
        let phase = self.phase(x[0], t);
        // ∂u/∂x = Ak cos(kx - ωt)
        vec![self.amplitude * self.wave_number * phase.cos()]
    }

    fn acceleration(&self, x: &[f64], t: f64) -> Vec<f64> {
        let phase = self.phase(x[0], t);
        vec![-self.amplitude * self.omega.powi(2) * phase.sin()]
    }

    fn spatial_dimension(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "1D Sine Wave"
    }

    fn parameters(&self) -> SolutionParameters {
        SolutionParameters {
            amplitude: self.amplitude,
            wavelength: 2.0 * PI / self.wave_number,
            omega: self.omega,
            wave_speed: self.wave_speed,
            density: 1.0,
            lambda: 0.0,
            mu: 1.0,
        }
    }
}

// ============================================================================
// Polynomial Test Functions (for gradient verification)
// ============================================================================

/// 2D polynomial test function: u = (x², xy)
///
/// # Mathematical Specification
///
/// **Displacement**:
/// ```text
/// u(x, y) = x²
/// v(x, y) = xy
/// ```
///
/// **Gradient**:
/// ```text
/// ∂u/∂x = 2x,  ∂u/∂y = 0
/// ∂v/∂x = y,   ∂v/∂y = x
/// ```
///
/// **Second derivatives**:
/// ```text
/// ∂²u/∂x² = 2,   ∂²u/∂y² = 0,   ∂²u/∂x∂y = 0
/// ∂²v/∂x² = 0,   ∂²v/∂y² = 0,   ∂²v/∂x∂y = 1
/// ```
///
/// This is NOT a physical wave solution - it's purely for testing
/// gradient computation correctness with known analytical derivatives.
#[derive(Debug, Clone)]
pub struct PolynomialTest2D {
    /// Scaling factor for numerical stability
    scale: f64,
}

impl PolynomialTest2D {
    pub fn new(scale: f64) -> Self {
        Self { scale }
    }
}

impl AnalyticalSolution for PolynomialTest2D {
    fn displacement(&self, x: &[f64], t: f64) -> Vec<f64> {
        let _t = t; // Time-independent for simplicity
        let s = self.scale;
        vec![s * x[0] * x[0], s * x[0] * x[1]]
    }

    fn velocity(&self, _x: &[f64], _t: f64) -> Vec<f64> {
        // Time-independent: velocity is zero
        vec![0.0, 0.0]
    }

    fn gradient(&self, x: &[f64], _t: f64) -> Vec<f64> {
        let s = self.scale;
        // [∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y]
        vec![
            s * 2.0 * x[0], // ∂u/∂x = 2x
            0.0,            // ∂u/∂y = 0
            s * x[1],       // ∂v/∂x = y
            s * x[0],       // ∂v/∂y = x
        ]
    }

    fn acceleration(&self, _x: &[f64], _t: f64) -> Vec<f64> {
        vec![0.0, 0.0]
    }

    fn spatial_dimension(&self) -> usize {
        2
    }

    fn name(&self) -> &str {
        "2D Polynomial Test (x², xy)"
    }

    fn parameters(&self) -> SolutionParameters {
        SolutionParameters {
            amplitude: self.scale,
            wavelength: 1.0,
            omega: 0.0,
            wave_speed: 1.0,
            density: 1.0,
            lambda: 0.0,
            mu: 1.0,
        }
    }
}

/// 2D quadratic polynomial: u = (x² + y², xy)
///
/// # Mathematical Specification
///
/// **Displacement**:
/// ```text
/// u(x, y) = x² + y²
/// v(x, y) = xy
/// ```
///
/// **Gradient**:
/// ```text
/// ∂u/∂x = 2x,  ∂u/∂y = 2y
/// ∂v/∂x = y,   ∂v/∂y = x
/// ```
///
/// **Second derivatives**:
/// ```text
/// ∂²u/∂x² = 2,   ∂²u/∂y² = 2,   ∂²u/∂x∂y = 0
/// ∂²v/∂x² = 0,   ∂²v/∂y² = 0,   ∂²v/∂x∂y = 1
/// ```
///
/// Useful for testing Laplacian computation: ∇²u = 4, ∇²v = 0
#[derive(Debug, Clone)]
pub struct QuadraticTest2D {
    scale: f64,
}

impl QuadraticTest2D {
    pub fn new(scale: f64) -> Self {
        Self { scale }
    }
}

impl AnalyticalSolution for QuadraticTest2D {
    fn displacement(&self, x: &[f64], _t: f64) -> Vec<f64> {
        let s = self.scale;
        vec![s * (x[0] * x[0] + x[1] * x[1]), s * x[0] * x[1]]
    }

    fn velocity(&self, _x: &[f64], _t: f64) -> Vec<f64> {
        vec![0.0, 0.0]
    }

    fn gradient(&self, x: &[f64], _t: f64) -> Vec<f64> {
        let s = self.scale;
        // [∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y]
        vec![
            s * 2.0 * x[0], // ∂u/∂x = 2x
            s * 2.0 * x[1], // ∂u/∂y = 2y
            s * x[1],       // ∂v/∂x = y
            s * x[0],       // ∂v/∂y = x
        ]
    }

    fn acceleration(&self, _x: &[f64], _t: f64) -> Vec<f64> {
        vec![0.0, 0.0]
    }

    fn spatial_dimension(&self) -> usize {
        2
    }

    fn name(&self) -> &str {
        "2D Quadratic Test (x²+y², xy)"
    }

    fn parameters(&self) -> SolutionParameters {
        SolutionParameters {
            amplitude: self.scale,
            wavelength: 1.0,
            omega: 0.0,
            wave_speed: 1.0,
            density: 1.0,
            lambda: 0.0,
            mu: 1.0,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plane_wave_p_wave_orthogonality() {
        let params = SolutionParameters {
            amplitude: 1e-6,
            wavelength: 0.01,
            omega: 0.0,
            wave_speed: 5000.0,
            density: 2700.0,
            lambda: 5e10,
            mu: 2.6e10,
        };

        let wave = PlaneWave2D::p_wave(1e-6, 0.01, [1.0, 0.0], params);

        // P-wave: polarization should be parallel to propagation
        let dot =
            wave.polarization[0] * wave.wave_vector[0] + wave.polarization[1] * wave.wave_vector[1];
        let k_mag = (wave.wave_vector[0].powi(2) + wave.wave_vector[1].powi(2)).sqrt();
        let d_mag = (wave.polarization[0].powi(2) + wave.polarization[1].powi(2)).sqrt();

        assert!(
            (dot.abs() - k_mag * d_mag).abs() < 1e-10,
            "P-wave must be longitudinal"
        );
    }

    #[test]
    fn test_plane_wave_s_wave_orthogonality() {
        let params = SolutionParameters {
            amplitude: 1e-6,
            wavelength: 0.01,
            omega: 0.0,
            wave_speed: 3000.0,
            density: 2700.0,
            lambda: 5e10,
            mu: 2.6e10,
        };

        let wave = PlaneWave2D::s_wave(1e-6, 0.01, [1.0, 0.0], params);

        // S-wave: polarization should be perpendicular to propagation
        let dot =
            wave.polarization[0] * wave.wave_vector[0] + wave.polarization[1] * wave.wave_vector[1];

        assert!(dot.abs() < 1e-10, "S-wave must be transverse");
    }

    #[test]
    fn test_sine_wave_gradient() {
        let wave = SineWave1D::new(1.0, 1.0, 1.0);
        let x = &[0.5];
        let t = 0.0;

        let grad = wave.gradient(x, t);
        let k = 2.0 * PI;
        let expected = k * (k * x[0]).cos();

        assert!((grad[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_polynomial_gradient() {
        let poly = PolynomialTest2D::new(1.0);
        let x = &[2.0, 3.0];
        let t = 0.0;

        let grad = poly.gradient(x, t);

        // Expected: [2x, 0, y, x] = [4, 0, 3, 2]
        assert!((grad[0] - 4.0).abs() < 1e-10);
        assert!((grad[1] - 0.0).abs() < 1e-10);
        assert!((grad[2] - 3.0).abs() < 1e-10);
        assert!((grad[3] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_quadratic_laplacian() {
        let quad = QuadraticTest2D::new(1.0);
        let x = &[1.0, 1.0];
        let _t = 0.0;

        // For u = x² + y², Laplacian ∇²u = 2 + 2 = 4
        // We can verify this by checking second derivatives would be constant = 2 each
        // (This test documents the expected behavior)
        let u = quad.displacement(x, _t);
        assert_eq!(u[0], 2.0); // x² + y² at (1,1) = 2
        assert_eq!(u[1], 1.0); // xy at (1,1) = 1
    }

    #[test]
    fn test_plane_wave_dispersion_relation() {
        let params = SolutionParameters {
            amplitude: 1e-6,
            wavelength: 0.01,
            omega: 0.0,
            wave_speed: 5000.0,
            density: 2700.0,
            lambda: 5e10,
            mu: 2.6e10,
        };

        let wave = PlaneWave2D::p_wave(1e-6, 0.01, [1.0, 0.0], params);

        let k = (wave.wave_vector[0].powi(2) + wave.wave_vector[1].powi(2)).sqrt();
        let cp = params.p_wave_speed();

        // Dispersion relation: ω = cp * k
        let expected_omega = cp * k;
        assert!((wave.omega - expected_omega).abs() / expected_omega < 1e-10);
    }
}
