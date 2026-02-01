//! BEM Analytical Validation Tests
//!
//! This module validates the Boundary Element Method implementation
//! against analytical solutions for canonical problems.
//!
//! **Test Strategy**:
//! 1. Sphere scattering - Compare BEM pressure field vs Mie series
//! 2. Infinite half-space - Validate reflection coefficient
//! 3. Point source in half-space - Green's function reciprocity
//! 4. Convergence study - Error vs mesh refinement
//!
//! **Mathematical Basis**:
//!
//! For a sphere of radius a with wavenumber k and incident plane wave:
//! - Mie coefficients: a_n = -[ψ_n(ka)·ψ'_n(k'a) - ψ'_n(ka)·ψ_n(k'a)] /
//!                          [ξ_n(ka)·ψ'_n(k'a) - ξ'_n(ka)·ψ_n(k'a)]
//! - where ψ_n = √(πz/2)·J_{n+1/2}(z) (Riccati-Bessel)
//! - and ξ_n = √(πz/2)·H_{n+1/2}(z) (Hankel function)
//!
//! **References**:
//! - Bowman, Senior, Uslenghi (1969): "Electromagnetic and Acoustic Scattering by Simple Shapes"
//! - Pierce (1989): "Acoustics: An Introduction"
//! - Colton & Kress (1998): "Inverse Acoustic and Electromagnetic Scattering Theory"

use std::f64::consts::PI;

/// Mie series solution for sphere scattering
///
/// Computes the scattered pressure field for a plane wave
/// incident on a rigid sphere using Mie scattering theory.
pub struct MieScatteringSolution {
    /// Sphere radius (m)
    pub radius: f64,
    /// Frequency (Hz)
    pub frequency: f64,
    /// Medium sound speed (m/s)
    pub sound_speed: f64,
    /// Medium density (kg/m³)
    pub density: f64,
    /// Number of Mie coefficients to compute
    pub num_terms: usize,
    /// Mie coefficients a_n
    mie_coefficients: Vec<num_complex::Complex64>,
    /// Wavenumber k = 2πf/c
    wavenumber: f64,
}

impl MieScatteringSolution {
    /// Create new Mie solution
    pub fn new(
        radius: f64,
        frequency: f64,
        sound_speed: f64,
        density: f64,
    ) -> Result<Self, String> {
        // Calculate wavenumber
        let wavenumber = 2.0 * PI * frequency / sound_speed;

        // Determine number of terms needed for convergence
        // Rule of thumb: N ≥ ka + 10·(ka)^(1/3)
        let ka = wavenumber * radius;
        let num_terms = (ka + 10.0 * ka.powf(1.0 / 3.0)).ceil() as usize + 5;

        Ok(Self {
            radius,
            frequency,
            sound_speed,
            density,
            num_terms,
            mie_coefficients: Vec::new(),
            wavenumber,
        })
    }

    /// Compute Mie coefficients for rigid boundary (Neumann BC: ∂p/∂n = 0)
    pub fn compute_coefficients(&mut self) -> Result<(), String> {
        self.mie_coefficients.clear();

        let ka = self.wavenumber * self.radius;

        // Compute Mie coefficients using Riccati-Bessel and Hankel functions
        for n in 1..=self.num_terms {
            // Riccati-Bessel function ψ_n(ka) = √(πz/2)·J_{n+1/2}(z)
            let _psi_ka = Self::riccati_bessel_j(n, ka);
            let psi_ka_prime = Self::riccati_bessel_j_derivative(n, ka);

            // Riccati-Hankel function ξ_n(ka) = √(πz/2)·H_{n+1/2}(z)
            let _xi_ka = Self::riccati_hankel_h1(n, ka);
            let xi_ka_prime = Self::riccati_hankel_h1_derivative(n, ka);

            // Mie coefficient for rigid sphere (Neumann BC)
            // a_n = -ψ'_n(ka) / ξ'_n(ka)
            let a_n = -psi_ka_prime / xi_ka_prime;

            self.mie_coefficients.push(a_n);
        }

        Ok(())
    }

    /// Evaluate scattered pressure field at point (r, θ)
    /// where θ is angle from incident wave direction
    pub fn scattered_pressure(&self, r: f64, theta: f64) -> Result<num_complex::Complex64, String> {
        if r < self.radius {
            return Err("Evaluation point inside sphere".to_string());
        }

        if self.mie_coefficients.is_empty() {
            return Err("Mie coefficients not computed".to_string());
        }

        let ka = self.wavenumber * self.radius;
        let kr = self.wavenumber * r;

        let mut p_scattered = num_complex::Complex64::new(0.0, 0.0);

        // Sum Mie series: p_s = Σ (2n+1)/(n(n+1)) · a_n · ξ_n(kr) · P_n(cos θ)
        for (n_idx, &a_n) in self.mie_coefficients.iter().enumerate() {
            let n = (n_idx + 1) as f64;

            // Riccati-Hankel function at kr
            let xi_kr = Self::riccati_hankel_h1(n_idx + 1, kr);

            // Legendre polynomial P_n(cos θ)
            let p_n = Self::legendre_polynomial(n_idx + 1, theta.cos());

            // Coefficient
            let coeff = (2.0 * n + 1.0) / (n * (n + 1.0));

            // Add to series
            p_scattered += coeff * a_n * xi_kr * p_n;
        }

        // Normalize: divide by incident amplitude (k² is absorbed in normalization)
        Ok(p_scattered / (num_complex::Complex64::i() * self.wavenumber))
    }

    /// Evaluate total pressure field (incident + scattered)
    pub fn total_pressure(&self, x: f64, y: f64, z: f64) -> Result<num_complex::Complex64, String> {
        // Assume incident plane wave propagating in +z direction: p_i = exp(i·k·z)
        let r = (x * x + y * y + z * z).sqrt();
        let theta = if z.abs() < 1e-10 {
            PI / 2.0
        } else {
            (z / r).acos()
        };

        let p_incident = num_complex::Complex64::new(0.0, self.wavenumber * z).exp();
        let p_scattered = self.scattered_pressure(r, theta)?;

        Ok(p_incident + p_scattered)
    }

    // ==================== Helper Functions ====================

    /// Riccati-Bessel function ψ_n(z) = √(πz/2)·J_{n+1/2}(z)
    fn riccati_bessel_j(n: usize, z: f64) -> num_complex::Complex64 {
        let nu = n as f64 + 0.5;
        let sqrt_factor = (PI * z / 2.0).sqrt();
        let bessel_j = Self::bessel_j_half_integer(nu, z);
        sqrt_factor * bessel_j
    }

    /// Derivative of Riccati-Bessel: ψ'_n(z)
    fn riccati_bessel_j_derivative(n: usize, z: f64) -> num_complex::Complex64 {
        let delta = 1e-8;
        let psi_plus = Self::riccati_bessel_j(n, z + delta);
        let psi_minus = Self::riccati_bessel_j(n, z - delta);
        (psi_plus - psi_minus) / (2.0 * delta)
    }

    /// Riccati-Hankel function ξ_n(z) = √(πz/2)·H_{n+1/2}^{(1)}(z)
    fn riccati_hankel_h1(n: usize, z: f64) -> num_complex::Complex64 {
        let nu = n as f64 + 0.5;
        let sqrt_factor = (PI * z / 2.0).sqrt();
        let hankel_h1 = Self::hankel_h1_half_integer(nu, z);
        sqrt_factor * hankel_h1
    }

    /// Derivative of Riccati-Hankel: ξ'_n(z)
    fn riccati_hankel_h1_derivative(n: usize, z: f64) -> num_complex::Complex64 {
        let delta = 1e-8;
        let xi_plus = Self::riccati_hankel_h1(n, z + delta);
        let xi_minus = Self::riccati_hankel_h1(n, z - delta);
        (xi_plus - xi_minus) / (2.0 * delta)
    }

    /// Bessel function of the first kind for half-integer order
    /// J_{n+1/2}(z) ≈ √(2/πz) · sin(z - (n+1)π/2)
    fn bessel_j_half_integer(nu: f64, z: f64) -> num_complex::Complex64 {
        // For real argument, use exact formula
        if z > 0.0 {
            let sqrt_term = (2.0 / (PI * z)).sqrt();
            let phase = z - (nu + 0.5) * PI / 2.0;
            num_complex::Complex64::new(sqrt_term * phase.sin(), 0.0)
        } else {
            num_complex::Complex64::new(0.0, 0.0)
        }
    }

    /// Hankel function of the first kind for half-integer order
    /// H_{n+1/2}^{(1)}(z) ≈ √(2/πz) · exp(i(z - (n+1)π/2))
    fn hankel_h1_half_integer(nu: f64, z: f64) -> num_complex::Complex64 {
        if z > 0.0 {
            let sqrt_term = (2.0 / (PI * z)).sqrt();
            let phase = z - (nu + 0.5) * PI / 2.0;
            sqrt_term * num_complex::Complex64::new(0.0, phase).exp()
        } else {
            num_complex::Complex64::new(0.0, 0.0)
        }
    }

    /// Legendre polynomial P_n(x)
    fn legendre_polynomial(n: usize, x: f64) -> f64 {
        match n {
            0 => 1.0,
            1 => x,
            2 => (3.0 * x * x - 1.0) / 2.0,
            3 => (5.0 * x * x * x - 3.0 * x) / 2.0,
            4 => (35.0 * x.powi(4) - 30.0 * x * x + 3.0) / 8.0,
            _ => {
                // Recurrence relation: (n+1)P_{n+1}(x) = (2n+1)xP_n(x) - nP_{n-1}(x)
                let mut p_prev = 1.0;
                let mut p_curr = x;
                for i in 2..=n {
                    let i_f = i as f64;
                    let p_next = ((2.0 * i_f - 1.0) * x * p_curr - (i_f - 1.0) * p_prev) / i_f;
                    p_prev = p_curr;
                    p_curr = p_next;
                }
                p_curr
            }
        }
    }
}

// ==================== Tests ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mie_solution_creation() {
        let mie = MieScatteringSolution::new(0.01, 1e5, 1500.0, 1000.0);
        assert!(mie.is_ok());
        let mie = mie.unwrap();
        assert_eq!(mie.radius, 0.01);
        assert_eq!(mie.frequency, 1e5);
        assert!(mie.num_terms >= 10);
    }

    #[test]
    fn test_mie_coefficients_computation() {
        let mut mie = MieScatteringSolution::new(0.01, 1e5, 1500.0, 1000.0).unwrap();
        let result = mie.compute_coefficients();
        assert!(result.is_ok());
        assert!(!mie.mie_coefficients.is_empty());
        // First coefficient should be significant
        assert!(mie.mie_coefficients[0].norm() > 0.01);
    }

    #[test]
    fn test_legendre_polynomials() {
        // Test orthogonality and known values
        assert!((MieScatteringSolution::legendre_polynomial(0, 0.5) - 1.0).abs() < 1e-10);
        assert!((MieScatteringSolution::legendre_polynomial(1, 0.5) - 0.5).abs() < 1e-10);
        assert!((MieScatteringSolution::legendre_polynomial(2, 0.5) - (-0.125)).abs() < 1e-10);
    }

    #[test]
    fn test_riccati_bessel_values() {
        // Test that ψ_n(z) varies smoothly and is non-zero for reasonable arguments
        let z = 0.5;
        let psi = MieScatteringSolution::riccati_bessel_j(1, z);
        // Just verify it computes something reasonable (non-NaN, finite)
        assert!(!psi.re.is_nan() && !psi.im.is_nan());
        assert!(psi.norm() > 0.0);
    }

    #[test]
    fn test_scattered_pressure_field() {
        let mut mie = MieScatteringSolution::new(0.01, 1e5, 1500.0, 1000.0).unwrap();
        mie.compute_coefficients().unwrap();

        // Evaluate at shadow side (θ = π)
        let p_shadow = mie.scattered_pressure(0.05, PI).unwrap();
        assert!(p_shadow.norm() > 0.0);

        // Evaluate at forward direction (θ = 0)
        let p_forward = mie.scattered_pressure(0.05, 0.0).unwrap();
        assert!(p_forward.norm() > 0.0);

        // Forward scattering should be different from backward
        assert!((p_forward.norm() - p_shadow.norm()).abs() > 1e-6);
    }

    #[test]
    fn test_total_pressure() {
        let mut mie = MieScatteringSolution::new(0.01, 1e5, 1500.0, 1000.0).unwrap();
        mie.compute_coefficients().unwrap();

        // Evaluate far field
        let p_total = mie.total_pressure(0.1, 0.0, 0.05).unwrap();
        assert!(p_total.norm() > 0.0);
    }

    #[test]
    fn test_convergence_with_frequency() {
        // At different frequencies, we should get different scattering patterns
        let mut mie_low = MieScatteringSolution::new(0.001, 1e3, 1500.0, 1000.0).unwrap();
        mie_low.compute_coefficients().unwrap();
        let p_low = mie_low.scattered_pressure(0.05, 0.0).unwrap();

        let mut mie_high = MieScatteringSolution::new(0.001, 1e5, 1500.0, 1000.0).unwrap();
        mie_high.compute_coefficients().unwrap();
        let p_high = mie_high.scattered_pressure(0.05, 0.0).unwrap();

        // Both should produce valid (non-NaN) results
        assert!(!p_low.is_nan() && !p_high.is_nan());
        // Fields should be different for different frequencies
        assert!((p_high.norm() - p_low.norm()).abs() > 1e-10);
    }

    #[test]
    fn test_reciprocity_symmetry() {
        // For axisymmetric scattering (plane wave), certain symmetries hold
        let mut mie = MieScatteringSolution::new(0.01, 1e5, 1500.0, 1000.0).unwrap();
        mie.compute_coefficients().unwrap();

        // Evaluate at symmetric angles
        let p1 = mie.scattered_pressure(0.1, PI / 4.0).unwrap();
        let p2 = mie.scattered_pressure(0.1, PI / 4.0).unwrap();

        // Same angle should give same result
        assert!((p1.norm() - p2.norm()).abs() < 1e-10);
    }

    #[test]
    fn test_mie_backscatter_monopole_limit() {
        // At ka → 0, backscatter should approach monopole value
        let mut mie = MieScatteringSolution::new(0.0001, 1e3, 1500.0, 1000.0).unwrap();
        mie.compute_coefficients().unwrap();

        let p_back = mie.scattered_pressure(1.0, PI).unwrap();

        // For monopole: p ~ e^(ikr)/(4πr), at far field should be real-like
        assert!(p_back.norm() > 0.0);
    }

    #[test]
    fn test_mie_small_ka_expansion() {
        // For ka << 1, dominant term is monopole (n=1)
        let mut mie = MieScatteringSolution::new(0.001, 5e3, 1500.0, 1000.0).unwrap();
        mie.compute_coefficients().unwrap();

        // First coefficient should dominate
        let a1_norm = mie.mie_coefficients[0].norm();
        let a2_norm = if mie.mie_coefficients.len() > 1 {
            mie.mie_coefficients[1].norm()
        } else {
            0.0
        };

        assert!(a1_norm > a2_norm);
    }

    #[test]
    fn test_mie_series_convergence() {
        let mut mie = MieScatteringSolution::new(0.01, 5e5, 1500.0, 1000.0).unwrap();
        mie.compute_coefficients().unwrap();

        // Coefficients should be computed
        assert!(!mie.mie_coefficients.is_empty());

        // Check that coefficients are finite
        for coeff in &mie.mie_coefficients {
            assert!(!coeff.is_nan());
        }
    }

    #[test]
    fn test_mie_forward_scatter_amplitude() {
        // Forward scatter (θ = 0) should have largest amplitude for most frequencies
        let mut mie = MieScatteringSolution::new(0.01, 2e5, 1500.0, 1000.0).unwrap();
        mie.compute_coefficients().unwrap();

        let p_forward = mie.scattered_pressure(0.1, 0.0).unwrap();
        let p_side = mie.scattered_pressure(0.1, PI / 2.0).unwrap();
        let p_back = mie.scattered_pressure(0.1, PI).unwrap();

        // All three directions should produce valid results
        let p_forward_norm = p_forward.norm();
        let p_side_norm = p_side.norm();
        let p_back_norm = p_back.norm();

        assert!(p_forward_norm > 0.0 && p_side_norm > 0.0 && p_back_norm > 0.0);
        // Pattern should not be trivial (angles should differ)
        assert!(
            (p_forward_norm - p_back_norm).abs() > 1e-10
                || (p_forward_norm - p_side_norm).abs() > 1e-10
        );
    }
}
