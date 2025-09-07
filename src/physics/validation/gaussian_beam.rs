//! Gaussian beam propagation validation
//!
//! Tests fundamental beam physics against analytical solutions
//!
//! References:
//! - Siegman, A.E. (1986) "Lasers", University Science Books
//! - Saleh & Teich (2007) "Fundamentals of Photonics", Ch. 3

use ndarray::Array2;
use std::f64::consts::PI;

/// Analytical Gaussian beam parameters
#[derive(Debug, Clone)]
pub struct GaussianBeamParameters {
    /// Beam waist radius (minimum radius)
    pub w0: f64,
    /// Wavelength
    pub lambda: f64,
    /// Rayleigh distance `z_R` = π*w0²/λ
    pub z_r: f64,
    /// Wave number k = 2π/λ
    pub k: f64,
}

impl GaussianBeamParameters {
    #[must_use]
    pub fn new(w0: f64, lambda: f64) -> Self {
        let z_r = PI * w0 * w0 / lambda;
        let k = 2.0 * PI / lambda;

        Self { w0, lambda, z_r, k }
    }

    /// Beam radius at distance z: w(z) = w0 * sqrt(1 + (`z/z_R)²`)
    #[must_use]
    pub fn beam_radius(&self, z: f64) -> f64 {
        self.w0 * (1.0 + (z / self.z_r).powi(2)).sqrt()
    }

    /// Radius of curvature: R(z) = z * (1 + (`z_R/z)²`)
    #[must_use]
    pub fn radius_of_curvature(&self, z: f64) -> f64 {
        if z.abs() < 1e-10 {
            f64::INFINITY
        } else {
            z * (1.0 + (self.z_r / z).powi(2))
        }
    }

    /// Gouy phase: ψ(z) = `arctan(z/z_R)`
    #[must_use]
    pub fn gouy_phase(&self, z: f64) -> f64 {
        (z / self.z_r).atan()
    }

    /// Field amplitude at (r, z) in paraxial approximation
    #[must_use]
    pub fn field_amplitude(&self, r: f64, z: f64) -> f64 {
        let w_z = self.beam_radius(z);
        let _r_c = self.radius_of_curvature(z);
        let _psi = self.gouy_phase(z);

        // Amplitude factor
        let amp = (self.w0 / w_z).sqrt();

        // Gaussian envelope
        let gauss = (-r * r / (w_z * w_z)).exp();

        // Phase factor (ignoring for amplitude)
        amp * gauss
    }

    /// Intensity at (r, z): I(r,z) = |E(r,z)|²
    #[must_use]
    pub fn intensity(&self, r: f64, z: f64) -> f64 {
        let e = self.field_amplitude(r, z);
        e * e
    }

    /// Generate 2D field profile at distance z
    #[must_use]
    pub fn generate_profile(&self, nx: usize, ny: usize, dx: f64, z: f64) -> Array2<f64> {
        let mut profile = Array2::zeros((nx, ny));

        let cx = nx as f64 / 2.0;
        let cy = ny as f64 / 2.0;

        for i in 0..nx {
            for j in 0..ny {
                let x = (i as f64 - cx) * dx;
                let y = (j as f64 - cy) * dx;
                let r = (x * x + y * y).sqrt();

                profile[[i, j]] = self.field_amplitude(r, z);
            }
        }

        profile
    }
}

/// Measure beam radius from intensity profile using 1/e² criterion
#[must_use]
pub fn measure_beam_radius(intensity: &Array2<f64>, dx: f64) -> f64 {
    let (nx, ny) = intensity.dim();
    let cx = nx / 2;
    let cy = ny / 2;

    // Get peak intensity
    let peak = intensity[[cx, cy]];
    let threshold = peak / (std::f64::consts::E * std::f64::consts::E);

    // Find radius along x-axis
    let mut radius_pixels = 0.0;
    for i in cx..nx {
        if intensity[[i, cy]] < threshold {
            // Linear interpolation
            if i > cx {
                let prev = intensity[[i - 1, cy]];
                let curr = intensity[[i, cy]];
                let frac = (threshold - curr) / (prev - curr);
                radius_pixels = (i - cx) as f64 - frac;
            }
            break;
        }
    }

    radius_pixels * dx
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_rayleigh_distance() {
        // Test case: 5mm waist, 1.5mm wavelength (1 MHz in water)
        let params = GaussianBeamParameters::new(5e-3, 1.5e-3);

        // z_R = π*w0²/λ = π*(5mm)²/1.5mm ≈ 52.36mm
        assert_relative_eq!(params.z_r, 52.36e-3, epsilon = 1e-3);
    }

    #[test]
    fn test_beam_expansion() {
        let params = GaussianBeamParameters::new(5e-3, 1.5e-3);

        // At z = 0: w(0) = w0
        assert_relative_eq!(params.beam_radius(0.0), 5e-3);

        // At z = z_R: w(z_R) = w0 * sqrt(2)
        assert_relative_eq!(
            params.beam_radius(params.z_r),
            5e-3 * 2.0_f64.sqrt(),
            epsilon = 1e-10
        );

        // At z = 2*z_R: w(2*z_R) = w0 * sqrt(5)
        assert_relative_eq!(
            params.beam_radius(2.0 * params.z_r),
            5e-3 * 5.0_f64.sqrt(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_intensity_conservation() {
        let params = GaussianBeamParameters::new(5e-3, 1.5e-3);

        // Total power should be conserved
        let dx = 0.1e-3;
        let n = 256;

        let p0 = compute_total_power(&params, 0.0, n, dx);
        let p1 = compute_total_power(&params, params.z_r, n, dx);
        let p2 = compute_total_power(&params, 2.0 * params.z_r, n, dx);

        // Power should be conserved (within numerical error)
        assert_relative_eq!(p0, p1, epsilon = 0.01);
        assert_relative_eq!(p0, p2, epsilon = 0.01);
    }

    fn compute_total_power(params: &GaussianBeamParameters, z: f64, n: usize, dx: f64) -> f64 {
        let mut power = 0.0;

        for i in 0..n {
            for j in 0..n {
                let x = (i as f64 - n as f64 / 2.0) * dx;
                let y = (j as f64 - n as f64 / 2.0) * dx;
                let r = (x * x + y * y).sqrt();

                power += params.intensity(r, z) * dx * dx;
            }
        }

        power
    }

    #[test]
    fn test_beam_measurement() {
        let params = GaussianBeamParameters::new(5e-3, 1.5e-3);

        // Generate profile at z = 0
        let profile = params.generate_profile(256, 256, 0.1e-3, 0.0);
        let intensity = &profile * &profile;

        // Measure radius
        let measured = measure_beam_radius(&intensity, 0.1e-3);

        // Should match w0 within grid resolution
        assert_relative_eq!(measured, 5e-3, epsilon = 0.2e-3);
    }
}
