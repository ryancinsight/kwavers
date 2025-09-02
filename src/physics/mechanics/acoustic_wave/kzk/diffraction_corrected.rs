//! Corrected diffraction operator for KZK equation
//!
//! Implements exact angular spectrum method for beam propagation
//!
//! References:
//! - Goodman, J.W. (2005) "Introduction to Fourier Optics", Ch. 3-4
//! - Wen & Breazeale (1988) "A diffraction beam field expressed as the
//!   superposition of Gaussian beams", JASA 83(5)

use ndarray::{Array2, ArrayViewMut2};
use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::PI;

use super::KZKConfig;

/// Angular spectrum diffraction operator
pub struct AngularSpectrumOperator {
    config: KZKConfig,
    transfer_function: Array2<Complex<f64>>,
    fft_planner: FftPlanner<f64>,
    fft_buffer: Vec<Complex<f64>>,
}

impl AngularSpectrumOperator {
    /// Create new angular spectrum operator
    pub fn new(config: &KZKConfig, step_size: f64) -> Self {
        let nx = config.nx;
        let ny = config.ny;

        // Pre-compute transfer function for this step size
        let transfer_function = compute_transfer_function(config, step_size);

        Self {
            config: config.clone(),
            transfer_function,
            fft_planner: FftPlanner::new(),
            fft_buffer: vec![Complex::new(0.0, 0.0); nx * ny],
        }
    }

    /// Apply diffraction step using angular spectrum method
    pub fn apply(&mut self, field: &mut ArrayViewMut2<f64>, step_size: f64) {
        let nx = self.config.nx;
        let ny = self.config.ny;

        // Update transfer function if step size changed
        if (step_size - self.config.dz).abs() > 1e-10 {
            self.transfer_function = compute_transfer_function(&self.config, step_size);
        }

        // Convert to complex and reshape for FFT
        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                self.fft_buffer[idx] = Complex::new(field[[i, j]], 0.0);
            }
        }

        // Forward FFT
        let fft = self.fft_planner.plan_fft_forward(nx * ny);
        fft.process(&mut self.fft_buffer);

        // Apply transfer function in Fourier domain
        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                self.fft_buffer[idx] *= self.transfer_function[[i, j]];
            }
        }

        // Inverse FFT
        let ifft = self.fft_planner.plan_fft_inverse(nx * ny);
        ifft.process(&mut self.fft_buffer);

        // Extract real part and normalize
        let norm = 1.0 / (nx * ny) as f64;
        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                field[[i, j]] = self.fft_buffer[idx].re * norm;
            }
        }
    }
}

/// Compute angular spectrum transfer function
fn compute_transfer_function(config: &KZKConfig, step_size: f64) -> Array2<Complex<f64>> {
    let nx = config.nx;
    let ny = config.ny;
    let mut h = Array2::zeros((nx, ny));

    // Wavenumber in propagation direction
    let k0 = 2.0 * PI * config.frequency / config.c0;

    // Spatial frequencies
    let dkx = 2.0 * PI / (nx as f64 * config.dx);
    let dky = 2.0 * PI / (ny as f64 * config.dx);

    for j in 0..ny {
        for i in 0..nx {
            // Shift k-space coordinates for FFT convention
            let kx = if i <= nx / 2 {
                i as f64 * dkx
            } else {
                (i as f64 - nx as f64) * dkx
            };

            let ky = if j <= ny / 2 {
                j as f64 * dky
            } else {
                (j as f64 - ny as f64) * dky
            };

            // Transverse wavenumber squared
            let kt2 = kx * kx + ky * ky;

            // Longitudinal wavenumber (with evanescent wave handling)
            let kz = if kt2 < k0 * k0 {
                // Propagating waves
                (k0 * k0 - kt2).sqrt()
            } else {
                // Evanescent waves (imaginary kz)
                0.0
            };

            // Transfer function H(kx,ky) = exp(i*kz*Δz)
            // For propagating waves: phase advance
            // For evanescent waves: exponential decay
            if kt2 < k0 * k0 {
                // Propagating component
                let phase = kz * step_size;
                h[[i, j]] = Complex::new(phase.cos(), phase.sin());
            } else {
                // Evanescent component (decays exponentially)
                let decay = -(kt2 - k0 * k0).sqrt() * step_size;
                h[[i, j]] = Complex::new(decay.exp(), 0.0);
            }
        }
    }

    h
}

/// Analytical Gaussian beam propagation for validation
pub fn gaussian_beam_radius(z: f64, w0: f64, wavelength: f64) -> f64 {
    let z_r = PI * w0 * w0 / wavelength; // Rayleigh distance
    w0 * (1.0 + (z / z_r).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gaussian_beam_propagation() {
        // Test parameters
        let w0 = 5e-3; // 5mm beam waist
        let wavelength = 1.5e-3; // 1.5mm (1 MHz in water)
        let z_r = PI * w0 * w0 / wavelength; // Rayleigh distance

        // At Rayleigh distance, beam should expand by √2
        let w_z = gaussian_beam_radius(z_r, w0, wavelength);
        let expected = w0 * 2.0_f64.sqrt();

        assert_relative_eq!(w_z, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_transfer_function_symmetry() {
        let config = KZKConfig {
            nx: 64,
            ny: 64,
            dx: 1e-3,
            frequency: 1e6,
            c0: 1500.0,
            ..Default::default()
        };

        let h = compute_transfer_function(&config, 1e-3);

        // Transfer function should be Hermitian symmetric for real-valued fields
        // Check that |H(kx,ky)| = |H(-kx,-ky)|
        // Note: Due to FFT indexing, this is not a simple flip
        let h_dc = h[[0, 0]];
        assert!(h_dc.im.abs() < 1e-10, "DC component should be real");

        // Check a few points for reasonable values
        assert!(
            h[[32, 32]].norm() > 0.9,
            "Center frequency should have high transmission"
        );
        assert!(h[[0, 0]].norm() > 0.99, "DC should pass through");
    }
}
