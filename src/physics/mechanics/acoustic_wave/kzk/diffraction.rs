//! Diffraction operator for KZK equation
//!
//! Implements the angular spectrum method for beam diffraction.
//! Reference: Vecchio & Lewin (1994) "Finite amplitude acoustic propagation"

use ndarray::{Array2, ArrayViewMut2};
use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::PI;

use super::KZKConfig;

/// Diffraction operator using angular spectrum method
pub struct DiffractionOperator {
    /// Wavenumber squared in x direction
    kx2: Array2<f64>,
    /// Wavenumber squared in y direction
    ky2: Array2<f64>,
    /// FFT planner
    fft_planner: FftPlanner<f64>,
    /// Configuration
    config: KZKConfig,
}

impl DiffractionOperator {
    /// Create new diffraction operator
    pub fn new(config: &KZKConfig) -> Self {
        let mut kx2 = Array2::zeros((config.nx, config.ny));
        let mut ky2 = Array2::zeros((config.nx, config.ny));

        // Compute wavenumber arrays
        let dkx = 2.0 * PI / (config.nx as f64 * config.dx);
        let dky = 2.0 * PI / (config.ny as f64 * config.dx);

        for j in 0..config.ny {
            for i in 0..config.nx {
                let kx = if i <= config.nx / 2 {
                    i as f64 * dkx
                } else {
                    (i as f64 - config.nx as f64) * dkx
                };

                let ky = if j <= config.ny / 2 {
                    j as f64 * dky
                } else {
                    (j as f64 - config.ny as f64) * dky
                };

                kx2[[i, j]] = kx * kx;
                ky2[[i, j]] = ky * ky;
            }
        }

        Self {
            kx2,
            ky2,
            fft_planner: FftPlanner::new(),
            config: config.clone(),
        }
    }

    /// Apply diffraction for one step
    /// Solves: ∂p/∂z = (ic₀/2ω)∇⊥²p in frequency domain
    pub fn apply(&mut self, slice: &mut ArrayViewMut2<f64>, step_size: f64) {
        let nx = self.config.nx;
        let ny = self.config.ny;

        // Convert to complex for FFT
        let mut complex_field: Vec<Complex<f64>> = Vec::with_capacity(nx * ny);
        for j in 0..ny {
            for i in 0..nx {
                complex_field.push(Complex::new(slice[[i, j]], 0.0));
            }
        }

        // Forward FFT
        let fft = self.fft_planner.plan_fft_forward(nx * ny);
        fft.process(&mut complex_field);

        // Apply diffraction propagator in frequency domain
        // H(kx,ky) = exp(i * step_size * c₀ * (kx² + ky²) / (2k₀))
        let k0 = 2.0 * PI * self.config.frequency / self.config.c0;
        let factor = step_size * self.config.c0 / (2.0 * k0);

        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                let k_perp2 = self.kx2[[i, j]] + self.ky2[[i, j]];
                let phase = factor * k_perp2;

                // Multiply by propagator
                complex_field[idx] *= Complex::new(phase.cos(), phase.sin());
            }
        }

        // Inverse FFT
        let ifft = self.fft_planner.plan_fft_inverse(nx * ny);
        ifft.process(&mut complex_field);

        // Copy back to real array (normalized)
        let norm = 1.0 / (nx * ny) as f64;
        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                slice[[i, j]] = complex_field[idx].re * norm;
            }
        }
    }
}
