//! Angular spectrum propagation method
//!
//! Implements the angular spectrum method for wave propagation,
//! based on Goodman (2005): "Introduction to Fourier Optics"

use crate::error::KwaversResult;
use ndarray::Array2;
use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::PI;

/// Angular spectrum propagation method for forward/backward propagation
pub struct AngularSpectrum {
    /// Grid dimensions
    nx: usize,
    ny: usize,
    /// Grid spacing
    dx: f64,
    dy: f64,
    /// Wavenumber arrays
    kx: Array2<f64>,
    ky: Array2<f64>,
    /// FFT planner
    fft_planner: FftPlanner<f64>,
}

impl AngularSpectrum {
    /// Create new angular spectrum propagator
    pub fn new(nx: usize, ny: usize, dx: f64, dy: f64) -> Self {
        // Create wavenumber arrays
        let kx_1d = Self::create_k_vector(nx, dx);
        let ky_1d = Self::create_k_vector(ny, dy);

        // Create 2D wavenumber grids
        let mut kx = Array2::zeros((nx, ny));
        let mut ky = Array2::zeros((nx, ny));

        for i in 0..nx {
            for j in 0..ny {
                kx[[i, j]] = kx_1d[i];
                ky[[i, j]] = ky_1d[j];
            }
        }

        Self {
            nx,
            ny,
            dx,
            dy,
            kx,
            ky,
            fft_planner: FftPlanner::new(),
        }
    }

    /// Create k-space vector for FFT
    fn create_k_vector(n: usize, d: f64) -> Vec<f64> {
        let mut k = vec![0.0; n];
        let dk = 2.0 * PI / (n as f64 * d);

        for i in 0..n {
            if i <= n / 2 {
                k[i] = i as f64 * dk;
            } else {
                k[i] = (i as f64 - n as f64) * dk;
            }
        }

        k
    }

    /// Forward propagation using angular spectrum method
    pub fn forward_propagate(
        &mut self,
        field: &Array2<f64>,
        distance: f64,
        wavelength: f64,
    ) -> KwaversResult<Array2<f64>> {
        self.propagate(field, distance, wavelength, true)
    }

    /// Backward propagation using angular spectrum method
    pub fn backward_propagate(
        &mut self,
        field: &Array2<f64>,
        distance: f64,
        wavelength: f64,
    ) -> KwaversResult<Array2<f64>> {
        self.propagate(field, distance, wavelength, false)
    }

    /// Core propagation function
    fn propagate(
        &mut self,
        field: &Array2<f64>,
        distance: f64,
        wavelength: f64,
        forward: bool,
    ) -> KwaversResult<Array2<f64>> {
        let k = 2.0 * PI / wavelength;
        let sign = if forward { 1.0 } else { -1.0 };

        // Convert to complex for FFT
        let mut complex_field: Vec<Complex<f64>> = field
            .as_slice()
            .unwrap()
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        // Forward FFT
        let fft = self.fft_planner.plan_fft_forward(self.nx * self.ny);
        fft.process(&mut complex_field);

        // Apply propagation in k-space
        for i in 0..self.nx {
            for j in 0..self.ny {
                let idx = i * self.ny + j;
                let kx = self.kx[[i, j]];
                let ky = self.ky[[i, j]];
                let kz_sq = k * k - kx * kx - ky * ky;

                if kz_sq > 0.0 {
                    // Propagating waves
                    let kz = kz_sq.sqrt();
                    let phase = Complex::from_polar(1.0, sign * kz * distance);
                    complex_field[idx] *= phase;
                } else {
                    // Evanescent waves - exponential decay
                    let kz_imag = (-kz_sq).sqrt();
                    let decay = (-kz_imag * distance).exp();
                    complex_field[idx] *= decay;
                }
            }
        }

        // Inverse FFT
        let ifft = self.fft_planner.plan_fft_inverse(self.nx * self.ny);
        ifft.process(&mut complex_field);

        // Extract real part and normalize
        let mut result = Array2::zeros((self.nx, self.ny));
        for i in 0..self.nx {
            for j in 0..self.ny {
                let idx = i * self.ny + j;
                result[[i, j]] = complex_field[idx].re / (self.nx * self.ny) as f64;
            }
        }

        Ok(result)
    }
}