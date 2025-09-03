//! Proper 2D angular spectrum implementation
//!
//! Uses correct 2D FFT instead of 1D FFT on flattened array

use ndarray::{Array2, ArrayViewMut2};
use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::PI;

use super::KZKConfig;

/// Correct 2D angular spectrum operator
pub struct AngularSpectrum2D {
    config: KZKConfig,
    kx: Array2<f64>,
    ky: Array2<f64>,
    fft_planner: FftPlanner<f64>,
}

impl AngularSpectrum2D {
    #[must_use]
    pub fn new(config: &KZKConfig) -> Self {
        let nx = config.nx;
        let ny = config.ny;

        // Create k-space grids
        let mut kx = Array2::zeros((nx, ny));
        let mut ky = Array2::zeros((nx, ny));

        let dkx = 2.0 * PI / (nx as f64 * config.dx);
        let dky = 2.0 * PI / (ny as f64 * config.dx);

        for i in 0..nx {
            let kx_val = if i <= nx / 2 {
                i as f64 * dkx
            } else {
                (i as f64 - nx as f64) * dkx
            };

            for j in 0..ny {
                kx[[i, j]] = kx_val;
            }
        }

        for j in 0..ny {
            let ky_val = if j <= ny / 2 {
                j as f64 * dky
            } else {
                (j as f64 - ny as f64) * dky
            };

            for i in 0..nx {
                ky[[i, j]] = ky_val;
            }
        }

        Self {
            config: config.clone(),
            kx,
            ky,
            fft_planner: FftPlanner::new(),
        }
    }

    /// Apply angular spectrum propagation
    pub fn propagate(&mut self, field: &mut ArrayViewMut2<f64>, distance: f64) {
        let nx = self.config.nx;
        let ny = self.config.ny;
        let k0 = 2.0 * PI * self.config.frequency / self.config.c0;

        // Convert to complex
        let mut complex_field = Array2::zeros((nx, ny));
        for i in 0..nx {
            for j in 0..ny {
                complex_field[[i, j]] = Complex::new(field[[i, j]], 0.0);
            }
        }

        // 2D FFT: transform rows then columns
        complex_field = self.fft_2d_forward(complex_field);

        // Apply transfer function
        for i in 0..nx {
            for j in 0..ny {
                let kx = self.kx[[i, j]];
                let ky = self.ky[[i, j]];
                let kt2 = kx * kx + ky * ky;

                if kt2 < k0 * k0 {
                    // Propagating waves
                    let kz = (k0 * k0 - kt2).sqrt();
                    let phase = kz * distance;
                    complex_field[[i, j]] *= Complex::new(phase.cos(), phase.sin());
                } else {
                    // Evanescent waves - exponential decay
                    let alpha = (kt2 - k0 * k0).sqrt();
                    complex_field[[i, j]] *= (-alpha * distance).exp();
                }
            }
        }

        // Inverse 2D FFT
        complex_field = self.fft_2d_inverse(complex_field);

        // Extract real part
        for i in 0..nx {
            for j in 0..ny {
                field[[i, j]] = complex_field[[i, j]].re;
            }
        }
    }

    /// Forward 2D FFT
    fn fft_2d_forward(&mut self, mut data: Array2<Complex<f64>>) -> Array2<Complex<f64>> {
        let nx = data.shape()[0];
        let ny = data.shape()[1];

        // FFT along rows (axis 0)
        for j in 0..ny {
            let mut row_buffer: Vec<Complex<f64>> = (0..nx).map(|i| data[[i, j]]).collect();

            let fft = self.fft_planner.plan_fft_forward(nx);
            fft.process(&mut row_buffer);

            for i in 0..nx {
                data[[i, j]] = row_buffer[i];
            }
        }

        // FFT along columns (axis 1)
        for i in 0..nx {
            let mut col_buffer: Vec<Complex<f64>> = (0..ny).map(|j| data[[i, j]]).collect();

            let fft = self.fft_planner.plan_fft_forward(ny);
            fft.process(&mut col_buffer);

            for j in 0..ny {
                data[[i, j]] = col_buffer[j];
            }
        }

        data
    }

    /// Inverse 2D FFT
    fn fft_2d_inverse(&mut self, mut data: Array2<Complex<f64>>) -> Array2<Complex<f64>> {
        let nx = data.shape()[0];
        let ny = data.shape()[1];

        // IFFT along columns (axis 1)
        for i in 0..nx {
            let mut col_buffer: Vec<Complex<f64>> = (0..ny).map(|j| data[[i, j]]).collect();

            let ifft = self.fft_planner.plan_fft_inverse(ny);
            ifft.process(&mut col_buffer);

            for j in 0..ny {
                data[[i, j]] = col_buffer[j];
            }
        }

        // IFFT along rows (axis 0)
        for j in 0..ny {
            let mut row_buffer: Vec<Complex<f64>> = (0..nx).map(|i| data[[i, j]]).collect();

            let ifft = self.fft_planner.plan_fft_inverse(nx);
            ifft.process(&mut row_buffer);

            for i in 0..nx {
                data[[i, j]] = row_buffer[i];
            }
        }

        // Normalize
        let norm = 1.0 / (nx * ny) as f64;
        data.mapv(|c| c * norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_2d_angular_spectrum() {
        let config = KZKConfig {
            nx: 128,
            ny: 128,
            dx: 0.2e-3,
            frequency: 1e6,
            c0: 1500.0,
            ..Default::default()
        };

        let mut op = AngularSpectrum2D::new(&config);

        // Create Gaussian beam
        let beam_waist = 5e-3;
        let mut field = Array2::zeros((128, 128));

        for i in 0..128 {
            for j in 0..128 {
                let x = (i as f64 - 64.0) * config.dx;
                let y = (j as f64 - 64.0) * config.dx;
                let r2 = x * x + y * y;
                field[[i, j]] = (-r2 / (beam_waist * beam_waist)).exp();
            }
        }

        // Propagate small distance
        let mut field_view = field.view_mut();
        op.propagate(&mut field_view, 10e-3);

        // Should still have reasonable values (allow for phase shift)
        let center = field[[64, 64]];
        let center_magnitude = center.abs();
        assert!(
            center_magnitude > 0.5,
            "Center magnitude too low: {}",
            center_magnitude
        );
        assert!(
            center_magnitude <= 1.1,
            "Center magnitude too high: {}",
            center_magnitude
        );
    }
}
