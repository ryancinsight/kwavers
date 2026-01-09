use crate::math::fft::{fft_2d_complex, ifft_2d_complex, Complex64};
use ndarray::{Array2, ArrayViewMut2};
use std::f64::consts::PI;

use super::KZKConfig;

/// Correct 2D angular spectrum operator
pub struct AngularSpectrum2D {
    config: KZKConfig,
    kx: Array2<f64>,
    ky: Array2<f64>,
}

impl std::fmt::Debug for AngularSpectrum2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AngularSpectrum2D")
            .field("config", &self.config)
            .field("kx", &self.kx)
            .field("ky", &self.ky)
            .finish()
    }
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
                (i as i32 - nx as i32) as f64 * dkx
            };

            for j in 0..ny {
                kx[[i, j]] = kx_val;
            }
        }

        for j in 0..ny {
            let ky_val = if j <= ny / 2 {
                j as f64 * dky
            } else {
                (j as i32 - ny as i32) as f64 * dky
            };

            for i in 0..nx {
                ky[[i, j]] = ky_val;
            }
        }

        Self {
            config: config.clone(),
            kx,
            ky,
        }
    }

    /// Apply angular spectrum propagation
    pub fn propagate(&mut self, field: &mut ArrayViewMut2<f64>, distance: f64) {
        let nx = self.config.nx;
        let ny = self.config.ny;
        let k0 = 2.0 * PI * self.config.frequency / self.config.c0;

        // Convert to complex
        let complex_field = field.mapv(|x| Complex64::new(x, 0.0));

        // 2D FFT using central cache
        let mut complex_field_fft = fft_2d_complex(&complex_field);

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
                    complex_field_fft[[i, j]] *= Complex64::from_polar(1.0, phase);
                } else {
                    // Evanescent waves - exponential decay
                    let alpha = (kt2 - k0 * k0).sqrt();
                    complex_field_fft[[i, j]] *= (-alpha * distance).exp();
                }
            }
        }

        // Inverse 2D FFT: result is normalized
        let recovered = ifft_2d_complex(&complex_field_fft);

        // Extract real part
        for i in 0..nx {
            for j in 0..ny {
                field[[i, j]] = recovered[[i, j]].re;
            }
        }
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
