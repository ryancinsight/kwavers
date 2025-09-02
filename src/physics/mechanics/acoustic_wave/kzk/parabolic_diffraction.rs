//! KZK-specific diffraction operator
//!
//! Implements the parabolic approximation used in KZK equation,
//! NOT the full angular spectrum method
//!
//! References:
//! - Lee & Hamilton (1995) "Parametric array in air", Eq. 2.23
//! - Kamakura et al. (1992) "Nonlinear acoustic beam propagation"

use ndarray::{Array2, ArrayViewMut2};
use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::PI;

use super::{constants::*, KZKConfig};
use crate::constants::physics::SOUND_SPEED_WATER;

/// KZK parabolic diffraction operator
///
/// For KZK equation: ∂²p/∂z∂τ = (c₀/2)∇⊥²p
/// In frequency domain: ∂P/∂z = i(c₀/2ω)∇⊥²P
pub struct KzkDiffractionOperator {
    config: KZKConfig,
    kx2: Array2<f64>,
    ky2: Array2<f64>,
    fft_planner: FftPlanner<f64>,
}

impl KzkDiffractionOperator {
    pub fn new(config: &KZKConfig) -> Self {
        let nx = config.nx;
        let ny = config.ny;

        // Create k-space grids for transverse wavenumbers
        let mut kx2 = Array2::zeros((nx, ny));
        let mut ky2 = Array2::zeros((nx, ny));

        let dkx = 2.0 * PI / (nx as f64 * config.dx);
        let dky = 2.0 * PI / (ny as f64 * config.dx);

        // Standard FFT k-space ordering
        for i in 0..nx {
            let kx = if i <= nx / 2 {
                i as f64 * dkx
            } else {
                (i as f64 - nx as f64) * dkx
            };

            for j in 0..ny {
                kx2[[i, j]] = kx * kx;
            }
        }

        for j in 0..ny {
            let ky = if j <= ny / 2 {
                j as f64 * dky
            } else {
                (j as f64 - ny as f64) * dky
            };

            for i in 0..nx {
                ky2[[i, j]] = ky * ky;
            }
        }

        Self {
            config: config.clone(),
            kx2,
            ky2,
            fft_planner: FftPlanner::new(),
        }
    }

    /// Apply KZK diffraction step
    ///
    /// This implements the parabolic approximation:
    /// P(z+Δz) = P(z) * exp(-i(k_x² + k_y²)Δz/(2k₀))
    ///
    /// Note: This is different from angular spectrum which uses:
    /// P(z+Δz) = P(z) * exp(i*sqrt(k₀² - k_x² - k_y²)*Δz)
    pub fn apply(&mut self, field: &mut ArrayViewMut2<f64>, step_size: f64) {
        let nx = self.config.nx;
        let ny = self.config.ny;
        let k0 = 2.0 * PI * self.config.frequency / self.config.c0;

        // Convert to complex field
        let mut complex_field = Array2::zeros((nx, ny));
        for i in 0..nx {
            for j in 0..ny {
                complex_field[[i, j]] = Complex::new(field[[i, j]], 0.0);
            }
        }

        // 2D FFT
        complex_field = self.fft_2d_forward(complex_field);

        // Apply parabolic propagator in k-space
        // H(kx,ky) = exp(-i(kx² + ky²)Δz/(2k₀))
        for i in 0..nx {
            for j in 0..ny {
                let kt2 = self.kx2[[i, j]] + self.ky2[[i, j]];

                // Parabolic approximation phase
                let phase = -kt2 * step_size / (2.0 * k0);

                // Apply propagator
                complex_field[[i, j]] *= Complex::new(phase.cos(), phase.sin());
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

    /// Forward 2D FFT (proper implementation)
    fn fft_2d_forward(&mut self, mut data: Array2<Complex<f64>>) -> Array2<Complex<f64>> {
        let nx = data.shape()[0];
        let ny = data.shape()[1];

        // FFT along rows
        for j in 0..ny {
            let mut row_buffer: Vec<Complex<f64>> = (0..nx).map(|i| data[[i, j]]).collect();

            let fft = self.fft_planner.plan_fft_forward(nx);
            fft.process(&mut row_buffer);

            for i in 0..nx {
                data[[i, j]] = row_buffer[i];
            }
        }

        // FFT along columns
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

        // IFFT along columns
        for i in 0..nx {
            let mut col_buffer: Vec<Complex<f64>> = (0..ny).map(|j| data[[i, j]]).collect();

            let ifft = self.fft_planner.plan_fft_inverse(ny);
            ifft.process(&mut col_buffer);

            for j in 0..ny {
                data[[i, j]] = col_buffer[j];
            }
        }

        // IFFT along rows
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
    use crate::physics::validation::measure_beam_radius;
    use approx::assert_relative_eq;

    #[test]
    fn test_kzk_gaussian_beam() {
        // Test that KZK diffraction gives expected Gaussian beam spreading
        let config = KZKConfig {
            nx: DEFAULT_GRID_SIZE,
            ny: DEFAULT_GRID_SIZE,
            dx: DEFAULT_WAVELENGTH / 6.0, // 6 points per wavelength
            frequency: DEFAULT_FREQUENCY,
            c0: SOUND_SPEED_WATER,
            ..Default::default()
        };

        let mut op = KzkDiffractionOperator::new(&config);

        // Create Gaussian beam
        let beam_waist = DEFAULT_BEAM_WAIST;
        let mut field = Array2::zeros((config.nx, config.ny));

        for i in 0..config.nx {
            for j in 0..config.ny {
                let x = (i as f64 - config.nx as f64 / 2.0) * config.dx;
                let y = (j as f64 - config.ny as f64 / 2.0) * config.dx;
                let r2 = x * x + y * y;
                field[[i, j]] = (-r2 / (beam_waist * beam_waist)).exp();
            }
        }

        // Propagate to Rayleigh distance
        let wavelength = config.c0 / config.frequency;
        let z_r = PI * beam_waist * beam_waist / wavelength;
        let steps = 50;
        let dz = z_r / steps as f64;

        for _ in 0..steps {
            let mut field_view = field.view_mut();
            op.apply(&mut field_view, dz);
        }

        // Measure beam radius
        let intensity = &field * &field;
        let measured = measure_beam_radius(&intensity, config.dx);
        let expected = beam_waist * 2.0_f64.sqrt(); // At Rayleigh distance

        // Should match within 10% (parabolic approximation has some error)
        assert_relative_eq!(measured, expected, epsilon = 0.1 * expected);
    }
}
