//! Complex-valued parabolic diffraction operator for proper energy conservation

use crate::math::fft::{fft_2d_complex, ifft_2d_complex, Complex64};
use ndarray::{Array2, ArrayViewMut2};
use std::f64::consts::PI;

use super::KZKConfig;

/// Parabolic diffraction operator using complex-valued computations for energy preservation
pub struct ParabolicDiffractionOperator {
    config: KZKConfig,
    kx2: Array2<f64>,
    ky2: Array2<f64>,
}

impl std::fmt::Debug for ParabolicDiffractionOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParabolicDiffractionOperator")
            .field("config", &self.config)
            .field("kx2_shape", &self.kx2.shape())
            .field("ky2_shape", &self.ky2.shape())
            .finish()
    }
}

impl ParabolicDiffractionOperator {
    /// Create new complex parabolic diffraction operator
    #[must_use]
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
        }
    }

    /// Apply diffraction step to complex field
    pub fn apply_complex(&mut self, field: &mut ArrayViewMut2<Complex64>, step_size: f64) {
        let nx = self.config.nx;
        let ny = self.config.ny;
        let k0 = 2.0 * PI * self.config.frequency / self.config.c0;

        // Clone field for FFT
        let complex_field_owned = field.to_owned();

        // 2D FFT
        let mut complex_field = fft_2d_complex(&complex_field_owned);

        // Apply parabolic propagator in k-space
        // H(kx,ky) = exp(i(kx² + ky²)Δz/(2k₀))
        for i in 0..nx {
            for j in 0..ny {
                let kt2 = self.kx2[[i, j]] + self.ky2[[i, j]];

                // Parabolic approximation phase
                let phase = kt2 * step_size / (2.0 * k0);

                // Apply propagator H = exp(i*phase)
                complex_field[[i, j]] *= Complex64::from_polar(1.0, phase);
            }
        }

        // Inverse 2D FFT
        let inverted = ifft_2d_complex(&complex_field);

        // Copy back to field
        field.assign(&inverted);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::forward::nonlinear::kzk::constants::{
        DEFAULT_BEAM_WAIST, DEFAULT_FREQUENCY, DEFAULT_GRID_SIZE, DEFAULT_WAVELENGTH,
    };
    use crate::solver::validation::measure_beam_radius;
    use approx::assert_relative_eq;

    #[test]
    fn test_complex_energy_conservation() {
        let config = KZKConfig {
            nx: 64,
            ny: 64,
            dx: DEFAULT_WAVELENGTH / 8.0,
            frequency: DEFAULT_FREQUENCY,
            c0: 1500.0,
            ..Default::default()
        };

        let mut op = ParabolicDiffractionOperator::new(&config);

        // Create complex Gaussian beam
        let beam_waist = DEFAULT_BEAM_WAIST;
        let mut field = Array2::<Complex64>::zeros((config.nx, config.ny));

        for i in 0..config.nx {
            for j in 0..config.ny {
                let x = (i as f64 - config.nx as f64 / 2.0) * config.dx;
                let y = (j as f64 - config.ny as f64 / 2.0) * config.dx;
                let r2 = x * x + y * y;
                field[[i, j]] = Complex64::new((-r2 / (beam_waist * beam_waist)).exp(), 0.0);
            }
        }

        let initial_energy: f64 = field.iter().map(|c| c.norm_sqr()).sum();

        // Propagate
        let dz = DEFAULT_WAVELENGTH;
        let mut field_view = field.view_mut();
        op.apply_complex(&mut field_view, dz);

        let final_energy: f64 = field.iter().map(|c| c.norm_sqr()).sum();

        println!("Complex field energy conservation:");
        println!("Initial energy: {:.6}", initial_energy);
        println!("Final energy: {:.6}", final_energy);
        println!("Energy ratio: {:.6}", final_energy / initial_energy);

        // Energy should be conserved
        assert!((final_energy / initial_energy - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_complex_gaussian_beam_propagation() {
        let config = KZKConfig {
            nx: DEFAULT_GRID_SIZE,
            ny: DEFAULT_GRID_SIZE,
            dx: DEFAULT_WAVELENGTH / 6.0,
            frequency: DEFAULT_FREQUENCY,
            c0: 1500.0,
            ..Default::default()
        };

        let mut op = ParabolicDiffractionOperator::new(&config);

        // Create complex Gaussian beam
        let beam_waist = DEFAULT_BEAM_WAIST;
        let mut field = Array2::<Complex64>::zeros((config.nx, config.ny));

        for i in 0..config.nx {
            for j in 0..config.ny {
                let x = (i as f64 - config.nx as f64 / 2.0) * config.dx;
                let y = (j as f64 - config.ny as f64 / 2.0) * config.dx;
                let r2 = x * x + y * y;
                field[[i, j]] = Complex64::new((-r2 / (beam_waist * beam_waist)).exp(), 0.0);
            }
        }

        // Propagate to Rayleigh distance
        let wavelength = config.c0 / config.frequency;
        let z_r = PI * beam_waist * beam_waist / wavelength;
        let steps = 50;
        let dz = z_r / steps as f64;

        for _ in 0..steps {
            let mut field_view = field.view_mut();
            op.apply_complex(&mut field_view, dz);
        }

        // Measure beam radius from intensity
        let intensity = field.mapv(|c| c.norm_sqr());
        let measured = measure_beam_radius(&intensity, config.dx);
        let expected = beam_waist * 2.0_f64.sqrt();

        println!("Complex field Gaussian beam test:");
        println!("Measured radius: {:.3}mm", measured * 1000.0);
        println!("Expected radius: {:.3}mm", expected * 1000.0);
        println!(
            "Error: {:.1}%",
            (measured - expected).abs() / expected * 100.0
        );

        // Should match within 5% with proper complex field handling
        assert_relative_eq!(measured, expected, epsilon = 0.05 * expected);
    }
}
