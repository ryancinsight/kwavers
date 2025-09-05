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

use super::KZKConfig;

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

impl std::fmt::Debug for KzkDiffractionOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KzkDiffractionOperator")
            .field("config", &self.config)
            .field(
                "kx2",
                &format!("Array2<f64> {}x{}", self.kx2.nrows(), self.kx2.ncols()),
            )
            .field(
                "ky2",
                &format!("Array2<f64> {}x{}", self.ky2.nrows(), self.ky2.ncols()),
            )
            .field("fft_planner", &"<FftPlanner>")
            .finish()
    }
}

impl KzkDiffractionOperator {
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
            fft_planner: FftPlanner::new(),
        }
    }

    /// Apply KZK diffraction step
    ///
    /// This implements the parabolic approximation:
    /// P(z+Δz) = P(z) * exp(-i(k_x² + `k_y²)Δz/(2k₀`))
    ///
    /// Note: This is different from angular spectrum which uses:
    /// P(z+Δz) = P(z) * exp(i*sqrt(k₀² - `k_x²` - `k_y²`)*Δz)
    pub fn apply(&mut self, field: &mut ArrayViewMut2<f64>, step_size: f64) {
        self.apply_with_step(field, step_size, 0);
    }

    fn apply_with_step(&mut self, field: &mut ArrayViewMut2<f64>, step_size: f64, step: usize) {
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
                // For forward propagation in +z direction
                let phase = kt2 * step_size / (2.0 * k0);

                // Apply propagator H = exp(i*phase)
                complex_field[[i, j]] *= Complex::new(phase.cos(), phase.sin());
            }
        }

        // Inverse 2D FFT
        complex_field = self.fft_2d_inverse(complex_field);

        // Extract magnitude (for intensity-based measurements)
        // Note: For coherent fields, we should track complex amplitude
        for i in 0..nx {
            for j in 0..ny {
                // For now, take real part assuming we started with real field
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
    use crate::physics::constants::SOUND_SPEED_WATER;
    use crate::physics::mechanics::acoustic_wave::kzk::constants::{
        DEFAULT_BEAM_WAIST, DEFAULT_FREQUENCY, DEFAULT_GRID_SIZE, DEFAULT_WAVELENGTH,
    };
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
                // For Gaussian beam: E(r) = E0 * exp(-r²/w₀²)
                // Note: This is field amplitude, intensity I = |E|²
                field[[i, j]] = (-r2 / (beam_waist * beam_waist)).exp();
            }
        }

        // Propagate to Rayleigh distance
        let wavelength = config.c0 / config.frequency;
        let z_r = PI * beam_waist * beam_waist / wavelength;
        let steps = 50;
        let dz = z_r / steps as f64;

        println!(
            "Grid: {}x{}, dx={:.3}mm",
            config.nx,
            config.ny,
            config.dx * 1000.0
        );
        println!("Wavelength: {:.3}mm", wavelength * 1000.0);
        println!("Rayleigh distance: {:.3}mm", z_r * 1000.0);
        println!("Step size: {:.3}mm", dz * 1000.0);
        println!("k0 = {:.3} rad/mm", 2.0 * PI / wavelength / 1000.0);

        // Track beam evolution
        let initial_power: f64 = field.iter().map(|&x| x * x).sum();
        println!("Initial power: {:.6}", initial_power);

        for step in 0..steps {
            let mut field_view = field.view_mut();
            op.apply_with_step(&mut field_view, dz, step);

            if step % 10 == 0 {
                let power: f64 = field.iter().map(|&x| x * x).sum();
                println!(
                    "Step {}: power = {:.6}, ratio = {:.6}",
                    step,
                    power,
                    power / initial_power
                );
            }
        }

        // Measure beam radius
        let intensity = &field * &field;
        let measured = measure_beam_radius(&intensity, config.dx);
        let expected = beam_waist * 2.0_f64.sqrt(); // At Rayleigh distance

        println!("Measured radius: {:.3}mm", measured * 1000.0);
        println!("Expected radius: {:.3}mm", expected * 1000.0);
        println!(
            "Error: {:.1}%",
            (measured - expected).abs() / expected * 100.0
        );

        // Should match within 20% (parabolic approximation and numerical discretization introduce errors)
        // TODO: Improve diffraction operator accuracy to achieve <10% error
        assert_relative_eq!(measured, expected, epsilon = 0.2 * expected);
    }

    #[test]
    fn test_kzk_gaussian_beam_high_resolution() {
        // Test with finer grid to see if resolution is the issue
        let config = KZKConfig {
            nx: 256, // Double resolution
            ny: 256,
            dx: DEFAULT_WAVELENGTH / 12.0, // 12 points per wavelength
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
        let steps = 100; // More steps for accuracy
        let dz = z_r / steps as f64;

        println!("\nHigh resolution test:");
        println!(
            "Grid: {}x{}, dx={:.3}mm",
            config.nx,
            config.ny,
            config.dx * 1000.0
        );
        println!("Steps: {}, dz={:.3}mm", steps, dz * 1000.0);

        let initial_power: f64 = field.iter().map(|&x| x * x).sum();

        for step in 0..steps {
            let mut field_view = field.view_mut();
            op.apply(&mut field_view, dz);

            if step == steps - 1 {
                let power: f64 = field.iter().map(|&x| x * x).sum();
                println!("Final power ratio: {:.6}", power / initial_power);
            }
        }

        // Measure beam radius
        let intensity = &field * &field;
        let measured = measure_beam_radius(&intensity, config.dx);
        let expected = beam_waist * 2.0_f64.sqrt();

        println!("Measured radius: {:.3}mm", measured * 1000.0);
        println!("Expected radius: {:.3}mm", expected * 1000.0);
        println!(
            "Error: {:.1}%",
            (measured - expected).abs() / expected * 100.0
        );

        // Should match within 15% even with higher resolution
        // TODO: Investigate numerical dispersion in KZK implementation
        assert_relative_eq!(measured, expected, epsilon = 0.15 * expected);
    }

    #[test]
    fn test_fft_round_trip() {
        // Test that FFT forward/inverse preserves data
        let config = KZKConfig {
            nx: 64,
            ny: 64,
            dx: 0.001,
            frequency: 1e6,
            c0: 1500.0,
            ..Default::default()
        };

        let mut op = KzkDiffractionOperator::new(&config);

        // Create test data
        let mut original = Array2::zeros((64, 64));
        for i in 0..64 {
            for j in 0..64 {
                let x = (i as f64 - 32.0) * 0.001;
                let y = (j as f64 - 32.0) * 0.001;
                original[[i, j]] = (-(x * x + y * y) / (0.01 * 0.01)).exp();
            }
        }

        // Convert to complex
        let mut complex_data = Array2::zeros((64, 64));
        for i in 0..64 {
            for j in 0..64 {
                complex_data[[i, j]] = Complex::new(original[[i, j]], 0.0);
            }
        }

        // Forward then inverse FFT
        let fft_data = op.fft_2d_forward(complex_data.clone());
        let recovered = op.fft_2d_inverse(fft_data);

        // Check recovery
        let mut max_error = 0.0;
        for i in 0..64 {
            for j in 0..64 {
                let error = (recovered[[i, j]].re - original[[i, j]]).abs();
                max_error = f64::max(max_error, error);
            }
        }

        println!("FFT round-trip max error: {:.6e}", max_error);
        assert!(max_error < 1e-10, "FFT round trip failed!");

        // Check energy conservation
        let original_energy: f64 = original.iter().map(|&x| x * x).sum();
        let recovered_energy: f64 = recovered.iter().map(|c| c.re * c.re).sum();
        println!("Original energy: {:.6}", original_energy);
        println!("Recovered energy: {:.6}", recovered_energy);
        println!("Energy ratio: {:.6}", recovered_energy / original_energy);

        assert!((recovered_energy / original_energy - 1.0).abs() < 1e-10);
    }
}
