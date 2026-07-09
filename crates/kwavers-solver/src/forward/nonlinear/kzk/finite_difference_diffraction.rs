//! Diffraction operator for KZK equation
//!
//! Implements the angular spectrum method for beam diffraction.
//! Reference: Vecchio & Lewin (1994) "Finite amplitude acoustic propagation"

use apollo::{fft_1d_complex, ifft_1d_complex, Complex64};
use leto::Array1 as LetoArray1;
use leto::{
    Array2,
    ArrayViewMut2,
};

use super::KZKConfig;
use kwavers_core::constants::numerical::TWO_PI;

/// Diffraction operator using angular spectrum method
pub struct KzkDiffractionOperator {
    /// Wavenumber squared in x direction
    kx2: Array2<f64>,
    /// Wavenumber squared in y direction
    ky2: Array2<f64>,
    /// Configuration
    config: KZKConfig,
}

impl std::fmt::Debug for KzkDiffractionOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KzkDiffractionOperator")
            .field(
                "kx2",
                &format!("Array2<f64> {}x{}", self.kx2.nrows(), self.kx2.ncols()),
            )
            .field(
                "ky2",
                &format!("Array2<f64> {}x{}", self.ky2.nrows(), self.ky2.ncols()),
            )
            .field("fft_plan", &"<Apollo FFT plan>")
            .field("config", &self.config)
            .finish()
    }
}

impl KzkDiffractionOperator {
    /// Create new diffraction operator
    #[must_use]
    pub fn new(config: &KZKConfig) -> Self {
        let mut kx2 = Array2::zeros((config.nx, config.ny));
        let mut ky2 = Array2::zeros((config.nx, config.ny));

        // Compute wavenumber arrays
        let dkx = TWO_PI / (config.nx as f64 * config.dx);
        let dky = TWO_PI / (config.ny as f64 * config.dx);

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
            config: config.clone(),
        }
    }

    /// Apply diffraction for one step
    /// Solves: ∂p/∂z = (ic₀/2ω)∇⊥²p in frequency domain
    pub fn apply(&mut self, slice: &mut ArrayViewMut2<f64>, step_size: f64) {
        let nx = self.config.nx;
        let ny = self.config.ny;

        // Convert to complex for FFT
        let complex_field = LetoArray1::from_shape_vec(
            [nx * ny],
            (0..nx * ny)
                .map(|idx| {
                    let i = idx % nx;
                    let j = idx / nx;
                    Complex64::new(slice[[i, j]], 0.0)
                })
                .collect(),
        )
        .expect("KZK diffraction line shape must match its Leto FFT shape");

        // Forward FFT (1D of size nx * ny as in original implementation)
        let mut transformed = fft_1d_complex(&complex_field);

        // Apply diffraction propagator in frequency domain
        let k0 = TWO_PI * self.config.frequency / self.config.c0;
        let factor = -step_size / (2.0 * k0);

        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                let k_perp2 = self.kx2[[i, j]] + self.ky2[[i, j]];
                let phase = factor * k_perp2;

                // Multiply by propagator
                transformed[idx] *= Complex64::from_polar(1.0, phase);
            }
        }

        // Inverse FFT
        let inverted = ifft_1d_complex(&transformed);

        // Copy back to real array (normalized part is already in ifft_1d_complex)
        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                slice[[i, j]] = inverted[idx].re;
            }
        }
    }
}
