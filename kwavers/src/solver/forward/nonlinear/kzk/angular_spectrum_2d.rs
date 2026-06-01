use crate::math::fft::{Complex64, Fft2d, Shape2D, FFT_CACHE_2D};
use ndarray::{Array2, ArrayViewMut2, Zip};
use std::sync::Arc;

use super::KZKConfig;
use crate::core::constants::numerical::TWO_PI;

/// Correct 2D angular spectrum operator
pub struct AngularSpectrum2D {
    config: KZKConfig,
    kx: Array2<f64>,
    ky: Array2<f64>,
    /// Cached 2-D FFT plan for the transverse `(nx, ny)` slice shape.
    fft_plan: Arc<Fft2d>,
    /// Reusable complex field/spectrum buffer for in-place angular-spectrum propagation.
    scratch: Array2<Complex64>,
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

        let dkx = TWO_PI / (nx as f64 * config.dx);
        let dky = TWO_PI / (ny as f64 * config.dx);

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

        let fft_plan = FFT_CACHE_2D.get_or_create(Shape2D { nx, ny });
        let scratch = Array2::zeros((nx, ny));

        Self {
            config: config.clone(),
            kx,
            ky,
            fft_plan,
            scratch,
        }
    }

    /// Apply angular spectrum propagation using the cached 2-D FFT workspace.
    ///
    /// # Contract
    /// The angular-spectrum transfer function is diagonal in transverse
    /// wavenumber space. Therefore a single complex scratch buffer can hold
    /// the real input, its spectrum, and the inverse-transformed result without
    /// changing the mathematical operator. This removes one real-to-complex
    /// allocation plus one allocating forward and inverse FFT per propagation.
    pub fn propagate(&mut self, field: &mut ArrayViewMut2<f64>, distance: f64) {
        let k0 = TWO_PI * self.config.frequency / self.config.c0;

        Zip::from(&mut self.scratch)
            .and(field.view())
            .par_for_each(|s, &x| *s = Complex64::new(x, 0.0));

        self.fft_plan.forward_complex_inplace(&mut self.scratch);

        Zip::indexed(&mut self.scratch).par_for_each(|(i, j), value| {
            let kx = self.kx[[i, j]];
            let ky = self.ky[[i, j]];
            let kt2 = kx.mul_add(kx, ky * ky);

            if kt2 < k0 * k0 {
                let kz = (k0 * k0 - kt2).sqrt();
                let phase = kz * distance;
                *value *= Complex64::from_polar(1.0, phase);
            } else {
                let alpha = (kt2 - k0 * k0).sqrt();
                *value *= (-alpha * distance).exp();
            }
        });

        self.fft_plan.inverse_complex_inplace(&mut self.scratch);

        Zip::from(field)
            .and(&self.scratch)
            .par_for_each(|out, value| *out = value.re);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use crate::core::constants::numerical::MHZ_TO_HZ;

    #[test]
    fn test_2d_angular_spectrum() {
        let config = KZKConfig {
            nx: 128,
            ny: 128,
            dx: 0.2e-3,
            frequency: MHZ_TO_HZ,
            c0: SOUND_SPEED_WATER_SIM,
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
        let scratch_ptr = op.scratch.as_ptr();
        let mut field_view = field.view_mut();
        op.propagate(&mut field_view, 10e-3);
        let center_after_propagation = field_view[[64, 64]];
        op.propagate(&mut field_view, 0.0);
        assert_eq!(op.scratch.as_ptr(), scratch_ptr);
        assert!((field_view[[64, 64]] - center_after_propagation).abs() < 1e-10);

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
