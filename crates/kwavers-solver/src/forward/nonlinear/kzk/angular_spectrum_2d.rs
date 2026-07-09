use apollo::{fft_2d_complex_inplace, ifft_2d_complex_inplace, Complex64};
use leto::Array2 as LetoArray2;
use moirai_parallel::{enumerate_mut_with, Adaptive};
use leto::{
    Array2,
    ArrayViewMut2,
};

use super::KZKConfig;
use kwavers_core::constants::numerical::TWO_PI;

/// Correct 2D angular spectrum operator
pub struct AngularSpectrum2D {
    config: KZKConfig,
    kx: Array2<f64>,
    ky: Array2<f64>,
    /// Reusable complex field/spectrum buffer for in-place angular-spectrum propagation.
    scratch: LetoArray2<Complex64>,
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

        let scratch = LetoArray2::zeros([nx, ny]);

        Self {
            config: config.clone(),
            kx,
            ky,
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

        let scratch = self
            .scratch
            .as_slice_mut()
            .expect("invariant: KZK angular-spectrum scratch is standard-layout");
        if let Some(field_values) = field.as_slice() {
            enumerate_mut_with::<Adaptive, _, _>(scratch, |idx, s| {
                *s = Complex64::new(field_values[idx], 0.0);
            });
        } else {
            for (s, &x) in scratch.iter_mut().zip(field.iter()) {
                *s = Complex64::new(x, 0.0);
            }
        }

        fft_2d_complex_inplace(&mut self.scratch);

        let kx = self
            .kx
            .as_slice()
            .expect("invariant: KZK angular-spectrum kx is standard-layout");
        let ky = self
            .ky
            .as_slice()
            .expect("invariant: KZK angular-spectrum ky is standard-layout");
        let scratch = self
            .scratch
            .as_slice_mut()
            .expect("invariant: KZK angular-spectrum scratch is standard-layout");
        enumerate_mut_with::<Adaptive, _, _>(scratch, |idx, value| {
            let kx = kx[idx];
            let ky = ky[idx];
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

        ifft_2d_complex_inplace(&mut self.scratch);

        let scratch = self
            .scratch
            .as_slice()
            .expect("invariant: KZK angular-spectrum scratch is standard-layout");
        if let Some(field_values) = field.as_slice_mut() {
            enumerate_mut_with::<Adaptive, _, _>(field_values, |idx, out| {
                *out = scratch[idx].re;
            });
        } else {
            for (out, value) in field.iter_mut().zip(scratch.iter()) {
                *out = value.re;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use kwavers_core::constants::numerical::MHZ_TO_HZ;

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
