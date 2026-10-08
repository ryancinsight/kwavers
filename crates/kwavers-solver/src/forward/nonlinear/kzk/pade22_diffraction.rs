//! # Padé `[2,2]` diffraction operator
//!
//! Second-order rational approximation matching the exact Helmholtz propagator
//! through O(s³), valid for beam half-angles up to ~55°.
//!
//! ## Transfer function
//!
//! ```text
//! H_pade22(k_T) = exp(i φ₂₂)
//! φ₂₂ = k₀ Δz · (−s/2 + s²/4) / (1 − 3s/4 + s²/16)
//! ```
//!
//! The denominator 1 − 3s/4 + s²/16 ≥ 5/16 for all propagating modes (s ≤ 1),
//! ensuring numerical stability. Evanescent modes use amplitude decay.
//!
//! ## Validity
//!
//! The `[2,2]` Padé error is O(s⁴): accurate through ~55° beam half-angle.
//!
//! ## References
//!
//! - Collins MD (1993). J. Acoust. Soc. Am. 93(4), 1671–1682.

use apollo::{fft_2d_complex_inplace, ifft_2d_complex_inplace, Complex64 as ApolloComplex64};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_math::fft::Complex64;
use leto::Array2 as LetoArray2;
use leto::{Array2, ArrayViewMut2};
use moirai_parallel::{enumerate_mut_with, Adaptive};

use super::KZKConfig;

/// Padé `[2,2]` diffraction operator in the transverse spectral domain.
pub struct Pade22DiffractionOperator {
    config: KZKConfig,
    kt2: Array2<f64>,
    scratch: LetoArray2<ApolloComplex64>,
}

impl std::fmt::Debug for Pade22DiffractionOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pade22DiffractionOperator")
            .field("config", &self.config)
            .field("kt2_shape", &self.kt2.shape())
            .field("scratch", &self.scratch.shape())
            .finish()
    }
}

impl Pade22DiffractionOperator {
    /// Create a new Padé `[2,2]` diffraction operator.
    ///
    /// ## Contract
    ///
    /// The transverse wavenumber grid depends only on `(nx, ny, dx)` and can
    /// therefore be precomputed once and reused for all retarded-time slices.
    ///
    /// ## References
    ///
    /// - Collins MD (1993). J. Acoust. Soc. Am. 93(4), 1671–1682.
    #[must_use]
    pub fn new(config: &KZKConfig) -> Self {
        let nx = config.nx;
        let ny = config.ny;
        let dkx = TWO_PI / (nx as f64 * config.dx);
        let dky = TWO_PI / (ny as f64 * config.dx);

        let mut kt2 = Array2::zeros((nx, ny));
        for i in 0..nx {
            let kx = if i <= nx / 2 {
                i as f64 * dkx
            } else {
                (i as f64 - nx as f64) * dkx
            };
            let kx2 = kx * kx;

            for j in 0..ny {
                let ky = if j <= ny / 2 {
                    j as f64 * dky
                } else {
                    (j as f64 - ny as f64) * dky
                };
                kt2[[i, j]] = kx2 + ky * ky;
            }
        }

        let scratch = LetoArray2::<ApolloComplex64>::zeros([nx, ny]);

        Self {
            config: config.clone(),
            kt2,
            scratch,
        }
    }

    /// Apply the Padé `[2,2]` diffraction step to a complex field.
    ///
    /// ## Theorem
    ///
    /// For propagating modes (`k_T² < k₀²`) the Padé `[2,2]` propagator is a
    /// pure phase factor `exp(iφ₂₂)` and therefore preserves discrete spectral
    /// energy exactly up to FFT roundoff. Evanescent modes inherit the
    /// monotone-decay amplitude factor used by the exact wide-angle operator.
    ///
    /// ## Contract
    ///
    /// The hot path performs no heap allocation: copy into the pre-sized
    /// scratch buffer, in-place FFT, pointwise propagation, in-place IFFT,
    /// then copy back to the caller view.
    ///
    /// ## References
    ///
    /// - Collins MD (1993). J. Acoust. Soc. Am. 93(4), 1671–1682.
    ///
    /// # Panics
    ///
    /// Panics if a caller-supplied shape or an internal solver state violates
    /// the precondition required by this operation.
    pub fn apply_complex(&mut self, field: &mut ArrayViewMut2<Complex64>, step_size: f64) {
        let k0 = TWO_PI * self.config.frequency / self.config.c0;
        let k02 = k0 * k0;

        let scratch = self
            .scratch
            .as_slice_mut()
            .expect("invariant: Padé [2,2] diffraction scratch is standard-layout");
        if let Some(field_values) = field.as_slice() {
            enumerate_mut_with::<Adaptive, _, _>(scratch, |idx, s| {
                let value = field_values[idx];
                *s = ApolloComplex64::new(value.re, value.im);
            });
        } else {
            for (s, &value) in scratch.iter_mut().zip(field.as_view().iter()) {
                *s = ApolloComplex64::new(value.re, value.im);
            }
        }

        fft_2d_complex_inplace(&mut self.scratch);

        let kt2 = self
            .kt2
            .as_slice()
            .expect("invariant: Padé [2,2] diffraction kt2 is standard-layout");
        let scratch = self
            .scratch
            .as_slice_mut()
            .expect("invariant: Padé [2,2] diffraction scratch is standard-layout");
        enumerate_mut_with::<Adaptive, _, _>(scratch, |idx, s_val| {
            let kt2_val = kt2[idx];
            if kt2_val < k02 {
                let s = kt2_val / k02;
                let numerator = -s / 2.0 + s * s / 4.0;
                let denominator = 1.0 - 3.0 * s / 4.0 + s * s / 16.0;
                let phase = k0 * step_size * numerator / denominator;
                *s_val *= ApolloComplex64::from_polar(1.0, phase);
            } else {
                let decay = (kt2_val - k02).sqrt() * step_size;
                *s_val *= ApolloComplex64::new((-decay).exp(), 0.0);
            }
        });

        ifft_2d_complex_inplace(&mut self.scratch);

        let scratch = self
            .scratch
            .as_slice()
            .expect("invariant: Padé [2,2] diffraction scratch is standard-layout");
        if let Some(field_values) = field.as_mut_slice() {
            enumerate_mut_with::<Adaptive, _, _>(field_values, |idx, value| {
                let scratch_value = scratch[idx];
                *value = Complex64::new(scratch_value.re, scratch_value.im);
            });
        } else {
            for (([_, _], value), &s) in field
                .reborrow()
                .indexed_iter_mut()
                .expect("invariant: 2-D field view yields indexed iterator")
                .zip(scratch.iter())
            {
                *value = Complex64::new(s.re, s.im);
            }
        }
    }
}
