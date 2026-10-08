//! Wide-angle KZK diffraction operator using the exact Helmholtz propagator.
//!
//! # Wide-angle diffraction propagator
//!
//! In the retarded-time frame (τ = t − z/c₀) the KZK diffraction sub-step
//! is solved by the spectral transfer function:
//!
//! ```text
//! H_WA(k_T) = exp(i (kz − k₀) Δz)
//!
//! kz = √(k₀² − k_T²)   for k_T ≤ k₀  (propagating modes)
//! kz = 0; amplitude decay = exp(−√(k_T² − k₀²) Δz)   for k_T > k₀  (evanescent)
//! ```
//!
//! This is the exact solution of the Helmholtz equation reduced to the retarded
//! frame; the paraxial approximation kz−k₀ ≈ −k_T²/(2k₀) recovers the standard
//! KZK propagator.
//!
//! # Validity
//!
//! Valid for all propagating angles (0°–90°); not restricted to the paraxial
//! cone. Evanescent modes are correctly attenuated rather than oscillated.
//!
//! # Zero-allocation hot path
//!
//! Pre-allocates kT², a propagating/evanescent mask, and a complex scratch
//! buffer. Each `apply_complex` call: copy→FFT→multiply→IFFT→copy.

use apollo::{fft_2d_complex_inplace, ifft_2d_complex_inplace, Complex64 as ApolloComplex64};
use kwavers_math::fft::Complex64;
use leto::Array2 as LetoArray2;
use leto::{Array2, ArrayViewMut2};
use moirai_parallel::{enumerate_mut_with, Adaptive};

use super::KZKConfig;
use kwavers_core::constants::numerical::TWO_PI;

/// Wide-angle diffraction operator for exact Helmholtz marching in the
/// retarded-time frame.
pub struct WideAngleDiffractionOperator {
    config: KZKConfig,
    kt2: Array2<f64>,
    propagating_mask: Array2<bool>,
    scratch: LetoArray2<ApolloComplex64>,
}

impl std::fmt::Debug for WideAngleDiffractionOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WideAngleDiffractionOperator")
            .field("config", &self.config)
            .field("kt2_shape", &self.kt2.shape())
            .field("propagating_mask_shape", &self.propagating_mask.shape())
            .field("scratch", &self.scratch.shape())
            .finish()
    }
}

impl WideAngleDiffractionOperator {
    /// Create a new exact Helmholtz diffraction operator.
    ///
    /// ## Contract
    ///
    /// The transverse spectral grid is fixed by `(nx, ny, dx)`, so `k_T²` and
    /// the propagating/evanescent partition can be precomputed once and reused
    /// for every retarded-time slice.
    ///
    /// ## References
    ///
    /// - Christopher PT, Parker KJ (1991). J. Acoust. Soc. Am. 90(1), 488–499.
    /// - Lee Y-S, Hamilton MF (1995). J. Acoust. Soc. Am. 97(2), 906–917.
    #[must_use]
    pub fn new(config: &KZKConfig) -> Self {
        let nx = config.nx;
        let ny = config.ny;
        let dkx = TWO_PI / (nx as f64 * config.dx);
        let dky = TWO_PI / (ny as f64 * config.dx);
        let k0 = TWO_PI * config.frequency / config.c0;
        let k02 = k0 * k0;

        let mut kt2 = Array2::zeros((nx, ny));
        let mut propagating_mask = Array2::from_elem((nx, ny), false);

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
                let mode_kt2 = kx2 + ky * ky;
                kt2[[i, j]] = mode_kt2;
                propagating_mask[[i, j]] = mode_kt2 <= k02;
            }
        }

        let scratch = LetoArray2::<ApolloComplex64>::zeros([nx, ny]);

        Self {
            config: config.clone(),
            kt2,
            propagating_mask,
            scratch,
        }
    }

    /// Apply the exact wide-angle diffraction step to a complex field.
    ///
    /// ## Theorem
    ///
    /// In transverse-wavenumber space the retarded-frame Helmholtz update is
    /// diagonal. Propagating modes satisfy `|H_WA| = 1`, so their discrete
    /// spectral energy is preserved exactly up to FFT roundoff. Evanescent
    /// modes satisfy `0 < |H_WA| < 1`, so they decay monotonically with `Δz`.
    ///
    /// ## Contract
    ///
    /// The hot path performs no heap allocation: one copy into the pre-sized
    /// scratch buffer, in-place FFT, pointwise spectral multiplication,
    /// in-place IFFT, then copy back to the caller view.
    ///
    /// ## References
    ///
    /// - Christopher PT, Parker KJ (1991). J. Acoust. Soc. Am. 90(1), 488–499.
    /// - Tavakkoli J et al. (1998). IEEE TUFFC 45(4), 1069–1079.
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
            .expect("invariant: wide-angle KZK diffraction scratch is standard-layout");
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
            .expect("invariant: wide-angle KZK diffraction kt2 is standard-layout");
        let propagating_mask = self
            .propagating_mask
            .as_slice()
            .expect("invariant: wide-angle KZK diffraction mask is standard-layout");
        let scratch = self
            .scratch
            .as_slice_mut()
            .expect("invariant: wide-angle KZK diffraction scratch is standard-layout");
        enumerate_mut_with::<Adaptive, _, _>(scratch, |idx, s| {
            let mode_kt2 = kt2[idx];
            if propagating_mask[idx] {
                let kz = (k02 - mode_kt2).sqrt();
                let phase = (kz - k0) * step_size;
                *s *= ApolloComplex64::from_polar(1.0, phase);
            } else {
                let decay_exp = (-(mode_kt2 - k02).sqrt() * step_size).exp();
                *s *= ApolloComplex64::new(decay_exp, 0.0);
            }
        });

        ifft_2d_complex_inplace(&mut self.scratch);

        let scratch = self
            .scratch
            .as_slice()
            .expect("invariant: wide-angle KZK diffraction scratch is standard-layout");
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
