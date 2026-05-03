//! KZK parabolic diffraction sub-step operator.
//!
//! Implements the transverse diffraction term of the KZK equation using
//! the spectral propagator in the parabolic (paraxial) approximation.
//!
//! # Mathematical Derivation
//!
//! The KZK diffraction sub-step is:
//!
//! ```text
//! ∂²p/∂z∂τ = (c₀/2) ∇⊥²p
//! ```
//!
//! Fourier-transforming over retarded time τ with convention
//! `P̂(ω) = ∫ p(τ) e^{−iωτ} dτ`:
//!
//! - `∂p/∂τ ↔ iω P̂`
//! - `∇⊥²p ↔ −k_T² P̂`   (transverse spatial Fourier transform)
//!
//! Substituting:
//!
//! ```text
//! iω ∂P̂/∂z = −(c₀/2) k_T² P̂
//! ∂P̂/∂z    = −i k_T² / (2k₀) P̂      [k₀ = ω/c₀]
//! ```
//!
//! # Theorem (parabolic diffraction propagator)
//!
//! **Statement.** The unique bounded solution of
//! ```text
//!   ∂P̂/∂z = −i [k_T²/(2k₀)] P̂,   P̂(z=0) = P̂₀
//! ```
//! is
//! ```text
//!   P̂(z) = P̂₀ · exp(−i k_T² z / (2k₀))
//! ```
//!
//! **Proof.** This is a linear first-order ODE dP̂/dz = cP̂ with constant
//! coefficient c = −ik_T²/(2k₀).  Integrating: P̂(z) = P̂₀ · exp(cz).  QED.
//!
//! **Corollary.** |exp(−ik_T²z/(2k₀))| = 1 for all real k_T, k₀, z.
//! Diffraction is energy-conserving: it redistributes energy across
//! transverse wavenumbers without creating or destroying it (Parseval).
//!
//! # References
//!
//! - Aanonsen SI et al. (1984). J. Acoust. Soc. Am. 75(3), 749–768.
//!   DOI: 10.1121/1.390585
//! - Lee Y-S, Hamilton MF (1995). "Time-domain modeling of pulsed
//!   finite-amplitude sound beams." J. Acoust. Soc. Am. 97(2), 906–917.
//!   DOI: 10.1121/1.412000
//! - Christopher PT, Parker KJ (1991). "New approaches to nonlinear
//!   diffractive field propagation." J. Acoust. Soc. Am. 90(1), 488–499.
//!   DOI: 10.1121/1.401277

use crate::math::fft::{fft_2d_complex, ifft_2d_complex, Complex64};
use ndarray::{Array2, ArrayViewMut2};
use std::f64::consts::PI;

use super::KZKConfig;

#[cfg(test)]
mod tests;

/// KZK parabolic diffraction operator.
///
/// Applies the spectral propagator H(k_T) = exp(−i k_T² Δz/(2k₀)) to a
/// 2D real-valued field slice (one retarded-time plane at a time).
///
/// # Narrowband limitation
///
/// This operator uses a single wavenumber k₀ = 2πf₀/c₀ for all time slices,
/// which is exact for a continuous-wave (CW) or narrowband pulsed source
/// (bandwidth < ~20% of centre frequency).  For broadband pulses, each
/// frequency component requires its own k₀; see `AngularSpectrum2D` for that
/// case.
///
/// # Real-field representation
///
/// The input field is real-valued (pressure at a given retarded time).  The
/// spectral propagator produces a complex result; only the real part is
/// retained.  For narrowband CW sources, the imaginary part is quadrature and
/// the real-part extraction is accurate.  A complex-field alternative is
/// provided by `ParabolicDiffractionOperator` in `complex_parabolic_diffraction`.
///
/// # References
///
/// See module-level documentation.
pub struct KzkDiffractionOperator {
    config: KZKConfig,
    kx2: Array2<f64>,
    ky2: Array2<f64>,
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
                (i as i32 - nx as i32) as f64 * dkx
            };

            for j in 0..ny {
                kx2[[i, j]] = kx * kx;
            }
        }

        for j in 0..ny {
            let ky = if j <= ny / 2 {
                j as f64 * dky
            } else {
                (j as i32 - ny as i32) as f64 * dky
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

    /// Apply KZK diffraction step
    pub fn apply(&mut self, field: &mut ArrayViewMut2<f64>, step_size: f64) {
        self.apply_with_step(field, step_size, 0);
    }

    fn apply_with_step(&mut self, field: &mut ArrayViewMut2<f64>, step_size: f64, _step: usize) {
        let nx = self.config.nx;
        let ny = self.config.ny;
        let k0 = 2.0 * PI * self.config.frequency / self.config.c0;

        // Convert to complex field
        let complex_field = field.mapv(|x| Complex64::new(x, 0.0));

        // 2D FFT using central cache
        let mut complex_field_fft = fft_2d_complex(&complex_field);

        // Parabolic diffraction propagator (see module theorem):
        //
        //   H(k_T) = exp(−i k_T² Δz / (2k₀))
        //
        // Derived from ∂P̂/∂z = −i k_T²/(2k₀) P̂ (KZK diffraction sub-step in
        // retarded-time Fourier domain).  The sign is NEGATIVE; a positive sign
        // corresponds to the complex-conjugate propagator and must not be used.
        //
        // Refs: Lee & Hamilton (1995) eq. (4); Aanonsen et al. (1984) §3.
        for i in 0..nx {
            for j in 0..ny {
                let kt2 = self.kx2[[i, j]] + self.ky2[[i, j]];
                let phase = kt2 * step_size / (2.0 * k0);
                // exp(−i·phase): negative sign is physically correct
                complex_field_fft[[i, j]] *= Complex64::from_polar(1.0, -phase);
            }
        }

        // Inverse 2D FFT: result is complex
        let recovered = ifft_2d_complex(&complex_field_fft);

        // Extract real part
        for i in 0..nx {
            for j in 0..ny {
                field[[i, j]] = recovered[[i, j]].re;
            }
        }
    }

    #[cfg(test)]
    fn fft_2d_forward(&self, data: Array2<Complex64>) -> Array2<Complex64> {
        fft_2d_complex(&data)
    }

    #[cfg(test)]
    fn fft_2d_inverse(&self, data: Array2<Complex64>) -> Array2<Complex64> {
        ifft_2d_complex(&data)
    }
}
