//! Spectral filter for anti-aliasing.
//!
//! # Theory
//!
//! For nonlinear terms like u·∂u/∂x, the product in real space becomes
//! convolution in k-space, creating frequencies above the Nyquist limit.
//! The filter removes these components:
//!
//! ```text
//! û_filtered(k) = û(k) · H(k)
//! ```
//!
//! where H(k) is a window function (sharp cutoff or smooth taper).
//!
//! # Theorem: Separable modal filter
//!
//! Let `F₃ = Fₓ ⊗ Fᵧ ⊗ F_z` be the tensor-product DFT and let
//! `H(i,j,k)=Hₓ(i)Hᵧ(j)H_z(k)` with each one-dimensional transfer factor
//! bounded in `[0,1]`. The filtered field
//! `u_f = F₃⁻¹(H ⊙ F₃u)` preserves every retained Fourier mode exactly and
//! removes rejected modes without changing their phase. Constant fields are
//! invariant because `H(0,0,0)=1`.

use crate::fft::{fft_3d_array_into, ifft_3d_array_into, Complex64};
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::{Array3, ArrayView3};
use std::f64::consts::PI;

/// Types of spectral filters
#[derive(Debug, Clone, Copy)]
pub enum SpectralFilterType {
    /// Sharp cutoff at k_cutoff
    SharpCutoff,
    /// Smooth transition (Hamming window)
    Smooth,
    /// Exponential decay
    Exponential,
}

/// Spectral filter for anti-aliasing
///
/// Removes high-frequency components above a specified cutoff to prevent
/// aliasing errors in nonlinear simulations.
#[derive(Debug)]
pub struct SpectralFilter {
    /// Cutoff wavenumber (fraction of Nyquist)
    cutoff: f64,
    /// Filter type
    filter_type: SpectralFilterType,
}

impl SpectralFilter {
    /// Create a new spectral filter
    ///
    /// # Arguments
    ///
    /// * `cutoff` - Cutoff as fraction of Nyquist (typically 0.67 for 2/3 rule)
    /// * `filter_type` - Type of filter window
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(cutoff: f64, filter_type: SpectralFilterType) -> Self {
        Self {
            cutoff,
            filter_type,
        }
    }

    /// Apply the filter to a real-space field using a tensor-product FFT.
    ///
    /// The cutoff is interpreted per axis as a fraction of that axis Nyquist
    /// bin. The three one-dimensional transfer factors are multiplied, which
    /// implements the standard tensor-product modal de-aliasing contract used
    /// by Cartesian pseudospectral discretisations.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::InvalidInput` for empty arrays or invalid cutoff
    /// values.
    pub fn apply(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let shape = field.shape();
        validate_filter_input(self.cutoff, shape)?;

        let mut output = Array3::zeros(shape);
        let mut spectrum = Array3::zeros(shape);
        self.apply_into(field, &mut spectrum, &mut output)?;
        Ok(output)
    }

    /// Apply the filter using caller-owned spectrum and output workspaces.
    ///
    /// # Contract
    ///
    /// The real output buffer is first used as a staging copy for `field`, then
    /// overwritten by `IFFT(H ⊙ FFT(field))`. This removes the extra owned
    /// `field.to_owned()` copy from the convenience path and gives hot callers
    /// a zero-reallocation API:
    ///
    /// ```text
    /// out ← field
    /// spectrum ← FFT(out)
    /// spectrum`K` ← H(k) · spectrum`K`
    /// out ← IFFT(spectrum)
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`KwaversError::DimensionMismatch`] when either workspace does
    /// not match `field.shape()`, and `KwaversError::InvalidInput` for invalid
    /// cutoff values.
    pub fn apply_into(
        &self,
        field: ArrayView3<f64>,
        spectrum: &mut Array3<Complex64>,
        output: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let [nx, ny, nz] = field.shape();
        validate_filter_input(self.cutoff, field.shape())?;
        validate_workspace("spectrum", spectrum.shape(), field.shape())?;
        validate_workspace("output", output.shape(), field.shape())?;

        output.assign(&field);
        fft_3d_array_into(output, spectrum);

        for i in 0..nx {
            let hx = self.transfer_function(normalized_mode(i, nx), 1.0);
            for j in 0..ny {
                let hxy = hx * self.transfer_function(normalized_mode(j, ny), 1.0);
                for k in 0..nz {
                    spectrum[[i, j, k]] *=
                        hxy * self.transfer_function(normalized_mode(k, nz), 1.0);
                }
            }
        }

        ifft_3d_array_into(spectrum, output);
        Ok(())
    }

    /// Get filter transfer function H(k) for given wavenumber
    #[must_use]
    pub fn transfer_function(&self, k: f64, k_nyquist: f64) -> f64 {
        let k_normalized = k.abs() / k_nyquist;

        if k_normalized > self.cutoff {
            match self.filter_type {
                SpectralFilterType::SharpCutoff => 0.0,
                SpectralFilterType::Smooth => {
                    let transition = (k_normalized - self.cutoff) / (1.0 - self.cutoff);
                    0.5 * (1.0 + (PI * transition).cos())
                }
                SpectralFilterType::Exponential => {
                    let decay_rate = 10.0;
                    (-decay_rate * (k_normalized - self.cutoff).powi(2)).exp()
                }
            }
        } else {
            1.0
        }
    }
}

fn validate_filter_input(cutoff: f64, shape: [usize; 3]) -> KwaversResult<()> {
    let [nx, ny, nz] = shape;
    if nx == 0 || ny == 0 || nz == 0 {
        return Err(KwaversError::DimensionMismatch(
            "SpectralFilter requires all dimensions to be non-zero".to_owned(),
        ));
    }

    if !(0.0..=1.0).contains(&cutoff) || !cutoff.is_finite() {
        return Err(KwaversError::InvalidInput(format!(
            "SpectralFilter cutoff must be finite and within [0, 1], got {cutoff}"
        )));
    }

    Ok(())
}

fn validate_workspace(name: &str, actual: [usize; 3], expected: [usize; 3]) -> KwaversResult<()> {
    if actual != expected {
        return Err(KwaversError::DimensionMismatch(format!(
            "SpectralFilter {name} workspace shape {actual:?} must match field shape {expected:?}"
        )));
    }

    Ok(())
}

fn normalized_mode(index: usize, n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }

    let signed = if index <= n / 2 {
        index as isize
    } else {
        index as isize - n as isize
    };
    signed.unsigned_abs() as f64 / ((n / 2) as f64)
}
