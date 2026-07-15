//! Delay-multiply-and-sum (DMAS) beamforming for time-domain DAS.
//!
//! DMAS (Matrone et al. 2015) replaces the linear coherent sum of delay-and-sum
//! with the sum of sign-preserving pairwise products across the aperture. Because
//! each term couples two channels, energy that is coherent across the aperture
//! (the true focus) is reinforced while incoherent off-focus energy — which has
//! random relative sign between channels — averages toward zero, narrowing the
//! mainlobe and lowering sidelobes relative to DAS.
//!
//! This module is the **single source of truth** for the DMAS combination rule:
//! the active-imaging path here and the passive-acoustic-mapping path
//! (`signal_processing::pam`) both route through [`dmas_combine`].
//!
//! # Reference
//!
//! - Matrone, G., Savoia, A. S., Caliano, G., & Magenes, G. (2015). "The
//!   delay-multiply-and-sum beamforming algorithm in ultrasound B-mode image
//!   reconstruction." *IEEE Trans. Med. Imaging* 34(4), 940–949.

use super::das::align_channels;
use super::delay_reference::DelayReference;
use kwavers_core::error::KwaversResult;
use leto::Array3;

/// Sign-preserving DMAS combination of one aligned, apodized aperture column.
///
/// For apodized, delay-aligned samples `xᵢ`, let `ŝᵢ = sign(xᵢ)·√|xᵢ|` (the
/// sign-preserving square root, which keeps the pairwise product
/// `ŝᵢŝⱼ = sign(xᵢxⱼ)·√(|xᵢ||xⱼ|)` dimensionally consistent with a pressure).
/// Returns the sum of all distinct pairwise products in `O(N)` via the closed
/// form
///
/// ```text
/// y = Σ_{i<j} ŝᵢ ŝⱼ = ½ [ (Σᵢ ŝᵢ)² − Σᵢ ŝᵢ² ].
/// ```
///
/// A single channel (or empty aperture) has no pairs and returns `0`. The
/// signed root maps `xᵢ = 0` to `0` (`√0 = 0`), so zero samples contribute
/// nothing regardless of `signum`'s sign convention at zero.
#[must_use]
pub fn dmas_combine(apodized_samples: &[f64]) -> f64 {
    let mut sum_root = 0.0_f64; // Σᵢ ŝᵢ
    let mut sum_sq = 0.0_f64; // Σᵢ ŝᵢ²
    for &x in apodized_samples {
        let signed_root = x.signum() * x.abs().sqrt();
        sum_root += signed_root;
        sum_sq += signed_root * signed_root;
    }
    0.5 * sum_root.mul_add(sum_root, -sum_sq)
}

/// Time-domain delay-multiply-and-sum beamforming.
///
/// Aligns the aperture (shared SSOT with [`super::das::delay_and_sum`]), applies
/// `weights` per channel, and combines each output sample with [`dmas_combine`].
///
/// # Returns
/// DMAS-beamformed signal, shape `(1, 1, n_samples)`.
///
/// # Errors
/// - [`kwavers_core::error::KwaversError::InvalidInput`] on DAS contract
///   violations (see [`align_channels`]).
pub fn delay_and_sum_dmas(
    sensor_data: &Array3<f64>,
    sampling_frequency_hz: f64,
    delays_s: &[f64],
    weights: &[f64],
    reference: DelayReference,
) -> KwaversResult<Array3<f64>> {
    let aligned = align_channels(
        sensor_data,
        sampling_frequency_hz,
        delays_s,
        weights,
        reference,
    )?;
    let [n_elements, n_samples] = aligned.shape();

    let mut output = Array3::<f64>::zeros((1, 1, n_samples));
    let mut column = vec![0.0_f64; n_elements];
    for j in 0..n_samples {
        for (i, slot) in column.iter_mut().enumerate() {
            *slot = weights[i] * aligned[[i, j]];
        }
        output[[0, 0, j]] = dmas_combine(&column);
    }
    Ok(output)
}

#[cfg(test)]
mod tests;
