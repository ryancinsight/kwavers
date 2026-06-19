//! Coherence-factor adaptive weighting for time-domain delay-and-sum (DAS).
//!
//! Coherence factors are per-pixel quality weights that suppress incoherent
//! (off-axis clutter, reverberation, electronic-noise) energy in DAS images by
//! measuring how *coherent* the delay-aligned aperture data is at each output
//! sample. The weight multiplies the DAS output:
//! `y_cf[j] = CF[j] · y_das[j]`, with `CF[j] ∈ [0, 1]`.
//!
//! All estimators consume the **unapodized** delay-aligned aperture matrix from
//! [`super::das::align_channels`] (shape `(n_elements, n_samples)`): coherence
//! measures the wavefront, not the receive taper, so a perfectly coherent
//! wavefront yields `CF = 1` regardless of the DAS apodization (Mallart & Fink
//! 1994).
//!
//! # Variants
//!
//! - [`CoherenceFactor::Amplitude`] — Mallart & Fink (1994) amplitude coherence
//!   factor, the ratio of coherent to total aperture energy.
//! - [`CoherenceFactor::Sign`] — Camacho, Fritsch & Cruza (2009) sign coherence
//!   factor, robust to amplitude outliers.
//!
//! # References
//!
//! - Mallart, R., & Fink, M. (1994). "Adaptive focusing in scattering media
//!   through sound-speed inhomogeneities: The van Cittert–Zernike approach and
//!   focusing criterion." *J. Acoust. Soc. Am.* 96(6), 3721–3732. — coherence
//!   factor as the normalized coherent energy.
//! - Hollman, K. W., Rigby, K. W., & O'Donnell, M. (1999). "Coherence factor of
//!   speckle from a multi-row probe." *IEEE Ultrason. Symp.* — CF imaging use.
//! - Camacho, J., Parrilla, M., & Fritsch, C. (2009). "Phase coherence imaging."
//!   *IEEE Trans. Ultrason. Ferroelectr. Freq. Control* 56(5), 958–974. — sign
//!   and phase coherence factors.

use super::das::{align_channels, sum_aligned};
use super::delay_reference::DelayReference;
use kwavers_core::error::{KwaversError, KwaversResult};
use ndarray::{Array1, Array2, Array3};

/// Amplitude coherence factor from pre-accumulated aperture sums
/// (Mallart & Fink 1994).
///
/// `CF = |coherent_sum|² / (n · sum_of_squares)`, where `coherent_sum = Σᵢ xᵢ`
/// (or the magnitude of the complex/baseband coherent sum) and
/// `sum_of_squares = Σᵢ |xᵢ|²` is the **sum of per-element energies** (not the
/// square of the sum of magnitudes). The Cauchy–Schwarz inequality bounds the
/// ratio to `[0, 1]`, so a perfectly coherent aperture yields `1`. Returns `0`
/// for an empty or zero-energy aperture.
///
/// This is the canonical scalar entry point for accumulator-style beamformers
/// (e.g. SAFT) that build the coherent and energy sums voxel-by-voxel; the
/// matrix-column path ([`CoherenceFactor::weights`]) routes through it too.
#[must_use]
pub fn amplitude_coherence_from_sums(coherent_sum: f64, sum_of_squares: f64, n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let denom = (n as f64) * sum_of_squares;
    if denom <= 0.0 {
        return 0.0;
    }
    ((coherent_sum * coherent_sum) / denom).clamp(0.0, 1.0)
}

/// Per-output-sample coherence-factor estimator.
///
/// Each variant maps the delay-aligned aperture column `xⱼ = aligned[:, j]` to a
/// weight `CF[j] ∈ [0, 1]`: `1` for a perfectly coherent aperture, `0` for a
/// fully incoherent one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoherenceFactor {
    /// Mallart & Fink (1994) amplitude coherence factor:
    /// `CF = |Σᵢ xᵢ|² / (N · Σᵢ xᵢ²)`.
    ///
    /// The numerator is the coherent (in-phase) energy and the denominator is
    /// `N×` the incoherent (total) energy; the Cauchy–Schwarz inequality bounds
    /// the ratio to `[0, 1]`. Returns `0` for an all-zero column.
    Amplitude,
    /// Camacho et al. (2009) sign coherence factor:
    /// `SCF = (1 − √(1 − b̄²))^p`, where `b̄ = (1/N) Σᵢ sign(xᵢ)` and
    /// `p = sensitivity ≥ 1`.
    ///
    /// Depends only on the sign pattern across the aperture, so it is robust to
    /// amplitude outliers (bright reflectors, gain mismatch). `sensitivity`
    /// sharpens the taper; `1.0` is the canonical default.
    Sign {
        /// Exponent `p ≥ 1`; larger values reject incoherent energy more aggressively.
        sensitivity: f64,
    },
}

impl CoherenceFactor {
    /// Validate variant parameters.
    ///
    /// # Errors
    /// - [`KwaversError::InvalidInput`] if `Sign::sensitivity` is non-finite or `< 1`.
    pub fn validate(&self) -> KwaversResult<()> {
        if let CoherenceFactor::Sign { sensitivity } = *self {
            if !sensitivity.is_finite() || sensitivity < 1.0 {
                return Err(KwaversError::InvalidInput(format!(
                    "CoherenceFactor::Sign requires finite sensitivity >= 1.0; got {sensitivity}"
                )));
            }
        }
        Ok(())
    }

    /// Coherence weight for a single delay-aligned aperture column.
    #[must_use]
    fn weight_for_column(&self, column: &[f64]) -> f64 {
        let n = column.len();
        if n == 0 {
            return 0.0;
        }
        match *self {
            CoherenceFactor::Amplitude => {
                let mut coherent = 0.0_f64; // Σ xᵢ
                let mut sum_of_squares = 0.0_f64; // Σ xᵢ²
                for &x in column {
                    coherent += x;
                    sum_of_squares += x * x;
                }
                amplitude_coherence_from_sums(coherent, sum_of_squares, n)
            }
            CoherenceFactor::Sign { sensitivity } => {
                let mut sign_sum = 0.0_f64;
                for &x in column {
                    // Explicit three-way sign: f64::signum(0.0) is +1.0, but the
                    // sign coherence factor requires sign(0) := 0 (NaN ↦ 0 too).
                    sign_sum += if x > 0.0 {
                        1.0
                    } else if x < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                }
                let b = sign_sum / (n as f64); // b̄ ∈ [-1, 1]
                let base = 1.0 - (1.0 - b * b).max(0.0).sqrt();
                base.clamp(0.0, 1.0).powf(sensitivity)
            }
        }
    }

    /// Per-output-sample coherence weights in `[0, 1]`, length `aligned.ncols()`.
    ///
    /// `aligned` is the delay-aligned, **unapodized** aperture matrix from
    /// [`align_channels`], shape `(n_elements, n_samples)`.
    ///
    /// # Errors
    /// - [`KwaversError::InvalidInput`] on invalid variant parameters or an
    ///   empty aperture (`n_elements == 0`).
    pub fn weights(&self, aligned: &Array2<f64>) -> KwaversResult<Array1<f64>> {
        self.validate()?;
        let (n_elements, n_samples) = aligned.dim();
        if n_elements == 0 {
            return Err(KwaversError::InvalidInput(
                "CoherenceFactor::weights requires n_elements > 0".to_owned(),
            ));
        }
        let mut cf = Array1::<f64>::zeros(n_samples);
        let mut column = vec![0.0_f64; n_elements];
        for j in 0..n_samples {
            for (i, slot) in column.iter_mut().enumerate() {
                *slot = aligned[[i, j]];
            }
            cf[j] = self.weight_for_column(&column);
        }
        Ok(cf)
    }
}

/// Time-domain DAS with coherence-factor adaptive weighting.
///
/// Aligns the aperture once, computes the chosen coherence factor on the raw
/// (unapodized) aligned data, sums with `weights`, and scales each output sample
/// by its coherence weight.
///
/// # Returns
/// `(output, coherence_map)`:
/// - `output`: coherence-weighted beamformed signal, shape `(1, 1, n_samples)`.
/// - `coherence_map`: the per-sample `CF[j] ∈ [0, 1]`, length `n_samples`.
///
/// # Errors
/// - [`KwaversError::InvalidInput`] on DAS contract violations (see
///   [`align_channels`]) or invalid coherence parameters.
pub fn delay_and_sum_coherence(
    sensor_data: &Array3<f64>,
    sampling_frequency_hz: f64,
    delays_s: &[f64],
    weights: &[f64],
    reference: DelayReference,
    factor: CoherenceFactor,
) -> KwaversResult<(Array3<f64>, Array1<f64>)> {
    factor.validate()?;
    let aligned = align_channels(sensor_data, sampling_frequency_hz, delays_s, weights, reference)?;
    let cf = factor.weights(&aligned)?;
    let mut output = sum_aligned(&aligned, weights);
    for (j, &w) in cf.iter().enumerate() {
        output[[0, 0, j]] *= w;
    }
    Ok((output, cf))
}

#[cfg(test)]
mod tests;
