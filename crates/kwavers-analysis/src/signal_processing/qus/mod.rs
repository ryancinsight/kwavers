//! Quantitative ultrasound (QUS): spectral tissue characterization from RF.
//!
//! B-mode brightness is qualitative — it depends on gain, focus and the
//! machine. The **backscatter power spectrum** of gated RF, once normalized
//! against a reference phantom acquired with the same settings, is not: it
//! carries system-independent parameters that relate to tissue microstructure.
//!
//! # Pipeline
//!
//! 1. [`gated_spectrum`] — Welch power spectrum of an axial RF gate, averaged
//!    across the beams in the analysis window.
//! 2. [`normalize_to_reference`] — divide by a reference-phantom spectrum
//!    acquired identically, in dB. This is what cancels the transducer's
//!    response, the pulse shape and the system gain; an un-normalized spectrum
//!    is not a tissue measurement and the API does not offer one.
//! 3. [`backscatter_parameters`] — least-squares line fit over the usable
//!    bandwidth, giving the Lizzi–Feleppa parameters: spectral slope, spectral
//!    intercept and midband fit.
//!
//! # References
//! - Lizzi, F. L., et al. (1983). "Theoretical framework for spectrum analysis
//!   in ultrasonic tissue characterization." *J. Acoust. Soc. Am.* 73(4),
//!   1366–1373. — the slope/intercept/midband parameterization.
//! - Yao, L. X., Zagzebski, J. A., & Madsen, E. L. (1990). "Backscatter
//!   coefficient measurements using a reference phantom to extract depth-dependent
//!   instrumentation effects." *Ultrason. Imaging* 12(1), 58–70. — the
//!   reference-phantom normalization this module requires.
//! - `itkBackscatterImageFilter.hxx`, KitwareMedical/ITKUltrasound — the
//!   band-limited line fit and the slope-negation convention are matched to
//!   that implementation.

use kwavers_core::error::{KwaversError, KwaversResult};
use leto::{Array1, ArrayView2};

use super::spectral::{SpectralAnalysis, SpectralConfig};

mod attenuation;
mod backscatter;

#[cfg(test)]
mod tests;

pub use attenuation::{attenuation_from_spectra, AttenuationEstimate};
pub use backscatter::{backscatter_parameters, AnalysisBand, BackscatterParameters};

/// Power spectrum of an axial RF gate, averaged over the beams in the window.
///
/// `rf` is `[n_lines, n_samples]`, the same beam-space layout the B-mode
/// pipeline uses. The gate is `sample_range` along the axial axis; every beam
/// contributes one Welch periodogram and the result is their mean, which is the
/// standard variance reduction for a speckle-dominated estimate — a single
/// realization of a Rayleigh-scattering spectrum has ~100% standard deviation,
/// so averaging over independent beams is what makes the estimate usable.
///
/// Returns the one-sided spectrum, `fft_size/2 + 1` bins spanning
/// `0 … sample_rate/2`.
///
/// # Errors
/// Returns `InvalidInput` when the gate is empty or outside `rf`, when `rf` has
/// no beams, or when `sample_rate` is not positive.
///
/// # Panics
///
/// Panics if a caller-supplied shape or an internal analysis state violates
/// the precondition required by this operation.
pub fn gated_spectrum(
    rf: ArrayView2<f64>,
    sample_range: std::ops::Range<usize>,
    sample_rate_hz: f64,
    fft_size: usize,
) -> KwaversResult<Array1<f64>> {
    let [n_lines, n_samples] = rf.shape();
    if n_lines == 0 {
        return Err(KwaversError::InvalidInput(
            "RF window must contain at least one beam".to_owned(),
        ));
    }
    if sample_range.is_empty() || sample_range.end > n_samples {
        return Err(KwaversError::InvalidInput(format!(
            "gate {:?} is empty or exceeds the {n_samples} available samples",
            sample_range
        )));
    }
    if !sample_rate_hz.is_finite() || sample_rate_hz <= 0.0 {
        return Err(KwaversError::InvalidInput(
            "sample_rate_hz must be finite and positive".to_owned(),
        ));
    }

    let analysis = SpectralAnalysis::new(SpectralConfig {
        fft_size,
        // A gate is short by construction; overlapping segments inside it would
        // correlate the periodograms rather than add independent averages. The
        // averaging that matters here is across beams.
        overlap: 0.0,
    });

    let gate_len = sample_range.len();
    let mut accumulator: Option<Array1<f64>> = None;
    for line in 0..n_lines {
        let mut gate = Vec::with_capacity(gate_len);
        for sample in sample_range.clone() {
            gate.push(rf[[line, sample]]);
        }
        let gate = Array1::from_vec([gate_len], gate)
            .map_err(|error| KwaversError::InvalidInput(error.to_string()))?;
        let psd = analysis.compute_psd(gate.view(), sample_rate_hz)?;
        match accumulator.as_mut() {
            Some(sum) => {
                for (acc, &value) in sum.iter_mut().zip(psd.iter()) {
                    *acc += value;
                }
            }
            None => accumulator = Some(psd),
        }
    }

    let mut mean = accumulator.expect("invariant: n_lines > 0 checked above");
    let scale = 1.0 / n_lines as f64;
    for value in mean.iter_mut() {
        *value *= scale;
    }
    Ok(mean)
}

/// Reference-phantom normalization: `10·log₁₀(S_sample / S_reference)`, in dB.
///
/// The sample and reference spectra must come from the same transducer,
/// settings and depth, which is what makes the ratio cancel the system response
/// and leave a tissue-dependent quantity.
///
/// Bins where the reference has no power carry no information — the system did
/// not transmit there — and are returned as `f64::NEG_INFINITY` rather than a
/// finite value invented by a floor. [`backscatter_parameters`] restricts its
/// fit to a stated band, so these bins are excluded by construction rather than
/// silently biasing a regression.
///
/// # Errors
/// Returns `InvalidInput` when the two spectra differ in length or are empty.
pub fn normalize_to_reference(
    sample: &Array1<f64>,
    reference: &Array1<f64>,
) -> KwaversResult<Array1<f64>> {
    if sample.len() != reference.len() {
        return Err(KwaversError::DimensionMismatch(format!(
            "sample spectrum has {} bins, reference has {}",
            sample.len(),
            reference.len()
        )));
    }
    if sample.is_empty() {
        return Err(KwaversError::InvalidInput(
            "spectra must have at least one bin".to_owned(),
        ));
    }

    let values: Vec<f64> = sample
        .iter()
        .zip(reference.iter())
        .map(|(&s, &r)| {
            if r > 0.0 && s > 0.0 {
                10.0 * (s / r).log10()
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();
    Array1::from_vec([values.len()], values)
        .map_err(|error| KwaversError::InvalidInput(error.to_string()))
}
