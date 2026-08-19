//! Spectral-difference attenuation estimation.
//!
//! Estimates the frequency-dependent attenuation coefficient α(f) in dB/MHz/cm
//! from the depth-rate of the reference-normalized power spectrum.
//!
//! # Method
//!
//! For two analysis gates at depths `d₁ < d₂` (cm), the normalized spectra
//! `S₁(f)` and `S₂(f)` differ by the round-trip attenuation over the extra
//! propagation path `Δd = d₂ - d₁`:
//!
//! ```text
//! ΔS(f) = S₂(f) - S₁(f)    [dB, normalized]
//! α(f)  = -ΔS(f) / (2·Δd)   [dB/cm, one-way]
//! ```
//!
//! The factor of 2 accounts for the round-trip path. α(f) is modelled as
//! linear over the usable bandwidth:
//!
//! ```text
//! α(f) ≈ α₀ + β·f            [β in dB/(cm·Hz)]
//! ```
//!
//! and β (dB/MHz/cm after scaling) is the **attenuation coefficient slope**,
//! the tissue-characterization parameter from Yao et al. 1990.
//!
//! # ITK divergence
//!
//! `itkAttenuationImageFilter::ComputeAttenuation` computes the ratio of
//! spectral components rather than the difference of log-domain spectra.
//! This implementation follows Yao/Zagzebski/Madsen (1990) eq. (3), which
//! stays in the dB domain throughout. The divergence is recorded in
//! `gap_audit.md`.
//!
//! # References
//! - Yao, L. X., Zagzebski, J. A., & Madsen, E. L. (1990). "Backscatter
//!   coefficient measurements using a reference phantom to extract depth-dependent
//!   instrumentation effects." *Ultrason. Imaging* 12(1), 58–70.

use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array1;

use super::backscatter::AnalysisBand;

/// Attenuation coefficient estimated from two gated spectra at different depths.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttenuationEstimate {
    /// Frequency-independent offset of the linear α(f) fit, dB/cm.
    pub intercept_db_per_cm: f64,
    /// Attenuation coefficient slope, dB/(MHz·cm).
    ///
    /// Normal soft tissue: 0.3–0.6 dB/(MHz·cm).
    pub slope_db_per_mhz_cm: f64,
    /// Midband attenuation at the centre of the analysis band, dB/cm.
    pub midband_db_per_cm: f64,
    /// Number of frequency bins used in the fit.
    pub bins_used: usize,
}

/// Estimate the attenuation coefficient from two normalized spectra at different depths.
///
/// `spectrum_shallow` and `spectrum_deep` are reference-normalized spectra in dB
/// (from `gated_spectrum` → `normalize_to_reference`) at depths
/// `depth_shallow_cm < depth_deep_cm`. Each bin `i` is at frequency
/// `i · frequency_step_hz`.
///
/// Non-finite bins are dropped from the fit rather than rejected, consistent
/// with `backscatter_parameters`.
///
/// # Errors
///
/// Returns `InvalidInput` when:
/// - Either spectrum is empty or the two spectra differ in length.
/// - `frequency_step_hz` is not strictly positive.
/// - `depth_shallow_cm >= depth_deep_cm` or either depth is non-positive.
/// - Fewer than two finite bins fall inside `band`.
pub fn attenuation_from_spectra(
    spectrum_shallow: &Array1<f64>,
    spectrum_deep: &Array1<f64>,
    frequency_step_hz: f64,
    depth_shallow_cm: f64,
    depth_deep_cm: f64,
    band: AnalysisBand,
) -> KwaversResult<AttenuationEstimate> {
    if !frequency_step_hz.is_finite() || frequency_step_hz <= 0.0 {
        return Err(KwaversError::InvalidInput(
            "frequency_step_hz must be finite and positive".to_owned(),
        ));
    }
    let n = spectrum_shallow.len();
    if n == 0 {
        return Err(KwaversError::InvalidInput(
            "spectra must not be empty".to_owned(),
        ));
    }
    if spectrum_deep.len() != n {
        return Err(KwaversError::InvalidInput(format!(
            "spectra must have equal length ({n} vs {})",
            spectrum_deep.len(),
        )));
    }
    if !depth_shallow_cm.is_finite()
        || !depth_deep_cm.is_finite()
        || depth_shallow_cm <= 0.0
        || depth_deep_cm <= 0.0
    {
        return Err(KwaversError::InvalidInput(format!(
            "depths must be finite and positive, got shallow={depth_shallow_cm}, deep={depth_deep_cm}"
        )));
    }
    if depth_shallow_cm >= depth_deep_cm {
        return Err(KwaversError::InvalidInput(format!(
            "shallow depth ({depth_shallow_cm}) must be less than deep depth ({depth_deep_cm})"
        )));
    }

    let delta_d_cm = depth_deep_cm - depth_shallow_cm;

    // Collect in-band (f_MHz, α_db_per_cm) pairs.
    // α(f) = -ΔS(f) / (2·Δd) where ΔS = S_deep - S_shallow [dB].
    let mut freqs_mhz: Vec<f64> = Vec::new();
    let mut alphas: Vec<f64> = Vec::new();

    for (bin, (s_shallow, s_deep)) in spectrum_shallow
        .iter()
        .zip(spectrum_deep.iter())
        .enumerate()
    {
        let f_hz = bin as f64 * frequency_step_hz;
        if f_hz < band.low_hz || f_hz > band.high_hz {
            continue;
        }
        if !s_shallow.is_finite() || !s_deep.is_finite() {
            continue; // drop non-finite bins, consistent with backscatter_parameters
        }
        let delta_s_db = s_deep - s_shallow; // dB, negative when depth attenuates
        let alpha = -delta_s_db / (2.0 * delta_d_cm); // dB/cm, positive for absorbing tissue
        freqs_mhz.push(f_hz * 1e-6);
        alphas.push(alpha);
    }

    let bins_used = freqs_mhz.len();
    if bins_used < 2 {
        return Err(KwaversError::InvalidInput(format!(
            "analysis band [{}, {}] Hz contains {bins_used} usable bins; fit needs >= 2",
            band.low_hz, band.high_hz,
        )));
    }

    let (intercept_db_per_cm, slope_db_per_mhz_cm) = linear_fit(&freqs_mhz, &alphas);

    let centre_mhz = (band.low_hz + band.high_hz) * 0.5e-6;
    let midband_db_per_cm = intercept_db_per_cm + slope_db_per_mhz_cm * centre_mhz;

    Ok(AttenuationEstimate {
        intercept_db_per_cm,
        slope_db_per_mhz_cm,
        midband_db_per_cm,
        bins_used,
    })
}

/// OLS `y ≈ a + b·x`. Returns `(intercept, slope)`.
fn linear_fit(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let sum_x: f64 = xs.iter().sum();
    let sum_y: f64 = ys.iter().sum();
    let sum_xx: f64 = xs.iter().map(|x| x * x).sum();
    let sum_xy: f64 = xs.iter().zip(ys).map(|(x, y)| x * y).sum();
    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < f64::EPSILON {
        return (sum_y / n, 0.0);
    }
    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;
    (intercept, slope)
}
