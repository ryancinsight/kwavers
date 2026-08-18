//! Lizzi–Feleppa backscatter parameters from a normalized power spectrum.

use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array1;

/// The frequency band over which the line fit is performed.
///
/// A transducer only has usable sensitivity over part of its spectrum; outside
/// it the normalized spectrum is noise, and including those bins would drag the
/// regression. The band is stated explicitly rather than inferred, so a result
/// always carries the bandwidth it was measured over.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnalysisBand {
    /// Low edge, Hz (inclusive).
    pub low_hz: f64,
    /// High edge, Hz (inclusive).
    pub high_hz: f64,
}

impl AnalysisBand {
    /// Construct an analysis band.
    ///
    /// # Errors
    /// Returns `InvalidInput` when either edge is non-finite or negative, or
    /// when `low_hz >= high_hz`.
    pub fn try_new(low_hz: f64, high_hz: f64) -> KwaversResult<Self> {
        if !low_hz.is_finite() || !high_hz.is_finite() || low_hz < 0.0 || high_hz <= low_hz {
            return Err(KwaversError::InvalidInput(format!(
                "analysis band must satisfy 0 <= low < high, got [{low_hz}, {high_hz}]"
            )));
        }
        Ok(Self { low_hz, high_hz })
    }
}

/// Lizzi–Feleppa spectral parameters of a normalized backscatter spectrum.
///
/// All three are read off the same straight-line fit to the band-limited
/// normalized spectrum in dB, and are only comparable between acquisitions that
/// used the same reference phantom and analysis band.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BackscatterParameters {
    /// Spectral slope, dB/Hz, **negated** so that the usual decreasing spectrum
    /// yields a positive value. This matches ITK's convention
    /// (`itkBackscatterImageFilter`), which negates for the same reason.
    ///
    /// Relates to effective scatterer size: larger scatterers roll off faster.
    pub slope_db_per_hz: f64,
    /// Spectral intercept, dB — the fitted line extrapolated to zero frequency.
    pub intercept_db: f64,
    /// Midband fit, dB: the mean of the band-limited spectrum. Reported as the
    /// mean rather than the fitted line's midband value; the two coincide for a
    /// least-squares fit over uniformly spaced bins, and the mean is what ITK
    /// reports.
    pub midband_db: f64,
    /// Number of frequency bins the fit used.
    pub bins_used: usize,
}

/// Fit the Lizzi–Feleppa parameters over `band`.
///
/// `normalized_db` is a reference-normalized spectrum in dB (see
/// [`super::normalize_to_reference`]) whose bin `i` is at frequency
/// `i · frequency_step_hz`.
///
/// Non-finite bins — those the normalization marked as carrying no reference
/// power — are dropped rather than fitted, so a band that overlaps the
/// transducer's dead zone narrows the fit instead of poisoning it.
///
/// # Errors
/// Returns `InvalidInput` when `frequency_step_hz` is not positive, or when
/// fewer than two usable bins fall inside `band` — a line through fewer than
/// two points is not determined, and returning a fabricated slope there would
/// be worse than failing.
pub fn backscatter_parameters(
    normalized_db: &Array1<f64>,
    frequency_step_hz: f64,
    band: AnalysisBand,
) -> KwaversResult<BackscatterParameters> {
    if !frequency_step_hz.is_finite() || frequency_step_hz <= 0.0 {
        return Err(KwaversError::InvalidInput(
            "frequency_step_hz must be finite and positive".to_owned(),
        ));
    }

    let mut frequencies = Vec::new();
    let mut values = Vec::new();
    for (bin, &value) in normalized_db.iter().enumerate() {
        let frequency = bin as f64 * frequency_step_hz;
        if frequency >= band.low_hz && frequency <= band.high_hz && value.is_finite() {
            frequencies.push(frequency);
            values.push(value);
        }
    }

    let n = frequencies.len();
    if n < 2 {
        return Err(KwaversError::InvalidInput(format!(
            "analysis band [{}, {}] Hz contains {n} usable bins; a line fit needs at least 2",
            band.low_hz, band.high_hz
        )));
    }

    // Ordinary least squares in the centred variable, which keeps the normal
    // equations well conditioned: RF frequencies are ~1e6 while the spectrum is
    // ~1e1, so fitting against raw frequency squares a large number against a
    // small one and loses precision in the slope.
    let count = n as f64;
    let mean_frequency = frequencies.iter().sum::<f64>() / count;
    let mean_value = values.iter().sum::<f64>() / count;

    let mut covariance = 0.0;
    let mut variance = 0.0;
    for (&frequency, &value) in frequencies.iter().zip(values.iter()) {
        let df = frequency - mean_frequency;
        covariance += df * (value - mean_value);
        variance += df * df;
    }
    if variance <= 0.0 {
        return Err(KwaversError::InvalidInput(
            "analysis band collapsed to a single frequency; the slope is undetermined".to_owned(),
        ));
    }

    let fitted_slope = covariance / variance;
    Ok(BackscatterParameters {
        slope_db_per_hz: -fitted_slope,
        intercept_db: mean_value - fitted_slope * mean_frequency,
        midband_db: mean_value,
        bins_used: n,
    })
}
