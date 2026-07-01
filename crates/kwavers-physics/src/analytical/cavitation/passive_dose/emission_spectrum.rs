//! Analytical stable/inertial cavitation emission spectra for PAM figures.
//!
//! The model captures the spectral markers used by passive cavitation
//! detection: a harmonic comb plus subharmonic line for stable cavitation, and
//! the same lines over an elevated inharmonic broadband floor for inertial
//! cavitation.

/// Cavitation emission regime represented by the analytical PAM spectrum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CavitationEmissionRegime {
    /// Stable cavitation: harmonic and subharmonic line spectrum.
    Stable,
    /// Inertial cavitation: stable line spectrum plus broadband emission.
    Inertial,
}

const HARMONIC_COUNT: usize = 5;
const LINEWIDTH_FRACTION: f64 = 0.02;
const SUBHARMONIC_GAIN: f64 = 0.4;
const SUBHARMONIC_LINEWIDTH_SCALE: f64 = 0.5;
const INERTIAL_BROADBAND_GAIN: f64 = 0.15;
const INERTIAL_BROADBAND_CORNER: f64 = 3.0;

/// Normalized passive cavitation emission power spectrum.
///
/// The returned vector is a linear PSD normalized so the largest bin is one
/// after adding the requested finite signal-to-noise floor. The stable spectrum
/// is
/// `sum_n n^-1.5 L(f; n f0, 0.02 f0) + 0.4 L(f; f0/2, 0.01 f0)`.
/// The inertial spectrum adds a fourth-order broadband envelope
/// `0.15 / (1 + (f / (3 f0))^4)`.
///
/// Returns an empty vector when the frequency axis is empty, any frequency is
/// non-finite, `f0_hz <= 0`, or `snr_db` is non-finite.
#[must_use]
pub fn normalized_cavitation_emission_spectrum(
    freqs_hz: &[f64],
    f0_hz: f64,
    regime: CavitationEmissionRegime,
    snr_db: f64,
) -> Vec<f64> {
    if freqs_hz.is_empty()
        || !(f0_hz.is_finite() && f0_hz > 0.0)
        || !snr_db.is_finite()
        || freqs_hz.iter().any(|&f| !f.is_finite())
    {
        return Vec::new();
    }

    let linewidth_hz = LINEWIDTH_FRACTION * f0_hz;
    let mut spectrum = Vec::with_capacity(freqs_hz.len());
    let mut peak = 0.0_f64;

    for &frequency_hz in freqs_hz {
        let mut value = 0.0_f64;
        for harmonic in 1..=HARMONIC_COUNT {
            let harmonic_f = harmonic as f64;
            let amplitude = harmonic_f.powf(-1.5);
            value += amplitude * lorentzian_line(frequency_hz, harmonic_f * f0_hz, linewidth_hz);
        }
        value += SUBHARMONIC_GAIN
            * lorentzian_line(
                frequency_hz,
                0.5 * f0_hz,
                SUBHARMONIC_LINEWIDTH_SCALE * linewidth_hz,
            );

        if regime == CavitationEmissionRegime::Inertial {
            let normalized_frequency = frequency_hz / (INERTIAL_BROADBAND_CORNER * f0_hz);
            value += INERTIAL_BROADBAND_GAIN / (1.0 + normalized_frequency.powi(4));
        }

        peak = peak.max(value);
        spectrum.push(value);
    }

    if peak <= 0.0 {
        return vec![0.0; freqs_hz.len()];
    }

    let snr_linear = 10.0_f64.powf(snr_db / 10.0);
    let noise_floor = peak / snr_linear;
    let normalizer = peak + noise_floor;
    spectrum
        .into_iter()
        .map(|value| (value + noise_floor) / normalizer)
        .collect()
}

#[inline]
fn lorentzian_line(frequency_hz: f64, center_hz: f64, bandwidth_hz: f64) -> f64 {
    let half_width = 0.5 * bandwidth_hz;
    let numerator = half_width * half_width;
    numerator / ((frequency_hz - center_hz).powi(2) + numerator)
}
