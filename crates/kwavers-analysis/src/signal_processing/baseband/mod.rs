//! Analytic-signal I/Q demodulation for real RF channel data.
//!
//! For each real channel `x[n]`, this module forms its analytic signal
//! `z[n] = x[n] + j H{x}[n]` by one-sided FFT filtering, then translates it to
//! baseband as `z[n] exp(-j 2π f₀ n / f_s)`. The result retains the input's
//! `(channel, sample)` layout and is suitable for coherent complex DAS or
//! slow-time I/Q processing.

use apollo::{fft_1d_array, ifft_1d_complex};
use eunomia::Complex64;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::{Array1, Array2, ArrayView2};

/// Validated sampling configuration for analytic I/Q demodulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IqDemodulationConfig {
    sampling_frequency_hz: f64,
    center_frequency_hz: f64,
}

impl IqDemodulationConfig {
    /// Construct a demodulation configuration from SI frequencies.
    ///
    /// # Errors
    ///
    /// Returns [`KwaversError::InvalidInput`] unless both frequencies are finite
    /// and strictly positive and the carrier lies below the Nyquist frequency.
    pub fn new(sampling_frequency_hz: f64, center_frequency_hz: f64) -> KwaversResult<Self> {
        if !sampling_frequency_hz.is_finite() || sampling_frequency_hz <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "I/Q demodulation: sampling_frequency_hz must be finite and > 0".to_owned(),
            ));
        }
        if !center_frequency_hz.is_finite() || center_frequency_hz <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "I/Q demodulation: center_frequency_hz must be finite and > 0".to_owned(),
            ));
        }
        if center_frequency_hz >= 0.5 * sampling_frequency_hz {
            return Err(KwaversError::InvalidInput(
                "I/Q demodulation: center_frequency_hz must be below Nyquist".to_owned(),
            ));
        }
        Ok(Self {
            sampling_frequency_hz,
            center_frequency_hz,
        })
    }
}

/// Convert real RF channels to complex analytic baseband.
///
/// `rf` has shape `(channels, samples)` and the result has exactly that shape.
/// The Hilbert/FFT method assumes each input line is one finite uniformly
/// sampled record; it does not silently pad, truncate, or decimate data.
///
/// # Errors
///
/// Returns [`KwaversError::InvalidInput`] for an empty/non-finite RF matrix,
/// an invalid configuration, an address-space overflow, or allocation failure.
pub fn demodulate_rf_to_iq(
    rf: ArrayView2<'_, f64>,
    config: IqDemodulationConfig,
) -> KwaversResult<Array2<Complex64>> {
    let [channels, samples] = rf.shape();
    if channels == 0 || samples == 0 {
        return Err(KwaversError::InvalidInput(format!(
            "I/Q demodulation requires channels > 0 and samples > 0; got ({channels}, {samples})"
        )));
    }
    if !rf.iter().all(|sample| sample.is_finite()) {
        return Err(KwaversError::InvalidInput(
            "I/Q demodulation requires finite RF samples".to_owned(),
        ));
    }
    let value_count = channels.checked_mul(samples).ok_or_else(|| {
        KwaversError::InvalidInput("I/Q demodulation output shape overflows usize".to_owned())
    })?;
    let mut output = Vec::new();
    output.try_reserve_exact(value_count).map_err(|_| {
        KwaversError::InvalidInput("I/Q demodulation output allocation failed".to_owned())
    })?;
    for channel in 0..channels {
        let mut signal = Vec::new();
        signal.try_reserve_exact(samples).map_err(|_| {
            KwaversError::InvalidInput("I/Q demodulation channel allocation failed".to_owned())
        })?;
        signal.extend((0..samples).map(|sample| rf[[channel, sample]]));
        output.extend(analytic_baseband(signal, config)?);
    }
    Array2::from_shape_vec([channels, samples], output).map_err(|error| {
        KwaversError::InvalidInput(format!("I/Q demodulation output shape: {error}"))
    })
}

fn analytic_baseband(
    signal: Vec<f64>,
    config: IqDemodulationConfig,
) -> KwaversResult<Vec<Complex64>> {
    let n = signal.len();
    let mut spectrum = fft_1d_array(&Array1::from_shape_vec([n], signal).map_err(|error| {
        KwaversError::InvalidInput(format!("I/Q demodulation input shape: {error}"))
    })?);
    let half = n / 2;
    if n.is_multiple_of(2) {
        for value in spectrum.iter_mut().take(half).skip(1) {
            *value *= 2.0;
        }
        for value in spectrum.iter_mut().skip(half + 1) {
            *value = Complex64::new(0.0, 0.0);
        }
    } else {
        for value in spectrum.iter_mut().take(half + 1).skip(1) {
            *value *= 2.0;
        }
        for value in spectrum.iter_mut().skip(half + 1) {
            *value = Complex64::new(0.0, 0.0);
        }
    }
    let analytic = ifft_1d_complex(&spectrum);
    let mut baseband = Vec::new();
    baseband.try_reserve_exact(n).map_err(|_| {
        KwaversError::InvalidInput("I/Q demodulation baseband allocation failed".to_owned())
    })?;
    let angular_frequency = -TWO_PI * config.center_frequency_hz;
    let inv_sampling_frequency = 1.0 / config.sampling_frequency_hz;
    baseband.extend(analytic.iter().enumerate().map(|(sample, &value)| {
        let time_s = sample as f64 * inv_sampling_frequency;
        value * Complex64::new(0.0, angular_frequency * time_s).exp()
    }));
    Ok(baseband)
}

#[cfg(test)]
mod tests {
    use super::{demodulate_rf_to_iq, IqDemodulationConfig};
    use eunomia::Complex64;
    use leto::Array2;

    #[test]
    fn bin_centered_tone_demodulates_to_unit_dc() {
        let sampling_frequency_hz = 1024.0;
        let center_frequency_hz = 64.0;
        let samples = 1024_usize;
        let rf = Array2::from_shape_vec(
            [1, samples],
            (0..samples)
                .map(|sample| {
                    (std::f64::consts::TAU * center_frequency_hz * sample as f64
                        / sampling_frequency_hz)
                        .cos()
                })
                .collect(),
        )
        .unwrap();
        let iq = demodulate_rf_to_iq(
            rf.view(),
            IqDemodulationConfig::new(sampling_frequency_hz, center_frequency_hz).unwrap(),
        )
        .unwrap();
        // A radix-2 FFT performs fewer than 2N log₂N complex butterflies;
        // γ₁₃₁₀₇₂ covers those operations plus Hilbert scaling and phasor
        // rotation for N = 1024.
        let roundoff = 131_072.0 * f64::EPSILON;
        let bound = roundoff / (1.0 - roundoff);
        for &sample in iq.iter() {
            assert!((sample - Complex64::new(1.0, 0.0)).norm() <= bound);
        }
    }

    #[test]
    fn rejects_nonfinite_rf_and_invalid_frequency() {
        let rf = Array2::from_shape_vec([1, 2], vec![1.0, f64::NAN]).unwrap();
        let config = IqDemodulationConfig::new(4.0, 1.0).unwrap();
        let nonfinite_error = demodulate_rf_to_iq(rf.view(), config).unwrap_err();
        assert_eq!(
            nonfinite_error.to_string(),
            "Invalid input: I/Q demodulation requires finite RF samples"
        );
        let sampling_error = IqDemodulationConfig::new(0.0, 1.0).unwrap_err();
        assert_eq!(
            sampling_error.to_string(),
            "Invalid input: I/Q demodulation: sampling_frequency_hz must be finite and > 0"
        );
        let nyquist_error = IqDemodulationConfig::new(2.0, 1.0).unwrap_err();
        assert_eq!(
            nyquist_error.to_string(),
            "Invalid input: I/Q demodulation: center_frequency_hz must be below Nyquist"
        );
    }
}
