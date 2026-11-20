use std::f64::consts::PI;

/// Doppler frequency shift for a moving reflector.
///
/// ```text
/// Δf = 2·f₀·v·cos(θ) / c   `Hz`
/// ```
///
/// Positive for motion towards the transducer (θ < π/2).
///
/// # Arguments
/// * `v_m_s` – reflector speed [m/s]
/// * `theta_rad` – angle between flow direction and beam axis `rad`
/// * `f0_hz` – transmit centre frequency `Hz`
/// * `c` – sound speed [m/s]
///
/// # Reference
/// Szabo (2014) *Diagnostic Ultrasound Imaging*, §11.2.
#[must_use]
#[inline]
pub fn doppler_frequency_shift(v_m_s: f64, theta_rad: f64, f0_hz: f64, c: f64) -> f64 {
    2.0 * f0_hz * v_m_s * theta_rad.cos() / c
}

/// Deterministic Doppler spectrum and Kasai estimator payload.
#[derive(Debug, Clone, PartialEq)]
pub struct DopplerSpectrum {
    /// Slow-time sample positions `s`.
    pub slow_time_s: Vec<f64>,
    /// In-phase IQ component.
    pub iq_real: Vec<f64>,
    /// Quadrature IQ component.
    pub iq_imag: Vec<f64>,
    /// Shifted velocity axis [m/s].
    pub velocity_m_s: Vec<f64>,
    /// Shifted Doppler-spectrum power.
    pub power: Vec<f64>,
    /// True Doppler shift `Hz`.
    pub doppler_shift_hz: f64,
    /// Kasai-estimated Doppler shift `Hz`.
    pub estimated_shift_hz: f64,
    /// Kasai-estimated velocity [m/s].
    pub estimated_velocity_m_s: f64,
    /// Pulsed-wave Nyquist velocity [m/s].
    pub nyquist_velocity_m_s: f64,
}

/// Parameters for deterministic contrast-agent Doppler spectrum synthesis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContrastAgentDopplerConfig {
    /// Slow-time ensemble length.
    pub n_ensemble: usize,
    /// FFT zero-padding multiplier.
    pub fft_multiplier: usize,
    /// Pulse repetition frequency `Hz`.
    pub prf_hz: f64,
    /// Axial flow speed used for the Doppler shift [m/s].
    pub velocity_m_s: f64,
    /// Beam-to-flow angle `rad`.
    pub theta_rad: f64,
    /// Transmit center frequency `Hz`.
    pub f0_hz: f64,
    /// Speed of sound [m/s].
    pub sound_speed_m_s: f64,
    /// Bubble-scattering IQ amplitude.
    pub amplitude: f64,
}

/// Compute the Chapter 5 contrast-agent Doppler IQ series and finite-tone spectrum.
///
/// The IQ trace is `A exp(i 2π f_D n / PRF)`, where `A` is the bubble-scattering
/// amplitude supplied by the caller and `f_D = 2 f0 v cos(theta) / c`. The
/// spectrum is the exact finite-length DFT of that tone, evaluated at
/// `fft_multiplier * n_ensemble` bins and returned in fft-shifted order.
///
/// # Errors
///
/// Returns an error when the ensemble is shorter than two samples, the FFT
/// multiplier is zero, or any physical scalar is non-finite or non-positive
/// where positivity is required.
pub fn contrast_agent_doppler_spectrum(
    config: ContrastAgentDopplerConfig,
) -> Result<DopplerSpectrum, String> {
    if config.n_ensemble < 2 {
        return Err("n_ensemble must be at least two".to_owned());
    }
    if config.fft_multiplier == 0 {
        return Err("fft_multiplier must be greater than zero".to_owned());
    }
    if !(config.prf_hz.is_finite() && config.prf_hz > 0.0) {
        return Err("prf_hz must be finite and greater than zero".to_owned());
    }
    if !(config.f0_hz.is_finite() && config.f0_hz > 0.0) {
        return Err("f0_hz must be finite and greater than zero".to_owned());
    }
    if !(config.sound_speed_m_s.is_finite() && config.sound_speed_m_s > 0.0) {
        return Err("sound_speed_m_s must be finite and greater than zero".to_owned());
    }
    if !config.velocity_m_s.is_finite()
        || !config.theta_rad.is_finite()
        || !config.amplitude.is_finite()
    {
        return Err("velocity_m_s, theta_rad, and amplitude must be finite".to_owned());
    }
    let cos_theta = config.theta_rad.cos();
    if cos_theta.abs() < 1.0e-12 {
        return Err(
            "cos(theta_rad) must be nonzero for scalar Doppler velocity mapping".to_owned(),
        );
    }

    let n_fft = config
        .n_ensemble
        .checked_mul(config.fft_multiplier)
        .ok_or_else(|| "fft length overflow".to_owned())?;
    let doppler_shift_hz = doppler_frequency_shift(
        config.velocity_m_s,
        config.theta_rad,
        config.f0_hz,
        config.sound_speed_m_s,
    );
    let sample_period_s = 1.0 / config.prf_hz;
    let angular_step = 2.0 * PI * doppler_shift_hz * sample_period_s;
    let mut slow_time_s = Vec::with_capacity(config.n_ensemble);
    let mut iq_real = Vec::with_capacity(config.n_ensemble);
    let mut iq_imag = Vec::with_capacity(config.n_ensemble);
    for sample in 0..config.n_ensemble {
        let phase = angular_step * sample as f64;
        slow_time_s.push(sample as f64 * sample_period_s);
        iq_real.push(config.amplitude * phase.cos());
        iq_imag.push(config.amplitude * phase.sin());
    }

    let mut velocity_axis = Vec::with_capacity(n_fft);
    let mut power = Vec::with_capacity(n_fft);
    let half = n_fft / 2;
    for shifted_bin in 0..n_fft {
        let signed_bin = shifted_bin as isize - half as isize;
        let frequency_hz = signed_bin as f64 * config.prf_hz / n_fft as f64;
        let bin_angular_step = 2.0 * PI * frequency_hz * sample_period_s;
        let magnitude = finite_tone_dft_magnitude(
            config.amplitude,
            angular_step - bin_angular_step,
            config.n_ensemble,
        );
        velocity_axis
            .push(config.sound_speed_m_s * frequency_hz / (2.0 * config.f0_hz * cos_theta));
        power.push(magnitude * magnitude);
    }

    let r1_phase = angular_step;
    let estimated_shift_hz = r1_phase / (2.0 * PI * sample_period_s);
    let estimated_velocity_m_s =
        config.sound_speed_m_s * estimated_shift_hz / (2.0 * config.f0_hz * cos_theta);
    let nyquist_velocity_m_s =
        config.sound_speed_m_s * config.prf_hz / (4.0 * config.f0_hz * cos_theta);

    Ok(DopplerSpectrum {
        slow_time_s,
        iq_real,
        iq_imag,
        velocity_m_s: velocity_axis,
        power,
        doppler_shift_hz,
        estimated_shift_hz,
        estimated_velocity_m_s,
        nyquist_velocity_m_s,
    })
}

fn finite_tone_dft_magnitude(amplitude: f64, phase_step: f64, n_samples: usize) -> f64 {
    let denominator = (0.5 * phase_step).sin();
    if denominator.abs() < 1.0e-12 {
        return amplitude.abs() * n_samples as f64;
    }
    amplitude.abs() * ((0.5 * n_samples as f64 * phase_step).sin() / denominator).abs()
}

#[cfg(test)]
mod tests;
