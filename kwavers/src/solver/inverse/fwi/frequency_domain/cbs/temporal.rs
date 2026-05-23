//! PSTD temporal source and frequency-bin transfer identities.
//!
//! These functions encode the homogeneous leapfrog/k-space time discretization
//! used by the PSTD acquisition path.  The frequency-domain CBS operator uses
//! the same modal symbols, while clinical diagnostics use the finite-window
//! source/bin transfer to compare against time-domain acquisition data.

use crate::core::error::{KwaversError, KwaversResult};
use num_complex::Complex64;
use std::f64::consts::TAU;

/// Finite-window PSTD source/bin transfer configuration for one frequency.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PstdTemporalBinConfig {
    /// Continuous-wave drive frequency [Hz].
    pub frequency_hz: f64,
    /// PSTD time step [s].
    pub time_step_s: f64,
    /// Total simulated PSTD steps.
    pub total_steps: usize,
    /// First sample included in the frequency-domain bin.
    pub bin_start_step: usize,
    /// Scalar multiplier applied to the discrete source-signal difference.
    pub source_gain: f64,
}

/// PSTD pressure-source k-space correction `cos(c0 Δt |k| / 2)`.
#[must_use]
pub fn pstd_source_kappa_symbol(
    grid_wavenumber_rad_per_m: f64,
    time_step_s: f64,
    reference_sound_speed_m_s: f64,
) -> f64 {
    (0.5 * reference_sound_speed_m_s * time_step_s * grid_wavenumber_rad_per_m).cos()
}

/// PSTD leapfrog modal `θ² = 4 sin²(c0 Δt |k| / 2)`.
#[must_use]
pub fn pstd_modal_theta_squared(
    grid_wavenumber_rad_per_m: f64,
    time_step_s: f64,
    reference_sound_speed_m_s: f64,
) -> f64 {
    4.0 * (0.5 * reference_sound_speed_m_s * time_step_s * grid_wavenumber_rad_per_m)
        .sin()
        .powi(2)
}

/// Frequency-domain PSTD denominator
/// `[4 sin²(ω Δt / 2) - 4 sin²(c0 |k| Δt / 2)] / (c0 Δt)²`.
#[must_use]
pub fn pstd_leapfrog_symbol(
    reference_wavenumber_rad_per_m: f64,
    grid_wavenumber_rad_per_m: f64,
    time_step_s: f64,
    reference_sound_speed_m_s: f64,
) -> f64 {
    let scale = reference_sound_speed_m_s * time_step_s;
    let temporal = 4.0 * (0.5 * reference_wavenumber_rad_per_m * scale).sin().powi(2);
    let spatial = pstd_modal_theta_squared(
        grid_wavenumber_rad_per_m,
        time_step_s,
        reference_sound_speed_m_s,
    );
    (temporal - spatial) / (scale * scale)
}

/// Exact finite-window frequency bin for one PSTD modal unit source.
///
/// The scalar recurrence is
/// `p[n+1] = (2 - θ²) p[n] - p[n-1] + g (s[n] - s[n-1])`,
/// with zero initial state and `s[n] = sin(2π f n Δt)`.  The returned value is
/// `2 / M * Σ p[n+1] exp(-i 2π f n Δt)` over `n >= bin_start_step`.
///
/// # Errors
/// Returns an error when timing, source-gain, or modal parameters are invalid.
pub fn pstd_modal_frequency_bin_response(
    theta_squared: f64,
    config: PstdTemporalBinConfig,
) -> KwaversResult<Complex64> {
    validate_temporal_bin(theta_squared, config)?;
    let angular_step = TAU * config.frequency_hz * config.time_step_s;
    let mut previous = 0.0;
    let mut current = 0.0;
    let mut previous_signal = 0.0;
    let mut bin = Complex64::new(0.0, 0.0);

    for step in 0..config.total_steps {
        let signal = (angular_step * step as f64).sin();
        let source_scale = config.source_gain * (signal - previous_signal);
        let next = (2.0 - theta_squared).mul_add(current, -previous) + source_scale;
        if step >= config.bin_start_step {
            let phase = -angular_step * step as f64;
            bin += Complex64::new(phase.cos(), phase.sin()) * next;
        }
        previous = current;
        current = next;
        previous_signal = signal;
    }

    Ok(bin * (2.0 / (config.total_steps - config.bin_start_step) as f64))
}

fn validate_temporal_bin(theta_squared: f64, config: PstdTemporalBinConfig) -> KwaversResult<()> {
    if !theta_squared.is_finite() || theta_squared < 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "PSTD modal theta_squared must be finite and nonnegative, got {theta_squared}"
        )));
    }
    if !config.frequency_hz.is_finite() || config.frequency_hz <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "PSTD temporal frequency_hz must be positive and finite, got {}",
            config.frequency_hz
        )));
    }
    if !config.time_step_s.is_finite() || config.time_step_s <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "PSTD temporal time_step_s must be positive and finite, got {}",
            config.time_step_s
        )));
    }
    if config.total_steps == 0 || config.bin_start_step >= config.total_steps {
        return Err(KwaversError::InvalidInput(
            "PSTD temporal bin_start_step must lie inside total_steps".to_owned(),
        ));
    }
    if !config.source_gain.is_finite() {
        return Err(KwaversError::InvalidInput(format!(
            "PSTD temporal source_gain must be finite, got {}",
            config.source_gain
        )));
    }
    Ok(())
}
