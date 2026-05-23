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

impl PstdTemporalBinConfig {
    /// Validate finite-window bin parameters independent of a specific mode.
    ///
    /// # Errors
    /// Returns an error when timing or source-gain parameters are invalid.
    pub fn validate(self) -> KwaversResult<()> {
        validate_temporal_bin(0.0, self)
    }
}

/// Frequency-independent PSTD acquisition timing/source parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PstdTemporalTransferConfig {
    /// Scalar pressure-source amplitude [Pa].
    pub source_amplitude_pa: f64,
    /// Number of continuous-wave cycles simulated for each frequency.
    pub cycles_per_frequency: usize,
    /// Number of trailing cycles used for the complex frequency bin.
    pub frequency_bin_cycles: usize,
}

impl PstdTemporalTransferConfig {
    /// Validate acquisition transfer parameters.
    ///
    /// # Errors
    /// Returns an error when cycle counts or source amplitude are invalid.
    pub fn validate(self) -> KwaversResult<()> {
        validate_temporal_transfer(self)
    }

    /// Build the finite-window modal-bin config for one drive frequency.
    ///
    /// The source gain matches the additive PSTD pressure-source update used
    /// by the clinical dataset generator: `2 c0 Δt A / Δx`.
    ///
    /// # Errors
    /// Returns an error when frequency, sampling, spacing, or transfer
    /// parameters violate the PSTD acquisition contract.
    pub fn bin_config(
        self,
        frequency_hz: f64,
        time_step_s: f64,
        spacing_m: f64,
        reference_sound_speed_m_s: f64,
    ) -> KwaversResult<PstdTemporalBinConfig> {
        validate_frequency_sampling(frequency_hz, time_step_s)?;
        validate_temporal_transfer(self)?;
        if !spacing_m.is_finite() || spacing_m <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "PSTD temporal spacing_m must be positive and finite, got {spacing_m}"
            )));
        }
        if !reference_sound_speed_m_s.is_finite() || reference_sound_speed_m_s <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "PSTD temporal reference sound speed must be positive and finite, got {reference_sound_speed_m_s}"
            )));
        }
        let total_steps =
            pstd_time_steps_for_cycles(frequency_hz, time_step_s, self.cycles_per_frequency)?;
        let bin_start_step = pstd_frequency_bin_start_step(
            frequency_hz,
            time_step_s,
            self.cycles_per_frequency,
            self.frequency_bin_cycles,
        )?;
        let source_gain =
            2.0 * reference_sound_speed_m_s * time_step_s * self.source_amplitude_pa / spacing_m;
        Ok(PstdTemporalBinConfig {
            frequency_hz,
            time_step_s,
            total_steps,
            bin_start_step,
            source_gain,
        })
    }
}

/// PSTD time-step count for a continuous-wave drive over `cycles` cycles.
///
/// # Errors
/// Returns an error when sampling parameters are invalid or the count is not
/// representable.
pub fn pstd_time_steps_for_cycles(
    frequency_hz: f64,
    time_step_s: f64,
    cycles: usize,
) -> KwaversResult<usize> {
    validate_frequency_sampling(frequency_hz, time_step_s)?;
    if cycles == 0 {
        return Err(KwaversError::InvalidInput(
            "PSTD temporal cycles must be positive".to_owned(),
        ));
    }
    let raw = (cycles as f64 / (frequency_hz * time_step_s)).ceil();
    if !raw.is_finite() || raw > usize::MAX as f64 {
        return Err(KwaversError::InvalidInput(format!(
            "PSTD temporal time-step count is not representable for frequency {frequency_hz}"
        )));
    }
    Ok((raw as usize).max(2))
}

/// First sample of the trailing frequency bin.
///
/// # Errors
/// Returns an error when frequency or cycle parameters are invalid.
pub fn pstd_frequency_bin_start_step(
    frequency_hz: f64,
    time_step_s: f64,
    cycles_per_frequency: usize,
    frequency_bin_cycles: usize,
) -> KwaversResult<usize> {
    let transfer = PstdTemporalTransferConfig {
        source_amplitude_pa: 0.0,
        cycles_per_frequency,
        frequency_bin_cycles,
    };
    validate_temporal_transfer(transfer)?;
    let total_steps = pstd_time_steps_for_cycles(frequency_hz, time_step_s, cycles_per_frequency)?;
    let bin_steps = pstd_time_steps_for_cycles(frequency_hz, time_step_s, frequency_bin_cycles)?;
    Ok(total_steps.saturating_sub(bin_steps))
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

fn validate_temporal_transfer(config: PstdTemporalTransferConfig) -> KwaversResult<()> {
    if config.cycles_per_frequency == 0 {
        return Err(KwaversError::InvalidInput(
            "PSTD temporal cycles_per_frequency must be positive".to_owned(),
        ));
    }
    if config.frequency_bin_cycles == 0 || config.frequency_bin_cycles > config.cycles_per_frequency
    {
        return Err(KwaversError::InvalidInput(format!(
            "PSTD temporal frequency_bin_cycles must be in 1..={}, got {}",
            config.cycles_per_frequency, config.frequency_bin_cycles
        )));
    }
    if !config.source_amplitude_pa.is_finite() {
        return Err(KwaversError::InvalidInput(format!(
            "PSTD temporal source_amplitude_pa must be finite, got {}",
            config.source_amplitude_pa
        )));
    }
    Ok(())
}

fn validate_frequency_sampling(frequency_hz: f64, time_step_s: f64) -> KwaversResult<()> {
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "PSTD temporal frequency_hz must be positive and finite, got {frequency_hz}"
        )));
    }
    if !time_step_s.is_finite() || time_step_s <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "PSTD temporal time_step_s must be positive and finite, got {time_step_s}"
        )));
    }
    Ok(())
}
