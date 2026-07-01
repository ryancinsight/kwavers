//! Hann-windowed single-sided power spectral density for emission analysis.
//!
//! The rectangular-window DFT of a strongly harmonic emission leaks energy from
//! the dominant harmonic lines into every other bin, which would corrupt the
//! broadband (inertial) band of a passive-cavitation-dose decomposition. A Hann
//! window suppresses that spectral leakage (>30 dB sidelobe attenuation) so the
//! inter-line floor reflects genuine inharmonic emission. This is the standard
//! estimator used for passive-cavitation spectroscopy (Gyöngy & Coussios 2010).

use crate::analytical::cavitation::keller_miksis_rk4;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_math::fft::fft_1d_array;
use ndarray::Array1;

pub(super) const MAX_EXACT_F64_INTEGER: usize = 1usize << 53;

/// Single-sided Hann-windowed power spectral density of an emission series.
///
/// ```text
///   w[j]  = ½·(1 − cos(2πj/(N−1)))                 (Hann window)
///   X[k]  = Σ_j (x[j] − x̄)·w[j]·exp(−i2πkj/N)
///   S[k]  = c_k · |X[k]|² · Δt / Σ_j w[j]²          c_k = 1 (DC/Nyquist) else 2
/// ```
/// The `Σ w²` normalisation makes `S` a true power-preserving PSD comparable to
/// the rectangular-window [`super::super::bubble_power_spectrum`]; the band
/// energies that [`super::decompose_emission_spectrum`] integrates from it are
/// therefore leakage-suppressed.
///
/// # Arguments
/// * `signal` – emission time series (e.g. from [`super::bubble_acoustic_emission_pressure`])
/// * `dt_s`   – uniform sample interval [s]
/// * `n_fft`  – DFT length (≥ `signal.len()`; zero-padded)
///
/// Returns `(f_arr [Hz], psd)` over non-negative frequencies. Returns a pair of
/// empty vectors if `n_fft < 2` or `dt_s ≤ 0`.
///
/// # Note
/// Direct O(N²) DFT — exact, slow for large N. Callers decimate to a few
/// thousand samples before invoking.
#[must_use]
pub fn hann_windowed_power_spectrum(
    signal: &[f64],
    dt_s: f64,
    n_fft: usize,
) -> (Vec<f64>, Vec<f64>) {
    if n_fft < 2 || !(dt_s.is_finite() && dt_s > 0.0) {
        return (Vec::new(), Vec::new());
    }
    let n = n_fft;
    let n_f = n as f64;

    // Windowed, mean-removed, zero-padded record.
    let mut x = vec![0.0_f64; n];
    let count = signal.len().min(n);
    let mean: f64 = signal.iter().take(count).sum::<f64>() / count.max(1) as f64;
    let mut w_power = 0.0_f64; // Σ w² over the *populated* samples
    for j in 0..count {
        // Hann window across the populated record length `count`.
        let w = if count > 1 {
            0.5 * (1.0 - (TWO_PI * j as f64 / (count as f64 - 1.0)).cos())
        } else {
            1.0
        };
        x[j] = (signal[j] - mean) * w;
        w_power += w * w;
    }
    if w_power <= 0.0 {
        return (vec![0.0; n / 2 + 1], vec![0.0; n / 2 + 1]);
    }

    let n_pos = n / 2 + 1;
    let mut f_arr = vec![0.0_f64; n_pos];
    let mut psd = vec![0.0_f64; n_pos];
    for k in 0..n_pos {
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        let phase_step = TWO_PI * k as f64 / n_f;
        for (j, &xj) in x.iter().enumerate() {
            let phi = phase_step * j as f64;
            re += xj * phi.cos();
            im -= xj * phi.sin();
        }
        let mag_sq = re * re + im * im;
        let scale = if k == 0 || k == n / 2 { 1.0 } else { 2.0 };
        psd[k] = scale * mag_sq * dt_s / w_power;
        f_arr[k] = k as f64 / (n_f * dt_s);
    }
    (f_arr, psd)
}

/// Passive-cavitation band ratios from a monitoring window.
#[derive(Debug, Clone, PartialEq)]
pub struct PcdBandSignals {
    /// Subharmonic band power divided by fundamental band power.
    pub stable_signal: f64,
    /// Broadband band power divided by fundamental band power.
    pub inertial_signal: f64,
}

/// Normalized Keller-Miksis wall-velocity spectrum for Chapter 7 PCD figures.
#[derive(Debug, Clone, PartialEq)]
pub struct KellerMiksisPcdSpectrum {
    /// Single-sided frequency axis [Hz].
    pub frequency_hz: Vec<f64>,
    /// Normalized PSD in decibels relative to the maximum bin.
    pub normalized_psd_db: Vec<f64>,
    /// Subharmonic/fundamental band ratio.
    pub stable_signal: f64,
    /// Broadband/fundamental band ratio.
    pub inertial_signal: f64,
}

/// Closed-loop PCD controller trace driven by Keller-Miksis band ratios.
#[derive(Debug, Clone, PartialEq)]
pub struct KellerMiksisPcdControllerTrace {
    /// Pulse indices, one-based.
    pub pulse_index: Vec<f64>,
    /// Drive pressure applied at each pulse [kPa].
    pub pressure_kpa: Vec<f64>,
    /// Raw subharmonic/fundamental ratios.
    pub stable_signal: Vec<f64>,
    /// Raw broadband/fundamental ratios.
    pub inertial_signal: Vec<f64>,
    /// Stable signal normalized to its trace maximum for plotting.
    pub stable_signal_normalized: Vec<f64>,
    /// Inertial signal normalized to its trace maximum for plotting.
    pub inertial_signal_normalized: Vec<f64>,
}

/// Compute passive-cavitation stable and inertial band ratios from a signal.
///
/// `stable_signal` integrates the subharmonic band `[0.4 f0, 0.6 f0]` and
/// `inertial_signal` integrates the broadband band `[1.5 f0, 4.5 f0]`; both
/// are normalized by the fundamental band `[0.85 f0, 1.15 f0]`.
///
/// # Errors
///
/// Returns an error if the signal, timestep, or drive frequency is invalid, or
/// if the fundamental band has no usable energy.
pub fn pcd_band_signals(
    signal: &[f64],
    dt_s: f64,
    drive_frequency_hz: f64,
) -> Result<PcdBandSignals, String> {
    let (_freq, _psd, signals) = pcd_spectrum_from_signal(signal, dt_s, drive_frequency_hz)?;
    Ok(signals)
}

/// Compute a normalized PCD spectrum from a Keller-Miksis wall-velocity trace.
///
/// The Rust path owns the Hann window, FFT, band integration, and dB
/// normalization that Chapter 7 previously performed in Python.
///
/// # Errors
///
/// Returns an error if any physical, sampling, or band-power input is invalid.
#[allow(clippy::too_many_arguments)]
pub fn keller_miksis_pcd_spectrum(
    r0_m: f64,
    p_ac_pa: f64,
    drive_frequency_hz: f64,
    n_cycles: usize,
    n_per_cycle: usize,
    discard_cycles: usize,
    p0_pa: f64,
    rho: f64,
    sigma: f64,
    mu: f64,
    kappa: f64,
    vapor_pressure_pa: f64,
    sound_speed_m_s: f64,
) -> Result<KellerMiksisPcdSpectrum, String> {
    let (rdot, dt_s, discard_samples) = keller_miksis_wall_velocity(
        r0_m,
        p_ac_pa,
        drive_frequency_hz,
        n_cycles,
        n_per_cycle,
        discard_cycles,
        p0_pa,
        rho,
        sigma,
        mu,
        kappa,
        vapor_pressure_pa,
        sound_speed_m_s,
    )?;
    let signal = rdot
        .get(discard_samples..)
        .ok_or_else(|| "discard_cycles removes the whole signal".to_owned())?;
    let (frequency_hz, psd, signals) = pcd_spectrum_from_signal(signal, dt_s, drive_frequency_hz)?;
    let max_psd = psd.iter().copied().fold(0.0_f64, f64::max);
    if max_psd <= 0.0 || !max_psd.is_finite() {
        return Err("PCD spectrum has no finite positive power".to_owned());
    }
    let normalized_psd_db = psd
        .iter()
        .map(|&power| 10.0 * ((power / max_psd) + 1.0e-12).log10())
        .collect();

    Ok(KellerMiksisPcdSpectrum {
        frequency_hz,
        normalized_psd_db,
        stable_signal: signals.stable_signal,
        inertial_signal: signals.inertial_signal,
    })
}

/// Compute a Keller-Miksis PCD feedback-controller trace.
///
/// The controller samples one Keller-Miksis monitoring window per pulse,
/// computes SC/IC band ratios, then applies the Chapter 7 asymmetric pressure
/// law: broadband over `inertial_limit` backs off by `gamma_down`; otherwise
/// subharmonic below `stable_target` recruits by `gamma_up`.
///
/// # Errors
///
/// Returns an error if any physical, sampling, controller, or band-power input
/// is invalid.
#[allow(clippy::too_many_arguments)]
pub fn keller_miksis_pcd_controller_trace(
    r0_m: f64,
    drive_frequency_hz: f64,
    n_pulses: usize,
    initial_pressure_pa: f64,
    n_cycles_per_pulse: usize,
    n_per_cycle: usize,
    discard_cycles: usize,
    stable_target: f64,
    inertial_limit: f64,
    gamma_up: f64,
    gamma_down: f64,
    p_min_pa: f64,
    p_max_pa: f64,
    p0_pa: f64,
    rho: f64,
    sigma: f64,
    mu: f64,
    kappa: f64,
    vapor_pressure_pa: f64,
    sound_speed_m_s: f64,
) -> Result<KellerMiksisPcdControllerTrace, String> {
    validate_controller_inputs(
        n_pulses,
        initial_pressure_pa,
        stable_target,
        inertial_limit,
        gamma_up,
        gamma_down,
        p_min_pa,
        p_max_pa,
    )?;

    let mut pressure_pa = initial_pressure_pa.clamp(p_min_pa, p_max_pa);
    let mut pulse_index = Vec::with_capacity(n_pulses);
    let mut pressure_kpa = Vec::with_capacity(n_pulses);
    let mut stable_signal = Vec::with_capacity(n_pulses);
    let mut inertial_signal = Vec::with_capacity(n_pulses);

    for pulse in 0..n_pulses {
        let spectrum = keller_miksis_pcd_spectrum(
            r0_m,
            pressure_pa,
            drive_frequency_hz,
            n_cycles_per_pulse,
            n_per_cycle,
            discard_cycles,
            p0_pa,
            rho,
            sigma,
            mu,
            kappa,
            vapor_pressure_pa,
            sound_speed_m_s,
        )?;

        pulse_index.push((pulse + 1) as f64);
        pressure_kpa.push(pressure_pa / 1.0e3);
        stable_signal.push(spectrum.stable_signal);
        inertial_signal.push(spectrum.inertial_signal);

        if pulse + 1 < n_pulses {
            pressure_pa = if spectrum.inertial_signal > inertial_limit {
                pressure_pa * gamma_down
            } else if spectrum.stable_signal < stable_target {
                pressure_pa * gamma_up
            } else {
                pressure_pa
            }
            .clamp(p_min_pa, p_max_pa);
        }
    }

    let stable_signal_normalized = normalized_trace(&stable_signal);
    let inertial_signal_normalized = normalized_trace(&inertial_signal);

    Ok(KellerMiksisPcdControllerTrace {
        pulse_index,
        pressure_kpa,
        stable_signal,
        inertial_signal,
        stable_signal_normalized,
        inertial_signal_normalized,
    })
}

pub(super) fn hann_power_spectrum_fft(signal: &[f64], dt_s: f64) -> Option<(Vec<f64>, Vec<f64>)> {
    let n = signal.len();
    if !((2..=MAX_EXACT_F64_INTEGER).contains(&n) && dt_s.is_finite() && dt_s > 0.0) {
        return None;
    }
    let n_f = n as f64;
    let mean = signal.iter().sum::<f64>() / n_f;
    let mut windowed = Vec::with_capacity(n);
    for (j, &sample) in signal.iter().enumerate() {
        let j_f = j as f64;
        let w = 0.5 * (1.0 - (TWO_PI * j_f / (n_f - 1.0)).cos());
        windowed.push((sample - mean) * w);
    }
    let spectrum = fft_1d_array(&Array1::from_vec(windowed));
    let n_pos = n / 2 + 1;
    let mut freqs = Vec::with_capacity(n_pos);
    let mut psd = Vec::with_capacity(n_pos);
    for k in 0..n_pos {
        freqs.push(k as f64 / (n_f * dt_s));
        psd.push(spectrum[k].norm_sqr());
    }
    Some((freqs, psd))
}

fn pcd_spectrum_from_signal(
    signal: &[f64],
    dt_s: f64,
    drive_frequency_hz: f64,
) -> Result<(Vec<f64>, Vec<f64>, PcdBandSignals), String> {
    if !(drive_frequency_hz.is_finite() && drive_frequency_hz > 0.0) {
        return Err("drive_frequency_hz must be positive and finite".to_owned());
    }
    if !signal.iter().all(|value| value.is_finite()) {
        return Err("signal must contain only finite samples".to_owned());
    }
    let (freq, psd) = hann_power_spectrum_fft(signal, dt_s)
        .ok_or_else(|| "signal length and dt_s must define a valid FFT window".to_owned())?;

    let fundamental = band_power(
        &freq,
        &psd,
        0.85 * drive_frequency_hz,
        1.15 * drive_frequency_hz,
    );
    if fundamental <= 1.0e-30 || !fundamental.is_finite() {
        return Err("fundamental band power is not finite and positive".to_owned());
    }
    let subharmonic = band_power(
        &freq,
        &psd,
        0.40 * drive_frequency_hz,
        0.60 * drive_frequency_hz,
    );
    let broadband = band_power(
        &freq,
        &psd,
        1.50 * drive_frequency_hz,
        4.50 * drive_frequency_hz,
    );
    let signals = PcdBandSignals {
        stable_signal: subharmonic / fundamental,
        inertial_signal: broadband / fundamental,
    };

    Ok((freq, psd, signals))
}

#[allow(clippy::too_many_arguments)]
fn keller_miksis_wall_velocity(
    r0_m: f64,
    p_ac_pa: f64,
    drive_frequency_hz: f64,
    n_cycles: usize,
    n_per_cycle: usize,
    discard_cycles: usize,
    p0_pa: f64,
    rho: f64,
    sigma: f64,
    mu: f64,
    kappa: f64,
    vapor_pressure_pa: f64,
    sound_speed_m_s: f64,
) -> Result<(Vec<f64>, f64, usize), String> {
    validate_keller_miksis_inputs(
        r0_m,
        p_ac_pa,
        drive_frequency_hz,
        n_cycles,
        n_per_cycle,
        discard_cycles,
        p0_pa,
        rho,
        sigma,
        mu,
        kappa,
        vapor_pressure_pa,
        sound_speed_m_s,
    )?;
    let n_steps = n_cycles
        .checked_mul(n_per_cycle)
        .ok_or_else(|| "n_cycles * n_per_cycle overflows".to_owned())?;
    let t_end_s = n_cycles as f64 / drive_frequency_hz;
    let dt_s = t_end_s / n_steps as f64;
    let time: Vec<f64> = (0..=n_steps).map(|idx| idx as f64 * dt_s).collect();
    let (_radius, rdot) = keller_miksis_rk4(
        r0_m,
        0.0,
        p_ac_pa,
        drive_frequency_hz,
        &time,
        p0_pa,
        rho,
        sigma,
        mu,
        kappa,
        vapor_pressure_pa,
        sound_speed_m_s,
    );
    Ok((rdot, dt_s, discard_cycles * n_per_cycle))
}

fn band_power(freq: &[f64], psd: &[f64], f_min_hz: f64, f_max_hz: f64) -> f64 {
    freq.iter()
        .zip(psd)
        .filter_map(|(&f, &p)| {
            if f > f_min_hz && f < f_max_hz && p.is_finite() {
                Some(p.max(0.0))
            } else {
                None
            }
        })
        .sum()
}

fn normalized_trace(values: &[f64]) -> Vec<f64> {
    let max_value = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(0.0_f64, f64::max);
    let denom = max_value + 1.0e-30;
    values.iter().map(|&value| value / denom).collect()
}

#[allow(clippy::too_many_arguments)]
fn validate_keller_miksis_inputs(
    r0_m: f64,
    p_ac_pa: f64,
    drive_frequency_hz: f64,
    n_cycles: usize,
    n_per_cycle: usize,
    discard_cycles: usize,
    p0_pa: f64,
    rho: f64,
    sigma: f64,
    mu: f64,
    kappa: f64,
    vapor_pressure_pa: f64,
    sound_speed_m_s: f64,
) -> Result<(), String> {
    let scalars = [
        r0_m,
        p_ac_pa,
        drive_frequency_hz,
        p0_pa,
        rho,
        sigma,
        mu,
        kappa,
        vapor_pressure_pa,
        sound_speed_m_s,
    ];
    if !scalars.iter().all(|value| value.is_finite()) {
        return Err("Keller-Miksis PCD inputs must be finite".to_owned());
    }
    if r0_m <= 0.0 || drive_frequency_hz <= 0.0 || p0_pa <= 0.0 || rho <= 0.0 {
        return Err("r0_m, drive_frequency_hz, p0_pa, and rho must be positive".to_owned());
    }
    if sigma < 0.0 || mu < 0.0 || kappa <= 0.0 || sound_speed_m_s <= 0.0 {
        return Err("surface tension, viscosity, kappa, and sound speed are invalid".to_owned());
    }
    if n_cycles < 2 || n_per_cycle < 8 {
        return Err("n_cycles must be >= 2 and n_per_cycle must be >= 8".to_owned());
    }
    if discard_cycles >= n_cycles {
        return Err("discard_cycles must be less than n_cycles".to_owned());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_controller_inputs(
    n_pulses: usize,
    initial_pressure_pa: f64,
    stable_target: f64,
    inertial_limit: f64,
    gamma_up: f64,
    gamma_down: f64,
    p_min_pa: f64,
    p_max_pa: f64,
) -> Result<(), String> {
    let scalars = [
        initial_pressure_pa,
        stable_target,
        inertial_limit,
        gamma_up,
        gamma_down,
        p_min_pa,
        p_max_pa,
    ];
    if !scalars.iter().all(|value| value.is_finite()) {
        return Err("PCD controller inputs must be finite".to_owned());
    }
    if n_pulses == 0 {
        return Err("n_pulses must be positive".to_owned());
    }
    if stable_target < 0.0 || inertial_limit < 0.0 {
        return Err("stable_target and inertial_limit must be non-negative".to_owned());
    }
    if gamma_up < 1.0 || !(0.0..=1.0).contains(&gamma_down) {
        return Err("gamma_up must be >= 1 and gamma_down must be in [0, 1]".to_owned());
    }
    if p_min_pa <= 0.0 || p_max_pa < p_min_pa {
        return Err("pressure bounds must be positive with p_max_pa >= p_min_pa".to_owned());
    }
    Ok(())
}
