/// Compute the pressure field of a 1-D standing wave.
///
/// ```text
/// p(x, t) = p₀ · sin(k·x) · cos(ω·t)   `Pa`
/// ```
///
/// # Arguments
/// * `p0` – peak pressure amplitude `Pa`
/// * `k` – wavenumber [rad/m]
/// * `x_arr` – spatial positions `m`
/// * `omega_t` – phase `ω·t` `rad`
#[must_use]
#[inline]
pub fn standing_wave_1d(p0: f64, k: f64, x_arr: &[f64], omega_t: f64) -> Vec<f64> {
    let cos_wt = omega_t.cos();
    x_arr.iter().map(|&x| p0 * (k * x).sin() * cos_wt).collect()
}

/// Compute the pressure field of a 1-D plane wave.
///
/// ```text
/// p(x, t) = A · cos(k·x − ω·t)   `Pa`
/// ```
///
/// # Arguments
/// * `amplitude` – peak amplitude `Pa`
/// * `k` – wavenumber [rad/m]
/// * `x_arr` – spatial positions `m`
/// * `omega_t` – `ω·t` `rad`
#[must_use]
#[inline]
pub fn plane_wave_pressure_1d(amplitude: f64, k: f64, x_arr: &[f64], omega_t: f64) -> Vec<f64> {
    x_arr
        .iter()
        .map(|&x| amplitude * (k * x - omega_t).cos())
        .collect()
}

/// Compute pressure and particle velocity for a 1-D progressive plane wave.
///
/// ```text
/// p(x, t) = A · cos(k·x − ω·t)       `Pa`
/// u(x, t) = p(x, t) / (ρ c)          [m/s]
/// ```
///
/// # Errors
/// Returns an error when coordinates or scalar parameters are non-finite, or
/// when `density_kg_m3 <= 0` or `sound_speed_m_s <= 0`.
pub fn plane_wave_pressure_velocity_1d(
    amplitude_pa: f64,
    k: f64,
    x_arr: &[f64],
    omega_t: f64,
    density_kg_m3: f64,
    sound_speed_m_s: f64,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    if !(amplitude_pa.is_finite()
        && k.is_finite()
        && omega_t.is_finite()
        && density_kg_m3.is_finite()
        && sound_speed_m_s.is_finite())
    {
        return Err("plane-wave parameters must be finite".to_owned());
    }
    if density_kg_m3 <= 0.0 {
        return Err("plane-wave density_kg_m3 must be positive".to_owned());
    }
    if sound_speed_m_s <= 0.0 {
        return Err("plane-wave sound_speed_m_s must be positive".to_owned());
    }
    if !x_arr.iter().all(|value| value.is_finite()) {
        return Err("plane-wave coordinates must be finite".to_owned());
    }

    let impedance = density_kg_m3 * sound_speed_m_s;
    let pressure = plane_wave_pressure_1d(amplitude_pa, k, x_arr, omega_t);
    let velocity = pressure.iter().map(|&p| p / impedance).collect();
    Ok((pressure, velocity))
}

/// Gaussian-modulated cosine pulse over a 1-D coordinate axis.
///
/// ```text
/// g(x) = A exp[-(x - x0)^2 / (2 sigma^2)] cos[2 pi (x - x0) / lambda]
/// ```
///
/// # Arguments
/// * `x_arr` – spatial positions `m`
/// * `center_m` – pulse center `x0` `m`
/// * `sigma_m` – Gaussian standard deviation `m`
/// * `wavelength_m` – carrier wavelength `m`
/// * `amplitude_pa` – pressure amplitude `Pa`
///
/// # Errors
/// Returns an error when any input is non-finite, `sigma_m <= 0`, or
/// `wavelength_m <= 0`.
pub fn gaussian_modulated_pulse_1d(
    x_arr: &[f64],
    center_m: f64,
    sigma_m: f64,
    wavelength_m: f64,
    amplitude_pa: f64,
) -> Result<Vec<f64>, String> {
    use std::f64::consts::PI;
    if !(center_m.is_finite()
        && sigma_m.is_finite()
        && wavelength_m.is_finite()
        && amplitude_pa.is_finite())
    {
        return Err("Gaussian pulse parameters must be finite".to_owned());
    }
    if sigma_m <= 0.0 {
        return Err("Gaussian pulse sigma_m must be positive".to_owned());
    }
    if wavelength_m <= 0.0 {
        return Err("Gaussian pulse wavelength_m must be positive".to_owned());
    }
    if !x_arr.iter().all(|value| value.is_finite()) {
        return Err("Gaussian pulse coordinates must be finite".to_owned());
    }
    let inv_two_sigma2 = 1.0 / (2.0 * sigma_m * sigma_m);
    let k = 2.0 * PI / wavelength_m;
    Ok(x_arr
        .iter()
        .map(|&x| {
            let dx = x - center_m;
            amplitude_pa * (-(dx * dx) * inv_two_sigma2).exp() * (k * dx).cos()
        })
        .collect())
}

/// d'Alembert solution for a zero-initial-velocity 1-D pulse on a sorted axis.
///
/// For `p(x, 0) = g(x)` and `u(x, 0) = 0`, the wave equation solution is
/// `p(x, t) = 0.5 [g(x - ct) + g(x + ct)]`. Values shifted outside the sampled
/// coordinate range are zero, matching the finite plotted window used by the
/// book figures.
///
/// # Arguments
/// * `x_arr` – strictly increasing spatial positions `m`
/// * `initial_pressure` – `g(x)` sampled at `x_arr` `Pa`
/// * `shift_m` – propagation distance `c t` `m`
///
/// # Errors
/// Returns an error when the coordinate and pressure lengths differ, the axis
/// has fewer than two samples, values are non-finite, or the axis is not
/// strictly increasing.
pub fn dalembert_split_solution_1d(
    x_arr: &[f64],
    initial_pressure: &[f64],
    shift_m: f64,
) -> Result<Vec<f64>, String> {
    if x_arr.len() != initial_pressure.len() {
        return Err(
            "d'Alembert coordinate and pressure arrays must have matching lengths".to_owned(),
        );
    }
    if x_arr.len() < 2 {
        return Err("d'Alembert coordinate axis must contain at least two samples".to_owned());
    }
    if !shift_m.is_finite() {
        return Err("d'Alembert shift_m must be finite".to_owned());
    }
    if !x_arr.iter().all(|value| value.is_finite()) {
        return Err("d'Alembert coordinates must be finite".to_owned());
    }
    if !initial_pressure.iter().all(|value| value.is_finite()) {
        return Err("d'Alembert initial pressure samples must be finite".to_owned());
    }
    if !x_arr.windows(2).all(|window| window[1] > window[0]) {
        return Err("d'Alembert coordinate axis must be strictly increasing".to_owned());
    }
    Ok(x_arr
        .iter()
        .map(|&x| {
            0.5 * (interp_linear_sorted(x_arr, initial_pressure, x - shift_m)
                + interp_linear_sorted(x_arr, initial_pressure, x + shift_m))
        })
        .collect())
}

/// Real part of the spherical-wave Green's function (far-field).
///
/// ```text
/// p(r) = A · cos(k·r) / r   `Pa`
/// ```
///
/// Singularity at r = 0 is guarded: returns `f64::INFINITY` there.
///
/// # Reference
/// Pierce (1989) *Acoustics*, §1.6.
#[must_use]
#[inline]
pub fn spherical_wave_pressure(amplitude: f64, k: f64, r_arr: &[f64]) -> Vec<f64> {
    r_arr
        .iter()
        .map(|&r| {
            if r == 0.0 {
                f64::INFINITY
            } else {
                amplitude * (k * r).cos() / r
            }
        })
        .collect()
}

/// Normalized geometric spreading intensity envelopes.
///
/// For a three-dimensional point source, intensity decays as `1 / r^2`.
/// For a two-dimensional cylindrical source, intensity decays as `1 / r`.
/// Both envelopes are normalized by the first radius sample, so the returned
/// arrays start at one.
///
/// # Arguments
/// * `r_arr` - strictly positive finite radial distances `m`
///
/// # Errors
/// Returns an error when `r_arr` is empty or contains a non-finite or
/// non-positive radius.
pub fn geometric_spreading_intensity_envelopes(
    r_arr: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let Some(&r0) = r_arr.first() else {
        return Err("geometric spreading radius axis must not be empty".to_owned());
    };
    if !r_arr.iter().all(|value| value.is_finite() && *value > 0.0) {
        return Err("geometric spreading radii must be finite and positive".to_owned());
    }

    let spherical = r_arr
        .iter()
        .map(|&r| {
            let ratio = r0 / r;
            ratio * ratio
        })
        .collect();
    let cylindrical = r_arr.iter().map(|&r| r0 / r).collect();
    Ok((spherical, cylindrical))
}

fn interp_linear_sorted(x_arr: &[f64], values: &[f64], x: f64) -> f64 {
    let x0 = x_arr[0];
    let x1 = x_arr[x_arr.len() - 1];
    if x < x0 || x > x1 {
        return 0.0;
    }
    if x == x1 {
        return values[values.len() - 1];
    }
    let upper = x_arr.partition_point(|&value| value <= x);
    let lower = upper - 1;
    let span = x_arr[upper] - x_arr[lower];
    let frac = (x - x_arr[lower]) / span;
    values[lower] * (1.0 - frac) + values[upper] * frac
}

/// Plane-wave pressure reflection coefficient at a normal-incidence interface.
///
/// ```text
/// R = (Z₂ − Z₁) / (Z₂ + Z₁)
/// ```
///
/// # Reference
/// Kinsler et al. (2000) *Fundamentals of Acoustics*, §6.3.
#[must_use]
#[inline]
pub fn reflection_pressure_coeff(z1: f64, z2: f64) -> f64 {
    (z2 - z1) / (z2 + z1)
}

/// Plane-wave pressure transmission coefficient at a normal-incidence interface.
///
/// ```text
/// T = 2·Z₂ / (Z₂ + Z₁)
/// ```
///
/// # Reference
/// Kinsler et al. (2000) *Fundamentals of Acoustics*, §6.3.
#[must_use]
#[inline]
pub fn transmission_pressure_coeff(z1: f64, z2: f64) -> f64 {
    2.0 * z2 / (z2 + z1)
}

/// Power-law attenuation in Nepers/m.
///
/// ```text
/// α(f) = α₀ · f^y   [Np/m]
/// ```
///
/// # Arguments
/// * `f_hz` – frequencies `Hz`
/// * `alpha0` – attenuation coefficient [Np/m/Hz^y]
/// * `y` – power-law exponent (typically 1–2)
///
/// # Reference
/// Szabo (1994), *J. Acoust. Soc. Am.* 96, 491.
#[must_use]
#[inline]
pub fn power_law_attenuation_np_m(f_hz: &[f64], alpha0: f64, y: f64) -> Vec<f64> {
    f_hz.iter().map(|&f| alpha0 * f.powf(y)).collect()
}

/// Power-law attenuation in dB/cm.
///
/// ```text
/// α(f) = α₀ · f^y   [dB/cm],  f in MHz
/// ```
///
/// # Arguments
/// * `f_mhz` – frequencies `MHz`
/// * `alpha0` – attenuation coefficient [dB/(cm·MHz^y)]
/// * `y` – power-law exponent
#[must_use]
#[inline]
pub fn absorption_power_law_db_cm(f_mhz: &[f64], alpha0: f64, y: f64) -> Vec<f64> {
    f_mhz.iter().map(|&f| alpha0 * f.powf(y)).collect()
}

/// Stokes-Kirchhoff thermoviscous absorption coefficient [Np/m].
///
/// Classical result for a viscous, heat-conducting Newtonian fluid (Stokes 1845,
/// Kirchhoff 1868).  At each angular frequency `ω = 2πf`:
///
/// ```text
/// α_SK(ω) = δ · ω² / (2 · c₀³)   [Np/m]
/// ```
///
/// where `δ` [m²/s] is the acoustic diffusivity (also called the sound
/// diffusivity), combining shear viscosity, bulk viscosity, and thermal
/// conductivity contributions.
///
/// # Arguments
/// * `freqs_hz` – frequencies `Hz`
/// * `delta_m2_s` – acoustic diffusivity δ [m²/s]
/// * `c0` – small-signal sound speed [m/s]
///
/// # Reference
/// Pierce (1989) *Acoustics*, §10.1, Eq. 10.1.11.
#[must_use]
#[inline]
pub fn stokes_kirchhoff_absorption_np_m(freqs_hz: &[f64], delta_m2_s: f64, c0: f64) -> Vec<f64> {
    use std::f64::consts::PI;
    let two_c3 = 2.0 * c0 * c0 * c0;
    freqs_hz
        .iter()
        .map(|&f| {
            let omega = 2.0 * PI * f;
            delta_m2_s * omega * omega / two_c3
        })
        .collect()
}

/// Generate a Hann-windowed tone burst pressure waveform.
///
/// Computes the time-domain pressure waveform for a CW tone burst with a
/// Hann (raised-cosine) amplitude envelope over the full burst duration:
///
/// ```text
/// τ = n_cycles / f₀
/// w(t) = ½·(1 − cos(2π·t / τ))          for t ∈ [0, τ]
///        0                                otherwise
/// p(t) = A · w(t) · sin(2π·f₀·t)
/// ```
///
/// The Hann window ensures p(0) = p(τ) = 0 and provides −43 dB sidelobe
/// suppression in the frequency domain (Harris 1978).
///
/// # Arguments
/// * `t_arr` – time sample points `s`; need not start at zero
/// * `amplitude_pa` – peak pressure amplitude `Pa`
/// * `freq_hz` – carrier frequency `Hz`
/// * `n_cycles` – number of cycles in the burst (positive real)
///
/// # Reference
/// Harris (1978), *Proc. IEEE* 66, 51 — window functions for spectral analysis.
/// k-Wave Toolbox `tone_burst` (Treeby & Cox 2010) for burst generation convention.
#[must_use]
pub fn tone_burst_waveform(
    t_arr: &[f64],
    amplitude_pa: f64,
    freq_hz: f64,
    n_cycles: f64,
) -> Vec<f64> {
    use std::f64::consts::PI;
    if freq_hz <= 0.0 || n_cycles <= 0.0 {
        return vec![0.0; t_arr.len()];
    }
    let tau = n_cycles / freq_hz;
    let two_pi_f = 2.0 * PI * freq_hz;
    let two_pi_over_tau = 2.0 * PI / tau;
    t_arr
        .iter()
        .map(|&t| {
            if t < 0.0 || t > tau {
                0.0
            } else {
                let w = 0.5 * (1.0 - (two_pi_over_tau * t).cos());
                amplitude_pa * w * (two_pi_f * t).sin()
            }
        })
        .collect()
}

/// Generate a centered Hann-windowed tone burst on an existing time axis.
///
/// This matches diagnostic display figures that define a symmetric axial RF
/// pulse around `t = 0` and apply a discrete Hann window over the selected
/// samples:
///
/// ```text
/// τ = n_cycles / f₀
/// mask(t) = |t| < τ / 2
/// p(t_i) = A · hann(k/(N-1)) · sin(2πf₀t_i),  t_i in mask
/// ```
///
/// `N` is the number of samples satisfying the mask, not an analytically
/// inferred sample count. That preserves the plotted axial-PSF convention while
/// keeping pulse construction in Rust.
#[must_use]
pub fn centered_hann_tone_burst_waveform(
    t_arr: &[f64],
    amplitude_pa: f64,
    freq_hz: f64,
    n_cycles: f64,
) -> Vec<f64> {
    use kwavers_math::signal::window::hann;
    use std::f64::consts::PI;

    let mut out = vec![0.0; t_arr.len()];
    if !(amplitude_pa.is_finite()
        && freq_hz.is_finite()
        && freq_hz > 0.0
        && n_cycles.is_finite()
        && n_cycles > 0.0)
    {
        return out;
    }
    if !t_arr.iter().all(|sample| sample.is_finite()) {
        return out;
    }

    let half_duration_s = 0.5 * n_cycles / freq_hz;
    let active_indices: Vec<usize> = t_arr
        .iter()
        .enumerate()
        .filter_map(|(idx, &time_s)| (time_s.abs() < half_duration_s).then_some(idx))
        .collect();
    let n_active = active_indices.len();
    if n_active < 2 {
        return out;
    }

    let denominator = n_active as f64 - 1.0;
    let angular_frequency = 2.0 * PI * freq_hz;
    for (window_idx, signal_idx) in active_indices.into_iter().enumerate() {
        let weight = hann(window_idx as f64 / denominator);
        out[signal_idx] = amplitude_pa * weight * (angular_frequency * t_arr[signal_idx]).sin();
    }
    out
}

/// Superimpose Hann-windowed tone bursts at arbitrary start times (pulse train).
///
/// For each start time `t_start_k` in `t_starts`, a tone burst
/// `tone_burst_waveform(t − t_start_k, amplitude_pa, freq_hz, n_cycles)` is
/// evaluated and accumulated:
///
/// ```text
/// p(t) = Σₖ A · w(t − tₖ) · sin(2π·f₀·(t − tₖ)),   w as in tone_burst_waveform
/// ```
///
/// This generates arbitrary PRF schedules (regular, dual-PRF, dithered) without
/// performing any PRF computation in Python.
///
/// # Arguments
/// * `t_arr` – full simulation time axis `s`
/// * `amplitude_pa` – per-burst peak amplitude `Pa`
/// * `freq_hz` – carrier frequency `Hz`
/// * `n_cycles` – cycles per burst
/// * `t_starts` – burst start times `s`; any order, duplicates allowed
///
/// # Reference
/// Harris (1978), *Proc. IEEE* 66, 51.
/// Macoskey et al. (2018), *Ultrasound Med. Biol.* 44, 2971.
#[must_use]
pub fn pulse_train_waveform(
    t_arr: &[f64],
    amplitude_pa: f64,
    freq_hz: f64,
    n_cycles: f64,
    t_starts: &[f64],
) -> Vec<f64> {
    use std::f64::consts::PI;
    if freq_hz <= 0.0 || n_cycles <= 0.0 {
        return vec![0.0; t_arr.len()];
    }
    let tau = n_cycles / freq_hz;
    let two_pi_f = 2.0 * PI * freq_hz;
    let two_pi_over_tau = 2.0 * PI / tau;
    let mut out = vec![0.0_f64; t_arr.len()];
    for &t0 in t_starts {
        for (out_val, &t) in out.iter_mut().zip(t_arr.iter()) {
            let t_rel = t - t0;
            if t_rel >= 0.0 && t_rel <= tau {
                let w = 0.5 * (1.0 - (two_pi_over_tau * t_rel).cos());
                *out_val += amplitude_pa * w * (two_pi_f * t_rel).sin();
            }
        }
    }
    out
}

#[cfg(test)]
mod centered_hann_tone_burst_tests {
    use super::centered_hann_tone_burst_waveform;
    use kwavers_math::signal::window::hann;

    #[test]
    fn centered_hann_tone_burst_matches_discrete_window_contract() {
        let sample_rate_hz = 40.0e6;
        let freq_hz = 5.0e6;
        let n_cycles = 2.0;
        let dt_s = 1.0 / sample_rate_hz;
        let times: Vec<f64> = (0..241).map(|idx| -3.0e-6 + idx as f64 * dt_s).collect();

        let pulse = centered_hann_tone_burst_waveform(&times, 1.0, freq_hz, n_cycles);

        let half_duration_s = 0.5 * n_cycles / freq_hz;
        let active: Vec<usize> = times
            .iter()
            .enumerate()
            .filter_map(|(idx, &time_s)| (time_s.abs() < half_duration_s).then_some(idx))
            .collect();
        let mut expected = vec![0.0; times.len()];
        let denominator = active.len() as f64 - 1.0;
        for (window_idx, signal_idx) in active.into_iter().enumerate() {
            expected[signal_idx] = hann(window_idx as f64 / denominator)
                * (std::f64::consts::TAU * freq_hz * times[signal_idx]).sin();
        }

        for (observed, expected) in pulse.iter().zip(expected.iter()) {
            assert!(
                (observed - expected).abs() <= 1.0e-14,
                "observed={observed:e}, expected={expected:e}"
            );
        }
    }

    #[test]
    fn centered_hann_tone_burst_rejects_invalid_inputs_with_zero_trace() {
        let times = [0.0, 1.0e-6];

        assert_eq!(
            centered_hann_tone_burst_waveform(&times, 1.0, 0.0, 2.0),
            vec![0.0, 0.0]
        );
        assert_eq!(
            centered_hann_tone_burst_waveform(&[0.0, f64::NAN], 1.0, 1.0e6, 2.0),
            vec![0.0, 0.0]
        );
    }
}
