/// Compute the pressure field of a 1-D standing wave.
///
/// ```text
/// p(x, t) = p₀ · sin(k·x) · cos(ω·t)   [Pa]
/// ```
///
/// # Arguments
/// * `p0` – peak pressure amplitude [Pa]
/// * `k` – wavenumber [rad/m]
/// * `x_arr` – spatial positions [m]
/// * `omega_t` – phase `ω·t` [rad]
#[must_use]
#[inline]
pub fn standing_wave_1d(p0: f64, k: f64, x_arr: &[f64], omega_t: f64) -> Vec<f64> {
    let cos_wt = omega_t.cos();
    x_arr.iter().map(|&x| p0 * (k * x).sin() * cos_wt).collect()
}

/// Compute the pressure field of a 1-D plane wave.
///
/// ```text
/// p(x, t) = A · cos(k·x − ω·t)   [Pa]
/// ```
///
/// # Arguments
/// * `amplitude` – peak amplitude [Pa]
/// * `k` – wavenumber [rad/m]
/// * `x_arr` – spatial positions [m]
/// * `omega_t` – `ω·t` [rad]
#[must_use]
#[inline]
pub fn plane_wave_pressure_1d(amplitude: f64, k: f64, x_arr: &[f64], omega_t: f64) -> Vec<f64> {
    x_arr
        .iter()
        .map(|&x| amplitude * (k * x - omega_t).cos())
        .collect()
}

/// Real part of the spherical-wave Green's function (far-field).
///
/// ```text
/// p(r) = A · cos(k·r) / r   [Pa]
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
/// * `f_hz` – frequencies [Hz]
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
/// * `f_mhz` – frequencies [MHz]
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
/// * `freqs_hz` – frequencies [Hz]
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
/// * `t_arr` – time sample points [s]; need not start at zero
/// * `amplitude_pa` – peak pressure amplitude [Pa]
/// * `freq_hz` – carrier frequency [Hz]
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
/// * `t_arr` – full simulation time axis [s]
/// * `amplitude_pa` – per-burst peak amplitude [Pa]
/// * `freq_hz` – carrier frequency [Hz]
/// * `n_cycles` – cycles per burst
/// * `t_starts` – burst start times [s]; any order, duplicates allowed
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
