use apollo::fft_1d_leto;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_math::signal::window::hann;
use kwavers_math::special::bessel::jn;

/// Evaluate the normalised amplitude of the nth harmonic at nonlinear parameter σ.
///
/// Fubini (1935) showed that for a lossless plane wave in the pre-shock
/// regime (σ < 1):
/// ```text
/// Bₙ(σ) = 2/(n·σ) · Jₙ(n·σ)
/// ```
/// where Jₙ is the Bessel function of the first kind of order n.
///
/// # Arguments
/// * `n` – harmonic number (n ≥ 1)
/// * `sigma` – Fubini–Euler parameter (0 ≤ σ < 1)
///
/// # Reference
/// Hamilton & Blackstock (1998) *Nonlinear Acoustics*, §3.3.
#[must_use]
pub fn fubini_harmonic_amplitude(n: u32, sigma: f64) -> f64 {
    let n_f = n as f64;
    let x = n_f * sigma;
    if x.abs() < 1e-15 {
        return if n == 1 { 1.0 } else { 0.0 };
    }
    2.0 / x * jn(n, x)
}

/// Compute the Fubini harmonic spectrum for harmonics n = 1..=n_max at parameter σ.
///
/// Returns a `Vec<f64>` of length `n_max` where index 0 corresponds to n = 1.
#[must_use]
pub fn fubini_harmonic_spectrum(n_max: u32, sigma: f64) -> Vec<f64> {
    (1..=n_max)
        .map(|n| fubini_harmonic_amplitude(n, sigma))
        .collect()
}

/// Extract Hann-windowed harmonic amplitudes from row-major time traces.
///
/// Each row is multiplied by the symmetric Hann window and transformed with the
/// workspace FFT facade.  The returned matrix is row-major with shape
/// `n_traces × n_harmonics`; column `0` is the fundamental, column `1` is the
/// second harmonic, and so on.  The amplitude normalization matches the
/// standard one-sided sinusoid estimate `2·|X`K`| / Σw`.
///
/// # Errors
/// Returns an error when dimensions are inconsistent, scalar parameters are not
/// positive finite values, the trace is too short for a symmetric Hann window,
/// or any input sample is non-finite.
pub fn hann_windowed_harmonic_amplitudes(
    traces: &[f64],
    n_traces: usize,
    n_samples: usize,
    dt_s: f64,
    fundamental_hz: f64,
    n_harmonics: usize,
) -> Result<Vec<f64>, String> {
    if n_traces == 0 {
        return Err("n_traces must be positive".to_owned());
    }
    if n_samples < 2 {
        return Err("n_samples must be at least two for a symmetric Hann window".to_owned());
    }
    if n_harmonics == 0 {
        return Err("n_harmonics must be positive".to_owned());
    }
    if traces.len() != n_traces * n_samples {
        return Err("traces length must equal n_traces * n_samples".to_owned());
    }
    if !(dt_s.is_finite() && dt_s > 0.0) {
        return Err("dt_s must be positive and finite".to_owned());
    }
    if !(fundamental_hz.is_finite() && fundamental_hz > 0.0) {
        return Err("fundamental_hz must be positive and finite".to_owned());
    }
    if !traces.iter().all(|sample| sample.is_finite()) {
        return Err("traces must contain only finite samples".to_owned());
    }

    let window_denominator = n_samples as f64 - 1.0;
    let mut window = Vec::with_capacity(n_samples);
    for idx in 0..n_samples {
        window.push(hann(idx as f64 / window_denominator));
    }
    let window_sum = window.iter().sum::<f64>();
    if !(window_sum.is_finite() && window_sum > 0.0) {
        return Err("Hann window sum must be positive and finite".to_owned());
    }

    let df_hz = 1.0 / (n_samples as f64 * dt_s);
    let n_positive = n_samples / 2 + 1;
    let mut amplitudes = Vec::with_capacity(n_traces * n_harmonics);
    for row in traces.chunks_exact(n_samples) {
        let windowed: Vec<f64> = row
            .iter()
            .zip(window.iter())
            .map(|(&sample, &weight)| sample * weight)
            .collect();
        let fft_input = leto::Array1::from_shape_vec([n_samples], windowed)
            .expect("windowed trace length must match Leto FFT shape");
        let spectrum = fft_1d_leto(fft_input.view());
        for harmonic in 1..=n_harmonics {
            let bin = (harmonic as f64 * fundamental_hz / df_hz).round();
            let amp = if bin.is_finite() && bin >= 0.0 {
                let idx = bin as usize;
                if idx < n_positive {
                    2.0 * spectrum[idx].norm() / window_sum
                } else {
                    0.0
                }
            } else {
                0.0
            };
            amplitudes.push(amp);
        }
    }
    Ok(amplitudes)
}

/// Normalised amplitude of the nth harmonic in the **post-shock sawtooth (Fay)
/// region**, the complement of the pre-shock [`fubini_harmonic_amplitude`].
///
/// Past shock formation (σ ≥ 1) the lossless plane wave becomes a sawtooth whose
/// slope — and hence amplitude — decays as `1/(1+σ)` (energy lost at the shock).
/// The Fourier series of that sawtooth gives
/// ```text
/// Bₙ(σ) = 2 / (n·(1+σ))     (σ ≥ 1)
/// ```
/// so every harmonic decays as `1/n` (the sawtooth signature) and the whole
/// spectrum decays as `1/(1+σ)` with distance. Together with the Fubini solution
/// this is the Fay–Fubini pair connected by Blackstock (1966).
///
/// # Arguments
/// * `n` – harmonic number (n ≥ 1; `n = 0` returns 0)
/// * `sigma` – normalised distance σ = z/z_s (sawtooth region σ ≥ 1)
///
/// # Reference
/// Blackstock (1966), *J. Acoust. Soc. Am.* 39, 1019 (Fay–Fubini connection);
/// Hamilton & Blackstock (1998) *Nonlinear Acoustics*, §4.4 (sawtooth region).
#[must_use]
pub fn sawtooth_harmonic_amplitude(n: u32, sigma: f64) -> f64 {
    let denom = (1.0 + sigma) * n as f64;
    let valid = denom.is_finite() && denom > 0.0;
    if n == 0 || !valid {
        return 0.0;
    }
    2.0 / denom
}

/// Compute the post-shock **sawtooth (Fay)** harmonic spectrum for harmonics
/// n = 1..=n_max at parameter σ (index 0 ↔ n = 1).
#[must_use]
pub fn sawtooth_harmonic_spectrum(n_max: u32, sigma: f64) -> Vec<f64> {
    (1..=n_max)
        .map(|n| sawtooth_harmonic_amplitude(n, sigma))
        .collect()
}

/// Shock-formation distance for a sinusoidal plane wave (Fubini–Euler criterion).
///
/// ```text
/// x_s = ρ₀·c₀³ / (β·p₀·ω)   `m`
/// ```
///
/// # Arguments
/// * `p0_pa` – source pressure amplitude `Pa`
/// * `f0_hz` – fundamental frequency `Hz`
/// * `c0` – small-signal sound speed [m/s]
/// * `rho0` – ambient density [kg/m³]
/// * `beta` – nonlinearity parameter β = 1 + B/(2A)
///
/// # Reference
/// Blackstock (1966), *J. Acoust. Soc. Am.* 39, 1019.
#[must_use]
#[inline]
pub fn shock_formation_distance(p0_pa: f64, f0_hz: f64, c0: f64, rho0: f64, beta: f64) -> f64 {
    let omega = TWO_PI * f0_hz;
    rho0 * c0.powi(3) / (beta * p0_pa * omega)
}

/// Reconstruct the time-domain Fubini waveform from its harmonic series.
///
/// For a lossless plane wave in the pre-shock regime (σ < 1), the Fubini–Euler
/// solution gives the exact pressure waveform at a propagation depth where the
/// non-linearity parameter equals σ:
///
/// ```text
/// p(t) = p₀ · Σ_{n=1}^{n_max} Bₙ(σ) · sin(n·ω·t),   ω = 2π·f₀
/// ```
///
/// where `Bₙ(σ) = 2·Jₙ(n·σ) / (n·σ)` (Fubini 1935, Eq. 3.38).
///
/// # Arguments
/// * `t_arr` – time sample points `s`
/// * `p0_pa` – source pressure amplitude `Pa`
/// * `freq_hz` – fundamental frequency `Hz`
/// * `sigma` – Fubini–Euler parameter (0 ≤ σ < 1); clamped to 0.999
/// * `n_max` – highest harmonic order to include (≥ 1)
///
/// # Correctness
/// The Fubini series is exact for σ < 1. At σ → 1 the wave approaches shock
/// formation; higher `n_max` improves the approximation but σ must remain
/// strictly below 1.  Callers should not interpret the output as physically
/// valid for σ ≥ 1 (post-shock propagation requires the Rankine-Hugoniot jump
/// conditions and is outside the scope of this function).
///
/// # Reference
/// Hamilton & Blackstock (1998) *Nonlinear Acoustics*, §3.3, Eq. 3.38.
/// Fubini–Ghiron (1935), *Alta Frequenza* 4, 530.
#[must_use]
pub fn fubini_waveform(
    t_arr: &[f64],
    p0_pa: f64,
    freq_hz: f64,
    sigma: f64,
    n_max: u32,
) -> Vec<f64> {
    let sigma = sigma.clamp(0.0, 0.999_f64);
    let omega = TWO_PI * freq_hz;
    // Pre-compute harmonic amplitudes B_n(σ) = 2 J_n(n·σ) / (n·σ)
    let b_n: Vec<f64> = (1..=n_max)
        .map(|n| fubini_harmonic_amplitude(n, sigma))
        .collect();
    t_arr
        .iter()
        .map(|&t| {
            b_n.iter().enumerate().fold(0.0_f64, |acc, (i, &b)| {
                let n = (i + 1) as f64;
                acc + b * (n * omega * t).sin()
            }) * p0_pa
        })
        .collect()
}

/// Goldberg shock parameter for a pulsed plane wave, swept over pulse durations.
///
/// The Goldberg (Fubini–Euler) nonlinearity parameter at propagation distance
/// `x = c · τ` (one-burst path length) is:
///
/// ```text
/// σ(τ) = β · (2π·f / c) · (p₀ / (ρ·c²)) · (c · τ)
///       = β · 2π·f · p₀ · τ / (ρ·c²)
/// ```
///
/// σ < 1 → pre-shock (Fubini regime, valid for analytical waveform expansion).
/// σ = 1 → shock-formation distance reached within one burst duration.
/// σ > 1 → post-shock (Rankine–Hugoniot regime; Fubini series no longer exact).
///
/// # Arguments
/// * `pnp_pa` – peak negative pressure amplitude `Pa`
/// * `freq_hz` – fundamental frequency `Hz`
/// * `c` – small-signal sound speed [m/s]
/// * `rho` – ambient density [kg/m³]
/// * `beta` – nonlinearity parameter β = 1 + B/(2A)
/// * `tau_arr` – pulse durations `s` (one σ value per τ)
///
/// # Reference
/// Hamilton & Blackstock (1998) *Nonlinear Acoustics*, §3.3, Eq. 3.37.
/// Goldberg (1957), *Akust. Zh.* 3, 307.
#[must_use]
pub fn goldberg_shock_parameter_sweep(
    pnp_pa: f64,
    freq_hz: f64,
    c: f64,
    rho: f64,
    beta: f64,
    tau_arr: &[f64],
) -> Vec<f64> {
    // σ(τ) = β · (2πf/c) · ε · (c·τ),  ε = pnp/(ρc²)
    // Simplifies to: σ(τ) = β · 2πf · pnp · τ / (ρ·c²)
    let c2 = c * c;
    let coeff = if c2 <= 0.0 || rho <= 0.0 {
        0.0
    } else {
        beta * TWO_PI * freq_hz * pnp_pa / (rho * c2)
    };
    tau_arr.iter().map(|&tau| coeff * tau).collect()
}

/// Shock-enhanced absorption gain factor (phenomenological model).
///
/// In the pre-shock regime, acoustic absorption increases with nonlinearity
/// because energy is transferred to higher harmonics where attenuation scales
/// as n²·α₁.  The effective absorption gain relative to the linear value is
/// modelled as a sigmoid in the Goldberg parameter σ:
///
/// ```text
/// G(σ) = 1 + 9·σ / (σ + 1)
/// ```
///
/// G → 1 at σ = 0 (linear limit), G → 10 as σ → ∞ (fully shocked limit).
///
/// # Arguments
/// * `sigma_arr` – Goldberg shock parameters (element-wise)
///
/// # Reference
/// Hamilton & Blackstock (1998) *Nonlinear Acoustics*, §4.3 (absorption
/// enhancement in nonlinear waves).
#[must_use]
pub fn shock_enhanced_absorption_gain(sigma_arr: &[f64]) -> Vec<f64> {
    sigma_arr
        .iter()
        .map(|&s| {
            let s = s.max(0.0);
            1.0 + 9.0 * s / (s + 1.0)
        })
        .collect()
}

/// Effective pressure amplitude of a shock-distorted waveform.
///
/// In a shock-formed waveform the positive pressure peak (PPP) exceeds the
/// negative pressure peak (|PNP|) due to nonlinear steepening.  The effective
/// RMS-equivalent amplitude blends from PNP (linear limit, σ→0) to PPP
/// (fully-shocked limit, σ→∞) via a sigmoid in σ:
///
/// ```text
/// p_eff(σ) = p₋ · (1 + (p₊/p₋ − 1) · σ / (σ + 1))
/// ```
///
/// # Arguments
/// * `pnp_pa` – peak negative pressure amplitude |p⁻| `Pa`
/// * `ppp_pa` – peak positive pressure p⁺ `Pa` (≥ pnp for a shock)
/// * `sigma_arr` – Goldberg shock parameters (element-wise)
///
/// # Reference
/// Hamilton & Blackstock (1998) *Nonlinear Acoustics*, §3.3 (waveform
/// distortion and amplitude asymmetry in shocked plane waves).
#[must_use]
pub fn shock_waveform_pressure(pnp_pa: f64, ppp_pa: f64, sigma_arr: &[f64]) -> Vec<f64> {
    let ratio_minus_one = (ppp_pa / pnp_pa.max(f64::MIN_POSITIVE)) - 1.0;
    sigma_arr
        .iter()
        .map(|&s| {
            let s = s.max(0.0);
            pnp_pa * (1.0 + ratio_minus_one * s / (s + 1.0))
        })
        .collect()
}

/// Shock-enhanced volumetric heat-source density Q_eff [W/m³].
///
/// Combines the shock-distorted pressure amplitude model with the
/// shock-enhanced absorption gain to give the effective time-averaged heat
/// deposition at each Goldberg parameter value:
///
/// ```text
/// Q_eff(σ) = G(σ) · α · p_eff(σ)² / (ρ · c)
/// ```
///
/// where:
/// * `G(σ) = 1 + 9·σ/(σ+1)` is the absorption gain
/// * `p_eff(σ) = p_eff_arr`i`` is the effective pressure at index i
/// * `α·p²/(ρ·c) = 2α·I` is the standard linear heat-source density
///
/// # Arguments
/// * `p_eff_arr` – effective pressure amplitude array `Pa` (element-wise, same length as sigma_arr)
/// * `sigma_arr` – Goldberg shock parameters (element-wise)
/// * `alpha_np_m` – linear attenuation coefficient [Np/m]
/// * `rho` – density [kg/m³]
/// * `c` – sound speed [m/s]
///
/// # Reference
/// Hamilton & Blackstock (1998) *Nonlinear Acoustics*, §4.3.
/// Duck (1990) *Physical Properties of Tissue*, §5.2.
#[must_use]
pub fn shock_heat_source_density(
    p_eff_arr: &[f64],
    sigma_arr: &[f64],
    alpha_np_m: f64,
    rho: f64,
    c: f64,
) -> Vec<f64> {
    let rho_c = (rho * c).max(f64::MIN_POSITIVE);
    let n = p_eff_arr.len().min(sigma_arr.len());
    (0..n)
        .map(|i| {
            let s = sigma_arr[i].max(0.0);
            let gain = 1.0 + 9.0 * s / (s + 1.0);
            let p = p_eff_arr[i];
            gain * alpha_np_m * p * p / rho_c
        })
        .collect()
}

/// Rectangular-envelope Fubini waveform for a shock-formed millisecond pulse.
///
/// Generates the focal time-domain waveform of a shock-vapor histotripsy pulse:
/// a rectangular (hard-on / hard-off) envelope with the Fubini harmonic content
/// at parameter σ, modelling the focal pressure at a propagation depth where the
/// Goldberg nonlinearity parameter equals σ.
///
/// ```text
/// p(t) = p₀ · Σ_{n=1}^{n_max} Bₙ(σ) · sin(n·ω·(t − t_start))
///         for t ∈ [t_start, t_start + duration_s)
///         0 otherwise
/// ```
///
/// # Arguments
/// * `t_arr` – time sample points `s`
/// * `p0_pa` – peak pressure amplitude `Pa`
/// * `f0` – fundamental frequency `Hz`
/// * `duration_s` – pulse duration `s`
/// * `t_start` – burst start time `s`
/// * `sigma` – Fubini–Euler nonlinearity parameter (0 ≤ σ < 1); clamped to 0.999
/// * `n_max` – highest harmonic order to include
///
/// # Reference
/// Hamilton & Blackstock (1998) *Nonlinear Acoustics*, §3.3.
/// Khokhlova et al. (2014), *Int. J. Hyperthermia* 31, 145 (ms shock-vapor regime).
#[must_use]
pub fn shock_vapor_pulse_waveform(
    t_arr: &[f64],
    p0_pa: f64,
    f0: f64,
    duration_s: f64,
    t_start: f64,
    sigma: f64,
    n_max: u32,
) -> Vec<f64> {
    let sigma = sigma.clamp(0.0, 0.999_f64);
    let omega = TWO_PI * f0;
    let b_n: Vec<f64> = (1..=n_max)
        .map(|n| fubini_harmonic_amplitude(n, sigma))
        .collect();
    t_arr
        .iter()
        .map(|&t| {
            let t_rel = t - t_start;
            if t_rel < 0.0 || t_rel >= duration_s {
                return 0.0;
            }
            b_n.iter().enumerate().fold(0.0_f64, |acc, (i, &b)| {
                let n = (i + 1) as f64;
                acc + b * (n * omega * t_rel).sin()
            }) * p0_pa
        })
        .collect()
}

/// Compute harmonic evolution along propagation axis using the Westervelt / KZK
/// plane-wave solution with linear absorption (perturbation theory, first-order
/// successive-approximation for n = 2 harmonics, exact Fubini for higher
/// harmonics scaled by exponential absorption).
///
/// For the nth harmonic:
/// ```text
/// pₙ(z) = p₀ · Bₙ(σ(z)) · exp(−n²·α·z)
/// ```
/// where σ(z) = z / x_s and α is the absorption at the fundamental.
///
/// # Arguments
/// * `z_arr` – propagation distances `m`
/// * `p0` – source pressure `Pa`
/// * `f0` – fundamental frequency `Hz`
/// * `c0` – sound speed [m/s]
/// * `rho0` – density [kg/m³]
/// * `beta` – nonlinearity parameter β
/// * `alpha_np_m` – attenuation at fundamental [Np/m]
/// * `n_max` – highest harmonic to compute
///
/// Returns a 2-D Vec of shape `[n_z][n_harmonic]` (n_harmonic = n_max).
///
/// # Reference
/// Hamilton & Blackstock (1998) *Nonlinear Acoustics*, ch. 4.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn westervelt_harmonic_evolution(
    z_arr: &[f64],
    p0: f64,
    f0: f64,
    c0: f64,
    rho0: f64,
    beta: f64,
    alpha_np_m: f64,
    n_max: usize,
) -> Vec<Vec<f64>> {
    let omega = TWO_PI * f0;
    let x_s = rho0 * c0.powi(3) / (beta * p0 * omega);

    z_arr
        .iter()
        .map(|&z| {
            let sigma = (z / x_s).min(0.99);
            (1..=n_max)
                .map(|n| {
                    let b_n = fubini_harmonic_amplitude(n as u32, sigma);
                    // n-th harmonic is at n·f₀; for power-law α∝f², α_n = n²·α₁
                    // (Hamilton & Blackstock 1998 §4.3 eq. 4.3.9; Aanonsen et al. 1984 eq. 6)
                    let absorption = (-(n as f64).powi(2) * alpha_np_m * z).exp();
                    p0 * b_n * absorption
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod hann_windowed_harmonic_tests {
    use super::hann_windowed_harmonic_amplitudes;
    use kwavers_core::constants::numerical::TWO_PI;
    use kwavers_math::signal::window::hann;

    fn manual_harmonic_amplitude(
        signal: &[f64],
        dt_s: f64,
        fundamental_hz: f64,
        harmonic: usize,
    ) -> f64 {
        let n = signal.len();
        let df_hz = 1.0 / (n as f64 * dt_s);
        let bin = (harmonic as f64 * fundamental_hz / df_hz).round() as usize;
        let mut real = 0.0;
        let mut imag = 0.0;
        let mut window_sum = 0.0;
        for (idx, &sample) in signal.iter().enumerate() {
            let weight = hann(idx as f64 / (n as f64 - 1.0));
            window_sum += weight;
            let phase = -TWO_PI * bin as f64 * idx as f64 / n as f64;
            real += sample * weight * phase.cos();
            imag += sample * weight * phase.sin();
        }
        2.0 * real.hypot(imag) / window_sum
    }

    #[test]
    fn hann_windowed_harmonics_match_manual_dft_bins() {
        let n_samples = 64;
        let dt_s = 1.0 / 16_000.0;
        let fundamental_hz = 1_000.0;
        let mut traces = Vec::with_capacity(2 * n_samples);
        for row in 0..2 {
            for idx in 0..n_samples {
                let t = idx as f64 * dt_s;
                let signal = if row == 0 {
                    3.0 * (TWO_PI * fundamental_hz * t).sin()
                        + 0.5 * (TWO_PI * 2.0 * fundamental_hz * t).sin()
                } else {
                    1.5 * (TWO_PI * 3.0 * fundamental_hz * t).sin()
                };
                traces.push(signal);
            }
        }

        let amplitudes =
            hann_windowed_harmonic_amplitudes(&traces, 2, n_samples, dt_s, fundamental_hz, 3)
                .expect("invariant: finite exact-bin traces define harmonic amplitudes");

        assert_eq!(amplitudes.len(), 6);
        for row in 0..2 {
            let signal = &traces[row * n_samples..(row + 1) * n_samples];
            for harmonic in 1..=3 {
                let expected = manual_harmonic_amplitude(signal, dt_s, fundamental_hz, harmonic);
                let got = amplitudes[row * 3 + harmonic - 1];
                assert!(
                    (got - expected).abs() <= 1.0e-12,
                    "row {row} harmonic {harmonic}: got {got}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn hann_windowed_harmonics_reject_invalid_inputs() {
        assert!(hann_windowed_harmonic_amplitudes(&[0.0, 1.0], 1, 2, 0.0, 1.0, 1).is_err());
        assert!(hann_windowed_harmonic_amplitudes(&[0.0, f64::NAN], 1, 2, 1.0, 1.0, 1).is_err());
        assert!(hann_windowed_harmonic_amplitudes(&[0.0, 1.0], 1, 2, 1.0, 1.0, 0).is_err());
        assert!(hann_windowed_harmonic_amplitudes(&[0.0], 1, 2, 1.0, 1.0, 1).is_err());
    }
}

#[cfg(test)]
mod sawtooth_tests {
    use super::{
        fubini_harmonic_amplitude, sawtooth_harmonic_amplitude, sawtooth_harmonic_spectrum,
    };

    /// The sawtooth harmonics decay as 1/n (the defining sawtooth signature):
    /// Bₙ/B₁ = 1/n exactly, at any σ.
    #[test]
    fn sawtooth_harmonics_decay_as_inverse_n() {
        for &sigma in &[1.0, 3.0, 9.0] {
            let b1 = sawtooth_harmonic_amplitude(1, sigma);
            for n in 1..=8 {
                let bn = sawtooth_harmonic_amplitude(n, sigma);
                assert!(
                    (bn / b1 - 1.0 / n as f64).abs() < 1e-12,
                    "B{n}/B1 must be 1/{n} at σ={sigma}"
                );
            }
        }
    }

    /// The whole spectrum decays as 1/(1+σ) with distance, and matches the
    /// closed form Bₙ = 2/(n(1+σ)).
    #[test]
    fn sawtooth_amplitude_decays_with_distance() {
        // B₁(σ) = 2/(1+σ): 1.0 at σ=1, 0.5 at σ=3, 0.2 at σ=9.
        assert!((sawtooth_harmonic_amplitude(1, 1.0) - 1.0).abs() < 1e-12);
        assert!((sawtooth_harmonic_amplitude(1, 3.0) - 0.5).abs() < 1e-12);
        assert!((sawtooth_harmonic_amplitude(1, 9.0) - 0.2).abs() < 1e-12);

        // Closed-form check over a few (n, σ).
        for (n, sigma) in [(2u32, 1.0), (3, 4.0), (5, 2.5)] {
            let expected = 2.0 / (n as f64 * (1.0 + sigma));
            assert!((sawtooth_harmonic_amplitude(n, sigma) - expected).abs() < 1e-12);
        }
    }

    /// Spectrum helper matches the per-harmonic function; degenerate inputs give 0.
    #[test]
    fn sawtooth_spectrum_and_degenerate() {
        let spec = sawtooth_harmonic_spectrum(4, 2.0);
        assert_eq!(spec.len(), 4);
        for (i, &v) in spec.iter().enumerate() {
            assert!((v - sawtooth_harmonic_amplitude(i as u32 + 1, 2.0)).abs() < 1e-15);
        }
        assert_eq!(sawtooth_harmonic_amplitude(0, 2.0), 0.0);
        assert_eq!(sawtooth_harmonic_amplitude(1, -2.0), 0.0); // 1+σ ≤ 0
        assert_eq!(sawtooth_harmonic_amplitude(1, f64::NAN), 0.0);
    }

    /// At the shock point σ=1 the Fubini (pre-shock) fundamental is depleted
    /// (≈0.88) while the sawtooth (post-shock) form starts at 1.0 — the known
    /// Fay–Fubini discontinuity in the connection region (Blackstock 1966).
    #[test]
    fn fay_fubini_connection_at_shock() {
        let fubini_b1 = fubini_harmonic_amplitude(1, 1.0); // 2·J1(1) ≈ 0.880
        let sawtooth_b1 = sawtooth_harmonic_amplitude(1, 1.0); // 1.0
        assert!((fubini_b1 - 0.8801).abs() < 1e-3, "fubini B1={fubini_b1}");
        assert!((sawtooth_b1 - 1.0).abs() < 1e-12);
        assert!(sawtooth_b1 > fubini_b1, "sawtooth over-predicts near σ=1");
    }
}
