//! Frequency-weighted bubble-cloud cavitation emission source.
//!
//! # Physics
//!
//! Acoustically driven cavitation does not radiate a single tone. A bubble cloud
//! driven at fundamental `f₀` emits a structured spectrum (Neppiras 1980;
//! Leighton 1994 §4): the driven response at `f₀` and its harmonics, a
//! *subharmonic* line at `f₀/2` (period-doubling, the onset marker of stable
//! cavitation), and *ultraharmonic* lines at `(2n+1)f₀/2` — chiefly `3f₀/2` —
//! that grow with nonlinear/inertial activity. Passive acoustic mapping (PAM)
//! exploits exactly these markers: the subharmonic and ultraharmonic bands are
//! beamformed to localize cavitation while rejecting the strong driven
//! fundamental.
//!
//! This module synthesizes a deterministic emission waveform with the three
//! PAM-relevant spectral lines (`f₀/2`, `f₀`, `3f₀/2`), each given a
//! physically-ordered relative weight, modulated by a common Gaussian burst
//! envelope. The envelope width is chosen so the lines are spectrally resolved
//! (envelope bandwidth ≪ the `f₀/2` line spacing), which lets the per-band PAM
//! bandpass isolate the subharmonic from the ultraharmonic and reject the
//! fundamental.
//!
//! # Theorem (line resolution)
//!
//! A Gaussian burst `exp(-t²/2σ²)·sin(2πfₖt)` has spectral standard deviation
//! `Δf = 1/(2πσ)`. With `σ = N/f₀` (here `N = 2.5` fundamental periods),
//! `Δf = f₀/(2πN) ≈ 0.064 f₀`, which is far below the inter-line spacing
//! `f₀/2`, so the three lines are separable: the fundamental sits `≈ 7.8 Δf`
//! from each adjacent band centre, giving `exp(-7.8²/2) ≈ 5·10⁻¹⁴` leakage.
//!
//! # References
//! - Neppiras, E.A. (1980). *Acoustic cavitation.* Physics Reports 61(3), 159–251.
//! - Leighton, T.G. (1994). *The Acoustic Bubble*, §4.4 (subharmonic /
//!   ultraharmonic emission).
//! - Gyöngy & Coussios (2010), IEEE TBME 57(1) — PAM of these bands.

/// Relative amplitude of the subharmonic (`f₀/2`) cavitation line.
const CAV_WEIGHT_SUBHARMONIC: f64 = 0.5;
/// Relative amplitude of the driven fundamental (`f₀`) line (dominant).
const CAV_WEIGHT_FUNDAMENTAL: f64 = 1.0;
/// Relative amplitude of the ultraharmonic (`3f₀/2`) cavitation line.
const CAV_WEIGHT_ULTRAHARMONIC: f64 = 0.35;
/// Gaussian envelope standard deviation in fundamental periods. Sets the line
/// bandwidth `Δf = f₀/(2πN)`; `N = 2.5` keeps `Δf ≈ 0.064 f₀ ≪ f₀/2`.
const CAV_ENVELOPE_FUNDAMENTAL_CYCLES: f64 = 2.5;

/// Highest spectral line in the cavitation emission, as a multiple of `f₀`.
/// Used to size the FDTD grid refinement so the emission stays resolved.
pub(super) const CAV_MAX_LINE_MULTIPLE: f64 = 1.5;

/// Envelope standard deviation `σ` in seconds for fundamental `f₀`.
#[must_use]
pub(super) fn cavitation_envelope_sigma_s(fundamental_hz: f64) -> f64 {
    CAV_ENVELOPE_FUNDAMENTAL_CYCLES / fundamental_hz
}

/// Total temporal extent of the cavitation burst in seconds (`6σ`: the burst is
/// centred at `3σ` and has decayed to `exp(-4.5) ≈ 1%` by `6σ`).
#[must_use]
pub(super) fn cavitation_burst_duration_s(fundamental_hz: f64) -> f64 {
    6.0 * cavitation_envelope_sigma_s(fundamental_hz)
}

/// Synthesize the bubble-cloud emission waveform sampled at `dt_s` for
/// `time_steps` steps, driven at `fundamental_hz`. Normalised to unit peak
/// amplitude (the absolute scale is applied via the grid `source_scale`).
///
/// The waveform is `env(t)·Σₖ wₖ sin(2π fₖ (t − t₀))` with lines
/// `fₖ ∈ {f₀/2, f₀, 3f₀/2}`, weights `wₖ`, Gaussian envelope `env`, and burst
/// centre `t₀ = 3σ` so the signal rises smoothly from ≈ 0 at `t = 0`.
#[must_use]
pub(super) fn cavitation_emission_waveform(
    time_steps: usize,
    dt_s: f64,
    fundamental_hz: f64,
) -> Vec<f32> {
    let two_pi = std::f64::consts::TAU;
    let f_sub = 0.5 * fundamental_hz;
    let lines = [
        (f_sub, CAV_WEIGHT_SUBHARMONIC),
        (fundamental_hz, CAV_WEIGHT_FUNDAMENTAL),
        (CAV_MAX_LINE_MULTIPLE * fundamental_hz, CAV_WEIGHT_ULTRAHARMONIC),
    ];
    let sigma = cavitation_envelope_sigma_s(fundamental_hz);
    let t_center = 3.0 * sigma;

    let mut waveform = vec![0.0_f32; time_steps];
    let mut max_abs = 0.0_f64;
    for (step, sample) in waveform.iter_mut().enumerate() {
        let t = step as f64 * dt_s;
        let u = (t - t_center) / sigma;
        let envelope = (-0.5 * u * u).exp();
        let mut harmonic_sum = 0.0;
        for (frequency, weight) in lines {
            harmonic_sum += weight * (two_pi * frequency * (t - t_center)).sin();
        }
        let value = envelope * harmonic_sum;
        *sample = value as f32;
        max_abs = max_abs.max(value.abs());
    }
    if max_abs > 0.0 {
        for sample in &mut waveform {
            *sample = (f64::from(*sample) / max_abs) as f32;
        }
    }
    waveform
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::fft::apply_spectral_response_1d;
    use ndarray::Array1;

    /// The synthesized emission must contain the three cavitation lines
    /// (f₀/2, f₀, 3f₀/2) with the expected relative ordering and resolvable
    /// separation — verified by band-pass energy at each line centre.
    #[test]
    fn emission_contains_resolved_cavitation_lines() {
        let f0 = 250_000.0;
        let dt = 2.0e-8; // fs = 50 MHz ≫ 2·(3f0/2) Nyquist
        let n = 8192;
        let waveform = cavitation_emission_waveform(n, dt, f0);
        let trace = Array1::from_iter(waveform.iter().map(|&v| f64::from(v)));
        let fs = 1.0 / dt;

        // Narrow band-pass energy at a centre frequency.
        let band_energy = |center: f64| -> f64 {
            let bw = f0 / 16.0;
            let filtered = apply_spectral_response_1d(&trace, fs, |_, freq, nyq| {
                let f_eff = freq.min(2.0 * nyq - freq).max(0.0);
                let z = (f_eff - center) / bw;
                (-0.5 * z * z).exp()
            });
            filtered.iter().map(|&x| x * x).sum()
        };

        let e_sub = band_energy(0.5 * f0);
        let e_fund = band_energy(f0);
        let e_ultra = band_energy(1.5 * f0);
        // Energy at a frequency with no line (0.75 f0) must be far smaller —
        // confirms the lines are discrete and resolvable.
        let e_gap = band_energy(0.75 * f0);

        assert!(e_sub > 0.0 && e_fund > 0.0 && e_ultra > 0.0);
        assert!(
            e_fund > e_sub && e_sub > e_ultra,
            "line ordering must be fundamental > subharmonic > ultraharmonic: \
             fund={e_fund:.3e} sub={e_sub:.3e} ultra={e_ultra:.3e}"
        );
        assert!(
            e_sub > 50.0 * e_gap && e_ultra > 50.0 * e_gap,
            "cavitation lines must dominate the inter-line gap: \
             sub={e_sub:.3e} ultra={e_ultra:.3e} gap={e_gap:.3e}"
        );
    }
}
