//! Time-domain superposition of a microbubble population's acoustic emissions.
//!
//! A passive cavitation detector records the *coherent* sum of the pressure
//! radiated by every bubble in the focal volume, each arriving with its own
//! nucleation/propagation delay. Genuine broadband emission is an ensemble
//! effect: a single steady-state bubble radiates a line spectrum (harmonics of
//! `f₀`), but a polydisperse population whose bubbles ring up/down as impulsive
//! transients at *random nucleation times* radiates energy between the
//! harmonics — the broadband (inertial) signature. Summing power spectra is
//! wrong (it discards the inter-line structure); the superposition must be done
//! in the time domain before the spectrum is taken (Gyöngy & Coussios 2010).

/// Coherently superpose per-bubble emission series with per-bubble delays/gains.
///
/// ```text
///   y[t] = Σ_i gain_i · emission_i[t − delay_i]      (delay_i in samples)
/// ```
/// All `n_bubbles` per-bubble series share length `n_samples` and are laid out
/// row-major in `emissions` (`emissions[i*n_samples + s]`). Each is placed at
/// integer sample offset `delays`i`` and scaled by `gains`i``, accumulating into
/// an output buffer of length `out_len` (which must be at least
/// `n_samples + max(delays)` to avoid truncation). The returned series is fed to
/// [`super::hann_windowed_power_spectrum`] to obtain the ensemble PSD.
///
/// # Arguments
/// * `emissions` – `n_bubbles × n_samples` row-major per-bubble emission series
/// * `n_bubbles`, `n_samples` – grid dimensions of `emissions`
/// * `delays` – per-bubble nucleation/arrival delay `samples` (length `n_bubbles`)
/// * `gains` – per-bubble amplitude weight (length `n_bubbles`)
/// * `out_len` – length of the summed output buffer
///
/// Returns the summed ensemble emission of length `out_len`. Returns an empty
/// vector on any shape mismatch or if `out_len == 0`. A bubble whose placement
/// `delay_i + n_samples` exceeds `out_len` is clamped (its tail is truncated)
/// rather than panicking.
#[must_use]
pub fn ensemble_emission_superposition(
    emissions: &[f64],
    n_bubbles: usize,
    n_samples: usize,
    delays: &[usize],
    gains: &[f64],
    out_len: usize,
) -> Vec<f64> {
    if out_len == 0
        || n_bubbles == 0
        || n_samples == 0
        || emissions.len() != n_bubbles * n_samples
        || delays.len() != n_bubbles
        || gains.len() != n_bubbles
    {
        return Vec::new();
    }
    let mut out = vec![0.0_f64; out_len];
    for i in 0..n_bubbles {
        let base = i * n_samples;
        let d = delays[i];
        if d >= out_len {
            continue;
        }
        let g = gains[i];
        let n_copy = n_samples.min(out_len - d);
        let src = &emissions[base..base + n_copy];
        let dst = &mut out[d..d + n_copy];
        for (o, &s) in dst.iter_mut().zip(src.iter()) {
            *o += g * s;
        }
    }
    out
}
