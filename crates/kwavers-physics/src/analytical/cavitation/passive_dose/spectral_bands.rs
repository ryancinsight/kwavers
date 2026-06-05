//! Decomposition of a passive-cavitation emission spectrum into the
//! cavitation-relevant frequency bands used by clinical harmonic-dose
//! controllers (e.g. InsighTec Exablate Neuro BBB-opening monitoring).
//!
//! ## Spectral signatures (Gyöngy & Coussios 2010; Arvanitis 2012; Bader 2018)
//! Driven at fundamental `f₀`, a microbubble population emits at:
//! * `n·f₀`            – harmonics: nonlinear oscillation + nonlinear propagation
//! * `f₀/2`            – subharmonic: hallmark of *stable* (non-inertial) cavitation
//! * `(2k+1)·f₀/2`     – ultraharmonics (3f₀/2, 5f₀/2, …): stable cavitation
//! * inharmonic floor  – broadband noise: hallmark of *inertial* cavitation
//!
//! The stable-cavitation dose integrates the sub- + ultra-harmonic energy; the
//! inertial-cavitation dose integrates the broadband energy. Both are measured
//! *above a baseline noise floor* (the pre-microbubble control spectrum level).

/// Emission energy partitioned into cavitation-relevant spectral bands.
///
/// All four fields are spectral energies (PSD integrated over the assigned
/// bins, in PSD-units·Hz) measured above the supplied noise floor. They are
/// additive and mutually exclusive: every spectrum bin contributes to exactly
/// one band, so `fundamental + subharmonic + ultraharmonic + broadband` equals
/// the total above-floor spectral energy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CavitationBandEnergies {
    /// Energy in the harmonic comb `n·f₀` (n ≥ 1), including the fundamental.
    pub fundamental: f64,
    /// Energy in the subharmonic line `f₀/2`.
    pub subharmonic: f64,
    /// Energy in the ultraharmonic comb `(2k+1)·f₀/2` (k ≥ 1): 3f₀/2, 5f₀/2, …
    pub ultraharmonic: f64,
    /// Inharmonic broadband energy (every bin outside all narrowband windows).
    pub broadband: f64,
}

impl CavitationBandEnergies {
    /// Stable-cavitation emission = subharmonic + ultraharmonic energy.
    ///
    /// Excludes the harmonic comb, which is also produced by nonlinear
    /// propagation and is therefore not a clean cavitation marker.
    #[must_use]
    #[inline]
    pub fn stable_emission(&self) -> f64 {
        self.subharmonic + self.ultraharmonic
    }

    /// Inertial-cavitation emission = broadband energy.
    #[must_use]
    #[inline]
    pub fn inertial_emission(&self) -> f64 {
        self.broadband
    }
}

/// Decompose a PCD emission power spectrum into cavitation bands above a floor.
///
/// Each spectrum bin at frequency `f` is assigned to the nearest half-harmonic
/// line `k·f₀/2` (with `k = round(2f/f₀)`) when it falls within the half-window
/// `±rel_halfwidth·f₀`; otherwise it is broadband:
/// * `k` even (k ≥ 2) → `n·f₀` harmonic comb → `fundamental`
/// * `k = 1`          → `f₀/2`               → `subharmonic`
/// * `k` odd  (k ≥ 3) → `(2k+1)f₀/2`         → `ultraharmonic`
/// * outside all windows (incl. `k = 0`, the DC/sub-`f₀/2` region) → `broadband`
///
/// For each assigned bin the contribution is `max(psd − noise_floor, 0)·Δf`,
/// where `Δf` is the bin spacing — i.e. the energy is integrated *above* the
/// baseline noise floor, matching clinical passive-cavitation-dose practice.
///
/// # Arguments
/// * `freqs`         – frequency axis [Hz], uniformly spaced and ascending
/// * `psd`           – power spectral density at each frequency (same length)
/// * `f0_hz`         – fundamental drive frequency [Hz]
/// * `rel_halfwidth` – line half-window as a fraction of `f₀` (e.g. 0.05);
///   clamped to `(0, 0.25)` so adjacent half-harmonic windows (spacing `f₀/2`)
///   never overlap
/// * `noise_floor`   – baseline PSD subtracted from every bin (≥ 0)
///
/// Returns a zeroed [`CavitationBandEnergies`] if the inputs are empty, of
/// unequal length, or shorter than 2 bins (no resolvable `Δf`).
///
/// # Reference
/// Gyöngy M. & Coussios C.C. (2010) *J. Acoust. Soc. Am.* 128, 2403.
/// Arvanitis C.D. et al. (2012) *PLoS ONE* 7, e45783.
#[must_use]
pub fn decompose_emission_spectrum(
    freqs: &[f64],
    psd: &[f64],
    f0_hz: f64,
    rel_halfwidth: f64,
    noise_floor: f64,
) -> CavitationBandEnergies {
    let n = freqs.len();
    let zero = CavitationBandEnergies {
        fundamental: 0.0,
        subharmonic: 0.0,
        ultraharmonic: 0.0,
        broadband: 0.0,
    };
    if n < 2 || psd.len() != n || !(f0_hz.is_finite() && f0_hz > 0.0) {
        return zero;
    }
    let df = freqs[1] - freqs[0];
    if !(df.is_finite() && df > 0.0) {
        return zero;
    }
    // Clamp the window so adjacent half-harmonic lines (spacing f₀/2) stay
    // disjoint: a half-width of f₀/4 makes neighbouring windows just touch.
    let rel = rel_halfwidth.clamp(f64::MIN_POSITIVE, 0.249_999);
    let halfwidth_hz = rel * f0_hz;
    let floor = noise_floor.max(0.0);

    let mut bands = zero;
    for (&f, &s) in freqs.iter().zip(psd.iter()) {
        let energy = (s - floor).max(0.0) * df;
        if energy == 0.0 {
            continue;
        }
        // Nearest half-harmonic index k (line at k·f₀/2).
        let k = (2.0 * f / f0_hz).round();
        let line_hz = k * 0.5 * f0_hz;
        let on_line = k >= 1.0 && (f - line_hz).abs() <= halfwidth_hz;
        if !on_line {
            bands.broadband += energy;
            continue;
        }
        let k_int = k as i64;
        if k_int % 2 == 0 {
            bands.fundamental += energy; // k = 2,4,6 → n·f₀
        } else if k_int == 1 {
            bands.subharmonic += energy; // f₀/2
        } else {
            bands.ultraharmonic += energy; // 3f₀/2, 5f₀/2, …
        }
    }
    bands
}
