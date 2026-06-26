//! Acoustic wavelength + the BVD equivalent-circuit series-branch resonance + a generic
//! LC parallel-tank resonance kernel.
//!
//! This submodule carries the characteristic-number kernels: medium wavelength `λ = c / f`
//! for any medium (tissue, water, ...) plus the BVD-derived series-branch resonance that
//! pins the driver's centre frequency to the transducer, plus a companion parallel-tank
//! LC resonance kernel useful for input-impedance matching networks.
//!
//! **Two clearly distinguished kernels live here**:
//!
//! * [`bvd_series_resonance_hz`] — the BVD-equivalent-circuit series-branch motional
//!   resonance `f_s = 1/(2π√(L_s·C_s))` (serdeable from the transducer's L_m, C_m pair).
//! * [`bvd_anti_resonance_hz`] — the **canonical textbook BVD anti-resonance**
//!   `f_p = (1/2π)·√((C_s + C_0)/(L_s·C_s·C_0))` per Kino *Acoustic Waves* §3.4 / IEEE
//!   Std 176. Couples the motional series branch `L_s·C_s` with the static dielectric
//!   capacitance `C_0` (parallel-plate capacitance of the crystal between the electrodes).
//!   Sits strictly above the series-branch resonance for any real transducer; the gap
//!   `f_p − f_s` sets the electromechanical coupling coefficient
//!   `k² = 1 − (f_s/f_p)²` that drives the matching-network bandwidth ratio.
//!
//! Both kernels are pure-math (`f64`-in/`f64`-out, no state, no cross-slice dep) so they
//! feed straight into the slice facade's named `pub use` re-export chain.

/// Acoustic wavelength (m) for a medium speed (m/s) and frequency (Hz).
#[must_use]
pub fn wavelength_m(speed_m_s: f64, freq_hz: f64) -> f64 {
    if freq_hz <= 0.0 {
        return f64::INFINITY;
    }
    speed_m_s / freq_hz
}

/// Butterworth–Van Dyke series-branch resonance (Hz): `f_s = 1/(2π√(L_s·C_s))`. The driver's
/// centre frequency should match the transducer's `f_s` for efficient acoustic output.
#[must_use]
pub fn bvd_series_resonance_hz(ls_h: f64, cs_f: f64) -> f64 {
    if ls_h <= 0.0 || cs_f <= 0.0 {
        return f64::INFINITY;
    }
    1.0 / (2.0 * std::f64::consts::PI * (ls_h * cs_f).sqrt())
}

/// BVD-equivalent-circuit anti-resonance (Hz): `f_p = (1/2π) · √((C_s + C_0)/(L_s·C_s·C_0))`.
/// Sits **above** the series-branch resonance (the textbook BVD anti-resonance coupling the
/// static dielectric capacitance `C_0` with the motional series branch `L_s·C_s`). The gap
/// between `f_s = bvd_series_resonance_hz(L_s, C_s)` and this `f_p` sets the effective
/// electromechanical coupling coefficient `k² = 1 − (f_s/f_p)² which drives the bandwidth-
/// ratio of the matching-network and the achievable transducer figure-of-merit.
///
/// Callers pass the **motional-branch** parameters (same `L_s, C_s` pair used in
/// [`bvd_series_resonance_hz`]) plus the **static capacitance** `c0_f` of the transducer's
/// dielectric (the parallel-plate capacitance of the crystal between the electrodes). All
/// three must be > 0; returns `f64::INFINITY` otherwise (mirrors the series-branch fn's
/// degenerate-input contract).
///
/// **SSOT distinction with [`bvd_series_resonance_hz`]**: this fn couples `L_s·C_s·C_0` and
/// is the **canonical BVD anti-resonance** per Kino *Acoustic Waves* §3.4 / IEEE Std 176 — a
/// physically-realistic transducer resonance. The [`bvd_series_resonance_hz`] fn carries
/// only the motional series branch `L_s·C_s` and ignores the static dielectric `C_0`; both
/// are useful as design primitives but they solve different physical problems. The phase-3g
/// SSOT-distinction test pins the difference by feeding each fn distinct parameter tuples
/// and asserting `f_p > f_s` strictly for any positive `C_0`.
#[must_use]
pub fn bvd_anti_resonance_hz(ls_h: f64, cs_f: f64, c0_f: f64) -> f64 {
    if ls_h <= 0.0 || cs_f <= 0.0 || c0_f <= 0.0 {
        return f64::INFINITY;
    }
    let numerator = cs_f + c0_f;
    let denominator = ls_h * cs_f * c0_f;
    (1.0 / (2.0 * std::f64::consts::PI)) * (numerator / denominator).sqrt()
}
