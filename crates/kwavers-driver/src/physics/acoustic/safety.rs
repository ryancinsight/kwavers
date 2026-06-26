//! Acoustic safety metrics: mechanical index, tissue attenuation, intensity, ISPPA,
//! and round-trip attenuation.
//!
//! This submodule carries the FDA/regulatory safety-and-intensity kernels clustered together
//! because they all feed safety-budget reports that the device's compliance audit reads.
//!
//! * [`mechanical_index`] — FDA cavitation-safety metric. Diagnostic limit is 1.9;
//!   neuromodulation / therapy runs above it deliberately, so reporting it is a safety gate.
//! * [`tissue_attenuation_db`] — one-way (`α · f · z`) acoustic loss through soft tissue.
//!   Drives the source-pressure / drive-voltage budget.
//! * [`pressure_derating`] — linear pressure amplitude factor `10^(−dB/20)` from a dB loss.
//! * [`acoustic_intensity_w_per_m2`] — `I = p² / Z₀` from continuous RMS pressure (the
//!   spatial-temporal-average intensity).
//! * [`isppa_w_per_m2`] — **NEW**: spatial-peak pulse-average intensity from peak-negative
//!   pressure + duty factor. FDA Track-3 regulatory metric for pulsed operation; distinct
//!   from the continuous-time `acoustic_intensity_w_per_m2`.
//! * [`round_trip_attenuation_db`] — **NEW**: pulse-echo two-way (`2 · α · f · z`) loss;
//!   the relevant quantity for time-gain-compensation (TGC) curves and pulse-echo imaging.
//!
//! The five existing + two new kernels are pure-math (`f64`-in/`f64`-out, no state, no
//! cross-slice dep) so they feed straight into the slice facade's named `pub use`
//! re-export chain. The two new APIs are SSOT-distinct from the existing four (see the
//! SSOT-distinction test in `crate::physics::acoustic::tests`).

/// Mechanical Index — the FDA cavitation-safety metric: `MI = P_neg / √f_c`, peak rarefactional
/// pressure in MPa over √(centre frequency in MHz). Diagnostic limit is 1.9; neuromodulation/therapy
/// runs above it deliberately, so reporting it is a safety gate for the device.
#[must_use]
pub fn mechanical_index(p_neg_mpa: f64, freq_mhz: f64) -> f64 {
    if freq_mhz <= 0.0 {
        return f64::INFINITY;
    }
    p_neg_mpa / freq_mhz.sqrt()
}

/// Acoustic attenuation (dB) through tissue: `α · f · z` (soft tissue ≈ 0.5 dB/cm/MHz). Sets how
/// much source pressure — hence drive voltage — the neuromodulation target depth demands.
///
/// **One-way** form: suitable for transmit-only pressure-budget sizing. For pulse-echo
/// imaging, derive the round-trip form via [`round_trip_attenuation_db`].
#[must_use]
pub fn tissue_attenuation_db(alpha_db_cm_mhz: f64, freq_mhz: f64, depth_cm: f64) -> f64 {
    alpha_db_cm_mhz * freq_mhz * depth_cm
}

/// Pressure-derating factor (linear, ≤1) from `attenuation_db`: `10^(−dB/20)`.
#[must_use]
pub fn pressure_derating(attenuation_db: f64) -> f64 {
    10f64.powf(-attenuation_db / 20.0)
}

/// Acoustic intensity (W/m²) from RMS pressure (Pa) in a medium of characteristic impedance `z0`.
///
/// `I = p² / Z₀`, `Z₀ = ρ·c`. Typical values: water `Z₀ ≈ 1.48 MRayl`, tissue `≈ 1.54 MRayl`.
/// The ISPTA (spatial-peak temporal-average) or ISPPA (spatial-peak pulse-average) metrics for
/// safety are derived from this intensity at the focus. For therapy neuromodulation the target is
/// typically `I_sppa > 100 W/cm²` at the focus — see [`isppa_w_per_m2`].
///
/// **SSOT distinction**: this function returns the **continuous-RMS** intensity. For pulsed
/// operation use [`isppa_w_per_m2`], which takes peak-negative pressure + duty factor.
#[must_use]
pub fn acoustic_intensity_w_per_m2(p_rms_pa: f64, z0_rayl: f64) -> f64 {
    if z0_rayl <= 0.0 {
        return 0.0;
    }
    p_rms_pa * p_rms_pa / z0_rayl
}

/// **NEW**: Spatial-peak pulse-average intensity (W/m²) — FDA Track-3 regulatory metric.
///
/// `I_sppa = (p_neg²) / (2 · Z₀) · duty_factor` where `p_neg` is the peak rarefactional
/// (negative-going) pressure, `Z₀ = ρ·c` is the medium's characteristic impedance, and
/// `duty_factor` is the pulse-active fraction of the cycle (∈ [0, 1]).
///
/// Distinctions from [`acoustic_intensity_w_per_m2`]:
/// * This API takes **peak-negative** pressure (`p_neg`), not RMS — the 2 in the denominator
///   is `√2² = 2`, the inverse of the RMS-to-peak conversion (`p_rms = p_peak / √2`).
/// * This API multiplies by `duty_factor` — pulses spend part of the cycle quiescent, so the
///   time-averaged intensity drops linearly.
///
/// Boundary behaviour:
/// * `z0_rayl ≤ 0` ⇒ returns `0.0` (degenerate medium).
/// * `peak_negative_pa ≤ 0` OR `duty_factor ≤ 0` ⇒ returns `0.0` (no power delivered).
/// * `duty_factor > 1.0` ⇒ returns `f64::INFINITY` (caller error: caller fed an invalid
///   duty factor; the function refuses to compute a meaningless figure rather than silently
///   extrapolate).
///
/// **SSOT** (fixed in `crate::physics::acoustic::tests::ssot_distinction_isppa_vs_intensity`):
/// ISPPA and continuous-intensity are distinct quantities and must not be substituted for each
/// other at a call site. Both are required by the safety budget: ISPPA for short-pulse thermal
/// dose characterisation, continuous-intensity for steady-state heating.
#[must_use]
pub fn isppa_w_per_m2(peak_negative_pa: f64, z0_rayl: f64, duty_factor: f64) -> f64 {
    if z0_rayl <= 0.0 {
        return 0.0;
    }
    if peak_negative_pa <= 0.0 || duty_factor <= 0.0 {
        return 0.0;
    }
    if duty_factor > 1.0 {
        return f64::INFINITY;
    }
    peak_negative_pa * peak_negative_pa * duty_factor / (2.0 * z0_rayl)
}

/// **NEW**: Pulse-echo round-trip attenuation (dB): `2 · α · f · z`. The relevant quantity
/// in time-gain-compensation (TGC) curves and any pulse-echo imaging budget.
///
/// Distinction from [`tissue_attenuation_db`]: the one-way form (`α · f · z`) suits
/// transmit-only pressure budgets; pulse-echo imaging traverses the tissue twice (transmit
/// + receive), so the relevant figure is `2 · α · f · z`. The two are easily confused at a
/// call site — the SSOT-distinction test in `crate::physics::acoustic::tests` pins the
/// factor of two.
///
/// Boundary behaviour:
/// * `alpha_db_cm_mhz < 0` OR `freq_mhz < 0` OR `depth_cm < 0` ⇒ returns `f64::INFINITY`
///   (caller error: negative physical inputs).
#[must_use]
pub fn round_trip_attenuation_db(alpha_db_cm_mhz: f64, freq_mhz: f64, depth_cm: f64) -> f64 {
    if alpha_db_cm_mhz < 0.0 || freq_mhz < 0.0 || depth_cm < 0.0 {
        return f64::INFINITY;
    }
    2.0 * alpha_db_cm_mhz * freq_mhz * depth_cm
}
