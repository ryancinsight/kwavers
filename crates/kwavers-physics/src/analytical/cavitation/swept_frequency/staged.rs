//! Staged sonication frequency program: one slow up-and-down sweep across the
//! whole per-spot exposure.
//!
//! The sweep is *not* a fast carrier chirp repeated within a single pulse; it is
//! a single triangle in drive frequency spread over the entire sonication (the
//! per-spot pulse train). Its purpose is tied to the **stage of sonication**:
//!
//!   * **Build half** (stage 0 → ½): the drive frequency rises toward the
//!     cavitation-optimal frequency (the median nuclei Minnaert resonance), so
//!     the per-pulse engaged fraction — and hence cavitation activity — climbs to
//!     a peak at mid-sonication. Residual gas accumulates as the cloud builds.
//!   * **Wind-down half** (stage ½ → 1): the frequency falls back off resonance,
//!     so newly deposited cavitation tapers, *and* the falling drive acts as a
//!     clearing sweep that fragments the residual bubbles (gas-volume-conserving
//!     daughters dissolve faster, τ ∝ R²). The residual void fraction is driven
//!     back down so the **next** sonication starts from a clean, unshielded,
//!     un-pre-seeded field.
//!
//! This composes the per-pulse [`monochromatic_engaged_fraction`] (cavitation at
//! the staged frequency) with [`inter_pulse_residual_clearance`] (residual
//! evolution, with fragmentation enabled only in the wind-down half).

use crate::acoustics::bubble_dynamics::dissolution::GasDiffusionParams;

use super::clearance::inter_pulse_residual_clearance;
use super::engagement::{monochromatic_engaged_fraction, CavitationMedium, EngagementConfig};
use super::nuclei::NucleiSizeDistribution;

/// Triangular stage envelope `tri(s)` peaking at `s = ½`: `0 → 1 → 0`.
#[inline]
fn triangle(s: f64) -> f64 {
    if s <= 0.5 {
        2.0 * s
    } else {
        2.0 * (1.0 - s)
    }
}

/// Per-spot staged-sonication profile sampled once per pulse.
#[derive(Debug, Clone, PartialEq)]
pub struct StagedSonication {
    /// Stage fraction `s ∈ [0, 1]` of each pulse through the sonication.
    pub stage: Vec<f64>,
    /// Drive frequency `f(s)` of each pulse [Hz] — one up-down triangle.
    pub frequency_hz: Vec<f64>,
    /// Per-pulse cavitation activity = engaged nuclei fraction at `f(s)` ∈ [0, 1].
    pub cavitation_activity: Vec<f64>,
    /// Residual void fraction remaining at the *start of the next pulse* after
    /// each pulse's deposit and inter-pulse clearance.
    pub residual_void: Vec<f64>,
    /// Stage `s` at which cavitation activity peaks.
    pub peak_activity_stage: f64,
    /// Peak residual void fraction reached during the sonication.
    pub residual_peak: f64,
    /// Residual void fraction left at the end (carryover to the next sonication).
    pub residual_at_end: f64,
}

/// Run a single up-and-down frequency sweep across the sonication and return the
/// per-pulse cavitation-activity and residual-void profiles.
///
/// # Arguments
/// * `dist`, `medium` — nuclei population and liquid medium.
/// * `f_quiet_hz` — sweep endpoint frequency: the "quiet", low-cavitation
///   (high-threshold) frequency the drive starts and ends on, so the field is
///   left un-re-seeded for the next sonication.
/// * `f_peak_hz` — turn frequency at mid-sonication: the cavitation-optimal
///   frequency (see [`super::engagement::cavitation_optimal_frequency`]) where
///   activity peaks. May be above or below `f_quiet_hz`.
/// * `amplitude_pa` — per-pulse drive amplitude.
/// * `pulse_duration_s` — single-pulse ON duration.
/// * `n_pulses` — pulses in the per-spot train (≥ 1).
/// * `prf_hz` — pulse-repetition frequency (sets the inter-pulse interval).
/// * `void_deposit_per_activity` — void fraction deposited by a fully-active
///   pulse (scales the per-pulse residual source term).
/// * `residual_radius_m` — representative residual bubble radius for clearance.
/// * `clearing_fragment_count` — daughter multiplicity `N` produced by the
///   wind-down clearing sweep (`> 1`); the build half uses `N = 1` (no clearing).
/// * `diffusion` — Epstein–Plesset transport parameters.
/// * `cfg` — engagement-integration controls.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn staged_sonication_sweep(
    dist: &NucleiSizeDistribution,
    medium: &CavitationMedium,
    f_quiet_hz: f64,
    f_peak_hz: f64,
    amplitude_pa: f64,
    pulse_duration_s: f64,
    n_pulses: usize,
    prf_hz: f64,
    void_deposit_per_activity: f64,
    residual_radius_m: f64,
    clearing_fragment_count: f64,
    diffusion: GasDiffusionParams,
    cfg: &EngagementConfig,
) -> StagedSonication {
    let n = n_pulses.max(1);
    let interval_s = 1.0 / prf_hz.max(1e-9);
    let deposit = void_deposit_per_activity.max(0.0);
    let n_frag = clearing_fragment_count.max(1.0);

    let mut stage = Vec::with_capacity(n);
    let mut frequency_hz = Vec::with_capacity(n);
    let mut cavitation_activity = Vec::with_capacity(n);
    let mut residual_void = Vec::with_capacity(n);

    let mut running_void = 0.0_f64;
    let mut residual_peak = 0.0_f64;
    let mut peak_activity = -1.0_f64;
    let mut peak_activity_stage = 0.0_f64;

    for i in 0..n {
        let s = if n == 1 {
            0.5
        } else {
            i as f64 / (n - 1) as f64
        };
        // One up-down frequency triangle: f_low at the ends, f_peak at mid-stage.
        let f = f_quiet_hz + (f_peak_hz - f_quiet_hz) * triangle(s);
        // Per-pulse cavitation activity = engaged fraction at this stage frequency.
        let activity =
            monochromatic_engaged_fraction(dist, medium, f, amplitude_pa, pulse_duration_s, cfg);

        // Deposit this pulse's cavitation gas, then evolve over the inter-pulse
        // interval. Fragmentation (the clearing sweep) is active only in the
        // wind-down half (s > ½); the build half lets the cloud accumulate.
        running_void += deposit * activity;
        let frag = if s > 0.5 { n_frag } else { 1.0 };
        let c = inter_pulse_residual_clearance(
            running_void,
            residual_radius_m,
            interval_s,
            frag,
            diffusion,
        );
        running_void = c.void_fraction_swept;

        if activity > peak_activity {
            peak_activity = activity;
            peak_activity_stage = s;
        }
        residual_peak = residual_peak.max(running_void);

        stage.push(s);
        frequency_hz.push(f);
        cavitation_activity.push(activity);
        residual_void.push(running_void);
    }

    let residual_at_end = residual_void.last().copied().unwrap_or(0.0);
    StagedSonication {
        stage,
        frequency_hz,
        cavitation_activity,
        residual_void,
        peak_activity_stage,
        residual_peak,
        residual_at_end,
    }
}
