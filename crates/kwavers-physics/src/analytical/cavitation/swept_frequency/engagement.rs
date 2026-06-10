//! Swept-frequency nuclei engagement: how much of the bubble population a chirp
//! drives into inertial collapse versus a single tone.
//!
//! A nucleus is *engaged* when its peak expansion ratio `R_max/R₀` reaches the
//! inertial-collapse criterion (≈ 2; Flynn 1964, Apfel & Holland 1991). The
//! engaged *fraction* of the population is found by integrating the per-size
//! Keller–Miksis response over the [`NucleiSizeDistribution`]:
//!
//! * a **monochromatic** tone engages only the sizes whose resonance sits in the
//!   narrow band it dwells on;
//! * a **swept** drive sequentially matches the resonances across its covered
//!   band, engaging a strictly larger size population — provided the pulse is
//!   long enough (in carrier cycles) for the sweep to traverse the band.
//!
//! The pulse window caps both integrations, so a microsecond (≈ single-cycle)
//! pulse realizes no swept advantage (the sweep cannot move, and the bubble
//! cannot ring up), whereas a millisecond pulse realizes the full enhancement —
//! the ms-vs-µs asymmetry emerges from the dynamics, not an ad-hoc switch.

use kwavers_core::constants::cavitation::{
    POLYTROPIC_EXPONENT_AIR, SURFACE_TENSION_TISSUE, VAPOR_PRESSURE_WATER,
};
use kwavers_core::constants::fundamental::ATMOSPHERIC_PRESSURE;

use super::super::dynamics::keller_miksis_shelled_rk4;
use super::chirp::FrequencySweep;
use super::chirped_dynamics::chirped_keller_miksis_rk4;
use super::nuclei::NucleiSizeDistribution;

/// Liquid/medium parameters shared by every bubble in the population.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CavitationMedium {
    /// Ambient pressure `P₀` [Pa].
    pub p0_pa: f64,
    /// Liquid density `ρ` [kg/m³].
    pub rho: f64,
    /// Surface tension `σ` [N/m].
    pub sigma: f64,
    /// Liquid dynamic viscosity `μ` [Pa·s].
    pub mu: f64,
    /// Polytropic exponent `κ` of the gas.
    pub kappa: f64,
    /// Saturated vapor pressure `P_v` [Pa].
    pub p_v_pa: f64,
    /// Liquid sound speed `c` [m/s].
    pub c_liquid: f64,
}

impl CavitationMedium {
    /// Soft-tissue defaults: tissue surface tension, water-like density and
    /// viscosity, diatomic-gas polytropic exponent, 1540 m/s.
    #[must_use]
    pub fn soft_tissue() -> Self {
        Self {
            p0_pa: ATMOSPHERIC_PRESSURE,
            rho: 1050.0,
            sigma: SURFACE_TENSION_TISSUE,
            mu: 1.5e-3,
            kappa: POLYTROPIC_EXPONENT_AIR,
            p_v_pa: VAPOR_PRESSURE_WATER,
            c_liquid: 1540.0,
        }
    }
}

/// Engagement-integration controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EngagementConfig {
    /// Number of log-spaced nuclei radii sampled across the distribution.
    pub n_size_samples: usize,
    /// Half-width of the sampled size band in log-standard-deviations.
    pub n_sigma: f64,
    /// Inertial-collapse criterion `R_max/R₀` (≈ 2.0).
    pub inertial_ratio: f64,
    /// Time steps per carrier cycle (RK4 resolution).
    pub steps_per_cycle: usize,
    /// Cap on the monochromatic integration in carrier cycles (ring-up window).
    pub mono_cycles: f64,
    /// Cap on the swept integration in sweep periods.
    pub max_sweep_periods: f64,
    /// Hard cap on RK4 step count (cost/stability guard).
    pub max_steps: usize,
}

impl Default for EngagementConfig {
    fn default() -> Self {
        Self {
            n_size_samples: 41,
            n_sigma: 3.0,
            inertial_ratio: 2.0,
            steps_per_cycle: 64,
            mono_cycles: 60.0,
            max_sweep_periods: 2.0,
            max_steps: 200_000,
        }
    }
}

/// Result of an engagement comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EngagementResult {
    /// Number-weighted fraction engaged by the monochromatic tone ∈ [0, 1].
    pub mono_fraction: f64,
    /// Number-weighted fraction engaged by the swept drive ∈ [0, 1].
    pub swept_fraction: f64,
    /// Enhancement factor `swept / mono` (mono floored to avoid division by
    /// zero; ≥ 1 means the sweep engages more of the population).
    pub enhancement_factor: f64,
    /// Frequency band actually traversed by the sweep within the pulse [Hz].
    pub covered_band_hz: (f64, f64),
}

/// Uniform time samples `[0, total_s]` with `n_steps + 1` points.
fn linspace(total_s: f64, n_steps: usize) -> Vec<f64> {
    let n = n_steps.max(1);
    let dt = total_s / n as f64;
    (0..=n).map(|i| i as f64 * dt).collect()
}

/// Step count for a window of `duration_s` resolving carrier `f_resolve_hz`,
/// clamped to `[steps_per_cycle, max_steps]`.
fn step_count(duration_s: f64, f_resolve_hz: f64, cfg: &EngagementConfig) -> usize {
    let cycles = (duration_s * f_resolve_hz).max(1.0);
    ((cycles * cfg.steps_per_cycle as f64) as usize).clamp(cfg.steps_per_cycle, cfg.max_steps)
}

/// Fraction of the population a **monochromatic** tone at `freq_hz` engages
/// within a pulse of `pulse_duration_s`.
#[must_use]
pub fn monochromatic_engaged_fraction(
    dist: &NucleiSizeDistribution,
    medium: &CavitationMedium,
    freq_hz: f64,
    amplitude_pa: f64,
    pulse_duration_s: f64,
    cfg: &EngagementConfig,
) -> f64 {
    if !(freq_hz.is_finite() && freq_hz > 0.0 && amplitude_pa.is_finite() && amplitude_pa > 0.0) {
        return 0.0;
    }
    let window = pulse_duration_s.min(cfg.mono_cycles / freq_hz).max(0.0);
    if window <= 0.0 {
        return 0.0;
    }
    let t_arr = linspace(window, step_count(window, freq_hz, cfg));
    let (radii, weights) = dist.sample_radii(cfg.n_size_samples, cfg.n_sigma);
    let mut engaged = 0.0;
    let mut total = 0.0;
    for (r0, w) in radii.iter().zip(weights.iter()) {
        total += w;
        let (r, _) = keller_miksis_shelled_rk4(
            *r0,
            0.0,
            amplitude_pa,
            freq_hz,
            &t_arr,
            medium.p0_pa,
            medium.rho,
            medium.sigma,
            medium.mu,
            medium.kappa,
            medium.p_v_pa,
            0.0,
            medium.c_liquid,
        );
        let r_max = r.iter().copied().fold(*r0, f64::max);
        if r_max / *r0 >= cfg.inertial_ratio {
            engaged += w;
        }
    }
    if total > 0.0 {
        (engaged / total).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Fraction of the population the **swept** drive engages within a pulse of
/// `pulse_duration_s`.
#[must_use]
pub fn swept_engaged_fraction(
    dist: &NucleiSizeDistribution,
    medium: &CavitationMedium,
    sweep: &FrequencySweep,
    amplitude_pa: f64,
    pulse_duration_s: f64,
    cfg: &EngagementConfig,
) -> f64 {
    if !(amplitude_pa.is_finite() && amplitude_pa > 0.0) {
        return 0.0;
    }
    // Cap the integration at a few sweep periods (beyond that the sweep merely
    // repeats), but never beyond the pulse — the cycle-budget gate.
    let window = pulse_duration_s
        .min(cfg.max_sweep_periods * sweep.period_s)
        .max(0.0);
    if window <= 0.0 {
        return 0.0;
    }
    let f_resolve = sweep.f_start_hz.max(sweep.f_end_hz);
    let t_arr = linspace(window, step_count(window, f_resolve, cfg));
    let (radii, weights) = dist.sample_radii(cfg.n_size_samples, cfg.n_sigma);
    let mut engaged = 0.0;
    let mut total = 0.0;
    for (r0, w) in radii.iter().zip(weights.iter()) {
        total += w;
        let (r, _) = chirped_keller_miksis_rk4(
            sweep,
            amplitude_pa,
            *r0,
            0.0,
            &t_arr,
            medium.p0_pa,
            medium.rho,
            medium.sigma,
            medium.mu,
            medium.kappa,
            medium.p_v_pa,
            0.0,
            medium.c_liquid,
        );
        let r_max = r.iter().copied().fold(*r0, f64::max);
        if r_max / *r0 >= cfg.inertial_ratio {
            engaged += w;
        }
    }
    if total > 0.0 {
        (engaged / total).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Cavitation-optimal drive frequency: the frequency in `[f_lo_hz, f_hi_hz]`
/// that maximizes the engaged nuclei fraction at the given amplitude and pulse
/// duration.
///
/// At histotripsy sub-saturation amplitudes the engaged fraction is governed by
/// the inertial *threshold* (a lower frequency resonates larger nuclei, which
/// have a lower collapse threshold) as much as by resonance-matching to the
/// population median, so the optimum is found by a direct scan rather than
/// assumed to be the median Minnaert resonance. Returns `(f_opt_hz, fraction)`.
#[must_use]
// Physical drive parameters (medium, frequency bounds, amplitude, pulse duration,
// scan resolution, engagement config) are independent scalars with no cohesive
// sub-grouping; bundling them would obscure the call site without adding meaning.
// Matches the sibling sweep functions in this module.
#[allow(clippy::too_many_arguments)]
pub fn cavitation_optimal_frequency(
    dist: &NucleiSizeDistribution,
    medium: &CavitationMedium,
    f_lo_hz: f64,
    f_hi_hz: f64,
    amplitude_pa: f64,
    pulse_duration_s: f64,
    n_scan: usize,
    cfg: &EngagementConfig,
) -> (f64, f64) {
    let n = n_scan.max(2);
    let (lo, hi) = (f_lo_hz.min(f_hi_hz), f_lo_hz.max(f_hi_hz));
    let mut best_f = lo;
    let mut best = -1.0_f64;
    for i in 0..n {
        let f = lo + (hi - lo) * (i as f64 / (n - 1) as f64);
        let frac =
            monochromatic_engaged_fraction(dist, medium, f, amplitude_pa, pulse_duration_s, cfg);
        if frac > best {
            best = frac;
            best_f = f;
        }
    }
    (best_f, best.max(0.0))
}

/// Compare swept vs monochromatic engagement at matched amplitude and pulse
/// duration. The monochromatic reference tone is the sweep mean frequency.
#[must_use]
pub fn swept_vs_monochromatic_engagement(
    dist: &NucleiSizeDistribution,
    medium: &CavitationMedium,
    sweep: &FrequencySweep,
    amplitude_pa: f64,
    pulse_duration_s: f64,
    cfg: &EngagementConfig,
) -> EngagementResult {
    let mono = monochromatic_engaged_fraction(
        dist,
        medium,
        sweep.mean_frequency_hz(),
        amplitude_pa,
        pulse_duration_s,
        cfg,
    );
    let swept = swept_engaged_fraction(dist, medium, sweep, amplitude_pa, pulse_duration_s, cfg);
    EngagementResult {
        mono_fraction: mono,
        swept_fraction: swept,
        enhancement_factor: swept / mono.max(1e-9),
        covered_band_hz: sweep.covered_band_hz(pulse_duration_s),
    }
}
