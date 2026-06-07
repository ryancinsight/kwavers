//! Inter-pulse residual-bubble clearance under a low-amplitude sweep.
//!
//! Between pulses the residual cavitation bubbles dissolve by gas diffusion
//! (Epstein–Plesset); the dissolution time scales as `R₀²`, so a large residual
//! bubble persists and *shields* the focus on the next pulse (the void-fraction
//! impedance/sound-speed drop already coupled into the propagation model). A
//! low-amplitude clearing sweep drives residual bubbles through their resonance
//! and fragments them: gas volume is conserved at break-up, but the daughter
//! bubbles are smaller and — because `dR/dt` accelerates as `R` shrinks
//! (`τ ∝ R²`) — dissolve far faster. Net residual void fraction at the next
//! pulse is therefore lower with the sweep than without it.
//!
//! This module reuses the audited Epstein–Plesset [`DissolutionModel`] and its
//! RK4 integrator; the sweep enters only through the gas-volume-conserving
//! fragmentation of the initial radius.

use kwavers_core::constants::cavitation::{
    GAS_DIFFUSION_COEFFICIENT_TISSUE, OSTWALD_SOLUBILITY_AIR_WATER, SURFACE_TENSION_TISSUE,
};
use kwavers_core::constants::fundamental::ATMOSPHERIC_PRESSURE;

use crate::acoustics::bubble_dynamics::dissolution::{
    integrate_dissolution, DissolutionModel, EpsteinPlessetDissolution, GasDiffusionParams,
};

/// Gas-transport parameters for a residual gas bubble in soft tissue with the
/// given dissolved-gas saturation fraction (`f < 1` ⇒ undersaturated ⇒
/// dissolves). Uses the tissue gas diffusivity and tissue surface tension.
#[must_use]
pub fn tissue_gas_diffusion(saturation_fraction: f64) -> GasDiffusionParams {
    GasDiffusionParams {
        diffusivity: GAS_DIFFUSION_COEFFICIENT_TISSUE,
        ostwald_solubility: OSTWALD_SOLUBILITY_AIR_WATER,
        saturation_fraction,
        surface_tension: SURFACE_TENSION_TISSUE,
        ambient_pressure: ATMOSPHERIC_PRESSURE,
    }
}

/// Outcome of an inter-pulse interval: residual bubble size and void fraction
/// remaining at the start of the next pulse, passively versus with a clearing
/// sweep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterPulseClearance {
    /// Residual radius after passive dissolution over the interval [m].
    pub residual_radius_passive_m: f64,
    /// Residual radius of a daughter fragment after the interval [m].
    pub residual_radius_swept_m: f64,
    /// Residual void fraction with no sweep ∈ [0, initial].
    pub void_fraction_passive: f64,
    /// Residual void fraction with the clearing sweep ∈ [0, initial].
    pub void_fraction_swept: f64,
    /// Clearance gain `passive / swept` residual void fraction (> 1 ⇒ the sweep
    /// leaves less shielding residual).
    pub clearance_gain: f64,
}

/// Residual void fraction left after an inter-pulse interval, with and without a
/// fragmenting clearing sweep.
///
/// # Arguments
/// * `initial_void_fraction` – void fraction deposited by the pulse (≥ 0).
/// * `initial_radius_m` – representative residual bubble radius `R₀` [m].
/// * `interval_s` – inter-pulse (OFF) interval `1/PRF` [s].
/// * `fragment_count` – daughter-bubble multiplicity `N ≥ 1` produced by the
///   clearing sweep (gas-volume-conserving: `r_frag = R₀·N^(−1/3)`). `N = 1`
///   means no fragmentation (sweep inactive) and recovers the passive result.
/// * `params` – Epstein–Plesset gas-transport parameters (see
///   [`tissue_gas_diffusion`]).
///
/// The void fraction tracks total residual gas volume: number of bubbles ×
/// radius³, normalised to the initial single-bubble volume.
#[must_use]
pub fn inter_pulse_residual_clearance(
    initial_void_fraction: f64,
    initial_radius_m: f64,
    interval_s: f64,
    fragment_count: f64,
    params: GasDiffusionParams,
) -> InterPulseClearance {
    let beta0 = initial_void_fraction.max(0.0);
    let n = fragment_count.max(1.0);
    if !(initial_radius_m.is_finite()
        && initial_radius_m > 0.0
        && interval_s.is_finite()
        && interval_s > 0.0)
    {
        return InterPulseClearance {
            residual_radius_passive_m: initial_radius_m.max(0.0),
            residual_radius_swept_m: initial_radius_m.max(0.0),
            void_fraction_passive: beta0,
            void_fraction_swept: beta0,
            clearance_gain: 1.0,
        };
    }

    let model = EpsteinPlessetDissolution::new(params);
    // Integration step: resolve the interval finely while keeping the early-time
    // transient term well-sampled; floor at 1 µs.
    let dt = (interval_s / 2000.0).max(1.0e-6);
    let r_dissolved = 1.0e-9; // MIN_RADIUS — treat as fully cleared

    let residual_after = |r0: f64| -> f64 {
        let traj = integrate_dissolution(&model, r0, dt, interval_s, r_dissolved);
        // Last radius (or the dissolved floor if it crossed). If the model grew
        // the bubble (supersaturated), `dissolution_time` is None and the radius
        // is the final value.
        traj.radius.last().copied().unwrap_or(r0).max(0.0)
    };

    // Passive: one bubble of radius R₀.
    let r_passive = residual_after(initial_radius_m);
    let vol_ratio_passive = (r_passive / initial_radius_m).powi(3);
    let beta_passive = beta0 * vol_ratio_passive;

    // Swept: N daughters of radius R₀·N^(−1/3) (gas-volume conserving at break-up).
    let r_frag0 = initial_radius_m * n.powf(-1.0 / 3.0);
    let r_frag = residual_after(r_frag0);
    // Total residual gas volume = N · r_frag³, normalised to R₀³.
    let vol_ratio_swept = n * (r_frag / initial_radius_m).powi(3);
    let beta_swept = beta0 * vol_ratio_swept;

    let clearance_gain = if beta_swept > 0.0 {
        beta_passive / beta_swept
    } else if beta_passive > 0.0 {
        f64::INFINITY
    } else {
        1.0
    };

    InterPulseClearance {
        residual_radius_passive_m: r_passive,
        residual_radius_swept_m: r_frag,
        void_fraction_passive: beta_passive,
        void_fraction_swept: beta_swept,
        clearance_gain,
    }
}

/// Closed-form Epstein–Plesset dissolution time `R₀ → 0` [s] for the given
/// transport parameters (surface-tension-free quasi-static limit
/// `τ = R₀²/(2·D·L·(1−f))`), exposed for sizing the inter-pulse interval. Falls
/// back to the model's own estimate when available.
#[must_use]
pub fn residual_dissolution_time_s(r0_m: f64, params: GasDiffusionParams) -> Option<f64> {
    EpsteinPlessetDissolution::new(params).dissolution_time(r0_m)
}
