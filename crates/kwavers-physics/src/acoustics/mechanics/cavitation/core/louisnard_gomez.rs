//! Cycle-averaged rectified diffusion for strongly forced bubbles
//! (Fyrillas–Szeri / Louisnard–Gomez law).
//!
//! # Law
//!
//! Fyrillas & Szeri (1994, *J. Fluid Mech.* 277, 381–407), as applied to
//! inertially oscillating bubbles by Louisnard & Gomez (2003, *Phys. Rev.
//! E* 67, 036610, Eqs. (6)–(9), (12), (16)): the cycle-averaged growth
//! rate of the equilibrium radius `R0` is
//!
//! ```text
//! dR0/dt = [D·R_G·T·C0 / (p0·R0·(1 + 2α_S/3))] · (C∞* − ⟨pg/p0⟩₄) / I
//! ```
//!
//! with `α_S = 2σ/(p0·R0)`, `C∞*` the far-field saturation ratio, and the
//! `R⁴`-weighted cycle average (their Eq. (8))
//!
//! ```text
//! ⟨pg/p0⟩₄ = ⟨(pg/p0)·R⁴⟩ / ⟨R⁴⟩
//! ```
//!
//! The gas pressure follows the van der Waals isothermal law (their
//! Eq. (2)): `pg = pg0·(R0³ − h³)/(R³ − h³)` with `h = R0/8.76` and
//! `pg0 = p0 + 2σ/R0`. The dimensionless integral `I` (their Eq. (7)) is
//! evaluated numerically from the radius history via the App. A
//! substitution `u = 3s/Rmax*³`:
//!
//! ```text
//! I = (1/(3·Rmax*)) ∫₀^∞ du / ⟨(u + (R*/Rmax*)³)^(4/3)⟩
//! ```
//!
//! The law is the same physics as the Smith & Wang (2022, *J. Fluid
//! Mech.* 939, A28) history-integral form, but cycle-averaged *before*
//! the flux is evaluated, so the collapse phase is suppressed by the
//! `R⁴` weighting instead of amplified by a memory kernel. The Smith &
//! Wang form is **not** implemented here: the triplet-pulse prototype
//! (Build 6a, validation V2) found it returns unphysical per-cycle
//! fluxes (|Δn| ≫ bubble gas content) for inertially collapsing
//! bubbles, because its inner solution assumes the concentration
//! profile tracks the wall instantaneously, which fails when the wall
//! moves far faster than diffusion across the boundary layer.
//!
//! The caller supplies a radius history `R(t)` over (at least) one
//! steady cycle — e.g. from
//! [`KellerMiksisModel`](crate::acoustics::bubble_dynamics::KellerMiksisModel) —
//! because the law is a functional of the oscillation, not of the drive
//! amplitude alone. Louisnard & Gomez §IV: the growth rate is dominated
//! by the expansion phase, where the gas pressure is very low, which is
//! why the law is insensitive to collapse-phase modelling detail.
//!
//! # Validity regime
//!
//! Validated by Louisnard & Gomez at 20 kHz < f < 100 kHz, 1 bar < p <
//! 5 bar, `R0 < 15 µm`, near-saturated liquid. The rectified-diffusion
//! threshold (their Eq. (9), `C∞* = ⟨pg/p0⟩₄`) merges with the Blake
//! threshold above ~1.4 bar at near saturation: below it the bubble
//! dissolves, above it an inertial bubble grows.
//!
//! The triplet-pulse prototype (Build 6a) validated this implementation
//! route against the paper's Fig. 4 growth rates at the paper's own
//! conditions (26.5 kHz, saturated): all six anchors reproduced within
//! ~10% (223 vs ~200, 31 vs ~30, 95 vs ~100, 275 vs ~300, 28 vs ~30,
//! 9.5 vs ~10 µm/s). The paper's Fig. 6 growth times are reproduced
//! only partially (3–9× longer): Fig. 6 rests on the paper's analytical
//! Eq. (22) with the App. A fit for `I`, whose printed constants do not
//! reproduce the paper's own integral (A11) — direct evaluation gives
//! `I` ~5.4× larger. The numerical `I` used here reproduces (A11)
//! exactly and matches Fig. 7's exact-`I` values, so the Fig. 4 match
//! (exact numerical Eq. (16)) is the stronger validation.
//!
//! Applying the law to MHz / MPa tone bursts (as the formation model
//! does for the Iwasaki/Kuji bursts) is an extrapolation — 10× in
//! frequency and 10–20× in pressure beyond the validated range — made
//! with the numerical law evaluated on the actual `R(t)`, not with the
//! paper's closed-form approximations. Use
//! [`louisnard_gomez_regime`] to see which extrapolations a given call
//! makes, and [`louisnard_gomez_growth_rate_checked`] to refuse
//! out-of-regime inputs outright.
//!
//! # Relation to the Eller–Flynn rate
//!
//! [`rectified_diffusion_rate`](super::rectified_diffusion_rate) is the
//! small-amplitude (ε ≪ 1) steady rate, valid for `P_ac/P0 ≤ 3`. This
//! law is the strongly forced counterpart: it takes the full nonlinear
//! `R(t)` and remains meaningful when the bubble collapses inertially
//! every cycle, where the Eller–Flynn flux is meaningless. At burst
//! conditions (R0 = 5 µm, ~5–10 MPa, 1–1.25 MHz) this law's per-cycle
//! influx is 230–1500× smaller than a per-cycle application of the
//! Bader Eq. (7) single-pulse influx (see
//! [`bader_expansion_influx`](super::bader_expansion_influx)).

use kwavers_core::constants::fundamental::{BOLTZMANN, GAS_CONSTANT};
use kwavers_core::error::{KwaversError, KwaversResult, PhysicsError};

/// Lower frequency bound of the Louisnard & Gomez validated range [Hz]
/// (their study spans 20–100 kHz).
pub const LOUISNARD_GOMEZ_MIN_FREQUENCY_HZ: f64 = 20_000.0;
/// Upper frequency bound of the Louisnard & Gomez validated range [Hz].
pub const LOUISNARD_GOMEZ_MAX_FREQUENCY_HZ: f64 = 100_000.0;
/// Lower drive-pressure bound of the validated range [Pa] (1 bar).
pub const LOUISNARD_GOMEZ_MIN_PRESSURE_PA: f64 = 100_000.0;
/// Upper drive-pressure bound of the validated range [Pa] (5 bar).
pub const LOUISNARD_GOMEZ_MAX_PRESSURE_PA: f64 = 500_000.0;
/// Upper equilibrium-radius bound of the validated range [m] (15 µm).
pub const LOUISNARD_GOMEZ_MAX_RADIUS_M: f64 = 15e-6;

/// Medium / gas parameters for the Louisnard–Gomez law.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LouisnardGomezParams {
    /// Gas diffusivity in the liquid `D` [m²/s].
    pub diffusivity: f64,
    /// Saturation molar concentration of the gas in the liquid `C0`
    /// [mol/m³] (the concentration at saturation ratio 1).
    pub saturation_concentration: f64,
    /// Liquid temperature `T` [K].
    pub temperature: f64,
    /// Ambient pressure `p0` [Pa].
    pub ambient_pressure: f64,
    /// Surface tension `σ` [N/m].
    pub surface_tension: f64,
}

/// Cycle-averaged quantities of one steady oscillation cycle, as they
/// enter the Louisnard–Gomez law.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CycleAverageTerms {
    /// `R⁴`-weighted cycle-averaged gas pressure `⟨pg/p0⟩₄`
    /// (dimensionless; Louisnard & Gomez Eq. (8)). Compare against the
    /// saturation ratio: growth iff `C∞* > ⟨pg/p0⟩₄` (their Eq. (9)).
    pub pg_over_p0_r4: f64,
    /// The dimensionless integral `I` (their Eq. (7)), evaluated
    /// numerically from the radius history.
    pub integral_i: f64,
    /// Maximum radius over the cycle, in units of `R0` (`Rmax*`).
    pub r_max_star: f64,
    /// Eq. (12) approximation of `⟨pg/pg0⟩₄`:
    /// `(1 + α_S)·ξ·(R0/Rmax)³` with `ξ = 105/64`. A cross-check on
    /// [`Self::pg_over_p0_r4`] (divide by `pg0/p0 = 1 + α_S` before
    /// comparing); the numerical average is primary.
    pub pg_over_pg0_r4_eq12: f64,
}

/// Which Louisnard–Gomez validity bounds a call exceeds.
///
/// Returned by [`louisnard_gomez_regime`]. The formation model's burst
/// application (MHz, MPa) sets the frequency and pressure flags; that
/// extrapolation is deliberate and documented in the module docs, but
/// it must never be silent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LouisnardGomezRegime {
    /// Drive frequency outside 20–100 kHz.
    pub frequency_outside_validated: bool,
    /// Drive pressure amplitude outside 1–5 bar.
    pub pressure_outside_validated: bool,
    /// Equilibrium radius ≥ 15 µm.
    pub radius_outside_validated: bool,
    /// Saturation ratio outside (0, 1].
    pub saturation_outside_validated: bool,
}

impl LouisnardGomezRegime {
    /// True when every input is inside the validated range.
    #[must_use]
    pub fn in_validated_regime(&self) -> bool {
        !self.frequency_outside_validated
            && !self.pressure_outside_validated
            && !self.radius_outside_validated
            && !self.saturation_outside_validated
    }
}

/// Report which validated-regime bounds a Louisnard–Gomez call exceeds.
///
/// `acoustic_pressure` is the drive pressure *amplitude* [Pa] (the
/// quantity the paper's 1–5 bar range refers to), `frequency` the drive
/// frequency [Hz], `radius` the equilibrium radius `R0` [m], and
/// `saturation_ratio` the far-field saturation ratio `C∞*`.
#[must_use]
pub fn louisnard_gomez_regime(
    radius: f64,
    acoustic_pressure: f64,
    frequency: f64,
    saturation_ratio: f64,
) -> LouisnardGomezRegime {
    LouisnardGomezRegime {
        frequency_outside_validated: !(LOUISNARD_GOMEZ_MIN_FREQUENCY_HZ
            ..=LOUISNARD_GOMEZ_MAX_FREQUENCY_HZ)
            .contains(&frequency),
        pressure_outside_validated: !(LOUISNARD_GOMEZ_MIN_PRESSURE_PA
            ..=LOUISNARD_GOMEZ_MAX_PRESSURE_PA)
            .contains(&acoustic_pressure),
        // Explicit NaN disjunct, matching the NaN-tightness of the
        // `.contains()`-based flags above and the Bader checked
        // variant: a plain `radius >= MAX` test is false for NaN and
        // would wave a NaN radius through as in-regime. For every
        // non-NaN input the flag is unchanged. (Positivity of the
        // radius is the checked variant's separate reject, not this
        // flag's.)
        radius_outside_validated: radius >= LOUISNARD_GOMEZ_MAX_RADIUS_M || radius.is_nan(),
        saturation_outside_validated: !(saturation_ratio > 0.0 && saturation_ratio <= 1.0),
    }
}

/// Gas pressure inside the bubble from the van der Waals isothermal law
/// (Louisnard & Gomez Eq. (2)): `pg = pg0·(R0³ − h³)/(R³ − h³)` with
/// hard-core radius `h = R0/8.76` and `pg0 = p0 + 2σ/R0`.
///
/// `radius` is floored at `1.001·h` so a numerical overshoot below the
/// hard core cannot make `pg` negative.
#[must_use]
pub fn van_der_waals_gas_pressure(
    radius: f64,
    equilibrium_radius: f64,
    ambient_pressure: f64,
    surface_tension: f64,
) -> f64 {
    let r0 = equilibrium_radius;
    let h = r0 / 8.76;
    let pg0 = ambient_pressure + 2.0 * surface_tension / r0;
    let r = radius.max(h * 1.001);
    pg0 * (r0.powi(3) - h.powi(3)) / (r.powi(3) - h.powi(3))
}

/// Trapezoidal time-average of `values` over the samples `time`.
fn time_average(values: &[f64], time: &[f64]) -> f64 {
    let mut integral = 0.0;
    for i in 1..values.len() {
        integral += 0.5 * (values[i - 1] + values[i]) * (time[i] - time[i - 1]);
    }
    integral / (time[time.len() - 1] - time[0])
}

/// Cycle-averaged terms of the Louisnard–Gomez law from a radius
/// history over one steady cycle.
///
/// `radius` [m] and `time` [s] are the samples of one cycle (uniform
/// or not; trapezoidal averages are used), `equilibrium_radius` is
/// `R0` [m]. Degenerate inputs (fewer than 2 samples, non-positive
/// `R0`) return zeroed terms rather than NaN.
#[must_use]
pub fn cycle_average_terms(
    radius: &[f64],
    time: &[f64],
    equilibrium_radius: f64,
    ambient_pressure: f64,
    surface_tension: f64,
) -> CycleAverageTerms {
    let zeroed = CycleAverageTerms {
        pg_over_p0_r4: 0.0,
        integral_i: 0.0,
        r_max_star: 0.0,
        pg_over_pg0_r4_eq12: 0.0,
    };
    if radius.len() < 2 || radius.len() != time.len() || equilibrium_radius <= 0.0 {
        return zeroed;
    }
    let r0 = equilibrium_radius;
    let p0 = ambient_pressure;
    let alpha_s = 2.0 * surface_tension / (p0 * r0);

    let r_star: Vec<f64> = radius.iter().map(|&r| r / r0).collect();
    let r_max_star = r_star.iter().copied().fold(0.0_f64, f64::max);
    if r_max_star <= 0.0 {
        return zeroed;
    }

    // ⟨pg/p0⟩₄ = ⟨(pg/p0)·R*⁴⟩ / ⟨R*⁴⟩ (Eq. 8).
    let r4: Vec<f64> = r_star.iter().map(|&x| x.powi(4)).collect();
    let pg_r4: Vec<f64> = radius
        .iter()
        .zip(&r4)
        .map(|(&r, &w)| van_der_waals_gas_pressure(r, r0, p0, surface_tension) / p0 * w)
        .collect();
    let mean_r4 = time_average(&r4, time);
    let pg_over_p0_r4 = if mean_r4 > 0.0 {
        time_average(&pg_r4, time) / mean_r4
    } else {
        0.0
    };

    // I (Eq. 7) via u = 3s/Rmax*³: I = (1/(3·Rmax*)) ∫₀^∞ du/⟨(u+q)^(4/3)⟩
    // with q = (R*/Rmax*)³, on a log u-grid with the analytic u^(−4/3)
    // tail beyond u_max.
    let q: Vec<f64> = r_star.iter().map(|&x| (x / r_max_star).powi(3)).collect();
    let n_u = 400;
    let u_max = 1e4_f64;
    let log_lo = -8.0_f64;
    let log_hi = u_max.log10();
    let mut u_grid = Vec::with_capacity(n_u);
    let mut inv_g = Vec::with_capacity(n_u);
    for i in 0..n_u {
        let u = 10.0_f64.powf(log_lo + (log_hi - log_lo) * (i as f64) / ((n_u - 1) as f64));
        let vals: Vec<f64> = q.iter().map(|&qi| (u + qi).powf(4.0 / 3.0)).collect();
        let g = time_average(&vals, time);
        u_grid.push(u);
        inv_g.push(if g > 0.0 { 1.0 / g } else { 0.0 });
    }
    let mut integral = 0.0;
    for i in 1..n_u {
        integral += 0.5 * (inv_g[i - 1] + inv_g[i]) * (u_grid[i] - u_grid[i - 1]);
    }
    let integral_i = integral / (3.0 * r_max_star) + u_max.powf(-1.0 / 3.0) / r_max_star;

    // Eq. (12) cross-check: ⟨pg/pg0⟩₄ ≈ (1 + α_S)·ξ·(R0/Rmax)³, ξ = 105/64.
    let xi = 105.0 / 64.0;
    let pg_over_pg0_r4_eq12 = (1.0 + alpha_s) * xi / r_max_star.powi(3);

    CycleAverageTerms {
        pg_over_p0_r4,
        integral_i,
        r_max_star,
        pg_over_pg0_r4_eq12,
    }
}

/// Cycle-averaged growth rate `dR0/dt` [m/s] of the equilibrium radius
/// (Louisnard & Gomez Eq. (16)). Positive = growth, negative =
/// dissolution; the sign flips at the rectified-diffusion threshold
/// `C∞* = ⟨pg/p0⟩₄` (their Eq. (9)).
///
/// `radius`/`time` give one steady cycle of the oscillation (see
/// [`cycle_average_terms`]), `saturation_ratio` is the far-field
/// saturation ratio `C∞*`, and `params` carries the medium properties.
/// Degenerate inputs return 0.0 rather than NaN. This function does
/// **not** enforce the validated regime — see
/// [`louisnard_gomez_growth_rate_checked`] and the module docs on the
/// MHz/MPa burst extrapolation.
#[must_use]
pub fn louisnard_gomez_growth_rate(
    radius: &[f64],
    time: &[f64],
    equilibrium_radius: f64,
    saturation_ratio: f64,
    params: &LouisnardGomezParams,
) -> f64 {
    if equilibrium_radius <= 0.0
        || params.diffusivity <= 0.0
        || params.saturation_concentration <= 0.0
        || params.temperature <= 0.0
        || params.ambient_pressure <= 0.0
    {
        return 0.0;
    }
    let terms = cycle_average_terms(
        radius,
        time,
        equilibrium_radius,
        params.ambient_pressure,
        params.surface_tension,
    );
    if terms.integral_i <= 0.0 {
        return 0.0;
    }
    let r0 = equilibrium_radius;
    let p0 = params.ambient_pressure;
    let alpha_s = 2.0 * params.surface_tension / (p0 * r0);
    let prefactor =
        params.diffusivity * GAS_CONSTANT * params.temperature * params.saturation_concentration
            / (p0 * r0 * (1.0 + 2.0 * alpha_s / 3.0));
    prefactor * (saturation_ratio - terms.pg_over_p0_r4) / terms.integral_i
}

/// Per-cycle gas influx [molecules/cycle] under the Louisnard–Gomez
/// law: `Δn = (dR0/dt)·(dn/dR0)·(1/f)`, with `dn/dR0` from the
/// equilibrium gas-content map `n(R) = (p0 + 2σ/R)·(4π/3)·R³/(k_B·T)`
/// (the implicit residual-radius relation; see
/// [`equilibrium_gas_content`](super::equilibrium_gas_content)).
#[must_use]
pub fn louisnard_gomez_per_cycle_influx(
    radius: &[f64],
    time: &[f64],
    equilibrium_radius: f64,
    saturation_ratio: f64,
    frequency: f64,
    params: &LouisnardGomezParams,
) -> f64 {
    if frequency <= 0.0 {
        return 0.0;
    }
    let d_r0_dt =
        louisnard_gomez_growth_rate(radius, time, equilibrium_radius, saturation_ratio, params);
    let r0 = equilibrium_radius;
    let dn_dr0 = (4.0 * std::f64::consts::PI * params.ambient_pressure * r0 * r0
        + 16.0 * std::f64::consts::PI * params.surface_tension * r0 / 3.0)
        / (BOLTZMANN * params.temperature);
    d_r0_dt * dn_dr0 / frequency
}

/// Checked form of [`louisnard_gomez_growth_rate`]: same growth rate
/// [m/s], but rejects inputs outside the Louisnard & Gomez validated
/// regime instead of silently returning an extrapolation.
///
/// # Errors
/// - [`KwaversError::Physics`] with `PhysicsError::InvalidParameter`
///   when any input is degenerate (non-positive `R0`, `D`, `C0`, `T`,
///   `p0`; fewer than 2 history samples), or when
///   [`louisnard_gomez_regime`] reports the frequency, pressure,
///   radius, or saturation ratio outside the validated range
///   (20–100 kHz, 1–5 bar, `R0 < 15 µm`, `0 < C∞* ≤ 1`).
///   `acoustic_pressure` is the drive amplitude [Pa] the regime bounds
///   refer to; it is not otherwise used by the law, which is a
///   functional of the radius history.
pub fn louisnard_gomez_growth_rate_checked(
    radius: &[f64],
    time: &[f64],
    equilibrium_radius: f64,
    saturation_ratio: f64,
    acoustic_pressure: f64,
    frequency: f64,
    params: &LouisnardGomezParams,
) -> KwaversResult<f64> {
    let regime = louisnard_gomez_regime(
        equilibrium_radius,
        acoustic_pressure,
        frequency,
        saturation_ratio,
    );
    let reject = |parameter: &str, value: f64, reason: String| {
        KwaversError::Physics(PhysicsError::InvalidParameter {
            parameter: parameter.to_owned(),
            value,
            reason,
        })
    };
    // Each reject carries an explicit NaN disjunct: a bare
    // `x <= 0.0` test is false for NaN and would wave it through —
    // the NaN-tightness convention of the regime flags and the
    // Bader checked variant.
    if equilibrium_radius <= 0.0 || equilibrium_radius.is_nan() {
        return Err(reject(
            "equilibrium_radius",
            equilibrium_radius,
            "Equilibrium radius must be positive".to_owned(),
        ));
    }
    // Medium-parameter rejects, per this function's documented
    // contract: the unchecked law silently returns 0.0 for degenerate
    // D / C0 / T / p0, which a checked variant must not wave through
    // as Ok(0.0). NaN-tight for the same reason as above.
    if params.diffusivity <= 0.0 || params.diffusivity.is_nan() {
        return Err(reject(
            "diffusivity",
            params.diffusivity,
            "Diffusivity must be positive".to_owned(),
        ));
    }
    if params.saturation_concentration <= 0.0 || params.saturation_concentration.is_nan() {
        return Err(reject(
            "saturation_concentration",
            params.saturation_concentration,
            "Saturation concentration must be positive".to_owned(),
        ));
    }
    if params.temperature <= 0.0 || params.temperature.is_nan() {
        return Err(reject(
            "temperature",
            params.temperature,
            "Temperature must be positive".to_owned(),
        ));
    }
    if params.ambient_pressure <= 0.0 || params.ambient_pressure.is_nan() {
        return Err(reject(
            "ambient_pressure",
            params.ambient_pressure,
            "Ambient pressure must be positive".to_owned(),
        ));
    }
    if radius.len() < 2 || radius.len() != time.len() {
        return Err(reject(
            "radius history",
            radius.len() as f64,
            "A radius history of at least 2 samples over one cycle is required".to_owned(),
        ));
    }
    if regime.frequency_outside_validated {
        return Err(reject(
            "frequency",
            frequency,
            format!(
                "Louisnard-Gomez law validated at {LOUISNARD_GOMEZ_MIN_FREQUENCY_HZ}..={LOUISNARD_GOMEZ_MAX_FREQUENCY_HZ} Hz; got {frequency}"
            ),
        ));
    }
    if regime.pressure_outside_validated {
        return Err(reject(
            "acoustic_pressure",
            acoustic_pressure,
            format!(
                "Louisnard-Gomez law validated at {LOUISNARD_GOMEZ_MIN_PRESSURE_PA}..={LOUISNARD_GOMEZ_MAX_PRESSURE_PA} Pa; got {acoustic_pressure}"
            ),
        ));
    }
    if regime.radius_outside_validated {
        return Err(reject(
            "equilibrium_radius",
            equilibrium_radius,
            format!(
                "Louisnard-Gomez law validated for R0 < {LOUISNARD_GOMEZ_MAX_RADIUS_M} m; got {equilibrium_radius}"
            ),
        ));
    }
    if regime.saturation_outside_validated {
        return Err(reject(
            "saturation_ratio",
            saturation_ratio,
            "Louisnard-Gomez law validated for near-saturated liquid, 0 < C* <= 1".to_owned(),
        ));
    }
    Ok(louisnard_gomez_growth_rate(
        radius,
        time,
        equilibrium_radius,
        saturation_ratio,
        params,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acoustics::bubble_dynamics::{
        AdaptiveBubbleConfig, BubbleParameters, BubbleState, GasSpecies, ImplicitBubbleIntegrator,
        InstantaneousPressure, KellerMiksisModel,
    };
    use aequitas::systems::si::quantities::PressureRate;
    use eunomia::assert_relative_eq;
    use kwavers_core::constants::fundamental::ATMOSPHERIC_PRESSURE;

    const P0: f64 = ATMOSPHERIC_PRESSURE;

    fn paper_params() -> LouisnardGomezParams {
        // Louisnard & Gomez Fig. 4 conditions (air in water, 293 K).
        LouisnardGomezParams {
            diffusivity: 2.0e-9,
            saturation_concentration: 0.83,
            temperature: 293.0,
            ambient_pressure: P0,
            surface_tension: 0.073,
        }
    }

    #[test]
    fn test_van_der_waals_at_equilibrium() {
        // At R = R0 the van der Waals law returns pg0 = p0 + 2σ/R0.
        let r0 = 5e-6;
        let sigma = 0.073;
        let pg = van_der_waals_gas_pressure(r0, r0, P0, sigma);
        assert_relative_eq!(pg, P0 + 2.0 * sigma / r0, epsilon = 1e-9);
    }

    #[test]
    fn test_constant_history_exact_terms() {
        // R ≡ R0: R* ≡ 1, Rmax* = 1, q ≡ 1, so ⟨(u+1)^(4/3)⟩ = (u+1)^(4/3)
        // and I = (1/3)·∫₀^∞ (u+1)^(−4/3) du = (1/3)·3 = 1 exactly.
        // ⟨pg/p0⟩₄ = pg0/p0 = 1 + α_S.
        let r0 = 5e-6;
        let n = 200;
        let radius = vec![r0; n];
        let time: Vec<f64> = (0..n)
            .map(|i| i as f64 / (n - 1) as f64 / 26_500.0)
            .collect();
        let terms = cycle_average_terms(&radius, &time, r0, P0, 0.073);
        let alpha_s = 2.0 * 0.073 / (P0 * r0);
        assert_relative_eq!(terms.pg_over_p0_r4, 1.0 + alpha_s, epsilon = 1e-9);
        assert!(
            (terms.integral_i - 1.0).abs() < 0.01,
            "I for a constant history must be 1; got {}",
            terms.integral_i
        );
        assert_relative_eq!(terms.r_max_star, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_integral_i_against_direct_s_integration() {
        // Cross-check the u-substitution against a brute-force evaluation
        // of the defining integral I = ∫₀^∞ ds/⟨(3s + R*³)^(4/3)⟩ on a
        // synthetic expansion history (half-sine bump to R* = 6).
        let r0 = 5e-6;
        let n = 400;
        let radius: Vec<f64> = (0..n)
            .map(|i| {
                let phase = i as f64 / (n - 1) as f64;
                r0 * (1.0 + 5.0 * (std::f64::consts::PI * phase).sin())
            })
            .collect();
        let time: Vec<f64> = (0..n)
            .map(|i| i as f64 / (n - 1) as f64 / 26_500.0)
            .collect();
        let terms = cycle_average_terms(&radius, &time, r0, P0, 0.073);

        // Direct s-integration with the substitution-free definition.
        let r_star: Vec<f64> = radius.iter().map(|&r| r / r0).collect();
        let mut integral = 0.0;
        let s_max = 200.0;
        let n_s = 200_000;
        for i in 1..=n_s {
            let s_prev = s_max * (i - 1) as f64 / n_s as f64;
            let s = s_max * i as f64 / n_s as f64;
            let g = |sv: f64| {
                let vals: Vec<f64> = r_star
                    .iter()
                    .map(|&x| (3.0 * sv + x.powi(3)).powf(4.0 / 3.0))
                    .collect();
                time_average(&vals, &time)
            };
            integral += 0.5 * (1.0 / g(s_prev) + 1.0 / g(s)) * (s - s_prev);
        }
        // Tail beyond s_max: ⟨(3s)^(4/3)⟩ → integrand ~ (3s)^(−4/3),
        // ∫ = (3)^(−4/3) · 3 · s_max^(−1/3).
        integral += 3.0_f64.powf(-4.0 / 3.0) * 3.0 * s_max.powf(-1.0 / 3.0);
        assert!(
            (terms.integral_i - integral).abs() / integral < 0.05,
            "u-substitution I = {} vs direct s-integration {integral}",
            terms.integral_i
        );
    }

    #[test]
    fn test_regime_flags() {
        // Paper conditions: in regime.
        let r = louisnard_gomez_regime(5e-6, 200_000.0, 26_500.0, 1.0);
        assert!(r.in_validated_regime());
        // Formation-model burst conditions (1 MHz, 10 MPa): out of regime
        // in frequency and pressure, in regime in radius and saturation.
        let r = louisnard_gomez_regime(5e-6, 10_530_000.0, 1_000_000.0, 0.2);
        assert!(r.frequency_outside_validated);
        assert!(r.pressure_outside_validated);
        assert!(!r.radius_outside_validated);
        assert!(!r.saturation_outside_validated);
        assert!(!r.in_validated_regime());
    }

    #[test]
    fn test_checked_rejects_burst_extrapolation() {
        let params = paper_params();
        let n = 100;
        let radius = vec![5e-6; n];
        let time: Vec<f64> = (0..n)
            .map(|i| i as f64 / (n - 1) as f64 / 26_500.0)
            .collect();
        // In-regime call succeeds.
        let ok = louisnard_gomez_growth_rate_checked(
            &radius, &time, 5e-6, 1.0, 200_000.0, 26_500.0, &params,
        );
        assert!(ok.is_ok());
        // Burst extrapolation (1 MHz) is refused by the checked form.
        let err = louisnard_gomez_growth_rate_checked(
            &radius,
            &time,
            5e-6,
            0.2,
            10_530_000.0,
            1_000_000.0,
            &params,
        );
        assert!(err.is_err());
    }

    #[test]
    fn test_checked_rejects_nan_and_degenerate_params() {
        // Guard soundness (physics review 2026-11-27): a checked
        // variant must refuse NaN and degenerate medium parameters,
        // not return Ok(0.0) / Ok(NaN) — the unchecked law's silent
        // 0.0 is exactly what the checked form exists to prevent.
        let params = paper_params();
        let n = 100;
        let radius = vec![5e-6; n];
        let time: Vec<f64> = (0..n)
            .map(|i| i as f64 / (n - 1) as f64 / 26_500.0)
            .collect();
        let call = |r0: f64, params: &LouisnardGomezParams| {
            louisnard_gomez_growth_rate_checked(&radius, &time, r0, 1.0, 200_000.0, 26_500.0, params)
        };
        assert!(call(f64::NAN, &params).is_err());
        let mut q = params.clone();
        q.diffusivity = 0.0;
        assert!(call(5e-6, &q).is_err());
        q = params.clone();
        q.diffusivity = f64::NAN;
        assert!(call(5e-6, &q).is_err());
        q = params.clone();
        q.saturation_concentration = 0.0;
        assert!(call(5e-6, &q).is_err());
        q = params.clone();
        q.temperature = -1.0;
        assert!(call(5e-6, &q).is_err());
        q = params.clone();
        q.ambient_pressure = 0.0;
        assert!(call(5e-6, &q).is_err());
        // The regime classifier flags a NaN radius as outside.
        assert!(
            louisnard_gomez_regime(f64::NAN, 200_000.0, 26_500.0, 1.0).radius_outside_validated
        );
    }

    /// Keller–Miksis parameters at the Louisnard & Gomez Fig. 4
    /// closure (the prototype's validated implementation,
    /// formation/b6a_law.py): isothermal gas (κ = 1), vapour
    /// neglected, ρ = 1000 kg/m³, c = 1481 m/s (the paper's value),
    /// σ = 0.073 N/m, μ = 0.001 Pa·s.
    fn fig4_params(r0: f64) -> BubbleParameters {
        BubbleParameters {
            r0,
            p0: P0,
            rho_liquid: 1000.0,
            c_liquid: 1481.0,
            mu_liquid: 0.001,
            sigma: 0.073,
            pv: 0.0,
            gas_species: GasSpecies::Custom {
                gamma: 1.0,
                molecular_weight: 0.029,
            },
            use_thermal_effects: false,
            use_mass_transfer: false,
            driving_frequency: 26_500.0,
            ..Default::default()
        }
    }

    /// Integrator configuration for the Fig. 4 histories: tolerances
    /// an order tighter than the library defaults (the law reads the
    /// expansion phase of a stiff trajectory; the prototype's LSODA
    /// ground truth ran at rtol 1e-9), a step floor far below the
    /// collapse spike's timescale (LSODA itself needed steps to
    /// 1.6×10⁻¹⁷ s; a step starting at the turnaround needs ~10⁻¹⁹ s
    /// at these tolerances, so the floor must not bind), and a
    /// substep budget sized for sub-cycling through two collapses
    /// per main step at the strongest anchor.
    fn fig4_config(period: f64) -> AdaptiveBubbleConfig {
        AdaptiveBubbleConfig {
            dt_max: period,
            dt_min: 1e-20,
            rtol: 1e-8,
            atol: 1e-12,
            max_substeps: 100_000,
            ..AdaptiveBubbleConfig::default()
        }
    }

    /// Advance the library Keller–Miksis model by one output-grid
    /// step (T/400) under the Fig. 4 tone, following the integrator's
    /// zero-order-hold drive contract: instantaneous pressure
    /// p(t) = −p_ac·sin(ωt) and dp/dt = −p_ac·ω·cos(ωt) sampled at
    /// the step start.
    fn fig4_grid_step(
        integrator: &mut ImplicitBubbleIntegrator<'_, KellerMiksisModel>,
        state: &mut BubbleState,
        p_ac: f64,
        omega: f64,
        grid_dt: f64,
        t: f64,
    ) {
        let p = InstantaneousPressure::from_pascals(-p_ac * (omega * t).sin());
        let dp_dt = PressureRate::from_base(-p_ac * omega * (omega * t).cos());
        integrator
            .integrate_adaptive(state, p, dp_dt, grid_dt, t)
            .expect("library KM + implicit SDIRK2 must traverse the Fig. 4 cycles");
    }

    /// Integrate the library Keller–Miksis model at the Louisnard &
    /// Gomez Fig. 4 conditions and return the post-discard history
    /// window (cycles 5–60, uniform 400-points-per-cycle grid),
    /// matching the prototype's evaluation protocol (b6a_law.km_solve:
    /// 60 cycles, first 5 discarded, the law evaluated over the whole
    /// post-discard window — a last-cycle evaluation is poisoned at
    /// the strongest anchor by individual afterbounce-heavy cycles).
    ///
    /// Histories come from the library's own KM model driven by the
    /// implicit SDIRK2 integrator (KW-PHYS-009 item 5): the previous
    /// test-local explicit RK4 transcription is gone. The explicit
    /// step-doubling family cannot traverse the collapse spike
    /// without injecting energy (at (5 µm, 5 bar) it returned a
    /// window rate of ~10⁷ µm/s against the paper's ~300 µm/s, and
    /// it stalled outright at (3 µm, 2 bar)); the L-stable implicit
    /// path damps the unresolved spike instead, and the law's R⁴
    /// weighting reads the expansion phase it resolves.
    fn km_history(r0: f64, p_ac: f64, frequency: f64) -> (Vec<f64>, Vec<f64>) {
        let n_cycles = 60;
        let discard = 5;
        let pts_per_cycle = 400;
        let period = 1.0 / frequency;
        let grid_dt = period / pts_per_cycle as f64;
        let total_pts = n_cycles * pts_per_cycle;

        let params = fig4_params(r0);
        let model = KellerMiksisModel::new(params.clone());
        let mut state = BubbleState::new(&params);
        let mut integrator = ImplicitBubbleIntegrator::new(&model, fig4_config(period));
        let omega = 2.0 * std::f64::consts::PI * frequency;

        let mut grid_r = Vec::with_capacity(total_pts + 1);
        grid_r.push(state.radius);
        let mut t = 0.0;
        for _ in 0..total_pts {
            fig4_grid_step(&mut integrator, &mut state, p_ac, omega, grid_dt, t);
            t += grid_dt;
            grid_r.push(state.radius);
        }
        let start = discard * pts_per_cycle;
        let window_pts = total_pts - start;
        let radius = grid_r[start..=total_pts].to_vec();
        let time: Vec<f64> = (0..=window_pts).map(|i| i as f64 * grid_dt).collect();
        (radius, time)
    }

    /// Trajectory extremes (min radius, max |Ṙ|) over cycles
    /// 2..=n_cycles at the Fig. 4 conditions, from a fresh implicit
    /// integrator per cycle so the start-up cycle's extremes do not
    /// pollute the window. The output grid undersamples the ~ns
    /// collapse spike by construction, so the extremes come from the
    /// integrator's accepted sub-steps. LSODA reference,
    /// spike-resolved (solver internal nodes and fs-scale
    /// dense-output windows around the turnaround; two tolerance
    /// levels in exact agreement — physics review 2026-11-27): at
    /// (5 µm, 5 bar) Rmin = 0.016 µm, max |Ṙ| = 54,157 m/s; at
    /// (5 µm, 2 bar) Rmin = 0.0297 µm, max |Ṙ| = 20,842 m/s; at
    /// (3 µm, 2 bar) Rmin = 0.0149 µm, max |Ṙ| = 29,479 m/s.
    /// (Earlier reference readings — the B22 Part 3 diagnosis's
    /// 0.023 µm / 6044 m/s and the Part 3 build's 0.018 µm /
    /// ≈1.5×10⁴ m/s at the 2-bar anchors — were uniform-grid
    /// samplings that missed the ~ps-wide spike apex; their Rmax
    /// values were exact.)
    fn km_trajectory_extremes(r0: f64, p_ac: f64, frequency: f64, n_cycles: usize) -> (f64, f64) {
        let period = 1.0 / frequency;
        let pts_per_cycle = 400;
        let grid_dt = period / pts_per_cycle as f64;

        let params = fig4_params(r0);
        let model = KellerMiksisModel::new(params.clone());
        let mut state = BubbleState::new(&params);
        let omega = 2.0 * std::f64::consts::PI * frequency;

        let mut t = 0.0;
        let mut min_r = f64::INFINITY;
        let mut max_v: f64 = 0.0;
        for cycle in 0..n_cycles {
            let mut integrator = ImplicitBubbleIntegrator::new(&model, fig4_config(period));
            for _ in 0..pts_per_cycle {
                fig4_grid_step(&mut integrator, &mut state, p_ac, omega, grid_dt, t);
                t += grid_dt;
            }
            if cycle >= 1 {
                min_r = min_r.min(integrator.min_radius_seen());
                max_v = max_v.max(integrator.max_wall_speed_seen());
            }
        }
        (min_r, max_v)
    }

    #[test]
    fn test_louisnard_gomez_fig4_anchor() {
        // Louisnard & Gomez (2003) Fig. 4 anchor: R0 = 5 µm,
        // p = 2 bar at 26.5 kHz in saturated water grows at
        // ~100 µm/s. The paper's anchors are order-of-magnitude
        // reads (factor ~2 per the prototype's validation notes);
        // the prototype's LSODA gave 94.6 µm/s for this anchor.
        // Assert the factor-of-2 band. The history is produced by
        // the library's own KM + implicit SDIRK2 integrator
        // (KW-PHYS-009 item 5); the six-anchor sweep is the ignored
        // benchmark below.
        let params = paper_params();
        let (radius, time) = km_history(5e-6, 200_000.0, 26_500.0);
        let rate_um_s = louisnard_gomez_growth_rate(&radius, &time, 5e-6, 1.0, &params) * 1e6;
        assert!(
            rate_um_s > 50.0 && rate_um_s < 200.0,
            "Fig. 4 anchor R0 = 5 µm, p = 2 bar: got {rate_um_s} µm/s, paper ~100 µm/s"
        );
    }

    #[test]
    #[ignore = "expensive benchmark: six Keller-Miksis histories; run with -- --ignored"]
    fn test_louisnard_gomez_fig4_benchmark() {
        // Louisnard & Gomez (2003) Fig. 4 growth rates at 26.5 kHz,
        // saturated water — all six of the paper's anchors, with the
        // histories generated by the library's own KM model and the
        // implicit SDIRK2 integrator (KW-PHYS-009 item 5). Assert a
        // factor-of-2 band around each anchor and the correct
        // ordering in R0 and p. Prototype LSODA references:
        // (3, 2) 222.9, (5, 2) 94.6, (5, 5) 275.3 µm/s against the
        // paper's ~200 / ~100 / ~300 µm/s reads.
        let params = paper_params();
        let anchors: [(f64, f64, f64); 6] = [
            (3e-6, 200_000.0, 200.0),
            (5e-6, 200_000.0, 100.0),
            (10e-6, 200_000.0, 30.0),
            (20e-6, 200_000.0, 10.0),
            (5e-6, 130_000.0, 30.0),
            (5e-6, 500_000.0, 300.0),
        ];
        let mut rates = Vec::new();
        for (r0, p_ac, anchor_um_s) in anchors {
            let (radius, time) = km_history(r0, p_ac, 26_500.0);
            let rate_um_s = louisnard_gomez_growth_rate(&radius, &time, r0, 1.0, &params) * 1e6;
            rates.push(rate_um_s);
            assert!(
                rate_um_s > 0.5 * anchor_um_s && rate_um_s < 2.0 * anchor_um_s,
                "Fig. 4 anchor R0 = {r0} m, p = {p_ac} Pa: got {rate_um_s} µm/s, paper ~{anchor_um_s} µm/s"
            );
        }
        // Ordering: at 2 bar the rate falls with R0; at R0 = 5 µm
        // the rate rises with p (1.3 bar < 2 bar < 5 bar).
        assert!(rates[0] > rates[1] && rates[1] > rates[2] && rates[2] > rates[3]);
        assert!(rates[4] < rates[1] && rates[1] < rates[5]);

        // Trajectory check at the strongest anchor: the accepted
        // sub-steps must resolve the collapse spike to the
        // spike-resolved LSODA reference's scale (Rmin = 0.016 µm,
        // max |Ṙ| = 54,157 m/s), within a factor 2.5 either way —
        // an integrator that skips or damps the collapse entirely,
        // or one whose rebound runs away, fails here even if a
        // growth-rate band elsewhere hid it. (Library measurement,
        // physics review 2026-11-27: Rmin = 0.0160 µm,
        // max |Ṙ| = 54,157 m/s — the LSODA value to printed
        // precision, after the +4μ/(ρc) LHS term was restored.)
        let (min_r, max_v) = km_trajectory_extremes(5e-6, 500_000.0, 26_500.0, 3);
        assert!(
            min_r > 0.0064e-6 && min_r < 0.040e-6,
            "Fig. 4 (5 µm, 5 bar) trajectory: min radius {min_r} m must sit at the LSODA spike scale (0.016 µm, factor 2.5)"
        );
        assert!(
            max_v > 21_663.0 && max_v < 135_393.0,
            "Fig. 4 (5 µm, 5 bar) trajectory: max |Ṙ| {max_v} m/s must sit at the LSODA spike scale (54,157 m/s, factor 2.5)"
        );
    }
}
