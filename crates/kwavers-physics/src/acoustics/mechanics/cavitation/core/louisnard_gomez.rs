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
        radius_outside_validated: radius >= LOUISNARD_GOMEZ_MAX_RADIUS_M,
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
    if equilibrium_radius <= 0.0 {
        return Err(reject(
            "equilibrium_radius",
            equilibrium_radius,
            "Equilibrium radius must be positive".to_owned(),
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
        // In regime the checked form is the unchecked law, bit for bit.
        let checked = louisnard_gomez_growth_rate_checked(
            &radius, &time, 5e-6, 1.0, 200_000.0, 26_500.0, &params,
        )
        .unwrap();
        let plain = louisnard_gomez_growth_rate(&radius, &time, 5e-6, 1.0, &params);
        assert_eq!(checked.to_bits(), plain.to_bits());
        // Burst extrapolation (1 MHz) is refused by the checked form. The
        // frequency bound is tested before the pressure bound, so it is
        // the parameter the error names.
        let burst = louisnard_gomez_growth_rate_checked(
            &radius,
            &time,
            5e-6,
            0.2,
            10_530_000.0,
            1_000_000.0,
            &params,
        );
        assert!(matches!(
            burst,
            Err(KwaversError::Physics(PhysicsError::InvalidParameter {
                ref parameter,
                value,
                ..
            })) if parameter == "frequency" && value.to_bits() == 1_000_000.0_f64.to_bits()
        ));
    }

    /// Keller–Miksis RHS transcribed from the prototype's validated
    /// implementation (formation/b6a_law.py `km_rhs`): isothermal gas
    /// (κ = 1), vapour neglected, no elasticity (G = 0).
    ///
    /// A test-local integrator is used because the library's native
    /// KM cannot currently produce histories in this regime: its
    /// acceleration returns an error above wall Mach 0.95 and its
    /// adaptive integrator's stability monitor rejects |Ṙ| > 0.1·c,
    /// while the Fig. 4 collapses pass both bounds (the prototype's
    /// LSODA integration, rtol 1e-9, handles them; the F–S law
    /// suppresses the collapse phase by R⁴ weighting, so the law's
    /// outputs are insensitive to collapse detail). Filed as
    /// KW-PHYS-009.
    fn km_rhs_test(t: f64, r: f64, v: f64, r0: f64, p_ac: f64, f: f64) -> (f64, f64) {
        const RHO: f64 = 1000.0;
        const C_SND: f64 = 1481.0; // Louisnard & Gomez value
        const SIGMA: f64 = 0.073;
        const MU: f64 = 0.001;
        const KAPPA: f64 = 1.0;
        let r = r.max(r0 * 1e-3);
        let om = 2.0 * std::f64::consts::PI * f;
        let p_ac_t = -p_ac * (om * t).sin();
        let p_ac_dot = -p_ac * om * (om * t).cos();
        let pg = (P0 + 2.0 * SIGMA / r0) * (r0 / r).powf(3.0 * KAPPA);
        let pw = pg - 2.0 * SIGMA / r - 4.0 * MU * v / r;
        let p_inf = P0 + p_ac_t;
        let a = pw - p_inf;
        let d_a =
            -3.0 * KAPPA * pg * v / r + 2.0 * SIGMA * v / (r * r) + 4.0 * MU * v * v / (r * r)
                - p_ac_dot;
        let coef = (1.0 - v / C_SND) * r + 4.0 * MU / (RHO * C_SND);
        let rhs = (1.0 + v / C_SND) * a / RHO + (r / (RHO * C_SND)) * d_a
            - 1.5 * (1.0 - v / (3.0 * C_SND)) * v * v;
        (v, rhs / coef)
    }

    fn rk4_step_test(t: f64, r: f64, v: f64, h: f64, r0: f64, p_ac: f64, f: f64) -> (f64, f64) {
        let (k1r, k1v) = km_rhs_test(t, r, v, r0, p_ac, f);
        let (k2r, k2v) = km_rhs_test(
            t + 0.5 * h,
            r + 0.5 * h * k1r,
            v + 0.5 * h * k1v,
            r0,
            p_ac,
            f,
        );
        let (k3r, k3v) = km_rhs_test(
            t + 0.5 * h,
            r + 0.5 * h * k2r,
            v + 0.5 * h * k2v,
            r0,
            p_ac,
            f,
        );
        let (k4r, k4v) = km_rhs_test(t + h, r + h * k3r, v + h * k3v, r0, p_ac, f);
        (
            r + h / 6.0 * (k1r + 2.0 * k2r + 2.0 * k3r + k4r),
            v + h / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v),
        )
    }

    /// Integrate the test-local Keller–Miksis at the Louisnard &
    /// Gomez Fig. 4 conditions and return the post-discard
    /// history window (cycles 5–60, uniform 400-points-per-cycle
    /// grid), matching the prototype's evaluation protocol.
    /// Controller (each element earned by a failure mode
    /// against the prototype's LSODA ground truth): step-doubling
    /// RK4 at rtol 1e-10, atol (1e-14, 1e-8) (10× tighter than
    /// the prototype's LSODA tolerances — the explicit scheme
    /// needs them at the strongest anchor); the half-step result accepted as-is (the
    /// Richardson extrapolation injects energy in the stiff
    /// rebound); growth capped at 1.5× (a 5× cap jumps over the
    /// collapse onset); a relative-change rejection |ΔR| ≤ R/2
    /// (an unbounded accepted step overshoots past Rmin into
    /// unrecoverable R⁻³ stiffness); and a bounded ride-through
    /// at the step floor for the stiff instant itself — the
    /// explicit analogue of LSODA switching to BDF there (an
    /// unbounded floor-accept is poison: it once let a rebound
    /// step through that exploded the bubble to metres).
    ///
    /// The two strongest anchors are excluded from the sweep
    /// below: (3 µm, 2 bar), Rmax* = 33, stalls this driver in
    /// the stiff rebound, and at (5 µm, 5 bar) the driver's
    /// history is converged in tolerance yet departs
    /// qualitatively from LSODA's (window rate ~10⁷ µm/s vs
    /// 275 µm/s — the explicit scheme settles into a
    /// non-collapsing mode the implicit scheme does not).
    /// Both anchors' validation stands on the prototype's LSODA
    /// numbers (222.9 and 275.3 µm/s vs the paper's ~200 and
    /// ~300 µm/s reads). This is the same gap as KW-PHYS-009
    /// item 5 for the library's own integrator.
    fn km_history(r0: f64, p_ac: f64, frequency: f64) -> (Vec<f64>, Vec<f64>) {
        // Prototype protocol (b6a_law.km_solve): 60 cycles, the
        // first 5 discarded, and the law evaluated over the whole
        // post-discard window — NOT the last cycle alone. A
        // last-cycle evaluation is poisoned at the strongest
        // anchor (5 µm, 5 bar) by individual non-collapsing /
        // afterbounce-heavy cycles; the window average is what
        // the paper's Fig. 4 reports.
        let n_cycles = 60;
        let discard = 5;
        let pts_per_cycle = 400;
        let period = 1.0 / frequency;
        let t_end = n_cycles as f64 * period;
        let grid_dt = period / pts_per_cycle as f64;
        let total_pts = n_cycles * pts_per_cycle;
        let mut grid_r = Vec::with_capacity(total_pts + 1);
        let (mut r, mut v) = (r0, 0.0);
        let mut t = 0.0;
        grid_r.push(r);
        let mut h = grid_dt / 10.0;
        let mut next_grid = grid_dt;
        while t < t_end - 1e-15 {
            // Advance to the next grid point with step-doubling.
            let target = next_grid.min(t_end);
            let mut iterations = 0_u64;
            while t < target - 1e-15 {
                iterations += 1;
                assert!(
                    iterations < 20_000_000,
                    "KM test integrator stuck at t = {t:e}, h = {h:e}, R = {r:e}, V = {v:e}"
                );
                let hh = h.min(target - t);
                let (rf, vf) = rk4_step_test(t, r, v, hh, r0, p_ac, frequency);
                let (rh, vh) = rk4_step_test(t, r, v, hh / 2.0, r0, p_ac, frequency);
                let (rh, vh) = rk4_step_test(t + hh / 2.0, rh, vh, hh / 2.0, r0, p_ac, frequency);
                // A substage can go non-finite when the collapse
                // drives the wall through the Keller–Miksis
                // singularity (|Ṙ| → c makes the LHS coefficient
                // vanish). Reject and shrink — never let a NaN
                // reach the step-size factor (NaN comparisons are
                // false, which would otherwise grow h and loop
                // forever on the same interval).
                if !rh.is_finite() || !vh.is_finite() {
                    h = hh * 0.25;
                    continue;
                }
                // Step-doubling error estimate (un-divided
                // difference — the conservative acceptance test).
                // The half-step result is accepted as-is: the
                // Richardson-extrapolated value injects energy in
                // the stiff collapse rebound and runs the state
                // away (|Ṙ| → 10⁴ m/s at R → 50 nm).
                let err = ((rf - rh).abs() / (1e-14 + 1e-10 * rh.abs()))
                    .max((vf - vh).abs() / (1e-8 + 1e-10 * vh.abs()));
                if !err.is_finite() {
                    h = hh * 0.25;
                    continue;
                }
                // Relative-change rejection: no accepted step may
                // move R by more than half itself. Without it a
                // step entering the collapse overshoots past the
                // true Rmin (~1 µm) to tens of nm, after which the
                // R⁻³ gas law is unrecoverable stiffness.
                let change_ok = (rh - r).abs() <= 0.5 * r;
                if err <= 1.0 && change_ok {
                    r = rh;
                    v = vh;
                    t += hh;
                } else if hh <= 1e-13 && change_ok {
                    // Bounded ride-through at the step floor: the
                    // stiff instant (deepest collapse) cannot be
                    // resolved by any explicit step, but a floor
                    // step whose displacement is itself bounded
                    // moves the state negligibly — accept it and
                    // stay small, the explicit analogue of LSODA
                    // switching to BDF there. An UNBOUNDED
                    // floor-accept is poison (it let a rebound
                    // step through that exploded the bubble), so
                    // the change bound is the gate.
                    r = rh;
                    v = vh;
                    t += hh;
                    h = hh;
                    continue;
                }
                // Growth capped at 1.5×: a 5× cap lets h jump
                // over the collapse onset between checks.
                let factor = if err > 0.0 {
                    (0.9 * err.powf(-0.2)).clamp(0.2, 1.5)
                } else {
                    1.5
                };
                h = (hh * factor).max(1e-13);
            }
            grid_r.push(r);
            next_grid += grid_dt;
        }
        let start = discard * pts_per_cycle;
        let window_pts = total_pts - start;
        let radius = grid_r[start..=total_pts].to_vec();
        let time: Vec<f64> = (0..=window_pts).map(|i| i as f64 * grid_dt).collect();
        (radius, time)
    }

    #[test]
    fn test_louisnard_gomez_fig4_anchor() {
        // Louisnard & Gomez (2003) Fig. 4 anchor: R0 = 5 µm,
        // p = 2 bar at 26.5 kHz in saturated water grows at
        // ~100 µm/s. The paper's anchors are order-of-magnitude
        // reads (factor ~2 per the prototype's validation notes);
        // the prototype's LSODA gave 94.6 µm/s for this anchor.
        // Assert the factor-of-2 band. The four-anchor sweep is
        // the ignored benchmark below (KM histories are too
        // expensive for the always-on suite in debug builds);
        // two anchors are excluded from it — see km_history's
        // docs.
        let params = paper_params();
        let (radius, time) = km_history(5e-6, 200_000.0, 26_500.0);
        let rate_um_s = louisnard_gomez_growth_rate(&radius, &time, 5e-6, 1.0, &params) * 1e6;
        assert!(
            rate_um_s > 50.0 && rate_um_s < 200.0,
            "Fig. 4 anchor R0 = 5 µm, p = 2 bar: got {rate_um_s} µm/s, paper ~100 µm/s"
        );
    }

    #[test]
    #[ignore = "expensive benchmark: four Keller-Miksis histories; run with -- --ignored"]
    fn test_louisnard_gomez_fig4_benchmark() {
        // Louisnard & Gomez (2003) Fig. 4 growth rates at 26.5 kHz,
        // saturated water. Assert a factor-of-2 band around each
        // anchor and the correct ordering in R0 and p. Four of the
        // paper's six anchors; the two strongest are excluded
        // because their histories defeat the explicit test driver
        // (see km_history's docs): (3 µm, 2 bar) stalls it, and at
        // (5 µm, 5 bar) its converged history departs
        // qualitatively from LSODA's. Both are covered by the
        // prototype's LSODA integration (222.9 and 275.3 µm/s vs
        // the paper's ~200 and ~300 µm/s).
        let params = paper_params();
        let anchors: [(f64, f64, f64); 4] = [
            (5e-6, 200_000.0, 100.0),
            (10e-6, 200_000.0, 30.0),
            (20e-6, 200_000.0, 10.0),
            (5e-6, 130_000.0, 30.0),
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
        // the 2 bar rate exceeds the 1.3 bar rate.
        assert!(rates[0] > rates[1] && rates[1] > rates[2]);
        assert!(rates[0] > rates[3]);
    }
}
