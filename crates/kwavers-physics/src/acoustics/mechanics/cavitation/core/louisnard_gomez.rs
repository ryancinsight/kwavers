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
mod tests;
