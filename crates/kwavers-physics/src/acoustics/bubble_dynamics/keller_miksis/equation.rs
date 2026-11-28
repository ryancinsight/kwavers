use super::KellerMiksisModel;
use crate::acoustics::bubble_dynamics::{
    acoustic_pressure::InstantaneousPressure, bubble_state::BubbleState,
};
use aequitas::systems::si::quantities::PressureRate;
use kwavers_core::error::KwaversResult;

/// Floor on |(1 − Ṙ/c)| in the Keller–Miksis left-hand coefficient.
///
/// The coefficient (1 − Ṙ/c) is singular only at Ṙ = +c exactly. It is
/// regularised to this floor (sign preserved) so the acceleration stays
/// defined in a neighbourhood of the singularity instead of the model
/// erroring out: the true inertial-collapse trajectory of the equation
/// as written crosses Mach 1 every cycle (LSODA reference at the
/// Louisnard & Gomez Fig. 4 conditions reaches Mach ≈ 4.08 in
/// collapse; see KW-PHYS-009 item 5). Keller–Miksis is a
/// first-order-in-Mach theory, so states with Mach ≳ 1 are outside its
/// asymptotic validity — they are *flagged* via `state.mach_number`,
/// not errored, and the integrator's error control judges the steps.
const MACH_COEFFICIENT_FLOOR: f64 = 0.05;

/// Regularised (1 − Ṙ/c): clamped away from zero at
/// [`MACH_COEFFICIENT_FLOOR`], sign preserved.
fn regularised_one_minus_mach(v_c: f64) -> f64 {
    let x = 1.0 - v_c;
    if x.abs() < MACH_COEFFICIENT_FLOOR {
        if x < 0.0 {
            -MACH_COEFFICIENT_FLOOR
        } else {
            MACH_COEFFICIENT_FLOOR
        }
    } else {
        x
    }
}

/// Calculate bubble wall acceleration using Keller-Miksis equation.
///
/// Computes the spherical K-M ODE RHS only.  Non-spherical shape mode
/// evolution is coupled externally via
/// [`KellerMiksisModel::update_shape_stability`], which must be called
/// once per integration timestep after this function.
///
/// # Acoustic-drive contract
///
/// `p_acoustic` is the INSTANTANEOUS acoustic pressure `p_∞(t) − p0` [Pa]
/// at time `t`, and `dp_dt` is its time derivative `dp_∞/dt` [Pa/s]. The
/// Keller–Miksis equation is written in terms of a general far-field
/// pressure `p_∞(t)` (Keller & Miksis 1980, JASA 68(2):628–633, Eq. 3.4;
/// Prosperetti & Lezzi 1986, JFM 168:457–478); the sinusoidal form
/// `p_∞(t) = p0 ± P_A·sin(ωt + φ)` is the standard specialisation for
/// continuous harmonic drive, not part of the equation. Callers that have
/// an amplitude synthesise the tone themselves — cf. the analytical kernel
/// `keller_miksis_forced_rk4`, which takes a forcing closure returning
/// `(p, dp/dt)`. An earlier version of this function treated `p_acoustic`
/// as an amplitude and multiplied by `sin(ωt)` internally; that
/// double-modulated every caller that already passed an instantaneous
/// pressure (wave-coupled fields, the triplet-pulse port, the
/// passive-dose simulator), driving the bubble with a phase offset plus a
/// `2f0` term and no `f0` at all. The `t` argument is retained for
/// interface compatibility; the drive is fully specified by
/// (`p_acoustic`, `dp_dt`).
///
/// # Mach contract
///
/// This function does **not** error at high wall Mach number
/// (KW-PHYS-009 item 5): the (1 − Ṙ/c) coefficient is regularised
/// away from its Ṙ = +c singularity and the state records
/// `mach_number` so the regime is visible to callers. Integrate
/// inertial-collapse trajectories with the implicit SDIRK2 path
/// ([`ImplicitBubbleIntegrator`](crate::acoustics::bubble_dynamics::ImplicitBubbleIntegrator));
/// the explicit RK4 path cannot traverse the collapse spike without
/// injecting energy.
/// # Errors
/// - Propagates any `KwaversError` returned by called functions.
///
pub(crate) fn calculate_acceleration(
    model: &KellerMiksisModel,
    state: &mut BubbleState,
    p_acoustic: InstantaneousPressure,
    dp_dt: PressureRate,
    t: f64,
) -> KwaversResult<f64> {
    let r = state.radius;
    let v = state.wall_velocity;
    let c = model.params.c_liquid;

    // Formula boundary: the drive arrives typed; the equation below
    // works in pascals. `dp_dt` stays typed — the wall-pressure
    // derivative helper extracts it.
    let p_acoustic = p_acoustic.as_pascals();

    // Update Mach number based on wall velocity
    // Reference: Keller & Miksis (1980), Eq. 2.5
    let mach = v.abs() / c;
    state.mach_number = mach;

    // No Mach error (KW-PHYS-009 item 5): the equation as written is
    // singular only at Ṙ = +c exactly, and its true collapse
    // trajectory crosses Mach 1 every cycle, so erroring above
    // Mach 0.95 made the library unable to produce any history in
    // the inertial-collapse regime. The (1 − Ṙ/c) coefficient below
    // is regularised away from the singularity instead
    // ([`regularised_one_minus_mach`]); Mach ≳ 1 states are outside
    // the first-order theory's validity and are flagged via
    // `state.mach_number` for the caller / integrator to judge.

    // Far-field pressure p_∞(t) = p0 + p_acoustic(t); see the contract on
    // this function. The `t` argument is unused by the RHS: the drive is
    // fully specified by (`p_acoustic`, `dp_dt`).
    let _ = t;
    let p_inf = model.params.p0 + p_acoustic;

    // Internal gas pressure using polytropic relation or Van der Waals EOS.
    //
    // Non-thermal branch uses the correct vapor-pressure-aware polytropic closure
    // (Brennen 1995, §2.4; Prosperetti & Lezzi 1986):
    //   p_gas = (p0 + 2σ/r0 − pv)·(r0/r)^{3γ} + pv
    // Subtracting pv before scaling ensures only the non-condensable gas is
    // compressed polytropically; pv is then added back as the isothermal
    // vapor partial pressure, giving the correct equilibrium at r = r0 and
    // the correct pv floor as r → ∞.
    let gamma = state.gas_species.gamma();
    let p_gas = if !model.params.use_thermal_effects {
        let p_eq = model.params.p0 + 2.0 * model.params.sigma / model.params.r0 - model.params.pv;
        p_eq * (model.params.r0 / r).powf(3.0 * gamma) + model.params.pv
    } else {
        // Use Van der Waals equation with thermal effects
        model.calculate_vdw_pressure(state)?
    };

    // Bubble wall pressure (liquid side)
    // p_B = p_gas - 2σ/R - 4μṘ/R
    let surface_tension = model.params.surface_tension_pressure(r);
    let viscous_stress = model.params.viscous_wall_stress(v, r);
    let p_wall = p_gas - surface_tension - viscous_stress;

    // Time derivative of wall pressure for radiation damping term
    // This accounts for compressibility through the R/ρc × dp_B/dt term
    let dp_wall_dt = estimate_wall_pressure_derivative(model, state, p_gas, r, v, gamma, dp_dt)?;

    // Keller-Miksis equation components
    let v_c = v / c;
    let pressure_term = (1.0 + v_c) * (p_wall - p_inf) / model.params.rho_liquid;
    let radiation_term = r / (model.params.rho_liquid * c) * dp_wall_dt;
    let nonlinear_term = 1.5 * (1.0 - v_c / 3.0) * v * v;

    // Solve for R̈ from the K-M equation. The (1 − Ṙ/c) factor is
    // regularised away from its Ṙ = +c singularity (see
    // [`MACH_COEFFICIENT_FLOOR`]), so the coefficient can vanish
    // only through r itself. The +4μ/(ρc) term collects the R̈ part
    // of d/dt[4μṘ/R] onto the LHS (see
    // [`estimate_wall_pressure_derivative`]): with it, the
    // coefficient is the standard form (1 − Ṙ/c)·R + 4μ/(ρc)
    // carried by the validated prototype RHS (formation/b6a_law.py,
    // `km_rhs`: "The viscous term -4 mu Rdd/R inside d(p_w)/dt is
    // collected onto the LHS coefficient of Rdd (standard form)").
    let lhs_coeff = r * regularised_one_minus_mach(v_c)
        + 4.0 * model.params.mu_liquid / (model.params.rho_liquid * c);

    // Check for division by zero (structural: a non-positive radius)
    if lhs_coeff.abs() < 1e-12 {
        return Err(kwavers_core::error::PhysicsError::NumericalInstability {
            timestep: 0.0,
            cfl_limit: mach,
        }
        .into());
    }

    let acceleration = (pressure_term + radiation_term - nonlinear_term) / lhs_coeff;

    // Update state
    state.wall_acceleration = acceleration;
    state.pressure_internal = p_gas;
    state.pressure_liquid = p_wall;

    Ok(acceleration)
}

/// Estimate the net pressure time derivative for the KM radiation-damping term.
///
/// # Derivation
///
/// The Keller-Miksis radiation-damping term is:
/// ```text
/// R/(ρ_L·c) · d/dt[p_B − p_∞]
/// ```
/// where p_B = p_gas − 2σ/R − 4μṘ/R is the bubble-wall pressure and
/// p_∞ = p0 + p_acoustic(t) is the far-field pressure.
///
/// Differentiating:
/// ```text
/// d/dt[p_B − p_∞] = dp_gas/dt − d/dt[2σ/R] − d/dt[4μṘ/R] − dp_acoustic/dt
///
/// dp_gas/dt   = −3γ · (p_gas − p_v) · Ṙ / R   (polytropic part only:
///                 the vapour partial pressure p_v is isothermal —
///                 constant — so it carries no time derivative; the
///                 same split the Gilmore helper applies)
/// d/dt[2σ/R]  = −2σ · Ṙ / R²
/// d/dt[4μṘ/R] = 4μ · R̈ / R − 4μ · Ṙ² / R²   (this helper keeps the
///                 Ṙ² part; the R̈ part is collected onto the LHS
///                 coefficient of R̈ in [`calculate_acceleration`] as
///                 +4μ/(ρc) — the standard form, identical to the
///                 validated prototype RHS — so no term is dropped)
/// dp_acoustic/dt supplied externally as a [`PressureRate`] — the time
/// derivative of the INSTANTANEOUS far-field acoustic pressure (see the
/// contract on [`calculate_acceleration`]); it is used as-is, never
/// re-derived from an amplitude.
/// ```
///
/// # Reference
///
/// Keller JB, Miksis M (1980). "Bubble oscillations of large amplitude."
/// *J Acoust Soc Am* 68(2):628–633. Eq. (2.3).
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
#[allow(clippy::too_many_arguments)]
fn estimate_wall_pressure_derivative(
    model: &KellerMiksisModel,
    _state: &BubbleState,
    p_gas: f64,
    r: f64,
    v: f64,
    gamma: f64,
    dp_acoustic_dt: PressureRate,
) -> KwaversResult<f64> {
    // Formula boundary: the rate arrives typed; the derivative below
    // works in Pa/s.
    let dp_acoustic_dt = *dp_acoustic_dt.as_base();
    // Rate of change of internal gas pressure: dp_gas/dt =
    // −3γ (p_gas − p_v) Ṙ/R in the non-thermal closure, where
    // p_gas = p_eq·(r0/r)^{3γ} + p_v and only the non-condensable
    // part is polytropic (Brennen 1995, §2.4 — the closure stated on
    // `calculate_acceleration`). In the thermal branch p_gas is the
    // vdW mixture pressure, which has no additive p_v split, so the
    // full pressure is used (the polytropic form is an estimate
    // there in any case).
    let p_polytropic = if model.params.use_thermal_effects {
        p_gas
    } else {
        p_gas - model.params.pv
    };
    let dp_gas_dt = -3.0 * gamma * p_polytropic * v / r;

    // Rate of change of surface tension: d/dt(2σ/R) = −2σ Ṙ/R²
    let d_surface_dt = -2.0 * model.params.sigma * v / (r * r);

    // Rate of change of viscous stress, Ṙ² part: d/dt(4μṘ/R) =
    // 4μR̈/R − 4μṘ²/R². The R̈ part is not dropped — it is collected
    // onto the LHS coefficient in `calculate_acceleration`.
    let d_viscous_dt = -4.0 * model.params.mu_liquid * v * v / (r * r);

    // Net pressure derivative for radiation-damping term: d/dt[p_B − p_∞]
    // Subtracting dp_acoustic_dt accounts for the time-varying far-field pressure.
    let dp_net_dt = dp_gas_dt - d_surface_dt - d_viscous_dt - dp_acoustic_dt;

    Ok(dp_net_dt)
}
