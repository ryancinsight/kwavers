use super::KellerMiksisModel;
use crate::core::error::KwaversResult;
use crate::physics::acoustics::bubble_dynamics::bubble_state::BubbleState;

/// Calculate bubble wall acceleration using Keller-Miksis equation.
///
/// Computes the spherical K-M ODE RHS only.  Non-spherical shape mode
/// evolution is coupled externally via
/// [`KellerMiksisModel::update_shape_stability`], which must be called
/// once per integration timestep after this function.
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
///
pub(crate) fn calculate_acceleration(
    model: &KellerMiksisModel,
    state: &mut BubbleState,
    p_acoustic: f64,
    dp_dt: f64,
    t: f64,
) -> KwaversResult<f64> {
    let r = state.radius;
    let v = state.wall_velocity;
    let c = model.params.c_liquid;

    // Update Mach number based on wall velocity
    // Reference: Keller & Miksis (1980), Eq. 2.5
    let mach = v.abs() / c;
    state.mach_number = mach;

    // Check for numerical stability (wall velocity approaching sound speed)
    // K-M equation becomes singular when Ṙ → c
    if mach > 0.95 {
        return Err(crate::core::error::PhysicsError::NumericalInstability {
            timestep: 0.0,
            cfl_limit: mach,
        }
        .into());
    }

    // Acoustic forcing with proper phase
    let omega = 2.0 * std::f64::consts::PI * model.params.driving_frequency;
    let p_acoustic_inst = p_acoustic * (omega * t).sin();
    let p_inf = model.params.p0 + p_acoustic_inst;

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
        let p_eq =
            model.params.p0 + 2.0 * model.params.sigma / model.params.r0 - model.params.pv;
        p_eq * (model.params.r0 / r).powf(3.0 * gamma) + model.params.pv
    } else {
        // Use Van der Waals equation with thermal effects
        model.calculate_vdw_pressure(state)?
    };

    // Bubble wall pressure (liquid side)
    // p_B = p_gas - 2σ/R - 4μṘ/R
    let surface_tension = 2.0 * model.params.sigma / r;
    let viscous_stress = 4.0 * model.params.mu_liquid * v / r;
    let p_wall = p_gas - surface_tension - viscous_stress;

    // Time derivative of wall pressure for radiation damping term
    // This accounts for compressibility through the R/ρc × dp_B/dt term
    let dp_wall_dt =
        estimate_wall_pressure_derivative(model, state, p_gas, r, v, gamma, dp_dt, omega, t)?;

    // Keller-Miksis equation components
    let v_c = v / c;
    let pressure_term = (1.0 + v_c) * (p_wall - p_inf) / model.params.rho_liquid;
    let radiation_term = r / (model.params.rho_liquid * c) * dp_wall_dt;
    let nonlinear_term = 1.5 * (1.0 - v_c / 3.0) * v * v;

    // Solve for R̈ from the K-M equation
    let lhs_coeff = r * (1.0 - v_c);

    // Check for division by zero (should not occur if mach < 0.95)
    if lhs_coeff.abs() < 1e-12 {
        return Err(crate::core::error::PhysicsError::NumericalInstability {
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
/// dp_gas/dt   = −3γ · p_gas · Ṙ / R   (polytropic)
/// d/dt[2σ/R]  = −2σ · Ṙ / R²
/// d/dt[4μṘ/R] ≈ −4μ · Ṙ² / R²         (neglect R̈ term; first-order)
/// dp_acoustic/dt supplied externally as `dp_acoustic_dt`
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
    dp_acoustic_dt: f64,
    _omega: f64,
    _t: f64,
) -> KwaversResult<f64> {
    // Rate of change of internal gas pressure: dp_gas/dt = −3γ p_gas Ṙ/R
    let dp_gas_dt = -3.0 * gamma * p_gas * v / r;

    // Rate of change of surface tension: d/dt(2σ/R) = −2σ Ṙ/R²
    let d_surface_dt = -2.0 * model.params.sigma * v / (r * r);

    // Rate of change of viscous stress: d/dt(4μṘ/R) ≈ −4μ Ṙ²/R² (first-order)
    let d_viscous_dt = -4.0 * model.params.mu_liquid * v * v / (r * r);

    // Net pressure derivative for radiation-damping term: d/dt[p_B − p_∞]
    // Subtracting dp_acoustic_dt accounts for the time-varying far-field pressure.
    let dp_net_dt = dp_gas_dt - d_surface_dt - d_viscous_dt - dp_acoustic_dt;

    Ok(dp_net_dt)
}
