use super::KellerMiksisModel;
use crate::core::error::KwaversResult;
use crate::physics::acoustics::bubble_dynamics::bubble_state::BubbleState;

/// Calculate bubble wall acceleration using Keller-Miksis equation
/// TODO_AUDIT: P1 - Bubble Shape Instability - Add non-spherical deformation models (shape instabilities, jet formation) for realistic cavitation dynamics, replacing spherical approximation
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

    // Internal gas pressure using polytropic relation
    // For more accurate modeling with thermal effects, use Van der Waals EOS
    let gamma = state.gas_species.gamma();
    let p_gas = if !model.params.use_thermal_effects {
        // Polytropic: p_gas = p_eq * (R₀/R)^(3γ)
        let p_eq = model.params.p0 + 2.0 * model.params.sigma / model.params.r0;
        p_eq * (model.params.r0 / r).powf(3.0 * gamma)
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

/// Estimate time derivative of wall pressure
fn estimate_wall_pressure_derivative(
    model: &KellerMiksisModel,
    _state: &BubbleState,
    p_gas: f64,
    r: f64,
    v: f64,
    gamma: f64,
    _dp_acoustic_dt: f64,
    _omega: f64,
    _t: f64,
) -> KwaversResult<f64> {
    // Rate of change of internal gas pressure
    // From polytropic relation: p ∝ R^(-3γ)
    // dp_gas/dt = -3γ p_gas (dR/dt)/R
    let dp_gas_dt = -3.0 * gamma * p_gas * v / r;

    // Rate of change of surface tension: d/dt(2σ/R) = -2σ Ṙ/R²
    let d_surface_dt = -2.0 * model.params.sigma * v / (r * r);

    // Rate of change of viscous stress: d/dt(4μṘ/R) = 4μ(R̈/R - Ṙ²/R²)
    // Approximation: neglect R̈ term initially (iterative refinement possible)
    let d_viscous_dt = -4.0 * model.params.mu_liquid * v * v / (r * r);

    // Total wall pressure derivative
    let dp_wall_dt = dp_gas_dt - d_surface_dt - d_viscous_dt;

    Ok(dp_wall_dt)
}
