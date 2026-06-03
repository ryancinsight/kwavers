//! Vapor mass-transfer update for Keller-Miksis thermodynamics.

use super::super::KellerMiksisModel;
use kwavers_core::constants::numerical::{FOUR_PI, TWO_PI};
use kwavers_core::constants::{AVOGADRO, GAS_CONSTANT as R_GAS, M_WATER};
use kwavers_core::error::{KwaversResult, PhysicsError};
use crate::acoustics::bubble_dynamics::bubble_state::BubbleState;

/// Update vapor content through evaporation or condensation.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub(crate) fn update_mass_transfer(
    model: &KellerMiksisModel,
    state: &mut BubbleState,
    dt: f64,
) -> KwaversResult<()> {
    let p_sat = model.thermo_calc.vapor_pressure(state.temperature);
    let n_total = state.n_gas + state.n_vapor;
    let p_total = state.pressure_internal;
    let p_vapor = if n_total > 0.0 {
        p_total * (state.n_vapor / n_total)
    } else {
        0.0
    };

    let area = FOUR_PI * state.radius * state.radius;
    // Hertz-Knudsen molar flux [mol/s]: J_mol = α·A·ΔP / √(2πMRT)
    let sqrt_term = (TWO_PI * M_WATER * R_GAS * state.temperature).sqrt();
    let molar_flux = model.params.accommodation_coeff * area * (p_sat - p_vapor) / sqrt_term;
    // Convert mol/s → molecules: ΔN = J_mol · dt · N_A
    let dn_vapor = molar_flux * dt * AVOGADRO;

    // Equilibrium clamp. The Hertz-Knudsen flux drives the vapor partial pressure
    // toward saturation `p_sat`; the corresponding equilibrium vapor content at
    // the current volume and temperature (ideal gas) is
    //   N_eq = p_sat · V · N_A / (R · T).
    // An explicit Euler step must not OVERSHOOT this equilibrium — overshooting
    // adds vapor beyond saturation, which raises the internal pressure, expands
    // the bubble, enlarges the interfacial area, and evaporates still more,
    // producing a non-physical runaway (a 3 µm bubble blowing up to ~mm scale).
    // Clamping the step at `N_eq` keeps evaporation/condensation equilibrium-
    // limited (the kinetics set only how fast saturation is approached).
    let volume = state.volume();
    let n_vapor_eq = if state.temperature > 0.0 {
        p_sat * volume * AVOGADRO / (R_GAS * state.temperature)
    } else {
        state.n_vapor
    };
    let proposed = state.n_vapor + dn_vapor;
    let overshoots = (state.n_vapor - n_vapor_eq) * (proposed - n_vapor_eq) < 0.0;
    state.n_vapor = if overshoots { n_vapor_eq } else { proposed }.max(0.0);

    let n_total_new = state.n_gas + state.n_vapor;
    if n_total_new < 0.0 || n_total_new.is_nan() || n_total_new.is_infinite() {
        return Err(PhysicsError::InvalidParameter {
            parameter: "vapor_content".to_owned(),
            value: state.n_vapor,
            reason: format!(
                "Invalid vapor content: n_vapor={}, n_total={}",
                state.n_vapor, n_total_new
            ),
        }
        .into());
    }

    Ok(())
}
