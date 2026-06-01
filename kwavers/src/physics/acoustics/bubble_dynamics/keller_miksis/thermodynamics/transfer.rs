//! Vapor mass-transfer update for Keller-Miksis thermodynamics.

use super::super::KellerMiksisModel;
use crate::core::constants::numerical::{FOUR_PI, TWO_PI};
use crate::core::constants::{AVOGADRO, GAS_CONSTANT as R_GAS, M_WATER};
use crate::core::error::{KwaversResult, PhysicsError};
use crate::physics::acoustics::bubble_dynamics::bubble_state::BubbleState;

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

    state.n_vapor = (state.n_vapor + dn_vapor).max(0.0);

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
