//! Gas equation-of-state closures for Keller-Miksis thermodynamics.

use crate::core::constants::thermodynamic::{
    VAN_DER_WAALS_AIR, VAN_DER_WAALS_ARGON, VAN_DER_WAALS_NITROGEN, VAN_DER_WAALS_OXYGEN,
    VAN_DER_WAALS_XENON,
};
use crate::core::constants::{AVOGADRO, GAS_CONSTANT as R_GAS};
use crate::core::error::{KwaversResult, PhysicsError};
use crate::physics::acoustics::bubble_dynamics::bubble_state::{BubbleState, GasSpecies};

/// Calculate Van der Waals pressure for thermal effects.
pub(crate) fn calculate_vdw_pressure(state: &BubbleState) -> KwaversResult<f64> {
    let volume = (4.0 / 3.0) * std::f64::consts::PI * state.radius.powi(3);
    let n_total = state.n_gas + state.n_vapor;
    let (a, b) = match state.gas_species {
        GasSpecies::Air => VAN_DER_WAALS_AIR,
        GasSpecies::Argon => VAN_DER_WAALS_ARGON,
        GasSpecies::Xenon => VAN_DER_WAALS_XENON,
        GasSpecies::Nitrogen => VAN_DER_WAALS_NITROGEN,
        GasSpecies::Oxygen => VAN_DER_WAALS_OXYGEN,
        _ => VAN_DER_WAALS_AIR,
    };

    let a_si = a * 1e5 * 1e-6;
    let b_si = b * 1e-3;
    let n_moles = n_total / AVOGADRO;
    let excluded_volume = n_moles * b_si;

    if volume <= excluded_volume {
        return Err(PhysicsError::InvalidParameter {
            parameter: "bubble_volume".to_string(),
            value: volume,
            reason: format!(
                "Volume {} m3 must be greater than excluded volume {} m3",
                volume, excluded_volume
            ),
        }
        .into());
    }

    let p_ideal = n_moles * R_GAS * state.temperature / (volume - excluded_volume);
    let p_correction = a_si * n_moles * n_moles / (volume * volume);

    Ok(p_ideal - p_correction)
}
