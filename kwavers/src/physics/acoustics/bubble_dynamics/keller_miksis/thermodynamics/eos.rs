//! Gas equation-of-state closures for Keller-Miksis thermodynamics.

use crate::core::constants::thermodynamic::{
    VAN_DER_WAALS_AIR, VAN_DER_WAALS_ARGON, VAN_DER_WAALS_NITROGEN, VAN_DER_WAALS_OXYGEN,
    VAN_DER_WAALS_XENON,
};
use crate::core::constants::{AVOGADRO, GAS_CONSTANT as R_GAS};
use crate::core::error::{KwaversResult, PhysicsError};
use crate::physics::acoustics::bubble_dynamics::bubble_state::{BubbleState, GasSpecies};

/// Calculate Van der Waals pressure for thermal effects.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub(crate) fn calculate_vdw_pressure(state: &BubbleState) -> KwaversResult<f64> {
    let volume = (4.0 / 3.0) * std::f64::consts::PI * state.radius.powi(3);
    let n_total = state.n_gas + state.n_vapor;
    let (a, b) = match state.gas_species {
        GasSpecies::Air => VAN_DER_WAALS_AIR,
        GasSpecies::Argon => VAN_DER_WAALS_ARGON,
        GasSpecies::Xenon => VAN_DER_WAALS_XENON,
        GasSpecies::Nitrogen => VAN_DER_WAALS_NITROGEN,
        GasSpecies::Oxygen => VAN_DER_WAALS_OXYGEN,
        GasSpecies::Custom { .. } => VAN_DER_WAALS_AIR,
    };

    let a_si = a * 1e5 * 1e-6;
    let b_si = b * 1e-3;
    let n_moles = n_total / AVOGADRO;
    let excluded_volume = n_moles * b_si;

    if volume <= excluded_volume {
        return Err(PhysicsError::InvalidParameter {
            parameter: "bubble_volume".to_owned(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::acoustics::bubble_dynamics::bubble_state::{BubbleParameters, BubbleState};

    fn equilibrium_state() -> BubbleState {
        BubbleState::at_equilibrium(&BubbleParameters::default())
    }

    /// VdW pressure for a dilute gas approaches ideal gas law: p ≈ nRT/V.
    ///
    /// At large volume, excluded-volume and intermolecular attraction corrections
    /// become negligible: p_VdW → p_ideal = nRT/V.
    /// We verify that the VdW result is within 1% of the ideal gas value.
    #[test]
    fn vdw_pressure_approaches_ideal_gas_at_large_volume() {
        let mut state = equilibrium_state();
        // Large bubble (1 mm) so that VdW corrections are negligible
        state.radius = 1e-3;
        let p = calculate_vdw_pressure(&state).unwrap();

        // Ideal gas reference
        let volume = 4.0 / 3.0 * std::f64::consts::PI * state.radius.powi(3);
        let n_moles = (state.n_gas + state.n_vapor) / crate::core::constants::AVOGADRO;
        let p_ideal = n_moles * crate::core::constants::GAS_CONSTANT * state.temperature / volume;

        let rel_err = (p - p_ideal).abs() / p_ideal;
        assert!(
            rel_err < 0.01,
            "VdW pressure must be within 1% of ideal gas at R=1mm (rel_err={rel_err:.4})"
        );
    }

    /// VdW pressure is finite and positive for a physical state.
    #[test]
    fn vdw_pressure_positive_for_equilibrium_state() {
        let state = equilibrium_state();
        let p = calculate_vdw_pressure(&state).unwrap();
        assert!(
            p > 0.0 && p.is_finite(),
            "VdW pressure must be positive finite (got {p:.3e})"
        );
    }

    /// VdW returns Err when bubble volume ≤ excluded volume (unphysical collapse).
    ///
    /// Set radius near zero so volume < n·b (Van der Waals hard-sphere volume).
    #[test]
    fn vdw_pressure_errors_when_volume_below_excluded() {
        let mut state = equilibrium_state();
        // Radius so small the bubble volume is essentially zero
        state.radius = 1e-15;
        let result = calculate_vdw_pressure(&state);
        assert!(
            result.is_err(),
            "must return Err when volume <= excluded volume"
        );
    }
}
