//! Bubble temperature ODE update for Keller-Miksis thermodynamics.

use super::super::KellerMiksisModel;
use super::phase::{latent_heat_water_j_per_kg, p_sat_water_pa};
use kwavers_core::constants::fundamental::{GAS_CONSTANT as R_GAS, STEFAN_BOLTZMANN};
use kwavers_core::constants::numerical::{FOUR_PI, TWO_PI};
use kwavers_core::constants::thermodynamic::{
    EMISSIVITY_VAPOR, KELVIN_OFFSET_C, M_WATER, ROOM_TEMPERATURE_K, THERMAL_CONDUCTIVITY_AIR,
};
use kwavers_core::error::{KwaversResult, PhysicsError};
use crate::acoustics::bubble_dynamics::bubble_state::BubbleState;

/// Update bubble temperature through adiabatic, conductive, latent, and radiative terms.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub(crate) fn update_temperature(
    model: &KellerMiksisModel,
    state: &mut BubbleState,
    dt: f64,
) -> KwaversResult<()> {
    let r = state.radius;
    let v = state.wall_velocity;
    let t_bubble = state.temperature;
    let t_liquid = ROOM_TEMPERATURE_K;
    let gamma = state.gas_species.gamma();
    // Adiabatic compression: T·V^{γ-1} = const, V = (4/3)πR³ ⟹ dT/dt = -3(γ-1)·T·Ṙ/R
    // Factor of 3 comes from dV/V = 3·dR/R for a sphere (Brennen 1995 §2.22).
    let adiabatic_term = -3.0 * (gamma - 1.0) * t_bubble * v / r;

    let surface_area = FOUR_PI * r * r;
    // Total heat flow [W] via Fourier's law with boundary layer thickness ≈ R:
    //   Q = k · A · ΔT / R      [W·m²·K / (m·K·m)] = [W] ✓
    // Using surface_area alone (without /r) gives [W·m], not [W].
    let heat_flux_w = surface_area * THERMAL_CONDUCTIVITY_AIR * (t_bubble - t_liquid) / r;
    let n_total = state.n_gas + state.n_vapor;
    let n_moles = if n_total > 0.0 {
        n_total / kwavers_core::constants::AVOGADRO
    } else {
        return Ok(());
    };

    let c_v = model.molar_heat_capacity_cv(state);
    let heat_transfer_term = if n_moles > 0.0 && c_v > 0.0 {
        -heat_flux_w / (n_moles * c_v)
    } else {
        0.0
    };

    let latent_term = latent_temperature_rate(model, state, surface_area, n_moles, c_v, t_bubble);
    let radiation_term = radiation_temperature_rate(surface_area, n_moles, c_v, t_bubble, t_liquid);
    let dt_dt = adiabatic_term + heat_transfer_term + latent_term + radiation_term;
    let t_new = t_bubble + dt_dt * dt;

    if !(0.0..=50000.0).contains(&t_new) || t_new.is_nan() || t_new.is_infinite() {
        return Err(PhysicsError::InvalidParameter {
            parameter: "bubble_temperature".to_owned(),
            value: t_new,
            reason: format!(
                "Temperature {} K is outside valid range (0 K < T < 50000 K)",
                t_new
            ),
        }
        .into());
    }

    state.temperature = t_new;
    if t_new > state.max_temperature {
        state.max_temperature = t_new;
    }

    Ok(())
}

fn latent_temperature_rate(
    model: &KellerMiksisModel,
    state: &BubbleState,
    surface_area: f64,
    n_moles: f64,
    c_v: f64,
    t_bubble: f64,
) -> f64 {
    let t_liquid_k = ROOM_TEMPERATURE_K;
    let p_sat_liq = p_sat_water_pa(t_liquid_k - KELVIN_OFFSET_C);
    let n_total = state.n_gas + state.n_vapor;
    let p_vapor = if n_total > 0.0 {
        state.pressure_internal * (state.n_vapor / n_total)
    } else {
        0.0
    };

    let sqrt_term = (TWO_PI * M_WATER * R_GAS * t_liquid_k).sqrt();
    // Hertz-Knudsen mass flux [kg/s]: J_mass = α·A·ΔP·M / √(2πMRT)
    let mass_flux_kg_s =
        model.params.accommodation_coeff * surface_area * (p_sat_liq - p_vapor) * M_WATER
            / sqrt_term;
    let l_v = latent_heat_water_j_per_kg(t_bubble - KELVIN_OFFSET_C);

    if n_moles > 0.0 && c_v > 0.0 {
        -l_v * mass_flux_kg_s / (n_moles * c_v)
    } else {
        0.0
    }
}

fn radiation_temperature_rate(
    surface_area: f64,
    n_moles: f64,
    c_v: f64,
    t_bubble: f64,
    t_liquid: f64,
) -> f64 {
    if n_moles > 0.0 && c_v > 0.0 {
        let t4_diff = t_bubble.powi(4) - t_liquid.powi(4);
        let q_rad = EMISSIVITY_VAPOR * STEFAN_BOLTZMANN * t4_diff * surface_area;
        -q_rad / (n_moles * c_v)
    } else {
        0.0
    }
}
