//! Bubble temperature ODE update for Keller-Miksis thermodynamics.

use super::super::KellerMiksisModel;
use super::phase::latent_heat_water_j_per_kg;
use crate::acoustics::bubble_dynamics::bubble_state::BubbleState;
use kwavers_core::constants::fundamental::STEFAN_BOLTZMANN;
use kwavers_core::constants::numerical::FOUR_PI;
use kwavers_core::constants::thermodynamic::{
    EMISSIVITY_VAPOR, KELVIN_OFFSET_C, M_WATER, ROOM_TEMPERATURE_K, THERMAL_CONDUCTIVITY_AIR,
    T_CRITICAL_WATER,
};
use kwavers_core::constants::AVOGADRO;
use kwavers_core::error::{KwaversResult, PhysicsError};

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
        n_total / AVOGADRO
    } else {
        return Ok(());
    };

    let c_v = model.molar_heat_capacity_cv(state);
    let heat_transfer_term = if n_moles > 0.0 && c_v > 0.0 {
        -heat_flux_w / (n_moles * c_v)
    } else {
        0.0
    };

    let latent_term = latent_temperature_rate(model, state, n_moles, c_v, dt);
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

/// Latent-heat rate [K/s] from the vapour flux actually applied by
/// the mass update.
///
/// KW-PHYS-003 — two defects fixed here, both in the interface-temperature
/// rule that KW-PHYS-002 already applied to `p_sat`:
///
/// (A) `L_v` is evaluated at the **interface/liquid temperature** `T_w`
/// (here the room-temperature liquid, as `p_sat` already is), never at
/// the bulk gas temperature. The old code passed `t_bubble` (up to
/// ~3200 K in collapse) into a linear fit valid ~0–100 °C; the fit
/// goes negative above 1055.7 °C, flipping the sign of the latent term
/// during every violent collapse. `L_v` itself is clamped to `≥ 0`
/// with `L_v = 0` above `T_c` (`latent_heat_water_j_per_kg`, Watson's
/// rule).
///
/// (B) The term is **gated on `use_mass_transfer`**, and — the dn/dt
/// coupling — it is computed from the vapour change
/// `update_mass_transfer` ACTUALLY applies in this sub-step
/// (`applied_n_vapor`: the raw Hertz–Knudsen flux after the
/// equilibrium clamp and the zero floor), not from the raw flux. With
/// mass transfer off, `update_mass_transfer` never changes `n_vapor`,
/// so applying a latent-heat flux would be latent energy without mass
/// transfer, violating the mass–energy coupling of the interface
/// energy balance (Kreider et al. 2011, JASA 130:3511, Eq. 20); and
/// whenever the equilibrium clamp binds, the raw flux overstates the
/// applied mass transfer, so the latent term must follow the applied
/// Δn. The coupling needs no reordering of the operator split: the
/// temperature update runs before the mass update in the same
/// sub-step, but both evaluate `applied_n_vapor` from the same state
/// (the temperature update changes neither `n_vapor`, `R`, nor
/// `p_internal`), so the latent term sees exactly the Δn the mass
/// update is about to apply — up to the O(ΔT/T) difference in the
/// equilibrium clamp's gas-temperature dependence, which is
/// negligible in practice (see `applied_n_vapor`).
///
/// Kreider et al. 2011, §2B, Eq. 15: the Hertz–Knudsen flux is zero
/// above the critical temperature. The interface temperature here is
/// the (constant) liquid temperature `T_w`, so the switch-off below is
/// dormant while `T_w` is held at room temperature; it activates if a
/// varying wall temperature (Yasui 1996) is ever introduced. The bulk
/// gas temperature must NOT be used for this test — that is the
/// KW-PHYS-002 error pattern (`p_sat(T_gas)`), and gating the flux on
/// `T_gas > T_c` would zero the latent term during every collapse.
pub(crate) fn latent_temperature_rate(
    model: &KellerMiksisModel,
    state: &BubbleState,
    n_moles: f64,
    c_v: f64,
    dt: f64,
) -> f64 {
    // Defect (B): no mass transfer, no latent heat (Kreider Eq. 20).
    if !model.params.use_mass_transfer {
        return 0.0;
    }
    // Interface/liquid temperature at the bubble wall T_w (Kreider Eq. 15).
    let t_interface_k = ROOM_TEMPERATURE_K;
    // Kreider Eq. 15: the flux is zero above the critical temperature.
    if t_interface_k >= T_CRITICAL_WATER {
        return 0.0;
    }
    // No elapsed time, no latent exchange (also guards the /dt below).
    if dt <= 0.0 {
        return 0.0;
    }
    // Defect (B), dn/dt coupling: the latent term follows the APPLIED
    // vapour change — post-clamp, post-floor — never the raw flux.
    // `applied_n_vapor` is evaluated from the pre-temperature-update
    // state, which is the same state the mass update is about to apply
    // it to (see the fn docs above).
    let dn_applied = super::transfer::applied_n_vapor(model, state, dt) - state.n_vapor;
    // molecules per sub-step → mass rate [kg/s]:
    //   ṁ = (ΔN / N_A) · M_water / dt.
    let mass_rate_kg_s = dn_applied * M_WATER / (AVOGADRO * dt);
    // Defect (A): L_v at the interface temperature, clamped ≥ 0 with
    // L_v = 0 above T_c (Watson's rule) — never at the bulk gas T.
    let l_v = latent_heat_water_j_per_kg(t_interface_k - KELVIN_OFFSET_C);

    if n_moles > 0.0 && c_v > 0.0 {
        -l_v * mass_rate_kg_s / (n_moles * c_v)
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
