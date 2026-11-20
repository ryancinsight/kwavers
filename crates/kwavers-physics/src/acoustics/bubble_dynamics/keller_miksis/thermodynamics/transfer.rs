//! Vapor mass-transfer update for Keller-Miksis thermodynamics.
//!
//! Wall-temperature convention (KW-PHYS-002 fix): the saturation pressure
//! in the Hertz–Knudsen flux — and in the equilibrium clamp — is evaluated
//! at the LIQUID temperature at the bubble wall `T_w`, never at the bulk gas
//! temperature. This is the universal standard in the bubble-dynamics
//! literature:
//!
//! - Kreider et al. 2011, J. Acoust. Soc. Am. 130:3511–3530, §2B, Eq. (15):
//!   `ṅ_v = 4πR²σ̂(p_sat − f·p_i)/√(2πMℜT_w)`, "T_w is the liquid
//!   temperature at the bubble wall, p_sat is the saturated vapor pressure
//!   evaluated at T_w".
//! - Yasui 1997, Phys. Rev. E 56:6750–6760 (as reproduced in Kalmár et al.
//!   2022, Ultrason. Sonochem., Eqs. 19–20): the evaporation term uses
//!   `p_v*`, the saturated vapour pressure at the ambient/liquid temperature
//!   `T_∞`; the gas temperature appears only in the condensation-term
//!   denominator, never inside `p_sat`. Yasui 1996, J. Phys. Soc. Jpn.
//!   65:2830–2840, solves for the varying liquid temperature at the wall;
//!   `p_sat` follows that liquid-side temperature.
//! - Storey & Szeri 2000, Proc. R. Soc. Lond. A 456:1685–1709, §2(d):
//!   "`P_sat` is the saturation pressure at the temperature of the
//!   interface", with the warning that the Hertz–Knudsen law "is only valid
//!   for reasonable temperatures and pressures, and … loses validity during
//!   the terminal stages of the bubble collapse"; "the temperature of the
//!   interface exceeds the critical point for a very brief time".
//! - Fujikawa & Akamatsu 1980, J. Fluid Mech. 97:481–512 (origin of the
//!   non-equilibrium condensation treatment).
//!
//! The previous code evaluated `p_sat` at the bulk gas temperature. During
//! a violent collapse the gas temperature exceeds the critical temperature
//! of water (647 K), where the saturation curve is clamped at the critical
//! pressure (22.06 MPa) — ~10,000× the room-temperature value — so the flux
//! drove EVAPORATION during the very phase when condensation should occur,
//! and the equilibrium clamp inherited the same error. The result was net
//! vapour accumulation of ~+16 R0 per cycle (KW-PHYS-002). No paper uses
//! `p_sat(T_gas)`.
//!
//! The liquid temperature is held at `ROOM_TEMPERATURE_K` here, matching the
//! liquid temperature used by the temperature update (`update_temperature`
//! in `temperature.rs`, which already evaluated its latent-heat saturation
//! pressure at the liquid temperature — the mass and energy bookkeeping are
//! now consistent). A varying wall temperature (Yasui 1996) is a documented
//! refinement, not implemented here. The critical-temperature switch-off
//! (`ṅ_v = 0` for `T > T_c`, Kreider et al. 2011) IS implemented on the
//! latent-heat side (`temperature.rs::latent_temperature_rate`, KW-PHYS-003:
//! the flux is zeroed when the interface temperature exceeds `T_c`, and
//! `L_v` is clamped to `≥ 0` with `L_v = 0` above `T_c`); it is not repeated
//! here because this flux is already evaluated at the constant room-temperature
//! `T_w`, where the condition is vacuous. With `p_sat(T_w)` the flux is
//! already bounded and periodic (see the KW-PHYS-002 backlog entry).

use super::super::KellerMiksisModel;
use crate::acoustics::bubble_dynamics::bubble_state::BubbleState;
use kwavers_core::constants::numerical::{FOUR_PI, TWO_PI};
use kwavers_core::constants::thermodynamic::ROOM_TEMPERATURE_K;
use kwavers_core::constants::{AVOGADRO, GAS_CONSTANT as R_GAS, M_WATER};
use kwavers_core::error::{KwaversResult, PhysicsError};

/// Post-step vapour content [molecules] for one sub-step: the raw
/// Hertz–Knudsen step, clamped at the saturation equilibrium and
/// floored at zero.
///
/// This is the vapour change the mass update ACTUALLY applies, and it
/// is shared with the latent-heat term (`latent_temperature_rate` in
/// `temperature.rs`) so the temperature and mass equations can never
/// disagree about the flux (Kreider et al. 2011, JASA 130:3511,
/// Eq. 20: the interface energy balance couples latent heat to the
/// actual mass flux, not to the unclamped kinetic flux). Sharing the
/// helper also unifies the saturation pressure: the latent term used
/// to evaluate `p_sat` with the Antoine correlation
/// (`p_sat_water_pa`) while the mass update uses the Wagner equation
/// (`thermo_calc.vapor_pressure`), which differ by ~0.5% at 20 °C;
/// both now use Wagner.
///
/// One residual approximation: the two updates evaluate this helper
/// from the same state except for the gas temperature in the
/// equilibrium clamp — the operator split runs temperature before
/// mass in the same sub-step, so the latent term sees the
/// pre-temperature-update `T_gas` and the mass update the
/// post-update one. The two `n_eq` values differ by O(ΔT/T) per
/// sub-step, and only in sub-steps where the clamp binds. In
/// production runs the clamp binds only near ambient temperature
/// (measured Nov 19, 2026: all observed binds at T ≈ 293 K, none
/// above 400 K in the four-way split), where ΔT per sub-step is
/// negligible, so the residual disagreement is far below the
/// integrator's own splitting error.
pub(crate) fn applied_n_vapor(model: &KellerMiksisModel, state: &BubbleState, dt: f64) -> f64 {
    // Saturation pressure at the LIQUID temperature at the bubble wall T_w
    // (Kreider et al. 2011, Eq. 15) — see the module docs. The bulk gas
    // temperature must NOT be used here: above the critical temperature the
    // saturation curve is meaningless, and evaluating it at T_gas produced
    // the KW-PHYS-002 runaway (evaporation during collapse).
    let p_sat = model.thermo_calc.vapor_pressure(ROOM_TEMPERATURE_K);
    let n_total = state.n_gas + state.n_vapor;
    let p_total = state.pressure_internal;
    let p_vapor = if n_total > 0.0 {
        p_total * (state.n_vapor / n_total)
    } else {
        0.0
    };

    let area = FOUR_PI * state.radius * state.radius;
    // Hertz-Knudsen molar flux [mol/s] at the wall temperature
    // (Kreider et al. 2011, Eq. 15):
    //   J_mol = α·A·(p_sat(T_w) − p_vapor) / √(2πMRT_w)
    let sqrt_term = (TWO_PI * M_WATER * R_GAS * ROOM_TEMPERATURE_K).sqrt();
    let molar_flux = model.params.accommodation_coeff * area * (p_sat - p_vapor) / sqrt_term;
    // Convert mol/s → molecules: ΔN = J_mol · dt · N_A
    let dn_vapor = molar_flux * dt * AVOGADRO;

    // Equilibrium clamp. The Hertz-Knudsen flux drives the vapor partial pressure
    // toward saturation `p_sat(T_w)`; the corresponding equilibrium vapor content
    // at the current volume and GAS temperature (ideal gas) is
    //   N_eq = p_sat(T_w) · V · N_A / (R · T_gas).
    // (The saturation pressure is set by the wall temperature; the amount at
    // that partial pressure follows from the gas temperature.) An explicit
    // Euler step must not OVERSHOOT this equilibrium — overshooting adds vapor
    // beyond saturation, which raises the internal pressure, expands the
    // bubble, enlarges the interfacial area, and evaporates still more,
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
    if overshoots { n_vapor_eq } else { proposed }.max(0.0)
}

/// Update vapor content through evaporation or condensation.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub(crate) fn update_mass_transfer(
    model: &KellerMiksisModel,
    state: &mut BubbleState,
    dt: f64,
) -> KwaversResult<()> {
    state.n_vapor = applied_n_vapor(model, state, dt);

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
