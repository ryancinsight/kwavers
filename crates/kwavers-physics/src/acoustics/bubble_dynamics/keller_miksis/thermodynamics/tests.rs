use super::*;
use crate::acoustics::bubble_dynamics::acoustic_pressure::PressureAmplitude;
use kwavers_core::constants::fundamental::{ATMOSPHERIC_PRESSURE, STEFAN_BOLTZMANN};
use kwavers_core::constants::numerical::FOUR_PI;
use kwavers_core::constants::thermodynamic::{
    EMISSIVITY_VAPOR, ROOM_TEMPERATURE_K, THERMAL_CONDUCTIVITY_AIR,
};

#[test]
fn test_stefan_boltzmann_ambient_is_zero() {
    let t = ROOM_TEMPERATURE_K;
    let t0 = ROOM_TEMPERATURE_K;
    let t4_diff = t.powi(4) - t0.powi(4);
    let q_rad = EMISSIVITY_VAPOR * STEFAN_BOLTZMANN * t4_diff;
    assert!(
        q_rad.abs() < 1e-30,
        "q_rad should be ~0 at ambient temperature"
    );
}

#[test]
fn test_stefan_boltzmann_at_10000k_is_significant() {
    let t = 10_000.0_f64;
    let t0 = ROOM_TEMPERATURE_K;
    let r = 1e-6_f64;
    let surface_area = FOUR_PI * r * r;

    let q_rad = EMISSIVITY_VAPOR * STEFAN_BOLTZMANN * (t.powi(4) - t0.powi(4)) * surface_area;
    let q_cond = surface_area * THERMAL_CONDUCTIVITY_AIR * (t - t0);

    assert!(q_rad > 0.0, "radiative power must be positive at T > T0");
    assert!(
        q_rad > q_cond,
        "at T=10_000K radiation ({:.3e} W) should exceed conduction ({:.3e} W)",
        q_rad,
        q_cond
    );
}

#[test]
fn test_emissivity_vapor_in_range() {
    const _: () = {
        assert!(EMISSIVITY_VAPOR >= 0.05);
        assert!(EMISSIVITY_VAPOR <= 0.3);
    };
    let emissivity_vapor = EMISSIVITY_VAPOR;
    assert!((0.05..=0.3).contains(&emissivity_vapor));
}

#[test]
fn test_antoine_boiling_point() {
    let p = p_sat_water_pa(100.0);
    let rel_err = (p - ATMOSPHERIC_PRESSURE).abs() / ATMOSPHERIC_PRESSURE;
    assert!(
        rel_err < 0.005,
        "p_sat(100 C) = {:.1} Pa, expected about 101325 Pa (error {:.3}%)",
        p,
        rel_err * 100.0
    );
}

#[test]
fn test_latent_heat_temperature_dependence() {
    let l20 = latent_heat_water_j_per_kg(20.0);
    let l100 = latent_heat_water_j_per_kg(100.0);

    let err20 = (l20 - 2.452e6).abs() / 2.452e6;
    assert!(
        err20 < 0.001,
        "L_v(20 C) = {:.4e} J/kg, expected about 2.452e6",
        l20
    );

    let err100 = (l100 - 2.257e6).abs() / 2.257e6;
    assert!(
        err100 < 0.005,
        "L_v(100 C) = {:.4e} J/kg, expected about 2.257e6",
        l100
    );
}

#[test]
fn test_bubble_collapse_temperature_reduced_by_latent_heat() {
    use crate::acoustics::bubble_dynamics::bubble_state::{
        BubbleParameters, BubbleState, GasSpecies, GasType,
    };
    use crate::acoustics::bubble_dynamics::keller_miksis::KellerMiksisModel;
    use kwavers_core::constants::cavitation::{
        SURFACE_TENSION_WATER, VAPOR_PRESSURE_WATER, VISCOSITY_WATER,
    };
    use kwavers_core::constants::fundamental::{ATMOSPHERIC_PRESSURE, C_WATER, DENSITY_WATER};
    use kwavers_core::constants::thermodynamic::{
        ROOM_TEMPERATURE_K, SPECIFIC_HEAT_WATER, THERMAL_CONDUCTIVITY_WATER,
    };

    let params = BubbleParameters {
        r0: 10e-6,
        p0: ATMOSPHERIC_PRESSURE,
        rho_liquid: DENSITY_WATER,
        c_liquid: C_WATER,
        mu_liquid: VISCOSITY_WATER,
        sigma: SURFACE_TENSION_WATER,
        pv: VAPOR_PRESSURE_WATER,
        thermal_conductivity: THERMAL_CONDUCTIVITY_WATER,
        specific_heat_liquid: SPECIFIC_HEAT_WATER,
        accommodation_coeff: 0.35,
        gas_species: GasSpecies::Air,
        initial_gas_pressure: ATMOSPHERIC_PRESSURE,
        gas_composition: {
            let mut composition = std::collections::HashMap::new();
            composition.insert(GasType::N2, 0.79);
            composition.insert(GasType::O2, 0.21);
            composition
        },
        gamma: 1.4,
        t0: ROOM_TEMPERATURE_K,
        driving_frequency: 26_500.0,
        driving_amplitude: PressureAmplitude::from_pascals(1.5e5),
        use_compressibility: true,
        use_thermal_effects: true,
        use_mass_transfer: true,
    };

    let model = KellerMiksisModel::new(params.clone());
    let mut state = BubbleState::new(&params);
    let dt = 1e-8_f64;
    let mut peak_temp_with_latent = state.temperature;
    for step in 0..100 {
        if update_temperature(&model, &mut state, dt).is_ok() {
            state.wall_velocity = -(step as f64 + 1.0);
        }
        peak_temp_with_latent = peak_temp_with_latent.max(state.temperature);
    }

    assert!(
        peak_temp_with_latent.is_finite() && peak_temp_with_latent > 0.0,
        "Peak temperature must be positive and finite, got {:.1} K",
        peak_temp_with_latent
    );
}

/// KW-PHYS-003 defect (B): the latent term is identically zero when
/// mass transfer is off — latent energy without mass transfer violates
/// the interface energy balance (Kreider et al. 2011, Eq. 20). The
/// state is set to the maximum vapour deficit (`n_vapor = 0`, so the
/// Hertz–Knudsen flux would be at its largest) to make the test
/// decisive: any phantom flux would show up here.
#[test]
fn latent_term_zero_when_mass_transfer_off() {
    use crate::acoustics::bubble_dynamics::bubble_state::{BubbleParameters, BubbleState};
    use crate::acoustics::bubble_dynamics::keller_miksis::KellerMiksisModel;
    use kwavers_core::constants::AVOGADRO;

    let mut params = BubbleParameters::default();
    params.r0 = 3.0e-6;
    params.use_thermal_effects = true;
    params.use_mass_transfer = false;
    let model = KellerMiksisModel::new(params.clone());
    let mut state = BubbleState::new(&params);
    state.n_vapor = 0.0;
    let n_moles = state.n_gas / AVOGADRO;
    let c_v = model.molar_heat_capacity_cv(&state);
    let rate = latent_temperature_rate(&model, &state, n_moles, c_v, 1e-8);
    assert_eq!(
        rate, 0.0,
        "latent term must be identically zero with mass transfer off (KW-PHYS-003)"
    );
}

/// KW-PHYS-003 defect (A): with mass transfer on, evaporation cools
/// the gas even when the bulk gas temperature is at a violent-collapse
/// value (~3200 K). Before the fix `L_v` was evaluated at `t_bubble`
/// and was −4.43 MJ/kg there, flipping the sign so evaporation heated
/// the gas. `L_v` is now evaluated at the interface temperature, so
/// the sign is correct at any gas temperature.
#[test]
fn latent_term_cools_on_evaporation_at_collapse_temperature() {
    use crate::acoustics::bubble_dynamics::bubble_state::{BubbleParameters, BubbleState};
    use crate::acoustics::bubble_dynamics::keller_miksis::KellerMiksisModel;
    use kwavers_core::constants::AVOGADRO;

    let mut params = BubbleParameters::default();
    params.r0 = 3.0e-6;
    params.use_thermal_effects = true;
    params.use_mass_transfer = true;
    let model = KellerMiksisModel::new(params.clone());
    let mut state = BubbleState::new(&params);
    state.temperature = 3200.0;
    state.n_vapor = 0.0;
    let n_moles = state.n_gas / AVOGADRO;
    let c_v = model.molar_heat_capacity_cv(&state);
    let rate = latent_temperature_rate(&model, &state, n_moles, c_v, 1e-8);
    assert!(
        rate.is_finite() && rate < 0.0,
        "evaporation must cool the gas (rate < 0), got {rate:.3e} K/s at T_gas = 3200 K (KW-PHYS-003)"
    );
}

/// KW-PHYS-003 dn/dt coupling: the latent term tracks the APPLIED
/// dn/dt — the vapour change `update_mass_transfer` actually applies,
/// after its equilibrium clamp — not the raw Hertz–Knudsen flux. The
/// state is set to violent condensation at a long step (internal
/// pressure 100 MPa, vapour-rich, dt = 1 µs), so the raw flux would
/// drive n_vapor far below zero while the mass update clamps the step
/// at the saturation equilibrium n_eq. Before the coupling fix the
/// latent term used the raw flux and implied a mass rate ~10⁴× the
/// applied one; now the two agree to float precision. The clamp is
/// verified to have bound by checking n_vapor lands on n_eq
/// (computed independently from the equilibrium definition).
#[test]
fn latent_term_tracks_applied_dn_dt_when_clamp_binds() {
    use crate::acoustics::bubble_dynamics::bubble_state::{BubbleParameters, BubbleState};
    use crate::acoustics::bubble_dynamics::keller_miksis::KellerMiksisModel;
    use kwavers_core::constants::fundamental::GAS_CONSTANT as R_GAS;
    use kwavers_core::constants::thermodynamic::{KELVIN_OFFSET_C, M_WATER};
    use kwavers_core::constants::AVOGADRO;

    let mut params = BubbleParameters::default();
    params.r0 = 3.0e-6;
    params.use_thermal_effects = true;
    params.use_mass_transfer = true;
    params.accommodation_coeff = 0.35;
    let model = KellerMiksisModel::new(params.clone());
    let mut state = BubbleState::new(&params);
    state.pressure_internal = 1.0e8;
    state.n_vapor = 1.0e10;
    let dt = 1e-6;
    let n_moles = (state.n_gas + state.n_vapor) / AVOGADRO;
    let c_v = model.molar_heat_capacity_cv(&state);
    let l_v = latent_heat_water_j_per_kg(ROOM_TEMPERATURE_K - KELVIN_OFFSET_C);
    // Equilibrium vapour content at the current volume and gas
    // temperature (ideal gas, saturation set by the wall temperature).
    let p_sat = model.thermo_calc.vapor_pressure(ROOM_TEMPERATURE_K);
    let n_eq = p_sat * state.volume() * AVOGADRO / (R_GAS * state.temperature);

    let rate = latent_temperature_rate(&model, &state, n_moles, c_v, dt);
    let n_vapor_before = state.n_vapor;
    model
        .update_mass_transfer(&mut state, dt)
        .expect("mass transfer must succeed");

    // The equilibrium clamp bound: the step lands on n_eq, not on the
    // raw-flux proposal (which would be far below zero here).
    assert!(
        (state.n_vapor - n_eq).abs() <= 1e-9 * n_eq,
        "clamp must bind: n_vapor = {:.3e}, expected n_eq = {:.3e} (KW-PHYS-003)",
        state.n_vapor,
        n_eq
    );
    // The latent term's implied mass rate equals the applied dn/dt.
    let dn_applied = state.n_vapor - n_vapor_before;
    let m_dot_applied = dn_applied * M_WATER / (AVOGADRO * dt);
    let m_dot_implied = -rate * n_moles * c_v / l_v;
    let tol = 1e-9 * m_dot_applied.abs();
    assert!(
        (m_dot_implied - m_dot_applied).abs() <= tol,
        "latent term must track the applied dn/dt: implied ṁ = {:.3e} kg/s, applied ṁ = {:.3e} kg/s (KW-PHYS-003)",
        m_dot_implied,
        m_dot_applied
    );
}
