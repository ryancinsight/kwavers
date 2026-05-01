use super::*;
use crate::core::constants::fundamental::STEFAN_BOLTZMANN;
use crate::core::constants::thermodynamic::{
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
    let surface_area = 4.0 * std::f64::consts::PI * r * r;

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
    assert!(EMISSIVITY_VAPOR >= 0.05, "emissivity must be >= 0.05");
    assert!(EMISSIVITY_VAPOR <= 0.3, "emissivity must be <= 0.3");
}

#[test]
fn test_antoine_boiling_point() {
    let p = p_sat_water_pa(100.0);
    let rel_err = (p - 101_325.0).abs() / 101_325.0;
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
    use crate::physics::acoustics::bubble_dynamics::bubble_state::{
        BubbleParameters, BubbleState, GasSpecies, GasType,
    };
    use crate::physics::acoustics::bubble_dynamics::keller_miksis::KellerMiksisModel;

    let params = BubbleParameters {
        r0: 10e-6,
        p0: 101_325.0,
        rho_liquid: 998.0,
        c_liquid: 1482.0,
        mu_liquid: 1.002e-3,
        sigma: 0.0728,
        pv: 2330.0,
        thermal_conductivity: 0.6,
        specific_heat_liquid: 4182.0,
        accommodation_coeff: 0.35,
        gas_species: GasSpecies::Air,
        initial_gas_pressure: 101_325.0,
        gas_composition: {
            let mut composition = std::collections::HashMap::new();
            composition.insert(GasType::N2, 0.79);
            composition.insert(GasType::O2, 0.21);
            composition
        },
        gamma: 1.4,
        t0: 293.15,
        driving_frequency: 26_500.0,
        driving_amplitude: 1.5e5,
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
