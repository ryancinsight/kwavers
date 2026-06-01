use super::*;
use crate::core::constants::fundamental::{
    ATMOSPHERIC_PRESSURE, BOLTZMANN as BOLTZMANN_CONSTANT, PLANCK as PLANCK_CONSTANT,
};
use crate::core::constants::numerical::FOUR_PI;
use crate::core::constants::numerical::MPA_TO_PA;

#[test]
fn gaunt_factor_lower_bound() {
    let cases = [(1e12, 10_000.0), (1e15, 20_000.0), (1e18, 100_000.0)];
    for (freq, temp) in cases {
        let g = gaunt_factor_thermal(freq, temp);
        assert!(g >= 1.0, "g_ff({freq}, {temp}) = {g} < 1.0");
        assert!(g <= 10.0, "g_ff({freq}, {temp}) = {g} > 10.0");
    }
}

#[test]
fn gaunt_factor_hard_xray_limit() {
    let g = gaunt_factor_thermal(1e20, 10_000.0);
    assert_eq!(g, 1.0, "hard X-ray Gaunt factor must be 1.0");
}

#[test]
fn gaunt_factor_increases_with_temperature() {
    let freq = 1e14;
    let g1 = gaunt_factor_thermal(freq, 10_000.0);
    let g2 = gaunt_factor_thermal(freq, 50_000.0);
    assert!(g2 > g1, "g_ff should increase with T at fixed nu");
}

#[test]
fn saha_hydrogen_10000k_1atm() {
    let model = BremsstrahlungModel::default();
    let x = model.saha_ionization(10_000.0, ATMOSPHERIC_PRESSURE, 13.6);
    assert!(
        x > 0.01 && x < 0.10,
        "H ionization at 10,000K, 1 atm: x = {x:.4}, expected 1-10%"
    );
}

#[test]
fn saha_hydrogen_fully_ionized_at_50000k() {
    let model = BremsstrahlungModel::default();
    let x = model.saha_ionization(50_000.0, ATMOSPHERIC_PRESSURE, 13.6);
    assert!(x > 0.95, "H at 50,000 K must be >95% ionized, got {x:.4}");
}

#[test]
fn saha_output_in_valid_range() {
    let model = BremsstrahlungModel::default();
    for (t, p) in [(5_000.0, 1e5), (20_000.0, 1e5), (100_000.0, MPA_TO_PA)] {
        let x = model.saha_ionization(t, p, 13.6);
        assert!((0.0..=1.0).contains(&x), "x({t}K, {p}Pa) = {x}");
    }
}

#[test]
fn saha_increases_with_temperature() {
    let model = BremsstrahlungModel::default();
    let temps = [5_000.0, 8_000.0, 12_000.0, 20_000.0, 50_000.0];
    let fracs: Vec<f64> = temps
        .iter()
        .map(|&t| model.saha_ionization(t, ATMOSPHERIC_PRESSURE, 13.6))
        .collect();

    for i in 1..fracs.len() {
        assert!(
            fracs[i] >= fracs[i - 1],
            "Ionization must increase with T: x({}) = {:.5} < x({}) = {:.5}",
            temps[i],
            fracs[i],
            temps[i - 1],
            fracs[i - 1]
        );
    }
}

#[test]
fn plasma_state_charge_neutrality() {
    let state = PlasmaState::from_single_stage(20_000.0, 1e5, 13.6, 1.0);
    let rel_err =
        (state.electron_density - state.ion_density_z2).abs() / state.electron_density.max(1.0);
    assert!(rel_err < 1e-10, "Charge neutrality rel_err = {rel_err}");
}

#[test]
fn argon_plasma_first_ionization_dominant_at_20kk() {
    let state = PlasmaState::from_noble_gas(20_000.0, 1e5, NobleGas::Argon);
    assert!(state.ionization_fraction > 0.0);
    assert!(state.ionization_fraction < 1.0);
}

#[test]
fn argon_plasma_highly_ionized_at_100kk() {
    let state = PlasmaState::from_noble_gas(100_000.0, 1e5, NobleGas::Argon);
    assert!(
        state.ionization_fraction > 0.5,
        "Argon at 100,000 K should be >50% ionized, got {:.4}",
        state.ionization_fraction
    );
}

#[test]
fn emission_coefficient_positive_finite() {
    let model = BremsstrahlungModel::default();
    let j = model.emission_coefficient(1e15, 20_000.0, 1e24, 1e24);
    assert!(j > 0.0 && j.is_finite(), "emission coefficient = {j}");
}

#[test]
fn emission_coefficient_decreases_with_frequency() {
    let model = BremsstrahlungModel::default();
    let j_low = model.emission_coefficient(1e14, 20_000.0, 1e24, 1e24);
    let j_high = model.emission_coefficient(1e16, 20_000.0, 1e24, 1e24);
    assert!(j_high < j_low, "Emission must decrease with frequency");
}

#[test]
fn emission_coefficient_quadratic_in_density() {
    let model = BremsstrahlungModel {
        use_thermal_gaunt_factor: false,
        ..Default::default()
    };
    let j1 = model.emission_coefficient(1e15, 20_000.0, 1e24, 1e24);
    let j2 = model.emission_coefficient(1e15, 20_000.0, 2e24, 2e24);
    let ratio = j2 / j1;
    assert!((ratio - 4.0).abs() < 1e-10, "density scaling ratio={ratio}");
}

#[test]
fn emission_coefficient_magnitude_rybicki_lightman() {
    let model = BremsstrahlungModel {
        z_ion: 1.0,
        use_thermal_gaunt_factor: false,
        fixed_gaunt_factor: 1.0,
    };

    let temperature = 20_000.0_f64;
    let n_e = 1e24_f64;
    let frequency = 1e15_f64;
    let j_computed = model.emission_coefficient(frequency, temperature, n_e, n_e);

    // R&L 5.14b gives eps_nu (total, 4pi sr) = 6.8e-51 [W m^-3 Hz^-1 K^{1/2}] in SI.
    // Per-steradian: j_nu = eps_nu / (4pi) => C_ff_per_sr = 6.8e-51 / (4pi).
    let c_ff_ref = 6.8e-51_f64 / (FOUR_PI);
    let h_nu = PLANCK_CONSTANT * frequency;
    let k_t = BOLTZMANN_CONSTANT * temperature;
    let j_reference = c_ff_ref * n_e * n_e * temperature.powf(-0.5) * (-h_nu / k_t).exp();
    let rel_err = (j_computed - j_reference).abs() / j_reference;
    assert!(rel_err < 0.10, "rel_err = {rel_err:.3}");
}

#[test]
fn emission_from_temperature_pressure_increases_with_temperature() {
    let model = BremsstrahlungModel::default();
    let freq = 1e14;
    let pressure = 1e8;
    let e_ion = 15.76;

    let j1 = model.emission_from_temperature_pressure(freq, 20_000.0, pressure, e_ion);
    let j2 = model.emission_from_temperature_pressure(freq, 50_000.0, pressure, e_ion);
    assert!(j2 > j1, "Emission must increase with temperature");
}
