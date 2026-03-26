//! Tests for energy balance calculator

use super::*;
use crate::physics::acoustics::bubble_dynamics::bubble_state::BubbleState;
use crate::physics::acoustics::bubble_dynamics::BubbleParameters;
use uom::si::f64::{Power, Pressure};
use uom::si::power::watt;
use uom::si::pressure::pascal;

#[test]
fn test_energy_balance_equilibrium() {
    let params = BubbleParameters::default();
    let calculator = EnergyBalanceCalculator::new(&params);
    let state = BubbleState::new(&params);

    // At equilibrium with no wall motion, energy rate should be near zero
    let pressure = Pressure::new::<pascal>(101325.0);
    let heat_rate = Power::new::<watt>(0.0);
    let latent_rate = Power::new::<watt>(0.0);

    let energy_rate = calculator.calculate_energy_rate(&state, pressure, heat_rate, latent_rate);

    assert!(energy_rate.get::<watt>().abs() < 1e-10);
}

#[test]
fn test_heat_transfer_calculation() {
    let params = BubbleParameters::default();
    let calculator = EnergyBalanceCalculator::new(&params);
    let mut state = BubbleState::new(&params);

    // Set higher temperature
    state.temperature = 400.0; // K

    let heat_rate = calculator.calculate_heat_transfer_rate(&state, 10.0);

    // Heat rate is positive when bubble is hotter than liquid (heat flows out)
    // This is the standard convention: positive = heat out of bubble
    assert!(heat_rate.get::<watt>() > 0.0);
}

#[test]
fn test_chemical_reaction_energy() {
    let params = BubbleParameters::default();
    let calculator = EnergyBalanceCalculator::new(&params);
    let mut state = BubbleState::new(&params);

    // At low temperature, no chemical reactions
    state.temperature = 1000.0;
    let reaction_rate_low = calculator.calculate_chemical_reaction_rate(&state);
    assert_eq!(reaction_rate_low.get::<watt>(), 0.0);

    // At high temperature (T > 2000 K), chemical reactions occur
    state.temperature = 5000.0;
    state.n_vapor = 1e20; // Significant vapor content
    let reaction_rate_high = calculator.calculate_chemical_reaction_rate(&state);

    // Should be negative (endothermic, absorbs energy from bubble)
    assert!(reaction_rate_high.get::<watt>() < 0.0);
}

#[test]
fn test_plasma_ionization_energy() {
    let params = BubbleParameters::default();
    let calculator = EnergyBalanceCalculator::new(&params);
    let mut state = BubbleState::new(&params);

    // At low temperature, no ionization
    state.temperature = 5000.0;
    let ionization_rate_low = calculator.calculate_plasma_ionization_rate(&state);
    assert_eq!(ionization_rate_low.get::<watt>(), 0.0);

    // At extreme temperature (T > 10000 K), ionization occurs
    state.temperature = 15_000.0;
    state.n_gas = 1e20;
    let ionization_rate_high = calculator.calculate_plasma_ionization_rate(&state);

    // Should be negative (endothermic, absorbs energy)
    assert!(ionization_rate_high.get::<watt>() < 0.0);
}

#[test]
fn test_radiation_losses() {
    let params = BubbleParameters::default();
    let calculator = EnergyBalanceCalculator::new(&params);
    let mut state = BubbleState::new(&params);

    // At moderate temperature, negligible radiation
    state.temperature = 3000.0;
    let radiation_low = calculator.calculate_radiation_losses(&state);
    assert_eq!(radiation_low.get::<watt>(), 0.0);

    // At extreme temperature (T > 5000 K), significant radiation
    state.temperature = 10_000.0;
    state.radius = 5e-6; // 5 μm
    let radiation_high = calculator.calculate_radiation_losses(&state);

    // Should be negative (energy loss from bubble)
    assert!(radiation_high.get::<watt>() < 0.0);

    // Stefan-Boltzmann: Q ∝ T⁴, so doubling temperature increases radiation by 16×
    state.temperature = 20_000.0;
    let radiation_double = calculator.calculate_radiation_losses(&state);
    let ratio = radiation_double.get::<watt>() / radiation_high.get::<watt>();
    assert!((ratio - 16.0).abs() < 2.0); // Within 2x tolerance (T⁴ dominates)
}

#[test]
fn test_complete_energy_balance() {
    let params = BubbleParameters::default();
    let calculator = EnergyBalanceCalculator::new(&params);
    let mut state = BubbleState::new(&params);

    // Extreme collapse conditions (sonoluminescence regime)
    state.temperature = 12_000.0; // K
    state.radius = 3e-6; // 3 μm
    state.wall_velocity = -100.0; // m/s (compression)
    state.n_gas = 1e19;
    state.n_vapor = 5e18;

    let internal_pressure = 1e8; // 1000 bar
    let mass_transfer_rate = 1e-9; // kg/s
    let thermal_diffusivity = 1e-5; // m²/s

    let total_energy_rate = calculator.calculate_complete_energy_rate(
        &state,
        internal_pressure,
        mass_transfer_rate,
        thermal_diffusivity,
    );

    // Total should be finite and physical
    let energy_watt = total_energy_rate.get::<watt>();
    assert!(energy_watt.is_finite());

    // During violent compression, multiple energy sinks should be active
    let chemical = calculator
        .calculate_chemical_reaction_rate(&state)
        .get::<watt>();
    let plasma = calculator
        .calculate_plasma_ionization_rate(&state)
        .get::<watt>();
    let radiation = calculator.calculate_radiation_losses(&state).get::<watt>();

    // All should be non-zero in this regime
    assert!(chemical < 0.0); // Energy absorption
    assert!(plasma < 0.0); // Energy absorption
    assert!(radiation < 0.0); // Energy loss
}

#[test]
fn test_energy_balance_options() {
    let params = BubbleParameters::default();

    // Calculator with all effects disabled
    let calc_disabled = EnergyBalanceCalculator::with_options(&params, false, false, false);
    let mut state = BubbleState::new(&params);
    state.temperature = 15_000.0;

    // Should return zero for all advanced terms
    assert_eq!(
        calc_disabled
            .calculate_chemical_reaction_rate(&state)
            .get::<watt>(),
        0.0
    );
    assert_eq!(
        calc_disabled
            .calculate_plasma_ionization_rate(&state)
            .get::<watt>(),
        0.0
    );
    assert_eq!(
        calc_disabled
            .calculate_radiation_losses(&state)
            .get::<watt>(),
        0.0
    );

    // Calculator with all effects enabled
    let calc_enabled = EnergyBalanceCalculator::with_options(&params, true, true, true);
    state.n_gas = 1e20;
    state.n_vapor = 5e19;

    // Should return non-zero for all advanced terms
    assert_ne!(
        calc_enabled
            .calculate_chemical_reaction_rate(&state)
            .get::<watt>(),
        0.0
    );
    assert_ne!(
        calc_enabled
            .calculate_plasma_ionization_rate(&state)
            .get::<watt>(),
        0.0
    );
    assert_ne!(
        calc_enabled
            .calculate_radiation_losses(&state)
            .get::<watt>(),
        0.0
    );
}
