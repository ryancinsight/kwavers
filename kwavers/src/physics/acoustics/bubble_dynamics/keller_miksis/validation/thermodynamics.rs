//! Keller-Miksis thermodynamic auxiliary regression tests.
//!
//! Covers heat capacity, vapor mass transfer, adiabatic and conductive
//! temperature evolution, Van der Waals bubble pressure, and physical-
//! bound rejection on extreme states.

use crate::physics::acoustics::bubble_dynamics::bubble_state::{BubbleParameters, BubbleState};
use crate::physics::acoustics::bubble_dynamics::keller_miksis::KellerMiksisModel;

#[test]
fn test_heat_capacity_calculation() {
    let params = BubbleParameters::default();
    let model = KellerMiksisModel::new(params.clone());
    let state = BubbleState::new(&params);

    let cv = model.molar_heat_capacity_cv(&state);
    assert!(cv > 0.0, "Heat capacity should be positive");
}

#[test]
fn test_mass_transfer_evaporation() {
    let params = BubbleParameters {
        accommodation_coeff: 0.4,
        ..Default::default()
    };

    let model = KellerMiksisModel::new(params.clone());
    let mut state = BubbleState::new(&params);

    state.temperature = 350.0;
    state.pressure_internal = 101325.0;

    let n_vapor_initial = state.n_vapor;
    model.update_mass_transfer(&mut state, 1e-6)
        .expect("Mass transfer should succeed");
    assert!(
        state.n_vapor >= n_vapor_initial,
        "Evaporation should increase vapor content"
    );
}

#[test]
fn test_temperature_adiabatic_heating() {
    // Test adiabatic heating during compression, isolating the adiabatic term from
    // latent heat by setting accommodation_coeff = 0.
    //
    // With α = 0 the Hertz-Knudsen mass flux is zero, so dT/dt reduces to:
    //   (dT/dt) = -(γ-1)·T·(dR/dt)/R    [adiabatic]
    //           + heat conduction term     [≈ 0 when T ≈ T_liquid]
    //
    // For R₀ = 5 µm, Ṙ = -100 m/s, T = 293.15 K, γ = 1.4:
    //   dT/dt ≈ 0.4 × 293.15 × 100 / 5×10⁻⁶ ≈ 2.35×10⁹ K/s
    //
    // With dt = 1 ns: ΔT ≈ 2.35 K > 0  ✓
    //
    // Reference: Keller & Miksis (1980) J. Acoust. Soc. Am. 68(2):628–633.
    let params = BubbleParameters {
        accommodation_coeff: 0.0,
        ..Default::default()
    };
    let model = KellerMiksisModel::new(params.clone());
    let mut state = BubbleState::new(&params);

    state.wall_velocity = -100.0;
    let t_initial = state.temperature;

    model.update_temperature(&mut state, 1e-9)
        .expect("Temperature update should succeed (compression)");
    assert!(
        state.temperature > t_initial,
        "Compression must heat the gas (adiabatic term): T_init={:.4} K, T_final={:.4} K",
        t_initial,
        state.temperature
    );
}

#[test]
fn test_temperature_cooling() {
    let params = BubbleParameters::default();

    let model = KellerMiksisModel::new(params.clone());
    let mut state = BubbleState::new(&params);

    state.temperature = 350.0;
    state.wall_velocity = 0.0;
    state.n_gas = 1e15;

    let t_initial = state.temperature;
    model.update_temperature(&mut state, 1e-8)
        .expect("Temperature update should succeed (cooling)");
    let t_final = state.temperature;

    assert!(
        (t_final - t_initial).abs() < 10.0,
        "Temperature change should be reasonable: ΔT={}",
        t_final - t_initial
    );
}

#[test]
fn test_vdw_pressure_calculation() {
    let params = BubbleParameters {
        use_thermal_effects: true,
        ..Default::default()
    };

    let model = KellerMiksisModel::new(params.clone());
    let state = BubbleState::new(&params);

    let p_vdw = model.calculate_vdw_pressure(&state)
        .expect("VdW pressure calculation should succeed");
    assert!(p_vdw > 0.0, "VdW pressure should be positive");
    assert!(p_vdw.is_finite(), "VdW pressure should be finite");
}

#[test]
fn test_physical_bounds() {
    let params = BubbleParameters::default();
    let model = KellerMiksisModel::new(params.clone());
    let mut state = BubbleState::new(&params);

    state.temperature = 100000.0;

    let result = model.update_temperature(&mut state, 1.0);

    assert!(result.is_err(), "Extreme temperature should be rejected");
}
