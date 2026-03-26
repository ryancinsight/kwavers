use approx::assert_relative_eq;
use super::chemistry::PlasmaChemistry;
use super::reaction::{zeldovich_no_rate, PlasmaReaction};

#[test]
fn test_plasma_reaction_rate() {
    // Ground truth analytical derivation:
    // k = A * exp(-E_a / (R * T))
    // Given A = 1e10, E_a = 100,000 J/mol, R = 8.314462618 J/mol/K (CODATA)
    // At T = 300K: k ≈ 1e10 * exp(-100000 / (8.314462618 * 300)) ≈ 3.87966e-8
    // At T = 1000K: k ≈ 1e10 * exp(-100000 / (8.314462618 * 1000)) ≈ 59791.3
    
    let reaction = PlasmaReaction {
        name: "Test".to_string(),
        reactants: vec![("A".to_string(), 1.0)],
        products: vec![("B".to_string(), 1.0)],
        activation_energy: 100e3, // 100 kJ/mol
        pre_exponential: 1e10,
        temperature_exponent: 0.0,
    };

    let k_300 = reaction.rate_constant(300.0);
    let k_1000 = reaction.rate_constant(1000.0);

    // Assert against CODATA-derived values (R = 8.314462618)
    assert_relative_eq!(k_300, 3.8796566557e-8, epsilon = 1e-12);
    assert_relative_eq!(k_1000, 5.9791298866e4, epsilon = 1e-1);
}

#[test]
fn test_plasma_chemistry() {
    let mut plasma = PlasmaChemistry::new(3000.0, 101325.0);

    // Initial concentrations should be set
    assert!(plasma.concentrations.get("H2O").unwrap() > &0.0);
    assert!(plasma.concentrations.get("N2").unwrap() > &0.0);

    // Update should change concentrations
    let h2o_initial = *plasma.concentrations.get("H2O").unwrap();
    plasma.update(1e-6); // 1 microsecond
    let h2o_final = *plasma.concentrations.get("H2O").unwrap();

    assert!(h2o_final < h2o_initial); // Water should dissociate
    assert!(plasma.concentrations.get("OH").unwrap() > &1e-10); // OH should form
}

#[test]
fn test_zeldovich_no() {
    let rate = zeldovich_no_rate(2000.0, 0.01, 0.04); // mol/m³
    assert!(rate > 0.0);

    let rate_cold = zeldovich_no_rate(1000.0, 0.01, 0.04);
    assert_eq!(rate_cold, 0.0); // Too cold
}
