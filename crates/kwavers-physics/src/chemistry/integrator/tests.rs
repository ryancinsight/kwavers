use super::*;
use crate::chemistry::ros_plasma::radical_kinetics::RadicalReaction;

/// Constant first-order decay: y(t) = y0 * exp(-k*t).
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_rk45_constant_first_order_decay() {
    let mut kinetics = RadicalKinetics::new(7.0, 298.15);
    kinetics.reactions.clear();
    kinetics.reactions.push(RadicalReaction {
        name: "OH first-order decay".to_string(),
        reactants: vec![(ROSSpecies::HydroxylRadical, 1.0)],
        products: vec![],
        rate_constant: 0.01,
        activation_energy: 0.0,
        ph_factor: 0.0,
    });

    let mut integrator = RadicalIntegrator::with_tolerances(kinetics, 1e-8, 1e-18);
    integrator.h_max = 10.0;

    let mut concs = HashMap::new();
    concs.insert(ROSSpecies::HydroxylRadical, 1.0);

    let (result, stats) = integrator
        .integrate(&concs, 0.0, 100.0, 298.15, 7.0)
        .unwrap();
    let oh_final = result[&ROSSpecies::HydroxylRadical];
    let analytical = (-1.0_f64).exp();

    assert!(
        (oh_final - analytical).abs() < 1e-6,
        "[OH](100 s) = {oh_final:.10}, expected {analytical:.10}; accepted = {}",
        stats.steps_accepted
    );
}

/// OH self-recombination half-life: [OH](t) = [OH0] / (1 + 2k[OH0]t).
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_oh_recombination_half_life() {
    let mut kinetics = RadicalKinetics::new(7.0, 298.15);
    kinetics
        .reactions
        .retain(|r| r.name.contains("self-recombination"));
    assert_eq!(kinetics.reactions.len(), 1, "expected exactly one reaction");

    let mut integrator = RadicalIntegrator::with_tolerances(kinetics, 1e-4, 1e-14);
    integrator.h_max = 1e-5;

    let oh0 = 1e-6_f64;
    let mut concs = HashMap::new();
    concs.insert(ROSSpecies::HydroxylRadical, oh0);

    let k = 5.5e9_f64;
    let t_half_analytical = 1.0 / (2.0 * k * oh0);

    let (result, _) = integrator
        .integrate(&concs, 0.0, t_half_analytical, 298.15, 7.0)
        .unwrap();
    let oh_half = result[&ROSSpecies::HydroxylRadical];
    let relative_err = (oh_half - oh0 / 2.0).abs() / (oh0 / 2.0);

    assert!(
        relative_err < 0.01,
        "[OH](t_half) = {oh_half:.4e}, expected {:.4e}; err = {relative_err:.4}",
        oh0 / 2.0
    );
}

#[test]
fn test_concentrations_remain_non_negative() {
    let kinetics = RadicalKinetics::new(7.0, 298.15);
    let integrator = RadicalIntegrator::new(kinetics);

    let mut concs = HashMap::new();
    concs.insert(ROSSpecies::HydroxylRadical, 1e-5);

    let (result, _) = integrator
        .integrate(&concs, 0.0, 1e-6, 298.15, 7.0)
        .unwrap();

    for (&_species, &conc) in &result {
        assert!(conc >= 0.0, "negative concentration detected: {conc}");
    }
}

/// Oxygen atom conservation for 2 OH -> H2O2.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_oxygen_mass_conservation_oh_recombination() {
    let mut kinetics = RadicalKinetics::new(7.0, 298.15);
    kinetics
        .reactions
        .retain(|r| r.name.contains("self-recombination"));

    let integrator = RadicalIntegrator::with_tolerances(kinetics, 1e-8, 1e-20);

    let oh0 = 1e-6_f64;
    let mut concs = HashMap::new();
    concs.insert(ROSSpecies::HydroxylRadical, oh0);
    concs.insert(ROSSpecies::HydrogenPeroxide, 0.0);

    let t_half = 1.0 / (2.0 * 5.5e9 * oh0);
    let (result, _) = integrator
        .integrate(&concs, 0.0, t_half, 298.15, 7.0)
        .unwrap();

    let oh_f = result
        .get(&ROSSpecies::HydroxylRadical)
        .copied()
        .unwrap_or(0.0);
    let h2o2_f = result
        .get(&ROSSpecies::HydrogenPeroxide)
        .copied()
        .unwrap_or(0.0);

    let o_initial = oh0;
    let o_final = oh_f + 2.0 * h2o2_f;
    let rel_err = (o_final - o_initial).abs() / o_initial;

    assert!(
        rel_err < 1e-5,
        "Oxygen conservation error = {rel_err:.2e}; initial = {o_initial:.4e}, final = {o_final:.4e}"
    );
}
