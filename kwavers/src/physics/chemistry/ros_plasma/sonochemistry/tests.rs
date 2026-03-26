use super::model::{BubbleState, SonochemistryModel};
use super::super::ros_species::ROSSpecies;
use ndarray::Array3;

#[test]
fn test_zeldovich_ros_generation_rates() {
    use crate::physics::chemistry::ros_plasma::ros_species::calculate_ros_generation;

    let temperature = 2800.0;
    let pressure = 1e9;
    let water_vapor_fraction = 0.5;

    let rates = calculate_ros_generation(temperature, pressure, water_vapor_fraction);

    // Analytical derivation from Arrhenius: k_dissoc = 1e13 * exp(-E_a / T)
    // where E_a = 52,000 K for H2O dissociation
    let k_dissoc = 1e13 * f64::exp(-5.2e4 / temperature);
    
    // OH generation rate = k_dissoc * [H2O] * P / (R * T)
    let expected_oh_rate = k_dissoc * water_vapor_fraction * pressure / (8.314 * temperature);
    
    // H2O2 recombination rate = 1e-10 * [OH]^2
    let expected_h2o2_rate = 1e-10 * expected_oh_rate * expected_oh_rate;

    let actual_oh_rate = rates.get(&ROSSpecies::HydroxylRadical).unwrap();
    let actual_h2o2_rate = rates.get(&ROSSpecies::HydrogenPeroxide).unwrap();

    assert!((actual_oh_rate - expected_oh_rate).abs() < 1e-8 * expected_oh_rate, "OH rate must match Arrhenius prediction exactly");
    assert!((actual_h2o2_rate - expected_h2o2_rate).abs() < 1e-8 * expected_h2o2_rate, "H2O2 rate must match second-order formation exactly");
}

#[test]
fn test_ph_update() {
    let mut model = SonochemistryModel::new(3, 3, 3, 7.0);

    if let Some(h2o2) = model
        .ros_concentrations
        .get_mut(ROSSpecies::HydrogenPeroxide)
    {
        h2o2[[1, 1, 1]] = 1e-3; // 1 mM
    }

    let initial_ph = model.ph_field[[1, 1, 1]];
    model.update_ph(1.0);
    let final_ph = model.ph_field[[1, 1, 1]];

    // Mathematical Proof: pH starts at 7.0.
    // update_ph adds: dt * (-0.1 * h2o2/1e-3 + 0.5 * oh/1e-6)
    // h2o2 is 1e-3 -> -0.1 * 1.0 = -0.1
    // oh is 0.0 -> 0.0
    // new_ph = 7.0 + 1.0 * (-0.1) = 6.9
    let expected_ph = 6.9;
    assert!((final_ph - expected_ph).abs() < 1e-12, "pH update must strictly follow derived acidification rate");
}
