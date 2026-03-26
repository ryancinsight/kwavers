//! ROS generation models based on physical conditions

use super::types::ROSSpecies;
use std::collections::HashMap;

/// Calculate ROS generation rates based on bubble conditions
#[must_use]
pub fn calculate_ros_generation(
    temperature: f64,
    pressure: f64,
    water_vapor_fraction: f64,
) -> HashMap<ROSSpecies, f64> {
    let mut generation_rates = HashMap::new();

    // High temperature dissociation of water vapor
    if temperature > 2000.0 {
        // H₂O → H• + •OH
        let k_dissociation = 1e13 * (-5.2e4 / temperature).exp(); // Arrhenius
        let oh_rate = k_dissociation * water_vapor_fraction * pressure / (8.314 * temperature);
        generation_rates.insert(ROSSpecies::HydroxylRadical, oh_rate);
        generation_rates.insert(ROSSpecies::AtomicHydrogen, oh_rate);

        // O₂ dissociation
        if temperature >= 3000.0 {
            // O2 dissociation temperature threshold
            let o2_fraction = 0.21 * (1.0 - water_vapor_fraction); // Air composition
            let k_o2 = 1e14 * (-6.0e4 / temperature).exp();
            let o_rate = 2.0 * k_o2 * o2_fraction * pressure / (8.314 * temperature);
            generation_rates.insert(ROSSpecies::AtomicOxygen, o_rate);
        }
    }

    // Secondary reactions at intermediate temperatures
    if temperature > 1000.0 && temperature < 3000.0 {
        // H₂O₂ formation: 2•OH → H₂O₂
        let oh_conc = generation_rates
            .get(&ROSSpecies::HydroxylRadical)
            .unwrap_or(&0.0);
        let h2o2_rate = 1e-10 * oh_conc * oh_conc; // Second order
        generation_rates.insert(ROSSpecies::HydrogenPeroxide, h2o2_rate);

        // Superoxide formation: O₂ + e⁻ → O₂•⁻
        if temperature > 1500.0 {
            let ionization_fraction = (temperature / 20000.0).min(0.1);
            let o2_rate = 1e-12 * ionization_fraction * pressure / (8.314 * temperature);
            generation_rates.insert(ROSSpecies::Superoxide, o2_rate);
        }
    }

    generation_rates
}
