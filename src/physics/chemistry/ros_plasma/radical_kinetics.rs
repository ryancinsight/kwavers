//! Radical reaction kinetics in aqueous phase
//!
//! Reactions of ROS in the liquid surrounding the bubble

use super::ros_species::ROSSpecies;
use std::collections::HashMap;

/// Radical reaction in aqueous phase
#[derive(Debug, Clone)]
pub struct RadicalReaction {
    /// Reaction name
    pub name: String,
    /// Reactant species
    pub reactants: Vec<(ROSSpecies, f64)>,
    /// Product species
    pub products: Vec<(ROSSpecies, f64)>,
    /// Rate constant at 25°C (M⁻ⁿ⁺¹·s⁻¹ where n is reaction order)
    pub rate_constant: f64,
    /// Activation energy (J/mol)
    pub activation_energy: f64,
    /// pH dependence factor
    pub ph_factor: f64,
}

/// Radical kinetics solver
#[derive(Debug)]
pub struct RadicalKinetics {
    /// List of radical reactions
    pub reactions: Vec<RadicalReaction>,
    /// Current pH
    pub ph: f64,
    /// Temperature (K)
    pub temperature: f64,
}

impl RadicalKinetics {
    /// Create new radical kinetics solver
    pub fn new(ph: f64, temperature: f64) -> Self {
        let mut kinetics = Self {
            reactions: Vec::new(),
            ph,
            temperature,
        };

        kinetics.add_standard_reactions();
        kinetics
    }

    /// Add standard radical reactions in water
    fn add_standard_reactions(&mut self) {
        // Hydroxyl radical reactions
        self.reactions.push(RadicalReaction {
            name: "OH self-recombination".to_string(),
            reactants: vec![(ROSSpecies::HydroxylRadical, 2.0)],
            products: vec![(ROSSpecies::HydrogenPeroxide, 1.0)],
            rate_constant: 5.5e9, // M⁻¹·s⁻¹
            activation_energy: 0.0,
            ph_factor: 0.0,
        });

        // Superoxide reactions
        self.reactions.push(RadicalReaction {
            name: "Superoxide dismutation".to_string(),
            reactants: vec![(ROSSpecies::Superoxide, 2.0)],
            products: vec![
                (ROSSpecies::HydrogenPeroxide, 1.0),
                (ROSSpecies::SingletOxygen, 1.0),
            ],
            rate_constant: 1e8, // pH dependent
            activation_energy: 0.0,
            ph_factor: -1.0, // Faster at low pH
        });

        // Fenton-like reactions
        self.reactions.push(RadicalReaction {
            name: "H2O2 + OH".to_string(),
            reactants: vec![
                (ROSSpecies::HydrogenPeroxide, 1.0),
                (ROSSpecies::HydroxylRadical, 1.0),
            ],
            products: vec![(ROSSpecies::HydroperoxylRadical, 1.0)],
            rate_constant: 2.7e7,      // M⁻¹·s⁻¹
            activation_energy: 2800.0, // J/mol
            ph_factor: 0.0,
        });

        // Ozone reactions
        self.reactions.push(RadicalReaction {
            name: "Ozone + OH".to_string(),
            reactants: vec![(ROSSpecies::Ozone, 1.0), (ROSSpecies::HydroxylRadical, 1.0)],
            products: vec![(ROSSpecies::HydroperoxylRadical, 1.0)],
            rate_constant: 1.1e8, // M⁻¹·s⁻¹
            activation_energy: 0.0,
            ph_factor: 0.0,
        });

        // Singlet oxygen quenching
        self.reactions.push(RadicalReaction {
            name: "Singlet oxygen decay".to_string(),
            reactants: vec![(ROSSpecies::SingletOxygen, 1.0)],
            products: vec![],     // Decays to ground state O2
            rate_constant: 2.9e5, // s⁻¹ in water
            activation_energy: 0.0,
            ph_factor: 0.0,
        });
    }

    /// Calculate rate constant at current conditions
    pub fn rate_constant(&self, reaction: &RadicalReaction) -> f64 {
        let r_gas = 8.314;

        // Arrhenius temperature dependence
        let k_t = reaction.rate_constant
            * (-reaction.activation_energy * (1.0 / self.temperature - 1.0 / 298.15) / r_gas).exp();

        // pH dependence
        let ph_correction = if reaction.ph_factor != 0.0 {
            10f64.powf(reaction.ph_factor * (self.ph - 7.0))
        } else {
            1.0
        };

        k_t * ph_correction
    }

    /// Calculate reaction rates for given concentrations
    pub fn calculate_rates(
        &self,
        concentrations: &HashMap<ROSSpecies, f64>,
    ) -> HashMap<ROSSpecies, f64> {
        let mut rate_changes: HashMap<ROSSpecies, f64> = HashMap::new();

        for reaction in &self.reactions {
            let k = self.rate_constant(reaction);

            // Calculate reaction rate
            let mut rate = k;
            for (species, stoich) in &reaction.reactants {
                if let Some(&conc) = concentrations.get(species) {
                    rate *= conc.powf(*stoich);
                } else {
                    rate = 0.0;
                    break;
                }
            }

            // Apply stoichiometry
            for (species, stoich) in &reaction.reactants {
                *rate_changes.entry(*species).or_insert(0.0) -= rate * stoich;
            }

            for (species, stoich) in &reaction.products {
                *rate_changes.entry(*species).or_insert(0.0) += rate * stoich;
            }
        }

        rate_changes
    }

    /// Add scavenger reactions (e.g., for dosimetry)
    pub fn add_scavenger(
        &mut self,
        scavenger_name: &str,
        oh_rate: f64,
        other_rates: HashMap<ROSSpecies, f64>,
    ) {
        // OH scavenging
        if oh_rate > 0.0 {
            self.reactions.push(RadicalReaction {
                name: format!("{} + OH", scavenger_name),
                reactants: vec![(ROSSpecies::HydroxylRadical, 1.0)],
                products: vec![], // Products not tracked
                rate_constant: oh_rate,
                activation_energy: 0.0,
                ph_factor: 0.0,
            });
        }

        // Other ROS scavenging
        for (species, rate) in other_rates {
            if rate > 0.0 {
                self.reactions.push(RadicalReaction {
                    name: format!("{} + {:?}", scavenger_name, species),
                    reactants: vec![(species, 1.0)],
                    products: vec![],
                    rate_constant: rate,
                    activation_energy: 0.0,
                    ph_factor: 0.0,
                });
            }
        }
    }
}

/// Calculate hydroxyl radical yield (G-value) from energy deposition
pub fn calculate_oh_yield(energy_density: f64, ph: f64) -> f64 {
    // G-value for OH production (molecules/100 eV)
    let g_oh_neutral = 2.7; // At neutral pH

    // pH correction
    let ph_factor = if ph < 3.0 {
        0.5 // Reduced at very low pH
    } else if ph > 11.0 {
        1.5 // Increased at high pH
    } else {
        1.0
    };

    // Convert energy density (J/m³) to OH concentration (mol/m³)
    let ev_per_joule = 6.242e18;
    let molecules_per_mole = 6.022e23;

    g_oh_neutral * ph_factor * energy_density * ev_per_joule / (100.0 * molecules_per_mole)
}

/// Estimate radical diffusion length before recombination
pub fn radical_diffusion_length(species: ROSSpecies, concentration: f64) -> f64 {
    let d = species.diffusion_coefficient();
    let lifetime = species.lifetime_water();

    // For second-order recombination
    let k_recomb = match species {
        ROSSpecies::HydroxylRadical => 5.5e9, // M⁻¹·s⁻¹
        ROSSpecies::AtomicHydrogen => 1e10,
        _ => 1e8, // Default
    };

    // Effective lifetime considering recombination
    let conc_m = concentration / 1000.0; // Convert to M
    let tau_eff = 1.0 / (1.0 / lifetime + k_recomb * conc_m);

    // Diffusion length
    (2.0 * d * tau_eff).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radical_kinetics() {
        let kinetics = RadicalKinetics::new(7.0, 298.15);

        // Test rate constant calculation
        let oh_recomb = &kinetics.reactions[0];
        let k = kinetics.rate_constant(oh_recomb);
        assert!((k - 5.5e9).abs() < 1e7); // Should be close to literature value
    }

    #[test]
    fn test_ph_dependence() {
        let kinetics_acid = RadicalKinetics::new(3.0, 298.15);
        let kinetics_base = RadicalKinetics::new(11.0, 298.15);

        // Find superoxide dismutation reaction
        let reaction = kinetics_acid
            .reactions
            .iter()
            .find(|r| r.name.contains("dismutation"))
            .unwrap();

        let k_acid = kinetics_acid.rate_constant(reaction);
        let k_base = kinetics_base.rate_constant(reaction);

        assert!(k_acid > k_base); // Should be faster at low pH
    }

    #[test]
    fn test_oh_yield() {
        let energy = 1e6; // J/m³
        let yield_neutral = calculate_oh_yield(energy, 7.0);
        let yield_acid = calculate_oh_yield(energy, 2.0);

        assert!(yield_neutral > 0.0);
        assert!(yield_acid < yield_neutral); // Lower at acidic pH
    }

    #[test]
    fn test_diffusion_length() {
        let length = radical_diffusion_length(ROSSpecies::HydroxylRadical, 1e-3);
        assert!(length > 0.0 && length < 1e-6); // Should be sub-micron
    }
}
