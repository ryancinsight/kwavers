//! Plasma chemistry reactions in sonoluminescence
//!
//! High-temperature reactions occurring in the bubble interior during collapse

use super::ros_species::ROSSpecies;
use std::collections::HashMap;

/// Plasma reaction types
#[derive(Debug, Clone)]
pub struct PlasmaReaction {
    /// Reaction name
    pub name: String,
    /// Reactants and their stoichiometric coefficients
    pub reactants: Vec<(String, f64)>,
    /// Products and their stoichiometric coefficients
    pub products: Vec<(String, f64)>,
    /// Activation energy (J/mol)
    pub activation_energy: f64,
    /// Pre-exponential factor (units depend on reaction order)
    pub pre_exponential: f64,
    /// Temperature exponent for modified Arrhenius
    pub temperature_exponent: f64,
}

impl PlasmaReaction {
    /// Calculate rate constant at given temperature
    pub fn rate_constant(&self, temperature: f64) -> f64 {
        let r_gas = 8.314; // J/(mol·K)
        self.pre_exponential
            * temperature.powf(self.temperature_exponent)
            * (-self.activation_energy / (r_gas * temperature)).exp()
    }
}

/// Plasma chemistry model
#[derive(Debug, Debug)]
pub struct PlasmaChemistry {
    /// List of plasma reactions
    pub reactions: Vec<PlasmaReaction>,
    /// Species concentrations (mol/m³)
    pub concentrations: HashMap<String, f64>,
    /// Temperature (K)
    pub temperature: f64,
    /// Pressure (Pa)
    pub pressure: f64,
}

impl PlasmaChemistry {
    /// Create new plasma chemistry model
    pub fn new(temperature: f64, pressure: f64) -> Self {
        let mut chemistry = Self {
            reactions: Vec::new(),
            concentrations: HashMap::new(),
            temperature,
            pressure,
        };

        // Initialize with standard plasma reactions
        chemistry.add_standard_reactions();
        let _ = chemistry.initialize_concentrations();

        chemistry
    }

    /// Add standard plasma reactions for water vapor and air
    fn add_standard_reactions(&mut self) {
        // Water dissociation
        self.reactions.push(PlasmaReaction {
            name: "Water dissociation".to_string(),
            reactants: vec![("H2O".to_string(), 1.0)],
            products: vec![("H".to_string(), 1.0), ("OH".to_string(), 1.0)],
            activation_energy: 498e3, // J/mol
            pre_exponential: 2.2e10,
            temperature_exponent: 1.0,
        });

        // Oxygen dissociation
        self.reactions.push(PlasmaReaction {
            name: "Oxygen dissociation".to_string(),
            reactants: vec![("O2".to_string(), 1.0)],
            products: vec![("O".to_string(), 2.0)],
            activation_energy: 495e3, // J/mol
            pre_exponential: 3.6e18,
            temperature_exponent: -1.0,
        });

        // Nitrogen dissociation
        self.reactions.push(PlasmaReaction {
            name: "Nitrogen dissociation".to_string(),
            reactants: vec![("N2".to_string(), 1.0)],
            products: vec![("N".to_string(), 2.0)],
            activation_energy: 945e3, // J/mol
            pre_exponential: 7.0e21,
            temperature_exponent: -1.5,
        });

        // OH radical recombination
        self.reactions.push(PlasmaReaction {
            name: "OH recombination".to_string(),
            reactants: vec![("OH".to_string(), 2.0)],
            products: vec![("H2O2".to_string(), 1.0)],
            activation_energy: 0.0, // No barrier
            pre_exponential: 1.5e9,
            temperature_exponent: 0.0,
        });

        // H atom recombination
        self.reactions.push(PlasmaReaction {
            name: "H recombination".to_string(),
            reactants: vec![("H".to_string(), 2.0)],
            products: vec![("H2".to_string(), 1.0)],
            activation_energy: 0.0,
            pre_exponential: 1.0e10,
            temperature_exponent: 0.0,
        });

        // NO formation (Zeldovich mechanism)
        self.reactions.push(PlasmaReaction {
            name: "NO formation".to_string(),
            reactants: vec![("N".to_string(), 1.0), ("O2".to_string(), 1.0)],
            products: vec![("NO".to_string(), 1.0), ("O".to_string(), 1.0)],
            activation_energy: 315e3, // J/mol
            pre_exponential: 1.8e14,
            temperature_exponent: 0.0,
        });

        // Ozone formation
        self.reactions.push(PlasmaReaction {
            name: "Ozone formation".to_string(),
            reactants: vec![("O".to_string(), 1.0), ("O2".to_string(), 1.0)],
            products: vec![("O3".to_string(), 1.0)],
            activation_energy: 0.0,
            pre_exponential: 6.0e9,
            temperature_exponent: 0.0,
        });

        // Ionization reactions at very high temperatures
        if self.temperature > 10000.0 {
            // Electron impact ionization of hydrogen
            self.reactions.push(PlasmaReaction {
                name: "H ionization".to_string(),
                reactants: vec![("H".to_string(), 1.0)],
                products: vec![("H+".to_string(), 1.0), ("e-".to_string(), 1.0)],
                activation_energy: 1312e3, // J/mol (13.6 eV)
                pre_exponential: 1e15,
                temperature_exponent: 0.5,
            });
        }
    }

    /// Initialize species concentrations based on initial conditions
    fn initialize_concentrations(&mut self) -> crate::KwaversResult<()> {
        let r_gas = 8.314;
        if self.temperature <= 0.0 {
            return Err(crate::KwaversError::InvalidInput(
                "Temperature must be greater than 0 K to initialize concentrations".to_string(),
            ));
        }
        let total_conc = self.pressure / (r_gas * self.temperature);

        // Assume initial composition: 50% water vapor, 50% air
        self.concentrations
            .insert("H2O".to_string(), 0.5 * total_conc);
        self.concentrations
            .insert("N2".to_string(), 0.395 * total_conc); // 79% of air
        self.concentrations
            .insert("O2".to_string(), 0.105 * total_conc); // 21% of air

        // Initialize other species at trace levels
        self.concentrations
            .insert("H".to_string(), 1e-10 * total_conc);
        self.concentrations
            .insert("OH".to_string(), 1e-10 * total_conc);
        self.concentrations
            .insert("O".to_string(), 1e-10 * total_conc);
        self.concentrations
            .insert("N".to_string(), 1e-10 * total_conc);
        self.concentrations
            .insert("H2".to_string(), 1e-10 * total_conc);
        self.concentrations.insert("H2O2".to_string(), 0.0);
        self.concentrations.insert("NO".to_string(), 0.0);
        self.concentrations.insert("O3".to_string(), 0.0);

        if self.temperature > 10000.0 {
            self.concentrations
                .insert("H+".to_string(), 1e-15 * total_conc);
            self.concentrations
                .insert("e-".to_string(), 1e-15 * total_conc);
        }
        Ok(())
    }

    /// Update concentrations for one time step
    pub fn update(&mut self, dt: f64) {
        let mut rate_changes: HashMap<String, f64> = HashMap::new();

        // Calculate rates for all reactions
        for reaction in &self.reactions {
            let k = reaction.rate_constant(self.temperature);

            // Calculate reaction rate (assuming elementary reactions)
            let mut rate = k;
            for (species, coeff) in &reaction.reactants {
                if let Some(&conc) = self.concentrations.get(species) {
                    rate *= conc.powf(*coeff);
                } else {
                    rate = 0.0;
                    break;
                }
            }

            // Apply rate changes
            for (species, coeff) in &reaction.reactants {
                *rate_changes.entry(species.clone()).or_insert(0.0) -= rate * coeff;
            }

            for (species, coeff) in &reaction.products {
                *rate_changes.entry(species.clone()).or_insert(0.0) += rate * coeff;
            }
        }

        // Update concentrations
        for (species, rate_change) in rate_changes {
            if let Some(conc) = self.concentrations.get_mut(&species) {
                *conc = (*conc + rate_change * dt).max(0.0);
            }
        }
    }

    /// Get ROS concentrations from plasma species
    pub fn get_ros_concentrations(&self) -> HashMap<ROSSpecies, f64> {
        let mut ros = HashMap::new();

        // Map plasma species to ROS species
        if let Some(&oh) = self.concentrations.get("OH") {
            ros.insert(ROSSpecies::HydroxylRadical, oh);
        }
        if let Some(&h2o2) = self.concentrations.get("H2O2") {
            ros.insert(ROSSpecies::HydrogenPeroxide, h2o2);
        }
        if let Some(&o3) = self.concentrations.get("O3") {
            ros.insert(ROSSpecies::Ozone, o3);
        }
        if let Some(&o) = self.concentrations.get("O") {
            ros.insert(ROSSpecies::AtomicOxygen, o);
        }
        if let Some(&h) = self.concentrations.get("H") {
            ros.insert(ROSSpecies::AtomicHydrogen, h);
        }

        ros
    }

    /// Calculate degree of ionization
    pub fn ionization_fraction(&self) -> f64 {
        let total_conc: f64 = self.concentrations.values().sum();
        let electron_conc = self.concentrations.get("e-").unwrap_or(&0.0);

        if total_conc > 0.0 {
            electron_conc / total_conc
        } else {
            0.0
        }
    }

    /// Calculate equilibrium composition at given temperature
    pub fn equilibrium_composition(&mut self, max_iterations: usize, tolerance: f64) {
        for _ in 0..max_iterations {
            let old_concentrations = self.concentrations.clone();

            // Large time step to approach equilibrium
            self.update(1.0);

            // Check convergence
            let mut max_change: f64 = 0.0;
            for (species, &old_conc) in &old_concentrations {
                if let Some(&new_conc) = self.concentrations.get(species) {
                    let change = ((new_conc - old_conc) / old_conc.max(1e-20)).abs();
                    max_change = max_change.max(change);
                }
            }

            if max_change < tolerance {
                break;
            }
        }
    }
}

/// Calculate NO production rate (Zeldovich mechanism)
pub fn zeldovich_no_rate(temperature: f64, o2_conc: f64, n2_conc: f64) -> f64 {
    if temperature < 1800.0 {
        return 0.0; // Too cold for thermal NO
    }

    // Equilibrium O atom concentration
    let k_o2_diss = 3.6e18 * temperature.powf(-1.0) * (-495e3 / (8.314 * temperature)).exp();
    let o_eq = (k_o2_diss * o2_conc).sqrt();

    // NO formation rate
    let k_no = 1.8e14 * (-315e3 / (8.314 * temperature)).exp();
    k_no * o_eq * n2_conc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plasma_reaction_rate() {
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

        assert!(k_1000 > k_300); // Rate increases with temperature
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
}
