//! Plasma chemistry solver
//!
//! Models plasma-phase chemical kinetics within collapsing cavitation bubbles.

use super::super::ros_species::ROSSpecies;
use super::reaction::PlasmaReaction;
use std::collections::HashMap;

/// Plasma chemistry model
#[derive(Debug)]
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
    #[must_use]
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
    fn initialize_concentrations(&mut self) -> crate::core::error::KwaversResult<()> {
        let r_gas = 8.314;
        if self.temperature <= 0.0 {
            return Err(crate::core::error::KwaversError::InvalidInput(
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
    #[must_use]
    pub fn get_ros_concentrations(&self) -> HashMap<ROSSpecies, f64> {
        let mut ros = HashMap::new();

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
    #[must_use]
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
