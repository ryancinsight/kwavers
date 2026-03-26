//! ROS concentrations tracking and integration

use super::types::ROSSpecies;
use ndarray::Array3;
use std::collections::HashMap;

use crate::core::constants::chemistry::{
    HYDROGEN_PEROXIDE_WEIGHT, HYDROXYL_RADICAL_WEIGHT, NITRIC_OXIDE_WEIGHT, PEROXYNITRITE_WEIGHT,
    SINGLET_OXYGEN_WEIGHT, SUPEROXIDE_WEIGHT,
};

/// Container for ROS concentrations in the simulation
#[derive(Debug)]
pub struct ROSConcentrations {
    /// Concentration fields for each ROS species (mol/m³)
    pub fields: HashMap<ROSSpecies, Array3<f64>>,
    /// Total ROS concentration field
    pub total_ros: Array3<f64>,
    /// Grid dimensions
    pub(crate) shape: (usize, usize, usize),
}

impl ROSConcentrations {
    /// Create new ROS concentration container
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        let shape = (nx, ny, nz);
        let mut fields = HashMap::new();

        // Initialize all ROS species fields
        for species in [
            ROSSpecies::HydroxylRadical,
            ROSSpecies::HydrogenPeroxide,
            ROSSpecies::Superoxide,
            ROSSpecies::SingletOxygen,
            ROSSpecies::Ozone,
            ROSSpecies::HydroperoxylRadical,
            ROSSpecies::AtomicOxygen,
            ROSSpecies::AtomicHydrogen,
            ROSSpecies::Peroxynitrite,
            ROSSpecies::NitricOxide,
        ] {
            fields.insert(species, Array3::zeros(shape));
        }

        Self {
            fields,
            total_ros: Array3::zeros(shape),
            shape,
        }
    }

    /// Get concentration field for a specific species
    #[must_use]
    pub fn get(&self, species: ROSSpecies) -> Option<&Array3<f64>> {
        self.fields.get(&species)
    }

    /// Get mutable concentration field for a specific species
    pub fn get_mut(&mut self, species: ROSSpecies) -> Option<&mut Array3<f64>> {
        self.fields.get_mut(&species)
    }

    /// Update total ROS concentration
    pub fn update_total(&mut self) {
        self.total_ros.fill(0.0);
        for conc in self.fields.values() {
            self.total_ros = &self.total_ros + conc;
        }
    }

    /// Calculate oxidative stress index
    #[must_use]
    pub fn oxidative_stress_index(&self) -> f64 {
        let mut total_stress = 0.0;

        for (species, conc) in &self.fields {
            let weight = match species {
                ROSSpecies::HydroxylRadical => HYDROXYL_RADICAL_WEIGHT, // Most damaging
                ROSSpecies::HydrogenPeroxide => HYDROGEN_PEROXIDE_WEIGHT,
                ROSSpecies::Superoxide => SUPEROXIDE_WEIGHT,
                ROSSpecies::SingletOxygen => SINGLET_OXYGEN_WEIGHT,
                ROSSpecies::Peroxynitrite => PEROXYNITRITE_WEIGHT,
                ROSSpecies::NitricOxide => NITRIC_OXIDE_WEIGHT,
                ROSSpecies::Ozone => 0.3, // Moderate oxidative stress
                ROSSpecies::AtomicHydrogen => 0.1, // Low oxidative stress
                ROSSpecies::HydroperoxylRadical => 0.4, // Moderate stress
                ROSSpecies::AtomicOxygen => 0.5, // High reactivity
                ROSSpecies::NitrogenDioxide => 0.3, // Moderate stress
            };

            total_stress += weight * conc.sum();
        }

        total_stress / self.fields.values().next().map_or(1.0, |f| f.len() as f64)
    }

    /// Apply decay based on species lifetime
    pub fn apply_decay(&mut self, dt: f64) {
        for (species, conc) in &mut self.fields {
            let lifetime = species.lifetime_water();
            let decay_rate = 1.0 / lifetime;

            // Exponential decay: C(t+dt) = C(t) * exp(-dt/τ)
            conc.mapv_inplace(|c| c * (-dt * decay_rate).exp());
        }
    }
}
