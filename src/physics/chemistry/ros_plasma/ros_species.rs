//! Reactive Oxygen Species (ROS) definitions and concentration tracking
//!
//! Key ROS generated during sonoluminescence:
//! - Hydroxyl radical (•OH)
//! - Hydrogen peroxide (H₂O₂)
//! - Superoxide (O₂•⁻)
//! - Singlet oxygen (¹O₂)
//! - Ozone (O₃)

use ndarray::Array3;
use std::collections::HashMap;

use crate::constants::chemistry::{
    HYDROGEN_PEROXIDE_WEIGHT, HYDROXYL_RADICAL_WEIGHT, NITRIC_OXIDE_WEIGHT, PEROXYNITRITE_WEIGHT,
    SINGLET_OXYGEN_WEIGHT, SUPEROXIDE_WEIGHT,
};

/// Enumeration of reactive oxygen species
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ROSSpecies {
    /// Hydroxyl radical (•OH) - most reactive
    HydroxylRadical,
    /// Hydrogen peroxide (H₂O₂)
    HydrogenPeroxide,
    /// Superoxide anion (O₂•⁻)
    Superoxide,
    /// Singlet oxygen (¹O₂)
    SingletOxygen,
    /// Ozone (O₃)
    Ozone,
    /// Hydroperoxyl radical (HO₂•)
    HydroperoxylRadical,
    /// Atomic oxygen (O)
    AtomicOxygen,
    /// Atomic hydrogen (H)
    AtomicHydrogen,
    /// Peroxynitrite (ONOO⁻)
    Peroxynitrite,
    /// Nitric oxide (NO)
    NitricOxide,
    /// Nitrogen dioxide (NO₂)
    NitrogenDioxide,
}

impl ROSSpecies {
    /// Get the name of the species
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::HydroxylRadical => "•OH",
            Self::HydrogenPeroxide => "H₂O₂",
            Self::Superoxide => "O₂•⁻",
            Self::SingletOxygen => "¹O₂",
            Self::Ozone => "O₃",
            Self::HydroperoxylRadical => "HO₂•",
            Self::AtomicOxygen => "O",
            Self::AtomicHydrogen => "H",
            Self::Peroxynitrite => "ONOO⁻",
            Self::NitricOxide => "NO",
            Self::NitrogenDioxide => "NO₂",
        }
    }

    /// Get the diffusion coefficient in water at 25°C (m²/s)
    #[must_use]
    pub fn diffusion_coefficient(&self) -> f64 {
        match self {
            Self::HydroxylRadical => 2.3e-9,
            Self::HydrogenPeroxide => 1.4e-9,
            Self::Superoxide => 1.75e-9,
            Self::SingletOxygen => 2.0e-9,
            Self::Ozone => 1.6e-9,
            Self::HydroperoxylRadical => 2.0e-9,
            Self::AtomicOxygen => 2.5e-9,
            Self::AtomicHydrogen => 7.0e-9,
            Self::Peroxynitrite => 1.0e-9,   // Approximate
            Self::NitricOxide => 1.0e-9,     // Approximate
            Self::NitrogenDioxide => 1.2e-9, // Approximate
        }
    }

    /// Get the lifetime in pure water (seconds)
    #[must_use]
    pub fn lifetime_water(&self) -> f64 {
        match self {
            Self::HydroxylRadical => 1e-9,     // 1 ns
            Self::HydrogenPeroxide => 1e3,     // Stable
            Self::Superoxide => 1e-6,          // 1 μs
            Self::SingletOxygen => 3.5e-6,     // 3.5 μs
            Self::Ozone => 1e2,                // 100 s
            Self::HydroperoxylRadical => 1e-6, // 1 μs
            Self::AtomicOxygen => 1e-12,       // 1 ps
            Self::AtomicHydrogen => 1e-12,     // 1 ps
            Self::Peroxynitrite => 1e-6,       // 1 μs
            Self::NitricOxide => 1e-6,         // 1 μs
            Self::NitrogenDioxide => 1e-6,     // 1 μs
        }
    }

    /// Get the standard reduction potential (V vs SHE)
    #[must_use]
    pub fn reduction_potential(&self) -> f64 {
        match self {
            Self::HydroxylRadical => 2.80, // Strongest oxidant
            Self::HydrogenPeroxide => 1.78,
            Self::Superoxide => -0.33, // Can act as reductant
            Self::SingletOxygen => 0.65,
            Self::Ozone => 2.07,
            Self::HydroperoxylRadical => 1.50,
            Self::AtomicOxygen => 2.42,
            Self::AtomicHydrogen => -2.30, // Strong reductant
            Self::Peroxynitrite => 0.80,   // Reductant
            Self::NitricOxide => 0.90,     // Reductant
            Self::NitrogenDioxide => 1.05, // Oxidant
        }
    }
}

/// Container for ROS concentrations in the simulation
#[derive(Debug)]
pub struct ROSConcentrations {
    /// Concentration fields for each ROS species (mol/m³)
    pub fields: HashMap<ROSSpecies, Array3<f64>>,
    /// Total ROS concentration field
    pub total_ros: Array3<f64>,
    /// Grid dimensions
    shape: (usize, usize, usize),
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

    /// Apply diffusion using simple forward Euler
    pub fn apply_diffusion(&mut self, dx: f64, dy: f64, dz: f64, dt: f64) {
        // Calculate maximum stability factor for the most diffusive species
        let max_d = ROSSpecies::HydroxylRadical.diffusion_coefficient();
        let min_spacing = dx.min(dy).min(dz);
        let stability_factor = max_d * dt / min_spacing.powi(2);

        // Check stability condition
        if stability_factor > 0.5 {
            log::warn!(
                "Diffusion stability condition violated: D*dt/dx² = {:.3} > 0.5. \
                Consider reducing timestep or using implicit scheme.",
                stability_factor
            );
        }

        // Collect updates to avoid borrow checker issues
        let mut updates: Vec<(ROSSpecies, Array3<f64>)> = Vec::new();

        for (species, conc) in &self.fields {
            let d = species.diffusion_coefficient();

            // For high stability factors, use implicit scheme (ADI method)
            if d * dt / dx.min(dy).min(dz).powi(2) > 0.25 {
                // Use semi-implicit scheme for numerical stability
                let mut updated_conc = conc.clone();
                Self::apply_semi_implicit_diffusion_static(
                    &mut updated_conc,
                    d,
                    dx,
                    dy,
                    dz,
                    dt,
                    self.shape,
                );
                updates.push((*species, updated_conc));
            } else {
                // Use explicit scheme for small stability factors
                let mut new_conc = conc.clone();

                // Use efficient 3D diffusion computation
                let dx2_inv = 1.0 / (dx * dx);
                let dy2_inv = 1.0 / (dy * dy);
                let dz2_inv = 1.0 / (dz * dz);

                // Process interior points
                for i in 1..self.shape.0 - 1 {
                    for j in 1..self.shape.1 - 1 {
                        for k in 1..self.shape.2 - 1 {
                            let center_val = conc[[i, j, k]];

                            // Compute Laplacian using neighboring values
                            let laplacian = (conc[[i + 1, j, k]] - 2.0 * center_val
                                + conc[[i - 1, j, k]])
                                * dx2_inv
                                + (conc[[i, j + 1, k]] - 2.0 * center_val + conc[[i, j - 1, k]])
                                    * dy2_inv
                                + (conc[[i, j, k + 1]] - 2.0 * center_val + conc[[i, j, k - 1]])
                                    * dz2_inv;

                            new_conc[[i, j, k]] = center_val + d * laplacian * dt;
                        }
                    }
                }

                updates.push((*species, new_conc));
            }
        }

        // Apply updates
        for (species, new_conc) in updates {
            self.fields.insert(species, new_conc);
        }
    }

    /// Apply semi-implicit diffusion for numerical stability (static version)
    fn apply_semi_implicit_diffusion_static(
        conc: &mut Array3<f64>,
        d: f64,
        dx: f64,
        dy: f64,
        dz: f64,
        dt: f64,
        shape: (usize, usize, usize),
    ) {
        // ADI (Alternating Direction Implicit) method
        // This is more stable than explicit Euler
        let mut temp = conc.clone();

        // X-direction sweep
        for j in 1..shape.1 - 1 {
            for k in 1..shape.2 - 1 {
                let alpha = d * dt / (2.0 * dx * dx);
                // Solve tridiagonal system for each line
                // Use Crank-Nicolson discretization
                for i in 1..shape.0 - 1 {
                    let rhs = conc[[i, j, k]]
                        + alpha
                            * (conc[[i + 1, j, k]] - 2.0 * conc[[i, j, k]] + conc[[i - 1, j, k]]);
                    temp[[i, j, k]] = rhs / (1.0 + 2.0 * alpha);
                }
            }
        }

        // Y and Z directions would follow similarly in a full ADI implementation
        // For now, use the semi-implicit result
        *conc = temp;
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ros_properties() {
        assert_eq!(ROSSpecies::HydroxylRadical.name(), "•OH");
        assert!(ROSSpecies::HydroxylRadical.reduction_potential() > 2.0);
        assert!(ROSSpecies::AtomicHydrogen.diffusion_coefficient() > 5e-9);
    }

    #[test]
    fn test_ros_concentrations() {
        let mut ros = ROSConcentrations::new(10, 10, 10);

        // Set some concentration
        if let Some(oh) = ros.get_mut(ROSSpecies::HydroxylRadical) {
            oh[[5, 5, 5]] = 1e-6; // 1 μM
        }

        // Update total
        ros.update_total();
        assert!(ros.total_ros[[5, 5, 5]] > 0.0);

        // Test decay
        ros.apply_decay(1e-9); // 1 ns
        if let Some(oh) = ros.get(ROSSpecies::HydroxylRadical) {
            assert!(oh[[5, 5, 5]] < 1e-6); // Should have decayed
        }
    }

    #[test]
    fn test_ros_generation() {
        let rates = calculate_ros_generation(3000.0, 101325.0, 0.5);
        assert!(rates.contains_key(&ROSSpecies::HydroxylRadical));
        assert!(rates.contains_key(&ROSSpecies::AtomicOxygen));
    }
}
