//! Reactive Oxygen Species (ROS) definitions and concentration tracking
//!
//! Key ROS generated during sonoluminescence:
//! - Hydroxyl radical (•OH)
//! - Hydrogen peroxide (H₂O₂)
//! - Superoxide (O₂•⁻)
//! - Singlet oxygen (¹O₂)
//! - Ozone (O₃)

use ndarray::{Array3};
use std::collections::HashMap;

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
}

impl ROSSpecies {
    /// Get the name of the species
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
        }
    }
    
    /// Get the diffusion coefficient in water at 25°C (m²/s)
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
        }
    }
    
    /// Get the lifetime in pure water (seconds)
    pub fn lifetime_water(&self) -> f64 {
        match self {
            Self::HydroxylRadical => 1e-9,      // 1 ns
            Self::HydrogenPeroxide => 1e3,      // Stable
            Self::Superoxide => 1e-6,           // 1 μs
            Self::SingletOxygen => 3.5e-6,      // 3.5 μs
            Self::Ozone => 1e2,                 // 100 s
            Self::HydroperoxylRadical => 1e-6,  // 1 μs
            Self::AtomicOxygen => 1e-12,        // 1 ps
            Self::AtomicHydrogen => 1e-12,      // 1 ps
        }
    }
    
    /// Get the standard reduction potential (V vs SHE)
    pub fn reduction_potential(&self) -> f64 {
        match self {
            Self::HydroxylRadical => 2.80,      // Strongest oxidant
            Self::HydrogenPeroxide => 1.78,
            Self::Superoxide => -0.33,          // Can act as reductant
            Self::SingletOxygen => 0.65,
            Self::Ozone => 2.07,
            Self::HydroperoxylRadical => 1.50,
            Self::AtomicOxygen => 2.42,
            Self::AtomicHydrogen => -2.30,      // Strong reductant
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
        for (_, conc) in &self.fields {
            self.total_ros = &self.total_ros + conc;
        }
    }
    
    /// Calculate oxidative stress index based on ROS concentrations
    pub fn oxidative_stress_index(&self) -> Array3<f64> {
        let mut stress = Array3::zeros(self.shape);
        
        // Weight each ROS by its oxidative potential
        for (species, conc) in &self.fields {
            let weight = match species {
                ROSSpecies::HydroxylRadical => 10.0,     // Most damaging
                ROSSpecies::HydrogenPeroxide => 1.0,
                ROSSpecies::Superoxide => 2.0,
                ROSSpecies::SingletOxygen => 3.0,
                ROSSpecies::Ozone => 5.0,
                ROSSpecies::HydroperoxylRadical => 4.0,
                ROSSpecies::AtomicOxygen => 8.0,
                ROSSpecies::AtomicHydrogen => 0.1,       // Reductant
            };
            stress = stress + weight * conc;
        }
        
        stress
    }
    
    /// Apply decay based on species lifetime
    pub fn apply_decay(&mut self, dt: f64) {
        for (species, conc) in self.fields.iter_mut() {
            let lifetime = species.lifetime_water();
            let decay_rate = 1.0 / lifetime;
            
            // Exponential decay: C(t+dt) = C(t) * exp(-dt/τ)
            conc.mapv_inplace(|c| c * (-dt * decay_rate).exp());
        }
    }
    
    /// Apply diffusion using simple forward Euler
    pub fn apply_diffusion(&mut self, dx: f64, dy: f64, dz: f64, dt: f64) {
        for (species, conc) in &mut self.fields {
            let d = species.diffusion_coefficient();
            let mut new_conc = conc.clone();
            
            // Simple 3D diffusion (central differences)
            for i in 1..self.shape.0-1 {
                for j in 1..self.shape.1-1 {
                    for k in 1..self.shape.2-1 {
                        let laplacian = 
                            (conc[[i+1, j, k]] - 2.0 * conc[[i, j, k]] + conc[[i-1, j, k]]) / (dx * dx) +
                            (conc[[i, j+1, k]] - 2.0 * conc[[i, j, k]] + conc[[i, j-1, k]]) / (dy * dy) +
                            (conc[[i, j, k+1]] - 2.0 * conc[[i, j, k]] + conc[[i, j, k-1]]) / (dz * dz);
                        
                        new_conc[[i, j, k]] = conc[[i, j, k]] + d * laplacian * dt;
                    }
                }
            }
            
            *conc = new_conc;
        }
    }
}

/// Calculate ROS generation rates based on bubble conditions
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
        if temperature > 3000.0 {
            let o2_fraction = 0.21 * (1.0 - water_vapor_fraction); // Air composition
            let k_o2 = 1e14 * (-6.0e4 / temperature).exp();
            let o_rate = 2.0 * k_o2 * o2_fraction * pressure / (8.314 * temperature);
            generation_rates.insert(ROSSpecies::AtomicOxygen, o_rate);
        }
    }
    
    // Secondary reactions at intermediate temperatures
    if temperature > 1000.0 && temperature < 3000.0 {
        // H₂O₂ formation: 2•OH → H₂O₂
        let oh_conc = generation_rates.get(&ROSSpecies::HydroxylRadical).unwrap_or(&0.0);
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