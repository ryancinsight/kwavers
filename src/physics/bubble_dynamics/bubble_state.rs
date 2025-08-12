//! Bubble state and parameters
//!
//! Core data structures for bubble dynamics

use std::f64::consts::PI;
use crate::constants::thermodynamics::{R_GAS, AVOGADRO, T_AMBIENT};
use crate::constants::bubble_dynamics::{
    MIN_RADIUS, MAX_RADIUS, N2_FRACTION, O2_FRACTION,
    VDW_A_N2, VDW_B_N2, VDW_A_O2, VDW_B_O2,
    BAR_L2_TO_PA_M6, L_TO_M3
};

/// Complete state of a single bubble
#[derive(Debug, Clone)]
pub struct BubbleState {
    // Geometric properties
    pub radius: f64,              // Current radius [m]
    pub wall_velocity: f64,       // dR/dt [m/s]
    pub wall_acceleration: f64,   // d²R/dt² [m/s²]
    
    // Thermodynamic properties
    pub temperature: f64,         // Internal temperature [K]
    pub pressure_internal: f64,   // Internal pressure [Pa]
    pub pressure_liquid: f64,     // Liquid pressure at wall [Pa]
    
    // Gas content
    pub n_gas: f64,              // Number of gas molecules
    pub n_vapor: f64,            // Number of vapor molecules
    pub gas_species: GasSpecies, // Type of gas
    
    // Dynamic indicators
    pub is_collapsing: bool,     // True during collapse phase
    pub mach_number: f64,        // Wall Mach number
    pub compression_ratio: f64,   // R₀/R
    
    // History tracking
    pub max_temperature: f64,     // Maximum T reached
    pub max_compression: f64,     // Maximum compression
    pub collapse_count: u32,      // Number of collapses
}

/// Gas species in bubble
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GasSpecies {
    Air,
    Argon,
    Xenon,
    Nitrogen,
    Oxygen,
    Custom { gamma: f64, molecular_weight: f64 },
}

impl GasSpecies {
    /// Get polytropic index
    pub fn gamma(&self) -> f64 {
        match self {
            Self::Air => 1.4,
            Self::Argon => 5.0/3.0,
            Self::Xenon => 5.0/3.0,
            Self::Nitrogen => 1.4,
            Self::Oxygen => 1.4,
            Self::Custom { gamma, .. } => *gamma,
        }
    }
    
    /// Get molecular weight [kg/mol]
    pub fn molecular_weight(&self) -> f64 {
        match self {
            Self::Air => 0.029,
            Self::Argon => 0.040,
            Self::Xenon => 0.131,
            Self::Nitrogen => 0.028,
            Self::Oxygen => 0.032,
            Self::Custom { molecular_weight, .. } => *molecular_weight,
        }
    }
}

/// Physical parameters for bubble dynamics
#[derive(Debug, Clone)]
pub struct BubbleParameters {
    // Equilibrium properties
    pub r0: f64,                 // Equilibrium radius [m]
    pub p0: f64,                 // Ambient pressure [Pa]
    
    // Liquid properties
    pub rho_liquid: f64,         // Liquid density [kg/m³]
    pub c_liquid: f64,           // Sound speed in liquid [m/s]
    pub mu_liquid: f64,          // Dynamic viscosity [Pa·s]
    pub sigma: f64,              // Surface tension [N/m]
    pub pv: f64,                 // Vapor pressure [Pa]
    
    // Thermal properties
    pub thermal_conductivity: f64,     // k [W/(m·K)]
    pub specific_heat_liquid: f64,     // cp [J/(kg·K)]
    pub accommodation_coeff: f64,      // Thermal accommodation
    
    // Gas properties
    pub gas_species: GasSpecies,
    pub initial_gas_pressure: f64,     // Initial gas pressure [Pa]
    
    // Numerical parameters
    pub use_compressibility: bool,     // Use Keller-Miksis
    pub use_thermal_effects: bool,     // Include heat transfer
    pub use_mass_transfer: bool,       // Include evaporation/condensation
}

impl Default for BubbleParameters {
    fn default() -> Self {
        Self {
            // Water at 20°C with 5 μm argon bubble
            r0: 5e-6,
            p0: 101325.0,
            rho_liquid: 998.0,
            c_liquid: 1482.0,
            mu_liquid: 1.002e-3,
            sigma: 0.0728,
            pv: 2.33e3,
            thermal_conductivity: 0.6,
            specific_heat_liquid: 4182.0,
            accommodation_coeff: 0.04,
            gas_species: GasSpecies::Argon,
            initial_gas_pressure: 101325.0,
            use_compressibility: true,
            use_thermal_effects: true,
            use_mass_transfer: true,
        }
    }
}

impl BubbleState {
    /// Create new bubble state at equilibrium
    pub fn new(params: &BubbleParameters) -> Self {
        let gas_pressure = params.initial_gas_pressure + 2.0 * params.sigma / params.r0;
        let n_gas = estimate_molecule_count(gas_pressure, params.r0, 293.15);
        
        Self {
            radius: params.r0,
            wall_velocity: 0.0,
            wall_acceleration: 0.0,
            temperature: 293.15,
            pressure_internal: gas_pressure,
            pressure_liquid: params.p0,
            n_gas,
            n_vapor: 0.0,
            gas_species: params.gas_species,
            is_collapsing: false,
            mach_number: 0.0,
            compression_ratio: 1.0,
            max_temperature: 293.15,
            max_compression: 1.0,
            collapse_count: 0,
        }
    }
    
    /// Calculate bubble volume
    pub fn volume(&self) -> f64 {
        4.0 / 3.0 * PI * self.radius.powi(3)
    }
    
    /// Calculate bubble surface area
    pub fn surface_area(&self) -> f64 {
        4.0 * PI * self.radius.powi(2)
    }
    
    /// Check if bubble is in violent collapse
    pub fn is_violent_collapse(&self) -> bool {
        self.is_collapsing && (self.mach_number > 0.3 || self.compression_ratio > 5.0)
    }
    
    /// Get total molecule count
    pub fn total_molecules(&self) -> f64 {
        self.n_gas + self.n_vapor
    }
    
    /// Update compression ratio
    pub fn update_compression(&mut self, r0: f64) {
        self.compression_ratio = r0 / self.radius;
        if self.compression_ratio > self.max_compression {
            self.max_compression = self.compression_ratio;
        }
    }
    
    /// Update temperature tracking
    pub fn update_max_temperature(&mut self) {
        if self.temperature > self.max_temperature {
            self.max_temperature = self.temperature;
        }
    }
    
    /// Detect collapse phase transition
    pub fn update_collapse_state(&mut self) {
        let was_collapsing = self.is_collapsing;
        self.is_collapsing = self.wall_velocity < 0.0 && self.wall_acceleration < 0.0;
        
        // Count collapse events
        if !was_collapsing && self.is_collapsing {
            self.collapse_count += 1;
        }
    }
}

/// Estimate number of molecules from ideal gas law
fn estimate_molecule_count(pressure: f64, radius: f64, temperature: f64) -> f64 {
    const R_GAS: f64 = 8.314; // J/(mol·K)
    const AVOGADRO: f64 = 6.022e23;
    
    let volume = 4.0 / 3.0 * PI * radius.powi(3);
    let moles = pressure * volume / (R_GAS * temperature);
    moles * AVOGADRO
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bubble_state_creation() {
        let params = BubbleParameters::default();
        let state = BubbleState::new(&params);
        
        assert_eq!(state.radius, params.r0);
        assert_eq!(state.wall_velocity, 0.0);
        assert!(state.n_gas > 0.0);
        assert_eq!(state.gas_species, GasSpecies::Argon);
    }
    
    #[test]
    fn test_gas_properties() {
        assert_eq!(GasSpecies::Argon.gamma(), 5.0/3.0);
        assert_eq!(GasSpecies::Air.gamma(), 1.4);
        assert!((GasSpecies::Xenon.molecular_weight() - 0.131).abs() < 1e-6);
    }
    
    #[test]
    fn test_compression_tracking() {
        let params = BubbleParameters::default();
        let mut state = BubbleState::new(&params);
        
        state.radius = params.r0 / 10.0; // Compress to 1/10
        state.update_compression(params.r0);
        
        assert_eq!(state.compression_ratio, 10.0);
        assert_eq!(state.max_compression, 10.0);
    }
}