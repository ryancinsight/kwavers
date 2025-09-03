//! Bubble state and parameters
//!
//! Core data structures for bubble dynamics

use std::collections::HashMap;
use std::f64::consts::PI;

/// Gas type enumeration for composition specification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GasType {
    N2,  // Nitrogen
    O2,  // Oxygen
    Ar,  // Argon
    He,  // Helium
    Xe,  // Xenon
    CO2, // Carbon dioxide
    H2O, // Water vapor (handled separately in most cases)
}

impl GasType {
    /// Get Van der Waals constant a [bar·L²/mol²]
    #[must_use]
    pub fn vdw_a(&self) -> f64 {
        match self {
            Self::N2 => 1.370, // From literature
            Self::O2 => 1.382,
            Self::Ar => 1.355,
            Self::He => 0.0346,
            Self::Xe => 4.250,
            Self::CO2 => 3.658,
            Self::H2O => 5.537,
        }
    }

    /// Get Van der Waals constant b [L/mol]
    #[must_use]
    pub fn vdw_b(&self) -> f64 {
        match self {
            Self::N2 => 0.0387, // From literature
            Self::O2 => 0.0319,
            Self::Ar => 0.0320,
            Self::He => 0.0238,
            Self::Xe => 0.0510,
            Self::CO2 => 0.0427,
            Self::H2O => 0.0305,
        }
    }

    /// Get molecular weight [kg/mol]
    #[must_use]
    pub fn molecular_weight(&self) -> f64 {
        match self {
            Self::N2 => 0.028014,
            Self::O2 => 0.031998,
            Self::Ar => 0.039948,
            Self::He => 0.004003,
            Self::Xe => 0.131293,
            Self::CO2 => 0.044009,
            Self::H2O => 0.018015,
        }
    }

    /// Get heat capacity ratio (gamma)
    #[must_use]
    pub fn gamma(&self) -> f64 {
        match self {
            Self::N2 => 1.4,       // Diatomic
            Self::O2 => 1.4,       // Diatomic
            Self::Ar => 5.0 / 3.0, // Monatomic
            Self::He => 5.0 / 3.0, // Monatomic
            Self::Xe => 5.0 / 3.0, // Monatomic
            Self::CO2 => 1.289,    // Triatomic
            Self::H2O => 1.33,     // Triatomic
        }
    }
}

/// Complete state of a single bubble
#[derive(Debug, Clone)]
pub struct BubbleState {
    // Geometric properties
    pub radius: f64,            // Current radius [m]
    pub wall_velocity: f64,     // dR/dt [m/s]
    pub wall_acceleration: f64, // d²R/dt² [m/s²]

    // Thermodynamic properties
    pub temperature: f64,       // Internal temperature [K]
    pub pressure_internal: f64, // Internal pressure [Pa]
    pub pressure_liquid: f64,   // Liquid pressure at wall [Pa]

    // Gas content
    pub n_gas: f64,              // Number of gas molecules
    pub n_vapor: f64,            // Number of vapor molecules
    pub gas_species: GasSpecies, // Type of gas

    // Dynamic indicators
    pub is_collapsing: bool,    // True during collapse phase
    pub mach_number: f64,       // Wall Mach number
    pub compression_ratio: f64, // R₀/R

    // History tracking
    pub max_temperature: f64, // Maximum T reached
    pub max_compression: f64, // Maximum compression
    pub collapse_count: u32,  // Number of collapses
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
    #[must_use]
    pub fn gamma(&self) -> f64 {
        match self {
            Self::Air => 1.4,
            Self::Argon => 5.0 / 3.0,
            Self::Xenon => 5.0 / 3.0,
            Self::Nitrogen => 1.4,
            Self::Oxygen => 1.4,
            Self::Custom { gamma, .. } => *gamma,
        }
    }

    /// Get molecular weight [kg/mol]
    #[must_use]
    pub fn molecular_weight(&self) -> f64 {
        match self {
            Self::Air => 0.029,
            Self::Argon => 0.040,
            Self::Xenon => 0.131,
            Self::Nitrogen => 0.028,
            Self::Oxygen => 0.032,
            Self::Custom {
                molecular_weight, ..
            } => *molecular_weight,
        }
    }

    /// Get molar heat capacity at constant volume [J/(mol·K)]
    ///
    /// These are fundamental thermodynamic properties of the gas species,
    /// not derived from gamma to maintain consistency with real gas models.
    #[must_use]
    pub fn molar_heat_capacity_cv(&self) -> f64 {
        use crate::constants::thermodynamics::R_GAS;

        match self {
            // For diatomic gases (Air, N2, O2): Cv = (5/2)R
            Self::Air => 2.5 * R_GAS,      // 20.8 J/(mol·K)
            Self::Nitrogen => 2.5 * R_GAS, // 20.8 J/(mol·K)
            Self::Oxygen => 2.5 * R_GAS,   // 20.8 J/(mol·K)

            // For monatomic gases (Ar, Xe): Cv = (3/2)R
            Self::Argon => 1.5 * R_GAS, // 12.5 J/(mol·K)
            Self::Xenon => 1.5 * R_GAS, // 12.5 J/(mol·K)

            // For custom gas, derive from gamma (with documented assumption)
            Self::Custom { gamma, .. } => R_GAS / (gamma - 1.0),
        }
    }
}

/// Physical parameters for bubble dynamics
#[derive(Debug, Clone)]
pub struct BubbleParameters {
    // Equilibrium properties
    pub r0: f64, // Equilibrium radius [m]
    pub p0: f64, // Ambient pressure [Pa]

    // Liquid properties
    pub rho_liquid: f64, // Liquid density [kg/m³]
    pub c_liquid: f64,   // Sound speed in liquid [m/s]
    pub mu_liquid: f64,  // Dynamic viscosity [Pa·s]
    pub sigma: f64,      // Surface tension [N/m]
    pub pv: f64,         // Vapor pressure [Pa]

    // Thermal properties
    pub thermal_conductivity: f64, // k [W/(m·K)]
    pub specific_heat_liquid: f64, // cp [J/(kg·K)]
    pub accommodation_coeff: f64,  // Thermal accommodation

    // Gas properties
    pub gas_species: GasSpecies,
    pub initial_gas_pressure: f64, // Initial gas pressure [Pa]
    /// Gas composition: maps gas type to mole fraction
    /// Default is air (79% N2, 21% O2)
    pub gas_composition: HashMap<GasType, f64>,

    // Acoustic forcing parameters
    pub driving_frequency: f64, // Driving frequency [Hz]
    pub driving_amplitude: f64, // Pressure amplitude [Pa]

    // Numerical parameters
    pub use_compressibility: bool, // Use Keller-Miksis
    pub use_thermal_effects: bool, // Include heat transfer
    pub use_mass_transfer: bool,   // Include evaporation/condensation
}

impl Default for BubbleParameters {
    fn default() -> Self {
        // Default air composition
        let mut gas_composition = HashMap::new();
        gas_composition.insert(GasType::N2, 0.79);
        gas_composition.insert(GasType::O2, 0.21);

        Self {
            // Water at 20°C with 5 μm air bubble
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
            gas_species: GasSpecies::Air,
            initial_gas_pressure: 101325.0,
            gas_composition,
            driving_frequency: 26.5e3, // 26.5 kHz (typical medical ultrasound)
            driving_amplitude: 1e5,    // 100 kPa acoustic pressure
            use_compressibility: true,
            use_thermal_effects: true,
            use_mass_transfer: true,
        }
    }
}

impl BubbleParameters {
    /// Create parameters for pure gas bubble
    #[must_use]
    pub fn with_pure_gas(mut self, gas_type: GasType) -> Self {
        self.gas_composition.clear();
        self.gas_composition.insert(gas_type, 1.0);
        self
    }

    /// Calculate effective Van der Waals constants for gas mixture
    #[must_use]
    pub fn effective_vdw_constants(&self) -> (f64, f64) {
        let mut a_mix = 0.0;
        let mut b_mix = 0.0;

        // Use mixing rules for Van der Waals constants
        // a_mix = (Σ x_i * sqrt(a_i))^2 (geometric mean for a)
        // b_mix = Σ x_i * b_i (arithmetic mean for b)
        for (gas, &fraction) in &self.gas_composition {
            a_mix += fraction * gas.vdw_a().sqrt();
            b_mix += fraction * gas.vdw_b();
        }
        a_mix = a_mix.powi(2);

        (a_mix, b_mix)
    }
}

impl BubbleState {
    /// Create new bubble state at equilibrium
    #[must_use]
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

    /// Create bubble state at exact mechanical equilibrium
    /// This ensures zero acceleration for validation tests
    #[must_use]
    pub fn at_equilibrium(params: &BubbleParameters) -> Self {
        // At equilibrium for Rayleigh-Plesset equation:
        // p_gas - p_liquid - 2σ/R = 0
        // Therefore: p_gas = p_liquid + 2σ/R = p0 + 2σ/r0

        // The gas pressure at equilibrium (including vapor)
        let p_gas_total = params.p0 + 2.0 * params.sigma / params.r0;

        // Pure gas pressure (excluding vapor)
        let p_gas_pure = p_gas_total - params.pv;

        // For polytropic gas: p_gas = p_gas0 * (r0/r)^(3γ)
        // At equilibrium r = r0, so we need p_gas0 = p_gas_pure
        // This is stored implicitly through molecule count

        let n_gas = estimate_molecule_count(p_gas_pure, params.r0, 293.15);
        let n_vapor = estimate_molecule_count(params.pv, params.r0, 293.15);

        Self {
            radius: params.r0,
            wall_velocity: 0.0,
            wall_acceleration: 0.0,
            temperature: 293.15,
            pressure_internal: p_gas_total,
            pressure_liquid: params.p0,
            n_gas,
            n_vapor,
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
    #[must_use]
    pub fn volume(&self) -> f64 {
        4.0 / 3.0 * PI * self.radius.powi(3)
    }

    /// Calculate bubble surface area
    #[must_use]
    pub fn surface_area(&self) -> f64 {
        4.0 * PI * self.radius.powi(2)
    }

    /// Check if bubble is in violent collapse
    #[must_use]
    pub fn is_violent_collapse(&self) -> bool {
        self.is_collapsing && (self.mach_number > 0.3 || self.compression_ratio > 5.0)
    }

    /// Get total molecule count
    #[must_use]
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

    /// Calculate total mass of gas and vapor in the bubble
    #[must_use]
    pub fn mass(&self) -> f64 {
        const AVOGADRO: f64 = 6.022e23;
        let molecular_weight = self.gas_species.molecular_weight();
        let water_molecular_weight = 0.018; // kg/mol for water vapor

        // Mass of gas + mass of vapor
        (self.n_gas * molecular_weight + self.n_vapor * water_molecular_weight) / AVOGADRO
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
        assert_eq!(state.gas_species, GasSpecies::Air);
    }

    #[test]
    fn test_gas_properties() {
        assert_eq!(GasSpecies::Argon.gamma(), 5.0 / 3.0);
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
