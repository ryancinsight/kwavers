use super::gas_dynamics::GasSpecies;
use super::parameters::BubbleParameters;
use crate::core::constants::fundamental::{AVOGADRO, GAS_CONSTANT};
use std::f64::consts::PI;

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
        // p_internal - p_external - 2σ/R = 0
        // Therefore: p_internal = p_external + 2σ/R = p0 + 2σ/r0

        // Calculate the equilibrium internal pressure using force balance
        let p_internal_theoretical = params.p0 + 2.0 * params.sigma / params.r0;

        // Set up molecule counts based on this theoretical pressure
        // Split into gas and vapor components
        let p_gas_pure_eq = p_internal_theoretical - params.pv;

        // Calculate molecule counts using ideal gas law at equilibrium conditions
        let n_gas = estimate_molecule_count(p_gas_pure_eq, params.r0, 293.15);
        let n_vapor = estimate_molecule_count(params.pv, params.r0, 293.15);

        // Create initial state with theoretical equilibrium pressure
        // Note: The actual pressure during solving will be recalculated based on
        // molecule counts and thermal effects, which may differ slightly

        Self {
            radius: params.r0,
            wall_velocity: 0.0,
            wall_acceleration: 0.0,
            temperature: 293.15,
            pressure_internal: p_internal_theoretical, // Store theoretical value
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
        let molecular_weight = self.gas_species.molecular_weight();
        let water_molecular_weight = 0.018; // kg/mol for water vapor

        // Mass of gas + mass of vapor
        (self.n_gas * molecular_weight + self.n_vapor * water_molecular_weight) / AVOGADRO
    }
}

/// Estimate number of molecules from ideal gas law
fn estimate_molecule_count(pressure: f64, radius: f64, temperature: f64) -> f64 {
    let volume = 4.0 / 3.0 * PI * radius.powi(3);
    let moles = pressure * volume / (GAS_CONSTANT * temperature);
    moles * AVOGADRO
}
