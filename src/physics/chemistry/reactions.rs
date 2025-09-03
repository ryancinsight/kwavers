//! Chemical reaction types and kinetics
//!
//! This module defines the core reaction types and rate calculations
//! following SOLID principles with clear separation of concerns.

/// Represents a chemical reaction with its kinetic parameters
#[derive(Debug, Clone)]
pub struct ChemicalReaction {
    pub name: String,
    pub rate_constant: f64,
}

impl ChemicalReaction {
    /// Calculate rate constant at given conditions
    #[must_use]
    pub fn rate_constant(&self, _temperature: f64, _pressure: f64) -> f64 {
        self.rate_constant
    }
}

/// Reaction rate value
#[derive(Debug, Clone)]
pub struct ReactionRate {
    pub value: f64,
}

/// Chemical species with concentration
#[derive(Debug, Clone)]
pub struct Species {
    pub name: String,
    pub concentration: f64,
}

/// Type of chemical reaction
#[derive(Debug, Clone, PartialEq)]
pub enum ReactionType {
    Dissociation,
    Recombination,
    Oxidation,
    Reduction,
    Polymerization,
}

/// Thermal dependence model for reactions
#[derive(Debug, Clone)]
pub enum ThermalDependence {
    Arrhenius {
        activation_energy: f64,
        pre_exponential: f64,
    },
    PowerLaw {
        exponent: f64,
    },
    Constant,
}

/// Pressure dependence model for reactions
#[derive(Debug, Clone)]
pub enum PressureDependence {
    Linear { coefficient: f64 },
    Logarithmic { coefficient: f64 },
    Constant,
}

/// Light dependence model for photochemical reactions
#[derive(Debug, Clone)]
pub enum LightDependence {
    Linear { quantum_yield: f64 },
    Saturable { max_rate: f64, half_saturation: f64 },
    None,
}

/// Configuration for chemical reactions
#[derive(Debug, Clone)]
pub struct ChemicalReactionConfig {
    pub reaction_type: ReactionType,
    pub thermal_dependence: ThermalDependence,
    pub pressure_dependence: PressureDependence,
    pub light_dependence: LightDependence,
    pub rate_constant: f64,
    pub activation_energy: f64,
}

impl Default for ChemicalReactionConfig {
    fn default() -> Self {
        Self {
            reaction_type: ReactionType::Dissociation,
            thermal_dependence: ThermalDependence::Constant,
            pressure_dependence: PressureDependence::Constant,
            light_dependence: LightDependence::None,
            rate_constant: 1e-3,
            activation_energy: 0.0,
        }
    }
}
