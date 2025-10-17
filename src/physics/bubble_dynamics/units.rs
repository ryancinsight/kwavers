//! Unit-Safe Bubble Dynamics Types
//!
//! This module provides type-safe wrappers for bubble dynamics parameters
//! using the uom crate to prevent unit conversion errors at compile time.

use std::collections::HashMap;
use uom::si::dynamic_viscosity::pascal_second;
use uom::si::f64::{DynamicViscosity, Length, MassDensity, Pressure};
use uom::si::length::meter;
use uom::si::mass_density::kilogram_per_cubic_meter;
use uom::si::pressure::pascal;

// Physical constants for bubble dynamics
const WATER_DENSITY: f64 = 998.0; // kg/m³
const ATMOSPHERIC_PRESSURE: f64 = 101325.0; // Pa
const WATER_VISCOSITY: f64 = 1.002e-3; // Pa·s
const WATER_SURFACE_TENSION: f64 = 0.0728; // N/m
const WATER_VAPOR_PRESSURE: f64 = 2.33e3; // Pa at 20°C
const AIR_MOLECULAR_WEIGHT: f64 = 0.029; // kg/mol
const WATER_MOLECULAR_WEIGHT: f64 = 0.018; // kg/mol
const DEFAULT_BUBBLE_RADIUS: f64 = 5e-6; // m
const ACCOMMODATION_COEFFICIENT: f64 = 0.04;

/// Bubble parameters with SI units
#[derive(Debug, Clone)]
pub struct BubbleParameters {
    /// Initial bubble radius
    pub r0: Length,
    /// Ambient pressure
    pub p0: Pressure,
    /// Liquid density
    pub rho_liquid: MassDensity,
    /// Liquid sound speed
    pub c_liquid: f64, // m/s (no uom type for velocity yet)
    /// Liquid dynamic viscosity
    pub mu_liquid: DynamicViscosity,
    /// Surface tension
    pub sigma: f64, // N/m (no uom type for surface tension)
    /// Vapor pressure
    pub pv: Pressure,
    /// Polytropic exponent
    pub gamma: f64,
    /// Thermal conductivity
    pub k_thermal: f64, // W/(m·K)
    /// Accommodation coefficient
    pub accommodation_coeff: f64,
    /// Initial gas pressure
    pub initial_gas_pressure: Pressure,
    /// Gas composition
    pub gas_composition: GasComposition,
}

/// Gas composition for multi-species tracking
#[derive(Debug, Clone)]
pub struct GasComposition {
    /// Mole fractions of gas species
    pub mole_fractions: HashMap<String, f64>,
    /// Molecular weights [kg/mol]
    pub molecular_weights: HashMap<String, f64>,
}

impl Default for BubbleParameters {
    fn default() -> Self {
        let mut gas_composition = GasComposition {
            mole_fractions: HashMap::new(),
            molecular_weights: HashMap::new(),
        };

        // Default: Air bubble
        gas_composition
            .mole_fractions
            .insert("N2".to_string(), 0.78);
        gas_composition
            .mole_fractions
            .insert("O2".to_string(), 0.21);
        gas_composition
            .mole_fractions
            .insert("Ar".to_string(), 0.01);

        gas_composition
            .molecular_weights
            .insert("N2".to_string(), 0.028);
        gas_composition
            .molecular_weights
            .insert("O2".to_string(), 0.032);
        gas_composition
            .molecular_weights
            .insert("Ar".to_string(), 0.040);

        Self {
            r0: Length::new::<meter>(DEFAULT_BUBBLE_RADIUS),
            p0: Pressure::new::<pascal>(ATMOSPHERIC_PRESSURE),
            rho_liquid: MassDensity::new::<kilogram_per_cubic_meter>(WATER_DENSITY),
            c_liquid: 1500.0, // m/s
            mu_liquid: DynamicViscosity::new::<pascal_second>(WATER_VISCOSITY),
            sigma: WATER_SURFACE_TENSION, // N/m
            pv: Pressure::new::<pascal>(WATER_VAPOR_PRESSURE),
            gamma: 1.4,       // Diatomic gas
            k_thermal: 0.598, // W/(m·K) for water
            accommodation_coeff: ACCOMMODATION_COEFFICIENT,
            initial_gas_pressure: Pressure::new::<pascal>(ATMOSPHERIC_PRESSURE),
            gas_composition,
        }
    }
}

impl BubbleParameters {
    /// Create parameters for an air bubble in water
    #[must_use]
    pub fn air_in_water() -> Self {
        Self::default()
    }

    /// Create parameters for a vapor bubble (cavitation)
    #[must_use]
    pub fn vapor_bubble() -> Self {
        let mut params = Self::default();
        params.initial_gas_pressure = params.pv;

        // Pure water vapor
        params.gas_composition.mole_fractions.clear();
        params
            .gas_composition
            .mole_fractions
            .insert("H2O".to_string(), 1.0);
        params.gas_composition.molecular_weights.clear();
        params
            .gas_composition
            .molecular_weights
            .insert("H2O".to_string(), WATER_MOLECULAR_WEIGHT);

        params
    }

    /// Create parameters for ultrasound contrast agent
    #[must_use]
    pub fn contrast_agent(shell_elasticity: f64) -> Self {
        let mut params = Self::default();

        // Typical UCA: Perfluorocarbon gas
        params.gas_composition.mole_fractions.clear();
        params
            .gas_composition
            .mole_fractions
            .insert("C4F10".to_string(), 1.0);
        params.gas_composition.molecular_weights.clear();
        params
            .gas_composition
            .molecular_weights
            .insert("C4F10".to_string(), 0.238);

        // Smaller initial radius
        params.r0 = Length::new::<meter>(2e-6);

        // Modified surface tension due to shell
        params.sigma = shell_elasticity;

        params
    }
}

/// Convert dimensional to dimensionless parameters
#[derive(Debug)]
pub struct DimensionlessParameters {
    /// Reynolds number
    pub reynolds: f64,
    /// Weber number
    pub weber: f64,
    /// Cavitation number
    pub cavitation: f64,
    /// Peclet number
    pub peclet: f64,
}

impl DimensionlessParameters {
    #[must_use]
    pub fn from_bubble_params(params: &BubbleParameters, velocity_scale: f64) -> Self {
        let r0 = params.r0.get::<meter>();
        let rho = params.rho_liquid.get::<kilogram_per_cubic_meter>();
        let mu = params.mu_liquid.get::<pascal_second>();
        let p0 = params.p0.get::<pascal>();
        let pv = params.pv.get::<pascal>();

        Self {
            reynolds: rho * velocity_scale * r0 / mu,
            weber: rho * velocity_scale.powi(2) * r0 / params.sigma,
            cavitation: (p0 - pv) / (0.5 * rho * velocity_scale.powi(2)),
            peclet: velocity_scale * r0 / params.k_thermal,
        }
    }
}

/// Calculate effective molecular weight for gas mixture
#[must_use]
pub fn effective_molecular_weight(composition: &GasComposition) -> f64 {
    let mut m_eff = 0.0;
    for (species, mole_frac) in &composition.mole_fractions {
        if let Some(mol_weight) = composition.molecular_weights.get(species) {
            m_eff += mole_frac * mol_weight;
        }
    }

    if m_eff == 0.0 {
        // Default to air if no composition specified
        AIR_MOLECULAR_WEIGHT
    } else {
        m_eff
    }
}

/// Calculate specific heat ratio for gas mixture
/// 
/// Uses molecular weight-based heuristic to estimate γ (gamma).
/// This is a standard engineering approximation when detailed composition is unavailable.
/// 
/// - Light gases (H2, He): γ ≈ 1.66 (nearly monatomic)
/// - Diatomic gases (N2, O2, air): γ ≈ 1.4
/// - Heavy/polyatomic gases: γ ≈ 1.3
///
/// For precise calculations with known composition, use thermodynamic tables.
#[must_use]
pub fn specific_heat_ratio(composition: &GasComposition, _temperature: f64) -> f64 {
    let molecular_weight = effective_molecular_weight(composition);

    // Standard estimates based on molecular structure
    if molecular_weight < 0.010 {
        // Light gas (H2, He) - closer to monatomic
        1.66
    } else if molecular_weight < 0.040 {
        // Diatomic gas (N2, O2, Air)
        1.4
    } else {
        // Heavy/polyatomic gas
        1.3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_parameters() {
        let params = BubbleParameters::default();
        assert_eq!(params.r0.get::<meter>(), DEFAULT_BUBBLE_RADIUS);
        assert_eq!(params.p0.get::<pascal>(), ATMOSPHERIC_PRESSURE);
    }

    #[test]
    fn test_dimensionless_numbers() {
        let params = BubbleParameters::default();
        let velocity = 10.0; // m/s
        let dim = DimensionlessParameters::from_bubble_params(&params, velocity);

        // Check Reynolds number
        assert!(dim.reynolds > 0.0);
        assert!(dim.reynolds < 1000.0); // Should be in intermediate regime

        // Check Weber number
        assert!(dim.weber > 0.0);

        // Check cavitation number
        assert!(dim.cavitation > 0.0);
    }
}
