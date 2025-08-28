// physics/thermal/mod.rs - Unified thermal physics module

pub mod bioheat;
pub mod calculator;
pub mod dose;
pub mod properties;
pub mod source;

use crate::error::KwaversResult;
use crate::grid::Grid;
use ndarray::Array3;
use serde::{Deserialize, Serialize};

// Re-export main types - single implementation
pub use bioheat::{BioheatSolver, PennesEquation};
pub use calculator::ThermalCalculator;
pub use dose::{ThermalDose, ThermalDoseCalculator};
pub use properties::{ThermalProperties, TissueProperties};
pub use source::{HeatSource, ThermalSource};

/// Thermal configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalConfig {
    /// Enable Pennes bioheat equation
    pub use_bioheat: bool,
    /// Blood temperature (K)
    pub blood_temperature: f64,
    /// Blood perfusion rate (kg/m³/s)
    pub blood_perfusion: f64,
    /// Blood specific heat (J/kg/K)
    pub blood_specific_heat: f64,
    /// Thermal diffusivity (m²/s)
    pub thermal_diffusivity: f64,
}

impl Default for ThermalConfig {
    fn default() -> Self {
        Self {
            use_bioheat: true,
            blood_temperature: 310.0, // 37°C
            blood_perfusion: 0.5,
            blood_specific_heat: 3617.0,
            thermal_diffusivity: 1.4e-7,
        }
    }
}

/// Thermal field state
#[derive(Debug, Clone)]
pub struct ThermalState {
    pub temperature: Array3<f64>,
    pub heat_flux: (Array3<f64>, Array3<f64>, Array3<f64>),
    pub thermal_dose: Array3<f64>,
}

impl ThermalState {
    /// Create new thermal state
    pub fn new(grid: &Grid, initial_temperature: f64) -> Self {
        let shape = (grid.nx, grid.ny, grid.nz);

        Self {
            temperature: Array3::from_elem(shape, initial_temperature),
            heat_flux: (
                Array3::zeros(shape),
                Array3::zeros(shape),
                Array3::zeros(shape),
            ),
            thermal_dose: Array3::zeros(shape),
        }
    }
}

/// Named constants for thermal physics
pub mod constants {
    /// Celsius to Kelvin conversion
    pub const CELSIUS_TO_KELVIN: f64 = 273.15;

    /// Reference temperature for CEM43 calculation (°C)
    pub const CEM43_REFERENCE_TEMP: f64 = 43.0;

    /// Activation energy for protein denaturation (J/mol)
    pub const ACTIVATION_ENERGY: f64 = 630e3;

    /// Universal gas constant (J/mol/K)
    pub const GAS_CONSTANT: f64 = 8.314;

    /// Threshold for thermal damage (equivalent minutes at 43°C)
    pub const THERMAL_DAMAGE_THRESHOLD: f64 = 240.0;
}
