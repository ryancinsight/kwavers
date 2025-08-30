// thermal/properties.rs - Thermal material properties

use crate::grid::Grid;

/// Thermal properties trait
pub trait ThermalProperties {
    fn thermal_conductivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn specific_heat(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn specific_heat_capacity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        self.specific_heat(x, y, z, grid)
    }
    fn thermal_diffusivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
}

/// Tissue-specific thermal properties
#[derive(Debug))]
pub struct TissueProperties {
    pub conductivity: f64,
    pub specific_heat: f64,
    pub density: f64,
    pub perfusion_rate: f64,
}

impl TissueProperties {
    /// Properties for muscle tissue
    pub fn muscle() -> Self {
        Self {
            conductivity: 0.5,     // W/m/K
            specific_heat: 3421.0, // J/kg/K
            density: 1090.0,       // kg/m³
            perfusion_rate: 0.5,   // kg/m³/s
        }
    }

    /// Properties for fat tissue
    pub fn fat() -> Self {
        Self {
            conductivity: 0.21,    // W/m/K
            specific_heat: 2348.0, // J/kg/K
            density: 911.0,        // kg/m³
            perfusion_rate: 0.2,   // kg/m³/s
        }
    }

    /// Properties for liver tissue
    pub fn liver() -> Self {
        Self {
            conductivity: 0.52,    // W/m/K
            specific_heat: 3540.0, // J/kg/K
            density: 1079.0,       // kg/m³
            perfusion_rate: 0.8,   // kg/m³/s
        }
    }
}
