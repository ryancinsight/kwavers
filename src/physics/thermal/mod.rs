//! Thermal module for bioheat transfer
//!
//! Implements the Pennes bioheat equation for modeling temperature rise
//! in tissue during ultrasound exposure.
//!
//! References:
//! - Pennes (1948) "Analysis of tissue and arterial blood temperatures"
//! - Nyborg (1988) "Solutions of the bio-heat transfer equation"
//! - ter Haar & Coussios (2007) "High intensity focused ultrasound"

pub mod pennes;
pub mod perfusion;
pub mod properties;
pub mod thermal_dose;

pub use pennes::PennesSolver;
pub use thermal_dose::ThermalDose;

/// Thermal properties of tissue
#[derive(Debug, Clone)]
pub struct ThermalProperties {
    /// Thermal conductivity (W/m/K)
    pub k: f64,
    /// Specific heat capacity (J/kg/K)
    pub c: f64,
    /// Density (kg/m³)
    pub rho: f64,
    /// Blood perfusion rate (kg/m³/s)
    pub w_b: f64,
    /// Blood specific heat (J/kg/K)
    pub c_b: f64,
    /// Arterial blood temperature (°C)
    pub T_a: f64,
    /// Metabolic heat generation (W/m³)
    pub Q_m: f64,
}

impl Default for ThermalProperties {
    fn default() -> Self {
        // Default values for soft tissue
        Self {
            k: 0.5,      // W/m/K
            c: 3600.0,   // J/kg/K
            rho: 1050.0, // kg/m³
            w_b: 0.5,    // kg/m³/s (moderate perfusion)
            c_b: 3800.0, // J/kg/K (blood)
            T_a: 37.0,   // °C (body temperature)
            Q_m: 400.0,  // W/m³ (basal metabolism)
        }
    }
}

/// Common tissue types with thermal properties
pub mod tissues {
    use super::ThermalProperties;

    /// Liver tissue properties
    pub fn liver() -> ThermalProperties {
        ThermalProperties {
            k: 0.52,
            c: 3540.0,
            rho: 1060.0,
            w_b: 16.7, // High perfusion
            c_b: 3800.0,
            T_a: 37.0,
            Q_m: 33800.0,
        }
    }

    /// Muscle tissue properties
    pub fn muscle() -> ThermalProperties {
        ThermalProperties {
            k: 0.49,
            c: 3421.0,
            rho: 1090.0,
            w_b: 0.54,
            c_b: 3800.0,
            T_a: 37.0,
            Q_m: 684.0,
        }
    }

    /// Fat tissue properties
    pub fn fat() -> ThermalProperties {
        ThermalProperties {
            k: 0.21,
            c: 2348.0,
            rho: 911.0,
            w_b: 0.3, // Low perfusion
            c_b: 3800.0,
            T_a: 37.0,
            Q_m: 400.0,
        }
    }

    /// Tumor tissue properties (hypoxic)
    pub fn tumor() -> ThermalProperties {
        ThermalProperties {
            k: 0.55,
            c: 3600.0,
            rho: 1050.0,
            w_b: 0.2, // Poor perfusion
            c_b: 3800.0,
            T_a: 37.0,
            Q_m: 5000.0, // Higher metabolism
        }
    }
}
