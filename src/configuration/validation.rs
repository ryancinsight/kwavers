//! Validation and verification configuration

use serde::{Deserialize, Serialize};

/// Validation parameters for accuracy verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationParameters {
    /// Enable energy conservation checks
    pub check_energy: bool,
    /// Energy conservation tolerance
    pub energy_tolerance: f64,
    /// Enable mass conservation checks
    pub check_mass: bool,
    /// Mass conservation tolerance
    pub mass_tolerance: f64,
    /// Check CFL condition
    pub check_cfl: bool,
    /// Check numerical dispersion
    pub check_dispersion: bool,
    /// Maximum allowed dispersion error
    pub dispersion_tolerance: f64,
    /// Validation interval (timesteps)
    pub validation_interval: usize,
}

impl ValidationParameters {
    /// Validate validation parameters (meta!)
    pub fn validate(&self) -> crate::error::KwaversResult<()> {
        if self.energy_tolerance <= 0.0 || self.energy_tolerance >= 1.0 {
            return Err(crate::error::ConfigError::InvalidValue {
                parameter: "energy_tolerance".to_string(),
                value: self.energy_tolerance.to_string(),
                constraint: "Must be in (0, 1)".to_string(),
            }
            .into());
        }

        if self.mass_tolerance <= 0.0 || self.mass_tolerance >= 1.0 {
            return Err(crate::error::ConfigError::InvalidValue {
                parameter: "mass_tolerance".to_string(),
                value: self.mass_tolerance.to_string(),
                constraint: "Must be in (0, 1)".to_string(),
            }
            .into());
        }

        if self.dispersion_tolerance <= 0.0 || self.dispersion_tolerance >= 1.0 {
            return Err(crate::error::ConfigError::InvalidValue {
                parameter: "dispersion_tolerance".to_string(),
                value: self.dispersion_tolerance.to_string(),
                constraint: "Must be in (0, 1)".to_string(),
            }
            .into());
        }

        if self.validation_interval == 0 {
            return Err(crate::error::ConfigError::InvalidValue {
                parameter: "validation_interval".to_string(),
                value: "0".to_string(),
                constraint: "Must be positive".to_string(),
            }
            .into());
        }

        Ok(())
    }
}

impl Default for ValidationParameters {
    fn default() -> Self {
        Self {
            check_energy: true,
            energy_tolerance: 1e-3,
            check_mass: true,
            mass_tolerance: 1e-6,
            check_cfl: true,
            check_dispersion: false,
            dispersion_tolerance: 1e-2,
            validation_interval: 100,
        }
    }
}
