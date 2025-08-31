//! Medium properties configuration

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Medium properties parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediumParameters {
    /// Medium type
    pub medium_type: MediumType,
    /// Density in kg/m³
    pub density: f64,
    /// Sound speed in m/s (uniform)
    pub sound_speed: Option<f64>,
    /// Minimum sound speed for heterogeneous media
    pub sound_speed_min: Option<f64>,
    /// Maximum sound speed for heterogeneous media
    pub sound_speed_max: Option<f64>,
    /// Absorption coefficient in dB/cm/MHz^y
    pub absorption: f64,
    /// Absorption power law exponent
    pub absorption_power: f64,
    /// Nonlinearity parameter B/A
    pub nonlinearity: f64,
    /// Custom properties
    pub properties: HashMap<String, f64>,
}

/// Types of acoustic media
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MediumType {
    /// Homogeneous medium
    Homogeneous,
    /// Layered medium
    Layered,
    /// Random heterogeneous
    Random,
    /// Tissue phantom
    Tissue,
    /// Custom defined
    Custom,
}

impl MediumParameters {
    /// Validate medium parameters
    pub fn validate(&self) -> crate::error::KwaversResult<()> {
        if self.density <= 0.0 {
            return Err(crate::error::ConfigError::InvalidValue {
                parameter: "density".to_string(),
                value: self.density.to_string(),
                constraint: "Must be positive".to_string(),
            }
            .into());
        }

        if let Some(c) = self.sound_speed {
            if c <= 0.0 {
                return Err(crate::error::ConfigError::InvalidValue {
                    parameter: "sound_speed".to_string(),
                    value: c.to_string(),
                    constraint: "Must be positive".to_string(),
                }
                .into());
            }
        }

        if self.absorption < 0.0 {
            return Err(crate::error::ConfigError::InvalidValue {
                parameter: "absorption".to_string(),
                value: self.absorption.to_string(),
                constraint: "Must be non-negative".to_string(),
            }
            .into());
        }

        Ok(())
    }
}

impl Default for MediumParameters {
    fn default() -> Self {
        Self {
            medium_type: MediumType::Homogeneous,
            density: 1000.0, // Water
            sound_speed: Some(1500.0),
            sound_speed_min: None,
            sound_speed_max: None,
            absorption: 0.0022, // Water at 1 MHz
            absorption_power: 2.0,
            nonlinearity: 3.5, // Water
            properties: HashMap::new(),
        }
    }
}
