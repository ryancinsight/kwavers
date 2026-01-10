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
    /// Heterogeneous medium file (optional)
    #[serde(default)]
    pub tissue_file: Option<String>,
    /// Property maps for heterogeneous media (property name -> file path)
    #[serde(default)]
    pub property_maps: HashMap<String, String>,
    /// Layer configuration for layered media
    #[serde(default)]
    pub layers: Vec<LayerParameters>,
    /// Anisotropic tensor file
    #[serde(default)]
    pub tensor_file: Option<String>,
    /// Principal directions for anisotropic media (optional)
    #[serde(default)]
    pub principal_directions: Option<[f64; 3]>,
    /// Custom properties
    #[serde(default)]
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
    /// Heterogeneous medium (file- or map-driven)
    Heterogeneous,
    /// Anisotropic medium (tensor-driven)
    Anisotropic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerParameters {
    pub thickness: f64,
    pub density: f64,
    pub sound_speed: f64,
    pub absorption: f64,
    pub interface_type: InterfaceTypeParameters,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterfaceTypeParameters {
    Sharp,
    Smooth(f64),
    Gradient(f64),
}

impl MediumParameters {
    /// Validate medium parameters
    pub fn validate(&self) -> crate::domain::core::error::KwaversResult<()> {
        if self.density <= 0.0 {
            return Err(crate::domain::core::error::ConfigError::InvalidValue {
                parameter: "density".to_string(),
                value: self.density.to_string(),
                constraint: "Must be positive".to_string(),
            }
            .into());
        }

        if let Some(c) = self.sound_speed {
            if c <= 0.0 {
                return Err(crate::domain::core::error::ConfigError::InvalidValue {
                    parameter: "sound_speed".to_string(),
                    value: c.to_string(),
                    constraint: "Must be positive".to_string(),
                }
                .into());
            }
        }

        if let (Some(min), Some(max)) = (self.sound_speed_min, self.sound_speed_max) {
            if !(min.is_finite() && max.is_finite()) {
                return Err(crate::domain::core::error::ConfigError::InvalidValue {
                    parameter: "sound_speed_range".to_string(),
                    value: format!("{min}..{max}"),
                    constraint: "Must be finite".to_string(),
                }
                .into());
            }
            if min <= 0.0 || max <= 0.0 || min > max {
                return Err(crate::domain::core::error::ConfigError::InvalidValue {
                    parameter: "sound_speed_range".to_string(),
                    value: format!("{min}..{max}"),
                    constraint: "Must be positive and min <= max".to_string(),
                }
                .into());
            }
        }

        if self.absorption < 0.0 {
            return Err(crate::domain::core::error::ConfigError::InvalidValue {
                parameter: "absorption".to_string(),
                value: self.absorption.to_string(),
                constraint: "Must be non-negative".to_string(),
            }
            .into());
        }

        if !self.absorption_power.is_finite() || self.absorption_power < 0.0 {
            return Err(crate::domain::core::error::ConfigError::InvalidValue {
                parameter: "absorption_power".to_string(),
                value: self.absorption_power.to_string(),
                constraint: "Must be finite and non-negative".to_string(),
            }
            .into());
        }

        if !self.nonlinearity.is_finite() || self.nonlinearity < 0.0 {
            return Err(crate::domain::core::error::ConfigError::InvalidValue {
                parameter: "nonlinearity".to_string(),
                value: self.nonlinearity.to_string(),
                constraint: "Must be finite and non-negative".to_string(),
            }
            .into());
        }

        if let Some(mu_a) = self.properties.get("mu_a").copied() {
            if !mu_a.is_finite() || mu_a < 0.0 {
                return Err(crate::domain::core::error::ConfigError::InvalidValue {
                    parameter: "mu_a".to_string(),
                    value: mu_a.to_string(),
                    constraint: "Must be finite and non-negative".to_string(),
                }
                .into());
            }
        }

        if let Some(mu_s_prime) = self.properties.get("mu_s_prime").copied() {
            if !mu_s_prime.is_finite() || mu_s_prime < 0.0 {
                return Err(crate::domain::core::error::ConfigError::InvalidValue {
                    parameter: "mu_s_prime".to_string(),
                    value: mu_s_prime.to_string(),
                    constraint: "Must be finite and non-negative".to_string(),
                }
                .into());
            }
        }

        if matches!(self.medium_type, MediumType::Layered) && self.layers.is_empty() {
            return Err(crate::domain::core::error::ConfigError::InvalidValue {
                parameter: "layers".to_string(),
                value: "empty".to_string(),
                constraint: "At least one layer is required".to_string(),
            }
            .into());
        }

        for (idx, layer) in self.layers.iter().enumerate() {
            if !(layer.thickness.is_finite() && layer.thickness > 0.0) {
                return Err(crate::domain::core::error::ConfigError::InvalidValue {
                    parameter: format!("layers[{idx}].thickness"),
                    value: layer.thickness.to_string(),
                    constraint: "Must be finite and positive".to_string(),
                }
                .into());
            }
            if !(layer.density.is_finite() && layer.density > 0.0) {
                return Err(crate::domain::core::error::ConfigError::InvalidValue {
                    parameter: format!("layers[{idx}].density"),
                    value: layer.density.to_string(),
                    constraint: "Must be finite and positive".to_string(),
                }
                .into());
            }
            if !(layer.sound_speed.is_finite() && layer.sound_speed > 0.0) {
                return Err(crate::domain::core::error::ConfigError::InvalidValue {
                    parameter: format!("layers[{idx}].sound_speed"),
                    value: layer.sound_speed.to_string(),
                    constraint: "Must be finite and positive".to_string(),
                }
                .into());
            }
            if !(layer.absorption.is_finite() && layer.absorption >= 0.0) {
                return Err(crate::domain::core::error::ConfigError::InvalidValue {
                    parameter: format!("layers[{idx}].absorption"),
                    value: layer.absorption.to_string(),
                    constraint: "Must be finite and non-negative".to_string(),
                }
                .into());
            }
            match layer.interface_type {
                InterfaceTypeParameters::Sharp => {}
                InterfaceTypeParameters::Smooth(width)
                | InterfaceTypeParameters::Gradient(width) => {
                    if !(width.is_finite() && width > 0.0) {
                        return Err(crate::domain::core::error::ConfigError::InvalidValue {
                            parameter: format!("layers[{idx}].interface_width"),
                            value: width.to_string(),
                            constraint: "Must be finite and positive".to_string(),
                        }
                        .into());
                    }
                }
            }
        }

        if matches!(self.medium_type, MediumType::Anisotropic) {
            let Some(tensor_file) = self.tensor_file.as_deref() else {
                return Err(crate::domain::core::error::ConfigError::MissingParameter {
                    parameter: "tensor_file".to_string(),
                    section: "medium".to_string(),
                }
                .into());
            };
            if tensor_file.is_empty() {
                return Err(crate::domain::core::error::ConfigError::InvalidValue {
                    parameter: "tensor_file".to_string(),
                    value: "empty".to_string(),
                    constraint: "Must not be empty".to_string(),
                }
                .into());
            }
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
            tissue_file: None,
            property_maps: HashMap::new(),
            layers: Vec::new(),
            tensor_file: None,
            principal_directions: None,
            properties: HashMap::new(),
        }
    }
}
