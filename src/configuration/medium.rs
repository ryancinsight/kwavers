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

        if let (Some(min), Some(max)) = (self.sound_speed_min, self.sound_speed_max) {
            if !(min.is_finite() && max.is_finite()) {
                return Err(crate::error::ConfigError::InvalidValue {
                    parameter: "sound_speed_range".to_string(),
                    value: format!("{min}..{max}"),
                    constraint: "Must be finite".to_string(),
                }
                .into());
            }
            if min <= 0.0 || max <= 0.0 || min > max {
                return Err(crate::error::ConfigError::InvalidValue {
                    parameter: "sound_speed_range".to_string(),
                    value: format!("{min}..{max}"),
                    constraint: "Must be positive and min <= max".to_string(),
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

        if !self.absorption_power.is_finite() || self.absorption_power < 0.0 {
            return Err(crate::error::ConfigError::InvalidValue {
                parameter: "absorption_power".to_string(),
                value: self.absorption_power.to_string(),
                constraint: "Must be finite and non-negative".to_string(),
            }
            .into());
        }

        if !self.nonlinearity.is_finite() || self.nonlinearity < 0.0 {
            return Err(crate::error::ConfigError::InvalidValue {
                parameter: "nonlinearity".to_string(),
                value: self.nonlinearity.to_string(),
                constraint: "Must be finite and non-negative".to_string(),
            }
            .into());
        }

        if let Some(mu_a) = self.properties.get("mu_a").copied() {
            if !mu_a.is_finite() || mu_a < 0.0 {
                return Err(crate::error::ConfigError::InvalidValue {
                    parameter: "mu_a".to_string(),
                    value: mu_a.to_string(),
                    constraint: "Must be finite and non-negative".to_string(),
                }
                .into());
            }
        }

        if let Some(mu_s_prime) = self.properties.get("mu_s_prime").copied() {
            if !mu_s_prime.is_finite() || mu_s_prime < 0.0 {
                return Err(crate::error::ConfigError::InvalidValue {
                    parameter: "mu_s_prime".to_string(),
                    value: mu_s_prime.to_string(),
                    constraint: "Must be finite and non-negative".to_string(),
                }
                .into());
            }
        }

        if matches!(self.medium_type, MediumType::Layered) && self.layers.is_empty() {
            return Err(crate::error::ConfigError::InvalidValue {
                parameter: "layers".to_string(),
                value: "empty".to_string(),
                constraint: "At least one layer is required".to_string(),
            }
            .into());
        }

        for (idx, layer) in self.layers.iter().enumerate() {
            if !(layer.thickness.is_finite() && layer.thickness > 0.0) {
                return Err(crate::error::ConfigError::InvalidValue {
                    parameter: format!("layers[{idx}].thickness"),
                    value: layer.thickness.to_string(),
                    constraint: "Must be finite and positive".to_string(),
                }
                .into());
            }
            if !(layer.density.is_finite() && layer.density > 0.0) {
                return Err(crate::error::ConfigError::InvalidValue {
                    parameter: format!("layers[{idx}].density"),
                    value: layer.density.to_string(),
                    constraint: "Must be finite and positive".to_string(),
                }
                .into());
            }
            if !(layer.sound_speed.is_finite() && layer.sound_speed > 0.0) {
                return Err(crate::error::ConfigError::InvalidValue {
                    parameter: format!("layers[{idx}].sound_speed"),
                    value: layer.sound_speed.to_string(),
                    constraint: "Must be finite and positive".to_string(),
                }
                .into());
            }
            if !(layer.absorption.is_finite() && layer.absorption >= 0.0) {
                return Err(crate::error::ConfigError::InvalidValue {
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
                        return Err(crate::error::ConfigError::InvalidValue {
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
                return Err(crate::error::ConfigError::MissingParameter {
                    parameter: "tensor_file".to_string(),
                    section: "medium".to_string(),
                }
                .into());
            };
            if tensor_file.is_empty() {
                return Err(crate::error::ConfigError::InvalidValue {
                    parameter: "tensor_file".to_string(),
                    value: "empty".to_string(),
                    constraint: "Must not be empty".to_string(),
                }
                .into());
            }
        }

        Ok(())
    }

    pub fn to_factory_config(&self) -> crate::error::KwaversResult<crate::factory::MediumConfig> {
        self.validate()?;

        let mut properties = self.properties.clone();
        properties.insert("absorption_alpha".to_string(), self.absorption);
        properties.insert("absorption_power".to_string(), self.absorption_power);
        properties.insert("nonlinearity".to_string(), self.nonlinearity);

        let mu_a = self.properties.get("mu_a").copied().unwrap_or(0.0);
        let mu_s_prime = self.properties.get("mu_s_prime").copied().unwrap_or(0.0);

        match self.medium_type {
            MediumType::Homogeneous => {
                let sound_speed = self.sound_speed.ok_or_else(|| {
                    crate::error::ConfigError::MissingParameter {
                        parameter: "sound_speed".to_string(),
                        section: "medium".to_string(),
                    }
                })?;

                Ok(crate::factory::MediumConfig {
                    medium_type: crate::factory::MediumType::Homogeneous {
                        density: self.density,
                        sound_speed,
                        mu_a,
                        mu_s_prime,
                    },
                    properties,
                })
            }
            MediumType::Layered => {
                let layers =
                    self.layers
                        .iter()
                        .map(|layer| {
                            crate::factory::component::medium::types::LayerProperties {
                        thickness: layer.thickness,
                        density: layer.density,
                        sound_speed: layer.sound_speed,
                        absorption: layer.absorption,
                        interface_type: match layer.interface_type {
                            InterfaceTypeParameters::Sharp => {
                                crate::factory::component::medium::types::InterfaceType::Sharp
                            }
                            InterfaceTypeParameters::Smooth(width) => {
                                crate::factory::component::medium::types::InterfaceType::Smooth(
                                    width,
                                )
                            }
                            InterfaceTypeParameters::Gradient(width) => {
                                crate::factory::component::medium::types::InterfaceType::Gradient(
                                    width,
                                )
                            }
                        },
                    }
                        })
                        .collect::<Vec<_>>();

                Ok(crate::factory::MediumConfig {
                    medium_type: crate::factory::MediumType::Layered { layers },
                    properties,
                })
            }
            MediumType::Anisotropic => {
                let tensor_file = self.tensor_file.clone().ok_or_else(|| {
                    crate::error::ConfigError::MissingParameter {
                        parameter: "tensor_file".to_string(),
                        section: "medium".to_string(),
                    }
                })?;

                Ok(crate::factory::MediumConfig {
                    medium_type: crate::factory::MediumType::Anisotropic {
                        tensor_file,
                        principal_directions: self.principal_directions,
                    },
                    properties,
                })
            }
            MediumType::Heterogeneous
            | MediumType::Random
            | MediumType::Tissue
            | MediumType::Custom => Ok(crate::factory::MediumConfig {
                medium_type: crate::factory::MediumType::Heterogeneous {
                    tissue_file: self.tissue_file.clone(),
                    property_maps: self.property_maps.clone(),
                },
                properties,
            }),
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factory::MediumFactory;
    use crate::grid::Grid;

    #[test]
    fn medium_parameters_to_factory_config_homogeneous_applies_acoustic_properties() {
        let params = MediumParameters {
            medium_type: MediumType::Homogeneous,
            density: 1000.0,
            sound_speed: Some(1500.0),
            sound_speed_min: None,
            sound_speed_max: None,
            absorption: 0.5,
            absorption_power: 1.1,
            nonlinearity: 7.0,
            tissue_file: None,
            property_maps: HashMap::new(),
            layers: Vec::new(),
            tensor_file: None,
            principal_directions: None,
            properties: HashMap::new(),
        };

        let grid = Grid::new(4, 3, 2, 1e-4, 2e-4, 3e-4).unwrap();
        let factory_cfg = params.to_factory_config().unwrap();
        let medium = MediumFactory::create_medium(&factory_cfg, &grid).unwrap();

        assert_eq!(medium.density_array().dim(), (4, 3, 2));
        assert_eq!(medium.sound_speed_array().dim(), (4, 3, 2));

        assert_eq!(medium.absorption(0, 0, 0), 0.5);
        assert_eq!(medium.nonlinearity(0, 0, 0), 7.0);
    }

    #[test]
    fn medium_parameters_to_factory_config_requires_sound_speed_for_homogeneous() {
        let params = MediumParameters {
            medium_type: MediumType::Homogeneous,
            density: 1000.0,
            sound_speed: None,
            sound_speed_min: None,
            sound_speed_max: None,
            absorption: 0.5,
            absorption_power: 1.1,
            nonlinearity: 7.0,
            tissue_file: None,
            property_maps: HashMap::new(),
            layers: Vec::new(),
            tensor_file: None,
            principal_directions: None,
            properties: HashMap::new(),
        };

        assert!(params.to_factory_config().is_err());
    }

    #[test]
    fn medium_parameters_to_factory_config_layered_requires_layers() {
        let params = MediumParameters {
            medium_type: MediumType::Layered,
            density: 1000.0,
            sound_speed: Some(1500.0),
            sound_speed_min: None,
            sound_speed_max: None,
            absorption: 0.5,
            absorption_power: 1.1,
            nonlinearity: 7.0,
            tissue_file: None,
            property_maps: HashMap::new(),
            layers: Vec::new(),
            tensor_file: None,
            principal_directions: None,
            properties: HashMap::new(),
        };

        assert!(params.to_factory_config().is_err());
    }

    #[test]
    fn medium_parameters_to_factory_config_layered_builds_factory_layers() {
        let params = MediumParameters {
            medium_type: MediumType::Layered,
            density: 1000.0,
            sound_speed: Some(1500.0),
            sound_speed_min: None,
            sound_speed_max: None,
            absorption: 0.5,
            absorption_power: 1.1,
            nonlinearity: 7.0,
            tissue_file: None,
            property_maps: HashMap::new(),
            layers: vec![LayerParameters {
                thickness: 0.01,
                density: 1050.0,
                sound_speed: 1540.0,
                absorption: 0.7,
                interface_type: InterfaceTypeParameters::Sharp,
            }],
            tensor_file: None,
            principal_directions: None,
            properties: HashMap::new(),
        };

        let cfg = params.to_factory_config().unwrap();
        match cfg.medium_type {
            crate::factory::MediumType::Layered { layers } => {
                assert_eq!(layers.len(), 1);
                assert_eq!(layers[0].thickness, 0.01);
                assert_eq!(layers[0].density, 1050.0);
                assert_eq!(layers[0].sound_speed, 1540.0);
                assert_eq!(layers[0].absorption, 0.7);
            }
            _ => panic!("expected layered factory config"),
        }
    }

    #[test]
    fn medium_parameters_to_factory_config_anisotropic_requires_tensor_file() {
        let params = MediumParameters {
            medium_type: MediumType::Anisotropic,
            density: 1000.0,
            sound_speed: Some(1500.0),
            sound_speed_min: None,
            sound_speed_max: None,
            absorption: 0.5,
            absorption_power: 1.1,
            nonlinearity: 7.0,
            tissue_file: None,
            property_maps: HashMap::new(),
            layers: Vec::new(),
            tensor_file: None,
            principal_directions: None,
            properties: HashMap::new(),
        };

        assert!(params.to_factory_config().is_err());
    }

    #[test]
    fn medium_parameters_to_factory_config_heterogeneous_maps_files_and_maps() {
        let mut property_maps = HashMap::new();
        property_maps.insert("density".to_string(), "density.nii.gz".to_string());
        let params = MediumParameters {
            medium_type: MediumType::Heterogeneous,
            density: 1000.0,
            sound_speed: Some(1500.0),
            sound_speed_min: None,
            sound_speed_max: None,
            absorption: 0.5,
            absorption_power: 1.1,
            nonlinearity: 7.0,
            tissue_file: Some("tissue.nii.gz".to_string()),
            property_maps,
            layers: Vec::new(),
            tensor_file: None,
            principal_directions: None,
            properties: HashMap::new(),
        };

        let cfg = params.to_factory_config().unwrap();
        match cfg.medium_type {
            crate::factory::MediumType::Heterogeneous {
                tissue_file,
                property_maps,
            } => {
                assert_eq!(tissue_file.as_deref(), Some("tissue.nii.gz"));
                assert_eq!(
                    property_maps.get("density").map(String::as_str),
                    Some("density.nii.gz")
                );
            }
            _ => panic!("expected heterogeneous factory config"),
        }
    }
}
