//! Medium validation rules - Evidence-based physics constraints
//!
//! Implements comprehensive validation following literature standards

use super::config::{LayerParameters, MediumParameters, MediumType};
use crate::domain::core::error::{ConfigError, KwaversResult};

/// Specialized medium validator following Single Responsibility Principle  
#[derive(Debug)]
pub struct MediumValidator;

impl MediumValidator {
    /// Validate medium configuration with physics-based constraints
    pub fn validate(config: &MediumParameters) -> KwaversResult<()> {
        match config.medium_type {
            MediumType::Homogeneous => {
                let mu_a = config.properties.get("mu_a").copied().unwrap_or(0.0);
                let mu_s_prime = config.properties.get("mu_s_prime").copied().unwrap_or(0.0);

                Self::validate_homogeneous(
                    config.density,
                    config.sound_speed.unwrap_or(1500.0),
                    mu_a,
                    mu_s_prime,
                )?;
            }
            MediumType::Heterogeneous => {
                Self::validate_heterogeneous(&config.tissue_file)?;
            }
            MediumType::Layered => {
                Self::validate_layered(&config.layers)?;
            }
            MediumType::Anisotropic => {
                // Tensor file is Option in parameters, but valid Anisotropic requires it?
                // MediumParameters has tensor_file: Option<String>.
                // Check if we need to unwrap or assume valid if used.
                // Anisotropic builder used config.tensor_file.
                // We should validate it here.
                if let Some(file) = &config.tensor_file {
                    Self::validate_anisotropic(file)?;
                } else {
                    return Err(ConfigError::MissingParameter {
                        parameter: "tensor_file".to_string(),
                        section: "medium".to_string(),
                    }
                    .into());
                }
            }
            _ => {} // Other types like Custom/Random/Tissue might need validation too
        }
        Ok(())
    }

    /// Validate homogeneous medium properties
    fn validate_homogeneous(
        density: f64,
        sound_speed: f64,
        mu_a: f64,
        mu_s_prime: f64,
    ) -> KwaversResult<()> {
        // Physical bounds based on literature (Hamilton & Blackstock 1998)
        if density <= 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "density".to_string(),
                value: density.to_string(),
                constraint: "Density must be positive".to_string(),
            }
            .into());
        }

        if sound_speed <= 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "sound_speed".to_string(),
                value: sound_speed.to_string(),
                constraint: "Sound speed must be positive".to_string(),
            }
            .into());
        }

        // Realistic ranges for biological media
        const MIN_DENSITY: f64 = 1.0; // kg/m³ (air-like minimum)
        const MAX_DENSITY: f64 = 8000.0; // kg/m³ (bone maximum)
        const MIN_SOUND_SPEED: f64 = 200.0; // m/s (air-like minimum)
        const MAX_SOUND_SPEED: f64 = 6000.0; // m/s (bone maximum)

        if !(MIN_DENSITY..=MAX_DENSITY).contains(&density) {
            return Err(ConfigError::InvalidValue {
                parameter: "density".to_string(),
                value: density.to_string(),
                constraint: format!("Density must be within [{MIN_DENSITY}, {MAX_DENSITY}] kg/m³"),
            }
            .into());
        }

        if !(MIN_SOUND_SPEED..=MAX_SOUND_SPEED).contains(&sound_speed) {
            return Err(ConfigError::InvalidValue {
                parameter: "sound_speed".to_string(),
                value: sound_speed.to_string(),
                constraint: format!(
                    "Sound speed must be within [{MIN_SOUND_SPEED}, {MAX_SOUND_SPEED}] m/s"
                ),
            }
            .into());
        }

        // Optical properties validation
        if mu_a < 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "mu_a".to_string(),
                value: mu_a.to_string(),
                constraint: "Absorption coefficient must be non-negative".to_string(),
            }
            .into());
        }

        if mu_s_prime < 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "mu_s_prime".to_string(),
                value: mu_s_prime.to_string(),
                constraint: "Reduced scattering coefficient must be non-negative".to_string(),
            }
            .into());
        }

        Ok(())
    }

    /// Validate heterogeneous medium configuration
    fn validate_heterogeneous(tissue_file: &Option<String>) -> KwaversResult<()> {
        if let Some(file_path) = tissue_file {
            if file_path.is_empty() {
                return Err(ConfigError::InvalidValue {
                    parameter: "tissue_file".to_string(),
                    value: "empty".to_string(),
                    constraint: "Tissue file path cannot be empty".to_string(),
                }
                .into());
            }

            // Check file extension for supported formats
            let valid_extensions = [".nii", ".nii.gz", ".mat", ".h5"];
            if !valid_extensions.iter().any(|ext| file_path.ends_with(ext)) {
                return Err(ConfigError::InvalidValue {
                    parameter: "tissue_file".to_string(),
                    value: file_path.clone(),
                    constraint: format!(
                        "File must have one of these extensions: {}",
                        valid_extensions.join(", ")
                    ),
                }
                .into());
            }
        }
        Ok(())
    }

    /// Validate layered medium configuration
    fn validate_layered(layers: &[LayerParameters]) -> KwaversResult<()> {
        if layers.is_empty() {
            return Err(ConfigError::InvalidValue {
                parameter: "layers".to_string(),
                value: "empty".to_string(),
                constraint: "At least one layer is required".to_string(),
            }
            .into());
        }

        for (i, layer) in layers.iter().enumerate() {
            if layer.thickness <= 0.0 {
                return Err(ConfigError::InvalidValue {
                    parameter: format!("layer[{}].thickness", i),
                    value: layer.thickness.to_string(),
                    constraint: "Layer thickness must be positive".to_string(),
                }
                .into());
            }

            // Validate each layer's physical properties
            Self::validate_homogeneous(
                layer.density,
                layer.sound_speed,
                layer.absorption,
                0.0, // No scattering for simple layers
            )?;
        }

        Ok(())
    }

    /// Validate anisotropic medium configuration
    fn validate_anisotropic(tensor_file: &str) -> KwaversResult<()> {
        if tensor_file.is_empty() {
            return Err(ConfigError::InvalidValue {
                parameter: "tensor_file".to_string(),
                value: "empty".to_string(),
                constraint: "Tensor file path cannot be empty".to_string(),
            }
            .into());
        }

        // Validate tensor file format
        if !tensor_file.ends_with(".ten") && !tensor_file.ends_with(".mat") {
            return Err(ConfigError::InvalidValue {
                parameter: "tensor_file".to_string(),
                value: tensor_file.to_string(),
                constraint: "Tensor file must have .ten or .mat extension".to_string(),
            }
            .into());
        }

        Ok(())
    }
}
