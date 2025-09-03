//! Medium factory for creating acoustic media
//!
//! Follows Creator pattern - responsible for creating medium objects

use crate::error::{ConfigError, KwaversResult};
use crate::grid::Grid;
use crate::medium::{homogeneous::HomogeneousMedium, Medium};
use crate::physics::constants::{
    DENSITY_TISSUE, DENSITY_WATER, SOUND_SPEED_TISSUE, SOUND_SPEED_WATER,
};
use std::collections::HashMap;

/// Medium configuration
#[derive(Debug, Clone)]
pub struct MediumConfig {
    pub medium_type: MediumType,
    pub properties: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub enum MediumType {
    Homogeneous {
        density: f64,
        sound_speed: f64,
        mu_a: f64,
        mu_s_prime: f64,
    },
    Heterogeneous {
        tissue_file: Option<String>,
    },
}

impl MediumConfig {
    /// Validate medium configuration
    pub fn validate(&self) -> KwaversResult<()> {
        match &self.medium_type {
            MediumType::Homogeneous {
                density,
                sound_speed,
                mu_a,
                mu_s_prime,
            } => {
                if *density <= 0.0 {
                    return Err(ConfigError::InvalidValue {
                        parameter: "density".to_string(),
                        value: density.to_string(),
                        constraint: "Density must be positive".to_string(),
                    }
                    .into());
                }

                if *sound_speed <= 0.0 {
                    return Err(ConfigError::InvalidValue {
                        parameter: "sound_speed".to_string(),
                        value: sound_speed.to_string(),
                        constraint: "Sound speed must be positive".to_string(),
                    }
                    .into());
                }

                if *mu_a < 0.0 {
                    return Err(ConfigError::InvalidValue {
                        parameter: "mu_a".to_string(),
                        value: mu_a.to_string(),
                        constraint: "Absorption coefficient must be non-negative".to_string(),
                    }
                    .into());
                }

                if *mu_s_prime < 0.0 {
                    return Err(ConfigError::InvalidValue {
                        parameter: "mu_s_prime".to_string(),
                        value: mu_s_prime.to_string(),
                        constraint: "Reduced scattering coefficient must be non-negative"
                            .to_string(),
                    }
                    .into());
                }
            }
            MediumType::Heterogeneous { tissue_file } => {
                if let Some(file) = tissue_file {
                    if file.is_empty() {
                        return Err(ConfigError::InvalidValue {
                            parameter: "tissue_file".to_string(),
                            value: "empty".to_string(),
                            constraint: "Tissue file path must not be empty".to_string(),
                        }
                        .into());
                    }
                }
            }
        }
        Ok(())
    }
}

impl Default for MediumConfig {
    fn default() -> Self {
        Self {
            medium_type: MediumType::Homogeneous {
                density: DENSITY_WATER,
                sound_speed: SOUND_SPEED_WATER,
                mu_a: 0.0,
                mu_s_prime: 0.0,
            },
            properties: HashMap::new(),
        }
    }
}

/// Factory for creating media
#[derive(Debug)]
pub struct MediumFactory;

impl MediumFactory {
    /// Create a medium from configuration
    pub fn create_medium(config: &MediumConfig, grid: &Grid) -> KwaversResult<Box<dyn Medium>> {
        config.validate()?;

        match &config.medium_type {
            MediumType::Homogeneous {
                density,
                sound_speed,
                mu_a,
                mu_s_prime,
            } => {
                // Create homogeneous medium with the required parameters
                let medium =
                    HomogeneousMedium::new(*density, *sound_speed, *mu_a, *mu_s_prime, grid);
                Ok(Box::new(medium))
            }
            MediumType::Heterogeneous { tissue_file } => {
                // Heterogeneous medium requires loading from file
                Err(KwaversError::NotImplemented(format!(
                    "Heterogeneous medium loading from '{}' not yet implemented",
                    tissue_file.as_deref().unwrap_or("unknown")
                )))
            }
        }
    }

    /// Create a water medium with standard properties
    pub fn create_water(grid: &Grid) -> KwaversResult<Box<dyn Medium>> {
        let config = MediumConfig {
            medium_type: MediumType::Homogeneous {
                density: DENSITY_WATER,
                sound_speed: SOUND_SPEED_WATER,
                mu_a: 0.0,
                mu_s_prime: 0.0,
            },
            properties: HashMap::new(),
        };
        Self::create_medium(&config, grid)
    }

    /// Create a soft tissue medium with standard properties
    pub fn create_tissue(grid: &Grid) -> KwaversResult<Box<dyn Medium>> {
        let config = MediumConfig {
            medium_type: MediumType::Homogeneous {
                density: DENSITY_TISSUE,
                sound_speed: SOUND_SPEED_TISSUE,
                mu_a: 0.01,      // Typical tissue absorption
                mu_s_prime: 1.0, // Typical tissue scattering
            },
            properties: HashMap::new(),
        };
        Self::create_medium(&config, grid)
    }
}

use crate::error::KwaversError;
