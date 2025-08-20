//! Medium factory for creating acoustic media
//!
//! Follows Creator pattern - responsible for creating medium objects

use crate::error::{KwaversResult, ConfigError};
use crate::grid::Grid;
use crate::medium::{Medium, homogeneous::HomogeneousMedium};
use crate::constants::physics;
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
        mu_s_prime: f64 
    },
    Heterogeneous { 
        tissue_file: Option<String> 
    },
}

impl MediumConfig {
    /// Validate medium configuration
    pub fn validate(&self) -> KwaversResult<()> {
        match &self.medium_type {
            MediumType::Homogeneous { density, sound_speed, mu_a, mu_s_prime } => {
                if *density <= 0.0 {
                    return Err(ConfigError::InvalidValue {
                        parameter: "density".to_string(),
                        value: density.to_string(),
                        constraint: "Density must be positive".to_string(),
                    }.into());
                }

                if *sound_speed <= 0.0 {
                    return Err(ConfigError::InvalidValue {
                        parameter: "sound_speed".to_string(),
                        value: sound_speed.to_string(),
                        constraint: "Sound speed must be positive".to_string(),
                    }.into());
                }

                if *mu_a < 0.0 {
                    return Err(ConfigError::InvalidValue {
                        parameter: "mu_a".to_string(),
                        value: mu_a.to_string(),
                        constraint: "Absorption coefficient must be non-negative".to_string(),
                    }.into());
                }

                if *mu_s_prime < 0.0 {
                    return Err(ConfigError::InvalidValue {
                        parameter: "mu_s_prime".to_string(),
                        value: mu_s_prime.to_string(),
                        constraint: "Reduced scattering coefficient must be non-negative".to_string(),
                    }.into());
                }
            }
            MediumType::Heterogeneous { tissue_file } => {
                if let Some(file) = tissue_file {
                    if file.is_empty() {
                        return Err(ConfigError::InvalidValue {
                            parameter: "tissue_file".to_string(),
                            value: "empty".to_string(),
                            constraint: "Tissue file path must not be empty".to_string(),
                        }.into());
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
                density: physics::DENSITY_WATER,
                sound_speed: physics::SOUND_SPEED_WATER,
                mu_a: 0.0,
                mu_s_prime: 0.0,
            },
            properties: HashMap::new(),
        }
    }
}

/// Factory for creating media
pub struct MediumFactory;

impl MediumFactory {
    /// Create a medium from configuration
    pub fn create_medium(config: &MediumConfig, grid: &Grid) -> KwaversResult<Box<dyn Medium>> {
        config.validate()?;
        
        match &config.medium_type {
            MediumType::Homogeneous { density, sound_speed, mu_a, mu_s_prime } => {
                // Create water-based medium with custom density and sound speed
                let mut medium = HomogeneousMedium::water();
                // Override with custom values using a builder pattern would be better,
                // but for now we'll create with all parameters
                let medium = HomogeneousMedium::new(
                    *density,
                    *sound_speed,
                    1.0e-3,  // viscosity (water default)
                    0.0728,  // surface tension (water default)
                    101325.0,  // ambient pressure (1 atm)
                    2338.0,  // vapor pressure (water at 20°C)
                    1.4,  // polytropic index (air)
                    4180.0,  // specific heat (water)
                    0.6,  // thermal conductivity (water)
                    *mu_a,  // attenuation
                    3.5,  // nonlinearity (water B/A)
                );
                Ok(Box::new(medium))
            }
            MediumType::Heterogeneous { tissue_file } => {
                // Heterogeneous medium requires loading from file
                Err(KwaversError::NotImplemented(
                    format!("Heterogeneous medium loading from '{}' not yet implemented", 
                            tissue_file.as_deref().unwrap_or("unknown"))
                ))
            }
        }
    }
    
    /// Create a water medium with standard properties
    pub fn create_water(grid: &Grid) -> KwaversResult<Box<dyn Medium>> {
        let config = MediumConfig {
            medium_type: MediumType::Homogeneous {
                density: physics::DENSITY_WATER,
                sound_speed: physics::SOUND_SPEED_WATER,
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
                density: physics::DENSITY_TISSUE,
                sound_speed: physics::SOUND_SPEED_TISSUE,
                mu_a: 0.01,  // Typical tissue absorption
                mu_s_prime: 1.0,  // Typical tissue scattering
            },
            properties: HashMap::new(),
        };
        Self::create_medium(&config, grid)
    }
}

use crate::error::KwaversError;