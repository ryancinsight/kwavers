//! Medium type definitions and configurations
//!
//! Comprehensive type system for acoustic media following domain principles

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Medium configuration with comprehensive type safety
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediumConfig {
    pub medium_type: MediumType,
    pub properties: HashMap<String, f64>,
}

/// Strongly-typed medium variants preventing configuration errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MediumType {
    /// Homogeneous medium with uniform properties
    Homogeneous {
        density: f64,
        sound_speed: f64,
        mu_a: f64,       // Absorption coefficient
        mu_s_prime: f64, // Reduced scattering coefficient
    },
    /// Heterogeneous medium with spatial property variation
    Heterogeneous {
        tissue_file: Option<String>,
        property_maps: HashMap<String, String>,
    },
    /// Layered medium with discrete layers
    Layered { layers: Vec<LayerProperties> },
    /// Anisotropic medium with directional properties
    Anisotropic {
        tensor_file: String,
        principal_directions: Option<[f64; 3]>,
    },
}

/// Layer properties for layered media
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerProperties {
    pub thickness: f64,
    pub density: f64,
    pub sound_speed: f64,
    pub absorption: f64,
    pub interface_type: InterfaceType,
}

/// Interface types between layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterfaceType {
    Sharp,         // Discontinuous interface
    Smooth(f64),   // Smooth transition with width
    Gradient(f64), // Linear gradient over distance
}

impl MediumConfig {
    /// Create homogeneous medium configuration
    pub fn homogeneous(density: f64, sound_speed: f64, mu_a: f64, mu_s_prime: f64) -> Self {
        Self {
            medium_type: MediumType::Homogeneous {
                density,
                sound_speed,
                mu_a,
                mu_s_prime,
            },
            properties: HashMap::new(),
        }
    }

    /// Create heterogeneous medium from file
    pub fn heterogeneous(tissue_file: String) -> Self {
        Self {
            medium_type: MediumType::Heterogeneous {
                tissue_file: Some(tissue_file),
                property_maps: HashMap::new(),
            },
            properties: HashMap::new(),
        }
    }

    /// Validate configuration - maintains backward compatibility
    pub fn validate(&self) -> crate::error::KwaversResult<()> {
        super::validation::MediumValidator::validate(self)
    }
}

impl Default for MediumConfig {
    fn default() -> Self {
        Self::homogeneous(1000.0, 1500.0, 0.5, 10.0) // Water-like properties
    }
}
