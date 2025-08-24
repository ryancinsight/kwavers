//! Transducer element geometry module

use crate::error::{ConfigError, KwaversError, KwaversResult};

// Geometry constants
const MIN_KERF_RATIO: f64 = 0.05;
const MAX_KERF_RATIO: f64 = 0.3;
const MIN_ASPECT_RATIO: f64 = 0.5;
const MAX_ASPECT_RATIO: f64 = 20.0;

/// Element geometry and dimensions
#[derive(Debug, Clone)]
pub struct ElementGeometry {
    /// Element width [m]
    pub width: f64,
    /// Element height [m]
    pub height: f64,
    /// Element thickness [m]
    pub thickness: f64,
    /// Kerf width between elements [m]
    pub kerf: f64,
    /// Element pitch (width + kerf) [m]
    pub pitch: f64,
    /// Aspect ratio (width/thickness)
    pub aspect_ratio: f64,
    /// Element area [m²]
    pub area: f64,
    /// Element volume [m³]
    pub volume: f64,
}

impl ElementGeometry {
    /// Create new element geometry with validation
    pub fn new(width: f64, height: f64, thickness: f64, kerf: f64) -> KwaversResult<Self> {
        // Validate dimensions
        if width <= 0.0 || height <= 0.0 || thickness <= 0.0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "dimensions".to_string(),
                value: format!("{}x{}x{}", width, height, thickness),
                constraint: "Must be positive".to_string(),
            }));
        }

        if kerf < 0.0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "kerf".to_string(),
                value: kerf.to_string(),
                constraint: "Cannot be negative".to_string(),
            }));
        }

        let kerf_ratio = kerf / width;
        if kerf_ratio < MIN_KERF_RATIO {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "kerf_ratio".to_string(),
                value: kerf_ratio.to_string(),
                constraint: format!("Must be >= {}", MIN_KERF_RATIO),
            }));
        }

        if kerf_ratio > MAX_KERF_RATIO {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "kerf_ratio".to_string(),
                value: kerf_ratio.to_string(),
                constraint: format!("Must be <= {}", MAX_KERF_RATIO),
            }));
        }

        let aspect_ratio = width / thickness;
        if aspect_ratio < MIN_ASPECT_RATIO || aspect_ratio > MAX_ASPECT_RATIO {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "aspect_ratio".to_string(),
                value: aspect_ratio.to_string(),
                constraint: format!("Must be between {} and {}", MIN_ASPECT_RATIO, MAX_ASPECT_RATIO),
            }));
        }

        Ok(Self {
            width,
            height,
            thickness,
            kerf,
            pitch: width + kerf,
            aspect_ratio,
            area: width * height,
            volume: width * height * thickness,
        })
    }

    /// Calculate element capacitance
    pub fn capacitance(&self, permittivity: f64) -> f64 {
        permittivity * self.area / self.thickness
    }

    /// Calculate lateral mode frequency
    pub fn lateral_mode_frequency(&self, sound_speed: f64) -> f64 {
        sound_speed / (2.0 * self.width)
    }

    /// Calculate thickness mode frequency
    pub fn thickness_mode_frequency(&self, sound_speed: f64) -> f64 {
        sound_speed / (2.0 * self.thickness)
    }
}