//! Element Geometry Module
//!
//! Defines the physical geometry of transducer elements including
//! dimensions, spacing, and aspect ratios.

use crate::error::{ConfigError, KwaversError, KwaversResult};

/// Element geometry parameters for transducer design
///
/// Based on:
/// - Szabo (2014): Chapter 5 - Transducer Arrays
/// - Shung (2015): Section 3.4 - Array Transducers
#[derive(Debug, Clone)]
pub struct ElementGeometry {
    /// Element width (m)
    pub width: f64,
    /// Element height (m)
    pub height: f64,
    /// Element thickness (m)
    pub thickness: f64,
    /// Kerf width between elements (m)
    pub kerf: f64,
    /// Element pitch (center-to-center spacing) (m)
    pub pitch: f64,
    /// Aspect ratio (width/thickness)
    pub aspect_ratio: f64,
    /// Fill factor (active area / total area)
    pub fill_factor: f64,
}

impl ElementGeometry {
    /// Create element geometry with validation
    ///
    /// # Arguments
    /// * `width` - Element width in meters
    /// * `height` - Element height in meters  
    /// * `thickness` - Element thickness in meters
    /// * `kerf` - Kerf (gap) between elements in meters
    ///
    /// # Returns
    /// Validated element geometry
    ///
    /// # Errors
    /// Returns error if dimensions are invalid or aspect ratio out of range
    pub fn new(width: f64, height: f64, thickness: f64, kerf: f64) -> KwaversResult<Self> {
        // Validate dimensions
        if width <= 0.0 || height <= 0.0 || thickness <= 0.0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "element_dimensions".to_string(),
                value: format!("width={}, height={}, thickness={}", width, height, thickness),
                constraint: "All dimensions must be positive".to_string(),
            }));
        }

        if kerf < 0.0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "kerf".to_string(),
                value: kerf.to_string(),
                constraint: "Kerf must be non-negative".to_string(),
            }));
        }

        let pitch = width + kerf;
        let aspect_ratio = width / thickness;
        let fill_factor = width / pitch;

        // Validate aspect ratio (Hunt et al., 1983)
        if aspect_ratio < super::MIN_ASPECT_RATIO || aspect_ratio > super::MAX_ASPECT_RATIO {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "aspect_ratio".to_string(),
                value: aspect_ratio.to_string(),
                constraint: format!(
                    "Aspect ratio must be between {} and {}",
                    super::MIN_ASPECT_RATIO,
                    super::MAX_ASPECT_RATIO
                ),
            }));
        }

        // Validate kerf ratio
        let kerf_ratio = kerf / width;
        if kerf_ratio < super::MIN_KERF_RATIO || kerf_ratio > super::MAX_KERF_RATIO {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "kerf_ratio".to_string(),
                value: kerf_ratio.to_string(),
                constraint: format!(
                    "Kerf ratio must be between {} and {}",
                    super::MIN_KERF_RATIO,
                    super::MAX_KERF_RATIO
                ),
            }));
        }

        Ok(Self {
            width,
            height,
            thickness,
            kerf,
            pitch,
            aspect_ratio,
            fill_factor,
        })
    }

    /// Calculate element area
    pub fn area(&self) -> f64 {
        self.width * self.height
    }

    /// Calculate element volume
    pub fn volume(&self) -> f64 {
        self.width * self.height * self.thickness
    }

    /// Calculate resonance frequency based on thickness
    ///
    /// For thickness mode vibration: f = c / (2 * thickness)
    ///
    /// # Arguments
    /// * `sound_speed` - Speed of sound in the material (m/s)
    pub fn resonance_frequency(&self, sound_speed: f64) -> f64 {
        sound_speed / (2.0 * self.thickness)
    }

    /// Calculate lateral mode frequencies
    ///
    /// Returns (width mode, height mode) frequencies in Hz
    pub fn lateral_modes(&self, sound_speed: f64) -> (f64, f64) {
        let width_mode = sound_speed / (2.0 * self.width);
        let height_mode = sound_speed / (2.0 * self.height);
        (width_mode, height_mode)
    }

    /// Check if lateral modes are well separated from main resonance
    ///
    /// Lateral modes should be > 2x the main resonance for clean operation
    pub fn validate_mode_separation(&self, sound_speed: f64) -> bool {
        let main_freq = self.resonance_frequency(sound_speed);
        let (width_mode, height_mode) = self.lateral_modes(sound_speed);
        
        width_mode > 2.0 * main_freq && height_mode > 2.0 * main_freq
    }

    /// Calculate electrical capacitance of the element
    ///
    /// C = ε₀ * εᵣ * A / t
    ///
    /// # Arguments
    /// * `dielectric_constant` - Relative dielectric constant
    pub fn capacitance(&self, dielectric_constant: f64) -> f64 {
        const EPSILON_0: f64 = 8.854e-12; // F/m
        EPSILON_0 * dielectric_constant * self.area() / self.thickness
    }

    /// Calculate mechanical compliance
    pub fn compliance(&self, youngs_modulus: f64) -> f64 {
        self.thickness / (youngs_modulus * self.area())
    }
}