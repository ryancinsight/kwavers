//! Element Geometry Module
//!
//! Defines the physical geometry of transducer elements including
//! dimensions, spacing, and aspect ratios.

use aequitas::systems::si::quantities::{Area, Frequency, Length, Velocity, Volume};
use kwavers_core::error::{ConfigError, KwaversError, KwaversResult};

/// Element geometry parameters for transducer design
///
/// Based on:
/// - Szabo (2014): Chapter 5 - Transducer Arrays
/// - Shung (2015): Section 3.4 - Array Transducers
#[derive(Debug, Clone)]
pub struct ElementGeometry {
    /// Element width (m)
    pub width: Length,
    /// Element height (m)
    pub height: Length,
    /// Element thickness (m)
    pub thickness: Length,
    /// Kerf width between elements (m)
    pub kerf: Length,
    /// Element pitch (center-to-center spacing) (m)
    pub pitch: Length,
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
    pub fn new(
        width: Length,
        height: Length,
        thickness: Length,
        kerf: Length,
    ) -> KwaversResult<Self> {
        let width_m = width.into_base();
        let height_m = height.into_base();
        let thickness_m = thickness.into_base();
        let kerf_m = kerf.into_base();
        // Validate dimensions
        if width_m <= 0.0 || height_m <= 0.0 || thickness_m <= 0.0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "element_dimensions".to_owned(),
                value: format!("width={width_m}, height={height_m}, thickness={thickness_m}"),
                constraint: "All dimensions must be positive".to_owned(),
            }));
        }

        if kerf_m < 0.0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "kerf".to_owned(),
                value: kerf_m.to_string(),
                constraint: "Kerf must be non-negative".to_owned(),
            }));
        }

        let pitch_m = width_m + kerf_m;
        let aspect_ratio = width_m / thickness_m;
        let fill_factor = width_m / pitch_m;

        // Validate aspect ratio (Hunt et al., 1983)
        if !(super::MIN_ASPECT_RATIO..=super::MAX_ASPECT_RATIO).contains(&aspect_ratio) {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "aspect_ratio".to_owned(),
                value: aspect_ratio.to_string(),
                constraint: format!(
                    "Aspect ratio must be between {} and {}",
                    super::MIN_ASPECT_RATIO,
                    super::MAX_ASPECT_RATIO
                ),
            }));
        }

        // Validate kerf ratio
        let kerf_ratio = kerf_m / width_m;
        if !(super::MIN_KERF_RATIO..=super::MAX_KERF_RATIO).contains(&kerf_ratio) {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "kerf_ratio".to_owned(),
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
            pitch: Length::from_base(pitch_m),
            aspect_ratio,
            fill_factor,
        })
    }

    /// Calculate element area
    #[must_use]
    pub fn area(&self) -> Area {
        Area::from_base(self.width.into_base() * self.height.into_base())
    }

    /// Calculate element volume
    #[must_use]
    pub fn volume(&self) -> Volume {
        Volume::from_base(
            self.width.into_base() * self.height.into_base() * self.thickness.into_base(),
        )
    }

    /// Calculate resonance frequency based on thickness
    ///
    /// For thickness mode vibration: f = c / (2 * thickness)
    ///
    /// # Arguments
    /// * `sound_speed` - Speed of sound in the material (m/s)
    #[must_use]
    pub fn resonance_frequency(&self, sound_speed: Velocity) -> Frequency {
        Frequency::from_base(sound_speed.into_base() / (2.0 * self.thickness.into_base()))
    }

    /// Calculate lateral mode frequencies
    ///
    /// Returns (width mode, height mode) frequencies in Hz
    #[must_use]
    pub fn lateral_modes(&self, sound_speed: Velocity) -> (Frequency, Frequency) {
        let width_mode =
            Frequency::from_base(sound_speed.into_base() / (2.0 * self.width.into_base()));
        let height_mode =
            Frequency::from_base(sound_speed.into_base() / (2.0 * self.height.into_base()));
        (width_mode, height_mode)
    }

    /// Check if lateral modes are well separated from main resonance
    ///
    /// Lateral modes should be > 2x the main resonance for clean operation
    #[must_use]
    pub fn validate_mode_separation(&self, sound_speed: Velocity) -> bool {
        let main_freq = self.resonance_frequency(sound_speed).into_base();
        let (width_mode, height_mode) = self.lateral_modes(sound_speed);

        width_mode.into_base() > 2.0 * main_freq && height_mode.into_base() > 2.0 * main_freq
    }

    /// Calculate electrical capacitance of the element
    ///
    /// C = ε₀ * εᵣ * A / t
    ///
    /// # Arguments
    /// * `dielectric_constant` - Relative dielectric constant
    #[must_use]
    pub fn capacitance(&self, dielectric_constant: f64) -> f64 {
        use kwavers_core::constants::fundamental::VACUUM_PERMITTIVITY;
        VACUUM_PERMITTIVITY * dielectric_constant * self.area().into_base()
            / self.thickness.into_base()
    }

    /// Calculate mechanical compliance
    #[must_use]
    pub fn compliance(&self, youngs_modulus: f64) -> f64 {
        self.thickness.into_base() / (youngs_modulus * self.area().into_base())
    }
}
