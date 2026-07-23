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
        // Validate dimensions
        if width <= Length::from_base(0.0)
            || height <= Length::from_base(0.0)
            || thickness <= Length::from_base(0.0)
        {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "element_dimensions".to_owned(),
                value: format!(
                    "width={}, height={}, thickness={}",
                    width.as_base(),
                    height.as_base(),
                    thickness.as_base()
                ),
                constraint: "All dimensions must be positive".to_owned(),
            }));
        }

        if kerf < Length::from_base(0.0) {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "kerf".to_owned(),
                value: kerf.as_base().to_string(),
                constraint: "Kerf must be non-negative".to_owned(),
            }));
        }

        let pitch = width + kerf;
        let aspect_ratio = *width.as_base() / *thickness.as_base();
        let fill_factor = *width.as_base() / *pitch.as_base();

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
        let kerf_ratio = *kerf.as_base() / *width.as_base();
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
            pitch,
            aspect_ratio,
            fill_factor,
        })
    }

    /// Calculate element area
    #[must_use]
    pub fn area(&self) -> Area {
        self.width * self.height
    }

    /// Calculate element volume
    #[must_use]
    pub fn volume(&self) -> Volume {
        self.width * self.height * self.thickness
    }

    /// Calculate resonance frequency based on thickness
    ///
    /// For thickness mode vibration: f = c / (2 * thickness)
    ///
    /// # Arguments
    /// * `sound_speed` - Speed of sound in the material (m/s)
    #[must_use]
    pub fn resonance_frequency(&self, sound_speed: Velocity) -> Frequency {
        sound_speed / (self.thickness * 2.0)
    }

    /// Calculate lateral mode frequencies
    ///
    /// Returns (width mode, height mode) frequencies in Hz
    #[must_use]
    pub fn lateral_modes(&self, sound_speed: Velocity) -> (Frequency, Frequency) {
        let width_mode = sound_speed / (self.width * 2.0);
        let height_mode = sound_speed / (self.height * 2.0);
        (width_mode, height_mode)
    }

    /// Check if lateral modes are well separated from main resonance
    ///
    /// Lateral modes should be > 2x the main resonance for clean operation
    #[must_use]
    pub fn validate_mode_separation(&self, sound_speed: Velocity) -> bool {
        let main_freq = self.resonance_frequency(sound_speed);
        let (width_mode, height_mode) = self.lateral_modes(sound_speed);

        width_mode > main_freq * 2.0 && height_mode > main_freq * 2.0
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
        VACUUM_PERMITTIVITY * dielectric_constant * *self.area().as_base()
            / *self.thickness.as_base()
    }

    /// Calculate mechanical compliance
    #[must_use]
    pub fn compliance(&self, youngs_modulus: f64) -> f64 {
        *self.thickness.as_base() / (youngs_modulus * *self.area().as_base())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aequitas::systems::si::units::{Hertz, MeterPerSecond, Millimeter};

    fn test_geometry() -> ElementGeometry {
        ElementGeometry::new(
            Length::from_unit::<Millimeter>(0.8),
            Length::from_unit::<Millimeter>(6.0),
            Length::from_unit::<Millimeter>(0.4),
            Length::from_unit::<Millimeter>(0.08),
        )
        .expect("valid element geometry")
    }

    #[test]
    fn geometry_derives_typed_area_volume_and_frequency() {
        let geometry = test_geometry();
        let area = geometry.area();
        let volume = geometry.volume();
        let resonance = geometry.resonance_frequency(Velocity::from_unit::<MeterPerSecond>(4600.0));

        assert!(*area.as_base() > 0.0);
        assert!(*volume.as_base() > 0.0);
        assert_eq!(
            *resonance.as_base(),
            4600.0 / (2.0 * *geometry.thickness.as_base())
        );
        assert!(resonance > Frequency::from_unit::<Hertz>(0.0));
        assert_eq!(*geometry.pitch.as_base(), 0.00088);
    }
}
