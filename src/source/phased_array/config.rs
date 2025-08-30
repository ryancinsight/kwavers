//! Configuration for phased array transducers

use crate::constants::medium_properties::TISSUE_SOUND_SPEED;

/// Configuration for phased array transducer geometry and behavior
#[derive(Debug, Clone))]
pub struct PhasedArrayConfig {
    /// Number of elements in the array
    pub num_elements: usize,
    /// Element spacing [m]
    pub element_spacing: f64,
    /// Element width [m]
    pub element_width: f64,
    /// Element height [m]
    pub element_height: f64,
    /// Array center position (x, y, z) [m]
    pub center_position: (f64, f64, f64),
    /// Operating frequency [Hz]
    pub frequency: f64,
    /// Enable element cross-talk modeling
    pub enable_crosstalk: bool,
    /// Cross-talk coupling coefficient [0.0-1.0]
    pub crosstalk_coefficient: f64,
}

impl Default for PhasedArrayConfig {
    fn default() -> Self {
        // Standard linear array for medical ultrasound
        let wavelength = TISSUE_SOUND_SPEED / 2.5e6; // λ at 2.5 MHz

        Self {
            num_elements: 64,
            element_spacing: wavelength / 2.0, // λ/2 spacing for no grating lobes
            element_width: wavelength * 0.45,  // ~0.45λ width
            element_height: 10e-3,             // 10mm elevation
            center_position: (0.0, 0.0, 0.0),
            frequency: 2.5e6, // 2.5 MHz center frequency
            enable_crosstalk: true,
            crosstalk_coefficient: 0.1, // 10% coupling (typical)
        }
    }
}

impl PhasedArrayConfig {
    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.num_elements == 0 {
            return Err("Number of elements must be positive".to_string());
        }

        if self.element_spacing <= 0.0 {
            return Err("Element spacing must be positive".to_string());
        }

        if self.element_width <= 0.0 || self.element_width > self.element_spacing {
            return Err("Element width must be positive and less than spacing".to_string());
        }

        if self.frequency <= 0.0 {
            return Err("Frequency must be positive".to_string());
        }

        if self.crosstalk_coefficient < 0.0 || self.crosstalk_coefficient > 1.0 {
            return Err("Crosstalk coefficient must be between 0 and 1".to_string());
        }

        Ok(())
    }

    /// Calculate array aperture size
    pub fn aperture_size(&self) -> f64 {
        (self.num_elements as f64 - 1.0) * self.element_spacing + self.element_width
    }

    /// Calculate Nyquist sampling criterion
    pub fn satisfies_nyquist(&self, sound_speed: f64) -> bool {
        let wavelength = sound_speed / self.frequency;
        self.element_spacing <= wavelength / 2.0
    }
}
