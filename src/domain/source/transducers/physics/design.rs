//! Transducer Design Module
//!
//! Complete transducer design combining all components.

use super::{
    BackingLayer, DirectivityPattern, ElementGeometry, FrequencyResponse, MatchingLayer,
    PiezoMaterial, TransducerSensitivity,
};
use crate::domain::core::error::{ConfigError, KwaversError, KwaversResult};

/// Complete transducer design specification
#[derive(Debug, Clone)]
pub struct TransducerDesign {
    /// Element geometry
    pub geometry: ElementGeometry,
    /// Piezoelectric material
    pub piezo: PiezoMaterial,
    /// Backing layer
    pub backing: BackingLayer,
    /// Matching layers
    pub matching_layers: Vec<MatchingLayer>,
    /// Frequency response
    pub frequency_response: FrequencyResponse,
    /// Directivity pattern
    pub directivity: DirectivityPattern,
    /// Sensitivity characteristics
    pub sensitivity: TransducerSensitivity,
}

impl TransducerDesign {
    /// Design transducer for specific application
    ///
    /// # Arguments
    /// * `frequency` - Operating frequency (Hz)
    /// * `num_elements` - Number of array elements
    /// * `aperture` - Total aperture size (m)
    /// * `focal_length` - Optional focal length for focused transducer (m)
    pub fn design_for_application(
        frequency: f64,
        num_elements: usize,
        aperture: f64,
        focal_length: Option<f64>,
    ) -> KwaversResult<Self> {
        // Calculate element dimensions
        let pitch = aperture / num_elements as f64;
        let kerf = pitch * 0.1; // 10% kerf
        let width = pitch - kerf;

        // Height depends on focusing
        let height = if focal_length.is_some() {
            aperture / 4.0 // Smaller for focused
        } else {
            aperture / 2.0 // Larger for unfocused
        };

        // Calculate thickness for resonance at desired frequency
        let piezo = PiezoMaterial::pzt_5h();
        let thickness = piezo.sound_speed / (2.0 * frequency);

        let geometry = ElementGeometry::new(width, height, thickness, kerf)?;

        // Design backing for broadband response
        let backing = BackingLayer::tungsten_epoxy(5e-3);

        // Design matching layer
        let matching_layer = MatchingLayer::quarter_wave(
            frequency,
            piezo.acoustic_impedance,
            super::TISSUE_IMPEDANCE,
        );

        // Calculate frequency response
        let frequency_response = FrequencyResponse::from_klm_model(
            frequency,
            piezo.coupling_k33,
            piezo.mechanical_q,
            super::ELECTRICAL_Q,
            200,
        )?;

        // Calculate directivity pattern
        let directivity = DirectivityPattern::rectangular_element(width, height, frequency, 180);

        // Calculate sensitivity
        let sensitivity = TransducerSensitivity::from_parameters(
            piezo.coupling_k33,
            geometry.area(),
            piezo.acoustic_impedance * 1e6, // Convert to Pa·s/m
            frequency,
        );

        Ok(Self {
            geometry,
            piezo,
            backing,
            matching_layers: vec![matching_layer],
            frequency_response,
            directivity,
            sensitivity,
        })
    }

    /// Validate complete design
    pub fn validate(&self) -> KwaversResult<()> {
        // Check mode separation
        if !self
            .geometry
            .validate_mode_separation(self.piezo.sound_speed)
        {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "mode_separation".to_string(),
                value: "insufficient".to_string(),
                constraint: "Lateral modes too close to main resonance".to_string(),
            }));
        }

        // Check bandwidth
        if !self.frequency_response.validate_bandwidth(20.0) {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "bandwidth".to_string(),
                value: self.frequency_response.fractional_bandwidth.to_string(),
                constraint: "Fractional bandwidth less than 20%".to_string(),
            }));
        }

        // Check sensitivity
        if !self.sensitivity.validate_sensitivity(40.0) {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "sensitivity".to_string(),
                value: "insufficient".to_string(),
                constraint: "SNR less than 40 dB at typical imaging depth".to_string(),
            }));
        }

        Ok(())
    }

    /// Calculate focal characteristics (if focused)
    #[must_use]
    pub fn focal_characteristics(&self, focal_length: f64) -> (f64, f64, f64) {
        let wavelength = 1540.0 / self.frequency_response.center_frequency;
        let aperture = self.geometry.width * 64.0; // Assume 64 element array

        // Focal zone length (depth of field)
        let focal_zone = 7.0 * wavelength * (focal_length / aperture).powi(2);

        // Lateral resolution at focus
        let lateral_resolution = 1.22 * wavelength * focal_length / aperture;

        // F-number
        let f_number = focal_length / aperture;

        (focal_zone, lateral_resolution, f_number)
    }

    /// Generate design report
    #[must_use]
    pub fn design_report(&self) -> String {
        format!(
            "Transducer Design Report\n\
            ========================\n\
            Frequency: {:.2} MHz\n\
            Bandwidth: {:.1}%\n\
            Element Size: {:.2} x {:.2} mm\n\
            Thickness: {:.3} mm\n\
            Material: {:?}\n\
            Coupling: {:.2}\n\
            Beamwidth: {:.1}°\n\
            Sensitivity: {:.1} Pa/V at 1m\n\
            Efficiency: {:.1}%",
            self.frequency_response.center_frequency / 1e6,
            self.frequency_response.fractional_bandwidth,
            self.geometry.width * 1e3,
            self.geometry.height * 1e3,
            self.geometry.thickness * 1e3,
            self.piezo.material_type,
            self.piezo.coupling_k33,
            self.directivity.beamwidth_3db,
            self.sensitivity.transmit_sensitivity,
            self.sensitivity.efficiency,
        )
    }
}
