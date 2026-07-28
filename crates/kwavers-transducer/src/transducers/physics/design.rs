//! Transducer Design Module
//!
//! Complete transducer design combining all components.

use super::{
    BackingLayer, ElementGeometry, FrequencyResponse, MatchingLayer, PiezoMaterial,
    TransducerDirectivityPattern, TransducerSensitivity,
};
use aequitas::systems::si::quantities::{Dimensionless, Frequency, Length};
use kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE;
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::error::{ConfigError, KwaversError, KwaversResult};

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
    pub directivity: TransducerDirectivityPattern,
    /// Sensitivity characteristics
    pub sensitivity: TransducerSensitivity,
}

impl TransducerDesign {
    /// Design transducer for specific application
    ///
    /// # Arguments
    /// * `frequency` - Operating frequency
    /// * `num_elements` - Number of array elements
    /// * `aperture` - Total aperture size
    /// * `focal_length` - Optional focal length for focused transducer
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn design_for_application(
        frequency: Frequency,
        num_elements: usize,
        aperture: Length,
        focal_length: Option<Length>,
    ) -> KwaversResult<Self> {
        // Calculate element dimensions
        let aperture_m = aperture.into_base();
        let pitch_m = aperture_m / num_elements as f64;
        let kerf_m = pitch_m * 0.1; // 10% kerf
        let width_m = pitch_m - kerf_m;

        // Height depends on focusing
        let height = if focal_length.is_some() {
            aperture_m / 4.0 // Smaller for focused
        } else {
            aperture_m / 2.0 // Larger for unfocused
        };

        // Calculate thickness for resonance at desired frequency
        let piezo = PiezoMaterial::pzt_5h();
        let thickness_m = piezo.sound_speed.into_base() / (2.0 * frequency.into_base());

        let geometry = ElementGeometry::new(
            Length::from_base(width_m),
            Length::from_base(height),
            Length::from_base(thickness_m),
            Length::from_base(kerf_m),
        )?;

        // Design backing for broadband response
        let backing = BackingLayer::tungsten_epoxy(Length::from_base(5e-3));

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
        let directivity = TransducerDirectivityPattern::rectangular_element(
            width_m,
            height,
            frequency.into_base(),
            180,
        );

        // Calculate sensitivity
        let sensitivity = TransducerSensitivity::from_parameters(
            Dimensionless::from_base(piezo.coupling_k33),
            geometry.area(),
            piezo.acoustic_impedance,
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
    /// # Errors
    /// - Returns `KwaversError::Config` if the precondition for a Config-class constraint is violated.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        // Check mode separation
        if !self
            .geometry
            .validate_mode_separation(self.piezo.sound_speed)
        {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "mode_separation".to_owned(),
                value: "insufficient".to_owned(),
                constraint: "Lateral modes too close to main resonance".to_owned(),
            }));
        }

        // Check bandwidth
        if !self.frequency_response.validate_bandwidth(20.0) {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "bandwidth".to_owned(),
                value: self.frequency_response.fractional_bandwidth.to_string(),
                constraint: "Fractional bandwidth less than 20%".to_owned(),
            }));
        }

        // Check sensitivity
        if !self.sensitivity.validate_sensitivity(40.0) {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "sensitivity".to_owned(),
                value: "insufficient".to_owned(),
                constraint: "SNR less than 40 dB at typical imaging depth".to_owned(),
            }));
        }

        Ok(())
    }

    /// Calculate focal characteristics (if focused)
    #[must_use]
    pub fn focal_characteristics(&self, focal_length: Length) -> (Length, Length, f64) {
        let wavelength_m =
            SOUND_SPEED_TISSUE / self.frequency_response.center_frequency.into_base();
        let aperture_m = self.geometry.width.into_base() * 64.0; // Assume 64 element array
        let focal_length_m = focal_length.into_base();

        // Focal zone length (depth of field)
        let focal_zone_m = 7.0 * wavelength_m * (focal_length_m / aperture_m).powi(2);

        // Lateral resolution at focus
        let lateral_resolution_m = 1.22 * wavelength_m * focal_length_m / aperture_m;

        // F-number
        let f_number = focal_length_m / aperture_m;

        (
            Length::from_base(focal_zone_m),
            Length::from_base(lateral_resolution_m),
            f_number,
        )
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
            self.frequency_response.center_frequency.into_base() / MHZ_TO_HZ,
            self.frequency_response.fractional_bandwidth,
            self.geometry.width.into_base() * 1e3,
            self.geometry.height.into_base() * 1e3,
            self.geometry.thickness.into_base() * 1e3,
            self.piezo.material_type,
            self.piezo.coupling_k33,
            self.directivity.beamwidth_3db,
            self.sensitivity.transmit_sensitivity.into_base(),
            self.sensitivity.efficiency,
        )
    }
}
