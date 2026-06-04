//! Types for 2D transducer array: `ApodizationType`, `TransducerArray2DConfig`, `Array2dElement`.

pub use kwavers_math::signal::ApodizationType;
use kwavers_source::{
    Apodization, BlackmanApodization, GaussianApodization, HammingApodization, HanningApodization,
    RectangularApodization,
};

/// Create an apodization implementation from an `ApodizationType`.
pub(super) fn create_apodization(apo: &ApodizationType) -> Box<dyn Apodization> {
    match apo {
        ApodizationType::Uniform => Box::new(RectangularApodization),
        ApodizationType::Hanning => Box::new(HanningApodization),
        ApodizationType::Hamming => Box::new(HammingApodization),
        ApodizationType::Blackman => Box::new(BlackmanApodization),
        ApodizationType::Gaussian { sigma } => Box::new(GaussianApodization::new(*sigma)),
        ApodizationType::Kaiser { .. } => Box::new(HammingApodization),
    }
}

/// Configuration for a 2D transducer array
#[derive(Debug, Clone)]
pub struct TransducerArray2DConfig {
    /// Number of elements in the array
    pub number_elements: usize,
    /// Width of each element (m)
    pub element_width: f64,
    /// Length of each element (elevation direction) (m)
    pub element_length: f64,
    /// Spacing between element centers (m)
    pub element_spacing: f64,
    /// Radius of curvature (m) (INF for flat array)
    pub radius: f64,
    /// Center position of the array (x, y, z) (m)
    pub center_position: (f64, f64, f64),
}

impl Default for TransducerArray2DConfig {
    fn default() -> Self {
        Self {
            number_elements: 32,
            element_width: 0.3e-3,
            element_length: 10e-3,
            element_spacing: 0.5e-3,
            radius: f64::INFINITY,
            center_position: (0.0, 0.0, 0.0),
        }
    }
}

impl TransducerArray2DConfig {
    /// Validate configuration parameters
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn validate(&self) -> Result<(), String> {
        if self.number_elements == 0 {
            return Err("Number of elements must be positive".to_owned());
        }
        if self.element_width <= 0.0 {
            return Err("Element width must be positive".to_owned());
        }
        if self.element_length <= 0.0 {
            return Err("Element length must be positive".to_owned());
        }
        if self.element_spacing < self.element_width {
            return Err("Element spacing must be >= element width".to_owned());
        }
        if self.radius <= 0.0 && !self.radius.is_infinite() {
            return Err("Radius must be positive or infinite".to_owned());
        }
        Ok(())
    }

    /// Calculate total array aperture width
    #[must_use]
    pub fn aperture_width(&self) -> f64 {
        ((self.number_elements - 1) as f64).mul_add(self.element_spacing, self.element_width)
    }

    /// Check if element spacing satisfies Nyquist criterion
    #[must_use]
    pub fn satisfies_nyquist(&self, sound_speed: f64, frequency: f64) -> bool {
        let wavelength = sound_speed / frequency;
        self.element_spacing <= wavelength / 2.0
    }
}

/// Individual transducer element in 2D array
#[derive(Debug, Clone)]
pub struct Array2dElement {
    /// Element position (x, y, z) (m)
    pub position: (f64, f64, f64),
    /// Element width (m)
    pub width: f64,
    /// Element length (m)
    pub length: f64,
    /// Time delay for beamforming (s)
    pub time_delay: f64,
    /// Transmit apodization weight [0.0-1.0]
    pub transmit_weight: f64,
    /// Receive apodization weight [0.0-1.0]
    pub receive_weight: f64,
    /// Whether element is active
    pub is_active: bool,
}
