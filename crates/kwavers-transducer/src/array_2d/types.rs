//! Types for 2D transducer array: `ApodizationType`, `TransducerArray2DConfig`, `Array2dElement`.

use aequitas::systems::si::quantities::{Frequency, Length, Time, Velocity};
use aequitas::systems::si::units::{Hertz, Meter, MeterPerSecond};
pub use kwavers_math::ApodizationType;
use kwavers_source::{
    Apodization, BlackmanApodization, GaussianApodization, HammingApodization, HanningApodization,
    RectangularApodization, TukeyApodization,
};

/// Create an apodization implementation from an `ApodizationType`.
pub(super) fn create_apodization(apo: &ApodizationType) -> Box<dyn Apodization> {
    match apo {
        ApodizationType::Uniform => Box::new(RectangularApodization),
        ApodizationType::Hanning => Box::new(HanningApodization),
        ApodizationType::Hamming => Box::new(HammingApodization),
        ApodizationType::Blackman => Box::new(BlackmanApodization),
        ApodizationType::Tukey { r } => Box::new(TukeyApodization::new(*r)),
        ApodizationType::Gaussian { sigma } => Box::new(GaussianApodization::new(*sigma)),
        ApodizationType::Kaiser { .. } => Box::new(HammingApodization),
    }
}

/// Surface geometry of a two-dimensional array.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ArrayCurvature {
    /// Planar array surface.
    Flat,
    /// Cylindrical curvature across the azimuthal element axis.
    Cylindrical { radius: Length<f64> },
}

impl ArrayCurvature {
    pub(super) fn validate(self) -> Result<(), String> {
        match self {
            Self::Flat => Ok(()),
            Self::Cylindrical { radius } => {
                let radius_m = radius.in_unit::<Meter>();
                if radius_m.is_finite() && radius_m > 0.0 {
                    Ok(())
                } else {
                    Err(format!(
                        "Cylindrical curvature radius must be finite and positive, got {radius_m}"
                    ))
                }
            }
        }
    }

    /// Return the finite radius for cylindrical geometry.
    #[must_use]
    pub fn radius(self) -> Option<Length<f64>> {
        match self {
            Self::Flat => None,
            Self::Cylindrical { radius } => Some(radius),
        }
    }
}

/// Configuration for a 2D transducer array
#[derive(Debug, Clone)]
pub struct TransducerArray2DConfig {
    /// Number of elements in the array
    pub number_elements: usize,
    /// Width of each element.
    pub element_width: Length<f64>,
    /// Length of each element in the elevation direction.
    pub element_length: Length<f64>,
    /// Center-to-center spacing between adjacent elements.
    pub element_spacing: Length<f64>,
    /// Surface geometry across the azimuthal element axis.
    pub curvature: ArrayCurvature,
    /// Center position of the array `(x, y, z)`.
    pub center_position: [Length<f64>; 3],
}

impl Default for TransducerArray2DConfig {
    fn default() -> Self {
        Self {
            number_elements: 32,
            element_width: Length::from_unit::<Meter>(0.3e-3),
            element_length: Length::from_unit::<Meter>(10e-3),
            element_spacing: Length::from_unit::<Meter>(0.5e-3),
            curvature: ArrayCurvature::Flat,
            center_position: [Length::from_unit::<Meter>(0.0); 3],
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
        let width_m = self.element_width.in_unit::<Meter>();
        let length_m = self.element_length.in_unit::<Meter>();
        let spacing_m = self.element_spacing.in_unit::<Meter>();
        if !width_m.is_finite() || width_m <= 0.0 {
            return Err(format!(
                "Element width must be finite and positive, got {width_m}"
            ));
        }
        if !length_m.is_finite() || length_m <= 0.0 {
            return Err(format!(
                "Element length must be finite and positive, got {length_m}"
            ));
        }
        if !spacing_m.is_finite() || spacing_m < width_m {
            return Err(format!(
                "Element spacing must be finite and >= element width, got {spacing_m}"
            ));
        }
        if self
            .center_position
            .iter()
            .any(|coordinate| !coordinate.in_unit::<Meter>().is_finite())
        {
            return Err("Center position coordinates must be finite".to_owned());
        }
        self.curvature.validate()
    }

    /// Calculate total array aperture width
    #[must_use]
    pub fn aperture_width(&self) -> Length<f64> {
        let width_m = self.element_width.in_unit::<Meter>();
        let spacing_m = self.element_spacing.in_unit::<Meter>();
        Length::from_unit::<Meter>((self.number_elements - 1) as f64 * spacing_m + width_m)
    }

    /// Check if element spacing satisfies Nyquist criterion
    #[must_use]
    pub fn satisfies_nyquist(&self, sound_speed: Velocity<f64>, frequency: Frequency<f64>) -> bool {
        let sound_speed_m_s = sound_speed.in_unit::<MeterPerSecond>();
        let frequency_hz = frequency.in_unit::<Hertz>();
        let wavelength_m = sound_speed_m_s / frequency_hz;
        self.element_spacing.in_unit::<Meter>() <= wavelength_m / 2.0
    }
}

/// Individual transducer element in 2D array
#[derive(Debug, Clone)]
pub struct Array2dElement {
    /// Element position `(x, y, z)`.
    pub position: [Length<f64>; 3],
    /// Element width.
    pub width: Length<f64>,
    /// Element length.
    pub length: Length<f64>,
    /// Time delay for beamforming.
    pub time_delay: Time<f64>,
    /// Transmit apodization weight [0.0-1.0]
    pub transmit_weight: f64,
    /// Receive apodization weight [0.0-1.0]
    pub receive_weight: f64,
    /// Whether element is active
    pub is_active: bool,
}
