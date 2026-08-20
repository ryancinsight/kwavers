//! Builder pattern for 2D Transducer Array
//!
//! Provides a fluent API for constructing and configuring 2D transducer arrays
//! with proper validation and sensible defaults.
//!
//! # Example
//!
//! ```rust
//! use aequitas::systems::si::quantities::{Angle, Frequency, Length, Velocity};
//! use aequitas::systems::si::units::{Degree, Hertz, Meter, MeterPerSecond};
//! use kwavers_transducer::array_2d::{TransducerArray2DBuilder, ApodizationType};
//!
//! let array = TransducerArray2DBuilder::new()
//!     .with_elements(64)
//!     .with_spacing(Length::from_unit::<Meter>(0.3e-3))
//!     .with_frequency(Frequency::from_unit::<Hertz>(2.5e6))
//!     .with_focus(Length::from_unit::<Meter>(20e-3))
//!     .with_steering(Angle::from_unit::<Degree>(15.0))
//!     .with_apodization(ApodizationType::Hanning)
//!     .build(Velocity::from_unit::<MeterPerSecond>(1540.0))
//!     .unwrap();
//! ```

use super::{ApodizationType, ArrayCurvature, TransducerArray2D, TransducerArray2DConfig};
use aequitas::systems::si::quantities::{Angle, Frequency, Length, Velocity};
use aequitas::systems::si::units::{Hertz, Meter, MeterPerSecond, Radian};
use kwavers_core::constants::numerical::MHZ_TO_HZ;

/// Builder for 2D transducer arrays
///
/// Implements the builder pattern for ergonomic construction of transducer
/// arrays with validation at each step.
#[derive(Debug)]
pub struct TransducerArray2DBuilder {
    config: TransducerArray2DConfig,
    frequency: Option<Frequency<f64>>,
    focus_distance: Option<Length<f64>>,
    elevation_focus: Option<Length<f64>>,
    steering_angle: Angle<f64>,
    transmit_apodization: ApodizationType,
    receive_apodization: ApodizationType,
}

impl Default for TransducerArray2DBuilder {
    fn default() -> Self {
        Self {
            config: TransducerArray2DConfig::default(),
            frequency: None,
            focus_distance: None,
            elevation_focus: None,
            steering_angle: Angle::from_unit::<Radian>(0.0),
            transmit_apodization: ApodizationType::Uniform,
            receive_apodization: ApodizationType::Uniform,
        }
    }
}

impl TransducerArray2DBuilder {
    /// Create a new builder with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of elements
    #[must_use]
    pub fn with_elements(mut self, count: usize) -> Self {
        self.config.number_elements = count;
        self
    }

    /// Set element spacing
    #[must_use]
    pub fn with_spacing(mut self, spacing: Length<f64>) -> Self {
        self.config.element_spacing = spacing;
        // Element width is typically slightly less than spacing
        self.config.element_width = Length::from_unit::<Meter>(spacing.in_unit::<Meter>() * 0.9);
        self
    }

    /// Set element width (independent of spacing)
    #[must_use]
    pub fn with_element_width(mut self, width: Length<f64>) -> Self {
        self.config.element_width = width;
        self
    }

    /// Set element length (elevation dimension)
    #[must_use]
    pub fn with_element_length(mut self, length: Length<f64>) -> Self {
        self.config.element_length = length;
        self
    }

    /// Set operating frequency
    #[must_use]
    pub fn with_frequency(mut self, frequency: Frequency<f64>) -> Self {
        self.frequency = Some(frequency);
        self
    }

    /// Set center position
    #[must_use]
    pub fn at_position(mut self, x: Length<f64>, y: Length<f64>, z: Length<f64>) -> Self {
        self.config.center_position = [x, y, z];
        self
    }

    /// Set the surface curvature.
    #[must_use]
    pub fn with_curvature(mut self, curvature: ArrayCurvature) -> Self {
        self.config.curvature = curvature;
        self
    }

    /// Set a cylindrical radius of curvature.
    #[must_use]
    pub fn with_radius(self, radius: Length<f64>) -> Self {
        self.with_curvature(ArrayCurvature::Cylindrical { radius })
    }

    /// Set focus distance.
    #[must_use]
    pub fn with_focus(mut self, distance: Length<f64>) -> Self {
        self.focus_distance = Some(distance);
        self
    }

    /// Set elevation focus distance
    #[must_use]
    pub fn with_elevation_focus(mut self, distance: Length<f64>) -> Self {
        self.elevation_focus = Some(distance);
        self
    }

    /// Set steering angle in radians.
    #[must_use]
    pub fn with_steering(mut self, angle: Angle<f64>) -> Self {
        self.steering_angle = angle;
        self
    }

    /// Set transmit apodization
    #[must_use]
    pub fn with_apodization(mut self, apodization: ApodizationType) -> Self {
        self.transmit_apodization = apodization;
        self.receive_apodization = apodization;
        self
    }

    /// Set separate transmit and receive apodization
    #[must_use]
    pub fn with_apodization_separate(
        mut self,
        transmit: ApodizationType,
        receive: ApodizationType,
    ) -> Self {
        self.transmit_apodization = transmit;
        self.receive_apodization = receive;
        self
    }

    /// Build the transducer array
    ///
    /// # Arguments
    ///
    /// * `sound_speed` - Speed of sound in the medium
    ///
    /// # Returns
    ///
    /// Result containing the configured array or validation error
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Configuration is invalid
    /// - Sound speed is not positive
    pub fn build(self, sound_speed: Velocity<f64>) -> Result<TransducerArray2D, String> {
        let frequency = self.frequency.unwrap_or_else(|| {
            let spacing_m = self.config.element_spacing.in_unit::<Meter>();
            let sound_speed_m_s = sound_speed.in_unit::<MeterPerSecond>();
            if spacing_m > 0.0 {
                Frequency::from_unit::<Hertz>(sound_speed_m_s / (2.0 * spacing_m))
            } else {
                Frequency::from_unit::<Hertz>(MHZ_TO_HZ)
            }
        });
        let config = Self::nyquist_adjusted_config(self.config, sound_speed, frequency);

        let mut array = TransducerArray2D::new(config, sound_speed, frequency)?;

        // Apply beamforming settings
        if let Some(focus) = self.focus_distance {
            array.set_focus_distance(focus)?;
        }

        if let Some(elev_focus) = self.elevation_focus {
            array.set_elevation_focus_distance(elev_focus)?;
        }

        if self.steering_angle.in_unit::<Radian>() != 0.0 {
            array.set_steering_angle(self.steering_angle)?;
        }

        array.set_transmit_apodization(self.transmit_apodization);
        array.set_receive_apodization(self.receive_apodization);

        Ok(array)
    }

    /// Build with explicit frequency
    ///
    /// Use this when you need precise frequency control
    /// # Errors
    /// - Propagates any [`kwavers_core::error::KwaversError`] returned by called functions.
    ///
    pub fn build_with_frequency(
        self,
        sound_speed: Velocity<f64>,
        frequency: Frequency<f64>,
    ) -> Result<TransducerArray2D, String> {
        let config = Self::nyquist_adjusted_config(self.config, sound_speed, frequency);
        let mut array = TransducerArray2D::new(config, sound_speed, frequency)?;

        if let Some(focus) = self.focus_distance {
            array.set_focus_distance(focus)?;
        }

        if let Some(elev_focus) = self.elevation_focus {
            array.set_elevation_focus_distance(elev_focus)?;
        }

        if self.steering_angle.in_unit::<Radian>() != 0.0 {
            array.set_steering_angle(self.steering_angle)?;
        }

        array.set_transmit_apodization(self.transmit_apodization);
        array.set_receive_apodization(self.receive_apodization);

        Ok(array)
    }

    fn nyquist_adjusted_config(
        mut config: TransducerArray2DConfig,
        sound_speed: Velocity<f64>,
        frequency: Frequency<f64>,
    ) -> TransducerArray2DConfig {
        let spacing_m = config.element_spacing.in_unit::<Meter>();
        let max_spacing_m =
            sound_speed.in_unit::<MeterPerSecond>() / (2.0 * frequency.in_unit::<Hertz>());
        if spacing_m > max_spacing_m {
            config.element_spacing = Length::from_unit::<Meter>(max_spacing_m);
            config.element_width = Length::from_unit::<Meter>(0.9 * max_spacing_m);
        }
        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aequitas::systems::si::quantities::{Angle, Frequency, Length, Velocity};
    use aequitas::systems::si::units::{Degree, Hertz, Meter, MeterPerSecond};
    use kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE;

    #[test]
    fn test_builder_basic() {
        let array = TransducerArray2DBuilder::new()
            .with_elements(32)
            .with_spacing(Length::from_unit::<Meter>(0.3e-3))
            .build(Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_TISSUE))
            .unwrap();

        assert_eq!(array.num_elements(), 32);
    }

    #[test]
    fn test_builder_with_focus() {
        let array = TransducerArray2DBuilder::new()
            .with_elements(32)
            .with_spacing(Length::from_unit::<Meter>(0.3e-3))
            .with_focus(Length::from_unit::<Meter>(20e-3))
            .with_steering(Angle::from_unit::<Degree>(10.0))
            .build(Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_TISSUE))
            .unwrap();

        assert!((array.focus_distance().unwrap().in_unit::<Meter>() - 20e-3).abs() < 1e-10);
        assert!((array.steering_angle().in_unit::<Degree>() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_builder_frequency_optimization() {
        let array = TransducerArray2DBuilder::new()
            .with_elements(32)
            .with_frequency(Frequency::from_unit::<Hertz>(2.5e6))
            .build(Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_TISSUE))
            .unwrap();

        // Should satisfy Nyquist criterion
        assert!(array.satisfies_nyquist());
    }
}
