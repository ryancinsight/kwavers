//! Builder pattern for 2D Transducer Array
//!
//! Provides a fluent API for constructing and configuring 2D transducer arrays
//! with proper validation and sensible defaults.
//!
//! # Example
//!
//! ```rust
//! use kwavers::domain::source::array_2d::{TransducerArray2DBuilder, ApodizationType};
//!
//! let array = TransducerArray2DBuilder::new()
//!     .with_elements(64)
//!     .with_spacing(0.3e-3)
//!     .with_frequency(2.5e6)
//!     .with_focus(20e-3)
//!     .with_steering(15.0)
//!     .with_apodization(ApodizationType::Hanning)
//!     .build(1540.0)
//!     .unwrap();
//! ```

use super::{ApodizationType, TransducerArray2D, TransducerArray2DConfig};

/// Builder for 2D transducer arrays
///
/// Implements the builder pattern for ergonomic construction of transducer
/// arrays with validation at each step.
#[derive(Debug)]
pub struct TransducerArray2DBuilder {
    config: TransducerArray2DConfig,
    focus_distance: Option<f64>,
    elevation_focus: Option<f64>,
    steering_angle: f64,
    transmit_apodization: ApodizationType,
    receive_apodization: ApodizationType,
}

impl Default for TransducerArray2DBuilder {
    fn default() -> Self {
        Self {
            config: TransducerArray2DConfig::default(),
            focus_distance: None,
            elevation_focus: None,
            steering_angle: 0.0,
            transmit_apodization: ApodizationType::Rectangular,
            receive_apodization: ApodizationType::Rectangular,
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
    pub fn with_spacing(mut self, spacing: f64) -> Self {
        self.config.element_spacing = spacing;
        // Element width is typically slightly less than spacing
        self.config.element_width = spacing * 0.9;
        self
    }

    /// Set element width (independent of spacing)
    #[must_use]
    pub fn with_element_width(mut self, width: f64) -> Self {
        self.config.element_width = width;
        self
    }

    /// Set element length (elevation dimension)
    #[must_use]
    pub fn with_element_length(mut self, length: f64) -> Self {
        self.config.element_length = length;
        self
    }

    /// Set operating frequency
    #[must_use]
    pub fn with_frequency(mut self, frequency: f64) -> Self {
        // Adjust spacing based on wavelength if not explicitly set
        let wavelength = 1540.0 / frequency; // Using default sound speed
        let optimal_spacing = wavelength / 2.0;

        // Only update if current spacing is larger than optimal
        if self.config.element_spacing > optimal_spacing {
            self.config.element_spacing = optimal_spacing;
            self.config.element_width = optimal_spacing * 0.9;
        }

        self
    }

    /// Set center position
    #[must_use]
    pub fn at_position(mut self, x: f64, y: f64, z: f64) -> Self {
        self.config.center_position = (x, y, z);
        self
    }

    /// Set radius of curvature (for curved arrays)
    #[must_use]
    pub fn with_radius(mut self, radius: f64) -> Self {
        self.config.radius = radius;
        self
    }

    /// Set focus distance
    #[must_use]
    pub fn with_focus(mut self, distance: f64) -> Self {
        self.focus_distance = Some(distance);
        self
    }

    /// Set elevation focus distance
    #[must_use]
    pub fn with_elevation_focus(mut self, distance: f64) -> Self {
        self.elevation_focus = Some(distance);
        self
    }

    /// Set steering angle [degrees]
    #[must_use]
    pub fn with_steering(mut self, angle: f64) -> Self {
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
    /// * `sound_speed` - Speed of sound in medium [m/s]
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
    pub fn build(self, sound_speed: f64) -> Result<TransducerArray2D, String> {
        // Calculate frequency from wavelength and spacing
        let frequency = if self.config.element_spacing > 0.0 {
            let wavelength = 2.0 * self.config.element_spacing; // Nyquist sampling
            sound_speed / wavelength
        } else {
            1e6 // Default 1 MHz
        };

        let mut array = TransducerArray2D::new(self.config, sound_speed, frequency)?;

        // Apply beamforming settings
        if let Some(focus) = self.focus_distance {
            array.set_focus_distance(focus);
        }

        if let Some(elev_focus) = self.elevation_focus {
            array.set_elevation_focus_distance(elev_focus);
        }

        if self.steering_angle != 0.0 {
            array.set_steering_angle(self.steering_angle);
        }

        array.set_transmit_apodization(self.transmit_apodization);
        array.set_receive_apodization(self.receive_apodization);

        Ok(array)
    }

    /// Build with explicit frequency
    ///
    /// Use this when you need precise frequency control
    pub fn build_with_frequency(
        self,
        sound_speed: f64,
        frequency: f64,
    ) -> Result<TransducerArray2D, String> {
        let mut array = TransducerArray2D::new(self.config, sound_speed, frequency)?;

        if let Some(focus) = self.focus_distance {
            array.set_focus_distance(focus);
        }

        if let Some(elev_focus) = self.elevation_focus {
            array.set_elevation_focus_distance(elev_focus);
        }

        if self.steering_angle != 0.0 {
            array.set_steering_angle(self.steering_angle);
        }

        array.set_transmit_apodization(self.transmit_apodization);
        array.set_receive_apodization(self.receive_apodization);

        Ok(array)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_basic() {
        let array = TransducerArray2DBuilder::new()
            .with_elements(32)
            .with_spacing(0.3e-3)
            .build(1540.0)
            .unwrap();

        assert_eq!(array.num_elements(), 32);
    }

    #[test]
    fn test_builder_with_focus() {
        let array = TransducerArray2DBuilder::new()
            .with_elements(32)
            .with_spacing(0.3e-3)
            .with_focus(20e-3)
            .with_steering(10.0)
            .build(1540.0)
            .unwrap();

        assert!((array.focus_distance() - 20e-3).abs() < 1e-10);
        assert!((array.steering_angle() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_builder_frequency_optimization() {
        let array = TransducerArray2DBuilder::new()
            .with_elements(32)
            .with_frequency(2.5e6) // This should adjust spacing
            .build(1540.0)
            .unwrap();

        // Should satisfy Nyquist criterion
        assert!(array.satisfies_nyquist());
    }
}
