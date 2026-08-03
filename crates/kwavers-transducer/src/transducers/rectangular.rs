//! Rectangular transducer domain entity
//!
//! This module defines the physical properties of a rectangular transducer array,
//! decoupled from any specific solver implementation.

use aequitas::systems::si::quantities::{Frequency, Length, ReciprocalLength, Velocity};
use aequitas::systems::si::units::{Hertz, Meter, MeterPerSecond};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, KwaversResult};

/// Rectangular transducer description
#[derive(Debug, Clone)]
pub struct RectangularTransducer {
    /// Transducer width.
    pub width: Length<f64>,
    /// Transducer height.
    pub height: Length<f64>,
    /// Center frequency.
    pub frequency: Frequency<f64>,
    /// Number of elements (Nx, Ny)
    pub elements: (usize, usize),
}

impl RectangularTransducer {
    /// Get element size
    ///
    /// # Errors
    ///
    /// Returns [`KwaversError::InvalidInput`] when an element count or
    /// dimension is invalid.
    pub fn element_size(&self) -> KwaversResult<(Length<f64>, Length<f64>)> {
        let (nx, ny) = self.elements;
        if nx == 0 || ny == 0 {
            return Err(KwaversError::InvalidInput(
                "rectangular transducer element counts must be positive".into(),
            ));
        }

        let nx = u32::try_from(nx).map_err(|_| {
            KwaversError::InvalidInput("rectangular transducer x element count exceeds u32".into())
        })?;
        let ny = u32::try_from(ny).map_err(|_| {
            KwaversError::InvalidInput("rectangular transducer y element count exceeds u32".into())
        })?;

        let width = self.width.in_unit::<Meter>();
        let height = self.height.in_unit::<Meter>();
        if !width.is_finite() || width <= 0.0 || !height.is_finite() || height <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "rectangular transducer dimensions must be finite and positive".into(),
            ));
        }

        Ok((
            Length::from_unit::<Meter>(width / f64::from(nx)),
            Length::from_unit::<Meter>(height / f64::from(ny)),
        ))
    }

    /// Get wavenumber
    ///
    /// # Errors
    ///
    /// Returns [`KwaversError::InvalidInput`] when the frequency or sound
    /// speed is invalid.
    pub fn wavenumber(&self, c0: Velocity<f64>) -> KwaversResult<ReciprocalLength<f64>> {
        let frequency = self.frequency.in_unit::<Hertz>();
        let sound_speed = c0.in_unit::<MeterPerSecond>();
        if !frequency.is_finite() || frequency <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "rectangular transducer frequency must be finite and positive".into(),
            ));
        }
        if !sound_speed.is_finite() || sound_speed <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "rectangular transducer sound speed must be finite and positive".into(),
            ));
        }

        Ok((self.frequency / c0) * TWO_PI)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_geometry_and_wavenumber_preserve_si_values() {
        let transducer = RectangularTransducer {
            width: Length::from_unit::<Meter>(5.0e-3),
            height: Length::from_unit::<Meter>(3.0e-3),
            frequency: Frequency::from_unit::<Hertz>(2.0e6),
            elements: (10, 6),
        };

        let (width, height) = transducer.element_size().expect("valid geometry");
        assert_eq!(width.in_unit::<Meter>(), 5.0e-4);
        assert_eq!(height.in_unit::<Meter>(), 5.0e-4);

        let wavenumber = transducer
            .wavenumber(Velocity::from_unit::<MeterPerSecond>(1500.0))
            .expect("valid medium");
        assert!(
            (wavenumber.in_unit::<aequitas::systems::si::units::PerMeter>()
                - 2.0 * std::f64::consts::PI * 2.0e6 / 1500.0)
                .abs()
                < 1.0e-12
        );
    }

    #[test]
    fn invalid_geometry_and_medium_are_rejected() {
        let transducer = RectangularTransducer {
            width: Length::from_unit::<Meter>(5.0e-3),
            height: Length::from_unit::<Meter>(3.0e-3),
            frequency: Frequency::from_unit::<Hertz>(2.0e6),
            elements: (0, 6),
        };
        assert!(transducer.element_size().is_err());

        let transducer = RectangularTransducer {
            elements: (10, 6),
            ..transducer
        };
        assert!(transducer
            .wavenumber(Velocity::from_unit::<MeterPerSecond>(0.0))
            .is_err());
    }
}
