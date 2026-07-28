//! Beam steering and focusing control

use super::element::ElementConfiguration;
use aequitas::systems::si::quantities::{Frequency, Length, Pressure, Time, Velocity};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::constants::SOUND_SPEED_WATER_SIM;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_signal::Signal;
use leto::Array3;
use std::sync::Arc;

/// Focal point specification
#[derive(Debug, Clone, Copy)]
pub struct FocalPoint {
    /// Position in 3D space in SI base-unit metres.
    pub position: [Length<f64>; 3],
    /// Desired pressure amplitude at focus in pascals.
    pub amplitude: Pressure<f64>,
    /// Steering mode
    pub mode: SteeringMode,
}

/// Steering mode
#[derive(Debug, Clone, Copy)]
pub enum SteeringMode {
    /// Geometric focusing
    Geometric,
    /// Time reversal focusing
    PhotoacousticTimeReversal,
    /// Adaptive focusing
    Adaptive,
}

/// Steering controller for phased arrays
#[derive(Debug, Clone)]
pub struct SteeringController {
    /// Operating frequency in hertz.
    frequency: Frequency<f64>,
    /// Sound speed in metres per second.
    sound_speed: Velocity<f64>,
    /// Current focal point
    focal_point: Option<FocalPoint>,
}

impl SteeringController {
    /// Create new steering controller
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(frequency: Frequency<f64>) -> Self {
        Self {
            frequency,
            sound_speed: Velocity::from_base(SOUND_SPEED_WATER_SIM), // Water/tissue nominal
            focal_point: None,
        }
    }

    /// Set focal point and calculate delays
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn set_focus(
        &mut self,
        focal_point: FocalPoint,
        elements: &[ElementConfiguration],
    ) -> KwaversResult<()> {
        self.focal_point = Some(focal_point);

        // Calculate time delays for each element
        let wavelength = self.sound_speed.into_base() / self.frequency.into_base();

        for element in elements {
            let distance = calculate_distance(element.position, focal_point.position);
            let _phase_delay = TWO_PI * distance.into_base() / wavelength;
            // Phase would be set on mutable elements
        }

        Ok(())
    }

    /// Apply steering to field
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn apply_to_field(
        &self,
        field: &mut Array3<f64>,
        time: Time<f64>,
        grid: &Grid,
        signal: Arc<dyn Signal>,
        elements: &[ElementConfiguration],
    ) -> KwaversResult<()> {
        for element in elements {
            if !element.is_active() {
                continue;
            }

            // Apply element contribution with phase delay
            let phase = (TWO_PI * self.frequency.into_base())
                .mul_add(time.into_base(), element.phase_offset.into_base());
            let amplitude = element.amplitude * phase.sin();

            // Point source approximation: Adds field at discrete grid point
            // Full implementation: Spatial distribution via apodization function
            // Current: Adequate for hemispherical array geometric focusing
            let position = element.position.map(Length::into_base);
            if let Some((ix, iy, iz)) =
                grid.position_to_indices(position[0], position[1], position[2])
            {
                if ix < grid.nx && iy < grid.ny && iz < grid.nz {
                    field[[ix, iy, iz]] += amplitude * signal.amplitude(time.into_base());
                }
            }
        }

        Ok(())
    }
}

/// Calculate distance between two points
fn calculate_distance(p1: [Length<f64>; 3], p2: [Length<f64>; 3]) -> Length<f64> {
    let dx = p2[0].into_base() - p1[0].into_base();
    let dy = p2[1].into_base() - p1[1].into_base();
    let dz = p2[2].into_base() - p1[2].into_base();
    Length::from_base((dz.mul_add(dz, dy.mul_add(dy, dx.powi(2)))).sqrt())
}
