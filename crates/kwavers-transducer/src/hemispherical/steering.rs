//! Beam steering and focusing control

use super::element::ElementConfiguration;
use aequitas::systems::si::quantities::{Angle, Frequency, Length, Pressure, Velocity};
use aequitas::systems::si::units::{Hertz, Meter, MeterPerSecond, Radian};
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
    /// Position in 3D space (m)
    pub position: [Length<f64>; 3],
    /// Desired pressure amplitude at focus
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
    /// Operating frequency (Hz)
    frequency: Frequency<f64>,
    /// Sound speed (m/s)
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
            sound_speed: Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_WATER_SIM),
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
        elements: &mut [ElementConfiguration],
    ) -> KwaversResult<()> {
        self.focal_point = Some(focal_point);

        // Calculate time delays for each element
        let wavelength =
            self.sound_speed.in_unit::<MeterPerSecond>() / self.frequency.in_unit::<Hertz>();

        let focal_point_m = focal_point.position.map(|value| value.in_unit::<Meter>());
        for element in elements {
            let element_position = element.position.map(|value| value.in_unit::<Meter>());
            let distance = calculate_distance(element_position, focal_point_m);
            let phase_delay = Angle::from_unit::<Radian>(TWO_PI * distance / wavelength);
            element.set_phase(phase_delay);
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
        time: f64,
        grid: &Grid,
        signal: Arc<dyn Signal>,
        elements: &[ElementConfiguration],
    ) -> KwaversResult<()> {
        for element in elements {
            if !element.is_active() {
                continue;
            }

            // Apply element contribution with phase delay
            let phase = (TWO_PI * self.frequency.in_unit::<Hertz>())
                .mul_add(time, element.phase_offset.in_unit::<Radian>());
            let amplitude = element.amplitude.into_base() * phase.sin();

            // Point source approximation: Adds field at discrete grid point
            // Full implementation: Spatial distribution via apodization function
            // Current: Adequate for hemispherical array geometric focusing
            if let Some((ix, iy, iz)) = grid.position_to_indices(
                element.position[0].in_unit::<Meter>(),
                element.position[1].in_unit::<Meter>(),
                element.position[2].in_unit::<Meter>(),
            ) {
                if ix < grid.nx && iy < grid.ny && iz < grid.nz {
                    field[[ix, iy, iz]] += amplitude * signal.amplitude(time);
                }
            }
        }

        Ok(())
    }
}

/// Calculate distance between two points
fn calculate_distance(p1: [f64; 3], p2: [f64; 3]) -> f64 {
    (p2[2] - p1[2])
        .mul_add(
            p2[2] - p1[2],
            (p2[1] - p1[1]).mul_add(p2[1] - p1[1], (p2[0] - p1[0]).powi(2)),
        )
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use aequitas::systems::si::units::Pascal;

    #[test]
    fn steering_controller_default_sound_speed_is_water() {
        let controller = SteeringController::new(Frequency::from_unit::<Hertz>(650_000.0));
        assert!(controller.sound_speed.in_unit::<MeterPerSecond>() > 1400.0);
    }

    #[test]
    fn focal_point_pressure_round_trips() {
        let focal_point = FocalPoint {
            position: [
                Length::from_unit::<Meter>(0.0),
                Length::from_unit::<Meter>(0.0),
                Length::from_unit::<Meter>(0.05),
            ],
            amplitude: Pressure::from_unit::<Pascal>(1_000_000.0),
            mode: SteeringMode::Geometric,
        };
        assert!((focal_point.amplitude.in_unit::<Pascal>() - 1.0e6).abs() < 1.0);
    }
}
