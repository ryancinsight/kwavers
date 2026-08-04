//! Beam steering and focusing control

use super::element::ElementConfiguration;
use aequitas::systems::si::quantities::{Frequency, Pressure, Velocity};
use aequitas::systems::si::units::{Hertz, MeterPerSecond, Radian};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::constants::SOUND_SPEED_WATER_SIM;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_signal::Signal;
use leto::Array3;
use std::sync::Arc;

/// Focal point specification.
#[derive(Debug, Clone, Copy)]
pub struct FocalPoint {
    /// Target position in mesh coordinates `[x, y, z]` in metres.
    ///
    /// Kept as raw `[f64; 3]` — passed directly to the mesh layer as spatial coordinates.
    pub position: [f64; 3],
    /// Desired acoustic pressure magnitude at the focus.
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

/// Steering controller for phased arrays.
#[derive(Debug, Clone)]
pub struct SteeringController {
    /// Operating frequency.
    frequency: Frequency<f64>,
    /// Acoustic wave speed.
    sound_speed: Velocity<f64>,
    /// Current focal point
    focal_point: Option<FocalPoint>,
}

impl SteeringController {
    /// Create a new steering controller with the given operating frequency.
    ///
    /// Sound speed defaults to the water/tissue nominal value.
    #[must_use]
    pub fn new(frequency: Frequency<f64>) -> Self {
        Self {
            frequency,
            sound_speed: Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_WATER_SIM),
            focal_point: None,
        }
    }

    /// Set focal point and calculate element delays.
    ///
    /// # Errors
    /// Returns an error if an internal constraint is violated.
    pub fn set_focus(
        &mut self,
        focal_point: FocalPoint,
        elements: &[ElementConfiguration],
    ) -> KwaversResult<()> {
        self.focal_point = Some(focal_point);

        // Scalar extraction at formula boundary: wavelength = c / f (result in metres).
        let c = self.sound_speed.in_unit::<MeterPerSecond>();
        let f = self.frequency.in_unit::<Hertz>();
        let wavelength = c / f;

        for element in elements {
            let distance = calculate_distance(element.position, focal_point.position);
            let _phase_delay = TWO_PI * distance / wavelength;
            // Phase would be set on mutable elements
        }

        Ok(())
    }

    /// Apply steering to field.
    ///
    /// # Errors
    /// Returns an error if an internal constraint is violated.
    pub fn apply_to_field(
        &self,
        field: &mut Array3<f64>,
        time: f64,
        grid: &Grid,
        signal: Arc<dyn Signal>,
        elements: &[ElementConfiguration],
    ) -> KwaversResult<()> {
        // Scalar extraction at formula boundary.
        let freq = self.frequency.in_unit::<Hertz>();
        for element in elements {
            if !element.is_active() {
                continue;
            }

            // Apply element contribution with phase delay.
            // `phase_offset` is typed Angle<f64>; extract radians at the formula boundary.
            let phase_rad = element.phase_offset.in_unit::<Radian>();
            let phase = (TWO_PI * freq).mul_add(time, phase_rad);
            let amplitude = element.amplitude * phase.sin();

            // Point source approximation: adds field at the discrete grid point.
            if let Some((ix, iy, iz)) = grid.position_to_indices(
                element.position[0],
                element.position[1],
                element.position[2],
            ) {
                if ix < grid.nx && iy < grid.ny && iz < grid.nz {
                    field[[ix, iy, iz]] += amplitude * signal.amplitude(time);
                }
            }
        }

        Ok(())
    }
}

/// Calculate Euclidean distance between two mesh-coordinate points.
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
    use aequitas::systems::si::units::{Hertz, Pascal};

    #[test]
    fn steering_controller_default_sound_speed_is_water() {
        let ctrl = SteeringController::new(Frequency::from_unit::<Hertz>(650_000.0));
        assert!(ctrl.sound_speed.in_unit::<MeterPerSecond>() > 1400.0);
    }

    #[test]
    fn focal_point_pressure_round_trips() {
        let fp = FocalPoint {
            position: [0.0, 0.0, 0.05],
            amplitude: Pressure::from_unit::<Pascal>(1_000_000.0), // 1 MPa
            mode: SteeringMode::Geometric,
        };
        assert!((fp.amplitude.in_unit::<Pascal>() - 1.0e6).abs() < 1.0);
    }
}
