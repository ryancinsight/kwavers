//! Beam steering and focusing control

use super::element::ElementConfiguration;
use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::signal::Signal;
use ndarray::Array3;
use std::f64::consts::PI;
use std::sync::Arc;

/// Focal point specification
#[derive(Debug, Clone, Copy)]
pub struct FocalPoint {
    /// Position in 3D space (m)
    pub position: [f64; 3],
    /// Desired pressure amplitude at focus
    pub amplitude: f64,
    /// Steering mode
    pub mode: SteeringMode,
}

/// Steering mode
#[derive(Debug, Clone, Copy)]
pub enum SteeringMode {
    /// Geometric focusing
    Geometric,
    /// Time reversal focusing
    TimeReversal,
    /// Adaptive focusing
    Adaptive,
}

/// Steering controller for phased arrays
#[derive(Debug, Clone)]
pub struct SteeringController {
    /// Operating frequency (Hz)
    frequency: f64,
    /// Sound speed (m/s)
    sound_speed: f64,
    /// Current focal point
    focal_point: Option<FocalPoint>,
}

impl SteeringController {
    /// Create new steering controller
    #[must_use]
    pub fn new(frequency: f64) -> Self {
        Self {
            frequency,
            sound_speed: 1500.0, // Water/tissue
            focal_point: None,
        }
    }

    /// Set focal point and calculate delays
    pub fn set_focus(
        &mut self,
        focal_point: FocalPoint,
        elements: &[ElementConfiguration],
    ) -> KwaversResult<()> {
        self.focal_point = Some(focal_point);

        // Calculate time delays for each element
        let wavelength = self.sound_speed / self.frequency;

        for element in elements {
            let distance = calculate_distance(element.position, focal_point.position);
            let phase_delay = 2.0 * PI * distance / wavelength;
            // Phase would be set on mutable elements
        }

        Ok(())
    }

    /// Apply steering to field
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
            let phase = 2.0 * PI * self.frequency * time + element.phase_offset;
            let amplitude = element.amplitude * phase.sin();

            // Add to field at element position
            // This is simplified - actual implementation would use proper spatial distribution
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

/// Calculate distance between two points
fn calculate_distance(p1: [f64; 3], p2: [f64; 3]) -> f64 {
    ((p2[0] - p1[0]).powi(2) + (p2[1] - p1[1]).powi(2) + (p2[2] - p1[2]).powi(2)).sqrt()
}
