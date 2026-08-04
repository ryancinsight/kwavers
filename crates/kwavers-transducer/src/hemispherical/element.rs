//! Element configuration and state management

use aequitas::systems::si::quantities::{Angle, Length};
use aequitas::systems::si::units::Radian;

/// Element configuration for hemispherical arrays
#[derive(Debug, Clone)]
pub struct ElementConfiguration {
    /// Position in 3D space (mesh coordinates, m).
    ///
    /// Kept as raw `[f64; 3]` because it is passed directly to the mesh layer
    /// (`Grid::position_to_indices`) as a spatial coordinate — a formula/mesh boundary.
    pub position: [f64; 3],
    /// Unit normal vector pointing from the element toward the geometric focus.
    pub normal: [f64; 3],
    /// Element radius.
    pub radius: Length<f64>,
    /// Phase offset.
    pub phase_offset: Angle<f64>,
    /// Dimensionless amplitude scaling factor in `[0, 1]`.
    pub amplitude: f64,
    /// Element state
    pub state: ElementState,
}

impl ElementConfiguration {
    /// Create a new element with the given mesh position, normal, and physical radius.
    #[must_use]
    pub fn new(position: [f64; 3], normal: [f64; 3], radius: Length<f64>) -> Self {
        Self {
            position,
            normal,
            radius,
            phase_offset: Angle::from_unit::<Radian>(0.0),
            amplitude: 1.0,
            state: ElementState::Active,
        }
    }

    /// Check if element is active
    #[must_use]
    pub fn is_active(&self) -> bool {
        matches!(self.state, ElementState::Active)
    }

    /// Set element phase offset.
    pub fn set_phase(&mut self, phase: Angle<f64>) {
        self.phase_offset = phase;
    }

    /// Set element amplitude (clamped to `[0, 1]`).
    pub fn set_amplitude(&mut self, amplitude: f64) {
        self.amplitude = amplitude.clamp(0.0, 1.0);
    }
}

/// Element operational state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ElementState {
    /// Element is active and transmitting
    Active,
    /// Element is disabled
    Disabled,
    /// Element failed diagnostics
    Failed,
    /// Element is in sparse mode (selectively active)
    Sparse,
}
