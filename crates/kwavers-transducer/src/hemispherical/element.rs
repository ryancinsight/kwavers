//! Element configuration and state management

use aequitas::systems::si::quantities::{Angle, Dimensionless, Length};
use aequitas::systems::si::units::Radian;

/// Element configuration for hemispherical arrays
#[derive(Debug, Clone)]
pub struct ElementConfiguration {
    /// Position in 3D space (m)
    pub position: [Length<f64>; 3],
    /// Unit normal vector pointing from the element toward the geometric focus.
    pub normal: [f64; 3],
    /// Element radius (m)
    pub radius: Length<f64>,
    /// Phase offset (radians)
    pub phase_offset: Angle<f64>,
    /// Amplitude scaling factor
    pub amplitude: Dimensionless<f64>,
    /// Element state
    pub state: ElementState,
}

impl ElementConfiguration {
    /// Create new element
    #[must_use]
    pub fn new(position: [Length<f64>; 3], normal: [f64; 3], radius: Length<f64>) -> Self {
        Self {
            position,
            normal,
            radius,
            phase_offset: Angle::from_unit::<Radian>(0.0),
            amplitude: Dimensionless::from_base(1.0),
            state: ElementState::Active,
        }
    }

    /// Check if element is active
    #[must_use]
    pub fn is_active(&self) -> bool {
        matches!(self.state, ElementState::Active)
    }

    /// Set element phase
    pub fn set_phase(&mut self, phase: Angle<f64>) {
        self.phase_offset = phase;
    }

    /// Set element amplitude
    pub fn set_amplitude(&mut self, amplitude: Dimensionless<f64>) {
        let value = amplitude.into_base().clamp(0.0, 1.0);
        self.amplitude = Dimensionless::from_base(value);
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
