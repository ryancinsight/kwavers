//! Element configuration and state management

/// Element configuration for hemispherical arrays
#[derive(Debug, Clone)]
pub struct ElementConfiguration {
    /// Position in 3D space (m)
    pub position: [f64; 3],
    /// Normal vector (outward from hemisphere)
    pub normal: [f64; 3],
    /// Element radius (m)
    pub radius: f64,
    /// Phase offset (radians)
    pub phase_offset: f64,
    /// Amplitude scaling factor
    pub amplitude: f64,
    /// Element state
    pub state: ElementState,
}

impl ElementConfiguration {
    /// Create new element
    #[must_use]
    pub fn new(position: [f64; 3], normal: [f64; 3], radius: f64) -> Self {
        Self {
            position,
            normal,
            radius,
            phase_offset: 0.0,
            amplitude: 1.0,
            state: ElementState::Active,
        }
    }

    /// Check if element is active
    #[must_use]
    pub fn is_active(&self) -> bool {
        matches!(self.state, ElementState::Active)
    }

    /// Set element phase
    pub fn set_phase(&mut self, phase: f64) {
        self.phase_offset = phase;
    }

    /// Set element amplitude
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
