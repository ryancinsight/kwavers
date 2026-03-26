//! Incident acoustic field types and cloud state snapshots

use super::config::{CloudBubble, CloudConfig};

/// Incident acoustic field
#[derive(Debug, Clone)]
pub struct IncidentField {
    /// Pressure amplitude (Pa)
    pub pressure_amplitude: f64,
    /// Frequency (Hz)
    pub frequency: f64,
    /// Sound speed (m/s)
    pub sound_speed: f64,
    /// Propagation direction (unit vector)
    pub direction: [f64; 3],
    /// Phase offset (rad)
    pub phase_offset: f64,
}

impl IncidentField {
    /// Create plane wave field
    pub fn plane_wave(pressure: f64, frequency: f64, direction: [f64; 3]) -> Self {
        Self {
            pressure_amplitude: pressure,
            frequency,
            sound_speed: 1500.0,
            direction,
            phase_offset: 0.0,
        }
    }

    /// Calculate pressure at position and time
    ///
    /// Plane wave: p(x,t) = p₀ cos(k·x − ωt + φ)
    pub fn pressure_at(&self, position: [f64; 3], time: f64) -> f64 {
        let kx = (2.0 * std::f64::consts::PI * self.frequency / self.sound_speed)
            * (position[0] * self.direction[0]
                + position[1] * self.direction[1]
                + position[2] * self.direction[2]);

        let omega_t = 2.0 * std::f64::consts::PI * self.frequency * time;

        self.pressure_amplitude * (kx - omega_t + self.phase_offset).cos()
    }
}

/// Snapshot of cloud state at a time step
#[derive(Debug, Clone)]
pub struct CloudState {
    /// Active bubbles in the cloud
    pub bubbles: Vec<CloudBubble>,
    /// Time (s)
    pub time: f64,
}

/// Complete cloud dynamics response
#[derive(Debug)]
pub struct CloudResponse {
    /// Time series of cloud states
    pub time_steps: Vec<CloudState>,
    /// Configuration used
    pub config: CloudConfig,
}
