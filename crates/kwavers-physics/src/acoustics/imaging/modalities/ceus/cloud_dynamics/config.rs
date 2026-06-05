//! Cloud configuration and bubble type definitions

use super::super::microbubble::Microbubble;

/// Individual bubble in a cloud
#[derive(Debug, Clone)]
pub struct CloudBubble {
    /// Bubble properties
    pub properties: Microbubble,
    /// Position in space (x, y, z) in meters
    pub position: [f64; 3],
    /// Velocity (vx, vy, vz) in m/s
    pub velocity: [f64; 3],
    /// Current radius (m)
    pub current_radius: f64,
    /// Bubble ID for tracking
    pub id: usize,
    /// Whether bubble is still active
    pub active: bool,
}

/// Microbubble cloud configuration
#[derive(Debug, Clone)]
pub struct CloudConfig {
    /// Initial number of bubbles
    pub num_bubbles: usize,
    /// Bubble concentration (bubbles/m³)
    pub concentration: f64,
    /// Cloud volume dimensions (m)
    pub dimensions: [f64; 3],
    /// Time step for dynamics (s)
    pub dt: f64,
    /// Simulation duration (s)
    pub duration: f64,
    /// Critical distance for coalescence (m)
    pub coalescence_distance: f64,
    /// Enable bubble-bubble interactions
    pub enable_interactions: bool,
    /// Enable dissolution dynamics
    pub enable_dissolution: bool,
}

impl Default for CloudConfig {
    fn default() -> Self {
        Self {
            num_bubbles: 1000,
            concentration: 1e12, // 10^12 bubbles/m³ (typical for contrast agents)
            dimensions: [1e-3, 1e-3, 1e-3], // 1 mm³ volume
            dt: 1e-8,            // 10 ns (much smaller than acoustic period)
            duration: 1e-3,      // 1 ms
            coalescence_distance: 2e-6, // 2 μm
            enable_interactions: true,
            enable_dissolution: false,
        }
    }
}
