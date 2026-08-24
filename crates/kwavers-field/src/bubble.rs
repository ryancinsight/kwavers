//! Bubble field definitions
//!
//! Defines the state fields for bubble dynamics.

use kwavers_core::constants::fundamental::ATMOSPHERIC_PRESSURE;
use kwavers_core::constants::thermodynamic::ROOM_TEMPERATURE_K;
use leto::Array3;

/// Bubble state fields for interfacing with physics modules
#[derive(Debug)]
pub struct BubbleStateFields {
    /// Bubble radius field (m)
    pub radius: Array3<f64>,
    /// Bubble temperature field (K)
    pub temperature: Array3<f64>,
    /// Bubble pressure field (Pa)
    pub pressure: Array3<f64>,
    /// Bubble wall velocity field (m/s)
    pub velocity: Array3<f64>,
    /// Collapse indicator: 1 where the bubble is collapsing, 0 elsewhere
    pub is_collapsing: Array3<f64>,
    /// Ratio of current to initial bubble volume
    pub compression_ratio: Array3<f64>,
}

impl BubbleStateFields {
    /// Create bubble state fields for the given shape.
    ///
    /// Fields are zero-initialized except for `temperature` (room
    /// temperature), `pressure` (atmospheric pressure), and
    /// `compression_ratio` (unity), which seed the equilibrium state.
    #[must_use]
    pub fn new(shape: (usize, usize, usize)) -> Self {
        let shape = [shape.0, shape.1, shape.2];

        Self {
            radius: Array3::zeros(shape),
            temperature: Array3::from_elem(shape, ROOM_TEMPERATURE_K),
            pressure: Array3::from_elem(shape, ATMOSPHERIC_PRESSURE),
            velocity: Array3::zeros(shape),
            is_collapsing: Array3::zeros(shape),
            compression_ratio: Array3::from_elem(shape, 1.0),
        }
    }
}
