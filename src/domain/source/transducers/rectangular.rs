//! Rectangular transducer domain entity
//!
//! This module defines the physical properties of a rectangular transducer array,
//! decoupled from any specific solver implementation.

use std::f64::consts::PI;

/// Rectangular transducer description
#[derive(Debug, Clone)]
pub struct RectangularTransducer {
    /// Transducer width (m)
    pub width: f64,
    /// Transducer height (m)
    pub height: f64,
    /// Center frequency (Hz)
    pub frequency: f64,
    /// Number of elements (Nx, Ny)
    pub elements: (usize, usize),
}

impl RectangularTransducer {
    /// Get element size
    pub fn element_size(&self) -> (f64, f64) {
        (
            self.width / self.elements.0 as f64,
            self.height / self.elements.1 as f64,
        )
    }

    /// Get wavenumber
    pub fn wavenumber(&self, c0: f64) -> f64 {
        2.0 * PI * self.frequency / c0
    }
}
