//! Staggered grid implementation for Yee cell scheme
//!
//! This module implements the staggered grid positions used in the FDTD method
//! following Yee's scheme for enforcing divergence conditions.

/// Staggered grid positions for Yee cell
#[derive(Debug, Clone)]
pub struct StaggeredGrid {
    /// Pressure at cell centers
    pub pressure_pos: (f64, f64, f64),
    /// Velocity components at face centers
    pub vx_pos: (f64, f64, f64),
    pub vy_pos: (f64, f64, f64),
    pub vz_pos: (f64, f64, f64),
}

impl Default for StaggeredGrid {
    fn default() -> Self {
        Self {
            pressure_pos: (0.0, 0.0, 0.0), // Cell center
            vx_pos: (0.5, 0.0, 0.0),       // x-face center
            vy_pos: (0.0, 0.5, 0.0),       // y-face center
            vz_pos: (0.0, 0.0, 0.5),       // z-face center
        }
    }
}

impl StaggeredGrid {
    /// Create a new staggered grid with custom positions
    pub fn new(
        pressure_pos: (f64, f64, f64),
        vx_pos: (f64, f64, f64),
        vy_pos: (f64, f64, f64),
        vz_pos: (f64, f64, f64),
    ) -> Self {
        Self {
            pressure_pos,
            vx_pos,
            vy_pos,
            vz_pos,
        }
    }

    /// Get the offset for a specific field component
    pub fn get_offset(&self, component: FieldComponent) -> (f64, f64, f64) {
        match component {
            FieldComponent::Pressure => self.pressure_pos,
            FieldComponent::VelocityX => self.vx_pos,
            FieldComponent::VelocityY => self.vy_pos,
            FieldComponent::VelocityZ => self.vz_pos,
        }
    }
}

/// Field component types for staggered grid
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldComponent {
    Pressure,
    VelocityX,
    VelocityY,
    VelocityZ,
}