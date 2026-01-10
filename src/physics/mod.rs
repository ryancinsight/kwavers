//! Physics module for multi-physics simulation
//!
//! This module contains the core physics implementations for acoustic wave simulation,
//! including material properties, wave equations, and numerical constants.

pub mod acoustics;
pub mod chemistry;
pub mod optics;
pub mod thermal;
// pub mod factory; // Removed
pub mod constants {
    pub use crate::domain::core::constants::*;
}

pub use acoustics::*;

pub mod plugin {
    pub use crate::domain::plugin::*;
}

pub mod imaging {
    pub use crate::physics::acoustics::imaging::modalities::{ceus, elastography, ultrasound};
    pub use crate::physics::acoustics::imaging::*;
}

// Backward-compatible re-exports for moved modules
// These ensure existing code continues to work during refactoring

/// Re-export wave_propagation from its new location
pub use acoustics::analytical::propagation as wave_propagation;

/// Re-export bubble_dynamics from its new location (now at nonlinear root level files)
pub use acoustics::nonlinear as bubble_dynamics;

/// Re-export cavitation_control from its new location
pub use acoustics::nonlinear::cavitation_control;

/// Re-export phase_modulation from its new location  
pub use acoustics::analytical::patterns as phase_modulation;

/// Re-export sonoluminescence_detector from its new location in sensor module
pub use crate::domain::sensor::sonoluminescence as sonoluminescence_detector;
