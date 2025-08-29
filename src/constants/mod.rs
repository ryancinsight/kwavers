//! Physical and numerical constants used throughout the library

pub mod acoustic;
pub mod adaptive_integration;
pub mod bubble_dynamics;
pub mod cavitation;
pub mod chemistry;
pub mod elastic;
pub mod medium_properties;
pub mod numerical;
pub mod optical;
pub mod optics;
pub mod performance;
pub mod physics;
pub mod stability;
pub mod thermal;
pub mod thermodynamics;

// Re-export commonly used constants
pub use acoustic::{WATER_DENSITY, WATER_NONLINEARITY, WATER_SOUND_SPEED};
pub use numerical::{CFL_SAFETY_FACTOR, EPSILON};
pub use physics::{SPEED_OF_LIGHT, VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY};
