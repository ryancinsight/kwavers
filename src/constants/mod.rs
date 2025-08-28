//! Physical and numerical constants used throughout the library

pub mod numerical;
pub mod physics;
pub mod acoustic;
pub mod thermal;
pub mod optical;
pub mod elastic;
pub mod chemistry;
pub mod bubble_dynamics;
pub mod thermodynamics;
pub mod adaptive_integration;
pub mod performance;
pub mod stability;

// Re-export commonly used constants
pub use numerical::{EPSILON, CFL_SAFETY_FACTOR};
pub use physics::{VACUUM_PERMITTIVITY, VACUUM_PERMEABILITY, SPEED_OF_LIGHT};
pub use acoustic::{WATER_SOUND_SPEED, WATER_DENSITY, WATER_NONLINEARITY};