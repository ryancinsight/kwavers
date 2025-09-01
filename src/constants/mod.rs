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

pub mod performance;
pub mod physics;
pub mod physics_constants;
pub mod stability;
pub mod thermal;
pub mod thermodynamics;

// Re-export commonly used constants
pub use acoustic::{WATER_DENSITY, WATER_SOUND_SPEED};
// Alias for test compatibility
pub use self::acoustic::WATER_DENSITY as DENSITY_WATER;
pub use self::acoustic::WATER_SOUND_SPEED as SPEED_OF_SOUND_WATER;
pub use numerical::{CFL_SAFETY_FACTOR, EPSILON};
