//! Phase Shifting for Beam Steering and Dynamic Focusing
//!
//! Implements phase shifting techniques for electronic beam steering,
//! dynamic focusing, and multi-focus patterns.
//!
//! ## Architecture
//! - `core`: Fundamental types and utilities
//! - `shifter`: Core phase shifting functionality
//! - `beam`: Beam steering implementation
//! - `focus`: Dynamic focusing and multi-focus patterns
//!
//! ## References
//! - Wooh & Shi (1999): "A simulation study of the beam steering characteristics for linear phased arrays"
//! - Ebbini & Cain (1989): "Multiple-focus ultrasound phased-array pattern synthesis"
//! - Pernot et al. (2003): "3D real-time motion correction in high-intensity focused ultrasound"

pub mod array;
pub mod beam;
pub mod core;
pub mod focus;
pub mod shifter;

// Re-export main types
pub use array::{PerformanceMetrics, PhaseArray};
pub use beam::BeamSteering;
pub use core::{
    calculate_wavelength, normalize_phase, quantize_phase, wrap_phase, ShiftingStrategy,
    MAX_FOCAL_POINTS, MAX_STEERING_ANGLE, MIN_FOCAL_DISTANCE, SPEED_OF_SOUND,
};
pub use focus::{ApodizationType, DynamicFocusing};
pub use shifter::PhaseShifter;
