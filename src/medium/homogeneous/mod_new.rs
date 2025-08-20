//! Homogeneous medium module
//!
//! Provides a medium with uniform properties throughout the spatial domain.

mod constants;
mod float_key;
mod absorption_cache;
mod core;

pub use constants::*;
pub use core::HomogeneousMedium;

// Re-export for backward compatibility
pub use self::HomogeneousMedium as Homogeneous;