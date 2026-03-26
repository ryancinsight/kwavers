//! Nonlinear Wave Physics Implementations
//! Nonlinear Wave Physics Implementations
//!
//! This module provides concrete implementations of nonlinear acoustic wave physics,
//! including harmonic generation, shock formation, and parametric effects.

pub mod nonlinear; // Nonlinear wave equation specifications

// Re-export nonlinear equation specifications for convenience
pub use nonlinear::{
    burgers::*, harmonics::*, kzk::*, parameters::*, parametric::*, saturation::*, shock::*,
};
