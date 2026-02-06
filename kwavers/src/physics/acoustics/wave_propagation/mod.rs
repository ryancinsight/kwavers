//! Nonlinear Wave Physics Implementations
//!
//! This module provides concrete implementations of nonlinear acoustic wave physics,
//! including harmonic generation, shock formation, and parametric effects.

pub mod equations; // Nonlinear wave equation specifications

// Re-export nonlinear equation specifications for convenience
pub use equations::{
    HarmonicImaging, HighIntensityTherapy, Microbubble, NonlinearParameters,
    NonlinearWavePropagation, ParametricAcoustics, TissueType,
};
