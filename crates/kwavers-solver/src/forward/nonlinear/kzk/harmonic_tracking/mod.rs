//! Harmonic Generation and Tracking for KZK Equation.
//!
//! Tools for tracking harmonic content during nonlinear acoustic propagation,
//! enabling monitoring of energy transfer from fundamental to higher frequencies.
//!
//! ## References
//! - Aanonsen et al. (1984) "Distortion and harmonic generation in the nearfield"
//! - Tjøtta & Tjøtta (1980) "Nonlinear waves in fluids with viscosity and diffusivity"

#[cfg(test)]
mod tests;
pub mod tracker;
pub mod types;

pub use tracker::HarmonicTracker;
pub use types::{HarmonicAnalysis, HarmonicConfig, PredictionModel};
