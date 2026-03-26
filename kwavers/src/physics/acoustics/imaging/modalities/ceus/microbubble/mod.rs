//! Microbubble Dynamics for Contrast-Enhanced Ultrasound
//!
//! Implements the physics of microbubble oscillation, shell viscoelasticity,
//! and nonlinear acoustic response for CEUS contrast agents.
//!
//! ## Physics Overview
//!
//! Microbubbles oscillate nonlinearly under acoustic excitation, producing
//! harmonic and subharmonic emissions that enable sensitive detection of
//! blood flow and tissue perfusion.
//!
//! ## Mathematical Models
//!
//! ### Modified Rayleigh-Plesset Equation
//! R̈ + (3/2)Ṙ² = (1/ρ)(P_gas + P_shell - P_0 - P_acoustic)
//!
//! ### Shell Viscoelasticity
//! σ_shell = E_shell * ε + η_shell * ε̇
//!
//! ## References
//!
//! - Church (1995): "The effects of an elastic solid surface layer on the radial
//!   pulsations of gas bubbles." JASA 97(3), 1510-1521.
//! - de Jong et al. (2002): "Principles and recent developments in ultrasound
//!   contrast agents." Ultrasonics 40, 71-78.

pub mod dynamics;
pub mod response;

#[cfg(test)]
mod tests;

pub use dynamics::BubbleDynamics;
pub use response::BubbleResponse;

// Re-export standard domain models for convenience
pub use crate::domain::imaging::ultrasound::ceus::{Microbubble, MicrobubblePopulation};
