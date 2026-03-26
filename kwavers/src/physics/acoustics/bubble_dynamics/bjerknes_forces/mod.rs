//! Bjerknes Forces: Bubble-Bubble Interactions in Acoustic Fields
//!
//! This module implements primary and secondary Bjerknes forces that govern
//! the interactions between microbubbles in acoustic fields.
//!
//! ## Physical Background
//!
//! **Primary Bjerknes Force**:
//! Arises from the time-averaged pressure gradient in the acoustic field acting on
//! the oscillating bubble surface. Each bubble experiences a radiation force.
//!
//! ```text
//! F_primary = -4πR² ⟨p'(∂u'/∂z)⟩
//! ```
//! where ⟨⟩ denotes time averaging and R is bubble radius
//!
//! **Secondary Bjerknes Force**:
//! Results from the acoustic field scattered by one bubble affecting another.
//! This force can be attractive (bubbles oscillate in phase) or repulsive
//! (out of phase oscillations).
//!
//! ```text
//! F_secondary ≈ (6πρc₀/f) * V₁ * V₂ * cos(φ) / d²
//! ```
//! where V₁, V₂ are bubble volume amplitudes, φ is phase difference, d is separation
//!
//! ## References
//!
//! - Björknes, V. (1906). "Fields of force" Columbia University Press
//! - Crum, L. A. (1975). "Bjerknes forces on bubbles in a stationary sound field"
//! - Pozarowski, P. & Holyst, R. (2002) "Two and three body interactions"
//! - Garbin, V. et al. (2007) "Dynamics of coated bubbles monitored by individual pulse acoustography"

pub mod calculator;
pub mod kinematics;
pub mod primary;
pub mod secondary;
pub mod types;

#[cfg(test)]
mod tests;

pub use calculator::BjerknesCalculator;
pub use types::{BjerknesConfig, BjerknesForce, InteractionType};
