//! Bubble state and parameters
//!
//! Core data structures for bubble dynamics

mod gas_dynamics;
mod parameters;
mod state;

#[cfg(test)]
mod tests;

pub use gas_dynamics::{GasSpecies, GasType};
pub use parameters::{viscous_bubble_wall_stress, young_laplace_pressure, BubbleParameters};
pub use state::BubbleState;
