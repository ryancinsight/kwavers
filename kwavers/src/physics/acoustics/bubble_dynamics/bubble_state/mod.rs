//! Bubble state and parameters
//!
//! Core data structures for bubble dynamics

mod gas_dynamics;
mod parameters;
mod state;

#[cfg(test)]
mod tests;

pub use gas_dynamics::{GasSpecies, GasType};
pub use parameters::{young_laplace_pressure, viscous_bubble_wall_stress, BubbleParameters};
pub use state::BubbleState;
