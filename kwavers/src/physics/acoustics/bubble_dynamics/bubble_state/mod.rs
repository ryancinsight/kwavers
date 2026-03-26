//! Bubble state and parameters
//!
//! Core data structures for bubble dynamics

mod gas_dynamics;
mod parameters;
mod state;

#[cfg(test)]
mod tests;

pub use gas_dynamics::{GasSpecies, GasType};
pub use parameters::BubbleParameters;
pub use state::BubbleState;
